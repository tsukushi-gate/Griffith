"""
LogicSeqMixer 訓練スクリプト（/ seq2seq）

TSMixer 風の token-mixing / channel-mixing を論理ゲート化した LogicSeqMixer を、
`train_rddlgn.py` と **完全に同一の評価プロトコル** で学習・評価する。
RDDLGN との公平比較が目的なので、データ・teacher forcing・損失・精度・BLEU・
最適化設定は一切変更していない（変更点は本ファイル末尾のコメント参照）。

アーム:
    (i)  --cross-attention none            … 主アーム（cross-attention 無し）
    (ii) --cross-attention logic_single_head … 論理単一ヘッド cross-attention

使用例:
    # 極小スモーク（copy / tiny / CPU）
    uv run python experiments/train_logic_seq_mixer.py \\
        --task copy --config tiny -ni 20 -ef 10 --device cpu

    # copy タスク（RDDLGN tiny と比較）
    uv run python experiments/train_logic_seq_mixer.py \\
        --task copy --config tiny -ni 50000 -ef 5000

    # WMT'14 縮小ステップ（アーム(i)）
    uv run python experiments/train_logic_seq_mixer.py \\
        --task wmt14 --config paper --vocab-size 16000 --seq-len 16 \\
        --cross-attention none -ni 40000 -ef 2000 --bleu-every 10000

    # WMT'14 本走（アーム(ii)、論文と同じ 130K ステップ）
    uv run python experiments/train_logic_seq_mixer.py \\
        --task wmt14 --config paper --vocab-size 16000 --seq-len 16 \\
        --cross-attention logic_single_head -ni 130000 -ef 5000

    # Modal 投入（--background は必ず --detach とセット）
    uv run modal run --detach scripts/modal_app.py --background \\
        --script train_logic_seq_mixer.py \\
        --args "--task wmt14 --config paper -ni 40000 -ef 2000"
"""

import argparse
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import (
    checkpoint_paths,
    find_resumable_checkpoint,
    load_checkpoint,
    save_checkpoint,
    setup_checkpoint_dir,
)
from data import TASK_DATASETS, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD
from env_utils import load_dotenv
from wandb_utils import (
    LOSS_LOG_INTERVAL_SEC,
    LossLogPacer,
    add_experiment_args,
    count_logic_gates,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
    progress_metrics,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.models import LogicSeqMixer, LogicSeqMixerConfig

# .envから環境変数を読み込み
load_dotenv()

# 自動再開の際に「同じ実験か」を判定する args のキー。
# metadata['config']（モデル構成の全体）に加えて、データ・最適化・乱数の同一性を確認する。
AUTO_RESUME_ARGS_KEYS = (
    'task', 'config', 'cross_attention', 'seed', 'num_iterations',
    'vocab_size', 'seq_len', 'num_samples', 'num_test_samples',
    'lr', 'weight_decay', 'batch_size', 'label_smoothing',
    'patience', 'lr_factor', 'n_inputs', 'parameterization', 'tau',
    'selection', 'gumbel', 'positional', 'n_pos_bits', 'context_mode',
)


# ============================================================
# 訓練・評価（RDDLGNプロトコルと同一）
# ============================================================

def get_emb_reg_weight(step: int, ramp_start: int = 1000, ramp_end: int = 100000,
                       max_weight: float = 0.1) -> float:
    """二値化正則化の重みを線形ランプで計算（RDDLGNと同一）。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


def train_step(model, src, tgt, optimizer, step, label_smoothing=0.1):
    """1ステップの訓練。teacher forcing。"""
    model.train()
    vocab_size = model.config.vocab_size

    # teacher forcing: tgt[:-1]が入力、tgt[1:]がラベル
    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    logits, emb_loss = model(src, tgt_input)

    main_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt_label.reshape(-1),
        label_smoothing=label_smoothing,
        ignore_index=TOKEN_PAD,
    )

    emb_weight = get_emb_reg_weight(step)
    total_loss = main_loss + emb_weight * emb_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # emb_loss は requires_grad=True の0次元Tensor。float()直接変換は
    # UserWarning('Converting a tensor with requires_grad=True to a scalar')
    # を出すので detach してから取り出す。
    return total_loss.item(), main_loss.item(), emb_loss.detach().item()


def _iter_chunks(n: int, chunk_size: int):
    """[0, n) を chunk_size 単位の (start, end) に分割する。0以下なら分割しない。"""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        yield 0, n
        return
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


@torch.no_grad()
def evaluate(model, loader, device, use_soft=True, chunk_size=0):
    """トークンレベルの精度を計算（PADは無視）。

    use_soft=True は model.train()（soft）、False は model.eval()（hard）。

    chunk_size はメモリ管理のためのforward分割サイズ。**閾値や評価条件を緩める
    変更ではなく純粋なメモリ管理**で、各サンプルの forward は独立なので分割の
    有無で結果は厳密に同一になる（精度はPADマスク付きの一致数/総数の比のため、
    分割して加算しても同じ値）。hard eval(use_soft=False)はCUDAカーネル分岐が
    無く `(B×T, out_dim, 2^N)` を素で確保するため、ここを刻まないとOOMする。
    """
    was_training = model.training
    model.train(mode=use_soft)

    correct = 0
    total = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 訓練時と同じ形式にする（BOS/EOS付加）
        src_batch, tgt_full = prepare_batch(src, tgt)

        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        for start, end in _iter_chunks(src_batch.shape[0], chunk_size):
            logits, _ = model(src_batch[start:end], tgt_input[start:end])
            preds = logits.argmax(dim=-1)

            labels = tgt_label[start:end]
            mask = labels != TOKEN_PAD
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def greedy_decode(model, src, max_len, bos_token=TOKEN_BOS):
    """貪欲デコーディングでシーケンスを生成。

    意味論は `train_rddlgn.py` の `greedy_decode` と同一
    （BOSから開始／毎ステップ prefix 全体を再デコード／`logits[:, -1].argmax`／
    返り値はBOSを除く）。LogicSeqMixer の encoder はトークン系列
    `(B, S, d_enc)` を返すので、その受け取り方だけ本モデル用に書き下ろす。

    注意: `train_rddlgn.py` と違い `model.eval()` をハードコードしない。
    soft(relaxed) BLEU と hard(collapsed) BLEU の両方を測るため、
    soft/hard の切替は呼び出し元（`compute_bleu`）の責務とする。
    本モデルに dropout / BatchNorm は無いので train モードでの評価も安全。
    """
    batch = src.shape[0]
    device = src.device

    # エンコード（enc_states: (B, S, d_enc)）
    src_emb = model.src_embedding(src)
    enc_states = model.encoder(src_emb)

    # デコード（自己回帰）
    generated = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_emb = model.tgt_embedding(generated)
        logits = model.decoder(tgt_emb, enc_states)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated[:, 1:]  # BOSを除く


@torch.no_grad()
def compute_bleu(model, loader, tokenizer, device, max_samples=1000, use_soft=False,
                 chunk_size=0):
    """greedy decodeでBLEUスコアを計算（RDDLGNと同一プロトコル）。

    - `sacrebleu.corpus_bleu(hyps, [refs]).score`（tokenize引数を指定しない=既定13a）
    - `max_len = tgt.shape[1]`
    - `max_samples=1000`
    - `ref.strip()` が空のサンプルは除外
    この4条件は公平比較の生命線なので変更しない。

    Args:
        use_soft: True で soft(relaxed) BLEU、False で hard(collapsed) BLEU。
            `evaluate(..., use_soft=...)` と同じ流儀で `model.train(mode=use_soft)`
            により切り替える。RDDLGN論文は relaxed 5.00 / Collapsed 4.39 の2行を
            報告しており、その落差(collapse gap)を本モデルでも測るために必要。
        chunk_size: greedy decode の forward 分割サイズ（メモリ管理のみ）。
            各サンプルのデコードは独立なので、分割しても hyp/ref の集合・順序・
            対象サンプル数(max_samples)は一切変わらず BLEU は厳密に同一になる。
    """
    if tokenizer is None:
        # 合成タスク等（tokenizerが無い）は従来どおり静かにスキップ
        return None

    # tokenizerがあるのにsacrebleuが無いのは構成ミス。握り潰さず落とす
    # （7時間走ってからBLEUが取れていないと気づく事故を防ぐ）。
    import sacrebleu

    was_training = model.training
    model.train(mode=use_soft)
    hypotheses = []
    references = []
    count = 0

    for src, tgt in loader:
        if count >= max_samples:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        seq_len = tgt.shape[1]

        # forwardのみチャンク分割（サンプル独立なので結果は分割前と厳密に同一）
        preds = torch.cat(
            [greedy_decode(model, src[start:end], max_len=seq_len)
             for start, end in _iter_chunks(src.shape[0], chunk_size)],
            dim=0,
        )

        for i in range(src.shape[0]):
            if count >= max_samples:
                break
            hyp = tokenizer.decode(preds[i].tolist())
            ref = tokenizer.decode(tgt[i].tolist())
            if ref.strip():
                hypotheses.append(hyp)
                references.append(ref)
                count += 1

    model.train(mode=was_training)

    if not hypotheses:
        return 0.0

    return sacrebleu.corpus_bleu(hypotheses, [references]).score


_BYTES_PER_GB = 1024 ** 3


def reset_peak_memory(device: str) -> None:
    """ピークメモリ統計をリセットする（CUDA以外では何もしない）。"""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: str) -> tuple[float, float] | None:
    """(allocated, reserved) のピークをGB単位で返す。CUDA以外はNone。

    参考: experiments/bench_iwp.py の peak_memory_bytes（推定はしない方針を踏襲）。
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return (
        torch.cuda.max_memory_allocated() / _BYTES_PER_GB,
        torch.cuda.max_memory_reserved() / _BYTES_PER_GB,
    )


def require_bleu_deps(task: str) -> None:
    """wmt14 runでは起動時に sacrebleu の存在を検証し、無ければ即座に落とす。

    以前は `compute_bleu` 内で ImportError を握り潰して None を返していたため、
    BLEUが無言でスキップされ「7時間のGPU本走が終わってから成果物が無いと気づく」
    事故が起きうる。警告ではなく起動時エラーにして構造的に防ぐ。
    """
    if task != 'wmt14':
        return
    try:
        import sacrebleu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            'BLEU計測に必要な sacrebleu が見つかりません（--task wmt14 では必須）。\n'
            '  ローカル: 既存extrasを全て並べたうえで `uv sync ... --extra wmt` を実行してください\n'
            '           （uv sync は列挙しなかったextraを削除するため、消してはいけない'
            'extraを必ず全部並べること）\n'
            '  Modal   : scripts/modal_app.py の .uv_sync(extras=[...]) に "wmt" が'
            '含まれているか確認してください\n'
            '  wmt extra には datasets（WMT\'14データ取得）と sacrebleu（BLEU）が入っています。'
        ) from e


# ============================================================
# ログ
# ============================================================

def setup_logger(task, log_dir='logs'):
    """ログファイルを作成する（run_dir 側にも log.txt を出す）。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'seqmixer_{task}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    handles = [log_file]

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        for fh in handles:
            fh.write(f'[{ts}] {msg}\n')
            fh.flush()  # 長時間runでも log.txt を逐次追跡できるようにする

    def attach(path):
        """run_dir 決定後に log.txt を追加で開く。"""
        handles.append(open(path, 'a', buffering=1))

    def close():
        for fh in handles:
            fh.close()

    return log, attach, close


# ============================================================
# 設定構築
# ============================================================

def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。例: '3,8,16' -> [3, 8, 16]"""
    return [int(x.strip()) for x in s.split(',')]


# CLI dest → LogicSeqMixerConfig のフィールド名
_CONFIG_FIELD_MAP = {
    'vocab_size': 'vocab_size',
    'd_emb': 'd_emb',
    'seq_len': 'seq_len',
    'd_channels': 'd_channels',
    'channel_hidden': 'channel_hidden',
    'n_enc_blocks': 'n_enc_blocks',
    'n_dec_blocks': 'n_dec_blocks',
    'channel_depth': 'channel_depth',
    'token_kernels': 'token_kernels',
    'token_depth': 'token_depth',
    'gate': 'gate',
    'residual_mode': 'residual_mode',
    'residual_init': 'residual_init',
    'm_sizes': 'm_layer_sizes',
    'positional': 'positional',
    'n_pos_bits': 'n_pos_bits',
    'context_mode': 'context_mode',
    'cross_attention': 'cross_attention',
    'n_inputs': 'n_inputs',
    'parameterization': 'parameterization',
    'tau': 'tau',
    'selection': 'selection',
    'gumbel': 'gumbel',
}


def _model_choices(name: str) -> list[str] | None:
    """モデル側の列挙定数から argparse の choices を動的に引く。

    値域をスクリプト側にハードコードすると、モデル側に選択肢
    （例: context_mode の 'gather'）が追加されたとき CLI から指定できなくなる。
    定数が見つからない場合は None を返し（=choices制約なし）、値の妥当性検証は
    LogicSeqMixerConfig.__post_init__ に委ねる。
    """
    try:
        from griffith_logic.models import logic_seq_mixer as _mod
    except ImportError:
        return None
    values = getattr(_mod, name, None)
    return list(values) if values else None


def _cli_value(args, dest):
    """CLI値をconfigフィールド用の型へ変換する（列挙系はtupleへ）。

    LogicSeqMixerConfig の token_kernels / m_layer_sizes は tuple 型なので
    tuple に揃える。
    """
    value = getattr(args, dest)
    if dest in ('token_kernels', 'm_sizes') and isinstance(value, str):
        return tuple(parse_int_list(value))
    return value


def _explicit_dests(parser, args) -> set[str]:
    """既定値から変わっている引数のdest集合を返す（明示指定の近似判定）。

    プリセット(paper または tiny)使用時に、ユーザーが明示指定した項目だけを
    上書きするために使う（既定値のまま = プリセットを尊重）。
    """
    return {
        dest for dest in _CONFIG_FIELD_MAP
        if hasattr(args, dest) and getattr(args, dest) != parser.get_default(dest)
    }


def _set_if_exists(config, field: str, value, skipped: list[str]) -> None:
    """configに存在するフィールドのみ設定する。存在しなければ記録して無視。"""
    if value is None:
        return
    if hasattr(config, field):
        setattr(config, field, value)
    else:
        skipped.append(field)


# custom で未指定だと事故になりやすい（＝規模を直接決める）項目
_CUSTOM_CRITICAL_DESTS = ('vocab_size', 'seq_len', 'd_channels', 'm_sizes')


def build_config(args, parser) -> LogicSeqMixerConfig:
    """CLI引数から LogicSeqMixerConfig を構築する。

    custom の土台は **tiny()**。LogicSeqMixerConfig の dataclass 既定値は
    論文規模(vocab 16000 / d_channels 9216 / m_layer_sizes 400k,400k,480k)なので、
    それを土台にすると `--config custom --d-channels 48` だけの指定で
    128万ゲート級の巨大ヘッドが黙って付いてしまう（メモリ枯渇事故）。
    """
    skipped: list[str] = []

    if args.config == 'paper':
        config = LogicSeqMixerConfig.from_paper()
    else:
        # tiny / custom はどちらも tiny() を土台にする
        config = LogicSeqMixerConfig.tiny()

    # 明示指定された項目だけ上書きする（既定値のままなら土台の値を尊重）
    explicit = _explicit_dests(parser, args)
    for dest in explicit:
        _set_if_exists(config, _CONFIG_FIELD_MAP[dest], _cli_value(args, dest), skipped)
    # cross_attention と gumbel はアーム切替・学習挙動の要なので常に反映する
    _set_if_exists(config, 'cross_attention', args.cross_attention, skipped)
    _set_if_exists(config, 'gumbel', args.gumbel, skipped)

    if args.config == 'custom':
        unspecified = [d for d in _CUSTOM_CRITICAL_DESTS if d not in explicit]
        if unspecified:
            print('警告: --config custom で未指定の項目は custom既定(tiny由来)を採用します:')
            for dest in unspecified:
                field = _CONFIG_FIELD_MAP[dest]
                print(f'  --{dest.replace("_", "-")} 未指定 → '
                      f'{field}={getattr(config, field, "N/A")} (tiny由来)')

    if skipped:
        print(f'警告: LogicSeqMixerConfig に存在しない項目を無視しました: {sorted(set(skipped))}')

    _warn_projection_capacity(config)

    return config


def _warn_projection_capacity(config) -> None:
    """埋め込み射影のタップ数不足を警告する（エラーにはしない）。

    埋め込み+位置ビット(d_emb + pos_width)を d_channels へ落とす射影のタップ数は
    2×d_channels 本しかないため、`2*d_channels < d_emb + pos_width` だと入力の
    一部が構造的に拾われず、学習が「静かに」劣化する（実測済みの罠）。
    アブレーションで意図的に破ることもあるため警告に留める。
    """
    d_channels = getattr(config, 'd_channels', None)
    d_emb = getattr(config, 'd_emb', None)
    if d_channels is None or d_emb is None:
        return
    pos_width = getattr(config, 'pos_width', 0)
    need = d_emb + pos_width
    if 2 * d_channels < need:
        print(
            f'警告: 設計条件 2×d_channels >= d_emb + n_pos_bits を満たしていません '
            f'(2×{d_channels}={2 * d_channels} < {d_emb}+{pos_width}={need})。'
            f'射影のタップ数が足りず入力の一部が構造的に捨てられ、学習が静かに劣化します'
            f'（アブレーションで意図的なら無視してください）。'
        )


def prepare_batch(src, tgt, bos_token=TOKEN_BOS, eos_token=TOKEN_EOS):
    """ターゲットにBOS/EOSを付加する（tgt_full = [BOS, tgt, EOS]）。"""
    batch = src.shape[0]
    device = src.device
    bos = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)
    eos = torch.full((batch, 1), eos_token, dtype=torch.long, device=device)
    tgt_full = torch.cat([bos, tgt, eos], dim=1)
    return src, tgt_full


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ。"""
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='LogicSeqMixer 訓練スクリプト')

    # タスク設定
    parser.add_argument('--task', type=str, default='copy',
                        choices=list(TASK_DATASETS.keys()) + ['wmt14'],
                        help='タスク (default: copy)')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='訓練サンプル数 (default: 50000)')
    parser.add_argument('--num-test-samples', type=int, default=5000,
                        help='テストサンプル数 (default: 5000)')

    # モデル構成
    parser.add_argument('--config', type=str, default='tiny',
                        choices=['paper', 'tiny', 'custom'],
                        help='モデル構成プリセット (default: tiny)')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='語彙サイズ（wmt14では実語彙数で上書きされる）')
    parser.add_argument('--d-emb', type=int, default=None, help='埋め込みビット幅')
    parser.add_argument('--seq-len', type=int, default=None, help='系列長')
    parser.add_argument('--d-channels', type=int, default=None,
                        help='トークンあたりのチャネル(ビット)幅')
    parser.add_argument('--channel-hidden', type=int, default=None,
                        help='channel-mixing の中間幅')
    parser.add_argument('--n-enc-blocks', type=int, default=None,
                        help='エンコーダのmixerブロック数')
    parser.add_argument('--n-dec-blocks', type=int, default=None,
                        help='デコーダのmixerブロック数')
    parser.add_argument('--channel-depth', type=int, default=None,
                        help='channel-mixing LutStack の段数')
    parser.add_argument('--token-kernels', type=str, default='3,8,16',
                        help='token-mixing のカーネル幅（カンマ区切り, default: 3,8,16）')
    parser.add_argument('--token-depth', type=int, default=None,
                        help='token-mixing LutStack の段数')
    parser.add_argument('--gate', type=str, default=None, choices=['and', 'none'],
                        help='mixing出力のゲーティング方式')
    parser.add_argument('--residual-mode', type=str, default=None,
                        choices=['xor', 'none'], help='残差結合の方式')
    parser.add_argument('--residual-init', action='store_true',
                        help='LutLayerを恒等寄り初期化する')
    parser.add_argument('--m-sizes', type=str, default=None,
                        help='出力ヘッドM層のサイズ（カンマ区切り）')
    parser.add_argument('--positional', type=str, default=None,
                        choices=_model_choices('_POSITIONAL'),
                        help='位置符号化（none はアブレーション対照。既定はモデル側config）')
    parser.add_argument('--n-pos-bits', type=int, default=None,
                        help='位置ビット幅（positional=learned_bits時。既定はモデル側config）')
    parser.add_argument('--context-mode', type=str, default=None,
                        choices=_model_choices('_CONTEXT_MODES'),
                        help='encoder表現のdecoderへの渡し方（既定はモデル側config）')
    parser.add_argument('--cross-attention', type=str, default='none',
                        choices=['none', 'logic_single_head'],
                        help='アーム切替: none=主アーム(i) / logic_single_head=アーム(ii)')
    parser.add_argument('--n-inputs', type=int, default=None, choices=[2, 4, 6],
                        help='LUTのfan-in')
    parser.add_argument('--parameterization', type=str, default=None,
                        choices=['softmax', 'independent', 'iwp', 'warp'],
                        help='ゲート関数のパラメータ化')
    parser.add_argument('--tau', type=float, default=None, help='温度パラメータ')
    parser.add_argument('--selection', type=str, default=None,
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式')
    parser.add_argument('--no-gumbel', dest='gumbel', action='store_false', default=None,
                        help='Gumbel smoothingを無効化')

    # 訓練設定（RDDLGNと同一の既定値）
    parser.add_argument('--lr', type=float, default=0.05,
                        help='学習率 (default: 0.05, 論文値)')
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=130000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10000,
                        help='ReduceLROnPlateau patience (default: 10000)')
    parser.add_argument('--lr-factor', type=float, default=0.8,
                        help='学習率減衰係数 (default: 0.8)')
    parser.add_argument('--eval-chunk-size', type=int, default=8,
                        help='評価forwardの分割サイズ。訓練の-bsとは独立。'
                             'hard evalはCUDAカーネル分岐が無くOOMしやすいため既定8'
                             '（0以下で分割なし。結果は分割前と厳密に同一）')
    parser.add_argument('--bleu-every', type=int, default=0,
                        help='BLEUを測る間隔(step)。0=最終stepのみ(RDDLGNと同挙動, default: 0)')
    parser.add_argument('--log-every-steps', type=int, default=None,
                        help='train lossをwandbへ記録するstep間隔。未指定なら実測s/itから'
                             f'{int(LOSS_LOG_INTERVAL_SEC)}秒以内になる値を自動算出する'
                             '（重い評価の間隔 -ef / --bleu-every とは独立）')

    # チェックポイント
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='自動再開を無効化して必ず step 0 から開始する'
                             '（既定は自動再開ON。Modalのリトライで巻き戻らないため）')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None,
                        help='未指定なら自動検出 (cuda→mps→cpu)')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--data-dir', type=str, default='./data-wmt14',
                        help='WMTデータ保存先 (default: ./data-wmt14)')
    parser.add_argument('--job-type', type=str, default=None,
                        choices=['train', 'ablation', 'sweep', 'eval', 'smoke'],
                        help='wandb job_type（未指定なら wmt14 本走=train / それ以外=smoke）')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを使わない')

    # --notes / --hypothesis
    add_experiment_args(parser)
    return parser


# ============================================================
# メイン
# ============================================================

def main():
    # リダイレクト時(`> file 2>&1`)の標準出力はブロックバッファリングされ、
    # 長時間runの [iter ...] 行が終了時まで書き出されず進捗監視ができない。
    # 行バッファリングへ切り替えて逐次フラッシュさせる(Modal本走で必須)。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = build_parser()
    args = parser.parse_args()

    # デバイス自動検出（cuda→mps→cpu）
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # BLEU依存の事前検証（データ取得や学習に入る前に落とす）
    require_bleu_deps(args.task)

    # シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # モデル構成
    config = build_config(args, parser)

    if args.vocab_size is not None:
        config.vocab_size = args.vocab_size
    if args.seq_len is not None:
        config.seq_len = args.seq_len

    print('\n=== LogicSeqMixer 訓練 ===')
    print(f'タスク: {args.task}')
    print(f'アーム: cross_attention={config.cross_attention}')
    print(f'構成: {args.config}')
    print(f'位置符号化: positional={getattr(config, "positional", "N/A")} '
          f'(pos_width={getattr(config, "pos_width", "N/A")}, '
          f'n_pos_bits={getattr(config, "n_pos_bits", "N/A")})')
    print(f'context_mode: {getattr(config, "context_mode", "N/A")}')
    print(f'd_emb={getattr(config, "d_emb", "N/A")} '
          f'd_channels={getattr(config, "d_channels", "N/A")}')

    # データセット（RDDLGNと同一の取得経路）
    wmt_tokenizer = None
    if args.task == 'wmt14':
        from data.wmt14 import download_and_tokenize
        train_set, test_set, wmt_tokenizer = download_and_tokenize(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            data_dir=args.data_dir,
        )
        # 実語彙数を優先してconfigを上書き
        config.vocab_size = wmt_tokenizer.vocab_size
    else:
        dataset_cls = TASK_DATASETS[args.task]
        train_set = dataset_cls(config.vocab_size, config.seq_len, args.num_samples)
        test_set = dataset_cls(config.vocab_size, config.seq_len, args.num_test_samples)

    # M最終層がvocab_sizeで割り切れるか確認・調整（RDDLGNと同じ丸め流儀）
    if hasattr(config, 'm_layer_sizes') and config.m_layer_sizes:
        m_last = config.m_layer_sizes[-1]
        if m_last % config.vocab_size != 0:
            # tuple の場合があるのでコピーして作り直す（元の型を保つ）
            sizes = list(config.m_layer_sizes)
            group_factor = max(1, m_last // config.vocab_size)
            sizes[-1] = config.vocab_size * max(group_factor, 1)
            config.m_layer_sizes = (
                tuple(sizes) if isinstance(config.m_layer_sizes, tuple) else sizes
            )
            print(f'M最終層を {m_last} → {config.m_layer_sizes[-1]} に調整 '
                  f'(vocab_size={config.vocab_size} の倍数)')

    print(f'語彙サイズ: {config.vocab_size}')
    print(f'系列長: {config.seq_len}')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    print(f'訓練サンプル: {len(train_set):,}')
    print(f'テストサンプル: {len(test_set):,}')

    # モデル構築
    model = LogicSeqMixer(config, device=args.device).to(args.device)
    counts = model.count_parameters()
    gate_counts = model.count_gates()
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'総ロジックゲート数: {gate_counts["total"]:,}')

    # Optimizer / Scheduler（RDDLGNと同一）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.patience,
    )

    # ログ
    log, attach_log, close_log = setup_logger(args.task, args.log_dir)

    # チェックポイント
    ckpt_prefix = f'seqmixer_{args.task}_{args.config}_{args.cross_attention}'

    vocab_path = None
    if wmt_tokenizer is not None:
        vocab_path = os.path.join(args.data_dir, f'vocab_v{config.vocab_size}.json')

    ckpt_metadata = {
        'args': vars(args),
        'config': asdict(config),
        'task': args.task,
        'vocab_path': vocab_path,
    }

    # 自動再開の候補探索
    # Modal の自動リトライでは同じコマンドが再実行され、setup_checkpoint_dir が
    # 新しいタイムスタンプ付き run_dir を作るため、run_dir 内だけを見ても前試行の
    # latest.pt は見つからない。prefix 横断で走査し、metadata が整合する未完了runの
    # 最も進んだ ckpt を選ぶ（で約$25.3を溶かした事故の再発防止）。
    resume_path = args.resume
    resume_run_dir = None
    if resume_path is None and not args.no_auto_resume:
        candidate = find_resumable_checkpoint(
            ckpt_prefix,
            base_dir=args.checkpoint_dir,
            expected_metadata=ckpt_metadata,
            num_iterations=args.num_iterations,
            match_keys=('task', 'config'),
            args_match_keys=AUTO_RESUME_ARGS_KEYS,
        )
        if candidate is not None:
            resume_path = candidate['path']
            resume_run_dir = candidate['run_dir']
            print(f'自動再開: {resume_path} から step {candidate["step"]} を復元')
        else:
            print('自動再開: 候補なし。step 0 から開始')
    elif args.no_auto_resume:
        print('自動再開: --no-auto-resume により無効')

    # 再開時は既存 run_dir に書き続ける（log.txt と ckpt を分断させない）
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        latest_path, best_path = checkpoint_paths(run_dir)
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir(
            ckpt_prefix, base_dir=args.checkpoint_dir,
        )
    attach_log(os.path.join(run_dir, 'log.txt'))
    log(f'START task={args.task} config={args.config} '
        f'cross_attention={args.cross_attention} vocab={config.vocab_size} '
        f'seq_len={config.seq_len} lr={args.lr} bs={args.batch_size} '
        f'ni={args.num_iterations} device={args.device} '
        f'params={counts["total"]:,} gates={gate_counts["total"]:,}')

    # チェックポイントから復元
    # 注意: train_rddlgn.py は resume 後に best_test_acc=0.0 を再代入してしまう
    # 既知バグ(L606付近)がある。本スクリプトでは best をここで確定させ、
    # 訓練ループ開始前に再初期化しない（復元した best を正しく引き継ぐ）。
    start_step = 0
    best_test_acc = 0.0
    if resume_path:
        print(f'チェックポイントを読み込み: {resume_path}')
        ckpt_info = load_checkpoint(
            resume_path, model, optimizer, scheduler, device=args.device,
        )
        start_step = ckpt_info['step']
        best_test_acc = ckpt_info['best_metric']
        ckpt_metadata.update(ckpt_info.get('metadata', {}))
        kind = '手動 --resume' if args.resume else '自動再開'
        log(f'RESUME kind={kind} path={resume_path} step={start_step} '
            f'best_acc={best_test_acc:.4f} run_dir={run_dir}')
        print(f'  ステップ {start_step} から再開 (best_acc={best_test_acc:.4f})')
        if args.resume and resume_run_dir is None:
            print('  注意: --resume は新しい run_dir へ書き出すため log.txt が分断されます')
    else:
        log(f'RESUME kind=none step=0 run_dir={run_dir}')

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'seqmixer_{args.task}_{args.config}_{args.cross_attention}'
                f'_v{config.vocab_size}_ni{args.num_iterations}')
    job_type = args.job_type
    if job_type is None:
        job_type = 'train' if args.task == 'wmt14' else 'smoke'
    wandb_config = {**vars(args), **asdict(config)}
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=(['logic_seq_mixer', args.task, args.config,
                   f'arm:{args.cross_attention}']
                  + normalize_tags(model='logic_seq_mixer', dataset=args.task,
                                   seed=args.seed, series='TSMIX-1')),
            group='tsmix-1',
            job_type=job_type,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )
        if use_wandb:
            # ゲート数は1回だけログする
            log_metrics(count_logic_gates(model), step=start_step)

    remaining_iters = args.num_iterations - start_step

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()
    # 訓練区間のピークメモリ計測を開始（CUDA以外ではno-op）
    reset_peak_memory(args.device)
    # BLEU は soft(relaxed) / hard(collapsed) の2本を測る。
    # RDDLGN論文の relaxed 5.00 / Collapsed 4.39（gap -0.61）と直接比較するため。
    bleu_soft = None
    bleu_hard = None

    # 軽量ログ（train loss / 進捗 / スループット）のペーサ。
    # 重い評価(-ef, --bleu-every)とは完全に独立させ、壁時計で
    # LOSS_LOG_INTERVAL_SEC 以内に必ず1点出るようにする（SKILL.md §11）。
    pacer = LossLogPacer(every_steps=args.log_every_steps)
    if pacer.every_steps is not None:
        print(pacer.describe())

    pbar = tqdm(
        enumerate(load_n(train_loader, remaining_iters), start=start_step),
        desc='train',
        total=args.num_iterations,
        initial=start_step,
    )

    for step, (src, tgt) in pbar:
        src = src.to(args.device)
        tgt = tgt.to(args.device)

        # BOS/EOSを付加
        src_batch, tgt_full = prepare_batch(src, tgt)

        total_loss, main_loss, emb_loss = train_step(
            model, src_batch, tgt_full, optimizer, step,
            label_smoothing=args.label_smoothing,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{total_loss:.4f}',
            main=f'{main_loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            eta=eta_str,
        )

        # ------------------------------------------------------------
        # 軽量ログ（SKILL.md §11: train lossは壁時計で最低5分に1回）
        # 重い評価とは独立に、実測s/itから決めたstep間隔で loss と進捗を出す。
        # ------------------------------------------------------------
        done_iters = step + 1 - start_step
        should_light_log = pacer.should_log(
            done_iters, elapsed, is_last=step == args.num_iterations - 1)
        if pacer.resolved_now:
            print(f'\n{pacer.describe()}', flush=True)
            log(pacer.describe())

        light_metrics = None
        if use_wandb and should_light_log:
            light_metrics = {
                'iteration': step + 1,
                'loss/total': total_loss,
                'loss/main': main_loss,
                'loss/emb_reg': emb_loss,
                'loss/emb_reg_weight': get_emb_reg_weight(step),
                'lr': optimizer.param_groups[0]['lr'],
                **progress_metrics(done_iters, step + 1, args.num_iterations, elapsed),
            }

        # 評価
        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            # 直前の訓練区間のピークを確定させ、eval区間を別枠で計測する
            # （OOMは hard eval で起きるため、訓練/評価を分けて可視化する）
            train_peak = peak_memory_gb(args.device)
            reset_peak_memory(args.device)
            eval_start = time.time()

            test_acc_soft = evaluate(model, test_loader, args.device, use_soft=True,
                                     chunk_size=args.eval_chunk_size)
            test_acc_hard = evaluate(model, test_loader, args.device, use_soft=False,
                                     chunk_size=args.eval_chunk_size)

            # 精度が上がらなくなったらLR減衰（RDDLGNと同一: 1.0 - acc_soft を渡す）
            scheduler.step(1.0 - test_acc_soft)

            is_best = test_acc_soft > best_test_acc
            if is_best:
                best_test_acc = test_acc_soft

            save_checkpoint(latest_path, model, optimizer, scheduler,
                            step + 1, best_test_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler,
                                step + 1, best_test_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            iter_per_sec = (step + 1 - start_step) / max(elapsed, 1e-9)

            emb_w = get_emb_reg_weight(step)
            current_lr = optimizer.param_groups[0]['lr']

            # BLEU（wmt14のみ。既定は最終stepのみ=RDDLGNと同挙動）
            # soft(relaxed) と hard(collapsed) の両方を測り、その差
            # collapse_gap = soft - hard を記録する（RDDLGNは -0.61）。
            bleu_str = ''
            collapse_gap = None
            is_final = step == args.num_iterations - 1
            want_bleu = is_final or (
                args.bleu_every > 0 and (step + 1) % args.bleu_every == 0
            )
            if args.task == 'wmt14' and want_bleu:
                bleu_soft = compute_bleu(
                    model, test_loader, wmt_tokenizer, args.device, use_soft=True,
                    chunk_size=args.eval_chunk_size)
                bleu_hard = compute_bleu(
                    model, test_loader, wmt_tokenizer, args.device, use_soft=False,
                    chunk_size=args.eval_chunk_size)
                if bleu_soft is not None and bleu_hard is not None:
                    collapse_gap = bleu_soft - bleu_hard
                    bleu_str = (f'  BLEU(soft)={bleu_soft:.2f} '
                                f'BLEU(hard)={bleu_hard:.2f} gap={collapse_gap:+.2f}')

            # 評価の所要時間とピークメモリ（本走の見積もり・OOM可視化用）
            eval_sec = time.time() - eval_start
            eval_peak = peak_memory_gb(args.device)
            mem_str = f'  eval={eval_sec:.1f}s'
            if train_peak is not None and eval_peak is not None:
                mem_str += (f' mem(train)={train_peak[0]:.2f}/{train_peak[1]:.2f}GB'
                            f' mem(eval)={eval_peak[0]:.2f}/{eval_peak[1]:.2f}GB'
                            f' [allocated/reserved]')
            # 次の訓練区間のためにリセット
            reset_peak_memory(args.device)

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f}(w={emb_w:.4f})  '
                f'acc(soft)={test_acc_soft:.4f} acc(hard)={test_acc_hard:.4f}'
                f'{bleu_str}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  lr={current_lr:.5f}'
                f'  [{elapsed_str}<{remain_str}, {iter_per_sec:.1f}it/s, 終了予定 {eta_str}]'
                f'{mem_str}',
                flush=True,
            )

            log(f'ITER {step+1}/{args.num_iterations} '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f} '
                f'emb_w={emb_w:.4f} acc_soft={test_acc_soft:.4f} '
                f'acc_hard={test_acc_hard:.4f}{bleu_str} lr={current_lr:.5f} '
                f'best={str(is_best).lower()} elapsed={elapsed_str} eta={remain_str}'
                f'{mem_str}')

            if use_wandb:
                # 同じstepへ2回 log すると wandb のstep管理が読みにくくなるので、
                # 軽量ログが同じstepに当たったときは1レコードに合流させる。
                wandb_metrics = {
                    **(light_metrics or {}),
                    'iteration': step + 1,
                    'eval/seconds': eval_sec,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'loss/emb_reg': emb_loss,
                    'loss/emb_reg_weight': emb_w,
                    'acc/soft': test_acc_soft,
                    'acc/hard': test_acc_hard,
                    'best_test_acc': best_test_acc,
                    'lr': current_lr,
                }
                if bleu_str:
                    wandb_metrics['bleu/soft'] = bleu_soft
                    wandb_metrics['bleu/hard'] = bleu_hard
                    wandb_metrics['bleu/collapse_gap'] = collapse_gap
                    # 後方互換: RDDLGNのキー体系に合わせ `bleu` は hard と同値
                    wandb_metrics['bleu'] = bleu_hard
                if train_peak is not None and eval_peak is not None:
                    # 訓練/評価それぞれのピーク。`mem/peak_*` は両者の最大（OOM監視用）
                    wandb_metrics['mem/train_peak_allocated_gb'] = train_peak[0]
                    wandb_metrics['mem/train_peak_reserved_gb'] = train_peak[1]
                    wandb_metrics['mem/eval_peak_allocated_gb'] = eval_peak[0]
                    wandb_metrics['mem/eval_peak_reserved_gb'] = eval_peak[1]
                    wandb_metrics['mem/peak_allocated_gb'] = max(train_peak[0], eval_peak[0])
                    wandb_metrics['mem/peak_reserved_gb'] = max(train_peak[1], eval_peak[1])
                log_metrics(wandb_metrics, step=step + 1)
                light_metrics = None  # 合流済み。二重ログを避ける

        # 評価が無いstepの軽量ログ。stepは評価ログと同じ `step + 1` を使い、
        # ループ内で単調増加するので wandb のstep巻き戻し警告は起きない。
        if light_metrics is not None:
            log_metrics(light_metrics, step=step + 1)

    # 最終結果
    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度(soft): {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_acc={best_test_acc:.4f} bleu_soft={bleu_soft} '
        f'bleu_hard={bleu_hard} total_time={total_time}')
    close_log()

    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={
                'args': vars(args),
                'config': asdict(config),
                'best_test_acc': best_test_acc,
                'bleu_soft': bleu_soft,
                'bleu_hard': bleu_hard,
                'bleu_collapse_gap': (
                    None if (bleu_soft is None or bleu_hard is None)
                    else bleu_soft - bleu_hard
                ),
                # 後方互換: `bleu` は hard(collapsed) と同値
                'bleu': bleu_hard,
            },
            model=model,
        )
        wandb_finish()

    # デコード例
    print('\n=== デコード例 ===')
    model.eval()
    sample_src, sample_tgt = next(iter(test_loader))
    sample_src = sample_src[:4].to(args.device)
    sample_tgt = sample_tgt[:4].to(args.device)

    _, sample_tgt_full = prepare_batch(sample_src, sample_tgt)
    with torch.no_grad():
        sample_logits, _ = model(sample_src, sample_tgt_full[:, :-1])
        sample_preds = sample_logits.argmax(dim=-1)

    for i in range(min(4, sample_src.shape[0])):
        src_ids = sample_src[i].tolist()
        tgt_ids = sample_tgt[i].tolist()
        pred_ids = sample_preds[i, :len(tgt_ids)].tolist()

        if wmt_tokenizer is not None:
            print(f'  src:  {wmt_tokenizer.decode(src_ids)}')
            print(f'  tgt:  {wmt_tokenizer.decode(tgt_ids)}')
            print(f'  pred: {wmt_tokenizer.decode(pred_ids)}')
        else:
            match = '✓' if pred_ids == tgt_ids else '✗'
            print(f'  src:  {src_ids}')
            print(f'  tgt:  {tgt_ids}')
            print(f'  pred: {pred_ids}  {match}')
        print()


if __name__ == '__main__':
    main()
