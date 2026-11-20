"""
RDDLGN 訓練スクリプト（seq2seq）

使用例:
    # copyタスクでデバッグ（tiny構成）
    uv run python experiments/train_rddlgn.py --task copy --config tiny -ni 5000 -ef 500

    # reverseタスク
    uv run python experiments/train_rddlgn.py --task reverse --config tiny -ni 10000 -ef 1000

    # sortタスク
    uv run python experiments/train_rddlgn.py --task sort --config tiny -ni 10000 -ef 1000

    # カスタム構成
    uv run python experiments/train_rddlgn.py --task copy --vocab-size 50 --d-emb 16 \\
        --seq-len 4 --n-sizes 32,32 --k-sizes 48,32 --l-sizes 32,32 \\
        --p-sizes 48,32 --m-sizes 64,64,150 --tau 1.0 -ni 5000

    # 論文構成（スケールダウン）
    uv run python experiments/train_rddlgn.py --task copy --config paper --scale 0.01 -ni 1000

    # WMT'14翻訳タスク（論文と同じ、130Kステップ）
    uv run python experiments/train_rddlgn.py --task wmt14 --config paper -ni 130000 -ef 5000

    # WMT'14（スケールダウン）
    uv run python experiments/train_rddlgn.py --task wmt14 --config paper --scale 0.1 -ni 10000 -ef 1000
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
from data import SPECIAL_TOKENS, TASK_DATASETS, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD
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

from griffith_logic import RDDLGN, RDDLGNConfig

# .envから環境変数を読み込み
load_dotenv()

# 自動再開で「同じ実験か」を判定する args のキー（metadata['config'] の一致に加えて確認）
AUTO_RESUME_ARGS_KEYS = (
    'task', 'config', 'seed', 'num_iterations',
    'vocab_size', 'seq_len', 'num_samples', 'num_test_samples',
    'lr', 'weight_decay', 'batch_size', 'label_smoothing',
    'patience', 'lr_factor', 'n_inputs', 'parameterization', 'tau', 'selection',
)



# ============================================================
# 訓練・評価
# ============================================================

def get_emb_reg_weight(step: int, ramp_start: int = 1000, ramp_end: int = 100000,
                       max_weight: float = 0.1) -> float:
    """二値化正則化の重みを線形ランプで計算。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


def train_step(model, src, tgt, optimizer, step, label_smoothing=0.1,
               tau_sched=None, spectral_reg=0.0):
    """1ステップの訓練。teacher forcing。"""
    model.train()
    vocab_size = model.config.vocab_size

    # teacher forcing: tgt[:-1]が入力、tgt[1:]がラベル
    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    logits, emb_loss = model(src, tgt_input)

    # メイン損失
    main_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt_label.reshape(-1),
        label_smoothing=label_smoothing,
        ignore_index=TOKEN_PAD,
    )

    # 二値化正則化
    emb_weight = get_emb_reg_weight(step)
    total_loss = main_loss + emb_weight * emb_loss

    # Spectral正則化(DWN §3.4)
    if spectral_reg > 0:
        from griffith_logic.regularization import spectral_penalty
        total_loss = total_loss + spectral_reg * spectral_penalty(model)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if tau_sched is not None:
        tau_sched.step()

    return total_loss.item(), main_loss.item(), emb_loss.item()


def _iter_chunks(n: int, chunk_size: int):
    """[0, n) を chunk_size 単位の (start, end) に分割する。0以下なら分割しない。"""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        yield 0, n
        return
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


@torch.no_grad()
def evaluate(model, loader, device, use_soft=True, chunk_size=0):
    """トークンレベルの精度を計算。

    chunk_size は評価forwardの分割サイズ（純粋なメモリ管理で、閾値や評価条件を
    緩める変更ではない）。各サンプルのforwardは独立で、精度はPADマスク付きの
    一致数/総数の比なので、分割の有無で結果は厳密に同一になる。
    hard eval(use_soft=False)は `LutLayer` の推論時パスにCUDAカーネル分岐が無く
    `core/basis.py` の `_compute_basis_butterfly` が `(B×T, out_dim, 2^N)` を素で
    確保するため、ここを刻まないとOOMする。
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

            # PAD/BOS/EOSを含む特殊トークンは無視せず、全ラベルで精度計算
            # ただしPADは無視
            labels = tgt_label[start:end]
            mask = labels != TOKEN_PAD
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def greedy_decode(model, src, max_len, bos_token=TOKEN_BOS):
    """貪欲デコーディングでシーケンスを生成。

    注意: 以前は `model.eval()` をハードコードしていたため hard(collapsed) BLEU
    しか測れなかった。soft(relaxed)/hard の両方を測るため、soft/hard の切替は
    呼び出し元（`compute_bleu`）の責務とする。RDDLGN に dropout / BatchNorm は
    無いので train モードでの評価も安全。
    """
    batch = src.shape[0]
    device = src.device

    # エンコード
    src_emb = model.src_embedding(src)
    context = model.encoder(src_emb)

    # デコード（自己回帰）
    generated = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_emb = model.tgt_embedding(generated)
        logits = model.decoder(tgt_emb, context)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated[:, 1:]  # BOSを除く


@torch.no_grad()
def compute_bleu(model, loader, tokenizer, device, max_samples=1000, use_soft=False,
                 chunk_size=0):
    """greedy decodeでBLEUスコアを計算。

    - `sacrebleu.corpus_bleu(hyps, [refs]).score`（tokenize引数を指定しない=既定13a）
    - `max_len = tgt.shape[1]`
    - `max_samples=1000`
    - `ref.strip()` が空のサンプルは除外
    この4条件は公平比較の生命線なので変更しない。

    Args:
        tokenizer: WordTokenizer（decode用）。Noneなら計算スキップ。
        use_soft: True で soft(relaxed) BLEU、False で hard(collapsed) BLEU。
            `evaluate(..., use_soft=...)` と同じ流儀で `model.train(mode=use_soft)`
            により切り替える。
        chunk_size: greedy decode の forward 分割サイズ（メモリ管理のみ）。
            各サンプルのデコードは独立なので、分割しても hyp/ref の集合・順序・
            対象サンプル数(max_samples)は変わらず BLEU は厳密に同一になる。
    """
    if tokenizer is None:
        # 合成タスク等（tokenizerが無い）は従来どおり静かにスキップ
        return None

    # tokenizerがあるのにsacrebleuが無いのは構成ミス。握り潰さず落とす
    # （長時間走ってからBLEUが取れていないと気づく事故を防ぐ）。
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

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


_BYTES_PER_GB = 1024 ** 3


def reset_peak_memory(device: str) -> None:
    """ピークメモリ統計をリセットする（CUDA以外では何もしない）。"""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: str) -> tuple[float, float] | None:
    """(allocated, reserved) のピークをGB単位で返す。CUDA以外はNone。"""
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return (
        torch.cuda.max_memory_allocated() / _BYTES_PER_GB,
        torch.cuda.max_memory_reserved() / _BYTES_PER_GB,
    )


def require_bleu_deps(task: str) -> None:
    """wmt14 runでは起動時に sacrebleu の存在を検証し、無ければ即座に落とす。

    以前は `compute_bleu` 内で ImportError を握り潰して None を返していたため、
    BLEUが無言でスキップされ「長時間のGPU本走が終わってから成果物が無いと気づく」
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
            '  Modal   : scripts/modal_app.py の .uv_sync(extras=[...]) に "wmt" が'
            '含まれているか確認してください'
        ) from e


# ============================================================
# ログ
# ============================================================

def setup_logger(task, log_dir='logs'):
    """ログファイルを作成する（run_dir 側にも log.txt を出す）。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'rddlgn_{task}_{timestamp}.log')
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
# メイン
# ============================================================

def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。例: '64,64' -> [64, 64]"""
    return [int(x.strip()) for x in s.split(',')]


def build_config(args) -> RDDLGNConfig:
    """CLIArgumentsからRDDLGNConfigを構築。"""
    if args.config == 'paper':
        config = RDDLGNConfig.from_paper(scale=args.scale)
    elif args.config == 'tiny':
        config = RDDLGNConfig.tiny()
    else:
        # カスタム構成
        config = RDDLGNConfig(
            vocab_size=args.vocab_size,
            d_emb=args.d_emb,
            seq_len=args.seq_len,
            n_layer_sizes=parse_int_list(args.n_sizes),
            k_layer_sizes=parse_int_list(args.k_sizes),
            l_layer_sizes=parse_int_list(args.l_sizes),
            p_layer_sizes=parse_int_list(args.p_sizes),
            m_layer_sizes=parse_int_list(args.m_sizes),
            n_inputs=args.n_inputs,
            parameterization=args.parameterization,
            tau=args.tau,
        )

    # CLIで上書き可能な項目（プリセット使用時でも明示指定すれば上書き）
    if args.vocab_size is not None and args.config != 'custom':
        config.vocab_size = args.vocab_size
    if args.seq_len is not None and args.config != 'custom':
        config.seq_len = args.seq_len
    if args.n_inputs != 2:
        config.n_inputs = args.n_inputs
    if args.parameterization != 'softmax':
        config.parameterization = args.parameterization
    # WARP設定
    config.tau_warp = args.tau_warp
    config.gumbel = not args.no_gumbel
    # selection(soft/hard_st/gumbel_st)
    config.selection = args.selection

    return config


def prepare_batch(src, tgt, bos_token=TOKEN_BOS, eos_token=TOKEN_EOS):
    """ターゲットにBOS/EOSを付加する。"""
    batch = src.shape[0]
    device = src.device
    bos = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)
    eos = torch.full((batch, 1), eos_token, dtype=torch.long, device=device)
    # tgt_full = [BOS, tgt_tokens, EOS]
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


def interactive_rddlgn(model, config, task, tokenizer=None):
    """対話的推論デモ。"""
    model.eval()
    device = next(model.parameters()).device
    seq_len = config.seq_len

    if task == 'wmt14':
        print('\n=== RDDLGN 対話デモ (wmt14 En→De) ===')
        print('英語テキストを入力してください（"q"で終了）:')
        while True:
            try:
                text = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text.lower() == 'q':
                break
            ids = tokenizer.encode(text, seq_len)
            # seq_lenまでパディング
            ids += [TOKEN_PAD] * (seq_len - len(ids))
            src = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                preds = greedy_decode(model, src, max_len=seq_len)
            print(f'翻訳: {tokenizer.decode(preds[0].tolist())}')
    else:
        print(f'\n=== RDDLGN 対話デモ ({task}) ===')
        print(f'トークンID列を入力（カンマ区切り、範囲: {SPECIAL_TOKENS}〜{config.vocab_size - 1}、"q"で終了）:')
        while True:
            try:
                text = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text.lower() == 'q':
                break
            try:
                ids = [int(x.strip()) for x in text.split(',')]
            except ValueError:
                print('エラー: カンマ区切りの整数を入力してください')
                continue
            ids = ids[:seq_len]
            display_ids = list(ids)
            ids += [TOKEN_PAD] * (seq_len - len(ids))
            src = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                preds = greedy_decode(model, src, max_len=seq_len)
            pred_ids = [x for x in preds[0].tolist() if x not in (TOKEN_PAD, TOKEN_BOS, TOKEN_EOS)]
            print(f'入力:  {display_ids}')
            print(f'出力:  {pred_ids}')


def main():
    # リダイレクト時(`> file 2>&1`)の標準出力はブロックバッファリングされ、
    # 長時間runの [iter ...] 行が終了時まで書き出されず進捗監視ができない。
    # 行バッファリングへ切り替えて逐次フラッシュさせる(Modal本走で必須)。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description='RDDLGN 訓練スクリプト')

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
    parser.add_argument('--scale', type=float, default=1.0,
                        help='paper構成のスケール (default: 1.0)')

    # カスタム構成パラメータ
    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--d-emb', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--n-sizes', type=str, default='64,64')
    parser.add_argument('--k-sizes', type=str, default='96,64')
    parser.add_argument('--l-sizes', type=str, default='64,64')
    parser.add_argument('--p-sizes', type=str, default='96,64')
    parser.add_argument('--m-sizes', type=str, default='128,128,300')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'])
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau-warp', type=float, default=1.0,
                        help='WARP温度パラメータ（warp時のみ）')
    parser.add_argument('--no-gumbel', action='store_true',
                        help='WARPのGumbel smoothingを無効化')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式 (default: soft)')
    parser.add_argument('--tau-anneal', action='store_true',
                        help='TauAnnealScheduler(τ_b幾何減衰+ノイズ減衰)を有効化')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')

    # 訓練設定
    parser.add_argument('--lr', type=float, default=0.05,
                        help='学習率 (default: 0.05, 論文値)')
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=130000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--log-every-steps', type=int, default=None,
                        help='train lossをwandbへ記録するstep間隔。未指定なら実測s/itから'
                             f'{int(LOSS_LOG_INTERVAL_SEC)}秒以内になる値を自動算出する'
                             '（重い評価の間隔 -ef とは独立）')
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
                        help='BLEUを測る間隔(step)。0=最終stepのみ (default: 0)')

    # チェックポイント・推論
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='自動再開を無効化して必ず step 0 から開始する（既定は自動再開ON）')
    parser.add_argument('--interactive', action='store_true',
                        help='対話推論モード（--resume必須）')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--data-dir', type=str, default='./data-wmt14',
                        help='WMTデータ保存先 (default: ./data-wmt14)')
    parser.add_argument('--wandb-group', type=str, default='rddlgn',
                        help='wandb group (default: rddlgn)')
    parser.add_argument('--wandb-tags', type=str, default=None,
                        help='wandbへ追加するタグ（カンマ区切り）')
    parser.add_argument('--wandb-series', type=str, default=None,
                        help='taxonomyタグ series:<値>（例:）')
    parser.add_argument('--job-type', type=str, default='train',
                        choices=['train', 'ablation', 'sweep', 'eval', 'smoke'],
                        help='wandb job_type (default: train)')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを使わない')

    # --notes / --hypothesis
    add_experiment_args(parser)

    args = parser.parse_args()

    if args.interactive and not args.resume:
        parser.error('--interactive には --resume が必要です')

    # デバイス自動検出
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
    config = build_config(args)

    # vocab_size未指定時はカスタム構成のみデフォルト適用（プリセットは上書きしない）
    if args.vocab_size is None and args.config == 'custom':
        config.vocab_size = 100
    elif args.vocab_size is not None:
        config.vocab_size = args.vocab_size

    # seq_len未指定時はconfig値を使用
    if args.seq_len is not None:
        config.seq_len = args.seq_len

    # M最終層がvocab_sizeで割り切れるか確認・調整
    m_last = config.m_layer_sizes[-1]
    if m_last % config.vocab_size != 0:
        group_factor = max(1, m_last // config.vocab_size)
        config.m_layer_sizes[-1] = config.vocab_size * max(group_factor, 1)
        print(f'M最終層を {m_last} → {config.m_layer_sizes[-1]} に調整 '
              f'(vocab_size={config.vocab_size} の倍数)')

    print('\n=== RDDLGN 訓練 ===')
    print(f'タスク: {args.task}')
    print(f'語彙サイズ: {config.vocab_size}')
    print(f'埋め込み次元: {config.d_emb}')
    print(f'系列長: {config.seq_len}')
    print(f'N-layers: {config.n_layer_sizes}')
    print(f'K-layers: {config.k_layer_sizes}')
    print(f'L-layers: {config.l_layer_sizes}')
    print(f'P-layers: {config.p_layer_sizes}')
    print(f'M-layers: {config.m_layer_sizes}')

    # データセット
    wmt_tokenizer = None
    if args.task == 'wmt14':
        from data.wmt14 import download_and_tokenize
        train_set, test_set, wmt_tokenizer = download_and_tokenize(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            data_dir=args.data_dir,
        )
        # トークナイザーの実際の語彙サイズでconfigを更新
        config.vocab_size = wmt_tokenizer.vocab_size
    else:
        seq_len = config.seq_len
        dataset_cls = TASK_DATASETS[args.task]
        train_set = dataset_cls(config.vocab_size, seq_len, args.num_samples)
        test_set = dataset_cls(config.vocab_size, seq_len, args.num_test_samples)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    print(f'訓練サンプル: {len(train_set):,}')
    print(f'テストサンプル: {len(test_set):,}')

    # モデル構築
    model = RDDLGN(config, device=args.device).to(args.device)
    counts = model.count_parameters()
    total_gates = sum(
        m.num_neurons for m in model.modules()
        if hasattr(m, 'num_neurons')
    )
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'総ロジックゲート数: {total_gates:,} (LUT{config.n_inputs})')

    # Optimizer / Scheduler
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
    tau_sched = None
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        tau_sched = TauAnnealScheduler(
            model, tau_0=args.tau_b0, tau_min=args.tau_b_min,
            total_steps=args.num_iterations,
            anneal_noise=(args.selection == 'gumbel_st'),
        )

    # ログ
    log, attach_log, close_log = setup_logger(args.task, args.log_dir)

    # チェックポイント
    ckpt_prefix = f'rddlgn_{args.task}_{args.config}'

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
    # Modal の自動リトライでは setup_checkpoint_dir が新しい run_dir を作るため、
    # run_dir 内だけを見ても前試行の latest.pt は見つからない（step 0 へ巻き戻る）。
    resume_path = args.resume
    resume_run_dir = None
    if resume_path is None and not args.no_auto_resume and not args.interactive:
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
    log(f'START task={args.task} config={args.config} vocab={config.vocab_size} '
        f'd_emb={config.d_emb} seq_len={config.seq_len} '
        f'lr={args.lr} bs={args.batch_size} ni={args.num_iterations} '
        f'device={args.device} params={counts["total"]:,} gates={total_gates:,}')

    # チェックポイントから復元
    # 注意: 以前は訓練ループ開始直前に best_test_acc=0.0 を再代入しており、
    # resume で復元した best が捨てられていた（既知バグ）。ここで確定させ、
    # 訓練ループ開始前に再初期化しない。
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

        if args.interactive:
            interactive_rddlgn(model, config, args.task, tokenizer=wmt_tokenizer)
            return
    else:
        log(f'RESUME kind=none step=0 run_dir={run_dir}')

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'rddlgn_{args.task}_{args.config}'
                f'_v{config.vocab_size}_d{config.d_emb}_ni{args.num_iterations}')
    wandb_config = {**vars(args), **asdict(config)}
    extra_tags = [t.strip() for t in (args.wandb_tags or '').split(',') if t.strip()]
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=(['rddlgn', args.task, args.config] + extra_tags
                  + normalize_tags(model='rddlgn', dataset=args.task,
                                   seed=args.seed, series=args.wandb_series)),
            group=args.wandb_group,
            job_type=args.job_type,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )
        if use_wandb:
            # ゲート数は1回だけログする
            log_metrics(count_logic_gates(model), step=start_step)

    remaining_iters = args.num_iterations - start_step

    # 訓練ループ
    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()
    # 訓練区間のピークメモリ計測を開始（CUDA以外ではno-op）
    reset_peak_memory(args.device)
    # BLEU は soft(relaxed) / hard(collapsed) の2本を測る
    bleu_soft = None
    bleu_hard = None

    # 軽量ログ（train loss / 進捗 / スループット）のペーサ。
    # 重い評価(-ef)とは独立に、壁時計で5分以内に必ず1点出す（SKILL.md §11）。
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
            tau_sched=tau_sched, spectral_reg=args.spectral_reg,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{total_loss:.4f}',
            main=f'{main_loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            eta=eta_str,
        )

        # 軽量ログ（SKILL.md §11: train lossは壁時計で最低5分に1回）。
        # 評価とは独立に、実測s/itから決めたstep間隔で loss と進捗を出す。
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

            # validation精度の逆数をloss代わりに監視（精度が上がらなくなったらLR減衰）
            scheduler.step(1.0 - test_acc_soft)

            is_best = test_acc_soft > best_test_acc
            if is_best:
                best_test_acc = test_acc_soft

            # チェックポイント保存
            save_checkpoint(latest_path, model, optimizer, scheduler,
                            step + 1, best_test_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler,
                                step + 1, best_test_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1)
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            iter_per_sec = (step + 1) / elapsed

            emb_w = get_emb_reg_weight(step)
            current_lr = optimizer.param_groups[0]['lr']

            # BLEU（wmt14タスク時のみ。既定は最終stepのみ）
            # soft(relaxed) と hard(collapsed) の両方を測り、その差
            # collapse_gap = soft - hard を記録する。
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
                # 同じstepの軽量ログは1レコードに合流させる（二重ログを避ける）
                wandb_metrics = {
                    **(light_metrics or {}),
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'loss/emb_reg': emb_loss,
                    'loss/emb_reg_weight': emb_w,
                    'acc/soft': test_acc_soft,
                    'acc/hard': test_acc_hard,
                    'best_test_acc': best_test_acc,
                    'lr': current_lr,
                    'eval/seconds': eval_sec,
                }
                if bleu_str:
                    wandb_metrics['bleu/soft'] = bleu_soft
                    wandb_metrics['bleu/hard'] = bleu_hard
                    wandb_metrics['bleu/collapse_gap'] = collapse_gap
                    # 後方互換: 従来キー `bleu` は hard(collapsed) と同値
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
                light_metrics = None  # 合流済み

        # 評価が無いstepの軽量ログ。stepは評価ログと同じ `step + 1` で単調増加する
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

    # デコード例を表示
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
            # WMTタスク: テキストで表示
            print(f'  src:  {wmt_tokenizer.decode(src_ids)}')
            print(f'  tgt:  {wmt_tokenizer.decode(tgt_ids)}')
            print(f'  pred: {wmt_tokenizer.decode(pred_ids)}')
        else:
            # 合成タスク: ID列で表示
            match = '✓' if pred_ids == tgt_ids else '✗'
            print(f'  src:  {src_ids}')
            print(f'  tgt:  {tgt_ids}')
            print(f'  pred: {pred_ids}  {match}')
        print()


if __name__ == '__main__':
    main()
