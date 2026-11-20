"""
FPSeqMixer / FP GRU 訓練スクリプト（/ seq2seq の float 対照）

`train_logic_seq_mixer.py`（LogicSeqMixer）の **float(通常NN)対照** を、
**評価プロトコルを1つも変えずに** 学習・評価する。目的は次の切り分け:

    (a) FP BLEU − 論理版 soft BLEU  … 構造の表現力差（連続 vs 緩和論理）
    (b) FP BLEU − 論理版 hard BLEU  … 回路化の総コスト（構造差 + 離散化損）
    (c) FP GRU BLEU                 … 論文の GRU 5.41 を「我々の評価系」で
                                       測り直す換算基準（評価系交絡の定量化）

論理版から**変えていない**もの（公平比較の生命線。変更禁止）:
    - WMT'14 の取得経路 `download_and_tokenize(vocab_size, seq_len, data_dir)`
      と共有 `WordTokenizer`（PAD0/BOS1/EOS2/UNK3）
    - teacher forcing `[BOS, tgt, EOS]`、入力 `[:-1]` / ラベル `[1:]`、
      `label_smoothing=0.1`、`ignore_index=TOKEN_PAD`
    - 精度: PADを無視したトークン一致率
    - BLEU: `sacrebleu.corpus_bleu(hyps, [refs]).score`（tokenize引数なし=既定13a）、
      `max_len = tgt.shape[1]`、`max_samples=1000`、`ref.strip()` が空なら除外、
      greedy decode は BOSから開始／毎ステップ prefix 全体を再デコード／
      `logits[:, -1].argmax`／返り値はBOSを除く
    - optimizer=AdamW(betas=(0.9,0.999), eps=1e-8, weight_decay=0.001)、
      scheduler=ReduceLROnPlateau(mode='min', factor=0.8, patience=10000)、
      batch_size=64

論理版から**変えた**もの（FPで学習を成立させるための最小限。レポートに明記する）:
    - **学習率の既定を 0.05 → 1e-3 にする**。0.05 は論理版 LutLayer の5パラメータ化
      （実効勾配が小さくAdamWで大きなlrが要る）に合わせた値で、通常の
      `nn.Linear`+GELU+LayerNorm では発散する。既定値の根拠は lr 掃引スモークの実測。
    - **soft/hard の区別が無い**ので精度もBLEUも1本だけ（論理版は relaxed/collapsed
      の2本）。`bleu/soft` `bleu/hard` は同一値を書き、`bleu` にも同値を入れる
      （論理版・RDDLGN とキー体系を揃えて比較クエリを共通化するため）。
    - 二値化正則化 `emb_reg_loss` は論理版固有なので存在しない。

使用例:
    # 極小スモーク（copy / tiny / MPS自動検出）
    uv run python experiments/train_fp_seq_mixer.py \\
        --task copy --config tiny -ni 20 -ef 10

    # lr 掃引スモーク（WMT'14キャッシュ vocab1600 を流用）
    uv run python experiments/train_fp_seq_mixer.py \\
        --task wmt14 --config paper --vocab-size 1600 --seq-len 16 \\
        --lr 1e-3 -ni 300 -ef 300 --no-wandb

    # WMT'14 本走（アーム(i) = gather相当、cross-attention 無し）
    uv run python experiments/train_fp_seq_mixer.py \\
        --task wmt14 --config paper --vocab-size 16000 --seq-len 16 \\
        --cross-attention none -ni 40000 -ef 2000 --bleu-every 10000 --seed 42

    # FP GRU 対照（パラメータ数を FP Mixer に揃えた nn.GRU seq2seq）
    uv run python experiments/train_fp_seq_mixer.py --model gru \\
        --task wmt14 --vocab-size 16000 --seq-len 16 \\
        -ni 40000 -ef 2000 --bleu-every 10000 --seed 42

    # Modal 投入（--background は必ず --detach とセット）
    uv run modal run --detach scripts/modal_app.py --background \\
        --gpu A10G --timeout 86400 \\
        --script train_fp_seq_mixer.py \\
        --args "--task wmt14 --vocab-size 16000 --seq-len 16 -ni 40000 -ef 2000"
"""

import argparse
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn as nn
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
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
    progress_metrics,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.models import FPSeqMixer, FPSeqMixerConfig

# .envから環境変数を読み込み
load_dotenv()

# FPSeqMixerConfig.from_paper() の総パラメータ実測値（GRUを合わせる目標）
FP_MIXER_PAPER_PARAMS = 40_660_648

# 自動再開で「同じ実験か」を判定する args のキー（metadata['config'] の一致に加えて確認）
AUTO_RESUME_ARGS_KEYS = (
    'model', 'task', 'config', 'cross_attention', 'seed', 'num_iterations',
    'vocab_size', 'seq_len', 'num_samples', 'num_test_samples',
    'lr', 'weight_decay', 'batch_size', 'label_smoothing',
    'patience', 'lr_factor',
)


# ============================================================
# FP GRU 対照モデル
# ============================================================

@dataclass
class FPGRUConfig:
    """PyTorch標準 `nn.GRU` による encoder-decoder seq2seq の設定。

    パラメータ数を FP Mixer(40,660,648) と同程度へ揃えるのが唯一の設計制約で、
    埋め込み幅(d_emb=1024)・語彙・系列長は FP Mixer と共有し、`hidden` だけを
    :func:`solve_gru_hidden` で解いて合わせる。
    """

    vocab_size: int = 16000
    d_emb: int = 1024
    seq_len: int = 16
    hidden: int = 670
    n_layers: int = 2
    dropout: float = 0.0
    shared_vocab: bool = True
    # FPSeqMixerConfig と同名フィールドを持たせて呼び出し側を分岐なしにする
    cross_attention: str = 'none'

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f'n_layersは1以上: {self.n_layers}')
        if self.hidden < 1:
            raise ValueError(f'hiddenは1以上: {self.hidden}')
        if self.vocab_size < 1:
            raise ValueError(f'vocab_sizeは1以上: {self.vocab_size}')


def estimate_gru_parameters(config: FPGRUConfig) -> int:
    """FPGRUTranslator の総パラメータ数を実体化せず解析式で見積もる。

    `nn.GRU` の1層は 3H×in + 3H×H + 2×3H（bias_ih / bias_hh）である。
    decoder は入力に文脈ベクトル(H)を連結するので第1層の in が d_emb+H になる。
    """
    c = config
    h = c.hidden
    total = c.vocab_size * c.d_emb * (1 if c.shared_vocab else 2)

    def gru(in_dim: int) -> int:
        first = 3 * h * in_dim + 3 * h * h + 6 * h
        rest = (c.n_layers - 1) * (6 * h * h + 6 * h)
        return first + rest

    total += gru(c.d_emb)          # encoder
    total += gru(c.d_emb + h)      # decoder（入力に文脈を連結）
    total += h * c.vocab_size + c.vocab_size   # 出力ヘッド
    return total


def solve_gru_hidden(target: int, base: FPGRUConfig) -> int:
    """総パラメータ数が `target` に最も近くなる hidden を返す。

    :func:`estimate_gru_parameters` は hidden について単調増加なので二分探索で
    境界を求め、その左右2点のうち誤差の小さい方を採る。
    """
    from dataclasses import replace

    def total(h: int) -> int:
        return estimate_gru_parameters(replace(base, hidden=h))

    lo, hi = 1, 2
    while total(hi) < target:
        hi *= 2
        if hi > 10 ** 6:
            break
    while lo < hi:
        mid = (lo + hi) // 2
        if total(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    candidates = [h for h in (lo - 1, lo) if h >= 1]
    return min(candidates, key=lambda h: abs(total(h) - target))


class FPGRUTranslator(nn.Module):
    """`nn.GRU` の encoder-decoder seq2seq（Cho et al. 流の文脈連結型）。

    論文 RDDLGN の GRU ベースライン(BLEU 5.41)を**我々の評価系**で測り直すための
    対照。FPSeqMixer と同じ呼び出し面（:meth:`encode` / :meth:`decode_logits` /
    ``forward(src, tgt)`` / :meth:`count_parameters`）を持つので、
    `evaluate` / `greedy_decode` / `compute_bleu` を分岐なしで共有できる。

    因果性: decoder は単方向 GRU なので位置tの出力は t 以下のトークンにしか
    依存しない（構造的保証）。
    """

    def __init__(self, config: FPGRUConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        h = config.hidden

        self.src_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=0.02)
        if config.shared_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_emb)
            nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=0.02)

        self.encoder_rnn = nn.GRU(
            config.d_emb, h, num_layers=config.n_layers,
            batch_first=True, dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.decoder_rnn = nn.GRU(
            config.d_emb + h, h, num_layers=config.n_layers,
            batch_first=True, dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(h, config.vocab_size)
        self.to(device)

    def encode(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """(B, S) -> memory (B, n_layers, H)。encoderの最終隠れ状態。

        greedy decode は毎ステップ prefix 全体を再デコードするので、memory は
        「tgtに依存しない encoder 側の全情報」であればよい。
        """
        _, hidden = self.encoder_rnn(self.src_embedding(src_tokens))
        return hidden.transpose(0, 1).contiguous()      # (B, L, H)

    def decode_logits(
        self, tgt_tokens: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """(B, T), memory(B, L, H) -> logits (B, T, vocab_size)。"""
        hidden = memory.transpose(0, 1).contiguous()    # (L, B, H)
        ctx = memory[:, -1]                             # (B, H) 最終層の文脈
        tgt_emb = self.tgt_embedding(tgt_tokens)
        ctx_seq = ctx.unsqueeze(1).expand(-1, tgt_emb.shape[1], -1)
        out, _ = self.decoder_rnn(torch.cat([tgt_emb, ctx_seq], dim=-1), hidden)
        return self.head(out)

    def forward(
        self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor
    ) -> torch.Tensor:
        return self.decode_logits(tgt_tokens, self.encode(src_tokens))

    def count_parameters(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        counts['embedding'] = sum(p.numel() for p in self.src_embedding.parameters())
        if not self.config.shared_vocab:
            counts['embedding'] += sum(
                p.numel() for p in self.tgt_embedding.parameters())
        counts['encoder'] = sum(p.numel() for p in self.encoder_rnn.parameters())
        counts['decoder'] = sum(p.numel() for p in self.decoder_rnn.parameters())
        counts['head'] = sum(p.numel() for p in self.head.parameters())
        counts['total'] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        c = self.config
        return (f'vocab={c.vocab_size}, d_emb={c.d_emb}, hidden={c.hidden}, '
                f'n_layers={c.n_layers}')


# ============================================================
# 呼び出し面の統一（Mixer / GRU をここで吸収する）
# ============================================================

def model_encode(model, src_tokens: torch.Tensor) -> torch.Tensor:
    """src トークン -> memory。以降の decode に渡す tgt 非依存の表現。"""
    if isinstance(model, FPGRUTranslator):
        return model.encode(src_tokens)
    return model.encoder(model.src_embedding(src_tokens))


def model_decode_logits(model, tgt_tokens: torch.Tensor,
                        memory: torch.Tensor) -> torch.Tensor:
    """tgt トークン + memory -> logits。"""
    if isinstance(model, FPGRUTranslator):
        return model.decode_logits(tgt_tokens, memory)
    return model.decoder(model.tgt_embedding(tgt_tokens), memory)


# ============================================================
# 訓練・評価（論理版と同一プロトコル）
# ============================================================

def train_step(model, src, tgt, optimizer, label_smoothing=0.1):
    """1ステップの訓練。teacher forcing。"""
    model.train()
    vocab_size = model.config.vocab_size

    # teacher forcing: tgt[:-1]が入力、tgt[1:]がラベル
    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    logits = model(src, tgt_input)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt_label.reshape(-1),
        label_smoothing=label_smoothing,
        ignore_index=TOKEN_PAD,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _iter_chunks(n: int, chunk_size: int):
    """[0, n) を chunk_size 単位の (start, end) に分割する。0以下なら分割しない。"""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        yield 0, n
        return
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


@torch.no_grad()
def evaluate(model, loader, device, chunk_size=0):
    """トークンレベルの精度を計算（PADは無視）。論理版 `evaluate` と同一定義。

    FP には soft/hard の区別が無いので `use_soft` 引数を持たない（論理版は
    `model.train(mode=use_soft)` で緩和/離散を切り替えていた）。
    chunk_size は純粋なメモリ管理で、分割の有無で結果は厳密に同一になる。
    """
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        src_batch, tgt_full = prepare_batch(src, tgt)
        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        for start, end in _iter_chunks(src_batch.shape[0], chunk_size):
            logits = model(src_batch[start:end], tgt_input[start:end])
            preds = logits.argmax(dim=-1)

            labels = tgt_label[start:end]
            mask = labels != TOKEN_PAD
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def greedy_decode(model, src, max_len, bos_token=TOKEN_BOS):
    """貪欲デコーディング。論理版 `greedy_decode` と**意味論が同一**。

    BOSから開始／毎ステップ prefix 全体を再デコード／`logits[:, -1].argmax`／
    返り値はBOSを除く。encoder出力の受け取り方だけモデル種別で吸収する。
    """
    batch = src.shape[0]
    device = src.device

    memory = model_encode(model, src)

    generated = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model_decode_logits(model, generated, memory)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated[:, 1:]  # BOSを除く


@torch.no_grad()
def compute_bleu(model, loader, tokenizer, device, max_samples=1000, chunk_size=0):
    """greedy decodeでBLEUを計算（論理版・RDDLGN と同一プロトコル）。

    - `sacrebleu.corpus_bleu(hyps, [refs]).score`（tokenize引数を指定しない=既定13a）
    - `max_len = tgt.shape[1]`
    - `max_samples=1000`
    - `ref.strip()` が空のサンプルは除外
    この4条件は公平比較の生命線なので変更しない。

    FP には soft/hard の区別が無いのでBLEUは1本だけ（論理版は relaxed/collapsed の
    2本を測り、その差 collapse gap を報告していた）。
    """
    if tokenizer is None:
        return None

    # tokenizerがあるのにsacrebleuが無いのは構成ミス。握り潰さず落とす
    import sacrebleu

    was_training = model.training
    model.eval()
    hypotheses = []
    references = []
    count = 0

    for src, tgt in loader:
        if count >= max_samples:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        seq_len = tgt.shape[1]

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
    """(allocated, reserved) のピークをGB単位で返す。CUDA以外はNone。"""
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return (
        torch.cuda.max_memory_allocated() / _BYTES_PER_GB,
        torch.cuda.max_memory_reserved() / _BYTES_PER_GB,
    )


def require_bleu_deps(task: str) -> None:
    """wmt14 runでは起動時に sacrebleu の存在を検証し、無ければ即座に落とす。"""
    if task != 'wmt14':
        return
    try:
        import sacrebleu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            'BLEU計測に必要な sacrebleu が見つかりません（--task wmt14 では必須）。\n'
            '  ローカル: 既存extrasを全て並べたうえで `uv sync ... --extra wmt` を実行\n'
            '  Modal   : scripts/modal_app.py の .uv_sync(extras=[...]) に "wmt" があるか確認'
        ) from e


# ============================================================
# ログ
# ============================================================

def setup_logger(task, model_kind, log_dir='logs'):
    """ログファイルを作成する（run_dir 側にも log.txt を出す）。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'fp{model_kind}_{task}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    handles = [log_file]

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        for fh in handles:
            fh.write(f'[{ts}] {msg}\n')
            fh.flush()

    def attach(path):
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


# CLI dest → FPSeqMixerConfig のフィールド名
_CONFIG_FIELD_MAP = {
    'vocab_size': 'vocab_size',
    'd_emb': 'd_emb',
    'seq_len': 'seq_len',
    'd_model': 'd_model',
    'channel_hidden': 'channel_hidden',
    'n_enc_blocks': 'n_enc_blocks',
    'n_dec_blocks': 'n_dec_blocks',
    'channel_depth': 'channel_depth',
    'token_kernels': 'token_kernels',
    'token_depth': 'token_depth',
    'gate': 'gate',
    'residual_mode': 'residual_mode',
    'norm': 'norm',
    'positional': 'positional',
    'n_pos': 'n_pos',
    'context_mode': 'context_mode',
    'cross_attention': 'cross_attention',
    'dropout': 'dropout',
}


def _cli_value(args, dest):
    """CLI値をconfigフィールド用の型へ変換する（列挙系はtupleへ）。"""
    value = getattr(args, dest)
    if dest == 'token_kernels' and isinstance(value, str):
        return tuple(parse_int_list(value))
    return value


def _explicit_dests(parser, args) -> set[str]:
    """既定値から変わっている引数のdest集合を返す（明示指定の近似判定）。"""
    return {
        dest for dest in _CONFIG_FIELD_MAP
        if hasattr(args, dest) and getattr(args, dest) != parser.get_default(dest)
    }


def build_mixer_config(args, parser) -> FPSeqMixerConfig:
    """CLI引数から FPSeqMixerConfig を構築する（土台は paper / tiny）。"""
    if args.config == 'paper':
        config = FPSeqMixerConfig.from_paper()
    else:
        config = FPSeqMixerConfig.tiny()

    for dest in _explicit_dests(parser, args):
        value = _cli_value(args, dest)
        if value is not None:
            setattr(config, _CONFIG_FIELD_MAP[dest], value)
    # cross_attention はアーム切替の要なので常に反映する
    config.cross_attention = args.cross_attention

    config.__post_init__()   # 上書き後の値域を再検証する
    return config


def build_gru_config(args) -> FPGRUConfig:
    """CLI引数から FPGRUConfig を構築する（hidden は自動で予算合わせ）。"""
    config = FPGRUConfig(
        vocab_size=args.vocab_size if args.vocab_size else 16000,
        d_emb=args.d_emb if args.d_emb else 1024,
        seq_len=args.seq_len if args.seq_len else 16,
        n_layers=args.gru_layers,
        dropout=args.dropout if args.dropout else 0.0,
    )
    if args.gru_hidden is not None:
        config.hidden = args.gru_hidden
    else:
        # FP Mixer(paper) と同じ vocab/d_emb 前提の予算に合わせる
        config.hidden = solve_gru_hidden(args.param_budget, config)
    config.__post_init__()
    return config


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
    parser = argparse.ArgumentParser(
        description='FPSeqMixer / FP GRU 訓練スクリプト')

    # モデル種別
    parser.add_argument('--model', type=str, default='mixer',
                        choices=['mixer', 'gru'],
                        help='mixer=FPSeqMixer / gru=nn.GRU seq2seq対照 '
                             '(default: mixer)')

    # タスク設定
    parser.add_argument('--task', type=str, default='copy',
                        choices=list(TASK_DATASETS.keys()) + ['wmt14'],
                        help='タスク (default: copy)')
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-test-samples', type=int, default=5000)

    # モデル構成（mixer）
    parser.add_argument('--config', type=str, default='tiny',
                        choices=['paper', 'tiny', 'custom'],
                        help='モデル構成プリセット (default: tiny)')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='語彙サイズ（wmt14では実語彙数で上書きされる）')
    parser.add_argument('--d-emb', type=int, default=None, help='埋め込み幅')
    parser.add_argument('--seq-len', type=int, default=None, help='系列長')
    parser.add_argument('--d-model', type=int, default=None,
                        help='mixerの内部幅（FP版の唯一の自由度）')
    parser.add_argument('--channel-hidden', type=int, default=None)
    parser.add_argument('--n-enc-blocks', type=int, default=None)
    parser.add_argument('--n-dec-blocks', type=int, default=None)
    parser.add_argument('--channel-depth', type=int, default=None)
    parser.add_argument('--token-kernels', type=str, default='3,8,16',
                        help='token-mixing のカーネル幅（カンマ区切り, default: 3,8,16）')
    parser.add_argument('--token-depth', type=int, default=None)
    parser.add_argument('--gate', type=str, default=None, choices=['mul', 'none'])
    parser.add_argument('--residual-mode', type=str, default=None,
                        choices=['add', 'none'])
    parser.add_argument('--norm', type=str, default=None, choices=['layer', 'none'])
    parser.add_argument('--positional', type=str, default=None,
                        choices=['learned', 'none'])
    parser.add_argument('--n-pos', type=int, default=None)
    parser.add_argument('--context-mode', type=str, default=None,
                        choices=['broadcast', 'gather', 'concat_all'])
    parser.add_argument('--cross-attention', type=str, default='none',
                        choices=['none', 'single_head'],
                        help='アーム切替: none=主アーム(i) / single_head=アーム(ii)')
    parser.add_argument('--dropout', type=float, default=None)

    # モデル構成（gru）
    parser.add_argument('--gru-hidden', type=int, default=None,
                        help='GRU隠れ幅。未指定なら --param-budget に合わせて自動決定')
    parser.add_argument('--gru-layers', type=int, default=2,
                        help='GRUの層数 (default: 2 = mixerのenc/dec各2ブロックに対応)')
    parser.add_argument('--param-budget', type=int, default=FP_MIXER_PAPER_PARAMS,
                        help=f'GRUを合わせる目標パラメータ数 '
                             f'(default: {FP_MIXER_PAPER_PARAMS} = FP Mixer paper構成)')

    # 訓練設定
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学習率 (default: 1e-3)。論理版の 0.05 は LutLayer の'
                             '5パラメータ化に合わせた値で、FPでは発散するため変更した')
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=40000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10000,
                        help='ReduceLROnPlateau patience (default: 10000)')
    parser.add_argument('--lr-factor', type=float, default=0.8)
    parser.add_argument('--eval-chunk-size', type=int, default=0,
                        help='評価forwardの分割サイズ（純粋なメモリ管理。'
                             '0以下で分割なし。結果は分割前と厳密に同一）')
    parser.add_argument('--bleu-every', type=int, default=0,
                        help='BLEUを測る間隔(step)。0=最終stepのみ (default: 0)')
    parser.add_argument('--log-every-steps', type=int, default=None,
                        help='train lossをwandbへ記録するstep間隔。未指定なら実測s/itから'
                             f'{int(LOSS_LOG_INTERVAL_SEC)}秒以内になる値を自動算出する')

    # チェックポイント
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='自動再開を無効化して必ず step 0 から開始する（既定は自動再開ON）')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None,
                        help='未指定なら自動検出 (cuda→mps→cpu)')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--data-dir', type=str, default='./data-wmt14')
    parser.add_argument('--job-type', type=str, default=None,
                        choices=['train', 'ablation', 'sweep', 'eval', 'smoke'])
    parser.add_argument('--no-wandb', action='store_true', help='wandbを使わない')

    add_experiment_args(parser)
    return parser


# ============================================================
# メイン
# ============================================================

def main():
    # リダイレクト時の標準出力はブロックバッファリングされ、長時間runの進捗が
    # 終了時まで書き出されない。行バッファリングへ切り替える（Modal本走で必須）。
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

    require_bleu_deps(args.task)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    is_gru = args.model == 'gru'
    config = build_gru_config(args) if is_gru else build_mixer_config(args, parser)

    if args.vocab_size is not None:
        config.vocab_size = args.vocab_size
    if args.seq_len is not None:
        config.seq_len = args.seq_len

    print(f'\n=== FP {"GRU" if is_gru else "SeqMixer"} 訓練 ===')
    print(f'タスク: {args.task}')
    if is_gru:
        print(f'GRU: hidden={config.hidden} layers={config.n_layers} '
              f'd_emb={config.d_emb}')
    else:
        print(f'アーム: cross_attention={config.cross_attention}')
        print(f'構成: {args.config}')
        print(f'd_emb={config.d_emb} d_model={config.d_model} '
              f'hidden={config.hidden_width} context_mode={config.context_mode} '
              f'positional={config.positional}(n_pos={config.n_pos})')

    # データセット（論理版と同一の取得経路）
    wmt_tokenizer = None
    if args.task == 'wmt14':
        from data.wmt14 import download_and_tokenize
        train_set, test_set, wmt_tokenizer = download_and_tokenize(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            data_dir=args.data_dir,
        )
        config.vocab_size = wmt_tokenizer.vocab_size
    else:
        dataset_cls = TASK_DATASETS[args.task]
        train_set = dataset_cls(config.vocab_size, config.seq_len, args.num_samples)
        test_set = dataset_cls(config.vocab_size, config.seq_len, args.num_test_samples)

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
    if is_gru:
        model = FPGRUTranslator(config, device=args.device).to(args.device)
    else:
        model = FPSeqMixer(config, device=args.device).to(args.device)
    counts = model.count_parameters()
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    budget_ratio = (counts['total'] - FP_MIXER_PAPER_PARAMS) / FP_MIXER_PAPER_PARAMS
    print(f'  FP Mixer(paper) {FP_MIXER_PAPER_PARAMS:,} 比: {budget_ratio:+.2%}')

    # Optimizer / Scheduler（論理版と同一。lrの既定値だけFP向けに変更）
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

    log, attach_log, close_log = setup_logger(args.task, args.model, args.log_dir)

    arm = config.cross_attention if not is_gru else 'gru'
    ckpt_prefix = f'fp{args.model}_{args.task}_{args.config}_{arm}'

    vocab_path = None
    if wmt_tokenizer is not None:
        vocab_path = os.path.join(args.data_dir, f'vocab_v{config.vocab_size}.json')

    ckpt_metadata = {
        'args': vars(args),
        'config': asdict(config),
        'task': args.task,
        'vocab_path': vocab_path,
    }

    # 自動再開の候補探索（Modalの自動リトライは新しい run_dir を作るため prefix 横断で探す）
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

    if resume_run_dir is not None:
        run_dir = resume_run_dir
        latest_path, best_path = checkpoint_paths(run_dir)
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir(
            ckpt_prefix, base_dir=args.checkpoint_dir,
        )
    attach_log(os.path.join(run_dir, 'log.txt'))
    log(f'START model={args.model} task={args.task} config={args.config} '
        f'arm={arm} vocab={config.vocab_size} seq_len={config.seq_len} '
        f'lr={args.lr} bs={args.batch_size} ni={args.num_iterations} '
        f'device={args.device} params={counts["total"]:,}')

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
    run_name = (f'fp{args.model}_{args.task}_{args.config}_{arm}'
                f'_v{config.vocab_size}_lr{args.lr}_ni{args.num_iterations}')
    job_type = args.job_type
    if job_type is None:
        job_type = 'train' if args.task == 'wmt14' else 'smoke'
    wandb_config = {**vars(args), **asdict(config),
                    'fp_mixer_paper_params': FP_MIXER_PAPER_PARAMS,
                    'params_total': counts['total']}
    use_wandb = False
    if not args.no_wandb:
        model_tag = f'fp_{args.model}'
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=([model_tag, 'tsmix-1d', args.task, args.config, f'arm:{arm}']
                  + normalize_tags(model=model_tag, dataset=args.task,
                                   seed=args.seed, series='TSMIX-1')),
            group='tsmix-1',
            job_type=job_type,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    remaining_iters = args.num_iterations - start_step

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()
    reset_peak_memory(args.device)
    bleu = None

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

        src_batch, tgt_full = prepare_batch(src, tgt)

        loss = train_step(
            model, src_batch, tgt_full, optimizer,
            label_smoothing=args.label_smoothing,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.6f}',
            eta=eta_str,
        )

        # 軽量ログ（SKILL.md §11: train lossは壁時計で最低5分に1回）
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
                'loss/total': loss,
                'loss/main': loss,
                'lr': optimizer.param_groups[0]['lr'],
                **progress_metrics(done_iters, step + 1, args.num_iterations, elapsed),
            }

        # 評価
        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            train_peak = peak_memory_gb(args.device)
            reset_peak_memory(args.device)
            eval_start = time.time()

            test_acc = evaluate(model, test_loader, args.device,
                                chunk_size=args.eval_chunk_size)

            # 論理版は `1.0 - acc_soft` を渡していた。FPはacc1本なのでそれに対応
            scheduler.step(1.0 - test_acc)

            is_best = test_acc > best_test_acc
            if is_best:
                best_test_acc = test_acc

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
            current_lr = optimizer.param_groups[0]['lr']

            bleu_str = ''
            is_final = step == args.num_iterations - 1
            want_bleu = is_final or (
                args.bleu_every > 0 and (step + 1) % args.bleu_every == 0
            )
            if args.task == 'wmt14' and want_bleu:
                bleu = compute_bleu(model, test_loader, wmt_tokenizer, args.device,
                                    chunk_size=args.eval_chunk_size)
                if bleu is not None:
                    bleu_str = f'  BLEU={bleu:.2f}'

            eval_sec = time.time() - eval_start
            eval_peak = peak_memory_gb(args.device)
            mem_str = f'  eval={eval_sec:.1f}s'
            if train_peak is not None and eval_peak is not None:
                mem_str += (f' mem(train)={train_peak[0]:.2f}/{train_peak[1]:.2f}GB'
                            f' mem(eval)={eval_peak[0]:.2f}/{eval_peak[1]:.2f}GB'
                            f' [allocated/reserved]')
            reset_peak_memory(args.device)

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={loss:.4f}  acc={test_acc:.4f}'
                f'{bleu_str}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  lr={current_lr:.6f}'
                f'  [{elapsed_str}<{remain_str}, {iter_per_sec:.1f}it/s, 終了予定 {eta_str}]'
                f'{mem_str}',
                flush=True,
            )

            log(f'ITER {step+1}/{args.num_iterations} loss={loss:.4f} '
                f'acc={test_acc:.4f}{bleu_str} lr={current_lr:.6f} '
                f'best={str(is_best).lower()} elapsed={elapsed_str} eta={remain_str}'
                f'{mem_str}')

            if use_wandb:
                wandb_metrics = {
                    **(light_metrics or {}),
                    'iteration': step + 1,
                    'eval/seconds': eval_sec,
                    'loss/total': loss,
                    'loss/main': loss,
                    'acc/soft': test_acc,   # FPはsoft/hardの区別が無く同一値
                    'acc/hard': test_acc,
                    'best_test_acc': best_test_acc,
                    'lr': current_lr,
                }
                if bleu_str:
                    # 論理版・RDDLGN とキー体系を揃える（FPは1本なので同値を書く）
                    wandb_metrics['bleu'] = bleu
                    wandb_metrics['bleu/soft'] = bleu
                    wandb_metrics['bleu/hard'] = bleu
                    wandb_metrics['bleu/collapse_gap'] = 0.0
                if train_peak is not None and eval_peak is not None:
                    wandb_metrics['mem/train_peak_allocated_gb'] = train_peak[0]
                    wandb_metrics['mem/train_peak_reserved_gb'] = train_peak[1]
                    wandb_metrics['mem/eval_peak_allocated_gb'] = eval_peak[0]
                    wandb_metrics['mem/eval_peak_reserved_gb'] = eval_peak[1]
                    wandb_metrics['mem/peak_allocated_gb'] = max(train_peak[0], eval_peak[0])
                    wandb_metrics['mem/peak_reserved_gb'] = max(train_peak[1], eval_peak[1])
                log_metrics(wandb_metrics, step=step + 1)
                light_metrics = None

        if light_metrics is not None:
            log_metrics(light_metrics, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度: {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_acc={best_test_acc:.4f} bleu={bleu} total_time={total_time}')
    close_log()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={
                'args': vars(args),
                'config': asdict(config),
                'best_test_acc': best_test_acc,
                'bleu': bleu,
                'params_total': counts['total'],
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
        sample_logits = model(sample_src, sample_tgt_full[:, :-1])
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
