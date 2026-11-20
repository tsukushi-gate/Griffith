"""
RDDLGN 文字LM 訓練スクリプト(D1: attention有無の対照群 / Tiny Shakespeare)

LogicGPT(attention段階論理化)との**同一プロトコル比較**のためのベースライン。
RDDLGN(Elman型リカレント・attentionなし)を Tiny Shakespeare の次文字予測に適用し、
論文Table「attentionの寄与」の対照群を得る。

設計判断:
    - RDDLGN 本体(griffith_logic.models.rddlgn)は seq2seq 翻訳用の
      encoder-decoder 構成。次文字予測(自己回帰LM)では **encoder 側の再帰
      (K層の k_prev 伝播)を各時刻で回し、各ステップの隠れ状態から logits を出す**
      構成が自然(decoder は teacher-forcing 翻訳用途なので使わない)。
    - よって本スクリプトは models/rddlgn.py の再利用可能部品
      (LogicEmbedding / LutStack / GroupSum / _init_hidden)だけを組み合わせて
      char-LM 専用の薄いラッパ RDDLGNCharLM を定義する(rddlgn.py は読み取りのみ)。
    - **causal性は再帰構造から自動的に保証される**: 時刻 t の logits は入力 x[0..t]
      のみに依存し、未来 x[t+1..] を一切参照しない(LogicGPT の causal mask に相当)。
    - データローダ(load_shakespeare)・評価(evaluate, hard/soft)・サンプル生成・
      引数・wandb・checkpoint の作法は train_logic_gpt.py と**同一物をコピー**して
      同一プロトコルを保証する(唯一の相違はモデルと group/tags)。

パラメータ規模:
    LogicGPT 論理部(L3-C 総パラメータ)= 33,349。これに合わせて d_emb・層幅を選定:
      d_emb=32, N=[256,256], K=[512,448], M=[222,260]  → 総計 33,344(diff -5)。
    softmax(N=2) LUT は 1ニューロン=16パラメータ。埋め込み 65*32=2080 を除いた
    論理ニューロン数 1954 * 16 = 31,264 + 2,080 = 33,344。

使用例:
    # スモーク(CPUで2分以内、wandbオフライン)
    WANDB_MODE=offline uv run python experiments/train_rddlgn_charlm.py \\
        --max-steps 10 --eval-batches 5 --device cpu --run-dir _rddlgn_smoke

    # 本走(20000ステップ)
    uv run python experiments/train_rddlgn_charlm.py \\
        --notes "attention無の対照群(D1)" --hypothesis "..." --seed 0

    # Modal 実行
    uv run modal run scripts/modal_app.py --script train_rddlgn_charlm.py \\
        --args "--seed 0 --notes '...' --hypothesis '...'"
"""

import argparse
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
)
from wandb_utils import finish as wandb_finish

from griffith_logic import GroupSum, LogicEmbedding, LutStack
from griffith_logic.models.rddlgn import _init_hidden

load_dotenv()

# ============================================================
# Tiny Shakespeare(char-level LM)— train_logic_gpt.py と同一物をコピー
# (同一プロトコル比較を保証するため、意図的に共有せず複製している)
# ============================================================

SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
_DEFAULT_DATA_DIR = './data-shakespeare'
_SHAKESPEARE_SUBPATH = os.path.join('tinyshakespeare', 'input.txt')
_TRAIN_RATIO = 0.9


def prepare_shakespeare(data_dir: str = _DEFAULT_DATA_DIR) -> str:
    """Tiny Shakespeare の input.txt を用意しパスを返す(無ければDL)。

    DL先は data_dir(相対パス。Modalでは永続Volumeに載る)。DL失敗時は捏造せず
    明確に例外を投げる。
    """
    path = os.path.join(data_dir, _SHAKESPEARE_SUBPATH)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001 (DL失敗は原因を明示して再送出)
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロードに失敗しました: {SHAKESPEARE_URL} '
            f'(原因: {e})'
        ) from e
    if not data:
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロード内容が空です: {SHAKESPEARE_URL}'
        )
    with open(path, 'wb') as f:
        f.write(data)
    return path


class _CharWindowDataset(Dataset):
    """スライディング窓でchar-LMの(x, y)ペアを生成する。

    x = tokens[i : i+context_len]、y = tokens[i+1 : i+context_len+1]。
    """

    def __init__(self, data: torch.Tensor, context_len: int) -> None:
        self._data = data
        self._context_len = context_len

    def __len__(self) -> int:
        return max(0, self._data.numel() - self._context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._data[idx: idx + self._context_len + 1]
        return chunk[:-1], chunk[1:]


def load_shakespeare(
    batch_size: int,
    context_len: int,
    data_dir: str = _DEFAULT_DATA_DIR,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Tiny Shakespeare(char-level LM)の(train_loader, val_loader, meta)を返す。

    char語彙(stoi/itos, 語彙数V〜65)を構築し 90/10 でtrain/val分割、
    スライディング窓(context_len)で次トークン予測データを作る。
    """
    path = prepare_shakespeare(data_dir)
    with open(path, encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(ids.numel() * _TRAIN_RATIO)
    train_data = ids[:split]
    val_data = ids[split:]

    train_set = _CharWindowDataset(train_data, context_len)
    val_set = _CharWindowDataset(val_data, context_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    meta = {
        'task': 'lm',
        'seq_len': context_len,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    return train_loader, val_loader, meta


# ============================================================
# RDDLGN 文字LM モデル(rddlgn.py の部品を再利用した薄いラッパ)
# ============================================================

@dataclass
class CharLMConfig:
    """RDDLGN 文字LM のハイパーパラメータ。

    層幅の既定値は LogicGPT 論理部(33,349)に合わせた設計:
      d_emb=32, N=[256,256], K=[512,448], M=[222,260] → 総計 33,344(diff -5)。
    """

    vocab_size: int = 65
    d_emb: int = 32
    n_layer_sizes: list = field(default_factory=lambda: [256, 256])
    k_layer_sizes: list = field(default_factory=lambda: [512, 448])
    m_layer_sizes: list = field(default_factory=lambda: [222, 260])
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0
    tau: float = 1.0  # GroupSum の温度(logits = 群和 / tau)
    init_mode: str = 'zero'
    init_scale: float = 0.5

    def __post_init__(self) -> None:
        # M最終層は vocab_size で割り切れる必要がある(GroupSum の群化制約)
        m_last = self.m_layer_sizes[-1]
        if m_last % self.vocab_size != 0:
            raise ValueError(
                f'M最終層({m_last})は vocab_size({self.vocab_size})で割り切れる'
                f'必要があります(GroupSum の群化制約)'
            )


class RDDLGNCharLM(nn.Module):
    """RDDLGN(Elman型)による char-level 言語モデル。attentionなし。

    構成: LogicEmbedding(V,d) → N層(各トークンをバッチ化した前処理)
        → K層(k_prev を各時刻へ伝播する再帰。attention の代替)
        → 各時刻の隠れ状態を M層 → GroupSum で (B,T,V) の logits に変換。

    causal性: 時刻 t の logits は入力 x[0..t] のみに依存する(K層の再帰が
    過去のみを参照するため)。LogicGPT の causal mask に対応する。
    """

    def __init__(self, config: CharLMConfig, device: str = 'cpu') -> None:
        super().__init__()
        self.config = config

        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
        )

        # 埋め込み(sigmoid[0,1] / eval時はHeaviside{0,1})
        self.embedding = LogicEmbedding(config.vocab_size, config.d_emb)

        # N層: d_emb → n_layer_sizes(各トークン独立の前処理)
        self.n_layers = LutStack(config.d_emb, config.n_layer_sizes, **lut_kwargs)

        # K層: (N出力 + K出力) → k_layer_sizes(時間方向の再帰)
        n_out = config.n_layer_sizes[-1]
        k_out = config.k_layer_sizes[-1]
        self.k_layers = LutStack(n_out + k_out, config.k_layer_sizes, **lut_kwargs)

        # M層 + GroupSum: 各時刻の K隠れ状態 → vocab logits
        self.m_layers = LutStack(k_out, config.m_layer_sizes, **lut_kwargs)
        self.group_sum = GroupSum(k=config.vocab_size, tau=config.tau)

        self.k_out_dim = k_out
        self.init_mode = config.init_mode
        self.init_scale = config.init_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) 整数トークンID

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = x.shape
        device = x.device

        # 埋め込み → N層(全時刻をバッチ化)
        emb = self.embedding(x)  # (B, T, d_emb)
        h = self.n_layers(emb.reshape(batch * seq_len, -1))
        h = h.reshape(batch, seq_len, -1)  # (B, T, n_out)

        # K層: k_prev を時間方向へ伝播し、各時刻で logits を出す(自動的にcausal)
        k_prev = _init_hidden(
            batch, self.k_out_dim, device, self.init_mode, self.init_scale,
        )
        logits_list = []
        for t in range(seq_len):
            k_input = torch.cat([h[:, t, :], k_prev], dim=-1)
            k_prev = self.k_layers(k_input)
            m_t = self.m_layers(k_prev)
            logits_list.append(self.group_sum(m_t))

        return torch.stack(logits_list, dim=1)  # (B, T, V)

    def pop_aux_loss(self) -> torch.Tensor:
        """埋め込みの二値化正則化損失(LogicGPT の選択器aux損失に対応する補助損失)。

        直前 forward の埋め込み出力 x に対する mean(x*(1-x))。0/1 に近いほど小。
        """
        return self.embedding.binary_reg_loss()

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, greedy: bool = True,
    ) -> torch.Tensor:
        """自己回帰生成。再帰モデルのため各ステップで系列全体を再評価する。

        (200文字程度のサンプル生成専用。O(n^2) だが用途上問題ない。)
        """
        for _ in range(max_new_tokens):
            logits = self(idx)  # (B, T, V)
            next_logits = logits[:, -1, :]
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す(wandbログ用)。"""
        counts = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'n_layers': sum(p.numel() for p in self.n_layers.parameters()),
            'k_layers': sum(p.numel() for p in self.k_layers.parameters()),
            'm_layers': sum(p.numel() for p in self.m_layers.parameters()),
        }
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# ============================================================
# 評価・生成 — train_logic_gpt.py と同一物をコピー
# ============================================================

@torch.no_grad()
def evaluate(
    model, loader, device, max_batches: int | None = 20, mode: str = 'hard',
) -> float:
    """val loss を平均で返す。max_batches=Noneで全val評価。

    mode='hard': `model.eval()` で評価(LogicEmbedding/LutLayer が hard{0,1})。
    mode='soft': `model.train()` で評価(sigmoid/softmax の連続緩和)。
        soft と hard の差が離散化ギャップ Δ_disc = loss_hard - loss_soft(Mind the Gap)。
    どちらも no_grad。評価後は元の train/eval 状態に復帰する。

    注意: 部分評価(max_batches指定)はvalの先頭スライスに固定されるため、
    全val評価との数値比較には使えない。本走は --eval-batches 0(=全val)を使うこと。
    """
    was_training = model.training
    if mode == 'soft':
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train(mode=was_training)
    return total_loss / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def sample_text(model, meta: dict, device, n_chars: int = 200,
                seed_char: str = '\n') -> str:
    """学習済みモデルから n_chars 文字を greedy 生成しデコードする。"""
    was_training = model.training
    model.eval()
    stoi = meta['stoi']
    itos = meta['itos']
    start_id = stoi.get(seed_char, 0)
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_chars, greedy=True)
    ids = out[0].tolist()
    model.train(mode=was_training)
    return ''.join(itos[i] for i in ids)


def cycle(loader):
    """DataLoaderを無限に巡回するジェネレータ(ステップ数ベース訓練用)。"""
    while True:
        yield from loader


def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。例: '256,256' -> [256, 256]"""
    return [int(x.strip()) for x in s.split(',')]


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='RDDLGN 文字LM 訓練スクリプト(Tiny Shakespeare / char-LM)'
    )
    # 訓練ハイパラ(既定は train_logic_gpt.py と同一プロトコル)
    parser.add_argument('--max-steps', type=int, default=20000,
                        help='最大ステップ数 (default: 20000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='バッチサイズ (default: 64)')
    parser.add_argument('--context-len', type=int, default=64,
                        help='系列長(コンテキスト長) (default: 64)')
    parser.add_argument('--lr', type=float, default=3.0e-4,
                        help='学習率 (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--eval-freq', type=int, default=500,
                        help='評価頻度(ステップ) (default: 500)')
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='評価バッチ数の上限(0=全val)。未指定は全val評価')
    parser.add_argument('--aux-lambda', type=float, default=0.0,
                        help='埋め込み二値化正則化の重み(LogicGPTのaux_lambda相当、'
                             '既定0.0=純粋な対照群)')
    # モデル構成(既定は LogicGPT 論理部33,349に合わせた33,344設計)
    parser.add_argument('--d-emb', type=int, default=32)
    parser.add_argument('--n-sizes', type=str, default='256,256')
    parser.add_argument('--k-sizes', type=str, default='512,448')
    parser.add_argument('--m-sizes', type=str, default='222,260',
                        help='M層(最終層はvocab_sizeの倍数)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp', 'raw'])
    parser.add_argument('--tau', type=float, default=1.0,
                        help='GroupSum温度 (default: 1.0)')
    # 環境・記録
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help=f'データ保存先 (default: {_DEFAULT_DATA_DIR})')
    parser.add_argument('--seed', type=int, default=0, help='ランダムシード')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbを無効化する(スモーク用)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開(latest.pt等のパス指定)')
    add_experiment_args(parser)
    parser.add_argument('--run-dir', type=str, default=None,
                        help='チェックポイント保存先ディレクトリ名'
                             '(checkpoints/<NAME>/固定。未指定時はタイムスタンプ方式)')
    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = args.device
    print(f'デバイス: {device}')

    torch.manual_seed(args.seed)

    batch_size = args.batch_size
    context_len = args.context_len
    lr = args.lr
    weight_decay = args.weight_decay
    max_steps = args.max_steps
    eval_freq = args.eval_freq
    # 評価バッチ数: 0またはNone=全val評価(ladderプロトコル準拠)
    if args.eval_batches is not None:
        eval_batches = None if args.eval_batches == 0 else args.eval_batches
    else:
        eval_batches = None
    aux_lambda = args.aux_lambda

    # データ
    train_loader, val_loader, meta = load_shakespeare(
        batch_size=batch_size, context_len=context_len, data_dir=args.data_dir,
    )
    vocab_size = meta['vocab_size']
    print(f'\nデータ: Tiny Shakespeare  vocab_size={vocab_size} '
          f'context_len={context_len}')

    # モデル
    config = CharLMConfig(
        vocab_size=vocab_size,
        d_emb=args.d_emb,
        n_layer_sizes=parse_int_list(args.n_sizes),
        k_layer_sizes=parse_int_list(args.k_sizes),
        m_layer_sizes=parse_int_list(args.m_sizes),
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        tau=args.tau,
    )
    model = RDDLGNCharLM(config, device=device).to(device)
    counts = model.count_parameters()
    print('\nRDDLGN char-LM (attentionなし対照群)')
    print(f'  d_emb={config.d_emb} N={config.n_layer_sizes} '
          f'K={config.k_layer_sizes} M={config.m_layer_sizes}')
    print(f'  parameterization={config.parameterization} n_inputs={config.n_inputs}')
    for name, cnt in counts.items():
        print(f'  {name}: {cnt:,}')
    print(f'  総パラメータ数: {counts["total"]:,} (LogicGPT論理部 33,349 対比)')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # チェックポイント
    if args.run_dir:
        run_dir = os.path.join('checkpoints', args.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        latest_path = os.path.join(run_dir, 'latest.pt')
        best_path = os.path.join(run_dir, 'best.pt')
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir('rddlgn_charlm')

    # --resume
    start_step = 0
    best_val_loss = float('inf')
    resume_wandb_run_id = None
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = ckpt_info['step']
        best_val_loss = ckpt_info['best_metric']
        resume_wandb_run_id = ckpt_info.get('metadata', {}).get('wandb_run_id')
        extra_state = ckpt_info.get('extra_state', {})
        if 'torch_rng' in extra_state:
            torch.set_rng_state(extra_state['torch_rng'])
        print(f'  ステップ {start_step} から再開 (best_val_loss={best_val_loss:.4f})')

    # wandb
    use_wandb = False
    run_name = (f'rddlgn_charlm_d{config.d_emb}'
                f'_k{config.k_layer_sizes[-1]}_s{max_steps}')
    if not args.no_wandb:
        wandb_config = {
            'model_cfg': asdict(config), 'batch_size': batch_size,
            'context_len': context_len, 'lr': lr, 'weight_decay': weight_decay,
            'max_steps': max_steps, 'aux_lambda': aux_lambda, 'seed': args.seed,
        }
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=device,
            tags=['rddlgn', 'charlm', 'shakespeare']
            + normalize_tags(series='D1', model='rddlgn',
                             dataset='shakespeare', seed=args.seed),
            group='rddlgn_charlm',
            resume_run_id=resume_wandb_run_id,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    wandb_run_id = None
    if use_wandb:
        try:
            import wandb
            wandb_run_id = wandb.run.id
        except Exception:
            pass

    ckpt_metadata = {
        'config': asdict(config), 'wandb_run_id': wandb_run_id,
    }

    # 訓練ループ(ステップ数ベース)
    print(f'\n訓練開始: {max_steps}ステップ (評価={eval_freq}ステップごと、'
          f'開始ステップ={start_step + 1})')
    print('=' * 70)
    loss_fn = nn.CrossEntropyLoss()
    data_iter = cycle(train_loader)
    train_start = time.time()
    model.train()

    for step in range(start_step + 1, max_steps + 1):
        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B,T,V)
        main_loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        aux = model.pop_aux_loss()  # 埋め込み二値化正則化(補助損失)
        loss = main_loss + aux_lambda * aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 評価・チェックポイント・ハートビート
        if step % eval_freq == 0 or step == max_steps or step == 1:
            # hard(離散化)= 従来の val_loss。best判定はこちらで行う
            val_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='hard',
            )
            # soft(連続緩和)= model.train()評価。離散化ギャップを測る
            val_soft_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='soft',
            )
            # 離散化ギャップ Δ_disc = loss_hard - loss_soft(Mind the Gap)
            discretization_gap = val_loss - val_soft_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            extra_state = {'torch_rng': torch.get_rng_state()}
            save_checkpoint(latest_path, model, optimizer, None, step,
                            best_val_loss, ckpt_metadata, extra_state=extra_state)
            if is_best:
                save_checkpoint(best_path, model, optimizer, None, step,
                                best_val_loss, ckpt_metadata, extra_state=extra_state)
            if use_wandb:
                log_checkpoint_artifact(
                    latest_path, run_name, stage='eval',
                    metadata={'step': step, 'val_loss': val_loss},
                )

            elapsed = time.time() - train_start
            sec_per_step = elapsed / (step - start_step)
            remaining = sec_per_step * (max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            aux_val = float(aux.detach()) if torch.is_tensor(aux) else float(aux)

            print(
                f'[step {step:>6d}/{max_steps}] '
                f'train_loss={main_loss.item():.4f} '
                f'val_loss={val_loss:.4f} '
                f'val_soft={val_soft_loss:.4f} '
                f'gap={discretization_gap:.4f} '
                f'aux={aux_val:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            if use_wandb:
                log_metrics({
                    'step': step,
                    'train_loss': main_loss.item(),
                    'val_loss': val_loss,
                    'val_soft_loss': val_soft_loss,
                    'discretization_gap': discretization_gap,
                    'best_val_loss': best_val_loss,
                    'aux_loss': aux_val,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 70)
    print(f'訓練完了: best_val_loss={best_val_loss:.4f}  (総時間: {total_time})')

    # サンプル生成(200文字)をログ + wandb へ
    sample = sample_text(model, meta, device, n_chars=200)
    print('\n---- サンプル生成(200文字, greedy) ----')
    print(sample)
    print('---- ここまで ----')
    if use_wandb:
        try:
            import wandb
            wandb.log({'sample_text': wandb.Html(f'<pre>{sample}</pre>')})
        except Exception:
            pass

    # 重みを wandb Artifacts へ
    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'config': asdict(config),
                      'best_val_loss': best_val_loss, 'max_steps': max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
