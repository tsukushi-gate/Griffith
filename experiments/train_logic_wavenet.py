"""
LogicWaveNet 訓練スクリプト(dilated causal conv の論理ゲート化 / Tiny Shakespeare)

`griffith_logic.models.logic_wavenet.LogicWaveNet`(dilated causal conv を積層した
自己回帰 char-level 言語モデル)を Tiny Shakespeare で訓練する。attention を持たず、
ゲート付き活性 z=AND(filter, gate) と residual/skip 接続のみで系列をモデル化する
(D1 LogicGPT の attention 版に対する比較軸)。

Tiny Shakespeare の DL・語彙構築・窓生成は本スクリプト内で自作し、データは相対パス
./data-shakespeare/(train_logic_gpt.py と共有)に落とす。

使用例:
    # 本番(20000ステップ)
    uv run python experiments/train_logic_wavenet.py

    # スモーク(CPUで2分以内、wandb無し)
    uv run python experiments/train_logic_wavenet.py --max-steps 20 --no-wandb --device cpu

    # Modal 実行
    uv run modal run scripts/modal_app.py --script train_logic_wavenet.py --args "--max-steps 20000"
"""

import argparse
import os
import sys
import time
import urllib.request
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

from griffith_logic.models.logic_wavenet import LogicWaveNet, LogicWaveNetConfig

load_dotenv()

# ============================================================
# Tiny Shakespeare(char-level LM)— 自作(train_logic_gpt.py と同方式)
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
    """Tiny Shakespeare(char-level LM)の(train_loader, val_loader, meta)を返す。"""
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


def cycle(loader):
    """DataLoaderを無限に巡回するジェネレータ(ステップ数ベース訓練用)。"""
    while True:
        yield from loader


# ============================================================
# 評価・生成
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> float:
    """val loss(hard eval)を平均で返す。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
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


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LogicWaveNet 訓練スクリプト(Tiny Shakespeare / char-LM)'
    )
    parser.add_argument('--d-channels', type=int, default=64,
                        help='論理チャネル幅(埋め込みビット幅)')
    parser.add_argument('--n-layers', type=int, default=8,
                        help='dilated conv 層数(受容野=2^n_layers)')
    parser.add_argument('--kernel-size', type=int, default=2, help='conv タップ数')
    parser.add_argument('--n-inputs', type=int, default=2, help='LUTファンイン(2/4/6)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        help='LUTパラメータ化(softmax/independent/warp/...)')
    parser.add_argument('--skip-mode', type=str, default='or',
                        help='skip集約(or/xor)')
    parser.add_argument('--residual-mode', type=str, default='xor',
                        help='residualマージ(or/xor)')
    parser.add_argument('--conv-depth', type=int, default=1,
                        help='filter/gate 1本あたりのLutLayer段数')
    parser.add_argument('--context-len', type=int, default=128,
                        help='系列長(受容野以上を推奨)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--eval-freq', type=int, default=500)
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help=f'データ保存先 (default: {_DEFAULT_DATA_DIR})')
    parser.add_argument('--seed', type=int, default=0, help='ランダムシード')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbを無効化する(スモーク用)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開(latest.pt等のパス指定)')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='チェックポイント保存先ディレクトリ名'
                             '(checkpoints/<NAME>/固定。未指定時はタイムスタンプ方式)')
    add_experiment_args(parser)
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

    # データ
    train_loader, val_loader, meta = load_shakespeare(
        batch_size=args.batch_size, context_len=args.context_len,
        data_dir=args.data_dir,
    )
    vocab_size = meta['vocab_size']
    print(f'\nデータ: Tiny Shakespeare  vocab_size={vocab_size} '
          f'context_len={args.context_len}')

    # モデル
    config = LogicWaveNetConfig(
        vocab_size=vocab_size,
        d_channels=args.d_channels,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        skip_mode=args.skip_mode,
        residual_mode=args.residual_mode,
        conv_depth=args.conv_depth,
    )
    model = LogicWaveNet(config).to(device)
    counts = model.count_parameters()
    print(f'\nLogicWaveNet  d_channels={config.d_channels} '
          f'n_layers={config.n_layers} kernel_size={config.kernel_size}')
    print(f'  受容野(receptive field)={model.receptive_field}  '
          f'n_inputs={config.n_inputs} param={config.parameterization}')
    print(f'  skip={config.skip_mode} residual={config.residual_mode} '
          f'conv_depth={config.conv_depth}')
    print(f'  総パラメータ数: {counts["total"]:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # チェックポイント
    if args.run_dir:
        run_dir = os.path.join('checkpoints', args.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        latest_path = os.path.join(run_dir, 'latest.pt')
        best_path = os.path.join(run_dir, 'best.pt')
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir('logic_wavenet')

    # --resume: モデル/optimizer/ステップ/best_metricを復元
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
    run_name = (f'logic_wavenet_c{config.d_channels}_l{config.n_layers}'
                f'_k{config.kernel_size}_s{args.max_steps}')
    if not args.no_wandb:
        wandb_config = {
            'd_channels': config.d_channels, 'n_layers': config.n_layers,
            'kernel_size': config.kernel_size, 'n_inputs': config.n_inputs,
            'parameterization': config.parameterization,
            'skip_mode': config.skip_mode, 'residual_mode': config.residual_mode,
            'conv_depth': config.conv_depth,
            'receptive_field': model.receptive_field,
            'context_len': args.context_len, 'batch_size': args.batch_size,
            'lr': args.lr, 'weight_decay': args.weight_decay,
            'max_steps': args.max_steps, 'seed': args.seed,
        }
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=device,
            tags=['logic_wavenet', 'shakespeare',
                  f'l{config.n_layers}', f'k{config.kernel_size}',
                  f'param-{config.parameterization}']
            + normalize_tags(model='logic_wavenet', dataset='shakespeare', seed=args.seed),
            group='logic_wavenet',
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
        'model': 'logic_wavenet', 'wandb_run_id': wandb_run_id,
    }

    # 訓練ループ(ステップ数ベース)
    print(f'\n訓練開始: {args.max_steps}ステップ (評価={args.eval_freq}ステップごと、'
          f'開始ステップ={start_step + 1})')
    print('=' * 70)
    loss_fn = nn.CrossEntropyLoss()
    data_iter = cycle(train_loader)
    train_start = time.time()
    model.train()

    for step in range(start_step + 1, args.max_steps + 1):
        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B,T,V)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ハートビート(eval時に経過/残り/終了予定を表示)
        if step % args.eval_freq == 0 or step == args.max_steps or step == 1:
            val_loss = evaluate(model, val_loader, device)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # チェックポイント(eval毎にlatest.pt、best更新時はbest.pt)
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
            sec_per_step = elapsed / max(1, step - start_step)
            remaining = sec_per_step * (args.max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))

            print(
                f'[step {step:>6d}/{args.max_steps}] '
                f'train_loss={loss.item():.4f} '
                f'val_loss={val_loss:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            if use_wandb:
                log_metrics({
                    'step': step,
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
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
            metadata={'model': 'logic_wavenet',
                      'best_val_loss': best_val_loss, 'max_steps': args.max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
