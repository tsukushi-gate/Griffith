"""D3: 論理リザバーコンピューティング(LogicReservoir)の訓練スクリプト。

凍結ランダム論理ゲート回路(リザバー)に系列を流し込み、readout だけを学習する。
「学習不要の巨大ランダム論理回路 + 軽量 readout 学習」という超低コスト学習を実演する。

タスク:
    --task parity : 系列パリティ(1bit ストリームの1の数の偶奇。2クラス)
    --task smnist : Sequential-MNIST(MNIST を1画素ずつ流す。thermometer 符号化。10クラス)

注意: 系列パリティは記憶×XOR を要する reservoir 難タスク。凍結ランダム回路が
running-parity 特徴を持つかは reservoir 生成シード依存で、大きめのリザバー
(--reservoir-size 512 以上)と短い系列(--seq-len 3〜4)で収束しやすい
(--reservoir-seed を振る前提)。安定に 100% 到達する例は tests/test_logic_reservoir.py
の遅延ビット再生タスクを参照。

使用例:
    # スモーク(CPU で2分以内、wandb 無し)
    uv run python experiments/train_logic_reservoir.py --task parity --max-steps 50 --no-wandb --device cpu

    # 本番(parity, wandb 有効)
    uv run python experiments/train_logic_reservoir.py --task parity --max-steps 3000

    # Sequential-MNIST(GPU 推奨。1画素ずつ 784 ステップ流す)
    uv run python experiments/train_logic_reservoir.py --task smnist --max-steps 5000

    # Modal 実行
    uv run modal run scripts/modal_app.py --script train_logic_reservoir.py --args "--task smnist --max-steps 5000"
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
)
from wandb_utils import finish as wandb_finish

from griffith_logic.encoding.thermometer import ThermometerEncoder
from griffith_logic.models import LogicReservoir, LogicReservoirConfig

load_dotenv()

_DEFAULT_DATA_DIR = './data-mnist'


# ============================================================
# タスク: parity(系列パリティ)
# ============================================================
def sample_parity_batch(
    batch_size: int, seq_len: int, width: int, device: str, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1bit ストリームの系列パリティバッチを生成する。

    入力ビットは width 本に複製して入力駆動(リザバーへの fan-out)を確保する。
    xs:(B,T,width) {0,1}、y:(B,) ∈{0,1}(系列中の1の数の偶奇)。
    """
    bit = (torch.rand(batch_size, seq_len, 1, generator=generator) > 0.5).float()
    xs = bit.repeat(1, 1, width).to(device)
    y = (bit.sum(dim=(1, 2)).long() % 2).to(device)
    return xs, y


def make_parity_eval(seq_len: int, width: int, device: str, n: int = 512):
    """固定 seed の held-out パリティ評価セット。"""
    g = torch.Generator().manual_seed(20260709)
    return sample_parity_batch(n, seq_len, width, device, g)


# ============================================================
# タスク: Sequential-MNIST(1画素ずつ流す + thermometer 符号化)
# ============================================================
def load_smnist(
    thermo_bits: int, device: str, data_dir: str = _DEFAULT_DATA_DIR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """MNIST を (画素系列, ラベル) に変換して返す。

    各画素(グレースケール)を thermometer 符号化して input_size=thermo_bits の
    系列(T=784)にする。戻り値: (xs_tr, y_tr, xs_te, y_te, input_size)。
    xs:(N, 784, thermo_bits)。
    """
    from griffith_logic.datasets.vision import load_mnist

    # グレースケールのまま読み込む(二値化しない)。thermometer で多値符号化する。
    train_set, test_set = load_mnist(root=data_dir, binarize_threshold=None)

    def _flatten(ds):
        xs = torch.stack([img.reshape(-1) for img, _ in ds])  # (N, 784)
        ys = torch.tensor([label for _, label in ds], dtype=torch.long)
        return xs, ys

    x_tr, y_tr = _flatten(train_set)
    x_te, y_te = _flatten(test_set)

    # 全画素強度の分布から thermometer しきい値を学習(1特徴として共有)
    encoder = ThermometerEncoder(bits=thermo_bits, mode='distributive')
    encoder.fit(x_tr.reshape(-1, 1))

    def _encode(x):  # (N, 784) → (N, 784, thermo_bits)
        n, t = x.shape
        code = encoder.forward(x.reshape(-1, 1))  # (N*784, thermo_bits)
        return code.reshape(n, t, thermo_bits)

    return (
        _encode(x_tr).to(device), y_tr.to(device),
        _encode(x_te).to(device), y_te.to(device),
        thermo_bits,
    )


# ============================================================
# 評価
# ============================================================
@torch.no_grad()
def eval_hard_acc(model, xs, y, device, chunk: int = 256) -> float:
    """hard eval(離散化 readout)での分類精度。"""
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, xs.shape[0], chunk):
        logits = model(xs[i:i + chunk].to(device))
        correct += (logits.argmax(-1) == y[i:i + chunk].to(device)).sum().item()
    model.train(mode=was_training)
    return correct / xs.shape[0]


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='LogicReservoir 訓練スクリプト(D3: 論理リザバーコンピューティング)'
    )
    parser.add_argument('--task', type=str, default='parity',
                        choices=['parity', 'smnist'], help='タスク種別')
    parser.add_argument('--max-steps', type=int, default=3000, help='最大ステップ数')
    parser.add_argument('--batch-size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.02, help='学習率(readout のみ)')
    parser.add_argument('--eval-freq', type=int, default=200, help='評価頻度(ステップ)')
    parser.add_argument('--seq-len', type=int, default=4,
                        help='parity の系列長(smnist は 784 固定)')
    parser.add_argument('--width', type=int, default=16,
                        help='parity の入力複製数(リザバーへの fan-out 確保)')
    parser.add_argument('--thermo-bits', type=int, default=3,
                        help='smnist の画素あたり thermometer ビット数')
    parser.add_argument('--reservoir-size', type=int, default=512,
                        help='リザバー状態のビット幅(parity は 512 以上を推奨)')
    parser.add_argument('--depth', type=int, default=2, help='リザバー内 LutLayer 段数')
    parser.add_argument('--n-inputs', type=int, default=2, help='LUT 入力数(2/4/6)')
    parser.add_argument('--leak', type=float, default=1.0,
                        help='leaky 統合係数(1.0=完全置換, <1.0 で soft leaky)')
    parser.add_argument('--readout', type=str, default='group_sum',
                        choices=['group_sum', 'lut'], help='readout 種別')
    parser.add_argument('--reservoir-seed', type=int, default=0,
                        help='リザバー回路の生成シード')
    parser.add_argument('--seed', type=int, default=0, help='学習ランダムシード')
    parser.add_argument('--device', type=str, default=None, help='デバイス(既定は自動検出)')
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help=f'データ保存先 (default: {_DEFAULT_DATA_DIR})')
    parser.add_argument('--log-dir', type=str, default='logs', help='ログ出力先')
    parser.add_argument('--no-wandb', action='store_true', help='wandb を無効化(スモーク用)')
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

    # --- データ・タスク設定 ---
    if args.task == 'parity':
        input_size = args.width
        num_classes = 2
        seq_len = args.seq_len
        eval_xs, eval_y = make_parity_eval(seq_len, args.width, device)
        data_gen = torch.Generator().manual_seed(args.seed + 1)
        print(f'\nタスク: 系列パリティ  seq_len={seq_len} width={args.width}')
    else:
        x_tr, y_tr, x_te, y_te, input_size = load_smnist(
            args.thermo_bits, device, data_dir=args.data_dir)
        num_classes = 10
        seq_len = x_tr.shape[1]
        eval_xs, eval_y = x_te, y_te
        print(f'\nタスク: Sequential-MNIST  T={seq_len} '
              f'thermo_bits={args.thermo_bits} input_size={input_size}')

    # --- モデル ---
    cfg = LogicReservoirConfig(
        input_size=input_size, reservoir_size=args.reservoir_size,
        num_classes=num_classes, depth=args.depth, n_inputs=args.n_inputs,
        leak=args.leak, readout=args.readout, seed=args.reservoir_seed,
    )
    model = LogicReservoir(cfg).to(device)
    n_readout = sum(p.numel() for p in model.trainable_parameters())
    n_reservoir = sum(p.numel() for p in model.reservoir.parameters())
    print(f'\nリザバー: size={cfg.reservoir_size} depth={cfg.depth} '
          f'n_inputs={cfg.n_inputs} leak={cfg.leak} readout={cfg.readout}')
    print(f'  凍結リザバーパラメータ: {n_reservoir:,}(学習しない)')
    print(f'  学習対象 readout パラメータ: {n_readout:,}')

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    # --- チェックポイント ---
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'logic_reservoir_{args.task}')

    # --- wandb ---
    use_wandb = False
    if not args.no_wandb:
        run_name = (f'reservoir_{args.task}_rs{args.reservoir_size}'
                    f'_d{args.depth}_leak{args.leak}_{args.readout}_s{args.max_steps}')
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=device,
            tags=['logic_reservoir', args.task, f'readout-{args.readout}',
                  f'leak{args.leak}', f'rs{args.reservoir_size}']
            + normalize_tags(model='logic_reservoir', dataset=args.task, seed=args.seed),
            group='logic_reservoir',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    # --- 訓練ループ(ステップベース、readout のみ更新) ---
    print(f'\n訓練開始: {args.max_steps}ステップ (評価={args.eval_freq}ステップごと)')
    print('=' * 70)
    best_acc = 0.0
    train_start = time.time()
    n_train = x_tr.shape[0] if args.task == 'smnist' else None

    for step in range(1, args.max_steps + 1):
        model.train()
        if args.task == 'parity':
            xs, y = sample_parity_batch(
                args.batch_size, seq_len, args.width, device, data_gen)
        else:
            idx = torch.randint(0, n_train, (args.batch_size,))
            xs, y = x_tr[idx], y_tr[idx]

        logits = model(xs)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_freq == 0 or step == args.max_steps or step == 1:
            acc = eval_hard_acc(model, eval_xs, eval_y, device)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                torch.save({'model_state_dict': model.state_dict(),
                            'config': vars(cfg), 'step': step, 'acc': acc},
                           best_path)

            elapsed = time.time() - train_start
            sec_per_step = elapsed / step
            remaining = sec_per_step * (args.max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            print(
                f'[step {step:>6d}/{args.max_steps}] '
                f'train_loss={loss.item():.4f} eval_acc={acc:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            if use_wandb:
                log_metrics({
                    'step': step, 'train_loss': loss.item(),
                    'eval_acc': acc, 'best_acc': best_acc,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 70)
    print(f'訓練完了: best_eval_acc={best_acc:.4f}  (総時間: {total_time})')

    # --- 重みを wandb Artifacts へ ---
    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'task': args.task, 'config': vars(cfg),
                      'best_acc': best_acc, 'max_steps': args.max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
