"""LogicMixer(MLP-Mixerの論理ゲート化)の訓練スクリプト。

MLP-Mixer(Tolstikhin 2021)のtoken-mixing/channel-mixing MLPをLutLayerスタック
に置換した論理ゲート版モデル(griffith_logic.models.mlp_mixer.LogicMixer)を
画像分類タスクで訓練する。世界初の論理ゲート版MLP-Mixer(調査時点)。

使用例:
    # スモーク(CPU で2分以内、wandb 無し)
    uv run python experiments/train_mlp_mixer.py --dataset mnist --max-steps 30 --no-wandb --device cpu

    # 本番(MNIST, wandb 有効)
    uv run python experiments/train_mlp_mixer.py --dataset mnist --max-steps 5000

    # CIFAR-10
    uv run python experiments/train_mlp_mixer.py --dataset cifar10 --max-steps 8000

    # Modal 実行
    uv run modal run scripts/modal_app.py --script train_mlp_mixer.py --args "--dataset mnist --max-steps 5000"
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
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
from wandb_utils import finish as wandb_finish

from griffith_logic.models.mlp_mixer import LogicMixer, LogicMixerConfig

load_dotenv()

# データセットごとの画像仕様(image_size, in_channels, num_classes, patch_sizeの既定)
_DATASET_SPECS = {
    'mnist': {'image_size': 28, 'in_channels': 1, 'num_classes': 10, 'patch_size': 7},
    'cifar10': {'image_size': 32, 'in_channels': 3, 'num_classes': 10, 'patch_size': 8},
}


# ============================================================
# データ読み込み(生ピクセル値[0,1]のまま。LogicMixer内部でthermometer符号化する)
# ============================================================
def load_vision_raw(
    dataset: str, device: str, data_dir: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_tr, y_tr, x_te, y_te) を返す。xは(N,C,H,W)実数[0,1]、二値化しない。

    vision.py の load_mnist/load_cifar10 はLUT回路向けに二値化/閾値化済みの
    ため、内部でthermometer符号化するLogicMixerには使えない
    (load_mnistはbinarize_threshold=Noneで生値を返せるため流用する)。
    """
    from griffith_logic.datasets.vision import load_mnist, resolve_data_root

    if dataset == 'mnist':
        train_set, test_set = load_mnist(root=data_dir, binarize_threshold=None)
    elif dataset == 'cifar10':
        try:
            import torchvision
        except ImportError as e:
            raise ImportError(
                'cifar10にはtorchvisionが必要です(uv sync --extra cpu または --extra cu128)'
            ) from e
        root = resolve_data_root('data-cifar', data_dir)
        transform = torchvision.transforms.ToTensor()
        train_set = torchvision.datasets.CIFAR10(
            root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f'未対応データセット: {dataset}')

    def _stack(ds):
        xs = torch.stack([img for img, _ in ds])
        ys = torch.tensor([label for _, label in ds], dtype=torch.long)
        return xs, ys

    x_tr, y_tr = _stack(train_set)
    x_te, y_te = _stack(test_set)
    return x_tr.to(device), y_tr.to(device), x_te.to(device), y_te.to(device)


# ============================================================
# 評価
# ============================================================
@torch.no_grad()
def _eval_acc(model, xs, y, device, *, hard: bool, chunk: int = 256) -> float:
    """分類精度。hard=Trueならeval(離散化)、Falseならtrain(緩和)経路で測る。

    LutLayer.forward は ``self.training`` で soft(緩和LUT) / hard(argmax離散)
    を切り替えるため、モード切替だけで soft/hard の両方が測れる
    (convert_arith_to_native_dlgn.py の soft_acc/hard_acc と同じ流儀)。
    """
    was_training = model.training
    model.train(mode=not hard)
    correct = 0
    for i in range(0, xs.shape[0], chunk):
        logits = model(xs[i:i + chunk].to(device))
        correct += (logits.argmax(-1) == y[i:i + chunk].to(device)).sum().item()
    model.train(mode=was_training)
    return correct / xs.shape[0]


def eval_hard_acc(model, xs, y, device, chunk: int = 256) -> float:
    """hard eval(離散化)での分類精度。"""
    return _eval_acc(model, xs, y, device, hard=True, chunk=chunk)


def eval_soft_acc(model, xs, y, device, chunk: int = 256) -> float:
    """soft(学習時の緩和経路)での分類精度。"""
    return _eval_acc(model, xs, y, device, hard=False, chunk=chunk)


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='LogicMixer 訓練スクリプト(MLP-Mixerの論理ゲート化)'
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(_DATASET_SPECS), help='データセット')
    parser.add_argument('--max-steps', type=int, default=3000, help='最大ステップ数')
    parser.add_argument('--batch-size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--eval-freq', type=int, default=200, help='評価頻度(ステップ)')
    parser.add_argument('--log-every-steps', type=int, default=None,
                        help='train lossをwandbへ記録するstep間隔。未指定なら実測s/stepから'
                             f'{int(LOSS_LOG_INTERVAL_SEC)}秒以内になる値を自動算出する'
                             '(評価頻度 --eval-freq とは独立)')
    parser.add_argument('--patch-size', type=int, default=None,
                        help='パッチ一辺の画素数(既定はデータセットごとの標準値)')
    parser.add_argument('--encode-bits', type=int, default=8,
                        help='画素あたりのthermometer符号化ビット数')
    parser.add_argument('--d-channels', type=int, default=64, help='パッチ表現のビット幅')
    parser.add_argument('--n-blocks', type=int, default=4, help='mixerブロック段数')
    parser.add_argument('--depth', type=int, default=2,
                        help='各mixing LutStackの段数')
    parser.add_argument('--n-inputs', type=int, default=2, help='LUT入力数(2/4/6)')
    parser.add_argument('--embed-hidden', type=int, default=None,
                        help='パッチ埋め込みを2段LUT化する際の中間幅(既定Noneで1段)')
    parser.add_argument('--embed-connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help="埋め込み1段目の接続方式('unique'は全入力ビット被覆を保証)")
    parser.add_argument('--residual-mode', type=str, default='xor',
                        choices=['xor', 'none', 'alternate'],
                        help="ブロックの残差方式('none'は残差なし=素の直列)")
    parser.add_argument('--residual-init', action='store_true',
                        help='混合LutLayerを恒等(素通し)近傍で初期化(深いn_blocksの学習開始対策)')
    parser.add_argument('--head-group-width', type=int, default=16,
                        help='分類ヘッドのクラスあたりゲート数(既定16。増やすとヘッドの入力被覆率が上がる)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'raw', 'warp'],
                        help='LUTパラメータ化方式')
    parser.add_argument('--seed', type=int, default=0, help='学習ランダムシード')
    parser.add_argument('--device', type=str, default=None, help='デバイス(既定は自動検出)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='データ保存先(既定はデータセットごとのrepo直下ディレクトリを自動発見)')
    parser.add_argument('--log-dir', type=str, default='logs', help='ログ出力先')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化(スモーク用)')
    parser.add_argument('--resume', type=str, default=None,
                        help='再開元チェックポイント(latest.pt)のパス')
    parser.add_argument('--wandb-group', type=str, default='mlp_mixer',
                        help='wandb group(既定 mlp_mixer)')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='run名に付ける識別子(スイープのアーム名等)')
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

    # --- データ ---
    spec = _DATASET_SPECS[args.dataset]
    patch_size = args.patch_size or spec['patch_size']
    print(f'\nデータセット: {args.dataset}  '
          f'image_size={spec["image_size"]} patch_size={patch_size}')
    x_tr, y_tr, x_te, y_te = load_vision_raw(args.dataset, device, args.data_dir)
    n_train = x_tr.shape[0]
    print(f'  train={n_train} test={x_te.shape[0]}')

    # --- モデル ---
    cfg = LogicMixerConfig(
        image_size=spec['image_size'], patch_size=patch_size,
        in_channels=spec['in_channels'], encode_bits=args.encode_bits,
        d_channels=args.d_channels, n_blocks=args.n_blocks, depth=args.depth,
        n_inputs=args.n_inputs, parameterization=args.parameterization,
        num_classes=spec['num_classes'],
        embed_hidden=args.embed_hidden, embed_connections=args.embed_connections,
        head_group_width=args.head_group_width,
        residual_init=args.residual_init, residual_mode=args.residual_mode,
    )
    model = LogicMixer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    gate_info = count_logic_gates(model)
    n_gates = int(gate_info['gates/total'])
    print(f'\nLogicMixer: d_tokens={cfg.d_tokens} d_channels={cfg.d_channels} '
          f'n_blocks={cfg.n_blocks} depth={cfg.depth} n_inputs={cfg.n_inputs} '
          f'param={cfg.parameterization}')
    print(f'  総パラメータ数: {n_params:,}')
    # ユニークゲート数(重み共有のtoken-mix/patch_embedは1回だけ数える)
    print(f'  ユニークゲート数: {n_gates:,}  ({gate_info.get("gates/lut_type", "?")})')
    embed_coverage = model.embed_input_coverage()
    head_coverage = model.head_input_coverage()
    print(f'  パッチ埋め込み: {"2段" if args.embed_hidden else "1段"}'
          f'(hidden={args.embed_hidden}, conn={args.embed_connections}) '
          f'入力被覆率={embed_coverage:.3f}')
    print(f'  分類ヘッド: group_width={args.head_group_width} '
          f'入力被覆率={head_coverage:.3f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- チェックポイント ---
    run_dir, latest_path, best_path = setup_checkpoint_dir(f'mlp_mixer_{args.dataset}')

    # --- --resume: 既存チェックポイントから再開 ---
    start_step = 0
    best_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = int(ckpt['step'])
        best_acc = float(ckpt['best_metric'])
        print(f'\n再開: {args.resume} (step={start_step}, best_hard_acc={best_acc:.4f})')

    # --- wandb ---
    use_wandb = False
    if not args.no_wandb:
        tag_part = f'_{args.run_tag}' if args.run_tag else ''
        run_name = (f'mlp_mixer_{args.dataset}{tag_part}_p{patch_size}_dc{args.d_channels}'
                    f'_b{args.n_blocks}_d{args.depth}_n{args.n_inputs}_s{args.max_steps}')
        config_dict = dict(vars(args))
        config_dict['n_gates'] = n_gates
        config_dict['n_params'] = n_params
        config_dict['embed_input_coverage'] = embed_coverage
        config_dict['head_input_coverage'] = head_coverage
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=config_dict,
            model=model,
            device=device,
            tags=['mlp_mixer', args.dataset, f'n{args.n_inputs}', args.parameterization]
            + normalize_tags(model='mlp_mixer', dataset=args.dataset, seed=args.seed),
            group=args.wandb_group,
            job_type='train',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    # --- 訓練ループ(ステップベース) ---
    print(f'\n訓練開始: {args.max_steps}ステップ (評価={args.eval_freq}ステップごと)')
    print('=' * 70)
    train_start = time.time()
    gen = torch.Generator().manual_seed(args.seed + 1)
    # resume時は乱数列も進めて同じミニバッチ列の再消費を避ける
    for _ in range(start_step):
        torch.randint(0, n_train, (args.batch_size,), generator=gen)
    last_stats: dict[str, float] = {}
    # 軽量ログ(SKILL.md §11: train lossは壁時計で最低5分に1回)。
    # 評価頻度(--eval-freq)とは独立に、実測s/stepから決めた間隔でlossと進捗を出す。
    pacer = LossLogPacer(every_steps=args.log_every_steps)
    if pacer.every_steps is not None:
        print(pacer.describe())

    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        idx = torch.randint(0, n_train, (args.batch_size,), generator=gen)
        xs, y = x_tr[idx], y_tr[idx]

        logits = model(xs)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        done = step - start_step
        light_metrics = None
        if pacer.should_log(done, time.time() - train_start,
                            is_last=step == args.max_steps):
            if pacer.resolved_now:
                print(pacer.describe(), flush=True)
            light_elapsed = time.time() - train_start
            light_metrics = {
                'step': step,
                'train_loss': loss.item(),
                'steps_per_sec': done / max(light_elapsed, 1e-9),
                **progress_metrics(done, step, args.max_steps, light_elapsed),
            }

        if step % args.eval_freq == 0 or step == args.max_steps or step == 1:
            # soft(緩和)/hard(離散)を同一テストセットで測り、離散化ギャップを分離記録する
            hard_acc = eval_hard_acc(model, x_te, y_te, device)
            soft_acc = eval_soft_acc(model, x_te, y_te, device)
            gap = soft_acc - hard_acc
            acc = hard_acc  # 主指標はhard acc(実回路で得られる精度)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                torch.save({'model_state_dict': model.state_dict(),
                            'config': vars(cfg), 'step': step, 'acc': acc,
                            'soft_acc': soft_acc, 'gap': gap, 'n_gates': n_gates},
                           best_path)
            save_checkpoint(
                latest_path, model, optimizer, None, step, best_acc,
                metadata={'args': vars(args), 'config': vars(cfg), 'n_gates': n_gates},
            )

            elapsed = time.time() - train_start
            done = step - start_step
            sec_per_step = elapsed / max(done, 1)
            remaining = sec_per_step * (args.max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            print(
                f'[step {step:>6d}/{args.max_steps}] '
                f'train_loss={loss.item():.4f} '
                f'hard_acc={hard_acc:.4f} soft_acc={soft_acc:.4f} gap={gap:+.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            # 同じstepの軽量ログは1レコードに合流させる(二重ログを避ける)
            last_stats = {
                **(light_metrics or {}),
                'step': step, 'train_loss': loss.item(),
                'eval_acc': acc, 'best_acc': best_acc,
                'hard_acc': hard_acc, 'soft_acc': soft_acc,
                'discretization_gap': gap, 'n_gates': n_gates,
                'steps_per_sec': done / max(elapsed, 1e-9),
            }
            if use_wandb:
                log_metrics(last_stats)
                light_metrics = None  # 合流済み

        # 評価が無いstepの軽量ログ(step引数なし=wandbの自動stepに従う)
        if use_wandb and light_metrics is not None:
            log_metrics(light_metrics)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 70)
    print(f'訓練完了: best_hard_acc={best_acc:.4f} '
          f'(最終 soft={last_stats.get("soft_acc", float("nan")):.4f} '
          f'gap={last_stats.get("discretization_gap", float("nan")):+.4f}) '
          f'gates={n_gates:,}  (総時間: {total_time})')

    # --- 重みを wandb Artifacts へ ---
    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'dataset': args.dataset, 'config': vars(cfg),
                      'best_acc': best_acc, 'n_gates': n_gates,
                      'max_steps': args.max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
