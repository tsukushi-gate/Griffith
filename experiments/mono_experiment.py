"""
単調性仮説検証（AP法 vs BP法 × 深さ）の観測実験ドライバ。

仮説:
    「AP法（誤差ブロードキャスト型学習則）の成功条件はネットワークの単調性」
    同一アーキのフラットMLPを {AP法, BP法} × 深さ で学習し、学習中の
    単調度V・極性一貫性Cと最終精度を記録して仮説を定量検証する。

設計:
    - AP法とBP法を同一アーキ（parameterization='independent'）で比較し、
      差分を学習則だけに絞る。V/Cは同一定義（monotonicity.py）で測る。
    - AP法: APTrainer（autogradなし、weightsを生[0,1]確率として更新）。
    - BP法: Adam + CrossEntropy（independentはforwardでsigmoidを通す）。
    - 単調性指標の読み取りは ap_mode で切替（AP=生確率, BP=sigmoid）。

使用例:
    uv run python experiments/mono_experiment.py --method ap --depth 4 -k 2000 -e 15
    uv run python experiments/mono_experiment.py --method bp --depth 4 -k 2000 -e 15
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_interventions import DFAliteAPTrainer, MonotoneAPTrainer
from ap_trainer import APTrainer
from env_utils import load_dotenv
from monotonicity import MonotonicityRecorder, collect_lut_layers
from train import (
    DATASET_INFO,
    build_model,
    evaluate,
    load_dataset,
    setup_logger,
)
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_metrics,
)

load_dotenv()

# 崩壊判定: 最終精度が (この係数 / クラス数) 未満なら「ほぼランダム=崩壊」
COLLAPSE_RANDOM_FACTOR = 1.2
# AP法系(誤差ブロードキャスト型)の method 集合。BP法と挙動を分岐する。
AP_METHODS = {'ap', 'ap_monotone', 'ap_dfalite'}
# method依存の既定学習率(AP法系は同一lr=0.1)
DEFAULT_LR = {'ap': 0.1, 'bp': 0.01, 'ap_monotone': 0.1, 'ap_dfalite': 0.1}


def build_ap_trainer(method: str, model, lr: float):
    """method に応じた AP法系トレーナーを生成する。"""
    if method == 'ap_monotone':
        return MonotoneAPTrainer(model, lr=lr, update_mode=1)
    if method == 'ap_dfalite':
        return DFAliteAPTrainer(model, lr=lr, update_mode=1)
    return APTrainer(model, lr=lr, update_mode=1)


def resolve_device(device: str | None) -> str:
    """デバイス自動検出（cuda → mps → cpu）。"""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int) -> None:
    """torch/numpy/random 全てのシードを固定する。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_args(args: argparse.Namespace) -> None:
    """build_model/load_dataset が要求する属性を args に揃える。"""
    # --depth は num_layers として解釈する
    args.num_layers = args.depth

    # AP法系は independent パラメータ化を強制する
    if args.method in AP_METHODS and args.parameterization != 'independent':
        print(
            f'警告: AP法は parameterization=independent を要求します。'
            f'{args.parameterization} → independent に上書きします。'
        )
        args.parameterization = 'independent'

    # method依存の学習率既定
    if args.lr is None:
        args.lr = DEFAULT_LR[args.method]

    # build_model が warp 属性を参照する場合に備えて AttributeError を回避する
    # （independent 時は参照されないが、安全のため既定値を持たせる）
    if not hasattr(args, 'tau_warp'):
        args.tau_warp = 1.0
    if not hasattr(args, 'no_gumbel'):
        args.no_gumbel = False


def _record_to_history(rec: dict) -> dict:
    """recorder の返す dict を JSON/history 用スキーマへ整形する。"""
    return {
        'step': rec['step'],
        'V_mean': rec['V_mean'],
        'V_per_layer': rec['V_per_layer'],
        'C_mean_frac_positive': rec['C']['mean_frac_positive'],
        'C_mean_frac_mixed': rec['C']['mean_frac_mixed'],
    }


def log_record(
    rec: dict,
    step: int,
    train_loss: float | None,
    train_acc: float | None,
    use_wandb: bool,
) -> None:
    """単調性記録を wandb にログする（train指標は学習開始後のみ付与）。"""
    if not use_wandb:
        return
    metrics: dict = {
        'V_mean': rec['V_mean'],
        'C_mean_frac_positive': rec['C']['mean_frac_positive'],
        'C_mean_frac_mixed': rec['C']['mean_frac_mixed'],
    }
    for i, v in enumerate(rec['V_per_layer']):
        metrics[f'V_layer{i}'] = v
    if train_loss is not None:
        metrics['train_loss'] = train_loss
    if train_acc is not None:
        metrics['train_acc'] = train_acc
    log_metrics(metrics, step=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='単調性仮説検証（AP法 vs BP法 × 深さ）観測実験ドライバ'
    )
    parser.add_argument('--method', type=str, required=True,
                        choices=['ap', 'bp', 'ap_monotone', 'ap_dfalite'],
                        help='学習則（必須）。ap_monotone=単調制約AP, ap_dfalite=対照')
    parser.add_argument('--depth', type=int, default=4,
                        help='LUT層数（=num_layers, default: 4）')
    parser.add_argument('-k', '--num-neurons', type=int, default=2000,
                        help='層あたりニューロン数（default: 2000）')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数（default: 2）')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(DATASET_INFO.keys()),
                        help='データセット（default: mnist）')
    parser.add_argument('-e', '--epochs', type=int, default=15,
                        help='エポック数（default: 15）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率（既定: AP=0.1 / BP=0.01）')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ（default: 128）')
    parser.add_argument('-t', '--tau', type=float, default=10.0,
                        help='GroupSum温度（default: 10.0）')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード（default: 0）')
    parser.add_argument('--interval', type=int, default=50,
                        help='単調性記録のstep間隔（default: 50）')
    parser.add_argument('--n-paths', type=int, default=1000,
                        help='極性一貫性Cの経路サンプル数（default: 1000）')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='接続方式（default: random）')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数（default: 1.0）')
    parser.add_argument('--parameterization', type=str, default='independent',
                        choices=['softmax', 'independent', 'iwp', 'warp'],
                        help='パラメータ化方式（default: independent）')
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='1エポックのバッチ数上限（スモーク用, default: 無制限）')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス（default: 自動 cuda→mps→cpu）')
    parser.add_argument('--out-dir', type=str, default='results/mono',
                        help='結果JSON出力先（default: results/mono）')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化する')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_args(args)

    args.device = resolve_device(args.device)
    set_seed(args.seed)
    print(f'デバイス: {args.device}  method={args.method}  '
          f'depth={args.num_layers}  k={args.num_neurons}  lr={args.lr}')

    num_classes = DATASET_INFO[args.dataset]['num_classes']

    # データ・モデル構築
    train_loader, test_loader = load_dataset(args)
    model = build_model(args)

    # AP法系は APTrainer 系（内部で weights を[0,1]一様乱数へ再初期化）
    trainer = None
    optimizer = None
    loss_fn = None
    if args.method in AP_METHODS:
        trainer = build_ap_trainer(args.method, model, args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

    # 単調性記録フック（trainer初期化後の重みで初期状態を測るため後に生成）
    recorder = MonotonicityRecorder(
        model,
        collect_lut_layers(model),
        ap_mode=(args.method in AP_METHODS),
        interval=args.interval,
        n_paths=args.n_paths,
    )

    # wandb
    run_name = (f'mono_{args.method}_d{args.num_layers}'
                f'_k{args.num_neurons}_s{args.seed}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=[args.method, f'depth{args.num_layers}',
                  args.dataset, f'seed{args.seed}'],
            group='monotonicity',
        )

    # ログファイル
    log, log_file = setup_logger(f'mono_{args.method}', args.dataset)
    log(f'START method={args.method} depth={args.num_layers} '
        f'k={args.num_neurons} n_inputs={args.n_inputs} lr={args.lr} '
        f'epochs={args.epochs} seed={args.seed} device={args.device}')

    history: list[dict] = []
    epoch_acc: list[dict] = []

    # 学習開始前（step=0）に初期状態を記録
    global_step = 0
    rec = recorder.maybe_record(global_step)
    if rec is not None:
        history.append(_record_to_history(rec))
        log_record(rec, global_step, None, None, use_wandb)
        print(f'[step 0] V_mean={rec["V_mean"]:.4f} '
              f'C_pos={rec["C"]["mean_frac_positive"]:.4f} '
              f'C_mixed={rec["C"]["mean_frac_mixed"]:.4f}')

    final_soft = 0.0
    final_hard = 0.0
    best_hard = 0.0
    final_train_acc = 0.0

    # 学習ループ（global step をバッチ横断で加算）
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            if args.method in AP_METHODS:
                metrics = trainer.train_step(x, y)
                batch_loss = metrics['loss']
                batch_acc = metrics['accuracy']
            else:
                model.train()
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                batch_acc = (logits.argmax(dim=-1) == y).float().mean().item()

            global_step += 1
            n_batches += 1
            bs = y.shape[0]
            epoch_loss += batch_loss
            epoch_correct += batch_acc * bs
            epoch_total += bs

            rec = recorder.maybe_record(global_step)
            if rec is not None:
                history.append(_record_to_history(rec))
                log_record(rec, global_step, batch_loss, batch_acc, use_wandb)

            if args.max_train_batches is not None and n_batches >= args.max_train_batches:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        final_train_acc = train_acc

        # エポックごと評価: soft（連続値forward）/ hard（離散化）
        if args.method in AP_METHODS:
            soft = trainer.evaluate(test_loader, args.device, binarize=False)
            hard = trainer.evaluate(test_loader, args.device, binarize=True)
        else:
            soft = evaluate(model, test_loader, args.device, mode=True)
            hard = evaluate(model, test_loader, args.device, mode=False)

        final_soft = soft
        final_hard = hard
        best_hard = max(best_hard, hard)
        epoch_acc.append({'epoch': epoch, 'soft': soft, 'hard': hard})

        print(f'[epoch {epoch:>3d}/{args.epochs}] loss={avg_loss:.4f} '
              f'train_acc={train_acc:.4f} soft={soft:.4f} hard={hard:.4f} '
              f'best_hard={best_hard:.4f}')
        log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
            f'train_acc={train_acc:.4f} soft={soft:.4f} hard={hard:.4f} '
            f'best_hard={best_hard:.4f}')
        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'test_soft': soft,
                'test_hard': hard,
                'best_hard': best_hard,
                'train_acc_epoch': train_acc,
            }, step=global_step)

    # 崩壊判定: 最終 train_acc または test soft が (1.2/クラス数) 未満
    collapse_threshold = COLLAPSE_RANDOM_FACTOR / num_classes
    collapsed = bool(
        final_train_acc < collapse_threshold or final_soft < collapse_threshold
    )

    # 結果JSON出力
    result = {
        'method': args.method,
        'depth': args.num_layers,
        'seed': args.seed,
        'k': args.num_neurons,
        'n_inputs': args.n_inputs,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'parameterization': args.parameterization,
        'final_soft': final_soft,
        'final_hard': final_hard,
        'best_hard': best_hard,
        'collapsed': collapsed,
        'history': history,
        'epoch_acc': epoch_acc,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f'mono_{args.method}_d{args.num_layers}_k{args.num_neurons}_s{args.seed}.json',
    )
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'\n結果JSON: {out_path}')
    print(f'final_soft={final_soft:.4f} final_hard={final_hard:.4f} '
          f'best_hard={best_hard:.4f} collapsed={collapsed} '
          f'（記録点数={len(history)}）')
    log(f'DONE final_soft={final_soft:.4f} final_hard={final_hard:.4f} '
        f'best_hard={best_hard:.4f} collapsed={collapsed} out={out_path}')
    log_file.close()

    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
