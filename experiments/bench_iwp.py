"""IWP-1 ベンチ駆動スクリプト: OP vs IWP の同一条件比較。

論文 "Light Differentiable Logic Gate Networks" (arXiv:2510.03250, ICLR 2026) の
**IWP (Input-Wise Parametrization)** を当ライブラリで追試するためのベンチ足場。

アーム定義(事前監査の結論に従う):
    OP  — Original Parametrization = `parameterization='softmax'`(16ゲートsoftmax)
    IWP — 論文IWP = `parameterization='independent'` + `estimator='sin'`
          + `residual_init=True`。数学的に論文 Eq.11 と同一。
          **既存の `parameterization='iwp'`(リテラル積)は論文IWPとは別物であり、
          本ベンチの対象ではない**(混同禁止)。

収集する指標(すべて実測値。推定・補間・創作は一切しない):
    1. 最終 hard test accuracy / soft test accuracy
    2. discretization gap = soft acc − hard acc
    3. 収束速度 = 「同一seedのOPが到達した最高 hard acc」に各armが最初に到達した step 数
       (論文の 8.5x 主張の追試)。未到達なら null を記録し、決して埋めない
    4. step あたり wall-clock の中央値 / 総 wall-clock
    5. LUT重みの要素数(パラメータ数)とピークメモリ(取得できない環境では null)

!!! 重要な但し書き(wall-clock比較の公平性) !!!
    `estimator='sin'` は CUDA カーネル経路(sigmoid固定)を使えず Python フォールバック
    になる。したがって **wall-clock の OP vs IWP 比較は、推定器を揃えない限り不公平**
    である(IWP側が不利に出る)。一方 **収束 step 数の比較は step 基準なので
    この影響を受けない**。速度主張を読むときは必ずこの点を考慮すること。

使用例:
    uv run python experiments/bench_iwp.py --task xor --arm both --seeds 0 --dry-run
    uv run python experiments/bench_iwp.py --task xor --arm both --seeds 0 1 2
    uv run python experiments/bench_iwp.py --task mnist2k --arm both --seeds 0 1 2
    # GPU実行(Modal):
    uv run modal run --detach scripts/modal_app.py --background \
        --script bench_iwp.py --args "--task mnist8k --arm both --seeds 0 1 2"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# experiments/ を単体実行したときもリポジトリ直下をimport起点にする
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / 'experiments') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'experiments'))

from env_utils import load_dotenv  # noqa: E402
from wandb_utils import (  # noqa: E402
    add_experiment_args,
    finish_wandb,
    init_wandb,
    log_artifact,
    log_images,
    log_metrics,
    normalize_tags,
)

from griffith_logic.group_sum import GroupSum  # noqa: E402
from griffith_logic.layers.lut import LutLayer  # noqa: E402
from griffith_logic.training.config import DataConfig  # noqa: E402
from griffith_logic.training.runner import _resolve_data  # noqa: E402
from griffith_logic.training.tasks import make_task  # noqa: E402

JST = timezone(timedelta(hours=9))

# wandb group はIWP-1レーン固定(taxonomy規約: group=モデル族/実験系列)
WANDB_GROUP = 'iwp-1'

# タスク名 → (OPレシピ, IWPレシピ)。レシピが唯一のハイパーパラメータ真実源。
RECIPE_DIR = _REPO_ROOT / 'configs' / 'recipes'
TASKS: dict[str, tuple[str, str]] = {
    'xor': ('iwp1_xor_op.yaml', 'iwp1_xor_iwp.yaml'),
    'mnist2k': ('iwp1_mnist_2k_op.yaml', 'iwp1_mnist_2k_iwp.yaml'),
    'mnist8k': ('iwp1_mnist_8k_op.yaml', 'iwp1_mnist_8k_iwp.yaml'),
}

# 結果の保存先(相対パス規約: Modalでも無改造で動く)
RESULT_DIR = Path('results') / 'iwp1'

# 比較の対象外にしてよいOP/IWP差分キー(これ以外が食い違ったら不公平比較として即エラー)
ARM_DIFF_KEYS = frozenset({'parameterization', 'estimator', 'residual_init',
                           'ri_mu', 'ri_sigma'})


def auto_device() -> str:
    """デバイス自動検出(cuda→mps→cpu)。CPU固定のハードコードはしない。"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int) -> None:
    """torch/numpy/randomのseedを固定する。"""
    import random

    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def eta_line(elapsed: float, remaining: float) -> str:
    """`[経過<残り, 終了予定 HH:MM:SS]` 形式の進捗サフィックスを返す(JST)。"""
    eta = datetime.now(JST) + timedelta(seconds=max(0.0, remaining))
    return (f'[{timedelta(seconds=int(elapsed))}<{timedelta(seconds=int(remaining))}, '
            f'終了予定 {eta.strftime("%H:%M:%S")}]')


def load_recipe(path: Path) -> dict:
    """レシピYAMLを素のdictとして読む。

    pydantic の TrainConfig を通さないのは、`model.estimator` /
    `model.residual_init` がまだ ModelConfig(extra='forbid')に無く、
    並行レーンの実装完了を待たずにベンチを動かせるようにするため。
    """
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'レシピのルートはマッピングが必要です: {path}')
    return cfg


def assert_fair_pair(op_cfg: dict, iwp_cfg: dict) -> None:
    """OP/IWPレシピの差分が ARM_DIFF_KEYS のみであることを検証する(不公平比較の防止)。"""
    for section in ('data', 'optim', 'run'):
        a, b = op_cfg.get(section, {}), iwp_cfg.get(section, {})
        if a != b:
            raise ValueError(
                f'OP/IWPレシピの {section} が一致しません(不公平比較): '
                f'OP={a} / IWP={b}'
            )
    a, b = dict(op_cfg.get('model', {})), dict(iwp_cfg.get('model', {}))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    illegal = diff - ARM_DIFF_KEYS
    if illegal:
        raise ValueError(
            f'OP/IWPレシピの model 差分が許容キー外です(不公平比較): {sorted(illegal)}'
        )


def build_model(mcfg: dict, data, device: str) -> nn.Sequential:
    """レシピの model セクションからモデルを構築する。

    runner._build_model と同じ構造(Flatten + LutLayer×layers [+ GroupSum])だが、
    論文IWP用の estimator / residual_init / ri_mu / ri_sigma を LutLayer へ渡す。
    """
    kind = mcfg.get('kind', 'dlgn')
    k = int(mcfg['k'])
    n_layers = int(mcfg['layers'])
    llkw = dict(
        n_inputs=int(mcfg.get('n_inputs', 2)),
        device=device,
        connections=mcfg.get('connections', 'random'),
        parameterization=mcfg.get('parameterization', 'softmax'),
        estimator=mcfg.get('estimator', 'sigmoid'),
        residual_init=bool(mcfg.get('residual_init', False)),
        ri_mu=float(mcfg.get('ri_mu', 1.2)),
        ri_sigma=float(mcfg.get('ri_sigma', 0.25)),
    )
    layers: list[nn.Module] = [nn.Flatten()]
    if kind == 'dlgn':
        if data.kind != 'classification':
            raise ValueError(f'dlgnはclassification専用: {data.name}({data.kind})')
        if k % data.out_width != 0:
            raise ValueError(f'k={k} はクラス数 {data.out_width} で割り切れる必要があります')
        d = data.in_width
        for _ in range(n_layers):
            layers.append(LutLayer(d, k, **llkw))
            d = k
        layers.append(GroupSum(data.out_width, tau=float(mcfg.get('tau', 1.0))))
    elif kind == 'logic_mlp':
        if data.kind != 'binary_vector':
            raise ValueError(f'logic_mlpはbinary_vector専用: {data.name}({data.kind})')
        d = data.in_width
        for _ in range(n_layers - 1):
            layers.append(LutLayer(d, k, **llkw))
            d = k
        layers.append(LutLayer(d, data.out_width, **llkw))
    else:
        raise ValueError(f'未対応の model.kind: {kind}')
    return nn.Sequential(*layers).to(device)


def lut_param_count(model: nn.Module) -> int:
    """LUT重み(LutLayer.weights)の要素数を合算する。"""
    return sum(m.weights.numel() for m in model.modules() if isinstance(m, LutLayer))


def peak_memory_bytes(device: str) -> int | None:
    """ピークメモリ(bytes)。取得できない環境では None を返す(推定はしない)。"""
    if device == 'cuda':
        return int(torch.cuda.max_memory_allocated())
    if device == 'mps' and hasattr(torch, 'mps'):
        fn = getattr(torch.mps, 'driver_allocated_memory', None)
        if callable(fn):
            return int(fn())
    return None


def run_suffix(epochs_override: int | None) -> str:
    """成果物名のサフィックス。

    `--epochs` でエポックを縮めたスモークは本走と別名にし、スモーク結果が
    本走の正本JSON/図を上書きして混同されるのを防ぐ。
    """
    return '' if epochs_override is None else f'_smoke_ep{epochs_override}'


def _eval_pass(model: nn.Module, loader, task, device: str, *, hard: bool) -> dict:
    """評価パス。

    hard=True  — 離散化評価(eval mode = 重みargmax + 入力round)
    hard=False — soft評価(train mode = soft重み、入力はround無し)
    """
    was_training = model.training
    model.train(mode=not hard)
    sums: dict[str, float] = {}
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device)
            if hard:
                x = x.round()
            y = y.to(device)
            out = model(x)
            if not torch.isfinite(out).all():
                raise RuntimeError('評価出力に NaN/Inf が発生しました(発散)')
            metrics = task.metrics(out, y)
            n = y.shape[0]
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * n
            total += n
    model.train(mode=was_training)
    return {name: value / total for name, value in sums.items()}


def run_one(task_name: str, arm: str, seed: int, cfg: dict, args) -> dict:
    """1条件(task × arm × seed)を学習して実測メトリクスを返す。"""
    mcfg = cfg.get('model', {})
    rcfg = dict(cfg.get('run', {}))
    ocfg = cfg.get('optim', {})
    epochs = int(args.epochs) if args.epochs is not None else int(rcfg.get('epochs', 3))
    eval_interval = int(rcfg.get('eval_interval', 0))
    device = args.device or auto_device()

    set_seed(seed)
    data = _resolve_data(
        DataConfig(**cfg.get('data', {})), device,
    )
    task = make_task(data.kind, rcfg.get('loss'))
    model = build_model(mcfg, data, device)
    optims = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}
    optimizer = optims[ocfg.get('name', 'adam')](
        model.parameters(), lr=float(ocfg.get('lr', 0.01)),
    )

    steps_per_epoch = len(data.train_loader)
    total_steps = epochs * steps_per_epoch
    run_name = f'{task_name}_{arm}_k{mcfg.get("k")}_l{mcfg.get("layers")}_s{seed}'
    n_lut_params = lut_param_count(model)

    print(f'--- {task_name} / arm={arm} / seed={seed} ---')
    print(f'device={device}')
    print(f'データセット: {data.name}({data.kind}, in={data.in_width}, out={data.out_width})')
    print(f'モデル: {mcfg.get("kind")} k={mcfg.get("k")} layers={mcfg.get("layers")} '
          f'param={mcfg.get("parameterization")} estimator={mcfg.get("estimator", "sigmoid")} '
          f'residual_init={bool(mcfg.get("residual_init", False))}')
    print(f'LUT重み要素数: {n_lut_params}')
    print(f'学習: {epochs}エポック × {steps_per_epoch}ステップ = {total_steps}ステップ')

    use_wandb = bool(rcfg.get('wandb', False)) or args.wandb
    wandb_on = False
    if use_wandb:
        wandb_on = init_wandb(
            run_name=run_name,
            config_dict={'task': task_name, 'arm': arm, 'seed': seed,
                         'epochs': epochs, 'recipe': cfg},
            model=model,
            device=device,
            tags=['iwp', f'arm:{arm}', f'task:{task_name}']
            + normalize_tags(model=str(mcfg.get('parameterization')),
                             dataset=data.name, seed=seed, series='IWP-1'),
            group=WANDB_GROUP,
            job_type='train',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    history: list[dict] = []
    step_times: list[float] = []
    step = 0
    last_loss = float('nan')
    best_hard = 0.0
    start = time.time()
    last_print = start

    def do_eval(epoch: int) -> None:
        nonlocal best_hard
        hard = _eval_pass(model, data.eval_loader, task, device, hard=True)
        soft = _eval_pass(model, data.eval_loader, task, device, hard=False)
        best_hard = max(best_hard, hard['acc'])
        entry = {
            'step': step, 'epoch': epoch, 'loss': last_loss,
            'hard_acc': hard['acc'], 'soft_acc': soft['acc'],
            'gap': soft['acc'] - hard['acc'],
            'elapsed_sec': time.time() - start,
        }
        for name, value in hard.items():
            if name != 'acc':
                entry[f'hard_{name}'] = value
        for name, value in soft.items():
            if name != 'acc':
                entry[f'soft_{name}'] = value
        history.append(entry)
        if wandb_on:
            log_metrics({f'{k}': v for k, v in entry.items()})

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in data.train_loader:
                t0 = time.time()
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                out = model(x)
                loss = task.loss_fn(out, y)
                if not math.isfinite(loss.item()):
                    raise RuntimeError(
                        f'損失が NaN/Inf になりました(step={step}, arm={arm}, seed={seed})'
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_times.append(time.time() - t0)
                step += 1
                last_loss = loss.item()

                now = time.time()
                if now - last_print >= 10.0:
                    elapsed = now - start
                    print(f'[step {step}/{total_steps}] loss={last_loss:.4f}  '
                          + eta_line(elapsed, elapsed / step * (total_steps - step)))
                    last_print = now

                if eval_interval > 0 and step % eval_interval == 0:
                    do_eval(epoch)
                    model.train()

            if eval_interval == 0 or step % eval_interval != 0:
                do_eval(epoch)
                model.train()
            elapsed = time.time() - start
            print(f'[epoch {epoch}/{epochs}] loss={last_loss:.4f}  '
                  f'hard_acc={history[-1]["hard_acc"]:.4f}  '
                  f'soft_acc={history[-1]["soft_acc"]:.4f}  '
                  f'gap={history[-1]["gap"]:.4f}  '
                  + eta_line(elapsed, elapsed / epoch * (epochs - epoch)))
    except Exception:
        if wandb_on:
            finish_wandb(status='failed')
        raise

    total_wall = time.time() - start
    final = history[-1]
    result = {
        'task': task_name, 'arm': arm, 'seed': seed, 'device': device,
        'run_name': run_name,
        'recipe': cfg,
        'epochs': epochs, 'steps': step, 'steps_per_epoch': steps_per_epoch,
        # True なら --epochs でエポックを縮めたスモーク結果(本走の値ではない)
        'epochs_overridden': args.epochs is not None,
        'final_hard_acc': final['hard_acc'],
        'final_soft_acc': final['soft_acc'],
        'discretization_gap': final['gap'],
        'best_hard_acc': best_hard,
        'final_loss': last_loss,
        'total_wall_sec': total_wall,
        'median_step_sec': statistics.median(step_times) if step_times else None,
        'lut_param_count': n_lut_params,
        'peak_memory_bytes': peak_memory_bytes(device),
        'history': history,
        'finished_at': datetime.now(JST).isoformat(),
    }

    suffix = run_suffix(args.epochs)
    out_path = RESULT_DIR / f'{task_name}_{arm}_seed{seed}{suffix}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'結果を保存: {out_path}')

    if wandb_on:
        log_artifact(str(out_path), name=f'iwp1-{task_name}-{arm}-seed{seed}{suffix}',
                     type='bench-result', aliases=['latest'],
                     metadata={'task': task_name, 'arm': arm, 'seed': seed})
        finish_wandb(status='finished', final_metrics={
            k: v for k, v in result.items() if isinstance(v, int | float)
        })
    return result


def steps_to_target(history: list[dict], target: float) -> int | None:
    """hard_acc が target 以上になった最初の step。未到達なら None(埋めない)。"""
    for entry in history:
        if entry['hard_acc'] >= target:
            return int(entry['step'])
    return None


def summarize(results: dict[tuple[str, int], dict], task_name: str) -> dict:
    """OP/IWPの実測値から比較サマリを組み立てる(推定値は一切入れない)。

    収束速度は seed ごとに「同一seedのOPの最高 hard acc」を目標値として、
    各armが最初にそれへ到達した step を測る。OPアームが無い場合は null。
    """
    seeds = sorted({s for (_, s) in results})
    per_seed = []
    for seed in seeds:
        op = results.get(('op', seed))
        iwp = results.get(('iwp', seed))
        target = op['best_hard_acc'] if op is not None else None
        row: dict = {'seed': seed, 'op_best_hard_acc_target': target}
        for arm, res in (('op', op), ('iwp', iwp)):
            if res is None:
                row[arm] = None
                continue
            row[arm] = {
                'final_hard_acc': res['final_hard_acc'],
                'final_soft_acc': res['final_soft_acc'],
                'discretization_gap': res['discretization_gap'],
                'best_hard_acc': res['best_hard_acc'],
                'steps_to_op_best': (
                    None if target is None
                    else steps_to_target(res['history'], target)
                ),
                'total_wall_sec': res['total_wall_sec'],
                'median_step_sec': res['median_step_sec'],
                'lut_param_count': res['lut_param_count'],
                'peak_memory_bytes': res['peak_memory_bytes'],
            }
        per_seed.append(row)

    def mean_of(arm: str, key: str) -> float | None:
        vals = [r[arm][key] for r in per_seed
                if r.get(arm) and r[arm].get(key) is not None]
        return statistics.mean(vals) if vals else None

    aggregate = {}
    for arm in ('op', 'iwp'):
        aggregate[arm] = {
            key: mean_of(arm, key) for key in (
                'final_hard_acc', 'final_soft_acc', 'discretization_gap',
                'best_hard_acc', 'steps_to_op_best', 'total_wall_sec',
                'median_step_sec',
            )
        }
    op_steps = aggregate['op']['steps_to_op_best']
    iwp_steps = aggregate['iwp']['steps_to_op_best']
    aggregate['speedup_steps_op_over_iwp'] = (
        op_steps / iwp_steps if op_steps and iwp_steps else None
    )
    return {
        'task': task_name,
        'generated_at': datetime.now(JST).isoformat(),
        'per_seed': per_seed,
        'aggregate_mean': aggregate,
        'caveat': (
            "estimator='sin' はCUDAカーネル経路(sigmoid固定)を使えずPython"
            'フォールバックになるため、wall-clockのOP vs IWP比較は推定器を'
            '揃えない限り不公平。収束step数の比較はstep基準なので影響を受けない。'
        ),
    }


def make_figures(results: dict[tuple[str, int], dict], task_name: str,
                 suffix: str = '') -> list[Path]:
    """学習曲線(hard/soft acc)とgap推移の比較図を描いて保存パスを返す。

    matplotlibは任意extra(`uv sync --extra figures`)なので遅延importする。
    """
    try:
        import matplotlib
    except ImportError as e:  # 図はベンチの必須成果物なので握り潰さない
        raise RuntimeError(
            'matplotlibが見つかりません。`uv sync --extra dev --extra cpu '
            '--extra figures` で導入するか、--no-figures を指定してください'
        ) from e
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = RESULT_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {'op': 'tab:blue', 'iwp': 'tab:red'}
    paths: list[Path] = []
    for metric, ylabel, fname in (
        ('hard_acc', 'hard test accuracy', 'hard_acc'),
        ('soft_acc', 'soft test accuracy', 'soft_acc'),
        ('gap', 'discretization gap (soft − hard)', 'gap'),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        seen = set()
        for (arm, seed), res in sorted(results.items()):
            steps = [h['step'] for h in res['history']]
            vals = [h[metric] for h in res['history']]
            ax.plot(steps, vals, color=colors.get(arm, 'gray'), alpha=0.75,
                    label=arm.upper() if arm not in seen else None)
            seen.add(arm)
        ax.set_xlabel('step')
        ax.set_ylabel(ylabel)
        ax.set_title(f'IWP-1 {task_name}: OP vs IWP — {ylabel}')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f'{task_name}_{fname}{suffix}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
        print(f'図を保存: {path}')
    return paths


def log_summary_to_wandb(task_name: str, summary: dict, summary_path: Path,
                         fig_paths: list[Path], args) -> None:
    """比較サマリと図を専用のwandb run(job_type=eval)へ記録する。"""
    if not init_wandb(
        run_name=f'{task_name}_summary{run_suffix(args.epochs)}',
        config_dict={'task': task_name, 'seeds': args.seeds, 'arm': args.arm},
        device=args.device or auto_device(),
        tags=['iwp', 'summary', f'task:{task_name}']
        + normalize_tags(series='IWP-1'),
        group=WANDB_GROUP,
        job_type='eval',
        notes=args.notes,
        hypothesis=args.hypothesis,
    ):
        return
    if fig_paths:
        from PIL import Image
        log_images(
            f'iwp1/{task_name}',
            [Image.open(p) for p in fig_paths],
            captions=[p.name for p in fig_paths],
        )
    flat = {}
    for arm, metrics in summary['aggregate_mean'].items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if value is not None:
                    flat[f'{arm}/{key}'] = value
    if summary['aggregate_mean'].get('speedup_steps_op_over_iwp') is not None:
        flat['speedup_steps_op_over_iwp'] = \
            summary['aggregate_mean']['speedup_steps_op_over_iwp']
    if flat:
        log_metrics(flat)
    log_artifact(str(summary_path), name=f'iwp1-{summary_path.stem}',
                 type='bench-summary', aliases=['latest'],
                 metadata={'task': task_name})
    for path in fig_paths:
        log_artifact(str(path), name=f'iwp1-{task_name}-{path.stem}',
                     type='figure', aliases=['latest'])
    finish_wandb(status='finished', final_metrics=flat)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='IWP-1: OP vs IWP ベンチ(同一条件・複数シード)',
    )
    parser.add_argument('--task', choices=sorted(TASKS), required=True,
                        help='ベンチ対象タスク')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='実行するseed(既定: 0 1 2)')
    parser.add_argument('--arm', choices=('op', 'iwp', 'both'), default='both',
                        help='実行するアーム(既定: both)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス(既定: 自動検出 cuda→mps→cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='レシピのepochsを上書きする(スモーク用)')
    parser.add_argument('--wandb', action='store_true',
                        help='レシピの run.wandb=false を上書きしてwandbを有効化する')
    parser.add_argument('--no-figures', action='store_true',
                        help='比較図の生成をスキップする(matplotlib未導入環境用)')
    parser.add_argument('--dry-run', action='store_true',
                        help='学習せず実行計画のみ表示する')
    add_experiment_args(parser)
    args = parser.parse_args()

    load_dotenv()  # WANDB_API_KEY 等を .env から読む(既存環境変数は上書きしない)
    started = datetime.now(JST)
    device = args.device or auto_device()
    print('=' * 72)
    print(f'IWP-1 ベンチ: task={args.task} arm={args.arm} seeds={args.seeds}')
    print(f'開始(JST): {started.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'device={device}')
    print('但し書き: estimator=sin はCUDAカーネル経路(sigmoid固定)を使えず')
    print('          Pythonフォールバックになるため、wall-clockのOP vs IWP比較は')
    print('          推定器を揃えない限り不公平。収束step数はstep基準なので影響なし。')
    print('=' * 72)

    op_recipe, iwp_recipe = TASKS[args.task]
    cfgs = {
        'op': load_recipe(RECIPE_DIR / op_recipe),
        'iwp': load_recipe(RECIPE_DIR / iwp_recipe),
    }
    assert_fair_pair(cfgs['op'], cfgs['iwp'])
    arms = ('op', 'iwp') if args.arm == 'both' else (args.arm,)
    suffix = run_suffix(args.epochs)

    if args.dry_run:
        print('[dry-run] 実行計画:')
        for arm in arms:
            recipe = op_recipe if arm == 'op' else iwp_recipe
            mcfg = cfgs[arm].get('model', {})
            rcfg = cfgs[arm].get('run', {})
            epochs = args.epochs if args.epochs is not None else rcfg.get('epochs')
            for seed in args.seeds:
                print(f'  arm={arm} seed={seed} recipe={RECIPE_DIR / recipe}')
                print(f'    model: {mcfg}')
                print(f'    epochs={epochs} eval_interval={rcfg.get("eval_interval", 0)} '
                      f'wandb={bool(rcfg.get("wandb", False)) or args.wandb}')
                print('    出力: '
                      f'{RESULT_DIR / f"{args.task}_{arm}_seed{seed}{suffix}.json"}')
        print(f'  サマリ: {RESULT_DIR / f"{args.task}_summary{suffix}.json"}')
        print(f'  図: {RESULT_DIR / "figures"}/'
              f'{args.task}_{{hard_acc,soft_acc,gap}}{suffix}.png')
        print(f'  wandb group={WANDB_GROUP}')
        return 0

    results: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for seed in args.seeds:
            results[(arm, seed)] = run_one(args.task, arm, seed, cfgs[arm], args)

    summary = summarize(results, args.task)
    summary_path = RESULT_DIR / f'{args.task}_summary{suffix}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'サマリを保存: {summary_path}')

    fig_paths = [] if args.no_figures else make_figures(results, args.task, suffix)

    want_wandb = args.wandb or any(
        bool(cfgs[arm].get('run', {}).get('wandb', False)) for arm in arms
    )
    if want_wandb:
        log_summary_to_wandb(args.task, summary, summary_path, fig_paths, args)
    else:
        print('wandb: レシピの run.wandb=false のためサマリ記録をスキップ'
              '(--wandb で強制有効化できる)')

    print('=' * 72)
    print(json.dumps(summary['aggregate_mean'], ensure_ascii=False, indent=2))
    elapsed = (datetime.now(JST) - started).total_seconds()
    print(f'総時間: {timedelta(seconds=int(elapsed))}  '
          f'終了(JST): {datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")}')
    return 0


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)
    sys.exit(main())
