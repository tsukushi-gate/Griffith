"""計算型配線の較正アンカー(C3 / C4)の再走ドライバ。

configs/recipes/cwo_*.yaml の 7 本を seed ごとに順に回し、run ごとの結果 JSON を
results/ に落とす。`griffith launch --grid "run.seed=[0,1]" <recipe>` と同じ経路
(load_config → run_training)を通るので、CLI で 1 本ずつ回してもよい
(README の「1 本ずつ回す」を参照)。本スクリプトは 7 本 × 2 seed の順次実行と
結果の回収を 1 コマンドにまとめるだけで、学習の中身は一切変えない。

使用例:
    # スモーク(1 エポック、CPU、wandb なし。落ちないことの確認用)
    uv run python experiments/computed-wiring-overlay/run_calibration.py --smoke

    # 本走(7 レシピ × seed 0/1 = 14 run)
    uv run python experiments/computed-wiring-overlay/run_calibration.py --device cuda

    # C3 だけ / 特定のレシピだけ
    uv run python experiments/computed-wiring-overlay/run_calibration.py --anchor C3
    uv run python experiments/computed-wiring-overlay/run_calibration.py \
        --recipes cwo_cifar10_48k_sr2 --seeds 0

wandb: レシピ側で group=dlgn-ffn / job_type=calibration / tags / notes / hypothesis を
指定済み。project は環境変数 WANDB_PROJECT で切り替える(既定は
griffith-logic、探索は各自の project)。--no-wandb で無効化できる。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RECIPE_DIR = REPO_ROOT / 'configs' / 'recipes'

sys.path.insert(0, str(REPO_ROOT))

# アンカーごとのレシピ(anchors.json の並び順を保つ)。C3 → C4 の順で回す
ANCHOR_RECIPES: dict[str, list[str]] = {
    'C3': [
        'cwo_mnist_48k_p0',
        'cwo_mnist_48k_p1',
        'cwo_mnist_48k_p6',
        'cwo_mnist_96k_p6',
    ],
    'C4': [
        'cwo_cifar10_48k_p',
        'cwo_cifar10_48k_sr2',
        'cwo_cifar10_48k_r',
    ],
}
ALL_RECIPES = [name for names in ANCHOR_RECIPES.values() for name in names]


def log(msg: str) -> None:
    print(f'[{datetime.now(JST).strftime("%H:%M:%S")}] {msg}', flush=True)


def git_hash() -> str | None:
    """library のコミットハッシュ(結果 JSON に残して再走の同一性を追えるようにする)"""
    try:
        out = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def anchor_of(recipe: str) -> str | None:
    for anchor, names in ANCHOR_RECIPES.items():
        if recipe in names:
            return anchor
    return None


def build_overrides(args, recipe: str, seed: int) -> list[str]:
    """レシピへ重ねるドット記法の上書き(腕の定義は一切触らない)。"""
    overrides = [f'run.seed={seed}', f'run.run_name={recipe}_s{seed}']
    if args.device:
        overrides.append(f'run.device={args.device}')
    if args.no_wandb:
        overrides.append('run.wandb=false')
    if args.smoke:
        overrides += ['run.epochs=1', 'run.wandb_job_type=smoke']
        if args.limit:
            overrides.append(f'data.params.limit={args.limit}')
    if args.epochs is not None:
        overrides.append(f'run.epochs={args.epochs}')
    root = args.mnist_root if recipe.startswith('cwo_mnist') else args.cifar_root
    if root:
        overrides.append(f'data.params.root={root}')
    return overrides


def eta_line(done: int, total: int, elapsed: float) -> str:
    """[経過<残り, 終了予定 HH:MM:SS] 形式(training-script スキルのルール 2)"""
    remaining = elapsed / done * (total - done) if done else 0.0
    eta = datetime.now(JST) + timedelta(seconds=remaining)
    return (f'[{timedelta(seconds=int(elapsed))}<'
            f'{timedelta(seconds=int(remaining))}, '
            f'終了予定 {eta.strftime("%H:%M:%S")}]')


def run_one(args, recipe: str, seed: int, out_root: Path) -> dict:
    """1 run(レシピ × seed)を実行して結果辞書を返す。"""
    from griffith_logic.training.config import load_config
    from griffith_logic.training.runner import run_training

    recipe_path = RECIPE_DIR / f'{recipe}.yaml'
    if not recipe_path.is_file():
        raise FileNotFoundError(f'レシピが見つかりません: {recipe_path}')
    overrides = build_overrides(args, recipe, seed)
    cfg = load_config(str(recipe_path), overrides)
    out_dir = out_root / f'{recipe}_s{seed}'

    log(f'START {recipe} seed={seed} -> {out_dir}')
    if args.dry_run:
        return {'recipe': recipe, 'seed': seed, 'dry_run': True,
                'overrides': overrides}

    started = time.time()
    result = run_training(cfg, out_dir=str(out_dir))
    wall = time.time() - started

    record = {
        'recipe': recipe,
        'anchor': anchor_of(recipe),
        'seed': seed,
        'overrides': overrides,
        'wall_clock_s': wall,
        'hard_acc': result.get('hard_acc'),
        'best_hard_acc': result.get('best_hard_acc'),
        'final_loss': result.get('final_loss'),
        'steps': result.get('steps'),
        'out_dir': result.get('out_dir'),
        'checkpoint_latest': result.get('checkpoint_latest'),
        'checkpoint_best': result.get('checkpoint_best'),
        'netlist_path': result.get('netlist_path'),
        'expected_metrics': result.get('expected_metrics'),
        'smoke': bool(args.smoke),
        'library_commit': git_hash(),
        'device_requested': args.device,
        'python': platform.python_version(),
        'finished_at': datetime.now(JST).isoformat(),
    }
    log(f'DONE  {recipe} seed={seed} hard_acc={record["hard_acc"]} '
        f'best_hard_acc={record["best_hard_acc"]} wall={wall:.1f}s')
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='計算型配線の較正アンカー(C3 / C4)を再走する')
    parser.add_argument('--anchor', choices=sorted(ANCHOR_RECIPES),
                        help='アンカーで絞る(省略時は C3 と C4 の両方)')
    parser.add_argument('--recipes', nargs='+', metavar='NAME',
                        help=f'レシピ名で絞る(既定: {" ".join(ALL_RECIPES)})')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1],
                        help='seed(既定: 0 1)')
    parser.add_argument('--smoke', action='store_true',
                        help='1 エポックだけ回す(配管の確認用。判定には使えない)')
    parser.add_argument('--epochs', type=int,
                        help='エポック数を明示指定(--smoke より優先)')
    parser.add_argument('--limit', type=int,
                        help='--smoke 時に学習データを先頭 N 件へ絞る')
    parser.add_argument('--device', help='cuda / cpu(省略時はレシピと自動検出)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandb を無効にする(スモークの既定ではない)')
    parser.add_argument('--out-root', default='outputs/computed-wiring-overlay',
                        help='run の出力先(既定: outputs/computed-wiring-overlay)')
    parser.add_argument('--mnist-root', help='MNIST の格納先(省略時は自動発見)')
    parser.add_argument('--cifar-root', help='CIFAR-10 の格納先(省略時は自動発見)')
    parser.add_argument('--dry-run', action='store_true',
                        help='実行せず対象と上書きだけ表示する')
    args = parser.parse_args(argv)

    recipes = args.recipes or (
        ANCHOR_RECIPES[args.anchor] if args.anchor else ALL_RECIPES
    )
    unknown = [r for r in recipes if not (RECIPE_DIR / f'{r}.yaml').is_file()]
    if unknown:
        parser.error(f'未知のレシピ: {unknown}(configs/recipes/ を確認)')

    out_root = Path(args.out_root)
    results_dir = out_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(recipe, seed) for recipe in recipes for seed in args.seeds]
    log(f'対象 {len(jobs)} run(レシピ {len(recipes)} × seed {len(args.seeds)})'
        f'{" [SMOKE]" if args.smoke else ""}')

    started = time.time()
    failures: list[str] = []
    for i, (recipe, seed) in enumerate(jobs, 1):
        try:
            record = run_one(args, recipe, seed, out_root)
        except Exception as exc:  # noqa: BLE001 — 1 run の失敗で全体を止めない
            failures.append(f'{recipe}_s{seed}: {exc!r}')
            record = {'recipe': recipe, 'seed': seed, 'error': repr(exc),
                      'finished_at': datetime.now(JST).isoformat()}
            log(f'FAIL  {recipe} seed={seed}: {exc!r}')
        if not args.dry_run:
            path = results_dir / f'{recipe}_s{seed}.json'
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        log(f'進捗 {i}/{len(jobs)} {eta_line(i, len(jobs), time.time() - started)}')

    total = timedelta(seconds=int(time.time() - started))
    log(f'完了: {len(jobs) - len(failures)}/{len(jobs)} 成功、総時間 {total}')
    if failures:
        for line in failures:
            log(f'  失敗: {line}')
        return 1
    if not args.dry_run:
        log(f'集計: uv run python {HERE.name}/aggregate.py '
            f'--results {results_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
