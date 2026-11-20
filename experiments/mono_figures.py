"""
単調性仮説検証（AP法 vs BP法）の観測結果JSONから受け入れ基準の図2枚を生成する。

対象は results/mono/*.json のフラットMLPスキーマ。CNN由来のJSON（model=='cnn'、
history各点が per_module 形式）は図1/2の対象外としてスキップする。

生成する図:
    図1 fig1_V_layerwise_learning.png
        層別単調度違反Vの学習推移（AP成功/失敗/BPの3群を横3パネル）。
    図2 fig2_acc_vs_V_scatter.png
        最終精度 vs ネットワーク平均V の散布図（1点=1run、色=群）。

生成した2枚はローカル保存に加え wandb にも log_images でログする（--no-wandb で無効）。
CJKフォント問題を避けるため図中の文字（軸ラベル・凡例・タイトル）はすべて英語。

使用例:
    uv run python experiments/mono_figures.py --no-wandb
    uv run python experiments/mono_figures.py --results-dir results/mono
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')  # GUIなしのヘッドレス描画バックエンドを明示する

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_images

# --- 定数（閾値・色・レイアウト） ---
# AP成功/失敗を分ける最終softの閾値（これ未満なら失敗扱い）
SUCCESS_SOFT_THRESHOLD = 0.5
# 散布図に回帰線を引く最小点数
MIN_POINTS_FOR_REGRESSION = 5

# 3群の並び順（図1のパネル順もこの順）
GROUP_ORDER = ('AP-success', 'AP-fail', 'BP')
# 群の代表色（AP-success=緑, AP-fail=赤, BP=青）
GROUP_COLORS = {
    'AP-success': '#2ca02c',
    'AP-fail': '#d62728',
    'BP': '#1f77b4',
}
# 図1で層別ラインに使う濃淡グラデーション用カラーマップ（群色と同系統）
GROUP_CMAPS = {
    'AP-success': 'Greens',
    'AP-fail': 'Reds',
    'BP': 'Blues',
}

# 図1の1パネル幅・高さ（インチ）
PANEL_WIDTH = 5.0
PANEL_HEIGHT = 4.5
# 層別ラインのグラデ濃さレンジ（薄すぎ・濃すぎを避ける）
CMAP_LOW = 0.35
CMAP_HIGH = 0.9


def classify_group(run: dict) -> str:
    """runを 'AP-success' / 'AP-fail' / 'BP' の3群に分類する。

    - BP: method=='bp'
    - AP失敗: method=='ap' かつ (collapsed==True または final_soft < 閾値)
    - AP成功: method=='ap' かつ上記以外
    """
    if run.get('method') == 'bp':
        return 'BP'
    collapsed = bool(run.get('collapsed', False))
    final_soft = run.get('final_soft')
    failed = collapsed or (final_soft is None) or (final_soft < SUCCESS_SOFT_THRESHOLD)
    return 'AP-fail' if failed else 'AP-success'


def _is_flat_mlp(run: dict) -> bool:
    """フラットMLP由来のrunか判定する（CNN由来なら False）。

    model=='cnn' は除外。加えて history各点が per_module 形式（V_per_layerを
    持たない）ならフラットMLPではないとみなす。
    """
    if run.get('model') == 'cnn':
        return False
    history = run.get('history')
    if not history:
        return False
    first = history[0]
    # per_module 形式は V_per_layer（list）を持たない
    return isinstance(first.get('V_per_layer'), list)


def load_runs(results_dir: str) -> list[dict]:
    """results_dir 内のJSONを読み込み、フラットMLPのrunのみ返す。"""
    runs: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f'警告: 読み込み失敗のためスキップ {path}: {e}')
            continue
        if not _is_flat_mlp(run):
            print(f'スキップ（フラットMLP対象外）: {os.path.basename(path)}')
            continue
        run['_path'] = path
        runs.append(run)
    return runs


def group_runs(runs: list[dict]) -> dict[str, list[dict]]:
    """runリストを3群の辞書に分類する（キーは GROUP_ORDER 全て存在）。"""
    groups: dict[str, list[dict]] = {g: [] for g in GROUP_ORDER}
    for run in runs:
        groups[classify_group(run)].append(run)
    return groups


def _pick_representative(group: list[dict]) -> dict | None:
    """群の代表runを選ぶ（最深=depth最大、同率はhistoryが長い方）。"""
    if not group:
        return None
    return max(group, key=lambda r: (r.get('depth', 0), len(r.get('history', []))))


def _group_mean_curve(group: list[dict]) -> tuple[list[int], list[float]]:
    """群内全runの V_mean をstepごとに平均した推移を返す。

    runごとにstep格子が異なっても、同一stepの値を平均して結合する。
    """
    acc: dict[int, list[float]] = defaultdict(list)
    for run in group:
        for point in run.get('history', []):
            v_mean = point.get('V_mean')
            step = point.get('step')
            if v_mean is None or step is None:
                continue
            acc[step].append(float(v_mean))
    steps = sorted(acc)
    means = [sum(acc[s]) / len(acc[s]) for s in steps]
    return steps, means


def _plot_group_panel(ax, group_name: str, group: list[dict]) -> None:
    """図1の1パネル（1群分）を描画する。"""
    ax.set_title(f'{group_name} (n={len(group)})')
    ax.set_xlabel('training step')
    ax.set_ylabel('monotonicity violation V')

    if not group:
        ax.text(0.5, 0.5, 'no runs', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        return

    # 代表run（最深）の層別Vを濃淡グラデで描画する
    rep = _pick_representative(group)
    history = rep.get('history', []) if rep else []
    if history and isinstance(history[0].get('V_per_layer'), list):
        n_layers = len(history[0]['V_per_layer'])
        steps = [p['step'] for p in history]
        cmap = plt.get_cmap(GROUP_CMAPS[group_name])
        for layer in range(n_layers):
            frac = layer / max(n_layers - 1, 1)
            shade = CMAP_LOW + (CMAP_HIGH - CMAP_LOW) * frac
            ys = [p['V_per_layer'][layer] for p in history
                  if isinstance(p.get('V_per_layer'), list)
                  and len(p['V_per_layer']) > layer]
            xs = [p['step'] for p in history
                  if isinstance(p.get('V_per_layer'), list)
                  and len(p['V_per_layer']) > layer]
            ax.plot(xs, ys, color=cmap(shade), linewidth=1.2,
                    label=f'rep layer {layer}')
        depth = rep.get('depth', n_layers) if rep else n_layers
        ax.plot([], [], ' ', label=f'(rep depth={depth})')

    # 群平均のV_mean（全run）を太い黒線で重ねる
    mean_steps, mean_vals = _group_mean_curve(group)
    if mean_steps:
        ax.plot(mean_steps, mean_vals, color='black', linewidth=2.5,
                label='group mean V (all runs)', zorder=5)

    ax.legend(fontsize=7, loc='best')


def build_fig1(groups: dict[str, list[dict]], figures_dir: str) -> str:
    """図1（層別単調度違反Vの学習推移）を生成して保存パスを返す。"""
    n_panels = len(GROUP_ORDER)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(PANEL_WIDTH * n_panels, PANEL_HEIGHT),
    )
    if n_panels == 1:
        axes = [axes]
    for ax, group_name in zip(axes, GROUP_ORDER):
        _plot_group_panel(ax, group_name, groups[group_name])
    fig.suptitle('Layer-wise monotonicity violation V during training')
    fig.tight_layout()

    out_path = os.path.join(figures_dir, 'fig1_V_layerwise_learning.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _final_mean_v(run: dict) -> float | None:
    """historyの最後の V_mean（最終ネットワーク平均V）を返す。"""
    history = run.get('history')
    if not history:
        return None
    return history[-1].get('V_mean')


def build_fig2(runs: list[dict], figures_dir: str) -> str:
    """図2（最終精度 vs ネットワーク平均V の散布図）を生成して保存パスを返す。"""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.set_title('Final soft accuracy vs network mean V')
    ax.set_xlabel('final network mean V')
    ax.set_ylabel('final soft accuracy')

    xs_all: list[float] = []
    ys_all: list[float] = []
    for run in runs:
        x = _final_mean_v(run)
        y = run.get('final_soft')
        if x is None or y is None:
            continue
        group_name = classify_group(run)
        ax.scatter(x, y, color=GROUP_COLORS[group_name], s=60,
                   edgecolors='white', linewidths=0.5, zorder=3)
        depth = run.get('depth')
        if depth is not None:
            ax.annotate(f'd{depth}', (x, y), textcoords='offset points',
                        xytext=(5, 4), fontsize=7, color='dimgray')
        xs_all.append(float(x))
        ys_all.append(float(y))

    # 点数が十分なら全点にわたる回帰線を薄く重ねる
    if len(xs_all) >= MIN_POINTS_FOR_REGRESSION:
        import numpy as np
        coef = np.polyfit(xs_all, ys_all, 1)
        xline = np.linspace(min(xs_all), max(xs_all), 100)
        yline = coef[0] * xline + coef[1]
        ax.plot(xline, yline, color='gray', linestyle='--',
                linewidth=1.2, alpha=0.4, zorder=1, label='linear fit')

    # 群の凡例（色→群名）をプロキシで作る
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=GROUP_COLORS[g],
                   label=g)
        for g in GROUP_ORDER
    ]
    if len(xs_all) >= MIN_POINTS_FOR_REGRESSION:
        handles.append(
            plt.Line2D([], [], color='gray', linestyle='--', label='linear fit')
        )
    ax.legend(handles=handles, fontsize=8, loc='best')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_path = os.path.join(figures_dir, 'fig2_acc_vs_V_scatter.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _log_to_wandb(fig_paths: list[str], captions: list[str]) -> None:
    """生成した図をwandbにログする（PIL画像化して log_images）。"""
    from PIL import Image
    images = [Image.open(p).convert('RGB') for p in fig_paths]
    log_images('mono/figures', images, captions=captions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='単調性仮説検証の観測結果JSONから受け入れ基準の図2枚を生成する'
    )
    parser.add_argument('--results-dir', type=str, default='results/mono',
                        help='入力JSONのディレクトリ（default: results/mono）')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                        help='図の出力ディレクトリ（default: results/figures）')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化する（ローカル保存は常に行う）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    runs = load_runs(args.results_dir)
    groups = group_runs(runs)
    counts = {g: len(groups[g]) for g in GROUP_ORDER}
    print(f'読み込みrun数: {len(runs)}  群内訳: {counts}')

    fig1_path = build_fig1(groups, args.figures_dir)
    fig2_path = build_fig2(runs, args.figures_dir)
    print(f'図1: {fig1_path}')
    print(f'図2: {fig2_path}')

    # wandbログ（--no-wandb またはAPIキー未設定なら無効）
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name='mono_figures',
            config_dict={
                'results_dir': args.results_dir,
                'figures_dir': args.figures_dir,
                'n_runs': len(runs),
                **{f'n_{g}': counts[g] for g in GROUP_ORDER},
            },
            model=None,
            device='cpu',
            tags=['monotonicity', 'figures'],
            group='monotonicity',
            job_type='figures',
        )
    if use_wandb:
        _log_to_wandb(
            [fig1_path, fig2_path],
            captions=['fig1: layer-wise V during training',
                      'fig2: final acc vs network mean V'],
        )
        wandb_finish()


if __name__ == '__main__':
    main()
