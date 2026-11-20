"""attention梯子(sMNIST)の結果図を生成する。

各段の `results/{STAGE}_smnist/summary.csv`（列: seed,loss,ppl,acc,attn_entropy,
discretization_gap。末尾に mean/std 行）から mean/std を読み、次の2枚を作る。

    図1 fig_ladder_accuracy.png
        梯子 L0-FP→L0b→L1→L2→L3-C の hard acc を mean±std のエラーバー付き
        折れ線で。天井(L0-FP)に水平線。各点に値を注記。
    図2 fig_l3_entropy.png（★図1候補）
        L3の4案(C/S/B/A)の attention エントロピーを棒グラフ(mean±std)で。
        各棒に hard acc を注記。参照線 ln(28)=一様上限, ln(4)。

数値は summary.csv の mean/std 行の実測値のみを用いる（捏造しない）。CSVが無い
段はスキップし、その旨を図に注記する。画像はローカル保存に加え wandb にも
log_images でログする（--no-wandb またはAPIキー未設定ならローカル保存のみ）。

使用例:
    uv run python experiments/attention_ladder/src/make_figures.py
    uv run python experiments/attention_ladder/src/make_figures.py --no-wandb
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

# ------------------------------------------------------------
# import規約: experiments/ をパス追加(train.py と同じ流儀)
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LADDER_DIR = os.path.abspath(os.path.join(_SRC_DIR, '..'))           # attention_ladder/
_EXP_DIR = os.path.abspath(os.path.join(_SRC_DIR, '..', '..'))        # experiments/
for _p in (_EXP_DIR, _SRC_DIR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use('Agg')  # ヘッドレス描画バックエンドを明示する

import matplotlib.pyplot as plt  # noqa: E402
from env_utils import load_dotenv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from wandb_utils import finish as wandb_finish  # noqa: E402
from wandb_utils import init_wandb, log_images  # noqa: E402

# ------------------------------------------------------------
# 定数
# ------------------------------------------------------------
# 出力先(既定): attention_ladder/results と results/figures
_DEFAULT_RESULTS_DIR = os.path.join(_LADDER_DIR, 'results')

# 日本語表示用フォント候補(matplotlibのみ・追加依存なし。macOS標準を優先)
_JP_FONT_CANDIDATES = (
    'Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'YuGothic',
    'Arial Unicode MS', 'AppleGothic', 'Noto Sans CJK JP',
)

# 参照エントロピー線(nats)
_LN28 = math.log(28)   # 系列長28に対する一様分布の上限 ≒ 3.332
_LN4 = math.log(4)     # 4-way一様の目安 ≒ 1.386

# 図1: 梯子の並び(内部STAGE名, 表示ラベル)
_LADDER_STAGES: tuple[tuple[str, str], ...] = (
    ('L0', 'L0-FP'),
    ('L0b', 'L0b'),
    ('L1', 'L1'),
    ('L2', 'L2'),
    ('L3-C', 'L3-C'),
)

# 図2: L3の4案(内部STAGE名, 短ラベル, 説明, 棒色)
_L3_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ('L3-C', 'C', 'mux_gumbel\n(MUX選択器/本命)', '#2ca02c'),   # 緑=離散選択
    ('L3-S', 'S', 'softmax_int\n(softmax/対照)', '#d62728'),    # 赤=平坦
    ('L3-B', 'B', 'topk_ste\n(+多数決)', '#1f77b4'),           # 青
    ('L3-A', 'A', 'threshold', '#ff7f0e'),                     # 橙
)

# 図1 y軸(既定は仕様どおり0.6-1.0。実測が下限を割る場合のみ自動で下げる)
_Y_TOP = 1.0
_Y_BOTTOM_SPEC = 0.6
# 天井(L0-FP)水平線の表示値(RESULTS.md記載値。実測m=0.9778と一致)
_CEILING_LABEL = 0.9778
# 精度注記を点の下側に置く閾値(天井付近の点は下に逃がす)
_LABEL_BELOW_THRESHOLD = 0.93


@dataclass
class StageStats:
    """1段分の summary.csv 集計(mean/std 行 + シード数)。"""

    stage: str
    n_seeds: int
    mean: dict[str, float]
    std: dict[str, float]


# ------------------------------------------------------------
# フォント設定
# ------------------------------------------------------------
def _setup_japanese_font() -> str | None:
    """日本語表示可能なフォントを matplotlib に設定する。

    追加依存(japanize-matplotlib等)は使わず、システムにある候補フォントから
    最初に見つかったものを font.family に設定する。見つからなければ None を返す
    (英数字は描けるが日本語は豆腐になる旨を呼び出し側でprintする)。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の豆腐回避
            return name
    return None


# ------------------------------------------------------------
# CSV読み込み
# ------------------------------------------------------------
def _summary_path(results_dir: str, stage: str) -> str:
    """段名から summary.csv の絶対パスを返す。"""
    return os.path.join(results_dir, f'{stage}_smnist', 'summary.csv')


def _to_float(text: str) -> float | None:
    """CSVセルをfloat化する。空文字などは None。"""
    text = (text or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_stage_stats(results_dir: str, stage: str) -> StageStats | None:
    """段の summary.csv を読み、mean/std 行とシード数を返す。

    CSVが存在しない、または mean 行が無い場合は None(=スキップ対象)。
    """
    path = _summary_path(results_dir, stage)
    if not os.path.isfile(path):
        return None

    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    mean_row: dict[str, str] | None = None
    std_row: dict[str, str] | None = None
    n_seeds = 0
    for row in rows:
        seed_key = (row.get('seed') or '').strip()
        if seed_key == 'mean':
            mean_row = row
        elif seed_key == 'std':
            std_row = row
        elif seed_key.lstrip('-').isdigit():
            n_seeds += 1

    if mean_row is None:
        return None

    cols = ('loss', 'ppl', 'acc', 'attn_entropy', 'discretization_gap')
    mean = {c: v for c in cols if (v := _to_float(mean_row.get(c, ''))) is not None}
    std_src = std_row or {}
    std = {c: (_to_float(std_src.get(c, '')) or 0.0) for c in cols}
    return StageStats(stage=stage, n_seeds=n_seeds, mean=mean, std=std)


def load_all_stats(results_dir: str, stages: list[str]) -> dict[str, StageStats]:
    """指定段の集計を読み込む(見つからない段は結果に含めない)。"""
    out: dict[str, StageStats] = {}
    for stage in stages:
        stats = read_stage_stats(results_dir, stage)
        if stats is None:
            print(f'  スキップ(CSV無し/mean行無し): {stage}')
            continue
        out[stage] = stats
    return out


# ------------------------------------------------------------
# 図1: 梯子の精度保存
# ------------------------------------------------------------
def build_fig_ladder_accuracy(
    stats_by_stage: dict[str, StageStats], figures_dir: str,
) -> str:
    """梯子 L0-FP→…→L3-C の hard acc(mean±std)折れ線図を生成する。"""
    xs: list[int] = []
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    n_seeds_list: list[int] = []
    missing_labels: list[str] = []

    idx = 0
    for stage, disp in _LADDER_STAGES:
        stats = stats_by_stage.get(stage)
        if stats is None or 'acc' not in stats.mean:
            missing_labels.append(disp)
            continue
        xs.append(idx)
        labels.append(f'{disp}\n(n={stats.n_seeds})')
        means.append(stats.mean['acc'])
        stds.append(stats.std.get('acc', 0.0))
        n_seeds_list.append(stats.n_seeds)
        idx += 1

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    if xs:
        ax.errorbar(
            xs, means, yerr=stds, fmt='-o', color='#1f4e8c', ecolor='#1f4e8c',
            elinewidth=1.4, capsize=5, markersize=8, markerfacecolor='#3a7bd5',
            markeredgecolor='white', markeredgewidth=1.0, linewidth=2.0,
            zorder=3, label='hard acc (mean±std)',
        )
        # 各点に値を注記(天井付近は下側へ逃がす)
        for x, m, s, n in zip(xs, means, stds, n_seeds_list):
            text = f'{m:.3f}' if n <= 1 else f'{m:.3f}\n±{s:.3f}'
            below = m > _LABEL_BELOW_THRESHOLD
            ax.annotate(
                text, (x, m), textcoords='offset points',
                xytext=(0, -26 if below else 12),
                ha='center', va='top' if below else 'bottom',
                fontsize=9, color='#16324f',
            )

    # 天井(L0-FP)水平線
    ax.axhline(
        _CEILING_LABEL, color='#d62728', linestyle='--', linewidth=1.4,
        alpha=0.8, zorder=2, label=f'天井 {_CEILING_LABEL:.4f} (L0-FP)',
    )

    # y軸: 仕様は0.6-1.0。実測がそれを割る場合のみ下限を自動で下げて隠さない
    data_low = min((m - s for m, s in zip(means, stds)), default=_Y_BOTTOM_SPEC)
    y_bottom = _Y_BOTTOM_SPEC
    axis_note = ''
    if data_low < _Y_BOTTOM_SPEC:
        y_bottom = math.floor((data_low - 0.03) * 20) / 20  # 0.05刻みで下げる
        axis_note = f'（実測が0.6を下回るためy軸下限を{y_bottom:.2f}へ拡張）'
    ax.set_ylim(y_bottom, _Y_TOP)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel('梯子ステージ（全FP → 完全ゲート化）')
    ax.set_ylabel('hard acc（離散評価）')
    ax.set_title('精度保存: 全FP→完全ゲート化MUX-attention(sMNIST)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    # 注記(シード数・欠損段・軸拡張)
    notes: list[str] = []
    if missing_labels:
        notes.append('データ欠損(スキップ): ' + ', '.join(missing_labels))
    notes.append('値は各段 summary.csv の mean/std 行(実測)。n=シード数。')
    if axis_note:
        notes.append(axis_note)
    fig.text(0.01, 0.01, '　'.join(notes), fontsize=7.5, color='gray', ha='left')

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = os.path.join(figures_dir, 'fig_ladder_accuracy.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 図2: L3のattentionエントロピー対比
# ------------------------------------------------------------
def build_fig_l3_entropy(
    stats_by_stage: dict[str, StageStats], figures_dir: str,
) -> str:
    """L3の4案のattentionエントロピー棒グラフ(hard acc注記付き)を生成する。"""
    xs: list[int] = []
    heights: list[float] = []
    errs: list[float] = []
    accs: list[float | None] = []
    labels: list[str] = []
    colors: list[str] = []
    stages_used: list[str] = []
    missing_labels: list[str] = []

    idx = 0
    for stage, short, desc, color in _L3_VARIANTS:
        stats = stats_by_stage.get(stage)
        if stats is None or 'attn_entropy' not in stats.mean:
            missing_labels.append(short)
            continue
        xs.append(idx)
        heights.append(stats.mean['attn_entropy'])
        errs.append(stats.std.get('attn_entropy', 0.0))
        accs.append(stats.mean.get('acc'))
        labels.append(f'{short}: {desc}\n(n={stats.n_seeds})')
        colors.append(color)
        stages_used.append(stage)
        idx += 1

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    if xs:
        ax.bar(
            xs, heights, yerr=errs, width=0.62, color=colors, alpha=0.9,
            edgecolor='white', linewidth=1.0, capsize=6, zorder=3,
        )
        # 各棒に hard acc を注記
        for x, h, e, acc in zip(xs, heights, errs, accs):
            acc_text = f'hard acc\n{acc:.3f}' if acc is not None else 'acc n/a'
            ax.annotate(
                acc_text, (x, h + e), textcoords='offset points',
                xytext=(0, 8), ha='center', va='bottom', fontsize=9,
                color='#333333', fontweight='bold',
            )

    # 参照線: ln(28)=一様上限, ln(4)
    ax.axhline(_LN28, color='#7f7f7f', linestyle='--', linewidth=1.3, zorder=2)
    ax.axhline(_LN4, color='#9467bd', linestyle=':', linewidth=1.3, zorder=2)
    ax.text(len(xs) - 0.4, _LN28, f'  ln(28)={_LN28:.3f}（一様上限）',
            va='center', ha='left', fontsize=8.5, color='#5f5f5f')
    ax.text(len(xs) - 0.4, _LN4, f'  ln(4)={_LN4:.3f}',
            va='center', ha='left', fontsize=8.5, color='#6a4a9c')

    # C(MUX)=0 と S(softmax)の要点注記
    stage_to_x = {s: x for s, x in zip(stages_used, xs)}
    if 'L3-C' in stage_to_x:
        xc = stage_to_x['L3-C']
        ax.annotate(
            'エントロピー=0\nhard 1トークン選択\n=固定回路化 可',
            (xc, 0.0), textcoords='offset points', xytext=(0, 60),
            ha='center', va='bottom', fontsize=8.5, color='#1a7a1a',
            bbox=dict(boxstyle='round,pad=0.35', fc='#eaf7ea', ec='#2ca02c', lw=1.0),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2),
        )
    if 'L3-S' in stage_to_x:
        xs_ = stage_to_x['L3-S']
        hs = heights[xs.index(xs_)]
        ax.annotate(
            '平坦（softmaxで分散）\n＝固定回路化 不可（FP不可）',
            (xs_, hs), textcoords='offset points', xytext=(30, 34),
            ha='left', va='bottom', fontsize=8.5, color='#a11212',
            bbox=dict(boxstyle='round,pad=0.35', fc='#fdeaea', ec='#d62728', lw=1.0),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('attention エントロピー（nats）')
    ax.set_title('softmaxの平坦化 vs 離散選択器: attentionエントロピー対比(sMNIST)')
    ax.set_ylim(0, _LN28 + 0.6)
    ax.grid(True, axis='y', alpha=0.3)

    notes: list[str] = ['値は各案 summary.csv の mean/std 行(実測)。棒上=hard acc。']
    if missing_labels:
        notes.insert(0, 'データ欠損(スキップ): ' + ', '.join(missing_labels))
    fig.text(0.01, 0.01, '　'.join(notes), fontsize=7.5, color='gray', ha='left')

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = os.path.join(figures_dir, 'fig_l3_entropy.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# wandb
# ------------------------------------------------------------
def _log_figures_to_wandb(fig_ladder: str, fig_entropy: str) -> None:
    """生成PNGをPIL化して wandb に log_images する。"""
    from PIL import Image

    ladder_img = Image.open(fig_ladder).convert('RGB')
    entropy_img = Image.open(fig_entropy).convert('RGB')
    log_images('ladder/accuracy', [ladder_img],
               captions=['ladder hard acc (mean±std)'])
    log_images('ladder/l3_entropy', [entropy_img],
               captions=['L3 attention entropy (C/S/B/A)'])


# ------------------------------------------------------------
# 数値サマリの標準出力
# ------------------------------------------------------------
def _print_used_numbers(stats_by_stage: dict[str, StageStats]) -> None:
    """図に使った数値(mean±std)を確認用にprintする。"""
    print('\n=== 図に使った数値(summary.csv 実測 mean±std) ===')
    for stage, stats in stats_by_stage.items():
        acc_m = stats.mean.get('acc')
        acc_s = stats.std.get('acc', 0.0)
        ent_m = stats.mean.get('attn_entropy')
        ent_s = stats.std.get('attn_entropy', 0.0)
        acc_txt = f'{acc_m:.4f}±{acc_s:.4f}' if acc_m is not None else 'n/a'
        ent_txt = f'{ent_m:.4f}±{ent_s:.4f}' if ent_m is not None else 'n/a'
        print(f'  {stage:5s} n={stats.n_seeds}  '
              f'hard_acc={acc_txt}  attn_entropy={ent_txt}')


# ------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='attention梯子(sMNIST)の結果図2枚を生成する'
    )
    parser.add_argument(
        '--results-dir', type=str, default=_DEFAULT_RESULTS_DIR,
        help='入力 summary.csv 群のルート(既定: attention_ladder/results)',
    )
    parser.add_argument(
        '--figures-dir', type=str, default=None,
        help='図の出力先(既定: <results-dir>/figures)',
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='wandbログを無効化する(ローカルPNG保存は常に行う)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    figures_dir = os.path.abspath(
        args.figures_dir or os.path.join(results_dir, 'figures')
    )
    os.makedirs(figures_dir, exist_ok=True)

    font = _setup_japanese_font()
    if font is None:
        print('警告: 日本語フォント未検出。図中の日本語が豆腐になる可能性がある。')
    else:
        print(f'日本語フォント: {font}')

    # 図1(梯子)と図2(L3)に必要な全段を読み込む
    ladder_stages = [s for s, _ in _LADDER_STAGES]
    l3_stages = [s for s, _, _, _ in _L3_VARIANTS]
    all_stages = list(dict.fromkeys(ladder_stages + l3_stages))
    print(f'CSV読み込み(results_dir={results_dir})')
    stats_by_stage = load_all_stats(results_dir, all_stages)
    _print_used_numbers(stats_by_stage)

    fig_ladder = build_fig_ladder_accuracy(stats_by_stage, figures_dir)
    fig_entropy = build_fig_l3_entropy(stats_by_stage, figures_dir)
    print(f'\n図1(梯子精度): {fig_ladder}')
    print(f'図2(L3エントロピー): {fig_entropy}')

    # wandbログ(--no-wandb またはAPIキー未設定ならローカル保存のみ)
    load_dotenv()
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            'attention_ladder_figures',
            {
                'results_dir': results_dir,
                'figures_dir': figures_dir,
                'stages': ','.join(stats_by_stage.keys()),
            },
            job_type='analysis',
            group='attention_ladder',
            tags=['figures', 'smnist'],
        )
    if use_wandb:
        _log_figures_to_wandb(fig_ladder, fig_entropy)
        wandb_finish()
        print('wandb: 図をログした(finish済み)')
    else:
        print('wandb: 無効のためローカルPNG保存のみ（run未作成）')


if __name__ == '__main__':
    main()
