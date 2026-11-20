"""MIXER-1: 「ゲート数 vs hard accuracy」曲線(LogicMixer vs 純MLP DLGN)を描く。

入力は results/mixer1/runs.json(mixer1_collect.py の出力)のみ。数値を手打ちしない。
生成した図はローカル保存に加え wandb にも log_images でログする(--no-wandb で無効)。

使い方:
    uv run python experiments/mixer1_plot.py
    uv run python experiments/mixer1_plot.py --no-wandb
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 日本語ラベルが豆腐(□)にならないよう、環境にあるCJKフォントを優先設定する。
_JP_FONT_CANDIDATES = (
    'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic',
    'Noto Sans CJK JP', 'IPAexGothic', 'IPAPGothic', 'TakaoPGothic',
)


def _setup_japanese_font() -> str | None:
    """利用可能な日本語フォントをmatplotlibの既定にする。無ければNone。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return name
    return None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_images

load_dotenv()

# 純MLP型DLGNの参考値。**条件が揃っていない4本の別実験からの寄せ集め**であり、
# 同一プロトコルのスケーリング曲線ではない。
# (ゲート数, acc, ラベル, 出典条件)
_MLP_DLGN_REFS = [
    (6000, 0.9088, '6k: 0.9088',
     'D5 N0 scratch / 14,040 step / 二値化784入力'),
    (8000, 0.9739, '8k: 0.9739',
     'D5 Tier-G scratch k=2000x4 / 50k iter'),
    (16000, 0.9338, '16k: 0.9338',
     'BinaryBrain比較 B1 / 10 epoch(短期打切)'),
    (96000, 0.9774, '96k: 0.9774',
     'softmax N=2 / 50 epoch / hard acc明記'),
]

_CHANCE = 0.1135  # 学習に失敗したrunが張り付く値(soft acc = 最頻クラス相当)

MNIST_TRAIN_SIZE = 60000

# difflogic(Petersen et al. 2022, arXiv:2210.08277)系のMNIST到達点。
# **リポジトリ内の資料から転記した値のみ**。論文本文PDFはリポジトリに未取得のため、
# 「論文Table」ではなく出典の性質(公式README再現レシピ/当リポの再現実測/要約ノート)
# をラベルに明記する。推測値は置かない。
# 各要素: (ゲート数 or None, acc, ラベル, 学習予算)
_DIFFLOGIC_REFS: list[tuple[int | None, float, str, str]] = [
    # 公式READMEの "paper small" 再現レシピ k=8000×l=6=48,000ゲート
    # 出典: difflogic 公式README の "paper small" レシピ
    (48000, 0.9769, '48,000ゲート 97.69%(公式README再現レシピ)',
     '200k iter × bs100 ≈ 333 epoch'),
    # 当リポジトリでの同レシピ再現実測(hard eval)
    # 出典: 当リポジトリでの再現実測(hard eval)
    (48000, 0.9784, '48,000ゲート 97.84%(当リポの再現実測 hard)',
     '200k iter × bs100 ≈ 333 epoch'),
    # 要約ノートに残るMNIST到達点。単位が「384Kパラメータ」とされ、
    # ゲート数との対応が資料間で不一致(survey側は「384kゲート」と記載)。
    # 出典: Petersen et al., Deep Differentiable Logic Gate Networks (arXiv:2210.08277)
    (None, 0.9847, 'MNIST 98.47%(384K規模。予算の記載なし)', '予算記載なし'),
]


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _is_mnist_run(r: dict) -> bool:
    return (r.get('dataset') == 'mnist' and r.get('run_tag') != 'smoke'
            and isinstance(r.get('n_gates'), int)
            and isinstance(r.get('best_acc'), float)
            and (r.get('step') or 0) >= 15000)


def _head_width(r: dict) -> int:
    w = r.get('head_group_width')
    return w if isinstance(w, int) else 16


def plot_scaling(rows: list[dict], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 6.5))

    runs = [r for r in rows if _is_mnist_run(r)]
    mixer = [r for r in runs if r.get('model_kind') == 'mixer']
    baseline = sorted(
        [r for r in runs if r.get('model_kind') == 'mlp_dlgn'],
        key=lambda r: r['n_gates'])

    failed = [r for r in mixer if r['best_acc'] < 0.2]
    ok = [r for r in mixer if r['best_acc'] >= 0.2]
    narrow = [r for r in ok if _head_width(r) <= 16]
    wide = [r for r in ok if _head_width(r) > 16]

    # 条件を揃えた純MLP DLGN基準線(20k step / lr0.01 / bs128)
    if baseline:
        ax.plot([r['n_gates'] for r in baseline], [r['best_acc'] for r in baseline],
                marker='s', markersize=9, color='#d62728', linewidth=2.0, zorder=5,
                label='純MLP DLGN(条件を揃えた基準線: 20k step)')

    # 条件不揃いの参考値(線で結ばない)
    ax.scatter([g for g, _, _, _ in _MLP_DLGN_REFS],
               [a for _, a, _, _ in _MLP_DLGN_REFS],
               marker='*', s=200, facecolors='none', edgecolors='#d62728',
               linewidths=1.4, zorder=4,
               label='純MLP DLGN(docs参考値・条件不揃い)')
    for g, a, lab, _cond in _MLP_DLGN_REFS:
        ax.annotate(lab, (g, a), textcoords='offset points', xytext=(7, -3),
                    fontsize=7.5, color='#d62728', alpha=0.85)

    # LogicMixer: 既定ヘッド(group_width=16)
    ax.scatter([r['n_gates'] for r in narrow], [r['best_acc'] for r in narrow],
               marker='o', s=100, color='#8c8c8c', edgecolors='black',
               linewidths=0.6, zorder=4,
               label='LogicMixer(既定ヘッド: 160ゲート固定)')

    # LogicMixer: ヘッド拡張
    ax.scatter([r['n_gates'] for r in wide], [r['best_acc'] for r in wide],
               marker='o', s=150, color='#1f77b4', edgecolors='black',
               linewidths=0.8, zorder=6,
               label='LogicMixer(ヘッド拡張: head_group_width≥32)')
    for r in wide:
        ax.annotate(f'gw={_head_width(r)}', (r['n_gates'], r['best_acc']),
                    textcoords='offset points', xytext=(6, 6), fontsize=7.5,
                    color='#1f77b4')

    if failed:
        ax.scatter([r['n_gates'] for r in failed], [r['best_acc'] for r in failed],
                   marker='x', s=120, color='#7f7f7f', linewidths=2.0, zorder=4,
                   label='LogicMixer(学習せず: unique配線 / n_blocks=8)')

    # CDDLGN(train_cnn.py)を同条件で流した参考点。純MLP基準線とは別系統なので
    # 絶対に基準線に混ぜないこと(混ぜると基準線がV字に折れて誤読される)。
    cddlgn = [r for r in runs if r.get('model_kind') == 'cddlgn']
    if cddlgn:
        ax.scatter([r['n_gates'] for r in cddlgn], [r['best_acc'] for r in cddlgn],
                   marker='^', s=110, color='#9467bd', edgecolors='black',
                   linewidths=0.6, zorder=5,
                   label='CDDLGN k=1(同条件・参考。k=1は想定外の小構成)')

    # difflogic系の到達点(ゲート数が判明しているものだけ点で置く)
    dl = [(g, a) for g, a, _, _ in _DIFFLOGIC_REFS if g is not None]
    if dl:
        ax.scatter([g for g, _ in dl], [a for _, a in dl],
                   marker='D', s=90, color='#2ca02c', edgecolors='black',
                   linewidths=0.6, zorder=6,
                   label='difflogic 48,000ゲート(予算333 epoch≈我々の7.8倍)')

    ax.axhline(_CHANCE, color='#999999', linestyle=':', linewidth=1.0)
    ax.annotate('チャンスレベル', (700, _CHANCE + 0.015), fontsize=8, color='#666666')

    ax.set_xscale('log')
    ax.set_xlabel('ユニークゲート数(log)')
    ax.set_ylabel('hard accuracy (MNIST test)')
    ax.set_title('ゲート数 vs hard accuracy: LogicMixer vs 純MLP DLGN(MNIST 20k step)')
    ax.grid(alpha=0.3, which='both')
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='lower right', fontsize=8.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_coverage_ablation(rows: list[dict], out_path: str) -> str:
    """C: 被覆率を上げても精度が上がらないことを示す棒グラフ。"""
    wanted = ['dc64_d2', 'c_dc64_d2_e256r', 'c_dc64_d2_e256u', 'c_dc64_d2_e196u']
    by_tag: dict[str, dict] = {}
    for r in rows:
        tag = r.get('run_tag')
        if tag in wanted and isinstance(r.get('best_acc'), float):
            # 同一tagが複数(local/modal)ある場合は進んでいる方を採る
            prev = by_tag.get(tag)
            if prev is None or (r.get('step') or 0) > (prev.get('step') or 0):
                by_tag[tag] = r
    present = [t for t in wanted if t in by_tag]
    if not present:
        return ''

    labels, accs, colors, notes = [], [], [], []
    _LABEL = {
        'dc64_d2': '1段/random\n(被覆0.327)',
        'c_dc64_d2_e256r': '2段/random\n(被覆1.000)',
        'c_dc64_d2_e256u': '2段/unique h=256\n(被覆1.000)',
        'c_dc64_d2_e196u': '2段/unique h=196\n(被覆1.000)',
    }
    for tag in present:
        r = by_tag[tag]
        labels.append(_LABEL[tag])
        accs.append(r['best_acc'])
        colors.append('#7f7f7f' if r['best_acc'] < 0.2 else '#1f77b4')
        notes.append(f"{r.get('n_gates')}g / step {r.get('step')}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, accs, color=colors, edgecolor='black', linewidth=0.6)
    for bar, acc, note in zip(bars, accs, notes, strict=True):
        ax.annotate(f'{acc:.4f}\n{note}',
                    (bar.get_x() + bar.get_width() / 2, acc),
                    ha='center', va='bottom', fontsize=8)
    ax.axhline(_CHANCE, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_ylabel('hard accuracy (MNIST test)')
    ax.set_ylim(0.0, 0.85)
    ax.set_title('C: 埋め込みの入力被覆率を1.0にしても精度は上がらない(d_channels=64固定)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_learning_curves(history: list[dict], out_path: str) -> str:
    """hard_acc vs step の学習曲線(上軸=epoch)。まだ登り続けているかを読む図。"""
    fig, ax = plt.subplots(figsize=(11.0, 6.8))

    def _label(h: dict) -> str:
        gw = h.get('head_group_width')
        gates = h.get('n_gates')
        if h['model_kind'] == 'mlp_dlgn':
            return f'純MLP DLGN {gates}g'
        if h['model_kind'] == 'cddlgn':
            return f'CDDLGN {gates}g(参考)'
        tag = f'Mixer {gates}g'
        if isinstance(gw, int) and gw > 16:
            tag += f' (head gw={gw})'
        else:
            tag += ' (既定head)'
        return tag

    # 既定ヘッドのMixer=灰系、ヘッド拡張=青系、純MLP=赤系
    wide = [h for h in history
            if h['model_kind'] == 'mixer'
            and isinstance(h.get('head_group_width'), int)
            and h['head_group_width'] > 16]
    narrow = [h for h in history if h['model_kind'] == 'mixer' and h not in wide]
    # CDDLGNは別系統。純MLP基準線に混ぜない
    mlp = [h for h in history if h['model_kind'] == 'mlp_dlgn'
           and (h.get('max_steps') or 0) >= 1000]
    cddlgn = [h for h in history if h['model_kind'] == 'cddlgn'
              and (h.get('max_steps') or 0) >= 1000]

    blues = plt.cm.Blues(np.linspace(0.55, 0.95, max(len(wide), 1)))
    greys = plt.cm.Greys(np.linspace(0.35, 0.7, max(len(narrow), 1)))
    reds = plt.cm.Reds(np.linspace(0.5, 0.9, max(len(mlp), 1)))

    purples = plt.cm.Purples(np.linspace(0.55, 0.85, max(len(cddlgn), 1)))

    for group, colors, style, lw in (
        (sorted(narrow, key=lambda h: h.get('n_gates') or 0), greys, '-', 1.3),
        (sorted(mlp, key=lambda h: h.get('n_gates') or 0), reds, '--', 2.0),
        (sorted(cddlgn, key=lambda h: h.get('n_gates') or 0), purples, ':', 2.0),
        (sorted(wide, key=lambda h: h.get('n_gates') or 0), blues, '-', 2.4),
    ):
        for i, h in enumerate(group):
            xs = [p['step'] for p in h['points']]
            ys = [p['hard_acc'] for p in h['points']]
            ax.plot(xs, ys, style, color=colors[i], linewidth=lw,
                    label=_label(h), zorder=3 if style == '-' else 2)

    # difflogic系の到達点(水平破線)。予算が我々の約7.8倍である点に注意。
    # 3本が0.977〜0.985に密集するので、プロット上の注記ではなく凡例に入れる。
    for _gates, acc, lab, budget in _DIFFLOGIC_REFS:
        ax.axhline(acc, color='#2ca02c', linestyle='-.', linewidth=1.3, alpha=0.85,
                   label=f'difflogic {lab} / {budget}')

    ax.axhline(_CHANCE, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_xlabel('step')
    ax.set_ylabel('hard accuracy (MNIST test)')
    ax.set_title('学習曲線: LogicMixer(ヘッド拡張/既定) vs 純MLP DLGN(同一条件)')
    ax.grid(alpha=0.3)
    ax.set_ylim(0.0, 1.06)
    ax.legend(loc='lower right', fontsize=7.5, ncol=2)

    # 上軸にepoch(1 epoch = 60000/128 step)
    secax = ax.secondary_xaxis(
        'top',
        functions=(lambda s: s * 128 / MNIST_TRAIN_SIZE,
                   lambda e: e * MNIST_TRAIN_SIZE / 128))
    secax.set_xlabel('epoch (batch=128, MNIST train=60,000)')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='MIXER-1 図の生成')
    parser.add_argument('--runs', default='results/mixer1/runs.json')
    parser.add_argument('--history', default='results/mixer1/history.json')
    parser.add_argument('--out-dir', default='results/mixer1')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    font = _setup_japanese_font()
    print(f'日本語フォント: {font or "見つからず(ラベルが豆腐になる可能性)"}')

    rows = load_rows(args.runs)
    paths = []
    p1 = plot_scaling(rows, os.path.join(args.out_dir, 'gates_vs_hard_acc.png'))
    paths.append(p1)
    p2 = plot_coverage_ablation(
        rows, os.path.join(args.out_dir, 'coverage_ablation.png'))
    if p2:
        paths.append(p2)
    if os.path.exists(args.history):
        history = load_rows(args.history)
        paths.append(plot_learning_curves(
            history, os.path.join(args.out_dir, 'learning_curves.png')))
    for p in paths:
        print(f'保存: {p}')

    if not args.no_wandb:
        from PIL import Image
        used = init_wandb(
            run_name='mixer1_figures', config_dict={'n_runs': len(rows)},
            tags=['mlp_mixer', 'figures'], group='mixer-1', job_type='eval',
            notes='MIXER-1のゲート数vs精度図',
        )
        if used:
            log_images('mixer1/figures', [Image.open(p) for p in paths],
                       captions=[os.path.basename(p) for p in paths])
            wandb_finish()
            print('wandbへログしました')


if __name__ == '__main__':
    main()
