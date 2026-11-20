"""
 学習曲線の作図（BLEU / accuracy / loss）

**データ源は Modal Volume 上の `log.txt` を主とする。** wandb の `state` が誤って
`finished` を返した事故（レポート §4.6.6）を踏まえ、wandb が壊れていても図が
作れる経路を正本にしてある（`log.txt` は逐次フラッシュされるので走行中の run も
そこまでの部分曲線が描ける）。wandb は `--source wandb` で補助的に使える。

描く図:
    図A `bleu`  — BLEU の学習曲線。論理版は soft(実線)/hard(破線)を対にして
                  **崩壊ギャップ**が視覚的に読めるようにする。
                  **論文値の水平線は引かない**（評価系が違い比較不能なため）。
    図B `acc`   — token accuracy の学習曲線。**論文 RDDLGN の 30.9% だけ**
                  参考の水平線を引く（accuracy は BLEU より評価系に頑健と
                  考えられるため。ただし「参考・別評価系」と図中に明記）。
    図C `loss`  — loss/main の学習曲線（発散・非単調性の確認用）。

方針:
    - **平滑化・補間をしない**。ログに残っている点だけを打ち、線は薄く繋ぐだけ。
    - BLEU は `--bleu-every 10000` のため点が疎。**マーカーを大きくして疎さを隠さない**。
    - 背景色を明示し、ライト/ダークどちらでも読めるようにする。

使用例:
    # Volume から log.txt を取得（このスクリプトは取得しない。取得は下記コマンド）
    uv run modal volume get "${MD_MODAL_VOLUME:-griffith-workspace}" \\
        checkpoints/<run_dir>/log.txt results/tsmix1d/logs/<run_dir>.log

    # 図を生成（既定の入力ディレクトリは results/tsmix1d/logs）
    uv run python experiments/tsmix1_plot_curves.py

    # wandb にも図をログする
    uv run python experiments/tsmix1_plot_curves.py --wandb
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 日本語フォント候補（experiments/mixer1_plot.py と同じ流儀）
_JP_FONTS = (
    'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic',
    'Noto Sans CJK JP', 'IPAexGothic', 'TakaoGothic',
)

# 論文 RDDLGN の token accuracy（表B・別評価系。図Bの参考線にのみ使う）
_PAPER_RDDLGN_ACC = 0.309

# 背景・前景（ライト/ダーク両対応のため明示指定する）
_BG = '#ffffff'
_FG = '#1a1a1a'
_GRID = '#d8d8d8'


def setup_japanese_font() -> str | None:
    """利用可能な日本語フォントを rcParams へ設定し、その名前を返す。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONTS:
        if name in available:
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return name
    return None


# ============================================================
# log.txt のパース
# ============================================================

# 共通: ITER <step>/<total>
_RE_ITER = re.compile(r'ITER (\d+)/(\d+)')
# 論理版: main=... acc_soft=... acc_hard=... BLEU(soft)=... BLEU(hard)=...
_RE_MAIN = re.compile(r'\bmain=([0-9.]+)')
_RE_ACC_SOFT = re.compile(r'\bacc_soft=([0-9.]+)')
_RE_ACC_HARD = re.compile(r'\bacc_hard=([0-9.]+)')
# 埋め込み二値化正則化の重み(step 1,000→100,000 で 0.0→0.1 に線形ランプ)と損失
_RE_EMB_W = re.compile(r'\bemb_w=([0-9.]+)')
_RE_EMB_LOSS = re.compile(r'\bemb=([0-9.]+)')
_RE_BLEU_SOFT = re.compile(r'BLEU\(soft\)=([0-9.]+)')
_RE_BLEU_HARD = re.compile(r'BLEU\(hard\)=([0-9.]+)')
# FP版: loss=... acc=... BLEU=...
_RE_LOSS = re.compile(r'\bloss=([0-9.]+)')
_RE_ACC = re.compile(r'\bacc=([0-9.]+)')
_RE_BLEU = re.compile(r'(?<![(\w])BLEU=([0-9.]+)')
# DONE 行（完走の唯一の証拠。BLEUはここだけフル精度で残る）
_RE_DONE_BLEU = re.compile(r'\bbleu=([0-9.]+)')
_RE_DONE_BLEU_SOFT = re.compile(r'\bbleu_soft=([0-9.]+)')
_RE_DONE_BLEU_HARD = re.compile(r'\bbleu_hard=([0-9.]+)')
_RE_START_PARAMS = re.compile(r'params=([\d,]+)')
_RE_START_NI = re.compile(r'\bni=(\d+)')


@dataclass
class Curve:
    """1 run 分の学習曲線（log.txt のパース結果）。"""

    name: str
    path: str
    is_logic: bool = False
    total_steps: int | None = None
    params: int | None = None
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    acc_soft: list[float] = field(default_factory=list)
    acc_hard: list[float] = field(default_factory=list)
    emb_w: list[float] = field(default_factory=list)
    emb_loss: list[float] = field(default_factory=list)
    bleu_steps: list[int] = field(default_factory=list)
    bleu_soft: list[float] = field(default_factory=list)
    bleu_hard: list[float] = field(default_factory=list)
    done: bool = False
    final_bleu_soft: float | None = None
    final_bleu_hard: float | None = None

    @property
    def last_step(self) -> int:
        return self.steps[-1] if self.steps else 0

    def status(self) -> str:
        """完走したかを `log.txt` の DONE 行だけで判定する（wandb は信用しない）。"""
        if self.done:
            return f'完走({self.last_step:,} step)'
        total = f'/{self.total_steps:,}' if self.total_steps else ''
        return f'走行中({self.last_step:,}{total} step)'


def parse_log(path: str, name: str) -> Curve:
    """`log.txt` を1本パースして :class:`Curve` を返す。

    論理版(`acc_soft=`/`BLEU(soft)=`)と FP版(`acc=`/`BLEU=`)の両方式を自動判別する。
    **値は行に書かれているものだけを使い、欠損を補間しない。**
    """
    curve = Curve(name=name, path=path)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if 'START ' in line:
                m = _RE_START_PARAMS.search(line)
                if m:
                    curve.params = int(m.group(1).replace(',', ''))
                m = _RE_START_NI.search(line)
                if m:
                    curve.total_steps = int(m.group(1))
                continue
            if 'DONE ' in line:
                curve.done = True
                m = _RE_DONE_BLEU_SOFT.search(line)
                if m:
                    curve.final_bleu_soft = float(m.group(1))
                m = _RE_DONE_BLEU_HARD.search(line)
                if m:
                    curve.final_bleu_hard = float(m.group(1))
                m = _RE_DONE_BLEU.search(line)
                if m and curve.final_bleu_soft is None:
                    curve.final_bleu_soft = float(m.group(1))
                    curve.final_bleu_hard = float(m.group(1))
                continue

            m_iter = _RE_ITER.search(line)
            if not m_iter:
                continue
            step = int(m_iter.group(1))
            curve.total_steps = curve.total_steps or int(m_iter.group(2))

            m_soft = _RE_ACC_SOFT.search(line)
            if m_soft:  # 論理版
                curve.is_logic = True
                m_main = _RE_MAIN.search(line)
                m_hard = _RE_ACC_HARD.search(line)
                m_w = _RE_EMB_W.search(line)
                m_e = _RE_EMB_LOSS.search(line)
                if m_main and m_hard:
                    curve.steps.append(step)
                    curve.loss.append(float(m_main.group(1)))
                    curve.acc_soft.append(float(m_soft.group(1)))
                    curve.acc_hard.append(float(m_hard.group(1)))
                    curve.emb_w.append(float(m_w.group(1)) if m_w else float('nan'))
                    curve.emb_loss.append(
                        float(m_e.group(1)) if m_e else float('nan'))
                b_s = _RE_BLEU_SOFT.search(line)
                b_h = _RE_BLEU_HARD.search(line)
                if b_s and b_h:
                    curve.bleu_steps.append(step)
                    curve.bleu_soft.append(float(b_s.group(1)))
                    curve.bleu_hard.append(float(b_h.group(1)))
            else:  # FP版
                m_loss = _RE_LOSS.search(line)
                m_acc = _RE_ACC.search(line)
                if m_loss and m_acc:
                    curve.steps.append(step)
                    curve.loss.append(float(m_loss.group(1)))
                    curve.acc_soft.append(float(m_acc.group(1)))
                b = _RE_BLEU.search(line)
                if b:
                    curve.bleu_steps.append(step)
                    curve.bleu_soft.append(float(b.group(1)))
    return curve


# ============================================================
# 描画対象の定義（run_dir → 表示名・色）
# ============================================================

@dataclass(frozen=True)
class Series:
    run_dir: str
    label: str
    color: str


_SERIES: tuple[Series, ...] = (
    Series('fpmixer_wmt14_paper_none_20260909_230436',
           'FP Mixer gather相当 (float)', '#1f77b4'),
    Series('fpmixer_wmt14_paper_single_head_20260909_230505',
           'FP Mixer cross相当 (float)', '#17a2b8'),
    Series('fpgru_wmt14_tiny_gru_20260909_230548',
           'FP GRU (float, 較正用)', '#7f7f7f'),
    Series('seqmixer_wmt14_paper_none_20260909_104230',
           '論理版 gather', '#d62728'),
    Series('seqmixer_wmt14_paper_logic_single_head_20260909_104254',
           '論理版 broadcast+cross', '#ff7f0e'),
    Series('seqmixer_wmt14_paper_none_20260909_104208',
           '論理版 broadcast (5,000step)', '#9467bd'),
)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_FG, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=_FG, fontsize=10)
    ax.set_ylabel(ylabel, color=_FG, fontsize=10)
    ax.tick_params(colors=_FG, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def _new_fig(width: float = 9.6, height: float = 6.4):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=_BG)
    return fig, ax


def _finish(fig, ax, out_path: str, note: str, ncol: int = 3) -> None:
    """凡例を軸の下へ出し、注記を折り返して保存する（曲線が凡例で隠れないように）。"""
    ax.legend(fontsize=8.5, ncol=ncol, framealpha=0.0,
              loc='upper center', bbox_to_anchor=(0.5, -0.13))
    fig.text(0.012, 0.012, note, fontsize=7.5, color='#555555', va='bottom')
    n_lines = note.count('\n') + 1
    fig.tight_layout(rect=(0, 0.02 + 0.022 * n_lines, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=_BG)
    plt.close(fig)


def plot_bleu(curves: dict[str, Curve], out_path: str) -> None:
    """図A: BLEU の学習曲線。論理版は soft(実線)/hard(破線)を対にする。

    **論文値の水平線は引かない**。GRU 較正で +13.79 BLEU の評価系差が確認されており
    （レポート §4.9.6）、同じ図に置くと比較可能に見えてしまうためである。
    """
    fig, ax = _new_fig()
    for s in _SERIES:
        c = curves.get(s.run_dir)
        if c is None or not c.bleu_steps:
            continue
        if c.is_logic:
            ax.plot(c.bleu_steps, c.bleu_soft, '-o', color=s.color, markersize=7,
                    linewidth=1.4, alpha=0.9, label=f'{s.label} soft')
            ax.plot(c.bleu_steps, c.bleu_hard, '--s', color=s.color, markersize=6,
                    linewidth=1.4, alpha=0.7, label=f'{s.label} hard')
            # 崩壊ギャップを縦の細線で可視化
            for st, bs, bh in zip(c.bleu_steps, c.bleu_soft, c.bleu_hard, strict=True):
                ax.plot([st, st], [bh, bs], color=s.color, linewidth=1.0, alpha=0.35)
        else:
            ax.plot(c.bleu_steps, c.bleu_soft, '-o', color=s.color, markersize=7,
                    linewidth=1.4, alpha=0.9, label=s.label)
    _style_axes(ax, '図A: BLEU の学習曲線（我々の評価系。論文値と比較不能）',
                'step', 'BLEU (sacrebleu 13a)')
    _finish(fig, ax, out_path,
            'BLEUは --bleu-every 10000 のため点が疎（マーカー位置が実測点。'
            '平滑化・補間なし）。論理版は実線=soft / 破線=hard で、\n'
            '両者を結ぶ縦の細線が崩壊ギャップ。'
            '論文値の水平線は引いていない（評価系が違い比較不能なため。§4.9.6）。')


def plot_acc(curves: dict[str, Curve], out_path: str) -> None:
    """図B: token accuracy の学習曲線。論文 RDDLGN の 30.9% のみ参考線を引く。"""
    fig, ax = _new_fig()
    for s in _SERIES:
        c = curves.get(s.run_dir)
        if c is None or not c.steps:
            continue
        if c.is_logic:
            ax.plot(c.steps, c.acc_soft, '-', color=s.color, linewidth=1.6,
                    alpha=0.95, label=f'{s.label} soft')
            ax.plot(c.steps, c.acc_hard, '--', color=s.color, linewidth=1.4,
                    alpha=0.7, label=f'{s.label} hard')
        else:
            ax.plot(c.steps, c.acc_soft, '-', color=s.color, linewidth=1.6,
                    alpha=0.95, label=s.label)
    ax.axhline(_PAPER_RDDLGN_ACC, color='#2ca02c', linewidth=1.2,
               linestyle=':', alpha=0.9)
    ax.annotate('論文 RDDLGN acc 30.9%（参考・別評価系）',
                (0.012, _PAPER_RDDLGN_ACC + 0.006), xycoords=('axes fraction', 'data'),
                fontsize=8.5, color='#2ca02c')
    _style_axes(ax, '図B: token accuracy の学習曲線（PADを除く一致率）',
                'step', 'token accuracy')
    _finish(fig, ax, out_path,
            'accuracyはBLEUより評価系に頑健と考えられるため、論文値を参考の水平線として'
            '1本だけ引いている\n'
            '（あくまで参考。トークナイザが違えばトークン単位も違う）。'
            '図AのBLEUには同じ線を引いていない。')


def plot_loss(curves: dict[str, Curve], out_path: str) -> None:
    """図C: loss/main の学習曲線（発散・非単調性の確認用）。"""
    fig, ax = _new_fig()
    for s in _SERIES:
        c = curves.get(s.run_dir)
        if c is None or not c.steps:
            continue
        ax.plot(c.steps, c.loss, '-', color=s.color, linewidth=1.6,
                alpha=0.95, label=s.label)
    _style_axes(ax, '図C: loss/main の学習曲線',
                'step', 'cross entropy (label smoothing 0.1)')
    _finish(fig, ax, out_path,
            'FP版と論理版は学習率が違う（1e-3 / 0.05）ため、loss の絶対値の高低を'
            '優劣として読まないこと。\n曲線の形（発散・停滞・非単調性）を見るための図。')


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """標本ピアソン相関（点が2未満、または分散0なら None）。"""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return None if den == 0 else num / den


def plot_gap(curves: dict[str, Curve], out_path: str) -> dict[str, float]:
    """図D: 崩壊ギャップ(soft − hard)の推移と、埋め込み二値化正則化の重み emb_w。

    「soft の低下は emb_w のランプで soft が hard へ降りてきているだけで、
    崩壊ギャップは学習とともに縮小するはず」という仮説を検証するための図。
    **仮説に合わせて数字を作らない**。実測点だけを打つ。

    Returns:
        run_dir -> r(emb_w, accギャップ) の相関。仮説が正しければ**負**になる。
    """
    fig, ax = _new_fig()
    ax2 = ax.twinx()
    correlations: dict[str, float] = {}

    for s in _SERIES:
        c = curves.get(s.run_dir)
        if c is None or not c.is_logic or not c.steps:
            continue
        acc_gap = [a - h for a, h in zip(c.acc_soft, c.acc_hard, strict=True)]
        ax.plot(c.steps, acc_gap, '-', color=s.color, linewidth=1.7, alpha=0.95,
                label=f'{s.label} accギャップ')
        if c.bleu_steps:
            bleu_gap = [bs - bh for bs, bh
                        in zip(c.bleu_soft, c.bleu_hard, strict=True)]
            ax2.plot(c.bleu_steps, bleu_gap, '--o', color=s.color, markersize=7,
                     linewidth=1.3, alpha=0.6, label=f'{s.label} BLEUギャップ(右軸)')
        r = pearson(c.emb_w, acc_gap)
        if r is not None:
            correlations[s.run_dir] = r

    # emb_w のランプ(全論理版runで共通の線形ランプ)を薄く重ねる
    ref = next((curves[s.run_dir] for s in _SERIES
                if s.run_dir in curves and curves[s.run_dir].is_logic
                and curves[s.run_dir].emb_w), None)
    if ref is not None:
        ax.plot(ref.steps, [w / 10.0 for w in ref.emb_w], ':', color='#2ca02c',
                linewidth=1.6, label='emb_w ÷10（正則化ランプ・形の参照用）')

    _style_axes(ax, '図D: 崩壊ギャップ(soft − hard)の推移と正則化ランプ',
                'step', 'token accuracy のギャップ')
    ax2.set_ylabel('BLEU のギャップ（右軸・点は疎）', color=_FG, fontsize=10)
    ax2.tick_params(colors=_FG, labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=8.0, ncol=2,
              framealpha=0.0, loc='upper center', bbox_to_anchor=(0.5, -0.13))
    note = ('仮説「emb_wのランプでsoftがhardへ降り、崩壊ギャップは縮小する」の検証図。'
            '予測が正しければaccギャップは減少し\n'
            'r(emb_w, accギャップ)は負になるはずだが、実測は増加・正の相関である'
            '（§4.11.4）。BLEUギャップは点が2つしかない。')
    fig.text(0.012, 0.012, note, fontsize=7.5, color='#555555', va='bottom')
    fig.tight_layout(rect=(0, 0.066, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=_BG)
    plt.close(fig)
    return correlations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='の学習曲線を log.txt から作図する')
    parser.add_argument('--log-dir', type=str, default='results/tsmix1d/logs',
                        help='log.txt を置いたディレクトリ (default: results/tsmix1d/logs)')
    parser.add_argument('--out-dir', type=str, default='results/tsmix1d/figures',
                        help='図の出力先 (default: results/tsmix1d/figures)')
    parser.add_argument('--prefix', type=str, default='tsmix1-curves',
                        help='出力ファイル名の接頭辞')
    parser.add_argument('--wandb', action='store_true',
                        help='生成した図を wandb にもログする（PJ規約）')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    font = setup_japanese_font()
    print(f'日本語フォント: {font or "見つからず(ラベルが豆腐になる可能性)"}')
    print(f'JST: {datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")}')

    curves: dict[str, Curve] = {}
    for s in _SERIES:
        path = os.path.join(args.log_dir, f'{s.run_dir}.log')
        if not os.path.exists(path):
            print(f'  [skip] {s.run_dir}: {path} が無い')
            continue
        curve = parse_log(path, s.label)
        curves[s.run_dir] = curve
        bleu_str = (f'BLEU点={len(curve.bleu_steps)}'
                    + (f' 最終={curve.final_bleu_soft:.3f}'
                       if curve.final_bleu_soft is not None else ''))
        print(f'  {s.label}: {curve.status()} 評価点={len(curve.steps)} {bleu_str}'
              + (f' params={curve.params:,}' if curve.params else ''))

    if not curves:
        raise SystemExit(f'log が1本も見つかりません: {args.log_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        'bleu': os.path.join(args.out_dir, f'{args.prefix}-bleu.png'),
        'acc': os.path.join(args.out_dir, f'{args.prefix}-acc.png'),
        'loss': os.path.join(args.out_dir, f'{args.prefix}-loss.png'),
    }
    outputs['gap'] = os.path.join(args.out_dir, f'{args.prefix}-gap.png')
    plot_bleu(curves, outputs['bleu'])
    plot_acc(curves, outputs['acc'])
    plot_loss(curves, outputs['loss'])
    correlations = plot_gap(curves, outputs['gap'])
    for run_dir, r in correlations.items():
        verdict = '仮説と整合(負)' if r < 0 else '**仮説と反対(正)**'
        print(f'  r(emb_w, accギャップ) {run_dir}: {r:+.3f} → {verdict}')
    for key, path in outputs.items():
        print(f'図{key}: {os.path.abspath(path)}')

    if args.wandb:
        from env_utils import load_dotenv
        from PIL import Image
        from wandb_utils import finish as wandb_finish
        from wandb_utils import init_wandb, log_images, normalize_tags
        load_dotenv()
        use_wandb = init_wandb(
            run_name='tsmix1_curves',
            config_dict={'log_dir': args.log_dir, 'runs': list(curves)},
            tags=['tsmix-1d', 'figures'] + normalize_tags(series='TSMIX-1'),
            group='tsmix-1',
            job_type='eval',
        )
        if use_wandb:
            log_images(
                'tsmix1/curves',
                [Image.open(outputs[k]) for k in ('bleu', 'acc', 'loss', 'gap')],
                captions=['図A: BLEU', '図B: accuracy', '図C: loss/main',
                          '図D: 崩壊ギャップと正則化ランプ'],
            )
            wandb_finish()
            print('wandb へ図をログしました')


if __name__ == '__main__':
    main()
