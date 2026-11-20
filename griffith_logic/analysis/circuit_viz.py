"""回路図の可視化(VIZ-1): コードで書いた層をゲート+配線の図にする。

`LutLayer(2, 16)` という1行が、実際には「16個のゲート」と「そこへ入る配線」に
展開されていることを図にするモジュール。学習の前後で同じ図を描き分けられる。

    plot_circuit(model, resolve=False)  — 学習前/学習中。ノードは 'LUT' 表示で、
        16関数の混合がどれだけ確定に近いかを**色の濃さ**(分布エントロピー)で示す
    plot_circuit(model, resolve=True)   — 学習後。argmaxで確定した**ゲート名**
        (AND / XOR / !A / ...)をノード内に書く
    plot_abstract(model)                — ゲート数が多い網の俯瞰図(層=箱、配線=矢印)

配色の規約(プロジェクト共通):
    緑  — 論理ゲート(LutLayer のニューロン)
    橙  — 実数部(GroupSum のクラススコア)
    灰  — 入力ビット / 配線

ゲート名は 16 ゲートの標準定義(ビジュアルエディタと共通)に従い、真理値表 → ゲートID の
変換も同じ規約に従う。真理値表そのものは `export/discretize.truth_tables_from_lut_layer`
(= IR/Verilog が使う唯一の離散化経路)から取るので、図のラベルと生成HDLは必ず一致する。

配線の抽出ロジック(層の分解・本数集計)は実験ブランチ `experiment/wiring-viz` の
`analysis/wiring_viz.py` を出発点にしている。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..export.discretize import truth_tables_from_lut_layer

# N=2 の16ゲート名(ビジュアルエディタの GATE_NAMES と同一定義)
# (ゲートID = 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数)。
GATE_NAMES = {
    0: 'FALSE', 1: 'AND', 2: 'A>B', 3: 'A',
    4: 'B>A', 5: 'B', 6: 'XOR', 7: 'OR',
    8: 'NOR', 9: 'XNOR', 10: '!B', 11: 'A>=B',
    12: '!A', 13: 'B>=A', 14: 'NAND', 15: 'TRUE',
}

# 配色(プロジェクト記法)
COLOR_GATE = '#2e9e4f'      # 緑 — 論理ゲート
COLOR_REAL = '#e8850c'      # 橙 — 実数部(GroupSum)
COLOR_INPUT = '#8a8a8a'     # 灰 — 入力ビット
COLOR_WIRE = '#9aa0a6'      # 灰 — 配線

# 全ゲート描画の上限。これを超えたら plot_abstract へフォールバックする
MAX_FULL_GATES = 256

# 未確定ゲートの塗りの薄さ(confidence=0 のときの alpha)
_ALPHA_MIN = 0.12
_ALPHA_MAX = 1.0


@dataclass
class LayerNode:
    """1カラム(入力ビット列 / LutLayer / GroupSum)。"""

    name: str
    width: int
    kind: str  # 'input' | 'lut' | 'group_sum'
    module: torch.nn.Module | None = None


@dataclass
class CircuitSpec:
    """描画に必要な情報だけへ落としたモデルの構造。"""

    columns: list[LayerNode] = field(default_factory=list)

    @property
    def total_gates(self) -> int:
        """論理ゲート(LutLayerニューロン)の総数。"""
        return sum(c.width for c in self.columns if c.kind == 'lut')


# ============================================================
# 抽出
# ============================================================

def _is_lut_layer(module: torch.nn.Module) -> bool:
    """LutLayer 相当か(循環importを避けるためダックタイピングで判定)。"""
    return (hasattr(module, 'indices')
            and hasattr(module, 'in_dim')
            and hasattr(module, 'out_dim')
            and hasattr(module, 'n_inputs'))


def _is_group_sum(module: torch.nn.Module) -> bool:
    """GroupSum 相当か。"""
    return type(module).__name__ == 'GroupSum' and hasattr(module, 'k')


def extract_circuit(model: torch.nn.Module) -> CircuitSpec:
    """モデルを「入力ビット列 → LutLayer列 → GroupSum」のカラム列へ落とす。

    Args:
        model: LutLayer を含む nn.Module(Sequential を想定)

    Returns:
        CircuitSpec

    Raises:
        AssertionError: LutLayer が無い、または直列に繋がっていない場合
    """
    luts: list[torch.nn.Module] = []
    group_sum = None
    for _, m in model.named_modules():
        if _is_lut_layer(m):
            luts.append(m)
        elif _is_group_sum(m) and group_sum is None:
            group_sum = m
    assert luts, 'LutLayer が1つも見つかりません'

    spec = CircuitSpec()
    in_dim = int(luts[0].in_dim)
    spec.columns.append(LayerNode('input', in_dim, 'input'))

    prev_width = in_dim
    for i, lut in enumerate(luts):
        assert int(lut.in_dim) == prev_width, (
            f'layer{i}: in_dim({int(lut.in_dim)}) が直前カラムの幅({prev_width})と'
            '一致しません。直列構成のみ描画できます'
        )
        spec.columns.append(
            LayerNode(f'layer{i}', int(lut.out_dim), 'lut', lut)
        )
        prev_width = int(lut.out_dim)

    if group_sum is not None:
        k = int(group_sum.k)
        assert prev_width % k == 0, (
            f'GroupSum: 最終層の幅({prev_width})がクラス数({k})で割り切れない'
        )
        spec.columns.append(LayerNode('group_sum', k, 'group_sum', group_sum))
    return spec


def gate_labels(layer: torch.nn.Module) -> list[str]:
    """LutLayer の各ニューロンの確定ゲート名を返す(argmax後の離散回路の名前)。

    真理値表は `export/discretize.truth_tables_from_lut_layer` から取るので、
    IR / Verilog が焼くゲートと必ず同じものに名前を付けることになる。
    N=2 では16ゲート名、N>2 では真理値表の16進表記(`LUT:xx`)を返す。

    Args:
        layer: LutLayer

    Returns:
        長さ out_dim の名前リスト
    """
    tables = truth_tables_from_lut_layer(layer)
    if int(layer.n_inputs) != 2:
        return [f'LUT:{int(t):x}' for t in tables]
    return [GATE_NAMES[tt_to_gate_id_n2(int(t))] for t in tables]


def tt_to_gate_id_n2(tt: int) -> int:
    """N=2真理値表(bit k = butterfly基底順)を16ゲートIDへ変換する。

    ビジュアルエディタ側の同名関数と同一規約
    (ゲートID = 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数)。

    Args:
        tt: 真理値表(bit k = 入力パターン k の出力)

    Returns:
        0-15 のゲートID
    """
    b00 = tt & 1
    b10 = (tt >> 1) & 1
    b01 = (tt >> 2) & 1
    b11 = (tt >> 3) & 1
    return (b00 << 3) | (b01 << 2) | (b10 << 1) | b11


def gate_confidence(layer: torch.nn.Module) -> torch.Tensor:
    """各ニューロンの「決まり具合」を [0,1] で返す(1=完全に確定)。

    softmax パラメータ化では正規化エントロピーの補数 `1 - H/ln(K)`。
    確率解釈のないパラメータ化(warp)は中立値 0.5 を返す。
    (ビジュアルエディタの neuron_uncertainty と同じ定義の裏返し)

    Args:
        layer: LutLayer

    Returns:
        (out_dim,) の float テンソル
    """
    import torch.nn.functional as F

    with torch.no_grad():
        w = layer.weights.detach().to(torch.float32).cpu()
        param = layer.parameterization
        if param == 'softmax':
            probs = F.softmax(w, dim=-1)
            k = int(probs.shape[-1])
            if k <= 1:
                return torch.ones(probs.shape[0])
            p = probs.clamp_min(1e-12)
            entropy = -(p * p.log()).sum(dim=-1) / math.log(k)
            return (1.0 - entropy).clamp(0.0, 1.0)
        if param in ('independent', 'iwp', 'raw'):
            p = torch.sigmoid(w) if param != 'raw' else w.clamp(0.0, 1.0)
            p = p.clamp(1e-7, 1.0 - 1e-7)
            h = -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)
            return (1.0 - h.mean(dim=-1)).clamp(0.0, 1.0)
    return torch.full((int(layer.out_dim),), 0.5)


# ============================================================
# 描画
# ============================================================

def _new_ax(ax, width: float, height: float):
    """ax が None なら figure ごと作る。"""
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=(width, height))
    return ax


def _ys(width: int) -> list[float]:
    """幅 width のカラムのゲートを [0,1] に等間隔配置する。"""
    return [(j + 0.5) / max(width, 1) for j in range(width)]


def plot_circuit(
    model: torch.nn.Module,
    resolve: bool = False,
    ax=None,
    title: str | None = None,
    auto_abstract: bool = True,
):
    """モデルを左→右の層状回路図として描く。

    Args:
        model: LutLayer(+GroupSum)からなるモデル
        resolve: True で各ゲートを argmax 確定し**ゲート名**を書く(学習後の図)。
            False では 'LUT' 表示のまま、16関数の混合の確からしさを色の濃さで示す
        ax: 描画先の Axes(None なら新規作成)
        title: 図タイトル(None なら resolve に応じた既定文言)
        auto_abstract: ゲート総数が MAX_FULL_GATES を超えたとき
            `plot_abstract` の俯瞰図へ自動フォールバックする(既定True)

    Returns:
        描画した Axes
    """
    from matplotlib.patches import FancyBboxPatch

    spec = extract_circuit(model)
    if auto_abstract and spec.total_gates > MAX_FULL_GATES:
        return plot_abstract(model, ax=ax, title=title)

    n_col = len(spec.columns)
    max_width = max(c.width for c in spec.columns)
    ax = _new_ax(ax, 2.2 + 2.0 * n_col, max(4.0, 0.30 * max_width + 1.5))

    row_h = 1.0 / max(max_width, 1)
    box_h = min(0.62 * row_h, 0.09)
    box_w = 0.46
    font = max(3.5, min(9.0, 110.0 / max(max_width, 1)))

    # --- 配線(ノードの下に敷く) ---
    for ci, col in enumerate(spec.columns):
        if col.kind == 'lut':
            idx = col.module.indices.detach().cpu().to(torch.int64)
            src = spec.columns[ci - 1]
            for o in range(col.width):
                for j in range(int(col.module.n_inputs)):
                    s = int(idx[j, o])
                    ax.plot(
                        [ci - 1 + box_w / 2, ci - box_w / 2],
                        [_ys(src.width)[s], _ys(col.width)[o]],
                        color=COLOR_WIRE, lw=0.6, alpha=0.55, zorder=1,
                    )
        elif col.kind == 'group_sum':
            src = spec.columns[ci - 1]
            group = src.width // col.width
            for c in range(col.width):
                for i in range(group):
                    ax.plot(
                        [ci - 1 + box_w / 2, ci - box_w / 2],
                        [_ys(src.width)[c * group + i], _ys(col.width)[c]],
                        color=COLOR_REAL, lw=0.6, alpha=0.45, zorder=1,
                    )

    # --- ノード ---
    for ci, col in enumerate(spec.columns):
        ys = _ys(col.width)
        if col.kind == 'lut':
            labels = gate_labels(col.module) if resolve else ['LUT'] * col.width
            conf = (torch.ones(col.width) if resolve
                    else gate_confidence(col.module))
        elif col.kind == 'group_sum':
            labels = [f'Σ{c}' for c in range(col.width)]
            conf = torch.ones(col.width)
        else:
            labels = [f'x{j}' for j in range(col.width)]
            conf = torch.ones(col.width)

        face = {'lut': COLOR_GATE, 'group_sum': COLOR_REAL,
                'input': COLOR_INPUT}[col.kind]
        for j, y in enumerate(ys):
            alpha = _ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * float(conf[j])
            ax.add_patch(FancyBboxPatch(
                (ci - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle=f'round,pad=0,rounding_size={box_h / 2:.4f}',
                facecolor=face, alpha=alpha, edgecolor='#333333', lw=0.5,
                zorder=2, mutation_aspect=1.0,
            ))
            # 薄い塗りの上でも読めるよう文字は常に不透明の黒
            ax.text(ci, y, labels[j], ha='center', va='center',
                    fontsize=font, color='#111111', zorder=3)

        caption = {'lut': f'LutLayer\n({col.width} gates)',
                   'group_sum': f'GroupSum\n(k={col.width})',
                   'input': f'input\n({col.width} bits)'}[col.kind]
        ax.text(ci, -0.055, caption, ha='center', va='top', fontsize=9)

    if title is None:
        title = ('after training — each gate resolved to one logic function'
                 if resolve else
                 'before/while training — each gate is a mixture of 16 functions'
                 ' (paler = less decided)')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.7, n_col - 0.3)
    ax.set_ylim(-0.16, 1.06)
    ax.set_axis_off()
    return ax


def plot_abstract(model: torch.nn.Module, ax=None, title: str | None = None):
    """層=箱・配線=本数付き矢印の俯瞰図(ゲート数が多いモデル向け)。

    experiment/wiring-viz の `plot_abstract_wiring` を、Sequential構成向けに
    簡約して移植したもの。

    Args:
        model: LutLayer(+GroupSum)からなるモデル
        ax: 描画先の Axes(None なら新規作成)
        title: 図タイトル

    Returns:
        描画した Axes
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    spec = extract_circuit(model)
    n_col = len(spec.columns)
    ax = _new_ax(ax, 2.2 + 1.8 * n_col, 4.5)

    box_w = 0.5
    widths = [c.width for c in spec.columns]
    max_h = max(math.log10(w + 1) for w in widths)
    heights = [1.6 * math.log10(w + 1) / max_h for w in widths]

    for i, (col, h) in enumerate(zip(spec.columns, heights, strict=True)):
        face = {'lut': COLOR_GATE, 'group_sum': COLOR_REAL,
                'input': COLOR_INPUT}[col.kind]
        ax.add_patch(Rectangle(
            (i - box_w / 2, -h / 2), box_w, h,
            facecolor=face, alpha=0.35, edgecolor='#333333', lw=1.2, zorder=2,
        ))
        ax.text(i, 0, f'{col.name}\n{col.width}', ha='center', va='center',
                fontsize=9, zorder=3)

    for i in range(1, n_col):
        col = spec.columns[i]
        if col.kind == 'lut':
            n_wires = col.width * int(col.module.n_inputs)
        else:
            n_wires = spec.columns[i - 1].width
        ax.add_patch(FancyArrowPatch(
            (i - 1 + box_w / 2, 0), (i - box_w / 2, 0),
            arrowstyle='-|>', mutation_scale=14, color='#444444', lw=1.4,
            zorder=1,
        ))
        ax.text(i - 0.5, 0.06, f'{n_wires:,} wires', ha='center', va='bottom',
                fontsize=8)

    ax.set_title(title or f'abstract circuit — {spec.total_gates:,} gates',
                 fontsize=11)
    ax.set_xlim(-0.8, n_col - 0.2)
    ax.set_ylim(-max(heights) / 2 - 0.4, max(heights) / 2 + 0.5)
    ax.set_axis_off()
    return ax
