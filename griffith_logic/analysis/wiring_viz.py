"""論理ゲート配線の可視化(VIZ-1)。

DLGN系モデルの「どの層のどのゲートが、どこから入力を受け取っているか」を
機械的に抽出(`extract_wiring_spec`)し、2種類の図に描く。

    plot_full_wiring     — 全配線を1本ずつ描く詳細図(ゲート=点、配線=線分)
    plot_abstract_wiring — 層=長方形・配線=矢印に集約した俯瞰図

入力バイパス(IALGN型)のように隠れ層の入力が
``concat([h_prev, x, o_0, ...])`` という「プール」になっている構造を、
モデルのクラス名に依存せず**インデックス範囲の分解**として扱うのが要点。

用語:
    プール  — 隠れ層 LutLayer の入力ベクトル全体(幅 = in_dim)
    セグメント — プールを構成する各ソースの連続インデックス範囲
    配線本数 — `LutLayer.indices` (n_inputs, out_dim) のうち該当セグメントを
               指すエントリ数(= 実際に引かれている線の数)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# 描画色(ソースの種類ごと)
COLOR_SERIAL = '#888888'   # 直列(直前の層 → その層)
COLOR_INPUT = '#d62728'    # 入力バイパス(input → 隠れ層)
COLOR_BYPASS = '#1f77b4'   # 中間層バイパス(過去の層 → 隠れ層)

INPUT_NAME = 'input'
GROUP_SUM_NAME = 'group_sum'
INTERFACE_NAME = 'interface'
# 「入力側の source」として赤で描く層名(ハイブリッドの二値化界面を含む)
INPUT_LIKE_NAMES = (INPUT_NAME, INTERFACE_NAME)

# 全配線図のalpha自動調整(本数が多いほど薄くする)
_ALPHA_MIN = 0.02
_ALPHA_MAX = 0.5
_ALPHA_SCALE = 5000.0


@dataclass
class LayerNode:
    """1つの層(または入力ベクトル)。

    Attributes:
        name: 層名。'input' / 'layer0' / 'layer1' / 'group_sum' など
        width: ゲート数(input なら入力次元、group_sum ならクラス数)
    """

    name: str
    width: int


@dataclass
class WiringEdge:
    """あるソース層から、ある層への配線の集計。

    Attributes:
        src: ソースの LayerNode.name
        dst: 接続先の LayerNode.name
        n_wires: 配線本数
        detail: (n_inputs, out_dim) の int64。detail[slot, j] は
            「dst のゲート j の slot 番目の入力が src のどのインデックスから
            来ているか」を **src ローカルのインデックス**で表す。
            該当ソースでないエントリは -1(=マスク)。全配線図の描画に使う。
    """

    src: str
    dst: str
    n_wires: int
    detail: torch.Tensor | None = None


@dataclass
class WiringSpec:
    """モデル全体の配線構造。"""

    layers: list[LayerNode] = field(default_factory=list)
    edges: list[WiringEdge] = field(default_factory=list)

    def node(self, name: str) -> LayerNode:
        for n in self.layers:
            if n.name == name:
                return n
        raise KeyError(f'未知の層名: {name}')

    def index_of(self, name: str) -> int:
        for i, n in enumerate(self.layers):
            if n.name == name:
                return i
        raise KeyError(f'未知の層名: {name}')


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
    return type(module).__name__ == 'GroupSum' and hasattr(module, 'k')


def _collect_modules(model: torch.nn.Module):
    """LutLayer と GroupSum を定義順(= named_modules の順)で拾う。"""
    luts, group_sum = [], None
    for _, m in model.named_modules():
        if _is_lut_layer(m):
            luts.append(m)
        elif _is_group_sum(m) and group_sum is None:
            group_sum = m
    return luts, group_sum


def _default_segments(
    in_dim: int,
    prev_name: str,
    prev_width: int,
    input_width: int,
    past: list[LayerNode],
    dst: str,
) -> list[tuple[int, int, str]]:
    """プール(幅 in_dim)を [(オフセット, 幅, ソース名), ...] に分解する。

    InputBypassDLGN のプール構成 ``[h_prev(k), x(in_dim), o_0(k), ...]`` を
    クラス名に依存せず幅の一致だけで復元する汎用ロジック。
    先頭は必ず直前の層。以降は「元入力 → 古い層から順」に幅を突き合わせる。
    どれとも一致しない残りは `<dst>_extra` という合成ノードにまとめる
    (例: anchor選択学習モードの合成列)。
    """
    assert in_dim >= prev_width, (
        f'{dst}: in_dim({in_dim}) が直前層の幅({prev_width})より小さい。'
        'プール分解できない構造です'
    )
    segments = [(0, prev_width, prev_name)]
    cursor = prev_width
    candidates = [(INPUT_NAME, input_width)] + [(p.name, p.width) for p in past]
    for name, width in candidates:
        if cursor >= in_dim:
            break
        if width <= 0 or cursor + width > in_dim:
            continue
        segments.append((cursor, width, name))
        cursor += width
    if cursor < in_dim:
        segments.append((cursor, in_dim - cursor, f'{dst}_extra'))
    return segments


def count_wires(indices: torch.Tensor, offset: int, width: int):
    """セグメント [offset, offset+width) を指す配線の本数と detail を返す。

    `wiring_spec()` を自前実装するモデルからも使える公開ヘルパ。

    Returns:
        (n_wires, detail): detail は indices と同形で、該当セグメント外は -1、
        該当分は src ローカルインデックス(= indices - offset)。
    """
    idx = indices.detach().to('cpu').to(torch.int64)
    mask = (idx >= offset) & (idx < offset + width)
    detail = torch.where(mask, idx - offset, torch.full_like(idx, -1))
    return int(mask.sum()), detail


def extract_wiring_spec(
    model: torch.nn.Module,
    pool_map: dict[str, list[tuple[int, int, str]]] | None = None,
) -> WiringSpec:
    """モデルから配線構造を抽出する。

    Args:
        model: DLGN系モデル。`wiring_spec()` メソッドを持つ場合はそれに委譲する
            (ハイブリッドモデル用の拡張点)。持たない場合は
            LutLayer 列 + GroupSum の直列構造として推論する。
        pool_map: 層名 → [(オフセット, 幅, ソース名), ...] のプール分解の上書き。
            自動分解が意図と異なる特殊構造のときに使う。

    Returns:
        WiringSpec
    """
    if hasattr(model, 'wiring_spec'):
        spec = model.wiring_spec()
        assert isinstance(spec, WiringSpec), (
            f'wiring_spec() は WiringSpec を返す必要があります: {type(spec)}'
        )
        return spec

    luts, group_sum = _collect_modules(model)
    assert luts, 'LutLayer が1つも見つかりません'

    input_width = int(luts[0].in_dim)
    spec = WiringSpec(layers=[LayerNode(INPUT_NAME, input_width)])

    prev_name, prev_width = INPUT_NAME, input_width
    for i, lut in enumerate(luts):
        dst = f'layer{i}'
        in_dim, out_dim = int(lut.in_dim), int(lut.out_dim)
        # 過去の中間層(直前の層を除く)。プール後半の候補になる
        past = [n for n in spec.layers
                if n.name not in (INPUT_NAME, prev_name)
                and not n.name.endswith('_extra')]
        if pool_map is not None and dst in pool_map:
            segments = list(pool_map[dst])
        elif in_dim == prev_width:
            segments = [(0, prev_width, prev_name)]
        else:
            segments = _default_segments(
                in_dim, prev_name, prev_width, input_width, past, dst,
            )

        spec.layers.append(LayerNode(dst, out_dim))
        for offset, width, src in segments:
            if not any(n.name == src for n in spec.layers):
                # 合成セグメント(例: anchor合成列)はノードとして登録しておく。
                # dst の直前に挿すと直列エッジの隣接判定が崩れるため input の直後に置く
                spec.layers.insert(1, LayerNode(src, width))
            n_wires, detail = count_wires(lut.indices, offset, width)
            if n_wires == 0:
                continue
            spec.edges.append(WiringEdge(src, dst, n_wires, detail))

        prev_name, prev_width = dst, out_dim

    if group_sum is not None:
        k = int(group_sum.k)
        assert prev_width % k == 0, (
            f'GroupSum: 最終層の幅({prev_width})がクラス数({k})で割り切れない'
        )
        per_class = prev_width // k
        # クラス c は連続する per_class 個のゲートを合計する
        detail = torch.arange(prev_width, dtype=torch.int64).reshape(k, per_class).T
        spec.layers.append(LayerNode(GROUP_SUM_NAME, k))
        spec.edges.append(
            WiringEdge(prev_name, GROUP_SUM_NAME, prev_width, detail.contiguous())
        )

    return spec


# ============================================================
# 描画
# ============================================================

def _pyplot():
    """Aggバックエンドで matplotlib を遅延import する(figures extra)。"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _edge_color(spec: WiringSpec, edge: WiringEdge) -> str:
    """ソースの種類で色を決める。"""
    if spec.index_of(edge.src) == spec.index_of(edge.dst) - 1:
        return COLOR_SERIAL
    if edge.src in INPUT_LIKE_NAMES:
        return COLOR_INPUT
    return COLOR_BYPASS


def _ypos(index: torch.Tensor | int, width: int):
    """ゲート番号を [0, 1] の縦位置に正規化する。"""
    if isinstance(index, int):
        return (index + 0.5) / max(width, 1)
    return (index.to(torch.float64) + 0.5) / max(width, 1)


def plot_full_wiring(
    spec: WiringSpec,
    out_path: str,
    max_gates_per_layer: int | None = None,
) -> str:
    """全配線図(ゲート=点、配線=線分)を描いて PNG 保存する。

    ゲート数が多くても間引かずに全本数を描き、視認性は alpha の自動調整で担保する
    (本数が多いほど薄くなる)。`max_gates_per_layer` を指定した場合のみ各層の
    先頭N個に制限し、その旨を図中に注記する。

    Returns:
        out_path
    """
    import os

    from matplotlib.collections import LineCollection

    plt = _pyplot()

    limit = max_gates_per_layer
    drawn = {n.name: (n.width if limit is None else min(n.width, limit))
             for n in spec.layers}

    # detail を持つエッジ(=1本ずつ描けるもの)だけで alpha を決める。
    # 密結合NN層(detail=None)の本数を混ぜると論理ゲート側が見えなくなるため。
    drawable = [e for e in spec.edges if e.detail is not None]
    total_wires = sum(e.n_wires for e in drawable)
    alpha = min(_ALPHA_MAX, max(_ALPHA_MIN, _ALPHA_SCALE / max(total_wires, 1)))

    fig, ax = plt.subplots(figsize=(2.0 + 1.9 * len(spec.layers), 8))

    # 配線(ソース種別ごとに LineCollection をまとめて追加)
    by_color: dict[str, list] = {}
    for edge in spec.edges:
        if edge.detail is None:
            # 密結合(全結合)エッジ: 個別の線は描かず帯 + 本数ラベルで示す
            x0, x1 = spec.index_of(edge.src), spec.index_of(edge.dst)
            ax.fill_betweenx([0.0, 1.0], x0, x1, color='#bbbbbb', alpha=0.25,
                             zorder=0)
            ax.text((x0 + x1) / 2, 1.02, f'dense\n{edge.n_wires:,} wires',
                    ha='center', va='bottom', fontsize=8, color='#555555')
            continue
        color = _edge_color(spec, edge)
        x_src = spec.index_of(edge.src)
        x_dst = spec.index_of(edge.dst)
        src_w, dst_w = spec.node(edge.src).width, spec.node(edge.dst).width
        d = edge.detail
        slots, gates = torch.nonzero(d >= 0, as_tuple=True)
        src_idx = d[slots, gates]
        if limit is not None:
            keep = (gates < limit) & (src_idx < limit)
            gates, src_idx = gates[keep], src_idx[keep]
        if gates.numel() == 0:
            continue
        ys = _ypos(src_idx, src_w).numpy()
        yd = _ypos(gates, dst_w).numpy()
        seg = [((x_src, a), (x_dst, b)) for a, b in zip(ys, yd, strict=True)]
        by_color.setdefault(color, []).extend(seg)

    for color, segs in by_color.items():
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=0.5, alpha=alpha, zorder=1,
        ))

    # ゲート(点)と層ラベル
    for i, node in enumerate(spec.layers):
        n = drawn[node.name]
        ys = [(j + 0.5) / max(node.width, 1) for j in range(n)]
        ax.scatter([i] * n, ys, s=3, color='black', zorder=2)
        label = f'{node.name}\n({node.width})'
        ax.text(i, -0.06, label, ha='center', va='top', fontsize=9)

    ax.set_xlim(-0.5, len(spec.layers) - 0.5)
    ax.set_ylim(-0.12, 1.18)
    ax.set_axis_off()

    handles = [
        plt.Line2D([], [], color=COLOR_SERIAL, lw=2, label='serial (prev layer)'),
        plt.Line2D([], [], color=COLOR_INPUT, lw=2, label='input bypass'),
        plt.Line2D([], [], color=COLOR_BYPASS, lw=2, label='hidden bypass'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    n_dense = sum(e.n_wires for e in spec.edges if e.detail is None)
    note = f'wires={total_wires:,}  alpha={alpha:.3f}'
    if n_dense:
        note += f'  (+{n_dense:,} dense wires shown as bands)'
    if limit is not None:
        note += f'  (showing first {limit} gates per layer)'
    ax.set_title(f'full wiring — {note}', fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_abstract_wiring(spec: WiringSpec, out_path: str) -> str:
    """抽象配線図(層=長方形、直列=直線矢印、バイパス=曲線矢印)を PNG 保存する。

    バイパス矢印は層の上を迂回させ、太さを本数に比例させることで
    「入力がどの層まで直接届いているか」が一目で分かるようにする。

    Returns:
        out_path
    """
    import os

    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt = _pyplot()

    n_layers = len(spec.layers)
    box_w = 0.5
    widths = [n.width for n in spec.layers]
    max_h = max(math.log10(max(w, 1) + 1) for w in widths)
    heights = [1.6 * math.log10(max(w, 1) + 1) / max_h for w in widths]

    fig, ax = plt.subplots(figsize=(2.2 + 1.7 * n_layers, 6))

    for i, (node, h) in enumerate(zip(spec.layers, heights, strict=True)):
        ax.add_patch(Rectangle(
            (i - box_w / 2, -h / 2), box_w, h,
            facecolor='#eeeeee', edgecolor='black', lw=1.2, zorder=2,
        ))
        ax.text(i, 0, f'{node.name}\n{node.width}', ha='center', va='center',
                fontsize=9, zorder=3)

    max_wires = max((e.n_wires for e in spec.edges), default=1)
    top = max(heights) / 2

    for edge in spec.edges:
        color = _edge_color(spec, edge)
        i, j = spec.index_of(edge.src), spec.index_of(edge.dst)
        # 密結合NN層(数十万本)と論理ゲート層(数百本)が同居しても両方見えるよう
        # 太さは対数スケールにする
        lw = 1.0 + 4.0 * (math.log10(1 + edge.n_wires)
                          / math.log10(1 + max(max_wires, 1)))
        serial = (j == i + 1)
        if serial:
            arrow = FancyArrowPatch(
                (i + box_w / 2, 0), (j - box_w / 2, 0),
                arrowstyle='-|>', mutation_scale=14, color='black',
                lw=lw, zorder=1,
            )
            ax.add_patch(arrow)
            ax.text((i + j) / 2, 0.08, f'{edge.n_wires:,}', ha='center',
                    va='bottom', fontsize=8, color='black')
        else:
            # 上を迂回する曲線。距離が遠いほど大きく膨らませる
            rad = -0.18 - 0.06 * (j - i)
            arrow = FancyArrowPatch(
                (i, top), (j, top),
                arrowstyle='-|>', mutation_scale=14, color=color, lw=lw,
                connectionstyle=f'arc3,rad={rad}', zorder=1,
            )
            ax.add_patch(arrow)
            bulge = top + 0.55 * abs(rad) * (j - i) + 0.06
            ax.text((i + j) / 2, bulge, f'{edge.src}→{edge.dst}: {edge.n_wires:,}',
                    ha='center', va='bottom', fontsize=8, color=color)

    has_bypass = any(spec.index_of(e.dst) != spec.index_of(e.src) + 1
                     for e in spec.edges)
    ax.set_xlim(-0.8, n_layers - 0.2)
    # バイパス矢印が無ければ上の余白を詰める
    ax.set_ylim(-max(heights) / 2 - 0.4, top + (1.4 if has_bypass else 0.3))
    ax.set_axis_off()
    ax.set_title('abstract wiring (box height ∝ log(width), arrow width ∝ wires)',
                 fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
