"""配線可視化(VIZ-1)のテスト。

抽出(extract_wiring_spec)は matplotlib 非依存なので常に実行し、
描画(plot_*)は figures extra(matplotlib)がある環境でのみ実行する。
"""

import importlib.util

import pytest
import torch

from griffith_logic import GroupSum, LutLayer
from griffith_logic.analysis.wiring_viz import (
    COLOR_INPUT,
    LayerNode,
    WiringEdge,
    WiringSpec,
    extract_wiring_spec,
    plot_abstract_wiring,
    plot_full_wiring,
)

HAS_MPL = importlib.util.find_spec('matplotlib') is not None
requires_mpl = pytest.mark.skipif(not HAS_MPL, reason='matplotlibが必要(figures extra)')

IN_DIM = 16
K = 8
NUM_CLASSES = 2


def build_plain(num_layers: int = 3) -> torch.nn.Module:
    """純DLGN(直列のみ)。"""
    layers: list[torch.nn.Module] = [torch.nn.Flatten()]
    layers.append(LutLayer(in_dim=IN_DIM, out_dim=K, n_inputs=2,
                           connections='unique'))
    for _ in range(num_layers - 1):
        layers.append(LutLayer(in_dim=K, out_dim=K, n_inputs=2,
                               connections='random'))
    layers.append(GroupSum(NUM_CLASSES, tau=1.0))
    return torch.nn.Sequential(*layers)


class BypassNet(torch.nn.Module):
    """隠れ層が concat([h_prev(K), x(IN_DIM)]) を見る2層モデル(明示配線)。

    slot0 = 前層の同インデックス(spine)、slot1 = 元入力 → input バイパスが
    ちょうど K 本になることを検証できる。
    """

    def __init__(self):
        super().__init__()
        self.layer0 = LutLayer(in_dim=IN_DIM, out_dim=K, n_inputs=2,
                               connections='unique')
        conn = torch.stack([
            torch.arange(K, dtype=torch.int64),
            K + torch.arange(K, dtype=torch.int64) % IN_DIM,
        ])
        self.hidden = LutLayer(in_dim=K + IN_DIM, out_dim=K, n_inputs=2,
                               connections=conn)
        self.group_sum = GroupSum(NUM_CLASSES, tau=1.0)

    def forward(self, x):
        h = self.layer0(x)
        h = self.hidden(torch.cat([h, x], dim=-1))
        return self.group_sum(h)


def test_extract_plain_is_serial_only():
    spec = extract_wiring_spec(build_plain(num_layers=3))
    assert [n.name for n in spec.layers] == [
        'input', 'layer0', 'layer1', 'layer2', 'group_sum',
    ]
    assert [n.width for n in spec.layers] == [IN_DIM, K, K, K, NUM_CLASSES]

    # 直列(前層 → その層)のみ: srcは常に直前の層
    names = [n.name for n in spec.layers]
    for e in spec.edges:
        assert names.index(e.src) == names.index(e.dst) - 1, (e.src, e.dst)

    lut_edges = [e for e in spec.edges if e.dst != 'group_sum']
    assert len(lut_edges) == 3
    for e in lut_edges:
        assert e.n_wires == 2 * K  # n_inputs × out_dim が全て直前層を指す
    gs = [e for e in spec.edges if e.dst == 'group_sum'][0]
    assert gs.n_wires == K


def test_extract_bypass_counts_input_wires():
    spec = extract_wiring_spec(BypassNet())
    names = [n.name for n in spec.layers]
    assert names == ['input', 'layer0', 'layer1', 'group_sum']

    edges = {(e.src, e.dst): e for e in spec.edges}
    # 層0は入力を全スロットから受ける
    assert edges[('input', 'layer0')].n_wires == 2 * K
    # 隠れ層は slot0=spine(K本) / slot1=input バイパス(K本)
    assert edges[('layer0', 'layer1')].n_wires == K
    assert edges[('input', 'layer1')].n_wires == K

    # detail はマスク付き: 該当ソース以外は -1
    d = edges[('input', 'layer1')].detail
    assert d.shape == (2, K)
    assert bool((d[0] == -1).all())          # slot0 は input 由来でない
    assert bool((d[1] >= 0).all())           # slot1 は全て input 由来
    assert int(d[1].max()) < IN_DIM


def test_pool_map_overrides_segmentation():
    """pool_map でプール分解を明示上書きできる。"""
    spec = extract_wiring_spec(
        BypassNet(), pool_map={'layer1': [(0, K + IN_DIM, 'layer0')]},
    )
    edges = {(e.src, e.dst): e for e in spec.edges}
    assert ('input', 'layer1') not in edges
    assert edges[('layer0', 'layer1')].n_wires == 2 * K


def test_wiring_spec_protocol_is_used():
    """モデルが wiring_spec() を持てばそれに委譲する。"""
    custom = WiringSpec(
        layers=[LayerNode('input', 4), LayerNode('layer0', 2)],
        edges=[WiringEdge('input', 'layer0', 4)],
    )

    class WithProtocol(torch.nn.Module):
        def wiring_spec(self):
            return custom

    assert extract_wiring_spec(WithProtocol()) is custom


def test_input_bypass_edge_is_colored_red():
    from griffith_logic.analysis.wiring_viz import _edge_color

    spec = extract_wiring_spec(BypassNet())
    edge = [e for e in spec.edges if (e.src, e.dst) == ('input', 'layer1')][0]
    assert _edge_color(spec, edge) == COLOR_INPUT


def _hybrid_front_dlgn():
    """experiments/train_hybrid_front.py の HybridFrontDLGN を import する。"""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                    'experiments'))
    torchvision = pytest.importorskip('torchvision')  # noqa: F841
    from train_hybrid_front import HybridFrontDLGN

    return HybridFrontDLGN


def test_hybrid_wiring_spec_reports_bypass_edges():
    """HybridFrontDLGN.wiring_spec() が界面/中間層バイパスを本数付きで返す。"""
    hybrid_cls = _hybrid_front_dlgn()
    model = hybrid_cls(
        in_dim=64, num_classes=2, front_widths=[32, 8],
        dlgn_widths=[10, 20, 40], n_inputs=2, expand_fill='bypass', seed=0,
    )
    spec = extract_wiring_spec(model)
    assert [n.name for n in spec.layers] == [
        'input', 'front0(NN)', 'front1(NN)', 'interface',
        'dlgn0', 'dlgn1', 'dlgn2', 'group_sum',
    ]
    edges = {(e.src, e.dst): e for e in spec.edges}

    # 前段NNは全結合(detail無し = 全配線図では帯で表現)
    assert edges[('input', 'front0(NN)')].n_wires == 64 * 32
    assert edges[('input', 'front0(NN)')].detail is None
    # 二値化界面は要素ごと1対1
    assert edges[('front1(NN)', 'interface')].n_wires == 8

    # 余剰ポートのバイパス: dlgn1 はポート40 - 前層10 = 30本が界面から
    assert edges[('dlgn0', 'dlgn1')].n_wires == 10
    assert edges[('interface', 'dlgn1')].n_wires == 40 - 10
    # dlgn2 はポート80 - 前層20 = 60本が界面 + dlgn0 に分かれる
    assert edges[('dlgn1', 'dlgn2')].n_wires == 20
    aux = (edges[('interface', 'dlgn2')].n_wires
           + edges[('dlgn0', 'dlgn2')].n_wires)
    assert aux == 80 - 20

    # 後段DLGNのエッジは detail を持ち、全配線図で1本ずつ描ける
    d = edges[('interface', 'dlgn2')].detail
    assert d.shape == (2, 40)
    assert int(d.max()) < 8


def test_hybrid_wiring_spec_without_bypass_is_serial():
    """expand-fill=none の圧縮構成ではバイパスエッジが出ない。"""
    hybrid_cls = _hybrid_front_dlgn()
    model = hybrid_cls(
        in_dim=64, num_classes=2, front_widths=[32],
        dlgn_widths=[16, 8], n_inputs=2, expand_fill='none', seed=0,
    )
    spec = extract_wiring_spec(model)
    names = [n.name for n in spec.layers]
    for e in spec.edges:
        assert names.index(e.src) == names.index(e.dst) - 1, (e.src, e.dst)


@requires_mpl
@pytest.mark.parametrize('build', [build_plain, BypassNet])
def test_plot_full_wiring_writes_png(tmp_path, build):
    spec = extract_wiring_spec(build())
    out = str(tmp_path / 'full.png')
    assert plot_full_wiring(spec, out) == out
    assert (tmp_path / 'full.png').stat().st_size > 0


@requires_mpl
def test_plot_full_wiring_respects_max_gates(tmp_path):
    spec = extract_wiring_spec(build_plain())
    out = str(tmp_path / 'full_limited.png')
    plot_full_wiring(spec, out, max_gates_per_layer=3)
    assert (tmp_path / 'full_limited.png').stat().st_size > 0


@requires_mpl
@pytest.mark.parametrize('build', [build_plain, BypassNet])
def test_plot_abstract_wiring_writes_png(tmp_path, build):
    spec = extract_wiring_spec(build())
    out = str(tmp_path / 'abstract.png')
    assert plot_abstract_wiring(spec, out) == out
    assert (tmp_path / 'abstract.png').stat().st_size > 0
