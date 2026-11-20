"""バタフライ因子化QAT → fan-in=2 native DLGN転写(multilut Phase 3)。

密線形層 W をバタフライ因子積 X⁽¹⁾…X⁽ᴸ⁾(各因子は行あたり非ゼロ≤2)へ
構造制約し、ternary量子化(STE)+二値活性(HeavisideSTE)でQATする。
forward自体が離散関数(soft/hardギャップが構造的にゼロ)なので、
確定後は各因子ニューロンが2入力閾値ゲートとなり、
threshold_truth_tables → gate_id_from_tt_n2 で16ネイティブゲートへ
**ロスレス**転写できる。

意味論のground truth: core/reference.py::reference_butterfly_forward。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..core.reference import reference_quantize_ternary
from .teacher_models import HeavisideSTE


def butterfly_pairs(width: int, level: int) -> np.ndarray:
    """バタフライ配線: 出力iの入力ペア (i, i XOR 2^level)。

    Args:
        width: 層幅(2の冪)
        level: バタフライ段(0 ≤ level < log2(width))

    Returns:
        (width, 2) の入力indexペア
    """
    assert width & (width - 1) == 0, f'幅は2の冪のみ: {width}'
    stride = 1 << level
    assert stride < width, (level, width)
    idx = np.arange(width)
    return np.stack([idx, idx ^ stride], axis=1)


class ButterflyFactor(nn.Module):
    """1バタフライ因子: y_i = 1[a_q·x_u + b_q·x_v + θ > 0](ternary STE)。

    a/bはfloat master重みで、forwardごとに因子内2W本を束ねた
    absmean ternary量子化(per='tensor')をSTEで適用する。
    行あたり非ゼロ≤2はバタフライ配線が構造的に保証する。
    """

    def __init__(self, width: int, level: int) -> None:
        super().__init__()
        self.width = width
        self.level = level
        pairs = butterfly_pairs(width, level)
        self.register_buffer('u', torch.as_tensor(pairs[:, 0]))
        self.register_buffer('v', torch.as_tensor(pairs[:, 1]))
        # flat butterfly風: 恒等寄り(a≈1, b小)初期化で深い積の信号を保つ
        self.a = nn.Parameter(torch.ones(width)
                              + 0.1 * torch.randn(width))
        self.b = nn.Parameter(0.5 * torch.randn(width))
        self.theta = nn.Parameter(-0.5 * torch.ones(width))

    def quantized_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """STE ternary量子化後の (a_q, b_q)(forward値)。"""
        w = torch.cat([self.a, self.b])
        q, scale, _ = reference_quantize_ternary(w, per='tensor')
        w_q = w + (q * scale - w).detach()
        return w_q[:self.width], w_q[self.width:]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a_q, b_q = self.quantized_weights()
        pre = a_q * h[:, self.u] + b_q * h[:, self.v] + self.theta
        return HeavisideSTE.apply(pre)


class ButterflyMLP(nn.Module):
    """バタフライ因子スタック+GroupSum readoutの分類器。

    入力は幅W(2の冪)へゼロパディング。levels = n_blocks·log2(W) 段の
    因子を重ね、最終ビット列の先頭 usable 本(kの倍数)をGroupSumで
    クラススコアへ集約する。隠れはすべてfan-in=2の閾値ゲート。
    """

    def __init__(self, in_width: int, num_classes: int,
                 n_blocks: int = 1, tau: float = 1.0,
                 min_width: int = 2) -> None:
        super().__init__()
        from ..group_sum import GroupSum
        width = max(1 << math.ceil(math.log2(in_width)), min_width)
        self.in_width = in_width
        self.width = width
        self.num_classes = num_classes
        log2w = int(math.log2(width))
        self.factors = nn.ModuleList([
            ButterflyFactor(width, lv % log2w)
            for lv in range(n_blocks * log2w)
        ])
        self.usable = width - (width % num_classes)
        self.group_sum = GroupSum(num_classes, tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(x.shape[0], -1)
        if h.shape[1] < self.width:
            # 巡回複製パディング(ゼロ埋めは大半のワイヤを定数化し
            # 勾配が枯れる: XOR診断で0.75停滞→複製で8/8収束)
            idx = torch.arange(self.width, device=h.device) % h.shape[1]
            h = h[:, idx]
        for f in self.factors:
            h = f(h)
        return self.group_sum(h[:, :self.usable])


def transcribe_butterfly(model: ButterflyMLP):
    """ButterflyMLPをfan-in=2 lut_layer列のIRへロスレス転写する。

    各因子ニューロンの確定値 1[a_q·x_u + b_q·x_v + θ > 0] を
    threshold_truth_tables で厳密列挙し、n_inputs=2 のlut_layerに置く。
    gate_id_from_tt_n2 のヒストグラムをmetadataに記録する。

    Returns:
        IRGraph('x'→pad定数→因子lut_layer列→選択identity→group_sum→'y')
    """
    from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
    from ..ir.weights import gate_id_from_tt_n2
    from .binarize import _lut_layer_from_cols
    from .transcribe import threshold_truth_tables

    width = model.width
    nodes, edges = [], []
    prev_id, prev_width = 'x', model.in_width

    # 入力の巡回複製パディング(forwardと同一意味論のidentity LUT)
    if model.in_width < width:
        cols = [(0x2, [j % model.in_width]) for j in range(width)]
        pad_node = _lut_layer_from_cols('bf_pad', model.in_width, cols,
                                        {'kind': 'butterfly_pad'})
        nodes.append(pad_node)
        edges.append(IREdge(src='x', dst='bf_pad'))
        prev_id, prev_width = 'bf_pad', width

    gate_hist: dict[int, int] = {}
    with torch.no_grad():
        for li, f in enumerate(model.factors):
            a_q, b_q = f.quantized_weights()
            w2 = torch.stack([a_q, b_q], dim=1).double()   # (W, 2)
            tts = threshold_truth_tables(w2, f.theta.double())
            pairs = np.stack([f.u.numpy(), f.v.numpy()], axis=0)  # (2, W)
            for tt in tts:
                gid = gate_id_from_tt_n2(int(tt))
                gate_hist[gid] = gate_hist.get(gid, 0) + 1
            node_id = f'bf_f{li}'
            from ..ir.serialize import tt_to_hex
            nodes.append(IRNode(
                id=node_id,
                type='lut_layer',
                params={
                    'in_width': prev_width,
                    'out_width': width,
                    'n_inputs': 2,
                    'indices': pairs.astype(np.int64),
                    'truth_tables': tt_to_hex([int(t) for t in tts]),
                },
                train_info={'parameterization': 'softmax',
                            'selection': 'soft'},
            ))
            edges.append(IREdge(src=prev_id, dst=node_id))
            prev_id, prev_width = node_id, width

    # readout: 先頭usable本の選択identity + group_sum
    sel_cols = [(0x2, [j]) for j in range(model.usable)]
    sel = _lut_layer_from_cols('bf_sel', prev_width, sel_cols,
                               {'kind': 'butterfly_select'})
    gs = IRNode(
        id='bf_gs',
        type='group_sum',
        params={'in_width': model.usable, 'k': model.num_classes,
                'tau': 1.0},
    )
    nodes += [sel, gs]
    edges += [IREdge(src=prev_id, dst='bf_sel'),
              IREdge(src='bf_sel', dst='bf_gs'),
              IREdge(src='bf_gs', dst='y')]

    return IRGraph(
        metadata={
            'source': 'convert.transcribe_butterfly',
            'butterfly': {
                'width': width,
                'n_factors': len(model.factors),
                'gate_histogram': {str(k): v
                                   for k, v in sorted(gate_hist.items())},
                'n_luts_total': len(model.factors) * width
                + model.usable + (width if model.in_width < width else 0),
            },
        },
        inputs=[IRPort(id='x', width=model.in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=model.num_classes,
                        semantics='class_scores')],
        nodes=nodes,
        edges=edges,
    )
