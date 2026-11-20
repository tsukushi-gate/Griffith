"""学習済みLutLayer → 真理値表の万能離散化。

パラメータ化方式ごとの離散化規則を複製せず、
「eval()モードのレイヤ自身に全2^Nパターンを流して出力を読む」
方式を採る。全パラメータ化を分岐なしで正しくカバーし、
将来のパラメータ化追加にも自動追従する。

実現方法: レイヤをdeepcopyし、接続を恒等(indices[j,o]=j)・
in_dim=n_inputs に差し替えたプローブレイヤに全パターンを入力する。
レイヤ本来のeval離散化コードパス(layers/lut.py)がそのまま実行される。

rawパラメータ化はevalでも連続値(BinaryBrain意味論)のため、
ここで初めて >0.5 で二値化する(HW展開規則)。
"""

import copy

import numpy as np
import torch

from ..core.estimators import estimator_rho
from ..layers.lut import LutLayer


@torch.no_grad()
def truth_tables_from_composition(layer: LutLayer, post: list) -> np.ndarray:
    """LutLayer+後続の要素独立変換列の合成真理値表を抽出する。

    truth_tables_from_bb_block(export/lowering.py)のプローブfoldingを
    一般化したもの。レイヤと後続変換をeval()で合成し、全2^Nパターンを
    流して真理値表化する(合成の数式を複製しない)。

    前提: postの各変換が特徴ごとに独立(BatchNorm/Binarize/クランプ等)で
    あること。LayerNormのような特徴横断の変換は出力ビットoがレイヤの
    他の出力にも依存するため折り畳めない(適用禁止)。

    Args:
        layer: プローブ対象のLutLayer
        post: レイヤ出力へ順に適用するnn.Module列(空リスト可)

    Returns:
        (out_dim,) のobject配列(python int)。要素の bit k は
        butterfly基底順の入力パターンk(bit j=スロットjの値)に対する出力
    """
    n = layer.n_inputs
    out_dim = layer.out_dim

    probe = copy.deepcopy(layer).to('cpu')
    probe.eval()
    # 接続を恒等化: 各ニューロンのスロットjが入力jを見る
    probe.indices = torch.arange(n, dtype=torch.int64).unsqueeze(1) \
        .expand(n, out_dim).contiguous()
    probe.in_dim = n
    if getattr(probe, 'wiring', 'fixed') == 'learnable':
        # 学習配線は「どの入力を見るか」でありゲート機能(真理値表)とは独立。
        # プローブでは配線を恒等固定に落とさないとLearnableMappingが
        # n幅のプローブ入力へ元のin_dim添字でアクセスして壊れる(SM10)
        probe.wiring = 'fixed'
    if probe.parameterization == 'warp':
        probe.gumbel = False

    patterns = torch.tensor(
        [[float((k >> j) & 1) for j in range(n)] for k in range(2 ** n)]
    )
    outs = probe(patterns)          # (2^n, out_dim) — eval離散化パスを実行
    for m in post:
        pm = copy.deepcopy(m).to('cpu')
        pm.eval()
        outs = pm(outs)
    bits = (outs > 0.5).to(torch.int64)  # rawはここで初めて二値化

    tables = []
    for o in range(out_dim):
        t = 0
        for k in range(2 ** n):
            t |= int(bits[k, o].item()) << k
        tables.append(t)
    return np.array(tables, dtype=object)


@torch.no_grad()
def truth_tables_from_lut_layer(layer: LutLayer) -> np.ndarray:
    """LutLayerの全ニューロンの真理値表を抽出する。

    Returns:
        (out_dim,) のobject配列(python int)。要素の bit k は
        butterfly基底順の入力パターンk(bit j=スロットjの値)に対する出力
    """
    return truth_tables_from_composition(layer, [])


@torch.no_grad()
def truth_tables_from_dwn(layer) -> np.ndarray:
    """DWNLayerの全LUT(RAMノード)の真理値表を抽出する。

    DWNのeval意味論(layers/dwn.py):
        addr = Σ_j 1[gathered_j > 0] << j   (DWNFunction.forward: LSB=スロット0)
        y_u  = 1[luts[u, addr] > 0]         (BinarizeSTEFunction: 閾値0)
    つまり bit k = 1[luts[u, k] > 0]。アドレスのビットj=スロットjなので
    butterfly基底順(bit j=入力スロットjの値)ともそのまま一致する。

    Returns:
        (n_luts,) のobject配列(python int)
    """
    bits = (layer.luts.detach().cpu() > 0).to(torch.int64)  # (n_luts, 2^n)
    tables = []
    for u in range(bits.shape[0]):
        t = 0
        for k in range(bits.shape[1]):
            t |= int(bits[u, k].item()) << k
        tables.append(t)
    return np.array(tables, dtype=object)


@torch.no_grad()
def truth_tables_from_ltt(layer) -> np.ndarray:
    """LTTLayer(線形閾値ゲート)の真理値表を全 2^n パターンの列挙で抽出する。

    パターン k のビット j = 入力スロット j の値(butterfly 基底順 = DWN のアドレス規約と
    同じ)。bit k = 1[Σ_j w[u, j] · bit(k, j) + b[u] > 0](layers/ltt.py の eval と同式)。
    内部調査リポジトリ先行実験 `LTTLayer.truth_tables` と同じ列挙。

    Returns:
        (out_dim,) のobject配列(python int)
    """
    n = layer.n_inputs
    w = layer.w.detach().cpu()
    b = layer.b.detach().cpu()
    patterns = torch.tensor(
        [[(k >> j) & 1 for j in range(n)] for k in range(2 ** n)], dtype=w.dtype,
    )                                                           # (2^n, n)
    raw = patterns @ w.T + b                                    # (2^n, out_dim)
    hard = (raw > 0).T.to(torch.int64)                          # (out_dim, 2^n)
    tables = []
    for u in range(hard.shape[0]):
        t = 0
        for k in range(hard.shape[1]):
            t |= int(hard[u, k].item()) << k
        tables.append(t)
    return np.array(tables, dtype=object)


@torch.no_grad()
def truth_tables_from_conv(layer) -> np.ndarray:
    """LutConv2dの全内部ゲートの真理値表を抽出する。

    convのeval離散化規則(conv.py::_tree_forward_eval)はLutLayerと異なる
    (softmaxはone_hot(argmax)、N=2でも16ゲートモードなし)ため、
    プローブ再利用ではなくconv自身の規則を鏡写しにする。
    unrollされたIRとconv.eval()の等価性テスト(tests/test_ir_export.py)が
    規則の一致を恒久的に保証する。

    Returns:
        (out_channels, n_internal) のobject配列(python int)。
        ビット規約はbutterfly基底順(bit k = パターンk)
    """
    w = layer.weights  # (out_ch, n_internal, 2^N) or (out_ch, n_internal, N)
    n = layer.n_inputs
    n_entries = 2 ** n

    if layer.parameterization == 'iwp':
        # p>0.5のリテラル積 = 単一minterm
        p = (torch.sigmoid(w) > 0.5).to(torch.int64)  # (oc, ni, N)
        # minterm k = Σ_j p_j << j、tt = 1 << k
        k = (p << torch.arange(n)).sum(dim=-1)  # (oc, ni)
        tables = [[1 << int(k[oc, g]) for g in range(w.shape[1])]
                  for oc in range(w.shape[0])]
        return np.array(tables, dtype=object)

    if layer.parameterization in ('independent', 'independent_ste'):
        # 推定器 ρ(sigmoid / sin。conv.py::_tree_forward_eval の鏡写し)
        rho = estimator_rho(w, getattr(layer, 'estimator', 'sigmoid'))
        hard = (rho > 0.5).to(torch.int64)  # (oc, ni, 2^N)
    elif layer.parameterization == 'independent_dwn':
        # sigmoid を挟まない規約（conv.py::_tree_forward_eval の鏡写し）
        hard = (w > 0.0).to(torch.int64)
    elif layer.parameterization == 'softmax16g':
        # argmaxで16ゲートを選択 → gate_tableの4エントリ真理値表
        # (conv.py::_tree_forward_eval のsoftmax16g規則を鏡写し)
        gate_idx = w.argmax(dim=-1)  # (oc, ni)
        hard = layer.gate_table.to(torch.int64)[gate_idx]  # (oc, ni, 4)
    else:  # softmax: one_hot(argmax)
        hard = torch.nn.functional.one_hot(
            w.argmax(dim=-1), n_entries,
        ).to(torch.int64)

    tables = []
    for oc in range(w.shape[0]):
        row = []
        for g in range(w.shape[1]):
            t = 0
            for k in range(n_entries):
                t |= int(hard[oc, g, k].item()) << k
            row.append(t)
        tables.append(row)
    return np.array(tables, dtype=object)


@torch.no_grad()
def truth_tables_16gate_direct(layer: LutLayer) -> np.ndarray:
    """16ゲートモード(softmax N=2)のargmax直接読み(照合テスト用)。

    truth_tables_from_lut_layer と一致することをテストで担保する。
    """
    assert layer._use_16gate, '16ゲートモード専用'
    gate_idx = layer.weights.argmax(dim=-1)          # (out_dim,)
    table = layer.gate_table[gate_idx].to(torch.int64)  # (out_dim, 4)
    tables = []
    for o in range(layer.out_dim):
        t = 0
        for k in range(4):
            t |= int(table[o, k].item()) << k
        tables.append(t)
    return np.array(tables, dtype=object)
