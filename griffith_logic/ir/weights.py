"""真理値表 → LutLayer重みの復元(E2)。

IRの真理値表(butterfly基底順の2^Nビット整数)から、各パラメータ化方式の
weightsテンソルを合成する。合成後のレイヤをeval離散化パス
(export/discretize.truth_tables_from_lut_layer)に通すと元の真理値表が
ビット厳密に復元されることをtests/test_ir_build.pyで担保する。

パラメータ化ごとの表現可能性:
    softmax16g(softmax N=2) — 全16関数を表現可能
    independent / raw       — 任意の真理値表を表現可能
    softmax(N>2)            — popcount(tt) <= 1 のみ(単一minterm or FALSE)
    iwp                     — popcount(tt) == 1 のみ(単一minterm)
    warp                    — 任意の真理値表を表現可能(WH係数へ厳密変換)
"""

import numpy as np
import torch

from ..core.basis import _compute_wh_basis_butterfly
from ..core.connections import build_reverse_index
from .serialize import tt_from_hex

DEFAULT_LOGIT_M = 10.0  # 離散化が確実に安定するロジット振幅

# 任意の真理値表を直接編集できるパラメータ化(エディタのtt編集可否判定用)
TT_EDITABLE = {'softmax16g', 'independent', 'raw'}


def gate_id_from_tt_n2(tt: int) -> int:
    """N=2真理値表(bit k=butterfly基底順)を16ゲートIDへ変換する。

    ゲートID規約: 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数
    (core/basis.py の _build_gate_table_n2 の逆写像)。
    """
    b00 = tt & 1
    b10 = (tt >> 1) & 1
    b01 = (tt >> 2) & 1
    b11 = (tt >> 3) & 1
    return (b00 << 3) | (b01 << 2) | (b10 << 1) | b11


def _popcount(t: int) -> int:
    return bin(t).count('1')


def weights_from_truth_tables(
    parameterization: str,
    n_inputs: int,
    tts: list[int],
    *,
    logit_m: float = DEFAULT_LOGIT_M,
) -> torch.Tensor:
    """真理値表列からLutLayer.weightsを合成する。

    Args:
        parameterization: 'softmax16g' | 'softmax' | 'independent' |
            'raw' | 'iwp' | 'warp'('softmax'+N=2は16ゲートモード扱い)
        n_inputs: LUT入力数N
        tts: (out_dim,) の真理値表整数列(bit k=butterfly基底順)
        logit_m: 離散化マージンのロジット振幅

    Returns:
        (out_dim, パラメータ数) のfloat32テンソル

    Raises:
        ValueError: 指定パラメータ化で表現不能な真理値表
    """
    n_entries = 1 << n_inputs
    limit = 1 << n_entries
    tts = [int(t) for t in tts]
    out_dim = len(tts)
    for t in tts:
        if t < 0 or t >= limit:
            raise ValueError(f'真理値表が値域外: {t} (N={n_inputs})')

    is_16gate = parameterization == 'softmax16g' or (
        parameterization == 'softmax' and n_inputs == 2
    )

    if is_16gate:
        if n_inputs != 2:
            raise ValueError(f'softmax16gはN=2専用: N={n_inputs}')
        w = torch.full((out_dim, 16), -logit_m)
        for o, t in enumerate(tts):
            w[o, gate_id_from_tt_n2(t)] = logit_m
        return w

    if parameterization == 'softmax':
        # N>2 softmax: 離散化が「確率>0.5の単一エントリ」なので
        # 表現できるのは単一minterm(popcount=1)かFALSE(popcount=0)のみ
        w = torch.full((out_dim, n_entries), -logit_m)
        for o, t in enumerate(tts):
            pc = _popcount(t)
            if pc == 0:
                continue  # 全エントリ-M → softmax一様(1/2^N < 0.5) → FALSE
            if pc != 1:
                raise ValueError(
                    f'softmax N>2では表現不能な真理値表: {hex(t)} (popcount={pc})'
                )
            w[o, t.bit_length() - 1] = logit_m
        return w

    if parameterization == 'independent':
        w = torch.full((out_dim, n_entries), -logit_m)
        for o, t in enumerate(tts):
            for k in range(n_entries):
                if (t >> k) & 1:
                    w[o, k] = logit_m
        return w

    if parameterization == 'raw':
        # raw(BinaryBrain): 重み=[0,1]の値そのもの。HW展開規則(>0.5)に合わせ
        # ビットを1.0/0.0で直接書き込む
        w = torch.zeros(out_dim, n_entries)
        for o, t in enumerate(tts):
            for k in range(n_entries):
                if (t >> k) & 1:
                    w[o, k] = 1.0
        return w

    if parameterization == 'iwp':
        # IWP: 単一mintermのみ。k=そのmintermのビットパターンとして
        # スロットjの重み = +M(bit j=1) / -M(bit j=0)
        w = torch.empty(out_dim, n_inputs)
        for o, t in enumerate(tts):
            if _popcount(t) != 1:
                raise ValueError(
                    f'iwpでは表現不能な真理値表: {hex(t)} (popcount={_popcount(t)})'
                )
            k = t.bit_length() - 1
            for j in range(n_inputs):
                w[o, j] = logit_m if (k >> j) & 1 else -logit_m
        return w

    if parameterization == 'warp':
        # WARP: 目標符号 g[k] = 2*tt_k - 1 に対し θ = Hᵀ·g·M/2^N。
        # H[k,i] = (-1)^popcount(i&k) は±1直交行列なので logit(k) = M·g[k]。
        # ±1の積和はfp32で厳密(整数値が2^24未満)
        patterns = torch.tensor(
            [[float((k >> j) & 1) for j in range(n_inputs)] for k in range(n_entries)]
        )
        h = _compute_wh_basis_butterfly(
            patterns.unsqueeze(1), n_inputs
        ).squeeze(1)  # (2^N, 2^N): h[k, i]
        bits = torch.tensor(
            [[float((t >> k) & 1) for k in range(n_entries)] for t in tts]
        )  # (out_dim, 2^N)
        g = 2.0 * bits - 1.0
        return (g @ h) * (logit_m / n_entries)

    raise ValueError(f'未知のパラメータ化: {parameterization}')


def weights_from_truth_tables_fair_init(
    n_inputs: int,
    tts: list[int],
    *,
    transcribed_logit: float = 3.0,
    residual_bias: float = 5.0,
) -> torch.Tensor:
    """真理値表列を「残差初期化を保った非飽和」softmax16gロジットへ変換する(追加)。

    `weights_from_truth_tables(is_16gate=True)`(既定logit_m=10.0の飽和one-hot、
    以下「純粋転写init」)は2つの問題を持つ:
        (1) ±10.0の飽和ロジットはsoftmaxをほぼ完全なone-hotへ潰し、
            勾配がほぼ0になる(飽和saturationによる学習停止)
        (2) 転写先の1ゲートへ全質量を置くため、LutConv2d._residual_init が
            仕込む「ゲートA(パススルー)への事前バイアス」を完全に上書きする
            (深いtree convでの勾配減衰対策が消える)

    本関数は転写ゲート(target gate)のロジットへ**中庸**な正の値
    (`transcribed_logit`、既定はDEFAULT_LOGIT_M=10.0よりずっと小さい3.0)のみを
    加算し、他は0のまま(飽和なし)にする。さらにゲートA
    (真理値表[0,1,0,1]、`conv.py::_gate_a_index`と同じ探索則)のロジットへ
    `residual_bias`を**加算**する(転写ゲートがAならtranscribed_logitと
    合算、Aでなければ2ピークのsoftmaxになる)。

    これにより初期状態は「転写された関数へ強くバイアスされているが飽和は
    しておらず(gradが流れる)、かつ残差パススルー事前分布も残る」
    softmax混合になる。bit-exactではない(意図的、`weights_from_truth_tables`
    の飽和one-hot経路とは別物として使い分ける)。

    Args:
        n_inputs: LUT入力数N(softmax16gはN=2専用)
        tts: (out_dim,) の真理値表整数列(bit k=butterfly基底順)
        transcribed_logit: 転写ゲートのロジットに加える値(既定3.0、
            DEFAULT_LOGIT_M=10.0より小さい非飽和値)
        residual_bias: ゲートA(パススルー)のロジットに加える値
            (既定5.0、LutConv2d._residual_initのz3と同じ既定値)

    Returns:
        (out_dim, 16) のfloat32テンソル(softmax16g logits)

    Raises:
        ValueError: n_inputs != 2
    """
    if n_inputs != 2:
        raise ValueError(f'fair_initはsoftmax16g(N=2)専用: N={n_inputs}')
    n_entries = 1 << n_inputs
    limit = 1 << n_entries
    tts = [int(t) for t in tts]
    out_dim = len(tts)
    for t in tts:
        if t < 0 or t >= limit:
            raise ValueError(f'真理値表が値域外: {t} (N={n_inputs})')

    # ゲートA([0,1,0,1])のゲートIDをgate_id_from_tt_n2で動的に特定する
    # (conv.py::_gate_a_index と同じ規約: tt=0b1010=10 → [0,1,0,1])
    a_gate_id = gate_id_from_tt_n2(0b1010)

    w = torch.zeros((out_dim, 16))
    for o, t in enumerate(tts):
        target_gate_id = gate_id_from_tt_n2(t)
        w[o, target_gate_id] += transcribed_logit
        w[o, a_gate_id] += residual_bias
    return w


def conv_weights_from_truth_tables(
    parameterization: str,
    n_inputs: int,
    tts: list[int],
    *,
    logit_m: float = DEFAULT_LOGIT_M,
) -> torch.Tensor:
    """真理値表列からLutConv2dのゲート重みを合成する。

    LutConv2dのeval離散化規則(conv.py::_tree_forward_eval)は
    LutLayerと異なり、softmaxはN=2でも16ゲートモードなしの
    one_hot(argmax)。そのためsoftmaxで表現可能なのは単一minterm
    (popcount==1)のみ(FALSEも不可: argmaxは必ず1エントリを選ぶ)。

    'softmax16g'(N=2専用)はLutLayerの16ゲートモードと同じ規則
    (conv.py::_tree_forward_eval のsoftmax16g分岐、
    export/discretize.py::truth_tables_from_conv の対応分岐と鏡写し)で、
    任意のN=2真理値表(16通り全て)を表現できる。`weights_from_truth_tables`
    の is_16gate 分岐(gate_id_from_tt_n2 経由)をそのまま再利用する。

    Args:
        parameterization: 'softmax' | 'independent' | 'iwp' | 'softmax16g'
        n_inputs: ゲート入力数N
        tts: (ゲート数,) の真理値表整数列(bit k=butterfly基底順)
        logit_m: 離散化マージンのロジット振幅

    Returns:
        (ゲート数, パラメータ数) のfloat32テンソル

    Raises:
        ValueError: 指定パラメータ化で表現不能な真理値表
    """
    if parameterization in ('independent', 'iwp', 'softmax16g'):
        return weights_from_truth_tables(
            parameterization, n_inputs, tts, logit_m=logit_m,
        )
    if parameterization != 'softmax':
        raise ValueError(f'convで未知のパラメータ化: {parameterization}')

    n_entries = 1 << n_inputs
    limit = 1 << n_entries
    tts = [int(t) for t in tts]
    w = torch.full((len(tts), n_entries), -logit_m)
    for o, t in enumerate(tts):
        if t < 0 or t >= limit:
            raise ValueError(f'真理値表が値域外: {t} (N={n_inputs})')
        if _popcount(t) != 1:
            raise ValueError(
                f'conv softmax(one_hot argmax)では表現不能な真理値表: '
                f'{hex(t)} (popcount={_popcount(t)})'
            )
        w[o, t.bit_length() - 1] = logit_m
    return w


@torch.no_grad()
def restore_lut_conv2d(conv, indices_flat, truth_tables_hex) -> None:
    """LutConv2dへIRのindices_flat/truth_tablesを書き戻す。

    Args:
        conv: 対象LutConv2d(out_channels/tree_depth/n_inputsはIRと一致していること)
        indices_flat: (out_channels, n_leaves) の受容野内フラット接続テーブル
        truth_tables_hex: (out_channels*n_internal,) のhex文字列列(または整数列)。
            oc優先(oc-major)で木のレベル順=LutConv2d.weightsのdim1順
    """
    idx = torch.as_tensor(np.asarray(indices_flat, dtype=np.int64))
    if tuple(idx.shape) != (conv.out_channels, conv.n_leaves):
        raise ValueError(
            f'indices_flat形状 {tuple(idx.shape)} != '
            f'({conv.out_channels}, {conv.n_leaves})'
        )
    receptive = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    if idx.numel() > 0 and (idx.min() < 0 or idx.max() >= receptive):
        raise ValueError(f'indices_flatが受容野({receptive})を超えている')

    tts = tt_from_hex(truth_tables_hex)
    n_gates = conv.out_channels * conv.n_internal
    if len(tts) != n_gates:
        raise ValueError(
            f'truth_tables数({len(tts)}) != '
            f'out_channels*n_internal({n_gates})'
        )

    conv.indices_flat.copy_(idx.to(conv.indices_flat.device))
    w = conv_weights_from_truth_tables(conv.parameterization, conv.n_inputs, tts)
    w = w.reshape(conv.out_channels, conv.n_internal, -1)
    conv.weights.copy_(w.to(conv.weights.device, conv.weights.dtype))


@torch.no_grad()
def restore_lut_conv2d_fair_init(
    conv, indices_flat, truth_tables_hex, *,
    transcribed_logit: float = 3.0, residual_bias: float = 5.0,
) -> None:
    """LutConv2d(parameterization='softmax16g')へ、残差初期化を保った
    非飽和の転写initを書き戻す(追加、`restore_lut_conv2d`の飽和
    one-hot版とは別経路)。

    `restore_lut_conv2d`はビット厳密(logit_m=10.0飽和one-hot)だが
    (1)勾配がほぼ消える(2)LutConv2d.residual_initのゲートAバイアスを
    完全に上書きする、という2つの副作用を持つ。本関数は
    `weights_from_truth_tables_fair_init`で「転写ゲートへの中庸バイアス+
    ゲートAへの残差バイアスの加算」ロジットを合成し書き戻す。
    意図的にbit-exactではない(呼び出し側で`assert_conv_rebuild_bit_exact`
    のような厳密性assertを課さないこと)。

    Args:
        conv: 対象LutConv2d(parameterization='softmax16g'必須)
        indices_flat: (out_channels, n_leaves) の受容野内フラット接続テーブル
        truth_tables_hex: (out_channels*n_internal,) のhex文字列列(または整数列)
        transcribed_logit: 転写ゲートロジットへの加算値(既定3.0)
        residual_bias: ゲートAロジットへの加算値(既定5.0、_residual_initのz3既定と同じ)

    Raises:
        ValueError: conv.parameterization != 'softmax16g'、または形状不整合
    """
    if conv.parameterization != 'softmax16g':
        raise ValueError(
            f"fair_initはparameterization='softmax16g'専用: {conv.parameterization}"
        )
    idx = torch.as_tensor(np.asarray(indices_flat, dtype=np.int64))
    if tuple(idx.shape) != (conv.out_channels, conv.n_leaves):
        raise ValueError(
            f'indices_flat形状 {tuple(idx.shape)} != '
            f'({conv.out_channels}, {conv.n_leaves})'
        )
    receptive = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    if idx.numel() > 0 and (idx.min() < 0 or idx.max() >= receptive):
        raise ValueError(f'indices_flatが受容野({receptive})を超えている')

    tts = tt_from_hex(truth_tables_hex)
    n_gates = conv.out_channels * conv.n_internal
    if len(tts) != n_gates:
        raise ValueError(
            f'truth_tables数({len(tts)}) != '
            f'out_channels*n_internal({n_gates})'
        )

    conv.indices_flat.copy_(idx.to(conv.indices_flat.device))
    w = weights_from_truth_tables_fair_init(
        conv.n_inputs, tts,
        transcribed_logit=transcribed_logit, residual_bias=residual_bias,
    )
    w = w.reshape(conv.out_channels, conv.n_internal, -1)
    conv.weights.copy_(w.to(conv.weights.device, conv.weights.dtype))


@torch.no_grad()
def restore_dwn_layer(layer, indices, truth_tables_hex) -> None:
    """DWNLayerへIRのindices/truth_tablesを書き戻す。

    **非可逆な復元**: lutsは真理値表ビットの±1符号化(bit=1→+1.0,
    bit=0→−1.0)で合成する。eval意味論(「>0」の二値化)では元レイヤと
    ビット厳密に等価だが、元のluts実数値やLearnableMappingの重み分布は
    IRに保存されないため復元されない(訓練を再開すると元とは別物になる)。

    Args:
        layer: 対象DWNLayer(in_bits/n_luts/n_inputsはIRと一致していること)
        indices: (n_inputs, n_luts) の接続テーブル(list/ndarray/Tensor)
        truth_tables_hex: (n_luts,) のhex文字列列(または整数列)
    """
    idx = torch.as_tensor(np.asarray(indices, dtype=np.int64))
    if tuple(idx.shape) != (layer.n_inputs, layer.n_luts):
        raise ValueError(
            f'indices形状 {tuple(idx.shape)} != '
            f'({layer.n_inputs}, {layer.n_luts})'
        )
    if idx.numel() > 0 and (idx.min() < 0 or idx.max() >= layer.in_bits):
        raise ValueError('indicesが入力幅を超えている')

    if layer.mapping_mode == 'learnable':
        # 列q=u*n_inputs+j のargmaxが indices[j,u] を選ぶone-hot重みを合成
        # (LearnableMappingFunction.forwardのハード選択と等価)
        w = torch.zeros_like(layer.learnable_mapping.weights)
        flat = idx.T.reshape(-1)  # flat[u*n_inputs+j] = indices[j,u]
        w[flat, torch.arange(flat.numel())] = 1.0
        layer.learnable_mapping.weights.copy_(w)
    else:
        layer.indices.copy_(idx.to(layer.indices.device))

    tts = tt_from_hex(truth_tables_hex)
    if len(tts) != layer.n_luts:
        raise ValueError(f'truth_tables数({len(tts)}) != n_luts({layer.n_luts})')
    limit = 1 << layer.n_entries
    if any(t < 0 or t >= limit for t in tts):
        raise ValueError(f'真理値表が値域外 (N={layer.n_inputs})')
    bits = torch.tensor(
        [[float((t >> k) & 1) for k in range(layer.n_entries)] for t in tts]
    )
    layer.luts.copy_(
        (bits * 2.0 - 1.0).to(layer.luts.device, layer.luts.dtype)
    )


@torch.no_grad()
def restore_lut_layer(layer, indices, truth_tables_hex) -> None:
    """LutLayerへIRのindices/truth_tablesを書き戻す。

    indicesの差し替え時は逆引きバッファ(rev_starts/rev_flat_y/rev_flat_slot)
    をbuild_reverse_indexで必ず再計算する(漏れるとCUDA backwardが壊れる)。

    Args:
        layer: 対象LutLayer(in/out/n_inputsはIRと一致していること)
        indices: (n_inputs, out_dim) の接続テーブル(list/ndarray/Tensor)
        truth_tables_hex: (out_dim,) のhex文字列列(または整数列)
    """
    idx = torch.as_tensor(np.asarray(indices, dtype=np.int64))
    if tuple(idx.shape) != (layer.n_inputs, layer.out_dim):
        raise ValueError(
            f'indices形状 {tuple(idx.shape)} != ({layer.n_inputs}, {layer.out_dim})'
        )
    if idx.numel() > 0 and (idx.min() < 0 or idx.max() >= layer.in_dim):
        raise ValueError('indicesが入力幅を超えている')

    layer.indices.copy_(idx.to(layer.indices.device))
    rev_starts, rev_flat_y, rev_flat_slot = build_reverse_index(
        layer.indices, layer.in_dim
    )
    layer.rev_starts = rev_starts
    layer.rev_flat_y = rev_flat_y
    layer.rev_flat_slot = rev_flat_slot

    param = 'softmax16g' if getattr(layer, '_use_16gate', False) \
        else layer.parameterization
    tts = tt_from_hex(truth_tables_hex)
    if len(tts) != layer.out_dim:
        raise ValueError(f'truth_tables数({len(tts)}) != out_dim({layer.out_dim})')
    w = weights_from_truth_tables(param, layer.n_inputs, tts)
    layer.weights.copy_(w.to(layer.weights.device, layer.weights.dtype))
