"""学習後二値化→XNOR-popcount閾値の厳密ゲートlowering。

事前学習MLPの重みを±1へ二値化(PTB: Post-Training Binarization)した
bipolar BNNを、近似なしで論理ゲートIRへlowerする。精度損失はPTB段階
(実数→±1)のみに限定され、lowering自体はビット厳密 — これが本モジュールの
価値であり、テスト(tests/l1_math/test_binarize_lowering.py)で保証する。

変換の意味論(このモジュールが厳密一致を保証する参照):
    二値化: q = sign(W)、スケール α = mean|W|(XNOR-Net)
    隠れ層: y_i = 1[Σ_j q_ij·(2x_j-1) + b'_i > 0]   (x∈{0,1})
            バイアスはα単位で整数化: b'_i = round(b_i/α)

    XNORカウント閾値への帰着(厳密):
        m_ij = XNOR(x_j, [q_ij>0])  (x_jがq_ijの符号に一致するとき1)
        q_ij·(2x_j-1) = +1 (m_ij=1) / -1 (m_ij=0)
        ⇒ Σ_j q_ij(2x_j-1) = P_i - (n - P_i) = 2·P_i - n
           (P_i = popcount(m_i) = XNORカウント、n = in_width)
        発火条件: 2·P_i - n + b'_i > 0 ⟺ P_i > (n - b'_i)/2
                ⟺ P_i ≥ floor((n - b'_i)/2) + 1 =: θ_i
        ゆえに y_i = 1[P_i ≥ θ_i]。

    XNORビット生成: 縮退LUT層(transcribe.pyの流儀)。
        q_ij>0 なら m_ij = x_j          (identity, tt=0x8)
        q_ij<0 なら m_ij = NOT x_j       (not, tt=0x1)
    カウント比較: popcount_threshold_nodes(加算木+閾値比較)。
    最終層: transcribe._final_layer_nodes(argmax厳密保存構成)を再利用。
        ±1×α の実数重みを渡すと sign 量子化で q と一致するため厳密。

制約(v1):
    - 対象はnn.Linear列のMLP。Conv・多ビット活性は未対応
    - 回路規模(ニューロン毎に独立popcount木)はv1では最適化しない
      (metadataに総lut数・段数を記録するのみ)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex
from .transcribe import TT_IDENTITY, TT_NOT, TT_ZERO, _final_layer_nodes

# 全加算器LUT(n_inputs=3、butterfly順 addr bit j = スロットj)。
# sum = a^b^c(パリティ)、carry = majority(a,b,c)。
# addr=(a,b,c): a=bit0, b=bit1, c=bit2。popcount(addr)で判定する。
_FA_SUM_TT = sum(1 << a for a in range(8) if bin(a).count('1') % 2 == 1)   # 0b10010110
_FA_CARRY_TT = sum(1 << a for a in range(8) if bin(a).count('1') >= 2)     # 0b11101000
# 半加算器LUT(n_inputs=2)。sum = a^b、carry = a&b。
_HA_SUM_TT = 0b0110   # a^b: addr1(a=1),addr2(b=1)
_HA_CARRY_TT = 0b1000  # a&b: addr3

# LutLayer(layers/lut.py)が受け付けるfan-in。build_torch互換のため
# 全lut_layerノードはこの値のみを使う(回復学習経路の前提)
_ALLOWED_N_INPUTS = (2, 4, 6)


def _pad_n_inputs(k: int) -> int:
    """実fan-in k をLutLayer対応値(2/4/6)へ切り上げる。"""
    for n in _ALLOWED_N_INPUTS:
        if n >= k:
            return n
    raise ValueError(f'fan-in {k} は6を超える(LutLayer非対応)')


def _extend_tt(tt: int, k: int, n_inputs: int) -> int:
    """k個の実スロット(0..k-1)で定義された真理値表を、n_inputsスロットへ
    拡張する(出力はスロット k..n_inputs-1 に非依存)。

    未使用スロットは有効indexへ複製配線するが、出力がそれらに依存しないよう
    tt'[addr] = tt[addr & ((1<<k)-1)] とする。これにより異なるfan-inの
    ゲート(FA=3/HA=2/passthrough=1)を同一lut_layer(n_inputs=切り上げ値)へ
    混載してもビット厳密になる。
    """
    if k >= n_inputs:
        return tt
    mask = (1 << k) - 1
    out = 0
    for addr in range(1 << n_inputs):
        if (tt >> (addr & mask)) & 1:
            out |= 1 << addr
    return out


def _lut_layer_from_cols(
    node_id: str,
    in_width: int,
    cols: list[tuple[int, list[int]]],
    train_info: dict,
) -> IRNode:
    """(tt, srcスロット列)の列リストからlut_layerノードを構築する。

    n_inputsは実fan-inの最大値を_pad_n_inputsで2/4/6へ切り上げる
    (build_torch→LutLayerのassert対応)。各列のttは_extend_ttで
    上位スロット非依存に拡張し、未使用スロットはsrcs[0]へ複製配線する。

    N>2ではparameterization='independent'をtrain_infoに指定する:
    既定のsoftmaxはN>2で単一minterm真理値表しか表現できず、
    全加算器(パリティ)等がbuild_torch→restore_lut_layerで復元不能なため
    (BinaryBrain方式のエントリ独立ロジットは任意真理値表を厳密に表現できる)。

    Args:
        cols: [(tt, [入力列index, ...]), ...]。len(srcs)=その列の実fan-in
    """
    n_inputs = _pad_n_inputs(max(len(s) for _, s in cols))
    out_width = len(cols)
    indices = np.zeros((n_inputs, out_width), dtype=np.int64)
    tts: list[int] = []
    for col, (tt, srcs) in enumerate(cols):
        k = len(srcs)
        for j in range(n_inputs):
            indices[j, col] = srcs[j] if j < k else srcs[0]
        tts.append(_extend_tt(tt, k, n_inputs))
    if n_inputs > 2:
        train_info = {**train_info, 'parameterization': 'independent'}
    return IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': indices,
            'truth_tables': tt_to_hex(tts),
        },
        train_info=train_info,
    )


def binarize_weights(
    weight: torch.Tensor, method: str = 'sign'
) -> tuple[torch.Tensor, float]:
    """(out, in)実数重み→±1のint64行列とスケールα=mean|W|(XNOR-Net)。

    method='sign'のみv1。将来の拡張余地:
        'billm_like' — BiLLM(2024)風の顕著重み保持+残差2値化。
        salientマスクとブロック別スケールを返す拡張版として追加予定。

    Args:
        weight: (out, in) の実数重み
        method: 二値化方式('sign'のみ)

    Returns:
        (q, alpha):
            q     (out, in) int64 の±1行列(sign(0)=0を+1へ寄せる)
            alpha float スケール mean|W|(XNOR-Netの最適スケール)

    Raises:
        ValueError: 未対応のmethod
    """
    if method != 'sign':
        raise ValueError(f"methodは'sign'のみ(v1): {method}")
    # sign(0)=0 を +1 に寄せる(境界の実数重み0は正符号扱い)
    q = torch.where(weight >= 0, 1, -1).to(torch.int64)
    alpha = float(weight.abs().mean().clamp_min(1e-12))
    return q, alpha


def quantize_ternary(
    weight: torch.Tensor, per: str = 'channel', eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(out, in)実数重み→BitNet b1.58式 absmean ternary量子化。

    量子化(core/reference.py::reference_quantize_ternary と厳密同一):
        scale = mean(|W|)                       (absmean, per単位で平均)
        q     = clip(round(W / scale), -1, +1)  (∈ {-1, 0, +1})
        mask  = (q != 0)                         (残存接続マスク)
    実効重み(dequant)は q·scale。事前学習/量子化済みモデルの重みを
    {-1,0,+1}×scale へ正規化し、0マスク(=枝刈り結果)を取り出す入口。

    binarize_weights(1bit sign, 0マスクなし=密)との対比が本実験の中心
    対照(0マスクの有無が転写崖を回避するかを分離する)。

    Args:
        weight: (out, in) の実数重み
        per: 'channel'(各出力行ごとにscale) | 'tensor'(全体で1つ)
        eps: scale の下限クランプ(全ゼロ行のNaN回避)

    Returns:
        (q, scale, mask):
            q     (out, in) int64 の {-1, 0, +1} 行列
            scale per='channel'なら(out, 1)、'tensor'なら0次元 tensor
            mask  (out, in) bool の残存接続マスク

    Raises:
        ValueError: 未対応の per
    """
    if per == 'channel':
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # (out, 1)
    elif per == 'tensor':
        scale = weight.abs().mean().clamp_min(eps)                     # scalar
    else:
        raise ValueError(f"perは'channel'|'tensor'のみ: {per}")
    q = torch.clamp(torch.round(weight / scale), -1.0, 1.0).to(torch.int64)
    mask = q != 0
    return q, scale, mask


def extract_ternary_mask(
    model: nn.Module, per: str = 'channel',
) -> dict[str, torch.Tensor]:
    """モデルの各Linear/Conv2d重みから ternary 0マスクを復元する。

    公開 ternary チェックポイントは {-1,0,+1}×scale で float 保存されるため、
    quantize_ternary で再量子化した mask が残存接続(枝刈り済み構造)を表す。
    Conv2d は (out, C·kh·kw) へ平坦化して量子化し、元形状へ戻す。

    Args:
        model: 対象モデル(named_modules を走査)
        per: quantize_ternary の per

    Returns:
        {module_name: (out, ...) bool マスク}(Linear/Conv2d層のみ)
    """
    masks: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Conv2d):
            w = module.weight.detach()
            w2 = w.reshape(w.shape[0], -1)  # conv → (out, C·kh·kw)、linearは不変
            _, _, mask = quantize_ternary(w2, per=per)
            masks[name] = mask.reshape(w.shape)
    return masks


def popcount_threshold_nodes(
    width: int,
    thresholds: list[int],
    node_id_prefix: str,
    *,
    in_width: int,
    input_cols: np.ndarray,
) -> tuple[list[IRNode], list[IREdge], str, int]:
    """幅widthのビット列に対し y_i = 1[popcount(x) >= thresholds_i] を計算する
    lut_layerノード列(carry-save加算木+閾値比較)を構築する。

    設計(ビット厳密):
        1. 初段: input_cols[i] の width 本のビットを plane 0 のワイヤとみなす。
        2. carry-save reduction: 各段で plane 内のワイヤを全加算器
           (3→sum+carry)・半加算器(2→sum+carry)で圧縮する。各段は
           1つの lut_layer。全 plane が1本になるまで繰り返し、popcount の
           2進表現 b_0..b_{L-1}(L=ceil(log2(width+1)))を得る。
        3. 比較段: 各出力 i について C(b) >= θ_i を、上位ビットから
           (gt, eq) 状態を伝播する逐次比較器で厳密判定する(段数=L)。

    全lut_layerノードの n_inputs は LutLayer規約(2/4/6)へ切り上げる
    (_lut_layer_from_cols。build_torch互換、回復学習経路の前提)。
    実fan-in 3 のゲート(全加算器・比較器)は n_inputs=4 となり、未使用
    スロットは複製配線+真理値表の非依存拡張(_extend_tt)で厳密性を保つ。

    段数は O(log width)。LUT数の目安: 加算部 ≈ width(各ビットは高々1回
    加算器を通る、キャリー伝播込みで O(width))、比較部は出力ごとに
    L 段 × ブロック幅(2+L)。

    Args:
        width: カウント対象のビット幅(=1本の入力ビット数)
        thresholds: 出力ごとの閾値 θ_i(len = 出力数)
        node_id_prefix: 生成ノードidの接頭辞
        in_width: これらのノードの入力(prev_id)幅
        input_cols: (n_out, width) — 出力iのwidth本の入力列index
            (prev_idの出力列を指す)。全出力で同一なら複製して渡す

    Returns:
        (nodes, edges, last_node_id, out_width):
            out_width = 出力数(= len(thresholds))
    """
    n_out = len(thresholds)
    assert input_cols.shape == (n_out, width), (
        f'input_cols形状 {input_cols.shape} != ({n_out}, {width})'
    )
    # popcountは[0, width]。θ≤0は常時真、θ>widthは常時偽なので
    # [0, width+1]へクランプする(意味は保存: C>=0は常真、C>=width+1は常偽)。
    thresholds = [max(0, min(int(t), width + 1)) for t in thresholds]

    nodes: list[IRNode] = []
    edges: list[IREdge] = []

    # --- 初段: 各出力の width 本のビットを plane 0 へ配置する恒等コピー ---
    # すべての出力の全ビットを1つのlut_layerに並べる(出力ごとにブロック化)。
    # ワイヤ名 'g{i}p0w{w}' → このノードの出力列index。
    total0 = n_out * width
    node0 = _lut_layer_from_cols(
        f'{node_id_prefix}_p0',
        in_width,
        [(0x2, [int(input_cols[i, w])])  # identity(実fan-in 1)
         for i in range(n_out) for w in range(width)],
        {'kind': 'popcount_init'},
    )
    nodes.append(node0)

    # 各出力を独立に reduction する。ワイヤは (out i, plane p) 単位。
    # node0の各出力列index(グローバル)を各出力のplane0ワイヤに割り当て
    planes_per_out: list[dict[int, list[str]]] = []
    wire_slot: dict[str, int] = {}
    for i in range(n_out):
        planes: dict[int, list[str]] = {0: []}
        for w in range(width):
            wname = f'g{i}p0w{w}'
            wire_slot[wname] = i * width + w
            planes[0].append(wname)
        planes_per_out.append(planes)

    # reduction を全出力並列で進める。各段で全出力のワイヤを1ノードに集約する。
    prev_id = node0.id
    prev_out_width = total0
    stage = 0
    # 各出力が単一planeあたり1本になるまで(全体で最大plane本数)続ける
    while any(any(len(ws) > 1 for ws in po.values()) for po in planes_per_out):
        node_id = f'{node_id_prefix}_r{stage}'
        # (plane, tt, srcs, out_i)。emitはローカル関数を使わず直接appendする
        out_cols: list[tuple[int, int, list[str], int]] = []
        new_planes_per_out: list[dict[int, list[str]]] = [
            {} for _ in range(n_out)
        ]

        def _emit(out_i, plane, tt, srcs, *, cols=out_cols,
                  npo=new_planes_per_out, nid=node_id):
            col = len(cols)
            cols.append((plane, tt, srcs, out_i))
            npo[out_i].setdefault(plane, []).append(f'{nid}:i{out_i}c{col}')

        for i in range(n_out):
            for p in sorted(planes_per_out[i]):
                bucket = list(planes_per_out[i][p])
                while len(bucket) >= 3:
                    a, b, c = bucket.pop(), bucket.pop(), bucket.pop()
                    _emit(i, p, _FA_SUM_TT, [a, b, c])
                    _emit(i, p + 1, _FA_CARRY_TT, [a, b, c])
                if len(bucket) == 2:
                    a, b = bucket.pop(), bucket.pop()
                    _emit(i, p, _HA_SUM_TT, [a, b])
                    _emit(i, p + 1, _HA_CARRY_TT, [a, b])
                elif len(bucket) == 1:
                    _emit(i, p, 0x2, [bucket.pop()])

        nodes.append(_lut_layer_from_cols(
            node_id,
            prev_out_width,
            [(tt, [wire_slot[s] for s in srcs])
             for _plane, tt, srcs, _oi in out_cols],
            {'kind': 'popcount_csa', 'stage': stage},
        ))
        edges.append(IREdge(src=prev_id, dst=node_id))
        out_width = len(out_cols)

        # slot更新
        wire_slot = {
            f'{node_id}:i{oi}c{col}': col
            for col, (_p, _t, _s, oi) in enumerate(out_cols)
        }
        planes_per_out = new_planes_per_out
        prev_id = node_id
        prev_out_width = out_width
        stage += 1

    # ここで各出力は plane ごとに高々1本 = popcountの2進表現。
    # L = 最大plane+1。各出力の各planeの単一ワイヤをカウントビットとする。
    max_plane = 0
    for po in planes_per_out:
        for p in po:
            if po[p]:
                max_plane = max(max_plane, p)
    # 閾値(最大 width+1)を表現できるビット数も確保する(θの上位ビット欠落防止)
    n_count_bits = max(max_plane + 1, max(t.bit_length() for t in thresholds), 1)

    # --- 比較段: C(b) >= θ_i を逐次比較器で厳密判定 ---
    # Lビットの2進数 C と定数 θ の比較 1[C>=θ] を、上位ビットから
    # (gt, eq) 状態を伝播する逐次比較器で組む(段数=L、可読性重視)。
    #   gt: 上位ビットで C>θ が確定 / eq: 上位ビットが全一致
    #   各ゲートは実fan-in 3(LutLayer規約 2/4/6 へは builder が切り上げ)。
    # 詳細は _threshold_compare_nodes のdocstring参照。

    cmp_nodes, cmp_edges, last_id, out_w = _threshold_compare_nodes(
        n_out=n_out,
        n_count_bits=n_count_bits,
        thresholds=thresholds,
        planes_per_out=planes_per_out,
        wire_slot=wire_slot,
        prev_id=prev_id,
        prev_out_width=prev_out_width,
        node_id_prefix=node_id_prefix,
    )
    nodes += cmp_nodes
    edges += cmp_edges
    return nodes, edges, last_id, out_w


def _threshold_compare_nodes(
    *,
    n_out: int,
    n_count_bits: int,
    thresholds: list[int],
    planes_per_out: list[dict[int, list[str]]],
    wire_slot: dict[str, int],
    prev_id: str,
    prev_out_width: int,
    node_id_prefix: str,
) -> tuple[list[IRNode], list[IREdge], str, int]:
    """カウントの2進表現 b_0..b_{L-1} と定数閾値 θ_i を比較して
    1[C >= θ_i] を出力するLUT列を構築する(厳密)。

    上位ビットから (gt, eq) 状態を伝播する逐次比較器。状態は2ビットを
    1つのLUT出力列で表さず、gt/eqを別々の列として持ち回り、各段で
    実fan-in 3 のLUT (gt_prev, eq_prev, b_k) → gt_new / eq_new を評価する
    (LutLayer規約により n_inputs=4 へ切り上げ)。最終段で ge = gt or eq。

    C >= θ は「上位から見て最初に異なるビットでCが1」または「全ビット一致」。
    gt_new = gt_prev or (eq_prev and b_k and not t_k)
    eq_new = eq_prev and (b_k == t_k)
    初期状態 gt=0, eq=1。ge = gt_final or eq_final。
    """
    nodes: list[IRNode] = []
    edges: list[IREdge] = []

    # b_k の入力列index(prev_idの出力列)。存在しないビットは定数0。
    # 定数0が要るので、まず「gt=0, eq=1」の初期状態ノードと、必要なら定数0
    # ビットを供給するノードを作る。ここでは各出力のカウントビットを
    # まとめて参照するため、prev_idの列index表を作る。
    bit_col: list[list[int | None]] = []  # bit_col[i][k] = 列index or None
    for i in range(n_out):
        cols: list[int | None] = []
        for k in range(n_count_bits):
            ws = planes_per_out[i].get(k, [])
            if ws:
                cols.append(wire_slot[ws[0]])
            else:
                cols.append(None)  # 定数0
        bit_col.append(cols)

    # 状態初期化ノード(lut_layerは単一入力なので、各段のノードは1つの入力
    # ノードしか参照できない)。そこで比較状態(gt, eq)と全カウントビット
    # b_0..b_{L-1} を1ブロックに載せ、段ごとにそのブロックを更新して持ち回る。
    # ブロック = [gt, eq, b_0, ..., b_{L-1}](各出力ぶん連結)。
    # 定数生成の1入力LUT: 定数0=tt 0x0、定数1=tt 0x3、identity=tt 0x2。
    init_id = f'{node_id_prefix}_cmp_init'
    block = 2 + n_count_bits  # gt, eq, bits
    init_out = n_out * block
    init_cols: list[tuple[int, list[int]]] = []
    for i in range(n_out):
        init_cols.append((0x0, [0]))  # gt=0(定数0)
        init_cols.append((0x3, [0]))  # eq=1(定数1)
        for k in range(n_count_bits):
            c = bit_col[i][k]
            if c is None:
                init_cols.append((0x0, [0]))  # 未生成planeは定数0
            else:
                init_cols.append((0x2, [c]))  # identity(実fan-in 1)
    nodes.append(_lut_layer_from_cols(
        init_id, prev_out_width, init_cols, {'kind': 'cmp_init'},
    ))
    edges.append(IREdge(src=prev_id, dst=init_id))

    cur_id = init_id
    cur_width = init_out
    # 各出力のブロック内 slot: gt=0, eq=1, b_k = 2+k
    # 上位ビットから比較する。t_k = θのビットk。
    for step, k in enumerate(reversed(range(n_count_bits))):
        node_id = f'{node_id_prefix}_cmp{step}'
        out_block = 2 + n_count_bits  # 状態を保ちつつビットも持ち越す
        step_cols: list[tuple[int, list[int]]] = []
        for i in range(n_out):
            t_k = (thresholds[i] >> k) & 1
            gt_slot = i * block + 0
            eq_slot = i * block + 1
            bk_slot = i * block + 2 + k
            # gt_new = gt or (eq and b_k and not t_k)
            # 入力スロット: 0=gt, 1=eq, 2=b_k(実fan-in 3 → builderが4へ拡張)
            step_cols.append((_gt_new_tt(t_k), [gt_slot, eq_slot, bk_slot]))
            # eq_new = eq and (b_k == t_k)
            step_cols.append((_eq_new_tt(t_k), [gt_slot, eq_slot, bk_slot]))
            # ビットの持ち越し(identity)。以降の段が下位ビットを参照する。
            for kk in range(n_count_bits):
                step_cols.append((0x2, [i * block + 2 + kk]))
        nodes.append(_lut_layer_from_cols(
            node_id, cur_width, step_cols, {'kind': 'cmp_step', 'bit': k},
        ))
        edges.append(IREdge(src=cur_id, dst=node_id))
        cur_id = node_id
        cur_width = n_out * out_block
        block = out_block  # 次段のブロック規約は同じ

    # 最終: ge = gt or eq。各出力1列。OR(2入力) tt=0b1110(addr!=0で1)。
    final_id = f'{node_id_prefix}_ge'
    nodes.append(_lut_layer_from_cols(
        final_id,
        cur_width,
        [(0b1110, [i * block + 0, i * block + 1]) for i in range(n_out)],
        {'kind': 'cmp_ge'},
    ))
    edges.append(IREdge(src=cur_id, dst=final_id))
    return nodes, edges, final_id, n_out


def _gt_new_tt(t_k: int) -> int:
    """gt_new = gt or (eq and b_k and not t_k) の3入力真理値表。
    スロット: 0=gt, 1=eq, 2=b_k。addr bit0=gt, bit1=eq, bit2=b_k。"""
    tt = 0
    for addr in range(8):
        gt = addr & 1
        eq = (addr >> 1) & 1
        bk = (addr >> 2) & 1
        val = gt or (eq and bk and (1 - t_k))
        if val:
            tt |= 1 << addr
    return tt


def _eq_new_tt(t_k: int) -> int:
    """eq_new = eq and (b_k == t_k) の3入力真理値表。
    スロット: 0=gt, 1=eq, 2=b_k。"""
    tt = 0
    for addr in range(8):
        eq = (addr >> 1) & 1
        bk = (addr >> 2) & 1
        val = eq and (int(bk == t_k))
        if val:
            tt |= 1 << addr
    return tt


def _xnor_layer_nodes(
    q: torch.Tensor,
    node_id: str,
    in_width: int,
) -> tuple[IRNode, np.ndarray]:
    """XNORビット m_ij = XNOR(x_j, [q_ij>0]) を生成する縮退LUT層。

    q_ij>0 → identity(x_j)、q_ij<0 → NOT(x_j)。出力幅 = out_dim*in_dim。
    出力列 (i*in_dim + j) が m_ij。

    Returns:
        (lut_node, cols): cols[(i, j)] = このノードの出力列index
    """
    out_dim, in_dim = q.shape
    width = out_dim * in_dim
    indices = np.zeros((2, width), dtype=np.int64)
    tts: list[int] = []
    cols = np.zeros((out_dim, in_dim), dtype=np.int64)
    for i in range(out_dim):
        for j in range(in_dim):
            col = i * in_dim + j
            cols[i, j] = col
            indices[0, col] = j
            indices[1, col] = j
            tts.append(TT_IDENTITY if int(q[i, j]) > 0 else TT_NOT)
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': width,
            'n_inputs': 2,
            'indices': indices,
            'truth_tables': tt_to_hex(tts),
        },
        train_info={'kind': 'xnor_bits'},
    )
    return node, cols


def _hidden_thresholds(q: torch.Tensor, bias: torch.Tensor,
                       alpha: float) -> list[int]:
    """XNORカウント閾値 θ_i = floor((n - b'_i)/2) + 1 を導出する。

    n = in_width、b'_i = round(bias_i / alpha)。
    発火条件 P_i >= θ_i(P_i = XNORカウント)。導出はモジュールdocstring参照。
    """
    n = q.shape[1]
    thetas: list[int] = []
    for i in range(q.shape[0]):
        b_prime = int(round(float(bias[i]) / alpha)) if alpha > 0 else 0
        theta = (n - b_prime) // 2 + 1  # floor division(Pythonは負でも床)
        thetas.append(int(theta))
    return thetas


def quantize_threshold_layer(
    weight: torch.Tensor, bias: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """隠れ閾値ニューロンのニューロン別スケール整数量子化(')。

    fan-in 32段の閾値ニューロン y_i = 1[Σ_j w_ij x_j + b_i > 0] を、
    ニューロンごとの独立スケール s_i で整数化する。sign(bits=1)より高分解能な
    重み表現を許し、加算木loweringの回路規模と量子化損失のトレードオフを開く。

    量子化(bits>=2):
        s_i   = max_j|w_ij| / qmax        (qmax = 2^(bits-1) - 1、ニューロン別)
        q_ij  = round(w_ij / s_i)          (∈ [-qmax, qmax] の整数)
        b'_i  = round(b_i / s_i)           (同一スケールでバイアス整数化)

    発火条件は Σ_j q_ij x_j + b'_i > 0(スケール s_i>0 で不等号不変)。
    bits=1 は sign 量子化(binarize_weights と一致、s_i=mean|w_i|でバイアス整数化)
    にフォールバックする(lower_int_mlp が bits=1 で lower_bnn_mlp と整合するため)。

    Args:
        weight: (out, in) の実数重み
        bias:   (out,) の実数バイアス
        bits:   量子化ビット幅(>=1)

    Returns:
        (q, beta):
            q    (out, in) int64 — 整数量子化重み
            beta (out,)     int64 — 整数量子化バイアス b'_i

    Raises:
        ValueError: bits < 1
    """
    if bits < 1:
        raise ValueError(f'bitsは1以上: {bits}')
    weight = weight.to(torch.float64)
    bias = bias.to(torch.float64)
    if bits == 1:
        # sign量子化: q=±1、s_i=mean|w_i|(XNOR-Net、ニューロン別)でバイアス整数化。
        q = torch.where(weight >= 0, 1, -1).to(torch.int64)
        s = weight.abs().mean(dim=1).clamp_min(1e-12)            # (out,)
        beta = torch.round(bias / s).to(torch.int64)
        return q, beta
    qmax = (1 << (bits - 1)) - 1
    s = weight.abs().amax(dim=1).clamp_min(1e-12) / qmax          # (out,)
    q = torch.round(weight / s[:, None]).to(torch.int64)
    q = q.clamp(-qmax, qmax)
    beta = torch.round(bias / s).to(torch.int64)
    return q, beta


def _unary_threshold_layer_nodes(
    q: torch.Tensor,
    node_id: str,
    in_width: int,
) -> tuple[IRNode, np.ndarray, int, list[int]]:
    """整数量子化重み q から unary 展開ビット層を構築する(加算木の入力)。

    ニューロン i の count_i = Σ_j q_ij·x_j + C0_i を popcount で数えられるよう、
    各接続を |q_ij| 本の unary ビットへ展開する:
        q_ij > 0 → identity(x_j) を q_ij 本(x_j=1 のとき +1 を count に寄与)
        q_ij < 0 → NOT(x_j)     を |q_ij| 本(x_j=0 のとき +1、= C0_i の一部)
    これにより popcount(neuron i の全ビット) = Σ_j q_ij·x_j + C0_i = count_i。

    ニューロンごとの unary 幅 W_i = Σ_j|q_ij| は不揃いなので、全ニューロンを
    最大幅 W_max へ定数0ビット(TT_ZERO)でパディングする(popcount 不変)。
    後段 popcount_threshold_nodes は一様幅 input_cols を要求するため。

    Returns:
        (lut_node, input_cols, width_max, c0):
            lut_node    unary+padding の lut_layer(n_inputs=2 縮退LUT)
            input_cols  (out, W_max) int64 — 各ニューロンの W_max 本の出力列index
            width_max   W_max
            c0          [C0_i] = [Σ_j|q^-_ij|]
    """
    out_dim, in_dim = q.shape
    qn = q.to(torch.int64).cpu().numpy()
    # 各ニューロンの unary ビット (src_j, tt) 列と C0_i を作る
    per_neuron: list[list[tuple[int, int]]] = []
    c0: list[int] = []
    for i in range(out_dim):
        bits: list[tuple[int, int]] = []
        c0_i = 0
        for j in range(in_dim):
            v = int(qn[i, j])
            if v > 0:
                bits += [(j, TT_IDENTITY)] * v
            elif v < 0:
                bits += [(j, TT_NOT)] * (-v)
                c0_i += -v
        per_neuron.append(bits)
        c0.append(c0_i)
    width_max = max(1, max(len(b) for b in per_neuron))

    total = out_dim * width_max
    indices = np.zeros((2, total), dtype=np.int64)
    tts: list[int] = []
    input_cols = np.zeros((out_dim, width_max), dtype=np.int64)
    for i in range(out_dim):
        bits = per_neuron[i]
        padded = bits + [(0, TT_ZERO)] * (width_max - len(bits))
        for w, (src, tt) in enumerate(padded):
            col = i * width_max + w
            input_cols[i, w] = col
            indices[0, col] = src
            indices[1, col] = src
            tts.append(tt)
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': total,
            'n_inputs': 2,
            'indices': indices,
            'truth_tables': tt_to_hex(tts),
        },
        train_info={'kind': 'int_unary_bits'},
    )
    return node, input_cols, width_max, c0


def lower_int_mlp(
    model: nn.Module,
    *,
    weight_bits_hidden: int = 4,
    weight_bits_final: int = 8,
) -> IRGraph:
    """nn.Linear列のMLPを整数量子化閾値ニューロンとして厳密に回路化する(')。

    lower_bnn_mlp の一般化。隠れ層はニューロン別スケール整数量子化
    (quantize_threshold_layer, bits=weight_bits_hidden)→ unary展開
    (identity/NOT LUT)→ popcount加算木+per-neuron閾値で厳密lowering。
    最終層は transcribe._final_layer_nodes(argmax厳密保存)で lowering する。

    意味論(このモジュールが厳密一致を保証する参照):
        隠れ: count_i = Σ_j q_ij·x_j + C0_i(C0_i = Σ_j|q^-_ij|)
              y_i = 1[count_i >= θ_i]、θ_i = C0_i − b'_i + 1
              (⟺ Σ_j q_ij·x_j + b'_i > 0、= 量子化閾値ニューロンの発火判定)
        θ_i のクランプ(θ≤0=常真、θ>W_i=常偽)は popcount_threshold_nodes 既存
        挙動([0, width+1] クランプ)へ委譲する。unary幅は最大幅へ定数0
        パディングするが、パディングは popcount 不変なので閾値意味論を保つ。

    bits=1(weight_bits_hidden=1)では sign 量子化にフォールバックし、
    隠れニューロンの発火判定が lower_bnn_mlp と同一関数になる
    (θ_i = C0_i − b'_i + 1 と XNOR版 θ の等価性はテストで確認)。

    Args:
        model: nn.Linearを1つ以上含むモジュール(children順、活性化は発火判定
            として吸収)。隠れ層が無い単層でも動く(最終層のみ)
        weight_bits_hidden: 隠れ層の量子化ビット幅(>=1)
        weight_bits_final:  最終層の量子化ビット幅(>=1、transcribe側の共有スケール)

    Returns:
        検証可能なIRGraph(入力'x'→unary→popcount→...→final→group_sum→'y')。
        metadata['source']='convert.lower_int_mlp'、回路統計・bits を含む。

    Raises:
        ValueError: nn.Linearが見つからない
    """
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError('量子化対象のnn.Linearが見つからない')

    in_width = linears[0].in_features
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id = 'x'
    prev_width = in_width
    hidden_widths: list[int] = []

    for li, lin in enumerate(linears[:-1]):
        w = lin.weight.detach().to(torch.float64)
        b = (lin.bias.detach().to(torch.float64) if lin.bias is not None
             else torch.zeros(w.shape[0], dtype=torch.float64))
        q, beta = quantize_threshold_layer(w, b, weight_bits_hidden)
        out_dim = q.shape[0]

        # unary 展開層(identity/NOT + 定数0パディング)
        un_id = f'n{li}_unary'
        un_node, input_cols, width_max, c0 = _unary_threshold_layer_nodes(
            q, un_id, prev_width
        )
        nodes.append(un_node)
        edges.append(IREdge(src=prev_id, dst=un_id))
        hidden_widths.append(width_max)

        # 閾値 θ_i = C0_i − b'_i + 1(count_i >= θ_i ⟺ Σ q x + b' > 0)
        thetas = [int(c0[i]) - int(beta[i].item()) + 1 for i in range(out_dim)]

        pc_nodes, pc_edges, pc_last, _ = popcount_threshold_nodes(
            width=width_max,
            thresholds=thetas,
            node_id_prefix=f'n{li}_pc',
            in_width=un_node.params['out_width'],
            input_cols=input_cols,
        )
        pc_edges = [IREdge(src=un_id, dst=f'n{li}_pc_p0')] + pc_edges
        nodes += pc_nodes
        edges += pc_edges

        prev_id = pc_last
        prev_width = out_dim

    # 最終層: transcribe._final_layer_nodes(共有スケール整数量子化)
    final = linears[-1]
    fw = final.weight.detach().to(torch.float64)
    fb = (final.bias.detach().to(torch.float64) if final.bias is not None
          else torch.zeros(fw.shape[0], dtype=torch.float64))
    lut_node, gs_node, g = _final_layer_nodes(
        fw, fb, weight_bits=weight_bits_final,
        node_id=f'n{len(linears) - 1}_final',
    )
    nodes += [lut_node, gs_node]
    edges += [
        IREdge(src=prev_id, dst=lut_node.id),
        IREdge(src=lut_node.id, dst=gs_node.id),
        IREdge(src=gs_node.id, dst='y'),
    ]

    total_luts = sum(
        n.params['out_width'] for n in nodes if n.type == 'lut_layer'
    )
    return IRGraph(
        metadata={
            'source': 'convert.lower_int_mlp',
            'quantization': 'per_neuron_scale',
            'weight_bits_hidden': int(weight_bits_hidden),
            'weight_bits_final': int(weight_bits_final),
            'circuit': {
                'total_lut_outputs': int(total_luts),
                'n_lut_layers': int(sum(1 for n in nodes
                                        if n.type == 'lut_layer')),
                'group_width': g,
                'hidden_unary_widths': hidden_widths,
            },
        },
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=final.out_features,
                        semantics='class_scores')],
        nodes=nodes,
        edges=edges,
    )


def lower_bnn_mlp(
    model: nn.Module,
    *,
    per_neuron_threshold: bool = True,
) -> IRGraph:
    """nn.Linear列のMLPをsign二値化し、全層を厳密に回路化する(C2)。

    二値化意味論(bipolar BNN、モジュールdocstring参照)を厳密に実装する。
    隠れ層は XNORビット生成(縮退LUT)→ popcount閾値比較、最終層は
    transcribe._final_layer_nodes(argmax厳密保存、sign量子化=±1と同一)。

    Args:
        model: nn.Linearを1つ以上含むモジュール(children順、活性化は発火
            判定として吸収)。最低2層(隠れ1+最終1)を想定するが、単層でも動く
            (その場合はpopcount経路なしで最終層のみ)
        per_neuron_threshold: Trueでニューロンごとに閾値θ_iを導出(既定)。
            Falseでも同式だが将来の共有閾値最適化のフック(v1は同挙動)

    Returns:
        検証可能なIRGraph(入力'x'→XNOR→popcount→...→final→group_sum→'y')。
        metadata['source']='convert.lower_bnn_mlp'、回路統計を含む。

    Raises:
        ValueError: nn.Linearが見つからない
    """
    del per_neuron_threshold  # v1では常にper-neuron導出(フックのみ)
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError('二値化対象のnn.Linearが見つからない')

    in_width = linears[0].in_features
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id = 'x'
    prev_width = in_width
    n_stages = 0

    for li, lin in enumerate(linears[:-1]):
        w = lin.weight.detach().to(torch.float64)
        b = (lin.bias.detach().to(torch.float64) if lin.bias is not None
             else torch.zeros(w.shape[0], dtype=torch.float64))
        q, alpha = binarize_weights(w)
        out_dim, in_dim = q.shape
        thetas = _hidden_thresholds(q, b, alpha)

        # XNORビット層
        xnor_id = f'n{li}_xnor'
        xnor_node, cols = _xnor_layer_nodes(q, xnor_id, prev_width)
        nodes.append(xnor_node)
        edges.append(IREdge(src=prev_id, dst=xnor_id))
        n_stages += 1

        # popcount閾値層(各ニューロンiの in_dim 本の m_ij を集計)
        input_cols = np.zeros((out_dim, in_dim), dtype=np.int64)
        for i in range(out_dim):
            for j in range(in_dim):
                input_cols[i, j] = cols[i, j]
        pc_nodes, pc_edges, pc_last, _ = popcount_threshold_nodes(
            width=in_dim,
            thresholds=thetas,
            node_id_prefix=f'n{li}_pc',
            in_width=xnor_node.params['out_width'],
            input_cols=input_cols,
        )
        # popcountノードの入力は xnor_node。pc_nodesの先頭エッジ src を接続。
        # popcount_threshold_nodes内は prev_id を仮に持たないので、最初の
        # ノード(_p0)への入力エッジをここで xnor_id から張る。
        pc_edges = [IREdge(src=xnor_id, dst=f'n{li}_pc_p0')] + pc_edges
        nodes += pc_nodes
        edges += pc_edges
        n_stages += len(pc_nodes)

        prev_id = pc_last
        prev_width = out_dim  # popcount出力 = 各ニューロン1ビット

    # 最終層: transcribe._final_layer_nodes(sign量子化=±1と同一)
    final = linears[-1]
    fw = final.weight.detach().to(torch.float64)
    fb = (final.bias.detach().to(torch.float64) if final.bias is not None
          else torch.zeros(fw.shape[0], dtype=torch.float64))
    lut_node, gs_node, g = _final_layer_nodes(
        fw, fb, weight_bits=1, node_id=f'n{len(linears) - 1}_final'
    )
    nodes += [lut_node, gs_node]
    edges += [
        IREdge(src=prev_id, dst=lut_node.id),
        IREdge(src=lut_node.id, dst=gs_node.id),
        IREdge(src=gs_node.id, dst='y'),
    ]
    n_stages += 2

    total_luts = sum(
        n.params['out_width'] for n in nodes if n.type == 'lut_layer'
    )
    return IRGraph(
        metadata={
            'source': 'convert.lower_bnn_mlp',
            'binarization': 'sign',
            'circuit': {
                'total_lut_outputs': int(total_luts),
                'n_lut_layers': int(sum(1 for n in nodes
                                        if n.type == 'lut_layer')),
                'group_width': g,
            },
        },
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=final.out_features,
                        semantics='class_scores')],
        nodes=nodes,
        edges=edges,
    )
