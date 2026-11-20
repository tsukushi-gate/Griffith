"""事前学習MLP→LUT網IRの直接転写。

学習済みの実数値MLP(nn.Linear列)を、再学習なしで論理ゲートIRへ変換する。
NullaNet(2018)のISF/don't-care列挙とLogicNets(2020)のfan-in制約を骨格に、
ANN→SNN変換の層別data-based較正(Diehl 2015)を疎化バイアス再較正として取り込む。

変換の意味論(このモジュールが厳密一致を保証する参照):
    隠れ層: y_i = 1[Σ_j W_ij x_j + b_i > 0]   (x∈{0,1}、発火判定の二値化)
    最終層: score_c = Σ_j q_cj x_j + β_c       (q=W/sの整数量子化、β=b/sの整数量子化)
            argmax_c score_c を保存する形でIR化する

最終層のIR表現(argmax厳密保存):
    count_c = Σ_{q>0} q·x_j + Σ_{q<0} |q|·(1-x_j) + ones_c
            = score_c - β_c + C0_c + ones_c     (C0_c = Σ|q^-|)
    ones_c = K + β_c - C0_c、K = max_c(C0_c - β_c) と選ぶと
    count_c = score_c + K (全クラス共通オフセット) → argmax不変。
    x_j / NOT x_j / 定数0 / 定数1 はすべて n_inputs=2 の縮退LUT
    (tt: identity=0x8, not=0x1, zero=0x0, one=0xF)で表し、
    クラス群を同幅にゼロパディングして group_sum(k=C) で集計する。

制約(v1):
    - 対象はLinear列(活性化は発火判定として吸収)。Convは`transcribe_conv2d`参照
    - 隠れ活性の量子化は1bit(多ビットthermometerは将来拡張)
    - 較正データは決定的変換前提(datasets/vision.pyと同じ不変条件)

CNN→Conv-DLGN転写(`transcribe_conv2d`)の概念的注意点:
    Conv-DLGN(conv.py::LutConv2d)のtreeゲートは受容野の中から
    n_inputs^tree_depth本だけを read-once N分木でサンプルする。
    一方nn.Conv2dのカーネルは受容野**全ビット**に対する閾値関数であり、
    read-once木で厳密表現できる保証は一般に無い(fan-in制約前のMLP転写と
    同じ構図)。ゆえにtranscribe_conv2dは**近似**構造転写であり、
    「良い近似初期値→soft磨きで回復」というパラダイム(MLP版の同じ)
    の対象になる。ビット厳密性を主張できるのは以下の2点のみ:
        (1) 内部整合性: 転写ノードのsimulateが、構築した真理値表が定義する
            関数と厳密に一致すること(IRの読み書きにバグが無いことの保証)
        (2) 近似誤差: サンプルされたリーフだけを見るtree関数が、
            受容野全体を見るbinarize済みnn.Conv2d関数とどれだけ一致するか
            (較正データ上のagreement rateとして報告する。0を主張しない)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex

# binarize.pyはこのモジュール(TT_IDENTITY等)を逆にimportするため、
# モジュールレベルでの相互importは循環になる。binarize_weightsは
# transcribe_conv2d内で遅延importする(関数実行時には両モジュールとも
# 初期化済みなので安全)。

# 縮退LUT(n_inputs=2、両スロット同一index)の真理値表。
# addrは(x,x)なので到達するのは0(x=0)と3(x=1)のみ。未到達bit1,2は0に固定
TT_IDENTITY = 0x8  # bit3=1: x=1で1
TT_NOT = 0x1       # bit0=1: x=0で1
TT_ZERO = 0x0
TT_ONE = 0xF

DONT_CARE_POLICIES = ('zero', 'one', 'majority')


@dataclass
class TranscribeSpec:
    """転写仕様(スイープ軸)。

    Attributes:
        n_inputs: 隠れ層LUTのfan-in(LutLayer対応値: 2/4/6)
        weight_bits: 最終層の整数量子化ビット幅(1=sign。共有スケールs=max|W|/qmax)
        isf: Trueなら較正データが励起したアドレスのみ厳密評価(NullaNet ISF)
        dont_care: ISFで未観測アドレスを埋める方策('zero'|'one'|'majority')
    """

    n_inputs: int = 6
    weight_bits: int = 1
    isf: bool = False
    dont_care: str = 'zero'

    def __post_init__(self):
        if self.n_inputs not in (2, 4, 6):
            raise ValueError(f'n_inputsは2/4/6のみ: {self.n_inputs}')
        if self.weight_bits < 1:
            raise ValueError(f'weight_bitsは1以上: {self.weight_bits}')
        if self.dont_care not in DONT_CARE_POLICIES:
            raise ValueError(
                f'dont_careは{DONT_CARE_POLICIES}のみ: {self.dont_care}'
            )


def sparsify_fanin(
    weight: torch.Tensor,
    k: int,
    *,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """各ニューロンを|W|上位k接続へ疎化し、落とした接続をバイアスへ再較正する。

    再較正はDiehl 2015のdata-based較正の疎化版:
        bias_shift_i = Σ_{j∉S_i} W_ij · mean(x_j)   (較正データの平均活性)
    較正データなしでは0(落とした接続の寄与を単純に無視)。

    Args:
        weight: (out, in) の実数重み
        k: 残す接続数(in以下にクランプ)
        calib_inputs: (B, in) の較正入力(0/1)。省略可

    Returns:
        (indices, w_sel, bias_shift):
            indices (k, out) int64 ndarray — IR lut_layerのindices規約(スロット, ニューロン)
            w_sel   (out, k) — 選択された重み
            bias_shift (out,) — バイアスへ加算する再較正項
    """
    out_dim, in_dim = weight.shape
    k = min(k, in_dim)
    order = weight.abs().argsort(dim=1, descending=True)  # (out, in)
    sel = order[:, :k]                                    # (out, k)
    w_sel = weight.gather(1, sel)

    if calib_inputs is not None:
        x_mean = torch.as_tensor(
            np.asarray(calib_inputs), dtype=weight.dtype
        ).mean(dim=0)                                     # (in,)
        total = weight @ x_mean                           # (out,)
        kept = (w_sel * x_mean[sel]).sum(dim=1)           # (out,)
        bias_shift = total - kept
    else:
        bias_shift = torch.zeros(out_dim, dtype=weight.dtype)

    indices = sel.T.contiguous().cpu().numpy().astype(np.int64)  # (k, out)
    return indices, w_sel, bias_shift


def sparsify_from_mask(
    weight: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    *,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """外部0マスク(ternary量子化由来)で残す接続を決める疎化版sparsify_fanin。

    sparsify_fanin が |W|上位k本を選ぶのに対し、本関数は **mask=True の接続
    のみ**を候補とし、そのうち |W| 上位k本を選ぶ(ternaryの0=枝刈り済み構造を
    尊重する。追加の magnitude 枝刈りをしない)。マスク内非ゼロ数がkを超える
    ニューロンは |W| 上位kへ切り捨て、切り捨て分は sparsify_fanin と同じ
    data-based較正でバイアスへ再較正する。k未満のニューロンは残りスロットを
    重み0のダミー接続で埋める(LutLayer規約 n_inputs 固定のため)。

    Args:
        weight: (out, in) の実効重み(ternaryなら q·scale。mask=False位置は0前提)
        mask: (out, in) bool。True=残す候補
        k: LUT fan-in(残す接続数の上限、in以下にクランプ)
        calib_inputs: (B, in) の較正入力(0/1)。省略可

    Returns:
        (indices, w_sel, bias_shift): sparsify_fanin と同一規約
            indices (k, out) int64、w_sel (out, k)、bias_shift (out,)
    """
    out_dim, in_dim = weight.shape
    k = min(k, in_dim)
    # mask=True を優先: masked-out は |W| を -1 に落として選外へ回す
    score = weight.abs().clone()
    score[~mask] = -1.0
    order = score.argsort(dim=1, descending=True)   # (out, in)
    sel = order[:, :k]                               # (out, k)
    sel_valid = mask.gather(1, sel)                  # (out, k) bool: 実接続か埋めか
    w_sel = weight.gather(1, sel) * sel_valid        # 埋めスロットは重み0

    if calib_inputs is not None:
        x_mean = torch.as_tensor(
            np.asarray(calib_inputs), dtype=weight.dtype
        ).mean(dim=0)                                # (in,)
        total = weight @ x_mean                      # (out,)(masked-out=0前提)
        kept = (w_sel * x_mean[sel]).sum(dim=1)      # (out,)
        bias_shift = total - kept                    # 切り捨て接続の寄与
    else:
        bias_shift = torch.zeros(out_dim, dtype=weight.dtype)

    indices = sel.T.contiguous().cpu().numpy().astype(np.int64)  # (k, out)
    return indices, w_sel, bias_shift


def _all_patterns(n: int) -> torch.Tensor:
    """butterfly基底順の全入力パターン(2^n, n)。pattern[k, j] = bit j of k"""
    ks = torch.arange(1 << n)
    return ((ks[:, None] >> torch.arange(n)[None, :]) & 1).float()


def threshold_truth_tables(w_sel: torch.Tensor, bias: torch.Tensor) -> list[int]:
    """閾値ゲート y=1[w·x+b>0] を全列挙して真理値表化する(厳密)。

    Args:
        w_sel: (out, k) の重み
        bias: (out,) のバイアス

    Returns:
        (out,) の真理値表整数列(bit addr = butterfly基底順パターンaddrの出力)
    """
    k = w_sel.shape[1]
    patterns = _all_patterns(k).to(w_sel.dtype)           # (2^k, k)
    acts = w_sel @ patterns.T + bias[:, None]             # (out, 2^k)
    bits = (acts > 0).to(torch.int64)
    weights = [1 << a for a in range(1 << k)]
    return [
        int(sum(w for w, b in zip(weights, row.tolist(), strict=True) if b))
        for row in bits
    ]


def enumerate_truth_table(fn, n_inputs: int) -> int:
    """任意の二値関数 fn: (n,)の0/1ベクトル → 0/1 を全列挙して真理値表化する。

    テスト・カスタム部品用の汎用版(閾値ゲートはthreshold_truth_tablesが高速)。
    """
    tt = 0
    for addr in range(1 << n_inputs):
        x = [(addr >> j) & 1 for j in range(n_inputs)]
        if int(fn(x)) & 1:
            tt |= 1 << addr
    return tt


def isf_truth_tables(
    w_sel: torch.Tensor,
    bias: torch.Tensor,
    observed_addrs: np.ndarray,
    dont_care: str = 'zero',
) -> list[int]:
    """観測アドレスのみ厳密評価するISF列挙(NullaNet 2018)。

    較正データが励起したアドレスは閾値ゲートの厳密値、未観測はdon't-care
    として方策で埋める。閾値ゲートでは全列挙も安価だが、
    「観測分布への特化がどれだけ効くか」を測る対照として実装する。

    Args:
        w_sel: (out, k) の重み
        bias: (out,) のバイアス
        observed_addrs: (B, out) の較正データ由来アドレス行列
        dont_care: 'zero' | 'one' | 'majority'(観測値の多数決)

    Returns:
        (out,) の真理値表整数列
    """
    if dont_care not in DONT_CARE_POLICIES:
        raise ValueError(f'dont_careは{DONT_CARE_POLICIES}のみ: {dont_care}')
    exact = threshold_truth_tables(w_sel, bias)
    out_dim, k = w_sel.shape
    n_entries = 1 << k
    tts = []
    for o in range(out_dim):
        obs = np.unique(observed_addrs[:, o])
        t_exact = exact[o]
        obs_bits = [(t_exact >> int(a)) & 1 for a in obs]
        if dont_care == 'zero':
            fill = 0
        elif dont_care == 'one':
            fill = 1
        else:  # majority
            fill = 1 if (obs_bits and sum(obs_bits) * 2 >= len(obs_bits)) else 0
        t = ((1 << n_entries) - 1) if fill else 0
        obs_set = set(int(a) for a in obs)
        for addr in obs_set:
            bit = (t_exact >> addr) & 1
            t = (t | (1 << addr)) if bit else (t & ~(1 << addr))
        tts.append(t)
    return tts


def _quantize_final(weight: torch.Tensor, bias: torch.Tensor,
                    weight_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """最終層の共有スケール整数量子化。argmax比較のためスケールsは層全体で共有。

    weight_bits=1はsign量子化(XNOR-Net)、>1は q=round(W/s)、
    s = max|W| / (2^(bits-1) - 1)。バイアスも同一スケールで整数化する。
    """
    if weight_bits == 1:
        q = weight.sign().to(torch.int64)
        s = weight.abs().mean().clamp_min(1e-12)  # XNOR-Net α=mean|W|(bias用)
    else:
        qmax = (1 << (weight_bits - 1)) - 1
        s = weight.abs().max().clamp_min(1e-12) / qmax
        q = torch.round(weight / s).to(torch.int64)
    beta = torch.round(bias / s).to(torch.int64)
    return q, beta


def _final_layer_nodes(
    weight: torch.Tensor,
    bias: torch.Tensor,
    weight_bits: int,
    node_id: str,
) -> tuple[IRNode, IRNode, int]:
    """最終Linear→(縮退LUT展開層, group_sum)ノード対を構築する。

    モジュールdocstringのargmax厳密保存構成。戻り値は
    (lut_layerノード, group_sumノード, クラス群幅g)。
    """
    q, beta = _quantize_final(weight, bias, weight_bits)
    n_classes, in_dim = q.shape
    c0 = (-q.clamp(max=0)).sum(dim=1)                     # C0_c = Σ|q^-|
    kk = (c0 - beta).max()                                # K
    ones = (kk + beta - c0).tolist()                      # ones_c ≥ 0

    # クラスごとのビット列(index, tt)を構築
    per_class: list[list[tuple[int, int]]] = []
    for c in range(n_classes):
        bits: list[tuple[int, int]] = []
        for j in range(in_dim):
            v = int(q[c, j])
            if v > 0:
                bits += [(j, TT_IDENTITY)] * v
            elif v < 0:
                bits += [(j, TT_NOT)] * (-v)
        bits += [(0, TT_ONE)] * int(ones[c])
        per_class.append(bits)

    g = max(1, max(len(b) for b in per_class))  # 全ゼロ量子化でも幅1を確保
    indices = np.zeros((2, n_classes * g), dtype=np.int64)
    tts: list[int] = []
    for c, bits in enumerate(per_class):
        padded = bits + [(0, TT_ZERO)] * (g - len(bits))
        for i, (src, _tt) in enumerate(padded):
            col = c * g + i
            indices[0, col] = src
            indices[1, col] = src
        tts.extend(tt for _, tt in padded)

    lut_node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_dim,
            'out_width': n_classes * g,
            'n_inputs': 2,
            'indices': indices,
            'truth_tables': tt_to_hex(tts),
        },
        train_info={'kind': 'transcribe_final', 'weight_bits': weight_bits},
    )
    gs_node = IRNode(
        id=f'{node_id}_gs',
        type='group_sum',
        params={'in_width': n_classes * g, 'k': n_classes, 'tau': 1.0},
    )
    return lut_node, gs_node, g


def _simulate_hidden(h: np.ndarray, indices: np.ndarray,
                     tts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """転写済み隠れ層を較正データに適用する(層別較正伝播用)。

    Returns:
        (出力(B, out)の0/1, アドレス行列(B, out))
    """
    n = indices.shape[0]
    gathered = h[:, indices]                              # (B, n, out)
    weights = (1 << np.arange(n, dtype=np.int64))[None, :, None]
    addr = (gathered * weights).sum(axis=1)               # (B, out)
    tt_arr = np.asarray(tts, dtype=np.uint64)
    out = ((tt_arr[None, :] >> addr.astype(np.uint64)) & np.uint64(1))
    return out.astype(np.int64), addr


def transcribe_mlp(
    model: nn.Module,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
    spec: TranscribeSpec | None = None,
    masks: list[torch.Tensor] | None = None,
) -> IRGraph:
    """学習済みMLP(Linear列)をLUT網IRへ転写する(training-free)。

    Args:
        model: nn.Linearを1つ以上含むモジュール(children順に処理。
            活性化モジュールは発火判定二値化として吸収し個別には解釈しない)
        calib_inputs: (B, in_width) の0/1較正データ。疎化バイアス再較正と
            ISF列挙に使う。Noneなら較正なし(ISF不可)
        spec: 転写仕様(省略時は既定)
        masks: 隠れ層ごとの (out, in) bool 残存接続マスク(ternary量子化由来)。
            指定時は sparsify_fanin(|W|上位k)でなく sparsify_from_mask で
            マスク内から選ぶ。len は隠れLinear数(linears[:-1])と一致必須

    Returns:
        検証可能なIRGraph(入力ポート'x'→lut_layer列→group_sum→出力ポート'y')

    Raises:
        ValueError: Linearが見つからない / ISF指定で較正データなし /
            masks長が隠れ層数と不一致
    """
    spec = spec or TranscribeSpec()
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        # 直下にない場合は全モジュールから探す(Sequential入れ子等)
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError('転写対象のnn.Linearが見つからない')
    if spec.isf and calib_inputs is None:
        raise ValueError('isf=Trueにはcalib_inputsが必要')
    if masks is not None and len(masks) != len(linears) - 1:
        raise ValueError(
            f'masks長({len(masks)})は隠れLinear数({len(linears) - 1})と一致必須'
        )

    in_width = linears[0].in_features
    h = None
    if calib_inputs is not None:
        h = np.asarray(
            calib_inputs.detach().cpu().numpy()
            if isinstance(calib_inputs, torch.Tensor) else calib_inputs
        ).astype(np.int64)
        assert set(np.unique(h)).issubset({0, 1}), '較正データは0/1のみ'

    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id = 'x'

    for li, lin in enumerate(linears[:-1]):
        w = lin.weight.detach().to(torch.float64)
        b = (lin.bias.detach().to(torch.float64) if lin.bias is not None
             else torch.zeros(w.shape[0], dtype=torch.float64))
        if masks is not None:
            indices, w_sel, b_shift = sparsify_from_mask(
                w, masks[li].to(w.device), spec.n_inputs, calib_inputs=h
            )
        else:
            indices, w_sel, b_shift = sparsify_fanin(
                w, spec.n_inputs, calib_inputs=h
            )
        b_eff = b + b_shift

        # fan-in不足はスロット複製(index 0)+重み0でLutLayer規約(2/4/6)に合わせる
        k_actual = indices.shape[0]
        if k_actual < spec.n_inputs:
            pad = spec.n_inputs - k_actual
            indices = np.vstack(
                [indices, np.zeros((pad, indices.shape[1]), dtype=np.int64)]
            )
            w_sel = torch.cat(
                [w_sel, torch.zeros(w_sel.shape[0], pad, dtype=w_sel.dtype)],
                dim=1,
            )

        if spec.isf:
            _, addrs = _simulate_hidden(
                h, indices, threshold_truth_tables(w_sel, b_eff)
            )
            tts = isf_truth_tables(w_sel, b_eff, addrs, spec.dont_care)
        else:
            tts = threshold_truth_tables(w_sel, b_eff)

        node_id = f'n{li}_lut'
        nodes.append(IRNode(
            id=node_id,
            type='lut_layer',
            params={
                'in_width': lin.in_features,
                'out_width': lin.out_features,
                'n_inputs': spec.n_inputs,
                'indices': indices,
                'truth_tables': tt_to_hex(tts),
            },
            train_info={
                'kind': 'transcribe_hidden',
                'isf': spec.isf,
                'dont_care': spec.dont_care if spec.isf else None,
            },
        ))
        edges.append(IREdge(src=prev_id, dst=node_id))
        prev_id = node_id

        if h is not None:
            h, _ = _simulate_hidden(h, indices, tts)

    final = linears[-1]
    w = final.weight.detach().to(torch.float64)
    b = (final.bias.detach().to(torch.float64) if final.bias is not None
         else torch.zeros(w.shape[0], dtype=torch.float64))
    lut_node, gs_node, g = _final_layer_nodes(
        w, b, spec.weight_bits, f'n{len(linears) - 1}_final'
    )
    nodes += [lut_node, gs_node]
    edges += [
        IREdge(src=prev_id, dst=lut_node.id),
        IREdge(src=lut_node.id, dst=gs_node.id),
        IREdge(src=gs_node.id, dst='y'),
    ]

    return IRGraph(
        metadata={
            'source': 'convert.transcribe_mlp',
            'spec': {
                'n_inputs': spec.n_inputs,
                'weight_bits': spec.weight_bits,
                'isf': spec.isf,
                'dont_care': spec.dont_care,
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


# ============================================================
# CNN→Conv-DLGN構造転写(C1-conv)
# ============================================================

def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Conv2d + BatchNorm2d(eval統計)を1つのbias付きConv2dへ融合する(W13a-5)。

    標準的なBN畳み込み(推論時)融合式:
        w' = w · γ / √(var + ε)
        b' = β − mean · γ / √(var + ε) + b · γ / √(var + ε)
            (元convにbiasが無ければ b=0 として扱う)

    融合前後で `bn(conv(x))`(bn.eval()時)と `fused(x)` が数値的に一致する
    (テストで検証)。SmallUNetのような Conv-BN-ReLU ブロックを
    transcribe_conv2d(nn.Conv2d単体を要求)へ渡す前段として使う。

    Args:
        conv: 融合元のnn.Conv2d(bias有無どちらでも可)
        bn: 融合元のnn.BatchNorm2d(eval統計 running_mean/running_var を使う。
            呼び出し側で bn.eval() 済みであること)

    Returns:
        新規のnn.Conv2d(bias=True、convと同じin/out/kernel/stride/padding)
    """
    if conv.out_channels != bn.num_features:
        raise ValueError(
            f'conv.out_channels({conv.out_channels}) != '
            f'bn.num_features({bn.num_features})'
        )
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    )
    with torch.no_grad():
        gamma = bn.weight if bn.weight is not None else torch.ones(bn.num_features)
        beta = bn.bias if bn.bias is not None else torch.zeros(bn.num_features)
        mean = bn.running_mean
        var = bn.running_var
        scale = gamma / torch.sqrt(var + bn.eps)  # (out_channels,)

        conv_bias = (conv.bias if conv.bias is not None
                    else torch.zeros(conv.out_channels, dtype=conv.weight.dtype))

        fused.weight.copy_(
            conv.weight.to(torch.float64) * scale.to(torch.float64).view(-1, 1, 1, 1)
        )
        fused.bias.copy_(
            beta.to(torch.float64) + (conv_bias.to(torch.float64)
             - mean.to(torch.float64)) * scale.to(torch.float64)
        )
    return fused


def _generate_tree_leaf_indices(
    out_channels: int, n_leaves: int, receptive_field: int, seed: int,
) -> np.ndarray:
    """出力チャンネルごとに受容野内フラット位置をn_leaves本サンプルする。

    conv.py::_generate_tree_connections と同じ「チャンネル・高さ・幅を
    それぞれ一様ランダム」ではなく、ここではreceptive_field(=C*kH*kW)の
    フラット位置を直接一様サンプルする(transcribe_conv2dはunfold後の
    受容野ベクトルに対する閾値関数を近似するため、位置の生成源がどちらでも
    indices_flatの意味は同じ)。決定的RandomState(seed)を使い、
    receptive_field <= n_leaves なら重複なし(全域網羅)、それ以外は
    復元抽出(同一チャンネルを木の複数リーフが参照してよい、CDDLGN Eq.3と同じ)。

    Returns:
        (out_channels, n_leaves) int64 — indices_flat(0..receptive_field-1)
    """
    rng = np.random.RandomState(seed)
    idxf = np.empty((out_channels, n_leaves), dtype=np.int64)
    for oc in range(out_channels):
        if receptive_field <= n_leaves:
            # 全位置を敷き詰め、残りを復元抽出で埋める(小受容野の網羅性優先)
            base = np.arange(receptive_field, dtype=np.int64)
            extra = rng.randint(0, receptive_field,
                                size=n_leaves - receptive_field)
            idxf[oc] = np.concatenate([base, extra])[:n_leaves]
        else:
            idxf[oc] = rng.choice(receptive_field, size=n_leaves, replace=False)
    return idxf


def _majority_fill_gate(
    children_bits: np.ndarray, target: np.ndarray, n_inputs: int,
) -> tuple[int, np.ndarray]:
    """n_inputs本の子シグナルからターゲットを最良近似する真理値表を較正データの
    多数決で選び、その真理値表を較正データへ適用した出力を返す(貪欲な層別構築)。

    isf_truth_tables の dont_care='majority' と同じ発想(観測アドレスでの
    多数決)を、MLP閾値ゲートではなく任意の子シグナル→ターゲット対に適用する
    一般化。アドレスaに較正サンプルが1件も落ちない場合はターゲット全体の
    多数決で埋める(dont_careの外挿)。

    Args:
        children_bits: (n_inputs, B) 子シグナル(0/1)
        target: (B,) 近似したいターゲット(0/1)
        n_inputs: ゲート入力数

    Returns:
        (tt, gate_out): ttは2^n_inputsビット真理値表整数、gate_outは
        (B,) 較正データへこのゲートを適用したときの0/1出力(次段の子シグナルに使う)
    """
    n_entries = 1 << n_inputs
    weights = (1 << np.arange(n_inputs, dtype=np.int64))
    addr = (children_bits.T.astype(np.int64) * weights[None, :]).sum(axis=1)  # (B,)
    global_majority = int(round(float(target.mean()))) if target.size else 0
    tt = 0
    for a in range(n_entries):
        mask = addr == a
        if mask.any():
            bit = 1 if target[mask].mean() >= 0.5 else 0
        else:
            bit = global_majority
        if bit:
            tt |= 1 << a
    gate_out = ((tt >> addr) & 1).astype(np.int64)
    return tt, gate_out


def _binarized_conv_target(
    weight: torch.Tensor, bias: torch.Tensor, rf_bits: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, float]:
    """nn.Conv2dカーネルを二値化した閾値ゲートの厳密出力(全受容野)を返す。

    binarize.py の隠れ層意味論(y=1[Σq(2x-1)+b'>0])をconvカーネルへ適用した
    もの。これがtranscribe_conv2dが近似しようとする「ground truth」関数。

    Args:
        weight: (out_ch, C*kH*kW) 実数カーネル(平坦化済み)
        bias: (out_ch,) 実数バイアス
        rf_bits: (B, C*kH*kW) 較正用の受容野パッチ(0/1)

    Returns:
        (target, q, alpha):
            target (out_ch, B) int64 0/1 — 全受容野で見た厳密二値化出力
            q (out_ch, C*kH*kW) int64 ±1
            alpha float — XNOR-Netスケール
    """
    from .binarize import binarize_weights  # 遅延import(循環回避)
    q, alpha = binarize_weights(weight)
    b_prime = torch.round(bias.to(torch.float64) / alpha)
    rf = torch.as_tensor(rf_bits, dtype=torch.float64)  # (B, R)
    signed = 2.0 * rf - 1.0  # (B, R)
    scores = signed @ q.to(torch.float64).T + b_prime[None, :]  # (B, out_ch)
    target = (scores > 0).to(torch.int64).T.numpy()  # (out_ch, B)
    return target, q, alpha


def transcribe_conv2d(
    layer: nn.Conv2d,
    *,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
    spec: TranscribeSpec | None = None,
    tree_depth: int = 3,
    n_inputs: int = 2,
    seed: int = 42,
) -> tuple[IRNode, dict]:
    """学習済みnn.Conv2dをConv-DLGNのconv2d IRNodeへ近似構造転写する。

    モジュールdocstring「CNN→Conv-DLGN転写の概念的注意点」を参照。
    Conv-DLGNのtreeは受容野全体でなくn_inputs^tree_depth本のサンプル
    リーフしか見ないため、この転写は**近似**である(MLP版のfan-in疎化と
    同じ構図)。手順:
        1. カーネルをsign二値化(binarize_weights)し、受容野全体を見る
           厳密閾値ゲート(_binarized_conv_target)を「ground truth」とする
        2. 出力チャンネルごとにn_leaves=n_inputs^tree_depth本の受容野内
           フラット位置を決定的にサンプル(_generate_tree_leaf_indices)
        3. 較正データ上でボトムアップに木を構築する。各内部ゲートは
           子シグナル→ground truthを最良近似する真理値表を多数決で選ぶ
           (_majority_fill_gate、貪欲な層別回路探索)。選んだゲートの
           較正データ上のhard出力を次段の子シグナルとして伝播する

    Args:
        layer: 転写対象のnn.Conv2d(stride=1のみ対応、v1)
        calib_inputs: (B, C, H, W) の0/1較正データ(numpy/torch)。
            近似構築とpaddingの内容除外に使う。Noneだと構築不能(ValueError)
        spec: 予約(現状未使用。TranscribeSpecの重みビット等はconvには非適用)
        tree_depth: 木の深さd(LutConv2d.tree_depthと同じ)
        n_inputs: ゲート入力数(2/4/6のみ、LutConv2d制約)
        seed: リーフサンプリング・多数決タイブレークの決定的シード

    Returns:
        (node, info):
            node: 'conv2d' IRNode(train_info.parameterization='independent'。
                内部ゲートは任意真理値表になり得るため、単一minterm制約の
                softmaxでは表現不能な場合がある)
            info: {'approx_agreement': 較正データ上の近似一致率(全チャンネル平均),
                   'per_channel_agreement': (out_ch,) 一致率,
                   'n_leaves': int, 'n_internal': int}

    Raises:
        ValueError: stride!=1、n_inputs不正、calib_inputsなし、grouped/depthwise
    """
    del spec  # v1: conv2dには重み量子化ビット等が非適用。将来拡張用に予約
    if n_inputs not in (2, 4, 6):
        raise ValueError(f'n_inputsは2/4/6のみ: {n_inputs}')
    if layer.groups != 1:
        raise ValueError(f'grouped/depthwise convは未対応: groups={layer.groups}')
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,) * 2
    if stride != (1, 1):
        raise ValueError(f'transcribe_conv2dはstride=1のみ対応: stride={stride}')
    if calib_inputs is None:
        raise ValueError('transcribe_conv2dにはcalib_inputsが必要'
                         '(近似構築・agreement測定に使う)')

    kH, kW = layer.kernel_size
    pH, pW = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,) * 2
    in_ch = layer.in_channels
    out_ch = layer.out_channels
    receptive_field = in_ch * kH * kW

    x = calib_inputs.detach().cpu().numpy() if isinstance(calib_inputs, torch.Tensor) \
        else np.asarray(calib_inputs)
    x = x.astype(np.float64)
    assert set(np.unique(x)).issubset({0, 1}), '較正データは0/1のみ'
    assert x.ndim == 4 and x.shape[1] == in_ch, (
        f'calib_inputs形状は(B,{in_ch},H,W)が必要: {x.shape}'
    )
    in_h, in_w = int(x.shape[2]), int(x.shape[3])

    # unfold(padding込み)で受容野パッチ (B*L, R) を得る。paddingで補われた
    # 位置は0固定(F.unfoldの標準ゼロパディング)。
    x_t = torch.as_tensor(x, dtype=torch.float64)
    patches = F.unfold(x_t, (kH, kW), padding=(pH, pW))  # (B, R, L)
    b_, r_, l_ = patches.shape
    rf_bits = patches.permute(0, 2, 1).reshape(b_ * l_, r_).numpy()  # (B*L, R)
    # 空間位置が多いと較正コストが跳ねるため、行をサブサンプルして一定件数に抑える
    max_rows = 20000
    if rf_bits.shape[0] > max_rows:
        sel = np.random.RandomState(seed).choice(
            rf_bits.shape[0], size=max_rows, replace=False,
        )
        rf_bits = rf_bits[sel]

    weight_flat = layer.weight.detach().to(torch.float64).reshape(out_ch, -1)
    bias = (layer.bias.detach().to(torch.float64) if layer.bias is not None
            else torch.zeros(out_ch, dtype=torch.float64))
    target, _q, _alpha = _binarized_conv_target(weight_flat, bias, rf_bits)  # (out_ch,B)

    n_leaves = n_inputs ** tree_depth
    n_internal = tree_depth if n_inputs == 1 else (n_leaves - 1) // (n_inputs - 1)
    idxf = _generate_tree_leaf_indices(out_ch, n_leaves, receptive_field, seed)

    tts_by_channel: list[list[int]] = []
    agreement = np.empty(out_ch, dtype=np.float64)
    for oc in range(out_ch):
        leaves = rf_bits[:, idxf[oc]].T.astype(np.int64)  # (n_leaves, B)
        target_oc = target[oc]  # (B,)
        current = leaves
        gate_tts: list[int] = []
        for _level in range(tree_depth):
            n_groups = current.shape[0] // n_inputs
            next_signals = np.empty((n_groups, current.shape[1]), dtype=np.int64)
            for g in range(n_groups):
                children = current[g * n_inputs:(g + 1) * n_inputs]
                tt, gate_out = _majority_fill_gate(children, target_oc, n_inputs)
                gate_tts.append(tt)
                next_signals[g] = gate_out
            current = next_signals
        tts_by_channel.append(gate_tts)
        agreement[oc] = float((current[0] == target_oc).mean())

    tts_flat = [t for row in tts_by_channel for t in row]
    assert len(tts_flat) == out_ch * n_internal

    node = IRNode(
        id='transcribed_conv',
        type='conv2d',
        params={
            'in_shape': [in_ch, in_h, in_w],
            'out_channels': out_ch,
            'kernel_size': [kH, kW],
            'stride': [1, 1],
            'padding': [pH, pW],
            'tree_depth': tree_depth,
            'n_inputs': n_inputs,
            'indices_flat': idxf,
            'truth_tables': tt_to_hex(tts_flat),
        },
        train_info={
            'kind': 'transcribe_conv2d',
            # 任意真理値表を厳密表現できるパラメータ化(単一minterm制約の
            # softmaxでは多数決構築の真理値表を表現できない場合があるため)
            'parameterization': 'independent',
            'seed': seed,
        },
    )
    info = {
        'approx_agreement': float(agreement.mean()),
        'per_channel_agreement': agreement,
        'n_leaves': n_leaves,
        'n_internal': n_internal,
        'receptive_field': receptive_field,
    }
    return node, info
