"""LUT層の純autograd参照実装(テストのground truth)。

カスタムautograd Function・butterfly分解・CUDAカーネルを一切使わず、
論文の数式を素朴なtorch演算でそのまま書き下す。勾配はautogradが
自動導出するため、この実装の正しさはL1の数式テスト
(分割単位性・真理値表・積公式)だけで担保できる。

各カスタムFunction(LutFunction系)の出力・勾配は、この参照実装との
照合(L2: reference equivalence)で検証する。

数式:
    一般化LUT(確率的緩和):
        y = Σ_{k=0}^{2^N-1} f(W)[k] × Π_{j=0}^{N-1} (x_j if bit(k,j)=1 else 1-x_j)
    'iwp'(入力ごと1パラメータのリテラル積。**論文 arXiv:2510.03250 の IWP とは別物**。
    論文IWPに相当するのは 'independent'):
        y = Π_{j=0}^{N-1} [p_j x_j + (1-p_j)(1-x_j)],  p_j = sigmoid(α_j)
    WARP(Walsh-Hadamard方式):
        y = sigmoid(Σ_k θ_k φ_k(1-2x) / τ),  φ_k(s) = Π_{j: bit(k,j)=1} s_j
"""

import torch
import torch.nn.functional as F

from .estimators import estimator_rho


def reference_gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """接続テーブルに従い入力を収集: (batch, in_dim) → (batch, out_dim, n_inputs)"""
    batch = x.shape[0]
    n_inputs, out_dim = indices.shape
    return x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)


def reference_basis(gathered: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """確率的基底関数(素朴な積の直接計算)。

    basis[..., k] = Π_{j=0}^{N-1} (x_j if bit(k,j)=1 else 1-x_j)

    インデックス規約はbutterfly実装と同一: kのビットjが1なら因子はx_j。
    """
    terms = []
    for k in range(2 ** n_inputs):
        t = torch.ones_like(gathered[..., 0])
        for j in range(n_inputs):
            xj = gathered[..., j]
            t = t * (xj if (k >> j) & 1 else 1.0 - xj)
        terms.append(t)
    return torch.stack(terms, dim=-1)


def reference_wh_basis(gathered: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """Walsh-Hadamard基底(素朴な部分集合積の直接計算)。

    φ_k(s) = Π_{j: bit(k,j)=1} s_j,  s = 1 - 2x
    """
    s = 1.0 - 2.0 * gathered
    terms = []
    for k in range(2 ** n_inputs):
        t = torch.ones_like(s[..., 0])
        for j in range(n_inputs):
            if (k >> j) & 1:
                t = t * s[..., j]
        terms.append(t)
    return torch.stack(terms, dim=-1)


def reference_lut_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
    parameterization: str,
    gate_table: torch.Tensor | None = None,
    tau: float = 1.0,
    estimator: str = 'sigmoid',
) -> torch.Tensor:
    """LUT層forwardの参照実装(全パラメータ化対応、訓練時soft評価)。

    parameterization='independent' は論文 "Light Differentiable Logic Gate
    Networks" (arXiv:2510.03250) の IWP と数学的に同一である:

        g_ω(p, q) = Σ_k ω_k · e_k,   ω_k = ρ(Ω_k)              (Eq.11)
        e_k = Π_j (x_j if bit(k,j)=1 else 1-x_j)

    推定器 ρ は既定で sigmoid、estimator='sin' で論文§C.1.1 の
    sin01(Ω)=0.5+0.5·sin(Ω) を用いる(論文の実験設定はこちら)。

    Args:
        x: (batch, in_dim)、値は[0,1]
        weights: パラメータ化ごとの形状
            softmax/independent/raw/warp: (out_dim, 2^N)
            softmax16ゲートモード: (out_dim, 16) + gate_table
            iwp: (out_dim, N)
        indices: (n_inputs, out_dim)
        parameterization: 'softmax' | 'independent' | 'raw' | 'iwp' | 'warp'
            ※ 'iwp' は入力ごと1パラメータのリテラル積であり、**論文 arXiv:2510.03250
              の IWP ではない**(論文IWPは上記のとおり 'independent')
        gate_table: (16, 4)。softmax N=2 の16ゲートモードでのみ指定
        tau: WARPの温度
        estimator: independentの推定器ρ('sigmoid' | 'sin')

    Returns:
        (batch, out_dim)
    """
    gathered = reference_gather(x, indices)

    # 'iwp': 入力ごと1パラメータのリテラル積(丸めると単一minterm)。
    # **論文 arXiv:2510.03250 の IWP とは別物**で、2入力では16ゲート中4種しか
    # 表現できない(素通し G(A,B)=A も XOR も不可)。論文IWPは 'independent' の方。
    if parameterization == 'iwp':
        p = torch.sigmoid(weights)  # (out_dim, N)
        factors = p.unsqueeze(0) * gathered + (1.0 - p.unsqueeze(0)) * (1.0 - gathered)
        return factors.prod(dim=-1)

    if parameterization == 'warp':
        wh = reference_wh_basis(gathered, n_inputs)
        logit = (wh * weights.unsqueeze(0)).sum(dim=-1) / tau
        return torch.sigmoid(logit)

    basis = reference_basis(gathered, n_inputs)

    if parameterization == 'softmax':
        w = F.softmax(weights, dim=-1)
        if gate_table is not None:
            w = w @ gate_table.to(w.dtype)
    elif parameterization == 'independent':
        w = estimator_rho(weights, estimator)
    elif parameterization == 'raw':
        w = weights
    else:
        raise ValueError(f'未知のparameterization: {parameterization}')

    return (basis * w.unsqueeze(0)).sum(dim=-1)


def reference_warp_entry_logits(
    weights: torch.Tensor, n_inputs: int,
) -> torch.Tensor:
    """WARP係数θから各真理値表エントリのロジットを計算する(素朴ループ)。

    L[o, e] = Σ_i θ[o, i] · Π_{j: bit(i,j)=1} (1 - 2·bit(e,j))

    エントリeのビットjは入力パターンのx_j(LSB=slot 0、リポジトリ規約)。
    WARPのsoft forwardをbinary頂点eで評価したロジット(τ=1)と厳密一致する。
    """
    out_dim = weights.shape[0]
    n_entries = 2 ** n_inputs
    logits = torch.zeros(out_dim, n_entries,
                         dtype=weights.dtype, device=weights.device)
    for e in range(n_entries):
        for i in range(n_entries):
            sign = 1.0
            for j in range(n_inputs):
                if (i >> j) & 1:
                    sign *= (1.0 - 2.0 * ((e >> j) & 1))
            logits[:, e] += weights[:, i] * sign
    return logits


def reference_lut_warp_hard_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """WARP Hard-ST forwardの参照実装(閾値意味論、改訂)。

    エントリロジットの多重線形拡張を閾値化する:
        z = Σ_e basis_e(x) · (L[o,e] + noise[o,e]),  y = 1[z > 0]

    binary入力では z = L[o, addr(x)] なので per-entry hard表と厳密一致。
    連続入力でも出力はbinaryで、noise=0のとき z = θ·φ(s(x)) となり
    evalの離散化 1[θ·φ(s)>0] と**入力の連続性によらず**厳密一致する
    (旧・多重線形拡張版は連続入力で連続出力となりevalと乖離した —
    RDDLGN等の連続埋め込みスタックでhard崩壊の原因)。
    noise=Noneでhard_st、logisticノイズでgumbel_st。
    """
    gathered = reference_gather(x, indices)
    basis = reference_basis(gathered, n_inputs)
    logits = reference_warp_entry_logits(weights, n_inputs)
    if noise is not None:
        logits = logits + noise
    z = (basis * logits.unsqueeze(0)).sum(dim=-1)
    return (z > 0).to(basis.dtype)


def reference_lut_warp_hard_st_surrogate(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
    tau_b: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """WARP Hard-STのbackward契約の参照: σ(z/τ_b)によるsoft関数。

    z = Σ_e basis_e·(L+noise)_e(閾値意味論のロジット)。
    LutFunctionWARPHardSTのbackwardは本関数の純autograd勾配と
    厳密一致しなければならない(STE契約)。
    """
    gathered = reference_gather(x, indices)
    basis = reference_basis(gathered, n_inputs)
    logits = reference_warp_entry_logits(weights, n_inputs)
    if noise is not None:
        logits = logits + noise
    z = (basis * logits.unsqueeze(0)).sum(dim=-1)
    return torch.sigmoid(z / tau_b)


def reference_ltt_pre_activation(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, indices: torch.Tensor,
) -> torch.Tensor:
    """LTT(線形閾値ゲート)の前活性 z の参照実装(素朴な二重ループ)。

        z[i, j] = Σ_k w[j, k] · x[i, indices[k, j]] + b[j]

    Args:
        x: (batch, in_dim)、値は {0,1}(実数でも式はそのまま)
        w: (out_dim, n_inputs)
        b: (out_dim,)
        indices: (n_inputs, out_dim)

    Returns:
        (batch, out_dim)
    """
    n_inputs, out_dim = indices.shape
    rows = []
    for j in range(out_dim):
        z = b[j].expand(x.shape[0]).clone()
        for k in range(n_inputs):
            z = z + w[j, k] * x[:, int(indices[k, j])]
        rows.append(z)
    return torch.stack(rows, dim=1)


def reference_ltt_forward(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, indices: torch.Tensor,
    hard: bool = True, scale: float = 1.0,
) -> torch.Tensor:
    """LTT の forward 参照: hard=True で 1[z > 0]、False で σ(z / scale)(soft 評価)。"""
    z = reference_ltt_pre_activation(x, w, b, indices)
    if hard:
        return (z > 0).to(x.dtype)
    return torch.sigmoid(z / scale)


def reference_ltt_ste_surrogate(z: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    """LTT の代理勾配 dy/dz の参照(式をそのまま書く。layers/ltt.py と独立)。

        identity: 1
        clip:     1[|z| < s]
        sigmoid:  4·σ(z/s)·(1 − σ(z/s))
    """
    if mode == 'identity':
        return torch.ones_like(z)
    if mode == 'clip':
        return (z.abs() < scale).to(z.dtype)
    if mode == 'sigmoid':
        s = 1.0 / (1.0 + torch.exp(-z / scale))
        return 4.0 * s * (1.0 - s)
    raise ValueError(f'未知のste mode: {mode}')


def reference_ltt_grads(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, indices: torch.Tensor,
    grad_output: torch.Tensor, mode: str, scale: float,
) -> tuple:
    """LTT の STE 学習経路の (y, grad_x, grad_w, grad_b) 参照。

    y = 1[z > 0] の値と、dL/dz = grad_output · surrogate(z) を前活性の autograd で
    x / w / b へ流した勾配(カスタム Function を使わない)。
    """
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True)
    z = reference_ltt_pre_activation(xr, wr, br, indices)
    y = (z > 0).to(x.dtype)
    gz = grad_output * reference_ltt_ste_surrogate(z.detach(), mode, scale)
    gx, gw, gb = torch.autograd.grad(z, (xr, wr, br), grad_outputs=gz)
    return y.detach(), gx, gw, gb


def reference_soft_merge(inputs: list, mode: str) -> torch.Tensor:
    """soft論理マージの参照実装(IR v0.2 buildと同一式の素朴な逐次fold)。

    or:  y = 1-(1-a)(1-b)  ({0,1}入力でビットORと厳密一致)
    xor: y = a+b-2ab       ({0,1}入力でビットXORと厳密一致)
    sum: y = a+b           (整数和。論理領域の終端専用)
    """
    out = inputs[0]
    for t in inputs[1:]:
        if mode == 'or':
            out = 1 - (1 - out) * (1 - t)
        elif mode == 'xor':
            out = out + t - 2 * out * t
        elif mode == 'sum':
            out = out + t
        else:
            raise ValueError(f'未知のmergeモード: {mode}')
    return out


def reference_xnor_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """XNOR類似度soft attentionの参照実装(素朴なループ計算)。

    score[b,i,j] = Σ_d [q[b,i,d]·k[b,j,d] + (1-q[b,i,d])(1-k[b,j,d])]
    attn = softmax(score/τ, dim=-1)、 out[b,i] = Σ_j attn[b,i,j]·v[b,j]

    Args:
        q: (B, S_q, D)、k: (B, S_k, D)、v: (B, S_k, D_v)。値は[0,1]

    Returns:
        (B, S_q, D_v)
    """
    batch, s_q, _ = q.shape
    s_k = k.shape[1]
    scores = torch.zeros(batch, s_q, s_k, dtype=q.dtype, device=q.device)
    for b in range(batch):
        for i in range(s_q):
            for j in range(s_k):
                xnor = q[b, i] * k[b, j] + (1.0 - q[b, i]) * (1.0 - k[b, j])
                scores[b, i, j] = xnor.sum()
    attn = F.softmax(scores / tau, dim=-1)
    out = torch.zeros(batch, s_q, v.shape[-1], dtype=v.dtype, device=v.device)
    for b in range(batch):
        for i in range(s_q):
            for j in range(s_k):
                out[b, i] = out[b, i] + attn[b, i, j] * v[b, j]
    return out


def reference_xnor_attention_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """XNOR類似度causal(自己回帰)attentionの参照実装(素朴なループ計算)。

    reference_xnor_attentionと同じscore式に、位置iはj<=iのみを見る制約
    (未来位置j>iを完全除外)を加えたもの。デコーダ専用attention
    (LogicAttention(causal=True))のground truth。S_q==S_kの自己注意を想定
    (j>iの禁止はクエリ位置とキー位置が同じ時間軸を指すことが前提)。

    score[b,i,j] = Σ_d [q[b,i,d]·k[b,j,d] + (1-q[b,i,d])(1-k[b,j,d])]  (j<=iのみ)
    attn = softmax(score/τ, dim=-1)  (j>iはsoftmax計算から除外=重み0)
    out[b,i] = Σ_{j<=i} attn[b,i,j]·v[b,j]

    Args:
        q: (B, S, D)、k: (B, S, D)、v: (B, S, D_v)。値は[0,1]。S_q==S_k==S。

    Returns:
        (B, S, D_v)
    """
    batch, s_q, _ = q.shape
    s_k = k.shape[1]
    assert s_q == s_k, f'causalはS_q==S_kの自己注意を想定: {s_q} vs {s_k}'
    neg_inf = torch.finfo(q.dtype).min
    scores = torch.full((batch, s_q, s_k), neg_inf, dtype=q.dtype, device=q.device)
    for b in range(batch):
        for i in range(s_q):
            for j in range(i + 1):  # j<=iのみスコアを計算(未来は-infのまま)
                xnor = q[b, i] * k[b, j] + (1.0 - q[b, i]) * (1.0 - k[b, j])
                scores[b, i, j] = xnor.sum()
    attn = F.softmax(scores / tau, dim=-1)
    out = torch.zeros(batch, s_q, v.shape[-1], dtype=v.dtype, device=v.device)
    for b in range(batch):
        for i in range(s_q):
            for j in range(i + 1):
                out[b, i] = out[b, i] + attn[b, i, j] * v[b, j]
    return out


def reference_grads(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
    parameterization: str,
    grad_output: torch.Tensor,
    gate_table: torch.Tensor | None = None,
    tau: float = 1.0,
    estimator: str = 'sigmoid',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(出力, grad_x, grad_weights) をautogradで一括計算するヘルパ。

    カスタムFunctionのbackwardとの照合テスト用。
    estimator は independent(=論文IWP)の推定器ρの選択に用いる。
    """
    x = x.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    y = reference_lut_forward(
        x, weights, indices, n_inputs, parameterization,
        gate_table=gate_table, tau=tau, estimator=estimator,
    )
    grad_x, grad_w = torch.autograd.grad(y, (x, weights), grad_outputs=grad_output)
    return y.detach(), grad_x, grad_w


def reference_quantize_ternary(
    weight: torch.Tensor,
    per: str = 'channel',
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BitNet b1.58式 absmean ternary量子化の参照実装(素朴な直接計算)。

    量子化の意味論(ground truth):
        scale = mean(|W|)                          (absmean, per単位で平均)
        q     = clip(round(W / scale), -1, +1)     (∈ {-1, 0, +1})
        mask  = (q != 0)                           (残存接続の0/1マスク)
    W ≈ q · scale が量子化モデルの実効重み(dequant)。roundはtorch既定の
    round-half-to-even。BitNet b1.58 の weight_quant と同一(per='tensor')。

    per='channel' は各出力ニューロン(行)ごとにscaleを取り、'tensor'は
    全体で1つのscalarを使う。ゼロ重み行(scale=0)は eps でクランプする。

    Args:
        weight: (out, in) の実数重み
        per: 'channel'(行ごと) | 'tensor'(全体)
        eps: scale の下限クランプ

    Returns:
        (q, scale, mask):
            q     (out, in) float の {-1, 0, +1} 行列
            scale per='channel'なら(out, 1)、'tensor'ならスカラー(0次元)tensor
            mask  (out, in) bool の残存接続マスク(q != 0)

    Raises:
        ValueError: 未対応の per
    """
    if per == 'channel':
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # (out, 1)
    elif per == 'tensor':
        scale = weight.abs().mean().clamp_min(eps)                     # scalar
    else:
        raise ValueError(f"perは'channel'|'tensor'のみ: {per}")
    q = torch.clamp(torch.round(weight / scale), -1.0, 1.0)
    mask = q != 0
    return q, scale, mask


def reference_butterfly_forward(
    x: torch.Tensor,
    a_list: list,
    b_list: list,
    theta_list: list,
    pairs_list: list,
) -> torch.Tensor:
    """バタフライQAT因子列のforward参照実装(素朴ループ・純autograd)。

    各levelの意味論(ground truth):
        w_q = w + (quantize_ternary(w, per='tensor').q · scale − w).detach()
              (2W本の重み[a; b]を束ねて1つのabsmeanスケール、STE)
        y_i = HeavisideSTE(a_q[i]·x[u_i] + b_q[i]·x[v_i] + θ[i])
    各因子は行あたり非ゼロ≤2(バタフライ配線)なので、確定後は
    2入力閾値ゲート = 16ネイティブゲートへロスレス写像できる。

    Args:
        x: (B, W) の0/1入力
        a_list/b_list/theta_list: levelごとの (W,) パラメータ
        pairs_list: levelごとの (W, 2) 入力index(u, v)

    Returns:
        (B, W) の0/1出力(最終level)
    """
    from ..convert.teacher_models import HeavisideSTE

    h = x
    for a, b, theta, pairs in zip(
            a_list, b_list, theta_list, pairs_list, strict=True):
        w = torch.cat([a, b])
        q, scale, _ = reference_quantize_ternary(w, per='tensor')
        w_q = w + (q * scale - w).detach()
        a_q, b_q = w_q[:len(a)], w_q[len(a):]
        cols = []
        for i in range(h.shape[1]):
            u, v = int(pairs[i][0]), int(pairs[i][1])
            pre = a_q[i] * h[:, u] + b_q[i] * h[:, v] + theta[i]
            cols.append(HeavisideSTE.apply(pre))
        h = torch.stack(cols, dim=1)
    return h


def reference_conv_learnable_mapping_forward(
    candidate: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """LutConv2d用Learnable Mappingのforward参照実装。

    DWN論文§3.2 Learnable Mapping（griffith_logic/layers/dwn.py の
    LearnableMappingFunction）を、ツリー（出力チャンネル）次元oでバッチ化
    したもの。denseの I(W,x)_i = x_{argmax(W[:,i])} を、ツリーごとに独立の
    重み列 weights[o,:,i] で選択する:

        leaves[b,o,i] = candidate[b, o, argmax_j weights[o,j,i]]

    素朴な三重ループ（o, i, argmax）で直接計算する（効率は不要、明瞭さ優先）。

    Args:
        candidate: (batch, out_channels, n_candidates) ツリーごとの受容野候補
        weights: (out_channels, n_candidates, n_leaves) ツリーごとの選択重み

    Returns:
        (batch, out_channels, n_leaves)
    """
    out_channels, _n_candidates, n_leaves = weights.shape
    batch = candidate.shape[0]
    out = torch.zeros(
        batch, out_channels, n_leaves,
        dtype=candidate.dtype, device=candidate.device,
    )
    for o in range(out_channels):
        for i in range(n_leaves):
            j = int(weights[o, :, i].argmax().item())
            out[:, o, i] = candidate[:, o, j]
    return out


def reference_conv_learnable_mapping_grads(
    candidate: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conv用Learnable Mappingの(forward, grad_candidate, grad_weights)参照実装。

    forwardは argmax によるハード選択で不連続（gradcheck不可）のため、
    backwardはdense版LearnableMappingFunctionの式(公式実装torch_dwn/mapping.py
    準拠)をツリー次元oでバッチ化した閉形式で直接計算する（素朴な明示式、
    autograd不使用）:

        ∂L/∂weights[o,j,i] = Σ_b (2·candidate[b,o,j] − 1) · G[b,o,i]
        ∂L/∂candidate[b,o,j] = Σ_i G[b,o,i] · softmax_{dim=j}(weights[o,:,i]/τ)[j]

    Args:
        candidate: (batch, out_channels, n_candidates)
        weights: (out_channels, n_candidates, n_leaves)
        tau: backwardのsoftmax温度
        grad_output: (batch, out_channels, n_leaves) 上流勾配G

    Returns:
        (forward出力, grad_candidate, grad_weights)
    """
    y = reference_conv_learnable_mapping_forward(candidate, weights)
    soft = torch.softmax(weights / tau, dim=1)  # (out_channels, n_candidates, n_leaves)
    grad_candidate = torch.einsum('boi,ori->bor', grad_output, soft)
    grad_weights = torch.einsum('bor,boi->ori', 2.0 * candidate - 1.0, grad_output)
    return y, grad_candidate, grad_weights


def reference_conv_learnable_wiring_forward(
    x: torch.Tensor,
    mapping_weights: torch.Tensor,
    mapping_indices: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_table: torch.Tensor,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    n_inputs: int,
    tree_depth: int,
) -> torch.Tensor:
    """LutConv2d(wiring='learnable', parameterization='softmax16g')の
    forward参照実装(素朴unfold + Learnable Mapping + soft木)。

    パイプライン: F.unfold → 候補受容野をgather →
    reference_conv_learnable_mapping_forward（ツリーごとargmax選択）→
    葉（leaves）→ softmax16g soft木（実装_tree_forward_trainと同式）でLUT評価。

    Args:
        x: (B, C, H, W) 入力、値は[0,1]
        mapping_weights: (out_channels, n_candidates, n_leaves)
        mapping_indices: (out_channels, n_candidates) unfold後の候補フラットindex
        gate_weights: (out_channels, n_internal, 16) softmax16gロジット
        gate_table: (16, 4) 真理値表
        kernel_size, stride, padding: (h, w)タプル
        n_inputs: ツリーのゲート入力数（softmax16gはn_inputs=2専用）
        tree_depth: ツリー深さ

    Returns:
        (B, out_channels, H_out, W_out)
    """
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    B, _C, H, W = x.shape
    patches = F.unfold(x, (kH, kW), padding=(pH, pW), stride=(sH, sW))
    L = patches.shape[2]
    out_channels = mapping_weights.shape[0]

    candidate = patches[:, mapping_indices]  # (B, out_channels, n_candidates, L)
    candidate = candidate.permute(0, 3, 1, 2).reshape(B * L, out_channels, -1)
    leaves = reference_conv_learnable_mapping_forward(candidate, mapping_weights)

    current = leaves
    gate_offset = 0
    for _level in range(tree_depth):
        n_groups = current.shape[-1] // n_inputs
        grouped = current[..., :n_groups * n_inputs].reshape(
            *current.shape[:-1], n_groups, n_inputs,
        )
        level_weights = gate_weights[:, gate_offset:gate_offset + n_groups, :]
        basis = reference_basis(grouped, n_inputs)
        s = F.softmax(level_weights, dim=-1)
        w = s @ gate_table.to(s.dtype)
        current = (basis * w.unsqueeze(0)).sum(dim=-1)
        gate_offset += n_groups
    out = current.squeeze(-1)

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    return out.reshape(B, L, out_channels).permute(0, 2, 1).reshape(
        B, out_channels, H_out, W_out,
    )


def reference_causal_window(
    x: torch.Tensor,
    kernel_size: int,
    causal: bool = True,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Toeplitz型トークン混合の「窓gather」参照実装。

    各時刻 t について、長さ kernel_size の時間窓を最終次元へ並べる。

        causal=True : 窓は [t-kernel_size+1, ..., t](最後の要素が現在時刻)
        causal=False: 中心窓 [t-(kernel_size-1)//2, ..., t+kernel_size//2]

    範囲外の位置は pad_value で埋める。素朴な二重ループ(時刻×窓内オフセット)で
    書き下しており、``unfold``/``F.pad`` を使わない(ground truthは素朴さが価値)。
    純autograd演算のみなので x について微分可能。

    Args:
        x: (B, T, C) の実数テンソル
        kernel_size: 窓幅(1以上)
        causal: 因果窓(True)か中心窓(False)か
        pad_value: 範囲外を埋める値

    Returns:
        (B, T, C, kernel_size) のテンソル

    Raises:
        ValueError: xが3次元でない、または kernel_size < 1
    """
    if x.dim() != 3:
        raise ValueError(f'xは(B, T, C)の3次元テンソル: {tuple(x.shape)}')
    if kernel_size < 1:
        raise ValueError(f'kernel_sizeは1以上: {kernel_size}')

    seq_len = x.shape[1]
    # 窓の左端オフセット(t からいくつ過去へ遡るか)
    left = kernel_size - 1 if causal else (kernel_size - 1) // 2

    per_time = []
    for t in range(seq_len):
        per_offset = []
        for j in range(kernel_size):
            src = t - left + j
            if 0 <= src < seq_len:
                per_offset.append(x[:, src, :])
            else:
                per_offset.append(torch.full_like(x[:, 0, :], pad_value))
        per_time.append(torch.stack(per_offset, dim=-1))  # (B, C, K)
    return torch.stack(per_time, dim=1)  # (B, T, C, K)


def reference_soft_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """soft AND ゲート: a*b。a,b∈{0,1} で厳密に論理ANDと一致する。"""
    return a * b
