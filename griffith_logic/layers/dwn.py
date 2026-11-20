"""DWN (Differentiable Weightless Neural Networks) 層。

Bacellar et al., "Differentiable Weightless Neural Networks", ICML 2024.
公式実装 github.com/alanbacellar/DWN (torch_dwn) と照合済み。

数式:
    アドレス関数: A(U, a) = U_{δ(a)}(U ∈ ℝ^{2^n}、δ: 二進列→整数。
                  スロットjがビットj、LSB=スロット0。二値化は「>0」)
    FD(テーブル勾配、one-hot):
        ∂A/∂U_i(U, a) = 1[i = δ(a)]
    EFD(入力勾配):
        ∂A/∂a_j(U, a) = Σ_{k∈{0,1}^n} (−1)^{1−k_j} · c(H(k,a,j)) · A(U, k)
        H(k,a,j) = ビットjを除外した k と a のハミング距離
        係数 c(H):
            'paper'    — 1/(H+1)                        (論文 式(3))
            'official' — α·β^H, α=0.5·0.75^{n−1}, β=1/3 (公式CUDAカーネル既定値)
    出力二値化(STE): y = 1[A(U, a) > 0] ∈ {0,1}、backwardは素通し
    Learnable Mapping (論文§3.2):
        forward:  I(W, x)_i = x_{argmax(W[:,i])}
        backward: ∂L/∂W = (2x−1)^T G,  ∂L/∂x = G · softmax_{dim=0}(W/τ)^T

公式実装で解決した仕様(テストに焼き込み済み):
    - EFDの A(U,k) はテーブルの生の実数値(二値化しない)
    - a は「>0」で二値化した入力アドレス
    - forward時のLUT出力は実数値のままで、後段のSTE(閾値0)で{0,1}化
    - テーブル初期化は U(−1,1)、訓練時は毎forwardで[−1,1]にclamp
"""

import torch
import torch.nn as nn

from ..core.connections import get_random_connections

# 公式実装の既定値(torch_dwn/lut_layer.py)
_OFFICIAL_ALPHA_BASE = 0.5   # α = 0.5 · 0.75^{n−1}
_OFFICIAL_ALPHA_DECAY = 0.75
_OFFICIAL_BETA = 0.25 / 0.75  # β = 1/3
_DEFAULT_LM_TAU = 0.001       # LearnableMappingのsoftmax温度


def _popcount(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """整数テンソルの立っているビット数(下位n_bitsのみ)"""
    shifts = torch.arange(n_bits, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).sum(dim=-1)


def build_efd_coefficients(
    n_inputs: int,
    mode: str = 'paper',
    alpha: float | None = None,
    beta: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """EFD係数テンソル C[a, j, k] = (−1)^{1−k_j} · c(H(k,a,j)) を事前計算する。

    ∂A/∂a_j(U, a) = Σ_k C[δ(a), j, k] · U_k と縮約するだけでEFD勾配が得られる。

    Args:
        n_inputs: LUT入力数 n
        mode: 'paper'(1/(H+1)) | 'official'(α·β^H)
        alpha, beta: mode='official'の係数(None時は公式既定値)
        dtype, device: 出力テンソルの型と配置

    Returns:
        (2^n, n, 2^n) 係数テンソル
    """
    if mode not in ('paper', 'official'):
        raise ValueError(f"mode は 'paper' | 'official' のいずれか: {mode}")

    size = 2 ** n_inputs
    a = torch.arange(size).view(size, 1, 1)
    j = torch.arange(n_inputs).view(1, n_inputs, 1)
    k = torch.arange(size).view(1, 1, size)

    mask = ~(1 << j)  # ビットj除外
    h = _popcount((a & mask) ^ (k & mask), n_inputs)  # (2^n, n, 2^n)

    if mode == 'paper':
        coef = 1.0 / (h.to(torch.float64) + 1.0)
    else:
        if alpha is None:
            alpha = _OFFICIAL_ALPHA_BASE * _OFFICIAL_ALPHA_DECAY ** (n_inputs - 1)
        if beta is None:
            beta = _OFFICIAL_BETA
        coef = alpha * torch.full_like(h, beta, dtype=torch.float64) ** h

    sign = torch.where((k >> j) & 1 == 1, 1.0, -1.0).to(torch.float64)
    return (sign * coef).to(dtype=dtype, device=device)


class DWNFunction(torch.autograd.Function):
    """DWNのLUT評価: forward=整数アドレスgather、backward=FD+EFD。

    forwardは不連続(gather)のため、勾配は有限差分近似:
        grad_weights[u, i] = Σ_b 1[i = δ(a_bu)] · G[b, u]         (FD)
        grad_gathered[b, u, j] = G[b, u] · Σ_k C[δ(a_bu), j, k] · U_u[k]  (EFD)
    """

    @staticmethod
    def forward(ctx, gathered: torch.Tensor, weights: torch.Tensor,
                c_efd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gathered: (batch, n_luts, n) 収集済み入力(「>0」で二値化される)
            weights: (n_luts, 2^n) LUTテーブル(実数値)
            c_efd: (2^n, n, 2^n) build_efd_coefficients()の係数

        Returns:
            (batch, n_luts) テーブルの実数値(二値化は後段のSTEが担当)
        """
        n_inputs = gathered.shape[-1]
        pow2 = (2 ** torch.arange(n_inputs, device=gathered.device)).view(1, 1, -1)
        addr = ((gathered > 0).long() * pow2).sum(dim=-1)  # (batch, n_luts)
        ctx.save_for_backward(weights, c_efd, addr)
        lut_idx = torch.arange(weights.shape[0], device=weights.device)
        return weights[lut_idx.unsqueeze(0), addr]

    @staticmethod
    def backward(ctx, grad_output):
        weights, c_efd, addr = ctx.saved_tensors
        n_luts = weights.shape[0]
        grad_gathered = grad_weights = None

        if ctx.needs_input_grad[0]:
            # EFD: T[u, a, j] = Σ_k C[a, j, k] · U_u[k] を作り、実アドレスでgather
            table = torch.einsum('ajk,uk->uaj', c_efd.to(weights.dtype), weights)
            lut_idx = torch.arange(n_luts, device=weights.device).unsqueeze(0)
            grad_gathered = grad_output.unsqueeze(-1) * table[lut_idx, addr]

        if ctx.needs_input_grad[1]:
            # FD: アクセスしたエントリへのone-hot散布
            grad_weights = torch.zeros_like(weights)
            lut_idx = torch.arange(n_luts, device=weights.device).unsqueeze(0)
            grad_weights.index_put_(
                (lut_idx.expand_as(addr), addr), grad_output, accumulate=True,
            )

        return grad_gathered, grad_weights, None


class BinarizeSTEFunction(torch.autograd.Function):
    """出力二値化STE: forward = 1[x > 0]、backward = 素通し(公式実装準拠)"""

    @staticmethod
    def forward(ctx, x):
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LearnableMappingFunction(torch.autograd.Function):
    """Learnable Mappingのforward/backward(論文§3.2、公式mapping.py準拠)。

    forward:  I(W, x)_i = x_{argmax(W[:,i])}(ハードargmax選択)
    backward: ∂L/∂W = (2x−1)^T G
              ∂L/∂x = G · softmax_{dim=0}(W/τ)^T
    """

    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=0)  # (out_features,)
        ctx.save_for_backward(x, weights)
        ctx.tau = tau
        return x[:, mapping]

    @staticmethod
    def backward(ctx, grad_output):
        x, weights = ctx.saved_tensors
        grad_weights = (2.0 * x - 1.0).T @ grad_output
        grad_x = grad_output @ torch.softmax(weights / ctx.tau, dim=0).T
        return grad_x, grad_weights, None


class LearnableMapping(nn.Module):
    """入力→LUTの接続を勾配で学習する層。

    訓練中は重み行列 W ∈ ℝ^{in×out} を保持し、forwardは既定で
    x_{argmax(W[:,i])} のハード選択(推論時のオーバーヘッドなし)。

    漸進配線ハード化(SparseLUT arXiv 2503.12829の2フェーズを適応):
        forward_mode='soft' では x @ softmax(W/tau_soft, dim=0) のsoft混合
        forwardになる(純autograd、選択の不連続ジャンプ無し)。
        ProgressiveWiringSchedulerがtau_softを幾何減衰させ、訓練後半で
        forward_mode='hard'(従来のargmax+STE)へ切り替える。
        hard選択の配線学習が崩壊したことへの対策レバー。
        evalは常にhard argmax(forward_modeは訓練時のみ参照)。

    Args:
        in_features: 入力ビット数 P
        out_features: 出力ビット数 Q(= LUT数 × LUT入力数)
        tau: backwardのsoftmax温度(公式既定値 0.001)
    """

    FORWARD_MODES = ('hard', 'soft')

    def __init__(self, in_features: int, out_features: int,
                 tau: float = _DEFAULT_LM_TAU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.forward_mode = 'hard'  # 'hard' | 'soft'(漸進ハード化のフェーズ1)
        self.tau_soft = 1.0         # softモードのforward温度(スケジューラが減衰)
        # 公式実装と同じ一様[0,1)初期化
        self.weights = nn.Parameter(torch.rand(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.forward_mode == 'soft':
            return x @ torch.softmax(self.weights / self.tau_soft, dim=0)
        return LearnableMappingFunction.apply(x, self.weights, self.tau)

    def extra_repr(self):
        mode = (f', mode={self.forward_mode}/τ={self.tau_soft:.3g}'
                if self.forward_mode != 'hard' else '')
        return (f'in={self.in_features}, out={self.out_features}, '
                f'tau={self.tau}{mode}')


class DWNLayer(nn.Module):
    """DWNのLUT層: 二値入力 → n入力LUT×n_luts → 二値出力。

    Args:
        in_bits: 入力ビット数
        n_luts: LUT(RAMノード)数 = 出力ビット数
        n_inputs: LUTあたりの入力数 n
        mapping: 'random'(固定ランダム接続) | 'learnable'(Learnable Mapping)
        efd: EFD係数 'paper'(1/(H+1)) | 'official'(α·β^H)
        alpha, beta: efd='official'の係数(None時は公式既定値)
        ste: 出力を{0,1}に二値化する(公式既定値True)
        clamp_luts: 訓練時に毎forwardでテーブルを[−1,1]へclamp(公式既定値True)
        lm_tau: Learnable Mappingのsoftmax温度
    """

    MAPPINGS = ('random', 'learnable')

    def __init__(
        self,
        in_bits: int,
        n_luts: int,
        n_inputs: int = 6,
        mapping: str = 'random',
        efd: str = 'paper',
        alpha: float | None = None,
        beta: float | None = None,
        ste: bool = True,
        clamp_luts: bool = True,
        lm_tau: float = _DEFAULT_LM_TAU,
        device: str = 'cpu',
    ):
        super().__init__()
        assert mapping in self.MAPPINGS, (
            f'mapping は {self.MAPPINGS} のいずれか: {mapping}'
        )
        self.in_bits = in_bits
        self.n_luts = n_luts
        self.n_inputs = n_inputs
        self.n_entries = 2 ** n_inputs
        self.mapping_mode = mapping
        self.efd_mode = efd
        self.ste = ste
        self.clamp_luts = clamp_luts

        # LUTテーブル: 一様U(−1,1)初期化(公式実装準拠)
        self.luts = nn.Parameter(
            torch.rand(n_luts, self.n_entries, device=device) * 2.0 - 1.0
        )
        self.register_buffer(
            'c_efd',
            build_efd_coefficients(n_inputs, mode=efd, alpha=alpha, beta=beta,
                                   device=device),
        )

        if mapping == 'learnable':
            self.learnable_mapping = LearnableMapping(
                in_bits, n_luts * n_inputs, tau=lm_tau,
            ).to(device)
        else:
            self.register_buffer(
                'indices',
                get_random_connections(in_bits, n_luts, n_inputs, device),
            )

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """入力を各LUTのnスロットへ収集: (batch, in_bits) → (batch, n_luts, n)"""
        if self.mapping_mode == 'learnable':
            mapped = self.learnable_mapping(x)  # (batch, n_luts*n)
            return mapped.reshape(x.shape[0], self.n_luts, self.n_inputs)
        # indices: (n_inputs, n_luts) — reference_gatherと同一規約
        return x[:, self.indices.T.reshape(-1)].reshape(
            x.shape[0], self.n_luts, self.n_inputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_bits) 二値入力 {0,1}

        Returns:
            (batch, n_luts) 二値出力 {0,1}(ste=False時はテーブル実数値)
        """
        if not torch.compiler.is_compiling():
            assert x.shape[-1] == self.in_bits, (x.shape[-1], self.in_bits)

        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1.0, 1.0)

        gathered = self._gather(x)
        y = DWNFunction.apply(gathered, self.luts, self.c_efd)
        if self.ste:
            y = BinarizeSTEFunction.apply(y)
        return y

    def extra_repr(self):
        return (
            f'in_bits={self.in_bits}, n_luts={self.n_luts}, '
            f'n_inputs={self.n_inputs}, entries={self.n_entries}, '
            f'mapping={self.mapping_mode}, efd={self.efd_mode}, ste={self.ste}'
        )
