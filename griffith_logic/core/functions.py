"""LUT層のカスタムautograd Function群(全パラメータ化)。"""

import functools

import torch
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from ..cuda import get_cuda_module
from ..cuda.transpose import fast_transpose
from .basis import _build_wh_entry_matrix
from .estimators import estimator_rho, estimator_rho_prime

# autocast(混合精度)対応。
#
# 融合CUDAカーネル版のautograd Function(Cuda*)のforwardに _custom_fwd_follow_autocast、
# backwardに custom_bwd(device_type='cuda') を付与する。PyTorch公式の
# 「master weightはfp32のまま、reduced precisionで計算する」パターン
# (https://pytorch.org/docs/stable/notes/amp_examples.html#custom-autograd-functions)
# に従うが、キャスト先は固定せず**呼び出し時点のautocastのdtypeに合わせる**:
#   - torch.autocast(device_type='cuda', dtype=D) が有効な区間で呼ばれた場合のみ、
#     forward直前にx/weights等の浮動小数点CUDAテンソルをDへキャストし、
#     autocastを切った状態でforward本体を実行する(D=bfloat16ならbf16、
#     D=float16ならfp16でカーネルが動く)。int64のindices/rev_*やNoneの
#     gate_tableは影響を受けず、float64テンソルもキャスト対象外
#     (gradcheck等のfloat64経路は不変)。nn.Parameter自体はfp32のまま
#     Optimizerに握られ、キャスト後の値はforward内のローカル変数としてのみDになる。
#   - autocastが無効な通常経路(既定)では完全にno-opで、既存のfp32/fp64呼び出しの
#     数値・分岐は一切変化しない。
# Python参照実装(LutFunction等、core/reference.py含む)は対象外。


def _cuda_autocast_dtype() -> torch.dtype:
    """CUDA autocastの現在のdtype(torchのバージョン差を吸収)。"""
    get_dtype = getattr(torch, 'get_autocast_dtype', None)
    if get_dtype is not None:
        return get_dtype('cuda')
    return torch.get_autocast_gpu_dtype()  # 古いtorch向け


def _cuda_autocast_enabled() -> bool:
    """CUDA autocastが有効か(torchのバージョン差を吸収)。"""
    try:
        return torch.is_autocast_enabled('cuda')
    except TypeError:  # 古いtorchは引数なし(=CUDA)のみ
        return torch.is_autocast_enabled()


def _custom_fwd_follow_autocast(fwd):
    """custom_fwd(cast_inputs=<現在のautocast dtype>) を呼び出しごとに適用する。

    custom_fwd の cast_inputs はデコレート時に固定されるため、autocastが有効な
    呼び出しでのみ、その時点のdtypeでラップし直して実行する(無効時は
    cast_inputs なしの custom_fwd = 素通し)。
    """
    plain = custom_fwd(fwd, device_type='cuda')

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if _cuda_autocast_enabled():
            dtype = _cuda_autocast_dtype()
            return custom_fwd(fwd, device_type='cuda', cast_inputs=dtype)(*args, **kwargs)
        return plain(*args, **kwargs)

    return decorate_fwd


_CUSTOM_BWD = custom_bwd(device_type='cuda')


class LutFunction(torch.autograd.Function):
    """
    LUT層のカスタムautograd（softmaxパラメータ化）。

    N=2ではdifflogic互換の16ゲートモードを使用:
    - weights: (out_dim, 16) — 16種Boolean関数に対するsoftmax
    - gate_table: (16, 4) — 各ゲートの真理値表（butterfly基底順）
    - effective_w = softmax(weights) @ gate_table で4基底の有効重みを計算

    N>2ではgate_table=Noneで従来の基底直接softmax。

    メリット:
    - butterfly中間状態のみ保存（autogradデフォルトより省メモリ）
    - softmax backward を解析的に計算（グラフ構築なし）
    - butterfly逆伝播で数値的に安定な勾配計算（除算なし）
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, gate_table):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. Butterfly basis（中間状態を保存）
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j+1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # 3. Softmax重み → 有効基底重み
        s = F.softmax(weights, dim=-1)
        if gate_table is not None:
            # 16ゲートモード: (out_dim, 16) @ (16, 4) = (out_dim, 4)
            w = s @ gate_table
        else:
            w = s

        # 4. 出力
        output = torch.einsum('boe,oe->bo', basis, w)

        ctx.save_for_backward(gathered, basis, s, w, indices, *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]
        ctx.gate_table = gate_table

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        basis = saved[1]
        s = saved[2]      # softmax(weights)
        w = saved[3]      # effective weights (= s @ gate_table or s)
        indices = saved[4]
        intermediates = list(saved[5:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        gate_table = ctx.gate_table
        batch, out_dim = grad_output.shape

        # grad w.r.t. effective weights
        grad_eff = torch.einsum('bo,boe->oe', grad_output, basis)

        if gate_table is not None:
            # 16ゲートモード: chain rule through gate_table
            # grad_s = grad_eff @ gate_table.T → (out_dim, 16)
            grad_s = grad_eff @ gate_table.T
        else:
            grad_s = grad_eff

        # softmax Jacobian: grad_weights = s * (grad_s - dot(grad_s, s))
        dot = (grad_s * s).sum(dim=-1, keepdim=True)
        grad_weights = s * (grad_s - dot)

        # grad_x
        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_basis = grad_output ⊗ w（effective weights）
            grad_basis = torch.einsum('bo,oe->boe', grad_output, w)

            # butterfly逆伝播: 各入力jの勾配を安定に計算
            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                xj = gathered[:, :, j:j+1]

                # d(output)/d(x_j) = (grad_hi - grad_lo) · intermediates[j]
                grad_gathered[:, :, j] = ((grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)

                # 前段へ伝播: grad_old = grad_lo*(1-xj) + grad_hi*xj
                grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

            # scatter_addでgathered → x へ勾配を集約
            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None, None


class LutFunctionHardST(torch.autograd.Function):
    """
    Hard-ST（Straight-Through）によるLUT層のカスタムautograd。

    CAGE論文 (Kim, 2026) の "forward alignment" を実装:
    - Forward: argmax による hard selection（推論と同一の計算）
    - Backward: softmax(weights / τ_b) によるSTE勾配近似

    Selection gap = 0 を保証し、収束速度を大幅に向上させる。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, gate_table, tau_b):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. Butterfly basis（中間状態を保存）
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j+1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # 3. Hard selection: argmax → Boolean論理ゲート1つを選択
        if gate_table is not None:
            gate_idx = weights.argmax(dim=-1)
            w = gate_table[gate_idx]
        else:
            idx = weights.argmax(dim=-1)
            w = F.one_hot(idx, weights.shape[-1]).to(basis.dtype)

        # 4. 出力
        output = torch.einsum('boe,oe->bo', basis, w)

        ctx.save_for_backward(gathered, basis, weights, indices, *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]
        ctx.gate_table = gate_table
        ctx.tau_b = tau_b

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        basis = saved[1]
        weights = saved[2]
        indices = saved[3]
        intermediates = list(saved[4:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        gate_table = ctx.gate_table
        tau_b = ctx.tau_b
        batch, out_dim = grad_output.shape

        # STE逆伝播: softmax(weights / τ_b) で勾配近似
        s = F.softmax(weights / tau_b, dim=-1)
        if gate_table is not None:
            w_eff = s @ gate_table
        else:
            w_eff = s

        # grad w.r.t. effective weights
        grad_eff = torch.einsum('bo,boe->oe', grad_output, basis)

        if gate_table is not None:
            grad_s = grad_eff @ gate_table.T
        else:
            grad_s = grad_eff

        # softmax Jacobian（温度τ_b込み）
        dot = (grad_s * s).sum(dim=-1, keepdim=True)
        grad_weights = s * (grad_s - dot) / tau_b

        # grad_x
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_basis = torch.einsum('bo,oe->boe', grad_output, w_eff)

            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                xj = gathered[:, :, j:j+1]
                grad_gathered[:, :, j] = ((grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)
                grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None, None, None


class LutFunctionWARPHardST(torch.autograd.Function):
    """
    WARPパラメータ化用のHard-ST/Gumbel-ST(閾値意味論)。

    従来のLutFunctionHardSTはargmax over weightsで単一エントリを選ぶため、
    WH係数θに適用するとwarpの離散化(1[θ·φ(s)>0])と全く異なる関数になる
    (単一minterm崩壊)。本Functionはエントリロジット経由で閾値化する:

    forward:
        L = θ @ H  (H[i,e] = (-1)^popcount(i&e)、エントリロジット)
        z = Σ_e basis_e(x) · (L + noise)[o,e]  (ロジットの多重線形拡張)
        y = 1[z > 0]
        binary入力では z = (L+noise)[o, addr(x)] なので per-entryのhard表
        サンプルと厳密一致。連続入力でも出力はbinaryで、noise=0のとき
        z = θ·φ(s(x)) となりevalの離散化 1[θ·φ(s)>0] と**入力の連続性に
        よらず**厳密一致する(真のselection gap 0)。
        【改訂の経緯】旧版は「hard表w=1[L>0]の多重線形拡張 Σ basis·w」を
        出力しており、binary入力では同値だが連続入力(RDDLGN等の埋め込み
        スタック)では連続出力となりevalの閾値化と乖離、再帰で増幅されて
        hard崩壊を起こした(charlm n=4でsoft2.87/hard11.06 BPC)。
        noise=Noneでhard_st、logisticノイズ×gumbel_scaleでgumbel_st。

    backward(STE):
        ỹ = σ(z / τ_b) によるsoft関数の勾配と厳密一致。
        θへは ∂z/∂L = basis、∂L/∂θ = H^T で連鎖。noiseは定数扱い。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, tau_b, noise):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. 確率的基底butterfly(中間状態を保存)
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j+1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # 3. エントリロジット(+ノイズ)
        h = _build_wh_entry_matrix(n_inputs).to(
            device=weights.device, dtype=weights.dtype)
        logit = weights @ h  # (out_dim, 2^n)
        if noise is not None:
            logit = logit + noise

        # 4. ロジットの多重線形拡張 → 閾値化
        z = torch.einsum('boe,oe->bo', basis, logit)
        output = (z > 0).to(basis.dtype)

        ctx.save_for_backward(gathered, basis, logit, z, h, indices,
                              *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]
        ctx.tau_b = tau_b

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        basis = saved[1]
        logit = saved[2]
        z = saved[3]
        h = saved[4]
        indices = saved[5]
        intermediates = list(saved[6:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        tau_b = ctx.tau_b
        batch, out_dim = grad_output.shape

        # STE: ỹ = σ(z/τ_b) をsoft代理としてautograd勾配を解析計算
        sig = torch.sigmoid(z / tau_b)
        grad_z = grad_output * sig * (1.0 - sig) / tau_b  # (batch, out)

        # ∂z/∂L = basis → grad_L = Σ_batch grad_z × basis、∂L/∂θ = H^T
        grad_logit = torch.einsum('bo,boe->oe', grad_z, basis)
        grad_weights = grad_logit @ h.T

        # grad_x: z = Σ basis_e·L_e のbutterfly逆伝播(表=ロジット)
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_basis = torch.einsum('bo,oe->boe', grad_z, logit)

            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                xj = gathered[:, :, j:j+1]
                grad_gathered[:, :, j] = (
                    (grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)
                grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim,
                                 device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None, None, None


class LutFunctionIndependent(torch.autograd.Function):
    """
    LUT層のカスタムautograd（独立確率パラメータ化 / BinaryBrain方式）。

    softmax（競合）ではなく推定器ρ（独立）で各エントリの重みを計算する。
    各エントリの勾配が独立に計算され、softmax Jacobianが不要。

    これは論文 arXiv:2510.03250 の IWP (Eq.11) と数学的に同一:
        y = Σ_k ρ(Ω_k) · e_k(x)
    estimator='sigmoid'（既定）で ρ=σ、'sin' で ρ=0.5+0.5·sin（論文§C.1.1）。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, estimator='sigmoid'):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. Butterfly basis（中間状態を保存）
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j+1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # 3. 推定器ρによる重み（各エントリ独立に[0,1]）
        w = estimator_rho(weights, estimator)

        # 4. 出力
        output = torch.einsum('boe,oe->bo', basis, w)

        ctx.save_for_backward(
            gathered, basis, w, weights, indices, *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]
        ctx.estimator = estimator

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        basis = saved[1]
        w = saved[2]        # ρ(weights)
        weights = saved[3]  # 生パラメータΩ（ρ'の評価に必要）
        indices = saved[4]
        intermediates = list(saved[5:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        estimator = ctx.estimator
        batch, out_dim = grad_output.shape

        # grad_weights: 推定器の逆伝播を解析的に計算
        # output = einsum(basis, ρ(Ω)) → dρ/dΩ を掛ける
        grad_w_sig = torch.einsum('bo,boe->oe', grad_output, basis)
        grad_weights = grad_w_sig * estimator_rho_prime(weights, estimator)

        # grad_x: butterfly逆伝播（softmax版と同一）
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_basis = torch.einsum('bo,oe->boe', grad_output, w)

            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                xj = gathered[:, :, j:j+1]

                grad_gathered[:, :, j] = ((grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)
                grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        # 末尾のNoneはestimator引数ぶん
        return grad_x, grad_weights, None, None, None


class LutFunctionRaw(torch.autograd.Function):
    """
    LUT層のカスタムautograd（rawパラメータ化 / BinaryBrain方式）。

    重みをそのまま[0,1]の値として使用（sigmoid変換なし）。
    BinaryBrainのStochasticOperation_Lut_Forward/Backwardと等価。
    optimizer step後にclamp(0,1)で値域を維持する。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. Butterfly basis（中間状態を保存）
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j+1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # 3. 重みをそのまま使用（変換なし）
        w = weights

        # 4. 出力
        output = torch.einsum('boe,oe->bo', basis, w)

        ctx.save_for_backward(gathered, basis, w, indices, *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        basis = saved[1]
        w = saved[2]
        indices = saved[3]
        intermediates = list(saved[4:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        batch, out_dim = grad_output.shape

        # grad_weights: 直接勾配（sigmoid Jacobian不要）
        grad_weights = torch.einsum('bo,boe->oe', grad_output, basis)

        # grad_x: butterfly逆伝播
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_basis = torch.einsum('bo,oe->boe', grad_output, w)

            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                xj = gathered[:, :, j:j+1]

                grad_gathered[:, :, j] = ((grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)
                grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None


class LutFunctionIWP(torch.autograd.Function):
    """
    LUT層のカスタムautograd（入力ごとに1パラメータの「リテラル積」方式）。

    各入力ごとに独立なパラメータ α_j を持ち、出力を積の形で計算:
        output = Π_{j=0}^{N-1} [p_j × x_j + (1 - p_j) × (1 - x_j)]
        where p_j = sigmoid(α_j)

    Butterfly基底関数が不要で、N回の乗算のみで計算完了。
    パラメータ数: (out_dim, n_inputs) — softmax/independentの (out_dim, 2^n_inputs) から大幅削減。

    **注意: これは論文 arXiv:2510.03250 の IWP (Input-Wise Parametrization) とは
    別物である**（名前が紛らわしいが数式が異なる）。本方式は丸めると
    「各入力のリテラル(x_j または 1-x_j)の積」= 単一mintermにしかならず、
    表現力が厳しく制限される。2入力では16ゲート中4種（minterm系: AND,
    A∧¬B, ¬A∧B, ¬A∧¬B）しか表現できず、**素通し G(A,B)=A も XOR も
    表現不能**である。

    論文の IWP に相当するのは `parameterization='independent'` の方で、
    そちらは 2^N エントリの真理値表を直接パラメータ化した
        g = Σ_k ρ(Ω_k) · e_k,  e_k = Π_j (x_j if bit(k,j)=1 else 1-x_j)   (Eq.11)
    であり、2入力16ゲートすべてを表現できる（論文IWPと数学的に厳密同一）。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. 入力ごとの確率
        p = torch.sigmoid(weights)  # (out_dim, n_inputs)

        # 3. 各入力の「一致度」: p_j*x_j + (1-p_j)*(1-x_j)
        factors = p.unsqueeze(0) * gathered + (1.0 - p.unsqueeze(0)) * (1.0 - gathered)
        # factors: (batch, out_dim, n_inputs)

        # 4. 積で出力
        output = factors.prod(dim=-1)  # (batch, out_dim)

        ctx.save_for_backward(gathered, p, factors, indices)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        gathered, p, factors, indices = ctx.saved_tensors
        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        batch, out_dim = grad_output.shape

        # Π_{i≠j} factor_i をprefix/suffix積（exclusive cumprod）で計算。
        # output/factor_j 方式は factor_j≈0（x_jが二値端かつp_j飽和）で
        # 勾配が数学的に破綻するため除算を使わない
        prefix = torch.ones_like(factors)
        suffix = torch.ones_like(factors)
        for j in range(1, n_inputs):
            prefix[..., j] = prefix[..., j - 1] * factors[..., j - 1]
            suffix[..., n_inputs - 1 - j] = (
                suffix[..., n_inputs - j] * factors[..., n_inputs - j]
            )
        partial = prefix * suffix  # (batch, out_dim, n_inputs)

        grad_out_expanded = grad_output.unsqueeze(-1)  # (batch, out_dim, 1)

        # grad_weights (α_j):
        # ∂output/∂α_j = Π_{i≠j} factor_i × (2x_j - 1) × p_j × (1 - p_j)
        d_factor_dp = 2.0 * gathered - 1.0  # (batch, out_dim, n_inputs)
        d_p_dalpha = p * (1.0 - p)          # (out_dim, n_inputs)
        grad_weights = (
            grad_out_expanded * partial * d_factor_dp * d_p_dalpha.unsqueeze(0)
        ).sum(dim=0)

        # grad_x
        grad_x = None
        if ctx.needs_input_grad[0]:
            # ∂factor_j/∂x_j = 2p_j - 1
            d_factor_dx = 2.0 * p.unsqueeze(0) - 1.0  # (1, out_dim, n_inputs)
            grad_gathered = grad_out_expanded * partial * d_factor_dx
            # (batch, out_dim, n_inputs)

            # scatter_addでgathered → x へ勾配を集約
            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None


class LutFunctionWARP(torch.autograd.Function):
    """
    LUT層のカスタムautograd（WARPパラメータ化）。

    Walsh-Hadamard変換に基づく微分可能Boolean関数学習。
    Gerlach et al., "WARP Logic Neural Networks", 2025 (arXiv:2602.03527)

    - weights: (out_dim, 2^n_inputs) — WH係数θ（制約なし実数）
    - 出力 = σ(Σ θ_i φ_i^W(1-2x) / τ)

    特性:
    - パラメータ効率: 最小パラメータ数（2^n個）で任意のBoolean関数を正確に表現
    - 正確な勾配: 近似勾配不要（DWNのfinite differenceより安定）
    - 制約なし最適化: softmaxの確率simplex制約やIWPの[0,1]制約が不要
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, tau, gumbel_training):
        batch = x.shape[0]
        out_dim = indices.shape[1]

        # 1. 入力収集
        gathered = x[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # 2. WH basis butterfly（中間状態を保存）
        s = 1.0 - 2.0 * gathered  # [0,1] → [+1,-1]
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            sj = s[:, :, j:j+1]
            basis = torch.cat([basis, basis * sj], dim=-1)

        # 3. ロジット = (θ · WH基底) / τ
        logit = (basis * weights.unsqueeze(0)).sum(-1) / tau

        # 4. Gumbel-Sigmoid smoothing（訓練時のオプション）
        if gumbel_training:
            u = torch.rand(2, *logit.shape, device=logit.device, dtype=logit.dtype)
            u = u.clamp(1e-6, 1.0 - 1e-6)
            # Gumbel-Sigmoid: logistic noise = log(u/(1-u))
            logit = logit + u[0].log() - (1.0 - u[0]).log() - u[1].log() + (1.0 - u[1]).log()

        # 5. Sigmoid出力
        output = torch.sigmoid(logit)

        ctx.save_for_backward(gathered, s, basis, weights, output, indices, *intermediates)
        ctx.n_inputs = n_inputs
        ctx.in_dim = x.shape[1]
        ctx.tau = tau

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        gathered = saved[0]
        s = saved[1]
        basis = saved[2]
        weights = saved[3]
        output = saved[4]
        indices = saved[5]
        intermediates = list(saved[6:])

        n_inputs = ctx.n_inputs
        in_dim = ctx.in_dim
        tau = ctx.tau
        batch, out_dim = grad_output.shape

        # sigmoid微分: ∂L/∂logit = grad_output × σ'(logit)
        sig_deriv = output * (1.0 - output)
        grad_logit = grad_output * sig_deriv  # (batch, out_dim)

        # ∂L/∂weights = Σ_batch (∂L/∂logit × WH基底) / τ
        grad_weights = (grad_logit.unsqueeze(-1) * basis).sum(dim=0) / tau

        # ∂L/∂WH基底 → WH butterfly逆伝播
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_basis = grad_logit.unsqueeze(-1) * weights.unsqueeze(0) / tau

            grad_gathered = torch.zeros_like(gathered)
            grad_cur = grad_basis
            for j in reversed(range(n_inputs)):
                half = grad_cur.shape[-1] // 2
                grad_lo = grad_cur[..., :half]
                grad_hi = grad_cur[..., half:]

                sj = s[:, :, j:j+1]

                # s_jにかかる勾配（intermediates[j]はstep j前のbasis）
                grad_sj = (grad_hi * intermediates[j]).sum(dim=-1)
                # s_j = 1 - 2x_j → ∂s_j/∂x_j = -2
                grad_gathered[:, :, j] = grad_sj * (-2.0)

                # 前段への伝播
                grad_cur = grad_lo + grad_hi * sj

            # scatter_addでgathered → x へ勾配を集約
            flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
            flat_grad = grad_gathered.reshape(batch, -1)
            grad_x = torch.zeros(batch, in_dim, device=gathered.device, dtype=gathered.dtype)
            grad_x.scatter_add_(1, flat_indices, flat_grad)

        return grad_x, grad_weights, None, None, None, None


class CudaLutFunction(torch.autograd.Function):
    """
    LUT層のCUDAカーネル版autograd（softmaxパラメータ化）。

    butterfly基底をCUDAスレッドごとにレジスタ上で計算し、
    中間テンソルの保存を完全に排除してメモリ使用量を大幅削減。
    difflogicのCUDAカーネル設計を N=2,4,6 に一般化。

    N=2では16ゲートモード（gate_table != None）でdifflogicと完全互換。
    CUDAカーネルは変更不要: effective_w (out_dim, 4) を渡すだけ。
    """

    @staticmethod
    @_custom_fwd_follow_autocast
    def forward(ctx, x, weights, indices, n_inputs, rev_starts, rev_flat_y,
                rev_flat_slot, gate_table):
        cuda = get_cuda_module()

        # softmax重みを事前計算 → 有効基底重みに変換
        s = F.softmax(weights, dim=-1)
        if gate_table is not None:
            # 16ゲートモード: (out_dim, 16) @ (16, 4) = (out_dim, 4)
            w = (s @ gate_table).contiguous()
        else:
            w = s.contiguous()

        # (batch, in_dim) → (in_dim, batch) に転置（CUDAメモリコアレッシング最適化）
        # 連鎖内側(前段がLUT層)では x.t() が既にcontiguousでコピーは発生しない。
        # 実コピーが必要な境界のみ Triton のタイル転置が使われる
        x_t = fast_transpose(x)
        indices_c = indices.contiguous()

        # CUDAカーネルで forward: 出力は (out_dim, batch)
        y_t = cuda.forward(x_t, indices_c, w, n_inputs)

        # backward用に保存: x, weights, indices, 逆引きインデックスのみ
        # 【x_t を保存しない理由(実測して却下)】
        # backward の x.t().contiguous() を省くため x の代わりに x_t を保存する案を
        # 検討したが、x_t は forward〜backward の間ずっと生存するためピークメモリが
        # 増える。8層×dim4096×batch512 の訓練1ステップで実測 114.19MB → 124.68MB
        # (+9.2%)。速度よりメモリを優先し不採用(tests/l2_grad/test_cuda_memory_transpose.py)。
        ctx.save_for_backward(
            x, weights, indices_c, rev_starts, rev_flat_y, rev_flat_slot
        )
        ctx.n_inputs = n_inputs
        ctx.gate_table = gate_table

        return y_t.t()  # (batch, out_dim)

    @staticmethod
    @_CUSTOM_BWD
    def backward(ctx, grad_output):
        x, weights, indices, rev_starts, rev_flat_y, rev_flat_slot = ctx.saved_tensors
        n_inputs = ctx.n_inputs
        gate_table = ctx.gate_table
        cuda = get_cuda_module()

        # indices は forward で contiguous 化済み(ここでの再呼び出しは不要)
        # 前段がLUT層の場合 x.t() は既にcontiguousなので、fast_transpose は
        # コピーを伴わない(実コピーが起きるのは入力が非LUT層由来の最初の層のみ)
        x_t = fast_transpose(x)
        grad_y_t = fast_transpose(grad_output)

        # backward_w: CUDAカーネルでバッチリダクション
        # grad_eff[k] = Σ_batch grad_output * basis[k] — 4基底の勾配
        grad_eff = cuda.backward_w(x_t, indices, grad_y_t, n_inputs)

        s = F.softmax(weights, dim=-1)

        if gate_table is not None:
            # 16ゲートモード: gate_table経由で4→16に変換
            grad_s = grad_eff @ gate_table.T  # (out_dim, 16)
        else:
            grad_s = grad_eff

        # softmax jacobian: grad_weights = s * (grad_s - dot(grad_s, s))
        dot = (grad_s * s).sum(dim=-1, keepdim=True)
        grad_weights = s * (grad_s - dot)

        # backward_x: CUDAカーネル（逆引きインデックス使用）
        # effective_wを渡す（gate_table変換済み）
        grad_x = None
        if ctx.needs_input_grad[0]:
            if gate_table is not None:
                w = (s @ gate_table).contiguous()
            else:
                w = s.contiguous()
            grad_x_t = cuda.backward_x(
                x_t, indices, w, grad_y_t,
                rev_starts, rev_flat_y, rev_flat_slot, n_inputs,
            )
            grad_x = grad_x_t.t()

        return grad_x, grad_weights, None, None, None, None, None, None


class CudaLutFunctionIndependent(torch.autograd.Function):
    """
    LUT層のCUDAカーネル版autograd（独立確率パラメータ化 / BinaryBrain方式）。

    butterfly計算はsoftmax版と共通のCUDAカーネルを使用。
    重み変換のみsigmoidに変更。
    """

    @staticmethod
    @_custom_fwd_follow_autocast
    def forward(ctx, x, weights, indices, n_inputs, rev_starts, rev_flat_y, rev_flat_slot):
        cuda = get_cuda_module()

        w = torch.sigmoid(weights).contiguous()
        x_t = fast_transpose(x)
        indices_c = indices.contiguous()

        # forward カーネルは重み適用済みテンソルを受け取るため共通
        y_t = cuda.forward(x_t, indices_c, w, n_inputs)

        # x_t を保存しない理由は CudaLutFunction.forward のコメント参照(実測でメモリ増)
        ctx.save_for_backward(
            x, weights, indices_c, rev_starts, rev_flat_y, rev_flat_slot
        )
        ctx.n_inputs = n_inputs

        return y_t.t()

    @staticmethod
    @_CUSTOM_BWD
    def backward(ctx, grad_output):
        x, weights, indices, rev_starts, rev_flat_y, rev_flat_slot = ctx.saved_tensors
        n_inputs = ctx.n_inputs
        cuda = get_cuda_module()

        # indices は forward で contiguous 化済み
        x_t = fast_transpose(x)
        grad_y_t = fast_transpose(grad_output)

        # backward_w: CUDAカーネルで grad_w_sig = Σ_batch grad_output * basis[k]
        grad_w_sig = cuda.backward_w(x_t, indices, grad_y_t, n_inputs)

        # sigmoid jacobian: grad_weights = grad_w_sig * sigmoid(W) * (1 - sigmoid(W))
        w = torch.sigmoid(weights)
        grad_weights = grad_w_sig * w * (1.0 - w)

        # backward_x
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x_t = cuda.backward_x(
                x_t, indices, w.contiguous(), grad_y_t,
                rev_starts, rev_flat_y, rev_flat_slot, n_inputs,
            )
            grad_x = grad_x_t.t()

        return grad_x, grad_weights, None, None, None, None, None


class CudaLutFunctionIWP(torch.autograd.Function):
    """
    LUT層のCUDAカーネル版autograd（IWP: Input-Wise Parameterization / LDDLGN方式）。

    Python版(LutFunctionIWP)と同一の数式:
        output = Π_{j=0}^{N-1} [p_j x_j + (1-p_j)(1-x_j)],  p_j = sigmoid(α_j)

    softmax/independent版と異なり、butterfly基底(2^N要素)を一切使わない。
    カーネルはN要素のレジスタ配列のみで完結するため、N=6でも既存softmax/
    independentカーネルのような大きなレジスタ圧はかからない（設計ノート参照:
    lut_kernels.cu の「IWP用カーネル」セクション）。

    保存テンソル最小化(recompute方針): x, weights, indices, 逆引きインデックス
    のみを保存し、p=sigmoid(weights)やfactor類はbackwardで再計算する
    （CudaLutFunctionと同じ流儀）。
    """

    @staticmethod
    @_custom_fwd_follow_autocast
    def forward(ctx, x, weights, indices, n_inputs, rev_starts, rev_flat_y,
                rev_flat_slot):
        cuda = get_cuda_module()

        p = torch.sigmoid(weights).contiguous()
        x_t = x.t().contiguous()

        y_t = cuda.iwp_forward(x_t, indices.contiguous(), p, n_inputs)

        ctx.save_for_backward(x, weights, indices, rev_starts, rev_flat_y, rev_flat_slot)
        ctx.n_inputs = n_inputs

        return y_t.t()

    @staticmethod
    @_CUSTOM_BWD
    def backward(ctx, grad_output):
        x, weights, indices, rev_starts, rev_flat_y, rev_flat_slot = ctx.saved_tensors
        n_inputs = ctx.n_inputs
        cuda = get_cuda_module()

        p = torch.sigmoid(weights)
        p_c = p.contiguous()
        x_t = x.t().contiguous()
        grad_y_t = grad_output.t().contiguous()

        # backward_w: Σ_batch grad_y·partial_j·(2x_j-1)（sigmoid Jacobian未適用の生勾配）
        grad_alpha_raw = cuda.iwp_backward_w(x_t, indices.contiguous(), p_c, grad_y_t, n_inputs)

        # sigmoid jacobian: grad_weights = grad_alpha_raw * p * (1 - p)
        grad_weights = grad_alpha_raw * p * (1.0 - p)

        # backward_x: CUDAカーネル（逆引きインデックス使用、prefix/suffix積で除算なし）
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x_t = cuda.iwp_backward_x(
                x_t, indices.contiguous(), p_c, grad_y_t,
                rev_starts, rev_flat_y, rev_flat_slot, n_inputs,
            )
            grad_x = grad_x_t.t()

        return grad_x, grad_weights, None, None, None, None, None


class CudaLutFunctionWARP(torch.autograd.Function):
    """
    LUT層のCUDAカーネル版autograd（WARP: Walsh-Hadamard係数パラメータ化）。

    Python版(LutFunctionWARP)と同一の数式:
        z = Σ_e θ_e φ_e^W(1-2x) / τ  (+ noise)
        y = σ(z)

    対象は selection='soft' の学習時経路のみ。hard_st/gumbel_st
    （LutFunctionWARPHardST、エントリロジット閾値化の別意味論）は対象外
    （layers/lut.py 側で selection='soft' のときのみ本クラスへディスパッチする）。

    Gumbelノイズはこのクラスの外（呼び出し側 = layers/lut.py）で、Python版
    LutFunctionWARPと厳密に同一のtorch.randベース乱数生成により作られ、
    既にlogit加算用のテンソル（またはNone）として渡される。本クラスは
    「渡されたら加算するだけ」のカーネルを呼ぶのみで、乱数生成そのものには
    一切関与しない（RNG消費順の厳密一致がparityテストの前提）。

    保存テンソル最小化(recompute方針): x, θ(度数キャップ適用済み), indices,
    逆引きインデックスに加え出力y（sigmoid微分 y·(1-y) の計算に必要）を保存する。
    yは(batch, out_dim)のみで、basis/中間状態(O(2^N)倍の要素数)より遥かに
    小さいためCudaLutFunction/CudaLutFunctionIWPと同じ「省メモリ」の設計思想を
    保つ。noise自体はbackwardで不要（sigmoid微分はyから直接計算できるため
    保存しない）。

    backward_w/backward_xのCUDAカーネルはτに依存しない「生の」勾配を返し
    （WH-basis逆伝播がθ・grad_logitに対して線形であることを利用）、
    Python側で最後に/τするだけで数学的に同値にする
    （softmax/independentカーネルがsoftmax/sigmoid Jacobianを事後適用する
    のと同じ流儀）。bwd_xは逆引きCSR（rev_starts/rev_flat_y/rev_flat_slot）
    を使用する。
    """

    @staticmethod
    @_custom_fwd_follow_autocast
    def forward(ctx, x, weights, indices, n_inputs, tau, noise,
                rev_starts, rev_flat_y, rev_flat_slot):
        cuda = get_cuda_module()

        theta = weights.contiguous()
        x_t = x.t().contiguous()
        noise_t = noise.t().contiguous() if noise is not None else None

        y_t = cuda.warp_forward(x_t, indices.contiguous(), theta,
                                 1.0 / tau, noise_t, n_inputs)
        output = y_t.t()

        ctx.save_for_backward(x, weights, indices, output,
                              rev_starts, rev_flat_y, rev_flat_slot)
        ctx.n_inputs = n_inputs
        ctx.tau = tau

        return output

    @staticmethod
    @_CUSTOM_BWD
    def backward(ctx, grad_output):
        x, weights, indices, output, rev_starts, rev_flat_y, rev_flat_slot = \
            ctx.saved_tensors
        n_inputs = ctx.n_inputs
        tau = ctx.tau
        cuda = get_cuda_module()

        # sigmoid微分: dL/dz = grad_output × y × (1-y)
        # （noiseはzへの定数加算のみなので勾配計算に寄与しない。∂z/∂noise は
        #   backward対象外＝noiseへの勾配自体を返さない設計と整合）
        sig_deriv = output * (1.0 - output)
        grad_logit = (grad_output * sig_deriv).contiguous()

        x_t = x.t().contiguous()
        grad_logit_t = grad_logit.t().contiguous()
        theta = weights.contiguous()

        # backward_w: カーネルはτ非依存の生の勾配 Σ_batch grad_logit・basis を返す
        grad_theta_raw = cuda.warp_backward_w(
            x_t, indices.contiguous(), grad_logit_t, n_inputs)
        grad_weights = grad_theta_raw / tau

        # backward_x: CUDAカーネル（逆引きインデックス使用、τ未適用の生の∂z/∂x）
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x_t_raw = cuda.warp_backward_x(
                x_t, indices.contiguous(), theta, grad_logit_t,
                rev_starts, rev_flat_y, rev_flat_slot, n_inputs,
            )
            grad_x = grad_x_t_raw.t() / tau

        return grad_x, grad_weights, None, None, None, None, None, None, None
