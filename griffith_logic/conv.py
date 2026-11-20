
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .conv_samplers import SAMPLER_SPECS, build_leaf_selection, leaf_indices_flat
from .core.basis import _build_gate_table_n2
from .core.estimators import ESTIMATORS, estimator_rho
from .lut_layer import GradFactor


def _pair(x):
    """intをタプルに変換するヘルパー"""
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


class _BinarizeTableSTE(torch.autograd.Function):
    """真理値表の二値化STE: forward = 1[p > しきい値]、backward = 素通し。

    DWN(Bacellar et al., ICML 2024)が表を二値のまま学習するのと同じ発想を、
    木のゲートに適用する。閾値は呼び出し側の規約に合わせる
    ('independent_ste' は sigmoid 後なので 0.5、'independent_dwn' は
    生パラメータなので 0)。
    """

    @staticmethod
    def forward(ctx, p: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (p > threshold).to(p.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def _compute_basis(grouped: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """
    Butterfly分解による基底関数計算。

    Args:
        grouped: (..., n_inputs) 入力テンソル
        n_inputs: LUTの入力数

    Returns:
        (..., 2^n_inputs) 基底関数テンソル
    """
    basis = torch.ones(
        *grouped.shape[:-1], 1,
        device=grouped.device, dtype=grouped.dtype,
    )
    for j in range(n_inputs):
        xj = grouped[..., j:j+1]
        basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)
    return basis


def _generate_tree_connections(
    out_channels: int,
    n_leaves: int,
    in_channels: int,
    kernel_size_h: int,
    kernel_size_w: int,
    input_channels_per_tree: int,
    device: str = 'cpu',
):
    """
    ツリーのリーフ接続をランダム生成する（CDDLGN Eq. 3準拠）。

    各ツリー（出力チャンネル）に対し、リーフが受容野のどの位置を参照するかを決定。
    C_M: チャンネルインデックス、C_H: 高さオフセット、C_W: 幅オフセット

    Args:
        out_channels: 出力チャンネル数（ツリー数）
        n_leaves: ツリーあたりのリーフ数
        in_channels: 入力チャンネル数
        kernel_size_h: カーネル高さ
        kernel_size_w: カーネル幅
        input_channels_per_tree: 各ツリーが参照する入力チャンネル数の上限
        device: 計算デバイス

    Returns:
        indices_flat: (out_channels, n_leaves) unfold後のフラットインデックス
    """
    # 各ツリーが参照するチャンネルをランダムに選択
    actual_ch = min(input_channels_per_tree, in_channels)

    C_M = torch.zeros(out_channels, n_leaves, dtype=torch.int64, device=device)
    C_H = torch.zeros(out_channels, n_leaves, dtype=torch.int64, device=device)
    C_W = torch.zeros(out_channels, n_leaves, dtype=torch.int64, device=device)

    for k in range(out_channels):
        # このツリーが使うチャンネルを選択（最大 input_channels_per_tree 個）
        if in_channels <= actual_ch:
            available_channels = torch.arange(in_channels, device=device)
        else:
            available_channels = torch.randperm(in_channels, device=device)[:actual_ch]

        # 各リーフに対しチャンネル・高さ・幅をランダムに割り当て
        ch_idx = torch.randint(0, actual_ch, (n_leaves,), device=device)
        C_M[k] = available_channels[ch_idx]
        C_H[k] = torch.randint(0, kernel_size_h, (n_leaves,), device=device)
        C_W[k] = torch.randint(0, kernel_size_w, (n_leaves,), device=device)

    # unfold後のフラットインデックスに変換
    # unfoldの出力は (B, C*kH*kW, L) で、次元1のインデックスは c*kH*kW + h*kW + w
    indices_flat = C_M * kernel_size_h * kernel_size_w + C_H * kernel_size_w + C_W

    return indices_flat


def _generate_tree_candidate_indices(
    out_channels: int,
    in_channels: int,
    kernel_size_h: int,
    kernel_size_w: int,
    input_channels_per_tree: int,
    device: str = 'cpu',
):
    """
    学習配線（wiring='learnable'）用の候補受容野インデックスを生成する。

    各ツリー（出力チャンネル）が参照可能な入力チャンネル部分集合
    （_generate_tree_connectionsと同じランダム選択方式）× カーネル窓の
    全位置を列挙し、ConvLearnableMappingFunctionがargmax選択で
    n_leaves個の葉を選ぶ候補集合とする（DWN論文§3.2のLearnable Mapping、
    dense版layers/dwn.py::LearnableMappingのツリー次元バッチ化）。

    固定配線（_generate_tree_connections）はn_leaves個を直接ランダム
    サンプルするのに対し、学習配線はチャンネル部分集合×カーネル窓の
    全候補位置を列挙してからargmaxで選択を学習する点が異なる。

    Args:
        out_channels: 出力チャンネル数（ツリー数）
        in_channels: 入力チャンネル数
        kernel_size_h: カーネル高さ
        kernel_size_w: カーネル幅
        input_channels_per_tree: 各ツリーが参照する入力チャンネル数の上限
        device: 計算デバイス

    Returns:
        indices: (out_channels, actual_ch*kH*kW) unfold後のフラットインデックス
    """
    actual_ch = min(input_channels_per_tree, in_channels)
    n_candidates = actual_ch * kernel_size_h * kernel_size_w
    indices = torch.zeros(
        out_channels, n_candidates, dtype=torch.int64, device=device
    )

    for k in range(out_channels):
        if in_channels <= actual_ch:
            available_channels = torch.arange(in_channels, device=device)
        else:
            available_channels = torch.randperm(in_channels, device=device)[:actual_ch]

        # このツリーが参照可能なチャンネル×カーネル窓の全位置を列挙
        ch = available_channels.repeat_interleave(kernel_size_h * kernel_size_w)
        hw = torch.arange(
            kernel_size_h * kernel_size_w, device=device
        ).repeat(actual_ch)
        hh = hw // kernel_size_w
        ww = hw % kernel_size_w
        indices[k] = ch * kernel_size_h * kernel_size_w + hh * kernel_size_w + ww

    return indices


class ConvLearnableMappingFunction(torch.autograd.Function):
    """LutConv2d用Learnable Mapping（DWN論文§3.2をツリー/出力チャンネル
    次元oでバッチ化したもの）。

    dense版LearnableMappingFunction（griffith_logic/layers/dwn.py）は
    入力全体で1つの重み行列 W(in_features, out_features) を共有するが、
    conv版は畳み込みのツリーごとに独立の配線を学習する必要があるため、
    重みにツリー次元o（=out_channels）を追加してバッチ化する。
    各ツリーoは、そのツリー専用の候補受容野
    （input_channels_per_tree*kH*kW次元）からn_leaves個の葉への選択を
    独立に学習する。

    forward:  leaves[b,o,i] = candidate[b,o, argmax_j weights[o,j,i]]
              （ハードargmax選択、公式実装と同じくsoftmax不使用）
    backward: ∂L/∂weights[o,j,i]  = Σ_b (2·candidate[b,o,j]−1)·G[b,o,i]
              ∂L/∂candidate[b,o,j] = Σ_i G[b,o,i]·softmax_{dim=1}(weights[o,:,i]/τ)[j]
              （dense版と同一の式をツリー次元oでバッチ化しただけ。
              o=1に潰せばdense版LearnableMappingFunctionの式と一致する）

    ground truthはcore/reference.py::reference_conv_learnable_mapping_forward /
    reference_conv_learnable_mapping_grads（素朴ループ+直接式）。
    """

    @staticmethod
    def forward(ctx, candidate: torch.Tensor, weights: torch.Tensor, tau: float):
        """
        Args:
            candidate: (batch, out_channels, n_candidates) ツリーごとの受容野候補
            weights: (out_channels, n_candidates, n_leaves) ツリーごとの選択重み
            tau: backwardのsoftmax温度

        Returns:
            (batch, out_channels, n_leaves)
        """
        mapping = weights.argmax(dim=1)  # (out_channels, n_leaves)
        ctx.save_for_backward(candidate, weights)
        ctx.tau = tau
        batch = candidate.shape[0]
        idx = mapping.unsqueeze(0).expand(batch, -1, -1)
        return candidate.gather(2, idx)

    @staticmethod
    def backward(ctx, grad_output):
        candidate, weights = ctx.saved_tensors
        grad_candidate = grad_weights = None

        if ctx.needs_input_grad[0]:
            soft = torch.softmax(weights / ctx.tau, dim=1)  # (O, n_cand, n_leaves)
            grad_candidate = torch.einsum('boi,ori->bor', grad_output, soft)

        if ctx.needs_input_grad[1]:
            grad_weights = torch.einsum(
                'bor,boi->ori', 2.0 * candidate - 1.0, grad_output,
            )

        return grad_candidate, grad_weights, None


class LutConv2d(nn.Module):
    """
    論理ゲートツリー畳み込み層（CDDLGN Section 3, Eq. 2-3）。

    各カーネルはN入力LUTの完全木で構成される。
    木の深さdでN^d個のリーフ入力を受容野からランダムに選択し、
    ツリー状にLUT演算を適用して1出力を生成する。

    同じゲート重みが全空間位置で共有される（畳み込みの性質）。
    リーフの受容野内位置は既定で固定ランダム（wiring='fixed'）だが、
    wiring='learnable'でDWN論文§3.2のLearnable Mappingにより勾配で
    学習することもできる（配線もゲート重みも全空間位置で共有）。

    CDDLGNの数式（Eq. 3）:
        A'[k, i, j] = tree_k(A[C_M[k,:], C_H[k,:]+i, C_W[k,:]+j])
    """

    PARAMETERIZATIONS = ('softmax', 'independent', 'independent_ste',
                         'independent_dwn', 'iwp', 'softmax16g')

    WIRINGS = ('fixed', 'learnable')

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        tree_depth: int = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        residual_init: bool = True,
        input_channels_per_tree: int = 2,
        parameterization: str = 'softmax',
        wiring: str = 'fixed',
        lm_tau: float = 0.001,
        sampler: str = 'random',
        layer_idx: int = 0,
        estimator: str = 'sigmoid',
        ri_mu: float = 1.2,
        ri_sigma: float = 0.25,
        generator: torch.Generator | None = None,
    ):
        """
        Args:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数（ツリー数）
            kernel_size: 受容野のサイズ
            tree_depth: ツリーの深さ（d）。リーフ数 = n_inputs^d
            stride: 畳み込みストライド
            padding: ゼロパディング
            n_inputs: 各ゲートの入力数（2がCDDLGN準拠）
            device: 計算デバイス
            grad_factor: 勾配スケーリング係数（GradFactorはunfold前のxに適用
                されるため、wiring='fixed'/'learnable'どちらでも配線より
                前段でxの勾配をスケールする。適用順序: x → grad_factor →
                unfold → 配線（固定gather or Learnable Mapping）→ ツリー）
            residual_init: Residual Initialization を使用するか
            input_channels_per_tree: 各ツリーが参照する入力チャンネル数の上限
                （wiring='learnable'時は候補受容野のチャンネル数としても使う）
            parameterization: パラメータ化方式
                'softmax' — エントリ間競合（DLGN方式、デフォルト）
                'independent' — 各エントリ独立（BinaryBrain方式）
                'iwp' — 入力ごと1パラメータのリテラル積（単一minterm、
                    LDDLGN方式）。**論文 arXiv:2510.03250 の IWP とは別物**
                    で、2入力では16ゲート中4種しか表現できない（素通しも
                    XORも不可）。論文IWPに相当するのは 'independent'
            wiring: 入力配線の方式（DWN論文 arXiv 2410.11112 §3.2、
                dense版LutLayerのwiring引数と同じ機構をツリー次元でバッチ化）
                'fixed' — 初期化時に固定した接続（既定、従来挙動）
                'learnable' — ConvLearnableMappingFunction（勾配で配線を
                    学習。各ツリーが候補受容野input_channels_per_tree*kH*kW
                    次元からn_leaves個の葉への選択を独立に学習する）
            lm_tau: wiring='learnable'時のLearnable Mapping逆伝播softmax
                温度（DWN公式実装の既定値0.001）
            sampler: 葉の読み座標の決め方(conv_samplers.py)。
                'random' — 既定。従来の固定ランダム(チャネル部分集合 × 窓内一様)
                'random8' / 'cross5' / 'dilated8' / 'checker13' — 内部調査リポジトリ
                    先行実験のサンプラ。窓(kernel_size)と木(tree_depth)は腕ごとに
                    決まっており(SAMPLER_SPECS)、引数と食い違えば ValueError。
                    cross5 / dilated8 / checker13 は座標が (出力チャネル, layer_idx) の
                    計算可能関数(座標ストアゼロ)。2 入力木・wiring='fixed' 専用
            layer_idx: 層番号(パターン系サンプラの位相を層でずらすのに使う)
            estimator: parameterization='independent' / 'independent_ste' のエントリ
                確率の推定器 ρ。'sigmoid'(既定)/ 'sin'(0.5 + 0.5·sin。論文 IWP
                arXiv:2510.03250 §C.1.1 = 先行実験の既定)。他の parameterization では
                'sigmoid' のみ
            ri_mu / ri_sigma: estimator='sin' のときの Residual Initialization
                Ω ~ N(±μ, σ)(第1入力 a のpass-throughへ寄せる。既定 μ=1.2, σ=0.25 =
                論文 §C.1.1 / 先行実験 IWP_RI_DEFAULTS['sin'])
            generator: sampler='random8' の乱数と estimator='sin' の RI に使う generator
                (None ならグローバル RNG)
        """
        super().__init__()

        assert n_inputs in (2, 4, 6), f'n_inputs は 2, 4, 6 のいずれか: {n_inputs}'
        assert parameterization in self.PARAMETERIZATIONS, (
            f'parameterization は {self.PARAMETERIZATIONS} のいずれか: {parameterization}'
        )
        assert wiring in self.WIRINGS, (
            f'wiring は {self.WIRINGS} のいずれか: {wiring}'
        )
        assert estimator in ESTIMATORS, (
            f'estimator は {ESTIMATORS} のいずれか: {estimator}'
        )
        if estimator != 'sigmoid' and parameterization not in (
                'independent', 'independent_ste'):
            raise ValueError(
                f"estimator='{estimator}' は parameterization='independent' / "
                f"'independent_ste' 専用です(指定: '{parameterization}')"
            )
        if parameterization == 'softmax16g':
            assert n_inputs == 2, (
                "parameterization='softmax16g' は n_inputs=2 のみ対応 "
                f"(16ゲート真理値表がN=2専用): n_inputs={n_inputs}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.tree_depth = tree_depth
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.n_inputs = n_inputs
        self.n_entries = 2 ** n_inputs
        self.grad_factor = grad_factor
        self.input_channels_per_tree = input_channels_per_tree
        self.parameterization = parameterization
        self.wiring = wiring
        self.lm_tau = lm_tau
        self.sampler = sampler
        self.layer_idx = int(layer_idx)
        self.estimator = estimator
        self.ri_mu = float(ri_mu)
        self.ri_sigma = float(ri_sigma)

        # ツリー構造のサイズ
        self.n_leaves = n_inputs ** tree_depth
        # 内部ノード数 = (N^d - 1) / (N - 1)
        if n_inputs == 1:
            self.n_internal = tree_depth
        else:
            self.n_internal = (self.n_leaves - 1) // (n_inputs - 1)

        # 学習パラメータ: パラメータ化方式に応じてサイズが異なる
        if parameterization == 'iwp':
            # IWP: ゲートあたりn_inputs個のパラメータ
            self.weights = nn.Parameter(
                torch.randn(out_channels, self.n_internal, n_inputs, device=device)
            )
        elif parameterization == 'softmax16g':
            # softmax16g: ゲートあたり16個のロジット（N=2の16 Booleanゲート）。
            # 真理値表は softmax(w) @ gate_table で構成され、全16ゲートを表現できる
            # （旧'softmax'の4-simplex制約を解消）。
            self.weights = nn.Parameter(
                torch.randn(out_channels, self.n_internal, 16, device=device)
            )
            self.register_buffer('gate_table', _build_gate_table_n2().to(device))
        elif parameterization == 'independent_dwn':
            # DWN 公式実装に合わせ [-1,1] の一様乱数で初期化する
            # （randn だと初期値の大半が clamp で ±1 に張り付き、
            #   どの表エントリも勾配1ステップでは反転できなくなる）
            self.weights = nn.Parameter(
                torch.rand(out_channels, self.n_internal, self.n_entries,
                           device=device) * 2.0 - 1.0
            )
        else:
            # softmax/independent: ゲートあたり2^n_inputs個のパラメータ
            self.weights = nn.Parameter(
                torch.randn(out_channels, self.n_internal, self.n_entries, device=device)
            )

        # 接続テーブル: リーフが受容野のどの位置を参照するか
        # wiring='learnable'時も従来通り登録する（推論時フォールバック/
        # reach-map互換のため。dense版LutLayerのindicesバッファと同じ方針）
        if sampler == 'random':
            indices_flat = _generate_tree_connections(
                out_channels, self.n_leaves,
                in_channels, self.kernel_size[0], self.kernel_size[1],
                input_channels_per_tree, device,
            )
        else:
            indices_flat = self._sampled_indices_flat(
                sampler, in_channels, out_channels, input_channels_per_tree, generator,
            ).to(device)
        self.register_buffer('indices_flat', indices_flat)

        # 配線学習（DWN論文§3.2、dense版LutLayer(wiring='learnable')と
        # 同じ機構をツリー次元でバッチ化）
        if wiring == 'learnable':
            candidate_indices = _generate_tree_candidate_indices(
                out_channels, in_channels, self.kernel_size[0], self.kernel_size[1],
                input_channels_per_tree, device,
            )
            self.register_buffer('wiring_candidate_indices', candidate_indices)
            n_candidates = candidate_indices.shape[1]
            # 公式実装と同じ一様[0,1)初期化（dense版LearnableMappingと同一）
            self.mapping_weights = nn.Parameter(
                torch.rand(out_channels, n_candidates, self.n_leaves, device=device)
            )

        # Residual Initialization（Section 3.2）— IWPでは非対応
        if residual_init and parameterization != 'iwp':
            self._residual_init(generator=generator)

    def _sampled_indices_flat(self, sampler: str, in_channels: int, out_channels: int,
                              num_chn: int, generator) -> torch.Tensor:
        """先行実験のサンプラで葉座標を決め、unfold 後の平坦添字にする。"""
        if sampler not in SAMPLER_SPECS:
            raise ValueError(f'未知のサンプラ: {sampler!r}(対応: random / {sorted(SAMPLER_SPECS)})')
        spec = SAMPLER_SPECS[sampler]
        if self.n_inputs != 2:
            raise ValueError(f"sampler='{sampler}' は 2 入力木専用です(n_inputs={self.n_inputs})")
        if self.wiring != 'fixed':
            raise ValueError(f"sampler='{sampler}' は wiring='fixed' 専用です")
        if self.kernel_size != (spec['kernel'], spec['kernel']):
            raise ValueError(
                f"sampler='{sampler}' の窓は {spec['kernel']}×{spec['kernel']} です"
                f'(kernel_size={self.kernel_size})'
            )
        if self.n_leaves != 2 ** spec['depth']:
            raise ValueError(
                f"sampler='{sampler}' の木は深さ {spec['depth']}(葉 {2 ** spec['depth']})です"
                f'(tree_depth={self.tree_depth})'
            )
        sel_ch, sel_dy, sel_dx = build_leaf_selection(
            sampler, out_channels, in_channels, layer_idx=self.layer_idx,
            num_chn=num_chn, generator=generator,
        )
        return leaf_indices_flat(sel_ch, sel_dy, sel_dx, *self.kernel_size)

    def _residual_init(self, z3: float = 5.0, generator=None):
        """
        Residual Initialization（CDDLGN Section 3.2）。

        各ゲートをゲート'A'（第1入力のパススルー）にバイアスして初期化。
        z_3 = 5, 他の全z_i = 0 → softmax後 ゲート'A'の確率 ≈ 90%。

        'softmax16g': 16ゲートのうち真理値表[0,1,0,1]（ゲート'A'）のロジットに
        z3を設定。P('A')≈90% → soft真理値表 tt ≈ 0.9·[0,1,0,1] + 残りの混合。
        この tt は第1入力aをほぼそのまま通すため、深いツリーでも活性が減衰しない
        （旧'softmax'は softmax([0,z3,0,z3])=[0,.5,0,.5] で tt≈0.5·a と2倍減衰した）。

        'softmax'（4エントリ）: 従来どおり bit(k,0)==1 のエントリにz3を設定。

        'independent' / 'independent_ste' × estimator='sin': sin(5.0) は符号が
        反転して破綻するため、論文 IWP(arXiv:2510.03250 §C.1.1)の Ω 空間 RI
        Ω ~ N(±μ, σ)（bit 0 が立つエントリ = 第1入力が 1 なら +μ、他は −μ。
        ρ(+1.2) ≈ 0.97 / ρ(−1.2) ≈ 0.03）を使う（先行実験 `init_omega` の RI_SIGNS と同じ）。
        """
        with torch.no_grad():
            if (self.parameterization in ('independent', 'independent_ste')
                    and self.estimator == 'sin'):
                e = torch.arange(self.n_entries)
                sign = torch.where((e & 1).bool(), self.ri_mu, -self.ri_mu)
                noise = torch.randn(
                    self.weights.shape, generator=generator, dtype=self.weights.dtype,
                ) * self.ri_sigma
                self.weights.copy_((noise + sign.to(noise.dtype)).to(self.weights.device))
                return
            self.weights.zero_()
            if self.parameterization == 'softmax16g':
                a_idx = self._gate_a_index()
                self.weights[:, :, a_idx] = z3
            else:
                for k in range(self.n_entries):
                    if k & 1:  # bit 0 が立っている = 第1入力が1
                        self.weights[:, :, k] = z3

    def _gate_a_index(self) -> int:
        """gate_table中で真理値表[0,1,0,1]（ゲート'A'=第1入力パススルー）を持つ
        行インデックスを検索（ハードコードせず動的に特定）。"""
        target = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=self.gate_table.dtype,
                              device=self.gate_table.device)
        matches = (self.gate_table == target).all(dim=1).nonzero().flatten()
        assert matches.numel() == 1, f'ゲートA[0,1,0,1]が一意に見つからない: {matches}'
        return int(matches.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, height, width) 入力テンソル、値は[0,1]

        Returns:
            (batch, out_channels, H_out, W_out) 出力テンソル
        """
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # 1. Unfold → パッチ抽出
        # (B, C*kH*kW, L) where L = H_out * W_out
        patches = F.unfold(x, (kH, kW), padding=(pH, pW), stride=(sH, sW))
        L = patches.shape[2]

        # 出力の空間サイズを計算
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        # 2. リーフ入力の収集
        if self.wiring == 'learnable':
            # wiring_candidate_indices: (out_channels, n_candidates)
            # candidate実体化 (B, out_channels, n_candidates, L) はout_channels×
            # n_candidatesに比例して巨大化する（検出hm_convでOOMの原因）。
            # torch.utils.checkpointでこのcandidate生成〜Learnable Mapping適用を
            # 「非保存区間」にする: forward時はpatches（out_channelsに依存しない
            # 小さいテンソル）のみをctxに保持し、backward時にcandidateを再計算する
            # （数値は不変。ConvLearnableMappingFunction自体は無改造＝
            # tests/l2_grad/test_conv_learnable_wiring_grad.pyの直接呼び出し契約を
            # 破らない）。
            def _learnable_leaves(patches_, indices_, weights_):
                # patches[:, wiring_candidate_indices]: (B, out_channels, n_candidates, L)
                candidate = patches_[:, indices_]
                # (B, out_channels, n_candidates, L) → (B, L, out_channels, n_candidates)
                candidate = candidate.permute(0, 3, 1, 2)
                # (B*L, out_channels, n_candidates)
                candidate = candidate.reshape(B * L, self.out_channels, -1)
                # Learnable Mapping（DWN論文§3.2）でn_leaves個の葉へハード選択。
                # train/eval問わず常に決定的argmax選択（dense版wiring='learnable'の
                # 既定forward_mode='hard'と同じ; soft段階的ハード化は未対応）
                return ConvLearnableMappingFunction.apply(
                    candidate, weights_, self.lm_tau,
                )

            leaves = checkpoint(
                _learnable_leaves, patches, self.wiring_candidate_indices,
                self.mapping_weights, use_reentrant=False,
            )
        else:
            # indices_flat: (out_channels, n_leaves)
            # patches[:, indices_flat]: (B, out_channels, n_leaves, L)
            leaves = patches[:, self.indices_flat]
            # (B, out_channels, n_leaves, L) → (B, L, out_channels, n_leaves)
            leaves = leaves.permute(0, 3, 1, 2)
            # (B*L, out_channels, n_leaves)
            leaves = leaves.reshape(B * L, self.out_channels, self.n_leaves)

        # 3. ツリー演算
        if self.training:
            output = self._tree_forward_train(leaves)
        else:
            output = self._tree_forward_eval(leaves)

        # 4. 出力整形: (B*L, out_channels) → (B, out_channels, H_out, W_out)
        output = output.reshape(B, L, self.out_channels)
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)

        return output

    def _tree_forward_train(self, leaves: torch.Tensor) -> torch.Tensor:
        """
        訓練時のツリー演算。連続緩和で勾配を流す。

        Args:
            leaves: (B*L, out_channels, n_leaves)

        Returns:
            (B*L, out_channels) ツリーの出力
        """
        if self.parameterization == 'independent_dwn':
            # DWN 公式実装に倣い、表パラメータを [-1,1] に収める。
            # ループの外で1回だけ（スライスのin-place更新はautogradの
            # バージョン管理と衝突するため）。
            with torch.no_grad():
                self.weights.clamp_(-1.0, 1.0)

        current = leaves
        gate_offset = 0

        for _level in range(self.tree_depth):
            n_groups = current.shape[-1] // self.n_inputs
            # N個ずつグループ化: (..., n_groups, N)
            grouped = current[..., :n_groups * self.n_inputs].reshape(
                *current.shape[:-1], n_groups, self.n_inputs
            )

            # このレベルのゲート重み: (out_channels, n_groups, ...)
            level_weights = self.weights[:, gate_offset:gate_offset + n_groups, :]

            if self.parameterization == 'iwp':
                # IWP: 積の形で計算（basis不要）
                # level_weights: (out_channels, n_groups, n_inputs)
                p = torch.sigmoid(level_weights)
                factors = p.unsqueeze(0) * grouped + (1.0 - p.unsqueeze(0)) * (1.0 - grouped)
                current = factors.prod(dim=-1)
            else:
                # softmax/independent/softmax16g: basis + 加重和
                basis = _compute_basis(grouped, self.n_inputs)
                if self.parameterization == 'independent':
                    w = estimator_rho(level_weights, self.estimator)
                elif self.parameterization == 'independent_ste':
                    # independent は表が連続のまま学習され、最後に丸めると
                    # 全ゲートが反転して崩壊する（実測: soft 0.68 に対し hard 0.00）。
                    # DWN と同じく forward で表を二値化し、勾配は素通しする。
                    # これで訓練時の forward が評価時と完全に一致し、
                    # 離散化ギャップが構造的にゼロになる。
                    w = _BinarizeTableSTE.apply(estimator_rho(level_weights, self.estimator))
                elif self.parameterization == 'independent_dwn':
                    # independent_ste は勾配が sigmoid'(x) 倍されるため、
                    # 学習が進んで |x| が育つと勾配が消える。DWN 公式実装は
                    # sigmoid を使わず生パラメータを [-1,1] にclampして 1[w>0]
                    # で二値化する（勾配は減衰しない）。その規約に揃えたもの。
                    # clamp は _tree_forward_train の冒頭で済ませている。
                    w = _BinarizeTableSTE.apply(level_weights, 0.0)
                elif self.parameterization == 'softmax16g':
                    # 16ゲートロジット → softmax混合 → gate_table経由で4基底の
                    # 有効真理値表に変換（全16 Booleanゲートを表現可能）
                    s = F.softmax(level_weights, dim=-1)
                    w = s @ self.gate_table.to(s.dtype)
                else:
                    w = F.softmax(level_weights, dim=-1)
                current = (basis * w.unsqueeze(0)).sum(dim=-1)

            gate_offset += n_groups

        return current.squeeze(-1)

    def _tree_forward_eval(self, leaves: torch.Tensor) -> torch.Tensor:
        """
        推論時のツリー演算。離散化でBoolean論理に崩壊。

        Args:
            leaves: (B*L, out_channels, n_leaves)

        Returns:
            (B*L, out_channels) ツリーの出力
        """
        current = leaves
        gate_offset = 0

        for _level in range(self.tree_depth):
            n_groups = current.shape[-1] // self.n_inputs
            grouped = current[..., :n_groups * self.n_inputs].reshape(
                *current.shape[:-1], n_groups, self.n_inputs
            )

            level_weights = self.weights[:, gate_offset:gate_offset + n_groups, :]

            if self.parameterization == 'iwp':
                # IWP: sigmoid→round→リテラルの積
                p = (torch.sigmoid(level_weights) > 0.5).to(grouped.dtype)
                factors = p.unsqueeze(0) * grouped + (1.0 - p.unsqueeze(0)) * (1.0 - grouped)
                current = factors.prod(dim=-1)
            else:
                basis = _compute_basis(grouped, self.n_inputs)
                if self.parameterization in ('independent', 'independent_ste'):
                    w = (estimator_rho(level_weights, self.estimator) > 0.5).to(basis.dtype)
                elif self.parameterization == 'independent_dwn':
                    w = (level_weights > 0.0).to(basis.dtype)
                elif self.parameterization == 'softmax16g':
                    # argmaxで16ゲートから1つ選択 → その真理値表（4エントリ、離散Boolean）
                    gate_idx = level_weights.argmax(-1)  # (out_channels, n_groups)
                    w = self.gate_table.to(basis.dtype)[gate_idx]  # (..., 4)
                else:
                    w = F.one_hot(
                        level_weights.argmax(-1), self.n_entries
                    ).to(basis.dtype)
                current = (basis * w.unsqueeze(0)).sum(dim=-1)

            gate_offset += n_groups

        return current.squeeze(-1)

    @property
    def num_gates(self) -> int:
        """総ゲート数（全ツリーの内部ノード数の合計）"""
        return self.out_channels * self.n_internal

    def extra_repr(self):
        wiring_str = (
            f', wiring={self.wiring}/τ={self.lm_tau}'
            if self.wiring == 'learnable' else ''
        )
        sampler_str = f', sampler={self.sampler}' if self.sampler != 'random' else ''
        est_str = f', est={self.estimator}' if self.estimator != 'sigmoid' else ''
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, tree_depth={self.tree_depth}, '
            f'stride={self.stride}, padding={self.padding}, '
            f'n_inputs={self.n_inputs}, n_leaves={self.n_leaves}, '
            f'n_internal={self.n_internal}, '
            f'input_ch_per_tree={self.input_channels_per_tree}, '
            f'param={self.parameterization}{est_str}{wiring_str}{sampler_str}'
        )
