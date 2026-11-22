"""N入力LUT(Look-Up Table)層。"""

import math
import os
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.basis import (
    _build_gate_table_n2,
    _compute_basis_butterfly,
    _compute_wh_basis_butterfly,
)
from ..core.connections import (
    build_reverse_index,
    get_random_connections,
    get_unique_connections,
)
from ..core.estimators import estimator_rho
from ..core.functions import (
    CudaLutFunction,
    CudaLutFunctionIndependent,
    CudaLutFunctionIWP,
    CudaLutFunctionWARP,
    LutFunction,
    LutFunctionHardST,
    LutFunctionIndependent,
    LutFunctionIWP,
    LutFunctionRaw,
    LutFunctionWARP,
    LutFunctionWARPHardST,
)
from ..core.ste import GradFactor
from ..cuda import cuda_available

# Gumbelノイズ(Mind the Gap)の数値安定用: U~Uniform を [eps, 1-eps] にクランプ。
# log(-log(0)) の発散を避ける。causal_attention の _GUMBEL_EPS と同趣旨。
_GUMBEL_EPS = 1e-9

# iwp/warp の融合CUDAカーネルを使わず、Python実装(LutFunctionIWP/LutFunctionWARP)へ
# 戻すための環境変数。値はカンマ区切りのパラメータ化名(例: 'iwp', 'warp', 'iwp,warp')。
# 融合カーネル導入前の結果の再現や、カーネル有無のA/B比較に使う。
# 呼び出しごとに読む(テストや比較スクリプトから同一プロセス内で切り替えられるように)。
_ENV_PYTHON_PATH = 'MD_LUT_PYTHON_PATH'
_PYTHON_PATH_PARAMS = ('iwp', 'warp')


def _python_path_forced(parameterization: str) -> bool:
    """環境変数 MD_LUT_PYTHON_PATH で、このパラメータ化のPython実装が強制されているか。"""
    value = os.environ.get(_ENV_PYTHON_PATH)
    if not value:
        return False
    names = {v.strip().lower() for v in value.split(',')}
    return parameterization in _PYTHON_PATH_PARAMS and parameterization in names


def _randn_init(shape, device, generator):
    """重み初期化用のN(0,1)。

    generator=None のときは従来コードと同一の `torch.randn(..., device=device)`
    （bitwise不変）。generator指定時はCPUで生成してからdeviceへ移し、
    デバイスに依らず同じ初期重みになるようにする。
    """
    if generator is None:
        return torch.randn(*shape, device=device)
    return torch.randn(*shape, generator=generator).to(device)


class LutLayer(nn.Module):
    """
    N入力LUT（Look-Up Table）層。

    一般化LUTの数式:
        y = Σ_{k=0}^{2^N-1} f(W)[k] × Π_{j=0}^{N-1} (x_j if bit(k,j)=1 else 1-x_j)

    パラメータ化方式:
        - 'softmax': f = softmax（エントリ間が競合、合計1）— DLGN方式
        - 'independent': f = ρ（各エントリ独立、合計≠1）— BinaryBrain方式。
          論文 arXiv:2510.03250 の IWP (Eq.11) と数学的に同一
          （ω_k = ρ(Ω_k) をエントリ確率とする一般化LUT）
        - 'iwp': 入力ごとに1パラメータのリテラル積（単一ミンターム）。
          **論文 arXiv:2510.03250 の IWP とは別物**（16ゲート中4種しか
          表現できない）。論文のIWPは 'independent' に相当する
        - 'raw': 重みを直接[0,1]で保持 — BinaryBrain方式
        - 'warp': Walsh-Hadamard基底（制約なし実数、Gumbel smoothing）— WARP方式

    N=2でdifflogicの16ゲートsoftmax加重と等価。
    N=4,6でBinaryBrainのLUT4/LUT6と同じ表現力。
    N=4,6ではwarpパラメータ化が安定学習に推奨（Gerlach et al., 2025）。
    """

    PARAMETERIZATIONS = ('softmax', 'independent', 'iwp', 'raw', 'warp')

    # independent(=論文IWP)のエントリ確率推定器 ρ: ℝ→[0,1]
    ESTIMATORS = ('sigmoid', 'sin')

    SELECTIONS = ('soft', 'hard_st', 'gumbel_st')

    WIRINGS = ('fixed', 'learnable')

    INIT_MODES = ('randn', 'ri', 'ri_randn', 'rri', 'uniform')

    # 16ゲートID規約（core/basis.py _build_gate_table_n2）での自明ゲート。
    # 定数2種と1入力関数（pass-through / 否定）4種は縮約で必ず消える
    A_GATE_ID = 3                            # A pass-through（残差初期化の対象）
    CONST_GATE_IDS = (0, 15)                 # FALSE / TRUE
    PASSTHROUGH_GATE_IDS = (3, 5, 10, 12)    # A / B / ¬B / ¬A
    TRIVIAL_GATE_IDS = (0, 15, 3, 5, 10, 12)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        connections: str = 'random',
        parameterization: str = 'softmax',
        selection: str = 'soft',
        gumbel_noise: bool = False,
        gumbel_tau: float = 1.0,
        residual_init: bool = False,
        wiring: str = 'fixed',
        lm_tau: float = 0.001,
        wh_degree_cap: int | None = None,
        estimator: str = 'sigmoid',
        ri_mu: float = 1.2,
        ri_sigma: float = 0.25,
        gate_mask: Sequence[int] | None = None,
        init_mode: str | None = None,
        init_scale: float = 5.0,
        generator: torch.Generator | None = None,
    ):
        """
        Args:
            in_dim: 入力次元数
            out_dim: 出力ニューロン数
            n_inputs: LUTの入力数（2, 4, 6）
            device: 計算デバイス
            grad_factor: 勾配スケーリング係数（深いモデルでは>1に）
            connections: 接続初期化方式 ('random' or 'unique')
            parameterization: パラメータ化方式
                'softmax' — エントリ間競合（DLGN方式、デフォルト）
                'independent' — 各エントリ独立（BinaryBrain方式）
                'iwp' — 入力ごと1パラメータのリテラル積（単一minterm）。
                    **論文 arXiv:2510.03250 の IWP とは別物**で、2入力では
                    16ゲート中4種しか表現できない（素通しもXORも不可）。
                    論文IWPに相当するのは 'independent'
            selection: 訓練時のゲート選択方式
                'soft' — softmax混合（デフォルト、従来方式）
                'hard_st' — Hard Straight-Through（CAGE論文方式）
            gumbel_noise: ゲート選択ロジットへのGumbelノイズ注入（既定False）。
                Mind the Gap (arXiv:2506.07500) の離散化ギャップ縮小手法。
                **train(soft)時のみ**有効、eval(hard)時は完全に無効（bitwise不変）。
                softmaxパラメータ化(selection='soft')のみ対応。既定Off時は
                既存の全挙動が乱数消費も含めbitwise不変（後方互換の最重要契約）。
            gumbel_tau: Gumbelノイズ注入時のsoftmax温度 τ（既定1.0）。
                Mind the Gap の π_i = softmax((w_i+g_i)/τ) の τ。既定1.0は
                従来のsoftmax(τ=1相当)と整合させた値。gumbel_noise=False時は無視。
                'gumbel_st' — Gumbel-ST（Mind the Gap, arXiv 2506.07500）。
                    forward: argmax(weights + gumbel_scale·G)の離散サンプル
                    （Gumbel-max、温度フリー）、backward: softmax逆伝播
                    （τ_b、hard_stと同一のSTE）。数学的には
                    「weights+ノイズへのHard-ST」と厳密に等価なので
                    LutFunctionHardSTを再利用する。gumbel_scale→0で
                    hard_stに退化。evalは従来どおり決定的argmax
            residual_init: Residual Initialization。Trueかつゲートごと16ロジットを
                持つパラメータ化（softmax N=2 の16ゲートモード）で、重みを0初期化し
                ゲート'A'（第1入力パススルー、真理値表[0,1,0,1]）のロジットを
                init_scale（既定5.0）に。
                → P('A')≈90%で恒等に近い初期化（深層化安定化）。既定Falseで
                従来のrandn初期化（既存レシピ・テストへの影響なし）。
            wiring: 入力接続の方式（DWN論文 arXiv 2410.11112 §3.2）
                'fixed' — 初期化時に固定した接続（既定、従来挙動）
                'learnable' — LearnableMapping（勾配で配線を学習、DWNLayerと同機構）
            lm_tau: wiring='learnable'時のLearnableMapping逆伝播softmax温度
                （DWN公式実装の既定値0.001）
            estimator: エントリ確率の推定器 ρ: ℝ→[0,1]。
                'sigmoid' — σ(Ω)（既定、従来挙動）
                'sin' — 0.5+0.5·sin(Ω)（論文 arXiv:2510.03250 §C.1.1 が
                    実験で採用した正弦推定器。飽和せず勾配が消えない）
                parameterization='independent' のときのみ 'sigmoid' 以外を許可。
            ri_mu: estimator='sin' 時のresidual initialization平均 μ（既定1.2）。
                Ω ~ N(±μ, σ)（論文 §C.1.1）。
            ri_sigma: 同じくresidual initializationの標準偏差 σ（既定0.25）。
            gate_mask: **除外する**ゲートIDの集合（16ゲートモード専用、既定None）。
                真理値表整数ID（core/basis.py の _build_gate_table_n2 の規約:
                op = 8·P00 + 4·P01 + 2·P10 + P11。op3 = A pass-through、
                {0,15} = 定数、{3,5,10,12} = pass-through/否定）を渡すと、
                そのゲートのロジットに −inf を加えて soft/hard_st/gumbel_st の
                全forward経路とevalのargmaxから排除する。勾配も厳密に0になる。
                空列・Noneは「除外なし」で従来と**bitwise不変**。
                n_inputs=2 かつ parameterization='softmax' 以外は ValueError。
            init_mode: 16ゲートモードのロジット初期化方式（既定None=従来動作）。
                None のときは residual_init に従う（True→'ri' / False→'randn'）。
                'randn'    — N(0,1)（従来の既定）
                'ri'       — 全ロジット0 + op3（A pass-through）に init_scale
                             （従来の residual_init=True。既定 5.0）
                'ri_randn' — N(0,1) に op3 だけ +init_scale（残差初期化の変種。
                             非op3の対称性が 'ri' と違う）
                'rri'      — N(0,1) に「ゲートごとに乱択した、gate_maskで許可
                             されている非自明1種（定数{0,15}・pass-through
                             {3,5,10,12}を除く）」へ +init_scale
                'uniform'  — 全ロジット0（16種等確率。入力勾配が厳密0になる）
                いずれのモードでも randn の消費量は同じなので、init_mode=None
                では従来と bitwise 不変。16ゲートモード以外は ValueError。
            init_scale: 残差初期化の加算量（既定5.0）。'ri'/'ri_randn'/'rri' の
                +αの大きさで、softmaxの初期確率を決める（16ゲート・N(0,1)の場合
                P(選択) ≈ e^α/(e^α+15) なので α=5 で ≈0.91、α=1 で ≈0.15）。
                社内先行実験 Phase 3 の OP 腕は `--residual_init` 既定
                1.0（train_cifar_ab.py:255 → model_iwp.py:79-80 の
                `w[:,3] += residual_init`）なので、先行実験を再現するなら 1.0。
                'randn'/'uniform' では使わない（無視）。0以下は ValueError。
            generator: 重み初期化に使う乱数生成器（既定None=グローバルRNG）。
                デバイス非依存の決定性のため**CPU Generator**を渡すこと
                （CPUで生成してからdeviceへ移す）。None時は従来の
                torch.randn(..., device=device) をそのまま呼ぶので bitwise 不変。
        """
        super().__init__()
        assert selection in self.SELECTIONS, (
            f'selection は {self.SELECTIONS} のいずれか: {selection}'
        )

        assert n_inputs in (2, 4, 6), f'n_inputs は 2, 4, 6 のいずれか: {n_inputs}'
        explicit_conn = isinstance(connections, torch.Tensor)
        assert explicit_conn or connections in ('random', 'unique'), connections
        assert parameterization in self.PARAMETERIZATIONS, (
            f'parameterization は {self.PARAMETERIZATIONS} のいずれか: {parameterization}'
        )
        assert wiring in self.WIRINGS, (
            f'wiring は {self.WIRINGS} のいずれか: {wiring}'
        )
        assert estimator in self.ESTIMATORS, (
            f'estimator は {self.ESTIMATORS} のいずれか: {estimator}'
        )
        # 推定器の差し替えはindependent(=論文IWP)経路のみ実装済み。
        # 他のパラメータ化はsigmoid固定の式・CUDAカーネルに依存するため、
        # 黙って誤った勾配を出さないよう明示的に弾く。
        if estimator != 'sigmoid' and parameterization != 'independent':
            raise NotImplementedError(
                f"estimator='{estimator}' は parameterization='independent' "
                f"のみ対応です（指定: '{parameterization}'）。"
                '論文 arXiv:2510.03250 のIWPに相当するのはindependentです。'
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_inputs = n_inputs
        self.n_entries = 2 ** n_inputs
        self.device = device
        self.grad_factor = grad_factor
        self.parameterization = parameterization
        self.estimator = estimator
        self.ri_mu = ri_mu
        self.ri_sigma = ri_sigma
        self.selection = selection
        self.wh_degree_cap = wh_degree_cap
        if wh_degree_cap is not None:
            assert parameterization == 'warp', (
                'wh_degree_capはwarpパラメータ化専用'
            )
        self.wiring = wiring
        self.tau_b = 1.0  # Hard/Gumbel-STの逆伝播温度（スケジューラから更新）
        self.gumbel_scale = 1.0  # Gumbel-STのノイズ振幅（0でhard_stに退化）

        # Gumbelノイズ設定（Mind the Gap）
        self.gumbel_noise = gumbel_noise
        self.gumbel_tau = gumbel_tau
        if gumbel_noise:
            # softmaxのゲート選択ロジットにのみ素直に対応する。
            # warpはWH係数θ→単一logit→sigmoid構造で「16ゲート上のsoftmax」に
            # 相当する分布を持たず、Mind the Gapの式が写像できないため未対応
            # （warpは別途 self.gumbel によるGumbel-Sigmoid平滑化を既に持つ）。
            # hard_stはforwardがargmaxで、緩和softmaxへのノイズ注入と両立しない。
            if parameterization != 'softmax':
                raise NotImplementedError(
                    "gumbel_noise は parameterization='softmax' のみ対応です"
                    f"（指定: '{parameterization}'）。warpはロジット構造が異なり"
                    'Mind the Gapのsoftmaxゲート選択ノイズを写像できないため、'
                    'warp独自のGumbel-Sigmoid（gumbel=True）を使用してください。'
                )
            if selection != 'soft':
                raise NotImplementedError(
                    "gumbel_noise は selection='soft' のみ対応です"
                    f"（指定: '{selection}'）。hard_stはforwardがargmax選択のため"
                    '緩和softmaxへのGumbelノイズ注入と両立しません。'
                )

        # Hard-ST/Gumbel-STはsoftmaxパラメータ化のみ対応
        if selection in ('hard_st', 'gumbel_st'):
            assert parameterization in ('softmax', 'warp'), (
                f'{selection}はsoftmax/warpパラメータ化のみ対応: {parameterization}'
            )

        # 16ゲートモード: softmax + N=2 でdifflogic互換
        self._use_16gate = (parameterization == 'softmax' and n_inputs == 2)

        # ゲート種ロジットマスク(先行実験)。除外IDのロジットへ−infを加算する。
        # 既定Noneではbufferを作らず_masked_weights()がself.weightsを素通しするので、
        # 既存挙動は乱数消費も含めbitwise不変(後方互換の最重要契約)。
        self.gate_mask: tuple[int, ...] = ()
        if gate_mask is not None:
            if not self._use_16gate:
                raise ValueError(
                    "gate_mask は n_inputs=2 かつ parameterization='softmax' "
                    f'(16ゲートモード)専用です: n_inputs={n_inputs}, '
                    f'parameterization={parameterization!r}'
                )
            ids = sorted({int(g) for g in gate_mask})
            if any(not 0 <= g <= 15 for g in ids):
                raise ValueError(f'gate_maskのゲートIDは0〜15の範囲: {ids}')
            if len(ids) >= 16:
                raise ValueError('gate_maskで16種すべてを除外することはできません')
            self.gate_mask = tuple(ids)
        if self.gate_mask:
            logit_mask = torch.zeros(16, device=device)
            logit_mask[list(self.gate_mask)] = float('-inf')
            self.register_buffer('gate_logit_mask', logit_mask)

        # 初期化方式(先行実験 S2)。Noneは従来どおり residual_init に従う
        if generator is not None and generator.device.type != 'cpu':
            raise ValueError(
                'generator は CPU Generator を渡してください'
                f'(指定: {generator.device})。デバイス非依存の決定性のため'
                'CPUで生成してからdeviceへ移す設計です'
            )
        if init_mode is not None:
            if init_mode not in self.INIT_MODES:
                raise ValueError(
                    f'init_mode は {self.INIT_MODES} のいずれか: {init_mode!r}'
                )
            if not self._use_16gate:
                raise ValueError(
                    "init_mode は n_inputs=2 かつ parameterization='softmax' "
                    f'(16ゲートモード)専用です: n_inputs={n_inputs}, '
                    f'parameterization={parameterization!r}'
                )
            if residual_init:
                raise ValueError(
                    'init_mode と residual_init は同時指定できません'
                    "（residual_init=True は init_mode='ri' と等価）"
                )
        if init_scale <= 0:
            raise ValueError(f'init_scale は正の値: {init_scale}')
        self.init_mode = init_mode
        self.init_scale = float(init_scale)

        # WARP設定
        self.tau_warp = 1.0    # WARP温度パラメータ
        self.gumbel = True     # Gumbel-Sigmoid smoothing

        # 学習パラメータ: パラメータ化方式に応じてサイズが異なる
        if parameterization == 'warp':
            # WARP: WH係数θ（制約なし実数）→ (out_dim, 2^n_inputs)
            self.weights = nn.Parameter(
                torch.zeros(out_dim, self.n_entries, device=device)
            )
            # Residual Init: pass-through関数に高確率を割り当て（深層化安定化）
            with torch.no_grad():
                p = 0.9
                logit_p = math.log(p / (1.0 - p))  # σ⁻¹(p) ≈ 2.197
                # WH butterfly順序でindex 2^{n-1} がs_{n-1}（最後の入力）に対応
                # θ < 0 でf(x)≈x_{n-1}（pass-through）
                self.weights[:, 2 ** (n_inputs - 1)] = -self.tau_warp * logit_p
            # 度数キャップ: popcount(index)>dのWH係数を恒等的に0へ
            # マスク=PolyLUT式(hard表は次数d PTF=多項式閾値関数に制限される。
            # Boolean関数のフーリエ次数制限ではない点に注意)。
            # soft/hard/eval全経路で有効(勾配もマスクされ高次係数は学習されない)
            if wh_degree_cap is not None:
                assert 0 <= wh_degree_cap <= n_inputs, wh_degree_cap
                idx = torch.arange(self.n_entries, device=device)
                deg = torch.zeros_like(idx)
                for j in range(n_inputs):
                    deg += (idx >> j) & 1
                self.register_buffer(
                    'wh_degree_mask',
                    (deg <= wh_degree_cap).to(torch.float32),
                )
        elif parameterization == 'iwp':
            # IWP: 入力ごとに1パラメータ → (out_dim, n_inputs)
            self.weights = nn.Parameter(
                _randn_init((out_dim, n_inputs), device, generator)
            )
            if residual_init:
                import warnings
                warnings.warn(
                    'residual_initはiwpパラメータ化では未対応(無視されます)',
                    stacklevel=2,
                )
        elif parameterization == 'raw':
            # Raw: BinaryBrain方式。重みを直接[0,1]で保持
            # BinaryBrainと同じ初期化: Normal(mean=0.5, std=0.01)
            if generator is None:
                raw_w = torch.empty(
                    out_dim, self.n_entries, device=device
                ).normal_(0.5, 0.01)
            else:
                raw_w = torch.empty(out_dim, self.n_entries).normal_(
                    0.5, 0.01, generator=generator
                ).to(device)
            self.weights = nn.Parameter(raw_w)
        elif self._use_16gate:
            # softmax + N=2: 16ゲートモード（difflogic互換）
            self.weights = nn.Parameter(
                _randn_init((out_dim, 16), device, generator)
            )
            self.register_buffer(
                'gate_table', _build_gate_table_n2().to(device)
            )
            # init_mode=None は従来どおり residual_init（bool）に従う。
            # どのモードでも randn の消費は上の1回だけなので後方互換が保たれる
            mode = init_mode if init_mode is not None else (
                'ri' if residual_init else 'randn'
            )
            self._apply_init_mode(mode, generator)
        else:
            # softmax(N>2)/independent: エントリごとに1パラメータ → (out_dim, 2^n_inputs)
            self.weights = nn.Parameter(
                _randn_init((out_dim, self.n_entries), device, generator)
            )
            if residual_init:
                # Residual Init(N>2対応)。warpと同じく最後の入力
                # x_{n-1} のpass-throughへバイアスする(entryのbit n-1)。
                # - independent: ±5ロジット → sigmoidでhard表も厳密に
                #   pass-through(表現可能)
                # - softmax N>2: 該当2^(n-1)エントリへ+5(合計1制約のため
                #   厳密なpass-throughは不可能で、soft出力は
                #   x_{n-1}方向へのバイアスに留まる=2^(n-1)倍の減衰)
                # estimator='sin' では sin(5.0) が符号反転して破綻するため、
                # 論文 arXiv:2510.03250 §C.1.1 のΩ空間RI Ω~N(±μ,σ) を使う。
                with torch.no_grad():
                    e = torch.arange(self.n_entries, device=device)
                    mask = ((e >> (n_inputs - 1)) & 1).bool()
                    if estimator == 'sin':
                        # ρ(+μ)≈0.97, ρ(-μ)≈0.03（μ=1.2）で厳密な
                        # pass-throughに近い、heavy-tailな確率的初期化
                        sign = torch.where(mask, ri_mu, -ri_mu).to(
                            self.weights.dtype)
                        self.weights.copy_(
                            torch.randn_like(self.weights) * ri_sigma
                            + sign.unsqueeze(0)
                        )
                    else:
                        self.weights.zero_()
                        if parameterization == 'independent':
                            self.weights[:, mask] = 5.0
                            self.weights[:, ~mask] = -5.0
                        else:  # softmax N>2
                            self.weights[:, mask] = 5.0

        # 接続テーブル: (n_inputs, out_dim)。Tensor直渡しで明示配線も可
        # (SAT witness配線の転写・strictペア配線などに使用)
        if explicit_conn:
            idx = connections.to(torch.int64).to(device)
            assert tuple(idx.shape) == (n_inputs, out_dim), (
                f'明示indicesの形状 {tuple(idx.shape)} != '
                f'({n_inputs}, {out_dim})'
            )
            assert idx.numel() == 0 or (
                idx.min() >= 0 and idx.max() < in_dim), '配線が入力幅を超過'
            self.register_buffer('indices', idx.contiguous())
        elif connections == 'random':
            self.register_buffer(
                'indices',
                get_random_connections(in_dim, out_dim, n_inputs, device)
            )
        else:
            self.register_buffer(
                'indices',
                get_unique_connections(in_dim, out_dim, n_inputs, device)
            )

        # 逆引きインデックス（将来のCUDA backward_x最適化用）
        rev_starts, rev_flat_y, rev_flat_slot = build_reverse_index(self.indices, in_dim)
        self.register_buffer('rev_starts', rev_starts)
        self.register_buffer('rev_flat_y', rev_flat_y)
        self.register_buffer('rev_flat_slot', rev_flat_slot)

        # 配線学習（DWN論文§3.2、DWNLayerと同じ機構）
        # indices bufferは学習配線時も維持する（推論時フォールバック/reach-map互換のため）
        if wiring == 'learnable':
            from .dwn import LearnableMapping

            self.lm_tau = lm_tau
            self.learnable_mapping = LearnableMapping(
                in_dim, out_dim * n_inputs, tau=lm_tau,
            ).to(device)

            # identity接続: LearnableMappingの出力(out_dim*n_inputs,)を
            # 既存のindicesベースgather機構( x[:, indices.T.reshape(-1)] )に
            # そのまま通すためのidentity写像。indices[j, o] = o*n_inputs + j。
            identity = torch.arange(
                out_dim * n_inputs, device=device, dtype=torch.int64
            ).reshape(out_dim, n_inputs).T.contiguous()
            self.register_buffer('_identity_indices', identity)

        self.num_neurons = out_dim

    def _apply_gumbel(self, weights: torch.Tensor) -> torch.Tensor:
        """Gumbelノイズ注入（Mind the Gap, arXiv:2506.07500）。

        softmaxゲート選択ロジット w に Gumbel(0,1) ノイズ g_i = -log(-log u_i)
        を加算し温度 τ で割る（後段のsoftmaxで π_i = softmax((w_i+g_i)/τ) となる）。
        softmaxはカスタムFunction(LutFunction/CudaLutFunction)内部で計算されるため、
        ここでは (w+g)/τ を作って渡すだけでよい。g は定数（detach相当。requires_grad
        なしのrand由来）なので、Functionが返す ∂L/∂(noised) はautogradが
        `+g`・`/τ` を通して self.weights へ正しく連鎖する（backward改変不要）。

        gumbel_noise=False または eval時は weights をそのまま返す（乱数も消費せず
        bitwise不変。既存挙動の後方互換を保証する）。
        """
        if not (self.gumbel_noise and self.training):
            return weights
        u = torch.rand_like(weights).clamp(_GUMBEL_EPS, 1.0 - _GUMBEL_EPS)
        g = -torch.log(-torch.log(u))
        return (weights + g) / self.gumbel_tau

    def _apply_init_mode(
        self, mode: str, generator: torch.Generator | None,
    ) -> None:
        """16ゲートモードのロジット初期化（weights は既に N(0,1) で埋まっている）。

        'randn' 以外は N(0,1) の上に決定的な操作を重ねる。
        residual の加算量は self.init_scale（既定5.0）で、16ゲート・N(0,1)なら
        P(選択) ≈ e^α / (e^α + 15)（α=5 → ≈0.91 / α=1 → ≈0.15）。
        社内先行実験 model_iwp.py:79-80 の `w[:,3] += residual_init` に
        対応し、先行実験 Phase 3 の OP 腕は α=1.0（train_cifar_ab.py:255 の既定）。
        """
        alpha = self.init_scale
        # ゲート'A'（真理値表[0,1,0,1]）の位置を gate_table から確認する
        gt = self.gate_table
        target = torch.tensor([0.0, 1.0, 0.0, 1.0],
                              dtype=gt.dtype, device=gt.device)
        matches = (gt == target).all(dim=1).nonzero().flatten()
        assert matches.numel() == 1, (
            f'ゲートA[0,1,0,1]が一意に見つからない: {matches}'
        )
        a_idx = int(matches.item())
        assert a_idx == self.A_GATE_ID, (a_idx, self.A_GATE_ID)

        with torch.no_grad():
            if mode == 'randn':
                return
            if mode == 'uniform':
                self.weights.zero_()
                return
            if mode == 'ri':
                # 従来の residual_init: 全ロジット0 + op3 に init_scale
                self.weights.zero_()
                self.weights[:, a_idx] = alpha
                return
            if mode == 'ri_randn':
                self.weights[:, a_idx] += alpha
                return
            if mode == 'rri':
                trivial = set(self.TRIVIAL_GATE_IDS)
                cands = [g for g in self.allowed_gate_ids() if g not in trivial]
                if not cands:
                    raise ValueError(
                        "init_mode='rri' には非自明ゲート"
                        f'（{sorted(set(range(16)) - trivial)}）が'
                        '1種以上許可されている必要があります'
                        f'（gate_mask={list(self.gate_mask)}）'
                    )
                pick = torch.randint(
                    len(cands), (self.out_dim,), generator=generator,
                )
                cand_t = torch.tensor(cands, dtype=torch.int64)
                chosen = cand_t[pick].to(self.weights.device)
                rows = torch.arange(self.out_dim, device=self.weights.device)
                self.weights[rows, chosen] += alpha
                return
        raise ValueError(f'未知の init_mode: {mode!r}')

    def allowed_gate_ids(self) -> tuple[int, ...]:
        """16ゲートモードで学習に使えるゲートID(gate_maskで除外されていないもの)。"""
        if not self._use_16gate:
            raise ValueError('allowed_gate_idsは16ゲートモード専用')
        excluded = set(self.gate_mask)
        return tuple(g for g in range(16) if g not in excluded)

    def _masked_weights(self) -> torch.Tensor:
        """ゲート種マスク適用後のロジット(マスク無しならself.weightsそのもの)。

        除外ゲートに−infを加えるので softmax の確率は厳密に0、
        argmax は許可集合からしか選ばず、勾配も厳密に0になる
        (少なくとも1種は許可されているので max は有限でNaNは出ない)。
        """
        mask = getattr(self, 'gate_logit_mask', None)
        if mask is None:
            return self.weights
        return self.weights + mask

    def _warp_weights(self) -> torch.Tensor:
        """WARP係数(度数キャップがあればマスク済み)を返す。"""
        mask = getattr(self, 'wh_degree_mask', None)
        if mask is not None:
            return self.weights * mask
        return self.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim) 入力テンソル、値は[0,1]

        Returns:
            (batch, out_dim) 出力テンソル
        """
        if not torch.compiler.is_compiling():
            assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # gate_table: 16ゲートモードでのみ使用
        gate_table = getattr(self, 'gate_table', None)

        if self.wiring == 'learnable':
            # LearnableMappingで入力を(out_dim*n_inputs)個の仮想入力へ写像し、
            # 既存のindicesベースgather機構をidentity接続として再利用する。
            # mapped[:, o*n_inputs + j] が出力oのスロットjに対応するよう、
            # LutFunction系が行う x[:, indices.T.reshape(-1)] と同じ並びにする。
            mapped_x = self.learnable_mapping(x)
            gather_x = mapped_x
            gather_indices = self._identity_indices
        else:
            gather_x = x
            gather_indices = self.indices

        if self.training:
            # WARP×Hard/Gumbel-ST: エントリロジット経由の正しいhard表。
            # LutFunctionHardSTのargmaxはWH係数に適用すると単一minterm崩壊
            # するため、専用Functionへルーティングする。
            if (self.parameterization == 'warp'
                    and self.selection in ('hard_st', 'gumbel_st')):
                noise = None
                if self.selection == 'gumbel_st' and self.gumbel_scale > 0:
                    # logisticノイズ(Gumbel差)= 閾値化のGumbel-ST対応物
                    u = torch.rand_like(self.weights).clamp(1e-6, 1 - 1e-6)
                    noise = self.gumbel_scale * (u.log() - (1.0 - u).log())
                return LutFunctionWARPHardST.apply(
                    gather_x, self._warp_weights(), gather_indices,
                    self.n_inputs, self.tau_b, noise,
                )

            # Hard-ST: argmax forward + STE backward（CAGE論文方式）
            if self.selection == 'hard_st':
                return LutFunctionHardST.apply(
                    gather_x, self._masked_weights(), gather_indices,
                    self.n_inputs, gate_table, self.tau_b,
                )

            # Gumbel-ST: argmax(weights+G) forward + STE backward
            # （= weights+ノイズへのHard-STと厳密等価。ノイズは定数なので
            #    勾配はweightsへそのまま流れる）
            if self.selection == 'gumbel_st':
                w = self._masked_weights()
                if self.gumbel_scale > 0:
                    u = torch.rand_like(w)
                    g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
                    w = w + self.gumbel_scale * g
                return LutFunctionHardST.apply(
                    gather_x, w, gather_indices, self.n_inputs,
                    gate_table, self.tau_b,
                )

            # Gumbelノイズ注入（softmaxのみ。既定Off時は _masked_weights() を素通し）
            weights = self._apply_gumbel(self._masked_weights())

            # CUDA対応環境ではカーネル版を使用（メモリ大幅削減）
            # 学習配線時はrev_*バッファがfixed接続専用のため、Pythonフォールバックへ委譲
            if (
                self.wiring == 'fixed'
                and gather_x.is_cuda
                and cuda_available()
                # warpはWH基底+sigmoidでsoftmax系カーネルと数理が異なるため
                # 除外必須(含めるとCudaLutFunctionがθをsoftmax重みとして誤解釈)
                and self.parameterization not in ('iwp', 'raw', 'warp')
                # CUDAカーネルはsigmoid固定。sin推定器では黙って誤った
                # 勾配になるためPythonフォールバックへ落とす
                and self.estimator == 'sigmoid'
            ):
                if self.parameterization == 'independent':
                    return CudaLutFunctionIndependent.apply(
                        gather_x, weights, gather_indices, self.n_inputs,
                        self.rev_starts, self.rev_flat_y, self.rev_flat_slot,
                    )
                return CudaLutFunction.apply(
                    gather_x, weights, gather_indices, self.n_inputs,
                    self.rev_starts, self.rev_flat_y, self.rev_flat_slot,
                    gate_table,
                )

            # IWP専用CUDAカーネル分岐。上のexclusion tuple
            # ('iwp','raw','warp')に'iwp'を残したまま、ここで明示的に
            # 専用Functionへディスパッチする(softmax意味論のCudaLutFunctionへ
            # 誤って流れ込まないことをtupleとこの分岐の両方で保証する)。
            if (
                self.wiring == 'fixed'
                and gather_x.is_cuda
                and cuda_available()
                and self.parameterization == 'iwp'
                and not _python_path_forced('iwp')
            ):
                return CudaLutFunctionIWP.apply(
                    gather_x, weights, gather_indices, self.n_inputs,
                    self.rev_starts, self.rev_flat_y, self.rev_flat_slot,
                )

            # WARP: WH基底 + sigmoid。
            # ここに到達するのは selection='soft' の場合のみ（hard_st/gumbel_st
            # は本関数の前段で LutFunctionWARPHardST へ既にディスパッチ済みで
            # returnしている）。CUDA環境では融合カーネル
            # (warp_forward/backward_w/backward_x)へ明示的に分岐する。
            # 除外tuple('iwp','raw','warp')は変更しない
            # （上のCudaLutFunction/CudaLutFunctionIndependent分岐には
            # 到達させない設計を維持: TestWarpDispatchRegression参照）。
            if self.parameterization == 'warp':
                if (
                    self.wiring == 'fixed'
                    and gather_x.is_cuda
                    and cuda_available()
                    and not _python_path_forced('warp')
                ):
                    theta = self._warp_weights()
                    noise = None
                    if self.gumbel:
                        # Python版LutFunctionWARPのGumbel-Sigmoidノイズ生成と
                        # 厳密に同一のtorch.rand呼び出し（形状・順序）にする
                        # (RNG消費順の一致がparityテストの前提)
                        batch = gather_x.shape[0]
                        u = torch.rand(
                            2, batch, self.out_dim,
                            device=theta.device, dtype=theta.dtype,
                        )
                        u = u.clamp(1e-6, 1.0 - 1e-6)
                        noise = (
                            u[0].log() - (1.0 - u[0]).log()
                            - u[1].log() + (1.0 - u[1]).log()
                        )
                    return CudaLutFunctionWARP.apply(
                        gather_x, theta, gather_indices, self.n_inputs,
                        self.tau_warp, noise,
                        self.rev_starts, self.rev_flat_y, self.rev_flat_slot,
                    )
                return LutFunctionWARP.apply(
                    gather_x, self._warp_weights(), gather_indices,
                    self.n_inputs, self.tau_warp, self.gumbel,
                )

            # Pythonフォールバック（CPU/MPS/IWP/Raw）
            if self.parameterization == 'iwp':
                return LutFunctionIWP.apply(
                    gather_x, weights, gather_indices, self.n_inputs
                )
            if self.parameterization == 'raw':
                return LutFunctionRaw.apply(
                    gather_x, weights, gather_indices, self.n_inputs
                )
            if self.parameterization == 'independent':
                return LutFunctionIndependent.apply(
                    gather_x, weights, gather_indices, self.n_inputs,
                    self.estimator,
                )
            return LutFunction.apply(
                gather_x, weights, gather_indices, self.n_inputs, gate_table,
            )

        # 推論時: 離散化（カスタムautograd不要）
        gathered = gather_x[:, gather_indices.T.reshape(-1)].reshape(
            gather_x.shape[0], self.out_dim, self.n_inputs
        )

        if self.parameterization == 'iwp':
            # IWP: sigmoid→round→リテラルの積
            p = (torch.sigmoid(self.weights) > 0.5).to(x.dtype)
            factors = p.unsqueeze(0) * gathered + (1.0 - p.unsqueeze(0)) * (1.0 - gathered)
            return factors.prod(dim=-1)

        if self.parameterization == 'warp':
            # WARP: WH基底でロジット計算 → 閾値0で離散化
            wh_basis = _compute_wh_basis_butterfly(gathered, self.n_inputs)
            logit = (wh_basis * self._warp_weights().unsqueeze(0)).sum(-1)
            return (logit > 0).to(x.dtype)

        basis = _compute_basis_butterfly(gathered, self.n_inputs)

        if self.parameterization == 'raw':
            # Raw: BinaryBrain方式。推論時も連続値の重みを使用
            # （ハードウェア展開時のみ離散化する）
            w = self.weights
        elif self.parameterization == 'independent':
            # 独立確率: ρ(Ω) > 0.5 で離散化（各エントリ独立、論文IWPの丸め）
            w = (estimator_rho(self.weights, self.estimator) > 0.5).to(x.dtype)
        elif gate_table is not None:
            # 16ゲートモード: argmaxで完全Boolean関数を1つ選択
            # (gate_maskがあれば除外ゲートは−infなので選ばれない)
            gate_idx = self._masked_weights().argmax(dim=-1)  # (out_dim,)
            w = gate_table[gate_idx]                 # (out_dim, 4) — binary truth table
        else:
            # N>2 softmax: 確率>0.5のエントリのみ1に丸めて離散化
            # （Σ=1のため1になるのは高々1エントリ = 単一mintermかFALSE）。
            # このパラメータ化はhard LUTの表現力が構造的に乏しく
            # 離散化ギャップが大きい。HW展開にはindependent/iwp/warpを推奨
            w = (F.softmax(self.weights, dim=-1) > 0.5).to(x.dtype)

        return torch.einsum('boe,oe->bo', basis, w)

    def _compute_basis(self, gathered: torch.Tensor) -> torch.Tensor:
        """テスト互換用のpublicインタフェース"""
        return _compute_basis_butterfly(gathered, self.n_inputs)

    def extra_repr(self):
        param_str = self.parameterization
        if self._use_16gate:
            param_str += '/16gate'
        if self.parameterization == 'warp':
            param_str += f'/τ={self.tau_warp}'
            if self.gumbel:
                param_str += '/gumbel'
        if self.parameterization == 'independent':
            param_str += f', est={self.estimator}'
        sel_str = f', sel={self.selection}' if self.selection != 'soft' else ''
        gumbel_str = (
            f', gumbel_noise(τ={self.gumbel_tau})' if self.gumbel_noise else ''
        )
        wiring_str = (
            f', wiring={self.wiring}/τ={self.lm_tau}'
            if self.wiring == 'learnable' else ''
        )
        mask_str = f', gate_mask={list(self.gate_mask)}' if self.gate_mask else ''
        init_str = (
            f', init={self.init_mode}(α={self.init_scale:g})'
            if self.init_mode is not None else ''
        )
        return (
            f'in={self.in_dim}, out={self.out_dim}, '
            f'n_inputs={self.n_inputs}, entries={self.n_entries}, '
            f'param={param_str}{sel_str}{gumbel_str}{wiring_str}'
            f'{mask_str}{init_str}'
        )
