"""LUTネットワークの「単調性」を測る指標(experiments配下のアダプタ)。

理論背景:
    GEVB(Clark et al. 2021)は「重み非負(=単調増加)」ネットワークで
    誤差ブロードキャスト更新の符号が勾配と一致することを証明した。
    LUTは反転関数を学習でき単調性が破れうるため、AP法(誤差ブロードキャスト型
    学習則)がCNN/深層で失敗する、という仮説の定量検証に用いる。

本モジュールは griffith_logic/ のコア実装を一切変更せず、
LutLayer の重み(buffer/parameter)を読み取るだけのアダプタとして実装する。

提供する3指標:
    1. monotonicity_violation_V — LUT単調度違反(連続値、層単位)
    2. polarity_consistency_C   — 極性一貫性(入力→出力経路の極性積の集計)
    3. MonotonicityRecorder      — 学習ループ用の記録フック
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LutConv2d, LutLayer
from griffith_logic.core.basis import _compute_wh_basis_butterfly

# 極性ラベル(mixed=非単調、+1=単調増加、-1=単調減少)
POLARITY_INCREASING = 1.0
POLARITY_DECREASING = -1.0
POLARITY_MIXED = 0.0


def collect_lut_layers(model: torch.nn.Module) -> list[LutLayer]:
    """モデル内の全 LutLayer を出現順に収集する。"""
    return [m for m in model.modules() if isinstance(m, LutLayer)]


def _entry_bits(n_inputs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """エントリindex e のビット分解 (2^n, n_inputs) を返す。

    ビット規約: e = Σ_j x_j * 2**j (入力0がLSB)。bits[e, j] = (e >> j) & 1。
    """
    n_entries = 2 ** n_inputs
    e = torch.arange(n_entries, device=device)
    j = torch.arange(n_inputs, device=device)
    return ((e.unsqueeze(1) >> j.unsqueeze(0)) & 1).to(dtype)


def _iwp_table(p: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """IWPパラメータ化の実効soft truth table。

    T[o, e] = Π_j (p[o,j] if bit(e,j)==1 else 1-p[o,j]) をベクトル化して計算。
    """
    # bits: (E, n) → (1, E, n)、p: (O, n) → (O, 1, n)
    bits = _entry_bits(n_inputs, p.device, p.dtype).unsqueeze(0)
    p_exp = p.unsqueeze(1)
    factors = bits * p_exp + (1.0 - bits) * (1.0 - p_exp)  # (O, E, n)
    return factors.prod(dim=-1)  # (O, E)


def _warp_table(layer: LutLayer, n_inputs: int) -> torch.Tensor:
    """WARPパラメータ化の実効soft truth table。

    2^n 個のbinaryコーナー入力に対しWH基底でロジットを組み、sigmoidを通す。
    WH基底のエントリ順は WARP forward と同一関数で計算するため W と整合する。
    """
    w = layer.weights.data  # (O, E)
    dtype = w.dtype
    # コーナー入力: (E_corner, 1, n) を batch=コーナー / out_dim=1 とみなす
    corners = _entry_bits(n_inputs, w.device, dtype)  # (E, n)、e=Σx_j2^j順
    gathered = corners.unsqueeze(1)  # (E, 1, n)
    wh = _compute_wh_basis_butterfly(gathered, n_inputs).squeeze(1)  # (E_corner, E_entry)
    logit = wh @ w.t()  # (E_corner, O)
    return torch.sigmoid(logit).t()  # (O, E_corner)


def lut_soft_table(layer: LutLayer, ap_mode: bool = False) -> torch.Tensor:
    """LUT層の実効soft truth table T ∈ [0,1]^(out_dim, 2^n) を計算する。

    エントリ順は e = Σ_j x_j * 2**j (入力0がLSB)で全指標に共通。

    Args:
        layer: 対象 LutLayer
        ap_mode: True なら AP学習器のモデル(weightsを生の[0,1]確率として扱う)

    Returns:
        (out_dim, 2^n_inputs) のテンソル。16ゲート時は (out_dim, 4)。
    """
    w = layer.weights.data
    n = layer.n_inputs

    # AP法: sigmoid等を通さず weights をそのまま[0,1]確率として使う
    if ap_mode:
        return w.clamp(0.0, 1.0)

    param = layer.parameterization
    if param == 'independent':
        return torch.sigmoid(w)
    if param == 'raw':
        return w.clamp(0.0, 1.0)
    if param == 'softmax':
        if getattr(layer, '_use_16gate', False):
            # 16ゲート: softmax重み × 真理値表(16,4) → (out_dim, 4)
            probs = torch.softmax(w, dim=-1)
            return probs @ layer.gate_table.to(w.dtype)
        return torch.softmax(w, dim=-1)
    if param == 'iwp':
        return _iwp_table(torch.sigmoid(w), n)
    if param == 'warp':
        return _warp_table(layer, n)
    raise ValueError(f'未知のパラメータ化: {param}')


def _pair_indices(n_entries: int, n_inputs: int, device: torch.device):
    """各入力jについて (e0, e1) ペアのインデックスを生成するジェネレータ。

    e0: bit_j==0 のエントリ集合、e1 = e0 | (1<<j)(他ビットは同一)。
    """
    e_all = torch.arange(n_entries, device=device)
    for j in range(n_inputs):
        bit = 1 << j
        e0 = e_all[(e_all & bit) == 0]  # (2^(n-1),)
        e1 = e0 | bit
        yield j, e0, e1


def monotonicity_violation_V(layer: LutLayer, ap_mode: bool = False) -> float:
    """LUT単調度違反 V(連続指標、層単位)。

    各入力変数jを0→1にしたとき出力が減少するペアの違反量を集計する。
        violation = Σ_pairs max(0, T[o,e0] − T[o,e1])
    (増加すべきなのに減少している量)を
    (ニューロン数 × 入力数 × 2^(n-1)) で割った正規化平均を返す。

    注意: この V は「単調増加からの逸脱」を測る。単調"減少"(NOT的)なLUTも
    V>0 になる。減少単調と非単調の区別は極性指標 C 側で行う。
    V=0 は完全単調増加を意味する。
    """
    t = lut_soft_table(layer, ap_mode)  # (O, E)
    n = layer.n_inputs
    n_entries = t.shape[1]

    total_violation = 0.0
    total_count = 0
    for _j, e0, e1 in _pair_indices(n_entries, n, t.device):
        # 増加すべき(e0<e1方向)のに減少している量
        violation = torch.clamp(t[:, e0] - t[:, e1], min=0.0)
        total_violation += float(violation.sum().item())
        total_count += violation.numel()

    return total_violation / total_count if total_count > 0 else 0.0


def mean_monotonicity_violation_V(
    lut_layers: list[LutLayer], ap_mode: bool = False
) -> float:
    """モデル全体の単調度違反 V の層平均。"""
    if not lut_layers:
        return 0.0
    return sum(monotonicity_violation_V(l, ap_mode) for l in lut_layers) / len(lut_layers)


def _table_polarity(
    t: torch.Tensor, n_inputs: int, eps: float = 1e-3
) -> torch.Tensor:
    """soft truth table T (rows, 2^n) から各(行, 入力j)の極性を分類する。

    - 全ペアで T[e0] ≤ T[e1]+eps  → +1(増加)
    - 全ペアで T[e0] ≥ T[e1]−eps  → −1(減少)
    - それ以外                    →  0(mixed=非単調)

    定数(無関係)入力は増加条件を満たすため +1 に分類する(仕様順序に従う)。
    LutLayer(ニューロン単位)とLutConv2d(ゲート単位)で共通に使う。
    """
    n_entries = t.shape[1]
    rows = t.shape[0]

    polarity = torch.zeros(rows, n_inputs, device=t.device)
    for j, e0, e1 in _pair_indices(n_entries, n_inputs, t.device):
        t0 = t[:, e0]
        t1 = t[:, e1]
        increasing = ((t1 - t0) >= -eps).all(dim=-1)  # T0 ≤ T1+eps
        decreasing = ((t0 - t1) >= -eps).all(dim=-1)  # T0 ≥ T1−eps
        col = torch.full((rows,), POLARITY_MIXED, device=t.device)
        # 増加条件を優先(定数入力は増加扱い)
        col = torch.where(decreasing & ~increasing,
                          torch.full_like(col, POLARITY_DECREASING), col)
        col = torch.where(increasing,
                          torch.full_like(col, POLARITY_INCREASING), col)
        polarity[:, j] = col
    return polarity


def layer_polarity(
    layer: LutLayer, ap_mode: bool = False, eps: float = 1e-3
) -> torch.Tensor:
    """各(ニューロン, 入力j)の極性を分類し (out_dim, n_inputs) テンソルで返す。

    - 全ペアで T[e0] ≤ T[e1]+eps  → +1(増加)
    - 全ペアで T[e0] ≥ T[e1]−eps  → −1(減少)
    - それ以外                    →  0(mixed=非単調)

    定数(無関係)入力は増加条件を満たすため +1 に分類する(仕様順序に従う)。
    """
    return _table_polarity(lut_soft_table(layer, ap_mode), layer.n_inputs, eps)


def polarity_consistency_C(
    model: torch.nn.Module,
    lut_layers: list[LutLayer],
    n_paths: int = 1000,
    ap_mode: bool = False,
    eps: float = 1e-3,
    seed: int | None = None,
) -> dict:
    """極性一貫性 C(経路指標)。

    出力層のニューロンから開始し、各層で入力スロットをランダム選択して
    indices[s, j] を辿り前層ニューロンへ向かう経路をサンプリングする。
    経路上の(LUT, 選択入力)の極性の積を経路極性とし、mixed辺を1つでも
    含む経路を「mixed経路」とする。

    各層 l を開始層として n_paths 本サンプリングし、層別に
    {frac_positive, frac_negative, frac_mixed} を集計して返す。

    Returns:
        {'per_layer': {l: {...}}, 'mean_frac_positive': ..., 'mean_frac_mixed': ...}
    """
    device = lut_layers[0].weights.device
    polarities = [layer_polarity(l, ap_mode, eps) for l in lut_layers]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    per_layer: dict[int, dict] = {}
    for l in range(len(lut_layers)):
        out_dim = lut_layers[l].out_dim
        # 開始ニューロン(層lの出力)をランダムサンプリング
        current = torch.randint(
            0, out_dim, (n_paths,), generator=generator, device=device
        )
        product = torch.ones(n_paths, device=device)
        has_mixed = torch.zeros(n_paths, dtype=torch.bool, device=device)

        # 層lから0まで逆向きに辿る
        for ll in range(l, -1, -1):
            layer = lut_layers[ll]
            n_in = layer.n_inputs
            slots = torch.randint(
                0, n_in, (n_paths,), generator=generator, device=device
            )
            edge_pol = polarities[ll][current, slots]  # (n_paths,)
            has_mixed |= (edge_pol == POLARITY_MIXED)
            product = product * edge_pol
            # 前層ニューロンへ: indices 形状 (n_inputs, out_dim)
            current = layer.indices[slots, current]

        # mixed経路は product に 0 が乗るため +1/−1 判定から自然に外れる
        frac_positive = float((product > 0.5).float().mean().item())
        frac_negative = float((product < -0.5).float().mean().item())
        frac_mixed = float(has_mixed.float().mean().item())
        per_layer[l] = {
            'frac_positive': frac_positive,
            'frac_negative': frac_negative,
            'frac_mixed': frac_mixed,
        }

    mean_positive = sum(v['frac_positive'] for v in per_layer.values()) / len(per_layer)
    mean_mixed = sum(v['frac_mixed'] for v in per_layer.values()) / len(per_layer)
    return {
        'per_layer': per_layer,
        'mean_frac_positive': mean_positive,
        'mean_frac_mixed': mean_mixed,
    }


# ============================================================
# CNN(CDD-LGN / LutConv2d)対応
# ============================================================

def collect_lut_conv_layers(model: torch.nn.Module) -> list[LutConv2d]:
    """モデル内の全 LutConv2d を出現順に収集する。"""
    return [m for m in model.modules() if isinstance(m, LutConv2d)]


def conv_soft_tables(conv_layer: LutConv2d, ap_mode: bool = False) -> torch.Tensor:
    """LutConv2d の全ゲートの実効soft truth table (G, 2^n) を計算する。

    weights (out_channels, n_internal, D) を (G, D) に reshape し
    (G = out_channels × n_internal、各行が1ゲート)、各ゲートに
    lut_soft_table と同じパラメータ化別式を適用する。
    conv は softmax でも per-entry(LutLayer の16ゲート分岐は無い)。

    ビット規約は既存と共通: e = Σ_j x_j·2^j(入力0=LSB)。

    Args:
        conv_layer: 対象 LutConv2d
        ap_mode: True なら AP学習器の重み扱い(生値を[0,1]へclamp)に合わせる

    Returns:
        (G, 2^n_inputs) のテンソル。iwp では sigmoid→リテラル積で 2^n へ展開。

    注意: iwp では weights 最終次元が n_inputs のため、ap_mode=True(clamp)は
    LutLayer.lut_soft_table と同様に (G, n_inputs) を返す(V算出には非対応)。
    実運用の AP-CNN は independent 固定のためこの分岐は通らない。
    """
    w = conv_layer.weights.data
    n = conv_layer.n_inputs
    w_flat = w.reshape(-1, w.shape[-1])  # (G, D)

    # AP法: sigmoid等を通さず weights をそのまま[0,1]確率として使う
    if ap_mode:
        return w_flat.clamp(0.0, 1.0)

    param = conv_layer.parameterization
    if param == 'independent':
        return torch.sigmoid(w_flat)
    if param == 'softmax':
        return torch.softmax(w_flat, dim=-1)
    if param == 'iwp':
        return _iwp_table(torch.sigmoid(w_flat), n)
    raise ValueError(f'未知のパラメータ化: {param}')


def conv_monotonicity_violation_V(
    conv_layer: LutConv2d, ap_mode: bool = False
) -> float:
    """LutConv2d の単調度違反 V(連続指標、モジュール単位)。

    conv_soft_tables で得た (G, 2^n) に monotonicity_violation_V と同式を適用し、
    (ゲート数 × 入力数 × 2^(n-1)) で正規化した平均を返す。V=0 は完全単調増加。
    """
    t = conv_soft_tables(conv_layer, ap_mode)  # (G, E)
    n = conv_layer.n_inputs
    n_entries = t.shape[1]

    total_violation = 0.0
    total_count = 0
    for _j, e0, e1 in _pair_indices(n_entries, n, t.device):
        violation = torch.clamp(t[:, e0] - t[:, e1], min=0.0)
        total_violation += float(violation.sum().item())
        total_count += violation.numel()

    return total_violation / total_count if total_count > 0 else 0.0


def _fractions_from_polarity(pol: torch.Tensor) -> dict:
    """極性テンソルから {frac_positive, frac_negative, frac_mixed} を集計する。"""
    if pol.numel() == 0:
        return {'frac_positive': 0.0, 'frac_negative': 0.0, 'frac_mixed': 0.0}
    return {
        'frac_positive': float((pol == POLARITY_INCREASING).float().mean().item()),
        'frac_negative': float((pol == POLARITY_DECREASING).float().mean().item()),
        'frac_mixed': float((pol == POLARITY_MIXED).float().mean().item()),
    }


def conv_layer_polarity_fractions(
    conv_layer: LutConv2d, ap_mode: bool = False, eps: float = 1e-3
) -> dict:
    """LutConv2d のゲート単位極性分布 {frac_positive, frac_negative, frac_mixed}。

    各ゲート×入力j を +1/−1/mixed に分類(基準は layer_polarity と同一)し、
    モジュール全体で割合を集計する。

    注: Dense の経路指標 C と異なり、ここでは木の経路積 C は計算しない。
    conv は各ツリーが全空間位置で共有され経路が膨大なため、
    ゲート単位(内部ノード単位)の極性分布で代替する。
    """
    t = conv_soft_tables(conv_layer, ap_mode)
    pol = _table_polarity(t, conv_layer.n_inputs, eps)  # (G, n_inputs)
    return _fractions_from_polarity(pol)


def record_model_monotonicity(
    model: torch.nn.Module, ap_mode: bool = False, eps: float = 1e-3
) -> dict:
    """モデル内の全 LutConv2d / LutLayer を出現順に走査し単調性を記録する。

    各モジュールについて V と極性割合を計算し、モデル全体の平均を付与する。
    Conv は conv_monotonicity_violation_V / conv_layer_polarity_fractions、
    Dense(LutLayer)は monotonicity_violation_V / layer_polarity を使う。

    Returns:
        {
          'per_module': [
            {'type': 'conv'|'dense', 'V': float,
             'frac_positive': float, 'frac_negative': float, 'frac_mixed': float},
            ...
          ],
          'V_mean': 全モジュールの V 平均,
          'mixed_mean': 全モジュールの frac_mixed 平均,
        }
    """
    per_module: list[dict] = []
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LutConv2d):
                v = conv_monotonicity_violation_V(m, ap_mode)
                fractions = conv_layer_polarity_fractions(m, ap_mode, eps)
                per_module.append({'type': 'conv', 'V': v, **fractions})
            elif isinstance(m, LutLayer):
                v = monotonicity_violation_V(m, ap_mode)
                fractions = _fractions_from_polarity(layer_polarity(m, ap_mode, eps))
                per_module.append({'type': 'dense', 'V': v, **fractions})

    if per_module:
        v_mean = sum(x['V'] for x in per_module) / len(per_module)
        mixed_mean = sum(x['frac_mixed'] for x in per_module) / len(per_module)
    else:
        v_mean = 0.0
        mixed_mean = 0.0

    return {'per_module': per_module, 'V_mean': v_mean, 'mixed_mean': mixed_mean}


class MonotonicityRecorder:
    """学習ループ用の単調性記録フック。

    interval ステップごとに V(層別+全体平均)と C(層別)を計算して dict を返し、
    内部履歴 self.history に追記する。それ以外のステップでは None を返す。

    V/C 計算は torch.no_grad かつ既存 weights の読み取りのみで行い、
    interval を粗く(既定10)取ることで学習オーバーヘッドを抑える。
    wandb 等へのログは呼び出し側の責務(本クラスはログAPI非依存)。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lut_layers: list[LutLayer],
        ap_mode: bool = False,
        interval: int = 10,
        n_paths: int = 1000,
    ):
        self.model = model
        self.lut_layers = lut_layers
        self.ap_mode = ap_mode
        self.interval = interval
        self.n_paths = n_paths
        self.history: list[dict] = []

    def maybe_record(self, step: int) -> dict | None:
        """step % interval == 0 のときだけ記録用 dict を返し、履歴に追記する。"""
        if step % self.interval != 0:
            return None

        with torch.no_grad():
            v_per_layer = [
                monotonicity_violation_V(l, self.ap_mode) for l in self.lut_layers
            ]
            v_mean = sum(v_per_layer) / len(v_per_layer) if v_per_layer else 0.0
            consistency = polarity_consistency_C(
                self.model, self.lut_layers,
                n_paths=self.n_paths, ap_mode=self.ap_mode, seed=step,
            )

        record = {
            'step': step,
            'V_per_layer': v_per_layer,
            'V_mean': v_mean,
            'C': consistency,
        }
        self.history.append(record)
        return record
