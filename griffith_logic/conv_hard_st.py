"""
CNN（LutConv2d）向けHard-ST実装。

CAGE論文 (Kim, 2026 "Align Forward, Adapt Backward") の
forward alignment原理をCDDLGNのツリー畳み込みに適用する。

既存のLutConv2dを変更せず、新しいクラスとして提供。

■ なぜCAGEはCNNに適用されてこなかったか

1. CAGE論文はフラットなLutLayer（1ニューロン=1ゲート選択）のみで検証。
   LutConv2dのツリー構造（1出力ch=tree_depthレベル×複数ゲート）は未検証。

2. ツリーでは各レベルで独立にargmax+STEを適用する必要があり、
   多レベルのbutterfly basis backwardを含むカスタムautogradが必要。

3. Hard-STはCUDAカーネル未対応で、フラットでもSoft-Mixの2倍遅い。

■ CNNへの適用が可能な理由

1. forward alignment原理はアーキテクチャ非依存。
   各レベルで独立にselection gap=0を達成できる。

2. ツリーの各レベルは独立したゲート選択であり、
   LutLayerのHard-STと同じ原理が各レベルに適用可能。

3. 勾配のレベル間伝播はbutterfly backpropで正確に計算可能。
   backward温度τ_bはCAGEスケジューラで適応的に制御。

■ 期待される効果

- Selection gap = 0（訓練forwardと推論forwardが同一）
- 検出モデルのmAP(hard)崩壊問題の根本的解決
- 収束速度の向上（フラットでは約2倍を確認済み）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import LutConv2d

# ============================================================
# ヘルパー関数
# ============================================================

def _compute_basis_with_intermediates(
    grouped: torch.Tensor, n_inputs: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Butterfly basis計算 + 各ステップの中間状態を返す。

    backwardのbutterfly逆伝播に中間状態が必要。
    """
    intermediates = []
    basis = torch.ones(
        *grouped.shape[:-1], 1,
        device=grouped.device, dtype=grouped.dtype,
    )
    for j in range(n_inputs):
        intermediates.append(basis)
        xj = grouped[..., j:j+1]
        basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)
    return basis, intermediates


def _butterfly_backward(
    grad_basis: torch.Tensor,
    grouped: torch.Tensor,
    intermediates: list[torch.Tensor],
    n_inputs: int,
) -> torch.Tensor:
    """
    Butterfly basisの逆伝播。

    basis = compute_basis(grouped) に対する grad_grouped を計算する。
    LutFunctionHardST (lut_layer.py) の逆伝播と同一のアルゴリズム。
    """
    grad_gathered = torch.zeros_like(grouped)
    grad_cur = grad_basis
    for j in reversed(range(n_inputs)):
        half = grad_cur.shape[-1] // 2
        grad_lo = grad_cur[..., :half]
        grad_hi = grad_cur[..., half:]
        xj = grouped[..., j:j+1]
        grad_gathered[..., j:j+1] = (
            (grad_hi - grad_lo) * intermediates[j]
        ).sum(dim=-1, keepdim=True)
        grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj
    return grad_gathered


# ============================================================
# カスタムAutograd関数
# ============================================================

class TreeHardSTFunction(torch.autograd.Function):
    """
    ツリー畳み込みのHard-ST autograd関数。

    Forward: 各レベルでargmax（推論と同一のhard selection）
    Backward: 各レベルでsoftmax(w/τ_b)によるSTE勾配近似

    LutFunctionHardST (lut_layer.py) のツリー版。
    フラット版との違い:
    - 複数レベルのゲート選択をループ処理
    - 各レベルのbutterfly backpropを逆順に適用
    - 入力勾配にはsoft weightsを使用（勾配安定性のため）
    """

    @staticmethod
    def forward(
        ctx,
        leaves: torch.Tensor,
        weights: torch.Tensor,
        n_inputs: int,
        tree_depth: int,
        n_entries: int,
        tau_b: float,
        gate_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            leaves: (BL, OC, n_leaves) リーフ活性値
            weights: (OC, n_internal, n_entries) ゲート重み。
                gate_table指定時は (OC, n_internal, 16)（softmax16g、
                n_entries=4固定）。
            n_inputs: ゲート入力数
            tree_depth: ツリーの深さ
            n_entries: 2^n_inputs（softmax16g時も4のまま=真理値表サイズ）
            tau_b: backward温度
            gate_table: softmax16g用の (16, n_entries) 真理値表。Noneなら
                旧来の4-simplex softmax（weights最終次元=n_entries）。
                指定時はweights最終次元=16（16ゲート全表現）で、
                argmaxで選んだ行をgate_table経由でn_entries次元へ写像する
                （LutFunctionHardSTの16gパスと同一契約、CDDLGN構造欠陥回避）。
        """
        current = leaves
        gate_offset = 0

        all_grouped = []
        level_info = []  # (gate_offset, n_groups)

        for _level in range(tree_depth):
            n_groups = current.shape[-1] // n_inputs
            grouped = current[..., :n_groups * n_inputs].reshape(
                *current.shape[:-1], n_groups, n_inputs
            )
            level_weights = weights[:, gate_offset:gate_offset + n_groups, :]

            # Basis計算（forwardではintermediatesは不要）
            basis = torch.ones(
                *grouped.shape[:-1], 1,
                device=grouped.device, dtype=grouped.dtype,
            )
            for j in range(n_inputs):
                xj = grouped[..., j:j+1]
                basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

            # Hard selection（推論と同一）
            if gate_table is not None:
                gate_idx = level_weights.argmax(-1)
                w_hard = gate_table[gate_idx]
            else:
                w_hard = F.one_hot(
                    level_weights.argmax(-1), n_entries,
                ).to(basis.dtype)
            current = (basis * w_hard.unsqueeze(0)).sum(dim=-1)

            all_grouped.append(grouped)
            level_info.append((gate_offset, n_groups))
            gate_offset += n_groups

        ctx.save_for_backward(*all_grouped, weights)
        ctx.n_inputs = n_inputs
        ctx.tree_depth = tree_depth
        ctx.n_entries = n_entries
        ctx.tau_b = tau_b
        ctx.level_info = level_info
        ctx.gate_table = gate_table

        return current.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        all_grouped = list(saved[:-1])
        weights = saved[-1]

        n_inputs = ctx.n_inputs
        tree_depth = ctx.tree_depth
        tau_b = ctx.tau_b
        level_info = ctx.level_info
        gate_table = ctx.gate_table

        # grad_currentの初期化（最終レベルの出力勾配）
        # 最終レベルのn_groups=1の場合: (BL, OC) → (BL, OC, 1)
        grad_current = grad_output.unsqueeze(-1)

        grad_weights = torch.zeros_like(weights)

        for level in reversed(range(tree_depth)):
            grouped = all_grouped[level]
            gate_offset, n_groups = level_info[level]
            level_weights = weights[:, gate_offset:gate_offset + n_groups, :]

            # Basisと中間状態を再計算
            basis, intermediates = _compute_basis_with_intermediates(
                grouped, n_inputs,
            )

            # STE: softmax(w / τ_b) で勾配近似（16g時は16次元のsoftmax）
            s = F.softmax(level_weights / tau_b, dim=-1)
            w_eff = s @ gate_table if gate_table is not None else s

            # 重み勾配
            # output[bl, oc, g] = Σ_e basis[bl,oc,g,e] * w_eff[oc,g,e]
            # → d/dw_eff[oc,g,e] = Σ_bl grad[bl,oc,g] * basis[bl,oc,g,e]
            grad_eff = torch.einsum('bog,boge->oge', grad_current, basis)

            if gate_table is not None:
                grad_s = grad_eff @ gate_table.T
            else:
                grad_s = grad_eff

            # Softmax Jacobian（温度τ_b込み）
            dot = (grad_s * s).sum(dim=-1, keepdim=True)
            grad_level_weights = s * (grad_s - dot) / tau_b
            grad_weights[:, gate_offset:gate_offset + n_groups, :] = (
                grad_level_weights
            )

            # 入力勾配（soft weightsを使用してbutterfly backprop）
            # LutFunctionHardSTと同様、入力勾配にはsoft weightsを使う
            # → 勾配が全ゲートに分散し、深いツリーでも安定
            grad_basis = torch.einsum('bog,oge->boge', grad_current, w_eff)

            grad_grouped = _butterfly_backward(
                grad_basis, grouped, intermediates, n_inputs,
            )

            # 前のレベルへの勾配（reshapeで次元を戻す）
            grad_current = grad_grouped.reshape(
                grad_grouped.shape[0], grad_grouped.shape[1], -1,
            )

        # grad_leaves: 最初のレベルのgrouped勾配をn_leaves次元に展開
        grad_leaves = grad_current

        return grad_leaves, grad_weights, None, None, None, None, None


# ============================================================
# CNN Hard-ST畳み込み層
# ============================================================

class LutConv2dHardST(LutConv2d):
    """
    Hard-ST版のLutConv2d。

    LutConv2dを継承し、訓練時のforward計算のみHard-STに差し替え。
    推論時はLutConv2dと同一（argmaxによるhard selection）。

    使い方:
        # LutConv2dの代わりに使用
        conv = LutConv2dHardST(3, 64, kernel_size=3, tree_depth=3, padding=1)

        # CAGEスケジューラと組み合わせ
        scheduler = CAGEConvScheduler(model)
        for batch in loader:
            loss = ...
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    SELECTIONS = ('hard_st', 'gumbel_st')

    def __init__(
        self,
        *args,
        tau_b: float = 1.0,
        selection: str = 'hard_st',
        gumbel_scale: float = 1.0,
        **kwargs,
    ):
        """
        LutConv2dと同じ引数 + tau_b/selection/gumbel_scale。

        parameterizationは'softmax'|'softmax16g'のみ対応
        （Hard-STはsoftmax系parameterizationでのみ動作。'softmax16g'は
        CDDLGNの4-simplex構造欠陥(パススルー/OR/XOR表現不能)を回避する
        16ゲート全表現。detection_model.py/segmentation_models.pyの
        既定'softmax16g'と揃える）。

        Args:
            tau_b: backward温度（既定1.0、スケジューラから更新される想定）
            selection: 'hard_st'（argmax forward、既定）|
                'gumbel_st'（argmax(weights+Gumbel(0,gumbel_scale))forward、
                Mind the Gap方式。backwardはhard_stと同一のSTE。
                gumbel_scale→0でhard_stに退化）
            gumbel_scale: selection='gumbel_st'時のGumbelノイズ振幅
        """
        param = kwargs.get('parameterization', 'softmax')
        if param not in ('softmax', 'softmax16g'):
            raise ValueError(
                'LutConv2dHardSTはsoftmax/softmax16g parameterizationのみ'
                f'対応。指定: {param}'
            )
        kwargs['parameterization'] = param
        assert selection in self.SELECTIONS, (
            f'selection は {self.SELECTIONS} のいずれか: {selection}'
        )
        super().__init__(*args, **kwargs)
        self.tau_b = tau_b
        self.selection = selection
        self.gumbel_scale = gumbel_scale

    def _tree_forward_train(self, leaves: torch.Tensor) -> torch.Tensor:
        """Hard-ST/Gumbel-STによる訓練時ツリー演算。"""
        weights = self.weights
        if self.selection == 'gumbel_st' and self.gumbel_scale > 0:
            # Gumbel-max: argmax(weights + Gumbel(0,1)*scale) は
            # 「weights+ノイズへのHard-ST」と厳密等価（layers/lut.pyの
            # 平坦版gumbel_stパスと同一パターン）。ノイズは定数扱いなので
            # 勾配はweightsへそのまま流れる。
            u = torch.rand_like(weights)
            g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
            weights = weights + self.gumbel_scale * g
        gate_table = getattr(self, 'gate_table', None)
        return TreeHardSTFunction.apply(
            leaves, weights,
            self.n_inputs, self.tree_depth, self.n_entries, self.tau_b,
            gate_table,
        )


# ============================================================
# CAGE スケジューラ（CNN用）
# ============================================================

class CAGEConvScheduler:
    """
    LutConv2dHardST用のCAGEスケジューラ。

    ネットワーク全体のゲート選択確信度に基づいてτ_bを適応的に調整:
    - 確信度低い（訓練初期）→ τ_b高い → 勾配を広く分散（探索）
    - 確信度高い（訓練後期）→ τ_b低い → 勾配を集中（活用）
    """

    def __init__(
        self,
        model: nn.Module,
        tau_max: float = 3.0,
        tau_min: float = 0.5,
        beta: float = 0.99,
    ):
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.beta = beta

        self.conv_layers = [
            m for m in model.modules() if isinstance(m, LutConv2dHardST)
        ]
        if not self.conv_layers:
            raise ValueError('モデルにLutConv2dHardSTが含まれていません')

        # ゲート候補数K（softmax16g時はweights最終次元=16、旧softmaxは
        # n_entries=2^n_inputs。weights.shape[-1]で両対応、平坦版
        # CAGESchedulerと同じ方針）
        self.K = self.conv_layers[0].weights.shape[-1]

        # EMA確信度の初期値: 1/K（均等分布 = 最低確信度）
        self.ema_confidence = 1.0 / self.K
        self.tau_b = tau_max

    @torch.no_grad()
    def step(self):
        """各訓練イテレーション後に呼び出し、τ_bを更新する。"""
        total_confidence = 0.0
        total_gates = 0
        for layer in self.conv_layers:
            # weights: (OC, n_internal, n_entries)
            probs = F.softmax(layer.weights, dim=-1)
            max_probs = probs.max(dim=-1).values
            total_confidence += max_probs.sum().item()
            total_gates += max_probs.numel()

        c = total_confidence / total_gates

        # EMA更新
        self.ema_confidence = (
            self.beta * self.ema_confidence + (1.0 - self.beta) * c
        )

        # τ_b計算（CAGE論文 Eq. 11）
        c_norm = (self.ema_confidence - 1.0 / self.K) / (1.0 - 1.0 / self.K)
        c_norm = max(0.0, min(1.0, c_norm))
        self.tau_b = self.tau_max - (self.tau_max - self.tau_min) * c_norm

        for layer in self.conv_layers:
            layer.tau_b = self.tau_b

    def state_dict(self):
        return {
            'ema_confidence': self.ema_confidence,
            'tau_b': self.tau_b,
        }

    def load_state_dict(self, state):
        self.ema_confidence = state['ema_confidence']
        self.tau_b = state['tau_b']
        for layer in self.conv_layers:
            layer.tau_b = self.tau_b
