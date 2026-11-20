"""
CAGE (Confidence-Adaptive Gate Exploration) スケジューラ。

CAGE論文 (Kim, 2026 "Align Forward, Adapt Backward") の
逆伝播温度τ_bの適応スケジューリングを実装する。

τ_bはネットワーク全体のゲート選択確信度に基づいて調整:
- 確信度低い（訓練初期）→ τ_b高い → 勾配を広く分散（探索）
- 確信度高い（訓練後期）→ τ_b低い → 勾配を集中（活用）
"""

import torch
import torch.nn.functional as F

from griffith_logic.conv_hard_st import LutConv2dHardST
from griffith_logic.lut_layer import LutLayer


class CAGEScheduler:
    """Hard-ST訓練用の適応温度スケジューラ。"""

    def __init__(
        self,
        model: torch.nn.Module,
        tau_max: float = 3.0,
        tau_min: float = 0.5,
        beta: float = 0.99,
    ):
        """
        Args:
            model: LutLayerを含むモデル
            tau_max: τ_bの上限（低確信度時、探索モード）
            tau_min: τ_bの下限（高確信度時、活用モード）
            beta: EMA係数（確信度の移動平均）
        """
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.beta = beta

        self.lut_layers = [
            m for m in model.modules() if isinstance(m, LutLayer)
        ]
        assert len(self.lut_layers) > 0, 'モデルにLutLayerが含まれていません'

        # ゲート候補数K（16ゲートモードなら16、それ以外はn_entries）
        first = self.lut_layers[0]
        self.K = first.weights.shape[-1]

        # EMA確信度の初期値: 1/K（均等分布 = 最低確信度）
        self.ema_confidence = 1.0 / self.K
        self.tau_b = tau_max

    @torch.no_grad()
    def step(self):
        """各訓練イテレーション後に呼び出し、τ_bを更新する。"""
        # 1. 各LutLayerのweightsからゲート選択確信度を計算
        total_confidence = 0.0
        total_neurons = 0
        for layer in self.lut_layers:
            probs = F.softmax(layer.weights, dim=-1)
            max_probs = probs.max(dim=-1).values
            total_confidence += max_probs.sum().item()
            total_neurons += max_probs.numel()

        # 2. ネットワーク平均確信度
        c = total_confidence / total_neurons

        # 3. EMA更新
        self.ema_confidence = self.beta * self.ema_confidence + (1.0 - self.beta) * c

        # 4. τ_b計算（論文 Eq. 11）
        c_norm = (self.ema_confidence - 1.0 / self.K) / (1.0 - 1.0 / self.K)
        c_norm = max(0.0, min(1.0, c_norm))
        self.tau_b = self.tau_max - (self.tau_max - self.tau_min) * c_norm

        # 5. 全LutLayerのτ_bを更新
        for layer in self.lut_layers:
            layer.tau_b = self.tau_b

    def state_dict(self):
        return {
            'ema_confidence': self.ema_confidence,
            'tau_b': self.tau_b,
        }

    def load_state_dict(self, state):
        self.ema_confidence = state['ema_confidence']
        self.tau_b = state['tau_b']


class TauAnnealScheduler:
    """Gumbel-ST/Hard-ST用の幾何アニールスケジューラ。

    Mind the Gap (arXiv 2506.07500) の温度アニーリングに対応する。
    τ_b(t) = τ_0 · (τ_min/τ_0)^(t/T) の幾何減衰。
    anneal_noise=Trueならgumbel_scaleも同じ比率で1→0へ減衰させ、
    訓練終盤はhard_st(決定的argmax)に漸近する。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tau_0: float = 1.0,
        tau_min: float = 0.1,
        total_steps: int = 6000,
        anneal_noise: bool = False,
        noise_0: float = 1.0,
    ):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.total_steps = max(1, total_steps)
        self.anneal_noise = anneal_noise
        self.noise_0 = noise_0
        self.t = 0

        self.lut_layers = [
            m for m in model.modules() if isinstance(m, LutLayer)
        ]
        assert len(self.lut_layers) > 0, 'モデルにLutLayerが含まれていません'
        self.tau_b = tau_0
        self._apply()

    def _apply(self) -> None:
        for layer in self.lut_layers:
            layer.tau_b = self.tau_b
            if self.anneal_noise and hasattr(layer, 'gumbel_scale'):
                layer.gumbel_scale = self.noise_scale

    @property
    def noise_scale(self) -> float:
        frac = min(1.0, self.t / self.total_steps)
        return self.noise_0 * (1.0 - frac)

    def step(self) -> None:
        """各訓練イテレーション後に呼び出す。"""
        self.t += 1
        frac = min(1.0, self.t / self.total_steps)
        self.tau_b = self.tau_0 * (self.tau_min / self.tau_0) ** frac
        self._apply()

    def state_dict(self):
        return {'t': self.t, 'tau_b': self.tau_b}

    def load_state_dict(self, state):
        self.t = state['t']
        self.tau_b = state['tau_b']
        self._apply()


class TauAnnealConvScheduler:
    """LutConv2dHardST（CNN Hard-ST/Gumbel-ST、conv_hard_st.py）用の
    幾何アニールスケジューラ。TauAnnealSchedulerのConv版（W13a-1）。

    τ_b(t) = τ_0 · (τ_min/τ_0)^(t/T) の幾何減衰。anneal_noise=Trueなら
    gumbel_scaleも同じ比率で1→0へ減衰させ、訓練終盤はhard_st(決定的argmax)
    に漸近する。gumbel_st選択時はhardgap知見によりアニール必須。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tau_0: float = 1.0,
        tau_min: float = 0.1,
        total_steps: int = 6000,
        anneal_noise: bool = False,
        noise_0: float = 1.0,
    ):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.total_steps = max(1, total_steps)
        self.anneal_noise = anneal_noise
        self.noise_0 = noise_0
        self.t = 0

        self.conv_layers = [
            m for m in model.modules() if isinstance(m, LutConv2dHardST)
        ]
        assert len(self.conv_layers) > 0, (
            'モデルにLutConv2dHardSTが含まれていません'
        )
        self.tau_b = tau_0
        self._apply()

    def _apply(self) -> None:
        for layer in self.conv_layers:
            layer.tau_b = self.tau_b
            if self.anneal_noise:
                layer.gumbel_scale = self.noise_scale

    @property
    def noise_scale(self) -> float:
        frac = min(1.0, self.t / self.total_steps)
        return self.noise_0 * (1.0 - frac)

    def step(self) -> None:
        """各訓練イテレーション後に呼び出す。"""
        self.t += 1
        frac = min(1.0, self.t / self.total_steps)
        self.tau_b = self.tau_0 * (self.tau_min / self.tau_0) ** frac
        self._apply()

    def state_dict(self):
        return {'t': self.t, 'tau_b': self.tau_b}

    def load_state_dict(self, state):
        self.t = state['t']
        self.tau_b = state['tau_b']
        self._apply()


class ProgressiveWiringScheduler:
    """LearnableMappingの漸進配線ハード化スケジューラ。

    SparseLUT (arXiv 2503.12829) の2フェーズ訓練をDWN LearnableMappingへ適応:
        フェーズ1 (t < switch_at·T): forward_mode='soft'
            (x @ softmax(W/τ_soft))。τ_softをτ_0→τ_minへ幾何減衰し、
            soft混合を徐々に選択へ収束させる(接続の不連続ジャンプ無し)
        フェーズ2 (t ≥ switch_at·T): forward_mode='hard'
            (従来のargmax+STE。推論と同一経路)

    hard選択(argmax)の配線学習がDLGN内で崩壊した対策レバー。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        total_steps: int,
        switch_at: float = 0.8,
        tau_0: float = 1.0,
        tau_min: float = 0.05,
    ):
        from griffith_logic.layers.dwn import LearnableMapping

        self.total_steps = max(1, total_steps)
        self.switch_step = int(self.total_steps * switch_at)
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.t = 0

        self.mappings = [
            m for m in model.modules() if isinstance(m, LearnableMapping)
        ]
        assert len(self.mappings) > 0, (
            'モデルにLearnableMappingが含まれていません'
            '(--wiring learnable と併用してください)'
        )
        self._apply()

    def _apply(self) -> None:
        if self.t < self.switch_step:
            frac = self.t / max(1, self.switch_step)
            tau = self.tau_0 * (self.tau_min / self.tau_0) ** frac
            for m in self.mappings:
                m.forward_mode = 'soft'
                m.tau_soft = tau
        else:
            for m in self.mappings:
                m.forward_mode = 'hard'

    def step(self) -> None:
        """各訓練イテレーション後に呼び出す。"""
        self.t += 1
        self._apply()

    def state_dict(self):
        return {'t': self.t}

    def load_state_dict(self, state):
        self.t = state['t']
        self._apply()
