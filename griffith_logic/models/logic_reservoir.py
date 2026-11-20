"""D3: 論理リザバーコンピューティング(Logic Reservoir Computing)。

固定(凍結)ランダム再帰論理回路(リザバー)に系列を流し込み、readoutだけを
学習する。凍結ランダム論理回路リザバー自体はRBNリザバー(Snyder+ 2013等)や
LUTNet-RC(IEEE ICFPT 2023)に先行がある(監査で確認)。本実装の固有性は
微分可能論理ゲートFW(LutLayer)への統合・同一IRからのHDL出力経路・他部品との
学習則共有にある。
超低コスト学習(巨大ランダム回路+軽量readout)という本ライブラリの物語と直結する。

echo state相当の性質(fading memory: 過去入力の影響が時間とともに減衰する)は、
以下2点で担保する:
    1. リザバーを **hard化した純ブール回路** として走らせ、状態を{0,1}に離散化
       (連続値の発散を持たない有界な状態空間)
    2. leaky統合(leak<1.0)で `s_t = leak*new + (1-leak)*s_{t-1}` の低域通過を掛け、
       過去状態への上書き速度を落として摂動を減衰させる

リザバーは LutLayer 群(生成後 requires_grad=False で凍結)。学習対象は readout のみ。
リザバーは勾配不要なので状態計算は no_grad + detach で行う。
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..group_sum import GroupSum
from ..layers.lut import LutLayer

# readout='lut' 時、クラスあたりに割り当てる LUT ニューロン数(GroupSum のグループ幅)。
# 大きいほど readout の表現力が上がるが、学習対象パラメータも増える。
_READOUT_GROUP = 16


@dataclass
class LogicReservoirConfig:
    """LogicReservoir のハイパーパラメータ。"""

    input_size: int            # 入力ビット幅(符号化後)
    reservoir_size: int        # リザバー状態のビット幅
    num_classes: int
    depth: int = 2             # リザバー内 LutLayer 段数
    n_inputs: int = 2
    leak: float = 1.0          # 1.0=完全置換, <1.0 で soft leaky 統合(mux(leak,新,旧)相当)
    readout: str = 'group_sum'  # 'group_sum'(学習線形) | 'lut'(LutLayer+GroupSum)
    seed: int = 0              # リザバー回路の生成シード(再現性の要)


class LogicReservoir(nn.Module):
    """凍結ランダム論理回路リザバー+学習 readout。

    リザバー: LutLayer 群(パラメータは生成後 requires_grad=False で凍結。さらに
    hard化=真理値表を{0,1}に離散化した純ブール回路として動かすことを既定とする)。
    状態更新: s_t = reservoir(cat([x_t, s_{t-1}]))
    readout のみ学習対象。
    """

    READOUTS = ('group_sum', 'lut')

    def __init__(self, config: LogicReservoirConfig) -> None:
        super().__init__()
        if config.readout not in self.READOUTS:
            raise ValueError(
                f'readout は {self.READOUTS} のいずれか: {config.readout}'
            )
        if config.depth < 1:
            raise ValueError(f'depth は1以上: {config.depth}')
        self.config = config

        # --- リザバー(凍結ランダム論理回路)を seed で決定的に生成 ---
        # 大域 RNG を汚さないよう save/restore で包む。これによりリザバー回路は
        # config.seed のみに依存し(readout 初期化の ambient seed から独立)、
        # 同一 seed → 同一回路、異なる seed → 異なる回路を保証する。
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        self.reservoir = nn.ModuleList()
        in_dim = config.input_size + config.reservoir_size
        for _ in range(config.depth):
            layer = LutLayer(
                in_dim=in_dim,
                out_dim=config.reservoir_size,
                n_inputs=config.n_inputs,
                parameterization='softmax',
                connections='random',
            )
            self.reservoir.append(layer)
            in_dim = config.reservoir_size
        torch.random.set_rng_state(rng_state)

        # リザバーは凍結(勾配不要)かつ常に hard(eval)経路で走らせる
        for p in self.reservoir.parameters():
            p.requires_grad_(False)
        for layer in self.reservoir:
            layer.eval()

        # --- readout(学習対象) ---
        if config.readout == 'group_sum':
            # 標準リザバー計算の線形 readout。GroupSum(固定 0/1 ブロック和)の
            # 学習可能な一般化として、状態ビットからクラスロジットへの線形写像を学ぶ。
            self.readout: nn.Module = nn.Linear(config.reservoir_size, config.num_classes)
        else:
            # 'lut': 学習可能な LutLayer で状態を非線形に混合 → GroupSum でロジット化
            hidden = config.num_classes * _READOUT_GROUP
            self.readout = nn.Sequential(
                LutLayer(
                    in_dim=config.reservoir_size,
                    out_dim=hidden,
                    n_inputs=config.n_inputs,
                    parameterization='softmax',
                    connections='random',
                ),
                GroupSum(k=config.num_classes, tau=float(_READOUT_GROUP) ** 0.5),
            )

    # --- リザバーは train/eval に関わらず常に hard(eval)経路を維持する ---
    def train(self, mode: bool = True) -> 'LogicReservoir':
        super().train(mode)
        for layer in self.reservoir:
            layer.eval()
        return self

    def _step(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        """1タイムステップのリザバー状態更新。

        x_t: (B, input_size) の {0,1}、s_prev: (B, reservoir_size)。
        """
        h = torch.cat([x_t, s_prev], dim=-1)
        for layer in self.reservoir:
            h = layer(h)
        new = h  # (B, reservoir_size)
        leak = self.config.leak
        if leak >= 1.0:
            # 完全置換 = 純ブール状態(離散{0,1})
            return new
        # soft leaky 統合(mux(leak, 新, 旧)相当)。過去状態を低域通過させ摂動を減衰。
        return leak * new + (1.0 - leak) * s_prev

    def states(self, xs: torch.Tensor) -> torch.Tensor:
        """全時刻のリザバー状態を返す(解析・可視化用)。

        Args:
            xs: (B, T, input_size) の論理値系列
        Returns:
            (B, T, reservoir_size)。leak=1.0 では {0,1} の離散状態。
        """
        if xs.dim() != 3 or xs.shape[-1] != self.config.input_size:
            raise ValueError(
                f'xs は (B, T, input_size={self.config.input_size}): '
                f'{tuple(xs.shape)}'
            )
        batch, seq_len, _ = xs.shape
        device = xs.device
        # 初期状態は 0(離散{0,1}の中立値)
        s = torch.zeros(batch, self.config.reservoir_size, device=device, dtype=xs.dtype)
        outs = []
        # リザバーは勾配不要。no_grad + detach で readout へ切り離す。
        with torch.no_grad():
            for t in range(seq_len):
                s = self._step(xs[:, t, :], s)
                outs.append(s)
        return torch.stack(outs, dim=1).detach()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """xs:(B,T,input_size) → (B,num_classes) logits(最終状態の readout)。"""
        final_state = self.states(xs)[:, -1, :]
        return self.readout(final_state)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """readout のみを返す(optimizer に渡す用)。"""
        return iter(self.readout.parameters())
