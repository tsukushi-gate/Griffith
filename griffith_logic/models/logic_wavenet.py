"""LogicWaveNet: dilated causal convolution の論理ゲート化。

WaveNet(van den Oord et al., 2016)は dilated causal conv を積層し、各層を
**ゲート付き活性 z = tanh(Wf*x) ⊙ σ(Wg*x)** と residual/skip 接続で結ぶ自己回帰
モデルである。本モジュールはこれを微分可能論理ゲート(LutLayer)で置き換えた実装。
中心となる主張は:

    ゲート付き活性 z = filter ⊙ gate は、{0,1}論理値の上では
    z = AND(filter, gate) と本質的に同型であり、論理演算と親和的である。

これは D2(LogicLSTM/GRU)の「ゲーティング=MUX/ANDマスク」と同じ物語であり、
attention を持たない自己回帰モデルとして D1(LogicGPT)との比較軸になる。

新規性の限定形(監査、判定=🟡近接あり): 「論理ゲートによる音声生成」一般は
初ではない。最近接は GSoC 2024 の bytebeat difflogic 音声(Ian Clester)で、こちらは
無状態 f(t) 型・非自己回帰である。本実装が主張するのは「**WaveNet 構造(dilated causal
conv + ゲート活性=ビットAND)の論理ゲートへの写像**」に限られる。「世界初」とは書かない。

構成: token埋め込み(ブール緩和) → dilated causal conv ブロック × n_layers
      → skip集約(OR/XOR) → FP LMヘッド。forward(idx:(B,T) long) -> (B,T,vocab)。

因果性の要: dilated conv は各位置 t の入力を [x_{t-(K-1)d}, ..., x_{t-d}, x_t] に
限定し、左ゼロパディングで未来を一切見ない(gather+左pad方式。roll はラップ
アラウンドで因果性を壊すため不採用、unfold は明示 pad+slice より読みにくいため不採用)。
埋め込み・ヘッドはFP固定(周辺FP方針、LogicGPTと同じ)。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..blocks.structural import Add
from ..layers.lut import LutLayer
from .rddlgn import LogicEmbedding

# dilated conv の左パディング値。ブール0(=論理False)で埋める。
# soft/hard いずれのモードでも {0,1} を保ち、eval時のLUT入力の離散性を壊さない
# (0.5=中立値で埋めると hard eval時に層入力が非ブールになるため不可)。
_PAD_VALUE = 0.0


class _ConvLutStack(nn.Module):
    """位置共有の LutLayer を depth 段積んだ最小スタック(filter/gate の計算に使う)。

    in_dim(=kernel_size*d_channels) → d_channels → ... → d_channels(計 depth 段)。
    全時間位置で重み共有(1次元 conv と同じ)。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        depth: int,
        n_inputs: int,
        parameterization: str,
        connections: str,
    ) -> None:
        super().__init__()
        assert depth >= 1, f'depth は1以上: {depth}'
        self.layers = nn.ModuleList()
        prev = in_dim
        for _ in range(depth):
            self.layers.append(
                LutLayer(
                    in_dim=prev,
                    out_dim=out_dim,
                    n_inputs=n_inputs,
                    parameterization=parameterization,
                    connections=connections,
                )
            )
            prev = out_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class LogicWaveNetConfig:
    """LogicWaveNet の構成。

    受容野 = (kernel_size-1) * (2^n_layers - 1) + 1。kernel_size=2 のとき 2^n_layers。
    dilation は層ごとに 1,2,4,...,2^(n_layers-1) と倍増する。
    """

    vocab_size: int
    d_channels: int = 64        # 論理チャネル幅(埋め込みビット幅)
    n_layers: int = 6           # dilation 1,2,4,... と倍増(受容野=2^n_layers)
    kernel_size: int = 2        # conv のタップ数(現在位置+過去(K-1)タップ)
    n_inputs: int = 2           # LUTのファンイン(2/4/6)
    parameterization: str = 'softmax'
    skip_mode: str = 'or'       # 全層 skip の集約(or/xor)
    residual_mode: str = 'xor'  # residual マージ(or/xor)。情報保存の観点でxor既定
    conv_depth: int = 1         # filter/gate 1本あたりの LutLayer 段数
    connections: str = 'random'


class LogicWaveNet(nn.Module):
    """dilated causal convolution の論理ゲート化。

    各層 l(dilation d=2^l):
        concat_t = [x_{t-(K-1)d}, ..., x_{t-d}, x_t]        (幅 K*d_channels、左pad因果)
        f = filter_stack(concat)                            (幅 d_channels)
        g = gate_stack(concat)                              (幅 d_channels)
        z = f * g                                           soft AND(eval時は厳密ビットAND)
        x <- residual_merge(x, z)                           次層入力(幅 d_channels)
        skip <- z                                           全層で集める

    出力: skip_agg = skip_merge(全skip) → FP LMヘッド → (B,T,vocab)。
    埋め込み(ブール緩和)とヘッド(FP Linear)は LogicGPT と同じ周辺FP方針。
    """

    def __init__(self, config: LogicWaveNetConfig) -> None:
        super().__init__()
        assert config.skip_mode in ('or', 'xor'), config.skip_mode
        assert config.residual_mode in ('or', 'xor'), config.residual_mode
        assert config.kernel_size >= 2, f'kernel_size は2以上: {config.kernel_size}'
        assert config.n_layers >= 1, f'n_layers は1以上: {config.n_layers}'
        self.config = config
        c = config.d_channels

        # token埋め込み: Embedding → sigmoid →(eval時 Heaviside)のブール緩和
        self.embed = LogicEmbedding(config.vocab_size, c)

        # 層ごとの dilation(1,2,4,...,2^(n_layers-1))
        self.dilations = [2 ** i for i in range(config.n_layers)]

        stack_kwargs = dict(
            depth=config.conv_depth,
            n_inputs=config.n_inputs,
            parameterization=config.parameterization,
            connections=config.connections,
        )
        conv_in = config.kernel_size * c
        # filter/gate スタック(各層 K*c → c)
        self.filters = nn.ModuleList(
            [_ConvLutStack(conv_in, c, **stack_kwargs) for _ in self.dilations]
        )
        self.gates = nn.ModuleList(
            [_ConvLutStack(conv_in, c, **stack_kwargs) for _ in self.dilations]
        )
        # residual / skip の論理マージ
        self.residual_merges = nn.ModuleList(
            [Add(config.residual_mode) for _ in self.dilations]
        )
        self.skip_merge = Add(config.skip_mode)

        # LMヘッド(FP固定)。ブール skip 特徴 → 語彙ロジット
        self.head = nn.Linear(c, config.vocab_size)

    @property
    def receptive_field(self) -> int:
        """位置tのlogitsが依存する過去区間の長さ(= (K-1)*(2^n_layers-1)+1)。"""
        k = self.config.kernel_size
        return (k - 1) * (2 ** self.config.n_layers - 1) + 1

    def _dilated_concat(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """位置tごとに [x_{t-(K-1)d}, ..., x_{t-d}, x_t] を連結する。

        左ゼロパディングで未来を見ない(因果)。roll(ラップ)や右パディングは
        因果性を壊すため使わない。

        Args:
            x: (B,T,C)
            dilation: このタップの dilation d
        Returns:
            (B,T,K*C)  最後の次元は [最遠過去タップ ... 現在] の順に連結
        """
        kernel = self.config.kernel_size
        batch, seq_len, channels = x.shape
        taps = []
        for k in range(kernel):
            # k=kernel-1 が現在(shift=0)、k=0 が最遠過去(shift=(K-1)*d)
            shift = (kernel - 1 - k) * dilation
            if shift == 0:
                taps.append(x)
            elif shift >= seq_len:
                # 全位置がパディング領域(過去が存在しない)
                taps.append(x.new_full((batch, seq_len, channels), _PAD_VALUE))
            else:
                pad = x.new_full((batch, shift, channels), _PAD_VALUE)
                taps.append(torch.cat([pad, x[:, : seq_len - shift, :]], dim=1))
        return torch.cat(taps, dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """(B,T) long → (B,T,vocab) logits(次トークン予測)。"""
        batch, seq_len = idx.shape
        x = self.embed(idx)  # (B,T,C) ブール緩和

        skips = []
        for dilation, filt, gate, res_merge in zip(
            self.dilations, self.filters, self.gates, self.residual_merges,
            strict=True,
        ):
            concat = self._dilated_concat(x, dilation)  # (B,T,K*C)
            flat = concat.reshape(batch * seq_len, -1)
            f = filt(flat)
            g = gate(flat)
            z = (f * g).reshape(batch, seq_len, -1)  # soft AND(eval時は厳密ビットAND)
            x = res_merge([x, z])                     # residual(幅不変)
            skips.append(z)

        skip_agg = self.skip_merge(skips)  # (B,T,C)
        return self.head(skip_agg)         # (B,T,vocab)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """自己回帰生成。(B,T)→(B,T+max_new_tokens)。LogicGPT.generate と同一契約。

        greedy=True または temperature<=0 で argmax(決定的)。それ以外は
        temperature スケール後に top_k フィルタ→softmax→multinomial でサンプル
        する(generatorで再現可能)。soft/hard は呼び出し側の train()/eval() 状態に従う。
        決定的な生成には eval() を推奨する。
        """
        for _ in range(max_new_tokens):
            logits = self(idx)          # (B,T,vocab)
            logits = logits[:, -1, :]   # 末尾位置のみ (B,vocab)

            if greedy or temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # (B,1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < thresh, float('-inf'))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(
                    probs, num_samples=1, generator=generator,
                )  # (B,1)

            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す(wandbログ用)。"""
        counts = {
            'embed': sum(p.numel() for p in self.embed.parameters()),
            'filters': sum(p.numel() for p in self.filters.parameters()),
            'gates': sum(p.numel() for p in self.gates.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


__all__ = ['LogicWaveNet', 'LogicWaveNetConfig']
