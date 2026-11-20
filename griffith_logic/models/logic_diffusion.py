"""D6: LogicBinaryDiffusion(XORノイズ離散拡散の完全論理ゲート化)。

BDPM "Binary Diffusion Probabilistic Model"(Kinakh & Voloshynovskiy,
arXiv:2501.13915)が提案したXORノイズ拡散を、**デノイザ本体まで論理ゲート
(LutConv2d)で構成**した「完全論理ゲート拡散」。BDPMのデノイザは通常のCNN/
Transformerであり、ゲート化されているのはノイズ過程の定義のみ(BDPM 論文の
キー数式・末尾を参照)。本実装はデノイザを LogicUNet
(完全論理ゲート版U-Net)にすることで、前向き過程(XOR)・逆過程(デノイズ)・
デノイザ内部演算のすべてがブール回路で閉じた拡散モデルを構成する(BDPM との差分)。

BDPM準拠のキー数式:
    - 前向き過程(BDPM §XORノイズ過程): x_t = x_0 ⊕ z_t
      z_t は各ビットの反転を表す二値ノイズ。反転率は t の関数として [0, 0.5]
      を単調に動く(ガウス拡散の x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ε の離散版)。
    - 目的関数(BDPM §二値クロスエントロピー): デノイザが x_0 を予測する
      x_0予測型を採り、二値クロスエントロピー(BCEWithLogits)で学習する。

設計判断(先行研究との差分):
    1. デノイザ g_θ を LogicUNet(全段 LutConv2d)に置換。BDPMは通常NN。
    2. 逆過程は「x_0予測 → 次のタイムステップの反転率で再ノイズ(q_sample)」
       という祖先サンプリングのXOR版で構成する(下記 sample() のdocstring参照)。
    3. eval時、デノイザは純ブール回路(hard)として動き、q_sample/逆過程を
       通じて中間・最終表現が厳密に {0,1} を保つ。

本実装が扱うのは単一ビットプレーン(二値画像)。BDPMのMBPR(多ビットプレーン
線形重み付け)や条件付け y_ε は範囲外(本走で拡張)。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logic_unet import LogicUNet, LogicUNetConfig


@dataclass
class LogicDiffusionConfig:
    """LogicBinaryDiffusion のハイパーパラメータ。"""

    image_size: int = 28
    in_channels: int = 1
    timesteps: int = 16            # 拡散ステップ数 T
    flip_prob_max: float = 0.5     # t=T でのビット反転率(0.5=完全ノイズ)
    base_channels: int = 32
    depth: int = 2                 # デノイザ(LogicUNet)の down/up 段数
    n_inputs: int = 2
    parameterization: str = 'softmax'
    time_embed_bits: int = 8       # t の二値符号(thermometer)をデノイザ入力へconcat


class LogicBinaryDiffusion(nn.Module):
    """XORノイズ離散拡散(デノイザ=完全論理ゲート)。

    - forward(x_t, t): x_0 の予測ロジット (B, C, H, W) を返す(x_0予測型)。
    - q_sample(x0, t): 前向き過程 x_t = x0 ⊕ Bernoulli(flip_prob(t))。
    - loss(x0): ランダム t で q_sample→デノイザ→BCEWithLogits。
    - sample(n): 完全ノイズから逐次デノイズして {0,1} 画像を生成。
    """

    def __init__(self, config: LogicDiffusionConfig) -> None:
        super().__init__()
        if config.timesteps < 1:
            raise ValueError(f'timesteps は1以上: {config.timesteps}')
        if not 0.0 < config.flip_prob_max <= 0.5:
            raise ValueError(f'flip_prob_max は (0, 0.5]: {config.flip_prob_max}')
        if config.time_embed_bits < 1:
            raise ValueError(f'time_embed_bits は1以上: {config.time_embed_bits}')
        self.config = config

        # x_t のチャンネル複製数。LutConv2d は各ツリーが入力チャンネルを
        # ランダムに2本だけ参照する(input_channels_per_tree=2)ため、画像1chが
        # 多数の時刻符号chに埋もれると「ツリーが画素を全く見ない」事態が生じ、
        # 低ノイズでの複写すら学べない。画像chを時刻符号chと同程度まで複製して
        # 参照確率を確保する(前向き過程・損失には影響しない純粋な入力配線の工夫)。
        self.image_repeat = max(1, round(config.time_embed_bits / config.in_channels))

        # デノイザ: LogicUNet(全段 LutConv2d)。入力は x_t(複製後)と時刻符号
        # (time_embed_bits)を channel 次元で concat したもの。出力は x_0 予測
        # ロジット(in_channels チャンネル)。skip接続はデノイズに有利なため有効化。
        denoiser_in = config.in_channels * self.image_repeat + config.time_embed_bits
        self.denoiser = LogicUNet(LogicUNetConfig(
            in_channels=denoiser_in,
            base_channels=config.base_channels,
            depth=config.depth,
            n_inputs=config.n_inputs,
            parameterization=config.parameterization,
            out_channels=config.in_channels,
            use_skip=True,
        ))

    # ------------------------------------------------------------------
    # スケジュール・時刻符号
    # ------------------------------------------------------------------

    def _as_time_tensor(self, t, batch: int, device) -> torch.Tensor:
        """t(int / スカラ / (B,) テンソル)を (B,) の long テンソルへ正規化する。"""
        if isinstance(t, int):
            return torch.full((batch,), t, dtype=torch.long, device=device)
        t_tensor = torch.as_tensor(t, device=device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(batch)
        return t_tensor.to(torch.long)

    def flip_prob(self, t) -> torch.Tensor:
        """反転率スケジュール(線形・単調増加)。

        flip_prob(t) = flip_prob_max * t / T。t=0 → 0(恒等)、t=T → flip_prob_max。
        BDPM の「反転率が t の関数として [0, 0.5] を単調に動く」に対応(線形版)。
        """
        frac = torch.as_tensor(t, dtype=torch.float32) / self.config.timesteps
        return self.config.flip_prob_max * frac

    def _time_bits(self, t, batch: int, height: int, width: int,
                   device, dtype) -> torch.Tensor:
        """時刻 t を thermometer 符号化し空間へブロードキャストする。

        thermometer(単調)符号: level = t/T * time_embed_bits とし、
        bit[i] = 1 (i < level)。t=0 で全0、t=T で全1。反転率が単調増加する
        前向き過程と符号が単調に対応し、LUT が「ノイズ強度」を素直に読める。
        戻り値: (B, time_embed_bits, H, W) の {0,1} テンソル。
        """
        nb = self.config.time_embed_bits
        t_tensor = self._as_time_tensor(t, batch, device).to(torch.float32)
        level = t_tensor / self.config.timesteps * nb          # (B,)
        idx = torch.arange(nb, device=device).view(1, nb)      # (1, nb)
        bits = (idx < level.view(batch, 1)).to(dtype)          # (B, nb) {0,1}
        return bits.view(batch, nb, 1, 1).expand(batch, nb, height, width)

    # ------------------------------------------------------------------
    # 前向き過程 / デノイザ
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t, *,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """前向き過程: x_t = x0 XOR Bernoulli(flip_prob(t))。{0,1} 厳守。

        XOR はビット演算として (x0 + noise) % 2 で厳密に計算する(x0, noise は
        {0,1})。q_sample は学習対象外のためhard演算でよい(BDPM準拠)。
        t は int / スカラ / (B,) テンソルを受け付ける。
        """
        x0 = x0.to(torch.float32)
        batch = x0.shape[0]
        fp = self.flip_prob(self._as_time_tensor(t, batch, x0.device))
        fp = fp.to(x0.device).view(batch, 1, 1, 1)
        rand = torch.rand(x0.shape, generator=generator, device=x0.device)
        noise = (rand < fp).to(x0.dtype)
        return (x0 + noise) % 2

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        """デノイザ: x_0 の予測ロジット (B, C, H, W) を返す(x_0予測型・BDPM準拠)。

        x_t と時刻 t の thermometer 符号(空間へブロードキャスト)を channel 次元
        で concat してデノイザ(LogicUNet)へ入力する。
        """
        batch, _channels, height, width = x_t.shape
        bits = self._time_bits(t, batch, height, width, x_t.device, x_t.dtype)
        # x_t を複製して画像chの参照確率を確保し、時刻符号と concat する
        x_rep = x_t.repeat(1, self.image_repeat, 1, 1)
        denoiser_in = torch.cat([x_rep, bits], dim=1)
        return self.denoiser(denoiser_in)

    # ------------------------------------------------------------------
    # 損失 / サンプリング
    # ------------------------------------------------------------------

    def loss(self, x0: torch.Tensor, *,
             generator: torch.Generator | None = None) -> dict:
        """ランダム t で q_sample→デノイザ→BCEWithLogits(x_0予測)。

        t は {1, .., T} から一様サンプル(t=0 は無ノイズで自明なため除外)。
        戻り値: {'loss': スカラ, 't_mean': サンプルした t の平均}。
        """
        x0 = x0.to(torch.float32)
        batch = x0.shape[0]
        t = torch.randint(
            1, self.config.timesteps + 1, (batch,),
            generator=generator, device=x0.device,
        )
        x_t = self.q_sample(x0, t, generator=generator)
        logits = self.forward(x_t, t)
        loss = F.binary_cross_entropy_with_logits(logits, x0)
        return {'loss': loss, 't_mean': t.to(torch.float32).mean()}

    @torch.no_grad()
    def sample(self, n: int, *, num_steps: int | None = None,
               generator: torch.Generator | None = None,
               return_trajectory: bool = False):
        """逆過程: 完全ノイズから逐次デノイズして (n, C, H, W) の {0,1} を生成する。

        逆過程の構成(x_0予測から x_{t-1} をどう作るか):
            前向き過程は x_t = x_0 ⊕ d_t、d_t ~ Bernoulli(p_t)(p_t=flip_prob(t))。
            各ビットが x_0 から反転しているかを表す指示子 d_t の marginal 反転率が p_t。
            逆過程は各ステップで
                1. x_0 を予測(デノイザ→ロジット→hard {0,1})、
                2. **反転チャネルの厳密な事後分布** q(d_{t-1} | d_t, x̂_0) から d_{t-1}
                   をサンプルし、x_{t-1} = x̂_0 ⊕ d_{t-1} を構成する。
            事後分布はビット反転を独立チャネルとみなして導く。t→t-1 の1ステップ
            反転率 β を合成則 (1-2p_t) = (1-2p_{t-1})(1-2β) から求め、
            現在の反転指示子 d_t = (x_t ≠ x̂_0) を尤度、p_{t-1} を事前として
                P(d_{t-1}=1 | d_t) ∝ P(d_t | d_{t-1}=1)·p_{t-1}
            で正規化する。これはガウス拡散の祖先サンプリング(x_0予測→事後分散で
            x_{t-1} を再構成)のXOR版に相当する。marginal で再ノイズする素朴案より
            付加ノイズが遥かに小さく、最終ステップ(p_{t-1}=0)では d_{t-1}=0 となって
            予測 x_0 がそのまま最終出力になる。完全ノイズ Bernoulli(0.5) から開始する。

        eval(hard)経路で動く: デノイザは純ブール回路、事後サンプリングはビット反転
        のため、軌跡・最終出力ともに厳密に {0,1} を保つ。

        Args:
            num_steps: 逆過程のステップ数(既定=timesteps)。timesteps より小さい
                値を指定すると [T, 1] を等間隔サンプルした時刻列で高速化する。
            return_trajectory: True で各ステップの中間 x(デノイズ経過)も返す。

        Returns:
            return_trajectory=False: 最終 x (n, C, H, W) {0,1}。
            return_trajectory=True: (最終 x, 軌跡リスト[(n,C,H,W){0,1}, ...])。
        """
        cfg = self.config
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        big_t = cfg.timesteps
        steps = num_steps or big_t
        # 降順の時刻列(T → 1)。num_steps < T なら等間隔サンプルする。
        ts = torch.linspace(big_t, 1, steps).round().to(torch.long).tolist()

        # 完全ノイズ Bernoulli(0.5) から開始
        shape = (n, cfg.in_channels, cfg.image_size, cfg.image_size)
        x = (torch.rand(shape, generator=generator, device=device) < 0.5).to(
            torch.float32)

        trajectory: list[torch.Tensor] = []
        for i, t in enumerate(ts):
            logits = self.forward(x, t)
            x0_hat = (logits > 0).to(torch.float32)   # hard x_0 予測 {0,1}
            next_t = ts[i + 1] if i + 1 < len(ts) else 0
            x = self._posterior_sample(x, x0_hat, t, next_t, generator)
            if return_trajectory:
                trajectory.append(x.clone())

        if was_training:
            self.train()
        if return_trajectory:
            return x, trajectory
        return x

    def _posterior_sample(self, x_t: torch.Tensor, x0_hat: torch.Tensor,
                          t: int, next_t: int,
                          generator: torch.Generator | None) -> torch.Tensor:
        """反転チャネルの事後分布 q(d_{t-1} | d_t, x̂_0) から x_{t-1} をサンプルする。

        全て {0,1}(反転指示子)で計算し、x_{t-1} = x̂_0 ⊕ d_{t-1} を返す。
        """
        p_t = float(self.flip_prob(t))
        p_prev = float(self.flip_prob(next_t))
        # 最終ステップ(p_prev=0): 反転なしで予測 x_0 をそのまま返す
        if p_prev <= 0.0:
            return x0_hat
        # 1ステップ反転率 β: (1-2p_t) = (1-2p_{t-1})(1-2β)
        beta = 0.5 * (1.0 - (1.0 - 2.0 * p_t) / (1.0 - 2.0 * p_prev))
        beta = min(max(beta, 0.0), 0.5)

        d_t = (x_t != x0_hat).to(torch.float32)  # 現在の反転指示子 {0,1}
        # P(d_t | d_{t-1}=1): d_t=1 なら (1-β)、d_t=0 なら β
        like1 = torch.where(d_t > 0.5, 1.0 - beta, beta)
        # P(d_t | d_{t-1}=0): d_t=1 なら β、d_t=0 なら (1-β)
        like0 = torch.where(d_t > 0.5, beta, 1.0 - beta)
        w1 = like1 * p_prev
        w0 = like0 * (1.0 - p_prev)
        prob1 = w1 / (w0 + w1).clamp(min=1e-12)

        rand = torch.rand(d_t.shape, generator=generator, device=d_t.device)
        d_prev = (rand < prob1).to(torch.float32)
        return (x0_hat + d_prev) % 2
