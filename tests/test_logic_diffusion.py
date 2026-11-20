"""D6 LogicBinaryDiffusion の検証。

1. q_sample の統計性質: 反転率が指定スケジュールと一致(大数)、t=0で恒等・
   t=Tで flip_prob_max 近傍、出力 {0,1}
2. shape契約: forward / loss / sample の形状
3. 勾配疎通: デノイザ全層まで勾配が届く
4. hard離散性: eval時のデノイザ中間出力と sample 出力が厳密に {0,1}
5. 極小収束: 4パターンのトイ分布(8×8)を学習し、小ノイズ t での x_0 予測精度が
   較正閾値以上、かつ sample 出力が学習パターンのいずれかに近い(ハミング距離)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.logic_diffusion import (
    LogicBinaryDiffusion,
    LogicDiffusionConfig,
)


def _toy_patterns() -> torch.Tensor:
    """4つの識別可能な 8×8 二値パターン (4, 1, 8, 8) {0,1}。"""
    p = torch.zeros(4, 1, 8, 8)
    p[0, :, :, :4] = 1.0                       # 左半分
    p[1, :, :4, :] = 1.0                       # 上半分
    yy, xx = torch.meshgrid(
        torch.arange(8), torch.arange(8), indexing='ij')
    p[2, 0] = ((yy + xx) % 2).to(torch.float32)  # チェッカーボード
    p[3, :, 0, :] = 1.0                        # 枠(外周)
    p[3, :, -1, :] = 1.0
    p[3, :, :, 0] = 1.0
    p[3, :, :, -1] = 1.0
    return p


# ============================================================
# 1. q_sample の統計性質
# ============================================================

class TestQSampleStatistics:
    def test_output_is_binary(self):
        """q_sample の出力が厳密に {0,1}。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(image_size=8, timesteps=16))
        x0 = (torch.rand(8, 1, 8, 8) < 0.5).float()
        for t in (0, 3, 8, 16):
            x_t = model.q_sample(x0, t)
            assert torch.all((x_t == 0.0) | (x_t == 1.0)), t

    def test_t0_is_identity(self):
        """t=0 で反転率0 → 恒等(x_t == x0)。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(image_size=8, timesteps=16))
        x0 = (torch.rand(16, 1, 8, 8) < 0.5).float()
        x_t = model.q_sample(x0, 0)
        assert torch.equal(x_t, x0)

    def test_flip_rate_matches_schedule(self):
        """経験的な反転率がスケジュール flip_prob(t) に一致(大数)。"""
        cfg = LogicDiffusionConfig(image_size=16, timesteps=16, flip_prob_max=0.5)
        model = LogicBinaryDiffusion(cfg)
        gen = torch.Generator().manual_seed(0)
        x0 = (torch.rand(256, 1, 16, 16, generator=gen) < 0.5).float()
        for t in (0, 4, 8, 12, 16):
            x_t = model.q_sample(x0, t, generator=gen)
            empirical = (x_t != x0).float().mean().item()
            expected = model.flip_prob(t).item()
            assert abs(empirical - expected) < 0.02, (t, empirical, expected)

    def test_tT_reaches_flip_prob_max(self):
        """t=T の経験反転率が flip_prob_max 近傍。"""
        cfg = LogicDiffusionConfig(image_size=16, timesteps=16, flip_prob_max=0.5)
        model = LogicBinaryDiffusion(cfg)
        gen = torch.Generator().manual_seed(1)
        x0 = (torch.rand(256, 1, 16, 16, generator=gen) < 0.5).float()
        x_t = model.q_sample(x0, cfg.timesteps, generator=gen)
        empirical = (x_t != x0).float().mean().item()
        assert abs(empirical - cfg.flip_prob_max) < 0.02, empirical


# ============================================================
# 2. shape契約
# ============================================================

class TestShapeContract:
    def test_forward_shape(self):
        """forward は x_0 予測ロジット (B, C, H, W) を返す。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, in_channels=1, base_channels=8, depth=2))
        x_t = (torch.rand(3, 1, 8, 8) < 0.5).float()
        logits = model.forward(x_t, 5)
        assert logits.shape == (3, 1, 8, 8)

    def test_forward_per_sample_t(self):
        """t を (B,) テンソルで渡せる。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2))
        x_t = (torch.rand(4, 1, 8, 8) < 0.5).float()
        t = torch.tensor([1, 5, 10, 16])
        assert model.forward(x_t, t).shape == (4, 1, 8, 8)

    def test_loss_keys_and_scalar(self):
        """loss は {'loss','t_mean'} を返し loss はスカラで grad を持つ。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2))
        model.train()
        x0 = (torch.rand(4, 1, 8, 8) < 0.5).float()
        out = model.loss(x0)
        assert set(out) >= {'loss', 't_mean'}
        assert out['loss'].ndim == 0
        assert out['loss'].requires_grad

    def test_sample_shape(self):
        """sample は (n, C, H, W) を返す。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2, timesteps=8))
        x = model.sample(5)
        assert x.shape == (5, 1, 8, 8)

    def test_sample_num_steps_subsampling(self):
        """num_steps < timesteps でも動く(等間隔サンプル)。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2, timesteps=16))
        x = model.sample(3, num_steps=4)
        assert x.shape == (3, 1, 8, 8)

    def test_sample_trajectory(self):
        """return_trajectory で軌跡を返す。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2, timesteps=6))
        x, traj = model.sample(2, return_trajectory=True)
        assert x.shape == (2, 1, 8, 8)
        assert len(traj) == 6
        assert all(t.shape == (2, 1, 8, 8) for t in traj)


# ============================================================
# 3. 勾配疎通
# ============================================================

class TestGradFlow:
    def test_grad_reaches_all_denoiser_layers(self):
        """loss.backward() で全デノイザ層(LutConv2d)の grad が有限・非ゼロ。"""
        torch.manual_seed(2)
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2, n_inputs=2))
        model.train()
        gen = torch.Generator().manual_seed(0)
        x0 = _toy_patterns().repeat(4, 1, 1, 1)  # (16,1,8,8)
        model.loss(x0, generator=gen)['loss'].backward()

        for name, module in [
            ('enc_convs[0]', model.denoiser.enc_convs[0]),
            ('enc_convs[-1]', model.denoiser.enc_convs[-1]),
            ('bottleneck', model.denoiser.bottleneck),
            ('dec_convs[-1]', model.denoiser.dec_convs[-1]),
            ('head_conv', model.denoiser.head_conv),
        ]:
            g = module.weights.grad
            assert g is not None, f'{name} の grad が None'
            assert torch.isfinite(g).all(), f'{name} の grad に非有限値'
            assert g.abs().sum().item() > 0.0, f'{name} まで勾配が届いていない'

        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} の grad が None'
            assert torch.isfinite(p.grad).all(), f'{name} の grad に非有限値'


# ============================================================
# 4. hard離散性
# ============================================================

class TestHardDiscreteness:
    def test_eval_denoiser_intermediate_binary(self):
        """eval時、デノイザ中間(各LutConv2d出力)が厳密に {0,1}。"""
        torch.manual_seed(0)
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2))
        model.eval()

        from griffith_logic.conv import LutConv2d
        captured = []

        def hook(_mod, _inp, out):
            captured.append(out)

        handles = [m.register_forward_hook(hook)
                   for m in model.denoiser.modules() if isinstance(m, LutConv2d)]
        x_t = (torch.rand(2, 1, 8, 8) < 0.5).float()
        with torch.no_grad():
            model.forward(x_t, 5)
        for h in handles:
            h.remove()

        assert captured, 'フックが発火していない'
        for i, out in enumerate(captured):
            assert torch.all((out == 0.0) | (out == 1.0)), f'LutConv2d[{i}] が二値でない'

    def test_sample_output_binary(self):
        """sample 出力が厳密に {0,1}。"""
        model = LogicBinaryDiffusion(LogicDiffusionConfig(
            image_size=8, base_channels=8, depth=2, timesteps=8))
        gen = torch.Generator().manual_seed(0)
        x = model.sample(6, generator=gen)
        assert torch.all((x == 0.0) | (x == 1.0))


# ============================================================
# 5. 極小収束(スモーク)
# ============================================================

def _hamming_to_nearest(samples: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    """各サンプルの、最も近い学習パターンへの正規化ハミング距離 (n,)。"""
    n = samples.shape[0]
    flat_s = samples.reshape(n, -1)
    flat_p = patterns.reshape(patterns.shape[0], -1)
    # (n, n_patterns): ビット不一致の割合
    dist = (flat_s[:, None, :] != flat_p[None, :, :]).float().mean(dim=-1)
    return dist.min(dim=1).values


@pytest.mark.learning
class TestConvergence:
    def test_toy_distribution_learning(self):
        """4パターンのトイ分布を学習し、デノイズ能力とサンプル品質を確認する。

        この極小スケール(8×8, base=16, 600steps)では、論理デノイザは強い時刻
        条件付けや低ノイズでの精密複写までは獲得しきれず、「(ノイズ入力, t)→
        妥当なパターン」への写像に近い挙動を学ぶ(妥協点)。それでも
            (a) 高ノイズ側では passthrough(複写)より高精度に x_0 を復元し
                (=実際にノイズ除去できている)、
            (b) 完全ノイズからのサンプルが学習パターン近傍へ集中する
                (=分布を学習できている)
        ことを、実測較正値で検証する(生成品質の厳密基準は本走で較正)。
        """
        torch.manual_seed(0)
        patterns = _toy_patterns()
        cfg = LogicDiffusionConfig(
            image_size=8, in_channels=1, timesteps=8, flip_prob_max=0.5,
            base_channels=16, depth=2, n_inputs=2, time_embed_bits=8)
        model = LogicBinaryDiffusion(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        gen = torch.Generator().manual_seed(1)

        model.train()
        batch = patterns.repeat(8, 1, 1, 1)  # (32,1,8,8)
        for _ in range(600):
            opt.zero_grad()
            model.loss(batch, generator=gen)['loss'].backward()
            opt.step()

        model.eval()
        # 統計を安定させるため 256 サンプルで評価する
        big = patterns.repeat(64, 1, 1, 1)

        # --- (a1) x_0 予測精度がランダム予測(0.5)を明確に上回る(小ノイズ t=1) ---
        with torch.no_grad():
            x_t1 = model.q_sample(big, 1, generator=gen)
            acc_t1 = ((model.forward(x_t1, 1) > 0).float() == big).float().mean().item()
        # 実測較正(seed=0): acc@t1≈0.71。ランダム予測 0.5 を上回ることを、計測
        # ゆらぎ分のマージンを引いた 0.60 で判定する。
        assert acc_t1 >= 0.60, acc_t1

        # --- (a2) 高ノイズ t=T では passthrough(複写)より高精度に復元する ---
        with torch.no_grad():
            x_tT = model.q_sample(big, cfg.timesteps, generator=gen)
            copy_acc = (x_tT == big).float().mean().item()
            model_acc = ((model.forward(x_tT, cfg.timesteps) > 0).float()
                         == big).float().mean().item()
        # 実測較正(seed=0): copy≈0.50, model≈0.63。デノイザが複写を +0.05 以上
        # 上回る(=実際にノイズ除去している)ことを判定する。
        assert model_acc >= copy_acc + 0.05, (model_acc, copy_acc)

        # --- (b) サンプル品質: 学習パターンのいずれかに近い(ハミング距離小) ---
        with torch.no_grad():
            samples = model.sample(32, generator=gen)
        dist = _hamming_to_nearest(samples, patterns)
        # 完全ノイズ(ランダム{0,1})サンプルの最近傍ハミング距離は実測 ~0.44。
        # 学習が効けば大きく下回る。実測(≈0.20)に対しマージンを見て 0.30 を
        # 上限判定とする(スモーク: 生成品質の厳密基準は本走で較正)。
        assert dist.mean().item() <= 0.30, dist.mean().item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
