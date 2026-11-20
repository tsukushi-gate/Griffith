"""L3: RDDLGN(Bührer 2025)の勾配安定性回帰テスト。

リカレント構造でも全LutLayerに勾配が減衰せず届くこと(論文の主要主張の
ひとつ: 全層で安定した勾配)を、copyタスクの1 backwardで検証する。

実測較正(tiny構成, seed 0/1/2):
    LutLayer 11層すべてに非ゼロ勾配、per-layer勾配ノルムの
    std/mean = 2.59〜2.63(±50%帯 → [1.3, 3.9] を回帰帯とする)。
    ※RDDLGN 解析レポート(内部)の ≈7.9 は論文フル構成での値。
      本テストはtiny構成の自己較正値を監視する。

高速(<5秒)のため@paperマーカーなしで常時スイートに含める。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import RDDLGN, RDDLGNConfig
from griffith_logic.layers.lut import LutLayer

RATIO_LOW, RATIO_HIGH = 1.3, 3.9   # 実測2.6の±50%
MIN_GRAD_NORM = 1e-9               # 勾配消失ガード(実測min≈1e-6)


class TestRDDLGNGradientStability:
    def _grad_norms(self, seed=0):
        torch.manual_seed(seed)
        cfg = RDDLGNConfig.tiny()
        model = RDDLGN(cfg)
        src = torch.randint(0, cfg.vocab_size, (16, cfg.seq_len))
        logits, emb_loss = model(src, src)  # copyタスク(teacher forcing)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), src.reshape(-1),
        ) + 0.1 * emb_loss
        loss.backward()

        norms = []
        for m in model.modules():
            if isinstance(m, LutLayer):
                assert m.weights.grad is not None, '勾配未到達のLutLayerが存在'
                norms.append(m.weights.grad.norm().item())
        return torch.tensor(norms)

    def test_all_layers_receive_gradient(self, seed):
        """全LutLayerに勾配が届き、消失していない"""
        norms = self._grad_norms()
        assert len(norms) == 11  # tiny構成のLutLayer数(構成変更の検知)
        assert norms.min().item() > MIN_GRAD_NORM, (
            f'勾配消失の疑い: min norm = {norms.min():.3e}'
        )

    def test_gradient_ratio_band(self, seed):
        """層間勾配ノルムのstd/mean比が較正帯域内(回帰監視)"""
        ratios = []
        for s in (0, 1, 2):
            norms = self._grad_norms(seed=s)
            ratios.append((norms.std() / norms.mean()).item())
        mean_ratio = sum(ratios) / len(ratios)
        assert RATIO_LOW <= mean_ratio <= RATIO_HIGH, (
            f'勾配安定帯から逸脱: std/mean = {mean_ratio:.3f} '
            f'(帯域 [{RATIO_LOW}, {RATIO_HIGH}]、実測較正)'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
