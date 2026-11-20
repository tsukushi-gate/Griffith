"""L4: blocks部品の学習収束。

    - Residual(concat/xor)を含むモデルがXORをhard acc=100%で学習
    - LogicTransformerBlockのコピータスクスモーク: 300stepでlossが
      有意に下がる(閾値は実測較正、docstring参照)
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.blocks import LogicTransformerBlock, Residual

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _xor_dataset():
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    y = torch.tensor([0, 1, 1, 0])
    return x, y


def _train_until(model, x, y, *, lr=0.05, max_steps=1000, check_every=50):
    """soft訓練し、hard evalで100%になったら早期終了。最終hard accを返す"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


class TestResidualLearning:
    def test_residual_concat_learns_xor(self, seed):
        """Residual(merge='concat')入りモデルでXOR hard acc=100%"""
        torch.manual_seed(0)
        model = nn.Sequential(
            Residual(LutLayer(2, 14), merge='concat'),  # 2 → 16
            LutLayer(16, 16),
            GroupSum(k=2, tau=(16 // 2) ** 0.5),
        )
        x, y = _xor_dataset()
        acc = _train_until(model, x, y)
        assert acc == 1.0, f'Residual(concat): XOR hard acc={acc}'

    def test_residual_xor_merge_learns_xor(self, seed):
        """Residual(merge='xor')入りモデルでXOR hard acc=100%"""
        torch.manual_seed(0)
        model = nn.Sequential(
            LutLayer(2, 16),
            Residual(LutLayer(16, 16), merge='xor'),
            LutLayer(16, 16),
            GroupSum(k=2, tau=(16 // 2) ** 0.5),
        )
        x, y = _xor_dataset()
        acc = _train_until(model, x, y)
        assert acc == 1.0, f'Residual(xor): XOR hard acc={acc}'


class TestTransformerCopySmoke:
    # 較正記録(build_seed=0で10seed実測):
    #   初期loss ≈ ln(8) = 2.08、300step後の最終lossは
    #   最悪0.706 / 最良0.671。最悪値に2倍の余裕を持たせて閾値1.4に固定
    #   (初期値2.08より十分低く「有意に下がった」ことを判定できる)
    LOSS_THRESHOLD = 1.4

    def test_copy_task_loss_drops(self, seed):
        """コピータスク(系列をそのまま出力)で300step学習しlossが有意に低下。

        設定: S=4, vocab=8(one-hot 8bit=d_model), バッチ256固定、
        LogicTransformerBlock(8, n_heads=2, mlp_ratio=2,
        norm='layernorm', merge='xor') + 位置ごとのLUT読み出し。
        """
        s_len, vocab, d_model = 4, 8, 8
        torch.manual_seed(0)
        block = LogicTransformerBlock(
            d_model, n_heads=2, mlp_ratio=2, norm='layernorm', merge='xor',
        )
        readout = nn.Sequential(
            LutLayer(d_model, 64),
            GroupSum(k=vocab, tau=(64 // vocab) ** 0.5),
        )
        params = list(block.parameters()) + list(readout.parameters())

        torch.manual_seed(42)
        tokens = torch.randint(0, vocab, (256, s_len))
        x = F.one_hot(tokens, vocab).float()  # (B, S, 8)
        target = tokens.reshape(-1)

        opt = torch.optim.Adam(params, lr=0.03)
        block.train()
        readout.train()
        loss_init = None
        for _ in range(300):
            opt.zero_grad()
            logits = readout(block(x).reshape(-1, d_model))
            loss = F.cross_entropy(logits, target)
            if loss_init is None:
                loss_init = loss.item()
            loss.backward()
            opt.step()
        loss_final = loss.item()

        assert torch.isfinite(loss).all(), 'lossが非有限'
        assert loss_final < self.LOSS_THRESHOLD, (
            f'コピータスクのlossが下がらない: '
            f'{loss_init:.3f} → {loss_final:.3f}(閾値{self.LOSS_THRESHOLD})'
        )
        assert loss_final < 0.7 * loss_init, (
            f'相対低下が不足: {loss_init:.3f} → {loss_final:.3f}'
        )

    def test_forward_shapes_and_finite(self, seed):
        """形状・勾配疎通・全有限(norm/merge組み合わせ)"""
        for norm in ('layernorm', 'batchnorm', 'none'):
            for merge in ('or', 'xor'):
                torch.manual_seed(1)
                block = LogicTransformerBlock(4, n_heads=1, norm=norm,
                                              merge=merge)
                block.train()
                x = torch.randint(0, 2, (3, 5, 4)).float()
                y = block(x)
                assert y.shape == (3, 5, 4), f'{norm}/{merge}: 形状不一致'
                assert torch.isfinite(y).all(), f'{norm}/{merge}: 非有限値'
                y.sum().backward()
                for name, p in block.named_parameters():
                    assert p.grad is not None, f'{norm}/{merge}: {name}のgradがNone'
                    assert torch.isfinite(p.grad).all(), \
                        f'{norm}/{merge}: {name}のgradに非有限値'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
