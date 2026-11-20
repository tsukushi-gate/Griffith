"""L5: LL-ViTミニ統合テスト。

「Attention・LayerNorm・埋め込みはPyTorch実数層のまま、
MLPだけLUTベース論理ゲート」というLL-ViT型ハイブリッドが
実際に学習する(損失が減り、実数側にも勾配が流れる)ことを検証。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import (
    LogicMLP,
    LogicSpec,
    replace_mlp_with_lut,
    split_param_groups,
)

DIM = 32
SEQ = 8
N_CLASSES = 4


class MiniViTBlock(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        return x + self.mlp(self.norm2(x))


class MiniViT(nn.Module):
    """2層・dim32のミニViT(パッチ埋め込みはLinearで代用)"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, DIM)
        self.pos = nn.Parameter(torch.randn(1, SEQ, DIM) * 0.02)
        self.blocks = nn.Sequential(MiniViTBlock(), MiniViTBlock())
        self.head = nn.Linear(DIM, N_CLASSES)

    def forward(self, x):  # (B, SEQ, 16)
        h = self.embed(x) + self.pos
        h = self.blocks(h)
        return self.head(h.mean(dim=1))


def _synthetic_task(n=256, seed=0):
    """トークン平均の象限でクラスが決まる合成分類タスク"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, SEQ, 16, generator=g)
    m = x.mean(dim=(1, 2) )
    s = x[:, :, 0].mean(dim=1)
    y = ((m > 0).long() * 2 + (s > 0).long()).clamp(max=N_CLASSES - 1)
    return x, y


class TestLLViTMini:
    @pytest.mark.learning
    def test_hybrid_conversion_and_training(self, seed):
        torch.manual_seed(7)
        model = MiniViT()
        hybrid, replaced = replace_mlp_with_lut(
            model,
            spec=LogicSpec(n_inputs=2, depth=2, neurons_per_output=8),
        )
        assert replaced == ['blocks.0.mlp', 'blocks.1.mlp']
        assert isinstance(hybrid.blocks[0].mlp, LogicMLP)
        # Attention側はPyTorchのまま
        assert isinstance(hybrid.blocks[0].attn, nn.MultiheadAttention)

        x, y = _synthetic_task()
        opt = torch.optim.AdamW(
            split_param_groups(hybrid, logic_lr=0.02, ann_lr=1e-3),
        )

        hybrid.train()
        losses = []
        for step in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(hybrid(x), y)
            loss.backward()

            if step == 0:
                # Attention側パラメータにも勾配が流れている
                attn_grads = [
                    p.grad.abs().sum().item()
                    for n, p in hybrid.named_parameters()
                    if 'attn' in n and p.grad is not None
                ]
                assert attn_grads and all(g > 0 for g in attn_grads), (
                    'Attention側に勾配が流れていない'
                )
                # LUT側にも勾配
                lut_grads = [
                    p.grad.abs().sum().item()
                    for n, p in hybrid.named_parameters()
                    if '.luts.' in n and p.grad is not None
                ]
                assert lut_grads and all(g > 0 for g in lut_grads), (
                    'LUT側に勾配が流れていない'
                )

            opt.step()
            losses.append(loss.item())

        # 損失が減少している(前半平均 > 後半平均)
        first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
        assert last < first * 0.9, (
            f'損失が減少していない: first={first:.4f} last={last:.4f}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
