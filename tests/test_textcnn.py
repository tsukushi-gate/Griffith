"""TextCNNModel のユニットテスト。"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.conv import LutConv2d

# train_textcnn.py のモデルクラスを直接テスト
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_textcnn import TextCNNModel

# ============================================================
# TextCNNModel: 基本テスト
# ============================================================

class TestTextCNNModel:
    """TextCNNModelの基本動作テスト。"""

    @pytest.fixture
    def small_model(self):
        """テスト用の小さいモデル。"""
        return TextCNNModel(
            vocab_size=100,
            d_emb=16,
            n_filters=4,
            kernel_sizes=[3, 5],
            dense_dims=[32, 16],
            n_classes=2,
            tree_depth=2,
            n_inputs=2,
            device='cpu',
        )

    def test_output_shape(self, small_model):
        """出力形状が (batch, n_classes) になるか。"""
        token_ids = torch.randint(0, 100, (4, 20))
        logits, emb_loss = small_model(token_ids)
        assert logits.shape == (4, 2)
        assert emb_loss.shape == ()

    def test_variable_batch_size(self, small_model):
        """異なるバッチサイズで動作するか。"""
        for bs in [1, 8, 16]:
            token_ids = torch.randint(0, 100, (bs, 20))
            logits, _ = small_model(token_ids)
            assert logits.shape == (bs, 2)

    def test_variable_seq_len(self, small_model):
        """異なる系列長で動作するか。"""
        for seq_len in [5, 20, 50]:
            token_ids = torch.randint(0, 100, (4, seq_len))
            logits, _ = small_model(token_ids)
            assert logits.shape == (4, 2)

    def test_gradient_flows(self, small_model):
        """勾配が全パラメータに流れるか。"""
        token_ids = torch.randint(0, 100, (4, 20))
        logits, emb_loss = small_model(token_ids)
        loss = logits.sum() + emb_loss
        loss.backward()

        for name, param in small_model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'
            assert not torch.all(param.grad == 0), f'{name} の勾配が全ゼロです'

    def test_eval_mode_discrete(self, small_model):
        """eval mode で離散化が動作するか。"""
        small_model.eval()
        token_ids = torch.randint(0, 100, (4, 20))
        logits, _ = small_model(token_ids)
        assert logits.shape == (4, 2)

    def test_train_eval_different(self, small_model):
        """train mode と eval mode で出力が異なるか。"""
        token_ids = torch.randint(0, 100, (4, 20))

        small_model.train()
        logits_train, _ = small_model(token_ids)

        small_model.eval()
        logits_eval, _ = small_model(token_ids)

        # 完全一致はしないはず（離散化の影響）
        assert not torch.allclose(logits_train, logits_eval)

    def test_embedding_reg_loss_range(self, small_model):
        """二値化正則化損失が [0, 0.25] の範囲にあるか。"""
        small_model.train()
        token_ids = torch.randint(0, 100, (4, 20))
        _, emb_loss = small_model(token_ids)
        assert emb_loss >= 0.0
        assert emb_loss <= 0.25  # max(x*(1-x)) = 0.25 at x=0.5

    def test_count_parameters(self, small_model):
        """count_parameters() が正しい内訳を返すか。"""
        counts = small_model.count_parameters()
        assert 'embedding' in counts
        assert 'conv_branches' in counts
        assert 'dense' in counts
        assert 'total' in counts
        assert counts['total'] == counts['embedding'] + counts['conv_branches'] + counts['dense']
        assert counts['total'] == sum(p.numel() for p in small_model.parameters())


# ============================================================
# LutConv2d を1D畳み込みとして使うテスト
# ============================================================

class TestConv2dAs1d:
    """LutConv2d を kernel_size=(k, d_emb) で使った際の動作確認。"""

    def test_output_width_is_one(self):
        """kernel_w = d_emb のとき W_out = 1 になるか。"""
        d_emb = 32
        conv = LutConv2d(
            in_channels=1, out_channels=8,
            kernel_size=(3, d_emb), tree_depth=2,
            padding=(1, 0), n_inputs=2,
        )
        x = torch.rand(2, 1, 10, d_emb)
        y = conv(x)
        # W_out = (d_emb + 0 - d_emb) // 1 + 1 = 1
        assert y.shape[3] == 1
        # H_out = (10 + 2 - 3) // 1 + 1 = 10
        assert y.shape[2] == 10
        assert y.shape == (2, 8, 10, 1)

    def test_different_kernel_sizes(self):
        """異なるカーネルサイズ(3,4,5)で正しい出力形状になるか。"""
        d_emb = 16
        seq_len = 20
        x = torch.rand(4, 1, seq_len, d_emb)

        for k in [3, 4, 5]:
            conv = LutConv2d(
                in_channels=1, out_channels=4,
                kernel_size=(k, d_emb), tree_depth=2,
                padding=(k // 2, 0), n_inputs=2,
            )
            y = conv(x)
            assert y.shape[3] == 1  # W_out は常に1
            assert y.shape[0] == 4  # バッチ
            assert y.shape[1] == 4  # out_channels

    def test_global_max_pool(self):
        """Global max pool で正しい形状になるか。"""
        import torch.nn.functional as F

        d_emb = 16
        conv = LutConv2d(
            in_channels=1, out_channels=8,
            kernel_size=(3, d_emb), tree_depth=2,
            padding=(1, 0), n_inputs=2,
        )
        x = torch.rand(4, 1, 20, d_emb)
        h = conv(x)  # (4, 8, 20, 1)
        pooled = F.adaptive_max_pool2d(h, (1, 1))  # (4, 8, 1, 1)
        flat = pooled.flatten(1)  # (4, 8)
        assert flat.shape == (4, 8)


# ============================================================
# パラメータ化方式テスト
# ============================================================

class TestParameterizations:
    """各パラメータ化方式でTextCNNが動作するか。"""

    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp'])
    def test_forward_pass(self, param):
        """全パラメータ化方式でforward passが通るか。"""
        model = TextCNNModel(
            vocab_size=50,
            d_emb=16,
            n_filters=4,
            kernel_sizes=[3],
            dense_dims=[16, 8],
            n_classes=2,
            tree_depth=2,
            n_inputs=2,
            parameterization=param,
        )
        token_ids = torch.randint(0, 50, (2, 10))
        logits, emb_loss = model(token_ids)
        assert logits.shape == (2, 2)

    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp'])
    def test_gradient_flows(self, param):
        """全パラメータ化方式で勾配が流れるか。"""
        model = TextCNNModel(
            vocab_size=50,
            d_emb=16,
            n_filters=4,
            kernel_sizes=[3],
            dense_dims=[16, 8],
            n_classes=2,
            tree_depth=2,
            n_inputs=2,
            parameterization=param,
        )
        token_ids = torch.randint(0, 50, (2, 10))
        logits, emb_loss = model(token_ids)
        loss = logits.sum() + emb_loss
        loss.backward()

        for name, param_tensor in model.named_parameters():
            assert param_tensor.grad is not None, f'{name} に勾配がありません ({param})'


# ============================================================
# 訓練ステップテスト
# ============================================================

class TestTrainingStep:
    """1ステップの訓練が正しく動作するか。"""

    def test_loss_decreases(self):
        """数ステップの訓練でlossが減少するか。"""
        model = TextCNNModel(
            vocab_size=50,
            d_emb=16,
            n_filters=4,
            kernel_sizes=[3, 5],
            dense_dims=[32, 16],
            n_classes=2,
            tree_depth=2,
            n_inputs=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        token_ids = torch.randint(0, 50, (16, 20))
        labels = torch.randint(0, 2, (16,))

        losses = []
        for step in range(20):
            model.train()
            logits, emb_loss = model(token_ids)
            loss = loss_fn(logits, labels) + 0.01 * emb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 最初の5ステップの平均 > 最後の5ステップの平均
        assert sum(losses[:5]) / 5 > sum(losses[-5:]) / 5
