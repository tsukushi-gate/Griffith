"""RDDLGNモジュールのテストスイート。"""

import torch

from griffith_logic import (
    RDDLGN,
    LogicEmbedding,
    LutStack,
    RDDLGNClassifier,
    RDDLGNConfig,
    RDDLGNDecoder,
    RDDLGNEncoder,
)

DEVICE = 'cpu'


def _tiny_config() -> RDDLGNConfig:
    return RDDLGNConfig.tiny()


# ---------------------------------------------------------------------------
# LogicEmbedding
# ---------------------------------------------------------------------------

class TestLogicEmbedding:

    def test_output_shape(self):
        emb = LogicEmbedding(100, 32)
        tokens = torch.randint(0, 100, (4, 8))
        out = emb(tokens)
        assert out.shape == (4, 8, 32)

    def test_output_range_train(self):
        emb = LogicEmbedding(100, 32)
        emb.train()
        tokens = torch.randint(0, 100, (4, 8))
        out = emb(tokens)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_discrete_eval(self):
        emb = LogicEmbedding(100, 32)
        emb.eval()
        tokens = torch.randint(0, 100, (4, 8))
        out = emb(tokens)
        # eval時は{0, 1}のみ
        unique = torch.unique(out)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_binary_reg_loss(self):
        emb = LogicEmbedding(100, 32)
        emb.train()
        tokens = torch.randint(0, 100, (4, 8))
        emb(tokens)
        loss = emb.binary_reg_loss()
        assert loss.shape == ()
        # sigmoid出力の x*(1-x) は最大0.25（x=0.5時）
        assert 0.0 <= loss.item() <= 0.25

    def test_binary_reg_loss_gradient(self):
        emb = LogicEmbedding(100, 32)
        emb.train()
        tokens = torch.randint(0, 100, (4, 8))
        emb(tokens)
        loss = emb.binary_reg_loss()
        loss.backward()
        assert emb.embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# LutStack
# ---------------------------------------------------------------------------

class TestLutStack:

    def test_output_shape(self):
        stack = LutStack(in_dim=32, layer_sizes=[64, 48], device=DEVICE)
        x = torch.rand(4, 32)
        out = stack(x)
        assert out.shape == (4, 48)
        assert stack.out_dim == 48

    def test_gradient_flow(self):
        stack = LutStack(in_dim=32, layer_sizes=[64, 48], device=DEVICE)
        x = torch.rand(4, 32, requires_grad=True)
        out = stack(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 32)

    def test_single_layer(self):
        stack = LutStack(in_dim=16, layer_sizes=[32], device=DEVICE)
        assert len(stack.layers) == 1
        x = torch.rand(2, 16)
        out = stack(x)
        assert out.shape == (2, 32)

    def test_three_layers(self):
        stack = LutStack(in_dim=16, layer_sizes=[32, 64, 48], device=DEVICE)
        assert len(stack.layers) == 3
        x = torch.rand(2, 16)
        out = stack(x)
        assert out.shape == (2, 48)


# ---------------------------------------------------------------------------
# RDDLGNEncoder
# ---------------------------------------------------------------------------

class TestRDDLGNEncoder:

    def test_output_shape(self):
        config = _tiny_config()
        enc = RDDLGNEncoder(config, device=DEVICE)
        embedded = torch.rand(4, config.seq_len, config.d_emb)
        context = enc(embedded)
        expected_dim = config.k_layer_sizes[-1]
        assert context.shape == (4, expected_dim)

    def test_gradient_flow(self):
        config = _tiny_config()
        enc = RDDLGNEncoder(config, device=DEVICE)
        embedded = torch.rand(4, config.seq_len, config.d_emb, requires_grad=True)
        context = enc(embedded)
        context.sum().backward()
        assert embedded.grad is not None

    def test_different_seq_len(self):
        """seq_lenが異なっても動作する"""
        config = _tiny_config()
        enc = RDDLGNEncoder(config, device=DEVICE)
        for sl in [2, 4, 8]:
            embedded = torch.rand(2, sl, config.d_emb)
            context = enc(embedded)
            assert context.shape == (2, config.k_layer_sizes[-1])


# ---------------------------------------------------------------------------
# RDDLGNDecoder
# ---------------------------------------------------------------------------

class TestRDDLGNDecoder:

    def test_output_shape(self):
        config = _tiny_config()
        dec = RDDLGNDecoder(config, device=DEVICE)
        tgt_emb = torch.rand(4, 3, config.d_emb)
        context = torch.rand(4, config.k_layer_sizes[-1])
        logits = dec(tgt_emb, context)
        assert logits.shape == (4, 3, config.vocab_size)

    def test_gradient_flow(self):
        config = _tiny_config()
        dec = RDDLGNDecoder(config, device=DEVICE)
        tgt_emb = torch.rand(4, 3, config.d_emb, requires_grad=True)
        context = torch.rand(4, config.k_layer_sizes[-1], requires_grad=True)
        logits = dec(tgt_emb, context)
        logits.sum().backward()
        assert tgt_emb.grad is not None
        assert context.grad is not None


# ---------------------------------------------------------------------------
# RDDLGN (E2E)
# ---------------------------------------------------------------------------

class TestRDDLGN:

    def test_forward(self):
        config = _tiny_config()
        model = RDDLGN(config, device=DEVICE)
        src = torch.randint(0, config.vocab_size, (4, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (4, config.seq_len))
        logits, emb_loss = model(src, tgt)
        assert logits.shape == (4, config.seq_len, config.vocab_size)
        assert emb_loss.shape == ()

    def test_one_training_step(self):
        config = _tiny_config()
        model = RDDLGN(config, device=DEVICE)
        model.train()

        src = torch.randint(0, config.vocab_size, (4, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (4, config.seq_len))
        labels = torch.randint(0, config.vocab_size, (4, config.seq_len))

        logits, emb_loss = model(src, tgt)
        main_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        total_loss = main_loss + 0.1 * emb_loss
        total_loss.backward()

        # 全パラメータに勾配があることを確認
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がない'

    def test_shared_vocab(self):
        config = _tiny_config()
        config.shared_vocab = True
        model = RDDLGN(config, device=DEVICE)
        assert model.src_embedding is model.tgt_embedding

    def test_separate_vocab(self):
        config = _tiny_config()
        config.shared_vocab = False
        model = RDDLGN(config, device=DEVICE)
        assert model.src_embedding is not model.tgt_embedding

    def test_count_parameters(self):
        config = _tiny_config()
        model = RDDLGN(config, device=DEVICE)
        counts = model.count_parameters()
        assert 'embedding' in counts
        assert 'n_layers' in counts
        assert 'k_layers' in counts
        assert 'l_layers' in counts
        assert 'p_layers' in counts
        assert 'm_layers' in counts
        assert 'total' in counts
        # totalが全パラメータの合計と一致
        total_actual = sum(p.numel() for p in model.parameters())
        assert counts['total'] == total_actual


# ---------------------------------------------------------------------------
# RDDLGNClassifier（テキスト分類向け）
# ---------------------------------------------------------------------------

class TestRDDLGNClassifier:

    def test_forward_shape(self):
        config = _tiny_config()
        n_classes = 5
        model = RDDLGNClassifier(config, n_classes, device=DEVICE)
        tokens = torch.randint(0, config.vocab_size, (4, config.seq_len))
        logits, emb_loss = model(tokens)
        assert logits.shape == (4, n_classes)
        assert emb_loss.shape == ()

    def test_emb_reg_finite(self):
        config = _tiny_config()
        model = RDDLGNClassifier(config, 5, device=DEVICE)
        tokens = torch.randint(0, config.vocab_size, (2, config.seq_len))
        _, emb_loss = model(tokens)
        assert torch.isfinite(emb_loss).all()

    def test_gradient_flows_to_embedding(self):
        config = _tiny_config()
        model = RDDLGNClassifier(config, 5, device=DEVICE)
        model.train()
        tokens = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, emb_loss = model(tokens)
        labels = torch.randint(0, 5, (2,))
        loss = torch.nn.functional.cross_entropy(logits, labels) + 0.1 * emb_loss
        loss.backward()
        assert model.embedding.embedding.weight.grad is not None

    def test_head_size_divisible_by_n_classes(self):
        """head_sizesの最終値がn_classesで割り切れない場合でも自動調整される。"""
        config = _tiny_config()
        n_classes = 7
        model = RDDLGNClassifier(config, n_classes, head_sizes=[64, 50], device=DEVICE)
        assert model.group_sum.k == n_classes
        # LutStackの最終層出力がn_classesで割り切れることを確認
        assert model.head.out_dim % n_classes == 0

    def test_custom_head_sizes(self):
        config = _tiny_config()
        n_classes = 4
        model = RDDLGNClassifier(config, n_classes, head_sizes=[32, 16], device=DEVICE)
        tokens = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, _ = model(tokens)
        assert logits.shape == (2, n_classes)


# ---------------------------------------------------------------------------
# Collapse（eval時の離散化）
# ---------------------------------------------------------------------------

class TestCollapse:

    def test_eval_produces_discrete_embedding(self):
        config = _tiny_config()
        model = RDDLGN(config, device=DEVICE)
        model.eval()
        src = torch.randint(0, config.vocab_size, (2, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (2, config.seq_len))
        with torch.no_grad():
            logits, _ = model(src, tgt)
        # logitsは実数値（GroupSumの出力）
        assert logits.shape == (2, config.seq_len, config.vocab_size)

    def test_eval_embedding_is_binary(self):
        config = _tiny_config()
        model = RDDLGN(config, device=DEVICE)
        model.eval()
        src = torch.randint(0, config.vocab_size, (2, config.seq_len))
        with torch.no_grad():
            emb_out = model.src_embedding(src)
        unique = torch.unique(emb_out)
        assert all(v in (0.0, 1.0) for v in unique.tolist())
