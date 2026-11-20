"""RDDLGNAPTrainerのテスト"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_trainer_rddlgn import (
    RDDLGNAPTrainer,
    _build_stack_reach_maps,
    _stack_forward,
)

from griffith_logic import RDDLGN, RDDLGNConfig


def _tiny_config():
    """テスト用の極小構成"""
    return RDDLGNConfig(
        vocab_size=10,
        d_emb=16,
        seq_len=4,
        n_layer_sizes=[32, 32],
        k_layer_sizes=[48, 32],
        l_layer_sizes=[32, 32],
        p_layer_sizes=[48, 32],
        m_layer_sizes=[64, 64, 100],  # 100 = 10 * 10
        tau=1.0,
        n_inputs=2,
        parameterization='independent',
        connections='random',
    )


class TestStackReachMaps:
    def test_reach_map_shape(self):
        """reach-mapの長さが妥当"""
        config = _tiny_config()
        model = RDDLGN(config)
        reach = _build_stack_reach_maps(model.decoder.m_layers)
        # M-layersは3層
        assert len(reach) == 3
        for src, dst in reach:
            assert src.shape == dst.shape
            assert src.dtype == torch.long

    def test_last_layer_identity(self):
        """最終層のreach-mapは恒等"""
        config = _tiny_config()
        model = RDDLGN(config)
        reach = _build_stack_reach_maps(model.decoder.m_layers)
        src, dst = reach[-1]
        assert torch.equal(src, dst)
        assert len(src) == model.decoder.m_layers.out_dim


class TestStackForward:
    def test_forward_shape(self):
        """LutStack forwardの出力形状"""
        config = _tiny_config()
        model = RDDLGN(config)
        x = torch.rand(2, 16)  # (batch, d_emb)
        out, caches = _stack_forward(model.encoder.n_layers, x)
        assert out.shape == (2, 32)  # n_layer_sizes[-1]
        assert len(caches) == 2  # 2層

    def test_output_in_01(self):
        """出力が[0,1]範囲"""
        config = _tiny_config()
        model = RDDLGN(config)
        # 重みを[0,1]に初期化
        for m in model.modules():
            if hasattr(m, 'weights') and isinstance(m.weights, torch.nn.Parameter):
                m.weights.data.uniform_(0, 1)
        x = torch.rand(3, 16)
        out, _ = _stack_forward(model.encoder.n_layers, x)
        assert (out >= 0).all() and (out <= 1).all()


class TestRDDLGNAPTrainer:
    def test_init_rejects_softmax(self):
        """softmaxパラメータ化を拒否"""
        config = _tiny_config()
        config.parameterization = 'softmax'
        model = RDDLGN(config)
        with pytest.raises(ValueError, match='independent'):
            RDDLGNAPTrainer(model, lr=0.1)

    def test_weights_initialized_in_01(self):
        """初期化後の重みが[0,1]"""
        config = _tiny_config()
        model = RDDLGN(config)
        trainer = RDDLGNAPTrainer(model, lr=0.1)
        for m in model.modules():
            if hasattr(m, 'weights') and isinstance(m.weights, torch.nn.Parameter):
                assert (m.weights.data >= 0).all()
                assert (m.weights.data <= 1).all()

    def test_train_step_runs(self):
        """train_stepがエラーなく実行される"""
        config = _tiny_config()
        model = RDDLGN(config)
        trainer = RDDLGNAPTrainer(model, lr=0.1)

        src = torch.randint(3, 10, (2, 4))
        # tgt_full: [BOS, tokens, EOS]
        bos = torch.full((2, 1), 1, dtype=torch.long)
        eos = torch.full((2, 1), 2, dtype=torch.long)
        tgt_tokens = torch.randint(3, 10, (2, 4))
        tgt_full = torch.cat([bos, tgt_tokens, eos], dim=1)

        metrics = trainer.train_step(src, tgt_full)
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_weights_change_after_step(self):
        """train_step後にLUT重みが変化する"""
        torch.manual_seed(42)
        config = _tiny_config()
        model = RDDLGN(config)
        trainer = RDDLGNAPTrainer(model, lr=0.1)

        # M-layersの重みを記録
        m_weights_before = [
            l.weights.data.clone()
            for l in model.decoder.m_layers.layers
        ]

        src = torch.randint(3, 10, (2, 4))
        bos = torch.full((2, 1), 1, dtype=torch.long)
        eos = torch.full((2, 1), 2, dtype=torch.long)
        tgt_tokens = torch.randint(3, 10, (2, 4))
        tgt_full = torch.cat([bos, tgt_tokens, eos], dim=1)

        trainer.train_step(src, tgt_full)

        for wb, layer in zip(m_weights_before, model.decoder.m_layers.layers):
            assert not torch.equal(wb, layer.weights.data), \
                'M-layers重みが更新されていない'

    def test_embedding_changes(self):
        """train_step後にEmbedding重みが変化する"""
        torch.manual_seed(42)
        config = _tiny_config()
        model = RDDLGN(config)
        trainer = RDDLGNAPTrainer(model, lr=0.1)

        emb_before = model.src_embedding.embedding.weight.data.clone()

        src = torch.randint(3, 10, (4, 4))
        bos = torch.full((4, 1), 1, dtype=torch.long)
        eos = torch.full((4, 1), 2, dtype=torch.long)
        tgt_tokens = torch.randint(3, 10, (4, 4))
        tgt_full = torch.cat([bos, tgt_tokens, eos], dim=1)

        trainer.train_step(src, tgt_full)

        emb_after = model.src_embedding.embedding.weight.data
        assert not torch.equal(emb_before, emb_after), \
            'Embedding重みが更新されていない'

    def test_evaluate_runs(self):
        """evaluateがエラーなく実行される"""
        config = _tiny_config()
        model = RDDLGN(config)
        trainer = RDDLGNAPTrainer(model, lr=0.1)

        from torch.utils.data import DataLoader, TensorDataset
        src_data = torch.randint(3, 10, (8, 4))
        tgt_data = torch.randint(3, 10, (8, 4))
        ds = TensorDataset(src_data, tgt_data)
        loader = DataLoader(ds, batch_size=4)

        acc = trainer.evaluate(loader, 'cpu')
        assert 0.0 <= acc <= 1.0
