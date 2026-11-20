"""blocks_to_ir(blocks部品モデル→IR)のテスト。

    - Residualダイヤモンド: validate通過+hard eval⇔simulate⇔build_torch
      の3段一致+Verilog文字列妥当性
    - LogicPositionalEncoding: constant+concat展開の往復一致
    - LutLayer→LogicBatchNorm→Binarize: fold前後のhard eval完全一致
    - 折り畳み不能なNorm→blackbox、LogicDropout→痕跡なし
    - LogicAttention/LogicTransformerBlock→blackbox+contract(validate通過)
    - 異常系: 素のConcat/Add、Residualのネスト、未対応モジュール
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import Binarize, GroupSum, LutLayer
from griffith_logic.blocks import (
    Add,
    Concat,
    LogicAttention,
    LogicBatchNorm,
    LogicDropout,
    LogicLayerNorm,
    LogicPositionalEncoding,
    LogicTransformerBlock,
    Residual,
    blocks_to_ir,
)
from griffith_logic.export.verilog import to_verilog
from griffith_logic.ir import build_torch, simulate, validate

BATCH = 64


def _binary_np(batch, width, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)


def _hard_forward(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32))
    return np.round(y.numpy()).astype(np.int64)


def _diamond_model(seed=11):
    """x(8) → lut → Residual(lut→lut, concat) → lut(16→8) のダイヤモンド"""
    torch.manual_seed(seed)
    return nn.Sequential(
        LutLayer(8, 8),
        Residual(nn.Sequential(LutLayer(8, 6), LutLayer(6, 8)),
                 merge='concat'),
        LutLayer(16, 8),
    )


class TestResidualDiamond:
    def test_validate_and_structure(self):
        graph = blocks_to_ir(_diamond_model(), 8, name='diamond')
        assert validate(graph) == []
        types = [n.type for n in graph.nodes]
        assert types.count('lut_layer') == 4
        assert types.count('concat') == 1
        # fanout: 先頭lutからinner先頭とconcat slot0の2本
        first = graph.nodes[0].id
        outs = [e for e in graph.edges if e.src == first]
        assert len(outs) == 2

    def test_three_way_equivalence(self):
        """3段一致: 元モデルhard eval ⇔ simulate ⇔ build_torch"""
        model = _diamond_model()
        graph = blocks_to_ir(model, 8)
        x = _binary_np(BATCH, 8)
        original = _hard_forward(model, x)
        sim = simulate(graph, x)
        rebuilt = _hard_forward(build_torch(graph), x)
        assert (original == sim).all(), '元モデルhard evalとsimulateが不一致'
        assert (sim == rebuilt).all(), 'simulateとbuild_torch往復が不一致'

    def test_verilog_string_valid(self):
        graph = blocks_to_ir(_diamond_model(), 8)
        src = to_verilog(graph)
        assert 'concat' in src and 'module logic_core' in src

    @pytest.mark.parametrize('merge', ['or', 'xor'])
    def test_merge_or_xor_roundtrip(self, merge, seed):
        model = nn.Sequential(
            LutLayer(8, 8),
            Residual(LutLayer(8, 8), merge=merge),
        )
        graph = blocks_to_ir(model, 8)
        assert validate(graph) == []
        add_nodes = [n for n in graph.nodes if n.type == 'add']
        assert len(add_nodes) == 1 and add_nodes[0].params['mode'] == merge
        x = _binary_np(BATCH, 8)
        assert (_hard_forward(model, x) == simulate(graph, x)).all()
        assert (_hard_forward(build_torch(graph), x)
                == simulate(graph, x)).all()

    def test_group_sum_semantics_propagated(self, seed):
        model = nn.Sequential(LutLayer(4, 8), GroupSum(k=2, tau=1.0))
        graph = blocks_to_ir(model, 4)
        assert graph.outputs[0].semantics == 'class_scores'


class TestPositionalEncoding:
    def test_constant_concat_expansion(self, seed):
        model = nn.Sequential(
            LogicPositionalEncoding(4, 3, code='gray'),
            LutLayer(17, 8),
        )
        graph = blocks_to_ir(model, 5)
        assert validate(graph) == []
        types = [n.type for n in graph.nodes]
        assert types.count('constant') == 1
        assert types.count('concat') == 1

        const = next(n for n in graph.nodes if n.type == 'constant')
        assert const.params['width'] == 12
        assert len(const.params['bits']) == 12

        x = _binary_np(BATCH, 5)
        assert (_hard_forward(model, x) == simulate(graph, x)).all()
        assert (_hard_forward(build_torch(graph), x)
                == simulate(graph, x)).all()

    def test_verilog_contains_hardwired_constant(self, seed):
        model = nn.Sequential(
            LogicPositionalEncoding(2, 2, code='binary'),
            LutLayer(8, 4),
        )
        graph = blocks_to_ir(model, 4)
        src = to_verilog(graph)
        # binaryコード表 [[0,0],[1,0]] → flatten [0,0,1,0] → 4'b0100
        assert "= 4'b0100;" in src


class TestBatchNormFold:
    def _folded_setup(self, seed=5):
        torch.manual_seed(seed)
        model = nn.Sequential(
            LutLayer(6, 8, parameterization='independent'),
            LogicBatchNorm(8),
            Binarize(),
        )
        # BNのrunning統計を実データで作る(fold意味論はeval時の統計に依存)
        model.train()
        for i in range(5):
            model(torch.randint(0, 2, (64, 6)).float())
        model.eval()
        return model

    def test_folds_to_single_lut_node(self, seed):
        model = self._folded_setup()
        graph = blocks_to_ir(model, 6)
        assert validate(graph) == []
        assert [n.type for n in graph.nodes] == ['lut_layer']
        assert graph.nodes[0].train_info['folded_from'] == \
            'LutLayer+LogicBatchNorm+Binarize'

    def test_fold_preserves_hard_eval_exact(self, seed):
        """fold前後のhard eval完全一致(全域: 2^6=64パターン全数)"""
        model = self._folded_setup()
        graph = blocks_to_ir(model, 6)
        x = np.array(
            [[(k >> j) & 1 for j in range(6)] for k in range(64)],
            dtype=np.int64,
        )
        assert (_hard_forward(model, x) == simulate(graph, x)).all(), \
            'fold前後でhard evalが不一致'

    def test_eval_deterministic(self, seed):
        """eval時の決定性: 同一入力で毎回同一、バッチ合成にも不変"""
        model = self._folded_setup()
        x = torch.randint(0, 2, (16, 6)).float()
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
            y3 = model(torch.cat([x, 1 - x]))[:16]
        assert torch.equal(y1, y2)
        assert torch.equal(y1, y3)


class TestNormBlackbox:
    def test_unfoldable_batchnorm_becomes_blackbox(self, seed):
        """Binarizeが続かないBN → blackbox(幅in/out付き)"""
        model = nn.Sequential(LutLayer(6, 8), LogicBatchNorm(8))
        graph = blocks_to_ir(model, 6)
        assert validate(graph) == []
        bb = [n for n in graph.nodes if n.type == 'blackbox']
        assert len(bb) == 1
        assert bb[0].params['torch_class'] == 'LogicBatchNorm'
        assert bb[0].params['in_width'] == 8
        assert bb[0].params['out_width'] == 8

    def test_layernorm_always_blackbox(self, seed):
        """LayerNormは特徴横断のためBinarizeが続いても折り畳まない"""
        model = nn.Sequential(LutLayer(6, 8), LogicLayerNorm(8), Binarize())
        graph = blocks_to_ir(model, 6)
        types = [n.type for n in graph.nodes]
        assert types == ['lut_layer', 'blackbox', 'binarize']


class TestDropoutNoTrace:
    def test_dropout_leaves_no_trace(self, seed):
        torch.manual_seed(3)
        with_drop = nn.Sequential(
            LutLayer(6, 8), LogicDropout(0.5, 'neutral'), LutLayer(8, 4),
        )
        graph = blocks_to_ir(with_drop, 6)
        assert [n.type for n in graph.nodes] == ['lut_layer', 'lut_layer']
        assert validate(graph) == []
        # eval時identityなのでIRのsimulateと元モデルのhard evalは一致
        x = _binary_np(BATCH, 6)
        assert (_hard_forward(with_drop, x) == simulate(graph, x)).all()


class TestAttentionBlackbox:
    def test_transformer_block_blackbox_contract(self, seed):
        torch.manual_seed(4)
        model = nn.Sequential(LogicTransformerBlock(4, n_heads=2))
        graph = blocks_to_ir(model, 12)  # S=3, d_model=4
        assert validate(graph) == []
        bb = graph.nodes[0]
        assert bb.type == 'blackbox'
        assert bb.params['torch_class'] == 'LogicTransformerBlock'
        assert bb.params['in_width'] == 12
        assert bb.params['out_width'] == 12
        assert 'contract' in bb.train_info

    def test_attention_blackbox_contract(self, seed):
        torch.manual_seed(5)
        model = nn.Sequential(LogicAttention(4, v_sizes=[8]))
        graph = blocks_to_ir(model, 12)
        bb = graph.nodes[0]
        assert bb.type == 'blackbox'
        assert bb.params['in_width'] == 12
        assert bb.params['out_width'] == 24  # S=3 × d_v=8

    def test_width_not_multiple_of_d_model(self, seed):
        model = nn.Sequential(LogicTransformerBlock(5))
        with pytest.raises(ValueError, match='倍数'):
            blocks_to_ir(model, 12)


class TestErrors:
    def test_bare_concat_rejected(self):
        with pytest.raises(NotImplementedError, match='Concat'):
            blocks_to_ir(nn.Sequential(Concat()), 4)

    def test_bare_add_rejected(self):
        with pytest.raises(NotImplementedError, match='Add'):
            blocks_to_ir(nn.Sequential(Add('or')), 4)

    def test_nested_residual_rejected(self, seed):
        model = nn.Sequential(
            Residual(Residual(LutLayer(4, 4), merge='xor'), merge='xor'),
        )
        with pytest.raises(NotImplementedError, match='ネスト'):
            blocks_to_ir(model, 4)

    def test_unsupported_module_rejected(self):
        with pytest.raises(NotImplementedError, match='Linear'):
            blocks_to_ir(nn.Sequential(nn.Linear(4, 4)), 4)

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match='生成しない'):
            blocks_to_ir(nn.Sequential(LogicDropout(0.1)), 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
