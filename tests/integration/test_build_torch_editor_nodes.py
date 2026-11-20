"""build_torch がエディタ経由で受ける新しい IR 形を組めること。

エディタは IR を build_torch で PyTorch モデルに組み、学習・評価・build 時の再抽出
(export_ir)を行う。本ブランチが足した IR 形について次を検証する:
    - thermometer の 8 入力 lut_layer → FixedLutLayer(eval が simulate と一致、再抽出で不変)
    - GroupSum(assign='stride') の非ネイティブ(1 入力恒等 lut_layer)→ FixedLutLayer、
      ネイティブ(permute)→ Permute。どちらも eval 一致、backward が恒等 LUT を通る
    - conv2d ノードの train_info.estimator(sigmoid 以外のときだけ)の往復
    - draft ノード(initialized=False)の構造化配線 train_info.connections + wiring が
      core.connections の生成器と一致
    - FixedLutLayer / Permute 自体の意味論(表引き = 期待値、順列)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from griffith_logic import FixedLutLayer, GroupSum, LutLayer, Permute
from griffith_logic.conv import LutConv2d
from griffith_logic.core.connections import (
    dedupe_connections,
    get_butterfly_connections,
    get_local2d_connections,
    get_rot_connections,
)
from griffith_logic.encoding.thermometer import ThermometerEncoder
from griffith_logic.export import export_ir
from griffith_logic.ir import (
    IREdge,
    IRGraph,
    IRNode,
    IRPort,
    build_torch,
    ir_from_dict,
    ir_to_dict,
    simulate,
    validate,
)
from griffith_logic.ir.build import NCHWAdapter

BATCH = 24


def _roundtrip_graph(ir):
    """dict 化 → 復元 → validate(エディタが PUT / build で通す経路)"""
    graph = ir_from_dict(ir_to_dict(ir))
    validate(graph)
    return graph


def _bits(batch, width, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).float()


def _int_scores(model, x):
    with torch.no_grad():
        return model(x).round().to(torch.int64).numpy()


# --- thermometer(8 入力 LUT) ---

FEATURES, BITS, CODE_BITS = 4, 3, 8


def _thermo_model():
    torch.manual_seed(0)
    enc = ThermometerEncoder(BITS, input_bits=CODE_BITS)
    return nn.Sequential(
        enc, LutLayer(FEATURES * BITS, 8, n_inputs=2), GroupSum(k=2, tau=1.0),
    ).eval()


def _codes(batch, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1 << CODE_BITS, (batch, FEATURES), generator=g)


def _code_bits(codes):
    """符号 c を LSB 先頭の CODE_BITS ビット(位置 d·B + j)に展開する"""
    j = torch.arange(CODE_BITS)
    return ((codes[:, :, None] >> j) & 1).reshape(codes.shape[0], -1).float()


class TestThermometerRebuild:
    def test_rebuilt_matches_simulate_and_torch(self):
        model = _thermo_model()
        graph = _roundtrip_graph(export_ir(model, input_width=FEATURES * CODE_BITS))
        core = build_torch(graph)
        assert isinstance(core[0], FixedLutLayer)
        assert core[0].n_inputs == CODE_BITS
        core.eval()

        codes = _codes(BATCH)
        bits = _code_bits(codes)
        q = codes.float() / ((1 << CODE_BITS) - 1)
        want = _int_scores(model, q)
        assert np.array_equal(_int_scores(core, bits), want)
        assert np.array_equal(simulate(graph, bits.numpy().astype(np.int64)), want)

    def test_reexport_keeps_node(self):
        model = _thermo_model()
        ir = export_ir(model, input_width=FEATURES * CODE_BITS)
        core = build_torch(_roundtrip_graph(ir)).eval()
        ir2 = export_ir(core, input_width=FEATURES * CODE_BITS)
        a, b = ir.nodes[0], ir2.nodes[0]
        assert (a.type, b.type) == ('lut_layer', 'lut_layer')
        assert b.params['n_inputs'] == CODE_BITS
        assert np.array_equal(np.asarray(a.params['indices']), np.asarray(b.params['indices']))
        assert list(a.params['truth_tables']) == list(b.params['truth_tables'])
        assert b.train_info == a.train_info
        assert b.train_info['kind'] == 'thermometer'

    def test_frozen_and_downstream_trains(self):
        core = build_torch(_roundtrip_graph(
            export_ir(_thermo_model(), input_width=FEATURES * CODE_BITS)))
        assert list(core[0].parameters()) == []
        core.train()
        loss = core(_code_bits(_codes(BATCH))).sum()
        loss.backward()
        assert core[1].weights.grad is not None
        assert torch.isfinite(loss)


# --- GroupSum(assign='stride') ---

IN_W, HID, K = 16, 12, 4


def _stride_model():
    torch.manual_seed(2)
    return nn.Sequential(
        LutLayer(IN_W, HID, n_inputs=2), GroupSum(k=K, tau=1.0, assign='stride'),
    ).eval()


class TestStrideRebuild:
    def test_non_native_identity_lut(self):
        model = _stride_model()
        graph = _roundtrip_graph(export_ir(model, input_width=IN_W))
        core = build_torch(graph).eval()
        assert isinstance(core[1], FixedLutLayer) and core[1].n_inputs == 1
        x = _bits(BATCH, IN_W)
        want = _int_scores(model, x)
        assert np.array_equal(_int_scores(core, x), want)
        assert np.array_equal(simulate(graph, x.numpy().astype(np.int64)), want)

    def test_backward_flows_through_identity_lut(self):
        core = build_torch(_roundtrip_graph(export_ir(_stride_model(), input_width=IN_W)))
        core.train()
        loss = core(_bits(BATCH, IN_W)).pow(2).sum()
        loss.backward()
        grad = core[0].weights.grad
        assert grad is not None and float(grad.abs().sum()) > 0

    def test_non_native_reexport_keeps_identity_node(self):
        ir = export_ir(_stride_model(), input_width=IN_W)
        core = build_torch(_roundtrip_graph(ir)).eval()
        ir2 = export_ir(core, input_width=IN_W)
        assert [n.type for n in ir2.nodes] == [n.type for n in ir.nodes]
        a, b = ir.nodes[1], ir2.nodes[1]
        assert np.array_equal(np.asarray(a.params['indices']), np.asarray(b.params['indices']))
        assert list(a.params['truth_tables']) == list(b.params['truth_tables'])
        assert b.train_info == a.train_info

    def test_native_permute(self):
        model = _stride_model()
        ir = export_ir(model, input_width=IN_W, native_conv=True)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'permute', 'group_sum']
        graph = _roundtrip_graph(ir)
        core = build_torch(graph).eval()
        assert isinstance(core[1], Permute)
        x = _bits(BATCH, IN_W)
        want = _int_scores(model, x)
        assert np.array_equal(_int_scores(core, x), want)
        assert np.array_equal(simulate(graph, x.numpy().astype(np.int64)), want)
        # 再抽出しても permute ノードのまま(params 不変)
        ir2 = export_ir(core, input_width=IN_W)
        assert [n.type for n in ir2.nodes] == ['lut_layer', 'permute', 'group_sum']
        assert ir2.nodes[1].params == ir.nodes[1].params


# --- conv2d の estimator ---

class TestConvEstimator:
    @staticmethod
    def _model(estimator):
        torch.manual_seed(3)
        conv = LutConv2d(
            1, 4, kernel_size=3, tree_depth=2, n_inputs=2,
            parameterization='independent', estimator=estimator,
        )
        return nn.Sequential(
            conv, nn.Flatten(), LutLayer(4 * 6 * 6, 8, n_inputs=2), GroupSum(k=2),
        ).eval()

    @staticmethod
    def _conv_of(core):
        return next(m.module for m in core.modules() if isinstance(m, NCHWAdapter))

    def test_sin_estimator_roundtrip(self):
        model = self._model('sin')
        ir = export_ir(model, input_shape=(1, 8, 8), native_conv=True)
        assert ir.nodes[0].train_info['estimator'] == 'sin'
        graph = _roundtrip_graph(ir)
        core = build_torch(graph).eval()
        assert self._conv_of(core).estimator == 'sin'
        x = torch.randint(0, 2, (BATCH, 1, 8, 8), generator=torch.Generator().manual_seed(4))
        x = x.float()
        want = _int_scores(model, x)
        assert np.array_equal(_int_scores(nn.Sequential(nn.Flatten(), core), x), want)
        assert np.array_equal(simulate(graph, x.reshape(BATCH, -1).numpy().astype(np.int64)), want)
        ir2 = export_ir(core, input_shape=(1, 8, 8), native_conv=True)
        assert ir2.nodes[0].train_info['estimator'] == 'sin'

    def test_default_sigmoid_has_no_key(self):
        ir = export_ir(self._model('sigmoid'), input_shape=(1, 8, 8), native_conv=True)
        assert 'estimator' not in ir.nodes[0].train_info
        assert self._conv_of(build_torch(_roundtrip_graph(ir))).estimator == 'sigmoid'


# --- draft ノードの構造化配線 ---

def _pass_through_tt(n_inputs):
    return '0x' + 'a' * ((1 << n_inputs) // 4)


def _placeholder_indices(in_w, out_w, n):
    return [[(o * n + j) % in_w for o in range(out_w)] for j in range(n)]


def _draft_graph(connections, wiring, *, in_w=32, out_w=64, n=2, initialized=False,
                 indices=None):
    node = IRNode(
        id='u1_lut_layer', type='lut_layer',
        params={
            'in_width': in_w, 'out_width': out_w, 'n_inputs': n,
            'indices': indices if indices is not None else _placeholder_indices(in_w, out_w, n),
            'truth_tables': [_pass_through_tt(n)] * out_w,
        },
        train_info={
            'parameterization': 'softmax', 'connections': connections, 'selection': 'soft',
            'initialized': initialized, **({'wiring': wiring} if wiring is not None else {}),
        },
    )
    gs = IRNode(id='u2_group_sum', type='group_sum',
                params={'in_width': out_w, 'k': 4, 'tau': 1.0})
    graph = IRGraph(
        metadata={'name': 'draft'},
        inputs=[IRPort(id='x', width=in_w)], outputs=[IRPort(id='y', width=4)],
        nodes=[node, gs],
        edges=[IREdge(src='x', dst='u1_lut_layer'), IREdge(src='u1_lut_layer', dst='u2_group_sum'),
               IREdge(src='u2_group_sum', dst='y')],
    )
    validate(graph)
    return graph


class TestDraftStructuredWiring:
    def test_rot_with_step(self):
        core = build_torch(_draft_graph('rot', {'s_a': 0, 's_b': 1, 'step': 3}))
        want = get_rot_connections(32, 64, 0, 1, 'cpu', step=3)
        assert torch.equal(core[0].indices, want)

    def test_butterfly_with_dedupe(self):
        core = build_torch(_draft_graph('butterfly', {'stride': 4, 'dedupe': True}, in_w=64))
        want = dedupe_connections(get_butterfly_connections(64, 64, 2, 4), 64)
        assert torch.equal(core[0].indices, want)

    def test_local2d_grid(self):
        core = build_torch(_draft_graph(
            'local2d', {'offsets': [[0, 0], [0, 1]], 'grid': [8, 8]}, in_w=64, out_w=40))
        want = get_local2d_connections(64, 40, [(0, 0), (0, 1)], grid=(8, 8))
        assert torch.equal(core[0].indices, want)

    def test_draft_trains_after_build(self):
        core = build_torch(_draft_graph('butterfly', {'stride': 2}, in_w=64))
        core.train()
        loss = core(_bits(BATCH, 64)).sum()
        loss.backward()
        assert torch.isfinite(loss)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match='未対応'):
            build_torch(_draft_graph('spiral', None))

    def test_missing_argument_rejected(self):
        with pytest.raises(ValueError, match='shifts'):
            build_torch(_draft_graph('rot_n', {'step': 2}))

    def test_initialized_node_keeps_ir_indices(self):
        """学習済みノードは wiring を無視して IR の indices を復元する"""
        idx = get_rot_connections(32, 64, 5, 7, 'cpu').tolist()
        core = build_torch(_draft_graph(
            'butterfly', {'stride': 4}, initialized=True, indices=idx))
        assert core[0].indices.tolist() == idx


# --- FixedLutLayer / Permute の意味論 ---

class TestFixedLutLayer:
    @staticmethod
    def _layer(n=3, out=5, in_dim=7, seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, in_dim, (n, out), generator=g)
        tts = torch.randint(0, 1 << (1 << n), (out,), generator=g).tolist()
        return FixedLutLayer(in_dim, out, n, idx, tts), idx, tts

    def test_soft_expectation_matches_bruteforce(self):
        layer, idx, tts = self._layer()
        g = torch.Generator().manual_seed(1)
        x = torch.rand(BATCH, 7, generator=g)
        got = layer(x)
        want = torch.zeros(BATCH, 5)
        for o in range(5):
            for k in range(8):
                if not (tts[o] >> k) & 1:
                    continue
                term = torch.ones(BATCH)
                for j in range(3):
                    v = x[:, idx[j, o]]
                    term = term * (v if (k >> j) & 1 else 1 - v)
                want[:, o] += term
        assert torch.allclose(got, want, atol=1e-6)

    def test_binary_lookup_equals_expectation(self):
        layer, _, _ = self._layer()
        x = _bits(BATCH, 7)
        with torch.no_grad():
            lookup = layer(x)
        soft = layer(x.clone().requires_grad_(True))
        assert torch.equal(lookup, soft.detach())
        assert set(lookup.unique().tolist()) <= {0.0, 1.0}

    def test_hex_tables_and_validation(self):
        layer = FixedLutLayer(4, 2, 1, [[0, 3]], ['0x2', '0x1'])
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        assert layer(x).tolist() == [[1.0, 1.0], [0.0, 0.0]]
        with pytest.raises(ValueError, match='形状'):
            FixedLutLayer(4, 2, 2, [[0, 1]], ['0x2', '0x2'])
        with pytest.raises(ValueError, match='超え'):
            FixedLutLayer(4, 2, 1, [[0, 4]], ['0x2', '0x2'])
        with pytest.raises(ValueError, match='収まり'):
            FixedLutLayer(4, 2, 1, [[0, 1]], ['0x4', '0x2'])


class TestPermute:
    def test_matches_torch_permute(self):
        p = Permute((3, 4), (1, 0))
        x = torch.arange(2 * 12).float().reshape(2, 12)
        want = x.reshape(2, 3, 4).permute(0, 2, 1).reshape(2, 12)
        assert torch.equal(p(x), want)
        assert p.out_shape == (4, 3)

    def test_invalid_perm_rejected(self):
        with pytest.raises(ValueError, match='順列'):
            Permute((3, 4), (0, 0))
