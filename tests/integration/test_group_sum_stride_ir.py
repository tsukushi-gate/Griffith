"""統合: GroupSum(assign='stride')の IR 化。

- 非ネイティブ抽出: 恒等 lut_layer(n_inputs=1)の転置 (m, k) → (k, m) + contig の group_sum
- ネイティブ抽出(native_conv=True): permute(in_shape [m, k], perm [1, 0]) + group_sum
- どちらも simulate が hard eval と一致し、非ネイティブは validate(target='hdl') を通る
- assign='contig'(既定)の IR は従来と同一(余分なノードが無い)
- レシピ model.group_assign=stride → runner が GroupSum(assign='stride') を組む
"""

import numpy as np
import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import ResolvedData, _build_model

K, WIDTH, IN = 4, 24, 16


def _model(assign='stride', seed=61):
    torch.manual_seed(seed)
    model = nn.Sequential(
        LutLayer(IN, WIDTH, n_inputs=2),
        LutLayer(WIDTH, WIDTH, n_inputs=2),
        GroupSum(k=K, tau=1.0, assign=assign),
    )
    return model.eval()


def _x(n=128, seed=3):
    return np.random.default_rng(seed).integers(0, 2, size=(n, IN)).astype(np.int64)


def _hard(model, x):
    with torch.no_grad():
        return np.round(model(torch.tensor(x, dtype=torch.float32)).numpy()).astype(np.int64)


class TestStrideExtraction:
    def test_non_native_is_identity_lut_permutation(self):
        model = _model()
        ir = export_ir(model, input_width=IN)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'lut_layer', 'lut_layer', 'group_sum']
        perm = ir.nodes[2]
        assert perm.id.endswith('_permute')
        assert perm.params['n_inputs'] == 1
        assert perm.params['in_width'] == WIDTH
        assert perm.params['out_width'] == WIDTH
        m = WIDTH // K
        idx = np.asarray(perm.params['indices']).reshape(-1).tolist()
        assert idx == [j * K + c for c in range(K) for j in range(m)]
        assert ir.nodes[3].params['k'] == K
        validate(ir, target='hdl')
        x = _x()
        assert np.array_equal(simulate(ir, x), _hard(model, x))

    def test_native_is_permute_node(self):
        model = _model()
        ir = export_ir(model, input_width=IN, native_conv=True)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'lut_layer', 'permute', 'group_sum']
        p = ir.nodes[2].params
        assert [int(v) for v in p['in_shape']] == [WIDTH // K, K]
        assert [int(v) for v in p['perm']] == [1, 0]
        validate(ir)
        x = _x()
        assert np.array_equal(simulate(ir, x), _hard(model, x))
        assert np.array_equal(simulate(ir, x), simulate(export_ir(model, input_width=IN), x))

    def test_stride_differs_from_contig_scores(self):
        """割当が違えばスコアも違う(一致テストが自明でないことの確認)"""
        x = _x()
        assert not np.array_equal(_hard(_model('stride'), x), _hard(_model('contig'), x))

    def test_contig_ir_unchanged(self):
        ir = export_ir(_model('contig'), input_width=IN)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'lut_layer', 'group_sum']
        assert 'lowered_from' not in (ir.nodes[-1].train_info or {})
        ir_native = export_ir(_model('contig'), input_width=IN, native_conv=True)
        assert [n.type for n in ir_native.nodes] == ['lut_layer', 'lut_layer', 'group_sum']


class TestRecipe:
    @staticmethod
    def _build(*overrides):
        cfg = load_config(None, ['model.kind=dlgn', 'model.k=8', 'model.layers=2', *overrides])
        data = ResolvedData(name='synthetic', kind='classification', in_width=IN,
                            out_width=2, train_loader=None, eval_loader=None)
        return cfg, _build_model(cfg.model, data, 'cpu', seed=0)

    def test_group_assign_stride(self):
        cfg, model = self._build('model.group_assign=stride')
        gs = [m for m in model.modules() if isinstance(m, GroupSum)][0]
        assert gs.assign == 'stride'
        model.eval()
        ir = export_ir(model, input_width=IN)
        x = _x()
        assert np.array_equal(simulate(ir, x), _hard(model, x))

    def test_default_is_contig(self):
        cfg, model = self._build()
        assert cfg.model.group_assign == 'contig'
        gs = [m for m in model.modules() if isinstance(m, GroupSum)][0]
        assert gs.assign == 'contig'
