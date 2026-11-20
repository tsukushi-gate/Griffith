"""統合: LTTLayerの IR 化。

- export_ir → lut_layer(n_inputs = n、真理値表は全パターン列挙)、validate(hdl) 通過
- simulate が eval forward(× τ の整数スコア)と一致。connections='learnable' は argmax 配線
- build_torch で LutLayer に戻しても hard eval が一致(IR の真理値表が自己完結)
"""

import numpy as np
import torch
import torch.nn as nn

from griffith_logic import GroupSum
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.layers.ltt import LTTLayer

IN, TAU = 16, 2.0


def _model(connections='random', seed=71):
    torch.manual_seed(seed)
    model = nn.Sequential(
        LTTLayer(IN, 24, n_inputs=6, connections=connections),
        LTTLayer(24, 24, n_inputs=6, connections='random'),
        GroupSum(k=4, tau=TAU),
    )
    return model.eval()


def _x(n=128, seed=3):
    return np.random.default_rng(seed).integers(0, 2, size=(n, IN)).astype(np.int64)


def _hard_scores(model, x):
    with torch.no_grad():
        return np.round(model(torch.tensor(x, dtype=torch.float32)).numpy() * TAU).astype(np.int64)


class TestLttIr:
    def test_nodes_and_simulate(self):
        model = _model()
        ir = export_ir(model, input_width=IN)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'lut_layer', 'group_sum']
        first = ir.nodes[0]
        assert first.params['n_inputs'] == 6
        assert first.params['in_width'] == IN and first.params['out_width'] == 24
        assert np.asarray(first.params['indices']).shape == (6, 24)
        assert first.train_info['kind'] == 'ltt'
        validate(ir, target='hdl')
        x = _x()
        assert np.array_equal(simulate(ir, x), _hard_scores(model, x))

    def test_learnable_mapping_uses_argmax(self):
        model = _model(connections='learnable')
        ir = export_ir(model, input_width=IN)
        idx = np.asarray(ir.nodes[0].params['indices'])
        assert np.array_equal(idx, model[0].effective_indices().numpy())
        x = _x(seed=5)
        assert np.array_equal(simulate(ir, x), _hard_scores(model, x))

    def test_build_torch_roundtrip(self):
        model = _model()
        ir = export_ir(model, input_width=IN)
        rebuilt = build_torch(ir).eval()
        x = _x(seed=9)
        with torch.no_grad():
            y = rebuilt(torch.tensor(x, dtype=torch.float32))
        assert np.array_equal(np.round(y.numpy() * TAU).astype(np.int64), _hard_scores(model, x))

    def test_not_trivially_constant(self):
        """一致テストが自明でないこと: 出力が入力で変わり、クラスも偏りきらない"""
        model = _model()
        scores = _hard_scores(model, _x())
        assert len({tuple(r) for r in scores.tolist()}) > 1
