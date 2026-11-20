"""統合: 構造化固定配線がレシピ(ConnectionsConfig)→ runner → LutLayer → IR まで通る。

- 全層指定 butterfly / local1d / local2d と、per_layer の rot_n / butterfly / local1d /
  local2d / rot(step) が、各層で core.connections の生成器の出力と bitwise 一致する
- 設定の検証(長さ・入力数の不整合)が ValueError になる
- export_ir → simulate が hard eval と一致する(配線は IR の indices 表にしか残らない)
"""

import numpy as np
import pytest
import torch

from griffith_logic.core.connections import (
    dedupe_connections,
    get_butterfly_connections,
    get_local1d_connections,
    get_local2d_connections,
    get_rot_connections,
    get_rot_n_connections,
    local2d_offset_schedule,
)
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.layers.lut import LutLayer
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import ResolvedData, _build_model


def _data(in_width, out_width=2):
    return ResolvedData(
        name='synthetic', kind='classification', in_width=in_width,
        out_width=out_width, train_loader=None, eval_loader=None,
    )


def _luts(model):
    return [m for m in model.modules() if isinstance(m, LutLayer)]


def _build(*overrides, in_width=16, k=8, layers=3, n_inputs=2, seed=0):
    cfg = load_config(None, [
        'model.kind=dlgn', f'model.k={k}', f'model.layers={layers}',
        f'model.n_inputs={n_inputs}', *overrides,
    ])
    return _build_model(cfg.model, _data(in_width), 'cpu', seed=seed)


def _config_error(*overrides, **kw):
    with pytest.raises(ValueError):
        _build(*overrides, **kw)


class TestButterflyRecipe:
    def test_two_input_default_strides(self):
        luts = _luts(_build('model.connections.kind=butterfly'))
        assert torch.equal(luts[0].indices, get_rot_n_connections(16, 8, [0, 1], step=2))
        assert torch.equal(luts[1].indices, get_butterfly_connections(8, 8, 2, 1))
        assert torch.equal(luts[2].indices, get_butterfly_connections(8, 8, 2, 2))

    def test_desc_order_and_explicit_strides(self):
        desc = _luts(_build('model.connections.kind=butterfly',
                            'model.connections.order=desc'))
        assert torch.equal(desc[1].indices, get_butterfly_connections(8, 8, 2, 2))
        assert torch.equal(desc[2].indices, get_butterfly_connections(8, 8, 2, 1))
        explicit = _luts(_build('model.connections.kind=butterfly',
                                'model.connections.strides=[3,5]'))
        assert torch.equal(explicit[1].indices, get_butterfly_connections(8, 8, 2, 3))
        assert torch.equal(explicit[2].indices, get_butterfly_connections(8, 8, 2, 5))

    def test_bf6_small_case_with_dedupe(self):
        """先行実験 BF6 の小ケース: in=24, W=12, n=6、3 層(gather + stride 1, 6)、dedupe"""
        luts = _luts(_build(
            'model.connections.kind=butterfly', 'model.connections.dedupe=true',
            in_width=24, k=12, layers=3, n_inputs=6))
        assert torch.equal(luts[0].indices, get_rot_n_connections(24, 12, range(6), step=6))
        assert torch.equal(luts[1].indices, get_butterfly_connections(12, 12, 6, 1))
        assert torch.equal(
            luts[2].indices,
            dedupe_connections(get_butterfly_connections(12, 12, 6, 6), 12))
        assert all(int(lut.indices.shape[0]) == 6 for lut in luts)

    def test_strides_length_checked(self):
        _config_error('model.connections.kind=butterfly', 'model.connections.strides=[1]')


class TestLocal1dRecipe:
    def test_strides_per_layer(self):
        luts = _luts(_build('model.connections.kind=local1d',
                            'model.connections.strides=[1,2,4]'))
        assert torch.equal(luts[0].indices, get_local1d_connections(16, 8, 2, 1))
        assert torch.equal(luts[1].indices, get_local1d_connections(8, 8, 2, 2))
        assert torch.equal(luts[2].indices, get_local1d_connections(8, 8, 2, 4))

    def test_strides_required(self):
        _config_error('model.connections.kind=local1d')
        _config_error('model.connections.kind=local1d', 'model.connections.strides=[1,2]')


class TestLocal2dRecipe:
    def test_image_layer0_and_ring_hidden(self):
        """n_inputs == 2·bits_per_cell なら層 0 は右隣・下隣(先行実験候補 C の層 0)"""
        luts = _luts(_build(
            'model.connections.kind=local2d', 'model.connections.grid=[3,4]',
            in_width=12, k=8, layers=3))
        assert torch.equal(
            luts[0].indices, get_local2d_connections(12, 8, [(1, 0), (0, 1)], (3, 4), 1))
        sched = local2d_offset_schedule(2, 2, 'contig')
        assert torch.equal(luts[1].indices, get_local2d_connections(8, 8, sched[0]))
        assert torch.equal(luts[2].indices, get_local2d_connections(8, 8, sched[1]))

    def test_default_self_cell_when_bits_equal_n_inputs(self):
        """n_inputs == bits_per_cell なら層 0 は自セル(2 bit × 1 セル)"""
        luts = _luts(_build(
            'model.connections.kind=local2d', 'model.connections.grid=[2,3]',
            'model.connections.bits_per_cell=2', in_width=12, k=8, layers=2))
        assert torch.equal(luts[0].indices, get_local2d_connections(12, 8, [(0, 0)], (2, 3), 2))

    def test_stride_assign_and_explicit_layer0(self):
        luts = _luts(_build(
            'model.connections.kind=local2d', 'model.connections.grid=[2,3]',
            'model.connections.bits_per_cell=2', 'model.connections.assign=stride',
            'model.connections.layer0_offsets=[[0,0]]', in_width=12, k=8, layers=3))
        assert torch.equal(luts[0].indices, get_local2d_connections(12, 8, [(0, 0)], (2, 3), 2))
        sched = local2d_offset_schedule(2, 2, 'stride')
        assert torch.equal(luts[1].indices, get_local2d_connections(8, 8, sched[0]))
        assert torch.equal(luts[2].indices, get_local2d_connections(8, 8, sched[1]))

    def test_l2d4_like_default_self_cell(self):
        """n_inputs == bits_per_cell なら層 0 は自セルの 4 bit(先行実験 L2D4)"""
        luts = _luts(_build(
            'model.connections.kind=local2d', 'model.connections.grid=[2,2]',
            'model.connections.bits_per_cell=4', in_width=16, k=8, layers=2, n_inputs=4))
        assert torch.equal(luts[0].indices, get_local2d_connections(16, 8, [(0, 0)], (2, 2), 4))
        assert torch.equal(
            luts[1].indices,
            get_local2d_connections(8, 8, local2d_offset_schedule(1, 4)[0]))

    def test_offsets_times_bits_must_equal_n_inputs(self):
        _config_error('model.connections.kind=local2d', 'model.connections.bits_per_cell=3',
                      in_width=12)
        _config_error('model.connections.kind=local2d',
                      'model.connections.layer0_offsets=[[1,0],[0,1],[1,1]]')


class TestPerLayerStructured:
    def test_new_kinds_per_layer(self):
        luts = _luts(_build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: rot_n, shifts: [0, 1], step: 2}, '
            '{kind: butterfly, stride: 3}, '
            '{kind: local2d, offsets: [[1, 0], [0, 1]]}]}'))
        assert torch.equal(luts[0].indices, get_rot_n_connections(16, 8, [0, 1], step=2))
        assert torch.equal(luts[1].indices, get_butterfly_connections(8, 8, 2, 3))
        assert torch.equal(luts[2].indices, get_local2d_connections(8, 8, [(1, 0), (0, 1)]))

    def test_rot_step_and_local1d_per_layer(self):
        luts = _luts(_build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: rot, s_a: 1, s_b: 3, step: 2}, {kind: local1d, stride: 2}, '
            '{kind: local1d, stride: 4, dedupe: true}]}'))
        assert torch.equal(luts[0].indices, get_rot_connections(16, 8, 1, 3, step=2))
        assert torch.equal(luts[1].indices, get_local1d_connections(8, 8, 2, 2))
        assert torch.equal(luts[2].indices, get_local1d_connections(8, 8, 2, 4))

    def test_top_level_rot_step_list(self):
        luts = _luts(_build(
            'model.connections.kind=rot', 'model.connections.s_a=[1,0,0]',
            'model.connections.s_b=[3,2,4]', 'model.connections.step=[2,1,1]'))
        assert torch.equal(luts[0].indices, get_rot_connections(16, 8, 1, 3, step=2))
        assert torch.equal(luts[1].indices, get_rot_connections(8, 8, 0, 2))
        assert torch.equal(luts[2].indices, get_rot_connections(8, 8, 0, 4))

    def test_rot_without_step_is_unchanged(self):
        """step を省略した rot は従来の生成器と bitwise 同一(既存レシピの不変)"""
        luts = _luts(_build(
            'model.connections.kind=rot', 'model.connections.s_a=[1,0,0]',
            'model.connections.s_b=[3,2,4]'))
        assert torch.equal(luts[0].indices, get_rot_connections(16, 8, 1, 3))

    def test_validation(self):
        _config_error('model.connections.kind=rot', 'model.connections.s_a=[1,0,0]',
                      'model.connections.s_b=[3,2,4]', 'model.connections.step=[2,1]')
        _config_error('model.connections={kind: per_layer, layers: ['
                      '{kind: rot_n, shifts: [0, 1, 2]}, {kind: perm}, {kind: perm}]}')
        _config_error('model.connections={kind: per_layer, layers: ['
                      '{kind: local2d, offsets: [[1, 0]]}, {kind: perm}, {kind: perm}]}')
        _config_error('model.connections={kind: per_layer, layers: ['
                      '{kind: butterfly}, {kind: perm}, {kind: perm}]}')
        _config_error('model.connections={kind: per_layer, layers: ['
                      '{kind: rot_n}, {kind: perm}, {kind: perm}]}')


class TestIrRoundTrip:
    @pytest.mark.parametrize(('overrides', 'kw'), [
        (('model.connections.kind=butterfly',), {}),
        (('model.connections.kind=butterfly', 'model.connections.dedupe=true'),
         dict(in_width=24, k=12, n_inputs=6)),
        (('model.connections.kind=local1d', 'model.connections.strides=[1,2,4]'), {}),
        (('model.connections.kind=local2d', 'model.connections.grid=[4,4]'), {}),
        (('model.connections.kind=local2d', 'model.connections.grid=[2,2]',
          'model.connections.bits_per_cell=4'), dict(n_inputs=4)),
        (('model.connections.kind=rot', 'model.connections.s_a=[0,0,0]',
          'model.connections.s_b=[1,2,3]', 'model.connections.step=[2,1,1]'), {}),
    ])
    def test_export_ir_simulate_matches_hard_eval(self, overrides, kw):
        in_width = kw.get('in_width', 16)
        model = _build(*overrides, **kw).eval()
        ir = export_ir(model, input_width=in_width)
        x = np.random.default_rng(0).integers(0, 2, size=(64, in_width)).astype(np.int64)
        with torch.no_grad():
            y = model(torch.tensor(x, dtype=torch.float32))
        hard = np.round(y.numpy()).astype(np.int64)
        assert np.array_equal(hard, simulate(ir, x))
