"""統合: conv-DLGNの組み立て → レシピ → IR。

- build_conv_dlgn の形(ブロック・flat・cdlgn 頭部則・rot2 の (r, s_b))が先行実験の値と一致
- 頭部 LutLayer の配線が get_rot_connections(step=r) と bitwise 一致、GroupSum が stride
- export_ir(native_conv=True / False)の simulate が hard eval と一致(validate 通過)
- model.kind=conv_dlgn のレシピから runner が同じモデルを組む。設定の検証が効く
"""

import numpy as np
import pytest
import torch

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.core.connections import get_rot_connections
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.models.conv_dlgn import (
    ConvDLGNConfig,
    build_conv_dlgn,
    conv_dlgn_shape,
    head_dims_cdlgn,
    rot2_schedule,
)
from griffith_logic.pooling import OrPooling2d
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import ResolvedData, _build_model

IN_SHAPE = (2, 8, 8)


def _cfg(**kw):
    base = dict(in_shape=IN_SHAPE, channels=(4, 8), sampler='dilated8', head_dims=(32, 16),
                num_classes=4, tau=2.0)
    base.update(kw)
    return ConvDLGNConfig(**base)


def _images(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, *IN_SHAPE), generator=g).float()


def _hard(model, x):
    """hard eval の整数スコア(GroupSum の出力 × τ。simulate の group_sum は τ で割らない)"""
    tau = [m for m in model if isinstance(m, GroupSum)][0].tau
    with torch.no_grad():
        return np.round(model(x).numpy() * tau).astype(np.int64)


class TestSchedules:
    def test_cdlgn_head_dims(self):
        assert head_dims_cdlgn(4096) == [4096, 40960, 20480, 10240]
        assert head_dims_cdlgn(32768, scale=0.25) == [32768, 81920, 40960, 20480]
        with pytest.raises(ValueError):
            head_dims_cdlgn(4096, scale=0.3)      # 最終層が 10 で割り切れない

    def test_rot2_schedule_exp098_values(self):
        """先行実験 rot2_schedule の docstring の値(cifar10_s / cifar10_m)"""
        assert rot2_schedule([4096, 40960, 20480, 10240]) == [(1, 8), (2, 203), (2, 1713)]
        assert rot2_schedule([32768, 327680, 163840, 81920]) == [(1, 13), (2, 573), (2, 8145)]

    def test_rot2_reduction_layers_cover_input_exactly_once(self):
        """縮小層(r=2・s_b 奇数)は被覆率 100%・fan-out 1(先行実験 selfcheck T10)"""
        dims = [4096, 40960, 20480, 10240]
        for layer, (r, s_b) in enumerate(rot2_schedule(dims)):
            if r == 1:
                continue
            idx = get_rot_connections(dims[layer], dims[layer + 1], 0, s_b, step=r)
            counts = torch.bincount(idx.reshape(-1), minlength=dims[layer])
            assert torch.equal(counts, torch.ones(dims[layer], dtype=torch.int64))


class TestBuild:
    def test_shape_and_modules(self):
        cfg = _cfg()
        shape = conv_dlgn_shape(cfg)
        assert shape['kernel'] == 5 and shape['padding'] == 2 and shape['tree_depth'] == 3
        assert shape['blocks'] == [(4, 4, 4), (8, 2, 2)]      # 8 → pool 4 → pool 2
        assert shape['flat'] == 32
        assert shape['head_dims'] == [32, 32, 16]
        model = build_conv_dlgn(cfg)
        kinds = [type(m).__name__ for m in model]
        assert kinds == ['LutConv2d', 'OrPooling2d', 'LutConv2d', 'OrPooling2d', 'Flatten',
                         'LutLayer', 'LutLayer', 'GroupSum']
        convs = [m for m in model if isinstance(m, LutConv2d)]
        assert [c.layer_idx for c in convs] == [0, 1]
        assert all(c.sampler == 'dilated8' and c.estimator == 'sin' for c in convs)
        gs = [m for m in model if isinstance(m, GroupSum)][0]
        assert gs.assign == 'stride' and gs.k == 4 and gs.tau == 2.0

    def test_head_rot2_wiring_matches_generator(self):
        model = build_conv_dlgn(_cfg())
        luts = [m for m in model if isinstance(m, LutLayer)]
        dims = [32, 32, 16]
        for layer, ((r, s_b), lut) in enumerate(zip(rot2_schedule(dims), luts, strict=True)):
            assert torch.equal(lut.indices, get_rot_connections(dims[layer], dims[layer + 1],
                                                                0, s_b, step=r))
            assert lut.estimator == 'sin' and lut.parameterization == 'independent'

    def test_stride_downsample_and_random_head(self):
        cfg = _cfg(downsample='stride', head_wiring='random', sampler='random8',
                   group_assign='contig')
        shape = conv_dlgn_shape(cfg)
        assert shape['blocks'] == [(4, 4, 4), (8, 2, 2)]
        model = build_conv_dlgn(cfg, generator=torch.Generator().manual_seed(0))
        assert not any(isinstance(m, OrPooling2d) for m in model)
        assert all(c.stride == (2, 2) for c in model if isinstance(c, LutConv2d))

    def test_default_cdlgn_head_when_head_dims_omitted(self):
        cfg = _cfg(head_dims=None, num_classes=2)
        assert conv_dlgn_shape(cfg)['head_dims'] == [32, 320, 160, 16]

    def test_invalid_configs(self):
        with pytest.raises(ValueError):
            conv_dlgn_shape(_cfg(head_dims=(32, 15)))          # 最終層が num_classes の倍数でない
        with pytest.raises(ValueError):
            conv_dlgn_shape(_cfg(sampler='grid9'))
        with pytest.raises(ValueError):
            conv_dlgn_shape(_cfg(head_wiring='perm'))
        with pytest.raises(ValueError):
            conv_dlgn_shape(_cfg(channels=(4, 8, 16, 32)))     # 空間が 0 になる


class TestIrRoundTrip:
    @pytest.mark.parametrize('sampler', ['dilated8', 'random8', 'cross5'])
    @pytest.mark.parametrize('native', [True, False])
    def test_export_simulate_matches_hard_eval(self, sampler, native):
        torch.manual_seed(7)
        model = build_conv_dlgn(_cfg(sampler=sampler),
                                generator=torch.Generator().manual_seed(7)).eval()
        ir = export_ir(model, input_shape=IN_SHAPE, native_conv=native)
        validate(ir)
        x = _images()
        flat = x.reshape(x.shape[0], -1).numpy().astype(np.int64)
        assert np.array_equal(simulate(ir, flat), _hard(model, x))
        if native:
            assert [n.type for n in ir.nodes][:4] == ['conv2d', 'or_pool2d', 'conv2d', 'or_pool2d']
            assert ir.nodes[-1].type == 'group_sum'
            assert ir.nodes[-2].type == 'permute'               # stride 割当の転置


class TestRecipe:
    @staticmethod
    def _data():
        return ResolvedData(name='synthetic', kind='classification',
                            in_width=int(np.prod(IN_SHAPE)), out_width=4,
                            train_loader=None, eval_loader=None)

    def _build(self, *overrides):
        cfg = load_config(None, [
            'model.kind=conv_dlgn', 'model.n_inputs=2', 'model.parameterization=independent',
            'model.estimator=sin', 'model.residual_init=true', 'model.tau=2.0',
            'model.group_assign=stride',
            'model.conv={in_shape: [2, 8, 8], channels: [4, 8], sampler: dilated8, '
            'head_dims: [32, 16]}',
            *overrides,
        ])
        return cfg, _build_model(cfg.model, self._data(), 'cpu', seed=0)

    def test_runner_builds_same_structure(self):
        cfg, model = self._build()
        assert cfg.model.conv.sampler == 'dilated8'
        kinds = [type(m).__name__ for m in model]
        assert kinds == ['LutConv2d', 'OrPooling2d', 'LutConv2d', 'OrPooling2d', 'Flatten',
                         'LutLayer', 'LutLayer', 'GroupSum']
        model.eval()
        ir = export_ir(model, input_shape=IN_SHAPE, native_conv=True)
        x = _images(seed=1)
        assert np.array_equal(simulate(ir, x.reshape(32, -1).numpy().astype(np.int64)),
                              _hard(model, x))

    def test_seed_reproducible_and_differs(self):
        _, a = self._build('model.conv.sampler=random8')
        _, b = self._build('model.conv.sampler=random8')
        assert torch.equal(a[0].indices_flat, b[0].indices_flat)
        assert torch.equal(a[0].weights, b[0].weights)

    def test_validation(self):
        with pytest.raises(ValueError):
            load_config(None, ['model.kind=conv_dlgn'])                           # conv 無し
        with pytest.raises(ValueError):
            load_config(None, ['model.kind=dlgn', 'model.conv={in_shape: [1, 8, 8]}'])
        with pytest.raises(ValueError):
            load_config(None, ['model.kind=conv_dlgn', 'model.n_inputs=4',
                               'model.conv={in_shape: [1, 8, 8]}'])
        with pytest.raises(ValueError):
            load_config(None, ['model.kind=conv_dlgn', 'model.parameterization=warp',
                               'model.conv={in_shape: [1, 8, 8]}'])
        with pytest.raises(ValueError):                                           # 入力幅不一致
            self._build('model.conv.in_shape=[1, 8, 8]')
