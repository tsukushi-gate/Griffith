"""tools/nas/mutate.py のテスト。

property-based(hypothesis)+固定fixtureで、各変異オペレータが
    (a) 変異後IRが validate() を通過する
    (b) 元IRを非破壊(適用前後でdict等価)
    (c) 同一seedで決定的
    (d) 幅整合(build_torch できる=形状が壊れていない)
であることを確認する。
"""

import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import build_torch, ir_from_dict, ir_to_dict, validate
from tools.nas.mutate import (
    MUTATION_OPS,
    mutate,
    mutate_conv_channels,
    mutate_insert_layer,
    mutate_lut_width,
    mutate_n_inputs,
    mutate_remove_layer,
)

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'ir_v01'


def _load_fixture(name: str):
    doc = json.loads((FIXTURE_DIR / name).read_text())
    return ir_from_dict(doc['ir'])


def _conv_ir():
    """conv2d→flatten→lut_layer→group_sum の小規模conv IR(poolなし)。

    conv_channels変異の後続=flattenがrewire対応型なので、この構成で
    チャネル増減の1ホップ整合をテストできる。
    """
    torch.manual_seed(7)
    c_in, h_in, w_in = 1, 6, 6
    conv = LutConv2d(c_in, 3, kernel_size=3, tree_depth=2, padding=0)
    h_out = w_in - 3 + 1
    model = nn.Sequential(
        conv,
        nn.Flatten(),
        LutLayer(3 * h_out * h_out, 8, n_inputs=2),
        GroupSum(k=2, tau=1.0),
    )
    model.eval()
    return export_ir(model, input_shape=(c_in, h_in, w_in), name='nas_conv_fixture',
                     native_conv=True)


@pytest.fixture(params=['xor.json', 'parity3.json'])
def small_ir(request):
    return _load_fixture(request.param)


@pytest.fixture
def conv_ir():
    return _conv_ir()


@pytest.fixture
def rewirable_ir():
    """後続lut_layerを持つ中間層があるIR(幅変異が必ず1候補は成立する)。

    xor.json は単一lut_layer→出力ポート直結なので、外部インターフェース
    (出力ポート幅)を変えない設計方針により幅変異の対象外
    (mutate_lut_widthが常にNoneを返す)。parity3.json は
    n1_lutlayer→n2_lutlayer→出力ポート、という中間接続を持つため
    n1_lutlayerの幅変異が可能。
    """
    return _load_fixture('parity3.json')


def _assert_non_destructive(before_dict, ir_after_call):
    assert before_dict == ir_after_call, '入力IRが変異操作で書き換えられている(非破壊違反)'


class TestLutWidthMutation:
    def test_validates_and_non_destructive(self, rewirable_ir):
        before = ir_to_dict(rewirable_ir)
        rng = random.Random(0)
        result = mutate_lut_width(rewirable_ir, rng)
        assert result is not None, '幅変異が1件も成立しなかった'
        validate(result)  # 例外なし
        _assert_non_destructive(before, ir_to_dict(rewirable_ir))
        # 元IRとは異なる新オブジェクトであること
        assert result is not rewirable_ir

    def test_deterministic_with_same_seed(self, rewirable_ir):
        r1 = mutate_lut_width(copy.deepcopy(rewirable_ir), random.Random(42))
        r2 = mutate_lut_width(copy.deepcopy(rewirable_ir), random.Random(42))
        assert ir_to_dict(r1) == ir_to_dict(r2)

    def test_width_actually_changed_and_buildable(self, rewirable_ir):
        rng = random.Random(1)
        result = mutate_lut_width(rewirable_ir, rng)
        assert result is not None
        # 少なくとも1ノードのout_widthが元と異なる
        before_widths = {n.id: n.params.get('out_width') for n in rewirable_ir.nodes
                          if n.type == 'lut_layer'}
        after_widths = {n.id: n.params.get('out_width') for n in result.nodes
                        if n.type == 'lut_layer'}
        assert before_widths != after_widths or len(result.nodes) != len(rewirable_ir.nodes)
        model = build_torch(result)
        model.eval()
        x = torch.zeros(2, result.inputs[0].width)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, result.outputs[0].width)

    def test_numpy_generator_rng_supported(self, rewirable_ir):
        rng = np.random.default_rng(3)
        result = mutate_lut_width(rewirable_ir, rng)
        assert result is not None
        validate(result)

    def test_output_port_successor_rejected(self):
        """xor.jsonは唯一のlut_layerが出力ポート直結なので幅変異は常にNone。"""
        ir = _load_fixture('xor.json')
        assert mutate_lut_width(ir, random.Random(0)) is None


class TestInsertLayer:
    def test_validates_and_non_destructive(self, small_ir):
        before = ir_to_dict(small_ir)
        rng = random.Random(0)
        result = mutate_insert_layer(small_ir, rng)
        assert result is not None
        validate(result)
        _assert_non_destructive(before, ir_to_dict(small_ir))
        assert len(result.nodes) == len(small_ir.nodes) + 1

    def test_pass_through_preserves_function(self, small_ir):
        """挿入層はpass-through初期化なので、hard evalの入出力関係が不変。"""
        rng = random.Random(5)
        result = mutate_insert_layer(small_ir, rng)
        assert result is not None

        n = small_ir.inputs[0].width
        xs = torch.tensor(
            [[(i >> b) & 1 for b in range(n)] for i in range(2 ** n)],
            dtype=torch.float32,
        )
        m_before = build_torch(small_ir)
        m_after = build_torch(result)
        m_before.eval()
        m_after.eval()
        with torch.no_grad():
            y_before = (m_before(xs) > 0.5).float()
            y_after = (m_after(xs) > 0.5).float()
        assert torch.equal(y_before, y_after)

    def test_deterministic_with_same_seed(self, small_ir):
        r1 = mutate_insert_layer(copy.deepcopy(small_ir), random.Random(9))
        r2 = mutate_insert_layer(copy.deepcopy(small_ir), random.Random(9))
        assert ir_to_dict(r1) == ir_to_dict(r2)


class TestRemoveLayer:
    def test_validates_and_non_destructive_on_insert_then_remove(self, small_ir):
        """挿入直後のIRには必ずin_width==out_widthの除去可能ノードがある。"""
        inserted = mutate_insert_layer(small_ir, random.Random(0))
        assert inserted is not None
        before = ir_to_dict(inserted)
        result = mutate_remove_layer(inserted, random.Random(1))
        assert result is not None
        validate(result)
        _assert_non_destructive(before, ir_to_dict(inserted))
        assert len(result.nodes) == len(inserted.nodes) - 1

    def test_no_removable_node_returns_none(self):
        ir = _load_fixture('xor.json')  # n_inputs=2,in_width=2,out_width=1 → 除去不可
        assert mutate_remove_layer(ir, random.Random(0)) is None

    def test_insert_then_remove_roundtrip_preserves_function(self, small_ir):
        rng_i, rng_r = random.Random(11), random.Random(12)
        inserted = mutate_insert_layer(small_ir, rng_i)
        assert inserted is not None
        removed = mutate_remove_layer(inserted, rng_r)
        assert removed is not None

        n = small_ir.inputs[0].width
        xs = torch.tensor(
            [[(i >> b) & 1 for b in range(n)] for i in range(2 ** n)],
            dtype=torch.float32,
        )
        m_before = build_torch(small_ir)
        m_after = build_torch(removed)
        m_before.eval()
        m_after.eval()
        with torch.no_grad():
            y_before = (m_before(xs) > 0.5).float()
            y_after = (m_after(xs) > 0.5).float()
        assert torch.equal(y_before, y_after)


class TestNInputsMutation:
    def test_validates_and_non_destructive(self, small_ir):
        before = ir_to_dict(small_ir)
        rng = random.Random(0)
        result = mutate_n_inputs(small_ir, rng)
        assert result is not None
        validate(result)
        _assert_non_destructive(before, ir_to_dict(small_ir))

    def test_n_inputs_changed_widths_unchanged(self, small_ir):
        rng = random.Random(2)
        result = mutate_n_inputs(small_ir, rng)
        assert result is not None
        before_shapes = [(n.params['in_width'], n.params['out_width'])
                         for n in small_ir.nodes if n.type == 'lut_layer']
        after_shapes = [(n.params['in_width'], n.params['out_width'])
                        for n in result.nodes if n.type == 'lut_layer']
        assert before_shapes == after_shapes
        changed = any(
            b.params['n_inputs'] != a.params['n_inputs']
            for b, a in zip(
                (n for n in small_ir.nodes if n.type == 'lut_layer'),
                (n for n in result.nodes if n.type == 'lut_layer'),
            )
        )
        assert changed

    def test_deterministic_with_same_seed(self, small_ir):
        r1 = mutate_n_inputs(copy.deepcopy(small_ir), random.Random(4))
        r2 = mutate_n_inputs(copy.deepcopy(small_ir), random.Random(4))
        assert ir_to_dict(r1) == ir_to_dict(r2)


class TestConvChannelsMutation:
    def test_validates_and_non_destructive(self, conv_ir):
        before = ir_to_dict(conv_ir)
        rng = random.Random(0)
        result = mutate_conv_channels(conv_ir, rng)
        assert result is not None
        validate(result)
        _assert_non_destructive(before, ir_to_dict(conv_ir))

    def test_channels_changed_and_buildable(self, conv_ir):
        rng = random.Random(6)
        result = mutate_conv_channels(conv_ir, rng)
        assert result is not None
        conv_before = next(n for n in conv_ir.nodes if n.type == 'conv2d')
        conv_after = next(n for n in result.nodes if n.type == 'conv2d')
        assert conv_before.params['out_channels'] != conv_after.params['out_channels']

        model = build_torch(result)
        model.eval()
        x = torch.zeros(2, result.inputs[0].width)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, result.outputs[0].width)

    def test_no_conv_returns_none(self, small_ir):
        assert mutate_conv_channels(small_ir, random.Random(0)) is None

    def test_deterministic_with_same_seed(self, conv_ir):
        r1 = mutate_conv_channels(copy.deepcopy(conv_ir), random.Random(8))
        r2 = mutate_conv_channels(copy.deepcopy(conv_ir), random.Random(8))
        assert ir_to_dict(r1) == ir_to_dict(r2)


class TestUnifiedEntrypoint:
    def test_mutate_picks_some_op_and_validates(self, small_ir):
        rng = random.Random(0)
        result = mutate(small_ir, rng)
        assert result is not None
        validate(result)

    def test_mutate_deterministic(self, small_ir):
        r1 = mutate(copy.deepcopy(small_ir), random.Random(123))
        r2 = mutate(copy.deepcopy(small_ir), random.Random(123))
        assert ir_to_dict(r1) == ir_to_dict(r2)

    def test_mutate_conv_channels_excluded_without_conv(self, small_ir):
        """opsを明示的にconv_channelsだけに絞ると、conv非搭載IRではNone。"""
        assert mutate(small_ir, random.Random(0), ops=['conv_channels']) is None

    def test_mutate_restricted_ops_respected(self, small_ir):
        rng = random.Random(0)
        result = mutate(small_ir, rng, ops=['n_inputs'])
        assert result is not None
        # n_inputsオペレータの効果(in/out幅不変)であることを確認
        before_shapes = [(n.params['in_width'], n.params['out_width'])
                         for n in small_ir.nodes if n.type == 'lut_layer']
        after_shapes = [(n.params['in_width'], n.params['out_width'])
                        for n in result.nodes if n.type == 'lut_layer']
        assert before_shapes == after_shapes

    def test_all_ops_registered(self):
        assert set(MUTATION_OPS.keys()) == {
            'lut_width', 'insert_layer', 'remove_layer', 'n_inputs', 'conv_channels',
        }


# --------------------------------------------------------------------------
# hypothesis: 複数seedにわたる決定性+validate通過のproperty test
# --------------------------------------------------------------------------

@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=2 ** 31 - 1),
       fixture_name=st.sampled_from(['xor.json', 'parity3.json']))
def test_mutate_property_validates_and_deterministic(seed, fixture_name):
    ir = _load_fixture(fixture_name)
    before = ir_to_dict(ir)

    r1 = mutate(copy.deepcopy(ir), random.Random(seed))
    r2 = mutate(copy.deepcopy(ir), random.Random(seed))

    # 元IRは非破壊のまま
    assert ir_to_dict(ir) == before

    if r1 is None:
        assert r2 is None
        return
    assert ir_to_dict(r1) == ir_to_dict(r2)
    validate(r1)  # 例外なし
