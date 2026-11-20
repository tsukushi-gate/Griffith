"""SegConvDLGN(U-Net 風 DAG)の IR 抽出・等価性テスト。

export/dag.export_dag_ir が torch.cat スキップ接続を持つ Conv-DLGN を
IR v0.2 の DAG へ写せること、および

    PyTorch hard eval ⇔ ir/simulate.py

の整数完全一致(3段等価性の中核 2 段)を固定する。SpatialGroupSum の
scale は argmax 不変で simulate は無視するため、torch 側の scale を 1.0 に
して生の整数スコア(グループ popcount)で厳密照合する。
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from segmentation_models import SegConvDLGN

from griffith_logic.export import export_dag_ir
from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir import (
    ir_from_dict,
    ir_to_dict,
    simulate,
    validate,
)


def _tiny_seg(k=4, in_channels=6, seed=0, wiring='fixed'):
    torch.manual_seed(seed)
    m = SegConvDLGN(
        in_channels=in_channels, k=k, tree_depth=2,
        readout_group=k, input_channels_per_tree=4,
        wiring=wiring,
    ).eval()
    m.readout.scale = 1.0  # 生の整数スコアで厳密照合するため
    return m


def _binary_images(batch, c, h, w, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, c, h, w), generator=g).float()


class TestSegDagExtract:
    def test_node_types_and_validate(self):
        """DAG 抽出は conv2d/or_pool2d/upsample2d/permute/concat/group_sum を生成し
        validate を通る(UpSample2d/SpatialGroupSumはネイティブノード化済み)"""
        ir = export_dag_ir(_tiny_seg(), (6, 16, 16), name='seg')
        assert validate(ir) == []
        types = [n.type for n in ir.nodes]
        # 3 段エンコーダ(conv+pool 2 回 + conv)、2 本の concat、readout group_sum
        assert types.count('conv2d') == 5      # c1,c2,c3,d2,d1
        assert types.count('or_pool2d') == 2   # p1,p2
        assert types.count('upsample2d') == 2  # u1,u2(旧: lut_layer全展開)
        assert types.count('permute') == 1     # readoutのチャネル並べ替え(旧: lut_layer全展開)
        assert types.count('lut_layer') == 0   # 恒等配線ノードはネイティブ化され0
        assert types.count('concat') == 2      # cat(u2,c2), cat(u1,c1)
        assert types.count('group_sum') == 1   # readout
        assert ir.inputs[0].width == 6 * 16 * 16

    def test_concat_slot_order(self):
        """concat のエッジは torch.cat の引数順(slot0=up枝, slot1=skip枝)"""
        ir = export_dag_ir(_tiny_seg(), (6, 16, 16))
        cat = next(n for n in ir.nodes if n.type == 'concat')
        slots = sorted(
            e.dst_slot for e in ir.edges if e.dst == cat.id)
        assert slots == [0, 1]


class TestSegDagEquivalence:
    @pytest.mark.parametrize('k,c,h', [(4, 6, 16), (2, 9, 16), (4, 3, 32)])
    def test_simulate_matches_hard_eval(self, k, c, h):
        """(核心)PyTorch hard eval ⇔ simulate の整数完全一致(スコア+マスク)"""
        m = _tiny_seg(k=k, in_channels=c)
        x4 = _binary_images(5, c, h, h)
        ir = export_dag_ir(m, (c, h, h))
        with torch.no_grad():
            y = m(x4).numpy()  # (B, 2, H, W) 生カウント(scale=1)
        y_flat = y.reshape(y.shape[0], -1).astype(np.int64)  # (o,y,x) flatten
        scores = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y_flat, scores), (
            f'max diff = {np.abs(y_flat - scores).max()}')
        # per-pixel argmax マスクも一致
        mask_torch = y.argmax(axis=1)
        mask_sim = scores.reshape(scores.shape[0], 2, h * h) \
            .argmax(axis=1).reshape(-1, h, h)
        assert np.array_equal(mask_torch, mask_sim)

    @pytest.mark.parametrize('k,c,h', [(4, 6, 16), (2, 9, 16), (4, 3, 32)])
    def test_simulate_matches_hard_eval_learnable_wiring(self, k, c, h):
        """(回帰)wiring='learnable'でも PyTorch hard eval ⇔ simulate 一致。

        export 抽出器が固定 indices_flat のまま学習配線 argmax 選択を無視すると、
        IR が別の入力を集めてマスクが壊れる(実測: 学習配線 seg で画素一致 54%→
        修正後 100%)。この回帰を固定する。
        """
        m = _tiny_seg(k=k, in_channels=c, wiring='learnable')
        x4 = _binary_images(5, c, h, h)
        ir = export_dag_ir(m, (c, h, h))
        with torch.no_grad():
            y = m(x4).numpy()
        y_flat = y.reshape(y.shape[0], -1).astype(np.int64)
        scores = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y_flat, scores), (
            f'学習配線 export 不一致: max diff = {np.abs(y_flat - scores).max()}')
        mask_torch = y.argmax(axis=1)
        mask_sim = scores.reshape(scores.shape[0], 2, h * h) \
            .argmax(axis=1).reshape(-1, h, h)
        assert np.array_equal(mask_torch, mask_sim)

    def test_dict_roundtrip_simulate_stable(self):
        """ir_to_dict → ir_from_dict 往復後も simulate 結果が不変"""
        m = _tiny_seg()
        x = _binary_images(4, 6, 16, 16).flatten(1).numpy().astype(np.int64)
        ir = export_dag_ir(m, (6, 16, 16))
        ir2 = ir_from_dict(ir_to_dict(ir))
        assert validate(ir2) == []
        assert np.array_equal(simulate(ir, x), simulate(ir2, x))

    def test_lower_to_primitives_preserves_scores(self):
        """conv2d/or_pool2d/upsample2d/permute を lut_layer 展開してもスコアが不変"""
        m = _tiny_seg()
        x = _binary_images(4, 6, 16, 16).flatten(1).numpy().astype(np.int64)
        ir = export_dag_ir(m, (6, 16, 16))
        low = lower_to_primitives(ir)
        assert all(
            n.type not in ('conv2d', 'or_pool2d', 'upsample2d', 'permute')
            for n in low.nodes
        )
        assert np.array_equal(simulate(ir, x), simulate(low, x))

    def test_native_ir_much_smaller_than_lowered(self):
        """ネイティブノードのJSONは全展開版より2桁以上小さい(肥大修正の固定)"""
        from griffith_logic.ir.serialize import ir_to_dict

        m = _tiny_seg()
        ir = export_dag_ir(m, (6, 16, 16))
        low = lower_to_primitives(ir)
        native_bytes = len(json.dumps(ir_to_dict(ir), separators=(',', ':')))
        lowered_bytes = len(json.dumps(ir_to_dict(low), separators=(',', ':')))
        assert native_bytes * 10 < lowered_bytes, (
            f'native={native_bytes}B lowered={lowered_bytes}B'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
