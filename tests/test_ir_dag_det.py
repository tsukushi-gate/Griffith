"""CenterNetLUT(多ヘッド dict 出力)の IR 抽出・等価性テスト。

export/dag.export_dag_ir が dict 出力(hm/size/off の 3 ヘッド)を持つ
検出モデルを、3 ヘッドを concat で束ねた単一出力 IR へ写せること、および

    PyTorch hard eval ⇔ ir/simulate.py

の整数完全一致(3 段等価性の中核 2 段)を固定する。SpatialGroupSum の
scale は argmax/デコードに効くが simulate は無視する(生 popcount を返す)ため、
torch 側の 3 ヘッドの scale をすべて 1.0 にして生の整数スコアで厳密照合する。
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from detection_model import CenterNetLUT

from griffith_logic.export import export_dag_ir
from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir import (
    ir_from_dict,
    ir_to_dict,
    simulate,
    validate,
)

HEAD_ORDER = ('hm', 'size', 'off')


def _tiny_det(n_classes=3, k=2, in_channels=3, canvas=16, seed=0,
              hm_group=2, size_group=2, offset_group=2, wiring='fixed',
              **extra_kw):
    torch.manual_seed(seed)
    m = CenterNetLUT(
        n_classes=n_classes, k=k, in_channels=in_channels,
        tree_depth=2, head_depth=2, n_inputs=2,
        hm_group=hm_group, size_group=size_group, offset_group=offset_group,
        input_channels_per_tree=2, parameterization='softmax16g',
        n_layers=3, wiring=wiring, **extra_kw,
    ).eval()
    # 生の整数スコア(グループ popcount)で simulate と厳密照合するため scale=1
    m.hm_sum.scale = 1.0
    m.size_sum.scale = 1.0
    m.off_sum.scale = 1.0
    return m


def _binary_images(batch, c, h, w, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, c, h, w), generator=g).float()


def _torch_heads_concat(model, x):
    """hard eval の 3 ヘッドを concat 規約(hm|size|off, NCHW flatten)で束ねる。"""
    with torch.no_grad():
        out = model(x)
    flats = [out[h].reshape(out[h].shape[0], -1).numpy() for h in HEAD_ORDER]
    return np.concatenate(flats, axis=1).astype(np.int64)


class TestDetDagExtract:
    def test_multi_head_concat_structure(self):
        """dict 出力は 3 group_sum を単一 concat へ束ね、validate を通る"""
        m = _tiny_det()
        ir = export_dag_ir(m, (3, 16, 16), name='det')
        assert validate(ir) == []
        assert ir.outputs[0].semantics == 'multi_head'
        # 出力へ流れるのは concat 一つ、その入力スロットは hm/size/off の 3 本
        cat = next(n for n in ir.nodes if n.type == 'concat')
        slots = sorted(e.dst_slot for e in ir.edges if e.dst == cat.id)
        assert slots == [0, 1, 2]
        # ヘッド分割メタが記録され、幅の合計 = 出力幅
        heads = ir.metadata['output_heads']
        assert [h['name'] for h in heads] == list(HEAD_ORDER)
        assert sum(h['width'] for h in heads) == ir.outputs[0].width

    def test_spatial_group_sum_is_native_permute(self):
        """3ヘッドのSpatialGroupSumはpermuteネイティブノード化され
        lut_layer全展開(1位置=1エントリ)は生成されない"""
        m = _tiny_det()
        ir = export_dag_ir(m, (3, 16, 16), name='det')
        types = [n.type for n in ir.nodes]
        assert types.count('permute') == 3     # hm_sum/size_sum/off_sum
        assert types.count('lut_layer') == 0
        assert types.count('group_sum') == 3

    def test_head_shapes(self):
        """各ヘッドの形状 (n_out, H/4, W/4) がメタに残る(hm=3, size/off=2ch)"""
        m = _tiny_det(n_classes=3, canvas=16)
        ir = export_dag_ir(m, (3, 16, 16))
        heads = {h['name']: h for h in ir.metadata['output_heads']}
        assert heads['hm']['shape'] == [3, 4, 4]
        assert heads['size']['shape'] == [2, 4, 4]
        assert heads['off']['shape'] == [2, 4, 4]


class TestDetDagEquivalence:
    @pytest.mark.parametrize('nc,k,canvas', [(3, 2, 16), (2, 2, 24), (4, 3, 16)])
    def test_simulate_matches_hard_eval(self, nc, k, canvas):
        """(核心)PyTorch hard eval ⇔ simulate の整数完全一致(3 ヘッド連結)"""
        m = _tiny_det(n_classes=nc, k=k, canvas=canvas)
        x = _binary_images(4, 3, canvas, canvas)
        ir = export_dag_ir(m, (3, canvas, canvas))
        y = _torch_heads_concat(m, x)
        scores = simulate(ir, x.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y, scores), (
            f'max diff = {np.abs(y - scores).max()}')

    @pytest.mark.parametrize('nc,k,canvas', [(3, 2, 16), (2, 2, 24), (4, 3, 16)])
    def test_simulate_matches_hard_eval_learnable_wiring(self, nc, k, canvas):
        """(回帰)wiring='learnable'でも PyTorch hard eval ⇔ simulate 一致。

        抽出器が固定 indices_flat のまま学習配線の argmax 選択を無視すると、
        IRが別入力を集めて検出が壊れる(実測: det デモで検出0件、カラー検出で
        ボックス散乱)。この回帰を固定する。
        """
        m = _tiny_det(n_classes=nc, k=k, canvas=canvas, wiring='learnable')
        x = _binary_images(4, 3, canvas, canvas)
        ir = export_dag_ir(m, (3, canvas, canvas))
        y = _torch_heads_concat(m, x)
        scores = simulate(ir, x.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y, scores), (
            f'学習配線 export 不一致: max diff = {np.abs(y - scores).max()}')

    def test_dict_roundtrip_and_lowering_stable(self):
        """dict 往復・lut_layer 展開後も simulate 結果が不変"""
        m = _tiny_det()
        x = _binary_images(3, 3, 16, 16).flatten(1).numpy().astype(np.int64)
        ir = export_dag_ir(m, (3, 16, 16))
        ir2 = ir_from_dict(ir_to_dict(ir))
        assert validate(ir2) == []
        assert np.array_equal(simulate(ir, x), simulate(ir2, x))
        low = lower_to_primitives(ir)
        assert all(
            n.type not in ('conv2d', 'or_pool2d', 'upsample2d', 'permute')
            for n in low.nodes
        )
        assert np.array_equal(simulate(ir, x), simulate(low, x))

    def test_native_ir_much_smaller_than_lowered(self):
        """ネイティブノードのJSONは全展開版より2桁以上小さい(肥大修正の固定)"""
        m = _tiny_det()
        ir = export_dag_ir(m, (3, 16, 16))
        low = lower_to_primitives(ir)
        native_bytes = len(json.dumps(ir_to_dict(ir), separators=(',', ':')))
        lowered_bytes = len(json.dumps(ir_to_dict(low), separators=(',', ':')))
        assert native_bytes * 10 < lowered_bytes, (
            f'native={native_bytes}B lowered={lowered_bytes}B'
        )


# ============================================================
# アーキ改造とIR抽出の互換性
# ============================================================
#
# export_dag_ir は torch.fx トレースの汎用DAG抽出器で、leaf module に
# 抽出器が登録された部品(LutConv2d/OrPooling2d/UpSample2d/SpatialGroupSum)と
# torch.cat(dim=1) のみで forward が構成されていればアーキ非依存にIR化できる。
# downsample='strided'(strideのLutConv2d)、neck='fpn'/'fpnpan'
# (UpSample2d+concat+LutConv2d)、decoupled_head(追加のLutConv2d分岐)は
# いずれもこの制約内に収まるため無改造でエクスポート可能なことを固定する。
# reg_bins>0 は forward内で F.softmax/view/sum を直接呼ぶため(登録extractorが
# 無い call_function/call_method)現状エクスポート非対応(別途extractor追加が
# 必要、完了報告に明記)。head_type='float' は
# scripts/export_webbench.py 側で明示的にエラーにする(このファイルの対象外)。

class TestDetDagW14ArchCompat:
    @pytest.mark.parametrize('extra_kw', [
        {'downsample': 'strided'},
        {'neck': 'fpn'},
        {'neck': 'fpnpan'},
        {'decoupled_head': True},
        {'neck': 'fpn', 'decoupled_head': True},
    ], ids=['downsample_strided', 'neck_fpn', 'neck_fpnpan',
            'decoupled_head', 'neck_fpn+decoupled_head'])
    def test_simulate_matches_hard_eval(self, extra_kw):
        """新アーキがtorch.fx抽出可能・PyTorch hard eval⇔simulate一致か"""
        m = _tiny_det(n_classes=3, k=2, canvas=16, **extra_kw)
        x = _binary_images(4, 3, 16, 16)
        ir = export_dag_ir(m, (3, 16, 16))
        y = _torch_heads_concat(m, x)
        scores = simulate(ir, x.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y, scores), (
            f'{extra_kw}: max diff = {np.abs(y - scores).max()}')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
