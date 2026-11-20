"""pretrain_cifar_encoder.py のテスト(W13a-4)。

エンコーダ部のsubmodule命名がCenterNetLUT/SegConvDLGNの該当層と一致し、
load_partial_state_dictでロード率100%(エンコーダ部分に限る)になることを
検証する。ネットワーク依存(CIFAR-10ダウンロード)は避け、モデル定義と
forward/backwardのみ小さいkでスモークする。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from checkpoint_utils import load_partial_state_dict
from detection_model import CenterNetLUT
from pretrain_cifar_encoder import (
    CifarPretrainDet,
    CifarPretrainSeg,
    build_model,
)
from segmentation_models import SegConvDLGN


class TestCifarPretrainDetForward:
    def test_forward_shape(self):
        model = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        x = torch.rand(2, 9, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_backward_all_grads(self):
        model = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        x = torch.rand(2, 9, 32, 32)
        y = model(x)
        y.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name

    def test_num_gates_positive(self):
        model = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        assert model.num_gates > 0


class TestCifarPretrainSegForward:
    def test_forward_shape(self):
        model = CifarPretrainSeg(in_channels=9, k=4, tree_depth=2, head_group=2)
        x = torch.rand(2, 9, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_backward_all_grads(self):
        model = CifarPretrainSeg(in_channels=9, k=4, tree_depth=2, head_group=2)
        x = torch.rand(2, 9, 32, 32)
        y = model(x)
        y.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name


class TestEncoderKeyNamesMatchDet:
    """CifarPretrainDetのエンコーダ命名がCenterNetLUT(n_layers=3)と一致するか"""

    def test_stage_keys_present(self):
        model = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        keys = set(model.state_dict().keys())
        assert 'stage1.0.weights' in keys
        assert 'stage2.0.weights' in keys
        assert 'stage3_pre.0.weights' in keys

    def test_partial_load_ratio_100_percent_on_encoder(self):
        """load_partial_state_dictでdet側のstage1/2/3_preが100%ロードされる"""
        enc = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        det = CenterNetLUT(
            n_classes=10, k=4, in_channels=9, tree_depth=2, head_depth=2,
            hm_group=2, size_group=2, offset_group=2, n_layers=3,
        )
        result = load_partial_state_dict(det, enc.state_dict())
        det_keys = det.state_dict()
        encoder_keys = {
            k for k in det_keys
            if k.startswith(('stage1.', 'stage2.', 'stage3_pre.'))
        }
        loaded = set(result['loaded_keys'])
        assert encoder_keys, 'エンコーダキーが見つかりません'
        assert encoder_keys <= loaded, encoder_keys - loaded
        # 非エンコーダ(検出ヘッド)キーはckpt側に無いので正しくスキップされる
        head_keys = set(det_keys) - encoder_keys
        assert head_keys <= set(result['skipped_missing'])
        assert result['skipped_shape_mismatch'] == []

    def test_encoder_weights_actually_copied(self):
        enc = CifarPretrainDet(in_channels=9, k=4, tree_depth=2, head_group=2)
        enc.stage1[0].weights.data.fill_(7.0)
        det = CenterNetLUT(
            n_classes=10, k=4, in_channels=9, tree_depth=2, head_depth=2,
            hm_group=2, size_group=2, offset_group=2, n_layers=3,
        )
        load_partial_state_dict(det, enc.state_dict())
        assert torch.equal(det.stage1[0].weights, enc.stage1[0].weights)


class TestEncoderKeyNamesMatchSeg:
    """CifarPretrainSegのエンコーダ命名がSegConvDLGNと一致するか"""

    def test_c_keys_present(self):
        model = CifarPretrainSeg(in_channels=9, k=4, tree_depth=2, head_group=2)
        keys = set(model.state_dict().keys())
        assert 'c1.weights' in keys
        assert 'c2.weights' in keys
        assert 'c3.weights' in keys

    def test_partial_load_ratio_100_percent_on_encoder(self):
        enc = CifarPretrainSeg(in_channels=9, k=4, tree_depth=2, head_group=2)
        seg = SegConvDLGN(in_channels=9, n_classes=2, k=4, tree_depth=2,
                          readout_group=2)
        result = load_partial_state_dict(seg, enc.state_dict())
        seg_keys = seg.state_dict()
        encoder_keys = {
            k for k in seg_keys if k.startswith(('c1.', 'c2.', 'c3.'))
        }
        loaded = set(result['loaded_keys'])
        assert encoder_keys, 'エンコーダキーが見つかりません'
        assert encoder_keys <= loaded, encoder_keys - loaded
        # d1/d2(デコーダ)はckpt側に無いので正しくスキップされる
        decoder_keys = set(seg_keys) - encoder_keys
        assert decoder_keys <= set(result['skipped_missing'])
        assert result['skipped_shape_mismatch'] == []

    def test_encoder_weights_actually_copied(self):
        enc = CifarPretrainSeg(in_channels=9, k=4, tree_depth=2, head_group=2)
        enc.c1.weights.data.fill_(7.0)
        seg = SegConvDLGN(in_channels=9, n_classes=2, k=4, tree_depth=2,
                          readout_group=2)
        load_partial_state_dict(seg, enc.state_dict())
        assert torch.equal(seg.c1.weights, enc.c1.weights)


class TestBuildModel:
    def _args(self, **overrides):
        import argparse
        ns = argparse.Namespace(
            target='det', k=4, tree_depth=2, head_group=2,
            input_channels_per_tree=None, wiring='fixed',
            device='cpu', grad_factor=1.0,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_det_default_input_channels_per_tree(self):
        model = build_model(self._args(target='det'), in_channels=9)
        assert isinstance(model, CifarPretrainDet)
        assert model.stage1[0].input_channels_per_tree == 2

    def test_seg_default_input_channels_per_tree(self):
        model = build_model(self._args(target='seg'), in_channels=9)
        assert isinstance(model, CifarPretrainSeg)
        assert model.c1.input_channels_per_tree == 4

    def test_explicit_input_channels_per_tree_overrides_default(self):
        model = build_model(
            self._args(target='det', input_channels_per_tree=3), in_channels=9)
        assert model.stage1[0].input_channels_per_tree == 3

    def test_wiring_learnable(self):
        model = build_model(self._args(target='seg', wiring='learnable'),
                            in_channels=9)
        assert model.c1.wiring == 'learnable'
        x = torch.rand(1, 9, 32, 32)
        y = model(x)
        y.sum().backward()
        assert model.c1.mapping_weights.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
