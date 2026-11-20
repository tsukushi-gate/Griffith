"""transcribe_unet_to_segdlgn.py のテスト(W13a-5)。

SmallUNet→SegConvDLGN構造転写PoC(c1のみ、既存パターン踏襲)。
ネットワーク依存(Oxford Petダウンロード)を避け、小さい合成SmallUNet+
ランダム較正データでcore関数(transcribe_c1_from_unet/build_*)をスモークする。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from segmentation_models import SegConvDLGN, SmallUNet
from transcribe_unet_to_segdlgn import (
    build_transcribed_segdlgn,
    build_wiring_only_segdlgn,
    remap_rgb_indices_to_thermometer,
    transcribe_c1_from_unet,
)


class TestRemapRgbIndicesToThermometer:
    def test_known_mapping(self):
        """3ch index -> thermometer(中央閾値)index の既知値照合"""
        import numpy as np
        # kernel 2x2、in_ch=3: index = c*4 + h*2 + w
        # c=0,pos(0,0)->idx0; c=1,pos(1,1)->idx7; c=2,pos(0,1)->idx9
        idx = np.array([[0, 7, 9]])
        remapped = remap_rgb_indices_to_thermometer(idx, 2, 2, n_thresholds=3)
        # mid = 3//2 = 1 → c_new = 1*3+c
        # c=0 -> c_new=3 -> 3*4+0=12
        # c=1 -> c_new=4 -> 4*4+3=19
        # c=2 -> c_new=5 -> 5*4+1=21
        assert remapped.tolist() == [[12, 19, 21]]

    def test_within_thermometer_receptive_bound(self):
        import numpy as np
        idx = np.array([[0, 1, 2, 3, 8, 11]])  # 3ch, kernel 2x2
        remapped = remap_rgb_indices_to_thermometer(idx, 2, 2, n_thresholds=3)
        receptive = 9 * 2 * 2  # 3*n_thresholds channels
        assert remapped.min() >= 0
        assert remapped.max() < receptive


class TestTranscribeC1FromUnet:
    def _unet(self, base=8, seed=0):
        torch.manual_seed(seed)
        unet = SmallUNet(in_channels=3, n_classes=2, base=base)
        unet.eval()
        return unet

    def test_k_matches_teacher_base(self):
        unet = self._unet(base=8)
        calib = torch.rand(32, 3, 16, 16)
        transcribed, info = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=2, seed=1,
        )
        assert transcribed['k'] == 8

    def test_agreement_in_valid_range(self):
        unet = self._unet(base=8)
        calib = torch.rand(32, 3, 16, 16)
        _, info = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=2, seed=1,
        )
        assert 0.0 <= info['approx_agreement'] <= 1.0
        assert len(info['per_channel_agreement']) == 8

    def test_indices_within_thermometer_bound(self):
        unet = self._unet(base=8)
        calib = torch.rand(32, 3, 16, 16)
        transcribed, _ = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=2, seed=1,
        )
        idx = transcribed['indices_flat']
        receptive = 9 * 3 * 3  # in_channels(9) * kernel(3x3, SmallUNet既定)
        assert idx.min() >= 0
        assert idx.max() < receptive

    def test_truth_tables_count_matches_gates(self):
        unet = self._unet(base=8)
        calib = torch.rand(32, 3, 16, 16)
        transcribed, _ = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=2, seed=1,
        )
        n_leaves = 2 ** 2  # n_inputs=2, tree_depth=2
        n_internal = (n_leaves - 1) // 1
        assert len(transcribed['truth_tables']) == transcribed['k'] * n_internal


class TestBuildTranscribedSegDLGN:
    def _transcribed(self, base=8, tree_depth=2):
        torch.manual_seed(0)
        unet = SmallUNet(in_channels=3, n_classes=2, base=base)
        unet.eval()
        calib = torch.rand(32, 3, 16, 16)
        transcribed, info = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=tree_depth, seed=1,
        )
        return transcribed, info

    def test_structural_model_is_segdlgn_with_matching_k(self):
        transcribed, _ = self._transcribed()
        model = build_transcribed_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4,
        )
        assert isinstance(model, SegConvDLGN)
        assert model.k == transcribed['k']
        assert model.c1.wiring == 'fixed'

    def test_structural_model_forward_backward(self):
        transcribed, _ = self._transcribed()
        model = build_transcribed_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4,
        )
        x = torch.randint(0, 2, (2, 9, 16, 16)).float()
        y = model(x)
        assert y.shape == (2, 2, 16, 16)
        y.sum().backward()

    def test_c1_indices_match_transcribed(self):
        transcribed, _ = self._transcribed()
        model = build_transcribed_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4,
        )
        import numpy as np
        expected = torch.as_tensor(
            np.asarray(transcribed['indices_flat'], dtype=np.int64))
        assert torch.equal(model.c1.indices_flat, expected)


class TestBuildWiringOnlySegDLGN:
    def _transcribed(self):
        torch.manual_seed(0)
        unet = SmallUNet(in_channels=3, n_classes=2, base=8)
        unet.eval()
        calib = torch.rand(32, 3, 16, 16)
        transcribed, _ = transcribe_c1_from_unet(
            unet, calib, n_thresholds=3, tree_depth=2, seed=1,
        )
        return transcribed

    def test_shares_wiring_with_structural_variant(self):
        transcribed = self._transcribed()
        structural = build_transcribed_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4, seed=5,
        )
        wiring_only = build_wiring_only_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4, seed=7,
        )
        assert torch.equal(structural.c1.indices_flat, wiring_only.c1.indices_flat)
        # 真理値表(weights)は転写結果を使わない(既定初期化のまま)ため異なる
        assert not torch.equal(structural.c1.weights, wiring_only.c1.weights)

    def test_forward_backward(self):
        transcribed = self._transcribed()
        model = build_wiring_only_segdlgn(
            transcribed, n_thresholds=3, tree_depth=2, readout_group=2,
            input_channels_per_tree=4,
        )
        x = torch.randint(0, 2, (2, 9, 16, 16)).float()
        y = model(x)
        y.sum().backward()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
