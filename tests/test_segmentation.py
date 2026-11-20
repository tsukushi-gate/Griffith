"""W1 セグメンテーション基盤のテスト(高速スイート)。

ネットワーク依存を避けるため、実データ(Oxford Pet)のダウンロードは行わず、
合成trimap/ランダム入力でデータ変換契約とモデルforward・1step学習を検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from griffith_logic.conv import LutConv2d
from griffith_logic.datasets.segmentation import (
    IGNORE_INDEX,
    ThermometerEncode3ch,
    trimap_to_binary_mask,
)

# ============================================================
# trimap → バイナリマスク
# ============================================================

class TestTrimapToBinaryMask:
    def _trimap(self):
        # 1=前景, 2=背景, 3=境界 を含む 2x3
        return torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.long)

    def test_border_foreground(self):
        mask = trimap_to_binary_mask(self._trimap(), border='foreground')
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_border_background(self):
        mask = trimap_to_binary_mask(self._trimap(), border='background')
        expected = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_border_ignore(self):
        mask = trimap_to_binary_mask(self._trimap(), border='ignore')
        expected = torch.tensor(
            [[1, 0, IGNORE_INDEX], [0, 1, IGNORE_INDEX]], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_channel_dim_squeezed(self):
        """(1,H,W)入力でも(H,W)を返す"""
        t = self._trimap().unsqueeze(0)
        mask = trimap_to_binary_mask(t)
        assert mask.shape == (2, 3)

    def test_invalid_border(self):
        with pytest.raises(AssertionError):
            trimap_to_binary_mask(self._trimap(), border='bogus')


# ============================================================
# thermometerエンコーディング
# ============================================================

class TestThermometerEncode3ch:
    def test_shape_and_binary(self):
        enc = ThermometerEncode3ch(3)
        x = torch.rand(3, 8, 8)
        y = enc(x)
        assert y.shape == (9, 8, 8)
        assert set(y.unique().tolist()) <= {0.0, 1.0}

    def test_threshold_monotonic(self):
        """閾値が上がるほどbitは非増加(bit_i >= bit_{i+1})"""
        enc = ThermometerEncode3ch(4)
        x = torch.rand(3, 8, 8)
        y = enc(x).reshape(4, 3, 8, 8)
        for i in range(3):
            assert (y[i] >= y[i + 1]).all()

    def test_threshold_values(self):
        """閾値式 (i+1)/(n+1) の確認(vision.ThresholdTransformと同一)"""
        enc = ThermometerEncode3ch(3)
        x = torch.full((3, 1, 1), 0.3)
        y = enc(x)  # 閾値 0.25/0.5/0.75 → [1,0,0] 各ch
        assert y[:3].sum() == 3  # 閾値0.25は全chで1
        assert y[3:].sum() == 0


# ============================================================
# SegConvDLGN
# ============================================================

class TestSegConvDLGN:
    @pytest.fixture()
    def model(self):
        from segmentation_models import SegConvDLGN
        torch.manual_seed(0)
        return SegConvDLGN(in_channels=9, k=4, readout_group=4)

    def test_forward_shape_train(self, model):
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        model.train()
        y = model(x)
        assert y.shape == (2, 2, 32, 32)

    def test_forward_shape_eval(self, model):
        """eval mode(離散LUT)でも同形状で動く"""
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 2, 32, 32)
        assert torch.isfinite(y).all()

    def test_all_convs_softmax16g(self, model):
        """全LutConv2dがsoftmax16g(旧softmaxは構造欠陥、スキル§4)"""
        convs = [m for m in model.modules() if isinstance(m, LutConv2d)]
        assert len(convs) == 5
        assert all(m.parameterization == 'softmax16g' for m in convs)

    def test_num_gates_positive(self, model):
        assert model.num_gates > 0
        assert model.num_gates == sum(
            m.num_gates for m in model.modules() if isinstance(m, LutConv2d))

    def test_one_train_step_updates_weights(self, model):
        """CE損失1stepで重みが更新される(ignore_index含む)"""
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        y = torch.randint(0, 2, (2, 32, 32))
        y[0, 0, 0] = 255  # ignore画素
        opt = torch.optim.AdamW(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
        before = model.c1.weights.detach().clone()
        model.train()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
        assert not torch.equal(before, model.c1.weights.detach())


class TestSegConvDLGNSelection:
    """selection={soft,hard_st,gumbel_st}本線統合(W13a-1)"""

    def _build(self, selection, **kw):
        from segmentation_models import SegConvDLGN
        torch.manual_seed(0)
        return SegConvDLGN(in_channels=9, k=4, readout_group=4,
                           selection=selection, **kw)

    def test_soft_default_matches_plain_lutconv2d(self):
        from griffith_logic.conv_hard_st import LutConv2dHardST
        model = self._build('soft')
        for m in model.modules():
            assert not isinstance(m, LutConv2dHardST)

    @pytest.mark.parametrize('selection', ['hard_st', 'gumbel_st'])
    def test_hard_st_uses_hardst_layers_softmax16g(self, selection):
        from griffith_logic.conv_hard_st import LutConv2dHardST
        model = self._build(selection)
        hardst_layers = [m for m in model.modules() if isinstance(m, LutConv2dHardST)]
        assert len(hardst_layers) == 5
        for layer in hardst_layers:
            assert layer.selection == selection
            # softmax16g構造欠陥回避(旧softmaxではなくsoftmax16gを使う、W13a-1)
            assert layer.parameterization == 'softmax16g'
        assert model.num_gates > 0

    @pytest.mark.parametrize('selection', ['soft', 'hard_st', 'gumbel_st'])
    def test_forward_backward_smoke(self, selection):
        model = self._build(selection)
        x = torch.randint(0, 2, (2, 9, 16, 16)).float()
        y = model(x)
        y.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    @pytest.mark.parametrize('selection', ['hard_st', 'gumbel_st'])
    def test_train_eval_gap_zero_on_binary_input(self, selection):
        model = self._build(selection)
        for m in model.modules():
            if hasattr(m, 'weights'):
                m.weights.data.normal_()
            if hasattr(m, 'gumbel_scale'):
                m.gumbel_scale = 0.0
        x = torch.randint(0, 2, (2, 9, 16, 16)).float()
        model.train()
        with torch.no_grad():
            y_train = model(x)
        model.eval()
        with torch.no_grad():
            y_eval = model(x)
        assert torch.equal(y_train, y_eval)

    def test_hard_st_wiring_learnable_coexist(self):
        model = self._build('hard_st', wiring='learnable')
        x = torch.randint(0, 2, (2, 9, 16, 16)).float()
        y = model(x)
        y.sum().backward()
        assert model.c1.mapping_weights.grad is not None

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError, match='selection'):
            self._build('bogus')

    def test_build_model_forwards_selection(self):
        """build_model(arch='convdlgn', selection=...)がSegConvDLGNへ配管されるか"""
        from segmentation_models import build_model

        from griffith_logic.conv_hard_st import LutConv2dHardST
        model = build_model('convdlgn', in_channels=9, n_classes=2,
                            k=4, readout_group=4, selection='hard_st')
        assert any(isinstance(m, LutConv2dHardST) for m in model.modules())


# ============================================================
# SmallUNet
# ============================================================

class TestSmallUNet:
    def test_forward_shape(self):
        from segmentation_models import SmallUNet
        model = SmallUNet(in_channels=3, base=8)
        x = torch.rand(2, 3, 64, 64)
        y = model(x)
        assert y.shape == (2, 2, 64, 64)


# ============================================================
# TeacherSeg(高解像度推論ラッパ、pretrained=Falseで重みDL回避)
# ============================================================

class TestTeacherSeg:
    def test_highres_wrapper_output_on_student_grid(self):
        """input_size=128で内部推論→出力は入力(student)格子64へdownされる。"""
        from segmentation_models import TeacherSeg
        model = TeacherSeg(backbone='lraspp', pretrained=False, input_size=128)
        model.eval()
        x = torch.rand(1, 3, 64, 64)  # 生[0,1]float
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64)  # student格子へ復帰
        assert torch.isfinite(y).all()

    def test_input_size_zero_no_upsample(self):
        """input_size=0は直接推論(旧TeacherLRASPP互換)。出力は入力格子。"""
        from segmentation_models import TeacherSeg
        model = TeacherSeg(backbone='lraspp', pretrained=False, input_size=0)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64)

    def test_teacher_lraspp_alias_state_dict_compatible(self):
        """旧TeacherLRASPPとTeacherSeg(lraspp,input_size=0)のkeyが一致(ckpt互換)。"""
        from segmentation_models import TeacherLRASPP, TeacherSeg
        a = TeacherLRASPP(pretrained=False)
        b = TeacherSeg(backbone='lraspp', pretrained=False, input_size=0)
        assert set(a.state_dict().keys()) == set(b.state_dict().keys())
        # 相互ロード可能
        b.load_state_dict(a.state_dict())

    def test_deeplabv3_backbone_builds_2class(self):
        from segmentation_models import TeacherSeg
        model = TeacherSeg(backbone='deeplabv3', pretrained=False, input_size=96)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64)

    def test_deeplabv3_has_aux_classifier(self):
        """回帰: pretrained=Falseでもaux_classifierが構築される
        (torchvisionはweights指定時のみaux_loss=True強制→pretrained有無で
        構造が食い違い、ckptロードで Unexpected key(s): net.aux_classifier.* に
        なっていた)。"""
        from segmentation_models import TeacherSeg
        model = TeacherSeg(backbone='deeplabv3', pretrained=False, input_size=224)
        assert model.net.aux_classifier is not None
        assert any(k.startswith('net.aux_classifier')
                   for k in model.state_dict().keys())

    def test_kd_teacher_ckpt_roundtrip_deeplabv3(self, tmp_path):
        """回帰: deeplabv3構成のsave→load_kd_teacher往復(GPU機Cクラッシュ再現)。"""
        from segmentation_models import TeacherSeg
        from train_segmentation import load_kd_teacher
        torch.manual_seed(0)
        teacher = TeacherSeg(backbone='deeplabv3', pretrained=False,
                             input_size=224)
        ckpt = {
            'model_state_dict': teacher.state_dict(),
            'optimizer_state_dict': {},
            'scheduler_state_dict': None,
            'step': 123,
            'best_metric': 0.6327,
            'metadata': {'args': {'teacher_backbone': 'deeplabv3',
                                  'teacher_input_size': 224}},
        }
        path = str(tmp_path / 'best.pt')
        torch.save(ckpt, path)
        loaded = load_kd_teacher(path, 'cpu')
        assert loaded.backbone == 'deeplabv3'
        assert loaded.input_size == 224
        assert not loaded.training  # eval固定
        assert all(not p.requires_grad for p in loaded.parameters())
        # 重みが実際に往復している(最終分類convで照合)
        assert torch.equal(loaded.net.classifier[-1].weight,
                           teacher.net.classifier[-1].weight)


# ============================================================
# 指標
# ============================================================

class TestSegMetrics:
    def test_perfect_prediction(self):
        from train_segmentation import SegMetrics
        m = SegMetrics()
        target = torch.randint(0, 2, (2, 8, 8))
        m.update(target.clone(), target)
        r = m.compute()
        assert r['miou'] == pytest.approx(1.0)
        assert r['pixel_acc'] == pytest.approx(1.0)

    def test_ignore_index_excluded(self):
        from train_segmentation import SegMetrics
        m = SegMetrics()
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 0] = IGNORE_INDEX  # 1行を無視
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        pred[0, 0] = 1  # 無視領域だけ外す
        m.update(pred, target)
        r = m.compute()
        assert r['pixel_acc'] == pytest.approx(1.0)

    def test_half_iou(self):
        from train_segmentation import SegMetrics
        m = SegMetrics()
        target = torch.tensor([[[1, 1, 0, 0]]])
        pred = torch.tensor([[[1, 0, 1, 0]]])
        m.update(pred, target)
        r = m.compute()
        # pet: TP=1 FP=1 FN=1 → IoU=1/3、Dice=2/4=0.5
        assert r['iou_pet'] == pytest.approx(1 / 3)
        assert r['dice'] == pytest.approx(0.5)


# ============================================================
# 多クラス指標(既知値でDice/IoU/mean_pixel_acc/boundary F1)
# ============================================================

class TestSegMetricsMultiClass:
    """3クラス既知ケースで手計算値と照合する。

    target = [[0,1],[2,255]] (255=ignore、除外)
    pred   = [[0,2],[2,0]]
    有効画素3つ: (0,0)T=0/P=0 正解, (0,1)T=1/P=2 誤り, (1,0)T=2/P=2 正解。
    """

    def _metrics(self):
        from train_segmentation import SegMetrics
        m = SegMetrics(n_classes=3)
        target = torch.tensor([[[0, 1], [2, 255]]])
        pred = torch.tensor([[[0, 2], [2, 0]]])
        m.update(pred, target)
        return m.compute()

    def test_iou_per_class_and_miou(self):
        r = self._metrics()
        assert r['iou_per_class'] == pytest.approx([1.0, 0.0, 0.5])
        assert r['miou'] == pytest.approx(0.5)

    def test_dice_per_class_and_mean(self):
        r = self._metrics()
        assert r['dice_per_class'] == pytest.approx([1.0, 0.0, 2 / 3])
        assert r['dice'] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)

    def test_pixel_acc_and_mean_pixel_acc(self):
        r = self._metrics()
        assert r['pixel_acc'] == pytest.approx(2 / 3)
        assert r['mean_pixel_acc'] == pytest.approx((1.0 + 0.0 + 1.0) / 3)

    def test_no_binary_only_keys(self):
        """n_classes!=2ではiou_petキーを出さない(後方互換キーの誤流用防止)"""
        r = self._metrics()
        assert 'iou_pet' not in r


class TestSegMetricsBoundaryF1:
    """境界F1(boundary IoU相当)を既知の縦境界パターンで検証する。"""

    def test_perfect_match_gives_f1_one(self):
        from train_segmentation import SegMetrics
        m = SegMetrics(n_classes=2, boundary_dilation=0)
        target = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        pred = target.clone()
        m.update(pred, target)
        r = m.compute()
        assert r['boundary_f1'] == pytest.approx(1.0)

    def test_shifted_boundary_recovered_by_dilation(self):
        """境界が1px水平シフトしたとき、dilation=0では罰則、dilation>=2で吸収される"""
        from train_segmentation import SegMetrics
        target = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        pred = torch.tensor([[[0, 1, 1, 1], [0, 1, 1, 1]]])

        m0 = SegMetrics(n_classes=2, boundary_dilation=0)
        m0.update(pred, target)
        r0 = m0.compute()
        assert r0['boundary_f1'] == pytest.approx(0.5)

        m2 = SegMetrics(n_classes=2, boundary_dilation=2)
        m2.update(pred, target)
        r2 = m2.compute()
        assert r2['boundary_f1'] == pytest.approx(1.0)
        assert r0['boundary_f1'] < r2['boundary_f1']


# ============================================================
# 多クラスモデル(n_classes駆動、手本=CenterNetLUT)
# ============================================================

class TestMultiClassModels:
    def test_seg_conv_dlgn_default_n_classes_backward_compat(self):
        from segmentation_models import SegConvDLGN
        torch.manual_seed(0)
        model = SegConvDLGN(in_channels=9, k=4, readout_group=4)
        assert model.n_classes == 2
        x = torch.randint(0, 2, (1, 9, 16, 16)).float()
        y = model(x)
        assert y.shape == (1, 2, 16, 16)

    def test_seg_conv_dlgn_voc_21_classes(self):
        from segmentation_models import SegConvDLGN
        torch.manual_seed(0)
        model = SegConvDLGN(in_channels=9, n_classes=21, k=4, readout_group=2)
        x = torch.randint(0, 2, (1, 9, 16, 16)).float()
        y = model(x)
        assert y.shape == (1, 21, 16, 16)
        assert model.d1.out_channels == 21 * 2

    def test_small_unet_voc_21_classes(self):
        from segmentation_models import SmallUNet
        model = SmallUNet(in_channels=3, n_classes=21, base=8)
        x = torch.rand(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 21, 32, 32)

    def test_build_model_plumbs_n_classes(self):
        from segmentation_models import build_model
        m_dlgn = build_model('convdlgn', in_channels=9, n_classes=21,
                             k=4, readout_group=2)
        m_unet = build_model('unet', in_channels=3, n_classes=21)
        assert m_dlgn.n_classes == 21
        assert m_unet.n_classes == 21


# ============================================================
# 強teacher化: n_classes=21で事前学習ヘッドを保持(置換スキップ)
# ============================================================

class TestPretrainedHeadPreservation:
    def test_lraspp_keeps_default_head_when_n_classes_21(self):
        """n_classes=21は事前学習ヘッドと同一クラス数→置換をスキップし、
        直接構築した骨格と(同一seedで)ビット一致する重みになる。"""
        from segmentation_models import _build_seg_backbone
        from torchvision.models import segmentation as tvseg
        torch.manual_seed(0)
        net_direct = tvseg.lraspp_mobilenet_v3_large(weights=None)
        torch.manual_seed(0)
        net_built = _build_seg_backbone('lraspp', pretrained=False, n_classes=21)
        assert torch.equal(net_direct.classifier.low_classifier.weight,
                           net_built.classifier.low_classifier.weight)
        assert torch.equal(net_direct.classifier.high_classifier.weight,
                           net_built.classifier.high_classifier.weight)

    def test_lraspp_replaces_head_when_n_classes_2(self):
        from segmentation_models import _build_seg_backbone
        net = _build_seg_backbone('lraspp', pretrained=False, n_classes=2)
        assert net.classifier.low_classifier.out_channels == 2
        assert net.classifier.high_classifier.out_channels == 2

    def test_deeplabv3_keeps_default_head_when_n_classes_21(self):
        from segmentation_models import _build_seg_backbone
        from torchvision.models import segmentation as tvseg
        torch.manual_seed(0)
        net_direct = tvseg.deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
        torch.manual_seed(0)
        net_built = _build_seg_backbone('deeplabv3', pretrained=False, n_classes=21)
        assert torch.equal(net_direct.classifier[-1].weight,
                           net_built.classifier[-1].weight)
        assert torch.equal(net_direct.aux_classifier[-1].weight,
                           net_built.aux_classifier[-1].weight)

    def test_deeplabv3_replaces_head_when_n_classes_2(self):
        from segmentation_models import _build_seg_backbone
        net = _build_seg_backbone('deeplabv3', pretrained=False, n_classes=2)
        assert net.classifier[-1].out_channels == 2
        assert net.aux_classifier[-1].out_channels == 2

    def test_teacher_seg_voc_21_classes_forward(self):
        from segmentation_models import TeacherSeg
        model = TeacherSeg(backbone='deeplabv3', pretrained=False,
                           input_size=64, n_classes=21)
        model.eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 21, 32, 32)

    def test_kd_teacher_ckpt_roundtrip_voc_21_classes(self, tmp_path):
        """load_kd_teacherがmetadata['args']['n_classes']=21を正しく復元する。"""
        from segmentation_models import TeacherSeg
        from train_segmentation import load_kd_teacher
        torch.manual_seed(0)
        teacher = TeacherSeg(backbone='deeplabv3', pretrained=False,
                             input_size=224, n_classes=21)
        ckpt = {
            'model_state_dict': teacher.state_dict(),
            'optimizer_state_dict': {},
            'scheduler_state_dict': None,
            'step': 1,
            'best_metric': 0.5,
            'metadata': {'args': {'teacher_backbone': 'deeplabv3',
                                  'teacher_input_size': 224, 'n_classes': 21}},
        }
        path = str(tmp_path / 'best.pt')
        torch.save(ckpt, path)
        loaded = load_kd_teacher(path, 'cpu')
        assert loaded.n_classes == 21
        assert loaded.net.classifier[-1].out_channels == 21


# ============================================================
# evaluate()のround_input(回帰: 連続float画像(unet/teacher-ft)を
# 誤ってx.round()で二値化(posterize)し、teacher hard mIoUが見かけ上
# 0.11程度まで落ちていたバグ)
# ============================================================

class _CaptureInputModel(torch.nn.Module):
    """forwardに渡されたxをそのまま記録するダミーモデル(round適用有無の検証用)。"""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.seen = None
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.seen = x.detach().clone()
        b, _, h, w = x.shape
        return torch.zeros(b, self.n_classes, h, w) + self.p


class TestEvaluateRoundInput:
    def _loader(self):
        x = torch.tensor([[[[0.2, 0.7], [0.5, 0.9]]]])  # (1,1,2,2) 連続float
        y = torch.zeros(1, 2, 2, dtype=torch.long)
        return [(x, y)], x

    def test_round_input_true_binarizes_hard_eval(self):
        """round_input=True(convdlgn向け)は mode=False で x.round()を適用する。"""
        from train_segmentation import evaluate
        model = _CaptureInputModel()
        loader, x = self._loader()
        evaluate(model, loader, 'cpu', mode=False, n_classes=2, round_input=True)
        assert torch.equal(model.seen, x.round())

    def test_round_input_false_keeps_continuous_float(self):
        """round_input=False(既定、unet/teacher-ft向け)は連続float画像を
        壊さずそのまま渡す(過去に発覚したposterizeバグの回帰防止)。"""
        from train_segmentation import evaluate
        model = _CaptureInputModel()
        loader, x = self._loader()
        evaluate(model, loader, 'cpu', mode=False, n_classes=2, round_input=False)
        assert torch.equal(model.seen, x)

    def test_default_round_input_is_false(self):
        """round_inputを明示しない呼び出しは丸めない(安全側デフォルト)。"""
        from train_segmentation import evaluate
        model = _CaptureInputModel()
        loader, x = self._loader()
        evaluate(model, loader, 'cpu', mode=False, n_classes=2)
        assert torch.equal(model.seen, x)

    def test_soft_mode_never_rounds_regardless_of_flag(self):
        from train_segmentation import evaluate
        model = _CaptureInputModel()
        loader, x = self._loader()
        evaluate(model, loader, 'cpu', mode=True, n_classes=2, round_input=True)
        assert torch.equal(model.seen, x)


class TestRenderSamplesRoundInput:
    def test_thermometer_false_keeps_continuous_float(self):
        """thermometer=False(unet/teacher-ft)は入力を丸めずモデルへ渡す。"""
        from train_segmentation import render_samples
        model = _CaptureInputModel(n_classes=3)
        x = torch.tensor([[[[0.2, 0.7], [0.5, 0.9]],
                           [[0.1, 0.4], [0.6, 0.3]],
                           [[0.9, 0.2], [0.1, 0.8]]]])  # (1,3,2,2)
        y = torch.zeros(1, 2, 2, dtype=torch.long)
        loader = [(x, y)]
        render_samples(model, loader, 'cpu', n_samples=1, thermometer=False,
                       n_classes=3)
        assert torch.equal(model.seen, x)

    def test_thermometer_true_rounds_input(self):
        from train_segmentation import render_samples
        model = _CaptureInputModel(n_classes=3)
        x = torch.tensor([[[[0.2, 0.7], [0.5, 0.9]],
                           [[0.1, 0.4], [0.6, 0.3]],
                           [[0.9, 0.2], [0.1, 0.8]]]])
        y = torch.zeros(1, 2, 2, dtype=torch.long)
        loader = [(x, y)]
        render_samples(model, loader, 'cpu', n_samples=1, thermometer=True,
                       n_classes=3)
        assert torch.equal(model.seen, x.round())


# ============================================================
# VOCSeg ローダ(ネットワーク不要=torchvision.datasets.VOCSegmentationを
# 合成データでモンキーパッチ)
# ============================================================

class TestVOCSeg:
    def _patch_voc(self, monkeypatch):
        from PIL import Image
        img = Image.new('RGB', (10, 8), color=(100, 150, 200))
        target = Image.new('P', (10, 8), color=0)
        px = target.load()
        for x in range(10):
            for y in range(8):
                if x < 3:
                    px[x, y] = 0       # background
                elif x < 7:
                    px[x, y] = 15      # person
                else:
                    px[x, y] = 255     # void

        class _FakeVOCSegmentation:
            def __init__(self, root, year, image_set, download):
                self._items = [(img, target)]

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        import torchvision
        monkeypatch.setattr(
            torchvision.datasets, 'VOCSegmentation', _FakeVOCSegmentation)

    def test_constants(self):
        from data.segmentation_voc import VOC_SEG_CLASSES, VOC_SEG_N_CLASSES
        assert VOC_SEG_N_CLASSES == 21
        assert len(VOC_SEG_CLASSES) == 21
        assert VOC_SEG_CLASSES[0] == 'background'
        assert VOC_SEG_CLASSES[-1] == 'tvmonitor'

    def test_float_encoding_shapes(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=8,
                   encoding='float')
        assert len(ds) == 1
        assert ds.in_channels == 3
        assert ds.n_classes == 21
        x, mask = ds[0]
        assert x.shape == (3, 8, 8)
        assert mask.shape == (8, 8)
        assert mask.dtype == torch.long

    def test_thermometer_encoding_channels(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=8,
                   encoding='thermometer', n_thresholds=3)
        assert ds.in_channels == 9
        x, mask = ds[0]
        assert x.shape == (9, 8, 8)
        assert set(x.unique().tolist()) <= {0.0, 1.0}

    def test_void_maps_to_ignore_index(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=10,
                   encoding='float')
        _, mask = ds[0]
        assert (mask == IGNORE_INDEX).any()
        assert (mask == 0).any()   # background
        assert (mask == 15).any()  # person

    def test_return_raw_gives_three_items(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=8,
                   encoding='thermometer', return_raw=True)
        item = ds[0]
        assert len(item) == 3
        x, mask, raw = item
        assert raw.shape == (3, 8, 8)

    def test_load_voc_seg_train_val_split(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import load_voc_seg
        train_set, val_set = load_voc_seg(root=str(tmp_path), img_size=8)
        assert len(train_set) == 1
        assert len(val_set) == 1


# ============================================================
# データ拡張: 幾何拡張(scale+crop+flip)の画像↔マスク整合、
# パディング領域=IGNORE_INDEX、測光拡張はtrainのみ・valは常に無変換
# ============================================================

def _stub_rng(uniform_val, randint_vals, random_val):
    """random.Random互換の決定的スタブ(uniform/randint/randomを固定値化)。"""
    vals = list(randint_vals)

    class _Stub:
        def uniform(self, a, b):
            return uniform_val

        def randint(self, a, b):
            return vals.pop(0)

        def random(self):
            return random_val

    return _Stub()


class TestRandomScaleCropFlip:
    def _img_mask(self, size):
        import numpy as np
        from PIL import Image
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        mask_arr = rng.integers(0, 21, size=(size, size), dtype=np.uint8)
        mask = Image.fromarray(mask_arr, mode='P')
        return img, mask

    def test_scale_one_flip_applies_identically_to_image_and_mask(self):
        """scale=1.0(resize実質no-op)+flip強制時、画像/マスクへ同一のflipが
        適用される(既知値=同じPIL resize/flipを手動再現した期待値と一致)。"""
        from data.segmentation_voc import random_scale_crop_flip
        img, mask = self._img_mask(4)
        rng = _stub_rng(uniform_val=1.0, randint_vals=[], random_val=0.1)  # flip

        img_out, mask_out = random_scale_crop_flip(img, mask, img_size=4, rng=rng)

        import numpy as np
        from PIL import Image
        img_expected = img.resize((4, 4), Image.BILINEAR).transpose(
            Image.FLIP_LEFT_RIGHT)
        mask_expected = mask.resize((4, 4), Image.NEAREST).transpose(
            Image.FLIP_LEFT_RIGHT)
        assert np.array_equal(np.array(img_out), np.array(img_expected))
        assert np.array_equal(np.array(mask_out), np.array(mask_expected))

    def test_scale_up_crop_offset_applies_identically_to_image_and_mask(self):
        """scale>1(パディング無し)+crop offset固定時、画像/マスクが同一領域から
        切り出される(既知値=同じresize/cropを手動再現した期待値と一致)。"""
        from data.segmentation_voc import random_scale_crop_flip
        img, mask = self._img_mask(8)
        # scale=1.5 → new_size=6、max_off=2 → randint呼び出し2回(off_x, off_y)
        rng = _stub_rng(uniform_val=1.5, randint_vals=[1, 2], random_val=0.9)  # no flip

        img_out, mask_out = random_scale_crop_flip(img, mask, img_size=4, rng=rng)

        import numpy as np
        from PIL import Image
        img_s = img.resize((6, 6), Image.BILINEAR)
        mask_s = mask.resize((6, 6), Image.NEAREST)
        img_expected = img_s.crop((1, 2, 5, 6))
        mask_expected = mask_s.crop((1, 2, 5, 6))
        assert np.array_equal(np.array(img_out), np.array(img_expected))
        assert np.array_equal(np.array(mask_out), np.array(mask_expected))

    def test_scale_down_pads_mask_with_ignore_index(self):
        """scale<1(パディング発生)時、パディング領域のマスクはIGNORE_INDEX、
        画像は黒(0,0,0)で埋まる(画像/マスクとも同一paste位置)。"""
        from data.segmentation_voc import random_scale_crop_flip
        img, mask = self._img_mask(8)
        # scale=0.5 → new_size=4、pad=4 → randint呼び出し2回(pad_l, pad_t)
        # (パディング後new_size=img_size=8なのでcrop offsetのrandintは呼ばれない)
        rng = _stub_rng(uniform_val=0.5, randint_vals=[1, 3], random_val=0.9)  # no flip

        img_out, mask_out = random_scale_crop_flip(img, mask, img_size=8, rng=rng)

        import numpy as np
        from PIL import Image
        img_s = img.resize((4, 4), Image.BILINEAR)
        mask_s = mask.resize((4, 4), Image.NEAREST)
        img_arr = np.array(img_out)
        mask_arr = np.array(mask_out)

        # 貼り付け領域(pad_l=1, pad_t=3, 4x4)は元のリサイズ結果と一致
        assert np.array_equal(img_arr[3:7, 1:5], np.array(img_s))
        assert np.array_equal(mask_arr[3:7, 1:5], np.array(mask_s))

        # 貼り付け領域外はimg=黒、mask=IGNORE_INDEX(255)
        outside = np.ones((8, 8), dtype=bool)
        outside[3:7, 1:5] = False
        assert (img_arr[outside] == 0).all()
        assert (mask_arr[outside] == IGNORE_INDEX).all()

    def test_default_rng_is_module_random_when_none(self):
        """rng省略時はmodule既定のrandomを使い、正常に動作する(非決定的だが
        エラーにならないこと・出力形状が期待通りであることのみ検証)。"""
        from data.segmentation_voc import random_scale_crop_flip
        img, mask = self._img_mask(6)
        img_out, mask_out = random_scale_crop_flip(img, mask, img_size=6)
        assert img_out.size == (6, 6)
        assert mask_out.size == (6, 6)


class TestVOCSegAugment:
    def _patch_voc(self, monkeypatch, size=(16, 16)):
        import numpy as np
        from PIL import Image
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        mask_arr = np.zeros((size[1], size[0]), dtype=np.uint8)
        mask_arr[:, :size[0] // 2] = 5
        mask_arr[:2, :] = 255  # void帯
        target = Image.fromarray(mask_arr, mode='P')

        class _FakeVOCSegmentation:
            def __init__(self, root, year, image_set, download):
                self._items = [(img, target)]

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        import torchvision
        monkeypatch.setattr(
            torchvision.datasets, 'VOCSegmentation', _FakeVOCSegmentation)

    def test_augment_default_off(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=8)
        assert ds.augment is False

    def test_augment_true_shapes_and_valid_mask_values(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=12,
                   encoding='float', augment=True)
        x, mask = ds[0]
        assert x.shape == (3, 12, 12)
        assert mask.shape == (12, 12)
        valid_values = set(range(21)) | {IGNORE_INDEX}
        assert set(mask.unique().tolist()) <= valid_values
        assert torch.isfinite(x).all()
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_augment_true_thermometer_still_binary(self, monkeypatch, tmp_path):
        """測光拡張(ColorJitter)後もthermometer化で{0,1}になる(範囲外に漏れない)。"""
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import VOCSeg
        ds = VOCSeg(root=str(tmp_path), image_set='train', img_size=12,
                   encoding='thermometer', n_thresholds=3, augment=True)
        x, mask = ds[0]
        assert x.shape == (9, 12, 12)
        assert set(x.unique().tolist()) <= {0.0, 1.0}

    def test_load_voc_seg_augment_train_only(self, monkeypatch, tmp_path):
        """augment=Trueはtrain_setのみ有効、val_setは常にaugment=False。"""
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import load_voc_seg
        train_set, val_set = load_voc_seg(
            root=str(tmp_path), img_size=8, augment=True)
        assert train_set.augment is True
        assert val_set.augment is False

    def test_load_voc_seg_augment_false_both_off(self, monkeypatch, tmp_path):
        self._patch_voc(monkeypatch)
        from data.segmentation_voc import load_voc_seg
        train_set, val_set = load_voc_seg(
            root=str(tmp_path), img_size=8, augment=False)
        assert train_set.augment is False
        assert val_set.augment is False


# ============================================================
# lrスケジューラ(build_lr_scheduler)
# ============================================================

class TestBuildLRScheduler:
    def test_none_returns_none(self):
        from train_segmentation import build_lr_scheduler
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        assert build_lr_scheduler(opt, 'none', epochs=10) is None

    def test_cosine_starts_at_base_lr_and_decays_to_near_zero(self):
        from train_segmentation import build_lr_scheduler
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=0.1)
        sched = build_lr_scheduler(opt, 'cosine', epochs=10)
        lrs = [opt.param_groups[0]['lr']]
        for _ in range(10):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])
        assert lrs[0] == pytest.approx(0.1)
        assert lrs[-1] == pytest.approx(0.0, abs=1e-6)
        assert lrs[5] < lrs[0]
        # 単調減少(cosineの前半)
        assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1))

    def test_invalid_schedule_raises(self):
        from train_segmentation import build_lr_scheduler
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        with pytest.raises(ValueError):
            build_lr_scheduler(opt, 'bogus', epochs=10)


class TestLoadN:
    """load_n(W13a-2、train_detection.pyのload_nと同型)"""

    def test_yields_exactly_n_batches(self):
        from train_segmentation import load_n
        loader = [1, 2, 3]
        out = list(load_n(loader, 7))
        assert out == [1, 2, 3, 1, 2, 3, 1]

    def test_n_smaller_than_loader(self):
        from train_segmentation import load_n
        loader = [1, 2, 3]
        assert list(load_n(loader, 2)) == [1, 2]


class TestBuildSelectionScheduler:
    """build_selection_scheduler(W13a-1、train_detection.pyと同型)"""

    def _model(self, selection, **kw):
        from segmentation_models import SegConvDLGN
        return SegConvDLGN(in_channels=9, k=4, readout_group=4,
                           selection=selection, **kw)

    def _args(self, **overrides):
        import argparse
        ns = argparse.Namespace(
            arch='convdlgn', selection='soft', tau_schedule=None,
            tau_b_init=1.0, tau_b_min=0.1, gumbel_scale_init=1.0,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_soft_returns_none(self):
        from train_segmentation import build_selection_scheduler
        model = self._model('soft')
        args = self._args(selection='soft')
        assert build_selection_scheduler(args, model, 100) is None

    def test_non_convdlgn_arch_returns_none(self):
        from train_segmentation import build_selection_scheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st', arch='unet')
        assert build_selection_scheduler(args, model, 100) is None

    def test_hard_st_default_is_none_schedule(self):
        """hard_st既定(--tau-schedule未指定)はtau_b固定(スケジューラNone)"""
        from train_segmentation import build_selection_scheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st')
        assert build_selection_scheduler(args, model, 100) is None

    def test_gumbel_st_default_is_anneal(self):
        """gumbel_st既定はTauAnnealConvScheduler(hardgap知見によりアニール必須)"""
        from cage_scheduler import TauAnnealConvScheduler
        from train_segmentation import build_selection_scheduler
        model = self._model('gumbel_st')
        args = self._args(selection='gumbel_st')
        sched = build_selection_scheduler(args, model, 50)
        assert isinstance(sched, TauAnnealConvScheduler)
        assert sched.total_steps == 50

    def test_explicit_cage_schedule(self):
        from train_segmentation import build_selection_scheduler

        from griffith_logic.conv_hard_st import CAGEConvScheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st', tau_schedule='cage')
        sched = build_selection_scheduler(args, model, 50)
        assert isinstance(sched, CAGEConvScheduler)
