"""セグメンテーションモデル定義。

- SegConvDLGN: LutConv2dエンコーダ/デコーダのU-Net風 Conv-DLGN(本命)。
  **全LutConv2dは parameterization='softmax16g'**(旧softmaxは構造欠陥、
  model-conversion-kdスキル§4)。ダウンサンプル=OrPooling2d、
  アップサンプル=UpSample2d(nearest、IR v0.2互換)、スキップ=チャネルconcat
  (IR concatノード互換)、読み出し=SpatialGroupSum(per-pixel 2クラス)。
- SmallUNet: CNNベースライン(2段エンコーダ+bottleneck、~1Mパラメータ)
- TeacherSeg: torchvision事前学習セグ骨格(lraspp/deeplabv3/fcn_resnet50)の
  2クラスヘッド付け替え + 高解像度推論ラッパ(入力をinput_sizeへup→推論→
  student格子へdown)。W3 KDのteacher(lift源泉)。TeacherLRASPPは旧64直接推論の
  後方互換エイリアス。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import LutConv2d, OrPooling2d, SpatialGroupSum, UpSample2d
from griffith_logic.conv_hard_st import LutConv2dHardST

SELECTIONS = ('soft', 'hard_st', 'gumbel_st')


class SegConvDLGN(nn.Module):
    """U-Net風 Conv-DLGN バイナリセグメンテーションモデル。

    構成(img 64、k=32、d=3の場合):
        c1 = LutConv2d(in_ch, k, 3x3, d) @64
        p1 = OrPool2 → 32
        c2 = LutConv2d(k, 2k, 3x3, d) @32
        p2 = OrPool2 → 16
        c3 = LutConv2d(2k, 4k, 3x3, d) @16
        u2 = UpSample2(c3) → 32、cat[u2, c2] = 6k
        d2 = LutConv2d(6k, 2k, 3x3, d) @32
        u1 = UpSample2(d2) → 64、cat[u1, c1] = 3k
        d1 = LutConv2d(3k, 2·g, 3x3, d) @64
        SpatialGroupSum(2, scale=1/τ) → (B, 2, 64, 64) per-pixelロジット

    τ = sqrt(g) 自動較正(train_cnn.py の τ∝√(n_outputs_per_class) 流儀)。

    n_classes(既定2、後方互換=Pet binary seg)を増やすと d1 の出力チャネルが
    n_classes * readout_group へ拡大する(手本=CenterNetLUT、n_classes駆動)。
    """

    N_CLASSES = 2  # 後方互換の既定値(2クラスbinary seg)

    def __init__(
        self,
        in_channels: int = 9,
        n_classes: int = N_CLASSES,
        k: int = 32,
        tree_depth: int = 3,
        readout_group: int = 32,
        input_channels_per_tree: int = 4,
        tau: float | None = None,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        wiring: str = 'fixed',
        selection: str = 'soft',
        n_inputs: int = 2,
        parameterization: str = 'softmax16g',
        tau_b: float = 1.0,
        gumbel_scale: float = 1.0,
    ):
        """
        n_inputs: 1ゲートの入力数(2|4|6)。2より大きくすると1ゲートが
            同時に見る変数が増える。検出では n=6 が唯一効いた軸だった
            (mAP 0.6237→0.6379、配信 0.6290 超え)。
        parameterization: パラメータ化方式。n_inputs>2 では softmax16g が
            使えない(16ゲート真理値表が2入力固有)ため 'independent_dwn' を使う。
            'independent'(連続のまま学習して最後に丸める)は離散化で崩壊するので
            使わないこと。
        wiring: LutConv2dの入力配線方式('fixed'|'learnable')。
        selection: 訓練時のゲート選択方式(W13a-1、既定'soft'で従来挙動と
            数値不変)。'hard_st'|'gumbel_st'時は全LutConv2dがLutConv2dHardST
            (conv_hard_st.py)へ切り替わり、selection gap 0の訓練になる。
            softmax16g対応(TreeHardSTFunctionのgate_table経由)なので
            構造欠陥は生じない。wiring='learnable'と併用可。
        tau_b: selection != 'soft' 時のHard-ST逆伝播温度初期値
        gumbel_scale: selection='gumbel_st'時のGumbelノイズ振幅初期値
        """
        super().__init__()
        if selection not in SELECTIONS:
            raise ValueError(f'selection は {SELECTIONS} のいずれか: {selection}')
        self.k = k
        self.n_classes = n_classes
        self.readout_group = readout_group
        self.selection = selection
        if tau is None:
            tau = max(1.0, math.sqrt(readout_group))
        self.tau = tau

        if n_inputs != 2 and parameterization == 'softmax16g':
            raise ValueError(
                'softmax16g は n_inputs=2 専用(16ゲート真理値表が2入力固有)。'
                f"n_inputs={n_inputs} では parameterization='independent_dwn' を"
                '使うこと'
            )
        self.n_inputs = n_inputs

        conv_kw = dict(
            kernel_size=3, tree_depth=tree_depth, padding=1,
            n_inputs=n_inputs, device=device, grad_factor=grad_factor,
            input_channels_per_tree=input_channels_per_tree,
            # 既定は softmax16g(スキル§4: 旧softmaxは構造欠陥)。
            # n_inputs>2 では上のガードにより independent_dwn が要求される。
            parameterization=parameterization,
            wiring=wiring,  # 'fixed'(既定、後方互換)|'learnable'
        )
        if selection != 'soft':
            conv_kw['selection'] = selection
            conv_kw['tau_b'] = tau_b
            conv_kw['gumbel_scale'] = gumbel_scale
        conv_cls = LutConv2d if selection == 'soft' else LutConv2dHardST

        self.c1 = conv_cls(in_channels, k, **conv_kw)
        self.p1 = OrPooling2d(2)
        self.c2 = conv_cls(k, 2 * k, **conv_kw)
        self.p2 = OrPooling2d(2)
        self.c3 = conv_cls(2 * k, 4 * k, **conv_kw)
        self.u2 = UpSample2d(2)
        self.d2 = conv_cls(6 * k, 2 * k, **conv_kw)
        self.u1 = UpSample2d(2)
        self.d1 = conv_cls(3 * k, self.n_classes * readout_group, **conv_kw)
        self.readout = SpatialGroupSum(self.n_classes, scale=1.0 / tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) の {0,1}(soft時は[0,1])
        → (B, 2, H, W) per-pixelロジット"""
        f1 = self.c1(x)                       # (B, k, H, W)
        f2 = self.c2(self.p1(f1))             # (B, 2k, H/2, W/2)
        f3 = self.c3(self.p2(f2))             # (B, 4k, H/4, W/4)
        g2 = self.d2(torch.cat([self.u2(f3), f2], dim=1))  # (B, 2k, H/2, W/2)
        g1 = self.d1(torch.cat([self.u1(g2), f1], dim=1))  # (B, n_classes*g, H, W)
        return self.readout(g1)               # (B, n_classes, H, W)

    @property
    def num_gates(self) -> int:
        """総ゲート数(全LutConv2dの内部ノード数=パラメータゲート数)"""
        return sum(m.num_gates for m in self.modules()
                   if isinstance(m, LutConv2d))


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """小U-Netベースライン(2段エンコーダ+bottleneck、base=48で~1Mパラメータ)。

    n_classes(既定2、後方互換)は最終1x1 convの出力チャネル数。
    """

    N_CLASSES = 2  # 後方互換の既定値(2クラスbinary seg)

    def __init__(self, in_channels: int = 3, n_classes: int = N_CLASSES, base: int = 48):
        super().__init__()
        b = base
        self.n_classes = n_classes
        self.enc1 = _double_conv(in_channels, b)
        self.enc2 = _double_conv(b, 2 * b)
        self.bott = _double_conv(2 * b, 4 * b)
        self.dec2 = _double_conv(4 * b + 2 * b, 2 * b)
        self.dec1 = _double_conv(2 * b + b, b)
        self.head = nn.Conv2d(b, self.n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.bott(self.pool(f2))
        u2 = F.interpolate(f3, scale_factor=2, mode='bilinear',
                           align_corners=False)
        g2 = self.dec2(torch.cat([u2, f2], dim=1))
        u1 = F.interpolate(g2, scale_factor=2, mode='bilinear',
                           align_corners=False)
        g1 = self.dec1(torch.cat([u1, f1], dim=1))
        return self.head(g1)


TEACHER_BACKBONES = ('lraspp', 'deeplabv3', 'fcn_resnet50')


# torchvisionセグ骨格の事前学習ヘッドが対応するクラス数(COCO_WITH_VOC_LABELS_V1
# =VOCの21クラス: background+20)。n_classesがこれと一致するときはヘッド置換を
# スキップし、事前学習済み分類headをそのまま使う(強teacher化)。
_PRETRAINED_HEAD_N_CLASSES = 21


def _build_seg_backbone(backbone: str, pretrained: bool, n_classes: int) -> nn.Module:
    """torchvisionセグ骨格を構築し、必要ならヘッドを n_classes へ付け替える。

    どの骨格も COCO_WITH_VOC_LABELS_V1(21クラス: background+VOC20)事前学習。
    n_classes == 21(事前学習ヘッドと一致)のときは最終分類convの置換を
    **スキップ**し、事前学習済みヘッドをそのまま保持する(VOC本番で事前学習の
    知識を最大限活用する強teacher化)。n_classes != 21(例: Pet binary seg=2)
    のときは従来通り最終分類convを n_classes へ差し替える。
    """
    from torchvision.models import segmentation as tvseg
    keep_pretrained_head = (n_classes == _PRETRAINED_HEAD_N_CLASSES)
    if backbone == 'lraspp':
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
        w = (LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
             if pretrained else None)
        net = tvseg.lraspp_mobilenet_v3_large(weights=w)
        if not keep_pretrained_head:
            low_ch = net.classifier.low_classifier.in_channels
            high_ch = net.classifier.high_classifier.in_channels
            net.classifier.low_classifier = nn.Conv2d(low_ch, n_classes, 1)
            net.classifier.high_classifier = nn.Conv2d(high_ch, n_classes, 1)
        return net
    # aux_loss=True を明示する(修正): torchvisionはweights指定時に
    # aux_loss=Trueへ上書きする(=aux_classifierあり)が、weights=None
    # (pretrained=False、KDロード側の骨格構築)では既定でauxなしになり、
    # 訓練時ckptとロード骨格の state_dict キーが食い違う
    # (Unexpected key(s): net.aux_classifier.*)。pretrainedに依らず
    # aux_loss=True で構造を決定的にし、save→load往復を保証する。
    if backbone == 'deeplabv3':
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
        w = (DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
             if pretrained else None)
        net = tvseg.deeplabv3_mobilenet_v3_large(weights=w, aux_loss=True)
    elif backbone == 'fcn_resnet50':
        from torchvision.models.segmentation import FCN_ResNet50_Weights
        w = (FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
             if pretrained else None)
        net = tvseg.fcn_resnet50(weights=w, aux_loss=True)
    else:
        raise ValueError(f'未対応 teacher backbone: {backbone} (対応: {TEACHER_BACKBONES})')
    if keep_pretrained_head:
        return net
    # DeepLabHead / FCNHead: classifier[-1] が最終分類conv、aux_classifier[-1] も同様
    in_ch = net.classifier[-1].in_channels
    net.classifier[-1] = nn.Conv2d(in_ch, n_classes, 1)
    if getattr(net, 'aux_classifier', None) is not None:
        aux_in = net.aux_classifier[-1].in_channels
        net.aux_classifier[-1] = nn.Conv2d(aux_in, n_classes, 1)
    return net


class TeacherSeg(nn.Module):
    """高解像度推論ラッパ付き torchvisionセグ teacher(W3 KDのlift源泉)。

    入力は生[0,1]float画像(teacher側で正規化=convert/teacher.py流儀)。
    forwardは (B, 3, H, W) を受け、内部で input_size × input_size へbilinear
    アップサンプルしてから推論し、ロジットを元の (H, W)(student格子)へ
    bilinearダウンサンプルして返す。LRASPP/DeepLabV3の出力ストライド8/16が
    小入力(64)ではタイル過ぎる問題を**入力側の高解像度化**で解消する
    (学生は64のまま=公平比較+web軽量要件を維持)。

    input_size が入力解像度と一致すれば(あるいは0なら)アップサンプルを行わない
    (= 旧 TeacherLRASPP と同一挙動、state_dict互換)。

    n_classes(既定2、後方互換=Pet binary seg)を21にすると事前学習ヘッド
    (COCO_WITH_VOC_LABELS_V1)をそのまま保持する強teacher化になる(_build_seg_backbone参照)。
    """

    N_CLASSES = 2  # 後方互換の既定値(2クラスbinary seg)
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = 'lraspp',
        pretrained: bool = True,
        input_size: int = 224,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_size = int(input_size)
        self.n_classes = n_classes
        self.net = _build_seg_backbone(backbone, pretrained, self.n_classes)
        self.register_buffer('mean', torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        z = x
        if self.input_size and (h != self.input_size or w != self.input_size):
            z = F.interpolate(
                z, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False)
        z = (z - self.mean) / self.std
        out = self.net(z)['out']  # (B, n_classes, in, in)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(
                out, size=(h, w), mode='bilinear', align_corners=False)
        return out


# 後方互換: 旧 TeacherLRASPP(64×64直接推論=input_size相当無し)。
# 既存ckpt/importが参照する可能性があるため薄いエイリアスとして残す。
class TeacherLRASPP(TeacherSeg):
    """[非推奨] TeacherSeg(backbone='lraspp', input_size=0) 相当(64直接推論)。"""

    def __init__(self, pretrained: bool = True):
        super().__init__(backbone='lraspp', pretrained=pretrained, input_size=0)


def build_model(arch: str, in_channels: int, n_classes: int = 2, **kw):
    """arch名 → モデル。train_segmentation.py から使用する。n_classesは全アーキ共通。"""
    if arch == 'convdlgn':
        return SegConvDLGN(in_channels=in_channels, n_classes=n_classes, **kw)
    if arch == 'unet':
        return SmallUNet(in_channels=in_channels, n_classes=n_classes)
    if arch == 'teacher-ft':
        return TeacherSeg(
            backbone=kw.get('teacher_backbone', 'lraspp'),
            pretrained=kw.get('pretrained', True),
            input_size=kw.get('teacher_input_size', 224),
            n_classes=n_classes,
        )
    raise ValueError(f'未対応アーキ: {arch}')
