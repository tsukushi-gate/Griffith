"""
CenterNet + LUT（微分可能論理ゲート）物体検出モデル。

バックボーン: LutConv2d + OrPooling2d
検出ヘッド: ヒートマップ + サイズ + オフセット

mAP頭打ち(0.4654、訓練長/容量/配線学習/selection/CIFAR事前学習/転写は
全て無効と確定)を受けた「素子がCNNに対して弱すぎる」仮説に基づく4件のアーキ改造。
**全て既定値では従来と数値不変**（既存テスト・チェックポイントに影響しない）。

  - downsample: 'orpool'(既定)|'strided' ダウンサンプルを学習可能にする
  - neck: 'none'(既定)|'fpn'|'fpnpan' 多スケール特徴融合neck
  - decoupled_head: bool(既定False) 分類/回帰ヘッドの特徴ブランチ分離
  - reg_bins: int(既定0)                    size/offのDFL風ビン分布回帰
  - head_type: 'lut'(既定)|'float' ハイブリッド(float検出ヘッド)

head_type='float' は IR エクスポート不可（scripts/export_webbench.py が
明示的にエラーにする。「学習配線exportが壊れてWebデモが動いていなかった」
事故の再発防止方針を踏襲）。
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import LutConv2d, OrPooling2d, UpSample2d
from griffith_logic.conv_hard_st import LutConv2dHardST
from griffith_logic.dwn_conv import DWNConv2d
from griffith_logic.spatial_group_sum import SpatialGroupSum

#(レバー4): 'dwn' は「離散のまま学習する」方式。
# soft/hard_st/gumbel_st はいずれも 16種ゲートに対する連続パラメータを学び
# 評価時に argmax で確定する（学習時と評価時で計算が違う）。
# 'dwn' は n入力RAMノードの表を有限差分(FD/EFD)で直接学ぶので、
# forward が学習時と評価時で完全に一致する（tests/l1_math/test_dwn_conv.py）。
SELECTIONS = ('soft', 'hard_st', 'gumbel_st', 'dwn')
DOWNSAMPLES = ('orpool', 'strided')
NECKS = ('none', 'fpn', 'fpnpan')
HEAD_TYPES = ('lut', 'float')


class CenterNetLUT(nn.Module):
    """
    CenterNet物体検出モデル（LUTベース）。

    バックボーン（Conv + OrPooling、出力ストライド4）と
    3つの検出ヘッド（ヒートマップ、サイズ、オフセット）で構成。

    n_layers で深さを選択:
      3: 元の3層（Conv→Pool→Conv→Pool→Conv）
      6: 6層（Stage2,3に同チャンネルConvを追加）
      8: 8層 + Stage3にOR Skip Connection
    """

    def __init__(
        self,
        n_classes: int = 10,
        k: int = 64,
        in_channels: int = 3,
        tree_depth: int = 3,
        head_depth: int = 3,
        n_inputs: int = 2,
        hm_group: int = 16,
        size_group: int = 16,
        offset_group: int = 4,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        input_channels_per_tree: int = 2,
        parameterization: str = 'softmax16g',
        n_layers: int = 3,
        wiring: str = 'fixed',
        selection: str = 'soft',
        dwn_n_inputs: int = 6,
        dwn_efd: str = 'paper',
        tau_b: float = 1.0,
        gumbel_scale: float = 1.0,
        downsample: str = 'orpool',
        neck: str = 'none',
        decoupled_head: bool = False,
        reg_bins: int = 0,
        head_type: str = 'lut',
    ):
        """
        Args:
            n_classes: 検出クラス数
            k: バックボーンの幅パラメータ
            in_channels: 入力チャンネル数（閾値二値化後）
            tree_depth: バックボーンのLUTツリーの深さ
            head_depth: 検出ヘッドのLUTツリーの深さ
            n_inputs: ゲートの入力数
            hm_group: ヒートマップヘッドのグループサイズ。LUT出力は離散(hard eval
                では0/1)なので、hm_group個の出力をSpatialGroupSumで平均して連続値
                [0,1]のヒートマップにする（hard評価でNMS/topKが機能する前提）。
            size_group: サイズヘッドのグループサイズ（reg_bins>0のときはビン
                あたりの投票グループサイズの意味に変わる）
            offset_group: オフセットヘッドのグループサイズ（同上）
            device: 計算デバイス
            grad_factor: 勾配スケーリング係数
            input_channels_per_tree: 各ツリーが参照する入力チャンネル数
            parameterization: パラメータ化方式（Conv-DLGNは'softmax16g'必須。
                旧'softmax'はパススルー/OR/XOR表現不能+residual減衰の構造欠陥）
            n_layers: バックボーンのConv層数（3, 6, 8）
            wiring: LutConv2dの入力配線方式（'fixed'|'learnable'）。
                既定'fixed'で従来挙動と同一（後方互換）
            selection: 訓練時のゲート選択方式（W13a-1、既定'soft'で従来挙動と
                数値不変）。'hard_st'|'gumbel_st'時は全LutConv2dが
                LutConv2dHardST（conv_hard_st.py）へ切り替わり、
                selection gap 0の訓練になる。wiring='learnable'と併用可
                （LutConv2dHardSTは_tree_forward_trainのみ差し替えるため、
                配線学習パスは無改造で共存する）
            tau_b: selection != 'soft' 時のHard-ST逆伝播温度初期値
                （既定1.0、CAGEConvScheduler/TauAnnealConvSchedulerで
                訓練中に更新される想定）
            gumbel_scale: selection='gumbel_st'時のGumbelノイズ振幅初期値
            downsample: バックボーンのダウンサンプル方式（既定'orpool'
                で従来挙動と数値不変）。'strided'ではOrPooling2d(2)の代わりに
                kernel_size=2/stride=2のLutConv2d（学習可能ダウンサンプル）を
                使用する。neckのtop-down経路のダウンサンプルにも同じ方式が
                使われる。
            neck: 多スケール特徴融合neck（既定'none'で従来挙動と数値
                不変=モジュール自体が構築されない）。バックボーンは
                stage1(stride2,k ch)/stage2(stride4,2k ch)/stage3
                (stride4,4k=feat_ch ch、既存の最終出力)の2ネイティブ解像度
                しか持たない（output_stride=4はtrain_detection.py全体で
                固定仮定されているため、neckでもこれを変えない）。
                'fpn': トップダウンでstage3をUpSample2dしstage1と
                concat+LutConv2dで融合(p1, stride2)、それをダウンサンプルで
                stride4へ戻しstage2/stage3とconcat+LutConv2dで最終特徴
                (feat_ch, stride4)を作る。'fpnpan': 'fpn'に加えp1から
                もう一度ボトムアップ精緻化する第2パスを追加(PAN風)。
            decoupled_head: 分類(hm)ヘッドと回帰(size,off)ヘッドの入力特徴
                ブランチを分離するか（既定False=共有feat=従来挙動）。
                Trueなら cls_branch/reg_branch という2本のLutConv2dで
                （neck適用後の）特徴を一度変換してから各ヘッドへ渡す。
            reg_bins: size/offをスカラー直接回帰でなく reg_bins 個のビン上の
                分布として出力し期待値を取る（DFL風、既定0=従来の
                スカラー回帰）。GT('size'/'offset')は
                detection_utils.encode_centernet_targets で既に [0,1) へ
                正規化済みのため、ビン中心は linspace(0,1,reg_bins) で汎用的に
                定義できる。head_type='float'との併用は不可（float回帰に
                ビン分布は不要なためValueError）。
            head_type: 検出ヘッドの型（既定'lut'で従来挙動と数値不変）。
                'float'はバックボーンはLutConv2dのまま検出ヘッド3本のみ通常の
                nn.Conv2dにするハイブリッド構成（SpatialGroupSumの量子化読み出し
                [0,0.31]圧縮を回避する現実解）。IRエクスポート不可
                （scripts/export_webbench.pyが明示的にエラーを出す）。
        """
        super().__init__()

        if n_layers not in (3, 6, 8):
            raise ValueError(f'n_layers は 3, 6, 8 のいずれか: {n_layers}')
        if selection not in SELECTIONS:
            raise ValueError(f'selection は {SELECTIONS} のいずれか: {selection}')
        if downsample not in DOWNSAMPLES:
            raise ValueError(f'downsample は {DOWNSAMPLES} のいずれか: {downsample}')
        if neck not in NECKS:
            raise ValueError(f'neck は {NECKS} のいずれか: {neck}')
        if head_type not in HEAD_TYPES:
            raise ValueError(f'head_type は {HEAD_TYPES} のいずれか: {head_type}')
        if reg_bins < 0:
            raise ValueError(f'reg_bins は0以上: {reg_bins}')
        if reg_bins > 0 and head_type == 'float':
            raise ValueError(
                "reg_bins > 0 は head_type='lut' 専用です"
                "（floatヘッドの連続回帰にビン分布は不要）"
            )

        self.n_classes = n_classes
        self.k = k
        self.n_layers = n_layers
        self.hm_group = hm_group
        self.size_group = size_group
        self.offset_group = offset_group
        self.selection = selection
        self.downsample_mode = downsample
        self.neck = neck
        self.decoupled_head = decoupled_head
        self.reg_bins = reg_bins
        self.head_type = head_type

        conv_kw = dict(
            n_inputs=n_inputs, device=device, grad_factor=grad_factor,
            input_channels_per_tree=input_channels_per_tree,
            parameterization=parameterization,
            wiring=wiring,
        )
        if selection != 'soft':
            conv_kw['selection'] = selection
            conv_kw['tau_b'] = tau_b
            conv_kw['gumbel_scale'] = gumbel_scale

        if selection == 'dwn':
            # DWN は「ゲート木」ではなく1個の n入力RAMノードなので
            # tree_depth を取らない。木の葉数 n_inputs**tree_depth を
            # そのままノードの入力数にすると表が 2^(n^d) で爆発するため、
            # n_inputs をそのままノード入力数として使う（既定6 → 表64）。
            conv_kw = dict(
                n_inputs=dwn_n_inputs,
                input_channels_per_tree=input_channels_per_tree,
                efd=dwn_efd,
            )

            def conv(in_ch, out_ch, *, kernel_size=3, padding=1, stride=1):
                return DWNConv2d(in_ch, out_ch, kernel_size=kernel_size,
                                 padding=padding, stride=stride, **conv_kw)
        else:
            conv_cls = LutConv2d if selection == 'soft' else LutConv2dHardST

            def conv(in_ch, out_ch, *, kernel_size=3, padding=1, stride=1):
                return conv_cls(in_ch, out_ch, kernel_size=kernel_size,
                                tree_depth=tree_depth, padding=padding,
                                stride=stride, **conv_kw)

        def downsample_layer(ch):
            """OrPooling2d(2)（既定）または stride=2 の LutConv2d（学習可能
            ダウンサンプル）。ch→ch（チャンネル数は変えない）。"""
            if downsample == 'orpool':
                return OrPooling2d(2)
            return conv(ch, ch, kernel_size=2, padding=0, stride=2)

        # バックボーン: 出力ストライド = 4
        # 全Variantで共通の3ステージ構成
        # Stage 1: フル解像度 → /2
        # Stage 2: 半分解像度 → /4
        # Stage 3: 1/4解像度（検出ヘッドが動作）

        if n_layers == 3:
            # 元の構成: Conv→Pool→Conv→Pool→Conv
            self.stage1 = nn.ModuleList([conv(in_channels, k), downsample_layer(k)])
            self.stage2 = nn.ModuleList([conv(k, 2 * k), downsample_layer(2 * k)])
            self.stage3_pre = nn.ModuleList([conv(2 * k, 4 * k)])
            self.stage3_post = nn.ModuleList()

        elif n_layers == 6:
            # Stage2,3に同チャンネルConvを追加
            self.stage1 = nn.ModuleList([conv(in_channels, k), downsample_layer(k)])
            self.stage2 = nn.ModuleList([
                conv(k, 2 * k),
                conv(2 * k, 2 * k),
                downsample_layer(2 * k),
            ])
            self.stage3_pre = nn.ModuleList([
                conv(2 * k, 4 * k),
                conv(4 * k, 4 * k),
                conv(4 * k, 4 * k),
            ])
            self.stage3_post = nn.ModuleList()

        elif n_layers == 8:
            # 全ステージに層追加 + Stage3にOR Skip
            self.stage1 = nn.ModuleList([
                conv(in_channels, k),
                conv(k, k),
                downsample_layer(k),
            ])
            self.stage2 = nn.ModuleList([
                conv(k, 2 * k),
                conv(2 * k, 2 * k),
                downsample_layer(2 * k),
            ])
            # Stage3前半: チャンネル拡大
            self.stage3_pre = nn.ModuleList([
                conv(2 * k, 4 * k),
                conv(4 * k, 4 * k),
            ])
            # Stage3後半: OR Skipで接続
            self.stage3_post = nn.ModuleList([
                conv(4 * k, 4 * k),
                conv(4 * k, 4 * k),
            ])

        feat_ch = 4 * k

        # Neck（FPN/FPN+PAN）。バックボーンの2ネイティブ解像度
        # （stage1=stride2/k ch、stage2・stage3=stride4/2k・4k ch）を
        # UpSample2d+concat+LutConv2dのトップダウン経路で融合し、
        # 最終特徴を必ずstride4・feat_ch chへ揃える（ヘッド側は無改造）。
        if neck != 'none':
            self.neck_up = UpSample2d(2)
            self.neck_td = conv(feat_ch + k, k)               # stride2, k ch
            self.neck_down = downsample_layer(k)                # stride2→4, k ch
            self.neck_fuse = conv(k + 2 * k + feat_ch, feat_ch)  # stride4, feat_ch
            if neck == 'fpnpan':
                # PAN風ボトムアップ第2パス: p1を再度ダウンサンプルし最終特徴と融合
                self.neck_down2 = downsample_layer(k)
                self.neck_fuse2 = conv(feat_ch + k, feat_ch)

        if selection == 'dwn':
            def head_conv(in_ch, out_ch, *, residual_init=True):
                # DWN に residual_init（恒等ゲート寄り初期化）は無い。
                # 表は一様U(−1,1)初期化（公式実装準拠）。
                return DWNConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                 **conv_kw)
        else:
            head_conv_cls = LutConv2d if selection == 'soft' else LutConv2dHardST

            def head_conv(in_ch, out_ch, *, residual_init=True):
                return head_conv_cls(
                    in_ch, out_ch, kernel_size=3, tree_depth=head_depth,
                    padding=1, residual_init=residual_init, **conv_kw,
                )

        # decoupled head: 分類(hm)/回帰(size,off)で特徴ブランチを分離
        if decoupled_head:
            self.cls_branch = head_conv(feat_ch, feat_ch)
            self.reg_branch = head_conv(feat_ch, feat_ch)

        if head_type == 'float':
            # ハイブリッド: バックボーンはLutConv2dのまま、検出ヘッド
            # のみ通常のfloat nn.Conv2dにして連続ロジットを直接出す
            # （SpatialGroupSumの量子化読み出しを回避）。IRエクスポート不可。
            self.hm_conv = nn.Conv2d(feat_ch, n_classes, kernel_size=3, padding=1)
            self.size_conv = nn.Conv2d(feat_ch, 2, kernel_size=3, padding=1)
            self.off_conv = nn.Conv2d(feat_ch, 2, kernel_size=3, padding=1)
            self.hm_sum = None
            self.size_sum = None
            self.off_sum = None
        else:
            # ヒートマップヘッド: LutConv2d(or HardST) → SpatialGroupSum
            # LUT出力はhard eval(離散LUT)で0/1になり、単一chだとヒートマップが
            # 二値化してNMS/topKが全ピークで同点になる。hm_group個を平均して
            # 連続値[0,1]の確率に変換する（サイズ/オフセットヘッドと同じ構造）。
            self.hm_conv = head_conv(feat_ch, n_classes * hm_group, residual_init=False)
            self.hm_sum = SpatialGroupSum(n_classes, scale=1.0 / hm_group)

            if reg_bins > 0:
                # DFL風分布回帰: 各回帰値(w,h / dx,dy)を reg_bins 個の
                # ビン上のsoft分布として出力し、ビン中心の期待値を最終予測とする。
                # GTが既に[0,1)に正規化済みのため、ビン中心も[0,1)一様分割で足りる。
                self.size_conv = head_conv(feat_ch, 2 * reg_bins * size_group)
                self.size_sum = SpatialGroupSum(2 * reg_bins, scale=1.0 / size_group)
                self.off_conv = head_conv(feat_ch, 2 * reg_bins * offset_group)
                self.off_sum = SpatialGroupSum(2 * reg_bins, scale=1.0 / offset_group)
                bin_centers = (torch.arange(reg_bins, dtype=torch.float32) + 0.5) / reg_bins
                self.register_buffer('_bin_centers', bin_centers)
            else:
                # サイズヘッド: LutConv2d(or HardST) → SpatialGroupSum
                # 2 * size_group チャンネル → グループ合計で2チャンネル (w, h)
                self.size_conv = head_conv(feat_ch, 2 * size_group)
                self.size_sum = SpatialGroupSum(2, scale=1.0 / size_group)

                # オフセットヘッド: LutConv2d(or HardST) → SpatialGroupSum
                # 2 * offset_group チャンネル → グループ合計で2チャンネル (dx, dy)
                self.off_conv = head_conv(feat_ch, 2 * offset_group)
                self.off_sum = SpatialGroupSum(2, scale=1.0 / offset_group)

    def _backbone_forward(self, x: torch.Tensor):
        """バックボーンを流し、neckが使う中間特徴も返す。

        Returns: (c1, c2, c3) = (stage1出力/stride2/k ch,
                                  stage2出力/stride4/2k ch,
                                  stage3出力/stride4/4k=feat_ch ch)
        neck='none'時はc1/c2を使わない（呼び出しコストのみで数値には無関係）。
        """
        feat = x
        for layer in self.stage1:
            feat = layer(feat)
        c1 = feat
        for layer in self.stage2:
            feat = layer(feat)
        c2 = feat
        for layer in self.stage3_pre:
            feat = layer(feat)

        # Stage3後半（n_layers=8のときOR Skipを適用）
        if len(self.stage3_post) > 0:
            residual = feat
            for layer in self.stage3_post:
                feat = layer(feat)
            feat = torch.max(feat, residual)
        c3 = feat
        return c1, c2, c3

    def _neck_forward(self, c1: torch.Tensor, c2: torch.Tensor,
                      c3: torch.Tensor) -> torch.Tensor:
        """FPN(top-down)/FPN+PAN(top-down→bottom-up)で最終特徴(stride4, feat_ch)を作る。"""
        up = self.neck_up(c3)                                     # stride4→2, feat_ch
        p1 = self.neck_td(torch.cat([up, c1], dim=1))             # stride2, k ch
        down = self.neck_down(p1)                                  # stride2→4, k ch
        feat = self.neck_fuse(torch.cat([down, c2, c3], dim=1))   # stride4, feat_ch
        if self.neck == 'fpnpan':
            down2 = self.neck_down2(p1)                            # stride2→4, k ch
            feat = self.neck_fuse2(torch.cat([feat, down2], dim=1))  # stride4, feat_ch
        return feat

    def _regress(self, conv_layer, sum_layer, feat: torch.Tensor) -> torch.Tensor:
        """size/offヘッドの共通読み出し。reg_bins=0なら従来通りSpatialGroupSumの
        スカラー値をそのまま返す。reg_bins>0ならビン分布softmax→期待値(DFL風)。"""
        raw = sum_layer(conv_layer(feat))
        if self.reg_bins == 0:
            return raw
        B, _, H, W = raw.shape
        probs = F.softmax(raw.view(B, 2, self.reg_bins, H, W), dim=2)
        centers = self._bin_centers.view(1, 1, self.reg_bins, 1, 1)
        return (probs * centers).sum(dim=2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W) 入力画像 [0,1]

        Returns:
            'hm': (B, n_classes, H/4, W/4) ヒートマップ
            'size': (B, 2, H/4, W/4) 正規化サイズ
            'off': (B, 2, H/4, W/4) サブピクセルオフセット
        """
        c1, c2, c3 = self._backbone_forward(x)
        feat = self._neck_forward(c1, c2, c3) if self.neck != 'none' else c3

        if self.decoupled_head:
            cls_feat = self.cls_branch(feat)
            reg_feat = self.reg_branch(feat)
        else:
            cls_feat = reg_feat = feat

        if self.head_type == 'float':
            hm = torch.sigmoid(self.hm_conv(cls_feat))
            size = self.size_conv(reg_feat)
            off = self.off_conv(reg_feat)
        else:
            hm = self.hm_sum(self.hm_conv(cls_feat))
            size = self._regress(self.size_conv, self.size_sum, reg_feat)
            off = self._regress(self.off_conv, self.off_sum, reg_feat)

        return {'hm': hm, 'size': size, 'off': off}

    @property
    def num_gates(self) -> int:
        """総ゲート数（重み共有。空間展開前）。

        selection='dwn' の DWNConv2d は LutConv2d のサブクラスではないので
        明示的に含める。含め忘れると配信サイズの見積りが 0 になる。
        """
        return sum(
            m.num_gates for m in self.modules()
            if isinstance(m, LutConv2d | DWNConv2d)
        )
