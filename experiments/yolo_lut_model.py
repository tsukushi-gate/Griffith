"""
LogicYOLO: anchor-free YOLO型物体検出の論理ゲート(LUT)化。

CenterNet-LUT(detection_model.CenterNetLUT)がガウシアン中心回帰なのに対し、
本モデルはYOLOX型の **decoupled head**(cls / obj / reg を分岐)+
**anchor-free 単一正例割当**(GT中心セルのみを正例)を論理ゲートで実装する。
物体検出のYOLO系論理化は前例がない(D4)。

構造上の差分(CenterNet-LUT との比較):
  - ヘッド: CenterNetは hm(クラス統合ヒートマップ)/size/off の3分岐。
    本モデルは cls / obj / reg の decoupled head(objectnessを分離)。
  - 割当: CenterNetはガウシアンでソフト化した中心ヒートマップ。
    本モデルはYOLOX流の **ハード単一正例**(GT中心セル=1、他=0)。
  - 回帰: CenterNetは SpatialGroupSum の連続和を L1 回帰。
    本モデルは **離散ビン分類(distribution / 期待値デコード)**。

離散ビン回帰の根拠:
  座標(中心オフセット・サイズ)を各Kビンの分布として表現し、期待値でデコードする
  (Generalized Focal Loss / Distribution Focal Loss, Li et al. 2020 の考え方)。
  LUT出力は[0,1]に収まるので、K本のLUT出力を「非負のビン重み」とみなし
  総和で正規化すれば、そのまま確率分布になる(softmaxロジット化が不要で、
  [0,1]出力と相性が良い)。連続L1回帰よりも多峰・不確実性を表現でき、
  ハード離散化後も各ビンが{0,1}に崩れて分布のargmax=デコード値を保てる。
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from detection_loss import centernet_focal_loss

from griffith_logic import LutConv2d, OrPooling2d
from griffith_logic.spatial_group_sum import SpatialGroupSum

# 回帰ビンの値域(デコード/エンコードで共通利用)
OFFSET_RANGE = (0.0, 1.0)   # 中心セル内サブピクセルオフセット dx, dy ∈ [0, 1)
SIZE_RANGE = (0.0, 1.0)     # グリッドサイズで正規化した w, h ∈ (0, 1]
EPS = 1e-6


# ============================================================
# 離散ビン ⇔ 連続値 の変換(thermometer/DFL的な分布表現)
# ============================================================

def value_to_bin_dist(
    v: torch.Tensor, lo: float, hi: float, n_bins: int,
) -> torch.Tensor:
    """連続値を K ビンの分布に変換する(隣接2ビンへの線形分配)。

    v ∈ [lo, hi] を t = (v-lo)/(hi-lo)·(K-1) ∈ [0, K-1] に写像し、
    floor(t) と ceil(t) の2ビンへ線形補間の重みで分配する(DFL方式)。
    分布の期待値は t に一致するため、往復(encode→decode)で値を厳密復元できる。

    Args:
        v: (...) 連続値
        lo, hi: 値域
        n_bins: ビン数 K

    Returns:
        (..., K) 各行が総和1の分布
    """
    t = (v - lo) / (hi - lo) * (n_bins - 1)
    t = t.clamp(0.0, n_bins - 1)
    lower = t.floor().long()
    upper = (lower + 1).clamp(max=n_bins - 1)
    w_upper = t - lower.to(t.dtype)
    w_lower = 1.0 - w_upper

    dist = torch.zeros(*v.shape, n_bins, device=v.device, dtype=t.dtype)
    dist.scatter_(-1, lower.unsqueeze(-1), w_lower.unsqueeze(-1))
    # scatter_add で ceil 側を加算(lower==upper の端点でも安全に合算)
    dist.scatter_add_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))
    return dist


def bin_dist_to_value(
    dist: torch.Tensor, lo: float, hi: float, n_bins: int,
) -> torch.Tensor:
    """ビン分布を期待値でデコードして連続値に戻す。

    Args:
        dist: (..., K) 非負のビン重み(正規化前でも可)
        lo, hi: 値域
        n_bins: ビン数 K

    Returns:
        (...) 連続値
    """
    p = dist / (dist.sum(dim=-1, keepdim=True) + EPS)
    idx = torch.arange(n_bins, device=dist.device, dtype=p.dtype)
    expected = (p * idx).sum(dim=-1)  # 期待ビンインデックス ∈ [0, K-1]
    return lo + expected / (n_bins - 1) * (hi - lo)


# ============================================================
# モデル
# ============================================================

class YoloLutModel(nn.Module):
    """YOLOX型 anchor-free decoupled head の論理ゲート実装。

    バックボーン(出力ストライド4)+ 3分岐 decoupled head:
      - cls: LutConv2d → SpatialGroupSum → (B, n_classes, Hs, Ws) ∈ [0,1]
      - obj: LutConv2d → SpatialGroupSum → (B, 1, Hs, Ws) ∈ [0,1]
      - reg: LutConv2d → (B, 4*K, Hs, Ws) ∈ [0,1](各座標K本のビン重み)
    """

    N_COORDS = 4  # dx, dy, w, h

    def __init__(
        self,
        n_classes: int = 10,
        k: int = 64,
        in_channels: int = 3,
        tree_depth: int = 3,
        head_depth: int = 3,
        n_inputs: int = 2,
        n_bins: int = 16,
        cls_group: int = 8,
        obj_group: int = 8,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        input_channels_per_tree: int = 2,
        parameterization: str = 'softmax',
    ):
        """
        Args:
            n_classes: 検出クラス数
            k: バックボーン幅パラメータ
            in_channels: 入力チャンネル数(閾値二値化後)
            tree_depth: バックボーンのLUTツリー深さ
            head_depth: ヘッドのLUTツリー深さ
            n_inputs: ゲート入力数
            n_bins: 回帰ビン数 K
            cls_group: clsヘッドのグループサイズ
            obj_group: objヘッドのグループサイズ
            device: 計算デバイス
            grad_factor: 勾配スケーリング係数
            input_channels_per_tree: 各ツリーが参照する入力チャンネル数
            parameterization: パラメータ化方式
        """
        super().__init__()
        self.n_classes = n_classes
        self.k = k
        self.n_bins = n_bins
        self.cls_group = cls_group
        self.obj_group = obj_group
        self.output_stride = 4

        conv_kw = dict(
            n_inputs=n_inputs, device=device, grad_factor=grad_factor,
            input_channels_per_tree=input_channels_per_tree,
            parameterization=parameterization,
        )

        def conv(in_ch, out_ch, depth):
            return LutConv2d(in_ch, out_ch, kernel_size=3,
                             tree_depth=depth, padding=1, **conv_kw)

        # バックボーン(出力ストライド=4、CenterNet-LUT n_layers=3 と同構成)
        self.stage1 = nn.ModuleList([conv(in_channels, k, tree_depth), OrPooling2d(2)])
        self.stage2 = nn.ModuleList([conv(k, 2 * k, tree_depth), OrPooling2d(2)])
        self.stage3 = nn.ModuleList([conv(2 * k, 4 * k, tree_depth)])
        feat_ch = 4 * k

        # decoupled head
        # cls: グループ和で[0,1]のクラス確率(多ラベルBCE前提)
        self.cls_conv = conv(feat_ch, n_classes * cls_group, head_depth)
        self.cls_sum = SpatialGroupSum(n_classes, scale=1.0 / cls_group)
        # obj: グループ和で[0,1]のobjectness
        self.obj_conv = conv(feat_ch, obj_group, head_depth)
        self.obj_sum = SpatialGroupSum(1, scale=1.0 / obj_group)
        # reg: 4座標 × K ビンの重み(グループ和せず、各ビンをそのまま出力)
        self.reg_conv = conv(feat_ch, self.N_COORDS * n_bins, head_depth)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """バックボーン特徴を返す(hard離散性テスト用に公開)。"""
        feat = x
        for layer in self.stage1:
            feat = layer(feat)
        for layer in self.stage2:
            feat = layer(feat)
        for layer in self.stage3:
            feat = layer(feat)
        return feat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W) 入力 [0,1]

        Returns:
            'cls': (B, n_classes, Hs, Ws) クラス確率 [0,1]
            'obj': (B, 1, Hs, Ws) objectness [0,1]
            'reg': (B, 4, K, Hs, Ws) 各座標のビン重み [0,1]
        """
        feat = self.backbone(x)
        cls = self.cls_sum(self.cls_conv(feat))
        obj = self.obj_sum(self.obj_conv(feat))
        reg = self.reg_conv(feat)
        B, _, Hs, Ws = reg.shape
        reg = reg.view(B, self.N_COORDS, self.n_bins, Hs, Ws)
        return {'cls': cls, 'obj': obj, 'reg': reg}

    @property
    def num_gates(self) -> int:
        return sum(
            m.num_gates for m in self.modules() if isinstance(m, LutConv2d)
        )


# ============================================================
# ターゲット割当(anchor-free 単一正例)
# ============================================================

def build_targets(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    output_stride: int,
    n_classes: int,
    n_bins: int,
) -> dict[str, torch.Tensor]:
    """GTからYOLOX型の割当ターゲットを生成する(GT中心セルのみ正例)。

    Args:
        gt_boxes: (B, max_obj, 4) [x1, y1, x2, y2] ピクセル座標
        gt_labels: (B, max_obj) クラスラベル
        mask: (B, max_obj) 有効オブジェクトマスク
        grid_h, grid_w: グリッド(ヘッド出力)サイズ
        output_stride: 出力ストライド
        n_classes: クラス数
        n_bins: 回帰ビン数

    Returns:
        'obj': (B, 1, Hs, Ws) 0/1
        'cls': (B, n_classes, Hs, Ws) 0/1(正例セルの正解クラスのみ1)
        'reg': (B, 4, K, Hs, Ws) 正例セルのビン分布(他は0)
        'pos': (B, 1, Hs, Ws) 正例マスク(0/1)
    """
    B = gt_boxes.shape[0]
    device = gt_boxes.device
    s = output_stride

    obj_t = torch.zeros(B, 1, grid_h, grid_w, device=device)
    cls_t = torch.zeros(B, n_classes, grid_h, grid_w, device=device)
    reg_t = torch.zeros(B, 4, n_bins, grid_h, grid_w, device=device)
    pos_t = torch.zeros(B, 1, grid_h, grid_w, device=device)

    for b in range(B):
        for i in range(gt_boxes.shape[1]):
            if mask[b, i] <= 0:
                continue
            x1, y1, x2, y2 = gt_boxes[b, i].tolist()
            label = int(gt_labels[b, i].item())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            cx_g = cx / s
            cy_g = cy / s
            gx = min(int(cx_g), grid_w - 1)
            gy = min(int(cy_g), grid_h - 1)

            obj_t[b, 0, gy, gx] = 1.0
            pos_t[b, 0, gy, gx] = 1.0
            cls_t[b, :, gy, gx] = 0.0
            cls_t[b, label, gy, gx] = 1.0

            dx = torch.tensor(cx_g - gx, device=device)
            dy = torch.tensor(cy_g - gy, device=device)
            w_norm = torch.tensor((w / s) / grid_w, device=device)
            h_norm = torch.tensor((h / s) / grid_h, device=device)

            reg_t[b, 0, :, gy, gx] = value_to_bin_dist(dx, *OFFSET_RANGE, n_bins)
            reg_t[b, 1, :, gy, gx] = value_to_bin_dist(dy, *OFFSET_RANGE, n_bins)
            reg_t[b, 2, :, gy, gx] = value_to_bin_dist(w_norm, *SIZE_RANGE, n_bins)
            reg_t[b, 3, :, gy, gx] = value_to_bin_dist(h_norm, *SIZE_RANGE, n_bins)

    return {'obj': obj_t, 'cls': cls_t, 'reg': reg_t, 'pos': pos_t}


# ============================================================
# 損失
# ============================================================

def _masked_bce(pred, target, mask, eps=EPS):
    """正例セルのみで多ラベルBCEを計算(clsヘッド用)。"""
    pred = pred.clamp(eps, 1.0 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    loss = loss * mask  # (B,1,H,W) を (B,C,H,W) にブロードキャスト
    n_pos = mask.sum().clamp(min=1.0)
    return loss.sum() / n_pos


class YoloLutLoss(nn.Module):
    """LogicYOLO 複合損失。

    L = L_obj(focal) + λ_cls·L_cls(正例BCE) + λ_reg·L_reg(ビンCE)

    - obj: CenterNet流ペナルティ付きfocal(ハード0/1ターゲットでは標準focalに帰着)
    - cls: 正例セルのみの多ラベルBCE(YOLOX同様 objectness と分離)
    - reg: 正例セルのみ、各座標ビン分布のクロスエントロピー(DFL)
    """

    N_COORDS = 4

    def __init__(
        self,
        n_classes: int = 10,
        n_bins: int = 16,
        output_stride: int = 4,
        lambda_cls: float = 1.0,
        lambda_reg: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.output_stride = output_stride
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg

    def forward(self, outputs, targets):
        """
        Args:
            outputs: モデル出力(cls / obj / reg)
            targets: データセットのGT辞書('gt_boxes','gt_labels','mask')

        Returns:
            (合計損失, 個別損失の辞書)
        """
        obj_pred = outputs['obj']
        cls_pred = outputs['cls']
        reg_pred = outputs['reg']  # (B,4,K,Hs,Ws) ∈ [0,1]
        B, _, Hs, Ws = obj_pred.shape

        tgt = build_targets(
            targets['gt_boxes'], targets['gt_labels'], targets['mask'],
            Hs, Ws, self.output_stride, self.n_classes, self.n_bins,
        )

        obj_loss = centernet_focal_loss(obj_pred, tgt['obj'])
        cls_loss = _masked_bce(cls_pred, tgt['cls'], tgt['pos'])

        # reg: ビン分布のCE(正例セルのみ)
        p = reg_pred / (reg_pred.sum(dim=2, keepdim=True) + EPS)  # (B,4,K,H,W)
        ce = -(tgt['reg'] * torch.log(p + EPS)).sum(dim=2)  # (B,4,H,W)
        ce = ce * tgt['pos']  # (B,1,H,W) をブロードキャスト
        n_pos = tgt['pos'].sum().clamp(min=1.0)
        reg_loss = ce.sum() / (n_pos * self.N_COORDS)

        total = obj_loss + self.lambda_cls * cls_loss + self.lambda_reg * reg_loss

        loss_dict = {
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total.item(),
        }
        return total, loss_dict


# ============================================================
# デコード
# ============================================================

def decode_reg(reg: torch.Tensor) -> torch.Tensor:
    """回帰ビン分布 (B,4,K,H,W) を連続値 (B,4,H,W) にデコードする。

    返り値のチャンネルは [dx, dy, w_norm, h_norm]。
    """
    dx = bin_dist_to_value(reg[:, 0].permute(0, 2, 3, 1), *OFFSET_RANGE, reg.shape[2])
    dy = bin_dist_to_value(reg[:, 1].permute(0, 2, 3, 1), *OFFSET_RANGE, reg.shape[2])
    w = bin_dist_to_value(reg[:, 2].permute(0, 2, 3, 1), *SIZE_RANGE, reg.shape[2])
    h = bin_dist_to_value(reg[:, 3].permute(0, 2, 3, 1), *SIZE_RANGE, reg.shape[2])
    return torch.stack([dx, dy, w, h], dim=1)  # (B,4,H,W)


def decode_yolo(
    outputs: dict[str, torch.Tensor],
    output_stride: int = 4,
    top_k: int = 100,
    threshold: float = 0.1,
) -> list[dict]:
    """モデル出力からバウンディングボックスを復号する。

    スコア = obj × cls をクラス別に計算し、nms_by_maxpool でピーク抽出、
    topk・閾値でフィルタ後、reg期待値デコードで box を得る。
    detection_utils と互換の検出結果リストを返す。
    """
    from detection_utils import nms_by_maxpool

    obj = outputs['obj']    # (B,1,Hs,Ws)
    cls = outputs['cls']    # (B,C,Hs,Ws)
    reg = outputs['reg']    # (B,4,K,Hs,Ws)
    B, C, Hs, Ws = cls.shape

    score = obj * cls  # (B,C,Hs,Ws) ブロードキャスト
    score = nms_by_maxpool(score)
    reg_val = decode_reg(reg)  # (B,4,Hs,Ws)

    results = []
    for b in range(B):
        scores_flat = score[b].reshape(-1)
        K = min(top_k, scores_flat.shape[0])
        topk_scores, topk_inds = scores_flat.topk(K)
        keep = topk_scores > threshold
        topk_scores = topk_scores[keep]
        topk_inds = topk_inds[keep]

        if topk_inds.numel() == 0:
            results.append({
                'boxes': torch.zeros(0, 4, device=cls.device),
                'scores': torch.zeros(0, device=cls.device),
                'labels': torch.zeros(0, dtype=torch.long, device=cls.device),
            })
            continue

        classes = topk_inds // (Hs * Ws)
        spatial = topk_inds % (Hs * Ws)
        ys = spatial // Ws
        xs = spatial % Ws

        dx = reg_val[b, 0, ys, xs]
        dy = reg_val[b, 1, ys, xs]
        w_norm = reg_val[b, 2, ys, xs]
        h_norm = reg_val[b, 3, ys, xs]

        cx = (xs.float() + dx) * output_stride
        cy = (ys.float() + dy) * output_stride
        w_px = w_norm * output_stride * Ws
        h_px = h_norm * output_stride * Hs

        boxes = torch.stack([
            cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2,
        ], dim=-1)
        results.append({
            'boxes': boxes, 'scores': topk_scores, 'labels': classes,
        })
    return results
