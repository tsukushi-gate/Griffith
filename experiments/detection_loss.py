"""
CenterNet用損失関数。

ペナルティ付きフォーカルロス（ヒートマップ）+ L1ロス（回帰）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def centernet_focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    CenterNet式ペナルティ付きフォーカルロス。

    LUT出力は既に[0,1]なのでsigmoid不要。

    Args:
        pred: (B, C, H, W) 予測ヒートマップ [0,1]
        gt: (B, C, H, W) 正解ヒートマップ [0,1]
        alpha: フォーカルロスのべき指数
        beta: ペナルティ項のべき指数

    Returns:
        スカラー損失値
    """
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)

    # 正例（gt == 1の位置）
    pos_mask = gt.eq(1).float()
    # 負例（gt < 1の位置）
    neg_mask = gt.lt(1).float()

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -(
        ((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
    )

    n_pos = pos_mask.sum()
    if n_pos == 0:
        return neg_loss.sum()

    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def reg_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    正例位置でのL1回帰損失。

    Args:
        pred: (B, 2, H, W) 予測値
        gt: (B, max_objects, 2) 正解値
        mask: (B, max_objects) 有効オブジェクトマスク
        indices: (B, max_objects) ヒートマップ上のフラットインデックス

    Returns:
        スカラー損失値
    """
    B, _, H, W = pred.shape
    pred_flat = pred.reshape(B, 2, H * W)

    # 各オブジェクトの位置から予測値を取得
    indices_expanded = indices.unsqueeze(1).expand(B, 2, indices.shape[1])
    pred_at_pos = pred_flat.gather(2, indices_expanded)
    pred_at_pos = pred_at_pos.permute(0, 2, 1)

    # マスク付きL1
    mask_expanded = mask.unsqueeze(2).expand_as(gt)
    loss = torch.abs(pred_at_pos - gt) * mask_expanded

    n_pos = mask.sum()
    if n_pos == 0:
        return loss.sum()

    return loss.sum() / n_pos


def _gather_at_pos(pred: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """密予測マップ (B,2,H,W) からGTピーク位置の値を取り出す → (B, max_objects, 2)。

    reg_l1_loss と同じ gather を切り出したもの。teacher の size/off 予測を
    GT位置で拾い、KD の回帰ターゲット(teacher予測)として再利用する。
    """
    B, _, H, W = pred.shape
    pred_flat = pred.reshape(B, 2, H * W)
    idx = indices.unsqueeze(1).expand(B, 2, indices.shape[1])
    return pred_flat.gather(2, idx).permute(0, 2, 1)


def centernet_hm_kd_loss(
    student_hm: torch.Tensor,
    teacher_hm: torch.Tensor,
    temperature: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """ヒートマップKD損失: per-(channel,pixel) の二値(Bernoulli)蒸留。

    CenterNet のヒートマップは「クラス間 softmax」ではなく、各(クラス,位置)が
    独立した objectness 確率(Bernoulli p ∈ [0,1])である。したがって分類/セグ
    の softmax-over-classes KL(make_dense_kd_loss)はそのまま流用できず、各要素を
    独立した二値分布として扱う必要がある。

    teacher/student はいずれも確率(sigmoid / SpatialGroupSum平均)で出力される。
    温度 T を掛けるためロジット空間へ写す(z = logit(p))。tempered 確率
    q = sigmoid(z / T) を作り、二値KL

        KL(Bern(q_t) || Bern(q_s))
          = q_t·log(q_t/q_s) + (1-q_t)·log((1-q_t)/(1-q_s))

    を全 (B,C,H,W) で平均し、Hinton 2015 の soft項勾配スケール補正として T² を
    掛ける(分類版 kd_kl_loss と同一の補正意図)。

    二重温度(standardize)について: セグKD(make_dense_kd_loss)は GroupSum の
    τ_gs がロジットを軟化するため standardize が必須だった。しかし CenterNet の
    ヒートマップヘッド(SpatialGroupSum, scale=1/hm_group)は**τを持たない単純な
    平均**で、出力は確率そのものである。z=logit(p) 変換後に掛かる軟化は KD 温度 T
    のみ(二重温度は発生しない)。かつ各チャンネルは独立二値なので、クラス次元
    Z-score 標準化は 2要素[0,z]に対して退化し magnitude 情報を壊す。よって本損失は
    standardize を行わない(セグKDとの設計上の差異)。

    Args:
        student_hm: (B, C, H, W) student ヒートマップ確率 [0,1]。
        teacher_hm: (B, C, H, W) teacher ヒートマップ確率 [0,1]。
        temperature: KD温度 T(>0)。
        eps: 確率クランプの下限/上限マージン。

    Returns:
        スカラー(全要素平均)。teacher==student のとき 0、常に非負。
    """
    T = float(temperature)
    ps = student_hm.clamp(eps, 1.0 - eps)
    pt = teacher_hm.clamp(eps, 1.0 - eps)
    # ロジット変換(logit(p) = log(p) - log(1-p))
    zs = torch.log(ps) - torch.log1p(-ps)
    zt = torch.log(pt) - torch.log1p(-pt)
    # tempered 確率のログを logsigmoid で数値安定に計算(clampより勾配が素直)
    log_qs = F.logsigmoid(zs / T)
    log_1mqs = F.logsigmoid(-zs / T)
    log_qt = F.logsigmoid(zt / T)
    log_1mqt = F.logsigmoid(-zt / T)
    qt = torch.sigmoid(zt / T)
    kl = qt * (log_qt - log_qs) + (1.0 - qt) * (log_1mqt - log_1mqs)
    return (kl * (T * T)).mean()


class CenterNetLoss(nn.Module):
    """
    CenterNet複合損失。

    L = L_heatmap + lambda_size * L_size + lambda_off * L_offset
    """

    def __init__(self, lambda_size: float = 0.1, lambda_off: float = 1.0):
        super().__init__()
        self.lambda_size = lambda_size
        self.lambda_off = lambda_off

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            outputs: モデル出力
                'hm': (B, n_classes, H, W)
                'size': (B, 2, H, W)
                'off': (B, 2, H, W)
            targets: 正解データ
                'heatmap': (B, n_classes, H, W)
                'size': (B, max_objects, 2)
                'offset': (B, max_objects, 2)
                'indices': (B, max_objects)
                'mask': (B, max_objects)

        Returns:
            (合計損失, 個別損失の辞書)
        """
        hm_loss = centernet_focal_loss(outputs['hm'], targets['heatmap'])
        size_loss = reg_l1_loss(
            outputs['size'], targets['size'],
            targets['mask'], targets['indices'],
        )
        off_loss = reg_l1_loss(
            outputs['off'], targets['offset'],
            targets['mask'], targets['indices'],
        )

        total = hm_loss + self.lambda_size * size_loss + self.lambda_off * off_loss

        loss_dict = {
            'hm_loss': hm_loss.item(),
            'size_loss': size_loss.item(),
            'off_loss': off_loss.item(),
            'total_loss': total.item(),
        }

        return total, loss_dict


class CenterNetKDLoss(nn.Module):
    """CenterNet 知識蒸留(KD)複合損失(W3-det、CNN teacher → LUT student)。

    各ヘッドで hard(GT)項と KD(teacher予測)項を α で混合する
    (model-conversion-kd スキル §2-3: 最適 α=0.25、離散論理 student には
    hard label の錨が要る):

        L_hm   = α·focal(student_hm, gt_hm) + (1-α)·hm_kd(student_hm, teacher_hm, T)
        L_size = α·L1(student_size, gt_size)@pos + (1-α)·L1(student_size, teacher_size)@pos
        L_off  = α·L1(student_off,  gt_off )@pos + (1-α)·L1(student_off,  teacher_off )@pos
        L = L_hm + λ_size·L_size + λ_off·L_off

    - ヒートマップKD: per-(channel,pixel) 二値KL(centernet_hm_kd_loss、上記docstring参照)。
      focal も hm_kd も「位置ごとに平均された objectness 損失」で同オーダーのため、
      追加のスケール係数無しで α が直接両者を釣り合わせる(seg版 make_dense_kd_loss の
      per-pixel-mean 正規化と同じ思想)。
    - 回帰KD: GTピーク位置(indices/mask)で teacher の size/off 予測を拾い、
      それを回帰ターゲットにした masked L1(hard項と同じマスク構造)。

    targets dict に teacher 予測を相乗りさせる(train_step がバッチ毎オンライン推論で
    注入する): 'teacher_hm' (B,C,H,W), 'teacher_size' (B,2,H,W), 'teacher_off' (B,2,H,W)。
    """

    def __init__(
        self,
        alpha: float = 0.25,
        temperature: float = 2.0,
        lambda_size: float = 0.1,
        lambda_off: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.lambda_size = lambda_size
        self.lambda_off = lambda_off

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        a = self.alpha
        mask = targets['mask']
        indices = targets['indices']

        # --- ヒートマップ: focal(hard) + 二値KL(KD) ---
        hm_hard = centernet_focal_loss(outputs['hm'], targets['heatmap'])
        hm_kd = centernet_hm_kd_loss(
            outputs['hm'], targets['teacher_hm'], temperature=self.temperature,
        )
        hm_loss = a * hm_hard + (1.0 - a) * hm_kd

        # --- サイズ: L1(GT) + L1(teacher予測) をGT位置で ---
        size_hard = reg_l1_loss(outputs['size'], targets['size'], mask, indices)
        size_kd = reg_l1_loss(
            outputs['size'], _gather_at_pos(targets['teacher_size'], indices),
            mask, indices,
        )
        size_loss = a * size_hard + (1.0 - a) * size_kd

        # --- オフセット: L1(GT) + L1(teacher予測) をGT位置で ---
        off_hard = reg_l1_loss(outputs['off'], targets['offset'], mask, indices)
        off_kd = reg_l1_loss(
            outputs['off'], _gather_at_pos(targets['teacher_off'], indices),
            mask, indices,
        )
        off_loss = a * off_hard + (1.0 - a) * off_kd

        total = hm_loss + self.lambda_size * size_loss + self.lambda_off * off_loss

        loss_dict = {
            'hm_loss': hm_loss.item(),
            'size_loss': size_loss.item(),
            'off_loss': off_loss.item(),
            'total_loss': total.item(),
            'hm_hard': hm_hard.item(),
            'hm_kd': hm_kd.item(),
        }

        return total, loss_dict
