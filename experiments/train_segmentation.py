"""セグメンテーション訓練スクリプト。

pet(Oxford-IIIT Pet、2クラス pet/bg)/ voc(Pascal VOC 2012、21クラス
background+20)の per-pixel 分類を3アーキで学習する:
- convdlgn:   SegConvDLGN(LutConv2d softmax16g、thermometer入力)— 本命
- unet:       SmallUNet(CNNベースライン、float入力)
- teacher-ft: torchvision事前学習セグ骨格(lraspp/deeplabv3/fcn_resnet50)
              fine-tune(KDのteacherになる)。--dataset voc かつ
              21クラス一致時は事前学習ヘッドをそのまま保持する(強teacher化)。

使用例:
    # Conv-DLGN scratch(GPU、Pet)
    uv run python experiments/train_segmentation.py --arch convdlgn -k 32 -e 60

    # VOC 21クラス CNNベースライン
    uv run python experiments/train_segmentation.py --dataset voc --arch unet \\
        --img-size 128 -e 40

    # VOC teacher fine-tune(事前学習ヘッド保持)
    uv run python experiments/train_segmentation.py --dataset voc \\
        --arch teacher-ft --teacher-backbone deeplabv3 -e 18

    # CPUスモーク
    WANDB_MODE=offline uv run python experiments/train_segmentation.py \\
        --arch convdlgn --epochs 1 --limit-samples 64 -bs 8 --device cpu
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import (
    load_partial_state_dict,
    save_checkpoint,
    setup_checkpoint_dir,
)
from data.segmentation_voc import VOC_SEG_CLASSES, VOC_SEG_N_CLASSES, load_voc_seg
from env_utils import load_dotenv
from seg_losses import SEG_LOSS_NAMES, build_seg_loss, compute_class_weights_from_loader
from segmentation_models import SegConvDLGN, TeacherSeg, build_model
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
)

from griffith_logic.convert.distill import make_dense_kd_loss
from griffith_logic.datasets.segmentation import (
    IGNORE_INDEX,
    load_oxford_pet_seg,
)

load_dotenv()

N_CLASSES = 2  # 後方互換の既定値(pet binary seg)。VOCは21(load_dataが解決)。


# ============================================================
# データ
# ============================================================

def load_data(args, kd: bool = False):
    """--dataset に応じて Pet(2クラス) / VOC(21クラス) を読み込む。

    Returns:
        (train_loader, test_loader, in_channels, n_classes, class_names)
    """
    encoding = 'thermometer' if args.arch == 'convdlgn' else 'float'
    if args.dataset == 'voc':
        train_set, test_set = load_voc_seg(
            root='./data-voc',
            img_size=args.img_size,
            encoding=encoding,
            n_thresholds=args.n_thresholds,
            download=False,
            augment=getattr(args, 'augment', False),
        )
        n_classes = VOC_SEG_N_CLASSES
        class_names = VOC_SEG_CLASSES
    else:
        train_set, test_set = load_oxford_pet_seg(
            root='./data-oxford-pet',
            img_size=args.img_size,
            encoding=encoding,
            n_thresholds=args.n_thresholds,
            border=args.border,
        )
        n_classes = 2
        class_names = ['background', 'pet']
    in_channels = train_set.in_channels

    if kd:
        # KD時は train のみ生float画像(teacher入力)も返す。test は2要素のまま
        # (evaluate/render_samples を無改造で通す)。
        train_set.return_raw = True

    if args.limit_samples is not None:
        n = args.limit_samples
        train_set = torch.utils.data.Subset(train_set, range(min(n, len(train_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(n, len(test_set))))

    use_pin = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin, drop_last=True, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin, drop_last=False, num_workers=args.num_workers,
    )
    return train_loader, test_loader, in_channels, n_classes, class_names


def load_kd_teacher(ckpt_path: str, device: str) -> TeacherSeg:
    """W3/W7b KD teacher を ckpt から復元する(no_grad/eval固定)。

    teacher-ft 時に保存した metadata['args'] から backbone/input_size/n_classes を
    解決して同一構造の TeacherSeg を組み、重みを読み込む(pretrained=Falseで
    骨格だけ用意→ckpt重みで上書き)。metadataに情報が無い旧ckptは
    lraspp/64直接推論/2クラス(Pet binary)とみなす。
    """
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    targs = (data.get('metadata', {}) or {}).get('args', {}) or {}
    backbone = targs.get('teacher_backbone', 'lraspp')
    input_size = targs.get('teacher_input_size', 0)  # 旧ckptは64直接推論(=0)
    n_classes = targs.get('n_classes', 2)  # 旧ckptはPet binary(2クラス)
    teacher = TeacherSeg(
        backbone=backbone, pretrained=False, input_size=input_size,
        n_classes=n_classes)
    teacher.load_state_dict(data['model_state_dict'])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f'KD teacher: backbone={backbone} input_size={input_size} '
          f'n_classes={n_classes} (best_metric={data.get("best_metric")})')
    return teacher


def build_lr_scheduler(optimizer, schedule: str, epochs: int):
    """--lr-schedule に応じたスケジューラを返す。

    'none'(既定)は None(スケジューラ未使用、既存挙動と同一)。
    'cosine' は torch.optim.lr_scheduler.CosineAnnealingLR(T_max=epochs)。
    引数名epochsは後方互換のため維持するが、実際にはtotal step数を渡す:
    --epochs時はepochs=エポック数(epoch単位でstep())、--n-iters時は
    epochs=イテレーション数(iteration単位でstep()、W13a-2)。warmupなし。
    """
    if schedule == 'none':
        return None
    if schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    raise ValueError(f'未対応 lr-schedule: {schedule} (対応: none, cosine)')


def parse_key_map(spec: str | None) -> list[tuple[str, str]] | None:
    """--init-key-map の "model側prefix=ckpt側prefix,..." 文字列をパースする
    (W13a-3、load_partial_state_dictのkey_map引数向け。train_detection.pyの
    parse_key_mapと同型)。"""
    if not spec:
        return None
    pairs = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        model_prefix, sep, ckpt_prefix = item.partition('=')
        if not sep:
            raise ValueError(
                f'--init-key-map の形式が不正です(model側=ckpt側): {item!r}'
            )
        pairs.append((model_prefix, ckpt_prefix))
    return pairs


def apply_init_from(model, init_from: str, init_key_map: str | None, device: str) -> dict | None:
    """--init-from 指定時、事前学習ckptから部分ロードする(W13a-3)。

    Returns:
        load_partial_state_dict()の結果dict、または未指定ならNone
    """
    if not init_from:
        return None
    print(f'--init-from: {init_from} から部分ロード')
    ckpt_data = torch.load(init_from, map_location=device, weights_only=False)
    ckpt_state = ckpt_data.get('model_state_dict', ckpt_data)
    key_map = parse_key_map(init_key_map)
    result = load_partial_state_dict(model, ckpt_state, key_map=key_map)
    print(f'  ロード率: {result["load_ratio"]:.1%} '
          f'({len(result["loaded_keys"])}/{result["total_keys"]}キー)')
    if result['skipped_shape_mismatch']:
        print(f'  形状不一致でスキップ: {len(result["skipped_shape_mismatch"])}キー')
    if result['skipped_missing']:
        print(f'  ckpt側に対応キー無しでスキップ: {len(result["skipped_missing"])}キー')
    return result


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ(複数epoch分を周回する)。

    train_detection.pyのload_nと同型(W13a-2)。epoch境界を跨いでも単純に
    loaderを再イテレートするため、epochモード(num_iterations=epochs*
    batches_per_epoch)でも従来の「1passずつのepoch」と等価になる。
    """
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


def build_selection_scheduler(args, model, num_iterations: int):
    """--selection/--tau-schedule に応じてτ_bスケジューラを構築する(W13a-1)。

    train_detection.pyのbuild_selection_schedulerと同型。selection='soft'
    またはarch!='convdlgn'ならNone。'hard_st'/'gumbel_st'で--tau-schedule
    未指定(None)のときは既定を自動選択する:
        hard_st   → 'none'(tau_b固定、--tau-b-init)
        gumbel_st → 'anneal'(hardgap知見によりアニール必須)
    """
    if args.arch != 'convdlgn' or args.selection == 'soft':
        return None
    schedule = args.tau_schedule
    if schedule is None:
        schedule = 'anneal' if args.selection == 'gumbel_st' else 'none'
    if schedule == 'none':
        return None
    if schedule == 'cage':
        from griffith_logic.conv_hard_st import CAGEConvScheduler
        return CAGEConvScheduler(
            model, tau_max=args.tau_b_init, tau_min=args.tau_b_min,
        )
    # schedule == 'anneal'
    from cage_scheduler import TauAnnealConvScheduler
    return TauAnnealConvScheduler(
        model, tau_0=args.tau_b_init, tau_min=args.tau_b_min,
        total_steps=num_iterations,
        anneal_noise=(args.selection == 'gumbel_st'),
        noise_0=args.gumbel_scale_init,
    )


# ============================================================
# 指標(mIoU / Dice / pixel acc / クラス別IoU / 境界IoU(boundary F1))
# ============================================================

def _boundary_mask(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """(H, W) long ラベル → (H, W) bool 境界マスク。

    4近傍のいずれかとラベルが異なる画素を境界とする(morphological gradient
    相当)。ignore画素自体は境界としてカウントしない(隣接クラスとの差分では
    境界扱いになりうるが、ignore画素自体はvalidマスクで除外する)。
    """
    valid = labels != ignore_index
    lbl = labels.clone()
    lbl[~valid] = -1  # ignoreは専用センチネルにして通常クラスと区別する
    b = torch.zeros_like(labels, dtype=torch.bool)
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    b[:, 1:] |= lbl[:, :-1] != lbl[:, 1:]
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    b[1:, :] |= lbl[:-1, :] != lbl[1:, :]
    return b & valid


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """(H, W) bool マスクを radius 画素だけ膨張する(max_pool2dベース)。"""
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    d = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return d.squeeze(0).squeeze(0) > 0.5


class SegMetrics:
    """混同行列ベースのセグメンテーション指標集計(ignore_index対応)。

    集計する指標:
        - miou: 全クラス平均IoU
        - iou_per_class / dice_per_class: クラス別IoU/Dice
        - dice: 全クラス平均Dice
        - pixel_acc: 全画素正解率(global)
        - mean_pixel_acc: クラス別再現率(recall)の平均(mAcc)
        - boundary_f1 / boundary_precision / boundary_recall: 境界IoU相当
          (boundary F1、GT/pred境界を`boundary_dilation`画素膨張してマッチング)
    """

    def __init__(self, n_classes: int = N_CLASSES, boundary_dilation: int = 2):
        self.n = n_classes
        self.boundary_dilation = boundary_dilation
        self.tp = torch.zeros(n_classes, dtype=torch.long)
        self.fp = torch.zeros(n_classes, dtype=torch.long)
        self.fn = torch.zeros(n_classes, dtype=torch.long)
        self.correct = 0
        self.total = 0
        # 境界F1用カウンタ(precision分子/分母、recall分子/分母)
        self.b_tp_prec = 0
        self.b_fp = 0
        self.b_tp_rec = 0
        self.b_fn = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred/target: (B, H, W) の long"""
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()

        # --- 境界指標(画像ごと、reshape前に計算) ---
        for b in range(pred_cpu.shape[0]):
            gt_b = _boundary_mask(target_cpu[b], IGNORE_INDEX)
            valid_b = target_cpu[b] != IGNORE_INDEX
            pd_b = _boundary_mask(pred_cpu[b], IGNORE_INDEX) & valid_b
            gt_dil = _dilate_mask(gt_b, self.boundary_dilation)
            pd_dil = _dilate_mask(pd_b, self.boundary_dilation)
            self.b_tp_prec += (pd_b & gt_dil).sum().item()
            self.b_fp += (pd_b & ~gt_dil).sum().item()
            self.b_tp_rec += (gt_b & pd_dil).sum().item()
            self.b_fn += (gt_b & ~pd_dil).sum().item()

        pred_flat = pred_cpu.reshape(-1)
        target_flat = target_cpu.reshape(-1)
        valid = target_flat != IGNORE_INDEX
        pred_flat, target_flat = pred_flat[valid], target_flat[valid]
        self.correct += (pred_flat == target_flat).sum().item()
        self.total += target_flat.numel()
        for c in range(self.n):
            p, t = pred_flat == c, target_flat == c
            self.tp[c] += (p & t).sum()
            self.fp[c] += (p & ~t).sum()
            self.fn[c] += (~p & t).sum()

    def compute(self) -> dict:
        tp, fp, fn = self.tp.float(), self.fp.float(), self.fn.float()
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom.clamp(min=1), torch.ones_like(tp))
        dice_denom = 2 * tp + fp + fn
        dice_c = torch.where(dice_denom > 0, 2 * tp / dice_denom.clamp(min=1),
                              torch.ones_like(tp))
        acc_denom = tp + fn
        class_acc = torch.where(acc_denom > 0, tp / acc_denom.clamp(min=1),
                                 torch.ones_like(tp))

        eps = 1e-8
        b_precision = self.b_tp_prec / max(self.b_tp_prec + self.b_fp, eps)
        b_recall = self.b_tp_rec / max(self.b_tp_rec + self.b_fn, eps)
        b_f1 = (2 * b_precision * b_recall / (b_precision + b_recall)
                if (b_precision + b_recall) > 0 else 0.0)

        result = {
            'miou': iou.mean().item(),
            'dice': dice_c.mean().item(),
            'pixel_acc': self.correct / max(self.total, 1),
            'mean_pixel_acc': class_acc.mean().item(),
            'iou_per_class': iou.tolist(),
            'dice_per_class': dice_c.tolist(),
            'iou_bg': iou[0].item(),
            'boundary_f1': b_f1,
            'boundary_precision': b_precision,
            'boundary_recall': b_recall,
        }
        if self.n == 2:
            # 後方互換: binary seg(pet)の旧キー
            result['iou_pet'] = iou[1].item()
        return result


@torch.no_grad()
def evaluate(model, loader, device, mode=False, desc=None, n_classes=N_CLASSES,
             round_input=False):
    """mode=True: soft(訓練モード=連続緩和)、False: hard(離散LUT)。

    round_input: Trueのとき mode=False(hard)で入力を{0,1}へ丸める。
    thermometer入力のconvdlgnのみ意味を持つ(丸めは冪等)。unet/teacher-ftは
    連続float RGB画像が入力なので丸めると画像が0/1に二値化(posterize)され
    破壊的(過去の修正で発覚: teacher hard_miouが0.11까지落ちるバグの原因)。
    呼び出し側で arch=convdlgn のときだけ True を渡すこと。
    """
    orig_mode = model.training
    model.train(mode=mode)
    metrics = SegMetrics(n_classes=n_classes)
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    for x, y in it:
        x = x.to(torch.float32).to(device)
        y = y.to(device)
        if not mode and round_input:
            x = x.round()  # hard評価は{0,1}入力(thermometerでは冪等)
        logits = model(x)
        metrics.update(logits.argmax(dim=1), y)
    model.train(mode=orig_mode)
    return metrics.compute()


# ============================================================
# 予測サンプル画像
# ============================================================

def _voc_colormap(n: int) -> np.ndarray:
    """Pascal VOC標準カラーマップ(bit-interleave方式)を n クラス分生成する。

    index=0は黒(background)、index 1..20 はVOCの標準色(devkit互換)。
    n>21でも同一アルゴリズムで一意な色を割り当てる(VOC以外/binary segにも
    汎用的に使える)。
    """
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


@torch.no_grad()
def render_samples(model, loader, device, n_samples=4, thermometer=False,
                   n_classes=N_CLASSES):
    """入力/GT/予測を横並びにしたPIL画像リストを返す(多クラスパレット描画)。"""
    from PIL import Image
    orig_mode = model.training
    model.eval()
    x, y = next(iter(loader))
    x = x.to(torch.float32).to(device)
    x_in = x.round() if thermometer else x  # 連続float(unet/teacher-ft)は丸めない
    pred = model(x_in).argmax(dim=1).cpu()
    model.train(mode=orig_mode)

    cmap = _voc_colormap(max(n_classes, 2))
    ignore_color = np.array([255, 255, 255], dtype=np.uint8)  # void可視化用(白)

    images = []
    for i in range(min(n_samples, x.shape[0])):
        xi = x[i].cpu()
        if thermometer:
            # (3*n_th, H, W) → 閾値ビット平均でRGB近似復元
            n_th = xi.shape[0] // 3
            rgb = xi.reshape(n_th, 3, *xi.shape[1:]).mean(dim=0)
        else:
            rgb = xi
        rgb = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

        gt_idx = y[i].cpu().numpy()
        pd_idx = pred[i].numpy()
        gt3 = cmap[np.clip(gt_idx, 0, n_classes - 1)].copy()
        gt3[gt_idx == IGNORE_INDEX] = ignore_color
        pd3 = cmap[np.clip(pd_idx, 0, n_classes - 1)]

        panel = np.concatenate([rgb, gt3, pd3], axis=1)
        images.append(Image.fromarray(panel))
    return images


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='セグメンテーション訓練(Pet binary / VOC 21クラス)')
    parser.add_argument('--dataset', type=str, default='pet',
                        choices=['pet', 'voc'],
                        help='pet=Oxford-IIIT Pet(2クラス、既定) / '
                             'voc=Pascal VOC 2012(21クラス)')
    parser.add_argument('--arch', type=str, required=True,
                        choices=['convdlgn', 'unet', 'teacher-ft'])
    parser.add_argument('--img-size', type=int, default=64,
                        choices=[64, 96, 128, 160])
    parser.add_argument('--n-thresholds', type=int, default=3,
                        help='thermometer閾値数(convdlgnのみ、default: 3)')
    parser.add_argument('--border', type=str, default='foreground',
                        choices=['foreground', 'background', 'ignore'],
                        help='trimap境界画素の扱い(--dataset pet専用、'
                             'default: foreground)')
    parser.add_argument('-k', type=int, default=32,
                        help='SegConvDLGN幅パラメータ (default: 32)')
    parser.add_argument('-d', '--tree-depth', type=int, default=3)
    parser.add_argument('--readout-group', type=int, default=32,
                        help='per-pixel読み出しのクラスあたりチャネル数')
    parser.add_argument('--input-channels-per-tree', type=int, default=4)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='1ゲートの入力数 (default: 2)。検出では n=6 が'
                             '唯一効いた軸だった。n>2 では --parameterization '
                             'independent_dwn が必須')
    parser.add_argument('--parameterization', type=str, default='softmax16g',
                        choices=['independent_dwn', 'independent_ste',
                                 'softmax16g'],
                        help='パラメータ化方式 (default: softmax16g、n=2専用)。'
                             'n>2 では independent_dwn を使う')
    parser.add_argument('--wiring', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='LutConv2dの入力配線方式 (default: fixed。'
                             'learnableでDWN方式の配線学習を有効化)')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='訓練時のゲート選択方式(--arch convdlgn専用、'
                             'W13a-1)。soft=既定(従来挙動と数値不変)、'
                             'hard_st=CAGE論文方式(selection gap 0)、'
                             'gumbel_st=Mind the Gap方式(要アニール、'
                             '--tau-schedule anneal自動適用)')
    parser.add_argument('--tau-b-init', type=float, default=1.0,
                        help='selection!=soft時のHard-ST逆伝播温度初期値'
                             '(default: 1.0)')
    parser.add_argument('--tau-b-min', type=float, default=0.1,
                        help='τ_bスケジューラの下限(anneal/cage共通、default: 0.1)')
    parser.add_argument('--gumbel-scale-init', type=float, default=1.0,
                        help='selection=gumbel_st時のGumbelノイズ振幅初期値'
                             '(default: 1.0)')
    parser.add_argument('--tau-schedule', type=str, default=None,
                        choices=['none', 'anneal', 'cage'],
                        help='τ_bスケジューラ(default: Noneで選択に応じ自動'
                             '決定。hard_st→none(固定tau_b)、gumbel_st→anneal'
                             '(hardgap知見によりアニール必須))')
    parser.add_argument('--tau', type=float, default=None,
                        help='SpatialGroupSum温度(Noneで√readout_group)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Noneでアーキ既定(convdlgn 0.02 / unet 1e-3 / '
                             'teacher-ft 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Noneでアーキ既定(convdlgn 0.002 / 他 1e-4)')
    parser.add_argument('--lr-schedule', type=str, default='none',
                        choices=['none', 'cosine'],
                        help='学習率スケジュール(既定none=固定lr。cosineは'
                             'CosineAnnealingLR、epoch単位、warmupなし)')
    parser.add_argument('--augment', action='store_true',
                        help='VOCのtrain分割にデータ拡張(scale/crop/flip/'
                             'colorjitter)を適用する(既定off、'
                             '--dataset voc専用)')
    parser.add_argument('--seg-loss', type=str, default='ce',
                        choices=list(SEG_LOSS_NAMES),
                        help='損失関数(既定ce=CrossEntropyLoss、後方互換。'
                             'weighted_ce/focal/dice/focal_dice は'
                             'VOC 21クラス背景collapse対策)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='focal loss のgamma(既定2.0、'
                             '--seg-loss focal/focal_dice)')
    parser.add_argument('--focal-dice-weight', type=float, default=0.5,
                        help='focal_dice内のdice項重み(既定0.5、'
                             'focal項の重みは1-this)')
    parser.add_argument('--class-weight-batches', type=int, default=None,
                        help='クラス重み(median-frequency balancing)計算に'
                             '使うtrainバッチ数上限(既定None=全バッチ走査)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数(既定40。--n-iters/-niと排他、W13a-2)')
    parser.add_argument('-ni', '--n-iters', type=int, default=None,
                        help='反復回数(--epochsと排他、W13a-2)。指定時は'
                             'イテレーションループで学習し、--lr-schedule '
                             'cosineもこのステップ数(T_max)基準で減衰する'
                             '(検出train_detection.pyの-ni/--num-iterations'
                             'と同型)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='--n-iters指定時の評価頻度(イテレーション単位、'
                             '既定: 1エポック相当=batches_per_epoch)')
    parser.add_argument('--grad-factor', type=float, default=1.0)
    parser.add_argument('--limit-samples', type=int, default=None,
                        help='スモーク用: train/testをN件に制限')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--init-from', type=str, default=None,
                        help='事前学習ckpt(例: pretrain_cifar_encoder.py出力)'
                             'から部分ロードして初期化する(W13a-3、--arch '
                             'convdlgn専用)。形状一致キーのみ重みをコピーする'
                             '(load_partial_state_dict)')
    parser.add_argument('--init-key-map', type=str, default=None,
                        help='--init-from用のキー名prefixリマップ。'
                             '"model側prefix=ckpt側prefix" をカンマ区切りで'
                             '複数指定可(例: "c1=encoder.c1,c2=encoder.c2")。'
                             '省略時はキー名がそのまま一致する前提(素通し)')
    # --- teacher(強teacher: 高解像度推論ラッパ) ---
    parser.add_argument('--teacher-backbone', type=str, default='lraspp',
                        choices=['lraspp', 'deeplabv3', 'fcn_resnet50'],
                        help='teacher-ft の骨格(既定 lraspp)')
    parser.add_argument('--teacher-input-size', type=int, default=224,
                        help='teacher内部推論解像度。入力をここへupし推論→student'
                             '格子へdown(既定 224、0で直接推論)')
    # --- KD(密予測知識蒸留、arch=convdlgn) ---
    parser.add_argument('--distill-teacher-ckpt', type=str, default=None,
                        help='強teacherのckpt。指定時 convdlgn を密予測KDで学習')
    parser.add_argument('--distill-alpha', type=float, default=0.25,
                        help='hard(CE)項の重み(既定0.25、スキル§2-3)')
    parser.add_argument('--distill-T', type=float, default=2.0,
                        help='KD温度T(既定2.0。バイナリはT=2がT=4を圧倒しうる§8.3-1)')
    parser.add_argument('--distill-pseudo-labels', action='store_true',
                        help='hard項GTをteacher argmaxに置換(ラベルフリー§2-4)')
    parser.add_argument('--distill-standardize',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='KD項ロジットをクラス次元Z-score標準化(既定on、§2-2)')
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # アーキ既定ハイパラ
    if args.lr is None:
        args.lr = {'convdlgn': 0.02, 'unet': 1e-3, 'teacher-ft': 1e-4}[args.arch]
    if args.weight_decay is None:
        args.weight_decay = {'convdlgn': 0.002, 'unet': 1e-4,
                             'teacher-ft': 1e-4}[args.arch]

    if args.augment and args.dataset != 'voc':
        print('警告: --augment は --dataset voc 専用です(petでは無視されます)')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # KD判定(convdlgn かつ teacher ckpt 指定時)
    is_kd = args.arch == 'convdlgn' and args.distill_teacher_ckpt is not None
    if args.distill_teacher_ckpt is not None and args.arch != 'convdlgn':
        raise SystemExit('--distill-teacher-ckpt は --arch convdlgn 専用です')
    if args.selection != 'soft' and args.arch != 'convdlgn':
        raise SystemExit('--selection hard_st/gumbel_st は --arch convdlgn 専用です')
    if args.init_from and args.arch != 'convdlgn':
        raise SystemExit('--init-from は --arch convdlgn 専用です')

    # エポック / イテレーション の解決(train_detection.py -ni と同型、W13a-2)
    use_epochs = args.n_iters is None
    if use_epochs and args.epochs is None:
        args.epochs = 40  # 従来既定

    # データ
    train_loader, test_loader, in_channels, n_classes, class_names = load_data(
        args, kd=is_kd)
    args.n_classes = n_classes  # ckpt_metadata(vars(args))経由でload_kd_teacherが参照
    dataset_label = ('Oxford-IIIT Pet binary seg' if args.dataset == 'pet'
                     else f'Pascal VOC 2012 seg ({n_classes}クラス)')
    print(f'データセット: {dataset_label} (img={args.img_size}, '
          f'border={args.border if args.dataset == "pet" else "n/a"})')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル, '
          f'入力ch: {in_channels}, クラス数: {n_classes}')

    # モデル
    model_kw = {}
    if args.arch == 'convdlgn':
        model_kw = dict(k=args.k, tree_depth=args.tree_depth,
                        readout_group=args.readout_group,
                        input_channels_per_tree=args.input_channels_per_tree,
                        tau=args.tau, device=args.device,
                        grad_factor=args.grad_factor,
                        wiring=getattr(args, 'wiring', 'fixed'),
                        selection=args.selection,
                        n_inputs=args.n_inputs,
                        parameterization=args.parameterization,
                        tau_b=args.tau_b_init,
                        gumbel_scale=args.gumbel_scale_init)
    elif args.arch == 'teacher-ft':
        model_kw = dict(teacher_backbone=args.teacher_backbone,
                        teacher_input_size=args.teacher_input_size)
    model = build_model(
        args.arch, in_channels, n_classes=n_classes, **model_kw
    ).to(args.device)

    # 事前学習ckptからの部分ロード初期化(W13a-3)
    init_result = apply_init_from(model, args.init_from, args.init_key_map, args.device)
    if init_result is not None:
        args.init_from_load_ratio = init_result['load_ratio']

    # KD teacher(convdlgn + ckpt指定時のみ)
    teacher = None
    if is_kd:
        teacher = load_kd_teacher(args.distill_teacher_ckpt, args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nアーキ: {args.arch}')
    print(f'総パラメータ数: {total_params:,}')
    if isinstance(model, SegConvDLGN):
        # n_inputs/parameterization は引数化されたので固定文字列にしない
        # (LUT2/softmax16g 決め打ちの表示は n=6 の実験で誤読を招いた)
        print(f'総ロジックゲート数: {model.num_gates:,} '
              f'(LUT{args.n_inputs}, {args.parameterization})')
        print(f'読み出し: group={args.readout_group}/クラス, τ={model.tau:.2f}')
        print(f'選択方式: {args.selection}'
              + (f' (τ_b={args.tau_b_init}, gumbel_scale={args.gumbel_scale_init})'
                 if args.selection != 'soft' else ''))
    print(model)

    # エポック / イテレーション数からlrスケジューラ総ステップ数を解決(W13a-2)
    batches_per_epoch = len(train_loader)
    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        eval_freq = batches_per_epoch  # 1エポックごと(従来と同一挙動)
        sched_total = args.epochs
    else:
        num_iterations = args.n_iters
        eval_freq = args.eval_freq if args.eval_freq is not None else batches_per_epoch
        sched_total = num_iterations

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_schedule, sched_total)
    if args.lr_schedule != 'none':
        print(f'lrスケジュール: {args.lr_schedule}')
    selection_scheduler = build_selection_scheduler(args, model, num_iterations)
    if selection_scheduler is not None:
        print(f'τ_bスケジューラ: {type(selection_scheduler).__name__} '
              f'(selection={args.selection})')
    # 不均衡対策損失: weighted_ce/focal/focal_dice はmedian-frequency
    # balancingのクラス重みが必要(train分割を走査、ignore画素は除外)。
    needs_class_weights = args.seg_loss in ('weighted_ce', 'focal', 'focal_dice')
    class_weights = None
    if needs_class_weights:
        print(f'クラス重み計算中(median-frequency balancing, '
              f'--seg-loss {args.seg_loss})...')
        class_weights = compute_class_weights_from_loader(
            train_loader, n_classes, ignore_index=IGNORE_INDEX,
            max_batches=args.class_weight_batches)
        print(f'クラス重み: {[round(w, 4) for w in class_weights.tolist()]}')
    seg_loss_fn = build_seg_loss(
        args.seg_loss, n_classes, ignore_index=IGNORE_INDEX,
        class_weights=class_weights, gamma=args.focal_gamma,
        focal_weight=1.0 - args.focal_dice_weight,
        dice_weight=args.focal_dice_weight,
    )
    if args.seg_loss != 'ce':
        print(f'seg-loss: {args.seg_loss} (gamma={args.focal_gamma}, '
              f'focal_dice_weight={args.focal_dice_weight})')

    if is_kd:
        # seg_loss='ce'(既定)は従来通りhard_loss_fn=None(make_dense_kd_loss内の
        # F.cross_entropy直呼び)にして後方互換を保つ。不均衡対策損失指定時のみ
        # hard(CE)項を差し替える(soft KL項はそのまま)。
        kd_hard_loss_fn = None if args.seg_loss == 'ce' else seg_loss_fn
        kd_loss_fn = make_dense_kd_loss(
            alpha=args.distill_alpha, temperature=args.distill_T,
            standardize=args.distill_standardize, ignore_index=IGNORE_INDEX,
            pseudo_labels=args.distill_pseudo_labels, kind='kl',
            hard_loss_fn=kd_hard_loss_fn,
        )
        print(f'KD: alpha={args.distill_alpha} T={args.distill_T} '
              f'standardize={args.distill_standardize} '
              f'pseudo_labels={args.distill_pseudo_labels} '
              f'seg_loss={args.seg_loss}')
    else:
        ce_loss_fn = seg_loss_fn

    # チェックポイント
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'seg_{args.arch}', base_dir=args.checkpoint_dir)
    print(f'チェックポイント保存先: {run_dir}/')
    log_path = os.path.join(run_dir, 'log.txt')
    log_file = open(log_path, 'a', buffering=1)

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    # wandb
    ds_suffix = '_voc' if args.dataset == 'voc' else ''
    steps_tag = f'e{args.epochs}' if use_epochs else f'it{args.n_iters}'
    if args.arch == 'convdlgn':
        kd_suffix = ''
        if is_kd:
            kd_suffix = f'_kdT{args.distill_T:g}'
            if args.distill_alpha != 0.25:
                kd_suffix += f'_a{args.distill_alpha:g}'
            if not args.distill_standardize:
                kd_suffix += '_nostd'
            if args.distill_pseudo_labels:
                kd_suffix += '_pseudo'
        aug_suffix = '_aug' if args.augment else ''
        loss_suffix = f'_{args.seg_loss}' if args.seg_loss != 'ce' else ''
        sel_suffix = f'_{args.selection}' if args.selection != 'soft' else ''
        run_name = (f'seg_{args.arch}_k{args.k}_g{args.readout_group}'
                    f'_img{args.img_size}_{steps_tag}{kd_suffix}'
                    f'{aug_suffix}{loss_suffix}{sel_suffix}{ds_suffix}')
    else:
        aug_suffix = '_aug' if args.augment else ''
        loss_suffix = f'_{args.seg_loss}' if args.seg_loss != 'ce' else ''
        run_name = (f'seg_{args.arch}_img{args.img_size}_{steps_tag}'
                    f'{aug_suffix}{loss_suffix}{ds_suffix}')
    tags = ['w1seg', args.arch]
    if args.dataset == 'voc':
        tags.append('voc')
    if is_kd:
        tags.append('kd')
        if args.distill_pseudo_labels:
            tags.append('pseudo')
    else:
        tags.append(f'img{args.img_size}')
    if args.augment:
        tags.append('aug')
    if args.seg_loss != 'ce':
        tags.append(f'loss-{args.seg_loss}')
    if args.selection != 'soft':
        tags += ['w13', f'sel-{args.selection}']
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=tags,
        group='detseg-web',
    )

    is_dlgn = args.arch == 'convdlgn'
    ckpt_metadata = {'args': vars(args)}
    steps_desc = (f'epochs={args.epochs}' if use_epochs
                 else f'n_iters={args.n_iters}')
    log(f'START arch={args.arch} img={args.img_size} lr={args.lr} '
        f'wd={args.weight_decay} bs={args.batch_size} {steps_desc} '
        f'params={total_params} seed={args.seed} device={args.device}')

    if use_epochs:
        print(f'\n訓練開始: {args.epochs}エポック '
              f'({batches_per_epoch}バッチ/エポック, 評価=1エポックごと)')
    else:
        print(f'\n訓練開始: {num_iterations:,}イテレーション '
              f'(評価={eval_freq:,}イテレーションごと)')
    print('=' * 70)

    best_miou = 0.0
    train_start = time.time()

    # 統一イテレーションループ(epoch/n-itersどちらも同一ループで処理、
    # W13a-2)。epochモードはnum_iterations=epochs*batches_per_epoch、
    # eval_freq=batches_per_epochなので、evalトリガーはエポック境界に一致し
    # 従来の「1エポックごとに評価」と数値的に等価になる。
    interval_loss = 0.0
    interval_count = 0
    pbar = tqdm(
        enumerate(load_n(train_loader, num_iterations)),
        desc='train', total=num_iterations,
    )
    model.train()
    for i, batch in pbar:
        if is_kd:
            x, y, x_raw = batch
            x_raw = x_raw.to(torch.float32).to(args.device)
        else:
            x, y = batch
        x = x.to(torch.float32).to(args.device)
        y = y.to(args.device)
        model.train()
        logits = model(x)
        if is_kd:
            with torch.no_grad():
                teacher_logits = teacher(x_raw)  # 生float→teacher(no_grad)
            loss = kd_loss_fn(logits, y, teacher_logits)
        else:
            loss = ce_loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if selection_scheduler is not None:
            selection_scheduler.step()
        interval_loss += loss.item()
        interval_count += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')

        if use_epochs:
            # epoch単位(warmupなし、従来挙動と同一cadence)
            if (i + 1) % batches_per_epoch == 0 and lr_scheduler is not None:
                lr_scheduler.step()
        elif lr_scheduler is not None:
            lr_scheduler.step()  # iteration単位(W13a-2)

        if (i + 1) % eval_freq != 0 and (i + 1) != num_iterations:
            continue

        avg_loss = interval_loss / max(interval_count, 1)
        interval_loss = 0.0
        interval_count = 0
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_now = (i + 1) / batches_per_epoch
        step_label = int(round(epoch_now)) if use_epochs else i + 1

        # 評価: convdlgnはhard(離散LUT、thermometer入力を{0,1}丸め)とsoft
        # (連続緩和)の両方、他(unet/teacher-ft)は連続float画像なので丸めない
        # (丸めるとRGB画像がposterizeされ破壊的、teacher mIoUが
        # 見かけ上0.11まで落ちるバグだった)
        hard = evaluate(model, test_loader, args.device, mode=False,
                        desc=f'eval(hard) {step_label}', n_classes=n_classes,
                        round_input=is_dlgn)
        if is_dlgn:
            soft = evaluate(model, test_loader, args.device, mode=True,
                            desc=f'eval(soft) {step_label}', n_classes=n_classes)
        else:
            soft = hard  # BN更新を避けるためCNN系はsoft=hard扱い

        hard_miou, soft_miou = hard['miou'], soft['miou']
        gap = soft_miou - hard_miou

        is_best = hard_miou > best_miou
        if is_best:
            best_miou = hard_miou

        step_now = i + 1
        save_checkpoint(latest_path, model, optimizer, None,
                        step_now, best_miou, ckpt_metadata)
        if is_best:
            save_checkpoint(best_path, model, optimizer, None,
                            step_now, best_miou, ckpt_metadata)

        # 時間・ETA(JST)
        elapsed = time.time() - train_start
        remaining = elapsed / (i + 1) * (num_iterations - (i + 1))
        eta_str = (datetime.now(JST)
                   + timedelta(seconds=remaining)).strftime('%H:%M:%S')
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remain_str = str(timedelta(seconds=int(remaining)))

        step_desc = (f'epoch {step_label:>3d}/{args.epochs}' if use_epochs
                    else f'iter {step_label:,}/{num_iterations:,}')
        print(f'[{step_desc}] loss={avg_loss:.4f}  '
              f'lr={cur_lr:.5f}  '
              f'hard_miou={hard_miou:.4f}  soft_miou={soft_miou:.4f}  '
              f'gap={gap:+.4f}  dice={hard["dice"]:.4f}  '
              f'pix_acc={hard["pixel_acc"]:.4f}  '
              f'bF1={hard["boundary_f1"]:.4f}'
              f'{"  ★ BEST" if is_best else ""}'
              f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]')
        log(f'{step_desc} loss={avg_loss:.4f} '
            f'hard_miou={hard_miou:.4f} soft_miou={soft_miou:.4f} '
            f'gap={gap:.4f} dice={hard["dice"]:.4f} '
            f'pixel_acc={hard["pixel_acc"]:.4f} '
            f'mean_pixel_acc={hard["mean_pixel_acc"]:.4f} '
            f'boundary_f1={hard["boundary_f1"]:.4f} '
            f'best={str(is_best).lower()}')

        if use_wandb:
            wandb_metrics = {
                'epoch': epoch_now,
                'iteration': i + 1,
                'loss': avg_loss,
                'lr': cur_lr,
                'hard_miou': hard_miou,
                'soft_miou': soft_miou,
                'miou_gap': gap,
                'hard_iou_bg': hard['iou_bg'],
                'hard_dice': hard['dice'],
                'hard_pixel_acc': hard['pixel_acc'],
                'hard_mean_pixel_acc': hard['mean_pixel_acc'],
                'hard_boundary_f1': hard['boundary_f1'],
                'hard_boundary_precision': hard['boundary_precision'],
                'hard_boundary_recall': hard['boundary_recall'],
                'soft_dice': soft['dice'],
                'soft_boundary_f1': soft['boundary_f1'],
                'best_hard_miou': best_miou,
            }
            if selection_scheduler is not None:
                wandb_metrics['tau_b'] = selection_scheduler.tau_b
            if 'iou_pet' in hard:  # 後方互換(pet binary segのみ)
                wandb_metrics['hard_iou_pet'] = hard['iou_pet']
            # クラス別IoU/Dice(名前付き、最終ステップのみwandbへ詳細ログ)
            if (i + 1) == num_iterations:
                for ci, cname in enumerate(class_names):
                    wandb_metrics[f'final_iou_class_{ci}_{cname}'] = (
                        hard['iou_per_class'][ci])
                    wandb_metrics[f'final_dice_class_{ci}_{cname}'] = (
                        hard['dice_per_class'][ci])
            log_metrics(wandb_metrics)
            try:
                imgs = render_samples(model, test_loader, args.device,
                                      thermometer=is_dlgn, n_classes=n_classes)
                log_images('seg/samples', imgs,
                           captions=[f'{step_desc} 入力|GT|pred'] * len(imgs))
            except Exception as e:  # 画像ログ失敗は訓練を止めない
                print(f'  (log_images失敗: {e})')

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終 best hard mIoU: {best_miou:.4f}  (総訓練時間: {total_time})')
    print(f'BEST_CHECKPOINT={best_path}')
    log(f'DONE best_hard_miou={best_miou:.4f} total_time={total_time} '
        f'best_ckpt={best_path}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_hard_miou': best_miou},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
