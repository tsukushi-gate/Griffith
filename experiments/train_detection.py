"""
CenterNet + LUT 物体検出 訓練スクリプト

合成Multi-MNISTデータセットでCenterNet物体検出モデルを訓練する。

使用例:
    # 1オブジェクト（位置推定の動作確認）
    uv run python experiments/train_detection.py -k 64 --max-objects 1 -e 50

    # 3オブジェクト
    uv run python experiments/train_detection.py -k 64 --max-objects 3 -e 100

    # イテレーションモード
    uv run python experiments/train_detection.py -k 64 -ni 20000 -ef 2000

    # 評価のみ（重みを指定して可視化）
    uv run python experiments/train_detection.py --eval outputs/.../checkpoints/best.pt
"""

import argparse
import glob
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, load_partial_state_dict, save_checkpoint
from data.detection import MultiMNISTDetection, collate_detection
from detection_loss import CenterNetKDLoss, CenterNetLoss
from detection_model import CenterNetLUT
from detection_model_cnn import CenterNetCNN
from detection_utils import compute_map, decode_detections, visualize_detections
from ema import ParamEMA
from env_utils import load_dotenv
from run_utils import RunDir
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
    log_model_checkpoint,
)

from griffith_logic import LutConv2d

load_dotenv()


# ============================================================
# ユーティリティ
# ============================================================

def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


def parse_key_map(spec: str | None) -> list[tuple[str, str]] | None:
    """--init-key-map の "model側prefix=ckpt側prefix,..." 文字列をパースする
    (W13a-3、load_partial_state_dictのkey_map引数向け)。"""
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


def show_vis_gui(vis_images, title, n_show=4):
    """Mac GUI でバウンディングボックス可視化画像を表示する。"""
    try:
        import matplotlib.pyplot as plt
        n_show = min(n_show, len(vis_images))
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
        if n_show == 1:
            axes = [axes]
        for ax, img in zip(axes, vis_images[:n_show]):
            ax.imshow(img)
            ax.axis('off')
        fig.suptitle(title)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    except ImportError:
        pass


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, images, targets, loss_fn, optimizer,
               teacher=None, teacher_images=None, grad_clip: float = 0.0):
    """1ステップの訓練。

    teacher を渡すと（KD）teacher_images でバッチ毎オンライン推論（no_grad）し、
    その予測 hm/size/off を targets へ 'teacher_hm'/'teacher_size'/'teacher_off'
    として注入してから loss_fn（CenterNetKDLoss）を呼ぶ。

    grad_clip > 0 で勾配ノルムクリップ。離散argmax前向き（hard_st）は
    ステップ毎にゲート選択が跳ぶため勾配が突発的に大きくなる。同リポジトリでは
    LogicGPT で grad-clip 1.0 + warmup がアーキ順位を逆転させた実測がある
    が、検出パスには未適用だった。
    """
    if teacher is not None:
        with torch.no_grad():
            t_out = teacher(teacher_images)
        targets = {
            **targets,
            'teacher_hm': t_out['hm'],
            'teacher_size': t_out['size'],
            'teacher_off': t_out['off'],
        }
    outputs = model(images)
    loss, loss_dict = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip and grad_clip > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        loss_dict = {**loss_dict, 'grad_norm': float(total_norm)}
    optimizer.step()
    return loss_dict


def load_kd_teacher_detection(ckpt_glob: str, device: str):
    """W3-det KD teacher（CNN CenterNet）を ckpt から復元する（no_grad/eval固定）。

    ckpt_glob は glob パターン可（実行時解決、最新mtime を採用）。teacher arch は
    ckpt の metadata['args'] から自動判別する（'cnn' 前提）。in_channels は
    state_dict の先頭 conv 重み形状から推定する（multimnist teacher=n_thresholds,
    VOC teacher=3 RGB のいずれでも自動対応）。

    Raises:
        FileNotFoundError: glob が一致しない場合。
        ValueError: teacher arch が cnn でない場合。
    """
    matches = sorted(glob.glob(ckpt_glob), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f'KD teacher ckpt が見つかりません: {ckpt_glob}')
    ckpt_path = matches[-1]
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    targs = (data.get('metadata', {}) or {}).get('args', {}) or {}
    arch = targs.get('arch', 'cnn')
    if arch != 'cnn':
        raise ValueError(f'KD teacher は arch=cnn 前提（実際: {arch}）: {ckpt_path}')

    state = data['model_state_dict']
    # 先頭 conv 重み [out, in, 3, 3] から in_channels を推定
    in_ch = state['backbone.0.0.weight'].shape[1]
    n_classes = state['hm_head.1.bias'].shape[0]

    teacher = CenterNetCNN(
        n_classes=n_classes,
        in_channels=in_ch,
        width=targs.get('cnn_width', 64),
        head_ch=targs.get('cnn_head_ch', 64),
    )
    teacher.load_state_dict(state)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f'KD teacher: {ckpt_path}')
    print(f'  arch=cnn in_ch={in_ch} n_classes={n_classes} '
          f'width={targs.get("cnn_width", 64)} '
          f'best_metric={data.get("best_metric")}')
    return teacher


def evaluate(model, loader, device, loss_fn, mode=False, output_stride=4,
             desc=None, max_batches=None, n_vis=32, n_classes=10,
             binary_input=True):
    """
    評価ループ。mAPと損失を計算する。

    Args:
        model: モデル
        loader: DataLoader
        device: デバイス
        loss_fn: 損失関数
        mode: True=softモード, False=hardモード（離散化）
        binary_input: モデルの入力が本来0/1か。True(既定)のとき hardモードで
            round を掛ける。生RGBを消費するCNNでは False を渡すこと
        output_stride: 出力ストライド
        desc: プログレスバーの説明
        max_batches: 評価バッチ数の上限（Noneで全バッチ）
        n_vis: 可視化する画像数（複数バッチから収集）

    Returns:
        (mAP, avg_loss, vis_correct, vis_incorrect)
        vis_correct: 全予測正解の画像リスト
        vis_incorrect: 不正解を含む画像リスト
    """
    orig_mode = model.training
    model.train(mode=mode)

    all_pred = []
    all_gt = []
    total_loss = 0.0
    n_batches = 0
    vis_correct = []
    vis_incorrect = []
    vis_count = 0

    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for images, targets in it:
            images = images.to(torch.float32).to(device)
            targets_device = {
                k: v.to(device) for k, v in targets.items()
            }

            # hardモードでの round は「入力が本来0/1」であることの安全網。
            # LUT student(thermometer二値化入力)では no-op だが、**生RGBを食う
            # CNN では画素を0/1に潰して精度を破壊する**(発見)。
            # 実画像タスクの CNN teacher の値がすべて過小評価されていた。
            if not mode and binary_input:
                images = images.round()

            outputs = model(images)
            _, loss_dict = loss_fn(outputs, targets_device)
            total_loss += loss_dict['total_loss']
            n_batches += 1

            # bbox復号（CPU上でmAP計算するためcpuに移動）
            detections = decode_detections(
                outputs['hm'], outputs['size'], outputs['off'],
                output_stride=output_stride,
                threshold=0.1,
            )
            for det in detections:
                all_pred.append({
                    'boxes': det['boxes'].cpu(),
                    'scores': det['scores'].cpu(),
                    'labels': det['labels'].cpu(),
                })

            # GT変換
            B = images.shape[0]
            for b in range(B):
                mask_b = targets['mask'][b] > 0
                all_gt.append({
                    'boxes': targets['gt_boxes'][b][mask_b].cpu(),
                    'labels': targets['gt_labels'][b][mask_b].cpu(),
                })

            # 可視化画像を生成（n_visに達するまで複数バッチから収集）
            if vis_count < n_vis:
                for b in range(B):
                    if vis_count >= n_vis:
                        break
                    mask_b = targets['mask'][b] > 0
                    vis_img, is_all_correct = visualize_detections(
                        image=images[b].cpu(),
                        gt_boxes=targets['gt_boxes'][b][mask_b].cpu(),
                        gt_labels=targets['gt_labels'][b][mask_b].cpu(),
                        pred_boxes=detections[b]['boxes'].cpu(),
                        pred_scores=detections[b]['scores'].cpu(),
                        pred_labels=detections[b]['labels'].cpu(),
                    )
                    if is_all_correct:
                        vis_correct.append(vis_img)
                    else:
                        vis_incorrect.append(vis_img)
                    vis_count += 1

            if max_batches and n_batches >= max_batches:
                break

    # mAP計算
    mAP, _ = compute_map(all_pred, all_gt, n_classes=n_classes, iou_threshold=0.5)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    model.train(mode=orig_mode)
    return mAP, avg_loss, vis_correct, vis_incorrect


def handle_eval_results(run_dir, use_wandb, device, step_label,
                        correct_hard, incorrect_hard,
                        correct_soft, incorrect_soft):
    """評価後の可視化画像をローカル保存・wandbログ・GUI表示する。"""
    # ローカル保存
    if correct_hard:
        run_dir.save_visualizations(correct_hard, 'hard_correct', step_label)
    if incorrect_hard:
        run_dir.save_visualizations(incorrect_hard, 'hard_incorrect', step_label)
    if correct_soft:
        run_dir.save_visualizations(correct_soft, 'soft_correct', step_label)
    if incorrect_soft:
        run_dir.save_visualizations(incorrect_soft, 'soft_incorrect', step_label)

    # wandbログ（正解・不正解を分けてログ）
    if use_wandb:
        if incorrect_hard:
            log_images('detection/hard_incorrect', incorrect_hard)
        if correct_hard:
            log_images('detection/hard_correct', correct_hard)
        if incorrect_soft:
            log_images('detection/soft_incorrect', incorrect_soft)
        if correct_soft:
            log_images('detection/soft_correct', correct_soft)

    # Mac GUI表示（不正解を優先的に表示）
    if device == 'mps':
        if incorrect_hard:
            show_vis_gui(incorrect_hard,
                         f'Step {step_label} - Hard INCORRECT ({len(incorrect_hard)}枚)')
        if correct_hard:
            show_vis_gui(correct_hard,
                         f'Step {step_label} - Hard CORRECT ({len(correct_hard)}枚)')


# ============================================================
# モデル構築
# ============================================================

def build_model(args, in_ch, n_classes):
    """--arch に応じて LUT版 / CNN版 CenterNet を構築する。"""
    if args.arch == 'cnn':
        model = CenterNetCNN(
            n_classes=n_classes,
            in_channels=in_ch,
            width=args.cnn_width,
            head_ch=args.cnn_head_ch,
            downsample=getattr(args, 'cnn_downsample', 'stride'),
        )
    else:
        model = CenterNetLUT(
            n_classes=n_classes,
            k=args.k,
            in_channels=in_ch,
            tree_depth=args.tree_depth,
            head_depth=args.head_depth,
            n_inputs=args.n_inputs,
            hm_group=args.hm_group,
            size_group=args.size_group,
            offset_group=args.offset_group,
            device=args.device,
            grad_factor=args.grad_factor,
            input_channels_per_tree=args.input_channels_per_tree,
            parameterization=args.parameterization,
            n_layers=args.n_layers,
            wiring=getattr(args, 'wiring', 'fixed'),
            selection=getattr(args, 'selection', 'soft'),
            dwn_n_inputs=getattr(args, 'dwn_n_inputs', 6),
            dwn_efd=getattr(args, 'dwn_efd', 'paper'),
            tau_b=getattr(args, 'tau_b_init', 1.0),
            gumbel_scale=getattr(args, 'gumbel_scale_init', 1.0),
            downsample=getattr(args, 'downsample', 'orpool'),
            neck=getattr(args, 'neck', 'none'),
            decoupled_head=getattr(args, 'decoupled_head', False),
            reg_bins=getattr(args, 'reg_bins', 0),
            head_type=getattr(args, 'head_type', 'lut'),
        )
    return model.to(args.device)


LADDER_PRESETS = {
    # CNN→DLGN帰属実験(W14ladder)。各段が奪うmAPを切り分けるため、
    # DLGN風の要素を1つずつ足し込む。--cnn-input/--cnn-downsample/
    # --downsample/--head-type を一括で上書きする(個別指定は無視)。
    'L0': dict(arch='cnn', cnn_input='raw', cnn_downsample='stride'),
    'L1': dict(arch='cnn', cnn_input='thermo', cnn_downsample='stride'),
    'L2': dict(arch='cnn', cnn_input='thermo', cnn_downsample='orpool'),
    'L3': dict(arch='lut', cnn_input='default', downsample='orpool', head_type='float'),
    'L4': dict(arch='lut', cnn_input='default', downsample='orpool', head_type='lut'),
}


def apply_ladder_preset(args, parser):
    """--ladder {L0..L4} でCNN→DLGN帰属実験の梯子を一括設定する(W14ladder)。

    各段の意図(mAP差 = その段で足し込んだ要素が奪う精度):
        L0: CenterNetCNNそのまま(float, 未二値化raw入力, strideダウンサンプル,
            連続ロジット読み出し) = 上限
        L1: L0 + 入力のみthermometer二値化 → 入口量子化の損失
        L2: L1 + ダウンサンプルをOrPooling2dに(stride conv→OrPool) →
            OR飽和の損失
        L3: L2相当のLUT化。バックボーンをLutConv2d+OrPooling2dにする一方、
            読み出しは連続float(head_type='float')のまま → 混合欠如の損失
            (LutConv2dはツリーあたりn_leaves個のランダムサンプリングした
            入力しか混ぜないのに対し、通常convは全入力チャンネルを混ぜる)
        L4: L3 + 読み出しをSpatialGroupSumに(既存CenterNetLUT相当) →
            量子化読み出しの損失

    L3/L4は数式的にCenterNetLUT(downsample='orpool', head_type=...)そのもの
    （流用、新規モデルコードは不要）。L0-L2はCenterNetCNNの
    downsample切替（W14ladder、本追加）で表現する。
    """
    if args.ladder == 'none':
        return
    preset = LADDER_PRESETS[args.ladder]
    for k, v in preset.items():
        setattr(args, k, v)
    print(f'[--ladder {args.ladder}] 設定を上書き: '
          + ', '.join(f'{k}={v}' for k, v in preset.items()))


def build_selection_scheduler(args, model, num_iterations: int):
    """--selection/--tau-schedule に応じてτ_bスケジューラを構築する(W13a-1)。

    selection='soft'ならNone(スケジューラ不要)。'hard_st'/'gumbel_st'では
    --tau-scheduleが未指定(None)のとき既定を自動選択する:
        hard_st   → 'none'（tau_b固定、--tau-b-init）
        gumbel_st → 'anneal'（hardgap知見によりアニール必須、
                    TauAnnealConvSchedulerでtau_b幾何減衰+gumbel_scale
                    を0へ減衰させ訓練終盤はhard_stへ漸近する）
    """
    if args.arch != 'lut' or args.selection == 'soft':
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
# データセット構築
# ============================================================

# データセットごとのクラス数（fewclass/faceは可変・固定1のためbuild_datasets内で決定）
DATASET_N_CLASSES = {'multimnist': 10, 'voc': 20, 'multimnist_color': 10}


def build_lr_scheduler(args, optimizer, use_epochs: bool):
    """LRスケジューラを進行単位（epoch or iteration）に合わせて構築する。

    ★: 旧実装は `T_max = args.epochs if args.epochs is not None else 100` で、
    `-ni`（イテレーション）モードでは T_max=100 の cosine を作った上に
    `scheduler.step()` をエポックループ内でしか呼んでいなかった。結果として
    **`-ni` 系の全run（D1-D6 /全部 / カラー face）が
    lr 固定のまま走っていた**（レポート上は "cosine" と記載されているのに）。
    唯一 cosine が実際に効いていたのは `-e 60` の run＝現行採用 0.5209 である。

    Returns:
        CosineAnnealingLR、または args.scheduler != 'cosine' なら None。
    """
    if args.scheduler != 'cosine':
        return None
    t_max = args.epochs if use_epochs else args.num_iterations
    if not t_max or t_max < 1:
        raise ValueError(f'スケジューラの T_max が不正: {t_max}')
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=args.eta_min,
    )


def _parse_thresholds(spec: str | None) -> tuple[float, ...]:
    """'0.15,0.35' → (0.15, 0.35)。None/空文字 → ()。"""
    if not spec:
        return ()
    return tuple(float(v) for v in spec.split(',') if v.strip())


def _resolve_input_mode(args) -> tuple[bool, int]:
    """--cnn-input に応じて (use_rgb, n_thresholds) を決定する(W14ladder)。

    'default'（既定）は既存挙動を完全維持する: use_rgb=(arch=='cnn')、
    n_thresholds=args.n_thresholds（数値不変）。
    'raw'/'thermo' は arch に依らず明示的に入力エンコードを上書きする
    （CNN→DLGN帰属実験の梯子でL0=raw/L1以降=thermoを切り替えるため）。
    グレースケール(--dataset multimnist)は use_rgb 機構を持たない
    （color=True専用）ため、raw は n_thresholds=0（ThresholdBinarize無し=
    生グレースケール）で表現する。use_rgb=True 側はcolor系データセット
    （voc/multimnist_color/fewclass）でのみ有効（grayscale分岐は
    use_rgb を渡さない=常に無視される）。
    """
    mode = getattr(args, 'cnn_input', 'default')
    if mode == 'raw':
        return True, 0
    if mode == 'thermo':
        return False, args.n_thresholds
    return (args.arch == 'cnn'), args.n_thresholds


def _kd_teacher_n_thresholds(ckpt_spec: str | None) -> int | None:
    """KD teacher ckpt が訓練時に使った --n-thresholds を読む。

    グレースケール（color=False）のKDでは teacher も student と同じ二値化入力を
    消費する設計だが、teacher の conv1 は自分が訓練された閾値数のチャネル数しか
    受け取れない。student だけ 31閾値（5bit）へ上げると
    `expected input to have 3 channels, but got 31` で落ちる。
    ここで teacher 側の閾値数を取得して dataset へ渡し、teacher 用入力だけ
    別に二値化させる。読めない場合は None（＝従来どおり共有）。
    """
    if ckpt_spec is None:
        return None
    paths = sorted(glob.glob(ckpt_spec))
    path = paths[0] if paths else ckpt_spec
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    return (ckpt.get('metadata', {}).get('args', {}) or {}).get('n_thresholds')


def build_datasets(args, output_stride, train=True, test=True, kd=False):
    """--dataset に応じて訓練/テストデータセットを構築する。

    kd=True のとき訓練データセットのみ teacher 入力を同梱する（__getitem__ が
    (student_image, targets, teacher_image) を返す）。評価はKD不要のため常に
    2つ組（student_image, targets）。fewclass の pseudo-label(--fewclass-pseudo-labels)
    は kd と独立（teacher予測を直接GT化する方式のためオンライン teacher forward は不要）。

    Returns:
        (train_dataset or None, test_dataset or None, n_classes)
    """
    if args.dataset == 'fewclass':
        from data.detection_fewclass import FewClassVOCDetection
        classes = tuple(
            c.strip() for c in args.fewclass_classes.split(',') if c.strip()
        )
        n_classes = len(classes)
        use_rgb, n_thr = _resolve_input_mode(args)
        limit = args.limit_samples if args.limit_samples > 0 else None
        common = dict(
            img_size=args.canvas_size,
            n_thresholds=n_thr,
            output_stride=output_stride,
            max_objects=args.max_objects,
            classes=classes,
            use_rgb=use_rgb,
            download=False,
            edge_thresholds=_parse_thresholds(getattr(args, 'edge_thresholds', None)),
            curv_thresholds=_parse_thresholds(getattr(args, 'curv_thresholds', None)),
            chroma_thresholds=_parse_thresholds(getattr(args, 'chroma_thresholds', None)),
        )
        train_ds = FewClassVOCDetection(
            train=True, limit=limit, kd_teacher_input=kd,
            pseudo_label_path=(args.fewclass_pseudo_labels or None),
            **common,
        ) if train else None
        test_ds = FewClassVOCDetection(
            train=False, limit=limit, **common,
        ) if test else None
        return train_ds, test_ds, n_classes

    if args.dataset == 'face':
        from data.detection_face import WiderFaceDetection
        n_classes = 1
        use_rgb, n_thr = _resolve_input_mode(args)
        limit = args.limit_samples if args.limit_samples > 0 else None
        # face専用引数は全て default=None。既定値の解決はここで行う
        # （run_evalのargs復元は getattr(...) is None のときのみ代入するため、
        # 非Noneの既定値だとparser側では --eval 時に復元されない）
        face_root = args.face_root if args.face_root is not None else './data-widerface'
        face_min_box_px = (
            args.face_min_box_px if args.face_min_box_px is not None else 4.0
        )
        face_max_blur = args.face_max_blur if args.face_max_blur is not None else 2
        common = dict(
            root=face_root,
            img_size=args.canvas_size,
            n_thresholds=n_thr,
            output_stride=output_stride,
            max_objects=args.max_objects,
            use_rgb=use_rgb,
            download=False,
            min_box_px=face_min_box_px,
            max_blur=face_max_blur,
            edge_thresholds=_parse_thresholds(getattr(args, 'edge_thresholds', None)),
            curv_thresholds=_parse_thresholds(getattr(args, 'curv_thresholds', None)),
            chroma_thresholds=_parse_thresholds(getattr(args, 'chroma_thresholds', None)),
            drop_invalid=not args.face_keep_invalid,
        )
        train_ds = WiderFaceDetection(
            train=True, limit=limit, **common) if train else None
        test_ds = WiderFaceDetection(
            train=False, limit=limit, **common) if test else None
        return train_ds, test_ds, n_classes

    n_classes = DATASET_N_CLASSES[args.dataset]

    use_rgb, n_thr = _resolve_input_mode(args)

    if args.dataset == 'voc':
        from data.detection_voc import VOCDetectionCenterNet
        # CNNは生RGB(3ch)、LUTは閾値二値化(3*n_thresholds ch)（--cnn-inputで上書き可）
        limit = args.limit_samples if args.limit_samples > 0 else None
        common = dict(
            img_size=args.canvas_size,
            n_thresholds=n_thr,
            output_stride=output_stride,
            max_objects=args.max_objects,
            n_classes=n_classes,
            use_rgb=use_rgb,
            download=False,
            limit=limit,
        )
        train_ds = VOCDetectionCenterNet(
            train=True, kd_teacher_input=kd, **common) if train else None
        test_ds = VOCDetectionCenterNet(train=False, **common) if test else None
    elif args.dataset == 'multimnist_color':
        # CNNは生RGB(3ch)、LUTは閾値二値化(3*n_thresholds ch)。VOCと同型の切替
        common = dict(
            canvas_size=args.canvas_size,
            min_objects=args.min_objects,
            max_objects=args.max_objects,
            n_thresholds=n_thr,
            output_stride=output_stride,
            color=True,
            use_rgb=use_rgb,
            # 低レベル特徴検出器は student(LUT)側の入力にのみ効く。use_rgb=True の
            # teacher(CNN)は生RGBを受けるため、指定しても無視される（非汚染）。
            edge_thresholds=_parse_thresholds(getattr(args, 'edge_thresholds', None)),
            curv_thresholds=_parse_thresholds(getattr(args, 'curv_thresholds', None)),
            chroma_thresholds=_parse_thresholds(getattr(args, 'chroma_thresholds', None)),
        )
        train_ds = MultiMNISTDetection(
            train=True, num_samples=args.num_train, seed=args.seed,
            kd_teacher_input=kd, **common,
        ) if train else None
        test_ds = MultiMNISTDetection(
            train=False, num_samples=args.num_test, seed=args.seed + 1,
            **common,
        ) if test else None
    else:
        # グレースケール(color=False)はuse_rgb機構が無い。raw入力(W14ladder L0)は
        # n_thresholds=0(ThresholdBinarize無し=生グレースケール)で表現する
        train_ds = MultiMNISTDetection(
            train=True,
            canvas_size=args.canvas_size,
            min_objects=args.min_objects,
            max_objects=args.max_objects,
            n_thresholds=n_thr,
            output_stride=output_stride,
            num_samples=args.num_train,
            seed=args.seed,
            kd_teacher_input=kd,
            kd_teacher_n_thresholds=(
                _kd_teacher_n_thresholds(args.distill_teacher_ckpt) if kd else None
            ),
            edge_thresholds=_parse_thresholds(getattr(args, 'edge_thresholds', None)),
            curv_thresholds=_parse_thresholds(getattr(args, 'curv_thresholds', None)),
        ) if train else None
        test_ds = MultiMNISTDetection(
            train=False,
            canvas_size=args.canvas_size,
            min_objects=args.min_objects,
            max_objects=args.max_objects,
            n_thresholds=n_thr,
            output_stride=output_stride,
            num_samples=args.num_test,
            seed=args.seed + 1,
            edge_thresholds=_parse_thresholds(getattr(args, 'edge_thresholds', None)),
            curv_thresholds=_parse_thresholds(getattr(args, 'curv_thresholds', None)),
        ) if test else None

    return train_ds, test_ds, n_classes


# ============================================================
# 評価専用モード
# ============================================================

def run_eval(args):
    """重みを読み込んで評価・可視化のみ実行する。"""
    # hard評価の入力 round は「入力が本来0/1」のときだけ掛ける。
    # 生RGBを食う CNN で round すると画素が潰れて精度を破壊するため。
    use_rgb, _ = _resolve_input_mode(args)
    print(f'評価モード: {args.eval}')

    # デバイス
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # チェックポイントからargs復元
    ckpt = torch.load(args.eval, map_location=args.device, weights_only=False)
    saved_args = ckpt.get('metadata', {}).get('args', {})

    # 保存されたargsでモデル・データセットのパラメータを復元
    for key in ['dataset', 'arch', 'k', 'tree_depth', 'head_depth', 'n_inputs',
                'parameterization', 'grad_factor', 'input_channels_per_tree',
                'hm_group', 'size_group', 'offset_group', 'n_layers', 'wiring',
                'selection', 'tau_b_init', 'tau_b_min', 'gumbel_scale_init',
                'cnn_width', 'cnn_head_ch', 'canvas_size', 'min_objects',
                'max_objects', 'n_thresholds', 'fewclass_classes',
                'fewclass_pseudo_labels', 'downsample', 'neck',
                'decoupled_head', 'reg_bins', 'head_type', 'cnn_input',
                'cnn_downsample', 'face_root', 'face_min_box_px',
                'face_max_blur', 'face_keep_invalid']:
        if key in saved_args and getattr(args, key.replace('-', '_'), None) is None:
            setattr(args, key.replace('-', '_'), saved_args[key])

    output_stride = 4

    # データセット
    _, test_dataset, n_classes = build_datasets(
        args, output_stride, train=False, test=True,
    )
    in_ch = test_dataset[0][0].shape[0]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_detection, drop_last=False,
    )

    # モデル構築・重み読み込み
    model = build_model(args, in_ch, n_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'モデル読み込み完了 (step={ckpt["step"]}, best_metric={ckpt["best_metric"]:.4f})')

    loss_fn = CenterNetLoss(
        lambda_size=args.lambda_size,
        lambda_off=args.lambda_off,
    )

    # 評価実行（evalモードでは多めに可視化）
    n_vis_eval = min(64, args.num_test)
    map_hard, eval_loss_hard, correct_hard, incorrect_hard = evaluate(
        model, test_loader, args.device, loss_fn, mode=False,
        output_stride=output_stride, desc='eval(hard)', n_vis=n_vis_eval,
        n_classes=n_classes,
        binary_input=not use_rgb,
    )
    map_soft, eval_loss_soft, correct_soft, incorrect_soft = evaluate(
        model, test_loader, args.device, loss_fn, mode=True,
        output_stride=output_stride, desc='eval(soft)', n_vis=n_vis_eval,
        n_classes=n_classes,
    )

    print(f'\nmAP(hard)={map_hard:.4f}  loss(hard)={eval_loss_hard:.4f}')
    print(f'mAP(soft)={map_soft:.4f}  loss(soft)={eval_loss_soft:.4f}')
    print(f'Hard可視化: {len(correct_hard)}枚正解, {len(incorrect_hard)}枚不正解')
    print(f'Soft可視化: {len(correct_soft)}枚正解, {len(incorrect_soft)}枚不正解')

    # 可視化画像をローカル保存
    eval_dir = os.path.join(os.path.dirname(args.eval), '..', 'visualizations')
    os.makedirs(eval_dir, exist_ok=True)
    for i, img in enumerate(correct_hard):
        img.save(os.path.join(eval_dir, f'eval_hard_correct_{i}.png'))
    for i, img in enumerate(incorrect_hard):
        img.save(os.path.join(eval_dir, f'eval_hard_incorrect_{i}.png'))
    for i, img in enumerate(correct_soft):
        img.save(os.path.join(eval_dir, f'eval_soft_correct_{i}.png'))
    for i, img in enumerate(incorrect_soft):
        img.save(os.path.join(eval_dir, f'eval_soft_incorrect_{i}.png'))
    print(f'可視化画像を保存: {os.path.normpath(eval_dir)}/')

    # Mac GUI表示（不正解を優先）
    if args.device == 'mps':
        if incorrect_hard:
            show_vis_gui(incorrect_hard,
                         f'Eval Hard INCORRECT ({len(incorrect_hard)}枚, mAP={map_hard:.4f})')
        if correct_hard:
            show_vis_gui(correct_hard,
                         f'Eval Hard CORRECT ({len(correct_hard)}枚, mAP={map_hard:.4f})')
        try:
            import matplotlib.pyplot as plt
            plt.show(block=True)
        except ImportError:
            pass


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CenterNet + LUT 物体検出 訓練スクリプト'
    )

    # モデル
    parser.add_argument('--arch', type=str, default='lut',
                        choices=['lut', 'cnn'],
                        help='アーキテクチャ: lut=CenterNetLUT(論理ゲート), '
                             'cnn=CenterNetCNN(通常Conv+BN+ReLU) (default: lut)')
    parser.add_argument('-k', type=int, default=64,
                        help='モデル幅パラメータ k (default: 64)')
    parser.add_argument('-d', '--tree-depth', type=int, default=3,
                        help='バックボーンのツリーの深さ (default: 3)')
    parser.add_argument('--head-depth', type=int, default=3,
                        help='検出ヘッドのツリーの深さ (default: 3)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax16g',
                        choices=['softmax', 'independent', 'independent_ste',
                                 'independent_dwn', 'iwp', 'softmax16g'],
                        help='パラメータ化方式 (default: softmax16g。'
                             '旧softmaxはConv-DLGNで構造欠陥あり非推奨。'
                             'n_inputs>2 では independent_ste を使う'
                             '(independent は離散化で崩壊する))')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')
    parser.add_argument('--input-channels-per-tree', type=int, default=2,
                        help='各ツリーが参照する入力チャンネル数 (default: 2)')
    parser.add_argument('--hm-group', type=int, default=16,
                        help='ヒートマップヘッドのグループサイズ (default: 16)')
    parser.add_argument('--size-group', type=int, default=16,
                        help='サイズヘッドのグループサイズ (default: 16)')
    parser.add_argument('--offset-group', type=int, default=16,
                        help='オフセットヘッドのグループサイズ (default: 16)')
    parser.add_argument('--n-layers', type=int, default=3, choices=[3, 6, 8],
                        help='バックボーンのConv層数 (default: 3)')
    parser.add_argument('--wiring', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='LutConv2dの入力配線方式 (default: fixed。'
                             'learnableでDWN方式の配線学習を有効化)')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st', 'dwn'],
                        help='訓練時のゲート選択方式(--arch lut専用、W13a-1)。'
                             'soft=既定(従来挙動と数値不変)、hard_st=CAGE論文'
                             '方式(selection gap 0)、gumbel_st=Mind the Gap'
                             '方式(要アニール、--tau-schedule anneal自動適用)、'
                             'dwn=DWN方式のRAMノード(レバー4。表を有限差分で'
                             '直接学ぶので学習時と評価時のforwardが完全一致)')
    parser.add_argument('--dwn-n-inputs', type=int, default=6,
                        help='--selection dwn のRAMノード入力数n(表サイズ2^n、'
                             'default: 6 → 64エントリ)')
    parser.add_argument('--dwn-efd', type=str, default='paper',
                        choices=['paper', 'official'],
                        help='--selection dwn のEFD係数(paper=1/(H+1)、'
                             'official=α·β^H)')
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
                             '(hardgap知見によりアニール必須)。明示指定で上書き'
                             '可。anneal=TauAnnealConvScheduler(幾何減衰+'
                             'gumbel_scaleも0へ減衰)、cage=CAGEConvScheduler'
                             '(確信度EMA適応、旧--cage相当)')
    parser.add_argument('--cnn-width', type=int, default=64,
                        help='CNN版バックボーンの基本チャンネル幅 (default: 64)')
    parser.add_argument('--cnn-head-ch', type=int, default=64,
                        help='CNN版ヘッドの中間チャンネル数 (default: 64)')
    parser.add_argument('--cnn-input', type=str, default='default',
                        choices=['default', 'raw', 'thermo'],
                        help='CNN版の入力エンコード(--arch cnn専用、W14ladder、'
                             'default: default=既存挙動をそのまま維持=数値不変)。'
                             "'raw'=未二値化の連続値入力(color系データセットは"
                             '生RGB=use_rgb相当、グレースケールmultimnistは'
                             'n_thresholds=0の生グレースケール)。'
                             "'thermo'=閾値二値化入力(LUT版と同じ"
                             'thermometer encoding)。CNN→DLGN帰属実験の梯子'
                             '(L0=raw/L1以降=thermo)で入口量子化の影響だけを'
                             '切り出すためのフラグ')
    parser.add_argument('--cnn-downsample', type=str, default='stride',
                        choices=['stride', 'orpool'],
                        help='CNN版バックボーンのダウンサンプル方式(--arch cnn'
                             '専用、W14ladder-L2、default: stride=従来挙動と'
                             '数値不変=2x2 stride2 convでダウンサンプル)。'
                             "'orpool'で(stride1 conv + OrPooling2d)に置換"
                             '(LUT版--downsample orpoolとの対応物、OR飽和の'
                             '影響だけを切り出して測る)')
    parser.add_argument('--ladder', type=str, default='none',
                        choices=['none', 'L0', 'L1', 'L2', 'L3', 'L4'],
                        help='CNN→DLGN帰属実験の梯子(W14ladder、default: '
                             'none=無効)。1段ごとにDLGN風の要素を1つずつ'
                             '足し込み、各段のmAP差でその要素が奪う精度を測る: '
                             'L0=CenterNetCNNそのまま(生/未二値化入力、stride'
                             'ダウンサンプル、連続float、上限)/ L1=入力のみ'
                             'thermometer二値化(入口量子化の損失)/ L2=L1+'
                             'ダウンサンプルをOrPooling2dに(OR飽和の損失)/ '
                             'L3=L2のLUT化(バックボーンLutConv2d+OrPool、'
                             '読み出しは連続float=head-type float、混合欠如の'
                             '損失)/ L4=L3+読み出しをSpatialGroupSumに(既存'
                             'CenterNetLUT相当、量子化読み出しの損失)。'
                             '指定時は--arch/--cnn-input/--cnn-downsample/'
                             '--downsample/--head-typeを一括で上書きする'
                             '(個別指定は無視される)')
    parser.add_argument('--downsample', type=str, default='orpool',
                        choices=['orpool', 'strided'],
                        help='バックボーンのダウンサンプル方式(--arch lut専用、'
                             'default: orpool=従来挙動と数値不変)。'
                             'stridedでOrPooling2dの代わりにstride=2の学習'
                             '可能LutConv2dを使用')
    parser.add_argument('--neck', type=str, default='none',
                        choices=['none', 'fpn', 'fpnpan'],
                        help='多スケール特徴融合neck(--arch lut専用、'
                             'default: none=従来挙動と数値不変)。fpn=UpSample2d'
                             '+concat+LutConv2dのトップダウン融合、fpnpan='
                             'fpnに加えボトムアップ精緻化(PAN風)も追加')
    parser.add_argument('--decoupled-head', action='store_true',
                        help='分類(hm)/回帰(size,off)ヘッドの入力特徴ブランチ'
                             'を分離する(--arch lut専用、default: '
                             'False=共有feat=従来挙動)')
    parser.add_argument('--reg-bins', type=int, default=0,
                        help='size/offをスカラー回帰でなくreg_bins個のビン上の'
                             '分布として出力し期待値を取る(DFL風、--arch lut'
                             '専用、default: 0=従来のスカラー回帰)')
    parser.add_argument('--head-type', type=str, default='lut',
                        choices=['lut', 'float'],
                        help='検出ヘッドの型(--arch lut専用、default: '
                             'lut=従来挙動と数値不変)。floatはバックボーンは'
                             'LutConv2dのまま検出ヘッドのみ通常nn.Conv2dに'
                             'するハイブリッド構成(読み出しの量子化[0,0.31]'
                             '圧縮を回避)。注意: floatヘッドのckptはIR'
                             'エクスポート不可(scripts/export_webbench.pyが'
                             '明示的にエラーを出す)')

    # データセット
    parser.add_argument('--dataset', type=str, default='multimnist',
                        choices=['multimnist', 'voc', 'multimnist_color',
                                 'fewclass', 'face'],
                        help='データセット: multimnist=合成Multi-MNIST(10クラス), '
                             'voc=Pascal VOC実データ(20クラス), '
                             'multimnist_color=カラー版Multi-MNIST(CIFAR-10背景+'
                             '色付き数字、10クラス), '
                             'fewclass=少クラスVOC実物体検出(--fewclass-classes'
                             'で指定), '
                             'face=WIDER FACE顔検出(1クラス、'
                             'scripts/fetch_widerface.pyで事前取得が必要) '
                             '(default: multimnist)')
    parser.add_argument('--fewclass-classes', type=str, default='person,car,dog',
                        help='--dataset fewclass 用のVOCクラス名(カンマ区切り、'
                             '並び順がクラスID) (default: person,car,dog)')
    parser.add_argument('--fewclass-pseudo-labels', type=str, default=None,
                        help='--dataset fewclass 用、gen_teacher_pseudolabels.py'
                             'が生成した.ptパス。指定時、訓練GTをteacher'
                             'pseudo-labelで置換する(W8 KD腕、CenterNetKDLossは'
                             '不使用=通常のCenterNetLossで学習)')
    parser.add_argument('--face-root', type=str, default=None,
                        help='--dataset face 用のWIDER FACEデータルート'
                             '(default: ./data-widerface。'
                             'scripts/fetch_widerface.py で事前取得すること)')
    parser.add_argument('--face-min-box-px', type=float, default=None,
                        help='--dataset face 用、リサイズ後の最小顔サイズpx'
                             '(min(w,h)がこれ未満の顔を除外) (default: 4.0)')
    parser.add_argument('--face-max-blur', type=int, default=None,
                        help='--dataset face 用、許容する最大blur属性値'
                             '(超過は除外) (default: 2)')
    parser.add_argument('--face-keep-invalid', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='--dataset face 用、invalid=1の顔を除外せず残す'
                             '(default: False=除外)')
    parser.add_argument('--limit-samples', type=int, default=0,
                        help='VOC/fewclassデータセット長の上限(スモーク用、0=全件) '
                             '(default: 0)')
    parser.add_argument('--canvas-size', type=int, default=96,
                        help='キャンバス/画像サイズ (default: 96。VOCは160推奨、'
                             'fewclassは128推奨)')
    parser.add_argument('--min-objects', type=int, default=1,
                        help='最小オブジェクト数 (default: 1)')
    parser.add_argument('--max-objects', type=int, default=3,
                        help='最大オブジェクト数 (default: 3)')
    parser.add_argument('--n-thresholds', type=int, default=3,
                        help='入力二値化の閾値数 (default: 3)')
    # Conv-DLGN §3.5 の低レベル特徴検出器(固定・非学習)。全て未指定なら従来と完全同一。
    parser.add_argument('--edge-thresholds', type=str, default=None,
                        help='1階差分(edge)検出器のしきい値をカンマ区切りで '
                             '(例: 0.15,0.35)。未指定で無効')
    parser.add_argument('--curv-thresholds', type=str, default=None,
                        help='2階差分(curvature)検出器のしきい値をカンマ区切りで。未指定で無効')
    parser.add_argument('--chroma-thresholds', type=str, default=None,
                        help='チャネル間比較・彩度のしきい値をカンマ区切りで '
                             '(multimnist_color 専用)。未指定で無効')
    parser.add_argument('--num-train', type=int, default=10000,
                        help='訓練サンプル数 (default: 10000)')
    parser.add_argument('--num-test', type=int, default=2000,
                        help='テストサンプル数 (default: 2000)')

    # 訓練
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.002,
                        help='Weight decay (default: 0.002)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='バッチサイズ (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数（--num-iterationsと排他）')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数（--epochsと排他）')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度')
    parser.add_argument('--lambda-size', type=float, default=0.1,
                        help='サイズ損失の重み (default: 0.1)')
    parser.add_argument('--lambda-off', type=float, default=1.0,
                        help='オフセット損失の重み (default: 1.0)')
    parser.add_argument('--ema-decay', type=float, default=0.0,
                        help='パラメータEMAの減衰率 (default: 0=無効)。'
                             '評価とckpt保存にのみ適用され、訓練は生の重みで進む。'
                             'hard_st の argmax 反転による評価振動の抑制狙い。'
                             '0.999 前後が定番')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='勾配ノルムクリップの閾値 (default: 0=無効)。'
                             'hard_st は離散argmaxでゲート選択が跳ぶため勾配が'
                             '突発的に大きくなる。1.0 が定番')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine'],
                        help='学習率スケジューラ (default: cosine)')
    parser.add_argument('--eta-min', type=float, default=1e-4,
                        help='CosineAnnealingLRの最小学習率 (default: 1e-4)')

    # KD(知識蒸留、--arch lut 専用)
    parser.add_argument('--distill-teacher-ckpt', type=str, default=None,
                        help='KD teacher(CNN CenterNet)のckptパス。glob可'
                             '(実行時解決、最新mtime採用)。指定時 LUT student を'
                             'KDで学習')
    parser.add_argument('--distill-alpha', type=float, default=0.25,
                        help='hard(GT)項の重みα (default: 0.25、スキル§2-3)')
    parser.add_argument('--distill-T', type=float, default=2.0,
                        help='ヒートマップKD温度T (default: 2.0。'
                             '二値的分布はT=2がT=4を圧倒しうる§8.3-1)')

    # その他
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='出力ベースディレクトリ (default: outputs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから再開')
    parser.add_argument('--eval', type=str, default=None,
                        help='評価のみ（チェックポイントパスを指定）')
    parser.add_argument('--init-from', type=str, default=None,
                        help='事前学習ckpt(例: pretrain_cifar_encoder.py出力)'
                             'から部分ロードして初期化する(W13a-3)。'
                             '--resumeと異なりoptimizer/step等は復元せず、'
                             '形状一致キーのみ重みをコピーする(load_partial_'
                             'state_dict)。--resumeと併用不可')
    parser.add_argument('--init-key-map', type=str, default=None,
                        help='--init-from用のキー名prefixリマップ。'
                             '"model側prefix=ckpt側prefix" をカンマ区切りで'
                             '複数指定可(例: "stage1.0=encoder.c1,stage2.0='
                             'encoder.c2")。省略時はキー名がそのまま一致する'
                             '前提(素通し)')

    args = parser.parse_args()
    apply_ladder_preset(args, parser)

    # 評価専用モード
    if args.eval:
        run_eval(args)
        return

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # KD判定(lut かつ teacher ckpt 指定時。CenterNetKDLossによるオンライン蒸留)
    is_kd = args.distill_teacher_ckpt is not None
    # hard評価の入力 round は「入力が本来0/1」のときだけ掛ける。
    # 生RGBを食う CNN で round すると画素が潰れて精度を破壊するため。
    use_rgb, _ = _resolve_input_mode(args)

    if is_kd and args.arch != 'lut':
        parser.error('--distill-teacher-ckpt は --arch lut 専用です')
    # fewclass の online KD を解禁(dataset 側に kd_teacher_input を追加した)。
    # 2段階蒸留(fasterrcnn疑似ラベル→中間CNN→DLGN)のため。pseudo-label 方式
    # (--fewclass-pseudo-labels)とは排他にする: 両方だと GT が二重に置換される。
    if is_kd and args.dataset == 'fewclass' and args.fewclass_pseudo_labels is not None:
        parser.error('--distill-teacher-ckpt と --fewclass-pseudo-labels は併用不可 '
                     '(GTの置換が二重になる)。どちらか一方を選ぶこと')
    if is_kd and args.dataset == 'face':
        parser.error('--distill-teacher-ckpt(CNN同taskオンラインKD)は --dataset '
                     'face と併用不可。face の __getitem__ は (image, targets) '
                     'の2つ組しか返さないため ValueError: not enough values to '
                     'unpack になる')
    if args.fewclass_pseudo_labels is not None and args.dataset != 'fewclass':
        parser.error('--fewclass-pseudo-labels は --dataset fewclass 専用です')
    if args.selection != 'soft' and args.arch != 'lut':
        parser.error('--selection hard_st/gumbel_st は --arch lut 専用です')
    # 低レベル特徴検出器を配管済みなのは RGB 3ch を持つ3データセットのみ。
    # 未配管のデータセットで黙って無視されると「効かなかった」と誤結論するので明示エラー。
    # （chroma はチャネル間比較なのでグレースケールでは原理的に無意味）
    # multimnist(グレースケール)は edge/curv のみ可。chroma はチャネル間比較なので
    # 1ch では原理的に無意味 → データセット側が ValueError を出す前にここで弾く。
    _LOWLEVEL_DATASETS = ('multimnist_color', 'fewclass', 'face', 'multimnist')
    if any((args.edge_thresholds, args.curv_thresholds, args.chroma_thresholds)):
        if args.dataset not in _LOWLEVEL_DATASETS:
            parser.error('--edge/--curv/--chroma-thresholds は --dataset '
                         f'{{{",".join(_LOWLEVEL_DATASETS)}}} 専用です')
        if args.arch != 'lut':
            parser.error('--edge/--curv/--chroma-thresholds は --arch lut 専用です '
                         '(CNN teacher は生RGBを受けるため無効)')
        if args.chroma_thresholds and args.dataset == 'multimnist':
            parser.error('--chroma-thresholds はRGB 3ch のデータセット専用です '
                         '(multimnist はグレースケール1ch)')
    if args.arch != 'lut' and (
        args.downsample != 'orpool' or args.neck != 'none'
        or args.decoupled_head or args.reg_bins != 0 or args.head_type != 'lut'
    ):
        parser.error('--downsample/--neck/--decoupled-head/--reg-bins/'
                     '--head-type は --arch lut 専用です')
    if args.arch != 'cnn' and (
        args.cnn_input != 'default' or args.cnn_downsample != 'stride'
    ):
        parser.error('--cnn-input/--cnn-downsample は --arch cnn 専用です')
    if args.init_from and args.resume:
        parser.error('--init-from と --resume は同時に指定できません')

    # データセット
    output_stride = 4
    train_dataset, test_dataset, n_classes = build_datasets(
        args, output_stride, train=True, test=True, kd=is_kd,
    )

    use_pin_memory = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_detection,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_detection,
        pin_memory=use_pin_memory, drop_last=False, num_workers=2,
    )

    # 入力チャンネル数はデータセットの実サンプルから決定
    # (multimnist=n_thresholds, VOC-LUT=3*n_thresholds, VOC-CNN=3)
    in_ch = train_dataset[0][0].shape[0]
    print(f'データセット: {args.dataset} 検出 ({n_classes}クラス)')
    print(f'キャンバス: {args.canvas_size}×{args.canvas_size}, '
          f'オブジェクト数(max): {args.max_objects}')
    print(f'閾値数: {args.n_thresholds} (入力チャンネル: {in_ch})')
    print(f'訓練: {len(train_dataset):,}サンプル, テスト: {len(test_dataset):,}サンプル')

    # モデル構築
    model = build_model(args, in_ch, n_classes)

    # パラメータ統計
    total_gates = model.num_gates
    total_params = sum(p.numel() for p in model.parameters())
    if args.arch == 'cnn':
        print(f'\nモデル構成: arch=cnn, width={args.cnn_width}, '
              f'head_ch={args.cnn_head_ch}')
    else:
        print(f'\nモデル構成: arch=lut, k={args.k}, tree_depth={args.tree_depth}, '
              f'n_inputs={args.n_inputs}, n_layers={args.n_layers}')
        print(f'パラメータ化: {args.parameterization}')
        print(f'選択方式: {args.selection}'
              + (f' (τ_b={args.tau_b_init}, gumbel_scale={args.gumbel_scale_init})'
                 if args.selection != 'soft' else ''))
        print(f'HMグループ: {args.hm_group}, サイズグループ: {args.size_group}, '
              f'オフセットグループ: {args.offset_group}')
        print(f'総ロジックゲート数: {total_gates:,} (LUT{args.n_inputs})')
    print(f'総パラメータ数: {total_params:,}')
    print(model)

    # 事前学習ckptからの部分ロード初期化(W13a-3)
    init_result = apply_init_from(model, args.init_from, args.init_key_map, args.device)
    if init_result is not None:
        args.init_from_load_ratio = init_result['load_ratio']

    # KD teacher(オンライン推論、no_grad/eval固定)
    teacher = None
    if is_kd:
        teacher = load_kd_teacher_detection(args.distill_teacher_ckpt, args.device)

    # 損失関数・オプティマイザ
    # 評価は常に通常CenterNetLoss(GTベース、scratchと同一プロトコル)
    eval_loss_fn = CenterNetLoss(
        lambda_size=args.lambda_size,
        lambda_off=args.lambda_off,
    )
    if is_kd:
        loss_fn = CenterNetKDLoss(
            alpha=args.distill_alpha,
            temperature=args.distill_T,
            lambda_size=args.lambda_size,
            lambda_off=args.lambda_off,
        )
        print(f'KD: alpha={args.distill_alpha} T={args.distill_T} '
              f'(hm=二値KL, size/off=teacher予測L1@GT位置)')
    else:
        loss_fn = eval_loss_fn
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # パラメータEMA。hard_st の推論は「ゲートlogitのargmax」で決まるため、
    # 拮抗ゲートが勾配ノイズで反転すると回路が跳ね、eval が振動する
    # （実測: eval間 sd 0.09〜0.16）。EMA を評価・保存にのみ適用する。
    param_ema = None
    if args.ema_decay and args.ema_decay > 0:
        param_ema = ParamEMA(model, decay=args.ema_decay)
        print(f'パラメータEMA: decay={args.ema_decay}（評価・ckpt保存に適用）')

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 100
        use_epochs = True

    # 学習率スケジューラ（構築はエポック/イテレーション解決の**後**であること）
    scheduler = build_lr_scheduler(args, optimizer, use_epochs)
    if scheduler is not None:
        unit = 'epoch' if use_epochs else 'iter'
        print(f'スケジューラ: CosineAnnealingLR(T_max={scheduler.T_max} [{unit}], '
              f'eta_min={args.eta_min})')

    # ランディレクトリ（統合出力）
    if args.dataset == 'voc':
        ds_tag = 'voc_'
    elif args.dataset == 'multimnist_color':
        ds_tag = 'color_'
    elif args.dataset == 'fewclass':
        n_fc = len(args.fewclass_classes.split(','))
        ds_tag = f'fc{n_fc}_'
    elif args.dataset == 'face':
        ds_tag = 'face_'
    else:
        ds_tag = ''
    is_pseudo_kd = args.dataset == 'fewclass' and args.fewclass_pseudo_labels is not None
    if args.arch == 'cnn':
        run_name = f'det_{ds_tag}cnn_w{args.cnn_width}_obj{args.max_objects}'
    else:
        kd_suffix = f'_kdT{args.distill_T:g}' if is_kd else ''
        pseudo_suffix = '_pseudo' if is_pseudo_kd else ''
        sel_suffix = f'_{args.selection}' if args.selection != 'soft' else ''
        run_name = (f'det_{ds_tag}lut_k{args.k}_d{args.tree_depth}_L{args.n_layers}'
                    f'_obj{args.max_objects}{kd_suffix}{pseudo_suffix}{sel_suffix}')
    run_dir = RunDir(run_name, base_dir=args.output_dir)
    run_dir.save_config({
        **vars(args),
        'total_gates': total_gates,
        'total_params': total_params,
        'model_architecture': str(model),
    })
    print(f'出力先: {run_dir.root}/')

    run_dir.log(
        f'START k={args.k} tree_depth={args.tree_depth} '
        f'n_inputs={args.n_inputs} n_layers={args.n_layers} '
        f'lr={args.lr} bs={args.batch_size} '
        f'max_objects={args.max_objects} canvas={args.canvas_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iterations=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device}'
    )

    # チェックポイントのメタデータ
    ckpt_metadata = {'args': vars(args)}

    best_map = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=args.device)
        start_step = ckpt_info['step']
        best_map = ckpt_info['best_metric']
        print(f'  ステップ {start_step} から再開 (best_mAP={best_map:.4f})')

    # wandb
    wandb_tags = ['w2det', args.arch, 'centernet', f'obj{args.max_objects}']
    if args.arch == 'lut' and args.selection != 'soft':
        wandb_tags += ['w13', f'sel-{args.selection}']
    if is_kd:
        wandb_tags.append('kd')
    if args.dataset == 'voc':
        wandb_tags.append('voc')
    if args.dataset == 'multimnist_color':
        wandb_tags.append('color')
    if args.dataset == 'fewclass':
        wandb_tags += ['w8det', 'fewclass']
        if is_pseudo_kd:
            wandb_tags.append('pseudo-kd')
    if args.dataset == 'face':
        wandb_tags += ['w17det', 'face', 'widerface']
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=wandb_tags,
        group='detseg-web',
    )
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                run_dir.save_wandb_run_id(wandb.run.id)
        except ImportError:
            pass

    batches_per_epoch = len(train_loader)

    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'\n訓練開始: {args.epochs}エポック '
              f'({batches_per_epoch}バッチ/エポック, '
              f'評価={eval_freq}エポックごと)')
    else:
        num_iterations = args.num_iterations
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'\n訓練開始: {num_iterations:,}イテレーション '
              f'(評価={eval_freq:,}イテレーションごと)')

    # τ_bスケジューラ(selection=hard_st/gumbel_st専用、W13a-1)
    selection_scheduler = build_selection_scheduler(args, model, num_iterations)
    if selection_scheduler is not None:
        print(f'τ_bスケジューラ: {type(selection_scheduler).__name__} '
              f'(selection={args.selection})')

    print('=' * 70)

    train_start = time.time()

    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses = {'hm_loss': 0, 'size_loss': 0, 'off_loss': 0, 'total_loss': 0}
            pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}', leave=False)

            for batch in pbar:
                teacher_images = None
                if is_kd:
                    images, targets, teacher_images = batch
                    teacher_images = teacher_images.to(torch.float32).to(args.device)
                else:
                    images, targets = batch
                images = images.to(torch.float32).to(args.device)
                targets_device = {k: v.to(args.device) for k, v in targets.items()}

                loss_dict = train_step(model, images, targets_device, loss_fn,
                                        optimizer, teacher=teacher,
                                        teacher_images=teacher_images,
                                        grad_clip=args.grad_clip)
                if selection_scheduler is not None:
                    selection_scheduler.step()
                if param_ema is not None:
                    param_ema.update(model)
                for key in epoch_losses:
                    epoch_losses[key] += loss_dict[key]

                elapsed_t = time.time() - train_start
                progress = (epoch - 1 + (pbar.n + 1) / len(train_loader)) / args.epochs
                remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
                pbar.set_postfix(loss=f'{loss_dict["total_loss"]:.4f}', eta=eta_str)

            avg_losses = {k: v / batches_per_epoch for k, v in epoch_losses.items()}

            # 学習率スケジューラのステップ
            if scheduler is not None:
                scheduler.step()

            # 評価
            if epoch % eval_freq == 0 or epoch == args.epochs:
                # EMA 有効時は評価・ckpt保存を EMA 重みで行う
                if param_ema is not None:
                    param_ema.swap_in(model)
                map_hard, eval_loss_hard, correct_hard, incorrect_hard = evaluate(
                    model, test_loader, args.device, eval_loss_fn, mode=False,
                    output_stride=output_stride,
                    desc=f'eval(hard) {epoch}/{args.epochs}',
                    n_classes=n_classes,
                    binary_input=not use_rgb,
                )
                map_soft, eval_loss_soft, correct_soft, incorrect_soft = evaluate(
                    model, test_loader, args.device, eval_loss_fn, mode=True,
                    output_stride=output_stride,
                    desc=f'eval(soft) {epoch}/{args.epochs}',
                    n_classes=n_classes,
                )

                n_correct = len(correct_hard)
                n_incorrect = len(incorrect_hard)
                print(f'  可視化: {n_correct}枚正解, {n_incorrect}枚不正解 '
                      f'(hard, {n_correct + n_incorrect}枚中)')

                is_best = map_hard > best_map
                if is_best:
                    best_map = map_hard

                # チェックポイント保存（ローカル）
                step_now = epoch * batches_per_epoch
                save_checkpoint(run_dir.latest_path, model, optimizer, None,
                                step_now, best_map, ckpt_metadata)
                if is_best:
                    save_checkpoint(run_dir.best_path, model, optimizer, None,
                                    step_now, best_map, ckpt_metadata)
                    if use_wandb:
                        log_model_checkpoint(
                            run_dir.best_path, 'detection-centernet-best',
                            metadata={'epoch': epoch, 'mAP_hard': map_hard},
                        )

                # メトリクスをローカルJSONLに保存
                current_lr = optimizer.param_groups[0]['lr']
                metrics = {
                    'epoch': epoch,
                    **avg_losses,
                    'mAP_hard': map_hard,
                    'mAP_soft': map_soft,
                    'best_mAP': best_map,
                    'eval_loss_hard': eval_loss_hard,
                    'eval_loss_soft': eval_loss_soft,
                    # lr を必ず記録する。「cosine と書いてあるのに実は固定」
                    # を二度と不可視にしないため（current_lr は L1418 で
                    # optimizer.param_groups[0]['lr'] から取っている）。
                    'lr': current_lr,
                    'n_vis_correct_hard': n_correct,
                    'n_vis_incorrect_hard': n_incorrect,
                }
                if selection_scheduler is not None:
                    metrics['tau_b'] = selection_scheduler.tau_b
                run_dir.log_metrics(metrics)

                # EMA スワップを解除して生の重みで訓練を継続する
                if param_ema is not None:
                    param_ema.swap_out(model)

                # 可視化画像のローカル保存・wandbログ・GUI表示
                handle_eval_results(
                    run_dir, use_wandb, args.device, step_label=epoch,
                    correct_hard=correct_hard, incorrect_hard=incorrect_hard,
                    correct_soft=correct_soft, incorrect_soft=incorrect_soft,
                )

                elapsed = time.time() - train_start
                sec_per_epoch = elapsed / epoch
                remaining = sec_per_epoch * (args.epochs - epoch)
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_losses["total_loss"]:.4f} '
                    f'(hm={avg_losses["hm_loss"]:.4f} '
                    f'sz={avg_losses["size_loss"]:.4f} '
                    f'off={avg_losses["off_loss"]:.4f})  '
                    f'mAP(hard)={map_hard:.4f}  '
                    f'mAP(soft)={map_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                run_dir.log(
                    f'EPOCH {epoch}/{args.epochs} '
                    f'loss={avg_losses["total_loss"]:.4f} '
                    f'hm={avg_losses["hm_loss"]:.4f} '
                    f'sz={avg_losses["size_loss"]:.4f} '
                    f'off={avg_losses["off_loss"]:.4f} '
                    f'mAP_hard={map_hard:.4f} mAP_soft={map_soft:.4f} '
                    f'best={str(is_best).lower()} elapsed={elapsed_str}'
                )

                if use_wandb:
                    log_metrics(metrics)
    else:
        pbar = tqdm(
            enumerate(load_n(train_loader, num_iterations)),
            desc='train', total=num_iterations,
        )
        for i, batch in pbar:
            teacher_images = None
            if is_kd:
                images, targets, teacher_images = batch
                teacher_images = teacher_images.to(torch.float32).to(args.device)
            else:
                images, targets = batch
            images = images.to(torch.float32).to(args.device)
            targets_device = {k: v.to(args.device) for k, v in targets.items()}

            model.train()
            loss_dict = train_step(model, images, targets_device, loss_fn,
                                    optimizer, teacher=teacher,
                                    teacher_images=teacher_images,
                                    grad_clip=args.grad_clip)
            if selection_scheduler is not None:
                selection_scheduler.step()
            # ★: イテレーションモードでも LR スケジューラを進める
            # （旧実装はここが無く -ni 系が全て lr 固定だった）
            if scheduler is not None:
                scheduler.step()
            if param_ema is not None:
                param_ema.update(model)

            epoch_now = (i + 1) / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / (i + 1)
            remaining_sec = sec_per_iter * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(
                loss=f'{loss_dict["total_loss"]:.4f}',
                epoch=f'{epoch_now:.1f}', eta=eta_str,
            )

            if (i + 1) % eval_freq == 0:
                # EMA 有効時は評価・ckpt保存を EMA 重みで行う
                # （hard_st の argmax 反転による eval 振動を抑えるため）
                if param_ema is not None:
                    param_ema.swap_in(model)
                map_hard, eval_loss_hard, correct_hard, incorrect_hard = evaluate(
                    model, test_loader, args.device, eval_loss_fn, mode=False,
                    output_stride=output_stride,
                    desc=f'eval(hard) {i+1}/{num_iterations}',
                    n_classes=n_classes,
                    binary_input=not use_rgb,
                )
                map_soft, eval_loss_soft, correct_soft, incorrect_soft = evaluate(
                    model, test_loader, args.device, eval_loss_fn, mode=True,
                    output_stride=output_stride,
                    desc=f'eval(soft) {i+1}/{num_iterations}',
                    n_classes=n_classes,
                )

                n_correct = len(correct_hard)
                n_incorrect = len(incorrect_hard)

                is_best = map_hard > best_map
                if is_best:
                    best_map = map_hard

                save_checkpoint(run_dir.latest_path, model, optimizer, None,
                                i + 1, best_map, ckpt_metadata)
                if is_best:
                    save_checkpoint(run_dir.best_path, model, optimizer, None,
                                    i + 1, best_map, ckpt_metadata)
                    if use_wandb:
                        log_model_checkpoint(
                            run_dir.best_path, 'detection-centernet-best',
                            metadata={'iteration': i + 1, 'mAP_hard': map_hard},
                        )

                # メトリクスをローカルJSONLに保存
                metrics = {
                    'iteration': i + 1,
                    'epoch': epoch_now,
                    **loss_dict,
                    'mAP_hard': map_hard,
                    'mAP_soft': map_soft,
                    'best_mAP': best_map,
                    'n_vis_correct_hard': n_correct,
                    'n_vis_incorrect_hard': n_incorrect,
                    # lr を必ず記録する。「cosine と書いてあるのに実は固定」
                    # を二度と不可視にしないため。
                    'lr': optimizer.param_groups[0]['lr'],
                }
                if selection_scheduler is not None:
                    metrics['tau_b'] = selection_scheduler.tau_b
                run_dir.log_metrics(metrics)

                # EMA スワップを解除して生の重みで訓練を継続する
                if param_ema is not None:
                    param_ema.swap_out(model)

                # 可視化画像のローカル保存・wandbログ・GUI表示
                handle_eval_results(
                    run_dir, use_wandb, args.device, step_label=i + 1,
                    correct_hard=correct_hard, incorrect_hard=incorrect_hard,
                    correct_soft=correct_soft, incorrect_soft=incorrect_soft,
                )

                elapsed = time.time() - train_start
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining_sec)))
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

                print(
                    f'\n[iter {i+1:,} / epoch {epoch_now:.1f}] '
                    f'loss={loss_dict["total_loss"]:.4f}  '
                    f'mAP(hard)={map_hard:.4f}  '
                    f'mAP(soft)={map_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                run_dir.log(
                    f'ITER {i+1}/{num_iterations} epoch={epoch_now:.1f} '
                    f'loss={loss_dict["total_loss"]:.4f} '
                    f'mAP_hard={map_hard:.4f} mAP_soft={map_soft:.4f} '
                    f'best={str(is_best).lower()} elapsed={elapsed_str}'
                )

                if use_wandb:
                    log_metrics(metrics)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終mAP(hard): {best_map:.4f}  (総訓練時間: {total_time})')
    run_dir.log(f'DONE best_mAP={best_map:.4f} total_time={total_time}')
    run_dir.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            run_dir.best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_map': best_map},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
