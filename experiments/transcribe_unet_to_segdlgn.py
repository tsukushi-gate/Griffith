"""SmallUNet(CNN) → SegConvDLGN 構造転写PoC(W13a-5)。

過去実測(model-conversion-kdスキル):=転写liftの実体はほぼ配線パターン
(関数保存の寄与は検出限界以下)、=境界〜負。本スクリプトはこの知見を
踏まえ「有界(bounded)なPoC」として、既存パターン(第1conv層のみ
transcribe_conv2dで構造転写し、残りは通常初期化のままrecover=通常学習に
委ねる)をSmallUNet→SegConvDLGNへ適用する。専用recoverループは作らず、
出力ckptは `train_segmentation.py --init-from`(W13a-3)でそのまま読める
形式にする(磨きはS5学習armが担当)。

■ スコープ(有界)と設計判断の根拠

1. **c1のみ転写、c2/c3/d2/d1は通常初期化のまま**:
   transcribe_conv2dはbinary(0/1)入力を要求する(LutConv2dが binary-domain
   のtree演算だから)。SmallUNetの中間活性(enc2/bottの入力)はBN+ReLU後の
   連続値であり、0/1較正データを用意できない(teacherの「真の」中間活性を
   二値化して使うのは意味論が破綻する近似になる)。も同じ理由で
   第1conv層のみを構造転写し、以降はrandom初期化+recover(通常学習)に
   委ねている。本PoCも同じ境界を踏襲する。

2. **BN fold**: SmallUNetは Conv→BN→ReLU の2連(_double_conv)。各段の
   「第1conv」(enc1[0]+enc1[1])をfold_conv_bnで1つのbias付きConv2dへ融合
   してからtranscribe_conv2dへ渡す(BNは変換対象外のため必須の前処理)。

3. **入力チャンネル数の不一致(RGB 3ch → thermometer 9ch)への対処**:
   SmallUNetの入力は生RGB(3ch, [0,1]float)、SegConvDLGN(c1)の入力は
   thermometer二値化(3*n_thresholds ch、既定9ch)。transcribe_conv2dは
   teacher(nn.Conv2d)のin_channelsに縛られるため、まずteacherの3ch空間で
   構造転写し(較正データはround()で0/1近似)、得られたindices_flatを
   「各RGBチャンネル → thermometerの中央閾値ビット」写像
   (c_new = mid_threshold*3 + rgb、ThermometerEncode3chのチャネル順
   [threshold0_RGB, threshold1_RGB, ...]に基づく)でthermometer空間へ
   再マップする。他の閾値ビットは無視される近似(agreementは3ch空間での
   値を報告、9ch空間での再マップ後agreementとは異なりうる点に注意)。

4. **out_channels(k)はteacher(SmallUNetのbase幅)から自動決定**:
   transcribe_conv2dはteacher層のout_channelsをそのまま転写するため、
   `--k`を独立指定せずteacherのbase幅(SmallUNet(base=...))と一致させる
   (c1のチャネル数不一致を避ける)。

5. **wiring='fixed'固定**: 転写はindices_flatバッファへ書き込むが、
   wiring='learnable'の実際のforwardはmapping_weights経由でindices_flatを
   参照しない(無効化される)。転写結果を反映させるため常にwiring='fixed'。

使用例:
    # 実SmallUNet ckptから転写(checkpoints/seg_unet_best.pt)
    uv run python experiments/transcribe_unet_to_segdlgn.py \\
        --unet-ckpt checkpoints/seg_unet_best.pt

    # wiring-onlyアブレーション込み(既定で両方出力)
    uv run python experiments/transcribe_unet_to_segdlgn.py \\
        --unet-ckpt checkpoints/seg_unet_best.pt --out-dir checkpoints/transcribed
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segmentation_models import SegConvDLGN, SmallUNet

from griffith_logic.conv import LutConv2d
from griffith_logic.convert.transcribe import fold_conv_bn, transcribe_conv2d
from griffith_logic.ir.weights import restore_lut_conv2d


def remap_rgb_indices_to_thermometer(
    indices_flat: np.ndarray, kernel_h: int, kernel_w: int, n_thresholds: int,
) -> np.ndarray:
    """3ch(RGB)空間の受容野フラット index を thermometer(3*n_thresholds ch)
    空間へ再マップする(W13a-5、docstring §3参照)。

    3ch側index = c*kH*kW + h*kW + w (c∈{0,1,2})。各RGBチャンネルを
    thermometerの中央閾値ビット(mid = n_thresholds // 2)のみへ対応付ける:
        c_new = mid*3 + c  (ThermometerEncode3chのチャネル順
        [threshold0_RGB, threshold1_RGB, ...] に基づく)
    h, w(カーネル内オフセット)はそのまま保持する。

    Args:
        indices_flat: (out_channels, n_leaves) の3ch空間index(int)
        kernel_h, kernel_w: カーネルサイズ
        n_thresholds: thermometer閾値数

    Returns:
        (out_channels, n_leaves) の thermometer空間index(int64)
    """
    idx = np.asarray(indices_flat, dtype=np.int64)
    hw = kernel_h * kernel_w
    c = idx // hw
    rest = idx % hw
    mid = n_thresholds // 2
    c_new = mid * 3 + c
    return c_new * hw + rest


def transcribe_c1_from_unet(
    unet: SmallUNet,
    calib_raw: torch.Tensor,
    *,
    n_thresholds: int = 3,
    tree_depth: int = 3,
    seed: int = 42,
) -> tuple[dict, dict]:
    """SmallUNetのenc1(Conv+BN)をBN fold→transcribe_conv2dでSegConvDLGN.c1へ
    構造転写する(W13a-5)。

    Args:
        unet: 訓練済みSmallUNet(eval()済み推奨。BN統計を使うため)
        calib_raw: (B, 3, H, W) の生[0,1]float較正画像(round()で0/1近似する)
        n_thresholds: SegConvDLGN側のthermometer閾値数
        tree_depth: 転写先LutConv2dのツリー深さ
        seed: 決定的シード(リーフサンプリング・多数決タイブレーク)

    Returns:
        (transcribed, info):
            transcribed: {
                'k': int（=teacher out_channels、SegConvDLGN構築に使う）,
                'indices_flat': (k, n_leaves) thermometer空間index,
                'truth_tables': hex文字列リスト,
            }
            info: transcribe_conv2dのinfo dict(approx_agreement等、3ch空間)
    """
    unet.eval()
    fused = fold_conv_bn(unet.enc1[0], unet.enc1[1])

    calib_bin = calib_raw.round().to(torch.float64)
    node, info = transcribe_conv2d(
        fused, calib_inputs=calib_bin, tree_depth=tree_depth, n_inputs=2,
        seed=seed,
    )

    kH, kW = node.params['kernel_size']
    indices_thermo = remap_rgb_indices_to_thermometer(
        node.params['indices_flat'], kH, kW, n_thresholds,
    )
    transcribed = {
        'k': node.params['out_channels'],
        'indices_flat': indices_thermo,
        'truth_tables': node.params['truth_tables'],
    }
    return transcribed, info


def build_transcribed_segdlgn(
    transcribed: dict,
    *,
    n_thresholds: int = 3,
    tree_depth: int = 3,
    readout_group: int = 32,
    input_channels_per_tree: int = 4,
    device: str = 'cpu',
    seed: int = 0,
) -> SegConvDLGN:
    """transcribe_c1_from_unetの出力をc1へ書き戻したSegConvDLGNを構築する。

    c2/c3/d2/d1は通常のSegConvDLGN初期化のまま(§本ファイルdocstring)。
    """
    torch.manual_seed(seed)
    k = transcribed['k']
    in_channels = 3 * n_thresholds
    model = SegConvDLGN(
        in_channels=in_channels, k=k, tree_depth=tree_depth,
        readout_group=readout_group,
        input_channels_per_tree=input_channels_per_tree,
        wiring='fixed', device=device,
    )
    restore_lut_conv2d(
        model.c1, transcribed['indices_flat'], transcribed['truth_tables'],
    )
    return model


def build_wiring_only_segdlgn(
    transcribed: dict,
    *,
    n_thresholds: int = 3,
    tree_depth: int = 3,
    readout_group: int = 32,
    input_channels_per_tree: int = 4,
    device: str = 'cpu',
    seed: int = 0,
) -> SegConvDLGN:
    """wiring-onlyアブレーション: c1のindices_flat(配線)だけ転写結果を使い、
    真理値表(weights)はSegConvDLGN既定の初期化(residual_init)のまま残す
    (知見「転写liftの実体はほぼ配線パターン」の検証用)。
    """
    torch.manual_seed(seed)
    k = transcribed['k']
    in_channels = 3 * n_thresholds
    model = SegConvDLGN(
        in_channels=in_channels, k=k, tree_depth=tree_depth,
        readout_group=readout_group,
        input_channels_per_tree=input_channels_per_tree,
        wiring='fixed', device=device,
    )
    idx = torch.as_tensor(
        np.asarray(transcribed['indices_flat'], dtype=np.int64),
    ).to(model.c1.indices_flat.device)
    with torch.no_grad():
        model.c1.indices_flat.copy_(idx)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='SmallUNet → SegConvDLGN 構造転写PoC(c1のみ、W13a-5)'
    )
    parser.add_argument('--unet-ckpt', type=str, required=True,
                        help='SmallUNet ckptパス(train_segmentation.py '
                             '--arch unet の保存形式)')
    parser.add_argument('--calib-n', type=int, default=256,
                        help='較正サンプル数(default: 256)')
    parser.add_argument('--tree-depth', type=int, default=3,
                        help='転写先LutConv2dのツリー深さ (default: 3)')
    parser.add_argument('--readout-group', type=int, default=32,
                        help='SegConvDLGNの読み出しグループサイズ (default: 32)')
    parser.add_argument('--input-channels-per-tree', type=int, default=4,
                        help='c2/c3/d2/d1(通常init)のinput_channels_per_tree '
                             '(default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='決定的シード (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='デバイス (default: cpu)')
    parser.add_argument('--out-dir', type=str, default='checkpoints/transcribed',
                        help='出力先ディレクトリ (default: checkpoints/transcribed)')
    args = parser.parse_args()

    ckpt = torch.load(args.unet_ckpt, map_location=args.device, weights_only=False)
    meta_args = ckpt.get('metadata', {}).get('args', {})
    base = meta_args.get('base', 48)
    n_thresholds = meta_args.get('n_thresholds', 3)
    img_size = meta_args.get('img_size', 64)
    n_classes = meta_args.get('n_classes', 2)

    unet = SmallUNet(in_channels=3, n_classes=n_classes, base=base)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    print(f'SmallUNet読み込み完了: base={base} n_thresholds={n_thresholds} '
          f'img_size={img_size}')

    # 較正データ(実Oxford Petがあれば使う、無ければランダム[0,1]で代用)
    torch.manual_seed(args.seed)
    try:
        from griffith_logic.datasets.segmentation import load_oxford_pet_seg
        train_set, _ = load_oxford_pet_seg(
            root='./data-oxford-pet', img_size=img_size, encoding='float',
            n_thresholds=n_thresholds, download=False,
        )
        n = min(args.calib_n, len(train_set))
        calib_raw = torch.stack([train_set[i][0] for i in range(n)])
        print(f'較正データ: Oxford Pet実データ {n}件')
    except Exception as e:  # データ未ダウンロード等はランダムで代用(PoC継続)
        print(f'較正データ: Oxford Pet読み込み失敗({e})、ランダム[0,1]画像で代用')
        calib_raw = torch.rand(args.calib_n, 3, img_size, img_size)

    transcribed, info = transcribe_c1_from_unet(
        unet, calib_raw, n_thresholds=n_thresholds,
        tree_depth=args.tree_depth, seed=args.seed,
    )
    print(f'c1構造転写完了: k={transcribed["k"]} '
          f'approx_agreement={info["approx_agreement"]:.4f} '
          f'(3ch teacher空間、n_leaves={info["n_leaves"]})')

    model_structural = build_transcribed_segdlgn(
        transcribed, n_thresholds=n_thresholds, tree_depth=args.tree_depth,
        readout_group=args.readout_group,
        input_channels_per_tree=args.input_channels_per_tree,
        device=args.device, seed=args.seed,
    )
    model_wiring_only = build_wiring_only_segdlgn(
        transcribed, n_thresholds=n_thresholds, tree_depth=args.tree_depth,
        readout_group=args.readout_group,
        input_channels_per_tree=args.input_channels_per_tree,
        device=args.device, seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    structural_path = os.path.join(args.out_dir, 'segdlgn_from_unet_structural.pt')
    wiring_only_path = os.path.join(args.out_dir, 'segdlgn_from_unet_wiring_only.pt')

    common_metadata = {
        'args': vars(args),
        'source_unet_ckpt': args.unet_ckpt,
        'approx_agreement_3ch': info['approx_agreement'],
        'per_channel_agreement_3ch': info['per_channel_agreement'].tolist(),
        'k': transcribed['k'],
        'n_thresholds': n_thresholds,
        'created_at': datetime.now(JST).isoformat(),
    }
    torch.save({
        'model_state_dict': model_structural.state_dict(),
        'step': 0, 'best_metric': 0.0,
        'metadata': {**common_metadata, 'variant': 'structural'},
    }, structural_path)
    torch.save({
        'model_state_dict': model_wiring_only.state_dict(),
        'step': 0, 'best_metric': 0.0,
        'metadata': {**common_metadata, 'variant': 'wiring_only'},
    }, wiring_only_path)

    print(f'保存(structural, c1関数保存): {structural_path}')
    print(f'保存(wiring_only, 配線のみ転写): {wiring_only_path}')
    print('使用例:')
    print(f'  uv run python experiments/train_segmentation.py --arch convdlgn '
          f'-k {transcribed["k"]} -d {args.tree_depth} '
          f'--init-from {structural_path}')


if __name__ == '__main__':
    main()
