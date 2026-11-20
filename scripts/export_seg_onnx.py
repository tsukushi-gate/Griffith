"""学習済み SmallUNet(セグ CNN ベースライン)→ ONNX(の ORT Web 比較用)。

DLGN(irflat)vs ONNX Runtime Web の同一 64x64 入力ベンチのため、SmallUNet の
best.pt を ONNX へ書き出す。入力は生 float[0,1] 画像(3,64,64)を
ImageNet 標準化する運用に合わせ、ここでは正規化を含めない素の logits を出力
(前処理はブラウザ側 conv_bench で行う)。

依存(onnx/onnxscript)は project extras に無いので ephemeral 実行推奨:
    uv run --with onnx --with onnxscript python scripts/export_seg_onnx.py \\
        checkpoints/seg_unet_best.pt tools/webbench/public/models/seg_unet.onnx
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments'))

from segmentation_models import SmallUNet  # noqa: E402


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/seg_unet_best.pt'
    out = (sys.argv[2] if len(sys.argv) > 2
           else 'tools/webbench/public/models/seg_unet.onnx')
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    data = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = SmallUNet(in_channels=3)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    dummy = torch.zeros(1, 3, size, size)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), out,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ONNX書き出し: {out} (best_miou={data.get("best_metric"):.4f}, '
          f'params={n_params:,}, size={Path(out).stat().st_size:,}B)')


if __name__ == '__main__':
    main()
