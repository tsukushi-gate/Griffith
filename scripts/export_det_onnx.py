"""学習済み CenterNetCNN(検出 teacher)→ ONNX(の ORT Web 比較用)。

DLGN(packed/irflat)vs ONNX Runtime Web の同一 96x96 入力ベンチのため、
CenterNetCNN の best.pt を ONNX へ書き出す。入力は二値化後 (T,96,96) の
[0,1] で、モデルは dict {hm(sigmoid), size(sigmoid), off(sigmoid)} を返す。
ONNX は tuple 出力(hm, size, off)へ写す(前処理・decode はブラウザ側)。

依存(onnx/onnxscript)は project extras に無いので ephemeral 実行推奨:
    uv run --with onnx --with onnxscript python scripts/export_det_onnx.py \\
        scratch_ckpt/cnn_w64_teacher_best.pt \\
        tools/webbench/public/models/det_cnn_teacher.onnx
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments'))

from detection_model_cnn import CenterNetCNN  # noqa: E402


class _TupleWrap(nn.Module):
    """dict 出力を (hm, size, off) tuple へ変換(ONNX の名前付き出力用)。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        o = self.model(x)
        return o['hm'], o['size'], o['off']


def main():
    ckpt = (sys.argv[1] if len(sys.argv) > 1
            else 'scratch_ckpt/cnn_w64_teacher_best.pt')
    out = (sys.argv[2] if len(sys.argv) > 2
           else 'tools/webbench/public/models/det_cnn_teacher.onnx')

    data = torch.load(ckpt, map_location='cpu', weights_only=False)
    a = (data.get('metadata', {}) or {}).get('args', {}) or {}
    n_thr = a.get('n_thresholds', 3)
    size = a.get('canvas_size', 96)
    is_color = a.get('dataset') in ('multimnist_color', 'fewclass', 'face')
    # color/fewclass(少クラス実物体検出)/face(WIDER FACE顔検出):
    # teacher(CNN)は生RGB 3chを消費(二値化ではなくLUT studentとの情報非対称を
    # 意図した設計)。グレースケールは従来通り二値化後 T ch
    in_channels = 3 if is_color else 1 * n_thr
    if a.get('dataset') == 'fewclass':
        n_classes = len([c for c in a.get('fewclass_classes', '').split(',') if c.strip()])
    elif a.get('dataset') == 'face':
        # face(WIDER FACE)は1クラス固定。a.get('n_classes', 10)は
        # --n-classes という argparse引数が存在しないため常に10へフォール
        # バックする罠があり、face専用分岐が必須(export_webbench.pyと同じ罠)
        n_classes = 1
    else:
        n_classes = a.get('n_classes', 10)
    model = CenterNetCNN(
        n_classes=n_classes, in_channels=in_channels,
        width=a.get('cnn_width', 64), head_ch=a.get('cnn_head_ch', 64))
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    wrapped = _TupleWrap(model).eval()

    dummy = torch.zeros(1, in_channels, size, size)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (dummy,), out,
        input_names=['input'], output_names=['hm', 'size', 'off'],
        dynamic_axes={'input': {0: 'batch'}, 'hm': {0: 'batch'},
                      'size': {0: 'batch'}, 'off': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ONNX書き出し: {out} (best_mAP={data.get("best_metric"):.4f}, '
          f'in_ch={in_channels}, size={size}, params={n_params:,}, '
          f'bytes={Path(out).stat().st_size:,})')


if __name__ == '__main__':
    main()
