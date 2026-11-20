"""MNIST テストデータの二値化 npz 生成(ROM/golden/精度gate の共通ソース)。

二値化はここで一度だけ確定する(ToTensor > 0.5、row-major 28x28 flatten =
学習時と同一規約。下流のスクリプトは npz を読むだけで推測しない)。

    uv run python tools/mdl-fpga/examples/mnist/gen_data.py [--n 1000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--out', default=str(HERE / 'mnist_test_1000.npz'))
    args = ap.parse_args()

    from griffith_logic.datasets.vision import load_mnist
    _, test_set = load_mnist(binarize_threshold=0.5)

    images = np.zeros((args.n, 784), dtype=np.uint8)
    labels = np.zeros(args.n, dtype=np.int64)
    for i in range(args.n):
        img, label = test_set[i]
        images[i] = img.reshape(-1).numpy().astype(np.uint8)  # row-major
        labels[i] = int(label)

    np.savez_compressed(args.out, images_bits=images, labels=labels)
    print(f'保存: {args.out}({args.n} 枚、二値化 >0.5、row-major)')
    print(f'  ラベル分布: {np.bincount(labels, minlength=10).tolist()}')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
