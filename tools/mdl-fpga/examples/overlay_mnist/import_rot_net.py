"""import_net.py の旧名(計算型配線 = rot 系だけを受け付ける薄いラッパ)。

overlay が保存型(自由配線)にも対応したので、取込器は配線種別を問わない
`import_net.py` に移した。本ファイルは旧コマンド列を壊さないための互換入口で、
`--wiring computed` を既定にした import_net.py をそのまま呼ぶ(rot / 引き伸ばし
rot の閉形式に外れる IR は従来どおり言い切って拒否する)。

    uv run python tools/mdl-fpga/examples/overlay_mnist/import_rot_net.py \\
        --ir path/to/mnist_p6_seed1.ir.json --expect-correct 954

新しく書くなら import_net.py を直接使う(既定 --wiring auto)。
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from import_net import main as _main   # noqa: E402


def main() -> int:
    if not any(a.startswith('--wiring') for a in sys.argv[1:]):
        sys.argv.append('--wiring=computed')
    return _main()


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
