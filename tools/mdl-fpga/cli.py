"""mdl-fpga CLI エントリポイント(自己完結ディレクトリ用の起動シム)。

    uv run python tools/mdl-fpga/cli.py doctor
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdl_fpga.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
