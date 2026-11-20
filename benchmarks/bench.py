"""ベンチマークスイートのスタンドアロン入口。

実体は griffith_logic/training/bench_cli.py(`griffith bench` と同一)。
使い方:
    uv run python benchmarks/bench.py configs/recipes/logic_xor.yaml --skip-train
    uv run python benchmarks/bench.py outputs/<run>/checkpoints/best.pt

結果は benchmarks/results/<日付>_<name>.json に固定スキーマ
{name, timestamp, config, metrics: {hard_acc, total_gates, latency_ms_b1,
latency_ms_b128}, env: {python, torch, platform}} で保存される。
"""

import sys

from griffith_logic.cli import main

if __name__ == '__main__':
    sys.exit(main(['bench', *sys.argv[1:]]))
