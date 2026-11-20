#!/usr/bin/env bash
# XOR を学習 → バンドル化 → samples.json 生成 → iOS デモへ取り込む(1コマンド)。
# 実行: bash examples/31_deploy_ios/setup_xor.sh   (リポジトリルートから)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TMPDIR:-/tmp}/griffith_ios_xor"
mkdir -p "$OUT"

uv run python examples/06_deploy_c/export_model.py            # out_examples/xor_model.json
uv run griffith deploy bundle out_examples/xor_model.json -o "$OUT/xor.bundle.json"
uv run python examples/31_deploy_ios/make_xor_samples.py out_examples/xor_model.json "$OUT/samples.json"
uv run python examples/31_deploy_ios/gen_model.py xor --bundle "$OUT/xor.bundle.json" --samples "$OUT/samples.json"
echo "次: xcodebuild -project examples/31_deploy_ios/GriffithDemo.xcodeproj -scheme GriffithDemo -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build"
