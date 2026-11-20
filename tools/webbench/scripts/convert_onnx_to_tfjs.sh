#!/usr/bin/env bash
# W6: ONNX (seg_unet / det_cnn_teacher) → TensorFlow.js graph model 変換。
#
# 2 段変換: ONNX --onnx2tf--> TF SavedModel --tensorflowjs_converter--> tfjs_graph_model
# 出力先: public/models/{seg_unet_tfjs,det_cnn_teacher_tfjs}/ (model.json + *.bin、gitignore下)
#
# プロジェクトの uv 環境は汚さない。スクラッチに 2 つの venv を作る:
#   - onnx2tf 用 (Python 3.12): onnx2tf + tensorflow(2.21) + tf_keras + onnxruntime
#   - tfjs 用   (Python 3.11): tensorflowjs==4.22.0  ※3.12 は tfdf の wheel が無い
#
# 依存地獄への対処(このスクリプトで自動適用):
#   (a) onnx2tf 2.6.x は既定 flatbuffer_direct が SavedModel を出さない
#       → `-tb tf_converter` バックエンドで SavedModel を出す(tf_keras 必須)。
#   (b) tensorflowjs 4.22 は import 時に tensorflow_decision_forests を読むが、
#       これが protobuf gencode / jax(ml_dtypes) を巻き込んで壊れる。素の CNN
#       変換には tfdf は不要なので __init__.py を無害化し、ml_dtypes>=0.5 を入れる。
#
# 使い方:  bash tools/webbench/scripts/convert_onnx_to_tfjs.sh [SCRATCH_DIR]
#   SCRATCH_DIR 省略時は $TMPDIR/webbench-tfconv。ONNX は public/models/ を読む。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # tools/webbench
MODELS="$HERE/public/models"
SCRATCH="${1:-${TMPDIR:-/tmp}/webbench-tfconv}"
mkdir -p "$SCRATCH"

PY312="$(command -v python3.12)"
PY311="$(command -v python3.11 || true)"
[ -z "$PY311" ] && { echo "python3.11 が必要(tfjs 4.22 は cp311 wheel のみ)"; exit 1; }
PIP() { if command -v uv >/dev/null; then uv pip install --python "$1" "${@:2}"; else "$1" -m pip install "${@:2}"; fi; }

# --- venv A: onnx2tf ---
VA="$SCRATCH/tfconv-venv"
if [ ! -x "$VA/bin/onnx2tf" ]; then
  "$PY312" -m venv "$VA"
  PIP "$VA/bin/python" onnx2tf tensorflow tf_keras onnx onnx-graphsurgeon sng4onnx onnxsim onnxruntime
fi

# --- venv B: tensorflowjs ---
VB="$SCRATCH/tfjs-venv"
if [ ! -x "$VB/bin/tensorflowjs_converter" ]; then
  "$PY311" -m venv "$VB"
  PIP "$VB/bin/python" "tensorflowjs==4.22.0"
  # tfdf を無害化(素の CNN 変換に不要。jax/protobuf 衝突を回避)
  TFDF="$(echo "$VB"/lib/python3.11/site-packages/tensorflow_decision_forests/__init__.py)"
  [ -f "$TFDF" ] && printf '__version__ = "stub"\n' > "$TFDF"
  PIP "$VB/bin/python" "ml_dtypes>=0.5.1"
fi

convert() {  # $1=onnx basename (no ext)  $2=tfjs out dir name
  local name="$1" out="$2"
  echo "=== $name: onnx2tf → SavedModel ==="
  rm -rf "$SCRATCH/${name}_tf"
  "$VA/bin/onnx2tf" -i "$MODELS/${name}.onnx" -o "$SCRATCH/${name}_tf" -tb tf_converter -osd
  echo "=== $name: SavedModel → tfjs_graph_model ==="
  rm -rf "$MODELS/$out"
  "$VB/bin/tensorflowjs_converter" --input_format=tf_saved_model \
    --output_format=tfjs_graph_model "$SCRATCH/${name}_tf" "$MODELS/$out"
  ls -la "$MODELS/$out"
}

convert seg_unet          seg_unet_tfjs
convert det_cnn_teacher   det_cnn_teacher_tfjs
echo "完了: $MODELS/{seg_unet_tfjs,det_cnn_teacher_tfjs}"
