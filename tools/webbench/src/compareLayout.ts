// W6: TF.js(onnx2tf 変換の graph model は NHWC)出力を DLGN/ONNX の NCHW
// 規約へ橋渡しする純関数群。ort/tfjs に依存しないのでユニットテスト可能。

/** logits(NHWC[1,H,W,C] または NCHW[1,C,H,W])→ per-pixel argmax ラベル[H*W]。
    出力空間は正方(H==W)なので、隣接して一致する 2 次元で並びを判定する:
    NHWC は index 1,2 が H,W(一致)、NCHW は index 2,3 が H,W(一致)。 */
export function argmaxSpatial(data: ArrayLike<number>, shape: number[]): Int32Array {
  const nhwc = shape.length === 4 && shape[1] === shape[2] && shape[3] !== shape[2];
  if (nhwc) {
    const H = shape[1];
    const W = shape[2];
    const C = shape[3];
    const hw = H * W;
    const out = new Int32Array(hw);
    for (let p = 0; p < hw; p++) {
      let best = 0;
      let bv = data[p * C];
      for (let c = 1; c < C; c++) {
        const v = data[p * C + c];
        if (v > bv) { bv = v; best = c; }
      }
      out[p] = best;
    }
    return out;
  }
  const C = shape[1];
  const H = shape[2];
  const W = shape[3];
  const hw = H * W;
  const out = new Int32Array(hw);
  for (let p = 0; p < hw; p++) {
    let best = 0;
    let bv = data[p];
    for (let c = 1; c < C; c++) {
      const v = data[c * hw + p];
      if (v > bv) { bv = v; best = c; }
    }
    out[p] = best;
  }
  return out;
}

/** NHWC 平面 data[p*C + c]([1,H,W,C])→ NCHW 平面 plane[c*H*W + p]。 */
export function nhwcToNchwPlane(
  data: ArrayLike<number>,
  H: number,
  W: number,
  C: number,
): Float64Array {
  const hw = H * W;
  const plane = new Float64Array(C * hw);
  for (let p = 0; p < hw; p++) {
    for (let c = 0; c < C; c++) plane[c * hw + p] = data[p * C + c];
  }
  return plane;
}

/** NCHW 特徴 bits[c*H*W + p] → NHWC feat[p*C + c]([1,H,W,C] 入力用)。 */
export function nchwToNhwc(
  bits: ArrayLike<number>,
  H: number,
  W: number,
  C: number,
): Float32Array {
  const hw = H * W;
  const out = new Float32Array(hw * C);
  for (let c = 0; c < C; c++) {
    for (let p = 0; p < hw; p++) out[p * C + c] = bits[c * hw + p];
  }
  return out;
}
