// 前処理(WB-4): int16復元→チャネル別標準化→thermometer二値化。
// Python側 datasets/har.py encode_thermometer とビット単位で一致する契約:
//   bit index = ((ch*L)+t)*T+thr、判定は strict greater(z > threshold)。
// 契約の検証は preproc.test.ts(golden fixture照合)。
import type { Preproc, TestData, TestMeta } from './types';

/** int16(little-endian)バッファを float32(生値)へ復元する */
export function decodeI16(buf: ArrayBuffer, meta: TestMeta): TestData {
  const i16 = new Int16Array(buf);
  const [n, c, l] = meta.shape.length === 3
    ? meta.shape
    : [meta.shape[0], meta.shape[1], 1];
  const x = new Float32Array(i16.length);
  const scale = meta.scale;
  for (let i = 0; i < i16.length; i++) x[i] = Math.fround(i16[i] * scale);
  return { meta, n, chLen: c * l, x };
}

/** 1サンプル(生値 [C*L])をthermometerビット列 [C*L*T] へ符号化する。
    outを渡すと再利用(アロケーション回避)。 */
export function encodeThermometer(
  xRow: Float32Array,
  p: Preproc,
  out?: Uint8Array,
): Uint8Array {
  const c = p.mean.length;
  const l = xRow.length / c;
  const t = p.thresholds[0].length;
  const bits = out ?? new Uint8Array(c * l * t);
  let bi = 0;
  for (let ch = 0; ch < c; ch++) {
    const mean = p.mean[ch];
    const invStd = 1 / p.std[ch];
    const thr = p.thresholds[ch];
    for (let ti = 0; ti < l; ti++) {
      const z = (xRow[ch * l + ti] - mean) * invStd;
      for (let k = 0; k < t; k++) {
        bits[bi++] = z > thr[k] ? 1 : 0;
      }
    }
  }
  return bits;
}

/** サンプルiの生値ビューを返す(コピーなし) */
export function sampleView(data: TestData, i: number): Float32Array {
  return data.x.subarray(i * data.chLen, (i + 1) * data.chLen);
}

/** 標準化のみ(ONNX側前処理)。出力 [C*L] float32 */
export function standardize(
  xRow: Float32Array,
  mean: number[],
  std: number[],
  out?: Float32Array,
): Float32Array {
  const c = mean.length;
  const l = xRow.length / c;
  const z = out ?? new Float32Array(xRow.length);
  for (let ch = 0; ch < c; ch++) {
    const m = mean[ch];
    const invStd = 1 / std[ch];
    for (let ti = 0; ti < l; ti++) {
      z[ch * l + ti] = (xRow[ch * l + ti] - m) * invStd;
    }
  }
  return z;
}

/** argmax(タイは最初の最大値=np.argmaxと同じ) */
export function argmax(scores: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}
