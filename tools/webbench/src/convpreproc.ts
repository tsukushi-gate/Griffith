// 画像入力の前処理(W4): チャネル別標準化 + thermometer二値化(NCHW)。
// Python側 scripts/export_webbench.py encode_image_thermometer とビット単位で
// 一致する契約:
//   出力ch = c*T + thr(rgb-major)、bit index = ((c*T+thr)*H + y)*W + x
//   判定は strict greater(z > threshold)
// 契約の検証は conv.test.ts(thermo_golden.json 照合)。
export interface ImageThermoParams {
  mean: number[]; // [C](RGBチャネル別)
  std: number[]; // [C]
  thresholds: number[][]; // [C][T](標準化後のしきい、ch内単調増加)
  height: number;
  width: number;
}

/** 1画像(生値 [C*H*W] NCHW)→ thermometerビット列 [C*T*H*W]。
    outを渡すと再利用(アロケーション回避)。 */
export function encodeImageThermometer(
  xRow: Float32Array,
  p: ImageThermoParams,
  out?: Uint8Array | Int32Array,
): Uint8Array | Int32Array {
  const c = p.mean.length;
  const hw = p.height * p.width;
  const t = p.thresholds[0].length;
  const bits = out ?? new Uint8Array(c * t * hw);
  for (let ch = 0; ch < c; ch++) {
    const mean = p.mean[ch];
    const invStd = 1 / p.std[ch];
    const thr = p.thresholds[ch];
    for (let k = 0; k < t; k++) {
      const outBase = (ch * t + k) * hw;
      const th = thr[k];
      for (let i = 0; i < hw; i++) {
        const z = (xRow[ch * hw + i] - mean) * invStd;
        bits[outBase + i] = z > th ? 1 : 0;
      }
    }
  }
  return bits;
}

// ---------------------------------------------------------------------------
// 低レベル特徴検出器つき前処理(preproc.kind === 'image_lowlevel')。
// Conv-DLGN(Petersen+ NeurIPS'24) §3.5 の "edge and curvature detector kernels
// with thresholds" に対応する固定(非学習)前処理。
//
// Python の griffith_logic/encoding/image_features.py::encode_image_features と
// **ブロック順・チャネル順が厳密に一致**する契約:
//   [thermo][edge][curv][chroma]
//   thermo: thr-major   → ch = k*C + c        (thermometer版の rgb-major とは違う)
//   edge  : offset → thr → sign(+,-) → c
//   curv  : offset → thr → sign(+,-) → c
//   chroma: pair → thr → sign(+,-)、その後 saturation を thr 順に
// thermo が thr-major なので、thermometer 版で必要だった rgb↔thr 置換permuteは
// **不要**(export 側で抑止済み)。契約の検証は convpreproc.test.ts。
export interface ImageLowLevelParams {
  mean: number[]; // [C]
  std: number[]; // [C]
  thresholds: number[][]; // [C][T] thermometer
  edge_thresholds: number[];
  curv_thresholds: number[];
  chroma_thresholds: number[];
  edge_offsets: number[][]; // [[dy,dx],...]
  curv_offsets: number[][];
  chroma_pairs: number[][]; // [[i,j],...]
  height: number;
  width: number;
}

/** replicate padding つきの (dy,dx) シフト読み出し: out[y,x] = src[y+dy, x+dx]。 */
function shiftAt(
  src: Float32Array,
  chBase: number,
  h: number,
  w: number,
  y: number,
  x: number,
  dy: number,
  dx: number,
): number {
  let yy = y + dy;
  let xx = x + dx;
  if (yy < 0) yy = 0;
  else if (yy >= h) yy = h - 1;
  if (xx < 0) xx = 0;
  else if (xx >= w) xx = w - 1;
  return src[chBase + yy * w + xx];
}

/** 1画像(生値 [C*H*W] NCHW、[0,1])→ 低レベル特徴ビット列。 */
export function encodeImageLowLevel(
  xRow: Float32Array,
  p: ImageLowLevelParams,
  out?: Uint8Array | Int32Array,
): Uint8Array | Int32Array {
  const C = p.mean.length;
  const h = p.height;
  const w = p.width;
  const hw = h * w;
  const T = p.thresholds[0]?.length ?? 0;
  const nEdge = C * p.edge_offsets.length * p.edge_thresholds.length * 2;
  const nCurv = C * p.curv_offsets.length * p.curv_thresholds.length * 2;
  const nChroma =
    p.chroma_pairs.length * p.chroma_thresholds.length * 2 + p.chroma_thresholds.length;
  const bits = out ?? new Uint8Array((C * T + nEdge + nCurv + nChroma) * hw);
  let ch = 0; // 出力チャネルカーソル

  // --- thermo(thr-major: k が外、c が内) ---
  for (let k = 0; k < T; k++) {
    for (let c = 0; c < C; c++) {
      const th = p.thresholds[c][k];
      const mean = p.mean[c];
      const invStd = 1 / p.std[c];
      const src = c * hw;
      const dst = ch++ * hw;
      for (let i = 0; i < hw; i++) {
        bits[dst + i] = (xRow[src + i] - mean) * invStd > th ? 1 : 0;
      }
    }
  }

  // --- edge: 1階差分 d = x[p+(dy,dx)] - x[p] ---
  // Python: for offset: for t: cat[(d>t) (C,H,W), (d<-t) (C,H,W)]
  // → チャネル順は offset → t → sign(+,-) → c。sign ブロックは C チャネル幅。
  for (const [dy, dx] of p.edge_offsets) {
    for (const t of p.edge_thresholds) {
      const chPos = ch;
      ch += C;
      const chNeg = ch;
      ch += C;
      for (let c = 0; c < C; c++) {
        const base = c * hw;
        const dstPos = (chPos + c) * hw;
        const dstNeg = (chNeg + c) * hw;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * w + x;
            const d = shiftAt(xRow, base, h, w, y, x, dy, dx) - xRow[base + i];
            bits[dstPos + i] = d > t ? 1 : 0;
            bits[dstNeg + i] = d < -t ? 1 : 0;
          }
        }
      }
    }
  }

  // --- curv: 2階差分 d = x[p-(dy,dx)] - 2*x[p] + x[p+(dy,dx)] ---
  for (const [dy, dx] of p.curv_offsets) {
    for (const t of p.curv_thresholds) {
      const chPos = ch;
      ch += C;
      const chNeg = ch;
      ch += C;
      for (let c = 0; c < C; c++) {
        const base = c * hw;
        const dstPos = (chPos + c) * hw;
        const dstNeg = (chNeg + c) * hw;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * w + x;
            const d =
              shiftAt(xRow, base, h, w, y, x, -dy, -dx) -
              2 * xRow[base + i] +
              shiftAt(xRow, base, h, w, y, x, dy, dx);
            bits[dstPos + i] = d > t ? 1 : 0;
            bits[dstNeg + i] = d < -t ? 1 : 0;
          }
        }
      }
    }
  }

  // --- chroma: チャネル間差 + 彩度(max-min)。各1チャネル ---
  for (const [ci, cj] of p.chroma_pairs) {
    for (const t of p.chroma_thresholds) {
      const dstPos = ch++ * hw;
      const dstNeg = ch++ * hw;
      const bi = ci * hw;
      const bj = cj * hw;
      for (let i = 0; i < hw; i++) {
        const d = xRow[bi + i] - xRow[bj + i];
        bits[dstPos + i] = d > t ? 1 : 0;
        bits[dstNeg + i] = d < -t ? 1 : 0;
      }
    }
  }
  // saturation は全 chroma ペアの後に、しきい値順でまとめて置かれる(Python 同順)
  for (const t of p.chroma_thresholds) {
    const dst = ch++ * hw;
    for (let i = 0; i < hw; i++) {
      let mx = -Infinity;
      let mn = Infinity;
      for (let c = 0; c < C; c++) {
        const v = xRow[c * hw + i];
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
      bits[dst + i] = mx - mn > t ? 1 : 0;
    }
  }

  if (ch !== C * T + nEdge + nCurv + nChroma) {
    throw new Error(
      `encodeImageLowLevel: チャネル数不一致 ch=${ch} != ${C * T + nEdge + nCurv + nChroma}`,
    );
  }
  return bits;
}

/** doc.preproc の最小共通形。kind でエンコーダを選ぶための判別用。 */
export interface ImagePreprocDoc {
  kind: string;
  mean: number[];
  std: number[];
  thresholds: number[][];
  height: number;
  width: number;
  edge_thresholds?: number[];
  curv_thresholds?: number[];
  chroma_thresholds?: number[];
  edge_offsets?: number[][];
  curv_offsets?: number[][];
  chroma_pairs?: number[][];
}

/** 画像前処理の唯一のディスパッチ点。
 *
 * `preproc.kind` を見て thermometer / 低レベル特徴つき を選ぶ。
 * **エンコーダを実装しても呼び出し側に繋がなければ配信できない**という穴を
 * 一箇所に閉じ込めるため、デモ/ベンチは必ずこの関数を通すこと
 * (で encodeImageLowLevel を実装+ゴールデン検証したのに
 *  detdemo が encodeImageThermometer 直呼びのままで、勝ったモデルを
 *  出荷できない状態になっていた)。
 */
export function encodeImageAuto(
  xRow: Float32Array,
  p: ImagePreprocDoc,
  out?: Uint8Array | Int32Array,
): Uint8Array | Int32Array {
  if (p.kind === 'image_lowlevel') {
    return encodeImageLowLevel(
      xRow,
      {
        mean: p.mean,
        std: p.std,
        thresholds: p.thresholds,
        edge_thresholds: p.edge_thresholds ?? [],
        curv_thresholds: p.curv_thresholds ?? [],
        chroma_thresholds: p.chroma_thresholds ?? [],
        edge_offsets: p.edge_offsets ?? [],
        curv_offsets: p.curv_offsets ?? [],
        chroma_pairs: p.chroma_pairs ?? [],
        height: p.height,
        width: p.width,
      },
      out,
    );
  }
  return encodeImageThermometer(xRow, p, out);
}

/** canvas RGBA(Uint8ClampedArray)→ NCHW float [3*H*W]([0,1]画素)。 */
export function rgbaToNchw(
  rgba: Uint8ClampedArray,
  h: number,
  w: number,
  out?: Float32Array,
): Float32Array {
  const hw = h * w;
  const x = out ?? new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    x[i] = rgba[i * 4] / 255;
    x[hw + i] = rgba[i * 4 + 1] / 255;
    x[2 * hw + i] = rgba[i * 4 + 2] / 255;
  }
  return x;
}
