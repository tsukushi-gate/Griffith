// CenterNet デコード。experiments/detection_utils.py の
//   nms_by_maxpool + decode_detections をビット等価に移植する。
//
// 入力は ConvRuntime.scores() が返す「3 ヘッド連結の生 popcount 整数列」
// (dag.py の多出力 concat 規約: hm|size|off、各 NCHW flatten)。ヘッドごとの
// offset/scale は DetReadout(export_webbench.py det-ckpt が埋める)で供給する:
//   hm_prob   = popcount * hm.scale     (= popcount / hm_group, [0,1])
//   size_norm = popcount * size.scale
//   off       = popcount * off.scale
//
// 一致検証(det_decode.test.ts): Python decode_detections が同一 hm/size/off
// (float64)から出力する箱列と、この decode の箱列が許容誤差内で一致する。
// Python 参照は float64・-inf パディング maxpool・位置依存 1e-6 タイブレークで
// あり、ここも float64(JS number)で同一手順を踏むため厳密一致に近い。
import type { DetReadout } from './types';

/** ピクセル座標(入力画像座標系)の検出箱。 */
export interface DetBoxPx {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cls: number;
  score: number;
}

/** 生 popcount 列 → ヘッドを実数値に展開した float64 平面を取り出す。
    戻り値 plane[o*H*W + y*W + x](NCHW flatten、o=チャネル)。 */
function headPlane(
  scores: ArrayLike<number>,
  head: { offset: number; n_out: number; h: number; w: number; scale: number },
): Float64Array {
  const n = head.n_out * head.h * head.w;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = scores[head.offset + i] * head.scale;
  return out;
}

/** 3×3 max pooling による NMS(CenterNet 方式)。detection_utils.nms_by_maxpool
    と同一: 位置依存 1e-6 ノイズでタイブレーク、境界は -inf パディング相当
    (窓内の有効近傍のみで max)。非ピークは 0 を返す。 */
function nmsByMaxpool(hm: Float64Array, C: number, H: number, W: number): Float64Array {
  const hw = H * W;
  const out = new Float64Array(C * hw);
  for (let c = 0; c < C; c++) {
    const base = c * hw;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const pos = y * W + x;
        const self = hm[base + pos] + pos * 1e-6;
        // 3×3 有効近傍の最大(タイブレーク込み)。-inf パディング=範囲外を除外。
        let mx = -Infinity;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            const npos = yy * W + xx;
            const v = hm[base + npos] + npos * 1e-6;
            if (v > mx) mx = v;
          }
        }
        // keep = (hmax == heatmap_tb) のとき元の値を残す(タイブレーク前の hm)
        out[base + pos] = mx === self ? hm[base + pos] : 0;
      }
    }
  }
  return out;
}

/** 単一画像(batch=1)の生スコア列をデコードして箱列(ピクセル座標)を返す。
    Python decode_detections と同じ topK→閾値→size/off 復元手順。 */
export function decodeDetections(
  scores: ArrayLike<number>,
  ro: DetReadout,
): DetBoxPx[] {
  const { hm: hmH, size: szH, off: offH } = ro.heads;
  const C = hmH.n_out;
  const H = hmH.h;
  const W = hmH.w;
  const hw = H * W;
  const stride = ro.output_stride;

  const hmProb = headPlane(scores, hmH);
  const sizeP = headPlane(scores, szH); // [2*hw] (w,h)
  const offP = headPlane(scores, offH); // [2*hw] (dx,dy)

  const nms = nmsByMaxpool(hmProb, C, H, W);

  // scores_flat(C*hw)を降順ソート→上位 K→閾値。topk と同順にするため
  // 値降順(同値は元インデックス昇順で安定)にする。
  const n = C * hw;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  // 部分ソートは不要(n は小さい)。安定降順ソート。
  const idxArr = Array.from(order);
  idxArr.sort((a, b) => (nms[b] - nms[a]) || (a - b));

  const K = Math.min(ro.top_k, n);
  const boxes: DetBoxPx[] = [];
  for (let r = 0; r < K; r++) {
    const flat = idxArr[r];
    const score = nms[flat];
    if (!(score > ro.threshold)) continue;
    const cls = Math.floor(flat / hw);
    const spatial = flat % hw;
    const yy = Math.floor(spatial / W);
    const xx = spatial % W;

    const offx = offP[spatial]; // off ch0
    const offy = offP[hw + spatial]; // off ch1
    const wNorm = sizeP[spatial]; // size ch0
    const hNorm = sizeP[hw + spatial]; // size ch1

    const cx = (xx + offx) * stride;
    const cy = (yy + offy) * stride;
    const wPx = wNorm * stride * W;
    const hPx = hNorm * stride * H;
    boxes.push({
      x1: cx - wPx / 2,
      y1: cy - hPx / 2,
      x2: cx + wPx / 2,
      y2: cy + hPx / 2,
      cls,
      score,
    });
  }
  return boxes;
}
