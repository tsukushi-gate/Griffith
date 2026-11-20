// 重み共有カーネルスイープ方式の conv2d 実行(packedエンジンの心臓部)。
//
// lowerConv.ts の「全空間展開」(k32モデルで4.6M LUT/枚)をやめ、conv層を
// 「out_channelごとの木プログラム(重み共有ゲート)× 空間窓スライド」として実行する。
// 空間位置(H*W)を Int32 ワードへ横パック(32位置=1ワード、各ビット=空間位置)し、
// ゲート演算をビット並列化する。境界/パディングは「窓外=0で gather → 元TTのまま評価」
// で処理する(TTをbit_j=0で評価するのは lowering._specialize_zero と厳密に一致するため、
// 出力は展開版とビット単位で同一)。
//
// 正しさの錨: conv_golden.json / seg_e2e_golden.json を packed/irflat 両エンジンで通す。
import { ttParse } from '../shared';
import type { Conv2dParams, PermuteParams, UpSample2dParams } from '../shared';
import type { NodeValue } from './irflat';

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** 2入力ゲート(tt: 0..15)をワード並列で適用。a=入力0(bit0), b=入力1(bit1)。
    codegen.ts gateExpr と同一の16式(ビット単位のブール演算)。 */
function gate2(tt: number, a: number, b: number): number {
  switch (tt & 0xf) {
    case 0x0: return 0;
    case 0x1: return ~(a | b);
    case 0x2: return a & ~b;
    case 0x3: return ~b;
    case 0x4: return ~a & b;
    case 0x5: return ~a;
    case 0x6: return a ^ b;
    case 0x7: return ~(a & b);
    case 0x8: return a & b;
    case 0x9: return ~(a ^ b);
    case 0xa: return a;
    case 0xb: return a | ~b;
    case 0xc: return b;
    case 0xd: return ~a | b;
    case 0xe: return a | b;
    case 0xf: return -1;
    default: return 0;
  }
}

/** 1つのタップ(入力ch c, 窓オフセット dy,dx)の gather 面を空間位置パックで作る。
    出力位置 po=(oy*WOut+ox) のビットに、入力位置 (oy*sH-pH+dy, ox*sW-pW+dx) の
    0/1(窓外は0)を書く。行ごと丸ごと窓外なら WOut ぶんスキップ。 */
function buildTapPlane(
  xd: Int32Array, cbase: number, dy: number, dx: number,
  H: number, W: number, HOut: number, WOut: number,
  sH: number, sW: number, pH: number, pW: number, wordsOut: number,
): Int32Array {
  const plane = new Int32Array(wordsOut);
  let po = 0;
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy * sH - pH + dy;
    if (y < 0 || y >= H) { po += WOut; continue; }
    const rowBase = cbase + y * W;
    for (let ox = 0; ox < WOut; ox++, po++) {
      const xx = ox * sW - pW + dx;
      if (xx < 0 || xx >= W) continue;
      if (xd[rowBase + xx]) plane[po >> 5] |= 1 << (po & 31);
    }
  }
  return plane;
}

/** conv2d をネイティブ実行(空間位置パック)。x は [batch*(C*H*W)] の 0/1。
    出力は lowering の展開版と同一の [batch*(OC*HOut*WOut)] NCHW flatten。 */
export function evalConv2dPacked(
  p: Conv2dParams, x: NodeValue, batch: number,
): NodeValue {
  const [, H, W] = p.in_shape;
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const OC = p.out_channels;
  const N = p.n_inputs;
  const depth = p.tree_depth;
  if (N !== 2) throw new Error(`packed conv は n_inputs=2 のみ対応(実際: ${N})`);
  const nLeaves = N ** depth;
  const nInternal = (nLeaves - 1) / (N - 1);
  const Lin = H * W;
  const Lout = HOut * WOut;
  const wordsOut = (Lout + 31) >> 5;
  const inW = x.width;
  const outW = OC * Lout;
  const khw = kH * kW;
  const idxf = p.indices_flat; // [OC][nLeaves]

  // per-oc,leaf のタップ(c,dy,dx)を先に解く(batch非依存)。
  const leafC = new Int32Array(OC * nLeaves);
  const leafDy = new Int32Array(OC * nLeaves);
  const leafDx = new Int32Array(OC * nLeaves);
  for (let oc = 0; oc < OC; oc++) {
    const row = idxf[oc];
    for (let k = 0; k < nLeaves; k++) {
      const pp = row[k];
      const rem = pp % khw;
      leafC[oc * nLeaves + k] = Math.floor(pp / khw);
      leafDy[oc * nLeaves + k] = Math.floor(rem / kW);
      leafDx[oc * nLeaves + k] = rem % kW;
    }
  }
  // 木の各ゲートTT(0..15)を数値化(batch/oc横断で1回)。
  const ttNum = new Uint8Array(OC * nInternal);
  for (let i = 0; i < ttNum.length; i++) {
    ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  }

  const data = new Int32Array(batch * outW);
  const leafPlanes: Int32Array[] = new Array(nLeaves);
  const maxBank = Math.max(1, nLeaves >> 1);
  const bankA = Array.from({ length: maxBank }, () => new Int32Array(wordsOut));
  const bankB = Array.from({ length: maxBank }, () => new Int32Array(wordsOut));
  const tapCache = new Map<number, Int32Array>();

  for (let b = 0; b < batch; b++) {
    const xoff = b * inW;
    const ooff = b * outW;
    tapCache.clear();

    for (let oc = 0; oc < OC; oc++) {
      const lb = oc * nLeaves;
      for (let k = 0; k < nLeaves; k++) {
        const c = leafC[lb + k];
        const dy = leafDy[lb + k];
        const dx = leafDx[lb + k];
        const key = (c * khw + dy * kW + dx); // (c,dy,dx) 一意
        let plane = tapCache.get(key);
        if (plane === undefined) {
          plane = buildTapPlane(
            x.data, xoff + c * Lin, dy, dx,
            H, W, HOut, WOut, sH, sW, pH, pW, wordsOut,
          );
          tapCache.set(key, plane);
        }
        leafPlanes[k] = plane;
      }

      // 木をボトムアップ評価(ping-pongバンクで再利用、割当なし)。
      let src = leafPlanes;
      let srcLen = nLeaves;
      let dstBank = bankA;
      let gOffset = 0;
      while (srcLen > 1) {
        const gOut = srcLen >> 1;
        for (let g = 0; g < gOut; g++) {
          const tt = ttNum[oc * nInternal + gOffset + g];
          const a = src[g << 1];
          const bp = src[(g << 1) + 1];
          const dst = dstBank[g];
          for (let wi = 0; wi < wordsOut; wi++) dst[wi] = gate2(tt, a[wi], bp[wi]);
        }
        gOffset += gOut;
        src = dstBank;
        srcLen = gOut;
        dstBank = dstBank === bankA ? bankB : bankA;
      }

      const outPlane = src[0];
      const base = ooff + oc * Lout;
      for (let po = 0; po < Lout; po++) {
        data[base + po] = (outPlane[po >> 5] >>> (po & 31)) & 1;
      }
    }
  }
  return { data, width: outW };
}

/** upsample2d(nearest)をネイティブ実行。indices配列を作らず直接
    out[c,y,x]=in[c,y//s,x//s]をコピーする(lowering._upsample2d_node_to_lut
    と同一index計算をwidth配列を経由せず実行するだけ)。 */
export function evalUpsample2dPacked(
  p: UpSample2dParams, x: NodeValue, batch: number,
): NodeValue {
  const [C, H, W] = p.in_shape;
  const s = p.scale;
  const HOut = H * s;
  const WOut = W * s;
  const outW = C * HOut * WOut;
  const inW = x.width;
  const data = new Int32Array(batch * outW);
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * outW;
    for (let c = 0; c < C; c++) {
      const cbaseIn = xo + c * H * W;
      const cbaseOut = oo + c * HOut * WOut;
      for (let y = 0; y < HOut; y++) {
        const yin = (y / s) | 0;
        const rowIn = cbaseIn + yin * W;
        const rowOut = cbaseOut + y * WOut;
        for (let xx = 0; xx < WOut; xx++) {
          data[rowOut + xx] = x.data[rowIn + ((xx / s) | 0)];
        }
      }
    }
  }
  return { data, width: outW };
}

/** permuteをネイティブ実行。出力位置→入力位置の写像(srcOf)を
    batch非依存で1回だけ構築し(conv2dの木プログラム事前計算と同様)、
    各サンプルはgatherのみ行う(lowering._permute_node_to_lutと同一
    index計算をindices配列に持たせず直接使う)。 */
export function evalPermutePacked(
  p: PermuteParams, x: NodeValue, batch: number,
): NodeValue {
  const dims = p.in_shape;
  const perm = p.perm;
  const n = dims.length;
  const width = dims.reduce((a, b) => a * b, 1);

  const inStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * dims[i + 1];
  const outDims = perm.map((q) => dims[q]);
  const outStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];

  const srcOf = new Int32Array(width);
  for (let outIdx = 0; outIdx < width; outIdx++) {
    let rem = outIdx;
    let src = 0;
    for (let i = 0; i < n; i++) {
      const coord = (rem / outStrides[i]) | 0;
      rem -= coord * outStrides[i];
      src += coord * inStrides[perm[i]];
    }
    srcOf[outIdx] = src;
  }

  const inW = x.width;
  const data = new Int32Array(batch * width);
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * width;
    for (let i = 0; i < width; i++) data[oo + i] = x.data[xo + srcOf[i]];
  }
  return { data, width };
}
