// 縮約済み dense 平坦ネットリスト(IRノード `lut_flat`)の実行器。
// runtime/c/src/mdl_kernels.c の `MDL_LUT_FLAT_BODY`(1枚ずつ)と
// `MDL_LUT_FLAT_PACKED_BODY`(バッチ軸パック)の TS ミラー。
//
// 信号番号の規約(griffith_logic/ir/schema.py の lut_flat 節が真実源):
//   0 = 定数0 / 1..in_width = 入力ビット / in_width+1+g = ゲート g の出力
//   ゲート g = [tt, a, b]、出力 = (tt >> ((sig[b]<<1)|sig[a])) & 1
//   outputs[i] = 2*信号番号 + 極性(極性1 = 反転、定数1 = 信号0の反転)
//
// lut_layer 経路(irflat.ts::evalLutLayer)は呼ぶたびに真理値表を ttParse
// し直すが、ここでは params オブジェクトごとに型付き配列へ一度だけ変換して
// WeakMap に持つ(ゲート表は数万要素あるので、毎回の変換が実行より重くなる)。
//
// C 版との違い(結果は同一): 信号表に載せる入力は**ゲート/出力が参照する
// 入力ビットだけ**に詰め直す(HAR k600 は 4,608 本中 668 本しか読まない。
// 入力の詰め替えが網の評価より重かったため、実測)。
import type { NodeValue } from './irflat';

export interface LutFlatParams {
  in_width: number;
  out_width: number;
  gates: number[][]; // [[tt, a, b], ...]
  outputs: number[];
}

export interface CompiledLutFlat {
  inW: number;
  /** 信号表に載せる入力ビットの元番号(信号 1+u ← 入力 inIdx[u]) */
  inIdx: Int32Array;
  nG: number;
  outW: number;
  tt: Uint8Array;
  a: Int32Array;
  b: Int32Array;
  o: Int32Array;
  /** a | b<<16(信号数 <= 65536 のときだけ。超えたら null で a/b を引く) */
  ab: Uint32Array | null;
  /** 1枚ずつ経路の信号表(使い回し。1+nIn+nG バイト) */
  sig: Uint8Array;
}

const compiled = new WeakMap<object, CompiledLutFlat>();

/** params → 実行用の型付き配列(params ごとに1度だけ作って WeakMap に持つ)。
    wasmLutFlat.ts もこの表(詰め直し後の信号番号)をそのままヒープへ載せる。 */
export function compileLutFlat(p: LutFlatParams): CompiledLutFlat {
  const hit = compiled.get(p);
  if (hit) return hit;
  const inW = p.in_width;
  const nG = p.gates.length;
  const outW = p.out_width;
  if (p.outputs.length !== outW) {
    throw new Error(`lut_flat: outputs 長 ${p.outputs.length} が out_width=${outW} と一致しない`);
  }
  // 参照される入力ビットだけを信号表へ載せる(番号の振り直し)
  const remap = new Int32Array(1 + inW + nG).fill(-1);
  const inIdxList: number[] = [];
  const nSigOrig = 1 + inW + nG;
  const touch = (sg: number): void => {
    if (sg >= 1 && sg <= inW && remap[sg] < 0) {
      remap[sg] = 1 + inIdxList.length;
      inIdxList.push(sg - 1);
    }
  };
  for (let g = 0; g < nG; g++) {
    const row = p.gates[g];
    const lim = inW + 1 + g;
    if (!(row[1] < lim && row[2] < lim && row[1] >= 0 && row[2] >= 0)) {
      throw new Error(`lut_flat: ゲート${g}が自分以降の信号を参照している(トポロジカル順違反)`);
    }
    touch(row[1]);
    touch(row[2]);
  }
  for (let i = 0; i < outW; i++) {
    const sg = p.outputs[i] >> 1;
    if (sg < 0 || sg >= nSigOrig) throw new Error(`lut_flat: outputs[${i}] が信号数 ${nSigOrig} を超える`);
    touch(sg);
  }
  // 元番号の昇順に並べ直す(詰め替えで入力行を前から順に読むため)
  inIdxList.sort((x, y) => x - y);
  for (let u = 0; u < inIdxList.length; u++) remap[1 + inIdxList[u]] = 1 + u;
  const nIn = inIdxList.length;
  remap[0] = 0;
  for (let g = 0; g < nG; g++) remap[inW + 1 + g] = 1 + nIn + g;
  const tt = new Uint8Array(nG);
  const a = new Int32Array(nG);
  const b = new Int32Array(nG);
  for (let g = 0; g < nG; g++) {
    const row = p.gates[g];
    tt[g] = row[0] & 15;
    a[g] = remap[row[1]];
    b[g] = remap[row[2]];
  }
  const o = new Int32Array(outW);
  for (let i = 0; i < outW; i++) {
    const r = p.outputs[i];
    o[i] = (remap[r >> 1] << 1) | (r & 1);
  }
  let ab: Uint32Array | null = null;
  if (1 + nIn + nG <= 65536) {
    ab = new Uint32Array(nG);
    for (let g = 0; g < nG; g++) ab[g] = (a[g] | (b[g] << 16)) >>> 0;
  }
  const c: CompiledLutFlat = {
    inW, inIdx: Int32Array.from(inIdxList), nG, outW, tt, a, b, o, ab,
    sig: new Uint8Array(1 + nIn + nG),
  };
  compiled.set(p, c);
  return c;
}

/** これ以上の枚数ならバッチ軸パック経路(32枚/語)を使う。
    HAR k600/k2400/k6000 で 3 枚から逆転(1 枚は 1枚ずつ経路の約半分の速さ、
    2 枚で 0.87〜0.89 倍、3 枚で 1.1〜1.15 倍、実測)。 */
export const LUT_FLAT_PACK_MIN_BATCH = 3;

/** 1枚ぶんのゲートを評価して c.sig を埋める(MDL_LUT_FLAT_BODY)。
    a/b は信号数 <= 65536 なら 1 語(a | b<<16)に詰めた表を引く
    (読む表が 3 本 → 2 本になり、HAR k6000 のゲート列で約 2 割速い、実測)。 */
function runScalar(c: CompiledLutFlat, x: ArrayLike<number>, xo: number): void {
  const { inIdx, nG, tt, sig } = c;
  const nIn = inIdx.length;
  const base = nIn + 1;
  sig[0] = 0;
  for (let u = 0; u < nIn; u++) sig[1 + u] = x[xo + inIdx[u]];
  const ab = c.ab;
  if (ab) {
    for (let g = 0; g < nG; g++) {
      const w = ab[g];
      sig[base + g] = (tt[g] >>> ((sig[w >>> 16] << 1) | sig[w & 0xffff])) & 1;
    }
  } else {
    const { a, b } = c;
    for (let g = 0; g < nG; g++) {
      sig[base + g] = (tt[g] >>> ((sig[b[g]] << 1) | sig[a[g]])) & 1;
    }
  }
}

/** バッチ軸パック(MDL_LUT_FLAT_PACKED_BODY)で全ゲートを評価し、信号表を返す。
    信号ごとに W=ceil(batch/32) 語を連続に持ち(信号メジャー)、
    1ゲート = W 語の butterfly-select。 */
function runPacked(c: CompiledLutFlat, x: ArrayLike<number>, batch: number): Int32Array {
  const { inW, inIdx, nG, tt, a, b } = c;
  const nIn = inIdx.length;
  const W = (batch + 31) >>> 5;
  const sig = new Int32Array((1 + nIn + nG) * W); // sig[0..W) = 定数0
  // 入力の詰め替え: 枚 s の入力 inIdx[u] → 信号 1+u の語 s>>5 のビット s&31
  for (let s = 0; s < batch; s++) {
    const xo = s * inW;
    const w = s >>> 5;
    const bit = s & 31;
    let p = W + w;
    for (let u = 0; u < nIn; u++, p += W) sig[p] |= (x[xo + inIdx[u]] & 1) << bit;
  }
  const gbase = (nIn + 1) * W;
  if (W === 1) {
    for (let g = 0; g < nG; g++) {
      const t = tt[g];
      const t0 = -(t & 1), t1 = -((t >> 1) & 1), t2 = -((t >> 2) & 1), t3 = -((t >> 3) & 1);
      const av = sig[a[g]], bv = sig[b[g]];
      const u = t0 ^ (av & (t0 ^ t1)), v = t2 ^ (av & (t2 ^ t3));
      sig[gbase + g] = u ^ (bv & (u ^ v));
    }
  } else {
    for (let g = 0; g < nG; g++) {
      const t = tt[g];
      const t0 = -(t & 1), t1 = -((t >> 1) & 1), t2 = -((t >> 2) & 1), t3 = -((t >> 3) & 1);
      const pa = a[g] * W, pb = b[g] * W, po = gbase + g * W;
      for (let w = 0; w < W; w++) {
        const av = sig[pa + w], bv = sig[pb + w];
        const u = t0 ^ (av & (t0 ^ t1)), v = t2 ^ (av & (t2 ^ t3));
        sig[po + w] = u ^ (bv & (u ^ v));
      }
    }
  }
  return sig;
}

function checkInput(c: CompiledLutFlat, x: { width: number }): void {
  if (x.width !== c.inW) {
    throw new Error(`lut_flat: 入力幅 ${x.width} が in_width=${c.inW} と一致しない`);
  }
}

/** lut_flat ノードを評価する([batch, in_width] の0/1 → [batch, out_width])。
    x.data は Int32Array でも生入力の Uint8Array でもよい(コピーせず読む)。
    `forcePacked` はテスト用(true=常にパック / false=常に1枚ずつ)。 */
export function evalLutFlat(
  p: LutFlatParams, x: { data: ArrayLike<number>; width: number }, batch: number,
  forcePacked?: boolean,
): NodeValue {
  const c = compileLutFlat(p);
  checkInput(c, x);
  const { outW, o } = c;
  const out = new Int32Array(batch * outW);
  if (forcePacked ?? batch >= LUT_FLAT_PACK_MIN_BATCH) {
    const sig = runPacked(c, x.data, batch);
    const W = (batch + 31) >>> 5;
    // 極性を語マスクで掛けて、枚ごとの0/1へ戻す
    for (let i = 0; i < outW; i++) {
      const r = o[i];
      const m = -(r & 1);
      const ps = (r >> 1) * W;
      for (let w = 0; w < W; w++) {
        const v = sig[ps + w] ^ m;
        const s0 = w << 5;
        const n = Math.min(32, batch - s0);
        for (let k = 0; k < n; k++) out[(s0 + k) * outW + i] = (v >>> k) & 1;
      }
    }
  } else {
    const sig = c.sig;
    for (let s = 0; s < batch; s++) {
      runScalar(c, x.data, s * c.inW);
      const oo = s * outW;
      for (let i = 0; i < outW; i++) {
        const r = o[i];
        out[oo + i] = sig[r >> 1] ^ (r & 1);
      }
    }
  }
  return { data: out, width: outW };
}

/** lut_flat → group_sum(k グループ)の融合: 出力ビット列を作らずに
    グループ内の popcount([batch, k] の整数スコア)を直接返す。
    結果は evalGroupSum(evalLutFlat(...)) と同一。パック経路は語ごとの
    垂直カウンタ(ビットスライス加算、packed.ts::groupSumScores32 と同じ手法)。 */
export function evalLutFlatGroupSum(
  p: LutFlatParams, x: { data: ArrayLike<number>; width: number }, batch: number,
  k: number, forcePacked?: boolean,
): NodeValue {
  const c = compileLutFlat(p);
  checkInput(c, x);
  const { outW, o } = c;
  if (k <= 0 || outW % k !== 0) throw new Error(`group_sum: 幅${outW}がk=${k}で割り切れない`);
  const gsz = outW / k;
  const scores = new Int32Array(batch * k);
  if (forcePacked ?? batch >= LUT_FLAT_PACK_MIN_BATCH) {
    const sig = runPacked(c, x.data, batch);
    const W = (batch + 31) >>> 5;
    const nPlanes = 32 - Math.clz32(gsz); // ceil(log2(gsz+1))
    const planes = new Int32Array(nPlanes);
    for (let w = 0; w < W; w++) {
      const s0 = w << 5;
      const n = Math.min(32, batch - s0);
      for (let g = 0; g < k; g++) {
        planes.fill(0);
        const i0 = g * gsz;
        for (let i = i0; i < i0 + gsz; i++) {
          const r = o[i];
          let carry = sig[(r >> 1) * W + w] ^ -(r & 1);
          for (let q = 0; carry !== 0 && q < nPlanes; q++) {
            const t = planes[q] & carry;
            planes[q] ^= carry;
            carry = t;
          }
        }
        for (let lane = 0; lane < n; lane++) {
          let cnt = 0;
          for (let q = 0; q < nPlanes; q++) cnt |= ((planes[q] >>> lane) & 1) << q;
          scores[(s0 + lane) * k + g] = cnt;
        }
      }
    }
  } else {
    const sig = c.sig;
    for (let s = 0; s < batch; s++) {
      runScalar(c, x.data, s * c.inW);
      for (let g = 0; g < k; g++) {
        let cnt = 0;
        const i0 = g * gsz;
        for (let i = i0; i < i0 + gsz; i++) {
          const r = o[i];
          cnt += sig[r >> 1] ^ (r & 1);
        }
        scores[s * k + g] = cnt;
      }
    }
  }
  return { data: scores, width: k };
}
