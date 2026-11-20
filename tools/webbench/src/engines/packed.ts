// (b) packedエンジン(WB-5、独自貢献の核): ビット並列(bit-parallel)実行。
// 32サンプル(64サンプル)を1ワードに縦パックし、各ゲートを分岐なしの
// ワード演算で評価する。difflogicのネイティブC実装(64サンプル/ワードで
// 100万MNIST画像/秒/コア)のブラウザ移植に相当する。
//
// ゲート評価(Shannon展開の分岐なし式、n入力一般):
//   out = OR_k ( mask_k & term_k ),  term_k = AND_j (bit j of k ? x_j : ~x_j)
//   mask_k = 真理値表ビットkが1なら全ビット1、0なら0(事前計算)
// n=2は out=(m0&~a&~b)|(m1&a&~b)|(m2&~a&b)|(m3&a&b) をインライン特殊化。
//
// group_sum(グループ内popcount)は垂直カウンタ(bit-sliced accumulation):
//   ワード列を桁上がり伝播でビット平面planesに加算し、
//   サンプルjのスコア = Σ_p ((planes[p]>>>j)&1)<<p。O(G·logG) word-ops。
//
// ブロック処理: 1ブロック=ワード幅サンプル。レイヤバッファは幅×1ワードで
// L1/L2に収まる(k=6000でも24KB)。
import { ttParse } from '../shared';
import type { LutLayerParams } from '../shared';
import type { ModelDoc, TestData } from '../types';
import { sampleView } from '../preproc';
import type { DenseLayers, Engine } from './types';
import { extractDense } from './types';

interface PackedLayer32 {
  k: number;
  n: number;
  idx: Int32Array; // [n * k](slot-major: idx[j*k+o])
  masks: Int32Array; // [2^n * k](masks[t*k+o] = TTビットtが1なら-1)
}

function buildLayers32(dense: DenseLayers): PackedLayer32[] {
  return dense.luts.map((p: LutLayerParams) => {
    const k = p.out_width;
    const n = p.n_inputs;
    const size = 1 << n;
    const idx = new Int32Array(n * k);
    for (let j = 0; j < n; j++) {
      for (let o = 0; o < k; o++) idx[j * k + o] = p.indices[j]?.[o] ?? 0;
    }
    const masks = new Int32Array(size * k);
    for (let o = 0; o < k; o++) {
      const tt = ttParse(p.truth_tables[o] ?? '0x0');
      for (let t = 0; t < size; t++) {
        masks[t * k + o] = (tt >> BigInt(t)) & 1n ? -1 : 0;
      }
    }
    return { k, n, idx, masks };
  });
}

/** 垂直カウンタ: グループ内ワード列を加算しサンプル別スコアへ展開する */
function groupSumScores32(
  x: Int32Array,
  groupK: number,
  nInBlock: number,
  scoresOut: Int32Array[], // [groupK] 個の Int32Array(wordBits)…呼び手が用意
): void {
  const groupSize = x.length / groupK;
  const nPlanes = 32 - Math.clz32(groupSize); // ceil(log2(G+1))
  const planes = new Int32Array(nPlanes);
  for (let g = 0; g < groupK; g++) {
    planes.fill(0);
    for (let i = 0; i < groupSize; i++) {
      let carry = x[g * groupSize + i];
      for (let p = 0; carry !== 0 && p < nPlanes; p++) {
        const t = planes[p] & carry;
        planes[p] ^= carry;
        carry = t;
      }
    }
    const out = scoresOut[g];
    for (let j = 0; j < nInBlock; j++) {
      let s = 0;
      for (let p = 0; p < nPlanes; p++) s |= ((planes[p] >>> j) & 1) << p;
      out[j] = s;
    }
  }
}

/** thermometer符号化+縦パック(ブロックの32サンプル→各信号1ワード)。
    encodeThermometer(preproc.ts)とビット単位で同一の判定(z > thr)。 */
function encodePackBlock32(
  data: TestData,
  from: number,
  nInBlock: number,
  model: ModelDoc,
  words: Int32Array, // [inWidth] 出力(zero済み前提でなくてよい: 上書き)
): void {
  words.fill(0);
  const p = model.preproc;
  const c = p.mean.length;
  const l = data.chLen / c;
  const t = p.thresholds[0].length;
  for (let j = 0; j < nInBlock; j++) {
    const row = sampleView(data, from + j);
    const bitJ = 1 << j;
    let bi = 0;
    for (let ch = 0; ch < c; ch++) {
      const mean = p.mean[ch];
      const invStd = 1 / p.std[ch];
      const thr = p.thresholds[ch];
      for (let ti = 0; ti < l; ti++) {
        const z = (row[ch * l + ti] - mean) * invStd;
        for (let k = 0; k < t; k++) {
          if (z > thr[k]) words[bi] |= bitJ;
          bi++;
        }
      }
    }
  }
}

export class Packed32Engine implements Engine {
  readonly name: string = 'dlgn-packed32';
  protected model: ModelDoc;
  protected dense: DenseLayers;
  private layers: PackedLayer32[];
  private buf0: Int32Array;
  private buf1: Int32Array;
  private inWords: Int32Array;
  private scoreBufs: Int32Array[];

  constructor(model: ModelDoc) {
    this.model = model;
    this.dense = extractDense(model.ir);
    this.layers = buildLayers32(this.dense);
    const maxW = Math.max(...this.layers.map((la) => la.k));
    this.buf0 = new Int32Array(maxW);
    this.buf1 = new Int32Array(maxW);
    this.inWords = new Int32Array(this.dense.inWidth);
    this.scoreBufs = Array.from(
      { length: this.dense.groupK },
      () => new Int32Array(32),
    );
  }

  /** 1ブロック(最大32サンプル)を評価しクラススコアをscoreBufsへ */
  private evalBlock(data: TestData, from: number, nInBlock: number): void {
    encodePackBlock32(data, from, nInBlock, this.model, this.inWords);
    let src: Int32Array = this.inWords;
    let dst = this.buf0;
    for (const la of this.layers) {
      const { k, n, idx, masks } = la;
      if (n === 2) {
        for (let o = 0; o < k; o++) {
          const a = src[idx[o]];
          const b = src[idx[k + o]];
          dst[o] =
            (masks[o] & ~a & ~b) |
            (masks[k + o] & a & ~b) |
            (masks[2 * k + o] & ~a & b) |
            (masks[3 * k + o] & a & b);
        }
      } else {
        const size = 1 << n;
        for (let o = 0; o < k; o++) {
          let out = 0;
          for (let t = 0; t < size; t++) {
            const m = masks[t * k + o];
            if (m === 0) continue;
            let term = -1;
            for (let j = 0; j < n; j++) {
              const x = src[idx[j * k + o]];
              term &= (t >> j) & 1 ? x : ~x;
            }
            out |= m & term;
          }
          dst[o] = out;
        }
      }
      const next = dst === this.buf0 ? this.buf1 : this.buf0;
      src = dst;
      dst = next;
    }
    const lastK = this.layers[this.layers.length - 1].k;
    groupSumScores32(
      src.subarray(0, lastK), this.dense.groupK, nInBlock, this.scoreBufs,
    );
  }

  async predict(data: TestData, from: number, count: number): Promise<Int32Array> {
    const preds = new Int32Array(count);
    const groupK = this.dense.groupK;
    for (let b = 0; b < count; b += 32) {
      const nInBlock = Math.min(32, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        let best = 0;
        for (let g = 1; g < groupK; g++) {
          if (this.scoreBufs[g][j] > this.scoreBufs[best][j]) best = g;
        }
        preds[b + j] = best;
      }
    }
    return preds;
  }

  scores(data: TestData, from: number, count: number): Int32Array[] {
    const out: Int32Array[] = [];
    const groupK = this.dense.groupK;
    for (let b = 0; b < count; b += 32) {
      const nInBlock = Math.min(32, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        const s = new Int32Array(groupK);
        for (let g = 0; g < groupK; g++) s[g] = this.scoreBufs[g][j];
        out.push(s);
      }
    }
    return out;
  }
}

// ============================================================
// 64bit変種(BigUint64Array)。BigIntのヒープ割当コストで32bit版より
// 遅い可能性が高く、それ自体を実測して報告する(WB-5の計測対象)。
// ============================================================

const M64 = (1n << 64n) - 1n; // NOT用マスク(BigIntは符号無限長のため)

interface PackedLayer64 {
  k: number;
  n: number;
  idx: Int32Array;
  masks: BigUint64Array;
}

export class Packed64Engine implements Engine {
  readonly name: string = 'dlgn-packed64';
  private model: ModelDoc;
  private dense: DenseLayers;
  private layers: PackedLayer64[];
  private buf0: BigUint64Array;
  private buf1: BigUint64Array;
  private inWords: BigUint64Array;
  private scoreBufs: Int32Array[];

  constructor(model: ModelDoc) {
    this.model = model;
    this.dense = extractDense(model.ir);
    this.layers = this.dense.luts.map((p: LutLayerParams) => {
      const k = p.out_width;
      const n = p.n_inputs;
      const size = 1 << n;
      const idx = new Int32Array(n * k);
      for (let j = 0; j < n; j++) {
        for (let o = 0; o < k; o++) idx[j * k + o] = p.indices[j]?.[o] ?? 0;
      }
      const masks = new BigUint64Array(size * k);
      for (let o = 0; o < k; o++) {
        const tt = ttParse(p.truth_tables[o] ?? '0x0');
        for (let t = 0; t < size; t++) {
          masks[t * k + o] = (tt >> BigInt(t)) & 1n ? M64 : 0n;
        }
      }
      return { k, n, idx, masks };
    });
    const maxW = Math.max(...this.layers.map((la) => la.k));
    this.buf0 = new BigUint64Array(maxW);
    this.buf1 = new BigUint64Array(maxW);
    this.inWords = new BigUint64Array(this.dense.inWidth);
    this.scoreBufs = Array.from(
      { length: this.dense.groupK },
      () => new Int32Array(64),
    );
  }

  private evalBlock(data: TestData, from: number, nInBlock: number): void {
    // encode+pack(64サンプル/ワード)
    this.inWords.fill(0n);
    const p = this.model.preproc;
    const c = p.mean.length;
    const l = data.chLen / c;
    const t = p.thresholds[0].length;
    for (let j = 0; j < nInBlock; j++) {
      const row = sampleView(data, from + j);
      const bitJ = 1n << BigInt(j);
      let bi = 0;
      for (let ch = 0; ch < c; ch++) {
        const mean = p.mean[ch];
        const invStd = 1 / p.std[ch];
        const thr = p.thresholds[ch];
        for (let ti = 0; ti < l; ti++) {
          const z = (row[ch * l + ti] - mean) * invStd;
          for (let k = 0; k < t; k++) {
            if (z > thr[k]) this.inWords[bi] |= bitJ;
            bi++;
          }
        }
      }
    }
    let src: BigUint64Array = this.inWords;
    let dst = this.buf0;
    for (const la of this.layers) {
      const { k, n, idx, masks } = la;
      if (n === 2) {
        for (let o = 0; o < k; o++) {
          const a = src[idx[o]];
          const b = src[idx[k + o]];
          const na = a ^ M64;
          const nb = b ^ M64;
          dst[o] =
            (masks[o] & na & nb) |
            (masks[k + o] & a & nb) |
            (masks[2 * k + o] & na & b) |
            (masks[3 * k + o] & a & b);
        }
      } else {
        const size = 1 << n;
        for (let o = 0; o < k; o++) {
          let out = 0n;
          for (let ti = 0; ti < size; ti++) {
            const m = masks[ti * k + o];
            if (m === 0n) continue;
            let term = M64;
            for (let j = 0; j < n; j++) {
              const x = src[idx[j * k + o]];
              term &= (ti >> j) & 1 ? x : x ^ M64;
            }
            out |= m & term;
          }
          dst[o] = out;
        }
      }
      const next = dst === this.buf0 ? this.buf1 : this.buf0;
      src = dst;
      dst = next;
    }
    // 垂直カウンタ(64bit)
    const lastK = this.layers[this.layers.length - 1].k;
    const groupK = this.dense.groupK;
    const groupSize = lastK / groupK;
    const nPlanes = 32 - Math.clz32(groupSize);
    const planes = new BigUint64Array(nPlanes);
    for (let g = 0; g < groupK; g++) {
      planes.fill(0n);
      for (let i = 0; i < groupSize; i++) {
        let carry = src[g * groupSize + i];
        for (let pp = 0; carry !== 0n && pp < nPlanes; pp++) {
          const tt = planes[pp] & carry;
          planes[pp] ^= carry;
          carry = tt;
        }
      }
      const out = this.scoreBufs[g];
      for (let j = 0; j < nInBlock; j++) {
        let s = 0;
        const bj = BigInt(j);
        for (let pp = 0; pp < nPlanes; pp++) {
          s |= Number((planes[pp] >> bj) & 1n) << pp;
        }
        out[j] = s;
      }
    }
  }

  async predict(data: TestData, from: number, count: number): Promise<Int32Array> {
    const preds = new Int32Array(count);
    const groupK = this.dense.groupK;
    for (let b = 0; b < count; b += 64) {
      const nInBlock = Math.min(64, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        let best = 0;
        for (let g = 1; g < groupK; g++) {
          if (this.scoreBufs[g][j] > this.scoreBufs[best][j]) best = g;
        }
        preds[b + j] = best;
      }
    }
    return preds;
  }

  scores(data: TestData, from: number, count: number): Int32Array[] {
    const out: Int32Array[] = [];
    const groupK = this.dense.groupK;
    for (let b = 0; b < count; b += 64) {
      const nInBlock = Math.min(64, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        const s = new Int32Array(groupK);
        for (let g = 0; g < groupK; g++) s[g] = this.scoreBufs[g][j];
        out.push(s);
      }
    }
    return out;
  }
}
