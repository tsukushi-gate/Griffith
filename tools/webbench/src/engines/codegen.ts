// (c) codegenエンジン(WB-5): IR→JSソース生成+new Function。
// 各ゲートを16種の最短式(a&b / a^b / ~(a|b) …)へ直書きコンパイルし、
// V8のJITに最適化させる。データはpacked32と同じ32サンプル/ワード縦パック。
// 1関数あたりのゲート数をCHUNK以下に分割(巨大関数のJIT不発対策)。
// new Function+初回実行(ウォームアップ)はコールドロード時間に計上する。
// 注意: CSP(unsafe-eval禁止)環境ではnew Functionが使えない
// (その場合はpacked32が保険。結果表に注記)。
import { ttParse } from '../shared';
import type { ModelDoc, TestData } from '../types';
import type { DenseLayers, Engine } from './types';
import { extractDense } from './types';

const CHUNK = 1500; // 1関数あたりの最大ゲート数(ステートメント数)

/** n=2の16ゲート最短式。s=srcバッファ変数名、a/bは添字式 */
function gateExpr(tt: number, a: string, b: string): string {
  switch (tt & 0xf) {
    case 0x0: return '0';
    case 0x1: return `~(${a}|${b})`;
    case 0x2: return `${a}&~${b}`;
    case 0x3: return `~${b}`;
    case 0x4: return `~${a}&${b}`;
    case 0x5: return `~${a}`;
    case 0x6: return `${a}^${b}`;
    case 0x7: return `~(${a}&${b})`;
    case 0x8: return `${a}&${b}`;
    case 0x9: return `~(${a}^${b})`;
    case 0xa: return a;
    case 0xb: return `${a}|~${b}`;
    case 0xc: return b;
    case 0xd: return `~${a}|${b}`;
    case 0xe: return `${a}|${b}`;
    case 0xf: return '-1';
    default: return '0';
  }
}

type LayerFn = (src: Int32Array, dst: Int32Array) => void;

function compileLayer(
  indices: number[][],
  truthTables: string[],
  outWidth: number,
): LayerFn[] {
  const fns: LayerFn[] = [];
  for (let start = 0; start < outWidth; start += CHUNK) {
    const end = Math.min(start + CHUNK, outWidth);
    const lines: string[] = [];
    for (let o = start; o < end; o++) {
      const tt = Number(ttParse(truthTables[o] ?? '0x0') & 0xfn);
      const a = `s[${indices[0]?.[o] ?? 0}]`;
      const b = `s[${indices[1]?.[o] ?? 0}]`;
      lines.push(`d[${o}]=${gateExpr(tt, a, b)};`);
    }
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    fns.push(new Function('s', 'd', lines.join('\n')) as LayerFn);
  }
  return fns;
}

export class CodegenEngine implements Engine {
  readonly name: string = 'dlgn-codegen';
  private packer: Packed32Base;
  private layerFns: LayerFn[][];
  private buf0: Int32Array;
  private buf1: Int32Array;

  constructor(model: ModelDoc) {
    this.packer = new Packed32Base(model);
    const dense = this.packer.dense;
    for (const p of dense.luts) {
      if (p.n_inputs !== 2) {
        throw new Error('codegenはn_inputs=2のみ対応(packedを使用)');
      }
    }
    this.layerFns = dense.luts.map((p) =>
      compileLayer(p.indices, p.truth_tables, p.out_width),
    );
    const maxW = Math.max(...dense.luts.map((p) => p.out_width));
    this.buf0 = new Int32Array(maxW);
    this.buf1 = new Int32Array(maxW);
  }

  /** ウォームアップ(JITコンパイル誘発)。コールドロード計測に含める */
  warmup(data: TestData): void {
    void this.evalBlock(data, 0, Math.min(32, data.n));
  }

  private evalBlock(data: TestData, from: number, nInBlock: number): Int32Array {
    const pk = this.packer;
    pk.encodePack(data, from, nInBlock);
    let src: Int32Array = pk.inWords;
    let dst = this.buf0;
    for (const fns of this.layerFns) {
      for (const fn of fns) fn(src, dst);
      const next = dst === this.buf0 ? this.buf1 : this.buf0;
      src = dst;
      dst = next;
    }
    pk.groupSum(src, nInBlock);
    return src;
  }

  async predict(data: TestData, from: number, count: number): Promise<Int32Array> {
    const preds = new Int32Array(count);
    const pk = this.packer;
    const groupK = pk.dense.groupK;
    for (let b = 0; b < count; b += 32) {
      const nInBlock = Math.min(32, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        let best = 0;
        for (let g = 1; g < groupK; g++) {
          if (pk.scoreBufs[g][j] > pk.scoreBufs[best][j]) best = g;
        }
        preds[b + j] = best;
      }
    }
    return preds;
  }

  scores(data: TestData, from: number, count: number): Int32Array[] {
    const out: Int32Array[] = [];
    const pk = this.packer;
    const groupK = pk.dense.groupK;
    for (let b = 0; b < count; b += 32) {
      const nInBlock = Math.min(32, count - b);
      this.evalBlock(data, from + b, nInBlock);
      for (let j = 0; j < nInBlock; j++) {
        const s = new Int32Array(groupK);
        for (let g = 0; g < groupK; g++) s[g] = pk.scoreBufs[g][j];
        out.push(s);
      }
    }
    return out;
  }
}

/** packed32のエンコード+group_sum部分だけを再利用するための内部クラス
    (packed.tsのprivate実装と重複させないための共有ヘルパ) */
class Packed32Base {
  readonly dense: DenseLayers;
  readonly inWords: Int32Array;
  readonly scoreBufs: Int32Array[];
  private model: ModelDoc;

  constructor(model: ModelDoc) {
    this.model = model;
    this.dense = extractDense(model.ir);
    this.inWords = new Int32Array(this.dense.inWidth);
    this.scoreBufs = Array.from(
      { length: this.dense.groupK },
      () => new Int32Array(32),
    );
  }

  encodePack(data: TestData, from: number, nInBlock: number): void {
    this.inWords.fill(0);
    const p = this.model.preproc;
    const c = p.mean.length;
    const l = data.chLen / c;
    const t = p.thresholds[0].length;
    for (let j = 0; j < nInBlock; j++) {
      const row = data.x.subarray(
        (from + j) * data.chLen, (from + j + 1) * data.chLen,
      );
      const bitJ = 1 << j;
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
  }

  groupSum(x: Int32Array, nInBlock: number): void {
    const lastK = this.dense.luts[this.dense.luts.length - 1].out_width;
    const groupK = this.dense.groupK;
    const groupSize = lastK / groupK;
    const nPlanes = 32 - Math.clz32(groupSize);
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
      const out = this.scoreBufs[g];
      for (let j = 0; j < nInBlock; j++) {
        let s = 0;
        for (let p = 0; p < nPlanes; p++) s |= ((planes[p] >>> j) & 1) << p;
        out[j] = s;
      }
    }
  }
}
