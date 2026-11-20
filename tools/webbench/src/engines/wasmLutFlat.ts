// lut_flat → group_sum 融合の WASM 実行ラッパー(wasm エンジン用)。
// JS 版(lutFlat.ts::evalLutFlatGroupSum)と同じ表(compileLutFlat の詰め直し後
// の信号番号)を params ごとに1度だけヒープへ載せ、呼ぶたびに入力バイト列だけを
// コピーして assembly/index*.ts::lutFlatGsAsm / lutFlatPackedGsAsm を回す。
// 実測(Node、HAR k6000): 1枚 JS 43.5µs → WASM 約27µs、32枚 JS 11µs/枚 →
// 約6µs/枚(公開リファレンス reference/deploy-optimizations の Web 節)。
//
// メモリ契約は wasmBitplaneOps.ts と同じ単純バンプアロケータ: alloc() は
// memory.grow を起こしうるので、**全 alloc 完了後にのみ**ビューを作る。
// 正しさの錨: JS 版との全ビット一致(wasmLutFlat.test.ts が実 WASM で突き合わせる)。
import { compileLutFlat, LUT_FLAT_PACK_MIN_BATCH } from './lutFlat';
import type { LutFlatParams } from './lutFlat';
import type { NodeValue } from './irflat';
import type { WasmModule } from './wasmConv';

interface Tables {
  tt: number; a: number; b: number; o: number; inIdx: number;
  nIn: number; nG: number; outW: number; inW: number;
}

interface Slot { ptr: number; bytes: number }

export class WasmLutFlatOps {
  private readonly tables = new WeakMap<object, Tables>();
  private readonly slots = new Map<string, Slot>();

  constructor(private readonly mod: WasmModule) {}

  /** この .wasm がカーネルを持っているか。*/
  get available(): boolean {
    return typeof this.mod.lutFlatGsAsm === 'function'
      && typeof this.mod.lutFlatPackedGsAsm === 'function';
  }

  private ensure(name: string, bytes: number): number {
    const cur = this.slots.get(name);
    if (cur && cur.bytes >= bytes) return cur.ptr;
    const ptr = this.mod.alloc(Math.max(8, bytes));
    this.slots.set(name, { ptr, bytes });
    return ptr;
  }

  private load(p: LutFlatParams): Tables {
    const hit = this.tables.get(p);
    if (hit) return hit;
    const c = compileLutFlat(p);
    const nIn = c.inIdx.length;
    const tt = this.mod.alloc(Math.max(8, c.nG));
    const a = this.mod.alloc(Math.max(8, c.nG * 4));
    const b = this.mod.alloc(Math.max(8, c.nG * 4));
    const o = this.mod.alloc(Math.max(8, c.outW * 4));
    const inIdx = this.mod.alloc(Math.max(8, nIn * 4));
    const buf = this.mod.memory.buffer;
    new Uint8Array(buf, tt, c.nG).set(c.tt);
    new Int32Array(buf, a, c.nG).set(c.a);
    new Int32Array(buf, b, c.nG).set(c.b);
    new Int32Array(buf, o, c.outW).set(c.o);
    new Int32Array(buf, inIdx, nIn).set(c.inIdx);
    const t: Tables = { tt, a, b, o, inIdx, nIn, nG: c.nG, outW: c.outW, inW: c.inW };
    this.tables.set(p, t);
    return t;
  }

  /** lut_flat → group_sum(k)を WASM で評価する。カーネル未搭載・割り切れない k
      では null(呼び出し側は JS 版へ落ち、JS 版がエラーを報告する)。 */
  groupSum(
    p: LutFlatParams, x: { data: ArrayLike<number>; width: number }, batch: number, k: number,
  ): NodeValue | null {
    if (!this.available || batch <= 0) return null;
    const t = this.load(p);
    if (x.width !== t.inW || k <= 0 || t.outW % k !== 0) return null;
    const inBytes = batch * t.inW;
    if (!ArrayBuffer.isView(x.data) || x.data.length < inBytes) return null;
    const packed = batch >= LUT_FLAT_PACK_MIN_BATCH;
    const W = (batch + 31) >>> 5;
    const nSig = 1 + t.nIn + t.nG;
    const nPlanes = 32 - Math.clz32(t.outW / k);
    // alloc は memory.grow を起こしうるので、ビュー作成の前に全て済ませる。
    const xPtr = this.ensure('x', inBytes);
    const sigPtr = this.ensure(packed ? 'sigW' : 'sig8', packed ? nSig * W * 4 : nSig);
    const plPtr = this.ensure('planes', nPlanes * 4);
    const scPtr = this.ensure('scores', batch * k * 4);
    const buf = this.mod.memory.buffer;
    const src = x.data as Uint8Array | Int32Array;
    new Uint8Array(buf, xPtr, inBytes).set(
      src.length === inBytes ? src : src.subarray(0, inBytes),
    );
    if (packed) {
      this.mod.lutFlatPackedGsAsm!(
        t.tt, t.a, t.b, t.o, t.inIdx, t.nIn, t.nG, t.outW, k,
        xPtr, t.inW, batch, sigPtr, plPtr, scPtr,
      );
    } else {
      this.mod.lutFlatGsAsm!(
        t.tt, t.a, t.b, t.o, t.inIdx, t.nIn, t.nG, t.outW, k,
        xPtr, t.inW, batch, sigPtr, scPtr,
      );
    }
    const data = new Int32Array(batch * k);
    data.set(new Int32Array(this.mod.memory.buffer, scPtr, batch * k));
    return { data, width: k };
  }
}
