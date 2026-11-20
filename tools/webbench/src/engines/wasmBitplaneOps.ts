// 非conv・非lut のビットプレーン演算をWASM側へ移した実行ラッパー
// (グラフ入力パック / or_pool2d)。単スレッドでもJSより速く、プールが
// 渡されればそのまま行/平面ブロックで分割実行される。
//
// C5後のフレーム内訳(scripts/profile_cpu.mjs、wasmエンジン)が動機:
//   pack.input  det 55.8% / detmono 18.8% / detface 63.4% / detreal 69.6% / seg 44.0%
//   or_pool2d   det 22.6% / detmono 42.6% / detface 21.1% / detreal 11.0% / seg  6.9%
//   conv.wasm   det 12.9% / detmono 23.4% / detface  8.6% / detreal  8.8% / seg 10.3%
// (=conv2dカーネルは既に1割前後で、支配項はこの2つだった。)
//
// メモリ契約は wasmConv.ts / wasmBitplaneConv.ts と同じ単純バンプアロケータ。
// スロットごとに「必要になった最大サイズ」を保持し、足りなくなったときだけ
// alloc() し直す(bump なので古い領域は解放されないが、増える回数は形状の
// 種類数ぶんに限られ、ウォームアップ中に収束する)。
// alloc() は memory.grow を起こしうるので、**全 alloc 完了後にのみ**
// 書き込み用ビューを作る(このファイルでも徹底する)。
//
// 正しさの錨: bitplane.ts::packPlanes / orPool2x2 との整数完全一致
// (wasmBitplaneOps.test.ts が実WASMを読んで両実装を突き合わせる)。
import { tailMask, wordsPerRow } from './bitplane';
import type { BitplaneValue, PackSource } from './bitplane';
import type { WasmModule } from './wasmConv';
import { KIND_ORPOOL, KIND_PACK } from './wasmThreadPool';
import type { WasmThreadPool } from './wasmThreadPool';

const SLOT_PACK_IN = 0;
const SLOT_PACK_OUT = 1;
const SLOT_POOL_IN = 2;
const SLOT_POOL_OUT = 3;
const N_SLOTS = 4;

interface Slot { ptr: number; bytes: number }

// --- マルチスレッド化の下限(実測較正) ---------------------------
// バリア1回(制御ブロック更新 → notify → Worker起床 → 完了カウンタ回収)の
// 固定費は、ワーカーを 2ms スピンさせた状態で **約18µs** だった。1ジョブの
// 実仕事がこれと同程度しかないと並列化は逆効果になる(実測: seg は
// 単スレッド0.375ms → 4スレッド0.572ms と逆転した)。そこで「単スレッドでの
// 推定所要時間」がバリア固定費を十分上回るジョブだけをプールへ回す。
//
// 推定に使うスループットはこのマシンでの実測値(profile_cpu.mjs):
//   pack    入力 約7バイト/ns  (det 1.59MB→0.24ms / detface 2.83MB→0.33ms)
//   or_pool 入力 約0.5ワード/ns(det 123kワード→0.25ms)
// いずれも「並列化するか否か」の判断にしか使わない(出力ビットには一切
// 影響しない)ので、他マシンでずれても正しさは変わらず、最悪でも
// 「効かない並列化をする / しない」が変わるだけ。
const PACK_BYTES_PER_US = 7000;
const ORPOOL_WORDS_PER_US = 500;
/** これ未満(µs)の推定仕事量ならプールを使わず単スレッドで処理する。
    18µs のバリア固定費に対して 4スレッドで 1.5倍以上の実効高速化が
    見込める下限( T/4 + 18 < T/1.5 ⇔ T > 43µs )に余裕を持たせた値。 */
const MT_MIN_US = 60;

export class WasmBitplaneOps {
  private readonly slots: (Slot | null)[] = new Array(N_SLOTS).fill(null);

  constructor(
    private readonly mod: WasmModule,
    /** null なら単スレッド(メインスレッドが全域を1回で処理)。 */
    private readonly pool: WasmThreadPool | null = null,
  ) {}

  /** このWASMビルドが C3 カーネルを持っているか(古い .wasm を読んだ場合の保険。
      false ならすべてJS実装のままになる)。 */
  get available(): boolean {
    return typeof this.mod.packPlanesAsm === 'function'
      && typeof this.mod.orPool2x2Asm === 'function';
  }

  /** 実効スレッド数(1 = 単スレッド)。 */
  get threads(): number {
    return this.pool?.threads ?? 1;
  }

  /** このジョブをワーカープールへ回すか(estUs = 単スレッド推定所要µs)。
      仕事量がスレッド数に満たない場合も分割しない(空シャードのバリア待ちが
      無駄になるだけ)。 */
  private usesPool(estUs: number, items: number): boolean {
    const p = this.pool;
    return p !== null && estUs >= MT_MIN_US && items >= p.threads;
  }

  private ensure(slot: number, bytes: number): number {
    const cur = this.slots[slot];
    if (cur && cur.bytes >= bytes) return cur.ptr;
    const ptr = this.mod.alloc(Math.max(8, bytes));
    this.slots[slot] = { ptr, bytes };
    return ptr;
  }

  /** グラフ入力(0/1バイト列)のビットプレーンパック。扱えない入力
      (Uint8Array 以外 / 長さ不足 / カーネル未搭載)では null を返し、
      呼び出し側は bitplane.ts::packPlanes(JS)へ落ちる。 */
  packPlanes(
    x: PackSource, batch: number, C: number, H: number, W: number,
  ): BitplaneValue | null {
    if (!this.available) return null;
    const data = x.data;
    if (!(data instanceof Uint8Array)) return null;
    if (x.width !== C * H * W) {
      throw new Error(`packPlanes(wasm): 幅不一致 (期待 ${C * H * W}, 実際 ${x.width})`);
    }
    const inBytes = batch * x.width;
    if (data.length < inBytes || inBytes === 0) return null;
    const wpr = wordsPerRow(W);
    const rows = batch * C * H;
    const outWords = rows * wpr;

    // alloc は memory.grow を起こしうるので、ビュー作成の前に全て済ませる。
    const inPtr = this.ensure(SLOT_PACK_IN, inBytes);
    const outPtr = this.ensure(SLOT_PACK_OUT, outWords * 4);
    const buf = this.mod.memory.buffer;
    new Uint8Array(buf, inPtr, inBytes).set(
      data.length === inBytes ? data : data.subarray(0, inBytes),
    );

    const run = (first: number, count: number): void => {
      if (count > 0) this.mod.packPlanesAsm!(inPtr, outPtr, first, count, W, wpr);
    };
    if (this.usesPool(inBytes / PACK_BYTES_PER_US, rows)) {
      this.pool!.run(KIND_PACK, [inPtr, outPtr, rows, W, wpr], rows, run);
    } else {
      run(0, rows);
    }

    const words = new Int32Array(outWords);
    words.set(new Int32Array(this.mod.memory.buffer, outPtr, outWords));
    return { words, C, H, W, wpr };
  }

  /** or_pool2d(kernel=2, stride=2, padding=0)。扱えない形状/カーネル未搭載
      では null を返し、呼び出し側は bitplane.ts::orPool2x2(JS)へ落ちる。 */
  orPool2x2(bp: BitplaneValue, batch: number): BitplaneValue | null {
    if (!this.available) return null;
    const { C, H, W, wpr } = bp;
    if (H < 2 || W < 2) return null;
    const HOut = ((H - 2) >> 1) + 1;
    const WOut = ((W - 2) >> 1) + 1;
    const wprOut = wordsPerRow(WOut);
    const maskOut = tailMask(WOut);
    const planes = batch * C;
    const inWords = planes * H * wpr;
    const outWords = planes * HOut * wprOut;
    if (bp.words.length < inWords || inWords === 0) return null;

    const inPtr = this.ensure(SLOT_POOL_IN, inWords * 4);
    const outPtr = this.ensure(SLOT_POOL_OUT, outWords * 4);
    const buf = this.mod.memory.buffer;
    new Int32Array(buf, inPtr, inWords).set(
      bp.words.length === inWords ? bp.words : bp.words.subarray(0, inWords),
    );

    const run = (first: number, count: number): void => {
      if (count > 0) {
        this.mod.orPool2x2Asm!(inPtr, outPtr, first, count, H, wpr, HOut, wprOut, maskOut);
      }
    };
    if (this.usesPool(inWords / ORPOOL_WORDS_PER_US, planes)) {
      this.pool!.run(
        KIND_ORPOOL, [inPtr, outPtr, planes, H, wpr, HOut, wprOut, maskOut], planes, run,
      );
    } else {
      run(0, planes);
    }

    const words = new Int32Array(outWords);
    words.set(new Int32Array(this.mod.memory.buffer, outPtr, outWords));
    return { words, C, H: HOut, W: WOut, wpr: wprOut };
  }
}
