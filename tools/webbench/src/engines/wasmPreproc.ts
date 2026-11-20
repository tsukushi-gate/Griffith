// 融合前処理(RGBA → ビットプレーン)のWASM実行ラッパー。
//
// cpuPreproc.ts の実行計画(記述子表 + 平面レイアウト)をそのまま
// assembly/index*.ts::preprocQuantAsm / preprocPackAsm へ渡す。
// フレームごとにWASMヒープへ載せるのは **RGBA だけ**(det 96x96 で 36KB、
// detface 128x128 で 64KB)で、従来の「0/1バイト列 1.6〜2.8MB を作って
// packPlanes が読み直す」経路が丸ごと消える。
//
// メモリ契約は wasmBitplaneOps.ts と同じ単純バンプアロケータ。記述子・qval は
// **構築時に1回だけ**書き、以後は上書きしない(モデル固定の定数)。
// alloc() は memory.grow を起こしうるので、**全 alloc 完了後にのみ**
// 書き込み用ビューを作る(このファイルでも徹底する)。
//
// シャード: 出力チャネル単位。チャネル平面は互いに素なのでスレッド数に
// よらず出力ビットは不変(packPlanesAsm と同じ流儀)。
//
// 正しさの錨: cpuPreproc.test.ts(JS参照 ⇔ scalar wasm ⇔ simd wasm ⇔
// encodeImageAuto+packPlanes の全ワード一致)。
import type { BitplaneValue } from './bitplane';
import type { CpuPreprocPlan } from './cpuPreproc';
import type { WasmModule } from './wasmConv';
import { KIND_PREPROC } from './wasmThreadPool';
import type { WasmThreadPool } from './wasmThreadPool';

/** ベクタ読み越し(最大16バイト)ぶんの末尾スラック。 */
const SLACK_BYTES = 64;

/** [と同じ較正の考え方] マルチスレッド化の下限。前処理1回の実仕事は
    det/detface で数百µs あるためバリア固定費(約18µs)には十分勝つが、
    小さいモデル(seg 64x64 × 93ch)では効かないこともあるので、
    packPlanes と同じく「単スレッド推定所要時間」で足切りする。
    推定スループットは実測ベース(出力ワードあたり約 3ns)。判断にしか
    使わないので他マシンでずれても出力ビットは一切変わらない。 */
const PREPROC_WORDS_PER_US = 330;
const MT_MIN_US = 60;

export class WasmPreprocOps {
  private readonly qPtr: number;
  private readonly bytePtr: number;
  private readonly descPtr: number;
  private readonly qvalPtr: number;
  private readonly rgbaPtr: number;
  private readonly outPtr: number;
  private readonly outWords: number;
  private readonly rgbaBytes: number;

  constructor(
    private readonly mod: WasmModule,
    private readonly plan: CpuPreprocPlan,
    /** null なら単スレッド(メインスレッドが全チャネルを1回で処理)。 */
    private readonly pool: WasmThreadPool | null = null,
  ) {
    if (!WasmPreprocOps.supported(mod)) {
      throw new Error('読み込んだ .wasm に前処理カーネルがありません');
    }
    const {
      channels, imageChannels, height: H, width: W, wpr, rowStride,
    } = plan;
    this.rgbaBytes = H * W * 4;
    this.outWords = channels * H * wpr;
    // alloc は memory.grow を起こしうるので、ビュー作成の前に全て済ませる。
    this.rgbaPtr = mod.alloc(this.rgbaBytes + SLACK_BYTES);
    this.qPtr = mod.alloc(imageChannels * H * rowStride * 4 + SLACK_BYTES);
    this.bytePtr = mod.alloc(imageChannels * H * W + SLACK_BYTES);
    this.descPtr = mod.alloc(plan.desc.length * 4);
    this.qvalPtr = mod.alloc(256 * 4);
    this.outPtr = mod.alloc(this.outWords * 4);
    const buf = mod.memory.buffer;
    new Int32Array(buf, this.descPtr, plan.desc.length).set(plan.desc);
    new Int32Array(buf, this.qvalPtr, 256).set(plan.qval);
  }

  /** このWASMビルドが C6 カーネルを持っているか(古い .wasm を読んだ場合の保険)。 */
  static supported(mod: WasmModule): boolean {
    return typeof mod.preprocQuantAsm === 'function'
      && typeof mod.preprocPackAsm === 'function';
  }

  /** 実効スレッド数(1 = 単スレッド)。 */
  get threads(): number {
    return this.pool?.threads ?? 1;
  }

  /** RGBA(batch枚ぶんの連結)→ ビットプレーン([batch][C][H][wpr])。 */
  pack(rgba: ArrayLike<number>, batch: number): BitplaneValue {
    const {
      channels, imageChannels, height: H, width: W, wpr, tail, rowStride, rowPad,
    } = this.plan;
    const need = batch * this.rgbaBytes;
    if (rgba.length < need) {
      throw new Error(`preproc(wasm): RGBAが短い(期待 ${need}, 実際 ${rgba.length})`);
    }
    const words = new Int32Array(batch * this.outWords);
    const byteStride = H * W;
    for (let b = 0; b < batch; b++) {
      const src = b * this.rgbaBytes;
      // memory.grow はもう起きない(全 alloc は constructor で完了済み)が、
      // 他のカーネルが grow する可能性があるので毎回ビューを作り直す。
      new Uint8Array(this.mod.memory.buffer, this.rgbaPtr, this.rgbaBytes).set(
        rgba instanceof Uint8Array || rgba instanceof Uint8ClampedArray
          ? rgba.subarray(src, src + this.rgbaBytes)
          : Array.prototype.slice.call(rgba, src, src + this.rgbaBytes),
      );
      this.mod.preprocQuantAsm!(
        this.rgbaPtr, this.qvalPtr, this.qPtr, this.bytePtr,
        H, W, rowStride, rowPad, byteStride, imageChannels,
      );
      const run = (first: number, count: number): void => {
        if (count > 0) {
          this.mod.preprocPackAsm!(
            this.qPtr, this.bytePtr, this.descPtr, this.outPtr, first, count,
            H, W, wpr, tail, rowStride, rowPad, byteStride, imageChannels,
          );
        }
      };
      if (this.usesPool(this.outWords / PREPROC_WORDS_PER_US, channels)) {
        this.pool!.run(
          KIND_PREPROC,
          [
            this.qPtr, this.bytePtr, this.descPtr, this.outPtr, channels,
            H, W, wpr, tail, rowStride, rowPad, byteStride, imageChannels,
          ],
          channels, run,
        );
      } else {
        run(0, channels);
      }
      words.set(
        new Int32Array(this.mod.memory.buffer, this.outPtr, this.outWords),
        b * this.outWords,
      );
    }
    return { words, C: channels, H, W, wpr };
  }

  private usesPool(estUs: number, items: number): boolean {
    const p = this.pool;
    return p !== null && estUs >= MT_MIN_US && items >= p.threads;
  }
}
