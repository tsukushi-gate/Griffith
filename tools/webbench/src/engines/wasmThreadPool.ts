// DLGN CPUエンジン(bitplane-wasm)のマルチスレッド実行基盤。
//
// 設計(調査メモ R1/R2 の指示に従う):
//   - **単一モジュール + 共有 WebAssembly.Memory**。メインスレッドが
//     `new WebAssembly.Memory({shared:true})` を作り、build/release-simd-mt.wasm
//     (asconfig の release-simd-mt ターゲット: --importMemory --sharedMemory)を
//     コンパイルして、Module と Memory をそのまま各Workerへ postMessage する。
//     Workerは同じメモリ上の**絶対ポインタ**で仕事をするので、フレームごとの
//     データ転送は一切発生しない({offset,length}すら送らない: 後述の
//     制御ブロックに書くだけ)。
//   - **手書きJS Workerプール + Atomics バリア**。ジョブ引数は共有の
//     制御ブロック(Int32Array over SharedArrayBuffer)に書き、世代カウンタを
//     1つ進めて Atomics.notify するだけ。postMessage の構造化複製コストを
//     ホットパスから完全に排除する(R1: 細かすぎる postMessage はビット演算の
//     速さを食い潰す)。
//   - **シャードは出力側で完全排他**(行ブロック/平面ブロック)。どのスレッドも
//     他スレッドの出力語に触れないため、**スレッド数によらず出力ビットは不変**。
//     境界計算 shardRange() は純関数として単体テストする(vitest には Worker が
//     無いため、整合性はこの純関数テスト + ブラウザ(Playwright)の整数一致の
//     2段構え)。
//   - crossOriginIsolated でない環境(GitHub Pages 等)や SAB 非対応環境では
//     `create()` が null を返し、呼び出し側は無改造の単スレッド経路へ落ちる。

/** 制御ブロック(共有 Int32Array)のインデックス。 */
export const IDX_GEN = 0; // 世代カウンタ(Workerはこれの変化を待つ)
export const IDX_DONE = 1; // 現世代を完了したWorker数
export const IDX_KIND = 2; // ジョブ種別
export const IDX_ARG0 = 4; // 以降がジョブ引数(種別ごとに意味が違う)
export const CTRL_WORDS = 32;

export const KIND_EXIT = 0;
/** args: [inPtr, outPtr, totalRows, W, wpr] */
export const KIND_PACK = 1;
/** args: [inPtr, outPtr, totalPlanes, H, wpr, HOut, wprOut, maskOut] */
export const KIND_ORPOOL = 2;
/** 融合前処理(RGBA→ビットプレーン)の出力チャネルシャード。
    args: [qPtr, bytePtr, descPtr, outPtr, totalChannels,
           H, W, wpr, tail, rowStride, rowPad, byteStride, cimg] */
export const KIND_PREPROC = 3;

/** 仕事量 total を nShards 等分したときの index 番目の範囲。
    余りは先頭のシャードから1つずつ配る(どのシャードも差は高々1)。
    total < nShards のときは count=0 のシャードが出るが、カーネル側は
    count=0 で何もしないので安全。 */
export function shardRange(
  total: number, nShards: number, index: number,
): { first: number; count: number } {
  if (nShards <= 1 || index < 0 || index >= nShards) {
    return index === 0 ? { first: 0, count: total } : { first: total, count: 0 };
  }
  const base = Math.floor(total / nShards);
  const rem = total % nShards;
  const first = index * base + Math.min(index, rem);
  const count = base + (index < rem ? 1 : 0);
  return { first, count };
}

/** シャード分割が「全域を過不足なく被覆し、互いに重ならない」ことの検査
    (整数一致の前提条件そのもの。テストから呼ぶ)。 */
export function shardCoversExactly(total: number, nShards: number): boolean {
  let next = 0;
  for (let i = 0; i < nShards; i++) {
    const { first, count } = shardRange(total, nShards, i);
    if (first !== next || count < 0) return false;
    next += count;
  }
  return next === total;
}

/** ORT wasm-factory.ts::isMultiThreadSupported と同じ2段判定。
    (1) SharedArrayBuffer の**構造化複製**が通るか(Firefox は型が存在しても
        crossOriginIsolated でないと複製で例外になる)、
    (2) スレッド命令(i32.atomic.load)入りの最小モジュールが validate できるか。 */
export function wasmThreadsSupported(): boolean {
  if (typeof SharedArrayBuffer !== 'function') return false;
  if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated) return false;
  try {
    new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
  } catch {
    return false;
  }
  try {
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1,
      3, 1, 1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
    ]));
  } catch {
    return false;
  }
}

/** 既定スレッド数は ORT / MediaPipe と同じ保守設定 min(4, ceil(cores/2))。
    big.LITTLE と帯域飽和を踏まえた値で、`hardwareConcurrency-1` は採らない
    (ユーザーが `?mt=N` で明示したときだけ上書きする)。 */
export function defaultThreadCount(): number {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(4, Math.ceil(cores / 2)));
}

/** メインスレッドが「自分のシャードを実行するための関数」。プールは
    Worker への通知と完了待ちだけを担当し、shard 0 は呼び出し側(メイン)が
    自分で実行する(pthreadpool と同じ流儀。1スレッド分のWorker生成を省ける)。 */
export type MainShardRunner = (first: number, count: number) => void;

export interface MtWasmBundle {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
  /** メインスレッド側インスタンスの exports(型は wasmConv.ts::WasmModule 互換)。 */
  instance: WebAssembly.Instance;
}

/** 共有メモリ版WASM(build/release-simd-mt.wasm)をコンパイル&インスタンス化する。
    メモリは asconfig の initial/maximum と一致させる(不一致は LinkError)。 */
export async function instantiateSharedWasm(bytes: ArrayBuffer): Promise<MtWasmBundle> {
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 8192, shared: true });
  const module = await WebAssembly.compile(bytes);
  const instance = await WebAssembly.instantiate(module, { env: { memory } });
  return { module, memory, instance };
}

/** メインスレッド側のWorkerプール。1ジョブ = 1バリア。 */
export class WasmThreadPool {
  private readonly workers: Worker[] = [];
  private readonly ctrl: Int32Array;
  private gen = 0;
  private disposed = false;

  private constructor(
    ctrlSab: SharedArrayBuffer,
    /** メイン込みの総スレッド数(= workers.length + 1)。 */
    readonly threads: number,
  ) {
    this.ctrl = new Int32Array(ctrlSab);
  }

  /** プールを生成する。threads<=1 / 非対応環境では null を返す
      (呼び出し側は単スレッド経路へ)。生成に失敗した場合も黙殺せず
      console.warn を残した上で null。 */
  static async create(
    bundle: MtWasmBundle, threads: number,
  ): Promise<WasmThreadPool | null> {
    if (threads <= 1) return null;
    if (!wasmThreadsSupported()) return null;
    const sab = new SharedArrayBuffer(CTRL_WORDS * 4);
    const pool = new WasmThreadPool(sab, threads);
    try {
      const readies: Promise<void>[] = [];
      for (let i = 1; i < threads; i++) {
        const w = new Worker(new URL('./wasmWorker.ts', import.meta.url), { type: 'module' });
        pool.workers.push(w);
        readies.push(new Promise<void>((res, rej) => {
          const to = setTimeout(() => rej(new Error(`worker${i} 初期化タイムアウト`)), 10_000);
          w.onmessage = (ev) => {
            if ((ev.data as { type?: string }).type === 'ready') {
              clearTimeout(to);
              res();
            }
          };
          w.onerror = (e) => {
            clearTimeout(to);
            rej(new Error(`worker${i} エラー: ${e.message}`));
          };
        }));
        w.postMessage({
          type: 'init',
          module: bundle.module,
          memory: bundle.memory,
          ctrl: sab,
          index: i,
          nThreads: threads,
        });
      }
      await Promise.all(readies);
      return pool;
    } catch (e) {
      console.warn('[wasmThreadPool] Workerプール生成に失敗→単スレッドへフォールバック:', e);
      pool.dispose();
      return null;
    }
  }

  /** 1ジョブを実行する。args は制御ブロックへ書かれ、Worker側は
      shardRange(total, threads, index) の担当範囲だけを処理する。
      shard 0 はこのメソッドの中でメインスレッドが実行する(=1スレッド分の
      Worker待ちが消え、レイテンシが最短になる)。 */
  run(kind: number, args: readonly number[], total: number, mainShard: MainShardRunner): void {
    if (this.disposed) throw new Error('wasmThreadPool: dispose済み');
    const c = this.ctrl;
    Atomics.store(c, IDX_KIND, kind);
    for (let i = 0; i < args.length; i++) Atomics.store(c, IDX_ARG0 + i, args[i]);
    Atomics.store(c, IDX_DONE, 0);
    this.gen += 1;
    Atomics.store(c, IDX_GEN, this.gen);
    Atomics.notify(c, IDX_GEN);

    const mine = shardRange(total, this.threads, 0);
    mainShard(mine.first, mine.count);

    // メインスレッドでは Atomics.wait が使えない(仕様で禁止)ので、
    // 完了カウンタをスピン待ちする。1024回ごとにだけ時計を見て、
    // Worker死亡時に無限ループしないよう上限を設ける。
    const need = this.workers.length;
    let spins = 0;
    const deadline = performance.now() + 5000;
    while (Atomics.load(c, IDX_DONE) < need) {
      spins += 1;
      if ((spins & 1023) === 0 && performance.now() > deadline) {
        throw new Error('wasmThreadPool: バリア待ちタイムアウト(Worker応答なし)');
      }
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const c = this.ctrl;
    Atomics.store(c, IDX_KIND, KIND_EXIT);
    Atomics.store(c, IDX_DONE, 0);
    this.gen += 1;
    Atomics.store(c, IDX_GEN, this.gen);
    Atomics.notify(c, IDX_GEN);
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
  }
}
