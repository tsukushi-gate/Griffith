// 計算Worker。メインスレッドから受け取った WebAssembly.Module と
// 共有 WebAssembly.Memory で自分専用のインスタンスを作り、以後は共有制御
// ブロック(Int32Array over SharedArrayBuffer)の世代カウンタを待って
// 「自分のシャードだけ」を実行する。
//
// フレームごとの postMessage は一切無い(init の1回だけ)。ジョブ引数は
// 制御ブロックから読む。出力は共有メモリ上の排他な行/平面ブロックなので、
// スレッド間で書き込みが競合することはない(=出力ビットはスレッド数に
//依らず不変)。
//
// 注意: このファイルは Worker グローバルスコープで動くため、DOM や
// window に触れてはいけない(cpuProfile.ts 等はインポートしない)。
import {
  IDX_ARG0, IDX_DONE, IDX_GEN, IDX_KIND, KIND_EXIT, KIND_ORPOOL, KIND_PACK, KIND_PREPROC,
  shardRange,
} from './wasmThreadPool';

interface ShardKernels {
  packPlanesAsm(
    inPtr: number, outPtr: number, rowFirst: number, rowCount: number,
    W: number, wpr: number,
  ): void;
  orPool2x2Asm(
    inPtr: number, outPtr: number, planeFirst: number, planeCount: number,
    H: number, wpr: number, HOut: number, wprOut: number, maskOut: number,
  ): void;
  /** 融合前処理(出力チャネルシャード)。古い .wasm 由来の
      インスタンスでは未定義になりうるので呼ぶ前に存在を確認する。 */
  preprocPackAsm?(
    qPtr: number, bytePtr: number, descPtr: number, outPtr: number,
    chFirst: number, chCount: number,
    H: number, W: number, wpr: number, tail: number,
    rowStride: number, rowPad: number, byteStride: number, cimg: number,
  ): void;
}

interface InitMessage {
  type: 'init';
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
  ctrl: SharedArrayBuffer;
  index: number;
  nThreads: number;
}

/** Atomics.wait へ入る前のスピン時間(ms)。
    [実測] 反復回数ベースの短いスピン(2万回≒40µs)だと、フレーム内の
    ジョブ間隔(0.1〜0.5ms)やフレーム間隔で必ず Atomics.wait へ落ちてしまい、
    futex の起床遅延がジョブあたり **50〜60µs** の固定費として乗る。seg のように
    1ジョブの実仕事が 15〜50µs しかないタスクでは、これだけで並列化の利得が
    完全に消える(実測 seg 0.375ms → 0.572ms と逆転した)。
    連続推論中はワーカーを起こしたままにするため、スピン予算を「ジョブ間隔
    より十分長く、アイドル時には確実に寝る」2ms に取る(30fpsデモなら
    デューティ比 6% 程度で、アイドル中はCPUを明け渡す)。 */
const SPIN_MS = 2;
/** 時計を見る間隔(Atomics.load のスピンに performance.now() を毎回混ぜると
    それ自体がスピンのコストになるため)。 */
const SPIN_CHECK_MASK = 1023;

function runJob(
  k: ShardKernels, ctrl: Int32Array, kind: number, index: number, nThreads: number,
): void {
  const a = (i: number): number => Atomics.load(ctrl, IDX_ARG0 + i);
  if (kind === KIND_PACK) {
    const [inPtr, outPtr, total, W, wpr] = [a(0), a(1), a(2), a(3), a(4)];
    const { first, count } = shardRange(total, nThreads, index);
    if (count > 0) k.packPlanesAsm(inPtr, outPtr, first, count, W, wpr);
    return;
  }
  if (kind === KIND_ORPOOL) {
    const [inPtr, outPtr, total, H, wpr, HOut, wprOut, maskOut] = [
      a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7),
    ];
    const { first, count } = shardRange(total, nThreads, index);
    if (count > 0) {
      k.orPool2x2Asm(inPtr, outPtr, first, count, H, wpr, HOut, wprOut, maskOut);
    }
    return;
  }
  if (kind === KIND_PREPROC) {
    const { first, count } = shardRange(a(4), nThreads, index);
    if (count > 0 && typeof k.preprocPackAsm === 'function') {
      k.preprocPackAsm(
        a(0), a(1), a(2), a(3), first, count,
        a(5), a(6), a(7), a(8), a(9), a(10), a(11), a(12),
      );
    }
    return;
  }
  // 未知の種別は「何もしない」で完了カウントだけ進める(メインを止めない)。
}

async function boot(msg: InitMessage): Promise<void> {
  const instance = await WebAssembly.instantiate(msg.module, { env: { memory: msg.memory } });
  const k = instance.exports as unknown as ShardKernels;
  const ctrl = new Int32Array(msg.ctrl);
  const { index, nThreads } = msg;
  (self as unknown as { postMessage(m: unknown): void }).postMessage({ type: 'ready' });

  let seen = Atomics.load(ctrl, IDX_GEN);
  for (;;) {
    let gen = Atomics.load(ctrl, IDX_GEN);
    let spins = 0;
    let spinUntil = performance.now() + SPIN_MS;
    while (gen === seen) {
      spins += 1;
      if ((spins & SPIN_CHECK_MASK) === 0 && performance.now() > spinUntil) {
        // スピン予算を使い切ったら寝る(タイムアウト付き: notifyを
        // 取りこぼしても必ず起きる)。起きたらまたスピン予算を配り直す。
        Atomics.wait(ctrl, IDX_GEN, seen, 50);
        spinUntil = performance.now() + SPIN_MS;
      }
      gen = Atomics.load(ctrl, IDX_GEN);
    }
    seen = gen;
    const kind = Atomics.load(ctrl, IDX_KIND);
    if (kind === KIND_EXIT) return;
    runJob(k, ctrl, kind, index, nThreads);
    Atomics.add(ctrl, IDX_DONE, 1);
    Atomics.notify(ctrl, IDX_DONE);
  }
}

self.onmessage = (ev: MessageEvent): void => {
  const m = ev.data as InitMessage;
  if (m && m.type === 'init') {
    boot(m).catch((e) => {
      // 黙殺しない(メイン側はバリアのタイムアウトで気づくが理由が要る)。
      console.error('[wasmWorker] 初期化/実行に失敗:', e);
    });
  }
};
