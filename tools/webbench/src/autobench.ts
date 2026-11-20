// 自動ベンチマークモードの共通ロジック。
// det/detmono/detface/detreal/seg の5エントリから同一の手順で呼ばれる:
//   URLクエリ ?autobench=1&backend=<js|wasm|webgpu>&frames=<n> を解釈し、
//   指定時は UI 操作なしで「DLGN ランタイム load() → ウォームアップ8回 →
//   frames 回計測(各 run() の inferMs、既存 timedRun/runDetFrameAsync と同じ
//   定義=前処理を除く推論のみ)」を行って window.__benchResult に結果を積み、
//   document.title を 'BENCH_DONE' にする(Playwright 側は waitForFunction /
//   タイトル変化を待機トリガに使う)。
//
// DLGN ランタイム(DetDlgnRuntime/SegDlgnRuntime、compareDet.ts/compareSeg.ts)は
// いずれも CompareRuntime<Out> 契約(load()→LoadInfo, run(rgba)→RunInfo<Out>)
// + actualBackend: DlgnBackend(load()後にフォールバック込みで確定)を持つため、
// このモジュールは型を絞った最小のダックタイピングインターフェースのみに依存し、
// det系/seg系どちらのランタイムクラスにも改造なしで使える。
import { summarize, tick } from './stats';
import type { LoadInfo } from './compare';
import type { DlgnBackend } from './compareSeg';
import { PIPELINE_DEPTH } from './engines/webgpuBitplane';
import type { WebgpuBreakdown } from './engines/webgpuGraph';

const VALID_BACKENDS: readonly DlgnBackend[] = ['js', 'wasm', 'webgpu'];

export interface AutobenchParams {
  task: string;
  backend: DlgnBackend;
  frames: number;
}

/** URLクエリを解釈する。`?autobench=1` が無ければ null(通常UIをそのまま動かす)。
    href を明示的に受け取ることでブラウザ外(vitest/node)でも単体テスト可能にする。 */
export function parseAutobenchParams(
  task: string,
  href: string = window.location.href,
): AutobenchParams | null {
  const q = new URL(href).searchParams;
  if (q.get('autobench') !== '1') return null;
  const backendRaw = q.get('backend');
  const backend: DlgnBackend = (VALID_BACKENDS as readonly string[]).includes(backendRaw ?? '')
    ? (backendRaw as DlgnBackend)
    : 'wasm';
  const framesRaw = Number(q.get('frames'));
  const frames = Number.isFinite(framesRaw) && framesRaw > 0 ? Math.floor(framesRaw) : 50;
  return { task, backend, frames };
}

/** DLGN ランタイム(DetDlgnRuntime/SegDlgnRuntime)が満たす最小契約。
    webgpu内訳計測の3メソッドは任意(実装が無いランタイムでは
    単に呼ばれない=no-op相当)。DetDlgnRuntime/SegDlgnRuntimeはモデル未ロード
    時にも安全なno-opとして実装している。 */
export interface AutobenchRuntime {
  load(): Promise<LoadInfo>;
  run(rgba: Uint8ClampedArray): Promise<{ inferMs: number }>;
  actualBackend: DlgnBackend;
  setWebgpuProfiling?(enabled: boolean): void;
  resetWebgpuProfilingStats?(): void;
  getWebgpuBreakdown?(): WebgpuBreakdown | null;
  /** フレームパイプライン(2フレームin-flight)対応か。
      未実装/false のランタイムでは pipelineMs は計測されない。 */
  supportsPipeline?(): boolean;
  submitFrame?(rgba: Uint8ClampedArray): Promise<{ inferMs: number }>;
  flushPipeline?(): Promise<void>;
  /** [計測専用] 前処理をループ外へ追い出すためのフック。
      逐次の inferMs(前処理を含まない)と同一区間でパイプラインを測るのに使う。 */
  prepareBenchInput?(rgba: Uint8ClampedArray): void;
  submitPreparedFrame?(): Promise<{ inferMs: number }>;
}

export interface BenchResult {
  task: string;
  backend: DlgnBackend; // 要求バックエンド(?backend=の値)
  actualBackend: DlgnBackend; // load()後の実バックエンド(フォールバック後)
  median: number;
  p95: number;
  min: number;
  n: number;
  loadMs: number;
  modelGzipBytes: number | null;
  /** actualBackend==='webgpu'の場合のみ設定。ノード別GPU時間の
      上位10件+CPU側分解(encode/writeBuffer/submit→map/readback copy/合計)。
      bench_matrix.mjs 等の既存消費側はフィールド追加のみで後方互換。 */
  breakdown?: WebgpuBreakdown;

  /** 逐次モードの1フレームあたり実測wall時間(前処理+推論+decode)。
      median/p95/min は「推論のみ(inferMs)」の統計なので、パイプラインの
      実効フレーム時間と直接比べるための同一定義の基準値としてこちらを使う。
      各フレームの計測値の総和/N(ループ中の tick() のマクロタスク待ちは含まない)。 */
  wallMs?: number;
  /** `?pipeline=1` かつ supportsPipeline() のときのみ設定。
      N=frames フレームをback-to-backで流した実効フレーム時間(全体時間/N)。
      レイテンシではなくスループット指標(1フレームぶんの表示遅延と引き換えに
      submit→mapAsyncの往復をGPU実行へ重ねた結果)。 */
  pipelineMs?: number;
  /** pipelineMs の計測フレーム数と同時in-flight数(報告の再現性のため)。 */
  pipelineFrames?: number;
  pipelineDepth?: number;
  /** 推論のみ(前処理をループ外へ追い出した)の実効フレーム時間。
      逐次の inferMs と**同一区間**なので meanInferMs と直接比較できる
      (=submit→mapAsyncの床がどこまで隠れたかの主指標)。 */
  pipelineInferMs?: number;
  /** 逐次 inferMs の平均(pipelineInferMs は平均なので中央値ではなく
      これと比べる。median は従来どおり残す)。 */
  meanInferMs?: number;
}

declare global {
  interface Window {
    __benchResult?: BenchResult;
  }
}

const WARMUP_ITERS = 8;

/** 計測フレームでのプロファイリングは、それ自体が resolveQuerySet と
    「2本目のmapAsync」を毎フレームに足すため median を汚す(webgpuは
    submit→mapAsync の往復がそのまま床になるので影響が出る)。
    既定は従来どおり有効(=内訳を必ず取る)のままにし、`?gpuprofile=0` を
    付けたときだけ無効化して「素の推論時間」を測れるようにする。
    parseAutobenchParams の戻り値には載せない(既存の解釈テストの契約を変えない)。 */
function profilingEnabledFromUrl(): boolean {
  try {
    return new URL(window.location.href).searchParams.get('gpuprofile') !== '0';
  } catch {
    return true;
  }
}

/** `?pipeline=1` でスループット計測(2フレームin-flight)を追加する。
    既定オフ(既存セルの計測内容を変えない)。parseAutobenchParams の戻り値には
    載せない(既存の解釈テストの契約を変えないため、gpuprofileと同じ扱い)。
    href を明示に取れるようにしてvitestから単体テストできるようにする。 */
export function pipelineEnabledFromUrl(href?: string): boolean {
  try {
    return new URL(href ?? window.location.href).searchParams.get('pipeline') === '1';
  } catch {
    return false;
  }
}

/** N フレームを「同時in-flight = depth」でback-to-backに流し、
    全体時間/N(実効フレーム時間)を返す。
    ループの形が肝: 次フレームを発行する前に **最古の** フレームだけをawaitする
    ことで、常に depth 本が飛んでいる状態を保つ(=submit→mapAsyncの往復が
    直前フレームのGPU実行と重なる)。 */
async function measurePipelineMs(
  rt: AutobenchRuntime,
  submit: () => Promise<unknown>,
  frames: number,
  depth: number,
): Promise<number> {
  // パイプライン専用のウォームアップ(リング確保・初回validationスコープ等)。
  for (let i = 0; i < WARMUP_ITERS; i++) await submit();
  await rt.flushPipeline?.();
  await tick();

  const pend: Promise<unknown>[] = [];
  const t0 = performance.now();
  for (let i = 0; i < frames; i++) {
    if (pend.length >= depth) await pend.shift();
    pend.push(submit());
  }
  await Promise.all(pend);
  const elapsed = performance.now() - t0;
  await rt.flushPipeline?.();
  return elapsed / frames;
}

/** 合成入力(内容は問わない、推論速度のみ計測するためデータ依存の分岐が
    無いDLGN推論では中身がゼロ埋めでも計測結果は変わらない。ただし念のため
    全ゼロ/全255のような退化パターンは避け、決定的な疑似ノイズにする)。 */
function syntheticRgba(w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < out.length; i++) out[i] = (i * 37 + 11) % 256;
  return out;
}

/** load() → ウォームアップ8回 → frames回計測 → window.__benchResult確定 →
    document.title='BENCH_DONE'。sizeOf は load() 後のモデル入力解像度を返す
    (DetRuntimeはmodelSize正方形、SegRuntimeはinW/inHで形が違うため呼び出し側が
    吸収する)。 */
export async function runAutobench(
  params: AutobenchParams,
  rt: AutobenchRuntime,
  sizeOf: () => { w: number; h: number },
): Promise<void> {
  const info = await rt.load();
  const { w, h } = sizeOf();
  const rgba = syntheticRgba(w, h);

  for (let i = 0; i < WARMUP_ITERS; i++) await rt.run(rgba);

  // webgpuバックエンドのみ、ウォームアップ後・本計測前にプロファイリングを
  // 有効化する(オプトイン。js/wasmバックエンドやgetWebgpuBreakdown未実装の
  // ランタイムでは呼ばれないためホットパスへのコスト追加は無い)。
  const isWebgpu = rt.actualBackend === 'webgpu';
  const wantProfiling = isWebgpu && profilingEnabledFromUrl();
  if (wantProfiling) {
    rt.setWebgpuProfiling?.(true);
    rt.resetWebgpuProfilingStats?.();
  }

  const samples: number[] = [];
  // wallSum は「前処理+推論+decode」の1フレーム実測の総和。
  // tick() のマクロタスク待ちを含めないためフレーム毎に区間を取る。
  let wallSum = 0;
  for (let i = 0; i < params.frames; i++) {
    const tw = performance.now();
    const r = await rt.run(rgba);
    wallSum += performance.now() - tw;
    samples.push(r.inferMs);
    if (i % 20 === 19) await tick();
  }
  const s = summarize(samples);
  const breakdown = wantProfiling ? (rt.getWebgpuBreakdown?.() ?? undefined) : undefined;

  // スループット指標。逐次のレイテンシ計測(上)を一切変えず、
  // 同じ入力・同じランタイムで追加の1周を回す。プロファイリングは
  // 切ってから測る(2本目のmapAsyncが隠したい床を復活させるため)。
  let pipelineMs: number | undefined;
  let pipelineInferMs: number | undefined;
  const wantPipeline = pipelineEnabledFromUrl()
    && typeof rt.submitFrame === 'function'
    && (rt.supportsPipeline?.() ?? false);
  if (wantPipeline) {
    if (wantProfiling) rt.setWebgpuProfiling?.(false);
    // (1) エンドツーエンド(前処理+推論+decode)= 実デモの1フレーム。wallMsと比較。
    pipelineMs = await measurePipelineMs(
      rt, () => rt.submitFrame!(rgba), params.frames, PIPELINE_DEPTH,
    );
    // (2) 推論のみ(前処理をループ外へ)= 逐次inferMsと同一区間。床の隠蔽度合いの主指標。
    if (rt.prepareBenchInput && rt.submitPreparedFrame) {
      rt.prepareBenchInput(rgba);
      pipelineInferMs = await measurePipelineMs(
        rt, () => rt.submitPreparedFrame!(), params.frames, PIPELINE_DEPTH,
      );
    }
  }

  window.__benchResult = {
    task: params.task,
    backend: params.backend,
    actualBackend: rt.actualBackend,
    median: s.median,
    p95: s.p95,
    min: s.min,
    n: s.n,
    loadMs: info.loadMs,
    modelGzipBytes: info.gzipBytes,
    breakdown,
    wallMs: wallSum / params.frames,
    meanInferMs: samples.reduce((a, b) => a + b, 0) / samples.length,
    pipelineMs,
    pipelineInferMs,
    pipelineFrames: wantPipeline ? params.frames : undefined,
    pipelineDepth: wantPipeline ? PIPELINE_DEPTH : undefined,
  };
  document.title = 'BENCH_DONE';
}
