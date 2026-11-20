// seg/det デモの 3 ランタイム(DLGN / ONNX Runtime Web / TensorFlow.js)
// side-by-side 比較の共通土台。各ランタイムを統一インターフェース
//   { name, load()→LoadInfo, run(rgba)→RunInfo } に揃え、seg_main / det_main が
// 同一入力を 3 本に流して出力(マスク/box)・サイズ・ロード・推論を並べる。
//
// 公平性の注記(UI にも明示):
//   - DLGN は thermometer 二値化入力(学習時と同一)を消費する。
//   - seg ONNX/TF.js(SmallUNet)は生 float[0,1] NCHW/NHWC 入力。
//   - det ONNX/TF.js(CenterNetCNN teacher)は DLGN と同じ thermometer 入力
//     (teacher も in_channels=T の二値特徴を学習時に消費している)。
// つまり各ランタイムは「学習時のネイティブ前処理」で動かす。
// 具体的なランタイム実装は compareSeg.ts / compareDet.ts。
//
// 上記の単発 run() 比較(seg_main/det_main のデモ表示用)に加えて、
// harness.ts(HAR時系列ベンチ)と同じ方法論
//   warmup → 単発レイテンシ p50/p95/min/max → 複数試行スループット中央値
//   → メモリ差分(usedJSHeapSize) → ペイロード(gzip transfer/decoded)
// を CompareRuntime<Out> にも適用する「計測モード」(measureRuntime)を追加する。
// バックエンド軸(ONNX wasm/webgpu, TF.js webgl/wasm/webgpu)は
// compareSeg.ts / compareDet.ts のランタイムクラスが backend 引数を持つ形で表現し、
// ここでは backend 文字列込みの結果レコードを返すだけに留める。
import { summarize, tick } from './stats';

// TF.js wasm バックエンドの .wasm 実体(セルフホスト、CDN 不使用)。
// Vite の `?url` 明示 URL import で dev/build 両対応のパスを解決する
// (onnxruntime-web は import.meta.url 基準の自己解決だが、tfjs-backend-wasm は
// setWasmPaths() に明示 URL を渡す必要がある)。
import tfjsWasmUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import tfjsWasmSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import tfjsWasmThreadedSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';

export const TFJS_WASM_PATHS: Record<string, string> = {
  'tfjs-backend-wasm.wasm': tfjsWasmUrl,
  'tfjs-backend-wasm-simd.wasm': tfjsWasmSimdUrl,
  'tfjs-backend-wasm-threaded-simd.wasm': tfjsWasmThreadedSimdUrl,
};

/** 1 ランタイムのロード結果(サイズ/時間/表示メタ)。 */
export interface LoadInfo {
  modelBytes: number | null; // モデル資産(非圧縮)
  gzipBytes: number | null; // 同 gzip 後(CompressionStream 実測)
  runtimeBytes: number | null; // 追加ランタイム(wasm 等)非圧縮。無ければ null
  // 追加ランタイムの gzip 後バイト数(モデルと同じ CompressionStream 実測)。
  // runtimeBytes は非圧縮なので、配信サイズ(gzipBytes基準)の大きい数字に足し
  // 合わせる場面ではこちらを使う(基準が揃った値でないと合算できないため)。
  // 計測できない/対象のランタイムが無い場合は null。
  runtimeGzipBytes: number | null;
  loadMs: number; // fetch 開始→実行可能までの実測
  paramsLabel: string; // "8.4M ゲート" / "1.06M params" 等
  note: string; // 前処理の注記(公平性)
  metricLabel: string; // "hard mIoU" / "hard mAP"
  metricValue: number | null;
}

export interface RunInfo<Out> {
  output: Out;
  inferMs: number;
}

/** 統一ランタイム抽象。Out は seg=Int32Array(ラベル)/det=DetBoxPx[]。
    dispose() は任意(webcamデモのようにランタイムを排他で切り替える場面でのみ
    呼ばれる。static比較ページ(seg_main.ts等)はページ寿命=ランタイム寿命なので
    呼ばない)。実装しないランタイムは「解放不要」として扱われる。 */
export interface CompareRuntime<Out> {
  readonly name: string;
  load(): Promise<LoadInfo>;
  run(rgba: Uint8ClampedArray): Promise<RunInfo<Out>>;
  dispose?(): void | Promise<void>;
}

/** ArrayBuffer/Uint8Array を gzip して圧縮後バイト数を返す(全ランタイム公平比較用)。
    CompressionStream 非対応環境では null。 */
export async function gzipSize(bytes: Uint8Array): Promise<number | null> {
  if (typeof CompressionStream === 'undefined') return null;
  try {
    const cs = new CompressionStream('gzip');
    const writer = cs.writable.getWriter();
    void writer.write(bytes as unknown as BufferSource);
    void writer.close();
    const reader = cs.readable.getReader();
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += (value as Uint8Array).byteLength;
    }
    return total;
  } catch {
    return null;
  }
}

/** performance resource timing から wasm 等の非圧縮バイトを拾う(最後の該当)。 */
export function resourceBytes(sub: string): number | null {
  const es = performance
    .getEntriesByType('resource')
    .filter((e) => e.name.includes(sub)) as PerformanceResourceTiming[];
  if (!es.length) return null;
  const last = es[es.length - 1];
  return last.decodedBodySize || last.transferSize || null;
}

/** performance resource timing から該当リソースの URL を特定し、実体を
    取得して gzip 後バイト数を返す(モデル(gzipSize)と同じ CompressionStream
    実測手法)。resourceBytes() が返す非圧縮値をそのまま gzipBytes 基準の配信
    サイズへ足すと基準がずれるため、この「同一手法で測った」値を使う。
    対象は既にロード済みでブラウザキャッシュに載っているため、ここでの再取得は
    実質追加のネットワーク往復にならない(モデル本体の gzipSize 計測と同じ
    パターン)。エントリが無い/取得失敗時は null。 */
export async function resourceGzipBytes(sub: string): Promise<number | null> {
  const es = performance
    .getEntriesByType('resource')
    .filter((e) => e.name.includes(sub)) as PerformanceResourceTiming[];
  if (!es.length) return null;
  const url = es[es.length - 1].name;
  try {
    const r = await fetch(url);
    const buf = await r.arrayBuffer();
    return await gzipSize(new Uint8Array(buf));
  } catch {
    return null;
  }
}

/** `navigator.gpu` の有無(同期チェック)。アダプタが実際に取れるかは別
    (collectEnvInfo が非同期で requestAdapter する)。 */
export function hasWebGpuApi(): boolean {
  return typeof navigator !== 'undefined'
    && 'gpu' in (navigator as unknown as { gpu?: unknown });
}

function heapMB(): number | null {
  const mem = (performance as unknown as {
    memory?: { usedJSHeapSize: number };
  }).memory;
  return mem ? mem.usedJSHeapSize / (1024 * 1024) : null;
}

export interface MeasureConfig {
  warmupIters: number;
  latencyIters: number;
  throughputImgs: number; // 1試行あたりのrun()回数
  throughputTrials: number;
  /** load()単体、および warmup+latency+throughput 全体それぞれに適用する
      タイムアウト(ms)。バックグラウンドタブのスロットリング等でwebgl
      readbackが実質ハングする既知の事象があるため、行列全体が止まらない
      よう安全弁として設ける(該当ランタイムのみerror扱いで先に進む)。 */
  loadTimeoutMs: number;
  runTimeoutMs: number;
}

export const DEFAULT_MEASURE_CONFIG: MeasureConfig = {
  warmupIters: 8,
  latencyIters: 40,
  throughputImgs: 15,
  throughputTrials: 5,
  loadTimeoutMs: 30_000,
  runTimeoutMs: 60_000,
};

/** Promise がタイムアウトした場合エラーで解決する(元のPromiseは中断できない
    ため、JS単一スレッド上でイベントループ自体が生きている限りタイムアウトは
    発火する。ハング元の処理自体はバックグラウンドに残り得るが、計測ループを
    先に進めることを優先する)。 */
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`計測タイムアウト(${ms}ms): ${label}`)), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer));
}

/** harness.ts(HAR時系列ベンチ)と同じ6指標のうち、CompareRuntime<Out>から
    測れる分(ペイロード/コールドロード/ウォームレイテンシ/スループット/
    精度メタ/メモリ差分)を1ランタイムぶんまとめたレコード。 */
export interface BackendMeasurement {
  runtime: string; // 表示名(バックエンド込み、例: "ONNX Runtime Web webgpu (SmallUNet)")
  backend: string; // 'js' | 'wasm' | 'webgpu' | 'webgl'
  status: 'ok' | 'error';
  error: string | null;
  modelBytes: number | null;
  gzipBytes: number | null;
  runtimeBytes: number | null;
  loadMs: number | null;
  paramsLabel: string | null;
  metricLabel: string | null;
  metricValue: number | null;
  warmLatency: { median: number; p95: number; min: number; max: number; n: number } | null;
  throughput: { samplesPerSec: number; trials: number[] } | null;
  memDeltaMB: number | null;
  notes: string[];
}

/** harness.ts と同じ方法論(warmup→単発レイテンシp50/p95→複数試行スループット
    中央値→メモリ差分)を CompareRuntime<Out> に適用する。ペイロード/精度メタは
    load() の LoadInfo から回収する(seg/det の run() はサンプル単位でバッチが
    無いため、スループットは run() を連続実行して測る)。
    失敗時(未対応バックエンド等)は status='error' で notes/error に記録し、
    呼び出し側(UI)は該当行を「利用不可」として表示する(graceful degrade)。 */
export async function measureRuntime<Out>(
  backend: string,
  rt: CompareRuntime<Out>,
  rgba: Uint8ClampedArray,
  cfg: MeasureConfig = DEFAULT_MEASURE_CONFIG,
  onProgress: (msg: string) => void = () => {},
): Promise<BackendMeasurement> {
  const heapBefore = heapMB();
  const notes: string[] = [];
  const empty = {
    modelBytes: null, gzipBytes: null, runtimeBytes: null, loadMs: null,
    paramsLabel: null, metricLabel: null, metricValue: null,
  };
  let info: LoadInfo;
  try {
    onProgress(`${rt.name}: ロード`);
    info = await withTimeout(rt.load(), cfg.loadTimeoutMs, `${rt.name} load()`);
  } catch (e) {
    return {
      runtime: rt.name, backend, status: 'error', error: String(e),
      ...empty, warmLatency: null, throughput: null, memDeltaMB: null, notes,
    };
  }
  if (info.note) notes.push(info.note);
  const loaded = {
    modelBytes: info.modelBytes, gzipBytes: info.gzipBytes,
    runtimeBytes: info.runtimeBytes, loadMs: info.loadMs,
    paramsLabel: info.paramsLabel, metricLabel: info.metricLabel,
    metricValue: info.metricValue,
  };
  try {
    const run = async (): Promise<{
      lat: number[]; trials: number[];
    }> => {
      onProgress(`${rt.name}: ウォームアップ ×${cfg.warmupIters}`);
      for (let i = 0; i < cfg.warmupIters; i++) await rt.run(rgba);

      onProgress(`${rt.name}: レイテンシ ×${cfg.latencyIters}`);
      const lat: number[] = [];
      for (let i = 0; i < cfg.latencyIters; i++) {
        const r = await rt.run(rgba);
        lat.push(r.inferMs);
        if (i % 20 === 19) await tick();
      }

      onProgress(`${rt.name}: スループット ×${cfg.throughputTrials}試行`);
      const trials: number[] = [];
      for (let t = 0; t < cfg.throughputTrials; t++) {
        await tick();
        const t0 = performance.now();
        for (let i = 0; i < cfg.throughputImgs; i++) await rt.run(rgba);
        trials.push((cfg.throughputImgs / (performance.now() - t0)) * 1000);
      }
      return { lat, trials };
    };
    const { lat, trials } = await withTimeout(run(), cfg.runTimeoutMs, `${rt.name} run()`);
    const sorted = [...trials].sort((a, b) => a - b);
    const heapAfter = heapMB();
    return {
      runtime: rt.name, backend, status: 'ok', error: null,
      ...loaded,
      warmLatency: summarize(lat),
      throughput: { samplesPerSec: sorted[Math.floor(sorted.length / 2)], trials },
      memDeltaMB: heapBefore != null && heapAfter != null ? heapAfter - heapBefore : null,
      notes,
    };
  } catch (e) {
    notes.push(`推論失敗: ${String(e)}`);
    return {
      runtime: rt.name, backend, status: 'error', error: String(e),
      ...loaded, warmLatency: null, throughput: null, memDeltaMB: null, notes,
    };
  }
}

/** 計測環境メタ(BenchReport 相当)。webgpuAdapter は requestAdapter() が
    成功した場合のみ非 null(vendor/architecture、UA文字列より詳細な実アダプタ有無)。 */
export interface EnvInfo {
  userAgent: string;
  hardwareConcurrency: number;
  crossOriginIsolated: boolean;
  webgpuApi: boolean; // navigator.gpu の有無
  webgpuAdapter: string | null; // requestAdapter成功時のみ
}

export async function collectEnvInfo(): Promise<EnvInfo> {
  let webgpuAdapter: string | null = null;
  if (hasWebGpuApi()) {
    try {
      const gpu = (navigator as unknown as {
        gpu: { requestAdapter: () => Promise<unknown> };
      }).gpu;
      const adapter = await gpu.requestAdapter() as
        | { info?: { vendor?: string; architecture?: string }; name?: string }
        | null;
      if (adapter) {
        webgpuAdapter = adapter.info
          ? `${adapter.info.vendor ?? '?'}/${adapter.info.architecture ?? '?'}`
          : (adapter.name ?? 'adapter(詳細不明)');
      }
    } catch {
      webgpuAdapter = null;
    }
  }
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated:
      typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    webgpuApi: hasWebGpuApi(),
    webgpuAdapter,
  };
}
