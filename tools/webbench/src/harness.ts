// 計測ハーネス(WB-6)。エンジンごとに
//   ①ペイロード(モデル+ランタイム、transfer/decoded)
//   ②コールドロード(fetch→init→初回推論)
//   ③ウォーム単発レイテンシ(median/p95)
//   ④スループット(samples/s、複数試行)
//   ⑤精度(y_test照合+混同行列)
//   ⑥メモリ(usedJSHeapSize差分、Chrome限定の参考値)
// を測る。キャッシュの影響を避けるためモデルfetchは cache:'no-store'。
// 注意: 同一ページ内で複数エンジンを順に測るためJIT/GC状態は完全独立では
// ない(方法論として結果JSONに記録)。
import { fetchModel } from './data';
import { CodegenEngine } from './engines/codegen';
import { InterpEngine } from './engines/interp';
import { OnnxEngine } from './engines/onnx';
import { Packed32Engine, Packed64Engine } from './engines/packed';
import type { Engine } from './engines/types';
import { summarize, tick } from './stats';
import type { TestData } from './types';

export type DlgnVariant = 'interp' | 'packed32' | 'packed64' | 'codegen';

export interface EngineSpec {
  name: string;
  kind: 'dlgn' | 'onnx';
  modelUrl: string;
  variant?: DlgnVariant;
  threads?: number;
  /** 精度/スループット計測に使うサンプル数上限(interpの低速対策)。
      省略時は全テストセット */
  maxSamples?: number;
}

export interface BenchConfig {
  warmupIters: number;
  latencyIters: number;
  throughputTrials: number;
}

export const DEFAULT_CONFIG: BenchConfig = {
  warmupIters: 50,
  latencyIters: 300,
  throughputTrials: 5,
};

export interface PayloadMetrics {
  modelTransferBytes: number | null; // gzip後(サーバがgzip配信時)
  modelDecodedBytes: number | null;
  runtimeTransferBytes: number | null; // ort wasm等(DLGNは0=ページJSのみ)
  runtimeDecodedBytes: number | null;
}

export interface EngineResult {
  name: string;
  kind: string;
  modelUrl: string;
  payload: PayloadMetrics;
  coldLoad: { fetchMs: number; initMs: number; firstInferMs: number; totalMs: number };
  warmLatency: { median: number; p95: number; min: number; max: number; n: number };
  throughput: {
    samplesPerSec: number; trials: number[]; nSamples: number; totalMsMedian: number;
  };
  accuracy: { acc: number; nCorrect: number; n: number };
  memDeltaMB: number | null;
  notes: string[];
}

function resourceBytes(urlSubstr: string): {
  transfer: number | null; decoded: number | null;
} {
  const entries = performance
    .getEntriesByType('resource')
    .filter((e) => e.name.includes(urlSubstr)) as PerformanceResourceTiming[];
  if (entries.length === 0) return { transfer: null, decoded: null };
  const last = entries[entries.length - 1];
  // transferSize=0はキャッシュ or CORS不透過
  return {
    transfer: last.transferSize || null,
    decoded: last.decodedBodySize || null,
  };
}

function heapMB(): number | null {
  const mem = (performance as unknown as {
    memory?: { usedJSHeapSize: number };
  }).memory;
  return mem ? mem.usedJSHeapSize / (1024 * 1024) : null;
}

async function createEngine(
  spec: EngineSpec,
  data: TestData,
): Promise<{ engine: Engine; fetchMs: number; initMs: number; notes: string[] }> {
  const notes: string[] = [];
  const t0 = performance.now();
  if (spec.kind === 'dlgn') {
    const model = await fetchModel(spec.modelUrl, true);
    const t1 = performance.now();
    let engine: Engine;
    switch (spec.variant ?? 'packed32') {
      case 'interp': engine = new InterpEngine(model); break;
      case 'packed32': engine = new Packed32Engine(model); break;
      case 'packed64': engine = new Packed64Engine(model); break;
      case 'codegen': {
        const cg = new CodegenEngine(model);
        cg.warmup(data); // JIT誘発もコールドロードに計上
        engine = cg;
        break;
      }
    }
    return { engine, fetchMs: t1 - t0, initMs: performance.now() - t1, notes };
  }
  // onnx: session.createがモデル+wasmランタイムのfetchを内包する
  const engine = new OnnxEngine(spec.name, spec.modelUrl, spec.threads ?? 1);
  await engine.init();
  const t1 = performance.now();
  if (engine.effectiveThreads() !== (spec.threads ?? 1)) {
    notes.push(
      `threads=${spec.threads}要求だが実効${engine.effectiveThreads()}` +
      '(crossOriginIsolated非対応)',
    );
  }
  return { engine, fetchMs: 0, initMs: t1 - t0, notes };
}

export async function runBench(
  spec: EngineSpec,
  data: TestData,
  cfg: BenchConfig = DEFAULT_CONFIG,
  onProgress: (msg: string) => void = () => {},
): Promise<EngineResult> {
  const heapBefore = heapMB();

  // ②コールドロード
  onProgress(`${spec.name}: コールドロード`);
  const coldStart = performance.now();
  const { engine, fetchMs, initMs, notes } = await createEngine(spec, data);
  const tInit = performance.now();
  await engine.predict(data, 0, 1);
  const tFirst = performance.now();
  const coldLoad = {
    fetchMs,
    initMs,
    firstInferMs: tFirst - tInit,
    totalMs: tFirst - coldStart,
  };

  // ①ペイロード(fetch完了後にresource timingから回収)
  const model = resourceBytes(spec.modelUrl);
  const runtime = spec.kind === 'onnx'
    ? resourceBytes('ort-wasm')
    : { transfer: 0, decoded: 0 };
  const payload: PayloadMetrics = {
    modelTransferBytes: model.transfer,
    modelDecodedBytes: model.decoded,
    runtimeTransferBytes: runtime.transfer,
    runtimeDecodedBytes: runtime.decoded,
  };

  // ③ウォーム単発レイテンシ
  onProgress(`${spec.name}: ウォーム単発 ×${cfg.latencyIters}`);
  for (let i = 0; i < cfg.warmupIters; i++) {
    await engine.predict(data, i % data.n, 1);
  }
  const lat: number[] = [];
  for (let i = 0; i < cfg.latencyIters; i++) {
    const t = performance.now();
    await engine.predict(data, i % data.n, 1);
    lat.push(performance.now() - t);
    if (i % 100 === 99) await tick();
  }

  // ④スループット+⑤精度
  const nSamples = Math.min(spec.maxSamples ?? data.n, data.n);
  if (nSamples < data.n) {
    notes.push(`精度/スループットは先頭${nSamples}件で計測(低速エンジンの時間対策)`);
  }
  onProgress(`${spec.name}: スループット ${nSamples}件 ×${cfg.throughputTrials}試行`);
  const trials: number[] = [];
  const totalMs: number[] = [];
  let preds: Int32Array<ArrayBufferLike> = new Int32Array(0);
  for (let tr = 0; tr < cfg.throughputTrials; tr++) {
    await tick();
    const t = performance.now();
    preds = await engine.predict(data, 0, nSamples);
    const dt = performance.now() - t;
    totalMs.push(dt);
    trials.push((nSamples / dt) * 1000);
    onProgress(
      `${spec.name}: 試行${tr + 1}/${cfg.throughputTrials} ` +
      `${Math.round((nSamples / dt) * 1000).toLocaleString()} samples/s`,
    );
  }
  const y = data.meta.y_test;
  let nCorrect = 0;
  for (let i = 0; i < nSamples; i++) {
    if (preds[i] === y[i]) nCorrect++;
  }

  const heapAfter = heapMB();
  const sortedTrials = [...trials].sort((a, b) => a - b);
  const sortedTotal = [...totalMs].sort((a, b) => a - b);
  return {
    name: spec.name,
    kind: spec.kind + (spec.variant ? `/${spec.variant}` : ''),
    modelUrl: spec.modelUrl,
    payload,
    coldLoad,
    warmLatency: summarize(lat),
    throughput: {
      samplesPerSec: sortedTrials[Math.floor(sortedTrials.length / 2)],
      trials,
      nSamples,
      totalMsMedian: sortedTotal[Math.floor(sortedTotal.length / 2)],
    },
    accuracy: { acc: nCorrect / nSamples, nCorrect, n: nSamples },
    memDeltaMB:
      heapBefore !== null && heapAfter !== null
        ? heapAfter - heapBefore
        : null,
    notes,
  };
}

export interface BenchReport {
  timestamp: string;
  userAgent: string;
  hardwareConcurrency: number;
  crossOriginIsolated: boolean;
  config: BenchConfig;
  engines: EngineResult[];
}

export function makeReport(
  engines: EngineResult[], cfg: BenchConfig,
): BenchReport {
  return {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated:
      typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    config: cfg,
    engines,
  };
}
