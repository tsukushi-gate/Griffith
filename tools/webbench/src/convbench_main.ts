// conv比較ベンチ: DLGN(irflat/ConvRuntime)vs ONNX Runtime Web(SmallUNet)。
// 同一 64x64 入力で 配信サイズ(gzip)/コールド起動/スループット(imgs/s) を測る。
// 結果は #out に JSON で出力(Browser pane / 手動で results/webbench へ保存)。
import * as ort from 'onnxruntime-web';

import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import { ConvRuntime } from './engines/conv';
import type { ConvEngine } from './engines/conv';
import type { ConvModelDoc } from './types';
import { asset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { REFERENCES, applyReferences } from './references';

const $ = (id: string): HTMLElement => document.getElementById(id)!;
const DLGN_URL = asset('/models/seg_convdlgn_k128.json');
const ONNX_URL = asset('/models/seg_unet.onnx');
const S = 64;
const WARMUP = 3;
const ITERS = 30;

function median(a: number[]): number {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

/** 決定的な擬似ランダム 64x64 RGBA 群(ワークロード。内容は速度に無関係)。 */
function makeInputs(n: number): Uint8ClampedArray[] {
  let seed = 12345;
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const inputs: Uint8ClampedArray[] = [];
  for (let k = 0; k < n; k++) {
    const rgba = new Uint8ClampedArray(S * S * 4);
    for (let i = 0; i < S * S; i++) {
      rgba[i * 4] = (rnd() * 255) | 0;
      rgba[i * 4 + 1] = (rnd() * 255) | 0;
      rgba[i * 4 + 2] = (rnd() * 255) | 0;
      rgba[i * 4 + 3] = 255;
    }
    inputs.push(rgba);
  }
  return inputs;
}

function resourceBytes(sub: string): { transfer: number | null; decoded: number | null } {
  const es = performance.getEntriesByType('resource')
    .filter((e) => e.name.includes(sub)) as PerformanceResourceTiming[];
  if (!es.length) return { transfer: null, decoded: null };
  const last = es[es.length - 1];
  return { transfer: last.transferSize || null, decoded: last.decodedBodySize || null };
}

/** DLGN(ConvRuntime)を指定エンジンで計測。fetchはengineに依らないので、
    docは呼び出し側から渡し、コールド起動の fetch は別途1回だけ計上する。 */
async function benchDlgn(
  engine: ConvEngine, doc: ConvModelDoc, fetchMs: number,
  inputs: Uint8ClampedArray[], log: (s: string) => void,
) {
  log(`DLGN(${engine}): 構築`);
  const tCtor0 = performance.now();
  const rt = new ConvRuntime(doc.ir, { engine });
  const p = doc.preproc;
  // doc.preproc をそのまま使う(kind と低レベル特徴パラメータを落とさない)
  const thermo = p;
  const nOut = doc.readout!.spatial!.n_out;
  const hw = S * S;
  const infer = (rgba: Uint8ClampedArray): Int32Array => {
    const nchw = rgbaToNchw(rgba, S, S);
    const bits = encodeImageAuto(nchw, thermo);
    const { data } = rt.scores(bits, 1);
    return rt.argmaxPerPixel(data, 1, nOut, hw);
  };
  const tInit = performance.now();
  infer(inputs[0]); // 初回推論(コールドに計上)
  const firstInferMs = performance.now() - tInit;
  const initMs = tInit - tCtor0; // 構築(lower/native)時間
  const cold = { fetchMs, initMs, firstInferMs,
    totalMs: fetchMs + initMs + firstInferMs };

  for (let i = 0; i < WARMUP; i++) infer(inputs[i % inputs.length]);
  log(`DLGN(${engine}): スループット ${ITERS}枚`);
  const per: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    infer(inputs[i % inputs.length]);
    per.push(performance.now() - t);
  }
  const medMs = median(per);
  const model = resourceBytes('seg_convdlgn_k128');
  return {
    engine: `DLGN ${engine} (SegConvDLGN k128)`,
    total_luts: doc.resources.total_luts,
    num_gates_weight_shared: (doc.eval as Record<string, unknown>)
      .num_gates_weight_shared ?? null,
    model_transfer_gzip_bytes: model.transfer,
    model_decoded_bytes: model.decoded,
    runtime_transfer_bytes: 0,
    coldMs: cold,
    perImageMsMedian: medMs,
    imgsPerSec: 1000 / medMs,
    hard_miou: (doc.eval as Record<string, unknown>).best_hard_miou ?? null,
  };
}

async function benchOnnx(inputs: Uint8ClampedArray[], log: (s: string) => void) {
  log('ORT: セッション作成(wasm)');
  ort.env.wasm.numThreads = 1;
  const cold0 = performance.now();
  const session = await ort.InferenceSession.create(ONNX_URL, {
    executionProviders: ['wasm'],
  });
  const tInit = performance.now();
  const hw = S * S;
  const infer = async (rgba: Uint8ClampedArray): Promise<Int32Array> => {
    const nchw = rgbaToNchw(rgba, S, S); // 生[0,1](SmallUNetは正規化なし)
    const tensor = new ort.Tensor('float32', Float32Array.from(nchw), [1, 3, S, S]);
    const out = await session.run({ input: tensor });
    const logits = out.logits.data as Float32Array; // [1,2,H,W]
    const labels = new Int32Array(hw);
    for (let px = 0; px < hw; px++) {
      labels[px] = logits[hw + px] > logits[px] ? 1 : 0;
    }
    return labels;
  };
  await infer(inputs[0]);
  const cold = { fetchMs: 0, initMs: tInit - cold0,
    firstInferMs: 0, totalMs: performance.now() - cold0 };

  for (let i = 0; i < WARMUP; i++) await infer(inputs[i % inputs.length]);
  log(`ORT: スループット ${ITERS}枚`);
  const per: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    await infer(inputs[i % inputs.length]);
    per.push(performance.now() - t);
  }
  const medMs = median(per);
  const model = resourceBytes('seg_unet.onnx');
  const wasm = resourceBytes('ort-wasm');
  // teacher の参照値は references.ts に集約(ONNXにメタデータが無いため)。
  const ref = REFERENCES.seg_unet_teacher;
  return {
    engine: 'ONNX Runtime Web wasm (SmallUNet)',
    params: ref.params,
    model_transfer_gzip_bytes: model.transfer,
    model_decoded_bytes: model.decoded,
    runtime_transfer_bytes: wasm.transfer,
    runtime_decoded_bytes: wasm.decoded,
    coldMs: cold,
    perImageMsMedian: medMs,
    imgsPerSec: 1000 / medMs,
    hard_miou: ref.metricValue,
  };
}

async function main(): Promise<void> {
  const log = (s: string): void => { $('status').textContent = s; };
  try {
    const inputs = makeInputs(ITERS);
    log('DLGN: モデル取得');
    const tF0 = performance.now();
    const doc = await fetch(DLGN_URL, { cache: 'no-store' })
      .then((r) => r.json()) as ConvModelDoc;
    const fetchMs = performance.now() - tF0;
    // 説明文の数値はロード済みモデルJSON/参照値テーブルから描画する。
    applyModelMeta(doc);
    applyReferences();
    const packed = await benchDlgn('packed', doc, fetchMs, inputs, log);
    const irflat = await benchDlgn('irflat', doc, fetchMs, inputs, log);
    const onnx = await benchOnnx(inputs, log);
    const report = {
      task: 'seg DLGN (packed vs irflat) vs ORT Web',
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      crossOriginIsolated:
        typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
      input: { height: S, width: S, iters: ITERS, warmup: WARMUP, batch: 1 },
      engines: [packed, irflat, onnx],
      speedup_packed_over_ort: onnx.perImageMsMedian / packed.perImageMsMedian,
      speedup_packed_over_irflat: irflat.perImageMsMedian / packed.perImageMsMedian,
    };
    log('完了');
    $('out').textContent = JSON.stringify(report, null, 2);
    (window as unknown as Record<string, unknown>).__BENCH__ = report;
  } catch (e) {
    log(`失敗: ${String(e)}`);
    $('out').textContent = String((e as Error).stack ?? e);
  }
}

void main();
