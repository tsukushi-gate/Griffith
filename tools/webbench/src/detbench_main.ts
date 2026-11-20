// 検出比較ベンチ: DLGN(packed/irflat, CenterNetLUT)vs
// ONNX Runtime Web(CenterNetCNN teacher)。同一 96×96 入力・同一 thermometer
// 前処理・同一 CenterNet decode で、配信サイズ(gzip)/コールド起動/
// スループット(imgs/s)を測る。結果は #out に JSON 出力(手動で
// results/webbench へ保存)。
import * as ort from 'onnxruntime-web';

import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import { decodeDetections } from './detdecode';
import { ConvRuntime } from './engines/conv';
import type { ConvEngine } from './engines/conv';
import type { DetReadout } from './types';
import type { DetModel } from './detdemo';
import { asset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { REFERENCES, applyReferences } from './references';

const $ = (id: string): HTMLElement => document.getElementById(id)!;
const DLGN_URL = asset('/models/det_multimnist_lut_k256.json');
const ONNX_URL = asset('/models/det_cnn_teacher.onnx');
const S = 96;
const WARMUP = 3;
const ITERS = 20;

function median(a: number[]): number {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

/** 決定的な擬似ランダム 96×96 グレースケール RGBA 群(速度は内容非依存)。 */
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
      const v = (rnd() * 255) | 0;
      rgba[i * 4] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
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

async function benchDlgn(
  engine: ConvEngine, doc: DetModel['doc'], fetchMs: number,
  inputs: Uint8ClampedArray[], log: (s: string) => void,
) {
  log(`DLGN(${engine}): 構築`);
  const tCtor0 = performance.now();
  const rt = new ConvRuntime(doc.ir, { engine });
  // doc.preproc をそのまま渡す。以前は mean/std/thresholds/height/width だけを
  // 抜き出しており、kind と低レベル特徴(edge/curv/chroma)のパラメータが欠落して
  // image_lowlevel モデルを thermometer として誤符号化していた。
  const thermo = doc.preproc;
  const ro = doc.readout;
  const infer = (rgba: Uint8ClampedArray): number => {
    const nchw = rgbaToNchw(rgba, S, S);
    const bits = encodeImageAuto(nchw, thermo);
    const { data } = rt.scores(bits, 1);
    return decodeDetections(data, ro).length;
  };
  const tInit = performance.now();
  infer(inputs[0]);
  const firstInferMs = performance.now() - tInit;
  const initMs = tInit - tCtor0;
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
  const model = resourceBytes('det_multimnist_lut_k256');
  const ev = doc.eval as Record<string, unknown>;
  return {
    engine: `DLGN ${engine} (CenterNetLUT k64)`,
    total_luts: doc.resources.total_luts,
    num_gates_weight_shared: ev.num_gates_weight_shared ?? null,
    model_transfer_gzip_bytes: model.transfer,
    model_decoded_bytes: model.decoded,
    runtime_transfer_bytes: 0,
    coldMs: cold,
    perImageMsMedian: medMs,
    imgsPerSec: 1000 / medMs,
    hard_map: ev.best_hard_map ?? null,
  };
}

/** ORT 出力(hm/size/off, 各 [1,C,24,24] Float32)を decode 用の連結列へ。 */
function ortReadout(C: number, H: number, W: number): DetReadout {
  const hw = H * W;
  return {
    kind: 'detection',
    heads: {
      hm: { offset: 0, n_out: C, h: H, w: W, scale: 1 },
      size: { offset: C * hw, n_out: 2, h: H, w: W, scale: 1 },
      off: { offset: (C + 2) * hw, n_out: 2, h: H, w: W, scale: 1 },
    },
    output_stride: 4, top_k: 100, threshold: 0.1, n_classes: C,
  };
}

async function benchOnnx(
  inputs: Uint8ClampedArray[], log: (s: string) => void,
) {
  log('ORT: セッション作成(wasm)');
  ort.env.wasm.numThreads = 1;
  const cold0 = performance.now();
  const session = await ort.InferenceSession.create(ONNX_URL, {
    executionProviders: ['wasm'],
  });
  const tInit = performance.now();
  const C = 10; const H = S / 4; const W = S / 4; const hw = H * W;
  const ro = ortReadout(C, H, W);
  // 修正: カラー teacher(det_cnn_teacher.onnx)は **生RGB 3ch** を消費する
  // (det.html の DetOnnxRuntime と同じ経路。compareDet.ts が
  //  「生RGBを直接消費するため thermometer パラメータは不要」と記す通り)。
  // 旧実装は thermometer ビットを流し込んでおり、teacher の入力仕様と食い違っていた。
  const infer = async (rgba: Uint8ClampedArray): Promise<number> => {
    const nchw = rgbaToNchw(rgba, S, S);
    const tensor = new ort.Tensor('float32', nchw, [1, 3, S, S]);
    const out = await session.run({ input: tensor });
    const hm = out.hm.data as Float32Array; // [1,C,H,W]
    const sz = out.size.data as Float32Array;
    const off = out.off.data as Float32Array;
    const scores = new Float64Array((C + 4) * hw);
    scores.set(hm, 0);
    scores.set(sz, C * hw);
    scores.set(off, (C + 2) * hw);
    return decodeDetections(scores, ro).length;
  };
  await infer(inputs[0]);
  const cold = { fetchMs: 0, initMs: tInit - cold0, firstInferMs: 0,
    totalMs: performance.now() - cold0 };

  for (let i = 0; i < WARMUP; i++) await infer(inputs[i % inputs.length]);
  log(`ORT: スループット ${ITERS}枚`);
  const per: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    await infer(inputs[i % inputs.length]);
    per.push(performance.now() - t);
  }
  const medMs = median(per);
  const model = resourceBytes('det_cnn_teacher.onnx');
  const wasm = resourceBytes('ort-wasm');
  // このページが読む ONNX_URL = det_cnn_teacher.onnx は det.html(カラー版)と
  // 同一ファイル。参照値を references.ts の1箇所に集約したので、
  // 「同じ重みで2ページが別の数字を出す」事故は構造的に起きない。
  const ref = REFERENCES.det_color_teacher;
  return {
    engine: 'ONNX Runtime Web wasm (CenterNetCNN teacher)',
    params: ref.params,
    model_transfer_gzip_bytes: model.transfer,
    model_decoded_bytes: model.decoded,
    runtime_transfer_bytes: wasm.transfer,
    runtime_decoded_bytes: wasm.decoded,
    coldMs: cold,
    perImageMsMedian: medMs,
    imgsPerSec: 1000 / medMs,
    hard_map: ref.metricValue,
  };
}

async function main(): Promise<void> {
  const log = (s: string): void => { $('status').textContent = s; };
  try {
    const inputs = makeInputs(ITERS);
    log('DLGN: モデル取得');
    const tF0 = performance.now();
    const doc = await fetch(DLGN_URL, { cache: 'no-store' })
      .then((r) => r.json()) as DetModel['doc'];
    const fetchMs = performance.now() - tF0;
    // 説明文の数値はロード済みモデルJSON/参照値テーブルから描画する。
    applyModelMeta(doc);
    applyReferences();
    const packed = await benchDlgn('packed', doc, fetchMs, inputs, log);
    const irflat = await benchDlgn('irflat', doc, fetchMs, inputs, log);
    let onnx: unknown = null;
    try {
      onnx = await benchOnnx(inputs, log);
    } catch (e) {
      log(`ORT スキップ: ${String(e)}`);
    }
    const engines: unknown[] = [packed, irflat];
    if (onnx) engines.push(onnx);
    const report = {
      task: 'det DLGN (packed vs irflat) vs ORT Web (CNN teacher)',
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      crossOriginIsolated:
        typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
      input: { height: S, width: S, iters: ITERS, warmup: WARMUP, batch: 1 },
      engines,
      speedup_packed_over_ort: onnx
        ? (onnx as { perImageMsMedian: number }).perImageMsMedian / packed.perImageMsMedian
        : null,
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
