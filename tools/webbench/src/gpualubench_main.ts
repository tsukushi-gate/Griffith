// WebGPU側の自己較正ルーフライン計測ページ(gpualubench.html)。
//
// scripts/gpu_calib/metal_calib.mm(純ALU較正)・gate2_ab.mm(gate2 A/B)の
// Metal側計測と対になる、WebGPU側の同型計測。node/vitestはWebGPUを実行
// できないため(wasmgpu_verify.html / mtverify.html と同じ理由)、
// ここが正式な実行経路。Playwright駆動は scripts/gpu_alu_bench.mjs。
//
//   (a) 純ALU較正: buildAluCalibWgsl(unroll) を dispatch し、timestamp query
//       でGPU時間を取る。(総命令数 ÷ GPU時間) = このマシンのWebGPU実測ALU
//       ピーク(instr/s)。3回測って mean/best/worst/stddev を報告する。
//   (b) gate2 A/B: switch(tt&0xf)版 vs 無分岐マスク式版を、実カーネルと同じ
//       「命令列を読んでスクラッチへread-modify-write」ループで比較する。
//       並列度2条件(低=32相当/高=1,048,576)×各8回、**出力の一致検査を必ず
//       行う**(全要素のbit-exact比較。ベンチに一致検査を入れ忘れて嘘の結論が
//       出た前例があるため)。
//
// 結果は window.__gpuAluBenchResult に積み、document.title を
// 'GPUALU_DONE'(timestamp-query非対応など致命的な場合は 'GPUALU_ERROR')にする。
import {
  ALU_OPS_PER_ITER,
  buildAluCalibWgsl,
  buildGate2AbWgsl,
  genGate2Program,
  scaleProgramForThreads,
} from './engines/gpuAluCalib';
import { requestWebgpuDevice } from './engines/webgpuConv';

const $ = (): HTMLElement => document.getElementById('status')!;
function log(line: string): void {
  $().textContent += `${line}\n`;
  // eslint-disable-next-line no-console -- Playwright側のconsoleログ収集にも載せる
  console.log(line);
}

function parseIntParam(name: string, dflt: number): number {
  const v = new URL(window.location.href).searchParams.get(name);
  if (v === null) return dflt;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : dflt;
}

// --- 汎用: 単発dispatchをtimestamp queryで包んで実行し、GPU経過時間(ns)を返す ---
interface TsHarness {
  querySet: GPUQuerySet;
  resolveBuf: GPUBuffer;
  readBuf: GPUBuffer;
}

function makeTsHarness(device: GPUDevice): TsHarness {
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolveBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  return { querySet, resolveBuf, readBuf };
}

/** dispatchOne 1回ぶんをtimestampで計測してGPU経過ns(bigint)を返す。
    withTimestamps=falseならウォームアップ用(タイムスタンプもmapAsyncもしない
    最短経路)。 */
async function runDispatch(
  device: GPUDevice,
  ts: TsHarness | null,
  encodeDispatch: (pass: GPUComputePassEncoder) => void,
): Promise<bigint> {
  const encoder = device.createCommandEncoder();
  const desc: GPUComputePassDescriptor = {};
  if (ts) {
    desc.timestampWrites = {
      querySet: ts.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }
  const pass = encoder.beginComputePass(desc);
  encodeDispatch(pass);
  pass.end();
  if (ts) {
    encoder.resolveQuerySet(ts.querySet, 0, 2, ts.resolveBuf, 0);
    encoder.copyBufferToBuffer(ts.resolveBuf, 0, ts.readBuf, 0, 16);
  }
  device.queue.submit([encoder.finish()]);
  if (!ts) {
    await device.queue.onSubmittedWorkDone();
    return 0n;
  }
  await ts.readBuf.mapAsync(GPUMapMode.READ);
  const raw = new BigUint64Array(ts.readBuf.getMappedRange().slice(0));
  ts.readBuf.unmap();
  const begin = raw[0];
  const end = raw[1];
  return end >= begin ? end - begin : 0n;
}

function dispatchCount1d(total: number, wgSize = 64): number {
  return Math.max(1, Math.ceil(total / wgSize));
}

// ============================================================
// (a) 純ALU較正カーネル
// ============================================================

interface AluTrial {
  gpuMs: number;
  instrPerSec: number;
}

interface AluCalibResult {
  unroll: number;
  opsPerIter: number;
  totalThreads: number;
  totalInstrPerThread: number;
  totalInstr: number;
  trials: AluTrial[];
  meanInstrPerSec: number;
  bestInstrPerSec: number;
  worstInstrPerSec: number;
  stddevPct: number;
  sanity: { out0: number; out1: number; outMid: number; outLast: number; folded: boolean };
}

async function runAluCalibration(
  device: GPUDevice, timestampOk: boolean,
  unroll: number, totalThreads: number, trials: number, warmups: number,
): Promise<AluCalibResult> {
  log(`\n== (a) 純ALU較正: unroll=${unroll} threads=${totalThreads} `
    + `trials=${trials}(+warmup${warmups}) ==`);
  const code = buildAluCalibWgsl(unroll);
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const outBuf = device.createBuffer({
    size: totalThreads * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniformBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([0x1234abcd, totalThreads, 0, 0]));
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outBuf } },
      { binding: 1, resource: { buffer: uniformBuf } },
    ],
  });
  const nGroups = dispatchCount1d(totalThreads);
  const encodeDispatch = (pass: GPUComputePassEncoder): void => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nGroups);
  };

  device.pushErrorScope('validation');
  for (let w = 0; w < warmups; w++) await runDispatch(device, null, encodeDispatch);
  const err = await device.popErrorScope();
  if (err) throw new Error(`ALU較正カーネルのvalidationエラー(黙殺せずthrow): ${err.message}`);

  const ts = timestampOk ? makeTsHarness(device) : null;
  const totalInstrPerThread = unroll * ALU_OPS_PER_ITER;
  const totalInstr = totalInstrPerThread * totalThreads;
  const trialResults: AluTrial[] = [];
  for (let t = 0; t < trials; t++) {
    const ns = await runDispatch(device, ts, encodeDispatch);
    const gpuMs = Number(ns) / 1e6;
    const instrPerSec = ns > 0n ? totalInstr / (Number(ns) / 1e9) : NaN;
    trialResults.push({ gpuMs, instrPerSec });
    log(`  trial ${t}: gpu_elapsed=${gpuMs.toFixed(4)}ms  instr/s=${instrPerSec.toExponential(4)}`);
  }
  const rates = trialResults.map((r) => r.instrPerSec);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const variance = rates.reduce((a, b) => a + (b - mean) ** 2, 0) / rates.length;
  const stddev = Math.sqrt(Math.max(0, variance));
  const best = Math.max(...rates);
  const worst = Math.min(...rates);
  log(`  mean instr/s = ${mean.toExponential(4)} `
    + `(best=${best.toExponential(4)} worst=${worst.toExponential(4)} `
    + `stddev/mean=${((100 * stddev) / mean).toFixed(2)}%)`);

  // sanity: 定数畳み込みで全スレッド同じ値になっていないか(=最適化で消えていないか)
  const sanityReadBuf = device.createBuffer({
    size: totalThreads * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, sanityReadBuf, 0, totalThreads * 4);
    device.queue.submit([enc.finish()]);
  }
  await sanityReadBuf.mapAsync(GPUMapMode.READ);
  const outArr = new Uint32Array(sanityReadBuf.getMappedRange().slice(0));
  sanityReadBuf.unmap();
  const out0 = outArr[0];
  const out1 = outArr[1];
  const outMid = outArr[Math.floor(totalThreads / 2)];
  const outLast = outArr[totalThreads - 1];
  const folded = out0 === out1 && out1 === outMid && outMid === outLast;
  log(`  sanity: out[0]=${out0} out[1]=${out1} out[mid]=${outMid} out[last]=${outLast}`
    + `${folded ? '  !! 全て同一値=定数畳み込みの疑い' : ''}`);

  return {
    unroll,
    opsPerIter: ALU_OPS_PER_ITER,
    totalThreads,
    totalInstrPerThread,
    totalInstr,
    trials: trialResults,
    meanInstrPerSec: mean,
    bestInstrPerSec: best,
    worstInstrPerSec: worst,
    stddevPct: (100 * stddev) / mean,
    sanity: { out0, out1, outMid, outLast, folded },
  };
}

// ============================================================
// (b) gate2 A/Bマイクロベンチ
// ============================================================

interface Gate2VariantResult {
  medianMs: number;
  meanMs: number;
  trials: number[];
}

interface Gate2ConfigResult {
  threads: number;
  switchResult: Gate2VariantResult;
  maskResult: Gate2VariantResult;
  outputsMatch: boolean;
  mismatchCount: number;
  speedup: number; // >1: maskの方が速い
  verdict: string;
}

interface Gate2AbResult {
  nInstr: number;
  nSlots: number;
  configs: Gate2ConfigResult[];
  allMatch: boolean;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

async function runGate2Variant(
  device: GPUDevice, timestampOk: boolean,
  pipeline: GPUComputePipeline, threads: number, nSlots: number, nInstr: number,
  instrBuf: GPUBuffer, trials: number, warmups: number,
): Promise<{ result: Gate2VariantResult; finalScratch: Uint32Array }> {
  const scratchBuf = device.createBuffer({
    size: nSlots * threads * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniformBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([nInstr, threads, nSlots, 0]));
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: scratchBuf } },
      { binding: 1, resource: { buffer: instrBuf } },
      { binding: 2, resource: { buffer: uniformBuf } },
    ],
  });
  const nGroups = dispatchCount1d(threads);
  const encodeDispatch = (pass: GPUComputePassEncoder): void => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nGroups);
  };

  device.pushErrorScope('validation');
  for (let w = 0; w < warmups; w++) await runDispatch(device, null, encodeDispatch);
  const err = await device.popErrorScope();
  if (err) throw new Error(`gate2 A/Bカーネルのvalidationエラー(黙殺せずthrow): ${err.message}`);

  const ts = timestampOk ? makeTsHarness(device) : null;
  const msTrials: number[] = [];
  for (let t = 0; t < trials; t++) {
    const ns = await runDispatch(device, ts, encodeDispatch);
    msTrials.push(Number(ns) / 1e6);
  }

  const readBuf = device.createBuffer({
    size: nSlots * threads * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(scratchBuf, 0, readBuf, 0, nSlots * threads * 4);
    device.queue.submit([enc.finish()]);
  }
  await readBuf.mapAsync(GPUMapMode.READ);
  const finalScratch = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  const mean = msTrials.reduce((a, b) => a + b, 0) / msTrials.length;
  return {
    result: { medianMs: median(msTrials), meanMs: mean, trials: msTrials },
    finalScratch,
  };
}

async function runGate2Ab(
  device: GPUDevice, timestampOk: boolean,
  nInstr: number, nSlots: number, trials: number, warmups: number,
): Promise<Gate2AbResult> {
  log(`\n== (b) gate2 A/B: n_instr=${nInstr} n_slots=${nSlots} `
    + `trials=${trials}(+warmup${warmups}) ==`);
  const code = buildGate2AbWgsl();
  const module = device.createShaderModule({ code });
  const pipelineSwitch = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main_switch' },
  });
  const pipelineMask = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main_mask' },
  });

  // gate2_ab.mm と同じ2並列度: 低=32スレッド相当 / 高=1,048,576(≒100万要素)。
  const PARALLELISM_CONFIGS = [32, 1 << 20];
  const rawProg = genGate2Program(nInstr, nSlots, 0xc0ffee);
  const configs: Gate2ConfigResult[] = [];
  let allMatch = true;

  for (const threads of PARALLELISM_CONFIGS) {
    const scaled = scaleProgramForThreads(rawProg, threads);
    const instrBuf = device.createBuffer({
      size: Math.max(4, scaled.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(instrBuf, 0, scaled as unknown as ArrayBufferView<ArrayBuffer>);

    const rs = await runGate2Variant(
      device, timestampOk, pipelineSwitch, threads, nSlots, nInstr, instrBuf, trials, warmups,
    );
    const rm = await runGate2Variant(
      device, timestampOk, pipelineMask, threads, nSlots, nInstr, instrBuf, trials, warmups,
    );

    let mismatchCount = 0;
    for (let i = 0; i < rs.finalScratch.length; i++) {
      if (rs.finalScratch[i] !== rm.finalScratch[i]) mismatchCount++;
    }
    const outputsMatch = mismatchCount === 0;
    allMatch = allMatch && outputsMatch;

    const speedup = rs.result.medianMs / rm.result.medianMs; // >1: maskの方が速い
    let verdict: string;
    if (!outputsMatch) verdict = '不一致(判定不能)';
    else if (speedup > 1.2) verdict = 'mask有意に速い(置換候補)';
    else if (speedup < 1 / 1.2) verdict = 'switch有意に速い';
    else verdict = '誤差内(switchで良い、書き換え不要)';

    log(`  threads=${threads}: switch median=${rs.result.medianMs.toFixed(4)}ms `
      + `mean=${rs.result.meanMs.toFixed(4)}ms`);
    log(`  threads=${threads}: mask   median=${rm.result.medianMs.toFixed(4)}ms `
      + `mean=${rm.result.meanMs.toFixed(4)}ms  speedup(switch/mask)=${speedup.toFixed(3)}x`);
    log(`  threads=${threads}: 出力一致=${outputsMatch ? `OK(bit-exact, ${rs.finalScratch.length}要素)` : `NG(${mismatchCount}要素不一致)`}  判定: ${verdict}`);

    configs.push({
      threads,
      switchResult: rs.result,
      maskResult: rm.result,
      outputsMatch,
      mismatchCount,
      speedup,
      verdict,
    });
  }

  log(`\n総合: 出力一致検査 ${allMatch ? '全パターンOK' : 'NG(要調査)'}`);
  return { nInstr, nSlots, configs, allMatch };
}

// ============================================================
// main
// ============================================================

export interface GpuAluBenchResult {
  timestampSupported: boolean;
  adapterInfo: { vendor: string | null; architecture: string | null; description: string | null } | null;
  aluCalibration: AluCalibResult | null;
  gate2Ab: Gate2AbResult | null;
  error?: string;
}

declare global {
  interface Window {
    __gpuAluBenchResult?: GpuAluBenchResult;
  }
}

async function main(): Promise<void> {
  $().textContent = '計測中…\n';
  try {
    const device = await requestWebgpuDevice();
    if (!device) {
      throw new Error('WebGPUデバイスを取得できない(navigator.gpu無しまたはアダプタ無し)');
    }
    const timestampSupported = device.features.has('timestamp-query');
    log(`timestamp-query対応: ${timestampSupported}`);
    if (!timestampSupported) {
      log('!! timestamp-query非対応環境。GPU時間は計測できない(黙殺せず明示する)');
    }

    let adapterInfo: GpuAluBenchResult['adapterInfo'] = null;
    try {
      const nav = navigator as unknown as { gpu?: GPU };
      const adapter = await nav.gpu?.requestAdapter();
      const info = adapter?.info
        ?? ((adapter as unknown as { requestAdapterInfo?: () => Promise<GPUAdapterInfo> })
          ?.requestAdapterInfo ? await (adapter as unknown as {
            requestAdapterInfo: () => Promise<GPUAdapterInfo>;
          }).requestAdapterInfo() : undefined);
      if (info) {
        adapterInfo = {
          vendor: info.vendor ?? null,
          architecture: info.architecture ?? null,
          description: info.description ?? null,
        };
      }
    } catch {
      adapterInfo = null;
    }

    const unroll = parseIntParam('unroll', 256);
    const aluThreads = parseIntParam('threads', 1 << 20);
    const aluTrials = parseIntParam('trials', 3);
    const aluWarmups = parseIntParam('warmups', 8);
    const nInstr = parseIntParam('instr', 2000);
    const nSlots = parseIntParam('slots', 32);
    const gTrials = parseIntParam('gtrials', 8);
    const gWarmups = parseIntParam('gwarmups', 4);

    const aluCalibration = await runAluCalibration(
      device, timestampSupported, unroll, aluThreads, aluTrials, aluWarmups,
    );
    const gate2Ab = await runGate2Ab(device, timestampSupported, nInstr, nSlots, gTrials, gWarmups);

    window.__gpuAluBenchResult = {
      timestampSupported, adapterInfo, aluCalibration, gate2Ab,
    };
    log('\nALL DONE');
    document.title = 'GPUALU_DONE';
  } catch (e) {
    const msg = String((e as Error).stack ?? e);
    log(`\nERROR: ${msg}`);
    window.__gpuAluBenchResult = {
      timestampSupported: false, adapterInfo: null, aluCalibration: null, gate2Ab: null, error: msg,
    };
    document.title = 'GPUALU_ERROR';
  }
}

void main();
