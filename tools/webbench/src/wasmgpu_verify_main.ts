// DLGN packed / packed-wasm / packed-webgpu の整数完全一致を実モデルで
// ブラウザ上で検証する(node/vitestはWebGPUを実行できないため、GPU実行系の
// 正しさ確認はこのページが正式な経路)。
// スコア(argmax前の生の整数出力)を全画素・全チャネルで比較する
// (evalConv2dPacked/wasmConv/webgpuConvが同一のビット結果を出すことの直接証拠)。
//
// ワードパック版executor 'bitplane-webgpu'(webgpuBitplane.ts)を
// 追加した。こちらは5タスク全モデル(seg + det4種)について
// 「CPU bitplane(bitplaneGraph.ts、conv_golden/seg_e2e_golden/det_e2e_golden で
// 整数一致が固定済み)」を参照実装として生スコアの完全一致を取る。
// bitplane-webgpu → packed-webgpu のフォールバックが黙って起きていないことも
// webgpuFallbackNote / dispatch数で確認する(黙殺根絶)。
import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import { loadSegModel, rgbFromB64 as segRgbFromB64 } from './segdemo';
import { loadDetModel, grayFromB64, rgbFromB64 as detRgbFromB64 } from './detdemo';
import { asset } from './basePath';
import { PIPELINE_DEPTH } from './engines/webgpuBitplane';
import type { ConvEngine, ConvRuntime, FrameInput } from './engines/conv';
import type { ImagePreprocDoc } from './convpreproc';

const $ = (): HTMLElement => document.getElementById('status')!;

function log(line: string): void {
  $().textContent += `${line}\n`;
}

/** 決定的な擬似ランダムRGBA(segbackends_main.tsと同じ生成器)。 */
function makeInput(size: number, seed: number): Uint8ClampedArray {
  let s = seed;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    rgba[i * 4] = (rnd() * 255) | 0;
    rgba[i * 4 + 1] = (rnd() * 255) | 0;
    rgba[i * 4 + 2] = (rnd() * 255) | 0;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function arraysEqual(a: Int32Array, b: Int32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function firstDiff(a: Int32Array, b: Int32Array): string {
  if (a.length !== b.length) return `長さ違い ${a.length} vs ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return `index=${i} ref=${a[i]} test=${b[i]}`;
  }
  return '(差分なし)';
}

/** モデル1本ぶんの「ロード → 入力エンコード → scoresAsync」を型で束ねる
    (seg/det でローダが違うだけで、以降の比較手順は完全に共通)。 */
interface LoadedModel {
  runtime: ConvRuntime;
  h: number;
  w: number;
  thermo: ImagePreprocDoc;
  /** doc に埋め込まれた実サンプル画像(RGBA)。合成ノイズだけでなく
      実画像でも前処理のビット一致を確認するために使う。 */
  sampleRgbas: Uint8ClampedArray[];
}

type Kind = 'seg' | 'det';

/** HWC uint8(3ch or 1ch)→ RGBA(A=255)。 */
function toRgba(src: Uint8Array, n: number, channels: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    if (channels === 3) {
      out[i * 4] = src[i * 3];
      out[i * 4 + 1] = src[i * 3 + 1];
      out[i * 4 + 2] = src[i * 3 + 2];
    } else {
      out[i * 4] = src[i];
      out[i * 4 + 1] = src[i];
      out[i * 4 + 2] = src[i];
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

async function loadAny(kind: Kind, url: string, engine: ConvEngine): Promise<LoadedModel> {
  if (kind === 'seg') {
    const m = await loadSegModel(url, engine);
    const n = m.h * m.w;
    const sampleRgbas = m.samples
      .filter((s) => s.height === m.h && s.width === m.w)
      .map((s) => toRgba(segRgbFromB64(s.rgb_b64, n), n, 3));
    return { runtime: m.runtime, h: m.h, w: m.w, thermo: m.thermo, sampleRgbas };
  }
  const m = await loadDetModel(url, engine);
  const n = m.h * m.w;
  const sampleRgbas = m.samples
    .filter((s) => s.height === m.h && s.width === m.w)
    .map((s) => (s.rgb_b64
      ? toRgba(detRgbFromB64(s.rgb_b64, n), n, 3)
      : toRgba(grayFromB64(s.gray_b64!, n), n, 1)));
  return { runtime: m.runtime, h: m.h, w: m.w, thermo: m.thermo, sampleRgbas };
}

/** CPU前処理(convpreproc.ts)で符号化したフレーム入力。 */
function cpuInput(m: LoadedModel, rgba: Uint8ClampedArray): FrameInput {
  const nchw = rgbaToNchw(rgba, m.h, m.w);
  return { kind: 'bits', bits: encodeImageAuto(nchw, m.thermo) };
}

async function scoresOf(m: LoadedModel, rgba: Uint8ClampedArray): Promise<Int32Array> {
  const nchw = rgbaToNchw(rgba, m.h, m.w);
  const bits = encodeImageAuto(nchw, m.thermo);
  return (await m.runtime.scoresAsync(bits, 1)).data;
}

interface Case {
  name: string;
  file: string;
  kind: Kind;
}

const CASES: Case[] = [
  { name: 'seg (SegConvDLGN k128)', file: 'seg_convdlgn_k128.json', kind: 'seg' },
  { name: 'det (CenterNetLUT k256)', file: 'det_multimnist_lut_k256.json', kind: 'det' },
  { name: 'detmono (mono k256)', file: 'det_multimnist_mono_lut_k256.json', kind: 'det' },
  { name: 'detface (WIDER FACE k256)', file: 'det_face_lut_k256.json', kind: 'det' },
  { name: 'detreal (fewclass k64)', file: 'det_fewclass_lut_k64.json', kind: 'det' },
];

/** 5タスク全モデル: CPU bitplane(参照)vs bitplane-webgpu の生スコア一致。*/
async function verifyBitplaneWebgpu(c: Case): Promise<boolean> {
  const url = asset(`/models/${c.file}`);
  log(`\n== ${c.name} : bitplane(CPU) vs bitplane-webgpu ==`);
  const ref = await loadAny(c.kind, url, 'bitplane');
  const gpu = await loadAny(c.kind, url, 'bitplane-webgpu');
  const note = gpu.runtime.webgpuFallbackNote;
  if (note) {
    log(`  !! フォールバック検出(bitplane-webgpuで走っていない): ${note}`);
    return false;
  }
  log(`  dispatch数(単一compute pass内)= ${gpu.runtime.webgpuDispatchCount}`);
  let ok = true;
  for (let trial = 0; trial < 3; trial++) {
    const rgba = makeInput(ref.w, 20260803 + trial * 6151);
    const a = await scoresOf(ref, rgba);
    const b = await scoresOf(gpu, rgba);
    const eq = arraysEqual(a, b);
    log(`  trial ${trial}: ${eq ? 'MATCH' : 'MISMATCH'} (width=${a.length})`);
    if (!eq) log(`    最初の不一致: ${firstDiff(a, b)}`);
    ok = ok && eq;
  }
  return ok;
}

/** GPU前処理(RGBA→ビット符号化のWGSLカーネル)の検証。
    2段階で見る:
      (a) packed planes(=前処理+packの出力そのもの)の全ワード一致。
          最終スコアより手前で比較するので、食い違ったチャネル/行が直接わかる。
      (b) 最終スコアの整数完全一致(逐次・パイプラインの両モード)。
    入力は「決定的ランダムRGBA(端の0/255を含む)」に加えて doc.demo_samples の
    実画像も流す(実画像は自然画像の分布=平坦領域が多く、閾値ちょうど付近の
    ビットが出やすいのでランダムとは別の当たり方をする)。 */
async function verifyGpuPreproc(c: Case): Promise<boolean> {
  const url = asset(`/models/${c.file}`);
  log(`\n== ${c.name} : CPU前処理 vs GPU前処理(gpupre) ==`);
  const m = await loadAny(c.kind, url, 'bitplane-webgpu');
  if (m.runtime.webgpuFallbackNote) {
    log(`  !! フォールバック検出: ${m.runtime.webgpuFallbackNote}`);
    return false;
  }
  if (!m.runtime.supportsGpuPreproc) {
    log(`  !! GPU前処理が無効: ${m.runtime.gpuPreprocNote ?? '(理由不明)'}`);
    return false;
  }
  const cpuBits = (encodeImageAuto(rgbaToNchw(makeInput(m.w, 1), m.h, m.w), m.thermo)).length;
  log(`  preproc.kind=${m.thermo.kind} / GPU前処理の転送 ${m.h * m.w * 4}B`
    + ` vs CPU前処理の転送 ${cpuBits}B(${(cpuBits / (m.h * m.w * 4)).toFixed(0)}倍)`);

  const inputs: { tag: string; rgba: Uint8ClampedArray }[] = [];
  for (let i = 0; i < 3; i++) {
    inputs.push({ tag: `random#${i}`, rgba: makeInput(m.w, 20260804 + i * 7919) });
  }
  // 端(0/255)を必ず含む合成入力(閾値境界と replicate padding を踏ませる)
  const extreme = makeInput(m.w, 4242);
  for (let i = 0; i < m.h * m.w; i += 3) {
    extreme[i * 4] = i % 2 === 0 ? 0 : 255;
    extreme[i * 4 + 1] = 255;
    extreme[i * 4 + 2] = 0;
  }
  inputs.push({ tag: 'extreme', rgba: extreme });
  m.sampleRgbas.slice(0, 3).forEach((rgba, i) => {
    inputs.push({ tag: `demo_sample#${i}`, rgba });
  });
  log(`  入力 ${inputs.length} 本(うち実画像 ${Math.min(3, m.sampleRgbas.length)} 本)`);

  let ok = true;
  for (const { tag, rgba } of inputs) {
    // (a) packed planes の全ワード一致
    const cpuPlanes = await m.runtime.debugPackedPlanes(cpuInput(m, rgba), 1);
    const gpuPlanes = await m.runtime.debugPackedPlanes({ kind: 'rgba', rgba }, 1);
    let badWord = -1;
    if (cpuPlanes.length !== gpuPlanes.length) {
      badWord = 0;
    } else {
      for (let i = 0; i < cpuPlanes.length; i++) {
        if (cpuPlanes[i] !== gpuPlanes[i]) { badWord = i; break; }
      }
    }
    // (b) 最終スコアの一致(逐次)
    const cpuScore = (await m.runtime.scoresFrameAsync(cpuInput(m, rgba), 1)).data;
    const gpuScore = (await m.runtime.scoresFrameAsync({ kind: 'rgba', rgba }, 1)).data;
    const eqScore = arraysEqual(cpuScore, gpuScore);
    const eqPlanes = badWord < 0;
    log(`  ${tag}: planes ${eqPlanes ? 'MATCH' : 'MISMATCH'}`
      + `(${cpuPlanes.length}ワード) / scores ${eqScore ? 'MATCH' : 'MISMATCH'}`);
    if (!eqPlanes) {
      log(`    最初の不一致ワード: index=${badWord} cpu=0x${(cpuPlanes[badWord] ?? 0).toString(16)}`
        + ` gpu=0x${(gpuPlanes[badWord] ?? 0).toString(16)}`);
    }
    if (!eqScore) log(`    最初の不一致スコア: ${firstDiff(cpuScore, gpuScore)}`);
    ok = ok && eqPlanes && eqScore;
  }

  // (c) パイプラインモードのRGBA発行でも同じ結果になること
  const pipeInputs = inputs.map((v) => v.rgba);
  const pend: Promise<{ data: Int32Array }>[] = [];
  const pipeOut: Int32Array[] = [];
  for (const rgba of pipeInputs) {
    if (pend.length >= PIPELINE_DEPTH) pipeOut.push((await pend.shift()!).data);
    pend.push(m.runtime.submitScoresFrameAsync({ kind: 'rgba', rgba }, 1));
  }
  while (pend.length > 0) pipeOut.push((await pend.shift()!).data);
  await m.runtime.flushPipeline();
  let pipeBad = 0;
  for (let i = 0; i < pipeInputs.length; i++) {
    const ref = (await m.runtime.scoresFrameAsync(cpuInput(m, pipeInputs[i]), 1)).data;
    if (!arraysEqual(ref, pipeOut[i])) {
      if (pipeBad === 0) log(`  pipeline frame ${i}: MISMATCH ${firstDiff(ref, pipeOut[i])}`);
      pipeBad++;
    }
  }
  log(`  パイプライン(RGBA発行)一致: ${pipeInputs.length - pipeBad}/${pipeInputs.length}`);
  return ok && pipeBad === 0;
}

/** CPU融合前処理(RGBA→ビットプレーンを1パス)の検証。
    GPU版と同じ2段構え:
      (a) packed planes(前処理+packの出力そのもの)の全ワード一致
      (b) 最終スコアの整数完全一致
    参照は従来経路(convpreproc.ts::encodeImageAuto → packPlanes)。
    'bitplane'(JS融合)と 'bitplane-wasm'(WASM融合、SIMD/MT込み)の両方を見る
    ので、ブラウザでしか走らない SIMD 版・ワーカープール分割もここで潰れる。 */
async function verifyCpuPreproc(c: Case): Promise<boolean> {
  const url = asset(`/models/${c.file}`);
  log(`\n== ${c.name} : CPU融合前処理(cpupre) ==`);
  let ok = true;
  for (const engine of ['bitplane', 'bitplane-wasm'] as ConvEngine[]) {
    const m = await loadAny(c.kind, url, engine);
    const rt = m.runtime;
    if (!rt.supportsCpuPreproc) {
      log(`  !! [${engine}] CPU融合前処理が無効: ${rt.cpuPreprocNote ?? '(理由不明)'}`);
      ok = false;
      continue;
    }
    log(`  [${engine}] 経路=${rt.preprocMode} / スレッド=${rt.actualThreads}`
      + ` / 入力 RGBA ${m.h * m.w * 4}B(従来は中間の0/1バイト列 ${rt.inWidth}B)`);
    const inputs: { tag: string; rgba: Uint8ClampedArray }[] = [];
    for (let i = 0; i < 3; i++) {
      inputs.push({ tag: `random#${i}`, rgba: makeInput(m.w, 20260806 + i * 7919) });
    }
    const extreme = makeInput(m.w, 4242);
    for (let i = 0; i < m.h * m.w; i += 3) {
      extreme[i * 4] = i % 2 === 0 ? 0 : 255;
      extreme[i * 4 + 1] = 255;
      extreme[i * 4 + 2] = 0;
    }
    inputs.push({ tag: 'extreme', rgba: extreme });
    m.sampleRgbas.slice(0, 3).forEach((rgba, i) => {
      inputs.push({ tag: `demo_sample#${i}`, rgba });
    });
    for (const { tag, rgba } of inputs) {
      const bitsInput = cpuInput(m, rgba);
      const refPlanes = rt.debugCpuPackedPlanes(bitsInput, 1);
      const gotPlanes = rt.debugCpuPackedPlanes({ kind: 'rgba', rgba }, 1);
      let badWord = -1;
      if (refPlanes.length !== gotPlanes.length) badWord = 0;
      else {
        for (let i = 0; i < refPlanes.length; i++) {
          if (refPlanes[i] !== gotPlanes[i]) { badWord = i; break; }
        }
      }
      const refScore = (await rt.scoresFrameAsync(bitsInput, 1)).data;
      const gotScore = (await rt.scoresFrameAsync({ kind: 'rgba', rgba }, 1)).data;
      const eqScore = arraysEqual(refScore, gotScore);
      const eqPlanes = badWord < 0;
      log(`    ${tag}: planes ${eqPlanes ? 'MATCH' : 'MISMATCH'}`
        + `(${refPlanes.length}ワード) / scores ${eqScore ? 'MATCH' : 'MISMATCH'}`);
      if (!eqPlanes) {
        log(`      最初の不一致ワード: index=${badWord}`
          + ` ref=0x${(refPlanes[badWord] ?? 0).toString(16)}`
          + ` got=0x${(gotPlanes[badWord] ?? 0).toString(16)}`);
      }
      if (!eqScore) log(`      最初の不一致スコア: ${firstDiff(refScore, gotScore)}`);
      ok = ok && eqPlanes && eqScore;
    }
  }
  return ok;
}

/** パイプラインモード(2フレームin-flight)で N フレームを流し、
    結果を**順番どおり**に集める。呼び出し規約どおり「次を発行する前に最古だけを
    await」することで常にdepth本を飛ばす。前処理はバックプレッシャ解除後に
    行う(呼び出し側バッファの再利用に依存しない安全な順序)。 */
async function scoresPipelined(
  m: LoadedModel, rgbas: Uint8ClampedArray[],
): Promise<Int32Array[]> {
  const pend: Promise<{ data: Int32Array }>[] = [];
  const results: Int32Array[] = [];
  for (const rgba of rgbas) {
    if (pend.length >= PIPELINE_DEPTH) results.push((await pend.shift()!).data);
    const nchw = rgbaToNchw(rgba, m.h, m.w);
    const bits = encodeImageAuto(nchw, m.thermo);
    pend.push(m.runtime.submitScoresAsync(bits, 1));
  }
  while (pend.length > 0) results.push((await pend.shift()!).data);
  await m.runtime.flushPipeline();
  return results;
}

/** 逐次モード(evalAll)とパイプラインモード(submitFrame)の
    整数完全一致ストレス。**フレーム毎に異なる入力**を50本流し、全フレームの
    生スコアが1要素も違わないことを確認する。
    狙い: in-flight 2フレームでの入力バッファ/中間バッファ/readbackリングの
    競合(=フレームNの結果にN+1の入力が混じる、リングの席を早く空けすぎて
    上書きされる、等)を検出する。バッファを共有してよい根拠(GPUキューの
    発行順実行保証)が実機で成り立っていることの直接証拠になる。 */
const PIPELINE_STRESS_FRAMES = 50;

async function verifyPipelineStress(c: Case): Promise<boolean> {
  const url = asset(`/models/${c.file}`);
  log(`\n== ${c.name} : 逐次 vs パイプライン(${PIPELINE_STRESS_FRAMES}フレーム, `
    + `in-flight=${PIPELINE_DEPTH}, 毎フレーム別入力) ==`);
  const m = await loadAny(c.kind, url, 'bitplane-webgpu');
  if (m.runtime.webgpuFallbackNote) {
    log(`  !! フォールバック検出: ${m.runtime.webgpuFallbackNote}`);
    return false;
  }
  if (!m.runtime.supportsPipeline) {
    log('  !! supportsPipeline=false(bitplane-webgpuで走っていない)');
    return false;
  }
  // 交互に大きく違う入力を作る(隣接フレームが似ていると取り違えを見逃す)。
  const inputs: Uint8ClampedArray[] = [];
  for (let i = 0; i < PIPELINE_STRESS_FRAMES; i++) {
    inputs.push(makeInput(m.w, 777_000 + i * 104_729 + (i % 2) * 5_000_003));
  }
  const seq: Int32Array[] = [];
  for (const rgba of inputs) seq.push(await scoresOf(m, rgba));
  const pipe = await scoresPipelined(m, inputs);

  if (pipe.length !== seq.length) {
    log(`  !! フレーム数不一致 seq=${seq.length} pipe=${pipe.length}`);
    return false;
  }
  let bad = 0;
  for (let i = 0; i < seq.length; i++) {
    if (!arraysEqual(seq[i], pipe[i])) {
      if (bad === 0) log(`  frame ${i}: MISMATCH ${firstDiff(seq[i], pipe[i])}`);
      bad++;
    }
  }
  // 「たまたま全フレーム同じ出力」だと一致してもストレスにならないので、
  // 入力差が出力差になっていることも確認する(検証自体の有効性チェック)。
  const distinct = seq.some((v) => !arraysEqual(v, seq[0]));
  log(`  一致: ${seq.length - bad}/${seq.length} フレーム`
    + ` / 入力差が出力差になっている: ${distinct ? 'yes' : 'NO(検証が無効)'}`);
  return bad === 0 && distinct;
}

/** 従来経路(packed / packed-wasm / packed-webgpu)の回帰確認(seg+detのみ)。 */
async function verifyPackedTrio(c: Case): Promise<boolean> {
  const url = asset(`/models/${c.file}`);
  log(`\n== ${c.name} : packed vs packed-wasm vs packed-webgpu ==`);
  const js = await loadAny(c.kind, url, 'packed');
  const wasm = await loadAny(c.kind, url, 'packed-wasm');
  const webgpu = await loadAny(c.kind, url, 'packed-webgpu');
  let ok = true;
  for (let trial = 0; trial < 3; trial++) {
    const rgba = makeInput(js.w, 314159 + trial * 271);
    const a = await scoresOf(js, rgba);
    const b = await scoresOf(wasm, rgba);
    const cc = await scoresOf(webgpu, rgba);
    const eqWasm = arraysEqual(a, b);
    const eqGpu = arraysEqual(a, cc);
    log(`  trial ${trial}: js vs wasm ${eqWasm ? 'MATCH' : 'MISMATCH'} `
      + `(width=${a.length}), js vs webgpu ${eqGpu ? 'MATCH' : 'MISMATCH'}`);
    if (!eqGpu) log(`    最初の不一致(webgpu): ${firstDiff(a, cc)}`);
    ok = ok && eqWasm && eqGpu;
  }
  return ok;
}

async function main(): Promise<void> {
  $().textContent = '検証中…\n';
  const results: Record<string, boolean> = {};
  try {
    for (const c of CASES) {
      results[`bitplane-webgpu:${c.name}`] = await verifyBitplaneWebgpu(c);
    }
    for (const c of CASES) {
      results[`gpupre:${c.name}`] = await verifyGpuPreproc(c);
    }
    for (const c of CASES) {
      results[`cpupre:${c.name}`] = await verifyCpuPreproc(c);
    }
    for (const c of CASES) {
      results[`pipeline-stress:${c.name}`] = await verifyPipelineStress(c);
    }
    for (const c of CASES.slice(0, 2)) {
      results[`packed-trio:${c.name}`] = await verifyPackedTrio(c);
    }
    const allOk = Object.values(results).every(Boolean);
    log('');
    for (const [k, v] of Object.entries(results)) log(`${v ? 'PASS' : 'FAIL'}  ${k}`);
    log(`\n${allOk ? 'ALL PASS' : 'FAIL'}`);
    $().className = allOk ? 'ok' : 'err';
    const w = window as unknown as Record<string, unknown>;
    w.__VERIFY_RESULT__ = allOk;
    w.__VERIFY_DETAIL__ = results;
  } catch (e) {
    log(`\nERROR: ${String((e as Error).stack ?? e)}`);
    $().className = 'err';
    (window as unknown as Record<string, unknown>).__VERIFY_RESULT__ = false;
  }
}

void main();
