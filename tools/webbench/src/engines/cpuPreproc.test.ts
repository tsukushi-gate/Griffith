// CPU融合前処理(RGBA → ビットプレーンを1パス)のビット完全一致検証。
//
// 契約: 融合実装(JS / scalar wasm / simd wasm)の出力ワード列が、従来経路
//   convpreproc.ts::encodeImageAuto → bitplane.ts::packPlanes
// と **1ワードも違わない**こと。encodeImageAuto 自体は Python との一致が
// thermo_golden.json / lowlevel_golden.json で既に固定されているので、
// ここが緑なら「配信されるビット」は従来と完全に同一である。
//
// 検証の広さ(1ビットでも違う設計は不採用、という要求に対応):
//   - 配信中の実モデル5本の preproc 設定(image_thermometer / image_lowlevel)
//   - ゴールデン由来パラメータ(mean/std≠0,1・負の閾値・6x7 等の非2冪サイズ
//     = 端数ワードと replicate padding を全周で踏む)
//   - 入力は 決定的ランダム / 端値飽和(0・255を規則的に混ぜる) / doc.demo_samples の実画像
//   - THERMO_STEP(単調な thermometer を k>=k0 へ畳んだ経路)と
//     LUT参照経路の**両方**(配信モデルは全て単調なので、LUT経路は記述子を
//     意図的に畳み戻して踏ませる)
//   - スカラーwasm ⇔ SIMD wasm ⇔ JS の三者一致
//   - チャネルシャード分割呼び出し(=マルチスレッド実行と同じ出力分割)の不変性
//   - ConvRuntime 経由の最終スコアが RGBA直投入と従来経路で完全一致
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { encodeImageAuto, rgbaToNchw } from '../convpreproc';
import type { ImagePreprocDoc } from '../convpreproc';
import { packPlanes } from './bitplane';
import { ConvRuntime } from './conv';
import { JsPreprocPacker, buildCpuPreprocPlan } from './cpuPreproc';
import type { CpuPreprocPlan } from './cpuPreproc';
import { PREPROC_DESC_STRIDE, PREPROC_OP_THERMO } from './gpuPreproc';
import { shardRange } from './wasmThreadPool';
import { WasmPreprocOps } from './wasmPreproc';
import { loadTestWasmModule, loadTestWasmSimdModule } from './wasmTestUtil';
import type { WasmModule } from './wasmConv';

const here = dirname(fileURLToPath(import.meta.url));
const modelsDir = join(here, '../../public/models');
const fixturesDir = join(here, '../fixtures');

const scalarMod: WasmModule | null = await loadTestWasmModule();
const simdMod: WasmModule | null = await loadTestWasmSimdModule();
const WASM_BUILDS: { name: string; mod: WasmModule }[] = [];
if (scalarMod && WasmPreprocOps.supported(scalarMod)) {
  WASM_BUILDS.push({ name: 'scalar', mod: scalarMod });
}
if (simdMod && WasmPreprocOps.supported(simdMod)) {
  WASM_BUILDS.push({ name: 'simd', mod: simdMod });
}

// ---------------------------------------------------------------------------
// 入力生成
// ---------------------------------------------------------------------------

/** 決定的な擬似ランダムRGBA(gpuPreproc.test.ts と同じ生成器)。
    端(0/255)を必ず含める: 閾値境界と replicate padding の両方を踏ませるため。 */
function makeRgba(h: number, w: number, seed: number): Uint8ClampedArray {
  let s = seed;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const rgba = new Uint8ClampedArray(h * w * 4);
  for (let i = 0; i < h * w; i++) {
    rgba[i * 4] = (rnd() * 256) | 0;
    rgba[i * 4 + 1] = (rnd() * 256) | 0;
    rgba[i * 4 + 2] = (rnd() * 256) | 0;
    rgba[i * 4 + 3] = 255;
  }
  for (let i = 0; i < h * w; i += 7) {
    rgba[i * 4] = 0;
    rgba[i * 4 + 1] = 255;
    rgba[i * 4 + 2] = i % 2 === 0 ? 255 : 0;
  }
  return rgba;
}

/** 端値だけで構成した入力(飽和・全黒・最大コントラスト隣接)。
    edge/curv の D が ±2^31, ±2^32 に達する = 固定小数の設計前提の境界を踏む。 */
function makeExtremeRgba(h: number, w: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(h * w * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = (x + y) % 2 === 0 ? 0 : 255; // 市松 = 隣接画素が常に最大差
      rgba[i * 4] = v;
      rgba[i * 4 + 1] = x % 3 === 0 ? 255 : 0;
      rgba[i * 4 + 2] = y % 3 === 0 ? 0 : 255;
      rgba[i * 4 + 3] = 255;
    }
  }
  return rgba;
}

function b64ToBytes(b64: string): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** HWC uint8(3ch or 1ch)→ RGBA(A=255)。 */
function toRgba(src: Uint8Array, n: number, channels: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = channels === 3 ? src[i * 3] : src[i];
    out[i * 4 + 1] = channels === 3 ? src[i * 3 + 1] : src[i];
    out[i * 4 + 2] = channels === 3 ? src[i * 3 + 2] : src[i];
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ---------------------------------------------------------------------------
// 参照実装(従来経路)
// ---------------------------------------------------------------------------

/** encodeImageAuto → packPlanes(=までの配信経路)のワード列。*/
function refWords(
  p: ImagePreprocDoc, rgbas: Uint8ClampedArray[],
): Int32Array {
  const hw = p.height * p.width;
  const parts = rgbas.map((rgba) => encodeImageAuto(rgbaToNchw(rgba, p.height, p.width), p));
  const C = parts[0].length / hw;
  const merged = new Uint8Array(parts.length * C * hw);
  parts.forEach((bits, i) => merged.set(bits as Uint8Array, i * C * hw));
  return packPlanes(
    { data: merged, width: C * hw }, rgbas.length, C, p.height, p.width,
  ).words;
}

function concatRgba(rgbas: Uint8ClampedArray[]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgbas.reduce((a, r) => a + r.length, 0));
  let o = 0;
  for (const r of rgbas) {
    out.set(r, o);
    o += r.length;
  }
  return out;
}

function firstDiff(a: ArrayLike<number>, b: ArrayLike<number>, plan: CpuPreprocPlan): string {
  if (a.length !== b.length) return `長さ違い ${a.length} vs ${b.length}`;
  const perCh = plan.height * plan.wpr;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      const ch = Math.floor((i % (plan.channels * perCh)) / perCh);
      const op = plan.desc[ch * PREPROC_DESC_STRIDE];
      return `word=${i}(ch=${ch}, op=${op}, y=${Math.floor((i % perCh) / plan.wpr)},`
        + ` j=${i % plan.wpr}) ref=0x${(a[i] >>> 0).toString(16)}`
        + ` got=0x${(b[i] >>> 0).toString(16)}`;
    }
  }
  return '(差分なし)';
}

function expectSameWords(
  ref: ArrayLike<number>, got: ArrayLike<number>, plan: CpuPreprocPlan, tag: string,
): void {
  let same = ref.length === got.length;
  if (same) {
    for (let i = 0; i < ref.length; i++) {
      if (ref[i] !== got[i]) {
        same = false;
        break;
      }
    }
  }
  expect(same, `${tag}: ${firstDiff(ref, got, plan)}`).toBe(true);
}

// ---------------------------------------------------------------------------
// 検証対象(配信モデル + ゴールデン由来パラメータ)
// ---------------------------------------------------------------------------

interface Case {
  name: string;
  p: ImagePreprocDoc;
  /** doc.demo_samples 由来の実画像(あれば)。 */
  samples: Uint8ClampedArray[];
  inputWidth: number | null;
  /** ConvRuntime 端到端テストに使うモデルJSONのパス(ゴールデン由来ケースはnull)。 */
  modelPath: string | null;
}

interface ModelDocLite {
  preproc: ImagePreprocDoc;
  ir: { inputs: { width: number }[] };
  demo_samples?: { rgb_b64?: string; gray_b64?: string; height: number; width: number }[];
}

function loadModelCases(): Case[] {
  const out: Case[] = [];
  if (!existsSync(modelsDir)) return out;
  for (const f of readdirSync(modelsDir)) {
    if (!f.endsWith('.json') || f === 'index.json') continue;
    let doc: ModelDocLite;
    try {
      doc = JSON.parse(readFileSync(join(modelsDir, f), 'utf8')) as ModelDocLite;
    } catch {
      continue;
    }
    const p = doc.preproc;
    if (!p || (p.kind !== 'image_thermometer' && p.kind !== 'image_lowlevel')) continue;
    const n = p.height * p.width;
    const samples = (doc.demo_samples ?? [])
      .filter((s) => s.height === p.height && s.width === p.width)
      .slice(0, 2)
      .map((s) => (s.rgb_b64
        ? toRgba(b64ToBytes(s.rgb_b64), n, 3)
        : toRgba(b64ToBytes(s.gray_b64!), n, 1)));
    out.push({
      name: f,
      p,
      samples,
      inputWidth: doc.ir?.inputs?.[0]?.width ?? null,
      modelPath: join(modelsDir, f),
    });
  }
  return out;
}

interface ThermoGolden {
  shape: [number, number, number, number];
  mean: number[];
  std: number[];
  thresholds: number[][];
}
interface LowLevelGolden extends ThermoGolden {
  edge_thresholds: number[];
  curv_thresholds: number[];
  chroma_thresholds: number[];
  edge_offsets: number[][];
  curv_offsets: number[][];
  chroma_pairs: number[][];
}

function loadGoldenCases(): Case[] {
  const out: Case[] = [];
  const tp = join(fixturesDir, 'thermo_golden.json');
  if (existsSync(tp)) {
    const g = JSON.parse(readFileSync(tp, 'utf8')) as ThermoGolden;
    const [, , h, w] = g.shape;
    out.push({
      name: `thermo_golden (${h}x${w}, mean/std≠0,1・負閾値)`,
      p: {
        kind: 'image_thermometer',
        mean: g.mean,
        std: g.std,
        thresholds: g.thresholds,
        height: h,
        width: w,
      },
      samples: [],
      inputWidth: null,
      modelPath: null,
    });
  }
  const lp = join(fixturesDir, 'lowlevel_golden.json');
  if (existsSync(lp)) {
    const g = JSON.parse(readFileSync(lp, 'utf8')) as LowLevelGolden;
    const [, , h, w] = g.shape;
    out.push({
      name: `lowlevel_golden (${h}x${w}, 非2冪サイズ)`,
      p: {
        kind: 'image_lowlevel',
        mean: g.mean,
        std: g.std,
        thresholds: g.thresholds,
        edge_thresholds: g.edge_thresholds,
        curv_thresholds: g.curv_thresholds,
        chroma_thresholds: g.chroma_thresholds,
        edge_offsets: g.edge_offsets,
        curv_offsets: g.curv_offsets,
        chroma_pairs: g.chroma_pairs,
        height: h,
        width: w,
      },
      samples: [],
      inputWidth: null,
      modelPath: null,
    });
  }
  return out;
}

/** 配信モデルにもゴールデンにも無い「境界の効く」設定を人工的に足す。
    - 常に真/常に偽になる thermometer 閾値(k0=0 / k0=256 の分岐)
    - 幅が4の倍数でない(=JSの4バイト同時比較が使えない経路)
    - |t| が 1 前後の edge/curv 閾値(cut が i32 の安全域を超え、
      SIMD経路がチャネル単位でスカラー(i64)へ落ちる分岐)
    - |dx|>1 のオフセット(replicate padding 幅の検証) */
function syntheticCases(): Case[] {
  return [
    {
      name: 'synthetic thermo (33x35, mean/std非自明, 常に真/偽の閾値)',
      p: {
        kind: 'image_thermometer',
        mean: [0.2, 0.5, -0.3],
        std: [0.7, 1.3, 0.4],
        thresholds: [
          [-5, -0.1, 0.35, 5],
          [-5, 0.02, 0.4, 5],
          [-5, 1.1, 2.0, 5],
        ],
        height: 33,
        width: 35,
      },
      samples: [],
      inputWidth: null,
      modelPath: null,
    },
    {
      name: 'synthetic lowlevel (13x37, |t|>=1 の閾値と |dx|>1 のオフセット)',
      p: {
        kind: 'image_lowlevel',
        mean: [0, 0.1, 0],
        std: [1, 0.9, 1.2],
        thresholds: [[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]],
        edge_thresholds: [0.0, 0.99, 1.0, 1.5],
        curv_thresholds: [0.5, 1.99, 2.5],
        chroma_thresholds: [0.001, 0.999],
        edge_offsets: [[0, 1], [2, 0], [1, -3], [0, 3]],
        curv_offsets: [[0, 1], [3, 2]],
        chroma_pairs: [[0, 1], [2, 0]],
        height: 13,
        width: 37,
      },
      samples: [],
      inputWidth: null,
      modelPath: null,
    },
  ];
}

const MODEL_CASES = loadModelCases();
const CASES = [...MODEL_CASES, ...loadGoldenCases(), ...syntheticCases()];

/** 1ケースぶんの入力集合(ランダム3本 + 端値飽和 + 実画像)。 */
function inputsOf(c: Case): { tag: string; rgba: Uint8ClampedArray }[] {
  const { height: h, width: w } = c.p;
  const list = [
    { tag: 'random#0', rgba: makeRgba(h, w, 20260804) },
    { tag: 'random#1', rgba: makeRgba(h, w, 7919) },
    { tag: 'random#2', rgba: makeRgba(h, w, 3) },
    { tag: 'extreme', rgba: makeExtremeRgba(h, w) },
  ];
  c.samples.forEach((rgba, i) => list.push({ tag: `demo_sample#${i}`, rgba }));
  return list;
}

// ---------------------------------------------------------------------------
// 1. JS融合実装 == 従来経路
// ---------------------------------------------------------------------------

describe('cpuPreproc(JS融合): encodeImageAuto + packPlanes と全ワード一致', () => {
  it('配信中の画像モデルを全て拾えている(検証が空回りしていない)', () => {
    expect(MODEL_CASES.length).toBeGreaterThanOrEqual(5);
    expect(MODEL_CASES.some((c) => c.p.kind === 'image_lowlevel')).toBe(true);
    expect(MODEL_CASES.some((c) => c.p.kind === 'image_thermometer')).toBe(true);
    expect(MODEL_CASES.some((c) => c.samples.length > 0)).toBe(true);
  });

  for (const c of CASES) {
    it(`${c.name}`, () => {
      const plan = buildCpuPreprocPlan(c.p);
      if (c.inputWidth !== null) {
        expect(plan.channels * plan.height * plan.width).toBe(c.inputWidth);
      }
      const packer = new JsPreprocPacker(plan);
      for (const { tag, rgba } of inputsOf(c)) {
        expectSameWords(refWords(c.p, [rgba]), packer.pack(rgba, 1).words, plan, tag);
      }
    });
  }

  it('batch>1 でもサンプル境界がずれない', () => {
    for (const c of CASES) {
      const plan = buildCpuPreprocPlan(c.p);
      const packer = new JsPreprocPacker(plan);
      const two = [makeRgba(c.p.height, c.p.width, 55), makeExtremeRgba(c.p.height, c.p.width)];
      expectSameWords(
        refWords(c.p, two), packer.pack(concatRgba(two), 2).words, plan, `${c.name} batch=2`,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// 2. thermometer の階段化(THERMO_STEP)とLUT経路
// ---------------------------------------------------------------------------

/** 記述子を「LUT参照の thermometer」へ畳み戻した計画(LUT経路を踏ませる用)。
    LUTワード(w8..w15)は buildPreprocDescriptors がCPUと同一式で埋めたまま
    なので、op を戻すだけで意味論は変わらない(=同じビットが出るはず)。 */
function forceLutPlan(plan: CpuPreprocPlan): CpuPreprocPlan {
  const desc = Int32Array.from(plan.desc);
  for (let ch = 0; ch < plan.channels; ch++) {
    const base = ch * PREPROC_DESC_STRIDE;
    if (desc[base] === 5) desc[base] = PREPROC_OP_THERMO;
  }
  return { ...plan, desc, stepChannels: 0, lutChannels: plan.stepChannels };
}

describe('cpuPreproc: thermometer の階段化と LUT 経路', () => {
  it('配信モデル/ゴールデンとも std>0 なので全 thermo チャネルが階段化される', () => {
    for (const c of CASES) {
      const plan = buildCpuPreprocPlan(c.p);
      expect(plan.stepChannels, c.name).toBeGreaterThan(0);
      expect(plan.lutChannels, c.name).toBe(0);
    }
  });

  it('LUT参照へ畳み戻しても同じビットが出る(LUT経路の検証)', () => {
    for (const c of CASES) {
      const plan = forceLutPlan(buildCpuPreprocPlan(c.p));
      const packer = new JsPreprocPacker(plan);
      const rgba = makeRgba(c.p.height, c.p.width, 999);
      expectSameWords(refWords(c.p, [rgba]), packer.pack(rgba, 1).words, plan, `${c.name} LUT`);
    }
  });
});

// ---------------------------------------------------------------------------
// 3. WASM(scalar/simd)== JS == 従来経路
// ---------------------------------------------------------------------------

describe.skipIf(WASM_BUILDS.length === 0)('cpuPreproc(WASM): scalar/simd とも全ワード一致', () => {
  it('scalar と simd の両ビルドがカーネルを持っている', () => {
    expect(WASM_BUILDS.map((b) => b.name)).toEqual(['scalar', 'simd']);
  });

  for (const { name, mod } of WASM_BUILDS) {
    for (const c of CASES) {
      it(`[${name}] ${c.name}`, () => {
        const plan = buildCpuPreprocPlan(c.p);
        const ops = new WasmPreprocOps(mod, plan);
        const js = new JsPreprocPacker(plan);
        for (const { tag, rgba } of inputsOf(c)) {
          const ref = refWords(c.p, [rgba]);
          const got = ops.pack(rgba, 1).words;
          expectSameWords(ref, got, plan, `${name}/${tag}`);
          expectSameWords(js.pack(rgba, 1).words, got, plan, `${name}/${tag} vs JS`);
        }
        // LUT経路(記述子を畳み戻したもの)も同じビットになること
        const lutPlan = forceLutPlan(plan);
        const lutOps = new WasmPreprocOps(mod, lutPlan);
        const rgba = makeRgba(c.p.height, c.p.width, 424242);
        expectSameWords(
          refWords(c.p, [rgba]), lutOps.pack(rgba, 1).words, plan, `${name}/LUT`,
        );
      });
    }

    it(`[${name}] batch>1`, () => {
      for (const c of CASES) {
        const plan = buildCpuPreprocPlan(c.p);
        const ops = new WasmPreprocOps(mod, plan);
        const two = [makeRgba(c.p.height, c.p.width, 55), makeExtremeRgba(c.p.height, c.p.width)];
        expectSameWords(
          refWords(c.p, two), ops.pack(concatRgba(two), 2).words, plan, `${name}/${c.name} b=2`,
        );
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 4. チャネルシャード分割(=マルチスレッド実行と同じ出力分割)の不変性
// ---------------------------------------------------------------------------

/** WASMヒープを使い捨てで確保して preprocPackAsm をシャード実行する
    (wasmBitplaneOps.test.ts と同じ流儀: Worker を使わずに
     「出力領域の排他分割」だけを再現する)。 */
function packViaShards(
  mod: WasmModule, plan: CpuPreprocPlan, rgba: Uint8ClampedArray, nShards: number,
): Int32Array {
  const {
    channels, imageChannels, height: H, width: W, wpr, tail, rowStride, rowPad,
  } = plan;
  const outWords = channels * H * wpr;
  const rgbaPtr = mod.alloc(H * W * 4 + 64);
  const qPtr = mod.alloc(imageChannels * H * rowStride * 4 + 64);
  const bytePtr = mod.alloc(imageChannels * H * W + 64);
  const descPtr = mod.alloc(plan.desc.length * 4);
  const qvalPtr = mod.alloc(256 * 4);
  const outPtr = mod.alloc(outWords * 4);
  const buf = mod.memory.buffer;
  new Int32Array(buf, descPtr, plan.desc.length).set(plan.desc);
  new Int32Array(buf, qvalPtr, 256).set(plan.qval);
  new Uint8Array(buf, rgbaPtr, H * W * 4).set(rgba);
  new Int32Array(buf, outPtr, outWords).fill(0x5a5a5a5a); // 未書き込みを検出する毒
  mod.preprocQuantAsm!(
    rgbaPtr, qvalPtr, qPtr, bytePtr, H, W, rowStride, rowPad, H * W, imageChannels,
  );
  for (let i = 0; i < nShards; i++) {
    const { first, count } = shardRange(channels, nShards, i);
    if (count > 0) {
      mod.preprocPackAsm!(
        qPtr, bytePtr, descPtr, outPtr, first, count,
        H, W, wpr, tail, rowStride, rowPad, H * W, imageChannels,
      );
    }
  }
  return Int32Array.from(new Int32Array(mod.memory.buffer, outPtr, outWords));
}

describe.skipIf(WASM_BUILDS.length === 0)('cpuPreproc(WASM): チャネルシャード分割で不変', () => {
  for (const { name, mod } of WASM_BUILDS) {
    it(`[${name}] 1/2/3/4/8分割いずれも従来経路と全ワード一致`, () => {
      for (const c of CASES) {
        const plan = buildCpuPreprocPlan(c.p);
        const rgba = makeRgba(c.p.height, c.p.width, 20260806);
        const ref = refWords(c.p, [rgba]);
        for (const n of [1, 2, 3, 4, 8]) {
          expectSameWords(ref, packViaShards(mod, plan, rgba, n), plan, `${name}/${c.name} n=${n}`);
        }
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 5. ConvRuntime 端到端: RGBA直投入 == 従来経路(最終スコアの整数一致)
// ---------------------------------------------------------------------------

interface ModelDocFull extends ModelDocLite {
  ir: { inputs: { width: number; id: string }[] };
}

describe('ConvRuntime: RGBA直投入(融合前処理)と従来経路の最終スコア一致', () => {
  for (const c of MODEL_CASES) {
    it(`${c.name}`, async () => {
      const doc = JSON.parse(readFileSync(c.modelPath!, 'utf8')) as ModelDocFull;
      const rgba = inputsOf(c)[3].rgba; // 端値飽和入力
      const bits = encodeImageAuto(rgbaToNchw(rgba, c.p.height, c.p.width), c.p);
      // 融合前処理あり(既定)/ なし(?cpupre=0 相当)で同じスコアになること
      const fused = new ConvRuntime(doc.ir as never, { engine: 'bitplane', preproc: c.p });
      expect(fused.supportsCpuPreproc, fused.cpuPreprocNote ?? '').toBe(true);
      expect(fused.preprocMode).toBe('cpu-js');
      const plain = new ConvRuntime(doc.ir as never, { engine: 'bitplane', cpuPreproc: false });
      expect(plain.supportsCpuPreproc).toBe(false);
      const ref = plain.scores(bits, 1).data;
      const got = fused.scoresFrame({ kind: 'rgba', rgba }, 1).data;
      expect(Array.from(got)).toEqual(Array.from(ref));
      // packed planes(スコアより手前)でも全ワード一致していること
      const planesRef = fused.debugCpuPackedPlanes({ kind: 'bits', bits }, 1);
      const planesGot = fused.debugCpuPackedPlanes({ kind: 'rgba', rgba }, 1);
      expect(Array.from(planesGot)).toEqual(Array.from(planesRef));

      if (scalarMod) {
        const w = await ConvRuntime.create(doc.ir as never, {
          engine: 'bitplane-wasm', preproc: c.p, wasmModule: scalarMod,
        });
        expect(w.preprocMode, w.cpuPreprocNote ?? '').toBe('cpu-wasm');
        expect(Array.from(w.scoresFrame({ kind: 'rgba', rgba }, 1).data)).toEqual(
          Array.from(ref),
        );
      }
    });
  }
});
