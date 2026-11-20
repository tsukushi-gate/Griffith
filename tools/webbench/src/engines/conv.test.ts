// W4 conv整数一致テスト:
//  1. conv_golden.json(Python ir/simulate.py が真実源)と ConvRuntime の
//     スコア完全一致(conv2d/or_pool2d/upsample/SpatialGroupSum/concat被覆)
//  2. thermo_golden.json: 画像thermometer前処理のビット順契約(Py↔TS)
//  3. property: lowerConv(TS展開)と手元素朴実装の窓計算一致(サイズ掃引)
// fixtureは scripts/export_webbench.py fixtures が生成する。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { Conv2dParams, IRGraphJson } from '../shared';
import { encodeImageThermometer } from '../convpreproc';
import type { ImageThermoParams } from '../convpreproc';
import { buildConvProgramLayout } from './bitplane';
import type { BundleProgramDict } from './bundleProgram';
import { ConvRuntime } from './conv';
import type { ConvEngine } from './conv';
import { lowerToPrimitives } from './lowerConv';
import { loadTestWasmModule } from './wasmTestUtil';
import type { WasmModule } from './wasmConv';

// packed-wasm: AssemblyScript製WASMカーネルもpacked/irflatと同じ
// goldenで整数完全一致を取る(build/release.wasmが無ければ自動スキップ)。
const wasmModule: WasmModule | null = await loadTestWasmModule();
const ENGINES: ConvEngine[] = wasmModule
  ? ['packed', 'irflat', 'packed-wasm', 'bitplane', 'bitplane-wasm']
  : ['packed', 'irflat', 'bitplane'];

/** engineに応じてConvRuntimeを構築する(packed-wasmは非同期初期化が要る)。 */
async function makeRuntime(ir: IRGraphJson, engine: ConvEngine): Promise<ConvRuntime> {
  if (engine === 'packed-wasm' || engine === 'bitplane-wasm') {
    return ConvRuntime.create(ir, { engine, wasmModule: wasmModule! });
  }
  return new ConvRuntime(ir, { engine });
}

const here = dirname(fileURLToPath(import.meta.url));

function b64ToBits(b64: string, nBits: number): Uint8Array {
  const raw = atob(b64);
  const bits = new Uint8Array(nBits);
  for (let i = 0; i < nBits; i++) {
    bits[i] = (raw.charCodeAt(i >> 3) >> (i & 7)) & 1;
  }
  return bits;
}

interface ConvGoldenCase {
  name: string;
  ir: IRGraphJson;
  in_width: number;
  out_width: number;
  n: number;
  x_bits_b64: string[];
  scores: number[][];
  preds: number[];
}

const convGoldenPath = join(here, '../fixtures/conv_golden.json');
const hasConvGolden = existsSync(convGoldenPath);

describe.skipIf(!hasConvGolden)('conv golden: Python simulateとの整数完全一致', () => {
  const golden = JSON.parse(readFileSync(convGoldenPath, 'utf8')) as {
    cases: ConvGoldenCase[];
  };

  for (const c of golden.cases) {
    for (const engine of ENGINES) {
      it(`[${engine}] ${c.name} (${c.n}サンプル, in=${c.in_width}, out=${c.out_width})`, async () => {
        const rt = await makeRuntime(c.ir, engine);
        // バッチ入力を組む
        const bits = new Int32Array(c.n * c.in_width);
        for (let i = 0; i < c.n; i++) {
          bits.set(b64ToBits(c.x_bits_b64[i], c.in_width), i * c.in_width);
        }
        const { data, width } = rt.scores(bits, c.n);
        expect(width).toBe(c.out_width);
        for (let i = 0; i < c.n; i++) {
          const got = Array.from(data.subarray(i * width, (i + 1) * width));
          expect(got, `${c.name} sample ${i}`).toEqual(c.scores[i]);
        }
      });

      it(`[${engine}] ${c.name}: バッチ=1逐次でも同一(バッチ独立性)`, async () => {
        const rt = await makeRuntime(c.ir, engine);
        for (const i of [0, c.n - 1]) {
          const bits = b64ToBits(c.x_bits_b64[i], c.in_width);
          const { data } = rt.scores(bits, 1);
          expect(Array.from(data)).toEqual(c.scores[i]);
        }
      });
    }
  }
});

interface ThermoGolden {
  shape: number[]; // [N,C,H,W]
  t: number;
  scale: number;
  mean: number[];
  std: number[];
  thresholds: number[][];
  x_i16: number[][];
  out_width: number;
  bits_b64: string[];
}

const thermoGoldenPath = join(here, '../fixtures/thermo_golden.json');
const hasThermoGolden = existsSync(thermoGoldenPath);

describe.skipIf(!hasThermoGolden)('画像thermometer前処理: ビット順契約(Py↔TS)', () => {
  const g = JSON.parse(readFileSync(thermoGoldenPath, 'utf8')) as ThermoGolden;
  const [n, c, h, w] = g.shape;
  const params: ImageThermoParams = {
    mean: g.mean,
    std: g.std,
    thresholds: g.thresholds,
    height: h,
    width: w,
  };

  it(`全${n}サンプルでビット列完全一致(bit=((c*T+thr)*H+y)*W+x)`, () => {
    for (let i = 0; i < n; i++) {
      const row = new Float32Array(c * h * w);
      for (let j = 0; j < row.length; j++) {
        row[j] = Math.fround(g.x_i16[i][j] * g.scale);
      }
      const bits = encodeImageThermometer(row, params);
      expect(bits.length).toBe(g.out_width);
      const expected = b64ToBits(g.bits_b64[i], g.out_width);
      expect(Array.from(bits), `sample ${i}`).toEqual(Array.from(expected));
    }
  });
});

// ============================================================
// property: lowerConvの窓/パディング計算をサイズ掃引で検証
// (or_pool2dの各出力 = 対応窓の入力OR、を素朴計算と照合)
// ============================================================

function orPoolIr(
  C: number, H: number, W: number, k: number, s: number, pad: number,
): IRGraphJson {
  const HOut = Math.floor((H + 2 * pad - k) / s) + 1;
  const WOut = Math.floor((W + 2 * pad - k) / s) + 1;
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: 'orpool-prop' },
    inputs: [{ id: 'x', width: C * H * W }],
    outputs: [{ id: 'y', width: C * HOut * WOut }],
    nodes: [{
      id: 'p1',
      type: 'or_pool2d',
      params: { in_shape: [C, H, W], kernel_size: k, stride: s, padding: pad },
      train_info: {},
      meta: {},
    }],
    edges: [
      { src: 'x', dst: 'p1' },
      { src: 'p1', dst: 'y' },
    ],
  };
}

function naiveOrPool(
  x: Uint8Array, C: number, H: number, W: number,
  k: number, s: number, pad: number,
): Int32Array {
  const HOut = Math.floor((H + 2 * pad - k) / s) + 1;
  const WOut = Math.floor((W + 2 * pad - k) / s) + 1;
  const out = new Int32Array(C * HOut * WOut);
  for (let c = 0; c < C; c++) {
    for (let oy = 0; oy < HOut; oy++) {
      for (let ox = 0; ox < WOut; ox++) {
        let v = 0;
        for (let dy = 0; dy < k; dy++) {
          for (let dx = 0; dx < k; dx++) {
            const y = oy * s - pad + dy;
            const x2 = ox * s - pad + dx;
            if (y >= 0 && y < H && x2 >= 0 && x2 < W) {
              v |= x[c * H * W + y * W + x2];
            }
          }
        }
        out[c * HOut * WOut + oy * WOut + ox] = v;
      }
    }
  }
  return out;
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('property: or_pool2d展開の窓/パディング一致(サイズ掃引)', () => {
  const cases = [
    { C: 2, H: 6, W: 6, k: 2, s: 2, pad: 0 },
    { C: 1, H: 7, W: 5, k: 2, s: 2, pad: 0 }, // 端数(床割り)
    { C: 3, H: 6, W: 6, k: 2, s: 1, pad: 0 }, // stride<kernel(重なり)
    { C: 2, H: 5, W: 5, k: 2, s: 2, pad: 1 }, // パディング
    { C: 1, H: 8, W: 8, k: 2, s: 2, pad: 1 },
  ];
  for (const c of cases) {
    for (const engine of ENGINES) {
      it(`[${engine}] C=${c.C} ${c.H}x${c.W} k=${c.k} s=${c.s} pad=${c.pad}`, async () => {
        const ir = orPoolIr(c.C, c.H, c.W, c.k, c.s, c.pad);
        const rt = await makeRuntime(ir, engine);
        const rand = rng(c.H * 100 + c.W * 10 + c.k);
        const x = new Uint8Array(c.C * c.H * c.W);
        for (let trial = 0; trial < 4; trial++) {
          for (let i = 0; i < x.length; i++) x[i] = rand() > 0.5 ? 1 : 0;
          const { data } = rt.scores(x, 1);
          const ref = naiveOrPool(x, c.C, c.H, c.W, c.k, c.s, c.pad);
          expect(Array.from(data), `trial ${trial}`).toEqual(Array.from(ref));
        }
      });
    }
  }
});

describe('lowerToPrimitives', () => {
  it('conv/pool無しIRはそのまま返す(参照同一)', () => {
    const ir: IRGraphJson = {
      format: 'magicdelta-ir',
      version: '0.2.0',
      metadata: {},
      inputs: [{ id: 'x', width: 4 }],
      outputs: [{ id: 'y', width: 2 }],
      nodes: [{
        id: 'g', type: 'group_sum',
        params: { in_width: 4, k: 2, tau: 1 }, train_info: {}, meta: {},
      }],
      edges: [{ src: 'x', dst: 'g' }, { src: 'g', dst: 'y' }],
    };
    expect(lowerToPrimitives(ir)).toBe(ir);
  });
});

// ============================================================
// property: bitplaneエンジン(evalConv2dBitplane)がpackedエンジン
// (evalConv2dPacked、既存の正しさの錨)とランダムconv IRで整数完全一致すること。
// 奇数W・stride2・pad0..2を掃引し、高速路(stride=1&WOut===W)と低速路
// (buildTapPlaneStrided)の両方を確実に踏む。
// ============================================================

function randomConvIr(
  C: number, H: number, W: number, OC: number,
  kH: number, kW: number, sH: number, sW: number, pH: number, pW: number,
  depth: number, rand: () => number,
): IRGraphJson {
  const nLeaves = 2 ** depth;
  const nInternal = nLeaves - 1;
  const indicesFlat: number[][] = [];
  const truthTables: string[] = [];
  for (let oc = 0; oc < OC; oc++) {
    const row: number[] = [];
    for (let k = 0; k < nLeaves; k++) row.push(Math.floor(rand() * C * kH * kW));
    indicesFlat.push(row);
    for (let g = 0; g < nInternal; g++) truthTables.push(`0x${Math.floor(rand() * 16).toString(16)}`);
  }
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: 'random-conv-prop' },
    inputs: [{ id: 'x', width: C * H * W }],
    outputs: [{ id: 'y', width: OC * HOut * WOut }],
    nodes: [{
      id: 'c1',
      type: 'conv2d',
      params: {
        in_shape: [C, H, W],
        out_channels: OC,
        kernel_size: [kH, kW],
        stride: [sH, sW],
        padding: [pH, pW],
        tree_depth: depth,
        n_inputs: 2,
        indices_flat: indicesFlat,
        truth_tables: truthTables,
      },
      train_info: {},
      meta: {},
    }],
    edges: [
      { src: 'x', dst: 'c1' },
      { src: 'c1', dst: 'y' },
    ],
  };
}

describe('property: bitplaneエンジン==packedエンジン(ランダムconv IR、奇数W/stride2/pad掃引)', () => {
  const cases = [
    // 高速路(stride=1・same padding風でWOut===W): 3x3 pad=1
    { C: 2, H: 5, W: 5, OC: 3, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 3 },
    { C: 1, H: 7, W: 33, OC: 2, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 2 }, // wpr境界
    { C: 3, H: 4, W: 9, OC: 2, kH: 1, kW: 1, sH: 1, sW: 1, pH: 0, pW: 0, depth: 1 }, // 1x1
    // 低速路(stride>1、WOut!==Wの端数掃引・pad0..2)
    { C: 2, H: 7, W: 7, OC: 2, kH: 3, kW: 3, sH: 2, sW: 2, pH: 0, pW: 0, depth: 2 },
    { C: 1, H: 9, W: 5, OC: 3, kH: 3, kW: 3, sH: 2, sW: 1, pH: 1, pW: 0, depth: 1 },
    { C: 2, H: 6, W: 33, OC: 2, kH: 3, kW: 3, sH: 2, sW: 2, pH: 2, pW: 2, depth: 2 },
    { C: 1, H: 5, W: 5, OC: 2, kH: 5, kW: 5, sH: 1, sW: 1, pH: 0, pW: 0, depth: 3 }, // WOut=1!==W(高速路対象外)
  ];
  for (const c of cases) {
    it(`C=${c.C} H=${c.H} W=${c.W} OC=${c.OC} k=(${c.kH},${c.kW}) s=(${c.sH},${c.sW}) pad=(${c.pH},${c.pW}) depth=${c.depth}`, async () => {
      const rand = rng(c.H * 1000 + c.W * 31 + c.OC * 7 + c.depth);
      const ir = randomConvIr(c.C, c.H, c.W, c.OC, c.kH, c.kW, c.sH, c.sW, c.pH, c.pW, c.depth, rand);
      const packed = new ConvRuntime(ir, { engine: 'packed' });
      const bitplane = new ConvRuntime(ir, { engine: 'bitplane' });
      for (let trial = 0; trial < 4; trial++) {
        const bits = new Int32Array(c.C * c.H * c.W);
        for (let i = 0; i < bits.length; i++) bits[i] = rand() > 0.5 ? 1 : 0;
        const a = packed.scores(bits, 1);
        const b = bitplane.scores(bits, 1);
        expect(b.width, `trial ${trial}`).toBe(a.width);
        expect(Array.from(b.data), `trial ${trial}`).toEqual(Array.from(a.data));
      }
    });
  }
});

// conv2d 命令列化(convProgram.ts)は「整数出力完全不変」が契約なので、
// 同じ IR を convProgram:true / false の両方で実行して完全一致を要求する
// (true が既定。false は従来の全ゲート評価経路)。JS('bitplane')と
// WASM('bitplane-wasm', scalar release.wasm)の両方で見る。ランダムIRは
// 高速路(stride=1・WOut===W)/strided路の両方と、パススルー/定数ttが偏った
// 「学習済みモデルに近い」分布を含める。
describe('conv2d命令列版 == 旧バランス木版(整数完全不変)', () => {
  const cases = [
    { C: 2, H: 5, W: 5, OC: 5, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 3 },
    { C: 1, H: 7, W: 33, OC: 4, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 3 },
    { C: 3, H: 4, W: 9, OC: 3, kH: 1, kW: 1, sH: 1, sW: 1, pH: 0, pW: 0, depth: 1 },
    { C: 2, H: 7, W: 7, OC: 4, kH: 3, kW: 3, sH: 2, sW: 2, pH: 0, pW: 0, depth: 2 },
    { C: 2, H: 6, W: 33, OC: 3, kH: 3, kW: 3, sH: 2, sW: 2, pH: 2, pW: 2, depth: 3 },
    { C: 1, H: 5, W: 5, OC: 2, kH: 5, kW: 5, sH: 1, sW: 1, pH: 0, pW: 0, depth: 3 },
  ];
  for (const c of cases) {
    it(`C=${c.C} ${c.H}x${c.W} OC=${c.OC} k=(${c.kH},${c.kW}) s=(${c.sH},${c.sW}) pad=(${c.pH},${c.pW}) depth=${c.depth}`, async () => {
      const rand = rng(c.H * 977 + c.W * 41 + c.OC * 13 + c.depth);
      const ir = randomConvIr(
        c.C, c.H, c.W, c.OC, c.kH, c.kW, c.sH, c.sW, c.pH, c.pW, c.depth, rand,
      );
      const legacy = new ConvRuntime(ir, { engine: 'bitplane', convProgram: false });
      const prog = new ConvRuntime(ir, { engine: 'bitplane', convProgram: true });
      expect(prog.convProgramFallbackNote).toBeNull();
      const wasmLegacy = wasmModule
        ? await ConvRuntime.create(ir, { engine: 'bitplane-wasm', wasmModule, convProgram: false })
        : null;
      const wasmProg = wasmModule
        ? await ConvRuntime.create(ir, { engine: 'bitplane-wasm', wasmModule, convProgram: true })
        : null;
      expect(wasmProg?.convProgramFallbackNote ?? null).toBeNull();
      for (let trial = 0; trial < 4; trial++) {
        const bits = new Int32Array(c.C * c.H * c.W);
        for (let i = 0; i < bits.length; i++) bits[i] = rand() > 0.5 ? 1 : 0;
        const want = legacy.scores(bits, 1);
        expect(Array.from(prog.scores(bits, 1).data), `js trial ${trial}`)
          .toEqual(Array.from(want.data));
        if (wasmLegacy && wasmProg) {
          expect(Array.from(wasmLegacy.scores(bits, 1).data), `wasm-legacy trial ${trial}`)
            .toEqual(Array.from(want.data));
          expect(Array.from(wasmProg.scores(bits, 1).data), `wasm-prog trial ${trial}`)
            .toEqual(Array.from(want.data));
        }
      }
    });
  }

  it('パススルー/定数に偏ったtt分布(学習済みモデル近似)でも一致し、実際に縮約される', async () => {
    const rand = rng(4242);
    // 残差初期化後の分布に寄せる: 0xa(素通し)と 0x0/0xf(定数)を多めに引く
    const skew = (): number => {
      const r = rand();
      return r < 0.5 ? 0xa : r < 0.6 ? 0x0 : r < 0.7 ? 0xf : Math.floor(rand() * 16);
    };
    const C = 3; const H = 8; const W = 8; const OC = 32; const depth = 3;
    const ir = randomConvIr(C, H, W, OC, 3, 3, 1, 1, 1, 1, depth, rand);
    const params = ir.nodes[0].params as unknown as { truth_tables: string[] };
    for (let i = 0; i < params.truth_tables.length; i++) {
      params.truth_tables[i] = `0x${skew().toString(16)}`;
    }
    const legacy = new ConvRuntime(ir, { engine: 'bitplane', convProgram: false });
    const prog = new ConvRuntime(ir, { engine: 'bitplane', convProgram: true });
    const layout = buildConvProgramLayout(ir.nodes[0].params as unknown as Conv2dParams);
    expect(layout.program.stats.gatesBefore).toBe(OC * (2 ** depth - 1));
    expect(layout.program.stats.gatesAfter).toBeLessThan(layout.program.stats.gatesBefore / 2);
    for (let trial = 0; trial < 6; trial++) {
      const bits = new Int32Array(C * H * W);
      for (let i = 0; i < bits.length; i++) bits[i] = rand() > 0.5 ? 1 : 0;
      expect(Array.from(prog.scores(bits, 1).data), `trial ${trial}`)
        .toEqual(Array.from(legacy.scores(bits, 1).data));
    }
  });
});

// ============================================================
// ConvRuntimeOptions.programs: 配布バンドルの命令列を注入した経路が
// ブラウザ側で再構築した経路と整数完全一致すること。src/fixtures/
// bundle_golden.json の bundle_doc(実conv2dノード2個・or_pool2d/upsample2d
// /permute/group_sum混在の実モデル形IR)を使う(無ければ自動スキップ)。
// ============================================================

interface BundleGoldenDoc {
  ir: IRGraphJson;
  programs: Record<string, BundleProgramDict>;
}

const bundleGoldenPath = join(here, '../fixtures/bundle_golden.json');
const hasBundleGolden = existsSync(bundleGoldenPath);

describe.skipIf(!hasBundleGolden)('ConvRuntimeOptions.programs: バンドル命令列の注入', () => {
  const golden = JSON.parse(readFileSync(bundleGoldenPath, 'utf8')) as {
    bundle_doc: BundleGoldenDoc;
  };
  const { ir, programs } = golden.bundle_doc;
  const inWidth = ir.inputs[0].width;
  const nodeIds = Object.keys(programs);
  const bundleEngines: ConvEngine[] = wasmModule ? ['bitplane', 'bitplane-wasm'] : ['bitplane'];

  function randomBits(seed: number, n: number): Int32Array {
    const rand = rng(seed);
    const bits = new Int32Array(n);
    for (let i = 0; i < n; i++) bits[i] = rand() > 0.5 ? 1 : 0;
    return bits;
  }

  async function makeRuntimeWithPrograms(
    engine: ConvEngine,
    programsArg: Record<string, BundleProgramDict> | null,
    opts: { convProgram?: boolean } = {},
  ): Promise<ConvRuntime> {
    if (engine === 'bitplane-wasm') {
      return ConvRuntime.create(ir, { engine, wasmModule: wasmModule!, programs: programsArg, ...opts });
    }
    return new ConvRuntime(ir, { engine, programs: programsArg, ...opts });
  }

  for (const engine of bundleEngines) {
    it(`[${engine}] programs注入時、注入無しと整数完全一致・provenanceは全て'bundle'`, async () => {
      const base = await makeRuntime(ir, engine);
      const withProg = await makeRuntimeWithPrograms(engine, programs);
      expect(withProg.convProgramFallbackNote).toBeNull();
      expect(withProg.bundleProgramNote).toBeNull();
      expect(withProg.programProvenance).not.toBeNull();
      for (const id of nodeIds) expect(withProg.programProvenance?.[id]).toBe('bundle');
      for (const trial of [1, 2, 3]) {
        const bits = randomBits(1000 + trial, inWidth);
        const want = base.scores(bits, 1);
        const got = withProg.scores(bits, 1);
        expect(Array.from(got.data), `trial ${trial}`).toEqual(Array.from(want.data));
      }
    });
  }

  it('破損したprogram dictはノード単位で再構築へフォールバックする(黙殺しない、スコアは不変)', async () => {
    const base = new ConvRuntime(ir, { engine: 'bitplane' });
    const corrupted: Record<string, BundleProgramDict> = JSON.parse(JSON.stringify(programs));
    const targetId = nodeIds[0];
    corrupted[targetId].instrs[0] = 99; // tt を範囲外にする
    const rt = new ConvRuntime(ir, { engine: 'bitplane', programs: corrupted });
    expect(rt.convProgramFallbackNote).toBeNull(); // ノード単位のフォールバックなので全体は落ちない
    expect(rt.bundleProgramNote).not.toBeNull();
    expect(rt.programProvenance?.[targetId]).toBe('built');
    for (const id of nodeIds.slice(1)) expect(rt.programProvenance?.[id]).toBe('bundle');
    const bits = randomBits(555, inWidth);
    expect(Array.from(rt.scores(bits, 1).data)).toEqual(Array.from(base.scores(bits, 1).data));
  });

  it('convProgram:false はprograms注入より優先される(旧経路・programProvenanceはnull)', () => {
    const base = new ConvRuntime(ir, { engine: 'bitplane', convProgram: false });
    const rt = new ConvRuntime(ir, { engine: 'bitplane', convProgram: false, programs });
    expect(rt.programProvenance).toBeNull();
    const bits = randomBits(333, inWidth);
    expect(Array.from(rt.scores(bits, 1).data)).toEqual(Array.from(base.scores(bits, 1).data));
  });
});
