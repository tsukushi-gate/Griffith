// WASM v128 SIMD版(build/release-simd.wasm、assembly/index-simd.ts)が
// scalar版(build/release.wasm、assembly/index.ts)とビット完全一致することを
// 検証する。WasmModuleのABIは両者で同一(alloc/evalConv2dBatch/
// evalConv2dBitplaneBatch)なので、'bitplane-wasm'/'packed-wasm'エンジンへ
// 注入するwasmModuleを差し替えるだけでSIMD経路をテストできる
// (bitplaneGraph.ts/conv.tsは無改造)。
//
// 正しさの錨:
//   1. conv_golden.json(Python simulateの真実源)とscalar/simd両方の
//      'bitplane-wasm'/'packed-wasm'が整数完全一致。
//   2. 実モデル(det_multimnist_mono_lut_k256/k640、public/models)で
//      scalar 'bitplane-wasm' と simd 'bitplane-wasm' が全出力ビット完全一致
//      (v128木縮約チャンク化が実運用サイズのgOut/tapRowsWords境界でも
//      ズレないことの確認)。
// build/release-simd.wasmが無ければ(pretest未実行環境)describe.skipIfで
// 自動スキップする。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { IRGraphJson } from '../shared';
import type { ConvModelDoc } from '../types';
import { ConvRuntime } from './conv';
import type { ConvEngine } from './conv';
import { loadTestWasmModule, loadTestWasmSimdModule } from './wasmTestUtil';
import type { WasmModule } from './wasmConv';

const scalarModule: WasmModule | null = await loadTestWasmModule();
const simdModule: WasmModule | null = await loadTestWasmSimdModule();
const ready = !!scalarModule && !!simdModule;

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
}

const convGoldenPath = join(here, '../fixtures/conv_golden.json');
const hasConvGolden = existsSync(convGoldenPath);

describe.skipIf(!ready || !hasConvGolden)(
  'bitplane-wasm/packed-wasm: SIMD(v128)版==scalar版==Python golden(整数完全一致)',
  () => {
    const golden = JSON.parse(readFileSync(convGoldenPath, 'utf8')) as { cases: ConvGoldenCase[] };
    const engines: ConvEngine[] = ['packed-wasm', 'bitplane-wasm'];

    for (const c of golden.cases) {
      for (const engine of engines) {
        it(`[${engine}] ${c.name} (${c.n}サンプル)`, async () => {
          const scalarRt = await ConvRuntime.create(c.ir, { engine, wasmModule: scalarModule! });
          const simdRt = await ConvRuntime.create(c.ir, { engine, wasmModule: simdModule! });
          const bits = new Int32Array(c.n * c.in_width);
          for (let i = 0; i < c.n; i++) {
            bits.set(b64ToBits(c.x_bits_b64[i], c.in_width), i * c.in_width);
          }
          const scalarOut = scalarRt.scores(bits, c.n);
          const simdOut = simdRt.scores(bits, c.n);
          expect(simdOut.width).toBe(scalarOut.width);
          expect(Array.from(simdOut.data)).toEqual(Array.from(scalarOut.data));
          // scalar側は既存conv.test.tsでPython goldenと突き合わせ済みだが、
          // ここでも直接照合して「両方goldenと一致」を明示する。
          for (let i = 0; i < c.n; i++) {
            const got = Array.from(simdOut.data.subarray(i * simdOut.width, (i + 1) * simdOut.width));
            expect(got, `${c.name} sample ${i}`).toEqual(c.scores[i]);
          }
        });
      }
    }
  },
);

// ============================================================
// 実モデル(検出DLGN、k256/k640): scalar 'bitplane-wasm' と simd
// 'bitplane-wasm' の全出力ビットが完全一致すること。gOut/tapRowsWordsが
// 4の倍数境界を跨ぐ実運用サイズでv128チャンク化+端数スカラーが
// ズレないことの確認(木の各段でnLeaves/gOutが2べきで変わるため、
// 端数境界を自然に何通りも踏む)。
// ============================================================
const modelsDir = join(here, '../../public/models');
const modelFiles = ['det_multimnist_mono_lut_k256.json', 'det_multimnist_mono_lut_k640.json'];

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

// ============================================================
// property: ランダムconv IRで、tapRowsWords(=HOut*tapWpr、木縮約の
// v128チャンク対象)が4ワード境界の前後(1,2,3,4,5,7,8,9ワード相当)を
// 踏むようにWOutを掃引し、simd==scalarを確認する。
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
    metadata: { name: 'random-conv-simd-prop' },
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

describe.skipIf(!ready)('property: bitplane-wasm simd==scalar(ワード境界の前後を掃引、depth掃引)', () => {
  const cases = [
    // W=32,64,96,128相当(wpr=1,2,3,4ちょうど)+端数(33,65,97)でtapRowsWordsが
    // 4ワード境界を跨ぐケースを広く踏む。depthも2..4で木の各段のgOutを掃引。
    { C: 1, H: 3, W: 31, OC: 2, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 2 },
    { C: 2, H: 3, W: 32, OC: 2, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 3 },
    { C: 1, H: 3, W: 33, OC: 3, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 2 },
    { C: 2, H: 3, W: 64, OC: 2, kH: 1, kW: 1, sH: 1, sW: 1, pH: 0, pW: 0, depth: 4 },
    { C: 1, H: 3, W: 65, OC: 2, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 4 },
    { C: 3, H: 5, W: 96, OC: 2, kH: 3, kW: 3, sH: 1, sW: 1, pH: 1, pW: 1, depth: 3 },
    { C: 1, H: 5, W: 97, OC: 2, kH: 3, kW: 3, sH: 2, sW: 2, pH: 1, pW: 1, depth: 2 }, // 低速路
  ];
  for (const c of cases) {
    it(`H=${c.H} W=${c.W} OC=${c.OC} depth=${c.depth}`, async () => {
      const rand = rng(c.H * 1000 + c.W * 31 + c.OC * 7 + c.depth);
      const ir = randomConvIr(c.C, c.H, c.W, c.OC, c.kH, c.kW, c.sH, c.sW, c.pH, c.pW, c.depth, rand);
      const scalarRt = await ConvRuntime.create(ir, { engine: 'bitplane-wasm', wasmModule: scalarModule! });
      const simdRt = await ConvRuntime.create(ir, { engine: 'bitplane-wasm', wasmModule: simdModule! });
      for (let trial = 0; trial < 3; trial++) {
        const bits = new Int32Array(c.C * c.H * c.W);
        for (let i = 0; i < bits.length; i++) bits[i] = rand() > 0.5 ? 1 : 0;
        const a = scalarRt.scores(bits, 1);
        const b = simdRt.scores(bits, 1);
        expect(b.width, `trial ${trial}`).toBe(a.width);
        expect(Array.from(b.data), `trial ${trial}`).toEqual(Array.from(a.data));
      }
    });
  }
});

for (const modelFile of modelFiles) {
  const modelPath = join(modelsDir, modelFile);
  const modelReady = ready && existsSync(modelPath);
  describe.skipIf(!modelReady)(`実モデル ${modelFile}: bitplane-wasm simd==scalar(乱数入力)`, () => {
    const doc = JSON.parse(readFileSync(modelPath, 'utf8')) as ConvModelDoc;
    const inWidth = doc.ir.inputs[0].width;

    it('乱数入力3件で全出力ビット完全一致', async () => {
      const scalarRt = await ConvRuntime.create(doc.ir, { engine: 'bitplane-wasm', wasmModule: scalarModule! });
      const simdRt = await ConvRuntime.create(doc.ir, { engine: 'bitplane-wasm', wasmModule: simdModule! });
      const rand = rng(inWidth + modelFile.length);
      for (let trial = 0; trial < 3; trial++) {
        const bits = new Int32Array(inWidth);
        for (let i = 0; i < inWidth; i++) bits[i] = rand() > 0.5 ? 1 : 0;
        const scalarOut = scalarRt.scores(bits, 1);
        const simdOut = simdRt.scores(bits, 1);
        expect(simdOut.width, `trial ${trial}`).toBe(scalarOut.width);
        expect(Array.from(simdOut.data), `trial ${trial}`).toEqual(Array.from(scalarOut.data));
      }
    });
  });
}
