// WB-5/W4 dense等価性テスト:
//  1. dense golden fixture(Python simulate.pyのスコア)と4実装(interp/
//     packed32/packed64/codegen)のスコア完全一致。HAR実データ由来の旧goldenを
//     合成dense IR+合成時系列に汎用化(契約は同一: thermometer+int16量子化)
//  2. ランダム小IRのproperty test(packed/codegen == interp を全ビット照合)
// fixtureは scripts/export_webbench.py fixtures が生成する。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { IRGraphJson } from '../shared';
import type { ModelDoc, TestData, TestMeta } from '../types';
import { CodegenEngine } from './codegen';
import { InterpEngine } from './interp';
import { Packed32Engine, Packed64Engine } from './packed';

const here = dirname(fileURLToPath(import.meta.url));

interface DenseGolden {
  model: ModelDoc;
  n: number;
  shape: number[]; // [N, C, L]
  scale: number;
  x_i16: number[][];
  bits_b64: string[];
  scores: number[][];
  preds: number[];
}

const goldenPath = join(here, '../fixtures/dense_golden.json');
const hasGolden = existsSync(goldenPath);

function goldenTestData(golden: DenseGolden): TestData {
  const chLen = golden.x_i16[0].length;
  const c = golden.model.preproc.mean.length;
  const meta: TestMeta = {
    shape: [golden.n, c, chLen / c],
    dtype: 'int16le',
    scale: golden.scale,
    labels: golden.model.labels,
    y_test: [],
    mean: golden.model.preproc.mean,
    std: golden.model.preproc.std,
  };
  const x = new Float32Array(golden.n * chLen);
  for (let i = 0; i < golden.n; i++) {
    for (let j = 0; j < chLen; j++) {
      // Python側と同じくint16*scaleで復元(float32丸め)
      x[i * chLen + j] = Math.fround(golden.x_i16[i][j] * golden.scale);
    }
  }
  return { meta, n: golden.n, chLen, x };
}

describe.skipIf(!hasGolden)('dense golden照合(Python simulate基準)', () => {
  const golden = JSON.parse(readFileSync(goldenPath, 'utf8')) as DenseGolden;
  const model = golden.model;
  const data = goldenTestData(golden);

  const engines = [
    () => new InterpEngine(model),
    () => new Packed32Engine(model),
    () => new Packed64Engine(model),
    () => new CodegenEngine(model),
  ];

  for (const make of engines) {
    const engine = make();
    it(`${engine.name}: スコア完全一致(${golden.n}サンプル)`, () => {
      const scores = engine.scores!(data, 0, golden.n);
      for (let i = 0; i < golden.n; i++) {
        expect(Array.from(scores[i]), `sample ${i}`).toEqual(golden.scores[i]);
      }
    });
  }

  it('preds一致', async () => {
    const engine = new Packed32Engine(model);
    const preds = await engine.predict(data, 0, golden.n);
    expect(Array.from(preds)).toEqual(golden.preds);
  });
});

// ============================================================
// property test: ランダム小IR(k≤64, l≤3)で packed/codegen == interp
// ============================================================

/** 疑似乱数(シード固定、mulberry32) */
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

function randomDenseIr(
  rand: () => number,
  inWidth: number,
  widths: number[],
  groupK: number,
): IRGraphJson {
  const nodes = [];
  const edges = [{ src: 'x', dst: 'n1' }];
  let prevW = inWidth;
  for (let li = 0; li < widths.length; li++) {
    const k = widths[li];
    const indices = [
      Array.from({ length: k }, () => Math.floor(rand() * prevW)),
      Array.from({ length: k }, () => Math.floor(rand() * prevW)),
    ];
    const tts = Array.from(
      { length: k },
      () => `0x${Math.floor(rand() * 16).toString(16)}`,
    );
    nodes.push({
      id: `n${li + 1}`,
      type: 'lut_layer' as const,
      params: {
        in_width: prevW, out_width: k, n_inputs: 2,
        indices, truth_tables: tts,
      },
      train_info: {},
      meta: {},
    });
    edges.push({ src: `n${li + 1}`, dst: `n${li + 2}` });
    prevW = k;
  }
  nodes.push({
    id: `n${widths.length + 1}`,
    type: 'group_sum' as const,
    params: { in_width: prevW, k: groupK, tau: 1 },
    train_info: {},
    meta: {},
  });
  edges.push({ src: `n${widths.length + 1}`, dst: 'y' });
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: 'random-test' },
    inputs: [{ id: 'x', width: inWidth }],
    outputs: [{ id: 'y', width: groupK }],
    nodes,
    edges,
  };
}

/** bit i = x[i] > 0.5 になる恒等的なpreproc(C=inWidth, L=1, T=1) */
function identityModel(ir: IRGraphJson): ModelDoc {
  const w = ir.inputs[0].width;
  return {
    ir,
    preproc: {
      kind: 'thermometer',
      input_kind: 'raw',
      mean: new Array(w).fill(0),
      std: new Array(w).fill(1),
      thresholds: Array.from({ length: w }, () => [0.5]),
      bit_order: 'ch,t,thr',
    },
    labels: [],
    resources: { total_luts: 0, logic_depth: 0 },
    eval: { hard_acc: 0, hard_acc_i16: 0, n_test: 0, confusion: [] },
  };
}

function randomBitsData(
  rand: () => number, n: number, w: number,
): TestData {
  const x = new Float32Array(n * w);
  for (let i = 0; i < x.length; i++) x[i] = rand() > 0.5 ? 1 : 0;
  return {
    meta: {
      shape: [n, w, 1], dtype: 'int16le', scale: 1,
      labels: [], y_test: [], mean: [], std: [],
    },
    n,
    chLen: w,
    x,
  };
}

describe('property: ランダムIRで4実装のスコア一致', () => {
  // 70サンプル(32/64ブロック境界をまたぐ)、幅はgroupKで割り切れる値
  const cases = [
    { seed: 1, inWidth: 24, widths: [48, 36], groupK: 3 },
    { seed: 2, inWidth: 16, widths: [64, 64, 32], groupK: 2 },
    { seed: 3, inWidth: 40, widths: [60], groupK: 6 },
  ];

  for (const c of cases) {
    it(`seed=${c.seed} widths=${c.widths.join('x')} k=${c.groupK}`, () => {
      const rand = rng(c.seed);
      const ir = randomDenseIr(rand, c.inWidth, c.widths, c.groupK);
      const model = identityModel(ir);
      const data = randomBitsData(rand, 70, c.inWidth);

      const ref = new InterpEngine(model).scores(data, 0, data.n);
      for (const engine of [
        new Packed32Engine(model),
        new Packed64Engine(model),
        new CodegenEngine(model),
      ]) {
        const got = engine.scores!(data, 0, data.n);
        for (let i = 0; i < data.n; i++) {
          expect(Array.from(got[i]), `${engine.name} sample ${i}`).toEqual(
            Array.from(ref[i]),
          );
        }
      }
    });
  }
});
