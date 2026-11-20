// 手描きデモ純関数のユニットテスト。2値化・確率正規化・縮小・IR判定。

import { describe, expect, it } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import { evalCircuit, type LayerProbs } from '../playground/evalCircuit';
import {
  argmax,
  binarize,
  browserEvalCost,
  downsampleGray,
  irInputWidth,
  isMnistShapedIR,
  MNIST_INPUT_WIDTH,
  normalizeProbs,
  outputSourceNodeId,
} from './drawUtils';

describe('binarize', () => {
  it('しきい値超のセルのみ1になる', () => {
    expect(binarize([0, 0.4, 0.6, 1], 0.5)).toEqual([0, 0, 1, 1]);
  });
  it('しきい値ちょうどは0(> 判定)', () => {
    expect(binarize([0.5, 0.5], 0.5)).toEqual([0, 0]);
  });
  it('しきい値変更で結果が変わる', () => {
    const gray = [0.3, 0.7];
    expect(binarize(gray, 0.2)).toEqual([1, 1]);
    expect(binarize(gray, 0.5)).toEqual([0, 1]);
    expect(binarize(gray, 0.9)).toEqual([0, 0]);
  });
});

describe('normalizeProbs', () => {
  it('合計が1になる', () => {
    const p = normalizeProbs([1, 2, 3]);
    const sum = p.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });
  it('全要素[0,1]かつ単調(大きいスコアほど高確率)', () => {
    const p = normalizeProbs([0, 1, 2]);
    expect(p[0]).toBeLessThan(p[1]);
    expect(p[1]).toBeLessThan(p[2]);
    for (const v of p) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
  it('全て等しいスコアなら一様分布', () => {
    const p = normalizeProbs([5, 5, 5, 5]);
    for (const v of p) expect(v).toBeCloseTo(0.25, 10);
  });
  it('空配列は空を返す', () => {
    expect(normalizeProbs([])).toEqual([]);
  });
  it('非有限値が混ざっても合計1・NaNを出さない', () => {
    const p = normalizeProbs([Infinity, NaN, 1]);
    const sum = p.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
    for (const v of p) expect(Number.isFinite(v)).toBe(true);
  });
  it('大きなスコアでもオーバーフローしない(数値安定)', () => {
    const p = normalizeProbs([1000, 1001]);
    const sum = p.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
    expect(p[1]).toBeGreaterThan(p[0]);
  });
});

describe('downsampleGray', () => {
  it('一様値はそのまま維持される', () => {
    const src = new Array<number>(4 * 4).fill(0.5);
    const out = downsampleGray(src, 4, 2);
    expect(out).toHaveLength(4);
    for (const v of out) expect(v).toBeCloseTo(0.5, 10);
  });
  it('2×2ブロック平均が正しい', () => {
    // 4×4 の左上2×2=1、それ以外0 → 2×2縮小で [1,0,0,0]
    const src = [
      1, 1, 0, 0,
      1, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ];
    expect(downsampleGray(src, 4, 2)).toEqual([1, 0, 0, 0]);
  });
  it('長さ不一致はthrow', () => {
    expect(() => downsampleGray([0, 1, 2], 4, 2)).toThrow();
  });
});

describe('argmax', () => {
  it('最大インデックスを返す', () => {
    expect(argmax([0.1, 0.7, 0.2])).toBe(1);
  });
  it('空は-1', () => {
    expect(argmax([])).toBe(-1);
  });
});

function mnistIr(lutOutWidth: number): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.1.0',
    metadata: { name: 't' },
    inputs: [{ id: 'x', width: MNIST_INPUT_WIDTH, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 10, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'n1',
        type: 'lut_layer',
        params: { in_width: MNIST_INPUT_WIDTH, out_width: lutOutWidth, n_inputs: 2, indices: [], truth_tables: [] },
        train_info: {},
        meta: {},
      },
      { id: 'n2', type: 'group_sum', params: { in_width: lutOutWidth, k: 10, tau: 1 }, train_info: {}, meta: {} },
    ],
    edges: [
      { src: 'x', dst: 'n1' },
      { src: 'n1', dst: 'n2' },
      { src: 'n2', dst: 'y' },
    ],
  } as unknown as IRGraphJson;
}

describe('IRヘルパ', () => {
  it('irInputWidth はポート幅の総和', () => {
    expect(irInputWidth(mnistIr(60))).toBe(MNIST_INPUT_WIDTH);
  });
  it('isMnistShapedIR は幅784単一入力のみtrue', () => {
    expect(isMnistShapedIR(mnistIr(60))).toBe(true);
    expect(isMnistShapedIR(null)).toBe(false);
    const narrow = mnistIr(60);
    narrow.inputs[0].width = 2;
    expect(isMnistShapedIR(narrow)).toBe(false);
  });
  it('outputSourceNodeId は出力ポートのソースノード', () => {
    expect(outputSourceNodeId(mnistIr(60))).toBe('n2');
  });
  it('browserEvalCost はlut_layer出力幅の総和', () => {
    expect(browserEvalCost(mnistIr(60))).toBe(60);
    expect(browserEvalCost(mnistIr(5000))).toBe(5000);
  });
});

// ライブ評価経路(evalCircuit + gateDist確率 → normalizeProbs)が
// 学習スナップショット(ゲート確率)の変化で出力クラス確率を変える。
describe('ライブ推論経路(gateDist→確率)', () => {
  // 2入力→2ゲート→group_sum(k=2) の最小分類IR
  function tinyClassifierIr(): IRGraphJson {
    return {
      format: 'magicdelta-ir',
      version: '0.1.0',
      metadata: { name: 'tiny' },
      inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
      outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
      nodes: [
        {
          id: 'lut',
          type: 'lut_layer',
          params: {
            in_width: 2,
            out_width: 2,
            n_inputs: 2,
            indices: [[0, 0], [1, 1]],
            truth_tables: ['0x0', '0x0'],
          },
          train_info: {},
          meta: {},
        },
        { id: 'gs', type: 'group_sum', params: { in_width: 2, k: 2, tau: 1 }, train_info: {}, meta: {} },
      ],
      edges: [
        { src: 'x', dst: 'lut' },
        { src: 'lut', dst: 'gs' },
        { src: 'gs', dst: 'y' },
      ],
    } as unknown as IRGraphJson;
  }

  function evalProbs(ir: IRGraphJson, probs: Record<string, LayerProbs>): number[] {
    const out = evalCircuit(ir, { x: [1, 1] }, 'soft', probs).output;
    return normalizeProbs(out);
  }

  it('ゲート確率スナップショットが変わると出力クラス確率が変わる', () => {
    const ir = tinyClassifierIr();
    // 真理値表ビット確率(bins=2^n=4): out_width×4。学習ステップ違いを模す2枚。
    const early: Record<string, LayerProbs> = {
      lut: { data: new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), bins: 4 },
    };
    const late: Record<string, LayerProbs> = {
      lut: { data: new Float32Array([0, 0, 0, 1, 1, 0, 0, 0]), bins: 4 },
    };
    const pEarly = evalProbs(ir, early);
    const pLate = evalProbs(ir, late);
    expect(pEarly.length).toBe(2);
    expect(pLate.length).toBe(2);
    // スナップショットが進むと分布が変化する(=学習が進むと結果が変わる)
    expect(Math.abs(pLate[0] - pEarly[0])).toBeGreaterThan(1e-6);
  });
});
