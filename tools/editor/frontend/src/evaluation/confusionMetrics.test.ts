// 混同行列メトリクスのユニットテスト。既知の混同から検算する。

import { describe, expect, it } from 'vitest';
import { buildConfusion, computeConfusionMetrics } from './confusionMetrics';

describe('computeConfusionMetrics', () => {
  it('対角のみなら全メトリクス=1', () => {
    // confusion[label][pred]
    const m = [
      [5, 0],
      [0, 3],
    ];
    const r = computeConfusionMetrics(m);
    expect(r.accuracy).toBe(1);
    expect(r.precision).toEqual([1, 1]);
    expect(r.recall).toEqual([1, 1]);
    expect(r.f1).toEqual([1, 1]);
    expect(r.support).toEqual([5, 3]);
  });

  it('既知の2クラス混同でprecision/recallが正しい', () => {
    // 実際0を10件中8件正解・2件を1と誤判定。実際1を10件中6件正解・4件を0と誤判定。
    const m = [
      [8, 2],
      [4, 6],
    ];
    const r = computeConfusionMetrics(m);
    // pred=0 列和=12, tp=8 → precision0 = 8/12
    expect(r.precision[0]).toBeCloseTo(8 / 12, 6);
    // 実際0 行和=10, tp=8 → recall0 = 0.8
    expect(r.recall[0]).toBeCloseTo(0.8, 6);
    // pred=1 列和=8, tp=6 → precision1 = 6/8
    expect(r.precision[1]).toBeCloseTo(6 / 8, 6);
    expect(r.recall[1]).toBeCloseTo(0.6, 6);
    expect(r.accuracy).toBeCloseTo(14 / 20, 6);
  });

  it('予測/実際0件のクラスは0で安全', () => {
    const m = [
      [0, 0],
      [0, 4],
    ];
    const r = computeConfusionMetrics(m);
    expect(r.precision[0]).toBe(0);
    expect(r.recall[0]).toBe(0);
    expect(r.f1[0]).toBe(0);
    expect(r.accuracy).toBe(1);
  });
});

describe('buildConfusion', () => {
  it('評価行からconfusion[label][pred]を集計する', () => {
    const items = [
      { label: 0, pred: 0 },
      { label: 0, pred: 1 },
      { label: 1, pred: 1 },
      { label: null, pred: 1 },
    ];
    const m = buildConfusion(items, 2);
    expect(m).toEqual([
      [1, 1],
      [0, 1],
    ]);
  });
});
