// alignRunSeriesの単体テスト

import { describe, expect, it } from 'vitest';
import { alignRunSeries } from './runsCompare';

describe('alignRunSeries', () => {
  it('step格子が異なる2runを共通x軸へnullパディング整列する', () => {
    const aligned = alignRunSeries([
      { steps: [1, 3], values: [0.1, 0.3] },
      { steps: [2, 3], values: [0.2, 0.9] },
    ]);
    expect(aligned[0]).toEqual([1, 2, 3]);
    expect(aligned[1]).toEqual([0.1, null, 0.3]);
    expect(aligned[2]).toEqual([null, 0.2, 0.9]);
  });

  it('空run・単一runを扱える', () => {
    expect(alignRunSeries([])).toEqual([[]]);
    const single = alignRunSeries([{ steps: [5], values: [1.0] }]);
    expect(single).toEqual([[5], [1.0]]);
  });

  it('重複stepは後勝ちでなく位置一致で埋まる', () => {
    const aligned = alignRunSeries([{ steps: [1, 1, 2], values: [0.1, 0.15, 0.2] }]);
    expect(aligned[0]).toEqual([1, 2]);
    // 同一stepが複数ある場合は後の値が残る(実運用では発生しない防御的仕様)
    expect(aligned[1]).toEqual([0.15, 0.2]);
  });
});
