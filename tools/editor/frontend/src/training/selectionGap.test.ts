// 選択ギャップ・ダッシュボード(§4-1)純粋ヘルパの単体テスト。

import { describe, expect, it } from 'vitest';
import { clampUncertainty, latestGap } from './selectionGap';

describe('clampUncertainty', () => {
  it('backend正規化済み値[0,1]はそのまま通す', () => {
    expect(clampUncertainty(0)).toBe(0);
    expect(clampUncertainty(0.42)).toBeCloseTo(0.42, 9);
    expect(clampUncertainty(1)).toBe(1);
  });

  it('範囲外・非有限値は[0,1]へ丸める', () => {
    expect(clampUncertainty(100)).toBe(1);
    expect(clampUncertainty(-1)).toBe(0);
    expect(clampUncertainty(Number.NaN)).toBe(0);
    expect(clampUncertainty(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe('latestGap', () => {
  it('最新の soft - hard をギャップとして返す', () => {
    const gap = latestGap(
      [
        [10, 0.9],
        [20, 0.95],
      ],
      [
        [10, 0.7],
        [20, 0.8],
      ],
    );
    expect(gap).not.toBeNull();
    expect(gap?.soft).toBe(0.95);
    expect(gap?.hard).toBe(0.8);
    expect(gap?.gap).toBeCloseTo(0.15, 6);
  });

  it('どちらかが空ならnull', () => {
    expect(latestGap([], [[10, 0.8]])).toBeNull();
    expect(latestGap([[10, 0.9]], [])).toBeNull();
  });
});
