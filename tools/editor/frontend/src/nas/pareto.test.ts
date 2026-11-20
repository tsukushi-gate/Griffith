// パレート抽出の純ロジックテスト。node環境(DOM不要)。

import { describe, expect, it } from 'vitest';
import { paretoFront, paretoLine, positionVsPareto } from './pareto';
import type { NasTrialRow } from './types';

function t(
  trial_id: number,
  acc: number,
  total_luts: number,
  extra: Partial<NasTrialRow> = {},
): NasTrialRow {
  return {
    trial_id,
    acc,
    best_acc: acc,
    total_luts,
    logic_depth: 1,
    gates: 1,
    num_params: 1,
    partial: false,
    state: 'done',
    parent_id: null,
    genome: { layers: 1, width: 4, n_inputs: 2 },
    ...extra,
  };
}

describe('paretoFront', () => {
  it('非劣解のみ返す', () => {
    const trials = [
      t(0, 0.5, 10), // 支配される
      t(1, 0.9, 5), // 非劣
      t(2, 1.0, 8), // 非劣(accで勝る)
      t(3, 0.9, 8), // trial1に支配される
    ];
    expect(paretoFront(trials)).toEqual([1, 2]);
  });

  it('partial:true は除外する(裁定6)', () => {
    const trials = [t(0, 1.0, 1, { partial: true }), t(1, 0.6, 3)];
    expect(paretoFront(trials)).toEqual([1]);
  });

  it('done でない trial は除外する', () => {
    const trials = [t(0, 0.0, 1, { state: 'failed' }), t(1, 0.7, 2)];
    expect(paretoFront(trials)).toEqual([1]);
  });

  it('空入力で空配列', () => {
    expect(paretoFront([])).toEqual([]);
  });
});

describe('paretoLine', () => {
  it('total_luts昇順で点列を返す', () => {
    const trials = [t(0, 1.0, 8), t(1, 0.9, 5)];
    const line = paretoLine(trials);
    expect(line.map((p) => p.trial_id)).toEqual([1, 0]);
  });
});

describe('positionVsPareto', () => {
  const trials = [t(1, 0.9, 5), t(2, 1.0, 8)];

  it('フロントに支配される点は behind', () => {
    const r = positionVsPareto({ acc: 0.8, total_luts: 10 }, trials);
    expect(r.position).toBe('behind');
    expect(r.dominatedBy).not.toBeNull();
  });

  it('支配されない点は ahead', () => {
    const r = positionVsPareto({ acc: 1.0, total_luts: 3 }, trials);
    expect(r.position).toBe('ahead');
    expect(r.dominatedBy).toBeNull();
  });

  it('acc未知は unknown', () => {
    const r = positionVsPareto({ acc: null, total_luts: 3 }, trials);
    expect(r.position).toBe('unknown');
  });

  it('候補ゼロは unknown', () => {
    const r = positionVsPareto({ acc: 0.9, total_luts: 3 }, []);
    expect(r.position).toBe('unknown');
  });
});
