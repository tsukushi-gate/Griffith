// @vitest-environment jsdom
// Trade-offグラフの描画+点クリックのテスト。

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => cleanup());
import { TradeoffChart } from './TradeoffChart';
import type { NasTrialRow } from './types';

function t(id: number, acc: number, luts: number, extra: Partial<NasTrialRow> = {}): NasTrialRow {
  return {
    trial_id: id, acc, best_acc: acc, total_luts: luts, logic_depth: 1, gates: 1,
    num_params: 1, partial: false, state: 'done', parent_id: null,
    genome: { layers: 1, width: 4, n_inputs: 2 }, ...extra,
  };
}

describe('TradeoffChart', () => {
  it('trialなしで空表示', () => {
    render(<TradeoffChart trials={[]} />);
    expect(screen.getByTestId('tradeoff-empty')).toBeTruthy();
  });

  it('点を描画しパレート点にフラグを立てる', () => {
    const trials = [t(0, 0.5, 10), t(1, 1.0, 2)];
    render(<TradeoffChart trials={trials} />);
    expect(screen.getByTestId('tradeoff-point-0').getAttribute('data-pareto')).toBe('false');
    expect(screen.getByTestId('tradeoff-point-1').getAttribute('data-pareto')).toBe('true');
  });

  it('点クリックで onSelectTrial が発火する', () => {
    const onSelect = vi.fn();
    render(<TradeoffChart trials={[t(3, 0.9, 4)]} onSelectTrial={onSelect} />);
    fireEvent.click(screen.getByTestId('tradeoff-point-3'));
    expect(onSelect).toHaveBeenCalledWith(3);
  });

  it('partial点は中空(fill=transparent)でパレート除外', () => {
    const trials = [t(0, 1.0, 1, { partial: true }), t(1, 0.6, 3)];
    render(<TradeoffChart trials={trials} />);
    expect(screen.getByTestId('tradeoff-point-0').getAttribute('fill')).toBe('transparent');
    expect(screen.getByTestId('tradeoff-point-0').getAttribute('data-pareto')).toBe('false');
  });
});
