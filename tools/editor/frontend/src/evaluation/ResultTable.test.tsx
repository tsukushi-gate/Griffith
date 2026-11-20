// @vitest-environment jsdom
// ResultTableのRTLテスト。仮想スクロール(全行はDOM化されない)、
// 確信度昇順ソート、行クリックコールバックを検証する。

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EvalItem } from './api';
import { ResultTable } from './ResultTable';

function makeItems(n: number): EvalItem[] {
  return Array.from({ length: n }, (_, i) => ({
    index: i,
    label: null,
    pred: null,
    label_bits: [i % 2],
    pred_bits: [i % 2],
    correct: true,
    soft_confidence: (n - i) / n, // indexが大きいほど確信度は小さい
    soft_margin: (n - i) / n,
    hard_soft_mismatch: false,
    input_bits: [i & 1, (i >> 1) & 1],
    image: null,
  }));
}

afterEach(cleanup);

describe('ResultTable', () => {
  it('数千件でも可視行だけDOM化される(仮想スクロール)', () => {
    render(<ResultTable items={makeItems(3000)} kind="binary_vector" />);
    const rows = screen.getAllByTestId(/^result-row-/);
    // 全3000行ではなく、ビューポート+overscan分のみ
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.length).toBeLessThan(100);
  });

  it('確信度ソートで最も迷うサンプル(確信度最小)が先頭に来る', () => {
    render(<ResultTable items={makeItems(50)} kind="binary_vector" />);
    fireEvent.click(screen.getByTestId('sort-confidence'));
    // 確信度最小=index49 が先頭付近に来る
    expect(screen.getByTestId('result-row-49')).toBeTruthy();
    // index0(確信度最大)は先頭ウィンドウから外れる
    expect(screen.queryByTestId('result-row-0')).toBeNull();
  });

  it('行クリックでコールバックが呼ばれる', () => {
    const onRowClick = vi.fn();
    render(<ResultTable items={makeItems(5)} kind="binary_vector" onRowClick={onRowClick} />);
    fireEvent.click(screen.getByTestId('result-row-2'));
    expect(onRowClick).toHaveBeenCalledWith(expect.objectContaining({ index: 2 }));
  });
});
