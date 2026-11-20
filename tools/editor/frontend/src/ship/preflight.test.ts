import { describe, expect, it } from 'vitest';
import type { PreflightResponse, ResourceEstimate } from '../api/rest';
import {
  blockerKindLabel,
  estimateRows,
  formatLutCounts,
  preflightBadge,
} from './preflight';

const est: ResourceEstimate = {
  total_luts: 3,
  lut_counts: { '2': 1, '3': 2 },
  logic_depth: 2,
  wire_count: 7,
  group_sums: 1,
  node_counts: { lut_layer: 3, group_sum: 1 },
  lowered: false,
};

describe('preflight純粋ヘルパ(§4-3)', () => {
  it('blockerKindLabel: 既知kindを日本語化、未知はそのまま', () => {
    expect(blockerKindLabel('blackbox')).toContain('blackbox');
    expect(blockerKindLabel('add_sum')).toContain('add');
    expect(blockerKindLabel('structure')).toBe('構造エラー');
    expect(blockerKindLabel('unknown_kind')).toBe('unknown_kind');
  });

  it('preflightBadge: 可否でtoneとテキストが変わる', () => {
    const ok: PreflightResponse = {
      hdl_exportable: true,
      blockers: [],
      warnings: [],
      estimate: est,
      draft_node_count: 0,
      verify_available: { iverilog: true, ghdl: false },
    };
    expect(preflightBadge(ok)).toEqual({ tone: 'ok', text: 'HDL出力可' });

    const blocked: PreflightResponse = {
      ...ok,
      hdl_exportable: false,
      blockers: [
        { node_id: 'bb', kind: 'blackbox', message: 'x' },
        { node_id: 'a', kind: 'add_sum', message: 'y' },
      ],
    };
    const badge = preflightBadge(blocked);
    expect(badge.tone).toBe('blocked');
    expect(badge.text).toContain('2');
  });

  it('formatLutCounts: n_inputs昇順に整形、空は "-"', () => {
    expect(formatLutCounts({ '3': 2, '2': 1 })).toBe('2入力×1, 3入力×2');
    expect(formatLutCounts({})).toBe('-');
  });

  it('estimateRows: 全6行を返し値が反映される', () => {
    const rows = estimateRows(est);
    expect(rows).toHaveLength(6);
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.value]));
    expect(byLabel['LUT総数']).toBe('3');
    expect(byLabel['論理深さ']).toBe('2');
    expect(byLabel['配線本数']).toBe('7');
    expect(byLabel['展開']).toContain('不要');
    expect(estimateRows({ ...est, lowered: true })[5].value).toContain('展開済');
  });
});
