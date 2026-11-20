// @vitest-environment jsdom
// 統計パネルのRTLテスト。GET /api/stats(モック)を購読して指標を表示し、
// 選択ノード型の寄与行がハイライトされることを確認する。

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';

const getStats = vi.fn();

vi.mock('../api/rest', () => ({
  getStats,
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
  ApiError: class ApiError extends Error {},
}));

const { useGraphStore } = await import('../store/graphStore');
const { StatsPanel } = await import('./StatsPanel');

function ir(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { editor: { ports: { x: { position: { x: 0, y: 0 } }, y: { position: { x: 9, y: 0 } } } } },
    inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'n1',
        type: 'lut_layer',
        params: { in_width: 2, out_width: 2, n_inputs: 2, indices: [[0, 0], [1, 1]], truth_tables: ['0x6', '0x6'] },
        train_info: {},
        meta: { editor: { position: { x: 4, y: 0 } } },
      },
    ],
    edges: [{ src: 'x', dst: 'n1' }, { src: 'n1', dst: 'y' }],
  };
}

const STATS = {
  total_luts: 42,
  lut_counts: { '2': 42 },
  logic_depth: 3,
  wire_count: 90,
  group_sums: 0,
  node_counts: { lut_layer: 1 },
  lowered: false,
  num_params: 168,
  partial: false,
};

beforeEach(() => {
  getStats.mockReset();
  getStats.mockResolvedValue(STATS);
  useGraphStore.getState().setIR(ir());
  useGraphStore.setState({ selectedNodeId: null });
});

afterEach(() => cleanup());

describe('StatsPanel', () => {
  it('統計を取得して主要指標を表示する', async () => {
    render(<StatsPanel />);
    await waitFor(() => expect(getStats).toHaveBeenCalled());
    expect((await screen.findByTestId('stat-total_luts')).textContent).toContain('42');
    expect(screen.getByTestId('stat-num_params').textContent).toContain('168');
    expect(screen.getByTestId('stat-node-lut_layer')).toBeTruthy();
  });

  it('選択ノード型の寄与行にハイライトclassが付く', async () => {
    useGraphStore.setState({ selectedNodeId: 'n1' });
    render(<StatsPanel />);
    const row = await screen.findByTestId('stat-node-lut_layer');
    expect(row.className).toContain('ring-primary/40');
  });
});
