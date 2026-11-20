// トポロジカル順序のテスト。
// 入力ポート → ノード(依存順) → 出力ポート の並びを検証する。

import { describe, expect, it } from 'vitest';
import type { IRGraphJson } from '../../ir/schema';
import { topologicalOrder } from './order';

function node(id: string): IRGraphJson['nodes'][number] {
  return { id, type: 'lut_layer', params: {}, train_info: {}, meta: {} };
}

function ir(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 2 }],
    outputs: [{ id: 'y', width: 2 }],
    // 定義順をわざと依存順と逆(b→a)にして、topoで是正されることを確認
    nodes: [node('b'), node('a')],
    edges: [
      { src: 'x', dst: 'a' },
      { src: 'a', dst: 'b' },
      { src: 'b', dst: 'y' },
    ],
  };
}

describe('topologicalOrder', () => {
  it('入力→依存順ノード→出力の順に並ぶ', () => {
    expect(topologicalOrder(ir())).toEqual(['x', 'a', 'b', 'y']);
  });

  it('循環があっても全ノードを1回ずつ含む(残りは末尾)', () => {
    const g = ir();
    g.edges.push({ src: 'b', dst: 'a' }); // a↔b循環
    const order = topologicalOrder(g);
    expect(order[0]).toBe('x');
    expect(order[order.length - 1]).toBe('y');
    expect(new Set(order).size).toBe(order.length);
    expect(order).toContain('a');
    expect(order).toContain('b');
  });
});
