// サブサーキット化の表示合成(applyGrouping)とメタデータ読取(readGroups)の単体テスト。

import type { Edge } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { EditorNode } from '../ir/irToGraph';
import type { IRGraphJson } from '../ir/schema';
import {
  GROUP_HEADER_PREFIX,
  GROUP_NODE_PREFIX,
  applyGrouping,
  groupIdFromNodeId,
  isGroupEdgeId,
  isGroupNodeId,
  readGroups,
  type EditorGroup,
} from './grouping';

function node(id: string, x = 0, y = 0): EditorNode {
  return {
    id,
    type: 'generic',
    position: { x, y },
    data: { label: id, subtitle: '', kind: 'blackbox' },
  } as EditorNode;
}

function edge(source: string, target: string): Edge {
  return { id: `${source}->${target}`, source, target };
}

// X → A → B → Y(A,Bをグループ化する構成)
const NODES = [node('X'), node('A', 100), node('B', 200, 40), node('Y', 300)];
const EDGES = [edge('X', 'A'), edge('A', 'B'), edge('B', 'Y')];

function group(overrides: Partial<EditorGroup>): EditorGroup {
  return {
    id: 'g1',
    name: 'sub',
    nodeIds: ['A', 'B'],
    collapsed: true,
    position: { x: 100, y: 0 },
    ...overrides,
  };
}

describe('id判定ヘルパ', () => {
  it('グループノードidを判定する', () => {
    expect(isGroupNodeId('grp:g1')).toBe(true);
    expect(isGroupNodeId('grph:g1')).toBe(true);
    expect(isGroupNodeId('A')).toBe(false);
    expect(isGroupEdgeId('grpE:g1->Y')).toBe(true);
    expect(groupIdFromNodeId('grp:g1')).toBe('g1');
    expect(groupIdFromNodeId('grph:g1')).toBe('g1');
    expect(groupIdFromNodeId('A')).toBeNull();
  });
});

describe('applyGrouping: 折りたたみ', () => {
  const { nodes, edges } = applyGrouping(NODES, EDGES, [group({})]);

  it('メンバーが隠れグループノードへ差し替わる', () => {
    const ids = nodes.map((n) => n.id);
    expect(ids).not.toContain('A');
    expect(ids).not.toContain('B');
    expect(ids).toContain('X');
    expect(ids).toContain('Y');
    expect(ids).toContain(`${GROUP_NODE_PREFIX}g1`);
  });

  it('内部エッジは隠れ、境界エッジはグループノードへ再配線される', () => {
    // A->B(内部)は消える
    expect(edges.find((e) => e.source === 'A' || e.target === 'B')).toBeUndefined();
    // X->A は X->grp:g1 へ
    expect(
      edges.some((e) => e.source === 'X' && e.target === `${GROUP_NODE_PREFIX}g1`),
    ).toBe(true);
    // B->Y は grp:g1->Y へ
    expect(
      edges.some((e) => e.source === `${GROUP_NODE_PREFIX}g1` && e.target === 'Y'),
    ).toBe(true);
  });

  it('グループノードに外部入出力数が載る', () => {
    const gnode = nodes.find((n) => n.id === `${GROUP_NODE_PREFIX}g1`)!;
    const data = gnode.data as { inputCount: number; outputCount: number; memberCount: number };
    expect(data.memberCount).toBe(2);
    expect(data.inputCount).toBe(1);
    expect(data.outputCount).toBe(1);
  });
});

describe('applyGrouping: 展開', () => {
  const { nodes, edges } = applyGrouping(NODES, EDGES, [group({ collapsed: false })]);

  it('メンバーは残りヘッダノードが追加される', () => {
    const ids = nodes.map((n) => n.id);
    expect(ids).toContain('A');
    expect(ids).toContain('B');
    expect(ids).toContain(`${GROUP_HEADER_PREFIX}g1`);
  });

  it('エッジは変更されない', () => {
    expect(edges).toEqual(EDGES);
  });
});

describe('applyGrouping: グループなし/空メンバー', () => {
  it('グループ0件は同一参照を返す', () => {
    const out = applyGrouping(NODES, EDGES, []);
    expect(out.nodes).toBe(NODES);
    expect(out.edges).toBe(EDGES);
  });

  it('存在しないメンバーだけのグループは無視される', () => {
    const out = applyGrouping(NODES, EDGES, [group({ nodeIds: ['Z'] })]);
    expect(out.nodes).toBe(NODES);
  });
});

describe('readGroups', () => {
  it('metadata.editor.groups を EditorGroup[] へ読む', () => {
    const ir = {
      metadata: {
        editor: {
          groups: {
            g1: { name: 'sub', nodeIds: ['A', 'B'], collapsed: true, position: { x: 1, y: 2 } },
          },
        },
      },
    } as unknown as IRGraphJson;
    const groups = readGroups(ir);
    expect(groups).toHaveLength(1);
    expect(groups[0]).toEqual({
      id: 'g1',
      name: 'sub',
      nodeIds: ['A', 'B'],
      collapsed: true,
      position: { x: 1, y: 2 },
    });
  });

  it('groups未設定は空配列', () => {
    expect(readGroups({ metadata: {} } as unknown as IRGraphJson)).toEqual([]);
    expect(readGroups(null)).toEqual([]);
  });
});
