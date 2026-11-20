// IRラウンドトリップテスト: irToFlow → flowToIr で意味的同値が保たれることを
// 埋め込みサンプルIR(SAMPLES全種)で検証する。
// 許容される差分: meta.editor.position / metadata.editor.ports の追加、
// train_info.connections のデフォルト補完('random')。

import { describe, expect, it } from 'vitest';
import { readGroups } from '../editor/grouping';
import { SAMPLES } from '../samples/sampleIr';
import type { IRGraphJson } from './schema';
import { flowToIr } from './graphToIr';
import { irToFlow } from './irToGraph';

/** metaからeditor namespaceを除いた残りを返す(editor.position追加の許容用) */
function withoutEditor(meta: Record<string, unknown>): Record<string, unknown> {
  const rest = { ...meta };
  delete rest.editor;
  return rest;
}

const IR_SAMPLES = SAMPLES.flatMap((s) => (s.ir ? [[s.key, s.ir] as const] : []));

describe.each(IR_SAMPLES)(
  'サンプル %s のラウンドトリップ',
  (_key, ir) => {
    const { nodes, edges } = irToFlow(ir);
    const out = flowToIr(nodes, edges, ir);

    it('format/versionが保全される', () => {
      expect(out.format).toBe(ir.format);
      expect(out.version).toBe(ir.version);
    });

    it('ノードのid/type/paramsが一致する', () => {
      expect(out.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params }))).toEqual(
        ir.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })),
      );
    });

    it('エッジが一致する', () => {
      expect(out.edges).toEqual(ir.edges);
    });

    it('入出力ポートが一致する', () => {
      expect(out.inputs).toEqual(ir.inputs);
      expect(out.outputs).toEqual(ir.outputs);
    });

    it('train_infoの既存キーが保全される', () => {
      for (const [i, n] of out.nodes.entries()) {
        // connectionsはdraft規約のデフォルト補完があるため元キーのみ検査
        expect(n.train_info).toEqual(
          expect.objectContaining(ir.nodes[i].train_info),
        );
      }
    });

    it('meta差分はeditor.positionの追加のみ', () => {
      for (const [i, n] of out.nodes.entries()) {
        expect(withoutEditor(n.meta)).toEqual(withoutEditor(ir.nodes[i].meta));
        const editor = n.meta.editor as { position: { x: number; y: number } };
        expect(typeof editor.position.x).toBe('number');
        expect(typeof editor.position.y).toBe('number');
      }
      expect(withoutEditor(out.metadata)).toEqual(withoutEditor(ir.metadata));
    });

    it('2周目のラウンドトリップで固定点になる', () => {
      const flow2 = irToFlow(out);
      const out2 = flowToIr(flow2.nodes, flow2.edges, out);
      expect(out2).toEqual(out);
    });
  },
);

// サブサーキット化: グループ定義(meta.editor.groups)が往復で保全されることを固定する。
describe('グループ情報のラウンドトリップ', () => {
  const base = IR_SAMPLES.find(([, ir]) => ir.nodes.length >= 2)?.[1] as IRGraphJson;

  it('前提: メンバー2個以上のサンプルが存在する', () => {
    expect(base).toBeDefined();
  });

  const withGroups: IRGraphJson = {
    ...base,
    metadata: {
      ...base.metadata,
      editor: {
        ...(base.metadata.editor as Record<string, unknown>),
        groups: {
          g1: {
            name: '半加算器',
            nodeIds: base.nodes.slice(0, 2).map((n) => n.id),
            collapsed: true,
            position: { x: 42, y: 7 },
          },
        },
      },
    },
  };

  const { nodes, edges } = irToFlow(withGroups);
  const out = flowToIr(nodes, edges, withGroups);

  it('グループ定義がそのまま保全される', () => {
    const groups = readGroups(out);
    expect(groups).toHaveLength(1);
    expect(groups[0]).toEqual({
      id: 'g1',
      name: '半加算器',
      nodeIds: withGroups.nodes.slice(0, 2).map((n) => n.id),
      collapsed: true,
      position: { x: 42, y: 7 },
    });
  });

  it('ノード構造・エッジはグループ有無で変わらない', () => {
    const plainFlow = irToFlow(base);
    const plain = flowToIr(plainFlow.nodes, plainFlow.edges, base);
    expect(out.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params }))).toEqual(
      plain.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })),
    );
    expect(out.edges).toEqual(plain.edges);
  });
});
