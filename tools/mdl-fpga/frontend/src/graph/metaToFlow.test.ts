import { describe, expect, it } from 'vitest';
import type { Meta, ProbeSpec } from '../api/types';
import { edgeId, metaToFlow } from './metaToFlow';
import { probeEdges, probeNodes, probeSourceId } from './probeEdges';

const META: Meta = {
  name: 't', in_width: 4, out_width: 1, score_bits: null, k: null,
  input_semantics: null, input_shape: null,
  nodes: [
    { id: 'l1', wire: 'w_l1', width: 2, type: 'lut_layer', alias: false, score_bits: null },
    { id: 'cat', wire: 'w_cat', width: 4, type: 'concat', alias: false, score_bits: null },
  ],
  inputs: [{ id: 'x', width: 4 }],
  outputs: [{ id: 'y', width: 1 }],
  edges: [
    { src: 'x', dst: 'l1', slot: 0 },
    { src: 'x', dst: 'cat', slot: 0 },
    { src: 'l1', dst: 'cat', slot: 1 },
    { src: 'cat', dst: 'y', slot: 0 },
  ],
  cycle: 0, workdir: '',
};

describe('metaToFlow', () => {
  it('ポート+ブロックのノードとエッジを生成する', () => {
    const { nodes, edges } = metaToFlow(META, {});
    expect(nodes.map((n) => n.id)).toEqual(['x', 'l1', 'cat', 'y']);
    expect(nodes.find((n) => n.id === 'x')!.type).toBe('port');
    expect(edges).toHaveLength(4);
  });
  it('多入力(concat)はスロットハンドル、単入力は無指定', () => {
    const { nodes, edges } = metaToFlow(META, {});
    const cat = nodes.find((n) => n.id === 'cat')!;
    expect((cat.data as { inSlots: number }).inSlots).toBe(2);
    const e = edges.find((e) => e.id === 'l1->cat#s1')!;
    expect(e.targetHandle).toBe('s1');
    expect(edges.find((e) => e.id === 'x->l1')!.targetHandle).toBeUndefined();
  });
  it('保存座標が優先される', () => {
    const { nodes } = metaToFlow(META, { l1: { x: 42, y: 7 } });
    expect(nodes.find((n) => n.id === 'l1')!.position).toEqual({ x: 42, y: 7 });
  });
  it('edgeId 規約は editor と同一(slot>0 で #s)', () => {
    expect(edgeId('a', 'b', 0)).toBe('a->b');
    expect(edgeId('a', 'b', 2)).toBe('a->b#s2');
  });
});

describe('system mode nodes', () => {
  it('system型ノードはblockとして描画され、labelとpathを保持する', () => {
    const meta: Meta = {
      ...META,
      nodes: [
        ...META.nodes,
        { id: 'img_idx', width: 3, type: 'system',
          path: 'u_sys.img_idx', label: 'ROM index' },
      ],
      edges: [...META.edges, { src: 'img_idx', dst: 'x', slot: 0 }],
    };
    const { nodes, edges } = metaToFlow(meta, {});
    const sys = nodes.find((n) => n.id === 'img_idx')!;
    expect(sys.type).toBe('block');
    expect((sys.data as { label: string }).label).toBe('ROM index');
    expect(edges.find((e) => e.id === 'img_idx->x')).toBeDefined();
  });
});

describe('probeEdges/probeNodes', () => {
  const probe = (patch: Partial<ProbeSpec>): ProbeSpec => ({
    id: 'p1', target: 'l1', slice: null, renderer: 'auto', args: {},
    pos: { x: 0, y: 0 }, pinned: false, label: null, depth: 1, scale: 1,
    ...patch,
  });
  it('ピン留めはキャンバスから除外される', () => {
    const ps = [probe({}), probe({ id: 'p2', pinned: true })];
    expect(probeNodes(ps).map((n) => n.id)).toEqual(['probe:p1']);
    const es = probeEdges(ps, new Set(['l1']));
    expect(es.map((e) => e.id)).toEqual(['probe-edge:p1']);
  });
  it('y_comb は出力ポート y に接続表示される', () => {
    expect(probeSourceId('y_comb')).toBe('y');
    const es = probeEdges([probe({ target: 'y_comb' })], new Set(['y']));
    expect(es[0]!.source).toBe('y');
  });
  it('孤児プローブのエッジは作らない(ノードは残す=エラーカード表示)', () => {
    const ps = [probe({ target: 'ghost' })];
    expect(probeEdges(ps, new Set(['l1']))).toHaveLength(0);
    expect(probeNodes(ps)).toHaveLength(1);
  });
  it('スライスはエッジラベルになる', () => {
    const es = probeEdges([probe({ slice: [7, 0] })], new Set(['l1']));
    expect(es[0]!.label).toBe('[7:0]');
  });
});
