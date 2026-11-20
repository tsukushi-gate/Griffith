import { describe, expect, it } from 'vitest';
import type { Meta } from '../api/types';
import type { LutParams } from '../store/detailStore';
import {
  EXPAND_MAX_WIDTH,
  applyExpansion,
  groupSize,
  layerSourceId,
} from './expandGraph';
import { metaToFlow } from './metaToFlow';

const META: Meta = {
  name: 't', in_width: 4, out_width: 2, score_bits: null, k: null,
  input_semantics: null, input_shape: null,
  nodes: [
    { id: 'l1', wire: 'w_l1', width: 2, type: 'lut_layer', alias: false, score_bits: null },
  ],
  inputs: [{ id: 'x', width: 4 }],
  outputs: [{ id: 'y', width: 2 }],
  edges: [
    { src: 'x', dst: 'l1', slot: 0 },
    { src: 'l1', dst: 'y', slot: 0 },
  ],
  cycle: 0, workdir: '',
};

const PARAMS: LutParams = {
  in_width: 4, out_width: 2, n_inputs: 2,
  indices: [[0, 2], [1, 3]],
  truth_tables: ['0x6', '0x8'],
};

describe('applyExpansion', () => {
  it('展開でグループ+子ゲート+ビット配線に差し替わる', () => {
    const flow = metaToFlow(META, {});
    const { nodes, edges } = applyExpansion(
      META, flow.nodes, flow.edges, ['l1'], { l1: PARAMS });

    const group = nodes.find((n) => n.id === 'l1')!;
    expect(group.type).toBe('lutGroup');
    expect(group.style).toEqual(groupSize(2));

    const gates = nodes.filter((n) => n.type === 'gate');
    expect(gates.map((n) => n.id)).toEqual(['l1::g0', 'l1::g1']);
    expect(gates[0]!.parentId).toBe('l1');
    expect((gates[0]!.data as { tt: number }).tt).toBe(0x6);
    expect((gates[1]!.data as { srcBits: number[] }).srcBits).toEqual([2, 3]);

    // 親の直後に子(React Flow v12 の順序要件)
    const ids = nodes.map((n) => n.id);
    expect(ids.indexOf('l1::g0')).toBe(ids.indexOf('l1') + 1);

    // バスエッジ x->l1 はビット配線 4 本(2 gates × 2 inputs)へ
    expect(edges.find((e) => e.id === 'x->l1')).toBeUndefined();
    const bitEdges = edges.filter((e) => e.id.startsWith('bit:l1:'));
    expect(bitEdges).toHaveLength(4);
    expect(bitEdges[0]!.source).toBe('x');
    expect((bitEdges[0]!.data as { srcBit: number }).srcBit).toBe(0);
    // 下流バス l1->y は残る
    expect(edges.find((e) => e.id === 'l1->y')).toBeDefined();
  });

  it('params 未取得・幅超過はそのまま(拒否)', () => {
    const flow = metaToFlow(META, {});
    const same = applyExpansion(META, flow.nodes, flow.edges, ['l1'], {});
    expect(same.nodes.find((n) => n.id === 'l1')!.type).toBe('block');

    const wide = { ...PARAMS, out_width: EXPAND_MAX_WIDTH + 1 };
    const same2 = applyExpansion(
      META, flow.nodes, flow.edges, ['l1'], { l1: wide });
    expect(same2.nodes.find((n) => n.id === 'l1')!.type).toBe('block');
  });

  it('layerSourceId は供給元を返す', () => {
    expect(layerSourceId(META, 'l1')).toBe('x');
    expect(layerSourceId(META, 'ghost')).toBeNull();
  });
});
