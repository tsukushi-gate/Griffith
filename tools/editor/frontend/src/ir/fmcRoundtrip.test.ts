// 構造化配線の引数(train_info.wiring)と group_sum の判定バイアス(params.bias)が
// irToFlow → flowToIr の往復で保たれ、random へ戻すと wiring が消えることを検証する。

import { describe, expect, it } from 'vitest';
import { flowToIr } from './graphToIr';
import type { LutLayerNodeData } from './irToGraph';
import { irToFlow } from './irToGraph';
import type { IRGraphJson } from './schema';

function ir(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 8, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 2, shape: null, encoding: 'class_scores', semantics: null }],
    nodes: [
      {
        id: 'lut',
        type: 'lut_layer',
        params: {
          in_width: 8,
          out_width: 8,
          n_inputs: 2,
          indices: [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 0, 1, 2, 3],
          ],
          truth_tables: new Array(8).fill('0x6'),
        },
        train_info: {
          parameterization: 'softmax',
          connections: 'butterfly',
          wiring: { stride: 4, dedupe: true },
          selection: 'soft',
        },
        meta: {},
      },
      {
        id: 'gs',
        type: 'group_sum',
        params: { in_width: 8, k: 2, tau: 1.0, bias: [0, 1] },
        train_info: {},
        meta: {},
      },
    ],
    edges: [
      { src: 'x', dst: 'lut' },
      { src: 'lut', dst: 'gs' },
      { src: 'gs', dst: 'y' },
    ],
  };
}

describe('構造化配線と判定バイアスの往復', () => {
  it('wiring と bias が保たれる', () => {
    const src = ir();
    const { nodes, edges } = irToFlow(src);
    const lutData = nodes.find((n) => n.id === 'lut')!.data as LutLayerNodeData;
    expect(lutData.connections).toBe('butterfly');
    expect(lutData.wiring).toEqual({ stride: 4, dedupe: true });

    const out = flowToIr(nodes, edges, src);
    const lut = out.nodes.find((n) => n.id === 'lut')!;
    expect(lut.train_info.connections).toBe('butterfly');
    expect(lut.train_info.wiring).toEqual({ stride: 4, dedupe: true });
    expect(lut.params.indices).toEqual(src.nodes[0].params.indices);
    const gs = out.nodes.find((n) => n.id === 'gs')!;
    expect(gs.params.bias).toEqual([0, 1]);
  });

  it('random へ戻すと wiring が消え、k を変えると bias が捨てられる', () => {
    const src = ir();
    const { nodes, edges } = irToFlow(src);
    const patched = nodes.map((n) => {
      if (n.id === 'lut') return { ...n, data: { ...n.data, connections: 'random' } } as typeof n;
      if (n.id === 'gs') return { ...n, data: { ...n.data, k: 4 } } as typeof n;
      return n;
    });
    const out = flowToIr(patched, edges, src);
    const lut = out.nodes.find((n) => n.id === 'lut')!;
    expect(lut.train_info.connections).toBe('random');
    expect('wiring' in lut.train_info).toBe(false);
    const gs = out.nodes.find((n) => n.id === 'gs')!;
    expect(gs.params.k).toBe(4);
    expect('bias' in gs.params).toBe(false);
  });

  it('全 0 の bias は params に出さない', () => {
    const src = ir();
    src.nodes[1].params.bias = [0, 0];
    const { nodes, edges } = irToFlow(src);
    const out = flowToIr(nodes, edges, src);
    expect('bias' in out.nodes.find((n) => n.id === 'gs')!.params).toBe(false);
  });
});
