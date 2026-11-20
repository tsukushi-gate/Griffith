// 多入力DAG(v0.2)のラウンドトリップテスト。
// ダイヤモンド型(fanout→2分岐→concat)と残差型(fanout→add)で
// graph→IR→graphの往復同一性とdst_slot省略規約を検証する。

import { describe, expect, it } from 'vitest';
import { flowToIr, slotFromHandle } from './graphToIr';
import {
  buildEditorEdge,
  irToFlow,
  RESIDUAL_EDGE_STYLE,
  type ConcatNodeData,
  type EditorNode,
} from './irToGraph';
import type { IRGraphJson, IRNodeJson } from './schema';

function lut(id: string, inWidth: number, outWidth: number): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: 2,
      indices: [
        new Array<number>(outWidth).fill(0),
        new Array<number>(outWidth).fill(1),
      ],
      truth_tables: new Array<string>(outWidth).fill('0x6'),
    },
    train_info: { parameterization: 'softmax', connections: 'random', selection: 'soft' },
    meta: {},
  };
}

/** ダイヤモンド型: x(4) →fanout→ [n1(4→3), n2(4→2)] → concat(3+2=5) → n3(5→1) → y */
const DIAMOND: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.2.0',
  metadata: { name: 'diamond' },
  inputs: [{ id: 'x', width: 4, shape: null, encoding: null, semantics: null }],
  outputs: [{ id: 'y', width: 1, shape: null, encoding: null, semantics: null }],
  nodes: [
    lut('n1', 4, 3),
    lut('n2', 4, 2),
    {
      id: 'c1',
      type: 'concat',
      params: { in_widths: [3, 2] },
      train_info: {},
      meta: {},
    },
    lut('n3', 5, 1),
  ],
  edges: [
    { src: 'x', dst: 'n1' },
    { src: 'x', dst: 'n2' },
    { src: 'n1', dst: 'c1' },
    { src: 'n2', dst: 'c1', dst_slot: 1 },
    { src: 'c1', dst: 'n3' },
    { src: 'n3', dst: 'y' },
  ],
};

/** 残差型: x(2) → n1(2→2) → add(xor) ←slot1─ x、add → y */
const RESIDUAL: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.2.0',
  metadata: { name: 'residual' },
  inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
  outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
  nodes: [
    lut('n1', 2, 2),
    {
      id: 'a1',
      type: 'add',
      params: { width: 2, mode: 'xor' },
      train_info: {},
      meta: {},
    },
  ],
  edges: [
    { src: 'x', dst: 'n1' },
    { src: 'n1', dst: 'a1' },
    { src: 'x', dst: 'a1', dst_slot: 1 },
    { src: 'a1', dst: 'y' },
  ],
};

/** conv/pool: x(16=1x4x4) → conv2d(oc2,k3) → (2,2,2)=8 → or_pool2d(k2,s2) → (2,1,1)=2 → y */
const CONV: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.2.0',
  metadata: { name: 'conv' },
  inputs: [{ id: 'x', width: 16, shape: null, encoding: null, semantics: null }],
  outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
  nodes: [
    {
      id: 'cv1',
      type: 'conv2d',
      params: {
        in_shape: [1, 4, 4],
        out_channels: 2,
        kernel_size: [3, 3],
        stride: [1, 1],
        padding: [0, 0],
        tree_depth: 1,
        n_inputs: 2,
        indices_flat: [
          [0, 1],
          [2, 3],
        ],
        truth_tables: ['0x6', '0xe'],
      },
      train_info: {},
      meta: {},
    },
    {
      id: 'p1',
      type: 'or_pool2d',
      params: { in_shape: [2, 2, 2], kernel_size: 2, stride: 2, padding: 0 },
      train_info: {},
      meta: {},
    },
  ],
  edges: [
    { src: 'x', dst: 'cv1' },
    { src: 'cv1', dst: 'p1' },
    { src: 'p1', dst: 'y' },
  ],
};

describe.each([
  ['diamond', DIAMOND],
  ['residual', RESIDUAL],
  ['conv', CONV],
] as const)('DAGラウンドトリップ: %s', (_key, ir) => {
  const { nodes, edges } = irToFlow(ir);
  const out = flowToIr(nodes, edges, ir);

  it('ノードのid/type/paramsが一致する', () => {
    expect(
      out.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })),
    ).toEqual(ir.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })));
  });

  it('エッジ(dst_slot含む)が一致する', () => {
    expect(out.edges).toEqual(ir.edges);
  });

  it('入出力ポートが一致する', () => {
    expect(out.inputs).toEqual(ir.inputs);
    expect(out.outputs).toEqual(ir.outputs);
  });

  it('2周目のラウンドトリップで固定点になる', () => {
    const flow2 = irToFlow(out);
    const out2 = flowToIr(flow2.nodes, flow2.edges, out);
    expect(out2).toEqual(out);
  });
});

describe('dst_slot省略規約(バックエンドschema.pyと同一)', () => {
  it('slot 0のエッジはdst_slotキー自体を出力しない', () => {
    const { nodes, edges } = irToFlow(DIAMOND);
    const out = flowToIr(nodes, edges, DIAMOND);
    const e0 = out.edges.find((e) => e.src === 'n1' && e.dst === 'c1')!;
    expect('dst_slot' in e0).toBe(false);
  });

  it('slot 1以降のエッジはdst_slotを持つ', () => {
    const { nodes, edges } = irToFlow(DIAMOND);
    const out = flowToIr(nodes, edges, DIAMOND);
    const e1 = out.edges.find((e) => e.src === 'n2' && e.dst === 'c1')!;
    expect(e1.dst_slot).toBe(1);
  });

  it('dst_slotなしのIRエッジはslot 0(targetHandle s0)として読む', () => {
    const { edges } = irToFlow(DIAMOND);
    const e0 = edges.find((e) => e.source === 'n1' && e.target === 'c1')!;
    expect(e0.targetHandle).toBe('s0');
  });
});

describe('flowエッジ変換(targetHandle/残差スタイル)', () => {
  it('多入力ノード行きはtargetHandleを明示し、slot>0はidに#s{slot}を付ける', () => {
    const e = buildEditorEdge('a', 'b', 2, 'concat');
    expect(e.id).toBe('a->b#s2');
    expect(e.targetHandle).toBe('s2');
    expect(e.style).toBeUndefined();
  });

  it('単一入力ノード行き(slot0)は既存規約のまま(targetHandleなし)', () => {
    const e = buildEditorEdge('a', 'b', 0, 'lut_layer');
    expect(e.id).toBe('a->b');
    expect(e.targetHandle).toBeUndefined();
  });

  it('addのslot1以降は残差スタイル(破線+slot-2色)になる', () => {
    const e = buildEditorEdge('x', 'a1', 1, 'add');
    expect(e.style).toEqual(RESIDUAL_EDGE_STYLE);
    const e0 = buildEditorEdge('n1', 'a1', 0, 'add');
    expect(e0.style).toBeUndefined();
  });

  it('irToFlowで残差サンプルのslot1エッジが破線スタイルを持つ', () => {
    const { edges } = irToFlow(RESIDUAL);
    const skip = edges.find((e) => e.source === 'x' && e.target === 'a1')!;
    expect(skip.style).toEqual(RESIDUAL_EDGE_STYLE);
    expect(skip.id).toBe('x->a1#s1');
  });
});

describe('slotFromHandle', () => {
  it('s{N}形式をslot番号へ写像し、なし/不正は0', () => {
    expect(slotFromHandle('s0')).toBe(0);
    expect(slotFromHandle('s12')).toBe(12);
    expect(slotFromHandle(null)).toBe(0);
    expect(slotFromHandle(undefined)).toBe(0);
    expect(slotFromHandle('out')).toBe(0);
  });
});

describe('concat幅の接続追従', () => {
  it('上流の出力幅変更がconcatのin_widths/出力幅へ伝播する', () => {
    const { nodes, edges } = irToFlow(DIAMOND);
    // n2の出力幅を2→4に変更(構造変更)
    const mutated = nodes.map((n) =>
      n.id === 'n2'
        ? ({ ...n, data: { ...n.data, outWidth: 4 } } as EditorNode)
        : n,
    );
    const out = flowToIr(mutated, edges, DIAMOND);
    const c1 = out.nodes.find((n) => n.id === 'c1')!;
    expect(c1.params.in_widths).toEqual([3, 4]);
    // 下流n3の導出in_widthも7へ(構造変更でdraft化)
    const n3 = out.nodes.find((n) => n.id === 'n3')!;
    expect(n3.params.in_width).toBe(7);
  });

  it('irToFlowのconcat表示データはparams由来', () => {
    const { nodes } = irToFlow(DIAMOND);
    const c1 = nodes.find((n) => n.id === 'c1')!;
    const d = c1.data as ConcatNodeData;
    expect(d.inWidths).toEqual([3, 2]);
    expect(d.outWidth).toBe(5);
  });
});
