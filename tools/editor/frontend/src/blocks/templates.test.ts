// 複合ブロックテンプレートの単体テスト。
// 各生成器のIR部分グラフが、入出力ポートへ接続後にshapeInference診断ゼロで
// あること、残差のダイヤモンド構造、id採番規約を検証する。

import { describe, expect, it } from 'vitest';
import { flowToIr } from '../ir/graphToIr';
import { irToFlow } from '../ir/irToGraph';
import { inferShapes } from '../ir/shapeInference';
import type { IRGraphJson } from '../ir/schema';
import {
  blockIoWidths,
  classifierHead,
  generateBlock,
  mlpBlock,
  residualBlock,
  templateToGraph,
  type BlockParams,
  type IdAllocator,
} from './templates';

/** テスト用のid採番(u{seq}_規約) */
function makeAlloc(start = 1): IdAllocator {
  let seq = start;
  return (suffix) => `u${seq++}_${suffix}`;
}

function diagnosticsOf(ir: IRGraphJson) {
  return inferShapes(ir.nodes, ir.edges, ir.inputs).diagnostics;
}

describe('mlpBlock', () => {
  it('幅列どおりのLutLayer連鎖を生成する', () => {
    const tpl = mlpBlock(8, [16, 4], 2, makeAlloc());
    expect(tpl.nodes.map((n) => n.type)).toEqual([
      'lut_layer',
      'lut_layer',
      'lut_layer',
    ]);
    expect(
      tpl.nodes.map((n) => [n.params.in_width, n.params.out_width]),
    ).toEqual([
      [8, 16],
      [16, 4],
      [4, 2],
    ]);
    expect(tpl.edges).toEqual([
      { src: tpl.nodes[0].id, dst: tpl.nodes[1].id },
      { src: tpl.nodes[1].id, dst: tpl.nodes[2].id },
    ]);
    expect(tpl.entryId).toBe(tpl.nodes[0].id);
    expect(tpl.exitId).toBe(tpl.nodes[2].id);
  });

  it('隠れ層なし(単層)も生成できる', () => {
    const tpl = mlpBlock(4, [], 2, makeAlloc());
    expect(tpl.nodes).toHaveLength(1);
    expect(tpl.edges).toEqual([]);
    expect(tpl.entryId).toBe(tpl.exitId);
  });

  it('全ノードがdraft(initialized=false)で採番はu{seq}_規約', () => {
    const tpl = mlpBlock(8, [16], 8, makeAlloc(5));
    for (const n of tpl.nodes) {
      expect(n.train_info.initialized).toBe(false);
      expect(n.id).toMatch(/^u\d+_/);
    }
    expect(tpl.nodes[0].id).toBe('u5_lutlayer');
    expect(tpl.nodes[1].id).toBe('u6_lutlayer');
  });

  it('接続後にshapeInference診断ゼロ', () => {
    const tpl = mlpBlock(8, [16, 4], 2, makeAlloc());
    expect(diagnosticsOf(templateToGraph('t', tpl, 8, 2))).toEqual([]);
  });
});

describe('residualBlock', () => {
  it('ダイヤモンド構造: entry→内側連鎖→add(slot0) + entry→add(slot1)', () => {
    const tpl = residualBlock(4, 2, 'xor', makeAlloc());
    const [entry, l1, l2, addNode] = tpl.nodes;
    expect(entry.id).toBe(tpl.entryId);
    expect(addNode.id).toBe(tpl.exitId);
    expect(addNode.type).toBe('add');
    expect(addNode.params).toEqual({ width: 4, mode: 'xor' });
    expect(tpl.edges).toEqual([
      { src: entry.id, dst: l1.id },
      { src: l1.id, dst: l2.id },
      { src: l2.id, dst: addNode.id }, // slot 0(dst_slot省略)
      { src: entry.id, dst: addNode.id, dst_slot: 1 }, // skip接続
    ]);
    // slot 0エッジはdst_slotキーを持たない(規約)
    expect('dst_slot' in tpl.edges[2]).toBe(false);
  });

  it('entryは恒等lut_layer(0xaパススルー・非draft)で幅維持', () => {
    const tpl = residualBlock(3, 1, 'or', makeAlloc());
    const entry = tpl.nodes[0];
    expect(entry.type).toBe('lut_layer');
    expect(entry.params.in_width).toBe(3);
    expect(entry.params.out_width).toBe(3);
    expect(entry.params.truth_tables).toEqual(['0xa', '0xa', '0xa']);
    expect(entry.params.indices).toEqual([
      [0, 1, 2],
      [0, 1, 2],
    ]);
    // 非draft: buildでランダム初期化されない(恒等を維持)
    expect(entry.train_info.initialized).not.toBe(false);
  });

  it('接続後にshapeInference診断ゼロ', () => {
    const tpl = residualBlock(8, 1, 'xor', makeAlloc());
    expect(diagnosticsOf(templateToGraph('t', tpl, 8, 8))).toEqual([]);
  });

  it('単体IR化してgraph→IR→graphの往復が固定点になる', () => {
    const tpl = residualBlock(4, 1, 'xor', makeAlloc());
    const ir = templateToGraph('t', tpl, 4, 4);
    const flow = irToFlow(ir);
    const out = flowToIr(flow.nodes, flow.edges, ir);
    expect(out.edges).toEqual(ir.edges);
    expect(
      out.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })),
    ).toEqual(ir.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })));
    const flow2 = irToFlow(out);
    expect(flowToIr(flow2.nodes, flow2.edges, out)).toEqual(out);
  });
});

describe('classifierHead', () => {
  it('LutLayer+GroupSumでout_widthはkの倍数(割り切り保証)', () => {
    const tpl = classifierHead(60, 10, 1.0, makeAlloc());
    const [lut, gs] = tpl.nodes;
    expect(lut.type).toBe('lut_layer');
    expect(gs.type).toBe('group_sum');
    expect((lut.params.out_width as number) % 10).toBe(0);
    expect(gs.params.in_width).toBe(lut.params.out_width);
    expect(gs.params.k).toBe(10);
    expect(gs.params.tau).toBe(1.0);
    expect(tpl.entryId).toBe(lut.id);
    expect(tpl.exitId).toBe(gs.id);
  });

  it('接続後にshapeInference診断ゼロ', () => {
    const tpl = classifierHead(60, 10, 1.0, makeAlloc());
    expect(diagnosticsOf(templateToGraph('t', tpl, 60, 10))).toEqual([]);
  });
});

describe('generateBlock / blockIoWidths', () => {
  it('kindごとの生成器へディスパッチし入出力幅が一致する', () => {
    const specs: BlockParams[] = [
      { kind: 'mlp', params: { inWidth: 8, hidden: [16], outWidth: 4 } },
      { kind: 'residual', params: { width: 6, innerLayers: 1, merge: 'or' } },
      { kind: 'classifier', params: { inWidth: 30, k: 5, tau: 0.5 } },
    ];
    for (const spec of specs) {
      const tpl = generateBlock(spec, makeAlloc());
      const io = blockIoWidths(spec);
      const ir = templateToGraph('t', tpl, io.in, io.out);
      expect(diagnosticsOf(ir)).toEqual([]);
    }
  });

  it('採番コールバックが全ノードで共有される(id衝突なし)', () => {
    const alloc = makeAlloc(1);
    const a = generateBlock(
      { kind: 'residual', params: { width: 2, innerLayers: 1, merge: 'xor' } },
      alloc,
    );
    const b = generateBlock(
      { kind: 'mlp', params: { inWidth: 2, hidden: [], outWidth: 2 } },
      alloc,
    );
    const ids = [...a.nodes, ...b.nodes].map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
