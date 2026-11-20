// composeDetailGraph(詳細ゲートビューの表示専用合成レイヤ)の単体テスト。
// 過去の「yしか表示されない」バグ(合成ノードのwidth/height欠落)の回帰テストを含む。

import { describe, expect, it } from 'vitest';
import { irToFlow } from '../ir/irToGraph';
import type { IRGraphJson, LutLayerParams } from '../ir/schema';
import { SAMPLE_PARITY3, SAMPLE_XOR } from '../samples/sampleIr';
import {
  BITS_HEADER,
  BITS_ROW,
  BITS_W,
  GATE_CARD_H,
  GATE_CARD_W,
} from './detailConfig';
import {
  composeDetailGraph,
  createDetailComposeCache,
  gateGridLayout,
  inputBitsSize,
  isLayerExpandable,
  isLayerExpanded,
} from './irToDetailGraph';

function lutParams(
  inWidth: number,
  outWidth: number,
  indices?: number[][],
  tts?: string[],
): LutLayerParams {
  return {
    in_width: inWidth,
    out_width: outWidth,
    n_inputs: 2,
    indices: indices ?? [
      new Array<number>(outWidth).fill(0),
      new Array<number>(outWidth).fill(1),
    ],
    truth_tables: tts ?? new Array<string>(outWidth).fill('0x6'),
  };
}

/** x → lut×n → y の直列IRを生成する */
function chainIr(inWidth: number, layers: LutLayerParams[]): IRGraphJson {
  const nodes = layers.map((p, i) => ({
    id: `n${i + 1}`,
    type: 'lut_layer' as const,
    params: p as unknown as Record<string, unknown>,
    train_info: { parameterization: 'softmax' },
    meta: {},
  }));
  const ids = nodes.map((n) => n.id);
  return {
    format: 'magicdelta-ir',
    version: '0.1.0',
    metadata: {},
    inputs: [{ id: 'x', width: inWidth, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 1, shape: null, encoding: null, semantics: null }],
    nodes,
    edges: [
      { src: 'x', dst: ids[0] },
      ...ids.slice(1).map((dst, i) => ({ src: ids[i], dst })),
      { src: ids[ids.length - 1], dst: 'y' },
    ],
  };
}

function compose(ir: IRGraphJson, overrides: Record<string, boolean> = {}) {
  const { nodes, edges } = irToFlow(ir);
  return composeDetailGraph(nodes, edges, ir, overrides);
}

describe('gateGridLayout', () => {
  it('16個までは1列(列優先)', () => {
    expect(gateGridLayout(1)).toMatchObject({ cols: 1, rows: 1 });
    expect(gateGridLayout(16)).toMatchObject({ cols: 1, rows: 16 });
  });

  it('17個で2列、上限は4列', () => {
    expect(gateGridLayout(17)).toMatchObject({ cols: 2, rows: 9 });
    expect(gateGridLayout(60)).toMatchObject({ cols: 4, rows: 15 });
    expect(gateGridLayout(200).cols).toBe(4);
  });
});

describe('isLayerExpandable / isLayerExpanded', () => {
  it('out_width<=128が手動展開の上限', () => {
    expect(isLayerExpandable(lutParams(4, 128))).toBe(true);
    expect(isLayerExpandable(lutParams(4, 129))).toBe(false);
  });

  it('自動展開はin/out両方が64以下の場合のみ', () => {
    expect(isLayerExpanded(lutParams(64, 64), undefined)).toBe(true);
    expect(isLayerExpanded(lutParams(65, 64), undefined)).toBe(false);
    expect(isLayerExpanded(lutParams(64, 65), undefined)).toBe(false);
  });

  it('手動overrideは自動判定より優先されるが展開可能上限は超えられない', () => {
    expect(isLayerExpanded(lutParams(100, 100), true)).toBe(true);
    expect(isLayerExpanded(lutParams(4, 4), false)).toBe(false);
    expect(isLayerExpanded(lutParams(4, 200), true)).toBe(false);
  });
});

describe('composeDetailGraph: XORサンプル', () => {
  const result = compose(SAMPLE_XOR);

  it('inputBits+lutLayerGroup+gate+yの4ノード構成になる', () => {
    expect(result.nodes.map((n) => n.type)).toEqual([
      'inputBits',
      'lutLayerGroup',
      'gate',
      'generic',
    ]);
    expect(result.nodes.map((n) => n.id)).toEqual([
      'x',
      'n1_lutlayer',
      'n1_lutlayer::g0',
      'y',
    ]);
  });

  it('全合成ノードにwidth/heightが付与される(「yしか表示されない」回帰)', () => {
    for (const n of result.nodes) {
      if (n.type === 'generic') continue; // 非合成ノードは実測に任せる
      expect(n.width, `${n.id}のwidth`).toBeGreaterThan(0);
      expect(n.height, `${n.id}のheight`).toBeGreaterThan(0);
    }
  });

  it('inputBitsノードはビット数に応じた実寸を持つ', () => {
    const bits = result.nodes.find((n) => n.type === 'inputBits')!;
    expect(bits.width).toBe(BITS_W);
    expect(bits.height).toBe(BITS_HEADER + 2 * BITS_ROW + 8);
    expect(inputBitsSize(2)).toEqual({ width: bits.width, height: bits.height });
  });

  it('ゲートカードは実寸定数どおり、グループ内に親子付けされる', () => {
    const gate = result.nodes.find((n) => n.type === 'gate')!;
    expect(gate.width).toBe(GATE_CARD_W);
    expect(gate.height).toBe(GATE_CARD_H);
    expect(gate.parentId).toBe('n1_lutlayer');
    expect(gate.data).toMatchObject({
      layerId: 'n1_lutlayer',
      neuron: 0,
      tt: '0x6',
      slotSources: [0, 1],
      showSlotLabels: false, // 上流(x)がビット展開済み
    });
  });

  it('グループ実寸がgroupSizesへ記録される', () => {
    const grid = gateGridLayout(1);
    expect(result.groupSizes.get('n1_lutlayer')).toEqual({
      width: grid.width,
      height: grid.height,
    });
    const group = result.nodes.find((n) => n.type === 'lutLayerGroup')!;
    expect(group.width).toBe(grid.width);
    expect(group.height).toBe(grid.height);
  });

  it('詳細エッジ(d:プレフィックス)がビット→スロットのhandle規約で張られる', () => {
    const detail = result.edges.filter((e) => e.id.startsWith('d:'));
    expect(detail).toHaveLength(2);
    expect(detail[0]).toMatchObject({
      id: 'd:n1_lutlayer:0:0',
      source: 'x',
      sourceHandle: 'b0',
      target: 'n1_lutlayer::g0',
      targetHandle: 's0',
    });
    expect(detail[1]).toMatchObject({
      id: 'd:n1_lutlayer:0:1',
      source: 'x',
      sourceHandle: 'b1',
      targetHandle: 's1',
    });
  });

  it('置き換え済みの集約エッジは除去され、y行きは残る', () => {
    const aggregate = result.edges.filter((e) => !e.id.startsWith('d:'));
    expect(aggregate.map((e) => e.id)).toEqual(['n1_lutlayer->y']);
    expect(result.overflowIds).toEqual([]);
  });
});

describe('composeDetailGraph: パリティ3(層間のゲート→ゲートエッジ)', () => {
  const result = compose(SAMPLE_PARITY3);

  it('両層が展開され、n2のゲートはn1のゲート出力(out handle)から受ける', () => {
    const toN2 = result.edges.filter((e) => e.id.startsWith('d:n2_lutlayer:'));
    expect(toN2).toHaveLength(2);
    expect(toN2[0]).toMatchObject({
      source: 'n1_lutlayer::g0',
      sourceHandle: 'out',
      target: 'n2_lutlayer::g0',
      targetHandle: 's0',
    });
    expect(toN2[1]).toMatchObject({
      source: 'n1_lutlayer::g1',
      sourceHandle: 'out',
      targetHandle: 's1',
    });
  });

  it('n1のゲートはxのビット(indices由来のb{src} handle)から受ける', () => {
    const toN1 = result.edges.filter((e) => e.id.startsWith('d:n1_lutlayer:'));
    // 2ゲート×2スロット。g1はindices[0][1]=2, indices[1][1]=2 → b2,b2
    expect(toN1.map((e) => e.sourceHandle)).toEqual(['b0', 'b1', 'b2', 'b2']);
    expect(toN1.every((e) => e.source === 'x')).toBe(true);
  });

  it('ゲート子ノード数はout_widthの合計と一致する', () => {
    expect(result.nodes.filter((n) => n.type === 'gate')).toHaveLength(3);
  });
});

describe('composeDetailGraph: しきい値と上限', () => {
  it('out_width=200の層は展開されない(手動overrideでも不可)', () => {
    const ir = chainIr(4, [lutParams(4, 200)]);
    const overridesCases: Record<string, boolean>[] = [{}, { n1: true }];
    for (const overrides of overridesCases) {
      const { nodes, edges } = irToFlow(ir);
      const result = composeDetailGraph(nodes, edges, ir, overrides);
      // 展開ゼロならstoreのノード/エッジをそのまま返す
      expect(result.nodes).toBe(nodes);
      expect(result.edges).toBe(edges);
      expect(result.nodes.some((n) => n.id.includes('::'))).toBe(false);
    }
  });

  it('総ノード数上限(1000)を超える層は展開を拒否しoverflowIdsへ記録する', () => {
    // 64幅×16層: 15層目まで(18+64×15=978)は展開、16層目(+64=1042)で上限超過
    const ir = chainIr(64, Array.from({ length: 16 }, () => lutParams(64, 64)));
    const result = compose(ir);
    expect(result.overflowIds).toEqual(['n16']);
    expect(result.nodes.some((n) => n.id === 'n15::g63')).toBe(true);
    expect(result.nodes.some((n) => n.id.startsWith('n16::'))).toBe(false);
    expect(result.nodes.find((n) => n.id === 'n16')?.type).toBe('lutLayer');
  });

  it('幅がINPUT_BITS_MAX超の入力ポートはビット展開されずスロットラベル表示になる', () => {
    // in=100はビット展開対象外、out=2は手動展開可能
    const ir = chainIr(100, [lutParams(100, 2, [[5, 7], [9, 11]], ['0x6', '0x8'])]);
    const { nodes, edges } = irToFlow(ir);
    const result = composeDetailGraph(nodes, edges, ir, { n1: true });
    expect(result.nodes.find((n) => n.id === 'x')?.type).toBe('generic');
    // 集約エッジx→n1は残り、d:エッジは張られない
    expect(result.edges.filter((e) => e.id.startsWith('d:'))).toEqual([]);
    expect(result.edges.some((e) => e.id === 'x->n1')).toBe(true);
    const g0 = result.nodes.find((n) => n.id === 'n1::g0')!;
    expect(g0.data).toMatchObject({ slotSources: [5, 9], showSlotLabels: true });
  });

  it('ir=nullならstoreのノード/エッジをそのまま返す', () => {
    const { nodes, edges } = irToFlow(SAMPLE_XOR);
    const result = composeDetailGraph(nodes, edges, null, {});
    expect(result.nodes).toBe(nodes);
    expect(result.edges).toBe(edges);
    expect(result.overflowIds).toEqual([]);
  });
});

describe('composeDetailGraph: 層別キャッシュ', () => {
  const ir = chainIr(4, [lutParams(4, 4), lutParams(4, 4), lutParams(4, 4)]);

  it('非構造変化(位置移動)の再合成では子ゲート/詳細エッジのオブジェクトを再利用する', () => {
    const cache = createDetailComposeCache();
    const { nodes, edges } = irToFlow(ir);
    const r1 = composeDetailGraph(nodes, edges, ir, {}, cache);
    // ドラッグを模擬: ノード配列/オブジェクトは新規、ir(params参照)は不変
    const moved = nodes.map((n) => ({
      ...n,
      position: { x: n.position.x + 50, y: n.position.y },
    }));
    const r2 = composeDetailGraph(moved, edges, ir, {}, cache);
    const g1 = r1.nodes.find((n) => n.id === 'n2::g1');
    const g2 = r2.nodes.find((n) => n.id === 'n2::g1');
    expect(g2).toBe(g1);
    const e1 = r1.edges.find((e) => e.id === 'd:n2:1:0');
    const e2 = r2.edges.find((e) => e.id === 'd:n2:1:0');
    expect(e2).toBe(e1);
  });

  it('params差し替え(tt編集相当)は当該層のみ再合成し、他層は再利用する', () => {
    const cache = createDetailComposeCache();
    const { nodes, edges } = irToFlow(ir);
    const r1 = composeDetailGraph(nodes, edges, ir, {}, cache);
    // updateLutParams相当: n2のparamsだけ新規オブジェクトへ差し替える
    const ir2: IRGraphJson = {
      ...ir,
      nodes: ir.nodes.map((n) =>
        n.id === 'n2'
          ? {
              ...n,
              params: {
                ...n.params,
                truth_tables: ['0x8', '0x8', '0x8', '0x8'],
              },
            }
          : n,
      ),
    };
    const r2 = composeDetailGraph(nodes, edges, ir2, {}, cache);
    const before = r1.nodes.find((n) => n.id === 'n2::g0')!;
    const after = r2.nodes.find((n) => n.id === 'n2::g0')!;
    expect(after).not.toBe(before);
    expect((after.data as { tt: string }).tt).toBe('0x8');
    // 依存が不変の層は同一オブジェクトのまま
    expect(r2.nodes.find((n) => n.id === 'n1::g0')).toBe(
      r1.nodes.find((n) => n.id === 'n1::g0'),
    );
    expect(r2.nodes.find((n) => n.id === 'n3::g0')).toBe(
      r1.nodes.find((n) => n.id === 'n3::g0'),
    );
  });

  it('上流の展開状態が変わった層は再合成される(showSlotLabels/エッジ源が変わるため)', () => {
    const cache = createDetailComposeCache();
    const { nodes, edges } = irToFlow(ir);
    const r1 = composeDetailGraph(nodes, edges, ir, {}, cache);
    // n1を畳む → n2の上流が詳細化済みでなくなる
    const r2 = composeDetailGraph(nodes, edges, ir, { n1: false }, cache);
    const g2 = r2.nodes.find((n) => n.id === 'n2::g0')!;
    expect(g2).not.toBe(r1.nodes.find((n) => n.id === 'n2::g0'));
    expect((g2.data as { showSlotLabels: boolean }).showSlotLabels).toBe(true);
    // n3はn2(展開のまま)が上流なので再利用される
    expect(r2.nodes.find((n) => n.id === 'n3::g0')).toBe(
      r1.nodes.find((n) => n.id === 'n3::g0'),
    );
    // 畳んだ層のキャッシュは破棄される(再展開時は作り直し)
    const r3 = composeDetailGraph(nodes, edges, ir, {}, cache);
    expect(r3.nodes.find((n) => n.id === 'n1::g0')).not.toBe(
      r1.nodes.find((n) => n.id === 'n1::g0'),
    );
  });

  it('キャッシュ再利用時も合成結果はキャッシュなしと同一内容になる', () => {
    const cache = createDetailComposeCache();
    const { nodes, edges } = irToFlow(ir);
    composeDetailGraph(nodes, edges, ir, {}, cache);
    const cached = composeDetailGraph(nodes, edges, ir, {}, cache);
    const plain = composeDetailGraph(nodes, edges, ir, {});
    expect(cached.nodes).toEqual(plain.nodes);
    expect(cached.edges).toEqual(plain.edges);
    expect(cached.overflowIds).toEqual(plain.overflowIds);
  });
});

describe('composeDetailGraph: 1000ゲート規模の合成性能', () => {
  // 64幅×15層=960ゲート、総ノード数 17+960=977 ≤ DETAIL_TOTAL_MAX(1000)
  const LAYERS = 15;
  const GATES = 64 * LAYERS;
  const ir = chainIr(64, Array.from({ length: LAYERS }, () => lutParams(64, 64)));

  it('960ゲートの初回合成が200ms未満で完了する(k=1000相当デモ)', () => {
    const { nodes, edges } = irToFlow(ir);
    const t0 = performance.now();
    const result = composeDetailGraph(nodes, edges, ir, {});
    const elapsed = performance.now() - t0;
    expect(result.overflowIds).toEqual([]);
    expect(result.nodes.filter((n) => n.type === 'gate')).toHaveLength(GATES);
    expect(elapsed).toBeLessThan(200);
  });

  it('大規模でも全合成ノードにwidth/heightが付与される(「yしか表示されない」回帰)', () => {
    const { nodes, edges } = irToFlow(ir);
    const result = composeDetailGraph(nodes, edges, ir, {});
    for (const n of result.nodes) {
      if (n.type === 'generic') continue;
      expect(n.width, `${n.id}のwidth`).toBeGreaterThan(0);
      expect(n.height, `${n.id}のheight`).toBeGreaterThan(0);
    }
  });

  it('キャッシュありの再合成(ドラッグ相当)も200ms未満で全ゲートを再利用する', () => {
    const cache = createDetailComposeCache();
    const { nodes, edges } = irToFlow(ir);
    const r1 = composeDetailGraph(nodes, edges, ir, {}, cache);
    const moved = nodes.map((n) => ({
      ...n,
      position: { x: n.position.x + 10, y: n.position.y },
    }));
    const t0 = performance.now();
    const r2 = composeDetailGraph(moved, edges, ir, {}, cache);
    const elapsed = performance.now() - t0;
    const gates1 = r1.nodes.filter((n) => n.type === 'gate');
    const gates2 = r2.nodes.filter((n) => n.type === 'gate');
    expect(gates2).toHaveLength(GATES);
    for (let i = 0; i < gates1.length; i++) {
      expect(gates2[i]).toBe(gates1[i]);
    }
    expect(elapsed).toBeLessThan(200);
  });
});
