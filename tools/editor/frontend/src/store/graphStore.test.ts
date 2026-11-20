// graphStoreの多入力DAG編集テスト:
// 接続可否(多入力slot/単一入力/ポート保護)、onConnectのslot自動割当、
// addSubgraphのテンプレート展開とparams保全、draft id採番を検証する。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { residualBlock } from '../blocks/templates';
import type { IRGraphJson, IRNodeJson } from '../ir/schema';
import {
  createDraftIdAllocator,
  isValidConnectionCheck,
  useGraphStore,
} from './graphStore';

vi.mock('../api/rest', () => ({
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
}));

const pos = (x: number, y: number) => ({ editor: { position: { x, y } } });

function lut(id: string, inWidth: number, outWidth: number, x: number): IRNodeJson {
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
    meta: pos(x, 0),
  };
}

/** x(2) → n1(2→2)、a1(add width2)は未接続。全ノード座標つき(elk回避) */
function baseIr(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {
      name: 't',
      editor: {
        ports: {
          x: { position: { x: 0, y: 0 } },
          y: { position: { x: 600, y: 0 } },
        },
      },
    },
    inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
    nodes: [
      lut('n1', 2, 2, 200),
      {
        id: 'a1',
        type: 'add',
        params: { width: 2, mode: 'xor' },
        train_info: {},
        meta: pos(400, 0),
      },
    ],
    edges: [{ src: 'x', dst: 'n1' }],
  };
}

beforeEach(() => {
  useGraphStore.getState().setIR(baseIr());
});

describe('isValidConnectionCheck(多入力対応)', () => {
  it('多入力ノードへの2本目の接続を許可する(空きslot)', () => {
    useGraphStore.getState().onConnect({
      source: 'n1',
      target: 'a1',
      sourceHandle: null,
      targetHandle: 's0',
    });
    expect(isValidConnectionCheck('x', 'a1', 's1')).toBe(true);
    expect(isValidConnectionCheck('x', 'a1', null)).toBe(true); // 自動割当
  });

  it('占有済みslotへの接続は拒否する', () => {
    useGraphStore.getState().onConnect({
      source: 'n1',
      target: 'a1',
      sourceHandle: null,
      targetHandle: 's0',
    });
    expect(isValidConnectionCheck('x', 'a1', 's0')).toBe(false);
  });

  it('単一入力ノードへの2本目は拒否のまま', () => {
    expect(isValidConnectionCheck('a1', 'n1')).toBe(false);
  });

  it('入力ポートへの接続/出力ポートからの接続は拒否(ポート保護)', () => {
    expect(isValidConnectionCheck('n1', 'x')).toBe(false);
    expect(isValidConnectionCheck('y', 'a1', 's0')).toBe(false);
  });

  it('循環になる接続は拒否する', () => {
    useGraphStore.getState().onConnect({
      source: 'n1',
      target: 'a1',
      sourceHandle: null,
      targetHandle: 's0',
    });
    // a1 → n1 は n1 → a1 が既存なので循環
    expect(isValidConnectionCheck('a1', 'n1')).toBe(false);
  });
});

describe('onConnect: slot割当とIR合成', () => {
  it('targetHandle指定でdst_slotへ写像される(slot0はキー省略)', () => {
    const s = useGraphStore.getState();
    s.onConnect({ source: 'n1', target: 'a1', sourceHandle: null, targetHandle: 's0' });
    useGraphStore
      .getState()
      .onConnect({ source: 'x', target: 'a1', sourceHandle: null, targetHandle: 's1' });
    const ir = useGraphStore.getState().ir!;
    const e0 = ir.edges.find((e) => e.src === 'n1' && e.dst === 'a1')!;
    const e1 = ir.edges.find((e) => e.src === 'x' && e.dst === 'a1')!;
    expect('dst_slot' in e0).toBe(false);
    expect(e1.dst_slot).toBe(1);
  });

  it('ハンドル未指定なら空きslotへ自動割当する', () => {
    useGraphStore
      .getState()
      .onConnect({ source: 'n1', target: 'a1', sourceHandle: null, targetHandle: null });
    useGraphStore
      .getState()
      .onConnect({ source: 'x', target: 'a1', sourceHandle: null, targetHandle: null });
    const edges = useGraphStore.getState().edges.filter((e) => e.target === 'a1');
    expect(edges.map((e) => e.targetHandle).sort()).toEqual(['s0', 's1']);
    // 残差エッジ(slot1)は破線スタイル
    const skip = edges.find((e) => e.targetHandle === 's1')!;
    expect(skip.style).toMatchObject({ strokeDasharray: '6 3' });
  });

  it('addの導出widthが上流(slot0)に追従し診断ゼロになる', () => {
    useGraphStore
      .getState()
      .onConnect({ source: 'n1', target: 'a1', sourceHandle: null, targetHandle: 's0' });
    useGraphStore
      .getState()
      .onConnect({ source: 'x', target: 'a1', sourceHandle: null, targetHandle: 's1' });
    useGraphStore
      .getState()
      .onConnect({ source: 'a1', target: 'y', sourceHandle: null, targetHandle: null });
    const st = useGraphStore.getState();
    const a1 = st.ir!.nodes.find((n) => n.id === 'a1')!;
    expect(a1.params.width).toBe(2);
    expect(st.diagnostics).toEqual([]);
  });
});

describe('addSubgraph', () => {
  it('テンプレートをドロップ位置へ展開しparams(恒等TT等)を保全する', () => {
    const tpl = residualBlock(2, 1, 'xor', createDraftIdAllocator());
    useGraphStore.getState().addSubgraph(tpl.nodes, tpl.edges, { x: 100, y: 300 });
    const st = useGraphStore.getState();

    // ノードがIRとflowの両方へ追加される
    for (const n of tpl.nodes) {
      expect(st.ir!.nodes.some((m) => m.id === n.id)).toBe(true);
      expect(st.nodes.some((m) => m.id === n.id)).toBe(true);
    }
    // 恒等entryのtruth_tablesが保全される(draft placeholderへ潰れない)
    const entry = st.ir!.nodes.find((n) => n.id === tpl.entryId)!;
    expect(entry.params.truth_tables).toEqual(['0xa', '0xa']);
    // 相対座標(entry=0,0)へドロップ位置がオフセットされる
    const entryPos = (entry.meta.editor as { position: { x: number; y: number } })
      .position;
    expect(entryPos).toEqual({ x: 100, y: 300 });
    // dst_slot=1のskipエッジがflow側でtargetHandle s1になる
    const skip = st.edges.find(
      (e) => e.source === tpl.entryId && e.target === tpl.exitId,
    )!;
    expect(skip.targetHandle).toBe('s1');
  });

  it('展開後にentry/exitを既存グラフへ接続すると診断ゼロ', () => {
    const tpl = residualBlock(2, 1, 'xor', createDraftIdAllocator());
    useGraphStore.getState().addSubgraph(tpl.nodes, tpl.edges, { x: 100, y: 300 });
    useGraphStore.getState().onConnect({
      source: 'n1',
      target: tpl.entryId,
      sourceHandle: null,
      targetHandle: null,
    });
    useGraphStore.getState().onConnect({
      source: tpl.exitId,
      target: 'y',
      sourceHandle: null,
      targetHandle: null,
    });
    // a1(未接続add)以外の診断が消える
    const st = useGraphStore.getState();
    const others = st.diagnostics.filter((d) => d.nodeId !== 'a1');
    expect(others).toEqual([]);
  });
});

describe('createDraftIdAllocator', () => {
  it('既存のu{seq}_idと衝突しない連番を払い出す', () => {
    useGraphStore.getState().addNodeFromPalette('concat', { x: 0, y: 200 });
    const alloc = createDraftIdAllocator();
    const id1 = alloc('lutlayer');
    const id2 = alloc('add');
    expect(id1).toBe('u2_lutlayer');
    expect(id2).toBe('u3_add');
    expect(useGraphStore.getState().nodes.some((n) => n.id === id1)).toBe(false);
  });
});
