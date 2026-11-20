// unitStore + graphStore の Unit 統合テスト。
// Unit定義の保存roundtrip、引数違い2インスタンス展開(幅整合)、
// 定義変更→再展開(id安定)、手編集インスタンスの edited 警告 を固定する。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import { unitInstancesOf, unitsOf, type EditorUnit } from '../blocks/unitTypes';
import { useGraphStore } from './graphStore';
import { useUnitStore } from './unitStore';

vi.mock('../api/rest', () => ({
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
}));

function baseIr(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { editor: {} },
    inputs: [{ id: 'x', width: 4, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 8, shape: null, encoding: null, semantics: null }],
    nodes: [],
    edges: [],
  };
}

/** 引数widthを out_width に代入する最小Unit。 */
function widthUnit(): EditorUnit {
  return {
    id: 'unit:lut',
    name: '幅可変LUT',
    args: [{ name: 'width', type: 'int', default: 8, min: 1 }],
    template: {
      nodes: [
        { id: 'in', type: 'unit-in', params: {}, meta: { editor: { position: { x: 0, y: 0 } } } },
        {
          id: 'lut0',
          type: 'lut_layer',
          params: {
            in_width: 4,
            out_width: { $arg: 'width' },
            n_inputs: 2,
            indices: [[0, 1], [2, 3]],
            truth_tables: ['0x6'],
          },
          train_info: { parameterization: 'softmax', connections: 'random', selection: 'soft' },
          meta: { editor: { position: { x: 220, y: 0 } } },
        },
        { id: 'out', type: 'unit-out', params: {}, meta: { editor: { position: { x: 440, y: 0 } } } },
      ],
      edges: [
        { src: 'in', dst: 'lut0' },
        { src: 'lut0', dst: 'out' },
      ],
    },
  };
}

beforeEach(() => {
  useGraphStore.getState().setIR(baseIr());
  useUnitStore.setState({ units: [], editing: null });
  useUnitStore.getState().loadFromActiveIr();
});

describe('unitStore: 定義の保存roundtrip', () => {
  it('saveEditingでmetadata.editor.unitsへ保存され再読込で復元', () => {
    useUnitStore.setState({ editing: widthUnit() });
    const { errors } = useUnitStore.getState().saveEditing();
    expect(errors).toEqual([]);

    const saved = useGraphStore.getState().ir!;
    expect(unitsOf(saved)).toHaveLength(1);
    expect(unitsOf(saved)[0].id).toBe('unit:lut');

    // roundtrip: 同一IRを再ロードしても定義が保持され、storeへ取り込める
    useGraphStore.getState().setIR(JSON.parse(JSON.stringify(saved)));
    useUnitStore.getState().loadFromActiveIr();
    expect(useUnitStore.getState().units).toHaveLength(1);
  });

  it('不正な定義(ネスト)はエラーを返し保存されない', () => {
    const u = widthUnit();
    u.template.nodes[1].meta = { editor: { unit: { unitId: 'a', instanceId: 'b' } } };
    useUnitStore.setState({ editing: u });
    const { errors } = useUnitStore.getState().saveEditing();
    expect(errors.length).toBeGreaterThan(0);
    expect(unitsOf(useGraphStore.getState().ir!)).toEqual([]);
  });
});

describe('graphStore.addUnitInstance: 引数違い2インスタンス', () => {
  it('ノード数一致・out_widthが各引数で解決される', () => {
    const g = useGraphStore.getState();
    g.addUnitInstance(widthUnit(), { width: 8 }, { x: 0, y: 0 });
    g.addUnitInstance(widthUnit(), { width: 16 }, { x: 400, y: 0 });

    const ir = useGraphStore.getState().ir!;
    const instances = unitInstancesOf(ir);
    expect(instances).toHaveLength(2);
    // 各インスタンス1ノード → 計2
    expect(ir.nodes).toHaveLength(2);

    const n1 = ir.nodes.find((n) => instances[0].expandedNodeIds.includes(n.id))!;
    const n2 = ir.nodes.find((n) => instances[1].expandedNodeIds.includes(n.id))!;
    expect(n1.params.out_width).toBe(8);
    expect(n2.params.out_width).toBe(16);
  });
});

describe('定義変更→再展開', () => {
  it('非editedは再展開しidが安定、editedはスキップして警告', () => {
    // まずUnit定義を保存(metadata.editor.unitsに載る=既存扱いになる)
    useUnitStore.setState({ editing: widthUnit() });
    useUnitStore.getState().saveEditing();

    const g = useGraphStore.getState();
    g.addUnitInstance(widthUnit(), { width: 8 }, { x: 0, y: 0 });
    g.addUnitInstance(widthUnit(), { width: 8 }, { x: 400, y: 0 });

    const before = unitInstancesOf(useGraphStore.getState().ir!);
    const inst2Id = before[1].expandedNodeIds[0];

    // インスタンス2のノードをデータ編集 → edited=trueになる
    useGraphStore.getState().updateNodeData(inst2Id, { label: 'touched' });
    const marked = unitInstancesOf(useGraphStore.getState().ir!);
    expect(marked[1].edited).toBe(true);
    expect(marked[0].edited).toBe(false);

    // 定義を再保存(既存 → 再展開伝播)
    useUnitStore.setState({ editing: widthUnit() });
    const { reexpand } = useUnitStore.getState().saveEditing();
    expect(reexpand.reexpandedInstanceIds).toEqual([before[0].instanceId]);
    expect(reexpand.skipped.map((s) => s.instanceId)).toEqual([before[1].instanceId]);

    // 非editedのノードidは安定(決定的採番)
    const after = unitInstancesOf(useGraphStore.getState().ir!);
    expect(after[0].expandedNodeIds).toEqual(before[0].expandedNodeIds);
  });
});
