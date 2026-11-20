// Unitデータモデル・展開・再展開の純関数テスト。
// 引数解決、定義バリデーション、2インスタンス幅整合、再展開id安定、
// edited警告、metadata roundtrip を固定する。

import { describe, expect, it } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import {
  expandUnit,
  reexpandUnit,
  resolveArgs,
  unitInstancesOf,
  unitTagOf,
  unitsOf,
  validateUnit,
  withUnitInstances,
  withUnits,
  type EditorUnit,
  type UnitInstance,
} from './unitTypes';

/** 最小Unit: lut_layer 1個 + 引数width(out_widthに代入)。unit-in/out付き。 */
function widthUnit(): EditorUnit {
  return {
    id: 'unit:lut',
    name: '幅可変LUT',
    args: [{ name: 'width', type: 'int', default: 8, min: 1 }],
    template: {
      nodes: [
        { id: 'in', type: 'unit-in', params: {} },
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
          train_info: { parameterization: 'softmax' },
          meta: { editor: { position: { x: 0, y: 0 } } },
        },
        { id: 'out', type: 'unit-out', params: {} },
      ],
      edges: [
        { src: 'in', dst: 'lut0' },
        { src: 'lut0', dst: 'out' },
      ],
    },
  };
}

function emptyIr(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 4, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 8, shape: null, encoding: null, semantics: null }],
    nodes: [],
    edges: [],
  };
}

describe('resolveArgs', () => {
  it('{$arg}をネスト構造ごと置換する', () => {
    const params = { out_width: { $arg: 'w' }, list: [{ $arg: 'w' }, 2], keep: 'x' };
    expect(resolveArgs(params, { w: 16 })).toEqual({
      out_width: 16,
      list: [16, 2],
      keep: 'x',
    });
  });
  it('未指定引数で例外', () => {
    expect(() => resolveArgs({ a: { $arg: 'missing' } }, {})).toThrow();
  });
});

describe('validateUnit', () => {
  it('妥当な定義はエラーなし', () => {
    expect(validateUnit(widthUnit())).toEqual([]);
  });
  it('ネストUnit(meta.editor.unitタグ付き実ノード)を検出', () => {
    const u = widthUnit();
    u.template.nodes[1].meta = { editor: { unit: { unitId: 'x', instanceId: 'y' } } };
    expect(validateUnit(u).some((e) => e.includes('ネスト'))).toBe(true);
  });
  it('enum引数のoptions欠落を検出', () => {
    const u = widthUnit();
    u.args = [{ name: 'mode', type: 'enum', default: 'a' }];
    expect(validateUnit(u).some((e) => e.includes('options'))).toBe(true);
  });
});

describe('expandUnit', () => {
  it('擬似IOが消え、由来タグ付き実ノードのみ生成される', () => {
    const exp = expandUnit(widthUnit(), { width: 8 }, 'uinst:1');
    expect(exp.nodes).toHaveLength(1);
    expect(exp.nodes[0].type).toBe('lut_layer');
    expect(unitTagOf(exp.nodes[0])).toEqual({ unitId: 'unit:lut', instanceId: 'uinst:1' });
    expect(exp.entryIds).toEqual([exp.nodes[0].id]);
    expect(exp.exitIds).toEqual([exp.nodes[0].id]);
    // 擬似IO由来のエッジは境界化して内部エッジには残らない
    expect(exp.edges).toEqual([]);
  });

  it('引数違い2インスタンスでノード数一致・out_widthが各引数で解決される', () => {
    const a = expandUnit(widthUnit(), { width: 8 }, 'uinst:1');
    const b = expandUnit(widthUnit(), { width: 16 }, 'uinst:2');
    expect(a.nodes.length).toBe(b.nodes.length);
    expect(a.nodes[0].params.out_width).toBe(8);
    expect(b.nodes[0].params.out_width).toBe(16);
    // idはinstanceIdで分離され衝突しない
    expect(a.nodes[0].id).not.toBe(b.nodes[0].id);
  });

  it('同一instanceIdの再展開でノードidが安定(決定的採番)', () => {
    const a = expandUnit(widthUnit(), { width: 8 }, 'uinst:1');
    const b = expandUnit(widthUnit(), { width: 32 }, 'uinst:1');
    expect(a.nodes[0].id).toBe(b.nodes[0].id);
  });
});

describe('reexpandUnit', () => {
  /** 2インスタンスを展開済みのIRを組み立てる。 */
  function irWith2Instances(): IRGraphJson {
    let ir = withUnits(emptyIr(), [widthUnit()]);
    const e1 = expandUnit(widthUnit(), { width: 8 }, 'uinst:1');
    const e2 = expandUnit(widthUnit(), { width: 16 }, 'uinst:2');
    const instances: UnitInstance[] = [
      {
        instanceId: 'uinst:1',
        unitId: 'unit:lut',
        argValues: { width: 8 },
        expandedNodeIds: e1.expandedNodeIds,
        edited: false,
        position: { x: 100, y: 0 },
      },
      {
        instanceId: 'uinst:2',
        unitId: 'unit:lut',
        argValues: { width: 16 },
        expandedNodeIds: e2.expandedNodeIds,
        edited: true, // 手編集済み
        position: { x: 400, y: 0 },
      },
    ];
    ir = { ...ir, nodes: [...e1.nodes, ...e2.nodes], edges: [...e1.edges, ...e2.edges] };
    return withUnitInstances(ir, instances);
  }

  it('定義変更で非editedのみ再展開、editedはスキップして警告', () => {
    const ir = irWith2Instances();
    // 定義変更: 引数widthのデフォルト変更(実質は展開時のargValuesを使う)
    const res = reexpandUnit(ir, widthUnit());
    expect(res.reexpandedInstanceIds).toEqual(['uinst:1']);
    expect(res.skipped.map((s) => s.instanceId)).toEqual(['uinst:2']);
  });

  it('再展開後もノードidが安定し外部エッジが生存する', () => {
    let ir = irWith2Instances();
    // 外部エッジ: 入力ポート x → uinst:1のノード、uinst:1のノード → 出力 y
    const inst1NodeId = unitInstancesOf(ir)[0].expandedNodeIds[0];
    ir = {
      ...ir,
      edges: [
        ...ir.edges,
        { src: 'x', dst: inst1NodeId },
        { src: inst1NodeId, dst: 'y' },
      ],
    };
    const res = reexpandUnit(ir, widthUnit());
    // 決定的採番なので同じidが再生成され、外部エッジは落ちない
    expect(res.ir.nodes.some((n) => n.id === inst1NodeId)).toBe(true);
    expect(res.droppedEdges).toEqual([]);
    expect(res.ir.edges.some((e) => e.src === 'x' && e.dst === inst1NodeId)).toBe(true);
    expect(res.ir.edges.some((e) => e.src === inst1NodeId && e.dst === 'y')).toBe(true);
  });
});

describe('metadata roundtrip', () => {
  it('units/unitInstancesがsave→load相当で保持される', () => {
    const unit = widthUnit();
    const exp = expandUnit(unit, { width: 8 }, 'uinst:1');
    const inst: UnitInstance = {
      instanceId: 'uinst:1',
      unitId: unit.id,
      argValues: { width: 8 },
      expandedNodeIds: exp.expandedNodeIds,
      edited: false,
    };
    let ir = withUnits(emptyIr(), [unit]);
    ir = withUnitInstances(ir, [inst]);
    // JSON往復(save→load)
    const roundtrip = JSON.parse(JSON.stringify(ir)) as IRGraphJson;
    expect(unitsOf(roundtrip)).toEqual([unit]);
    expect(unitInstancesOf(roundtrip)).toEqual([inst]);
  });
});
