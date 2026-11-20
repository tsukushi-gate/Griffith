// recurrentUnit(LSTM/GRU unroll生成器)の構造・tt定数・IO幅テスト。
// 等価性(scan_cellとのビット一致)は Python 側 tests/test_editor_recurrent_n7.py で担保する。
// ここでは「フロント生成物が正しい代数構造(ノード型・配線・tt定数)を持つ」ことを固定する。

import { describe, expect, it } from 'vitest';
import {
  AND_TT,
  NOT_TT,
  PASS_TT,
  RECURRENT_CELL_TYPES,
  unrollRecurrent,
  type RecurrentCellType,
} from './recurrentUnit';
import { blockIoWidths, generateBlock } from './templates';
import type { IRNodeJson } from '../ir/schema';

/** テスト用の単純なid採番(一意であればよい)。 */
function makeAlloc(): (s: string) => string {
  let n = 0;
  return (s: string) => `t${n++}_${s}`;
}

function ttOf(node: IRNodeJson): string[] {
  return node.params.truth_tables as string[];
}

function recTag(node: IRNodeJson): { cellType: string; unroll: number; step: number } | undefined {
  const editor = (node.meta?.editor ?? {}) as Record<string, unknown>;
  return editor.recurrent as { cellType: string; unroll: number; step: number } | undefined;
}

describe('unrollRecurrent 構造', () => {
  it('LSTM: entry/exit・初期状態2個(h0,c0)・由来タグを持つ', () => {
    const tpl = unrollRecurrent(
      { cellType: 'lstm', unroll: 3, inWidth: 4, hidden: 4, depth: 1 },
      makeAlloc(),
    );
    // entry は入力分配 lut_layer、exit は最終step h(lut_layer)
    const entry = tpl.nodes.find((n) => n.id === tpl.entryId)!;
    const exit = tpl.nodes.find((n) => n.id === tpl.exitId)!;
    expect(entry.type).toBe('lut_layer');
    expect(exit.type).toBe('lut_layer');

    // 初期状態 constant は LSTM で2個(h0,c0)
    const consts = tpl.nodes.filter((n) => n.type === 'constant');
    expect(consts.length).toBe(2);
    for (const c of consts) {
      expect(c.params.bits).toEqual([0, 0, 0, 0]); // hidden=4 の 0 初期化
    }

    // 全ノードに meta.editor.recurrent 由来タグ(cellType/unroll一致)
    for (const n of tpl.nodes) {
      const tag = recTag(n);
      expect(tag).toBeDefined();
      expect(tag!.cellType).toBe('lstm');
      expect(tag!.unroll).toBe(3);
    }

    // add(OR)ノードが各stepの c_t に対応して unroll 個
    const adds = tpl.nodes.filter((n) => n.type === 'add');
    expect(adds.length).toBe(3);
    for (const a of adds) expect(a.params.mode).toBe('or');
  });

  it('GRU: 初期状態は h0 のみ(constant 1個)', () => {
    const tpl = unrollRecurrent(
      { cellType: 'gru', unroll: 2, inWidth: 3, hidden: 5, depth: 1 },
      makeAlloc(),
    );
    const consts = tpl.nodes.filter((n) => n.type === 'constant');
    expect(consts.length).toBe(1);
    expect(consts[0].params.bits).toEqual([0, 0, 0, 0, 0]);
    const adds = tpl.nodes.filter((n) => n.type === 'add'); // h_t=mux が unroll 個
    expect(adds.length).toBe(2);
  });

  it('状態代数の tt 定数が仕様どおり(AND=0x8 / NOT=0x1 / PASS=0xa)', () => {
    const tpl = unrollRecurrent(
      { cellType: 'lstm', unroll: 1, inWidth: 2, hidden: 2, depth: 1 },
      makeAlloc(),
    );
    // AND固定lut(tt全て 0x8): LSTM unroll=1 では ic/t1/t2/h_t の4個
    const andLuts = tpl.nodes.filter(
      (n) => n.type === 'lut_layer' && ttOf(n).every((t) => t === AND_TT),
    );
    expect(andLuts.length).toBe(4);
    // NOT lut(W→W, indices両slot同一)を少なくとも1つ含む
    const notLuts = tpl.nodes.filter(
      (n) => n.type === 'lut_layer' && ttOf(n).every((t) => t === NOT_TT),
    );
    expect(notLuts.length).toBe(1); // notf 1個(LSTM, unroll=1)
    // slice/entry は PASS_TT
    const passLuts = tpl.nodes.filter(
      (n) => n.type === 'lut_layer' && ttOf(n).every((t) => t === PASS_TT),
    );
    expect(passLuts.length).toBeGreaterThanOrEqual(2); // entry + x_t スライス
  });

  it('ゲートは draft(initialized:false)で build時学習可能', () => {
    const tpl = unrollRecurrent(
      { cellType: 'lstm', unroll: 1, inWidth: 2, hidden: 3, depth: 2 },
      makeAlloc(),
    );
    // depth=2 のゲートが4種(f/i/o/c̃)×2段 = 8個の draft lut
    const drafts = tpl.nodes.filter(
      (n) => n.type === 'lut_layer' && n.train_info?.initialized === false,
    );
    expect(drafts.length).toBe(8);
  });
});

import { inferShapes } from '../ir/shapeInference';

describe('生成物の形状整合(inferShapes)', () => {
  it('x→entry→…→y に接続した standalone グラフは形状診断エラーゼロ', () => {
    const tpl = unrollRecurrent(
      { cellType: 'lstm', unroll: 3, inWidth: 4, hidden: 4, depth: 1 },
      makeAlloc(),
    );
    const total = 3 * 4;
    const inputs = [{ id: 'x', width: total, shape: null, encoding: null, semantics: null }];
    const edges = [
      { src: 'x', dst: tpl.entryId },
      ...tpl.edges,
      { src: tpl.exitId, dst: 'y' },
    ];
    const { diagnostics } = inferShapes(tpl.nodes, edges, inputs);
    const errors = diagnostics.filter((d) => d.level === 'error');
    expect(errors).toEqual([]);
  });

  it('GRU も接続時に形状診断エラーゼロ', () => {
    const tpl = unrollRecurrent(
      { cellType: 'gru', unroll: 3, inWidth: 2, hidden: 5, depth: 2 },
      makeAlloc(),
    );
    const total = 3 * 2;
    const inputs = [{ id: 'x', width: total, shape: null, encoding: null, semantics: null }];
    const edges = [
      { src: 'x', dst: tpl.entryId },
      ...tpl.edges,
      { src: tpl.exitId, dst: 'y' },
    ];
    const errors = inferShapes(tpl.nodes, edges, inputs).diagnostics.filter(
      (d) => d.level === 'error',
    );
    expect(errors).toEqual([]);
  });
});

describe('generateBlock / blockIoWidths 統合', () => {
  it('recurrent kind が generateBlock でディスパッチされる', () => {
    const tpl = generateBlock(
      { kind: 'recurrent', params: { cellType: 'lstm', unroll: 2, inWidth: 4, hidden: 4, depth: 1 } },
      makeAlloc(),
    );
    expect(tpl.nodes.length).toBeGreaterThan(0);
    expect(tpl.entryId).toBeTruthy();
    expect(tpl.exitId).toBeTruthy();
  });

  it('IO幅: in=unroll*inWidth / out=hidden', () => {
    const io = blockIoWidths({
      kind: 'recurrent',
      params: { cellType: 'gru', unroll: 5, inWidth: 3, hidden: 8, depth: 1 },
    });
    expect(io).toEqual({ in: 15, out: 8 });
  });

  it('全セル種別が展開でき、ノードidが一意', () => {
    for (const cellType of RECURRENT_CELL_TYPES as RecurrentCellType[]) {
      const tpl = unrollRecurrent(
        { cellType, unroll: 4, inWidth: 4, hidden: 4, depth: 1 },
        makeAlloc(),
      );
      const ids = tpl.nodes.map((n) => n.id);
      expect(new Set(ids).size).toBe(ids.length);
      // エッジ端点はすべてノードとして存在する
      const idSet = new Set(ids);
      for (const e of tpl.edges) {
        expect(idSet.has(e.src)).toBe(true);
        expect(idSet.has(e.dst)).toBe(true);
      }
    }
  });
});
