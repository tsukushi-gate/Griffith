// shapeInference(フロント側の形状推論・診断)の単体テスト。
// 幅整合/入力欠落/多重入力/group_sum割り切りの良否両ケースに加え、
// v0.2のarity表・slot診断・concat/add・conv/pool shape伝播を検証する。

import { describe, expect, it } from 'vitest';
import type { IRNodeJson, IRPortJson } from './schema';
import {
  conv2dOutShape,
  hasShapeErrors,
  inferShapes,
  nodeSlotInWidth,
  nodeWidths,
  orPool2dOutShape,
} from './shapeInference';

function lut(id: string, inWidth: number, outWidth: number): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: 2,
      indices: [[0], [1]],
      truth_tables: ['0x6'],
    },
    train_info: {},
    meta: {},
  };
}

function groupSum(id: string, inWidth: number, k: number): IRNodeJson {
  return {
    id,
    type: 'group_sum',
    params: { in_width: inWidth, k, tau: 1.0 },
    train_info: {},
    meta: {},
  };
}

const inputX = (width: number): IRPortJson[] => [
  { id: 'x', width, shape: null, encoding: null, semantics: null },
];

describe('nodeWidths', () => {
  it('lut_layerはin_width/out_width', () => {
    expect(nodeWidths(lut('n1', 4, 2))).toEqual({ in: 4, out: 2 });
  });

  it('group_sumはin_width/k', () => {
    expect(nodeWidths(groupSum('n1', 60, 10))).toEqual({ in: 60, out: 10 });
  });

  it('binarize/flattenは幅維持', () => {
    const bin: IRNodeJson = {
      id: 'b',
      type: 'binarize',
      params: { width: 8, threshold: 0.5 },
      train_info: {},
      meta: {},
    };
    expect(nodeWidths(bin)).toEqual({ in: 8, out: 8 });
    expect(nodeWidths({ ...bin, type: 'flatten' })).toEqual({ in: 8, out: 8 });
  });

  it('blackboxはin_width/out_width、paramsが無ければnull', () => {
    const bb: IRNodeJson = {
      id: 'bb',
      type: 'blackbox',
      params: { in_width: 3, out_width: 5 },
      train_info: {},
      meta: {},
    };
    expect(nodeWidths(bb)).toEqual({ in: 3, out: 5 });
    expect(nodeWidths({ ...bb, params: {} })).toEqual({ in: null, out: null });
  });
});

describe('inferShapes: 正常系', () => {
  it('整合したチェーンには診断が出ない', () => {
    const { widths, diagnostics } = inferShapes(
      [lut('n1', 2, 1)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'y' },
      ],
      inputX(2),
    );
    expect(diagnostics).toEqual([]);
    expect(widths.n1).toEqual({ in: 2, out: 1 });
    expect(hasShapeErrors(diagnostics)).toBe(false);
  });

  it('group_sumの割り切り(in_width % k == 0)は診断なし', () => {
    const { diagnostics } = inferShapes(
      [groupSum('gs', 6, 3)],
      [
        { src: 'x', dst: 'gs' },
        { src: 'gs', dst: 'y' },
      ],
      inputX(6),
    );
    expect(diagnostics).toEqual([]);
  });
});

describe('inferShapes: 幅整合', () => {
  it('入力ポート→ノードの幅不整合はerror(edgeId付き)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 1)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'y' },
      ],
      inputX(3),
    );
    const err = diagnostics.find((d) => d.edgeId === 'x->n1');
    expect(err?.level).toBe('error');
    expect(err?.nodeId).toBe('n1');
    expect(hasShapeErrors(diagnostics)).toBe(true);
  });

  it('ノード間の幅不整合はerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 4), lut('n2', 2, 1)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'n2' },
        { src: 'n2', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.edgeId === 'n1->n2');
    expect(err?.level).toBe('error');
  });

  it('出力ポート行きエッジは幅検査の対象外', () => {
    // n1(out=4) → y(width定義なし=ノードではない)は診断されない
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 4)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'y' },
      ],
      inputX(2),
    );
    expect(diagnostics.filter((d) => d.edgeId === 'n1->y')).toEqual([]);
  });
});

describe('inferShapes: 接続数', () => {
  it('入力エッジ欠落はerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 1)],
      [{ src: 'n1', dst: 'y' }],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.nodeId === 'n1' && d.level === 'error');
    expect(err?.message).toContain('入力エッジがない');
  });

  it('多重入力はerror(v0.1は単一入力のみ)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), lut('n2', 2, 2), lut('n3', 2, 1)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'n3' },
        { src: 'n2', dst: 'n3' },
        { src: 'n3', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find(
      (d) => d.nodeId === 'n3' && d.message.includes('複数'),
    );
    expect(err?.level).toBe('error');
  });
});

describe('inferShapes: group_sum割り切り', () => {
  it('in_width % k != 0はerror', () => {
    const { diagnostics } = inferShapes(
      [groupSum('gs', 7, 3)],
      [
        { src: 'x', dst: 'gs' },
        { src: 'gs', dst: 'y' },
      ],
      inputX(7),
    );
    const err = diagnostics.find(
      (d) => d.nodeId === 'gs' && d.message.includes('% k'),
    );
    expect(err?.level).toBe('error');
  });

  it('k=0はゼロ除算ガードにより割り切り診断を出さない', () => {
    const { diagnostics } = inferShapes(
      [groupSum('gs', 6, 0)],
      [
        { src: 'x', dst: 'gs' },
        { src: 'gs', dst: 'y' },
      ],
      inputX(6),
    );
    expect(diagnostics.filter((d) => d.message.includes('% k'))).toEqual([]);
  });
});

describe('inferShapes: 出力未使用の警告', () => {
  it('どこにも接続されないノードはwarning(errorではない)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 1)],
      [{ src: 'x', dst: 'n1' }],
      inputX(2),
    );
    const warn = diagnostics.find((d) => d.message.includes('未使用'));
    expect(warn?.level).toBe('warning');
    expect(warn?.nodeId).toBe('n1');
    // warningのみならエラー扱いにならない
    expect(hasShapeErrors(diagnostics)).toBe(false);
  });
});

// --- v0.2: 多入力DAG ---

function concat(id: string, inWidths: number[]): IRNodeJson {
  return {
    id,
    type: 'concat',
    params: { in_widths: inWidths },
    train_info: {},
    meta: {},
  };
}

function add(id: string, width: number, mode = 'xor'): IRNodeJson {
  return {
    id,
    type: 'add',
    params: { width, mode },
    train_info: {},
    meta: {},
  };
}

function constant(id: string, width: number, bits: number[]): IRNodeJson {
  return {
    id,
    type: 'constant',
    params: { width, bits },
    train_info: {},
    meta: {},
  };
}

describe('inferShapes: fanout許容(v0.2)', () => {
  it('同一srcから複数エッジを張っても診断は出ない', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), lut('n2', 2, 2)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'y' },
        { src: 'n2', dst: 'y2' },
      ],
      inputX(2),
    );
    expect(diagnostics).toEqual([]);
  });
});

describe('inferShapes: constant', () => {
  it('入力なしで固定幅出力を持つ正常系は診断なし', () => {
    const { widths, diagnostics } = inferShapes(
      [constant('c1', 3, [1, 0, 1])],
      [{ src: 'c1', dst: 'y' }],
      inputX(1),
    );
    expect(diagnostics).toEqual([]);
    expect(widths.c1).toEqual({ in: null, out: 3 });
  });

  it('入力エッジを持つconstantはerror', () => {
    const { diagnostics } = inferShapes(
      [constant('c1', 2, [1, 0])],
      [
        { src: 'x', dst: 'c1' },
        { src: 'c1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('入力エッジを持てません'));
    expect(err?.level).toBe('error');
  });

  it('bits長がwidthと一致しない場合はerror', () => {
    const { diagnostics } = inferShapes(
      [constant('c1', 3, [1, 0])],
      [{ src: 'c1', dst: 'y' }],
      inputX(1),
    );
    const err = diagnostics.find((d) => d.message.includes('0/1リスト'));
    expect(err?.level).toBe('error');
  });
});

describe('inferShapes: concat', () => {
  it('ダイヤモンド型(fanout→2分岐→concat)の正常系は診断なし', () => {
    const { widths, diagnostics } = inferShapes(
      [lut('n1', 4, 3), lut('n2', 4, 2), concat('c1', [3, 2]), lut('n3', 5, 1)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'c1' },
        { src: 'n2', dst: 'c1', dst_slot: 1 },
        { src: 'c1', dst: 'n3' },
        { src: 'n3', dst: 'y' },
      ],
      inputX(4),
    );
    expect(diagnostics).toEqual([]);
    expect(widths.c1).toEqual({ in: null, out: 5 });
  });

  it('slot別の幅不整合はerror(slot番号入りメッセージ)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 4, 3), lut('n2', 4, 4), concat('c1', [3, 2])],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'c1' },
        { src: 'n2', dst: 'c1', dst_slot: 1 },
        { src: 'c1', dst: 'y' },
      ],
      inputX(4),
    );
    const err = diagnostics.find((d) => d.edgeId === 'n2->c1');
    expect(err?.level).toBe('error');
    expect(err?.message).toContain('slot1');
  });

  it('slot重複はerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), lut('n2', 2, 2), concat('c1', [2, 2])],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'c1', dst_slot: 1 },
        { src: 'n2', dst: 'c1', dst_slot: 1 },
        { src: 'c1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('重複'));
    expect(err?.level).toBe('error');
    expect(err?.nodeId).toBe('c1');
  });

  it('slot欠番はerror(in_widths要素数ぶんの接続が必要)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), concat('c1', [2, 2, 2])],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'c1' },
        { src: 'n1', dst: 'c1', dst_slot: 2 },
        { src: 'c1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('slot 1 が欠番'));
    expect(err?.level).toBe('error');
  });

  it('範囲外slotはerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), lut('n2', 2, 2), concat('c1', [2, 2])],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'c1' },
        { src: 'n2', dst: 'c1', dst_slot: 5 },
        { src: 'c1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('範囲外'));
    expect(err?.level).toBe('error');
  });

  it('in_widthsが2要素未満はerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), concat('c1', [2])],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'c1' },
        { src: 'c1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('2要素以上'));
    expect(err?.level).toBe('error');
  });

  it('nodeSlotInWidthはconcatのslot別幅を返す', () => {
    const c = concat('c1', [3, 2]);
    expect(nodeSlotInWidth(c, 0)).toBe(3);
    expect(nodeSlotInWidth(c, 1)).toBe(2);
    expect(nodeSlotInWidth(c, 9)).toBe(null);
  });
});

describe('inferShapes: add', () => {
  it('残差型(fanout→add slot1)の正常系は診断なし', () => {
    const { widths, diagnostics } = inferShapes(
      [lut('n1', 2, 2), add('a1', 2)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'a1' },
        { src: 'x', dst: 'a1', dst_slot: 1 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    expect(diagnostics).toEqual([]);
    expect(widths.a1).toEqual({ in: 2, out: 2 });
  });

  it('3入力のaddも合法(slot連番なら)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), lut('n2', 2, 2), add('a1', 2)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'x', dst: 'n2' },
        { src: 'n1', dst: 'a1' },
        { src: 'n2', dst: 'a1', dst_slot: 1 },
        { src: 'x', dst: 'a1', dst_slot: 2 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    expect(diagnostics).toEqual([]);
  });

  it('入力1本のみはerror(2本以上必要)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), add('a1', 2)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'a1' },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('2本以上'));
    expect(err?.level).toBe('error');
    expect(err?.nodeId).toBe('a1');
  });

  it('slot欠番(0,2)はerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), add('a1', 2)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'a1' },
        { src: 'x', dst: 'a1', dst_slot: 2 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('欠番'));
    expect(err?.level).toBe('error');
  });

  it('slot幅不整合はerror(全slot同幅)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 3, 2), add('a1', 3)],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'a1' },
        { src: 'x', dst: 'a1', dst_slot: 1 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(3),
    );
    // n1(out=2) → a1(width=3) が不整合。x(3) → slot1 は整合
    const err = diagnostics.find((d) => d.edgeId === 'n1->a1');
    expect(err?.level).toBe('error');
  });

  it('未知のmodeはerror', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 2), add('a1', 2, 'nand')],
      [
        { src: 'x', dst: 'n1' },
        { src: 'n1', dst: 'a1' },
        { src: 'x', dst: 'a1', dst_slot: 1 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('addモード'));
    expect(err?.level).toBe('error');
  });

  it('width<=0はerror', () => {
    const { diagnostics } = inferShapes(
      [add('a1', 0)],
      [
        { src: 'x', dst: 'a1' },
        { src: 'x', dst: 'a1', dst_slot: 1 },
        { src: 'a1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('正整数'));
    expect(err?.level).toBe('error');
  });
});

describe('inferShapes: 単一入力ノードのslot制約', () => {
  it('slot1への接続はerror(slot 0のみ)', () => {
    const { diagnostics } = inferShapes(
      [lut('n1', 2, 1)],
      [
        { src: 'x', dst: 'n1', dst_slot: 1 },
        { src: 'n1', dst: 'y' },
      ],
      inputX(2),
    );
    const err = diagnostics.find((d) => d.message.includes('slot 0のみ'));
    expect(err?.level).toBe('error');
  });
});

// --- v0.2: conv/pool shape伝播 ---

function conv2d(id: string, params: Record<string, unknown>): IRNodeJson {
  return { id, type: 'conv2d', params, train_info: {}, meta: {} };
}

function orPool2d(id: string, params: Record<string, unknown>): IRNodeJson {
  return { id, type: 'or_pool2d', params, train_info: {}, meta: {} };
}

const CONV_PARAMS = {
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
};

describe('conv2d/or_pool2dのshape伝播', () => {
  it('conv2dOutShape: (1,4,4) k3 s1 p0 oc2 → (2,2,2)', () => {
    expect(conv2dOutShape(CONV_PARAMS)).toEqual([2, 2, 2]);
  });

  it('kernel/stride/paddingはスカラでも計算できる', () => {
    expect(
      conv2dOutShape({ ...CONV_PARAMS, kernel_size: 3, stride: 1, padding: 0 }),
    ).toEqual([2, 2, 2]);
  });

  it('orPool2dOutShape: (2,4,4) k2 s2 → (2,2,2)', () => {
    expect(
      orPool2dOutShape({ in_shape: [2, 4, 4], kernel_size: 2, stride: 2 }),
    ).toEqual([2, 2, 2]);
  });

  it('nodeWidthsはC*H*W flatten幅を返す', () => {
    expect(nodeWidths(conv2d('cv', CONV_PARAMS))).toEqual({ in: 16, out: 8 });
    expect(
      nodeWidths(orPool2d('p', { in_shape: [2, 2, 2], kernel_size: 2, stride: 2 })),
    ).toEqual({ in: 8, out: 2 });
  });

  it('conv→poolチェーンの正常系は診断なし(in_shapeから出力幅計算)', () => {
    const { diagnostics } = inferShapes(
      [
        conv2d('cv1', CONV_PARAMS),
        orPool2d('p1', { in_shape: [2, 2, 2], kernel_size: 2, stride: 2, padding: 0 }),
      ],
      [
        { src: 'x', dst: 'cv1' },
        { src: 'cv1', dst: 'p1' },
        { src: 'p1', dst: 'y' },
      ],
      inputX(16),
    );
    expect(diagnostics).toEqual([]);
  });

  it('入力幅とin_shapeのflatten幅不整合はerror', () => {
    const { diagnostics } = inferShapes(
      [conv2d('cv1', CONV_PARAMS)],
      [
        { src: 'x', dst: 'cv1' },
        { src: 'cv1', dst: 'y' },
      ],
      inputX(20),
    );
    const err = diagnostics.find((d) => d.edgeId === 'x->cv1');
    expect(err?.level).toBe('error');
  });

  it('in_shape不正(2要素)はerror', () => {
    const { diagnostics } = inferShapes(
      [conv2d('cv1', { ...CONV_PARAMS, in_shape: [4, 4] })],
      [
        { src: 'x', dst: 'cv1' },
        { src: 'cv1', dst: 'y' },
      ],
      inputX(16),
    );
    const err = diagnostics.find(
      (d) => d.nodeId === 'cv1' && d.message.includes('不正'),
    );
    expect(err?.level).toBe('error');
  });

  it('カーネルが大きすぎて出力形状が0以下になる場合はerror', () => {
    const { diagnostics } = inferShapes(
      [orPool2d('p1', { in_shape: [1, 2, 2], kernel_size: 4, stride: 1 })],
      [
        { src: 'x', dst: 'p1' },
        { src: 'p1', dst: 'y' },
      ],
      inputX(4),
    );
    const err = diagnostics.find((d) => d.message.includes('出力形状が不正'));
    expect(err?.level).toBe('error');
  });
});
