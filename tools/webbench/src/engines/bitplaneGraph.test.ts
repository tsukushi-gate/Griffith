// bitplaneGraph.ts のグラフ結合パスのproperty test。
//
// C1.5で bitplaneGraph は2つのノード融合/ネイティブ化を追加した:
//   1. SpatialGroupSum型 permute(in_shape=[nOut,gs,H,W], perm=[0,2,3,1])+
//      group_sum を、アンパックせずビットスライス加算で評価する
//   2. 空間形状の揃った concat をチャネル平面連結で評価する
// どちらも「アンパックして irflat.ts の既存実装へ委譲する」旧経路と
// 整数完全一致でなければならない(整数出力は不変が契約)。ここでは
// evalIrFlat(意味論の基準、simulate.py 移植)を参照に、ランダム小IRで
// 直接照合する。フォールバック経路(kが一致しないgroup_sum、permute出力を
// group_sum以外が消費するケース)も同じ参照で踏む。
import { describe, expect, it } from 'vitest';

import type { IRGraphJson, IRNodeJson } from '../shared';
import { evalIrBitplane } from './bitplaneGraph';
import { evalIrFlat } from './irflat';

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 3x3/stride1/pad1(出力H,Wが入力と同じ=bitplane高速路)のランダムconv2d。 */
function convNode(
  id: string, C: number, H: number, W: number, OC: number, depth: number,
  rand: () => number,
): IRNodeJson {
  const nLeaves = 2 ** depth;
  const indicesFlat: number[][] = [];
  const truthTables: string[] = [];
  for (let oc = 0; oc < OC; oc++) {
    const row: number[] = [];
    for (let k = 0; k < nLeaves; k++) row.push(Math.floor(rand() * C * 9));
    indicesFlat.push(row);
    for (let g = 0; g < nLeaves - 1; g++) {
      truthTables.push(`0x${Math.floor(rand() * 16).toString(16)}`);
    }
  }
  return {
    id,
    type: 'conv2d',
    params: {
      in_shape: [C, H, W],
      out_channels: OC,
      kernel_size: [3, 3],
      stride: [1, 1],
      padding: [1, 1],
      tree_depth: depth,
      n_inputs: 2,
      indices_flat: indicesFlat,
      truth_tables: truthTables,
    },
    train_info: {},
    meta: {},
  };
}

function permuteNode(id: string, nOut: number, gs: number, H: number, W: number): IRNodeJson {
  return {
    id,
    type: 'permute',
    params: { in_shape: [nOut, gs, H, W], perm: [0, 2, 3, 1] },
    train_info: {},
    meta: {},
  };
}

function groupSumNode(id: string, inWidth: number, k: number): IRNodeJson {
  return {
    id, type: 'group_sum', params: { in_width: inWidth, k, tau: 1 }, train_info: {}, meta: {},
  };
}

function graph(
  C: number, H: number, W: number, nodes: IRNodeJson[], edges: IRGraphJson['edges'],
  outWidth: number,
): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: 'bitplane-graph-prop' },
    inputs: [{ id: 'x', width: C * H * W }],
    outputs: [{ id: 'y', width: outWidth }],
    nodes,
    edges,
  };
}

/** evalIrBitplane と evalIrFlat の出力が整数完全一致することを確認する。 */
function expectSameAsFlat(ir: IRGraphJson, C: number, H: number, W: number, seed: number): void {
  const rand = rng(seed);
  for (let trial = 0; trial < 4; trial++) {
    const bits = new Int32Array(C * H * W);
    for (let i = 0; i < bits.length; i++) bits[i] = rand() > 0.5 ? 1 : 0;
    const want = evalIrFlat(ir, { x: bits }, 1).output;
    const got = evalIrBitplane(ir, { x: bits }, 1).output;
    expect(got.width, `trial ${trial}`).toBe(want.width);
    expect(Array.from(got.data), `trial ${trial}`).toEqual(Array.from(want.data));
  }
}

// ============================================================
// 1. SpatialGroupSum読み出しヘッド(permute+group_sum融合)
// ============================================================

describe('SpatialGroupSumヘッド融合: evalIrFlatと整数完全一致', () => {
  // gs=1 / 非2冪(3,5,7,31) / 実モデル値(16,32)、W=17/33はワード境界端数。
  const cases: { nOut: number; gs: number; H: number; W: number }[] = [
    { nOut: 1, gs: 1, H: 3, W: 17 },
    { nOut: 2, gs: 1, H: 2, W: 33 },
    { nOut: 3, gs: 3, H: 4, W: 17 },
    { nOut: 2, gs: 5, H: 3, W: 33 },
    { nOut: 1, gs: 7, H: 5, W: 5 },
    { nOut: 2, gs: 16, H: 3, W: 24 },
    { nOut: 2, gs: 31, H: 2, W: 32 },
    { nOut: 2, gs: 32, H: 2, W: 33 },
  ];
  for (const c of cases) {
    it(`nOut=${c.nOut} gs=${c.gs} H=${c.H} W=${c.W}`, () => {
      const C = 2;
      const OC = c.nOut * c.gs;
      const rand = rng(11 * c.gs + c.W * 7 + c.nOut);
      const ir = graph(C, c.H, c.W, [
        convNode('c1', C, c.H, c.W, OC, 2, rand),
        permuteNode('p1', c.nOut, c.gs, c.H, c.W),
        groupSumNode('g1', OC * c.H * c.W, c.nOut * c.H * c.W),
      ], [
        { src: 'x', dst: 'c1' },
        { src: 'c1', dst: 'p1' },
        { src: 'p1', dst: 'g1' },
        { src: 'g1', dst: 'y' },
      ], c.nOut * c.H * c.W);
      expectSameAsFlat(ir, C, c.H, c.W, 5000 + c.gs * 13 + c.W);
    });
  }

  it('3ヘッド+concat(det実モデルと同型)', () => {
    const C = 2;
    const H = 4;
    const W = 17;
    const rand = rng(31337);
    const heads: [string, number, number][] = [
      // [接頭辞, nOut, gs]
      ['h1', 3, 16],
      ['h2', 2, 16],
      ['h3', 2, 4],
    ];
    const nodes: IRNodeJson[] = [convNode('trunk', C, H, W, 8, 2, rand)];
    const edges: IRGraphJson['edges'] = [{ src: 'x', dst: 'trunk' }];
    let outWidth = 0;
    heads.forEach(([id, nOut, gs], i) => {
      const OC = nOut * gs;
      nodes.push(convNode(`${id}_c`, 8, H, W, OC, 2, rand));
      nodes.push(permuteNode(`${id}_p`, nOut, gs, H, W));
      nodes.push(groupSumNode(`${id}_g`, OC * H * W, nOut * H * W));
      edges.push({ src: 'trunk', dst: `${id}_c` });
      edges.push({ src: `${id}_c`, dst: `${id}_p` });
      edges.push({ src: `${id}_p`, dst: `${id}_g` });
      edges.push({ src: `${id}_g`, dst: 'outcat', dst_slot: i });
      outWidth += nOut * H * W;
    });
    nodes.push({ id: 'outcat', type: 'concat', params: {}, train_info: {}, meta: {} });
    edges.push({ src: 'outcat', dst: 'y' });
    expectSameAsFlat(graph(C, H, W, nodes, edges, outWidth), C, H, W, 777);
  });
});

// ============================================================
// 2. 融合しないフォールバック経路(正しさのアンカー)
// ============================================================

describe('SpatialGroupSum型permuteの非融合フォールバック', () => {
  it('kが nOut*H*W と一致しない group_sum は従来経路(アンパック)で一致する', () => {
    const C = 2;
    const H = 2;
    const W = 5;
    const nOut = 2;
    const gs = 4;
    const OC = nOut * gs;
    const rand = rng(4711);
    const ir = graph(C, H, W, [
      convNode('c1', C, H, W, OC, 2, rand),
      permuteNode('p1', nOut, gs, H, W),
      // in_width=80。融合形の k は nOut*H*W=20 だが、あえて 10 にする。
      groupSumNode('g1', OC * H * W, 10),
    ], [
      { src: 'x', dst: 'c1' },
      { src: 'c1', dst: 'p1' },
      { src: 'p1', dst: 'g1' },
      { src: 'g1', dst: 'y' },
    ], 10);
    expectSameAsFlat(ir, C, H, W, 909);
  });

  it('permute出力をgroup_sum以外(concat)が消費しても一致する', () => {
    const C = 2;
    const H = 3;
    const W = 17;
    const nOut = 2;
    const gs = 4;
    const OC = nOut * gs;
    const rand = rng(2024);
    const ir = graph(C, H, W, [
      convNode('c1', C, H, W, OC, 2, rand),
      permuteNode('p1', nOut, gs, H, W),
      convNode('c2', C, H, W, 3, 1, rand),
      { id: 'cat', type: 'concat', params: {}, train_info: {}, meta: {} },
    ], [
      { src: 'x', dst: 'c1' },
      { src: 'c1', dst: 'p1' },
      { src: 'x', dst: 'c2' },
      { src: 'p1', dst: 'cat', dst_slot: 0 },
      { src: 'c2', dst: 'cat', dst_slot: 1 },
      { src: 'cat', dst: 'y' },
    ], (OC + 3) * H * W);
    expectSameAsFlat(ir, C, H, W, 1234);
  });
});

// ============================================================
// 3. チャネル平面concat(UNet skip接続型)
// ============================================================

describe('空間形状が揃ったconcatの平面連結: evalIrFlatと整数完全一致', () => {
  for (const [H, W] of [[3, 17], [2, 33], [4, 32]] as const) {
    it(`H=${H} W=${W}: concat(2枝) → conv → ヘッド`, () => {
      const C = 2;
      const nOut = 2;
      const gs = 8;
      const rand = rng(6060 + W);
      const ir = graph(C, H, W, [
        convNode('a', C, H, W, 4, 2, rand),
        convNode('b', C, H, W, 6, 1, rand),
        { id: 'cat', type: 'concat', params: {}, train_info: {}, meta: {} },
        convNode('c', 10, H, W, nOut * gs, 2, rand),
        permuteNode('p', nOut, gs, H, W),
        groupSumNode('g', nOut * gs * H * W, nOut * H * W),
      ], [
        { src: 'x', dst: 'a' },
        { src: 'x', dst: 'b' },
        { src: 'a', dst: 'cat', dst_slot: 0 },
        { src: 'b', dst: 'cat', dst_slot: 1 },
        { src: 'cat', dst: 'c' },
        { src: 'c', dst: 'p' },
        { src: 'p', dst: 'g' },
        { src: 'g', dst: 'y' },
      ], nOut * H * W);
      expectSameAsFlat(ir, C, H, W, 8080 + W);
    });
  }
});
