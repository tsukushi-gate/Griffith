// 反例トレース純関数(§4-2)の単体テスト。
// 小さなIRフィクスチャで到達コーン計算・原因候補順位付けを検証する。

import { describe, expect, it } from 'vitest';
import type { IRGraphJson, IRPortJson } from '../ir/schema';
import {
  binaryEntropy,
  computeReachabilityCone,
  mismatchedOutputTargets,
  rankCauseCandidates,
} from './counterexampleTrace';

function port(id: string, width: number): IRPortJson {
  return { id, width, shape: null, encoding: null, semantics: null };
}

/**
 * フィクスチャ:
 *   x(2bit) → L1(lut_layer, 2ニューロン) → L2(lut_layer, 1ニューロン) → y
 *   別枝: z(1bit) → M1(lut_layer, 1ニューロン) → w   ← yの反例には無関係
 * yの到達コーンは {x, L1, L2} でありM1/z/wを含まない、を検証する。
 */
const FIXTURE: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.1.0',
  metadata: {},
  inputs: [port('x', 2), port('z', 1)],
  outputs: [port('y', 1), port('w', 1)],
  nodes: [
    {
      id: 'L1',
      type: 'lut_layer',
      params: { in_width: 2, out_width: 2, n_inputs: 2, indices: [[0, 0], [1, 1]], truth_tables: ['0x8', '0xe'] },
      train_info: {},
      meta: {},
    },
    {
      id: 'L2',
      type: 'lut_layer',
      params: { in_width: 2, out_width: 1, n_inputs: 2, indices: [[0], [1]], truth_tables: ['0x6'] },
      train_info: {},
      meta: {},
    },
    {
      id: 'M1',
      type: 'lut_layer',
      params: { in_width: 1, out_width: 1, n_inputs: 1, indices: [[0]], truth_tables: ['0x2'] },
      train_info: {},
      meta: {},
    },
  ],
  edges: [
    { src: 'x', dst: 'L1' },
    { src: 'L1', dst: 'L2' },
    { src: 'L2', dst: 'y' },
    { src: 'z', dst: 'M1' },
    { src: 'M1', dst: 'w' },
  ],
};

describe('binaryEntropy', () => {
  it('確定(0/1)は0、五分五分(0.5)は最大の1', () => {
    expect(binaryEntropy(0)).toBe(0);
    expect(binaryEntropy(1)).toBe(0);
    expect(binaryEntropy(0.5)).toBeCloseTo(1, 10);
    expect(binaryEntropy(0.11)).toBeLessThan(binaryEntropy(0.4));
  });
});

describe('mismatchedOutputTargets', () => {
  it('食い違う出力ポートの接続元ノードだけを起点にする', () => {
    // y(index0)のみ食い違い、w(index1)は一致
    const targets = mismatchedOutputTargets(FIXTURE, [1, 0], [0, 0]);
    expect([...targets]).toEqual(['L2']);
  });

  it('全一致なら起点なし', () => {
    expect(mismatchedOutputTargets(FIXTURE, [0, 0], [0, 0]).size).toBe(0);
  });
});

describe('computeReachabilityCone', () => {
  it('yの到達コーンはx,L1,L2で別枝(M1,z,w)を含まない', () => {
    const cone = computeReachabilityCone(FIXTURE, ['L2']);
    expect([...cone.nodes].sort()).toEqual(['L1', 'L2', 'x']);
    expect(cone.edges.has('x->L1')).toBe(true);
    expect(cone.edges.has('L1->L2')).toBe(true);
    expect(cone.nodes.has('M1')).toBe(false);
    expect(cone.edges.has('z->M1')).toBe(false);
  });
});

describe('rankCauseCandidates', () => {
  const cone = computeReachabilityCone(FIXTURE, ['L2']);

  it('食い違うゲートを迷いだけのゲートより上位にする', () => {
    // L1: neuron0はsoft/hard食い違い(1 vs 0.2)、neuron1は迷いのみ(1 vs 0.6=同側) / L2: 迷いのみ
    const hard = { L1: new Float32Array([1, 1]), L2: new Float32Array([1]) };
    const soft = { L1: new Float32Array([0.2, 0.6]), L2: new Float32Array([0.9]) };
    const ranked = rankCauseCandidates(FIXTURE, cone, hard, soft);
    expect(ranked[0]).toMatchObject({ nodeId: 'L1', neuron: 0, disagreement: true });
    expect(ranked.some((c) => c.nodeId === 'L1' && c.neuron === 1)).toBe(true);
    // 食い違いゲートのスコアは常に迷いのみゲートより高い
    const top = ranked[0].score;
    const others = ranked.filter((c) => !c.disagreement).map((c) => c.score);
    expect(others.every((s) => s < top)).toBe(true);
  });

  it('コーン外のゲート(M1)は候補に含めない', () => {
    const hard = { L1: new Float32Array([1, 0]), L2: new Float32Array([1]), M1: new Float32Array([0]) };
    const soft = { L1: new Float32Array([0.2, 0.5]), L2: new Float32Array([0.9]), M1: new Float32Array([0.5]) };
    const ranked = rankCauseCandidates(FIXTURE, cone, hard, soft);
    expect(ranked.some((c) => c.nodeId === 'M1')).toBe(false);
  });

  it('迷いも食い違いも無いゲートは除外する', () => {
    const hard = { L1: new Float32Array([1, 0]), L2: new Float32Array([1]) };
    const soft = { L1: new Float32Array([1, 0]), L2: new Float32Array([1]) };
    expect(rankCauseCandidates(FIXTURE, cone, hard, soft)).toEqual([]);
  });
});
