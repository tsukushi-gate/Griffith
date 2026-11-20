// ノード初期化プリセットの性質テスト。
// griffith_logic/ir/validate.py の制約(indices形状・値域・真理値表数/値域)を
// フロント生成物が満たすことを保証する。

import { describe, expect, it } from 'vitest';
import {
  conv2dInit,
  convInternalNodeCount,
  convReceptiveField,
  lutInit,
  mulberry32,
  passThroughTT,
  randomTT,
  type Conv2dInitSpec,
} from './nodeInit';

/** 真理値表hexの上限(2^(2^n))未満か。 */
function ttInRange(hex: string, nInputs: number): boolean {
  const v = BigInt(hex);
  const limit = 1n << BigInt(1 << nInputs);
  return v >= 0n && v < limit;
}

describe('passThroughTT / randomTT', () => {
  it('passThroughTT は 2^n/4 桁の a 連続(slot0素通し)', () => {
    expect(passThroughTT(2)).toBe('0xa');
    expect(passThroughTT(4)).toBe('0xaaaa');
    expect(passThroughTT(6)).toBe('0xaaaaaaaaaaaaaaaa');
  });

  it('randomTT は値域内(2^(2^n)未満)で決定的', () => {
    for (const n of [2, 3, 4, 6]) {
      const a = randomTT(n, mulberry32(1));
      const b = randomTT(n, mulberry32(1));
      expect(a).toBe(b); // seed固定で決定的
      expect(ttInRange(a, n)).toBe(true);
    }
  });

  it('n_inputs<2(2^n<4)はエラー', () => {
    expect(() => passThroughTT(1)).toThrow();
    expect(() => randomTT(1, mulberry32(1))).toThrow();
  });
});

describe('lutInit', () => {
  const cases: Array<[number, number, number]> = [
    [16, 8, 2],
    [12, 12, 2],
    [64, 100, 4],
    [7, 5, 2],
  ];

  it.each(cases)('random: 形状・値域がvalidate準拠 (in=%i,out=%i,n=%i)', (inW, outW, n) => {
    const { indices, truth_tables } = lutInit(inW, outW, n, 'random', mulberry32(42));
    expect(indices.length).toBe(n);
    for (const row of indices) {
      expect(row.length).toBe(outW);
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(inW);
      }
    }
    expect(truth_tables.length).toBe(outW);
    for (const tt of truth_tables) expect(ttInRange(tt, n)).toBe(true);
  });

  it('pass-through: 恒等結線(indices[j][o]=o%in)+slot0素通しtt', () => {
    const inW = 8;
    const outW = 8;
    const { indices, truth_tables } = lutInit(inW, outW, 2, 'pass-through', mulberry32(1));
    for (let o = 0; o < outW; o++) {
      for (let j = 0; j < 2; j++) expect(indices[j][o]).toBe(o % inW);
      expect(truth_tables[o]).toBe(passThroughTT(2));
    }
  });

  it('residual-init: 大半はpass-through、決定的', () => {
    const { truth_tables: a } = lutInit(32, 32, 2, 'residual-init', mulberry32(7));
    const { truth_tables: b } = lutInit(32, 32, 2, 'residual-init', mulberry32(7));
    expect(a).toEqual(b); // 決定的
    const passCount = a.filter((tt) => tt === passThroughTT(2)).length;
    expect(passCount).toBeGreaterThan(a.length / 2); // 微小摂動=大半は恒等
  });
});

describe('conv2dInit', () => {
  const spec: Conv2dInitSpec = {
    inShape: [1, 8, 8],
    outChannels: 4,
    kernelSize: 3,
    stride: 1,
    padding: 0,
    treeDepth: 2,
    nInputs: 2,
  };

  it('内部ノード数式(validate.pyと同一)', () => {
    expect(convInternalNodeCount(2, 2)).toBe(3); // (4-1)/(2-1)
    expect(convInternalNodeCount(2, 3)).toBe(7);
    expect(convInternalNodeCount(1, 5)).toBe(5); // n=1 は tree_depth
    expect(convInternalNodeCount(3, 2)).toBe(4); // (9-1)/(3-1)
  });

  it.each(['random', 'pass-through', 'residual-init'] as const)(
    'preset=%s: indices_flat/truth_tables がvalidate準拠',
    (preset) => {
      const p = conv2dInit(spec, preset, mulberry32(3));
      const rf = convReceptiveField(spec); // C*kH*kW = 9
      expect(rf).toBe(9);
      const nLeaves = spec.nInputs ** spec.treeDepth; // 4
      const nInternal = convInternalNodeCount(spec.nInputs, spec.treeDepth); // 3
      const idxf = p.indices_flat as number[][];
      expect(idxf.length).toBe(spec.outChannels);
      for (const row of idxf) {
        expect(row.length).toBe(nLeaves);
        for (const v of row) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThan(rf);
        }
      }
      const tts = p.truth_tables as string[];
      expect(tts.length).toBe(spec.outChannels * nInternal);
      for (const tt of tts) expect(ttInRange(tt, spec.nInputs)).toBe(true);
      // params は validate.py が期待するキーを持つ
      expect(p.kernel_size).toEqual([3, 3]);
      expect(p.stride).toEqual([1, 1]);
      expect(p.padding).toEqual([0, 0]);
    },
  );

  it('pass-through: 全内部ノードがslot0素通し', () => {
    const p = conv2dInit(spec, 'pass-through', mulberry32(9));
    const tts = p.truth_tables as string[];
    for (const tt of tts) expect(tt).toBe(passThroughTT(2));
  });
});
