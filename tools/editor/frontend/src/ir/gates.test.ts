// gates(N=2ゲートID変換・色ブレンド)の単体テスト。
// 真理値表の規約: bit k = butterfly基底順(k = s0 + 2·s1)。
// ゲートID: 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数。

import { describe, expect, it } from 'vitest';
import type { RGB } from '../theme/theme';
import { GATE_NAMES, blendToBg, ttToGateIdN2 } from './gates';

describe('ttToGateIdN2', () => {
  // [真理値表(butterfly基底順), 期待ゲートID, ゲート名]
  const CASES: Array<[number, number, string]> = [
    [0x0, 0, 'FALSE'],
    [0x1, 8, 'NOR'],
    [0x2, 2, 'A>B'],
    [0x3, 10, '!B'],
    [0x4, 4, 'B>A'],
    [0x5, 12, '!A'],
    [0x6, 6, 'XOR'],
    [0x7, 14, 'NAND'],
    [0x8, 1, 'AND'],
    [0x9, 9, 'XNOR'],
    [0xa, 3, 'A'],
    [0xb, 11, 'A>=B'],
    [0xc, 5, 'B'],
    [0xd, 13, 'B>=A'],
    [0xe, 7, 'OR'],
    [0xf, 15, 'TRUE'],
  ];

  it.each(CASES)('tt=%i → gateId=%i (%s)', (tt, expected, name) => {
    const id = ttToGateIdN2(tt);
    expect(id).toBe(expected);
    expect(GATE_NAMES[id]).toBe(name);
  });

  it('16真理値表が16ゲートIDへ全単射で対応する', () => {
    const ids = new Set(
      Array.from({ length: 16 }, (_, tt) => ttToGateIdN2(tt)),
    );
    expect(ids.size).toBe(16);
    for (let id = 0; id < 16; id++) expect(ids.has(id)).toBe(true);
  });
});

describe('blendToBg', () => {
  const red: RGB = [255, 0, 0];
  const blue: RGB = [0, 0, 255];

  it('t=0で背景色そのものになる', () => {
    expect(blendToBg(red, blue, 0)).toEqual([0, 0, 255]);
  });

  it('t=1で前景色そのものになる', () => {
    expect(blendToBg(red, blue, 1)).toEqual([255, 0, 0]);
  });

  it('t=0.5で中間色(四捨五入)になる', () => {
    expect(blendToBg(red, blue, 0.5)).toEqual([128, 0, 128]);
  });

  it('tが負なら0へクランプされる', () => {
    expect(blendToBg(red, blue, -1)).toEqual([0, 0, 255]);
  });

  it('tが1超なら1へクランプされる', () => {
    expect(blendToBg(red, blue, 2)).toEqual([255, 0, 0]);
  });
});
