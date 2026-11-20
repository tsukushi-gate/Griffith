// 真理値表検証テーブルの行列生成ロジックの単体テスト。

import { describe, expect, it } from 'vitest';
import { SAMPLE_MNIST_MINI, SAMPLE_PARITY3, SAMPLE_XOR } from '../samples/sampleIr';
import {
  enumerateTruthTable,
  expectedMatchesShape,
  inputBitsToPattern,
  matchInfo,
  patternToInputBits,
  rowsToExpected,
  totalInputWidth,
} from './truthTable';

describe('patternToInputBits', () => {
  it('butterfly規約(パターンkのbit j = (k>>j)&1)でポートへ分配する', () => {
    expect(patternToInputBits(SAMPLE_XOR, 0)).toEqual({ x: [0, 0] });
    expect(patternToInputBits(SAMPLE_XOR, 1)).toEqual({ x: [1, 0] });
    expect(patternToInputBits(SAMPLE_XOR, 2)).toEqual({ x: [0, 1] });
    expect(patternToInputBits(SAMPLE_XOR, 3)).toEqual({ x: [1, 1] });
  });

  it('inputBitsToPatternは逆変換になっている', () => {
    for (let k = 0; k < 8; k++) {
      expect(inputBitsToPattern(SAMPLE_PARITY3, patternToInputBits(SAMPLE_PARITY3, k))).toBe(k);
    }
  });
});

describe('enumerateTruthTable', () => {
  it('XOR: 4行すべてhard評価出力が真理値と一致する', () => {
    const rows = enumerateTruthTable(SAMPLE_XOR);
    expect(rows).toHaveLength(4);
    for (const row of rows) {
      expect(row.inputBits).toEqual([row.k & 1, (row.k >> 1) & 1]);
      expect(row.outputBits).toEqual([row.inputBits[0] ^ row.inputBits[1]]);
    }
  });

  it('パリティ3: 8行の出力がパリティと一致する', () => {
    const rows = enumerateTruthTable(SAMPLE_PARITY3);
    expect(rows).toHaveLength(8);
    for (const row of rows) {
      const [a, b, c] = row.inputBits;
      expect(row.outputBits).toEqual([a ^ b ^ c]);
    }
  });

  it('入力幅が上限を超えるIRはエラー', () => {
    expect(totalInputWidth(SAMPLE_MNIST_MINI)).toBe(784);
    expect(() => enumerateTruthTable(SAMPLE_MNIST_MINI)).toThrow(/全列挙上限/);
  });
});

describe('matchInfo / rowsToExpected', () => {
  it('現在出力のコピーは全行一致する', () => {
    const rows = enumerateTruthTable(SAMPLE_XOR);
    const expected = rowsToExpected(rows);
    const info = matchInfo(rows, expected);
    expect(info.matched).toBe(4);
    expect(info.total).toBe(4);
    expect(info.perRow).toEqual([true, true, true, true]);
  });

  it('期待値を1箇所反転すると一致数が減る', () => {
    const rows = enumerateTruthTable(SAMPLE_XOR);
    const expected = rowsToExpected(rows);
    expected[2] = [expected[2][0] ? 0 : 1];
    const info = matchInfo(rows, expected);
    expect(info.matched).toBe(3);
    expect(info.perRow[2]).toBe(false);
  });

  it('期待値がnullなら全行不一致', () => {
    const rows = enumerateTruthTable(SAMPLE_XOR);
    expect(matchInfo(rows, null).matched).toBe(0);
  });
});

describe('expectedMatchesShape', () => {
  it('行数・出力幅が同じならtrue、異なればfalse', () => {
    const rows = enumerateTruthTable(SAMPLE_XOR);
    const expected = rowsToExpected(rows);
    expect(expectedMatchesShape(expected, rows)).toBe(true);
    expect(expectedMatchesShape(null, rows)).toBe(false);
    expect(expectedMatchesShape(expected, null)).toBe(false);
    // 行数不一致(パリティ3の8行に対しXORの4行分期待値)
    const rows3 = enumerateTruthTable(SAMPLE_PARITY3);
    expect(expectedMatchesShape(expected, rows3)).toBe(false);
  });
});
