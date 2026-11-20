import { describe, expect, it } from 'vitest';
import { parseSlice, sliceLabel } from './parseSlice';

describe('parseSlice', () => {
  it('空文字は全体(null)', () => {
    expect(parseSlice('', 8)).toBeNull();
    expect(parseSlice('  ', 8)).toBeNull();
  });
  it('単一 bit と範囲', () => {
    expect(parseSlice('5', 8)).toEqual([5, 5]);
    expect(parseSlice('7:0', 8)).toEqual([7, 0]);
    expect(parseSlice('[3:1]', 8)).toEqual([3, 1]);
  });
  it('不正入力', () => {
    expect(() => parseSlice('abc', 8)).toThrow();
    expect(() => parseSlice('0:7', 8)).toThrow(); // msb < lsb
    expect(() => parseSlice('8', 8)).toThrow();   // 幅超過
  });
});

describe('sliceLabel', () => {
  it('表示ラベル', () => {
    expect(sliceLabel(null)).toBe('');
    expect(sliceLabel([5, 5])).toBe('[5]');
    expect(sliceLabel([7, 0])).toBe('[7:0]');
  });
});
