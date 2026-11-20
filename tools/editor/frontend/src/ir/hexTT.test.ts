// hexTT(真理値表hex文字列のビット操作)の単体テスト。
// N=6の真理値表は64bitでNumberの安全整数を超えるため、BigInt域のケースを必ず含める。

import { describe, expect, it } from 'vitest';
import { popcount, ttBit, ttParse, ttToggleBit, ttToHex } from './hexTT';

// N=6の全ビット立ち(64bit): Number.MAX_SAFE_INTEGERを超える値
const ALL_ONES_64 = '0xffffffffffffffff';
const ALL_ONES_64_VALUE = (1n << 64n) - 1n;

describe('ttParse', () => {
  it('0xプレフィックス付きhexをBigIntへ変換する', () => {
    expect(ttParse('0x6')).toBe(6n);
  });

  it('裸のhexも受け付ける', () => {
    expect(ttParse('6')).toBe(6n);
    expect(ttParse('ff')).toBe(255n);
  });

  it('大文字・前後空白を許容する', () => {
    expect(ttParse('  0xA  ')).toBe(10n);
    expect(ttParse('0XFF')).toBe(255n);
  });

  it('不正な文字列は0nへフォールバックする', () => {
    expect(ttParse('zz')).toBe(0n);
    expect(ttParse('')).toBe(0n);
    expect(ttParse('0x')).toBe(0n);
  });

  it('N=6の64bit値(安全整数超)を正確に扱う', () => {
    expect(ttParse(ALL_ONES_64)).toBe(ALL_ONES_64_VALUE);
    expect(ttParse('0x8000000000000000')).toBe(1n << 63n);
  });
});

describe('ttToHex', () => {
  it('小文字0xプレフィックス・ゼロ詰めなしで出力する', () => {
    expect(ttToHex(6n)).toBe('0x6');
    expect(ttToHex(0n)).toBe('0x0');
    expect(ttToHex(255n)).toBe('0xff');
  });

  it('64bit値をロスなく往復できる', () => {
    expect(ttToHex(ALL_ONES_64_VALUE)).toBe(ALL_ONES_64);
    expect(ttParse(ttToHex(1n << 63n))).toBe(1n << 63n);
  });
});

describe('ttBit', () => {
  it('XOR(0x6)の各ビットを読める', () => {
    // 0x6 = 0b0110: bit0=0, bit1=1, bit2=1, bit3=0
    expect(ttBit('0x6', 0)).toBe(0);
    expect(ttBit('0x6', 1)).toBe(1);
    expect(ttBit('0x6', 2)).toBe(1);
    expect(ttBit('0x6', 3)).toBe(0);
  });

  it('64bit境界のビット(k=63)を読める', () => {
    expect(ttBit('0x8000000000000000', 63)).toBe(1);
    expect(ttBit('0x8000000000000000', 62)).toBe(0);
    expect(ttBit(ALL_ONES_64, 63)).toBe(1);
  });

  it('値の範囲外のビットは0を返す', () => {
    expect(ttBit('0x6', 10)).toBe(0);
  });
});

describe('ttToggleBit', () => {
  it('ビットを立てる/落とす', () => {
    expect(ttToggleBit('0x6', 0)).toBe('0x7'); // 0→1
    expect(ttToggleBit('0x6', 1)).toBe('0x4'); // 1→0
  });

  it('2回トグルすると元に戻る', () => {
    expect(ttToggleBit(ttToggleBit('0x6', 2), 2)).toBe('0x6');
  });

  it('64bit域のビット(k=63)をトグルできる', () => {
    expect(ttToggleBit('0x0', 63)).toBe('0x8000000000000000');
    expect(ttToggleBit(ALL_ONES_64, 63)).toBe('0x7fffffffffffffff');
  });
});

describe('popcount', () => {
  it('立っているビット数を数える', () => {
    expect(popcount('0x0')).toBe(0);
    expect(popcount('0x6')).toBe(2);
    expect(popcount('0xa')).toBe(2);
    expect(popcount('0xf')).toBe(4);
  });

  it('64bit全ビット立ちで64を返す', () => {
    expect(popcount(ALL_ONES_64)).toBe(64);
  });

  it('不正入力は0を返す(ttParseのフォールバック)', () => {
    expect(popcount('not-hex')).toBe(0);
  });
});
