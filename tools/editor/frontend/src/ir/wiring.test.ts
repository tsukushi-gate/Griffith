// 構造化配線のフロント側定義のテスト: 既定値・文字列変換・統計の要約。

import { describe, expect, it } from 'vitest';
import {
  CONNECTION_CHOICES,
  defaultWiringSpec,
  describeWiringStats,
  isStructuredConnection,
  specValueToText,
  STRUCTURED_CONNECTIONS,
  textToSpecValue,
  WIRING_FIELDS,
} from './wiring';

describe('wiring: 種と既定値', () => {
  it('選択肢は random / unique + 構造化配線', () => {
    expect(CONNECTION_CHOICES.slice(0, 2)).toEqual(['random', 'unique']);
    for (const k of STRUCTURED_CONNECTIONS) {
      expect(isStructuredConnection(k)).toBe(true);
      expect(WIRING_FIELDS[k].length).toBeGreaterThan(0);
    }
    expect(isStructuredConnection('random')).toBe(false);
  });

  it('既定値は n_inputs に追従する(rot_n の shifts は 0..n-1、local2d の offsets は n 組)', () => {
    expect(defaultWiringSpec('rot_n', 3)).toEqual({ shifts: [0, 1, 2], step: 3 });
    expect(defaultWiringSpec('local2d', 2)).toEqual({ offsets: [[0, 0], [0, 1]], bits_per_cell: 1 });
    expect(defaultWiringSpec('rot', 2)).toEqual({ s_a: 0, s_b: 1, step: 1 });
    expect(defaultWiringSpec('butterfly', 6)).toEqual({ stride: 1 });
  });
});

describe('wiring: 文字列 ⇄ 値', () => {
  const field = (key: string, kind: 'int' | 'ints' | 'pairs' | 'grid', optional = false) => ({
    key,
    label: key,
    hint: '',
    kind,
    optional,
  });

  it('int / ints / pairs / grid を往復できる', () => {
    expect(specValueToText(3, field('step', 'int'))).toBe('3');
    expect(specValueToText([0, 1, 2], field('shifts', 'ints'))).toBe('0,1,2');
    expect(specValueToText([[0, 0], [1, -1]], field('offsets', 'pairs'))).toBe('0,0;1,-1');
    expect(specValueToText([8, 8], field('grid', 'grid'))).toBe('8,8');

    expect(textToSpecValue('3', field('step', 'int'))).toEqual({ ok: true, value: 3 });
    expect(textToSpecValue(' 0, 1 ,2', field('shifts', 'ints'))).toEqual({ ok: true, value: [0, 1, 2] });
    expect(textToSpecValue('0,0;1,-1', field('offsets', 'pairs'))).toEqual({
      ok: true,
      value: [[0, 0], [1, -1]],
    });
    expect(textToSpecValue('8,8', field('grid', 'grid'))).toEqual({ ok: true, value: [8, 8] });
  });

  it('不正な文字列は ok=false、省略可の空は未指定', () => {
    expect(textToSpecValue('x', field('step', 'int')).ok).toBe(false);
    expect(textToSpecValue('1.5', field('step', 'int')).ok).toBe(false);
    expect(textToSpecValue('0;1', field('offsets', 'pairs')).ok).toBe(false);
    expect(textToSpecValue('8', field('grid', 'grid')).ok).toBe(false);
    expect(textToSpecValue('', field('grid', 'grid', true))).toEqual({ ok: true, value: undefined });
    expect(textToSpecValue('', field('stride', 'int')).ok).toBe(false);
  });
});

describe('wiring: 統計の要約', () => {
  it('被覆・未使用・同一入力を 1 行にする', () => {
    const text = describeWiringStats(
      { coverage: 0.5, usage_hist: { '0': 6, '1': 6, '2': 0, '3+': 0 }, same_input: 1, n_gates: 8 },
      12,
    );
    expect(text).toContain('被覆 6/12(50.0%)');
    expect(text).toContain('未使用入力 6');
    expect(text).toContain('同一入力ゲート 1');
  });
});
