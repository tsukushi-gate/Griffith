// カードスタイル表の単一情報源テスト。
// 型別カテゴリ/頭文字/主要パラメータ抽出が仕様どおりであることを固定する。

import { describe, expect, it } from 'vitest';
import type { IRNodeJson } from '../../ir/schema';
import {
  cardCategoryStyle,
  cardStyleFor,
  mainParamsFor,
} from './CARD_STYLE_BY_TYPE';

describe('cardStyleFor', () => {
  it('NNC配色カテゴリを踏襲する(Input=黒/演算=青/活性化=赤/pool=茶)', () => {
    expect(cardStyleFor('input').category).toBe('input');
    expect(cardStyleFor('lut_layer').category).toBe('compute');
    expect(cardStyleFor('conv2d').category).toBe('compute');
    expect(cardStyleFor('group_sum').category).toBe('activation');
    expect(cardStyleFor('or_pool2d').category).toBe('pool');
    expect(cardStyleFor('concat').category).toBe('other');
  });

  it('頭文字は1字で、配色はカテゴリから引ける', () => {
    expect(cardStyleFor('lut_layer').letter).toBe('L');
    expect(cardCategoryStyle('input').bg).toBe('#2b2b2b');
    expect(cardCategoryStyle('lut_layer').bg).toBe(cardCategoryStyle('conv2d').bg);
  });
});

describe('mainParamsFor', () => {
  const mk = (type: IRNodeJson['type'], params: Record<string, unknown>): IRNodeJson => ({
    id: 'n',
    type,
    params,
    train_info: {},
    meta: {},
  });

  it('lut_layerは入力数と出力幅を出す', () => {
    expect(mainParamsFor(mk('lut_layer', { n_inputs: 2, out_width: 8 }))).toBe('N=2 out=8');
  });

  it('concatは入力幅の連結を出す', () => {
    expect(mainParamsFor(mk('concat', { in_widths: [3, 5] }))).toBe('3+5');
  });

  it('conv2dはカーネルと出力チャネルを出す', () => {
    expect(mainParamsFor(mk('conv2d', { kernel_size: [3, 3], out_channels: 16 }))).toBe(
      'k=3×3 oc=16',
    );
  });
});
