// 指標カードの数値整形。欠損は必ず「—」にする(嘘の既定値を出さない)。
import { describe, expect, it } from 'vitest';

import { MISSING } from '../modelMeta';
import { fmtBytesBig, fmtBytesInline, fmtMetricBig, fmtMs } from './format';

describe('指標カードの整形', () => {
  it('ミリ秒', () => {
    expect(fmtMs(12.34)).toEqual({ value: '12.3', unit: 'ms' });
    expect(fmtMs(0.372)).toEqual({ value: '0.37', unit: 'ms' });
    expect(fmtMs(null)).toEqual({ value: MISSING, unit: '' });
  });

  it('バイト数', () => {
    expect(fmtBytesBig(6363347)).toEqual({ value: '6.07', unit: 'MB' });
    expect(fmtBytesBig(36265)).toEqual({ value: '35.4', unit: 'KB' });
    expect(fmtBytesBig(512)).toEqual({ value: '512', unit: 'B' });
    expect(fmtBytesBig(null)).toEqual({ value: MISSING, unit: '' });
    expect(fmtBytesInline(36265)).toBe('35.4 KB');
    expect(fmtBytesInline(null)).toBe(MISSING);
  });

  it('精度はモデルJSONの実値をそのまま4桁で出す', () => {
    expect(fmtMetricBig(0.5).value).toBe('0.5000');
    expect(fmtMetricBig(null)).toEqual({ value: MISSING, unit: '' });
  });
});
