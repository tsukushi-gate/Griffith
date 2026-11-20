// decodeU8(u8量子化エンベロープのデコーダ)の単体テスト。
// backend monitor.py encode_u8 の逆変換: 値 = q/255 × scale。

import { describe, expect, it } from 'vitest';
import type { U8Envelope } from './rest';
import { decodeU8 } from './u8';

function env(data: string, scale: number, shape: number[] = []): U8Envelope {
  return { encoding: 'u8', shape, scale, data };
}

describe('decodeU8', () => {
  it('既知のbase64(バイト列[0,255,51])をデコードする', () => {
    // 'AP8z' = base64([0x00, 0xff, 0x33])
    const out = decodeU8(env('AP8z', 1, [3]));
    expect(out).toBeInstanceOf(Float32Array);
    expect(out).toHaveLength(3);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(1); // 255/255 × 1
    expect(out[2]).toBeCloseTo(51 / 255, 6); // = 0.2
  });

  it('scaleで値域が復元される', () => {
    const out = decodeU8(env('AP8z', 2.5));
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(2.5, 6);
    expect(out[2]).toBeCloseTo(0.2 * 2.5, 6);
  });

  it("最大量子値のみ('/w==' = [255])はscaleそのものへ戻る", () => {
    const out = decodeU8(env('/w==', 3));
    expect(out).toHaveLength(1);
    expect(out[0]).toBeCloseTo(3, 6);
  });

  it('scale=0なら全て0になる', () => {
    const out = decodeU8(env('AP8z', 0));
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });

  it('空データは長さ0の配列を返す', () => {
    expect(decodeU8(env('', 1))).toHaveLength(0);
  });

  it('任意のバイト列を量子化式(q/255×scale)どおりに復元する', () => {
    const bytes = [0, 1, 127, 128, 254, 255];
    const data = btoa(String.fromCharCode(...bytes));
    const out = decodeU8(env(data, 10));
    for (const [i, q] of bytes.entries()) {
      expect(out[i]).toBeCloseTo((q / 255) * 10, 5);
    }
  });

  it('同一エンベロープの再デコードはメモ化された同一インスタンスを返す', () => {
    const e = env('AP8z', 1, [3]);
    const first = decodeU8(e);
    expect(decodeU8(e)).toBe(first);
    // 内容が同じでも別オブジェクトなら独立にデコードされる(値は一致)
    const other = decodeU8(env('AP8z', 1, [3]));
    expect(other).not.toBe(first);
    expect(Array.from(other)).toEqual(Array.from(first));
  });
});
