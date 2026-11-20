// WB-4/W4 前処理テスト: dense golden fixtureのビット列(Python
// encode_dense_thermometer)と TS encodeThermometer の完全一致
// (ビット順契約 ((ch*L)+t)*T+thr の実地検証)。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { ModelDoc } from './types';
import { argmax, encodeThermometer } from './preproc';

const here = dirname(fileURLToPath(import.meta.url));

function b64ToBits(b64: string, nBits: number): Uint8Array {
  const raw = atob(b64);
  const bits = new Uint8Array(nBits);
  for (let i = 0; i < nBits; i++) {
    bits[i] = (raw.charCodeAt(i >> 3) >> (i & 7)) & 1;
  }
  return bits;
}

const goldenPath = join(here, 'fixtures/dense_golden.json');
const hasGolden = existsSync(goldenPath);

describe.skipIf(!hasGolden)('preproc: dense goldenとのビット一致', () => {
  interface Golden {
    model: ModelDoc;
    n: number;
    x_i16: number[][];
    scale: number;
    bits_b64: string[];
  }
  const golden = JSON.parse(readFileSync(goldenPath, 'utf8')) as Golden;
  const model = golden.model;

  it(`全${golden.n}サンプルでビット列完全一致`, () => {
    const chLen = golden.x_i16[0].length;
    const w = model.ir.inputs[0].width;
    for (let i = 0; i < golden.n; i++) {
      const row = new Float32Array(chLen);
      for (let j = 0; j < chLen; j++) {
        row[j] = Math.fround(golden.x_i16[i][j] * golden.scale);
      }
      const bits = encodeThermometer(row, model.preproc);
      const expected = b64ToBits(golden.bits_b64[i], w);
      expect(Array.from(bits), `sample ${i}`).toEqual(Array.from(expected));
    }
  });
});

describe('argmax', () => {
  it('タイは最初の最大値(np.argmax互換)', () => {
    expect(argmax([1, 3, 3, 2])).toBe(1);
    expect(argmax([5])).toBe(0);
    expect(argmax([-1, -1])).toBe(0);
  });
});
