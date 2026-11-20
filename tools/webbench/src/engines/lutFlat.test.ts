// lutFlat.ts のテスト。
//
// 契約:
//   - 1枚ずつ経路 / バッチ軸パック経路の双方が、schema.py の lut_flat 節を
//     素朴に写した参照実装と全ビット一致(語境界 31/32/33・64 を跨ぐ枚数を含む)
//   - 定数0/定数1(信号0の極性)・入力の直結出力・未参照入力が正しく扱われる
//   - 生入力(Uint8Array)を渡しても Int32Array と同じ結果(コピーなしで読む)
//   - evalIrBitplane / evalIrFlat の両グラフ実行器が lut_flat ノードを実行する
//   - 壊れたネットリスト(前方参照・出力長不一致・入力幅不一致)は例外
// 元IR(lut_layer列)との一致は Python 側(tests/deploy/test_dense_flat.py の
// verify-web)が simulate をゴールデンにして担う。
import { describe, expect, it } from 'vitest';

import type { IRGraphJson } from '../shared';
import { evalIrBitplane } from './bitplaneGraph';
import { evalIrFlat } from './irflat';
import { evalLutFlat } from './lutFlat';
import type { LutFlatParams } from './lutFlat';

/** 決定的な擬似乱数(mulberry32)。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNetlist(seed: number, inW: number, nG: number, outW: number): LutFlatParams {
  const r = rng(seed);
  const pick = (n: number) => Math.floor(r() * n);
  const gates: number[][] = [];
  for (let g = 0; g < nG; g++) {
    const lim = 1 + inW + g; // 自分より前の信号(0=定数0 を含む)
    gates.push([pick(16), pick(lim), pick(lim)]);
  }
  const nSig = 1 + inW + nG;
  const outputs: number[] = [];
  for (let i = 0; i < outW; i++) outputs.push(2 * pick(nSig) + pick(2));
  outputs[0] = 0; // 定数0
  if (outW > 1) outputs[1] = 1; // 定数1
  if (outW > 2) outputs[2] = 2 * 1 + 1; // 入力0の反転を直結
  return { in_width: inW, out_width: outW, gates, outputs };
}

/** schema.py の lut_flat 節をそのまま写した参照実装(1枚)。 */
function reference(p: LutFlatParams, x: ArrayLike<number>): number[] {
  const sig: number[] = [0];
  for (let i = 0; i < p.in_width; i++) sig.push(x[i] & 1);
  for (const [tt, a, b] of p.gates) sig.push((tt >> ((sig[b] << 1) | sig[a])) & 1);
  return p.outputs.map((r) => sig[r >> 1] ^ (r & 1));
}

function randomBits(seed: number, n: number): Int32Array {
  const r = rng(seed);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = r() < 0.5 ? 1 : 0;
  return out;
}

describe('evalLutFlat', () => {
  const inW = 23;
  const p = randomNetlist(1, inW, 140, 17);

  for (const batch of [1, 2, 5, 31, 32, 33, 64, 70]) {
    for (const packed of [false, true]) {
      it(`batch=${batch} ${packed ? 'packed' : 'scalar'} が参照実装と一致`, () => {
        const x = randomBits(100 + batch, batch * inW);
        const got = evalLutFlat(p, { data: x, width: inW }, batch, packed);
        expect(got.width).toBe(17);
        for (let s = 0; s < batch; s++) {
          const want = reference(p, x.subarray(s * inW, (s + 1) * inW));
          expect(Array.from(got.data.subarray(s * 17, (s + 1) * 17))).toEqual(want);
        }
      });
    }
  }

  it('既定の経路選択(枚数で自動)も同じ結果', () => {
    for (const batch of [1, 3, 4, 40]) {
      const x = randomBits(7 + batch, batch * inW);
      const auto = evalLutFlat(p, { data: x, width: inW }, batch);
      const scalar = evalLutFlat(p, { data: x, width: inW }, batch, false);
      expect(Array.from(auto.data)).toEqual(Array.from(scalar.data));
    }
  });

  it('生入力 Uint8Array を渡しても同じ結果', () => {
    const batch = 9;
    const x = randomBits(55, batch * inW);
    const u8 = Uint8Array.from(x);
    for (const packed of [false, true]) {
      const a = evalLutFlat(p, { data: x, width: inW }, batch, packed);
      const b = evalLutFlat(p, { data: u8, width: inW }, batch, packed);
      expect(Array.from(b.data)).toEqual(Array.from(a.data));
    }
  });

  it('参照されない入力が多くても正しい(入力の詰め直し)', () => {
    // 入力 200 本のうちゲートが読むのは 0..9 だけ
    const q: LutFlatParams = {
      in_width: 200,
      out_width: 4,
      gates: [[6, 1, 2], [8, 3, 10], [14, 11, 12], [1, 13, 5]],
      outputs: [2 * 201 + 0, 2 * 202 + 1, 2 * 204, 2 * 200 + 1],
    };
    const batch = 37;
    const x = randomBits(9, batch * 200);
    for (const packed of [false, true]) {
      const got = evalLutFlat(q, { data: x, width: 200 }, batch, packed);
      for (let s = 0; s < batch; s++) {
        const want = reference(q, x.subarray(s * 200, (s + 1) * 200));
        expect(Array.from(got.data.subarray(s * 4, (s + 1) * 4))).toEqual(want);
      }
    }
  });

  it('壊れたネットリストは例外', () => {
    const x = { data: new Int32Array(3), width: 3 };
    expect(() => evalLutFlat(
      { in_width: 3, out_width: 1, gates: [[6, 1, 4]], outputs: [8] }, x, 1,
    )).toThrow(/トポロジカル順/);
    expect(() => evalLutFlat(
      { in_width: 3, out_width: 2, gates: [[6, 1, 2]], outputs: [8] }, x, 1,
    )).toThrow(/out_width/);
    expect(() => evalLutFlat(
      { in_width: 3, out_width: 1, gates: [[6, 1, 2]], outputs: [100] }, x, 1,
    )).toThrow(/信号数/);
    expect(() => evalLutFlat(
      { in_width: 4, out_width: 1, gates: [[6, 1, 2]], outputs: [10] }, x, 1,
    )).toThrow(/入力幅/);
  });
});

describe('グラフ実行器が lut_flat ノードを実行する', () => {
  const inW = 12;
  const p = randomNetlist(3, inW, 60, 8);
  const ir = {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: inW }],
    outputs: [{ id: 'y', width: 2 }],
    nodes: [
      { id: 'f', type: 'lut_flat', params: p },
      { id: 'gs', type: 'group_sum', params: { in_width: 8, k: 2, tau: 1 } },
    ],
    edges: [{ src: 'x', dst: 'f' }, { src: 'f', dst: 'gs' }, { src: 'gs', dst: 'y' }],
  } as unknown as IRGraphJson;

  for (const batch of [1, 6, 33]) {
    it(`batch=${batch}: evalIrBitplane ⇔ evalIrFlat ⇔ 参照`, () => {
      const x = randomBits(200 + batch, batch * inW);
      const u8 = Uint8Array.from(x);
      const a = evalIrBitplane(ir, { x: u8 }, batch).output;
      const b = evalIrFlat(ir, { x }, batch).output;
      expect(Array.from(a.data)).toEqual(Array.from(b.data));
      for (let s = 0; s < batch; s++) {
        const bits = reference(p, x.subarray(s * inW, (s + 1) * inW));
        const want = [bits.slice(0, 4), bits.slice(4)].map((v) => v.reduce((m, n) => m + n, 0));
        expect(Array.from(a.data.subarray(s * 2, s * 2 + 2))).toEqual(want);
      }
    });
  }
});
