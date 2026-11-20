// wasmLutFlat.ts(lutFlatGsAsm / lutFlatPackedGsAsm)のテスト。
// 契約: release.wasm / release-simd.wasm の双方で、JS 版
// lutFlat.ts::evalLutFlatGroupSum と全ビット一致(1枚ずつ/パックの境界、
// 語境界 31/32/33/64 を跨ぐ枚数、Int32Array 入力、params の再利用)。
// JS 版自体は lutFlat.test.ts が参照実装と突き合わせている。
import { describe, expect, it } from 'vitest';

import { evalLutFlatGroupSum, LUT_FLAT_PACK_MIN_BATCH } from './lutFlat';
import type { LutFlatParams } from './lutFlat';
import { WasmLutFlatOps } from './wasmLutFlat';
import { loadTestWasmModule, loadTestWasmSimdModule } from './wasmTestUtil';

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
    const lim = 1 + inW + g;
    gates.push([pick(16), pick(lim), pick(lim)]);
  }
  const outputs: number[] = [];
  for (let i = 0; i < outW; i++) outputs.push(2 * pick(1 + inW + nG) + pick(2));
  outputs[0] = 1; // 定数1
  return { in_width: inW, out_width: outW, gates, outputs };
}

function randomBytes(seed: number, n: number): Uint8Array {
  const r = rng(seed);
  return Uint8Array.from({ length: n }, () => (r() < 0.5 ? 1 : 0));
}

const mods = {
  release: await loadTestWasmModule(),
  'release-simd': await loadTestWasmSimdModule(),
};

for (const [name, mod] of Object.entries(mods)) {
  describe.skipIf(!mod)(`WasmLutFlatOps(${name})`, () => {
    const ops = new WasmLutFlatOps(mod!);
    const inW = 37;
    const k = 5;
    const p = randomNetlist(11, inW, 300, 40);

    it('カーネルを持っている', () => {
      expect(ops.available).toBe(true);
    });

    for (const batch of [1, 2, LUT_FLAT_PACK_MIN_BATCH, 31, 32, 33, 64, 97]) {
      it(`batch=${batch}: JS 版と一致`, () => {
        const x = { data: randomBytes(batch, batch * inW), width: inW };
        const got = ops.groupSum(p, x, batch, k);
        const want = evalLutFlatGroupSum(p, x, batch, k);
        expect(got).not.toBeNull();
        expect(got!.width).toBe(k);
        expect(Array.from(got!.data)).toEqual(Array.from(want.data));
      });
    }

    it('Int32Array 入力・別 params の交互利用でも一致', () => {
      const q = randomNetlist(12, inW, 120, 10);
      for (let rep = 0; rep < 3; rep++) {
        for (const [pp, kk] of [[p, k], [q, 2]] as const) {
          const batch = 3 + rep * 20;
          const x = { data: Int32Array.from(randomBytes(rep + 50, batch * inW)), width: inW };
          expect(Array.from(ops.groupSum(pp, x, batch, kk)!.data))
            .toEqual(Array.from(evalLutFlatGroupSum(pp, x, batch, kk).data));
        }
      }
    });

    it('割り切れない k・幅不一致は null(JS 版へ委ねる)', () => {
      const x = { data: randomBytes(1, inW), width: inW };
      expect(ops.groupSum(p, x, 1, 3)).toBeNull();
      expect(ops.groupSum(p, { data: x.data, width: inW + 1 }, 1, k)).toBeNull();
    });
  });
}
