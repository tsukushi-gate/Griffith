// マルチスレッド化の正しさを vitest 側で担保できる範囲のテスト。
//
// vitest(node環境)には web Worker / crossOriginIsolated が無いため、
// 「実際に複数スレッドで走らせた結果」はここでは検証できない。そこで
//   (1) シャード境界の**純関数**(shardRange)が全域を過不足なく被覆すること
//   (2) WASMカーネルを**シャード分割して呼んだ結果**が、1回で全域を呼んだ
//       結果および JS 実装(bitplane.ts)と**ビット完全一致**すること
// の2つを固定する。(2)は Worker を使わずメインスレッドから分割呼び出しする
// だけで、実際のマルチスレッド実行と全く同じ「出力領域の排他分割」を再現する
// (各シャードは自分の行/平面ブロックにしか書かないので、実行順序も
// 同時性も結果に影響しない)。ブラウザでの実スレッド一致は Playwright
// (scripts/mt_equal.mjs)側で mt=0 vs mt=N の整数一致として確認する。
import { describe, expect, it } from 'vitest';

import { orPool2x2, packPlanes, tailMask, wordsPerRow } from './bitplane';
import { shardCoversExactly, shardRange } from './wasmThreadPool';
import { loadTestWasmModule, loadTestWasmSimdModule } from './wasmTestUtil';
import type { WasmModule } from './wasmConv';

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('shardRange: 分割境界の純関数', () => {
  it('全域を過不足なく被覆し、重ならない(総当たり)', () => {
    for (let total = 0; total <= 40; total++) {
      for (let n = 1; n <= 8; n++) {
        expect(shardCoversExactly(total, n), `total=${total} n=${n}`).toBe(true);
      }
    }
    for (const total of [96, 173, 256, 512, 1024, 2833]) {
      for (const n of [1, 2, 3, 4, 6, 8, 16]) {
        expect(shardCoversExactly(total, n), `total=${total} n=${n}`).toBe(true);
      }
    }
  });

  it('シャード間のサイズ差は高々1(負荷の偏りが出ない)', () => {
    for (const total of [7, 100, 173, 1000]) {
      for (const n of [2, 3, 4, 6]) {
        const counts = Array.from({ length: n }, (_, i) => shardRange(total, n, i).count);
        expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1);
      }
    }
  });

  it('nShards=1 は全域を1つ目に割り当てる', () => {
    expect(shardRange(100, 1, 0)).toEqual({ first: 0, count: 100 });
  });
});

/** WASMヒープを使い捨てで確保して packPlanesAsm をシャード実行する。 */
function packViaWasm(
  mod: WasmModule, u8: Uint8Array, batch: number, C: number, H: number, W: number,
  nShards: number,
): Int32Array {
  const wpr = wordsPerRow(W);
  const rows = batch * C * H;
  const inPtr = mod.alloc(u8.length);
  const outPtr = mod.alloc(rows * wpr * 4);
  new Uint8Array(mod.memory.buffer, inPtr, u8.length).set(u8);
  // 出力領域を意図的に汚しておく(未書き込み領域が残らないことの確認)。
  new Int32Array(mod.memory.buffer, outPtr, rows * wpr).fill(0x5a5a5a5a);
  for (let s = 0; s < nShards; s++) {
    const { first, count } = shardRange(rows, nShards, s);
    if (count > 0) mod.packPlanesAsm!(inPtr, outPtr, first, count, W, wpr);
  }
  return Int32Array.from(new Int32Array(mod.memory.buffer, outPtr, rows * wpr));
}

function orPoolViaWasm(
  mod: WasmModule, words: Int32Array, batch: number, C: number, H: number, W: number,
  nShards: number,
): { words: Int32Array; HOut: number; WOut: number; wprOut: number } {
  const wpr = wordsPerRow(W);
  const HOut = ((H - 2) >> 1) + 1;
  const WOut = ((W - 2) >> 1) + 1;
  const wprOut = wordsPerRow(WOut);
  const maskOut = tailMask(WOut);
  const planes = batch * C;
  const inWords = planes * H * wpr;
  const outWords = planes * HOut * wprOut;
  const inPtr = mod.alloc(inWords * 4);
  const outPtr = mod.alloc(outWords * 4);
  new Int32Array(mod.memory.buffer, inPtr, inWords).set(words);
  new Int32Array(mod.memory.buffer, outPtr, outWords).fill(0x5a5a5a5a);
  for (let s = 0; s < nShards; s++) {
    const { first, count } = shardRange(planes, nShards, s);
    if (count > 0) {
      mod.orPool2x2Asm!(inPtr, outPtr, first, count, H, wpr, HOut, wprOut, maskOut);
    }
  }
  return {
    words: Int32Array.from(new Int32Array(mod.memory.buffer, outPtr, outWords)),
    HOut,
    WOut,
    wprOut,
  };
}

const scalarMod = await loadTestWasmModule();
const simdMod = await loadTestWasmSimdModule();
const BUILDS: [string, WasmModule | null][] = [['scalar', scalarMod], ['simd', simdMod]];

for (const [name, mod] of BUILDS) {
  describe.skipIf(!mod || typeof mod.packPlanesAsm !== 'function')(
    `packPlanesAsm(${name}) ≡ bitplane.ts::packPlanes`,
    () => {
      const shapes: [number, number, number, number][] = [
        // [batch, C, H, W]
        [1, 1, 1, 32], [1, 3, 5, 96], [2, 4, 7, 128], [1, 7, 9, 64],
        [1, 2, 3, 17], [1, 2, 3, 31], [1, 2, 3, 33], [1, 5, 4, 5],
      ];
      for (const [batch, C, H, W] of shapes) {
        for (const nShards of [1, 2, 3, 4, 8]) {
          it(`b=${batch} C=${C} H=${H} W=${W} shards=${nShards}: 整数完全一致`, () => {
            const rand = rng(C * 131 + H * 17 + W * 7 + nShards);
            const n = batch * C * H * W;
            const u8 = new Uint8Array(n);
            // 0/1 以外のバイト値も混ぜる(非ゼロ→1 の意味論の検証)。
            const vals = [0, 0, 1, 2, 0x7f, 0x80, 0xff];
            for (let i = 0; i < n; i++) u8[i] = vals[Math.floor(rand() * vals.length)];
            const want = packPlanes({ data: u8, width: C * H * W }, batch, C, H, W);
            const got = packViaWasm(mod!, u8, batch, C, H, W, nShards);
            expect(Array.from(got)).toEqual(Array.from(want.words));
          });
        }
      }
    },
  );

  describe.skipIf(!mod || typeof mod.orPool2x2Asm !== 'function')(
    `orPool2x2Asm(${name}) ≡ bitplane.ts::orPool2x2`,
    () => {
      const shapes: [number, number, number, number][] = [
        [1, 1, 2, 2], [1, 3, 8, 96], [2, 5, 6, 64], [1, 8, 48, 48],
        [1, 4, 10, 33], [1, 4, 7, 31], [1, 6, 12, 17],
      ];
      for (const [batch, C, H, W] of shapes) {
        for (const nShards of [1, 2, 3, 4, 8]) {
          it(`b=${batch} C=${C} H=${H} W=${W} shards=${nShards}: 整数完全一致`, () => {
            const rand = rng(C * 977 + H * 53 + W * 11 + nShards);
            const n = batch * C * H * W;
            const u8 = new Uint8Array(n);
            for (let i = 0; i < n; i++) u8[i] = rand() > 0.5 ? 1 : 0;
            const bp = packPlanes({ data: u8, width: C * H * W }, batch, C, H, W);
            const want = orPool2x2(bp, batch);
            const got = orPoolViaWasm(mod!, bp.words, batch, C, H, W, nShards);
            expect(got.HOut).toBe(want.H);
            expect(got.WOut).toBe(want.W);
            expect(got.wprOut).toBe(want.wpr);
            expect(Array.from(got.words)).toEqual(Array.from(want.words));
          });
        }
      }
    },
  );
}

describe('環境判定', () => {
  it('vitest(node)ではWorkerプールを立てず単スレッド既定になる', async () => {
    const { wasmThreadsSupported } = await import('./wasmThreadPool');
    // node には crossOriginIsolated が無く、SAB の構造化複製も MessageChannel
    // 依存なので、環境によらず「例外を投げずに真偽を返す」ことだけを固定する。
    expect(typeof wasmThreadsSupported()).toBe('boolean');
  });
});
