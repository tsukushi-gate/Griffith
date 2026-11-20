// bitplane.ts の核心プリミティブのテスト(先に書く、TDD)。
// 各プリミティブをスカラー参照実装(または既存 packedConv.ts 実装)と
// 全域照合し、整数完全一致を確認する。
import { describe, expect, it } from 'vitest';

import type { NodeValue } from './irflat';
import { evalConcat, evalGroupSum } from './irflat';
import {
  buildSharedShiftPlane,
  buildTapPlaneBitplane,
  buildTapPlaneStrided,
  compressEven32LUT,
  compressEven32Shift,
  concatPlanes,
  copyPlanes,
  dilateX2,
  orPool2x2,
  packPlanes,
  shiftRowInto,
  spatialGroupSumBitplane,
  spreadEven16,
  tailMask,
  unpackPlanes,
  wordsPerRow,
} from './bitplane';
import type { BitplaneValue } from './bitplane';
import { evalPermutePacked, evalUpsample2dPacked } from './packedConv';
import type { UpSample2dParams } from '../shared';

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

function randomBits(n: number, rand: () => number): Int32Array {
  const data = new Int32Array(n);
  for (let i = 0; i < n; i++) data[i] = rand() > 0.5 ? 1 : 0;
  return data;
}

// ============================================================
// wordsPerRow / tailMask
// ============================================================

describe('wordsPerRow / tailMask', () => {
  it('W=32の倍数はぴったり、それ以外は端数分+1', () => {
    expect(wordsPerRow(32)).toBe(1);
    expect(wordsPerRow(33)).toBe(2);
    expect(wordsPerRow(1)).toBe(1);
    expect(wordsPerRow(96)).toBe(3);
    expect(wordsPerRow(31)).toBe(1);
  });
  it('tailMaskはW%32==0で-1(全ビット有効)、それ以外は下位ビットのみ', () => {
    expect(tailMask(32)).toBe(-1);
    expect(tailMask(96)).toBe(-1);
    expect(tailMask(5)).toBe(0b11111);
    expect(tailMask(33)).toBe(0b1);
    expect(tailMask(1)).toBe(0b1);
  });
});

// ============================================================
// 1. packPlanes / unpackPlanes ラウンドトリップ
// ============================================================

describe('packPlanes / unpackPlanes ラウンドトリップ', () => {
  const widths = [1, 5, 17, 24, 31, 32, 33, 48, 96];
  for (const W of widths) {
    for (const [C, H] of [[1, 1], [2, 3], [3, 5]] as [number, number][]) {
      it(`C=${C} H=${H} W=${W}: pack→unpackで元のbitと完全一致`, () => {
        const batch = 2;
        const rand = rng(C * 1000 + H * 100 + W);
        const data = randomBits(batch * C * H * W, rand);
        const nv: NodeValue = { data, width: C * H * W };
        const bp = packPlanes(nv, batch, C, H, W);
        expect(bp.wpr).toBe(wordsPerRow(W));
        expect(bp.words.length).toBe(batch * C * H * bp.wpr);
        const back = unpackPlanes(bp, batch);
        expect(back.width).toBe(C * H * W);
        expect(Array.from(back.data)).toEqual(Array.from(data));
      });
    }
  }

  // packPlanes は Uint8Array 入力かつ W%4===0 のとき4バイト同時の
  // 高速路(packRowU8)へ入る。素朴路(Int32Array 経由)との整数完全一致を、
  // 0/1 だけでなく **0/1以外のバイト値**(非ゼロ→1 の意味論)まで含めて固定する。
  describe('Uint8Array高速路 ≡ 素朴路', () => {
    for (const W of [4, 32, 64, 96, 128]) {
      for (const [C, H] of [[1, 1], [3, 5], [7, 9]] as [number, number][]) {
        it(`C=${C} H=${H} W=${W}: 0/1入力で完全一致`, () => {
          const batch = 2;
          const rand = rng(C * 7919 + H * 31 + W);
          const n = batch * C * H * W;
          const u8 = new Uint8Array(n);
          for (let i = 0; i < n; i++) u8[i] = rand() > 0.5 ? 1 : 0;
          const i32 = Int32Array.from(u8);
          const fast = packPlanes({ data: u8, width: C * H * W }, batch, C, H, W);
          const slow = packPlanes({ data: i32, width: C * H * W }, batch, C, H, W);
          expect(Array.from(fast.words)).toEqual(Array.from(slow.words));
        });
      }
    }

    it('0/1以外のバイト値(非ゼロ→1)でも完全一致', () => {
      const [C, H, W] = [5, 7, 96];
      const batch = 2;
      const rand = rng(20260803);
      const n = batch * C * H * W;
      const u8 = new Uint8Array(n);
      // 0 / 1 / 0x7f / 0x80 / 0xff の境界値を必ず混ぜる(bit7 の扱いの検証)。
      const vals = [0, 1, 2, 0x7f, 0x80, 0x81, 0xfe, 0xff];
      for (let i = 0; i < n; i++) u8[i] = vals[Math.floor(rand() * vals.length)];
      const i32 = Int32Array.from(u8);
      const fast = packPlanes({ data: u8, width: C * H * W }, batch, C, H, W);
      const slow = packPlanes({ data: i32, width: C * H * W }, batch, C, H, W);
      expect(Array.from(fast.words)).toEqual(Array.from(slow.words));
    });

    it('byteOffsetがずれたsubarrayでは素朴路へ落ちる(結果は同一)', () => {
      const [C, H, W] = [2, 3, 32];
      const n = C * H * W;
      const buf = new Uint8Array(n + 1);
      for (let i = 0; i < buf.length; i++) buf[i] = i % 3 === 0 ? 1 : 0;
      const shifted = buf.subarray(1); // byteOffset=1(4バイト境界外)
      const fast = packPlanes({ data: shifted, width: n }, 1, C, H, W);
      const slow = packPlanes({ data: Int32Array.from(shifted), width: n }, 1, C, H, W);
      expect(Array.from(fast.words)).toEqual(Array.from(slow.words));
    });
  });

  it('端数ビット(x>=W)は常に0(不変条件)', () => {
    const C = 2;
    const H = 3;
    const W = 5; // wpr=1, tail bits 5..31 must be 0
    const batch = 3;
    const rand = rng(42);
    const data = new Int32Array(batch * C * H * W).fill(1); // 全部1で埋める
    for (let i = 0; i < data.length; i++) data[i] = rand() > 0.3 ? 1 : 0;
    const nv: NodeValue = { data, width: C * H * W };
    const bp = packPlanes(nv, batch, C, H, W);
    const mask = tailMask(W);
    for (let i = 0; i < bp.words.length; i++) {
      expect(bp.words[i] & ~mask, `word[${i}]の端数ビットは0のはず`).toBe(0);
    }
  });
});

// ============================================================
// 2. shiftRowInto: スカラー参照実装と全域照合
// ============================================================

function shiftRowScalar(src: Int32Array, wpr: number, s: number): Int32Array {
  const total = wpr * 32;
  const out = new Int32Array(wpr);
  for (let x = 0; x < total; x++) {
    const sx = x + s;
    let bit = 0;
    if (sx >= 0 && sx < total) bit = (src[sx >>> 5] >>> (sx & 31)) & 1;
    if (bit) out[x >>> 5] |= 1 << (x & 31);
  }
  return out;
}

describe('shiftRowInto: スカラー参照実装との全域照合(s=-33..33)', () => {
  for (const wpr of [1, 2, 3, 4]) {
    it(`wpr=${wpr}: ランダム行 x s in [-33,33]`, () => {
      const rand = rng(wpr * 7 + 1);
      for (let trial = 0; trial < 6; trial++) {
        const src = new Int32Array(wpr);
        for (let j = 0; j < wpr; j++) src[j] = (rand() * 0x100000000) | 0;
        for (let s = -33; s <= 33; s++) {
          const dst = new Int32Array(wpr);
          shiftRowInto(dst, src, wpr, s);
          const want = shiftRowScalar(src, wpr, s);
          expect(Array.from(dst), `wpr=${wpr} s=${s} trial=${trial}`).toEqual(Array.from(want));
        }
      }
    });
  }

  it('s=0はコピー専用分岐でも一致', () => {
    const src = Int32Array.from([0x12345678, -1, 0]);
    const dst = new Int32Array(3);
    shiftRowInto(dst, src, 3, 0);
    expect(Array.from(dst)).toEqual(Array.from(src));
  });

  it('全1行のシフトも境界含め正しい(端の0埋めを確認)', () => {
    const wpr = 2;
    const src = Int32Array.from([-1, -1]);
    for (let s = -40; s <= 40; s++) {
      const dst = new Int32Array(wpr);
      shiftRowInto(dst, src, wpr, s);
      expect(Array.from(dst), `s=${s}`).toEqual(Array.from(shiftRowScalar(src, wpr, s)));
    }
  });

  it('dstOff/srcOffを使った大きい共有バッファ内の1行シフトも一致', () => {
    const wpr = 2;
    const rand = rng(99);
    const buf = new Int32Array(20);
    for (let i = 0; i < buf.length; i++) buf[i] = (rand() * 0x100000000) | 0;
    const rowOff = 6;
    const src = buf.slice(rowOff, rowOff + wpr);
    const dstBuf = new Int32Array(20);
    const dstOff = 10;
    for (let s = -35; s <= 35; s++) {
      shiftRowInto(dstBuf, buf, wpr, s, dstOff, rowOff);
      const want = shiftRowScalar(src, wpr, s);
      expect(Array.from(dstBuf.subarray(dstOff, dstOff + wpr)), `s=${s}`).toEqual(Array.from(want));
    }
  });
});

// ============================================================
// 3. buildTapPlaneBitplane(stride=1高速路) vs 素朴gather
// ============================================================

function naiveTapPlaneBits(
  xBits: Int32Array, H: number, W: number, HOut: number, WOut: number,
  dy: number, dx: number, pH: number, pW: number,
): Int32Array {
  // sH=sW=1 前提の素朴gather(packedConv.ts::buildTapPlaneと同じ幾何、
  // flatではなくrow×WOutの2D配列として返す)
  const out = new Int32Array(HOut * WOut);
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy - pH + dy;
    if (y < 0 || y >= H) continue;
    for (let ox = 0; ox < WOut; ox++) {
      const xx = ox - pW + dx;
      if (xx < 0 || xx >= W) continue;
      out[oy * WOut + ox] = xBits[y * W + xx];
    }
  }
  return out;
}

function bitplaneRowsToFlat(words: Int32Array, HOut: number, wpr: number, WOut: number): Int32Array {
  const out = new Int32Array(HOut * WOut);
  for (let oy = 0; oy < HOut; oy++) {
    for (let ox = 0; ox < WOut; ox++) {
      out[oy * WOut + ox] = (words[oy * wpr + (ox >>> 5)] >>> (ox & 31)) & 1;
    }
  }
  return out;
}

describe('buildTapPlaneBitplane: stride=1高速路(WOut===W)の素朴gather照合', () => {
  const cases = [
    { H: 4, W: 4, pH: 1, pW: 1 }, // 3x3 pad1相当のタップ位置を想定
    { H: 5, W: 33, pH: 1, pW: 1 }, // wpr=2境界
    { H: 6, W: 96, pH: 1, pW: 1 }, // wpr=3境界
    { H: 3, W: 7, pH: 0, pW: 0 },
    { H: 4, W: 17, pH: 2, pW: 2 },
  ];
  for (const c of cases) {
    // stride=1 same-padding: HOut=H, WOut=W (kernel=2p+1)
    const HOut = c.H;
    const WOut = c.W;
    for (const [dy, dx] of [[0, 0], [0, 1], [1, 0], [2 * c.pH, 2 * c.pW], [0, 2 * c.pW]]) {
      it(`H=${c.H} W=${c.W} pH=${c.pH} pW=${c.pW} dy=${dy} dx=${dx}`, () => {
        const rand = rng(c.H * 31 + c.W * 7 + dy * 3 + dx);
        const xBits = randomBits(c.H * c.W, rand);
        const wpr = wordsPerRow(c.W);
        const xWords = new Int32Array(c.H * wpr);
        for (let y = 0; y < c.H; y++) {
          for (let x = 0; x < c.W; x++) {
            if (xBits[y * c.W + x]) xWords[y * wpr + (x >>> 5)] |= 1 << (x & 31);
          }
        }
        const dst = new Int32Array(HOut * wpr);
        buildTapPlaneBitplane(xWords, 0, c.H, wpr, HOut, dy, dx, c.pH, c.pW, dst, 0);
        const got = bitplaneRowsToFlat(dst, HOut, wpr, WOut);
        const want = naiveTapPlaneBits(xBits, c.H, c.W, HOut, WOut, dy, dx, c.pH, c.pW);
        expect(Array.from(got)).toEqual(Array.from(want));
      });
    }
  }
});

// ============================================================
// 3.5 buildSharedShiftPlane: 行ごと shiftRowInto の参照実装と全域照合
// ============================================================

/** 最適化前の実装(行ごとに shiftRowInto を呼ぶ)。C5 で本体を連続ワード列の
    直線ループ+行境界フィックスアップへ置き換えたため、この素朴版を参照として
    残し「語単位で完全一致」を固定する(整数出力不変が契約)。 */
function sharedShiftPlaneRowwise(
  xWords: Int32Array, srcBase: number,
  H: number, wpr: number, pH: number, s: number,
  dst: Int32Array, dstBase: number,
): void {
  const nRows = H + 2 * pH;
  for (let r = 0; r < nRows; r++) {
    const y = r - pH;
    const outRowOff = dstBase + r * wpr;
    if (y < 0 || y >= H) {
      for (let j = 0; j < wpr; j++) dst[outRowOff + j] = 0;
    } else {
      shiftRowInto(dst, xWords, wpr, s, outRowOff, srcBase + y * wpr);
    }
  }
}

describe('buildSharedShiftPlane: 行ごとshiftRowInto版と語単位で完全一致', () => {
  // wpr=1(全語が行境界)/2/3/4、pH=0..2、シフトは s=dx-pW の実域を超えて広めに。
  for (const wpr of [1, 2, 3, 4]) {
    for (const pH of [0, 1, 2]) {
      it(`wpr=${wpr} pH=${pH}: s=-35..35 × ランダム入力`, () => {
        const rand = rng(wpr * 101 + pH * 13 + 5);
        const H = 5;
        for (let trial = 0; trial < 3; trial++) {
          // 大きな共有バッファの途中に置いて srcBase/dstBase 経路も踏む
          const srcBase = wpr * 2;
          const xWords = new Int32Array(srcBase + H * wpr + wpr);
          for (let i = 0; i < xWords.length; i++) xWords[i] = (rand() * 0x100000000) | 0;
          const planeWords = (H + 2 * pH) * wpr;
          const dstBase = 3;
          for (let s = -35; s <= 35; s++) {
            const got = new Int32Array(dstBase + planeWords + 3).fill(0x5a5a5a5a);
            const want = new Int32Array(dstBase + planeWords + 3).fill(0x5a5a5a5a);
            buildSharedShiftPlane(xWords, srcBase, H, wpr, pH, s, got, dstBase);
            sharedShiftPlaneRowwise(xWords, srcBase, H, wpr, pH, s, want, dstBase);
            expect(Array.from(got), `wpr=${wpr} pH=${pH} s=${s} trial=${trial}`)
              .toEqual(Array.from(want));
          }
        }
      });
    }
  }

  it('平面の外(dstBaseの前後)へ書き込まない', () => {
    const wpr = 3;
    const H = 4;
    const pH = 1;
    const rand = rng(777);
    const xWords = new Int32Array(H * wpr);
    for (let i = 0; i < xWords.length; i++) xWords[i] = (rand() * 0x100000000) | 0;
    const dstBase = 5;
    const planeWords = (H + 2 * pH) * wpr;
    for (const s of [-33, -1, 0, 1, 33]) {
      const dst = new Int32Array(dstBase + planeWords + 4).fill(0x1234abcd);
      buildSharedShiftPlane(xWords, 0, H, wpr, pH, s, dst, dstBase);
      for (let i = 0; i < dstBase; i++) expect(dst[i], `前ガード i=${i} s=${s}`).toBe(0x1234abcd);
      for (let i = dstBase + planeWords; i < dst.length; i++) {
        expect(dst[i], `後ガード i=${i} s=${s}`).toBe(0x1234abcd);
      }
    }
  });

  it('タップ (dy,dx) の読み出しが buildTapPlaneBitplane と一致する(共有平面の意味)', () => {
    // 共有平面の行 r=oy+dy から HOut 行読む = (dy,dx) タップ平面、という C1 の
    // 主張そのものを固定する(C5 の実装置き換え後も成り立つこと)。
    const H = 6;
    const W = 33; // wpr=2 で端数ビットあり
    const wpr = wordsPerRow(W);
    const pH = 1;
    const pW = 1;
    const HOut = H;
    const rand = rng(2024);
    const xWords = new Int32Array(H * wpr);
    const mask = tailMask(W);
    for (let y = 0; y < H; y++) {
      for (let j = 0; j < wpr; j++) xWords[y * wpr + j] = (rand() * 0x100000000) | 0;
      xWords[y * wpr + wpr - 1] &= mask; // 端数ビット0の不変条件
    }
    for (let dx = 0; dx <= 2 * pW; dx++) {
      const plane = new Int32Array((H + 2 * pH) * wpr);
      buildSharedShiftPlane(xWords, 0, H, wpr, pH, dx - pW, plane, 0);
      for (let dy = 0; dy <= 2 * pH; dy++) {
        const want = new Int32Array(HOut * wpr);
        buildTapPlaneBitplane(xWords, 0, H, wpr, HOut, dy, dx, pH, pW, want, 0);
        const got = plane.subarray(dy * wpr, dy * wpr + HOut * wpr);
        expect(Array.from(got), `dy=${dy} dx=${dx}`).toEqual(Array.from(want));
      }
    }
  });
});

// ============================================================
// 4. buildTapPlaneStrided(汎用パス) vs 素朴gather(stride>1含む)
// ============================================================

function naiveTapPlaneBitsGeneral(
  xBits: Int32Array, H: number, W: number, HOut: number, WOut: number,
  dy: number, dx: number, sH: number, sW: number, pH: number, pW: number,
): Int32Array {
  const out = new Int32Array(HOut * WOut);
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy * sH - pH + dy;
    if (y < 0 || y >= H) continue;
    for (let ox = 0; ox < WOut; ox++) {
      const xx = ox * sW - pW + dx;
      if (xx < 0 || xx >= W) continue;
      out[oy * WOut + ox] = xBits[y * W + xx];
    }
  }
  return out;
}

describe('buildTapPlaneStrided: 汎用パス(stride>1・WOut!==W)の素朴gather照合', () => {
  const cases = [
    { H: 7, W: 7, sH: 2, sW: 2, pH: 0, pW: 0, kH: 3, kW: 3 },
    { H: 9, W: 5, sH: 2, sW: 1, pH: 1, pW: 0, kH: 3, kW: 3 },
    { H: 6, W: 33, sH: 2, sW: 2, pH: 1, pW: 1, kH: 3, kW: 3 },
  ];
  for (const c of cases) {
    const HOut = Math.floor((c.H + 2 * c.pH - c.kH) / c.sH) + 1;
    const WOut = Math.floor((c.W + 2 * c.pW - c.kW) / c.sW) + 1;
    for (let dy = 0; dy < c.kH; dy++) {
      for (let dx = 0; dx < c.kW; dx++) {
        it(`H=${c.H} W=${c.W} s=(${c.sH},${c.sW}) pad=(${c.pH},${c.pW}) dy=${dy} dx=${dx}`, () => {
          const rand = rng(c.H * 13 + c.W * 5 + dy * 3 + dx + 1);
          const xBits = randomBits(c.H * c.W, rand);
          const wprIn = wordsPerRow(c.W);
          const xWords = new Int32Array(c.H * wprIn);
          for (let y = 0; y < c.H; y++) {
            for (let x = 0; x < c.W; x++) {
              if (xBits[y * c.W + x]) xWords[y * wprIn + (x >>> 5)] |= 1 << (x & 31);
            }
          }
          const wprOut = wordsPerRow(WOut);
          const dst = new Int32Array(HOut * wprOut);
          buildTapPlaneStrided(
            xWords, 0, c.H, c.W, wprIn, dy, dx, HOut, WOut, wprOut,
            c.sH, c.sW, c.pH, c.pW, dst, 0,
          );
          const got = bitplaneRowsToFlat(dst, HOut, wprOut, WOut);
          const want = naiveTapPlaneBitsGeneral(xBits, c.H, c.W, HOut, WOut, dy, dx, c.sH, c.sW, c.pH, c.pW);
          expect(Array.from(got)).toEqual(Array.from(want));
        });
      }
    }
  }
});

// ============================================================
// 5. orPool2x2: 2実装の相互一致 + 素朴OR-poolとの照合
// ============================================================

function naiveOrPool2x2(bits: Int32Array, C: number, H: number, W: number): {
  data: Int32Array; HOut: number; WOut: number;
} {
  const HOut = ((H - 2) >> 1) + 1;
  const WOut = ((W - 2) >> 1) + 1;
  const out = new Int32Array(C * HOut * WOut);
  for (let c = 0; c < C; c++) {
    for (let oy = 0; oy < HOut; oy++) {
      for (let ox = 0; ox < WOut; ox++) {
        let v = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            v |= bits[(c * H + 2 * oy + dy) * W + (2 * ox + dx)];
          }
        }
        out[(c * HOut + oy) * WOut + ox] = v;
      }
    }
  }
  return { data: out, HOut, WOut };
}

function bitplaneToFlat(bp: BitplaneValue, batch: number): Int32Array {
  return unpackPlanes(bp, batch).data;
}

describe('spreadEven16はcompressEven32の逆演算', () => {
  it('compress(spread(v)) === v (v<0x10000)', () => {
    const rand = rng(5);
    for (let i = 0; i < 500; i++) {
      const v = (rand() * 0x10000) | 0;
      expect(compressEven32LUT(spreadEven16(v))).toBe(v);
      expect(compressEven32Shift(spreadEven16(v))).toBe(v);
    }
  });
});

describe('compressEven32: LUT版とShift版の相互一致(全32bit網羅は重いので広くランダム+境界)', () => {
  it('相互一致(ランダム+全0/全1/交互パターン)', () => {
    const rand = rng(7);
    const samples: number[] = [0, -1, 0x55555555, 0xaaaaaaaa | 0, 0x0f0f0f0f, 0xf0f0f0f0 | 0];
    for (let i = 0; i < 2000; i++) samples.push((rand() * 0x100000000) | 0);
    for (const v of samples) {
      expect(compressEven32Shift(v), `v=${v}`).toBe(compressEven32LUT(v));
    }
  });
  it('参照(1ビットずつ抽出)とも一致', () => {
    const rand = rng(11);
    for (let i = 0; i < 500; i++) {
      const v = (rand() * 0x100000000) | 0;
      let want = 0;
      for (let i2 = 0; i2 < 16; i2++) want |= (((v >>> (2 * i2)) & 1) << i2);
      expect(compressEven32LUT(v)).toBe(want);
      expect(compressEven32Shift(v)).toBe(want);
    }
  });
});

describe('orPool2x2: 素朴OR-poolとの照合(LUT/Shift両実装) + 端数幅', () => {
  const cases = [
    { C: 1, H: 4, W: 4 },
    { C: 2, H: 6, W: 6 },
    { C: 3, H: 5, W: 5 }, // 奇数(端数行/列)
    { C: 1, H: 7, W: 33 }, // wpr境界+端数
    { C: 2, H: 9, W: 65 },
  ];
  for (const c of cases) {
    for (const impl of [
      { name: 'LUT', fn: compressEven32LUT },
      { name: 'Shift', fn: compressEven32Shift },
    ]) {
      it(`[${impl.name}] C=${c.C} H=${c.H} W=${c.W}`, () => {
        const batch = 2;
        const rand = rng(c.C * 100 + c.H * 10 + c.W);
        const data = randomBits(batch * c.C * c.H * c.W, rand);
        const nv: NodeValue = { data, width: c.C * c.H * c.W };
        const bp = packPlanes(nv, batch, c.C, c.H, c.W);
        const pooled = orPool2x2(bp, batch, impl.fn);
        const got = bitplaneToFlat(pooled, batch);
        for (let b = 0; b < batch; b++) {
          const bits = data.subarray(b * c.C * c.H * c.W, (b + 1) * c.C * c.H * c.W);
          const want = naiveOrPool2x2(bits, c.C, c.H, c.W);
          expect(pooled.H).toBe(want.HOut);
          expect(pooled.W).toBe(want.WOut);
          const gotB = got.subarray(b * c.C * want.HOut * want.WOut, (b + 1) * c.C * want.HOut * want.WOut);
          expect(Array.from(gotB), `batch ${b}`).toEqual(Array.from(want.data));
        }
      });
    }
  }
});

// ============================================================
// 6. dilateX2: evalUpsample2dPacked(旧実装)との照合
// ============================================================

describe('dilateX2: evalUpsample2dPacked(旧実装)との照合', () => {
  const cases = [
    { C: 1, H: 3, W: 3 },
    { C: 2, H: 4, W: 5 },
    { C: 3, H: 5, W: 17 }, // wpr境界
    { C: 1, H: 2, W: 32 },
    { C: 2, H: 3, W: 33 },
  ];
  for (const c of cases) {
    it(`C=${c.C} H=${c.H} W=${c.W}`, () => {
      const batch = 2;
      const rand = rng(c.C * 50 + c.H * 5 + c.W + 3);
      const data = randomBits(batch * c.C * c.H * c.W, rand);
      const nv: NodeValue = { data, width: c.C * c.H * c.W };
      const bp = packPlanes(nv, batch, c.C, c.H, c.W);
      const up = dilateX2(bp, batch);
      const got = unpackPlanes(up, batch).data;

      const params: UpSample2dParams = { in_shape: [c.C, c.H, c.W], scale: 2 };
      const want = evalUpsample2dPacked(params, nv, batch);
      expect(up.H).toBe(c.H * 2);
      expect(up.W).toBe(c.W * 2);
      expect(Array.from(got)).toEqual(Array.from(want.data));
    });
  }
});

// ============================================================
// 7. copyPlanes: チャネル選択コピー
// ============================================================

describe('copyPlanes: チャネル並べ替え/選択が素朴実装と一致', () => {
  it('チャネル逆順+重複選択', () => {
    const C = 4;
    const H = 3;
    const W = 5;
    const batch = 2;
    const rand = rng(123);
    const data = randomBits(batch * C * H * W, rand);
    const nv: NodeValue = { data, width: C * H * W };
    const bp = packPlanes(nv, batch, C, H, W);
    const order = [3, 0, 0, 2];
    const out = copyPlanes(bp, batch, order);
    const got = unpackPlanes(out, batch).data;
    for (let b = 0; b < batch; b++) {
      for (let o = 0; o < order.length; o++) {
        const srcC = order[o];
        for (let i = 0; i < H * W; i++) {
          const want = data[b * C * H * W + srcC * H * W + i];
          const g = got[b * order.length * H * W + o * H * W + i];
          expect(g).toBe(want);
        }
      }
    }
  });
});

// ============================================================
// concatPlanes: チャネル軸concatの平面連結
// ============================================================

describe('concatPlanes: 非パックconcat(evalConcat)と整数完全一致', () => {
  // W=17/33 はワード境界の端数ケース(端数ビットは常に0の不変条件)。
  for (const [H, W] of [[3, 17], [2, 33], [4, 32], [1, 5]] as const) {
    it(`H=${H} W=${W} の3入力(C=1/3/2)`, () => {
      const batch = 2;
      const Cs = [1, 3, 2];
      const rand = rng(4242 + W);
      const nvs: NodeValue[] = Cs.map((C) => ({
        data: randomBits(batch * C * H * W, rand),
        width: C * H * W,
      }));
      const bps = nvs.map((nv, i) => packPlanes(nv, batch, Cs[i], H, W));
      const got = unpackPlanes(concatPlanes(bps, batch), batch);
      const want = evalConcat(nvs, batch);
      expect(got.width).toBe(want.width);
      expect(Array.from(got.data)).toEqual(Array.from(want.data));
    });
  }
});

// ============================================================
// spatialGroupSumBitplane: 読み出しヘッドの融合ネイティブ評価
// ============================================================

/** 参照実装: irflat/packedConv の既存2ノード評価(permute → group_sum)を
    そのまま繋いだもの。これが意味論の基準(golden側の simulate.py と
    整数一致していることは conv.test.ts / seg_e2e.test.ts が保証済み)。 */
function refSpatialGroupSum(
  nv: NodeValue, batch: number, nOut: number, gs: number, H: number, W: number,
): NodeValue {
  const permuted = evalPermutePacked(
    { in_shape: [nOut, gs, H, W], perm: [0, 2, 3, 1] }, nv, batch,
  );
  return evalGroupSum(
    { in_width: nOut * gs * H * W, k: nOut * H * W, tau: 1 }, permuted, batch,
  );
}

describe('spatialGroupSumBitplane: permute+group_sum の既存経路と整数完全一致', () => {
  // gs: 1(平面1枚)/ 3・5(非2冪)/ 16(det実モデル)/ 32(seg実モデル)。
  // W: 17/33 はワード境界端数、32/64 はぴったり境界。
  const cases: [number, number, number, number][] = [
    // [nOut, gs, H, W]
    [1, 1, 3, 17],
    [2, 1, 2, 33],
    [3, 3, 4, 17],
    [2, 5, 3, 33],
    [2, 7, 2, 5],
    [1, 16, 5, 24],
    [10, 16, 3, 24],
    [2, 32, 3, 64],
    [2, 32, 2, 33],
    [1, 31, 2, 32],
  ];
  for (const [nOut, gs, H, W] of cases) {
    it(`nOut=${nOut} gs=${gs} H=${H} W=${W}`, () => {
      const batch = 2;
      const C = nOut * gs;
      const rand = rng(97 * gs + W + nOut);
      const nv: NodeValue = {
        data: randomBits(batch * C * H * W, rand),
        width: C * H * W,
      };
      const bp = packPlanes(nv, batch, C, H, W);
      const got = spatialGroupSumBitplane(bp, batch, nOut, gs);
      const want = refSpatialGroupSum(nv, batch, nOut, gs, H, W);
      expect(got.width).toBe(nOut * H * W);
      expect(got.width).toBe(want.width);
      expect(Array.from(got.data)).toEqual(Array.from(want.data));
    });
  }

  it('全1入力で各スコアが gs に一致する(繰り上がりの最上位平面まで検証)', () => {
    for (const gs of [1, 2, 3, 16, 31, 32]) {
      const nOut = 2;
      const H = 3;
      const W = 33;
      const C = nOut * gs;
      const data = new Int32Array(C * H * W).fill(1);
      const bp = packPlanes({ data, width: C * H * W }, 1, C, H, W);
      const got = spatialGroupSumBitplane(bp, 1, nOut, gs);
      expect(Array.from(got.data)).toEqual(new Array(nOut * H * W).fill(gs));
    }
  });
});
