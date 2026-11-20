// verifyCore.ts::verifyBundleCases のテスト(frozen contractの意味論)。
//
// 契約:
//   - bits_b64(packed LSB-first)/image_b64(raw uint8 CHW)のデコードが
//     js engine の直接実行と完全一致(pack/unpack往復)
//   - 全ケース一致なら ok=true・maxAbsDiff=0
//   - 1箇所でも不一致なら ok=false・error に「最初の不一致」の
//     case index/output index/expected/actualが入る
//   - 構造不正(mode不一致のフィールド欠落・image_b64長不正)は例外
//   - strictPrograms は常に true に固定される(opts側の指定を上書きする。
//     verify-webはフォールバックが許されない出荷ゲートのため)
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { loadBundle } from './lib';
import type { VerifyCaseFile } from './verifyCore';
import { verifyBundleCases } from './verifyCore';

const here = dirname(fileURLToPath(import.meta.url));
const bundleGoldenPath = join(here, '../fixtures/bundle_golden.json');
const hasBundleGolden = existsSync(bundleGoldenPath);

interface GoldenBundleDoc {
  ir: { inputs: { width: number }[] };
  demo_samples?: { height: number; width: number; rgb_b64: string }[];
  [k: string]: unknown;
}

function loadGoldenBundleDoc(): GoldenBundleDoc {
  const golden = JSON.parse(readFileSync(bundleGoldenPath, 'utf8')) as { bundle_doc: unknown };
  return JSON.parse(JSON.stringify(golden.bundle_doc)) as GoldenBundleDoc;
}

/** 決定的な擬似乱数(mulberry32)。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomBits(n: number, seed: number): Uint8Array {
  const rand = rng(seed);
  const bits = new Uint8Array(n);
  for (let i = 0; i < n; i++) bits[i] = rand() < 0.5 ? 1 : 0;
  return bits;
}

/** 0/1配列 → packed LSB-first base64(frozen contractの逆写像。
    verifyCore.ts内部のdecodeBitsB64と対になる、conv.test.ts::b64ToBitsの逆)。 */
function bitsToB64(bits: Uint8Array): string {
  const nBytes = Math.ceil(bits.length / 8);
  const bytes = new Uint8Array(nBytes);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) bytes[i >> 3] |= 1 << (i & 7);
  }
  let raw = '';
  for (const b of bytes) raw += String.fromCharCode(b);
  return btoa(raw);
}

/** HWC uint8 [h*w*3] → CHW uint8 base64(image_b64契約)。 */
function hwcToChwB64(rgbHwc: Uint8Array, h: number, w: number): string {
  const chw = new Uint8Array(3 * h * w);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < h * w; i++) chw[c * h * w + i] = rgbHwc[i * 3 + c];
  }
  let raw = '';
  for (const b of chw) raw += String.fromCharCode(b);
  return btoa(raw);
}

function bytesFromB64(b64: string): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

describe.skipIf(!hasBundleGolden)('verifyBundleCases: mode=bits', () => {
  it('js engineで計算したexpectedに対してok=true・maxAbsDiff=0(bits_b64のpack/unpack往復)', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    const inWidth = bundle.runtime.inWidth;
    const cases = [11, 22, 33].map((seed) => {
      const bits = randomBits(inWidth, seed);
      const { data } = bundle.scores(bits, 1);
      return { bits_b64: bitsToB64(bits), expected: Array.from(data) };
    });
    const file: VerifyCaseFile = { mode: 'bits', cases };
    const report = await verifyBundleCases(doc, file, { engine: 'js' });
    expect(report.ok).toBe(true);
    expect(report.maxAbsDiff).toBe(0);
    expect(report.engine).toBe('js');
    expect(report.mode).toBe('bits');
    expect(report.outputs).toEqual(cases.map((c) => c.expected));
  });

  it('1箇所でも不一致ならok=falseで最初の不一致を報告する', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    const inWidth = bundle.runtime.inWidth;
    const bitsA = randomBits(inWidth, 1);
    const bitsB = randomBits(inWidth, 2);
    const expectedA = Array.from(bundle.scores(bitsA, 1).data);
    const expectedB = Array.from(bundle.scores(bitsB, 1).data);
    // 2件目(index=1)の出力先頭を意図的に壊す
    const badIdx = 0;
    const wrongValue = expectedB[badIdx] + 1;
    const perturbed = [...expectedB];
    perturbed[badIdx] = wrongValue;
    const file: VerifyCaseFile = {
      mode: 'bits',
      cases: [
        { bits_b64: bitsToB64(bitsA), expected: expectedA },
        { bits_b64: bitsToB64(bitsB), expected: perturbed },
      ],
    };
    const report = await verifyBundleCases(doc, file, { engine: 'js' });
    expect(report.ok).toBe(false);
    expect(report.error).toBe(
      `case[1] output[${badIdx}]: expected=${wrongValue} actual=${expectedB[badIdx]}`,
    );
    expect(report.maxAbsDiff).toBe(1);
  });

  it('mode=bitsでbits_b64が無いケースは例外', async () => {
    const doc = loadGoldenBundleDoc();
    const file: VerifyCaseFile = { mode: 'bits', cases: [{ expected: [0] }] };
    await expect(verifyBundleCases(doc, file, { engine: 'js' })).rejects.toThrow(/bits_b64 が無い/);
  });
});

describe.skipIf(!hasBundleGolden)('verifyBundleCases: mode=image', () => {
  it('demo_samplesのrgb_b64をCHWへ変換したimage_b64でscoresRgbaと一致する', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    const sample = doc.demo_samples?.[0];
    expect(sample, 'bundle_golden.jsonにdemo_samplesが無い').toBeDefined();
    const { height: h, width: w, rgb_b64: rgbB64 } = sample!;
    const rgbHwc = bytesFromB64(rgbB64);
    const rgba = new Uint8ClampedArray(h * w * 4);
    for (let i = 0; i < h * w; i++) {
      rgba[i * 4] = rgbHwc[i * 3];
      rgba[i * 4 + 1] = rgbHwc[i * 3 + 1];
      rgba[i * 4 + 2] = rgbHwc[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
    const expected = Array.from(bundle.scoresRgba!(rgba).data);
    const file: VerifyCaseFile = {
      mode: 'image',
      cases: [{ image_b64: hwcToChwB64(rgbHwc, h, w), expected }],
    };
    const report = await verifyBundleCases(doc, file, { engine: 'js' });
    expect(report.ok).toBe(true);
    expect(report.maxAbsDiff).toBe(0);
    expect(report.mode).toBe('image');
  });

  it('image_b64のバイト数がchannels×H×Wと違えば例外', async () => {
    const doc = loadGoldenBundleDoc();
    const short = btoa('abc');
    const file: VerifyCaseFile = { mode: 'image', cases: [{ image_b64: short, expected: [0] }] };
    await expect(verifyBundleCases(doc, file, { engine: 'js' })).rejects.toThrow(/バイト数/);
  });

  it('mode=imageでimage_b64が無いケースは例外', async () => {
    const doc = loadGoldenBundleDoc();
    const file: VerifyCaseFile = { mode: 'image', cases: [{ expected: [0] }] };
    await expect(verifyBundleCases(doc, file, { engine: 'js' })).rejects.toThrow(/image_b64 が無い/);
  });
});

describe.skipIf(!hasBundleGolden)('verifyBundleCases: strictProgramsは常にtrue', () => {
  it('opts.strictPrograms=falseを渡しても壊れたprogramsは例外になる', async () => {
    const doc = loadGoldenBundleDoc();
    const programs = doc.programs as Record<string, { instrs: number[] }>;
    const nodeId = Object.keys(programs)[0];
    programs[nodeId].instrs[0] = 99; // tt範囲外に破壊
    const file: VerifyCaseFile = { mode: 'bits', cases: [] };
    await expect(verifyBundleCases(doc, file, { engine: 'js', strictPrograms: false }))
      .rejects.toThrow(/tt=99/);
  });
});
