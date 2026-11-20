// GPU前処理(RGBA→ビット符号化)の数値経路とチャネル順の検証。
//
// ここで固定するのは「記述子表(desc/qval)+その解釈手順」がCPUエンコーダ
// (convpreproc.ts::encodeImageAuto、既に lowlevel_golden/thermo_golden で
// Python とビット一致が固定済み)と**1ビットも違わない**こと。
// simulatePreprocBits は WGSL 本体と同じ手順で同じ記述子を読むので、
// ここが緑なら残る不確実性は「WGSLの写経ミス」だけになる
// (それは wasmgpu_verify.html が packed planes の全ワード比較で潰す)。
//
// 特に重要なのが「u8量子化のもとで f64 のCPU演算を厳密整数比較へ置き換えられる」
// という設計前提で、これは総当りで確認する(edge: 256^2、curv: 256^3)。
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { encodeImageAuto, rgbaToNchw } from '../convpreproc';
import type { ImagePreprocDoc } from '../convpreproc';
import { buildPreprocDescriptors, simulatePreprocBits } from './gpuPreproc';

const here = dirname(fileURLToPath(import.meta.url));
const modelsDir = join(here, '../../public/models');

/** 決定的な擬似ランダムRGBA(wasmgpu_verify_main.ts と同じ生成器)。
    端(0/255)を必ず含める: 閾値境界と replicate padding の両方を踏ませるため。 */
function makeRgba(h: number, w: number, seed: number): Uint8ClampedArray {
  let s = seed;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const rgba = new Uint8ClampedArray(h * w * 4);
  for (let i = 0; i < h * w; i++) {
    rgba[i * 4] = (rnd() * 256) | 0;
    rgba[i * 4 + 1] = (rnd() * 256) | 0;
    rgba[i * 4 + 2] = (rnd() * 256) | 0;
    rgba[i * 4 + 3] = 255;
  }
  // 極端値(飽和・全黒)を規則的に混ぜる
  for (let i = 0; i < h * w; i += 7) {
    rgba[i * 4] = 0;
    rgba[i * 4 + 1] = 255;
    rgba[i * 4 + 2] = i % 2 === 0 ? 255 : 0;
  }
  return rgba;
}

function firstDiff(a: ArrayLike<number>, b: ArrayLike<number>, hw: number): string {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return `index=${i}(ch=${Math.floor(i / hw)}, px=${i % hw}) cpu=${a[i]} gpu=${b[i]}`;
    }
  }
  return '(差分なし)';
}

interface ModelDocLite {
  preproc: ImagePreprocDoc;
  ir: { inputs: { width: number }[] };
}

function loadPreprocs(): { file: string; p: ImagePreprocDoc; inputWidth: number }[] {
  const out: { file: string; p: ImagePreprocDoc; inputWidth: number }[] = [];
  for (const f of readdirSync(modelsDir)) {
    if (!f.endsWith('.json') || f === 'index.json') continue;
    let doc: ModelDocLite;
    try {
      doc = JSON.parse(readFileSync(join(modelsDir, f), 'utf8')) as ModelDocLite;
    } catch {
      continue;
    }
    const p = doc.preproc;
    if (!p || (p.kind !== 'image_thermometer' && p.kind !== 'image_lowlevel')) continue;
    out.push({ file: f, p, inputWidth: doc.ir.inputs[0].width });
  }
  return out;
}

const PREPROCS = loadPreprocs();

describe('gpuPreproc: 記述子がCPUエンコーダとビット完全一致', () => {
  it('配信中の画像モデルを全て拾えている(検証が空回りしていない)', () => {
    expect(PREPROCS.length).toBeGreaterThanOrEqual(5);
    expect(PREPROCS.some((m) => m.p.kind === 'image_lowlevel')).toBe(true);
    expect(PREPROCS.some((m) => m.p.kind === 'image_thermometer')).toBe(true);
  });

  for (const { file, p, inputWidth } of PREPROCS) {
    it(`${file}(${p.kind}, ${p.height}x${p.width})`, () => {
      const d = buildPreprocDescriptors(p);
      // チャネル数は IR の入力幅から決まる幾何と一致していなければならない
      // (ここがずれると「動くが別物を食わせている」壊れ方をする)。
      expect(d.channels * p.height * p.width).toBe(inputWidth);
      const hw = p.height * p.width;
      for (const seed of [20260804, 991, 7]) {
        const rgba = makeRgba(p.height, p.width, seed);
        const cpu = encodeImageAuto(rgbaToNchw(rgba, p.height, p.width), p);
        const sim = simulatePreprocBits(d, rgba);
        expect(sim.length).toBe(cpu.length);
        // toEqual は2.8M要素で遅いので、まず高速比較して差分時のみ詳細を出す。
        let same = true;
        for (let i = 0; i < cpu.length; i++) {
          if (cpu[i] !== sim[i]) { same = false; break; }
        }
        expect(same, `seed=${seed}: ${firstDiff(cpu, sim, hw)}`).toBe(true);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// ゴールデン(Python由来)のパラメータでも一致すること。
// 配信中のモデルは全て mean=0/std=1・閾値が2冪という「優しい」設定なので、
// thermo の LUT 経路(任意の f64 係数 (x-mean)/std と負の閾値)は実質未検証に
// なる。thermo_golden.json は mean=-0.049…/std=0.9598…/閾値が負を含むので、
// そこを踏ませる。画素値は u8 量子化する(GPU前処理の入力契約)。
// ---------------------------------------------------------------------------
interface ThermoGolden {
  shape: [number, number, number, number];
  mean: number[];
  std: number[];
  thresholds: number[][];
}
interface LowLevelGolden {
  shape: [number, number, number, number];
  mean: number[];
  std: number[];
  thresholds: number[][];
  edge_thresholds: number[];
  curv_thresholds: number[];
  chroma_thresholds: number[];
  edge_offsets: number[][];
  curv_offsets: number[][];
  chroma_pairs: number[][];
}

function randomRgbaFlat(h: number, w: number, seed: number): Uint8ClampedArray {
  return makeRgba(h, w, seed);
}

describe('gpuPreproc: ゴールデン由来パラメータ(非自明なmean/std・負閾値)でも一致', () => {
  const fx = join(here, '../fixtures');

  it('thermo_golden のパラメータ(mean/std≠0,1、負の閾値)', () => {
    const g = JSON.parse(
      readFileSync(join(fx, 'thermo_golden.json'), 'utf8'),
    ) as ThermoGolden;
    const [, , h, w] = g.shape;
    const p: ImagePreprocDoc = {
      kind: 'image_thermometer',
      mean: g.mean,
      std: g.std,
      thresholds: g.thresholds,
      height: h,
      width: w,
    };
    // 前提が効いていること(mean=0/std=1 だとLUT経路の検証にならない)
    expect(g.mean.some((v) => v !== 0)).toBe(true);
    expect(g.std.some((v) => v !== 1)).toBe(true);
    expect(g.thresholds.some((row) => row.some((v) => v < 0))).toBe(true);
    const d = buildPreprocDescriptors(p);
    for (const seed of [11, 22, 33]) {
      const rgba = randomRgbaFlat(h, w, seed);
      const cpu = encodeImageAuto(rgbaToNchw(rgba, h, w), p);
      expect(Array.from(simulatePreprocBits(d, rgba))).toEqual(Array.from(cpu));
    }
  });

  it('lowlevel_golden のパラメータ(配信モデルとは別のT/オフセット構成)', () => {
    const g = JSON.parse(
      readFileSync(join(fx, 'lowlevel_golden.json'), 'utf8'),
    ) as LowLevelGolden;
    const [, , h, w] = g.shape;
    const p: ImagePreprocDoc = {
      kind: 'image_lowlevel',
      mean: g.mean,
      std: g.std,
      thresholds: g.thresholds,
      edge_thresholds: g.edge_thresholds,
      curv_thresholds: g.curv_thresholds,
      chroma_thresholds: g.chroma_thresholds,
      edge_offsets: g.edge_offsets,
      curv_offsets: g.curv_offsets,
      chroma_pairs: g.chroma_pairs,
      height: h,
      width: w,
    };
    const d = buildPreprocDescriptors(p);
    // 6x7 のような小さい非2冪サイズ = replicate padding を全周で踏む
    for (const seed of [101, 202, 303]) {
      const rgba = randomRgbaFlat(h, w, seed);
      const cpu = encodeImageAuto(rgbaToNchw(rgba, h, w), p);
      expect(Array.from(simulatePreprocBits(d, rgba))).toEqual(Array.from(cpu));
    }
  });
});

describe('gpuPreproc: 固定小数化の総当り検証(設計前提の直接確認)', () => {
  const SCALE = 2 ** 31;
  const Q: number[] = [];
  for (let k = 0; k < 256; k++) Q.push(Math.fround(k / 255) * SCALE);

  it('I(k)=f32(k/255)*2^31 は 0..2^31 の厳密整数(u32に収まる)', () => {
    for (let k = 0; k < 256; k++) {
      expect(Number.isInteger(Q[k]), `k=${k}`).toBe(true);
      expect(Q[k]).toBeGreaterThanOrEqual(0);
      expect(Q[k]).toBeLessThanOrEqual(0xffffffff);
    }
    // qval 表そのものも同じ値であること
    const d = buildPreprocDescriptors(PREPROCS[0].p);
    for (let k = 0; k < 256; k++) expect(d.qval[k]).toBe(Q[k]);
  });

  // 1階差分(edge / chroma / 彩度が共有する形)。全 (k1,k2) ペア × 実閾値。
  it('1階差分: 全256^2ペアで f64比較と整数比較が一致', () => {
    const thresholds = [0.1, 0.3, 0.2, 0.15, 0.4, 0.05, 1 / 255, 0.5, 0.03125, 0];
    for (const t of thresholds) {
      const cutGe = Math.floor(t * SCALE) + 1;
      const cutLe = Math.ceil(-t * SCALE) - 1;
      for (let k1 = 0; k1 < 256; k1++) {
        for (let k2 = 0; k2 < 256; k2++) {
          const d = Math.fround(k1 / 255) - Math.fround(k2 / 255);
          const D = Q[k1] - Q[k2];
          if ((d > t) !== (D >= cutGe)) {
            throw new Error(`edge+ 不一致 t=${t} k=(${k1},${k2}) d=${d} D=${D}`);
          }
          if ((d < -t) !== (D <= cutLe)) {
            throw new Error(`edge- 不一致 t=${t} k=(${k1},${k2}) d=${d} D=${D}`);
          }
        }
      }
    }
  });

  // 2階差分(curv)。D の値域が 33bit へ伸びる唯一の形なので全3つ組で確認する。
  it('2階差分: 全256^3三つ組で f64比較と整数比較が一致', () => {
    const t = 0.2; // 配信中の全モデルの curv_thresholds
    const cutGe = Math.floor(t * SCALE) + 1;
    const cutLe = Math.ceil(-t * SCALE) - 1;
    for (let k1 = 0; k1 < 256; k1++) {
      for (let k2 = 0; k2 < 256; k2++) {
        const partial = Q[k1] - 2 * Q[k2];
        const dPartial = Math.fround(k1 / 255) - 2 * Math.fround(k2 / 255);
        for (let k3 = 0; k3 < 256; k3++) {
          const d = dPartial + Math.fround(k3 / 255);
          const D = partial + Q[k3];
          if ((d > t) !== (D >= cutGe)) {
            throw new Error(`curv+ 不一致 k=(${k1},${k2},${k3}) d=${d} D=${D}`);
          }
          if ((d < -t) !== (D <= cutLe)) {
            throw new Error(`curv- 不一致 k=(${k1},${k2},${k3}) d=${d} D=${D}`);
          }
        }
      }
    }
  }, 60_000);
});

describe('gpuPreproc: 未対応構成は黙らず throw', () => {
  const base = PREPROCS[0].p;
  it('未知の kind', () => {
    expect(() => buildPreprocDescriptors({ ...base, kind: 'image_raw' }))
      .toThrow(/未対応の preproc.kind/);
  });
  it('std<=0(単調性の前提崩れ)', () => {
    expect(() => buildPreprocDescriptors({ ...base, std: base.std.map(() => 0) }))
      .toThrow(/std/);
  });
  it('mean/std の長さ不一致', () => {
    expect(() => buildPreprocDescriptors({ ...base, std: [...base.std, 1] }))
      .toThrow(/長さ不一致/);
  });
});
