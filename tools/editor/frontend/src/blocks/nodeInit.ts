// ノード初期化の純関数群。
// lut_layer / conv2d の indices・truth_tables を、初期化プリセット
//   - 'random'       : ランダム窓 + ランダム真理値表
//   - 'pass-through' : 恒等(slot0パススルー、tt=0xaa..)。lutは幅維持で恒等
//   - 'residual-init': 恒等 + 微小摂動(一部ゲートだけランダム化)
// に従って生成する。生成物はすべて griffith_logic/ir/schema.py・validate.py の
// paramsスキーマに載る妥当な値で、hard/softの整合は真理値表を定数として
// 完全指定するため追加のバック処理を要しない(フロント純関数で完結)。

import type { IRNodeJson } from '../ir/schema';

export type InitPreset = 'random' | 'pass-through' | 'residual-init';

export const INIT_PRESETS: InitPreset[] = ['random', 'pass-through', 'residual-init'];

export const INIT_PRESET_LABEL: Record<InitPreset, string> = {
  random: 'ランダム',
  'pass-through': 'パススルー(恒等)',
  'residual-init': '残差初期化(恒等+微小摂動)',
};

// residual-init で各出力ゲートをランダム化する既定確率(微小摂動)。
const DEFAULT_PERTURB = 0.15;

/** 決定的PRNG(mulberry32)。seed固定でテスト・再現に使う。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** slot0 パススルー真理値表(出力 = 入力スロット0)。'0x' + 'a'×(2^n/4)。 */
export function passThroughTT(nInputs: number): string {
  const nibbles = (1 << nInputs) / 4;
  if (nibbles < 1) {
    throw new Error(`n_inputsは2以上が必要(2^n>=4): ${nInputs}`);
  }
  return '0x' + 'a'.repeat(nibbles);
}

/** ランダム真理値表(2^n ビット)を hex文字列で生成する(nに依らずBigInt不要)。 */
export function randomTT(nInputs: number, rng: () => number): string {
  const nibbles = (1 << nInputs) / 4;
  if (nibbles < 1) {
    throw new Error(`n_inputsは2以上が必要(2^n>=4): ${nInputs}`);
  }
  let s = '';
  for (let i = 0; i < nibbles; i++) {
    s += Math.floor(rng() * 16).toString(16);
  }
  return '0x' + s;
}

/** [lo, hi) から重複なく count 個を選ぶ(候補が足りなければ重複許容)。 */
function sampleWindow(lo: number, hi: number, count: number, rng: () => number): number[] {
  const span = hi - lo;
  const picked: number[] = [];
  if (span <= 0) return new Array<number>(count).fill(Math.max(0, lo));
  if (span >= count) {
    const seen = new Set<number>();
    while (picked.length < count) {
      const v = lo + Math.floor(rng() * span);
      if (!seen.has(v)) {
        seen.add(v);
        picked.push(v);
      }
    }
  } else {
    for (let i = 0; i < count; i++) picked.push(lo + Math.floor(rng() * span));
  }
  return picked;
}

export interface LutInit {
  indices: number[][]; // (n_inputs, out_width)
  truth_tables: string[]; // (out_width,)
}

/**
 * lut_layer の indices・truth_tables をプリセットに従って生成する。
 * - random: 出力ごとに入力幅全域からランダムに n_inputs 本選び、tt もランダム。
 * - pass-through: 出力 o は入力 (o % in_width) を slot0 で素通し(幅一致なら恒等)。
 * - residual-init: pass-through と同じ結線で、一部の出力だけ tt をランダム化。
 */
export function lutInit(
  inWidth: number,
  outWidth: number,
  nInputs: number,
  preset: InitPreset,
  rng: () => number,
  perturb = DEFAULT_PERTURB,
): LutInit {
  const indices: number[][] = Array.from({ length: nInputs }, () =>
    new Array<number>(outWidth),
  );
  const truth_tables = new Array<string>(outWidth);

  for (let o = 0; o < outWidth; o++) {
    if (preset === 'random') {
      const picks = sampleWindow(0, inWidth, nInputs, rng);
      for (let j = 0; j < nInputs; j++) indices[j][o] = picks[j];
      truth_tables[o] = randomTT(nInputs, rng);
    } else {
      // pass-through / residual-init: slot0 に対象入力、他スロットも同じ入力を張る
      const target = inWidth > 0 ? o % inWidth : 0;
      for (let j = 0; j < nInputs; j++) indices[j][o] = target;
      const noisy = preset === 'residual-init' && rng() < perturb;
      truth_tables[o] = noisy ? randomTT(nInputs, rng) : passThroughTT(nInputs);
    }
  }
  return { indices, truth_tables };
}

// --- conv2d ---

/** int または [h,w] を [h,w] へ正規化(validate.pyの_pairと同一)。 */
function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [Number(v[0]), Number(v[1])] : [Number(v), Number(v)];
}

export interface Conv2dInitSpec {
  inShape: [number, number, number]; // [C,H,W]
  outChannels: number;
  kernelSize: number | [number, number];
  stride: number | [number, number];
  padding: number | [number, number];
  treeDepth: number;
  nInputs: number;
}

/** conv2d の内部ノード数(木のノード総数)。validate.py と同一式。 */
export function convInternalNodeCount(nInputs: number, treeDepth: number): number {
  const leaves = nInputs ** treeDepth;
  return nInputs === 1 ? treeDepth : (leaves - 1) / (nInputs - 1);
}

/** conv2d の受容野サイズ(C*kH*kW = indices_flat の値域上限)。 */
export function convReceptiveField(spec: Conv2dInitSpec): number {
  const [c] = spec.inShape;
  const [kh, kw] = pair(spec.kernelSize);
  return c * kh * kw;
}

/**
 * conv2d の params(indices_flat・truth_tables 含む)をプリセットに従って生成する。
 * indices_flat は常に受容野(局所窓)からランダムに葉位置を選ぶ。プリセットは
 * 真理値表のみを規定する(pass-through=各内部ノードが slot0 を素通し=葉0を出力)。
 */
export function conv2dInit(
  spec: Conv2dInitSpec,
  preset: InitPreset,
  rng: () => number,
  perturb = DEFAULT_PERTURB,
): Record<string, unknown> {
  const { inShape, outChannels, treeDepth, nInputs } = spec;
  const [kh, kw] = pair(spec.kernelSize);
  const [sh, sw] = pair(spec.stride);
  const [ph, pw] = pair(spec.padding);
  const rf = convReceptiveField(spec);
  const nLeaves = nInputs ** treeDepth;
  const nInternal = convInternalNodeCount(nInputs, treeDepth);

  const indices_flat: number[][] = [];
  for (let oc = 0; oc < outChannels; oc++) {
    indices_flat.push(sampleWindow(0, rf, nLeaves, rng));
  }

  const truth_tables: string[] = [];
  for (let oc = 0; oc < outChannels; oc++) {
    for (let node = 0; node < nInternal; node++) {
      if (preset === 'random') {
        truth_tables.push(randomTT(nInputs, rng));
      } else {
        const noisy = preset === 'residual-init' && rng() < perturb;
        truth_tables.push(noisy ? randomTT(nInputs, rng) : passThroughTT(nInputs));
      }
    }
  }

  return {
    in_shape: [inShape[0], inShape[1], inShape[2]],
    out_channels: outChannels,
    kernel_size: [kh, kw],
    stride: [sh, sw],
    padding: [ph, pw],
    tree_depth: treeDepth,
    n_inputs: nInputs,
    indices_flat,
    truth_tables,
  };
}

/** conv2d ノードJSONを生成する(パレット/ダイアログ用)。 */
export function conv2dNode(
  id: string,
  spec: Conv2dInitSpec,
  preset: InitPreset,
  rng: () => number,
  position: { x: number; y: number },
): IRNodeJson {
  return {
    id,
    type: 'conv2d',
    // conv2dのsoftmaxは one-hot(popcount=1)真理値表しか表現できないため、
    // 任意の生成真理値表(ランダム/パススルー)を許容する 'independent' を使う。
    params: conv2dInit(spec, preset, rng),
    train_info: { parameterization: 'independent', connections: 'random', selection: 'soft' },
    meta: { editor: { position } },
  };
}

/** or_pool2d ノードJSONを生成する(paramsのみ。indices/ttは持たない)。 */
export function orPool2dNode(
  id: string,
  inShape: [number, number, number],
  kernelSize: number,
  stride: number,
  padding: number,
  position: { x: number; y: number },
): IRNodeJson {
  return {
    id,
    type: 'or_pool2d',
    params: {
      in_shape: [inShape[0], inShape[1], inShape[2]],
      kernel_size: kernelSize,
      stride,
      padding,
    },
    train_info: {},
    meta: { editor: { position } },
  };
}
