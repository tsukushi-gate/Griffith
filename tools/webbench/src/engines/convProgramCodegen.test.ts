// conv2d 命令列 → 直線JSコード生成(convProgramCodegen.ts)のテスト。
//
// 契約は「整数出力完全不変」なので、テストは全て
// **命令列インタプリタ(BitplaneProgramConvKernel の既定経路)との全数照合**で書く:
//   - ランダム木 property: 深さ/OC/タップ数/tt分布/幾何(stride/padding/W)を
//     振ったConv2dParamsを、ランダムなビットプレーン入力で全ワード比較
//   - 実5モデル(public/models): conv2dノード単位 + ConvRuntime のスコア全数比較
//   - CHUNK不変性: 生成の分割点を変えても出力ワードが1ビットも変わらない
//   - CSP(unsafe-eval禁止)相当: `new Function` が throw する環境では
//     黙って壊れず、note を残してインタプリタ経路へ戻る
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { Conv2dParams, IRGraphJson } from '../shared';
import { BitplaneProgramConvKernel, packPlanes } from './bitplane';
import type { BitplaneValue } from './bitplane';
import { ConvRuntime } from './conv';
import { compileConvProgram } from './convProgramCodegen';
import { lowerToPrimitives } from './lowerConv';

const here = dirname(fileURLToPath(import.meta.url));

/** 決定的な擬似乱数(convProgram.test.ts と同じ mulberry32)。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface RandomConvOpts {
  C: number; H: number; W: number; OC: number;
  kernel: number; stride: number; padding: number; depth: number;
  /** 'residual' は学習済みモデルに近い(パススルー/定数が多い)tt分布。 */
  dist: 'uniform' | 'residual';
  seed: number;
}

function randomConvParams(o: RandomConvOpts): Conv2dParams {
  const rand = rng(o.seed);
  const nLeaves = 1 << o.depth;
  const nInternal = nLeaves - 1;
  const tapPool = o.C * o.kernel * o.kernel;
  const indices_flat: number[][] = [];
  for (let oc = 0; oc < o.OC; oc++) {
    const row: number[] = [];
    for (let k = 0; k < nLeaves; k++) row.push(Math.floor(rand() * tapPool));
    indices_flat.push(row);
  }
  const truth_tables: string[] = [];
  for (let i = 0; i < o.OC * nInternal; i++) {
    let tt: number;
    if (o.dist === 'uniform') {
      tt = Math.floor(rand() * 16) & 0xf;
    } else {
      const r = rand();
      tt = r < 0.55 ? 0xa : r < 0.62 ? 0xc : r < 0.68 ? 0x0
        : r < 0.74 ? 0xf : r < 0.78 ? 0x5 : r < 0.82 ? 0x3 : Math.floor(rand() * 16) & 0xf;
    }
    truth_tables.push(`0x${tt.toString(16)}`);
  }
  return {
    in_shape: [o.C, o.H, o.W],
    out_channels: o.OC,
    kernel_size: o.kernel,
    stride: o.stride,
    padding: o.padding,
    tree_depth: o.depth,
    n_inputs: 2,
    indices_flat,
    truth_tables,
  };
}

/** ランダムな 0/1 ビットプレーン入力([batch][C][H][wpr])。 */
function randomBitplane(C: number, H: number, W: number, batch: number, seed: number): BitplaneValue {
  const rand = rng(seed);
  const bits = new Uint8Array(batch * C * H * W);
  for (let i = 0; i < bits.length; i++) bits[i] = rand() < 0.5 ? 1 : 0;
  return packPlanes({ data: bits, width: C * H * W }, batch, C, H, W);
}

function expectSameWords(a: BitplaneValue, b: BitplaneValue, label: string): void {
  expect(b.C, `${label}: C`).toBe(a.C);
  expect(b.H, `${label}: H`).toBe(a.H);
  expect(b.W, `${label}: W`).toBe(a.W);
  expect(b.words.length, `${label}: 長さ`).toBe(a.words.length);
  for (let i = 0; i < a.words.length; i++) {
    if (a.words[i] !== b.words[i]) {
      throw new Error(`${label}: word[${i}] interp=${a.words[i]} codegen=${b.words[i]}`);
    }
  }
}

describe('convProgramCodegen: ランダム木でインタプリタと整数完全一致', () => {
  const cases: RandomConvOpts[] = [
    // 高速路(stride1/pad1/W=kernelの中心)+ 端数ビットあり(W=37)/なし(W=32)
    { C: 3, H: 9, W: 37, OC: 24, kernel: 3, stride: 1, padding: 1, depth: 3, dist: 'residual', seed: 1 },
    { C: 4, H: 8, W: 32, OC: 32, kernel: 3, stride: 1, padding: 1, depth: 3, dist: 'uniform', seed: 2 },
    // 深い木 + 大きめOC(oc書き出しがチャンク分割を主導するケース)
    { C: 5, H: 6, W: 40, OC: 64, kernel: 3, stride: 1, padding: 1, depth: 4, dist: 'residual', seed: 3 },
    // 1x1 conv(タップがチャネル方向のみ)
    { C: 8, H: 5, W: 33, OC: 16, kernel: 1, stride: 1, padding: 0, depth: 2, dist: 'uniform', seed: 4 },
    // strided路(WOut!==W なので高速路に乗らない)
    { C: 3, H: 8, W: 34, OC: 20, kernel: 3, stride: 2, padding: 1, depth: 3, dist: 'residual', seed: 5 },
    { C: 2, H: 7, W: 65, OC: 12, kernel: 2, stride: 2, padding: 0, depth: 2, dist: 'uniform', seed: 6 },
    // 定数/素通しに潰れやすい分布(ocOut が定数/タップ直結になる前置きチャンク)
    { C: 3, H: 4, W: 32, OC: 40, kernel: 3, stride: 1, padding: 1, depth: 2, dist: 'residual', seed: 7 },
  ];

  for (const c of cases) {
    it(`C=${c.C} ${c.H}x${c.W} OC=${c.OC} k=${c.kernel} s=${c.stride} d=${c.depth} ${c.dist}`, () => {
      const p = randomConvParams(c);
      const interp = new BitplaneProgramConvKernel(p);
      const cg = new BitplaneProgramConvKernel(p, { codegen: true });
      expect(cg.codegenNote, 'codegenが構築できていること').toBeNull();
      expect(cg.compiled).not.toBeNull();
      for (const batch of [1, 3]) {
        const x = randomBitplane(c.C, c.H, c.W, batch, c.seed * 977 + batch);
        expectSameWords(interp.evalAll(x, batch), cg.evalAll(x, batch), `batch=${batch}`);
      }
    });
  }

  it('CHUNK(1関数あたりステートメント数)を変えても出力は1ビットも変わらない', () => {
    const c = cases[2];
    const p = randomConvParams(c);
    const x = randomBitplane(c.C, c.H, c.W, 2, 4242);
    const ref = new BitplaneProgramConvKernel(p).evalAll(x, 2);
    const seen = new Set<number>();
    for (const chunkSize of [1, 2, 8, 64, 256, 1024, 1_000_000]) {
      const k = new BitplaneProgramConvKernel(p, { codegen: true, chunkSize });
      expect(k.compiled).not.toBeNull();
      seen.add(k.compiled!.stats.nChunks);
      expectSameWords(ref, k.evalAll(x, 2), `chunkSize=${chunkSize}`);
    }
    // 分割点が実際に動いていること(全部同じ本数なら不変性の検証にならない)
    expect(seen.size).toBeGreaterThan(1);
  });

  it('巻き上げ閾値(hoistMinUses)を変えても出力は変わらない', () => {
    const c = cases[0];
    const p = randomConvParams(c);
    const x = randomBitplane(c.C, c.H, c.W, 1, 31337);
    const ref = new BitplaneProgramConvKernel(p).evalAll(x, 1);
    for (const hoistMinUses of [2, 3, 1_000_000]) {
      const k = new BitplaneProgramConvKernel(p, { codegen: true, hoistMinUses });
      expectSameWords(ref, k.evalAll(x, 1), `hoistMinUses=${hoistMinUses}`);
    }
  });
});

describe('convProgramCodegen: 実モデル(public/models)でインタプリタと整数完全一致', () => {
  const MODELS: Record<string, string> = {
    det: 'det_multimnist_lut_k256.json',
    detmono: 'det_multimnist_mono_lut_k256.json',
    detface: 'det_face_lut_k256.json',
    detreal: 'det_fewclass_lut_k64.json',
    seg: 'seg_convdlgn_k128.json',
  };

  for (const [task, file] of Object.entries(MODELS)) {
    const path = join(here, '../../public/models', file);
    const run = existsSync(path) ? it : it.skip;
    // det/detmono は tree_depth=9(Shannon分解)まで含む実モデルで、
    // 縮約自体は速い(gate_stats.mjs実測: セルフチェック込みで200ms未満)が、
    // interp/codegen二重構築+全ノード(最大1280ch)の全数語比較という
    // テスト自体の総和コストが大きく、フルスイート並列実行時のCPU競合下では
    // 既定5000msに収まらないことがある(フォールバックを隠すための延長では
    // なく、実測される正味の計算量に対する妥当な予算)。
    run(`${task}: conv2dノード出力とスコアが一致する`, () => {
      const doc = JSON.parse(readFileSync(path, 'utf8')) as { ir: IRGraphJson };
      const lowered = lowerToPrimitives(doc.ir, { nativeConv: true, nativeOrPool: true });
      const convNodes = lowered.nodes.filter((nd) => nd.type === 'conv2d');
      expect(convNodes.length).toBeGreaterThan(0);

      for (const nd of convNodes) {
        const p = nd.params as unknown as Conv2dParams;
        const [C, H, W] = p.in_shape;
        const interp = new BitplaneProgramConvKernel(p);
        const cg = new BitplaneProgramConvKernel(p, { codegen: true });
        expect(cg.codegenNote, `${task}/${nd.id}`).toBeNull();
        const x = randomBitplane(C, H, W, 1, nd.id.length * 7919 + C);
        expectSameWords(interp.evalAll(x, 1), cg.evalAll(x, 1), `${task}/${nd.id}`);
      }

      // グラフ全体(ConvRuntime)でもスコアが完全一致する。
      const rand = rng(task.length * 13);
      const inWidth = doc.ir.inputs[0].width;
      const bits = new Uint8Array(inWidth);
      for (let i = 0; i < inWidth; i++) bits[i] = rand() < 0.5 ? 1 : 0;
      const base = new ConvRuntime(doc.ir, { engine: 'bitplane' });
      const gen = new ConvRuntime(doc.ir, { engine: 'bitplane', convCodegen: true });
      expect(gen.convCodegenNote, `${task}: codegen note`).toBeNull();
      expect(gen.convCodegenStats!.sourceChars).toBeGreaterThan(0);
      const a = base.scores(bits, 1);
      const b = gen.scores(bits, 1);
      expect(b.width).toBe(a.width);
      for (let i = 0; i < a.data.length; i++) {
        if (a.data[i] !== b.data[i]) {
          throw new Error(`${task}: score[${i}] interp=${a.data[i]} codegen=${b.data[i]}`);
        }
      }
    }, 20_000);
  }
});

describe('convProgramCodegen: new Function が使えない環境', () => {
  const p = randomConvParams({
    C: 3, H: 6, W: 33, OC: 16, kernel: 3, stride: 1, padding: 1, depth: 3,
    dist: 'residual', seed: 99,
  });

  it('compileConvProgram は throw を素通しする(呼び出し側が捕まえる契約)', () => {
    const orig = globalThis.Function;
    // CSP(unsafe-eval禁止)ブラウザと同じく new Function が投げる状況を作る。
    (globalThis as { Function: unknown }).Function = function Blocked(): never {
      throw new EvalError('Refused to evaluate a string as JavaScript (CSP)');
    };
    try {
      const L = new BitplaneProgramConvKernel(p).layout;
      expect(() => compileConvProgram(L)).toThrow(/CSP/);
    } finally {
      (globalThis as { Function: unknown }).Function = orig;
    }
  });

  it('カーネルは黙って壊れず note を残してインタプリタ経路へ戻る', () => {
    const x = randomBitplane(3, 6, 33, 1, 555);
    const ref = new BitplaneProgramConvKernel(p).evalAll(x, 1);
    const orig = globalThis.Function;
    (globalThis as { Function: unknown }).Function = function Blocked(): never {
      throw new EvalError('Refused to evaluate a string as JavaScript (CSP)');
    };
    let k: BitplaneProgramConvKernel;
    try {
      k = new BitplaneProgramConvKernel(p, { codegen: true });
    } finally {
      (globalThis as { Function: unknown }).Function = orig;
    }
    expect(k.compiled).toBeNull();
    expect(k.codegenNote).toMatch(/CSP/);
    expectSameWords(ref, k.evalAll(x, 1), 'CSPフォールバック');
  });
});
