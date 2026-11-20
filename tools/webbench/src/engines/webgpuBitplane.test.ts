// bitplane-webgpu のプラン構築(GPU非依存部分)を検証する。
// node環境ではWebGPUを実行できないので、ここで固定するのは
//   (a) レイアウト契約のヘルパー(wordsPerRow/tailMask/タップ表の詰め物/シフト分解)
//   (b) 実モデル5本に対する planBitplaneGraph の結果(dispatch数・融合の成立・
//       出力幅・生成WGSLの整合)
// の2つ。実際の数値一致(CPU bitplaneとの整数完全一致)は wasmgpu_verify.html を
// Playwrightで動かす経路が正式な錨(このファイルの守備範囲外)。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { ConvRuntime } from './conv';
import {
  packTap, planBitplaneGraph, splitShift, tailMaskU32, unpackTap, wordsPerRow,
} from './webgpuBitplane';
import { ttParse } from '../shared';
import type { IRGraphJson } from '../shared';
import type { ConvModelDoc } from '../types';

const here = dirname(fileURLToPath(import.meta.url));
const modelsDir = join(here, '../../public/models');

describe('レイアウト契約ヘルパー', () => {
  it('wordsPerRow = ceil(W/32)', () => {
    expect(wordsPerRow(1)).toBe(1);
    expect(wordsPerRow(16)).toBe(1);
    expect(wordsPerRow(24)).toBe(1);
    expect(wordsPerRow(32)).toBe(1);
    expect(wordsPerRow(33)).toBe(2);
    expect(wordsPerRow(64)).toBe(2);
    expect(wordsPerRow(96)).toBe(3);
    expect(wordsPerRow(128)).toBe(4);
  });

  it('tailMaskU32 は行末尾の有効ビットだけを立てる(W%32===0は全ビット)', () => {
    expect(tailMaskU32(32)).toBe(0xffffffff);
    expect(tailMaskU32(64)).toBe(0xffffffff);
    expect(tailMaskU32(24)).toBe(0x00ffffff);
    expect(tailMaskU32(16)).toBe(0x0000ffff);
    expect(tailMaskU32(1)).toBe(1);
  });

  it('splitShift は wordShift*32+bitShift===s かつ bitShift∈[0,32)(floor分解)', () => {
    for (let s = -70; s <= 70; s++) {
      const { ws, bs } = splitShift(s);
      expect(bs).toBeGreaterThanOrEqual(0);
      expect(bs).toBeLessThan(32);
      expect(ws * 32 + bs).toBe(s);
    }
    // 実モデル(k=3, pad=1)で現れる3通り
    expect(splitShift(-1)).toEqual({ ws: -1, bs: 31 });
    expect(splitShift(0)).toEqual({ ws: 0, bs: 0 });
    expect(splitShift(1)).toEqual({ ws: 0, bs: 1 });
  });

  it('packTap/unpackTap は往復する', () => {
    const cases = [
      { c: 0, dyRel: 0, ws: 0, bs: 0 },
      { c: 1023, dyRel: -1, ws: -1, bs: 31 },
      { c: 65535, dyRel: 31, ws: 15, bs: 31 },
      { c: 512, dyRel: -32, ws: -16, bs: 17 },
    ];
    for (const t of cases) {
      const packed = packTap(t.c, t.dyRel, t.ws, t.bs);
      expect(packed).toBeGreaterThanOrEqual(0);
      expect(packed).toBeLessThanOrEqual(0xffffffff);
      expect(unpackTap(packed)).toEqual(t);
    }
  });

  it('packTap は範囲外を黙って切り詰めずthrowする', () => {
    expect(() => packTap(0x10000, 0, 0, 0)).toThrow();
    expect(() => packTap(0, 0, 0, 32)).toThrow();
    expect(() => packTap(0, 32, 0, 0)).toThrow();
    expect(() => packTap(0, 0, 16, 0)).toThrow();
  });
});

// --- 実モデルに対するプラン構築 -------------------------------------------

interface ModelExpect {
  file: string;
  /** 単一compute pass内のdispatch数(融合後)。 */
  dispatches: number;
  /** pack へ融合される先頭チャネルpermuteがあるか。 */
  packFusesPermute: boolean;
  /** concat がエイリアス(dispatch 0)になるか。 */
  concatAliased: boolean;
}

const MODELS: ModelExpect[] = [
  // pack, conv1, pool2, conv3, pool4, conv5, conv6, sgs7, conv8, sgs9, conv10, sgs11
  { file: 'det_multimnist_lut_k256.json', dispatches: 12, packFusesPermute: false, concatAliased: true },
  { file: 'det_multimnist_mono_lut_k256.json', dispatches: 12, packFusesPermute: false, concatAliased: true },
  { file: 'det_face_lut_k256.json', dispatches: 12, packFusesPermute: false, concatAliased: true },
  // detreal(低レベル特徴入力へ更新): preproc.kind='image_lowlevel' では
  // thermometer の rgb-major⇔thr-major を吸収する先頭 chanperm ノードを前置しない
  // ため、pack への融合対象が無い。dispatch数は同じ12。
  { file: 'det_fewclass_lut_k64.json', dispatches: 12, packFusesPermute: false, concatAliased: true },
  // seg: pack(+chanperm), conv1, pool2, conv3, pool4, conv5, up6, conv8, up9, conv11, sgs12
  { file: 'seg_convdlgn_k128.json', dispatches: 11, packFusesPermute: true, concatAliased: true },
];

function loadIr(file: string): IRGraphJson {
  const doc = JSON.parse(readFileSync(join(modelsDir, file), 'utf8')) as ConvModelDoc;
  return doc.ir;
}

const haveModels = MODELS.every((m) => existsSync(join(modelsDir, m.file)));

describe.skipIf(!haveModels)('planBitplaneGraph(実モデル)', () => {
  for (const m of MODELS) {
    it(`${m.file}: 単一passのdispatch列へ落ちる`, () => {
      const ir = loadIr(m.file);
      // ConvRuntime の同期コンストラクタは lowering だけ行う(GPU初期化はしない)。
      const lowered = new ConvRuntime(ir, { engine: 'bitplane-webgpu' }).lowered;
      const plan = planBitplaneGraph(lowered);

      expect(plan.computes.length).toBe(m.dispatches);
      expect(plan.inputWidth).toBe(ir.inputs[0].width);
      expect(plan.output.width).toBe(ir.outputs[0].width);
      // 出力はバッファ内に収まる
      expect(plan.output.off + plan.output.width).toBeLessThanOrEqual(plan.output.total);

      // 融合の成立(黙って融合が外れて遅くなるのを防ぐ)
      const notes = plan.notes.join('\n');
      expect(notes.includes('pack融合')).toBe(m.packFusesPermute);
      expect(notes.includes('concat融合')).toBe(m.concatAliased);
      // SpatialGroupSum(permute+group_sum)は5モデルとも必ず融合される
      expect(notes).toContain('group_sum融合');

      // 各dispatchの健全性
      const ids = new Set<string>();
      for (const c of plan.computes) {
        expect(c.threadsPerSample).toBeGreaterThan(0);
        expect(c.code).toContain('@compute @workgroup_size(64)');
        expect(c.code).toContain('fn main(');
        // 同一dispatchでの読み書き衝突が無いこと(WebGPUのusage scope規則)
        expect(c.inputBufs).not.toContain(c.outputBuf);
        expect(ids.has(c.id)).toBe(false);
        ids.add(c.id);
      }
      // conv2d は or_pool2d を lut_layer 展開せずネイティブのまま持つ
      const types = plan.computes.map((c) => c.nodeType);
      expect(types[0]).toBe('pack');
      expect(types).toContain('conv2d');
      expect(types).toContain('spatial_group_sum');
      expect(types).not.toContain('lut_layer');
    });
  }

  it('or_pool2d はネイティブノードとして残り lut_layer 展開されない(det)', () => {
    const lowered = new ConvRuntime(loadIr('det_multimnist_lut_k256.json'), {
      engine: 'bitplane-webgpu',
    }).lowered;
    expect(lowered.nodes.some((n) => n.type === 'or_pool2d')).toBe(true);
    const plan = planBitplaneGraph(lowered);
    expect(plan.computes.filter((c) => c.nodeType === 'or_pool2d').length).toBe(2);
  });

  it('conv2dの静的配列は経路に応じた形を持つ(det n1)', () => {
    const ir = loadIr('det_multimnist_lut_k256.json');
    const lowered = new ConvRuntime(ir, { engine: 'bitplane-webgpu' }).lowered;
    const plan = planBitplaneGraph(lowered);
    const conv = plan.computes.find((c) => c.id === 'n1_lutconv2d')!;
    const n1 = ir.nodes.find((n) => n.id === 'n1_lutconv2d')! as unknown as {
      params: { out_channels: number; tree_depth: number };
    };
    const { out_channels: oc, tree_depth: depth } = n1.params;
    const nLeaves = 2 ** depth;

    if (nLeaves > 16) {
      // 深い木は簡約済み命令列経路。配列は
      // [tapVals, ocTapStart, instrFlat, ocInstr] の4本。
      expect(conv.staticArrays.length).toBe(4);
      expect(conv.staticArrays[1].length).toBe(oc);        // ocTapStart
      expect(conv.staticArrays[3].length).toBe(oc * 3);    // ocInstr(start,count,outRef)
      expect(conv.staticArrays[2].length % 4).toBe(0);     // 命令=4語
      // 簡約が効いていること(展開のままなら oc*(nLeaves-1) 命令になる)。
      const nInstr = conv.staticArrays[2].length / 4;
      expect(nInstr).toBeLessThan(oc * (nLeaves - 1) / 4);
      return;
    }

    // 浅い木は従来どおり完全展開経路。
    expect(conv.staticArrays[0].length).toBe(oc * nLeaves);
    expect(conv.staticArrays[1].length).toBe(oc * (nLeaves - 1));
    const staticArr0 = conv.staticArrays[0];
    let firstBad: { i: number; u: ReturnType<typeof unpackTap> } | null = null;
    for (let i = 0; i < staticArr0.length; i++) {
      const u = unpackTap(staticArr0[i]);
      const ok = u.c < 173 && Math.abs(u.dyRel) <= 1
        && (u.ws === -1 || u.ws === 0) && (u.bs === 0 || u.bs === 1 || u.bs === 31);
      if (!ok) { firstBad = { i, u }; break; }
    }
    expect(firstBad, `最初の違反タップ: ${JSON.stringify(firstBad)}`).toBeNull();
  });
});

describe('未対応構成は黙らずthrowする', () => {
  const base = (nodes: unknown[], edges: unknown[]): IRGraphJson => ({
    version: 1,
    inputs: [{ id: 'x', width: 4 }],
    outputs: [{ id: 'y' }],
    nodes,
    edges,
  } as unknown as IRGraphJson);

  it('未対応ノード型(lut_layer)', () => {
    const ir = base(
      [{
        id: 'l', type: 'lut_layer', params: {
          in_width: 4, out_width: 2, n_inputs: 1, indices: [[0, 1]], truth_tables: ['0x2', '0x2'],
        },
      }],
      [{ src: 'x', dst: 'l' }, { src: 'l', dst: 'y' }],
    );
    expect(() => planBitplaneGraph(ir)).toThrow(/未対応|空間ノードでない/);
  });

  it('出力がビットプレーンのまま終わるIR(アンパック未実装)', () => {
    const ir = base(
      [{
        id: 'c', type: 'conv2d', params: {
          in_shape: [1, 2, 2], out_channels: 1, kernel_size: [1, 1], stride: [1, 1],
          padding: [0, 0], n_inputs: 2, tree_depth: 1, indices_flat: [[0, 0]],
          truth_tables: ['0x8'],
        },
      }],
      [{ src: 'x', dst: 'c' }, { src: 'c', dst: 'y' }],
    );
    expect(() => planBitplaneGraph(ir)).toThrow(/整数値でない|未対応/);
  });
});

// 命令列経路(深い木)が完全展開と論理的に同一かを、WGSLを実行せずに
// 検証する。GPUカーネルが読むのと同じ静的配列を、同じ規約でJS側で解釈し、
// 元のバランス木を素朴評価した結果と全入力パターンで突き合わせる。
// GPUを持たないCIでも回るのが要点(GPU実機での一致は mtverify/wasmgpu_verify
// のページが別途見ている)。
describe.skipIf(!haveModels)('命令列経路の等価性', () => {
  /** 静的配列を「GPUカーネルと同じ手順で」解釈して1出力チャネルを評価する。 */
  function runProgram(
    sa: Uint32Array[], oc: number, tapVal: (packed: number) => number,
  ): number {
    const [tapVals, ocTapStart, instrFlat, ocInstr] = sa;
    const slots = new Uint32Array(64);
    const tapBase = ocTapStart[oc];
    const iStart = ocInstr[oc * 3];
    const iCount = ocInstr[oc * 3 + 1];
    const oRef = ocInstr[oc * 3 + 2];
    const resolve = (r: number): number => {
      const k = r & 3; const ix = r >>> 2;
      if (k === 2) return slots[ix];
      if (k === 1) return tapVal(tapVals[tapBase + ix]);
      return ix === 1 ? 0xffffffff : 0;
    };
    for (let i = 0; i < iCount; i++) {
      const o = (iStart + i) * 4;
      const tt = instrFlat[o];
      const av = resolve(instrFlat[o + 1]);
      const bv = resolve(instrFlat[o + 2]);
      // gate2: bit index = (b<<1)|a を32レーン並列で
      let r = 0;
      for (let bit = 0; bit < 32; bit++) {
        const a = (av >>> bit) & 1; const b = (bv >>> bit) & 1;
        r |= ((tt >>> ((b << 1) | a)) & 1) << bit;
      }
      slots[instrFlat[o + 3]] = r >>> 0;
    }
    return resolve(oRef) >>> 0;
  }

  it('det n1: 命令列の出力が元のバランス木と全パターンで一致する', () => {
    const ir = loadIr('det_multimnist_lut_k256.json');
    const node = ir.nodes.find((n) => n.id === 'n1_lutconv2d')! as unknown as {
      params: { out_channels: number; tree_depth: number;
        indices_flat: number[][]; truth_tables: (string | number)[] };
    };
    const { out_channels: OC, tree_depth: depth } = node.params;
    const nLeaves = 2 ** depth;
    if (nLeaves <= 16) return; // 浅い木では新経路に入らない

    const plan = planBitplaneGraph(
      new ConvRuntime(ir, { engine: 'bitplane-webgpu' }).lowered,
    );
    const sa = plan.computes.find((c) => c.id === 'n1_lutconv2d')!.staticArrays;

    // 元の木を素朴評価する参照実装(タップ値は「そのタップIDのビット」)。
    const nInternal = nLeaves - 1;
    const evalTree = (oc: number, bitOf: (pos: number) => number): number => {
      let cur = node.params.indices_flat[oc].map((pos) => bitOf(pos));
      let off = 0;
      while (cur.length > 1) {
        const next: number[] = [];
        for (let g = 0; g < cur.length / 2; g++) {
          const raw = node.params.truth_tables[oc * nInternal + off + g];
          const tt = Number(typeof raw === 'string' ? ttParse(raw) : raw) & 0xf;
          next.push((tt >>> ((cur[2 * g + 1] << 1) | cur[2 * g])) & 1);
        }
        off += cur.length / 2;
        cur = next;
      }
      return cur[0];
    };

    // 各ocのユニークタップ数は小さい(6入力LUT由来)。全パターンを総当りする。
    let checked = 0;
    for (let oc = 0; oc < Math.min(OC, 32); oc++) {
      const uniq = Array.from(new Set(node.params.indices_flat[oc]));
      expect(uniq.length).toBeLessThanOrEqual(12);
      for (let pat = 0; pat < (1 << uniq.length); pat++) {
        const bitOf = (pos: number) => (pat >>> uniq.indexOf(pos)) & 1;
        // 32レーン全部に同じパターンを敷く(レーン独立性は別テストが担保)。
        const want = evalTree(oc, bitOf) ? 0xffffffff : 0;
        const got = runProgram(sa, oc, (packed) => {
          // packTap の逆変換ではなく、ocごとのタップ並び順で対応付ける。
          const [tapVals, ocTapStart] = sa;
          const base = ocTapStart[oc];
          for (let t = 0; t < uniq.length; t++) {
            if (tapVals[base + t] === packed) return bitOf(uniq[t]) ? 0xffffffff : 0;
          }
          return 0;
        });
        expect(got >>> 0, `oc=${oc} pat=${pat}`).toBe(want >>> 0);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(0);
  }, 120000);
});
