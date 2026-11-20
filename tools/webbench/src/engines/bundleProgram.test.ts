// bundleProgram.ts(配布バンドルの programs 節 ↔ ConvProgram)のテスト。
//
// 契約:
//   - convProgramToDict ∘ convProgramFromDict は恒等(往復同一性)
//   - 壊れた dict は construct 時点で必ず例外になる(コーパス別に1本ずつ)
//   - Python(griffith_logic/deploy/programs.py::program_to_dict)が焼いた
//     dict と、TS側で同じ IR から buildConvProgram→convProgramToDict した
//     ものがバイト完全一致する(src/fixtures/bundle_golden.json、無ければskip)
//   - resolveConvProgram の3分岐(bundle採用/非strictフォールバック/strict再送出)
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { convTreeSpecOf } from './bitplane';
import {
  convProgramFromDict, convProgramToDict, resolveConvProgram,
} from './bundleProgram';
import type { BundleProgramDict } from './bundleProgram';
import {
  REF_KIND_INSTR, buildConvProgram, makeRef, refIdx, refKind,
} from './convProgram';
import type { ConvTreeSpec } from './convProgram';
import type { Conv2dParams, IRGraphJson } from '../shared';

/** 決定的な擬似乱数(mulberry32、convProgram.test.ts と同一)。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface RandomTreeOpts {
  OC: number;
  depth: number;
  nTapPool: number;
  dist: 'uniform' | 'residual';
  seed: number;
}

/** convProgram.test.ts::randomSpec と同一のランダム木生成(往復/検証テスト用に複製)。 */
function randomSpec(o: RandomTreeOpts): ConvTreeSpec {
  const rand = rng(o.seed);
  const nLeaves = 1 << o.depth;
  const nInternal = nLeaves - 1;
  const indicesFlat: number[][] = [];
  for (let oc = 0; oc < o.OC; oc++) {
    const row: number[] = [];
    for (let k = 0; k < nLeaves; k++) row.push(Math.floor(rand() * o.nTapPool));
    indicesFlat.push(row);
  }
  const truthTables = new Uint8Array(o.OC * nInternal);
  for (let i = 0; i < truthTables.length; i++) {
    if (o.dist === 'uniform') {
      truthTables[i] = Math.floor(rand() * 16) & 0xf;
    } else {
      const r = rand();
      truthTables[i] = r < 0.55 ? 0xa : r < 0.62 ? 0xc : r < 0.68 ? 0x0
        : r < 0.74 ? 0xf : r < 0.78 ? 0x5 : r < 0.82 ? 0x3 : Math.floor(rand() * 16) & 0xf;
    }
  }
  return { outChannels: o.OC, treeDepth: o.depth, indicesFlat, truthTables };
}

const ROUNDTRIP_CASES: RandomTreeOpts[] = [
  { OC: 24, depth: 3, nTapPool: 6, dist: 'residual', seed: 1 },
  { OC: 16, depth: 2, nTapPool: 3, dist: 'uniform', seed: 4 },
  { OC: 12, depth: 1, nTapPool: 2, dist: 'uniform', seed: 5 },
  { OC: 8, depth: 0, nTapPool: 4, dist: 'uniform', seed: 6 },
  { OC: 40, depth: 4, nTapPool: 9, dist: 'residual', seed: 7 },
];

describe('convProgramToDict ∘ convProgramFromDict は恒等(往復同一性)', () => {
  for (const c of ROUNDTRIP_CASES) {
    it(`OC=${c.OC} depth=${c.depth} taps=${c.nTapPool} ${c.dist} (seed=${c.seed})`, () => {
      const spec = randomSpec(c);
      const prog = buildConvProgram(spec);
      const dict = convProgramToDict(prog);
      const back = convProgramFromDict(dict, spec.outChannels);
      expect(back).toEqual(prog);
    });
  }
});

// ============================================================
// 構造検証: コーパス別の破損検出(1クラス1本)
// ============================================================

/** 検証テスト共通のベースdict(十分な命令数・非自明な依存関係を持つ木)。 */
function baseCase(): { spec: ConvTreeSpec; dict: BundleProgramDict } {
  const spec = randomSpec({ OC: 24, depth: 3, nTapPool: 6, dist: 'residual', seed: 1 });
  const prog = buildConvProgram(spec);
  expect(prog.instrs.length).toBeGreaterThan(2); // 以降の破損テストの前提
  return { spec, dict: convProgramToDict(prog) };
}

/** 深いコピー(テストごとに独立して汚せるように)。 */
function cloneDict(d: BundleProgramDict): BundleProgramDict {
  return JSON.parse(JSON.stringify(d)) as BundleProgramDict;
}

describe('convProgramFromDict: 構造検証(破損コーパス別)', () => {
  it('正常なdictは例外を投げない(破損テストの対照)', () => {
    const { spec, dict } = baseCase();
    expect(() => convProgramFromDict(dict, spec.outChannels)).not.toThrow();
  });

  it('instrs長が4*n_instrsと不一致なら例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs = bad.instrs.slice(0, bad.instrs.length - 1);
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/instrs長/);
  });

  it('ttが0..15の外なら例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs[0] = 99; // instr0のtt
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/tt=99/);
  });

  it('オペランド参照が前方参照でない(自己/後方参照)なら例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs[1] = makeRef(REF_KIND_INSTR, 0); // instr0.a を instr0自身への参照にする
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/前方参照でない/);
  });

  it('tapに重複があれば例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.taps = [...bad.taps, bad.taps[0]];
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/tapに重複/);
  });

  it('write_startがoc_outからの再計算値と不一致なら例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.write_start = bad.write_start.map((v, i) => (i === 1 ? v + 1 : v));
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/write_start/);
  });

  it('slotがn_slotsの外なら例外', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs[3] = bad.n_slots; // instr0のslotをn_slots(範囲外)にする
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/slot=/);
  });

  it('生存中の先行命令とスロットがエイリアスすると例外(Pythonの検証器には無い検査)', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    // 最初にINSTR参照をオペランドに持つ命令jを探し、その参照先kの生存中に
    // j自身のdest slotをkのslotへ書き換える(gate2RangeIntoの
    // 「dest≠生存中オペランド」前提を破る)。
    let target = -1;
    let ownerIdx = -1;
    for (let j = 0; j < dict.n_instrs; j++) {
      const a = dict.instrs[4 * j + 1];
      const b = dict.instrs[4 * j + 2];
      const ref = refKind(a) === REF_KIND_INSTR ? a : (refKind(b) === REF_KIND_INSTR ? b : -1);
      if (ref >= 0) {
        target = j;
        ownerIdx = refIdx(ref);
        break;
      }
    }
    expect(target).toBeGreaterThanOrEqual(0); // ベース木に非自明な依存があることの前提
    // 正しい割当では生存区間が重なる命令は必ず別スロットのはず(前提の再確認)。
    expect(dict.instrs[4 * target + 3]).not.toBe(dict.instrs[4 * ownerIdx + 3]);
    bad.instrs[4 * target + 3] = dict.instrs[4 * ownerIdx + 3];
    expect(() => convProgramFromDict(bad, spec.outChannels)).toThrow(/エイリアス/);
  });
});

// ============================================================
// resolveConvProgram の分岐(bundle採用/非strictフォールバック/strict再送出)
// ============================================================

describe('resolveConvProgram', () => {
  it('dが正常ならbundleを採用しnoteはnull', () => {
    const { spec, dict } = baseCase();
    const { program, provenance, note } = resolveConvProgram(spec, dict);
    expect(provenance).toBe('bundle');
    expect(note).toBeNull();
    expect(program.instrs.length).toBe(dict.n_instrs);
  });

  it('dが未指定ならbuildConvProgramへ(provenance=built, note=null)', () => {
    const { spec } = baseCase();
    const { provenance, note } = resolveConvProgram(spec, null);
    expect(provenance).toBe('built');
    expect(note).toBeNull();
  });

  it('dが壊れていれば(非strict既定)再構築へフォールバックし理由を記録', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs[0] = 99; // tt範囲外
    const { provenance, note } = resolveConvProgram(spec, bad);
    expect(provenance).toBe('built');
    expect(note).toMatch(/tt=99/);
  });

  it('strict指定時は壊れたdictで例外を再送出する', () => {
    const { spec, dict } = baseCase();
    const bad = cloneDict(dict);
    bad.instrs[0] = 99;
    expect(() => resolveConvProgram(spec, bad, { strict: true })).toThrow(/tt=99/);
  });
});

// ============================================================
// クロス言語バイト完全一致: Python programs.py::program_to_dict との照合
// ============================================================

interface ProgramsGoldenCase {
  name: string;
  ir: IRGraphJson;
  programs: Record<string, BundleProgramDict>;
}

const here = dirname(fileURLToPath(import.meta.url));
const bundleGoldenPath = join(here, '../fixtures/bundle_golden.json');
const hasBundleGolden = existsSync(bundleGoldenPath);

describe.skipIf(!hasBundleGolden)(
  'bundle_golden.json: Python programs.py::program_to_dict とのバイト完全一致',
  () => {
    const golden = JSON.parse(readFileSync(bundleGoldenPath, 'utf8')) as {
      programs_cases: ProgramsGoldenCase[];
    };

    for (const c of golden.programs_cases) {
      for (const nodeId of Object.keys(c.programs)) {
        it(`[${c.name}] ${nodeId}: dict完全一致(statsを含む)`, () => {
          const node = c.ir.nodes.find((n) => n.id === nodeId);
          expect(node, `${nodeId} がIRに存在しない`).toBeDefined();
          const spec = convTreeSpecOf(node!.params as unknown as Conv2dParams);
          const got = convProgramToDict(buildConvProgram(spec));
          expect(got).toEqual(c.programs[nodeId]);
        });
      }
    }
  },
);
