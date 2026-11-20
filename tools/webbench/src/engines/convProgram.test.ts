// conv2d 重み共有木の lossless 命令列化(convProgram.ts)のテスト。
//
// 契約は「整数出力完全不変」なので、テストは全て**全数照合**で書く:
//   - 16種truth tableの代数(swap/入力反転/定数畳み込み表/退化表)を
//     4パターン全数で定義どおりか照合する(表を手書きしない代わりの検証)
//   - ランダム木(タップ重複・定数化しやすいtt分布を含む)について、
//     縮約前のバランス木と縮約後の命令列を全入力パターンで突き合わせる
//   - 可換正規化の対合性(swap∘swap=恒等、可換ttは不動点)
//   - CSEの健全性(同一(tt,a,b)が2度発行されない/参照は必ず先行命令)
//   - スロット割当の健全性(生存区間が重なる命令は同じスロットを使わない)
// 実モデル(det/seg等)での evalConv2dBitplane との整数一致は
// conv.test.ts / seg_e2e.test.ts / detdecode.test.ts の ENGINES 行列が見る。
import { describe, expect, it } from 'vitest';

import {
  CONST_FOLD_A, CONST_FOLD_B, REF_KIND_CONST, REF_KIND_INSTR, REF_KIND_TAP,
  RED_BINARY, RED_CONST0, RED_CONST1, RED_ID, RED_NOT, SAME_FOLD, TT_DEGEN, TT_DEGEN_ON_B,
  TT_NOT, TT_NOT_A, TT_NOT_B, TT_SWAP, buildConvProgram, makeRef, refIdx, refKind, ttEval,
  ttIsCommutative, ttSwap, verifyConvProgram,
} from './convProgram';
import type { ConvProgram, ConvRef, ConvTreeSpec } from './convProgram';

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

/** bitplane.ts / assembly/index.ts の gate2 と同一の16式(0/1スカラー版)。 */
function gate2Scalar(tt: number, a: number, b: number): number {
  switch (tt & 0xf) {
    case 0x0: return 0;
    case 0x1: return (a | b) ^ 1;
    case 0x2: return a & (b ^ 1);
    case 0x3: return b ^ 1;
    case 0x4: return (a ^ 1) & b;
    case 0x5: return a ^ 1;
    case 0x6: return a ^ b;
    case 0x7: return (a & b) ^ 1;
    case 0x8: return a & b;
    case 0x9: return (a ^ b) ^ 1;
    case 0xa: return a;
    case 0xb: return a | (b ^ 1);
    case 0xc: return b;
    case 0xd: return (a ^ 1) | b;
    case 0xe: return a | b;
    default: return 1;
  }
}

describe('tt代数(16種truth tableの表)', () => {
  it('ttEval が gate2 の16式と全4パターンで一致する', () => {
    for (let tt = 0; tt < 16; tt++) {
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 2; b++) {
          expect(ttEval(tt, a, b), `tt=${tt} a=${a} b=${b}`).toBe(gate2Scalar(tt, a, b));
        }
      }
    }
  });

  it('TT_SWAP はオペランド入替、かつ対合(swap∘swap=恒等)', () => {
    for (let tt = 0; tt < 16; tt++) {
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 2; b++) {
          expect(ttEval(TT_SWAP[tt], a, b)).toBe(ttEval(tt, b, a));
        }
      }
      expect(ttSwap(ttSwap(tt))).toBe(tt);
    }
  });

  it('可換ttはswapの不動点、非可換ttは 0x2/0x4 と 0xb/0xd(+単項)', () => {
    const commutative: number[] = [];
    const nonCommutative: number[] = [];
    for (let tt = 0; tt < 16; tt++) (ttIsCommutative(tt) ? commutative : nonCommutative).push(tt);
    expect(commutative).toEqual([0x0, 0x1, 0x6, 0x7, 0x8, 0x9, 0xe, 0xf]);
    expect(nonCommutative).toEqual([0x2, 0x3, 0x4, 0x5, 0xa, 0xb, 0xc, 0xd]);
    // 単項(0x3,0x5,0xa,0xc)は縮約段で消えるので、命令列に残りうる非可換は
    // 0x2/0x4/0xb/0xd の4種のみ。
    expect(TT_SWAP[0x2]).toBe(0x4);
    expect(TT_SWAP[0xb]).toBe(0xd);
  });

  it('TT_NOT_A / TT_NOT_B は入力反転の吸収', () => {
    for (let tt = 0; tt < 16; tt++) {
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 2; b++) {
          expect(ttEval(TT_NOT_A[tt], a, b)).toBe(ttEval(tt, a ^ 1, b));
          expect(ttEval(TT_NOT_B[tt], a, b)).toBe(ttEval(tt, a, b ^ 1));
        }
      }
    }
  });

  it('定数畳み込み表は全16tt×定数0/1×左右で網羅的に正しい', () => {
    const apply = (code: number, x: number): number => {
      if (code === RED_CONST0) return 0;
      if (code === RED_CONST1) return 1;
      if (code === RED_ID) return x;
      expect(code).toBe(RED_NOT);
      return x ^ 1;
    };
    for (let tt = 0; tt < 16; tt++) {
      for (let k = 0; k < 2; k++) {
        // b が定数 k → 残る a についての簡約
        for (let a = 0; a < 2; a++) {
          expect(apply(CONST_FOLD_B[tt * 2 + k], a), `tt=${tt} b=${k} a=${a}`)
            .toBe(ttEval(tt, a, k));
        }
        // a が定数 k → 残る b についての簡約
        for (let b = 0; b < 2; b++) {
          expect(apply(CONST_FOLD_A[tt * 2 + k], b), `tt=${tt} a=${k} b=${b}`)
            .toBe(ttEval(tt, k, b));
        }
      }
    }
  });

  it('TT_DEGEN は 0x0/0xf/0xa/0xc/0x5/0x3 のみ非BINARYで、対象入力も正しい', () => {
    const degenerate: number[] = [];
    for (let tt = 0; tt < 16; tt++) if (TT_DEGEN[tt] !== RED_BINARY) degenerate.push(tt);
    expect(degenerate).toEqual([0x0, 0x3, 0x5, 0xa, 0xc, 0xf]);
    expect(TT_DEGEN[0x0]).toBe(RED_CONST0);
    expect(TT_DEGEN[0xf]).toBe(RED_CONST1);
    expect([TT_DEGEN[0xa], TT_DEGEN_ON_B[0xa]]).toEqual([RED_ID, 0]);
    expect([TT_DEGEN[0xc], TT_DEGEN_ON_B[0xc]]).toEqual([RED_ID, 1]);
    expect([TT_DEGEN[0x5], TT_DEGEN_ON_B[0x5]]).toEqual([RED_NOT, 0]);
    expect([TT_DEGEN[0x3], TT_DEGEN_ON_B[0x3]]).toEqual([RED_NOT, 1]);
  });

  it('SAME_FOLD は f(a,a) の簡約', () => {
    for (let tt = 0; tt < 16; tt++) {
      for (let a = 0; a < 2; a++) {
        const code = SAME_FOLD[tt];
        const got = code === RED_CONST0 ? 0 : code === RED_CONST1 ? 1 : code === RED_ID ? a : a ^ 1;
        expect(got, `tt=${tt} a=${a}`).toBe(ttEval(tt, a, a));
      }
    }
  });
});

// ============================================================
// ランダム木のproperty test
// ============================================================

interface RandomTreeOpts {
  OC: number;
  depth: number;
  /** タップIDの母数(小さいほど木内・木間でタップが重複しCSEが効く)。 */
  nTapPool: number;
  /** truth table の分布。'uniform' | 'residual'(0xa/0x0/0xf を多めに) */
  dist: 'uniform' | 'residual';
  seed: number;
}

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
      // 残差初期化後の学習済みモデルに近い分布(パススルー/定数が多い)
      const r = rand();
      truthTables[i] = r < 0.55 ? 0xa : r < 0.62 ? 0xc : r < 0.68 ? 0x0
        : r < 0.74 ? 0xf : r < 0.78 ? 0x5 : r < 0.82 ? 0x3 : Math.floor(rand() * 16) & 0xf;
    }
  }
  return { outChannels: o.OC, treeDepth: o.depth, indicesFlat, truthTables };
}

/** 元のバランス木を素朴評価する(bitplane.ts の木縮約と同じ順序)。 */
function evalTree(spec: ConvTreeSpec, oc: number, tapVal: Map<number, number>): number {
  const nLeaves = 1 << spec.treeDepth;
  const nInternal = nLeaves - 1;
  let cur: number[] = [];
  for (let k = 0; k < nLeaves; k++) cur.push(tapVal.get(spec.indicesFlat[oc][k]) as number);
  let gOffset = 0;
  while (cur.length > 1) {
    const next: number[] = [];
    for (let g = 0; g < cur.length >> 1; g++) {
      next.push(gate2Scalar(
        spec.truthTables[oc * nInternal + gOffset + g], cur[g * 2], cur[g * 2 + 1],
      ));
    }
    gOffset += next.length;
    cur = next;
  }
  return cur[0];
}

/** 命令列を素朴評価する(全命令を順に評価)。 */
function evalProgram(prog: ConvProgram, tapVal: Map<number, number>): number[] {
  const instrVal = new Int32Array(prog.instrs.length);
  const val = (r: ConvRef): number => {
    const kind = refKind(r);
    if (kind === REF_KIND_CONST) return refIdx(r);
    if (kind === REF_KIND_TAP) return tapVal.get(prog.taps[refIdx(r)]) as number;
    return instrVal[refIdx(r)];
  };
  for (let j = 0; j < prog.instrs.length; j++) {
    const ins = prog.instrs[j];
    instrVal[j] = gate2Scalar(ins.tt, val(ins.a), val(ins.b));
  }
  return prog.ocOut.map(val);
}

const RANDOM_CASES: RandomTreeOpts[] = [
  { OC: 24, depth: 3, nTapPool: 6, dist: 'residual', seed: 1 },
  { OC: 24, depth: 3, nTapPool: 40, dist: 'residual', seed: 2 },
  { OC: 16, depth: 3, nTapPool: 5, dist: 'uniform', seed: 3 },
  { OC: 16, depth: 2, nTapPool: 3, dist: 'uniform', seed: 4 },
  { OC: 12, depth: 1, nTapPool: 2, dist: 'uniform', seed: 5 },
  { OC: 8, depth: 0, nTapPool: 4, dist: 'uniform', seed: 6 },
  { OC: 40, depth: 4, nTapPool: 9, dist: 'residual', seed: 7 },
];

describe('buildConvProgram: ランダム木の全数一致', () => {
  for (const c of RANDOM_CASES) {
    it(`OC=${c.OC} depth=${c.depth} taps=${c.nTapPool} ${c.dist} (seed=${c.seed})`, () => {
      const spec = randomSpec(c);
      const prog = buildConvProgram(spec);
      // 全数照合は「元の木に現れる全タップ」で回す(命令列側では未使用タップが
      // 落とされているので、prog.taps だけで回すと落としたタップへの依存を
      // 見逃す)。
      const allTaps = [...new Set(([] as number[]).concat(
        ...Array.from({ length: c.OC }, (_, oc) => Array.from(spec.indicesFlat[oc] as number[])),
      ))];
      expect(prog.taps.length).toBeLessThanOrEqual(allTaps.length);
      expect(prog.taps.every((t) => allTaps.includes(t))).toBe(true);
      const nTaps = allTaps.length;
      if (nTaps <= 12) {
        for (let pat = 0; pat < 1 << nTaps; pat++) {
          const tapVal = new Map<number, number>();
          for (let i = 0; i < nTaps; i++) tapVal.set(allTaps[i], (pat >>> i) & 1);
          const got = evalProgram(prog, tapVal);
          for (let oc = 0; oc < c.OC; oc++) {
            expect(got[oc], `oc=${oc} pat=${pat}`).toBe(evalTree(spec, oc, tapVal));
          }
        }
      }
      // ロード時セルフチェック(oc単位のビット並列全数照合)も通ること
      expect(() => verifyConvProgram(spec, prog)).not.toThrow();
    });
  }

  it('ランダムなタップ大量ケースでも verifyConvProgram を通る', () => {
    const spec = randomSpec({ OC: 200, depth: 3, nTapPool: 500, dist: 'residual', seed: 11 });
    const prog = buildConvProgram(spec);
    expect(() => verifyConvProgram(spec, prog)).not.toThrow();
    expect(prog.stats.gatesAfter).toBeLessThan(prog.stats.gatesBefore);
  });
});

describe('buildConvProgram: 構造的健全性', () => {
  const spec = randomSpec({ OC: 64, depth: 3, nTapPool: 12, dist: 'residual', seed: 21 });
  const prog = buildConvProgram(spec);

  it('命令はトポロジカル順(参照は必ず先行命令)で、タップ参照は範囲内', () => {
    for (let j = 0; j < prog.instrs.length; j++) {
      for (const r of [prog.instrs[j].a, prog.instrs[j].b]) {
        const kind = refKind(r);
        expect([REF_KIND_CONST, REF_KIND_TAP, REF_KIND_INSTR]).toContain(kind);
        if (kind === REF_KIND_INSTR) expect(refIdx(r)).toBeLessThan(j);
        if (kind === REF_KIND_TAP) expect(refIdx(r)).toBeLessThan(prog.taps.length);
        if (kind === REF_KIND_CONST) expect(refIdx(r)).toBeLessThanOrEqual(1);
      }
    }
  });

  it('CSE: 同一(tt,a,b)の命令が2度現れない(NOTは a===b の単項として1度だけ)', () => {
    const seen = new Set<string>();
    for (const ins of prog.instrs) {
      const key = `${ins.tt}:${ins.a}:${ins.b}`;
      expect(seen.has(key), `重複命令 ${key}`).toBe(false);
      seen.add(key);
    }
  });

  it('可換正規化: 可換ttは a<b、非可換は 0x2/0x4/0xb/0xd のみ、単項ttは残らない', () => {
    for (const ins of prog.instrs) {
      if (ins.tt === TT_NOT && ins.a === ins.b) continue; // 単項NOT命令
      expect([0x1, 0x2, 0x4, 0x6, 0x7, 0x8, 0x9, 0xb, 0xd, 0xe]).toContain(ins.tt);
      expect(ins.a).not.toBe(ins.b);
      if (ttIsCommutative(ins.tt)) expect(ins.a).toBeLessThan(ins.b);
    }
  });

  it('dead命令が無い(全命令が ocOut から到達可能)', () => {
    const reach = new Uint8Array(prog.instrs.length);
    const stack: number[] = [];
    const push = (r: ConvRef): void => {
      if (refKind(r) === REF_KIND_INSTR && !reach[refIdx(r)]) {
        reach[refIdx(r)] = 1;
        stack.push(refIdx(r));
      }
    };
    for (const r of prog.ocOut) push(r);
    while (stack.length > 0) {
      const j = stack.pop() as number;
      push(prog.instrs[j].a);
      push(prog.instrs[j].b);
    }
    expect(Array.from(reach).filter((v) => v === 0).length).toBe(0);
  });

  it('スロット割当: 生存区間が重なる命令は同じスロットを共有しない', () => {
    const n = prog.instrs.length;
    const lastUse = new Int32Array(n);
    for (let j = 0; j < n; j++) lastUse[j] = j;
    for (let j = 0; j < n; j++) {
      for (const r of [prog.instrs[j].a, prog.instrs[j].b]) {
        if (refKind(r) === REF_KIND_INSTR) lastUse[refIdx(r)] = Math.max(lastUse[refIdx(r)], j);
      }
    }
    const owner = new Int32Array(prog.nSlots).fill(-1);
    for (let j = 0; j < n; j++) {
      const slot = prog.instrs[j].slot;
      expect(slot).toBeGreaterThanOrEqual(0);
      expect(slot).toBeLessThan(prog.nSlots);
      const prev = owner[slot];
      if (prev >= 0) expect(lastUse[prev], `slot=${slot} 命令${prev}の生存区間が命令${j}と重複`).toBeLessThan(j);
      owner[slot] = j;
      // オペランドのスロットとデスティネーションが衝突しないこと
      for (const r of [prog.instrs[j].a, prog.instrs[j].b]) {
        if (refKind(r) === REF_KIND_INSTR) expect(prog.instrs[refIdx(r)].slot).not.toBe(slot);
      }
    }
  });

  it('書き出しスケジュール: 全ocが1回だけ、値が確定した後に書かれる', () => {
    const { writeStart, writeOc } = prog;
    expect(writeStart.length).toBe(prog.instrs.length + 2);
    expect(writeStart[0]).toBe(0);
    expect(writeStart[writeStart.length - 1]).toBe(spec.outChannels);
    const written = new Set<number>();
    for (let bucket = 0; bucket < writeStart.length - 1; bucket++) {
      for (let i = writeStart[bucket]; i < writeStart[bucket + 1]; i++) {
        const oc = writeOc[i];
        expect(written.has(oc), `oc=${oc} が2回書かれた`).toBe(false);
        written.add(oc);
        const r = prog.ocOut[oc];
        if (bucket === 0) expect(refKind(r)).not.toBe(REF_KIND_INSTR);
        else expect(makeRef(REF_KIND_INSTR, bucket - 1)).toBe(r);
      }
    }
    expect(written.size).toBe(spec.outChannels);
  });

  it('統計の内訳が元ゲート数に一致する', () => {
    const s = prog.stats;
    expect(s.constFolded + s.aliasFolded + s.notFolded + s.cseHits + s.emitted)
      .toBe(s.gatesBefore);
    expect(s.gatesAfter).toBe(s.instrsCreated - s.deadRemoved);
    expect(s.gatesAfter).toBe(prog.instrs.length);
  });
});

describe('verifyConvProgram: 改竄した命令列を検出する', () => {
  const spec = randomSpec({ OC: 32, depth: 3, nTapPool: 10, dist: 'residual', seed: 31 });

  it('命令のttを1つ書き換えると例外になる', () => {
    const prog = buildConvProgram(spec);
    // 出力から到達可能な命令(=どこかのocの依存錐に入る)を1つ壊す
    const target = prog.instrs.length - 1;
    prog.instrs[target].tt = prog.instrs[target].tt ^ 0x6;
    expect(() => verifyConvProgram(spec, prog)).toThrow(/真理値表不一致|木の外のタップ/);
  });

  it('ocOut を別ocの出力へすり替えると例外になる', () => {
    const prog = buildConvProgram(spec);
    const differing = prog.ocOut.findIndex((r, i) => i > 0 && r !== prog.ocOut[0]);
    expect(differing).toBeGreaterThan(0);
    prog.ocOut[differing] = prog.ocOut[0];
    expect(() => verifyConvProgram(spec, prog)).toThrow();
  });
});
