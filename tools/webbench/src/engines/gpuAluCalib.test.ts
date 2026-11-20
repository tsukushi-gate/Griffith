// gpuAluCalib.ts の純関数(WGSL生成・命令列生成)をnode/vitestで検証する。
// GPU実行そのもの(timestamp query計測)はブラウザでしか確認できないため、
// ここでは「生成されるWGSLの構造」と「命令列生成の決定性・範囲」のみを見る
// (wasmgpu_verify.htmlが担うのと同じ役割分担: 幾何/生成ロジックはvitest、
// 実行時の正しさ・タイミングはPlaywright駆動のブラウザページ)。
import { describe, expect, it } from 'vitest';

import {
  ALU_OPS_PER_ITER,
  buildAluCalibWgsl,
  buildGate2AbWgsl,
  genGate2Program,
  scaleProgramForThreads,
} from './gpuAluCalib';

describe('buildAluCalibWgsl', () => {
  it('unroll回ぶんの反復行を生成し、依存連鎖を避ける8本のアキュムレータを持つ', () => {
    const code = buildAluCalibWgsl(4);
    const matches = code.match(/m = c \^ \d+u;/g);
    expect(matches).toHaveLength(4);
    for (const acc of ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']) {
      expect(code).toContain(acc);
    }
    // 末尾ストアは1回だけ(=ホットループ内にメモリアクセスが無い)
    expect(code.match(/outBuf\[idx\]/g)).toHaveLength(1);
  });

  it('反復ごとに異なるマスク定数を使う(同一値の繰り返しXORで畳まれる罠を回避)', () => {
    const code = buildAluCalibWgsl(8);
    const consts = [...code.matchAll(/c \^ (\d+)u/g)].map((m) => m[1]);
    expect(new Set(consts).size).toBe(8);
  });

  it('unroll<1は例外(黙って0命令のカーネルを組まない)', () => {
    expect(() => buildAluCalibWgsl(0)).toThrow();
  });

  it('ALU_OPS_PER_ITER はMetal側(metal_calib.mm)と同じ9(マスク生成1+適用8)', () => {
    expect(ALU_OPS_PER_ITER).toBe(9);
  });
});

describe('buildGate2AbWgsl', () => {
  it('switch/mask 両エントリポイントと共有gate2定義を含む', () => {
    const code = buildGate2AbWgsl();
    expect(code).toContain('fn gate2(tt: u32, a: u32, b: u32)'); // webgpuGraph.tsのGATE2_WGSL共有
    expect(code).toContain('fn gate2Mask(');
    expect(code).toContain('fn main_switch(');
    expect(code).toContain('fn main_mask(');
    expect(code).toContain('gate2(tt, a, b)');
    expect(code).toContain('gate2Mask(tt, a, b)');
  });
});

describe('genGate2Program', () => {
  it('tt∈[0,16)、slot∈[0,nSlots)の範囲に収まる', () => {
    const prog = genGate2Program(500, 32, 12345);
    expect(prog).toHaveLength(500);
    for (const ins of prog) {
      expect(ins.tt).toBeGreaterThanOrEqual(0);
      expect(ins.tt).toBeLessThan(16);
      for (const slot of [ins.aSlot, ins.bSlot, ins.outSlot]) {
        expect(slot).toBeGreaterThanOrEqual(0);
        expect(slot).toBeLessThan(32);
      }
    }
  });

  it('同じseedなら決定的(再現性)', () => {
    const a = genGate2Program(200, 16, 999);
    const b = genGate2Program(200, 16, 999);
    expect(a).toEqual(b);
  });

  it('全16種のttが実際に登場する(十分な命令数のとき)', () => {
    const prog = genGate2Program(2000, 32, 7);
    const seen = new Set(prog.map((p) => p.tt));
    expect(seen.size).toBe(16);
  });

  it('nInstr<1やnSlots<2は例外', () => {
    expect(() => genGate2Program(0, 32, 1)).toThrow();
    expect(() => genGate2Program(10, 1, 1)).toThrow();
  });
});

describe('scaleProgramForThreads', () => {
  it('スロット番号をthreads倍したオフセットへ展開する', () => {
    const prog = [{ tt: 6, aSlot: 0, bSlot: 3, outSlot: 1 }];
    const scaled = scaleProgramForThreads(prog, 1024);
    expect(Array.from(scaled)).toEqual([6, 0, 3 * 1024, 1 * 1024]);
  });

  it('4*nInstr個のu32になる', () => {
    const prog = genGate2Program(50, 8, 3);
    const scaled = scaleProgramForThreads(prog, 32);
    expect(scaled).toBeInstanceOf(Uint32Array);
    expect(scaled.length).toBe(50 * 4);
  });
});
