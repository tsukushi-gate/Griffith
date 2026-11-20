// particlePath(ポリライン幾何)の単体テスト。

import { describe, expect, it } from 'vitest';
import { buildPolyline, pointAtFraction } from './particlePath';

const pt = { x: 0, y: 0 };

describe('buildPolyline', () => {
  it('累積弧長と総弧長を計算する', () => {
    // (0,0)→(3,0)→(3,4): 長さ3+4=7
    const line = buildPolyline(new Float32Array([0, 0, 3, 0, 3, 4]))!;
    expect(line.length).toBeCloseTo(7, 5);
    expect(Array.from(line.cum)).toEqual([0, 3, 7]);
  });

  it('点が2未満・全長0はnullを返す', () => {
    expect(buildPolyline(new Float32Array([1, 2]))).toBeNull();
    expect(buildPolyline(new Float32Array([1, 2, 1, 2]))).toBeNull();
  });
});

describe('pointAtFraction', () => {
  const line = buildPolyline(new Float32Array([0, 0, 3, 0, 3, 4]))!;

  it('f=0は始点、f=1は終点を返す', () => {
    pointAtFraction(line, 0, pt);
    expect([pt.x, pt.y]).toEqual([0, 0]);
    pointAtFraction(line, 1, pt);
    expect([pt.x, pt.y]).toEqual([3, 4]);
  });

  it('セグメント途中を線形補間する(弧長3.5/7 → 2番目セグメントの先頭0.5)', () => {
    pointAtFraction(line, 0.5, pt);
    expect(pt.x).toBeCloseTo(3, 5);
    expect(pt.y).toBeCloseTo(0.5, 5);
  });

  it('第1セグメント内も弧長比で補間する(f=3/14 → x=1.5)', () => {
    pointAtFraction(line, 3 / 14, pt);
    expect(pt.x).toBeCloseTo(1.5, 5);
    expect(pt.y).toBeCloseTo(0, 5);
  });

  it('範囲外のfはクランプされる', () => {
    pointAtFraction(line, -1, pt);
    expect([pt.x, pt.y]).toEqual([0, 0]);
    pointAtFraction(line, 2, pt);
    expect([pt.x, pt.y]).toEqual([3, 4]);
  });
});
