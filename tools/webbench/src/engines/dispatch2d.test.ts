// 2Dディスパッチ分割の純関数テスト。
// k640 box0/0 の根因(1次元dispatchが maxComputeWorkgroupsPerDimension=65,535 を
// 超過: stage1=640*96*96=5,898,240要素 → 92,160 workgroups)を固定する。
import { describe, expect, it } from 'vitest';

import { dispatch2d } from './webgpuGraph';

const LIMIT = 65535;

describe('dispatch2d', () => {
  it('小規模は1次元のまま(y=1)', () => {
    const d = dispatch2d(1000, LIMIT);
    expect(d.x).toBe(Math.ceil(1000 / 64));
    expect(d.y).toBe(1);
    expect(d.xThreads).toBe(d.x * 64);
  });

  it('k640 stage1(5,898,240要素)は上限内の2Dへ分割される', () => {
    const total = 640 * 96 * 96;
    const d = dispatch2d(total, LIMIT);
    expect(d.x).toBeLessThanOrEqual(LIMIT);
    expect(d.y).toBeLessThanOrEqual(LIMIT);
    expect(d.y).toBeGreaterThan(1); // 従来はここで65,535超→validationエラー
    // 全要素をカバーする(スレッド総数 >= total)
    expect(d.x * 64 * d.y).toBeGreaterThanOrEqual(total);
  });

  it('ちょうど上限のWG数は y=1 に収まる', () => {
    const total = LIMIT * 64;
    const d = dispatch2d(total, LIMIT);
    expect(d.x).toBe(LIMIT);
    expect(d.y).toBe(1);
  });

  it('上限+1WGで y=2 へ割れる', () => {
    const total = (LIMIT + 1) * 64;
    const d = dispatch2d(total, LIMIT);
    expect(d.x).toBe(LIMIT);
    expect(d.y).toBe(2);
    expect(d.x * 64 * d.y).toBeGreaterThanOrEqual(total);
  });

  it('total=0/1でも最低1WG', () => {
    for (const t of [0, 1]) {
      const d = dispatch2d(t, LIMIT);
      expect(d.x).toBe(1);
      expect(d.y).toBe(1);
    }
  });

  it('WGSL側のインデックス復元(gid.y*xThreads+gid.x)が全要素を一意にカバー', () => {
    // 小さいlimitで実際に列挙して過不足なしを確認
    const total = 1000;
    const d = dispatch2d(total, 3, 64); // x<=3WG
    const seen = new Set<number>();
    for (let gy = 0; gy < d.y; gy++) {
      for (let gxThread = 0; gxThread < d.x * 64; gxThread++) {
        const gidx = gy * d.xThreads + gxThread;
        if (gidx < total) seen.add(gidx);
      }
    }
    expect(seen.size).toBe(total);
  });
});
