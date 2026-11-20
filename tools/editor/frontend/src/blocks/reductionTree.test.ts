// 付録E#9: 縮約ツリー(reduction tree)生成の性質テスト。
// 幅の単調縮約・窓被覆(各出力ゲートの入力が入力幅内)・層数・停止条件を検証する。

import { describe, expect, it } from 'vitest';
import { createDraftIdAllocator } from '../store/graphStore';
import type { LutLayerParams } from '../ir/schema';
import { reductionTreeLuts, reductionWidths } from './reductionTree';

describe('reductionWidths', () => {
  it('fanin=6 で W→ceil(W/6)→… と単調に縮約し minWidth で停止', () => {
    const w = reductionWidths(256, 6, 6);
    expect(w[0]).toBe(256);
    for (let i = 1; i < w.length; i++) expect(w[i]).toBeLessThan(w[i - 1]);
    expect(w[w.length - 1]).toBeLessThanOrEqual(6);
    // 各段は ceil(prev/6)(ただしminWidth下限)
    expect(w[1]).toBe(Math.ceil(256 / 6));
  });

  it('inWidth<=minWidth は縮約なし(単要素)', () => {
    expect(reductionWidths(4, 6, 6)).toEqual([4]);
  });
});

describe('reductionTreeLuts', () => {
  it('生成した各lut_layerが幅整合し、窓が入力幅内(validate準拠)', () => {
    const tpl = reductionTreeLuts(64, createDraftIdAllocator(), {
      fanin: 6,
      minWidth: 6,
      preset: 'random',
      seed: 123,
    });
    expect(tpl.nodes.length).toBeGreaterThanOrEqual(1);
    // 直列: 各ノードの in_width が前ノードの out_width と一致
    for (let i = 0; i < tpl.nodes.length; i++) {
      const p = tpl.nodes[i].params as unknown as LutLayerParams;
      expect(p.n_inputs).toBeLessThanOrEqual(6);
      expect(p.indices.length).toBe(p.n_inputs);
      // 窓被覆: 全indexが [0, in_width)
      for (const row of p.indices) {
        expect(row.length).toBe(p.out_width);
        for (const v of row) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThan(p.in_width);
        }
      }
      expect(p.truth_tables.length).toBe(p.out_width);
      if (i > 0) {
        const prev = tpl.nodes[i - 1].params as unknown as LutLayerParams;
        expect(p.in_width).toBe(prev.out_width);
      }
    }
    // 最終出力幅は minWidth 以下
    const last = tpl.nodes[tpl.nodes.length - 1].params as unknown as LutLayerParams;
    expect(last.out_width).toBeLessThanOrEqual(6);
    // entry/exit が端点
    expect(tpl.entryId).toBe(tpl.nodes[0].id);
    expect(tpl.exitId).toBe(tpl.nodes[tpl.nodes.length - 1].id);
    // 直列エッジ数 = ノード数-1
    expect(tpl.edges.length).toBe(tpl.nodes.length - 1);
  });

  it('pass-through preset は各ノードが initialized=true', () => {
    const tpl = reductionTreeLuts(36, createDraftIdAllocator(), {
      fanin: 6,
      minWidth: 6,
      preset: 'pass-through',
    });
    for (const n of tpl.nodes) {
      expect(n.train_info.initialized).toBe(true);
    }
  });
});
