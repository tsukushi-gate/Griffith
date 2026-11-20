// 配信サイズカードの「モデル+ランタイム」合算ロジック。
//
// 従来は大きい数字(配信サイズ)がモデルのみ(gzip基準)で、ランタイム(wasm等)
// は非圧縮のままサブラインに出るだけだった。合算するにはモデルとランタイムを
// 同じ基準(gzip実測どうし、または非圧縮どうし)で揃える必要がある
// (runtimeBytes は非圧縮、runtimeGzipBytes はモデルと同じ CompressionStream
// 実測なので、gzipBytes の有無で使い分ける)。
import { describe, expect, it } from 'vitest';

import { fmtBytesInline } from './format';
import { deliverBytesOf, sizeSublineOf } from './runtimeCard';

describe('deliverBytesOf: 配信サイズ(モデル+ランタイム)の基準を揃えた合算', () => {
  it('gzip実測が取れているとき: モデルgzip + ランタイムgzip を合算する(非圧縮とは混ぜない)', () => {
    const r = deliverBytesOf({
      modelBytes: 16_640_000, // 非圧縮(参考値、合算には使わない)
      gzipBytes: 422_400,
      runtimeBytes: 22_720_000, // 非圧縮のランタイム(合算には使わない)
      runtimeGzipBytes: 9_420,
    });
    expect(r.model).toBe(422_400);
    expect(r.runtime).toBe(9_420);
    expect(r.total).toBe(422_400 + 9_420);
    // 非圧縮どうしを混ぜていないことの確認(基準違いの数字を作らない)。
    expect(r.total).not.toBe(422_400 + 22_720_000);
  });

  it('ランタイムのgzipが未計測(null)のときはモデル単体にフォールバックする(NaN化しない)', () => {
    const r = deliverBytesOf({
      modelBytes: 16_640_000,
      gzipBytes: 422_400,
      runtimeBytes: 22_720_000,
      runtimeGzipBytes: null,
    });
    expect(r.runtime).toBeNull();
    expect(r.total).toBe(422_400);
    expect(Number.isNaN(r.total)).toBe(false);
  });

  it('ランタイムが存在しないカード(wasm不使用等)はモデル単体のまま', () => {
    const r = deliverBytesOf({
      modelBytes: 1_060_850,
      gzipBytes: 300_000,
      runtimeBytes: null,
      runtimeGzipBytes: null,
    });
    expect(r.runtime).toBeNull();
    expect(r.total).toBe(300_000);
  });

  it('CompressionStream非対応環境(gzipBytes===null)では非圧縮どうしで合算する', () => {
    const r = deliverBytesOf({
      modelBytes: 16_640_000,
      gzipBytes: null,
      runtimeBytes: 22_720_000,
      runtimeGzipBytes: null, // gzip非対応なら計測もできていないはず
    });
    expect(r.model).toBe(16_640_000);
    expect(r.runtime).toBe(22_720_000);
    expect(r.total).toBe(16_640_000 + 22_720_000);
  });

  it('モデル自体が欠損(エラー系)のときは null のまま', () => {
    const r = deliverBytesOf({
      modelBytes: null, gzipBytes: null, runtimeBytes: null, runtimeGzipBytes: null,
    });
    expect(r.total).toBeNull();
  });
});

// サブライン(内訳)の文言。以前は「非圧縮 {モデル単体}」を先頭に置いて
// いたため、配信サイズ(モデル+ランタイム合算)の方が非圧縮(モデル単体)より
// 大きく見える=矛盾しているように見える表示になっていた。先頭2項を足すと
// 必ず配信サイズ(deliverBytesOfのtotal)に一致する形へ揃える。
describe('sizeSublineOf: 配信サイズのサブライン(内訳)を組み立てる', () => {
  it('gzip実測+ランタイムgzipどちらも取れているとき: モデル+ランタイム / 非圧縮 の順', () => {
    const info = {
      modelBytes: 16_640_000, gzipBytes: 432_538, runtimeBytes: 22_720_000, runtimeGzipBytes: 3_686,
    };
    const { model, runtime, total } = deliverBytesOf(info);
    const sub = sizeSublineOf(info, model, runtime);
    expect(sub).toBe(
      `モデル ${fmtBytesInline(model)} + ランタイム ${fmtBytesInline(runtime)}`
      + ` / 非圧縮 ${fmtBytesInline(info.modelBytes)}`,
    );
    // 先頭2項(モデル+ランタイム)の合計が配信サイズ(大きい数字)と必ず一致する。
    expect((model ?? 0) + (runtime ?? 0)).toBe(total);
  });

  it('gzip実測はあるがランタイムgzipが未計測のとき: ランタイム項を省き、非圧縮のみ添える', () => {
    const info = {
      modelBytes: 1_060_850, gzipBytes: 300_000, runtimeBytes: null, runtimeGzipBytes: null,
    };
    const { model, runtime } = deliverBytesOf(info);
    const sub = sizeSublineOf(info, model, runtime);
    expect(sub).toBe(`モデル ${fmtBytesInline(model)} / 非圧縮 ${fmtBytesInline(info.modelBytes)}`);
  });

  it('CompressionStream非対応(gzipBytes===null)のとき: 非圧縮どうしの内訳のみ、重複注記は付けない', () => {
    const info = {
      modelBytes: 16_640_000, gzipBytes: null, runtimeBytes: 22_720_000, runtimeGzipBytes: null,
    };
    const { model, runtime } = deliverBytesOf(info);
    const sub = sizeSublineOf(info, model, runtime);
    expect(sub).toBe(`モデル ${fmtBytesInline(model)} + ランタイム ${fmtBytesInline(runtime)}`);
    expect(sub).not.toContain('非圧縮');
  });

  it('gzip非対応かつランタイムも無いときはモデル単体のみ(重複注記なし)', () => {
    const info = {
      modelBytes: 1_060_850, gzipBytes: null, runtimeBytes: null, runtimeGzipBytes: null,
    };
    const { model, runtime } = deliverBytesOf(info);
    expect(sizeSublineOf(info, model, runtime)).toBe(`モデル ${fmtBytesInline(model)}`);
  });

  it('モデルサイズ自体が欠損しているときは空文字', () => {
    expect(sizeSublineOf({ modelBytes: null, gzipBytes: null }, null, null)).toBe('');
  });
});
