// lib.ts::loadBundle / bundleDoc.ts::checkBundleDoc のテスト。
//
// 契約:
//   - kind判別(detection→seg→classify-image→dense の優先順)
//   - checkBundleDoc のバージョンポリシー(bundle.py::validate_bundleのミラー、
//     ~116-153行): bundle_format不一致/MAJOR不一致は例外、MINORが新しければ
//     警告のみ、bundle_formatが無いdocは「バンドルではない」警告付きで通す
//   - loadBundle().scores() が直接構築した ConvRuntime.scores() と完全一致
//     (このファイルは薄いラッパーであり自前実装を持たない、というlib.tsの契約)
//   - notes に conv2dプログラムの出所(bundle/built集計)が残る
//   - wasmエンジン(注入モジュール)がjsエンジンと出力完全一致
//   - strictPrograms: 壊れたprogramsは例外(フォールバック禁止)。非strictは
//     黙ってフォールバックする(既存のresolveConvProgram契約どおり)
//   - forceRebuildPrograms: バンドルのprogramsを無視して再構築しても出力は不変
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { ConvRuntime } from '../engines/conv';
import { loadTestWasmSimdModule } from '../engines/wasmTestUtil';
import type { WasmModule } from '../engines/wasmConv';
import { checkBundleDoc } from './bundleDoc';
import { loadBundle } from './lib';

const here = dirname(fileURLToPath(import.meta.url));
const bundleGoldenPath = join(here, '../fixtures/bundle_golden.json');
const hasBundleGolden = existsSync(bundleGoldenPath);

/** golden.bundle_doc のディープコピー(テストごとに独立して汚せるように)。 */
function loadGoldenBundleDoc(): Record<string, unknown> {
  const golden = JSON.parse(readFileSync(bundleGoldenPath, 'utf8')) as { bundle_doc: unknown };
  return JSON.parse(JSON.stringify(golden.bundle_doc)) as Record<string, unknown>;
}

/** 決定的な擬似乱数(mulberry32、bundleProgram.test.tsと同一)。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomBits(n: number, seed: number): Uint8Array {
  const rand = rng(seed);
  const bits = new Uint8Array(n);
  for (let i = 0; i < n; i++) bits[i] = rand() < 0.5 ? 1 : 0;
  return bits;
}

// release-simd.wasm(bitplane-wasmが使う)。無ければ以降のwasm系
// テストはdescribe.skipIfで自動スキップする(conv.test.tsと同じ流儀)。
const wasmModule: WasmModule | null = await loadTestWasmSimdModule();

describe.skipIf(!hasBundleGolden)('checkBundleDoc: バージョンポリシー', () => {
  it('bundle_format/bundle_versionが正しければ警告なしで通る', () => {
    const doc = loadGoldenBundleDoc();
    const { warnings } = checkBundleDoc(doc);
    expect(warnings).toEqual([]);
  });

  it('bundle_formatが無いdocは「バンドルではない」警告付きでそのまま通る(--slim)', () => {
    const doc = loadGoldenBundleDoc();
    delete doc.bundle_format;
    const { doc: out, warnings } = checkBundleDoc(doc);
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0]).toMatch(/バンドルではなく/);
    expect(out).toBe(doc);
  });

  it('bundle_formatが不正なら例外', () => {
    const doc = loadGoldenBundleDoc();
    doc.bundle_format = 'not-a-bundle';
    expect(() => checkBundleDoc(doc)).toThrow(/bundle_formatが不正/);
  });

  it('bundle_versionのMAJORが違えば例外', () => {
    const doc = loadGoldenBundleDoc();
    doc.bundle_version = '2.0';
    expect(() => checkBundleDoc(doc)).toThrow(/非対応/);
  });

  it('bundle_versionのMINORがこのリーダより新しければ警告のみ(例外にしない)', () => {
    const doc = loadGoldenBundleDoc();
    doc.bundle_version = '1.99';
    const { warnings } = checkBundleDoc(doc);
    expect(warnings.some((w) => w.includes('より新しい'))).toBe(true);
  });

  it('必須キー(ir)が無ければ例外', () => {
    const doc = loadGoldenBundleDoc();
    delete doc.ir;
    expect(() => checkBundleDoc(doc)).toThrow(/必須キーがない/);
  });

  it('オブジェクトでないJSONは例外', () => {
    expect(() => checkBundleDoc(null)).toThrow(/オブジェクトでない/);
    expect(() => checkBundleDoc([1, 2, 3])).toThrow(/オブジェクトでない/);
  });
});

describe.skipIf(!hasBundleGolden)('loadBundle: kind判別', () => {
  it('bundle_golden(readout.kind=spatial_scores)はkind=segと判別され、seg専用メソッドが生える', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.kind).toBe('seg');
    expect(bundle.segLabels).toBeTypeOf('function');
    expect(bundle.scoresRgba).toBeTypeOf('function');
    expect(bundle.detect).toBeUndefined();
  });

  it('readout.spatial=null(export_webbench.py::_readout_metaが形状を推定できない正当な出力)でも'
    + 'loadBundleは例外にせずkind=segのままscoresは動く。segLabelsは生えずnotesに理由が残る', async () => {
    const doc = loadGoldenBundleDoc();
    (doc.readout as Record<string, unknown>).spatial = null;
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.kind).toBe('seg');
    expect(bundle.segLabels).toBeUndefined();
    expect(bundle.scoresRgba).toBeTypeOf('function');
    expect(bundle.detect).toBeUndefined();
    // scores()自体はspatialに依存しないので普通に動く(verify-webはこれしか使わない)
    const bits = randomBits(bundle.runtime.inWidth, 42);
    expect(() => bundle.scores(bits, 1)).not.toThrow();
    expect(bundle.notes.some((n) => n.includes('readout.spatial 無し'))).toBe(true);
  });

  it('readoutを外し preproc.kind を非画像にすると kind=dense(画像専用メソッドが生えない)', async () => {
    const doc = loadGoldenBundleDoc();
    delete doc.readout;
    (doc.preproc as Record<string, unknown>).kind = 'thermometer';
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.kind).toBe('dense');
    expect(bundle.scoresRgba).toBeUndefined();
    expect(bundle.segLabels).toBeUndefined();
    expect(bundle.detect).toBeUndefined();
  });

  it('readoutを外すが preproc.kind は画像のままだと kind=classify-image', async () => {
    const doc = loadGoldenBundleDoc();
    delete doc.readout;
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.kind).toBe('classify-image');
    expect(bundle.scoresRgba).toBeTypeOf('function');
    expect(bundle.segLabels).toBeUndefined();
    expect(bundle.detect).toBeUndefined();
  });
});

describe.skipIf(!hasBundleGolden)('loadBundle: scoresが直接構築したConvRuntimeと一致', () => {
  it('js engine: bundle.scores() が new ConvRuntime(...).scores() と完全一致', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    const direct = new ConvRuntime(doc.ir as never, {
      engine: 'bitplane',
      programs: (doc.programs as never) ?? null,
    });
    const bits = randomBits(bundle.runtime.inWidth, 7);
    const a = bundle.scores(bits, 1);
    const b = direct.scores(bits, 1);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
    expect(a.width).toBe(b.width);
  });

  it('notesにconv2dプログラム出所の集計が入る(全ノードbundle採用)', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.notes.some((n) => n.includes('conv2dプログラム出所'))).toBe(true);
    expect(bundle.notes.some((n) => n.includes('bundle=2') && n.includes('built=0'))).toBe(true);
  });
});

describe.skipIf(!hasBundleGolden || !wasmModule)('loadBundle: wasmエンジン(注入モジュール)', () => {
  it('wasm engineはjsエンジンと出力が完全一致する', async () => {
    const doc = loadGoldenBundleDoc();
    const jsBundle = await loadBundle(doc, { engine: 'js' });
    const wasmBundle = await loadBundle(doc, { engine: 'wasm', wasmModule: wasmModule! });
    expect(wasmBundle.engine).toBe('wasm');
    const bits = randomBits(jsBundle.runtime.inWidth, 99);
    const a = jsBundle.scores(bits, 1);
    const b = wasmBundle.scores(bits, 1);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
  });
});

describe.skipIf(!hasBundleGolden)("loadBundle: engine='auto'のwasmフォールバック", () => {
  it('wasm初期化に失敗するとjsへ落ち、notesに理由が残る', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, {
      engine: 'auto',
      wasmUrl: new URL('file:///nonexistent-mdl-web-test.wasm'),
    });
    expect(bundle.engine).toBe('js');
    expect(bundle.notes.some((n) => n.includes('wasmエンジンの初期化に失敗'))).toBe(true);
  });
});

describe.skipIf(!hasBundleGolden)('loadBundle: strictPrograms / forceRebuildPrograms', () => {
  function corruptFirstProgram(doc: Record<string, unknown>): string {
    const programs = doc.programs as Record<string, { instrs: number[] }>;
    const nodeId = Object.keys(programs)[0];
    programs[nodeId].instrs[0] = 99; // tt(0..15)の範囲外に破壊
    return nodeId;
  }

  it('非strict既定: 壊れたprogramsは黙ってbuildConvProgramへフォールバックする', async () => {
    const doc = loadGoldenBundleDoc();
    corruptFirstProgram(doc);
    const bundle = await loadBundle(doc, { engine: 'js' });
    expect(bundle.notes.some((n) => n.includes('programs採用'))).toBe(true);
    expect(bundle.notes.some((n) => n.includes('built=1'))).toBe(true);
  });

  it('strictPrograms=true: 壊れたprogramsは例外(フォールバック禁止)', async () => {
    const doc = loadGoldenBundleDoc();
    corruptFirstProgram(doc);
    await expect(loadBundle(doc, { engine: 'js', strictPrograms: true }))
      .rejects.toThrow(/tt=99/);
  });

  it('strictPrograms=true: 正常なprogramsなら通る', async () => {
    const doc = loadGoldenBundleDoc();
    const bundle = await loadBundle(doc, { engine: 'js', strictPrograms: true });
    expect(bundle.kind).toBe('seg');
  });

  it('forceRebuildPrograms=true: バンドルのprogramsを無視して再構築しても出力は同一', async () => {
    const doc = loadGoldenBundleDoc();
    const normal = await loadBundle(doc, { engine: 'js' });
    const forced = await loadBundle(doc, { engine: 'js', forceRebuildPrograms: true });
    expect(forced.notes.some((n) => n.includes('forceRebuildPrograms=true'))).toBe(true);
    expect(forced.notes.some((n) => n.includes('built=2'))).toBe(true);
    const bits = randomBits(normal.runtime.inWidth, 3);
    expect(Array.from(forced.scores(bits, 1).data)).toEqual(
      Array.from(normal.scores(bits, 1).data),
    );
  });
});
