// mdlb.ts(`.mdlb` バイナリ形式のTSデコーダ)のテスト。
//
// 契約:
//   - Pythonで生成した実バイナリ(`fixtures/bundle_golden.mdlb`、
//     `griffith_logic/deploy/mdlb.py::write_mdlb` で
//     `tools/webbench/src/fixtures/bundle_golden.json` の `bundle_doc` から
// 生成、実装時に1回だけ生成してコミットした固定フィクスチャ)を
//     decodeMdlbで読むと、元のJSON(bundle_doc)と構造的に一致する
//     (cross-language golden test — Python⇔TSのバイト単位互換性そのもの
//     を検証する、単体でのTS内ラウンドトリップより強い契約)
//   - lib.ts::loadBundle は .mdlb の ArrayBuffer を渡しても、同じdocを
//     JSON.parse済みで渡した場合と完全一致するscores()を返す
//     (実行一致テスト)
//   - 故障注入(マジック/バージョン/セクション境界/チェックサム破壊)を検出する
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { decodeMdlb } from './mdlb';
import { loadBundle } from './lib';

const here = dirname(fileURLToPath(import.meta.url));
const goldenJsonPath = join(here, '../fixtures/bundle_golden.json');
const goldenMdlbPath = join(here, 'fixtures/bundle_golden.mdlb');
const hasFixtures = existsSync(goldenJsonPath) && existsSync(goldenMdlbPath);

function loadGoldenBundleDoc(): Record<string, unknown> {
  const golden = JSON.parse(readFileSync(goldenJsonPath, 'utf8')) as { bundle_doc: unknown };
  return JSON.parse(JSON.stringify(golden.bundle_doc)) as Record<string, unknown>;
}

function loadGoldenMdlbBytes(): Uint8Array {
  const buf = readFileSync(goldenMdlbPath);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/** 決定的な擬似乱数(loadBundle.test.tsと同一のmulberry32)。 */
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

describe.skipIf(!hasFixtures)('decodeMdlb: Python生成バイナリとのクロス言語ゴールデンテスト', () => {
  it('bundle_golden.mdlbをデコードするとbundle_golden.jsonのbundle_docと構造的に一致する', () => {
    const jsonDoc = loadGoldenBundleDoc();
    const decoded = decodeMdlb(loadGoldenMdlbBytes());
    expect(decoded).toEqual(jsonDoc);
  });

  it('ir.nodesのtruth_tables/indices_flatが復元される(TT/INDICESセクション経由)', () => {
    const jsonDoc = loadGoldenBundleDoc();
    const decoded = decodeMdlb(loadGoldenMdlbBytes());
    const jsonIr = jsonDoc.ir as { nodes: Array<{ id: string; type: string; params: Record<string, unknown> }> };
    const decodedIr = (decoded as { ir: typeof jsonIr }).ir;
    const convNodesJson = jsonIr.nodes.filter((n) => n.type === 'conv2d');
    expect(convNodesJson.length).toBeGreaterThan(0);
    for (const jn of convNodesJson) {
      const dn = decodedIr.nodes.find((n) => n.id === jn.id)!;
      expect(dn.params.truth_tables).toEqual(jn.params.truth_tables);
      expect(dn.params.indices_flat).toEqual(jn.params.indices_flat);
    }
  });

  it('programs節(命令列)が復元される', () => {
    const jsonDoc = loadGoldenBundleDoc();
    const decoded = decodeMdlb(loadGoldenMdlbBytes()) as { programs: Record<string, unknown> };
    expect(decoded.programs).toEqual(jsonDoc.programs);
  });

  it('demo_samples(base64画像を含む)が完全一致で復元される', () => {
    const jsonDoc = loadGoldenBundleDoc();
    const decoded = decodeMdlb(loadGoldenMdlbBytes()) as unknown as { demo_samples: unknown };
    expect(decoded.demo_samples).toEqual(jsonDoc.demo_samples);
  });
});

describe.skipIf(!hasFixtures)('loadBundle: .mdlbバイナリを直接読み込める', () => {
  it('ArrayBufferを渡すとJSON.parse済みdocと完全一致するscores()を返す', async () => {
    const jsonDoc = loadGoldenBundleDoc();
    const mdlbBytes = loadGoldenMdlbBytes();

    const fromJson = await loadBundle(jsonDoc, { engine: 'js' });
    const fromMdlbBuffer = await loadBundle(
      mdlbBytes.buffer.slice(mdlbBytes.byteOffset, mdlbBytes.byteOffset + mdlbBytes.byteLength),
      { engine: 'js' },
    );
    const bits = randomBits(fromJson.runtime.inWidth, 12345);
    const a = fromJson.scores(bits, 1);
    const b = fromMdlbBuffer.scores(bits, 1);
    expect(Array.from(b.data)).toEqual(Array.from(a.data));
    expect(b.width).toBe(a.width);
  });

  it('Uint8Arrayを渡しても同様に動く', async () => {
    const jsonDoc = loadGoldenBundleDoc();
    const mdlbBytes = loadGoldenMdlbBytes();

    const fromJson = await loadBundle(jsonDoc, { engine: 'js' });
    const fromMdlbU8 = await loadBundle(mdlbBytes, { engine: 'js' });
    const bits = randomBits(fromJson.runtime.inWidth, 777);
    const a = fromJson.scores(bits, 1);
    const b = fromMdlbU8.scores(bits, 1);
    expect(Array.from(b.data)).toEqual(Array.from(a.data));
  });

  it('kind判別やlabelsもJSON経由と一致する', async () => {
    const jsonDoc = loadGoldenBundleDoc();
    const mdlbBytes = loadGoldenMdlbBytes();
    const fromJson = await loadBundle(jsonDoc, { engine: 'js' });
    const fromMdlb = await loadBundle(mdlbBytes, { engine: 'js' });
    expect(fromMdlb.kind).toBe(fromJson.kind);
    expect(fromMdlb.doc.labels).toEqual(fromJson.doc.labels);
  });
});

describe('decodeMdlb: 故障注入(コンテナが壊れていたら例外)', () => {
  function goodBytes(): Uint8Array {
    if (!hasFixtures) {
      throw new Error('bundle_golden.mdlb fixtureが無い(このテストはhasFixturesでskipされるはず)');
    }
    return loadGoldenMdlbBytes();
  }

  it.skipIf(!hasFixtures)('マジック不正は例外', () => {
    const bad = new Uint8Array(goodBytes());
    bad[0] = 0x58; // 'X'
    expect(() => decodeMdlb(bad)).toThrow(/マジック/);
  });

  it.skipIf(!hasFixtures)('version不正は例外', () => {
    const bad = new Uint8Array(goodBytes());
    const view = new DataView(bad.buffer);
    view.setUint32(4, 999, true);
    expect(() => decodeMdlb(bad)).toThrow(/version/);
  });

  it.skipIf(!hasFixtures)('切り詰められたバイト列は例外', () => {
    const good = goodBytes();
    const truncated = good.subarray(0, Math.floor(good.length / 2));
    expect(() => decodeMdlb(truncated)).toThrow();
  });

  it.skipIf(!hasFixtures)('ヘッダ未満は例外', () => {
    const good = goodBytes();
    expect(() => decodeMdlb(good.subarray(0, 8))).toThrow(/ヘッダ/);
  });

  it.skipIf(!hasFixtures)('セクションのチェックサム破壊は例外', () => {
    const bad = new Uint8Array(goodBytes());
    bad[bad.length - 1] ^= 0xff;
    expect(() => decodeMdlb(bad)).toThrow(/チェックサム/);
  });
});
