// テストデータのfetch+復元(WB-6、W4でstemを汎用化)。
// dense系ベンチ用の共有テストデータ(<stem>.bin + <stem>_meta.json)。
// エクスポータが実モデルと共に生成する(実モデル接続は次ラウンド)。
import { decodeI16 } from './preproc';
import type { ModelDoc, TestData, TestMeta } from './types';
import { asset } from './basePath';

export async function fetchTestData(
  // 既定値は asset() で base を前置する。GitHub Pages 等のサブパス配信で
  // '/data' のままだとサイト直下を見て 404 になる(末尾スラッシュが無いため
  // 一括置換から漏れていた)。
  base = asset('/data'), stem = 'test',
): Promise<TestData> {
  const [meta, buf] = await Promise.all([
    fetch(`${base}/${stem}_meta.json`).then((r) => {
      if (!r.ok) throw new Error(`${stem}_meta.json: ${r.status}`);
      return r.json() as Promise<TestMeta>;
    }),
    fetch(`${base}/${stem}.bin`).then((r) => {
      if (!r.ok) throw new Error(`${stem}.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  return decodeI16(buf, meta);
}

/** モデルJSONのfetch(コールドロード計測用にcache無効オプション) */
export async function fetchModel(
  url: string, noStore = false,
): Promise<ModelDoc> {
  const r = await fetch(url, noStore ? { cache: 'no-store' } : undefined);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as ModelDoc;
}
