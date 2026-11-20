// シェルが描く操作UIのIDと、ページ側TSが getElementById で引くIDの突き合わせ。
//
// この改修でいちばん壊れやすいのが「サイドバーに #alpha を描き忘れて seg の
// 起動時に null 参照で落ちる」型の抜けなので、ソースを走査して機械的に検出する。
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { controlIds } from './controls';
import { PAGES } from './pages';

const src = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** ページ本文(HTML)が持つ要素。シェルの守備範囲ではない。 */
const OWNED_BY_HTML = new Set([
  'shell', 'grid', 'samples', 'tasks', 'video', 'overlay', 'stats', 'modelmeta',
  'position', 'status',
]);

/** ページ側TSがIDで引いている要素を列挙する。 */
function referencedIds(file: string): string[] {
  const text = readFileSync(resolve(src, file), 'utf8');
  const ids = new Set<string>();
  const patterns = [
    /getElementById\('([A-Za-z][\w-]*)'\)/g,
    /\$(?:<[^>]+>)?\('([A-Za-z][\w-]*)'\)/g,
    /bindBackendSelect\('[a-z]+', '([A-Za-z][\w-]*)'/g,
    /initialBackend\('([A-Za-z][\w-]*)'/g,
  ];
  for (const re of patterns) {
    for (const m of text.matchAll(re)) ids.add(m[1]);
  }
  return [...ids].sort();
}

/** ページ設定 → そのページの入口TS(検出4画面は共通実装を見る)。 */
const SOURCES: Record<string, string> = {
  index: 'index_main.ts',
  det: 'ui/detPage.ts',
  detmono: 'ui/detPage.ts',
  detface: 'ui/detPage.ts',
  detreal: 'ui/detPage.ts',
  seg: 'seg_main.ts',
  har: 'har_main.ts',
  webcam: 'webcam_main.ts',
};

describe('操作UIのIDがページ側TSの参照と一致する', () => {
  for (const [key, cfg] of Object.entries(PAGES)) {
    it(key, () => {
      const provided = new Set([...controlIds(cfg.controls), ...OWNED_BY_HTML]);
      const missing = referencedIds(SOURCES[key]).filter((id) => !provided.has(id));
      expect(missing, `${key}: シェルが描いていないID`).toEqual([]);
    });
  }
});

describe('操作UIの定義', () => {
  it('IDが重複しない', () => {
    for (const [key, cfg] of Object.entries(PAGES)) {
      const ids = controlIds(cfg.controls);
      expect(new Set(ids).size, key).toBe(ids.length);
    }
  });
});
