// tools/webbench/*.html が vite.config.ts の build.rollupOptions.input に
// 全て登録されていることを検証する(detmono.html が dist/ から欠落していた既知
// バグの再発防止 + 新規detface.htmlの登録漏れ防止)。dist/ を実際にbuildして
// 確認するのは重い(npm run build)ため、config自体を検証することで軽量に同じ
// 事故を検出する。
// 逆方向(登録されたinputが実在するか)と、サイドバーの導線が全て
// ビルド対象かも見る。ページ削除時に input を消し忘れると build が
// "Could not resolve entry" で落ちるため。
import { existsSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import viteConfig from '../vite.config';
import { NAV } from './ui/nav';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

function registeredPaths(): Set<string> {
  const input = viteConfig.build?.rollupOptions?.input;
  expect(input).toBeTypeOf('object');
  return new Set(
    Object.values(input as Record<string, string>).map((p) => resolve(p)),
  );
}

describe('vite.config.ts の input 登録', () => {
  it('tools/webbench 直下の全 *.html が rollupOptions.input に含まれる', () => {
    const htmlFiles = readdirSync(root).filter((f) => f.endsWith('.html'));
    expect(htmlFiles.length).toBeGreaterThan(0); // 列挙が空でないことを保証

    const registered = registeredPaths();
    const missing = htmlFiles.filter((f) => !registered.has(resolve(root, f)));
    expect(missing).toEqual([]);
  });

  it('登録された input のファイルが実在する(削除し忘れの検出)', () => {
    const dangling = [...registeredPaths()].filter((p) => !existsSync(p));
    expect(dangling).toEqual([]);
  });

  it('detmono.html と detface.html が明示的に登録されている(既知バグの回帰テスト)', () => {
    const registered = registeredPaths();
    expect(registered.has(resolve(root, 'detmono.html'))).toBe(true);
    expect(registered.has(resolve(root, 'detface.html'))).toBe(true);
  });

  it('サイドバーのリンク先が全てビルド対象になっている', () => {
    const registered = registeredPaths();
    const broken = NAV
      .map((n) => n.href.replace(/^\//, ''))
      .filter((f) => !registered.has(resolve(root, f)));
    expect(broken).toEqual([]);
  });
});
