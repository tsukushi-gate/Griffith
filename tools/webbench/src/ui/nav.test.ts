// サイドバーのタスク一覧。手コピーしていた <nav> を1箇所へ集約した回帰テスト。
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { NAV, navItems, navLabel } from './nav';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

describe('サイドバーのタスク一覧', () => {
  it('現在地はちょうど1件だけ強調される', () => {
    const items = navItems('detface');
    expect(items.filter((i) => i.current).map((i) => i.id)).toEqual(['detface']);
  });

  it('リンク先のHTMLが全て実在する', () => {
    const missing = NAV
      .map((n) => n.href.replace(/^\//, ''))
      .filter((f) => !existsSync(resolve(root, f)));
    expect(missing).toEqual([]);
  });

  it('削除したベンチ画面と開発用の検証画面は載せない', () => {
    const hrefs = NAV.map((n) => n.href);
    for (const gone of [
      '/detbench.html', '/convbench.html', '/detbackends.html', '/segbackends.html',
      '/wasmgpu_verify.html', '/mtverify.html',
    ]) {
      expect(hrefs).not.toContain(gone);
    }
  });

  it('idとラベルが1対1で、空文字が無い', () => {
    const ids = NAV.map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const n of NAV) {
      expect(navLabel(n.id)).toBe(n.label);
      expect(n.label.length).toBeGreaterThan(0);
      expect(n.hint.length).toBeGreaterThan(0);
    }
  });

  it('URLは basePath 経由なので配信ベースパスが前置される', () => {
    // ローカル(BASE_URL='/')では href と同一。サブパス配信では asset() が前置する。
    for (const item of navItems('index')) {
      expect(item.url.endsWith(item.href)).toBe(true);
    }
  });
});
