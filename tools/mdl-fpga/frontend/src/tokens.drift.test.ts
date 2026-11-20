// デザイントークンのドリフト監視: エディタ(tools/editor)の :root トークン名は
// すべてビューアの index.css にも存在しなければならない(UI統合の契約)。
// ビューア独自トークン(--edge-on 等)の追加は許容(片方向包含)。
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const HERE = import.meta.dirname;
const VIEWER_CSS = path.resolve(HERE, './index.css');
const EDITOR_CSS = path.resolve(
  HERE, '../../../editor/frontend/src/index.css');

function rootTokenNames(css: string): Set<string> {
  const m = css.match(/:root\s*\{([\s\S]*?)\n\}/);
  const body = m ? m[1] : '';
  return new Set(
    [...body.matchAll(/(--[\w-]+)\s*:/g)].map((x) => x[1]),
  );
}

describe('デザイントークン契約(エディタ ⊆ ビューア)', () => {
  it.skipIf(!existsSync(EDITOR_CSS))(
    'エディタの :root トークン名はビューアに全て存在する', () => {
      const editor = rootTokenNames(readFileSync(EDITOR_CSS, 'utf-8'));
      const viewer = rootTokenNames(readFileSync(VIEWER_CSS, 'utf-8'));
      expect(editor.size).toBeGreaterThan(30);   // 抽出が壊れていない
      const missing = [...editor].filter((t) => !viewer.has(t));
      expect(missing, 'エディタ側で追加されたトークンをビューアにも定義してください')
        .toEqual([]);
    });

  it('ビューアの .dark はライト(:root)と同じトークン名集合を持つ', () => {
    const css = readFileSync(VIEWER_CSS, 'utf-8');
    const light = rootTokenNames(css);
    const dm = css.match(/\.dark\s*\{([\s\S]*?)\n\}/);
    const dark = new Set(
      [...(dm ? dm[1] : '').matchAll(/(--[\w-]+)\s*:/g)].map((x) => x[1]),
    );
    // --radius はテーマ非依存なので除外
    const missing = [...light].filter(
      (t) => t !== '--radius' && !dark.has(t));
    expect(missing, 'ライトのみ定義のトークンがあります(.dark にも定義を)')
      .toEqual([]);
  });
});
