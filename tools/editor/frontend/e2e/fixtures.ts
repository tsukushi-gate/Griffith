// E2E共通fixture: 実行時クラッシュ(無限レンダーループ等)の自動検出。
//
// Reactの "Maximum update depth exceeded" は console.error で出るだけで throw
// しない(あるいはUncaught Errorとして pageerror にも出る)ため、UI条件だけをassertする
// 既存specでは素通りしてしまう。ここで page の pageerror / console.error を全spec共通で
// 監視し、レンダーループ系の致命エラーが1件でも出たらテストを失敗させる。
//
// 使い方: 各specは `@playwright/test` の代わりに `./fixtures` から test/expect を import する。
// _crashGuard は auto fixture なので、testが参照していなくても全testで有効になる。

import { test as base, expect } from '@playwright/test';

/** これらに一致したら即失敗させる致命パターン(レンダーループ/フック不整合/クラッシュ) */
const FATAL_PATTERNS: RegExp[] = [
  /Maximum update depth exceeded/i,
  /Too many re-renders/i,
  /Rendered more hooks than during the previous render/i,
  /Cannot update a component .* while rendering a different component/i,
  /Rendered fewer hooks than expected/i,
];

/** 無害なノイズ(致命判定から除外)。 */
const IGNORED_PATTERNS: RegExp[] = [
  /favicon\.ico/i,
  /ResizeObserver loop/i,
  /Download the React DevTools/i,
];

function isFatal(text: string): boolean {
  if (IGNORED_PATTERNS.some((re) => re.test(text))) return false;
  return FATAL_PATTERNS.some((re) => re.test(text));
}

export const test = base.extend<{ crashGuard: void }>({
  crashGuard: [
    async ({ page }, use) => {
      const fatal: string[] = [];
      const all: string[] = [];

      page.on('pageerror', (err) => {
        const text = `pageerror: ${err.message}`;
        all.push(text);
        if (isFatal(err.message)) fatal.push(text);
      });
      page.on('console', (msg) => {
        if (msg.type() !== 'error') return;
        const text = `console.error: ${msg.text()}`;
        all.push(text);
        if (isFatal(msg.text())) fatal.push(text);
      });

      await use();

      // 致命エラーが1件でもあれば失敗(全収集ログを添えて原因を明示する)
      expect(
        fatal,
        `実行時クラッシュ(レンダーループ等)を検出しました:\n` +
          `--- fatal ---\n${fatal.join('\n')}\n` +
          `--- all console/page errors ---\n${all.join('\n') || '(none)'}`,
      ).toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
