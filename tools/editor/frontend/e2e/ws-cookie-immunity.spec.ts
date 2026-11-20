// WS Cookie免疫の回帰E2E: localhostは全ポートでCookieを共有するため、他プロジェクトの
// 巨大Cookie(例: Supabaseのauth-token。4KB制限のため .0/.1 にチャンク分割される)が
// 学習WSのハンドシェイクに乗る。バックエンドのwebsocketsは1行8192B上限なので、
// vite /ws proxyがCookieを除去しないと 400 "line too long" で接続できない。
// 本specはその状況を再現し、ハンドシェイクが成立する(=onopen後にアプリ層の4004で
// 閉じる)ことを検証する。

import { expect, test } from './fixtures';

/** ブラウザの1Cookie上限(約4KB)未満に収めつつ、合計でヘッダ行8192Bを超えるサイズ */
const COOKIE_CHUNK_BYTES = 3200;
const COOKIE_CHUNKS = 3;

test('巨大なlocalhost Cookieがあっても学習WSのハンドシェイクが成立する', async ({ page }) => {
  await page.goto('/');

  await page.evaluate(
    ({ chunkBytes, chunks }) => {
      for (let i = 0; i < chunks; i++) {
        document.cookie = `fat_cookie.${i}=${'x'.repeat(chunkBytes)}; path=/`;
      }
    },
    { chunkBytes: COOKIE_CHUNK_BYTES, chunks: COOKIE_CHUNKS },
  );

  // Cookieが実際に積まれた(=ブラウザに拒否されていない)ことを前提確認
  const cookieLen = await page.evaluate(() => document.cookie.length);
  expect(cookieLen).toBeGreaterThan(8192);

  // 存在しないセッションへのWS: ハンドシェイクが通れば accept→close(4004) になる。
  // Cookieで400拒否されると onopen せず code=1006 で閉じる。
  const result = await page.evaluate(
    () =>
      new Promise<string>((resolve) => {
        const ws = new WebSocket(`ws://${window.location.host}/ws/train/e2e-nonexistent`);
        let opened = false;
        ws.onopen = () => {
          opened = true;
        };
        ws.onclose = (e) => resolve(`opened=${opened} code=${e.code}`);
        setTimeout(() => resolve(`opened=${opened} code=timeout`), 10_000);
      }),
  );
  expect(result).toBe('opened=true code=4004');
});
