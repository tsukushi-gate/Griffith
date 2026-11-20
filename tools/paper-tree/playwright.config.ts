// Playwright設定。バックエンド不要なのでvite devのみ起動する。
// 開発サーバ(5175)と衝突しないE2E専用ポート(6175)を既定にする。

import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const PORT = process.env.E2E_PORT ?? '6175';
const DIR = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    locale: 'ja-JP',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    // tree.jsonを最新化してから起動(他レーンのdata/が未完成でもdanglingは警告に降格)
    command: `npm run dev -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    cwd: DIR,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
