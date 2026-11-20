// Playwright E2E設定。
// uvicorn(バックエンド)とvite dev(フロント)をwebServerで両起動する。
// 開発サーバ(8765/5173)と衝突しないよう、E2E専用の代替ポート(9765/6173)を既定にする。
// 前提: リポジトリルートで `uv sync --extra dev --extra cpu --extra editor` 済みであること。

import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const FRONTEND_PORT = process.env.E2E_FRONTEND_PORT ?? '6173';
const BACKEND_PORT = process.env.E2E_BACKEND_PORT ?? '9765';

const FRONTEND_DIR = fileURLToPath(new URL('.', import.meta.url));
const REPO_ROOT = fileURLToPath(new URL('../../..', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  // バックエンドSTATEを共有するため直列実行(セッションの同時実行は1つのみ)
  fullyParallel: false,
  workers: 1,
  // flaky回避: 自動リトライ1回(CI/ローカル共通)
  retries: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${FRONTEND_PORT}`,
    trace: 'on-first-retry',
    locale: 'ja-JP',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      // バックエンド(cwd=リポジトリルート)。/api/healthで起動完了を判定する
      command: `uv run --no-sync uvicorn tools.editor.backend.main:app --port ${BACKEND_PORT}`,
      url: `http://127.0.0.1:${BACKEND_PORT}/api/health`,
      cwd: REPO_ROOT,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      // フロント(vite dev)。EDITOR_BACKEND_PORTでproxy先を代替ポートへ向ける
      command: `npm run dev -- --port ${FRONTEND_PORT} --strictPort`,
      url: `http://localhost:${FRONTEND_PORT}`,
      cwd: FRONTEND_DIR,
      env: { EDITOR_BACKEND_PORT: BACKEND_PORT },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
