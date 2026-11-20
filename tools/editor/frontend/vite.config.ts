// Vite設定: /apiと/ws(学習WebSocket)をバックエンド(EDITOR_BACKEND_PORT、既定8765)へ
// proxyする。学習WSは常に同一オリジン(=この/ws proxy)経由でバックエンドへ張るため、
// フロントのポートやバックエンドのポートがずれてもWSとRESTは同じバックエンドに追従する
// (別ホストへ直結したい場合のみ VITE_EDITOR_WS_BASE_URL でopt-in)。
/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

// E2E(playwright.config.ts)が開発サーバと衝突しない代替ポートへ差し替えるためのenv
const BACKEND_PORT = process.env.EDITOR_BACKEND_PORT ?? '8765';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // 純ロジックテストはnode環境で実行する。
  // DOMが必要なテストはファイル先頭の `// @vitest-environment jsdom` コメントで切り替える。
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
  build: {
    rollupOptions: {
      output: {
        // ベンダー分割: 初期ロードから重量級ライブラリを切り離す。
        // deck.gl系はAppShellのlazy importと合わせてL0表示時のみ取得される。
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined;
          // deck.glの間接依存(@loaders.gl/apache-arrow/h3-js等)も同チャンクへ
          if (
            /[\\/](@deck\.gl|@luma\.gl|@math\.gl|@probe\.gl|@loaders\.gl)[\\/]/.test(id) ||
            /[\\/](apache-arrow|h3-js|hammerjs|mjolnir\.js)[\\/]/.test(id)
          ) {
            return 'deckgl';
          }
          if (id.includes('/@xyflow/')) return 'xyflow';
          if (id.includes('/uplot/')) return 'uplot';
          // elkjsはautoLayoutの動的importからのみ参照される(専用チャンク必須。
          // 'vendor'に混ぜると初期ロードへ引き戻されてしまう)
          if (id.includes('/elkjs/')) return 'elk';
          return 'vendor';
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://127.0.0.1:${BACKEND_PORT}`,
        ws: true,
        changeOrigin: true,
        // localhostは全ポートでCookieを共有するため、他プロジェクトの巨大Cookie
        // (例: Supabaseのauth-token)がWSハンドシェイクに乗ると、バックエンドの
        // websockets(1行8192B上限)が 400 "line too long" で拒否してしまう。
        // 学習WSはCookieを使わないためproxyで除去して免疫化する(e2e:
        // ws-cookie-immunity.spec.ts)。
        configure: (proxy) => {
          proxy.on('proxyReqWs', (proxyReq) => proxyReq.removeHeader('cookie'));
        },
      },
    },
  },
});
