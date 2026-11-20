// mdl-fpga viewer のビルド/開発設定。
// dev: /api,/ws をバックエンド(既定 8788)へプロキシ(WS の Cookie 除去は
// editor の実証パターン=localhost 共有 Cookie 肥大で WS が 400 になる罠対策)。
// build: dist/ を FastAPI が静的配信する(エンドユーザに Node 不要)。
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { defineConfig } from 'vite';

const BACKEND = `http://127.0.0.1:${process.env.MDL_FPGA_PORT ?? '8788'}`;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          xyflow: ['@xyflow/react'],
          elk: ['elkjs/lib/elk.bundled.js'],
        },
      },
    },
  },
  server: {
    port: 5175,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': {
        target: BACKEND,
        ws: true,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReqWs', (proxyReq) => {
            proxyReq.removeHeader('cookie');
          });
        },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
} as Parameters<typeof defineConfig>[0]);
