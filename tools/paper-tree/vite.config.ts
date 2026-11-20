// Vite設定。バックエンドを持たない静的アプリなのでproxyは無い。
// devポートはeditor(5173)/その他と衝突しない5175を既定にする。
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    rollupOptions: {
      output: {
        // elkjsはlayout.tsの動的importからのみ参照される(初期ロードから分離)
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('/elkjs/')) return 'elk';
          if (id.includes('/@xyflow/')) return 'xyflow';
          return 'vendor';
        },
      },
    },
  },
  server: { port: 5175, strictPort: false },
});
