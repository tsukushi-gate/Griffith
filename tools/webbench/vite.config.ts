// webbench(WB-4): DLGN vs ONNX Runtime Web ブラウザ比較ツールのvite設定。
// - editor frontendのevalCircuit.ts等をroot外から相対importする(fs.allow)
// - onnxruntime-webはbundle版(既定export)を使用: JSローダ内蔵、.wasmは
//   import.meta.url基準で自己解決(CDN不使用・セルフホスト)
// - COOP/COEP: ort-wasmのマルチスレッド(SharedArrayBuffer)に必要
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';

const here = dirname(fileURLToPath(import.meta.url));

export const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// GitHub Pages はリポジトリ名のサブパス配下(/<repo>/)で配信されるため、
// base を '/' のままにすると全アセットが 404 になる。CI から
// VITE_BASE=/<リポジトリ名>/ を渡す。ローカル(dev/serve)は既定の '/'。
const base = process.env.VITE_BASE || '/';

export default defineConfig({
  base,
  // Tailwind CSS v4(CSS-first。tailwind.config.js は持たない)。
  // 走査対象は src/app.css の @source で明示している。
  plugins: [tailwindcss()],
  // ort bundleは.wasmをimport.meta.url基準で解決する。vite devのdep最適化に
  // 取り込まれるとURLが/.vite/deps/配下になり.wasmが404(HTML)になるため除外
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: crossOriginIsolationHeaders,
    fs: {
      // editor frontend の src をroot外importするために許可
      allow: [resolve(here, '../..')],
    },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  build: {
    // ort-wasm系はpublic/ortから実行時ロードするためバンドル解析から除外
    target: 'es2022',
    rollupOptions: {
      input: {
        index: resolve(here, 'index.html'),
        seg: resolve(here, 'seg.html'),
        det: resolve(here, 'det.html'),
        detmono: resolve(here, 'detmono.html'),
        detreal: resolve(here, 'detreal.html'),
        detface: resolve(here, 'detface.html'),
        detbench: resolve(here, 'detbench.html'),
        har: resolve(here, 'har.html'),
        webcam: resolve(here, 'webcam.html'),
        convbench: resolve(here, 'convbench.html'),
        segbackends: resolve(here, 'segbackends.html'),
        detbackends: resolve(here, 'detbackends.html'),
        // 検証用(ナビには出さない。scripts/mt_verify.mjs 等が直接URLで叩く)
        wasmgpuVerify: resolve(here, 'wasmgpu_verify.html'),
        mtverify: resolve(here, 'mtverify.html'),
        gpualubench: resolve(here, 'gpualubench.html'),
      },
    },
  },
  worker: {
    format: 'es',
  },
  test: {
    environment: 'node',
  },
});
