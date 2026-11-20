// Webランタイム(`runtime/web/dist`)のビルド設定。tools/webbench/vite.config.ts
// (比較ツール一式のマルチページビルド)とは別系統: こちらは配布バンドルを
// 読んで動く「ライブラリ本体」だけを3エントリで書き出す。
//   mdl_web.js  … src/deploy/lib.ts(loadBundle等の公開API、ブラウザ用ESM)
//   demo.js     … src/deploy/demo_main.ts(index.htmlが読む単一ページデモ)
//   verify.mjs  … src/deploy/verify_main.ts(node CLI、md deploy verify-web用)
//
// node:*組み込みは verify_main.ts(node:fs)だけが使う。mdl_web.js/demo.jsは
// ブラウザ専用でnode組み込みを一切importしないため、`external`で
// node:*を素通し(バンドルしない)にしても影響しない
// (Rollupは既定でnode組み込みを解決できずエラーになるため明示が必要)。
//
// closeBundleで build/release*.wasm と demo.html(→ dist/index.html)を
// dist直下へコピーする。ソースが無ければ即エラー(黙って空配布にしない)。
import { existsSync, copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import type { Plugin } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, '../../runtime/web/dist');

function copyRuntimeAssetsPlugin(): Plugin {
  return {
    name: 'copy-runtime-web-assets',
    closeBundle() {
      mkdirSync(outDir, { recursive: true });
      const copies: [string, string][] = [
        [resolve(here, 'build/release.wasm'), join(outDir, 'release.wasm')],
        [resolve(here, 'build/release-simd.wasm'), join(outDir, 'release-simd.wasm')],
        [resolve(here, 'src/deploy/demo.html'), join(outDir, 'index.html')],
      ];
      for (const [src, dst] of copies) {
        if (!existsSync(src)) {
          throw new Error(
            `copy-runtime-web-assets: コピー元が無い: ${src}`
            + '(npm run asbuild:release / asbuild:simd を先に実行すること)',
          );
        }
        copyFileSync(src, dst);
      }
    },
  };
}

export default defineConfig({
  plugins: [copyRuntimeAssetsPlugin()],
  // 既定のpublicDir('public/'、比較ツール用のmodels/mediapipe/dataで数百MB)を
  // 配布ライブラリへ巻き込まない(ここはビルド成果物のみを書き出す)。
  publicDir: false,
  build: {
    target: 'es2022',
    outDir,
    emptyOutDir: true,
    lib: {
      entry: {
        mdl_web: resolve(here, 'src/deploy/lib.ts'),
        demo: resolve(here, 'src/deploy/demo_main.ts'),
        verify: resolve(here, 'src/deploy/verify_main.ts'),
      },
      formats: ['es'],
    },
    rollupOptions: {
      external: (id: string) => id.startsWith('node:'),
      output: {
        entryFileNames: (chunk) => (chunk.name === 'verify' ? 'verify.mjs' : '[name].js'),
      },
    },
  },
  // wasmThreadPool.ts(engines/conv.ts経由で静的import)が
  // `new Worker(new URL('./wasmWorker.ts', import.meta.url), {type:'module'})`
  // を持つため、Workerチャンクをesモジュールとして書き出す
  // (threads=1既定ではWorker生成コード自体は実行されないが、Rollupの
  // Worker検出は静的解析なのでビルド時には必ず出力される)。
  worker: {
    format: 'es',
  },
});
