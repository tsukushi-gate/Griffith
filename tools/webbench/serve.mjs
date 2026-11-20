// gzip / brotli 対応の静的サーバ(WB-6)。ペイロード計測(transferSize)を
// 実配信条件で行うために使う(vite devサーバは非圧縮のため)。COOP/COEPヘッダ付き
// (ort-wasmマルチスレッドのcrossOriginIsolated要件)。
//   使い方: npm run build && npm run serve  →  http://localhost:8787
//
// **brotli を足したのは比較の公平性のため**。実CDN(Vercel/Cloudflare/CF)は
// 既定で brotli を返す。gzip だけで測ると **wasm を配る側が不当に重く見える**:
// ORT の wasm は gzip 2.93MB に対し brotli 1.84MB(1.6倍差)、DLGN 側の
// JSON は 0.30 → 0.28MB(1.07倍差)しか変わらない。圧縮の効きが両者で
// 違うので、片方だけ有利な条件で測ってはいけない。
//   BENCH_NO_BR=1 で brotli を無効化できる(gzip のみの旧条件を再現する用)。
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { createBrotliCompress, createGzip, constants as zc } from 'node:zlib';

// ROOT を上書きできるようにした。GitHub Pages はリポジトリ名の
// サブパス配下で配信されるため、`dist` を <root>/<repo>/ に置いた構造で
// 事前検証する用途に使う(SERVE_ROOT=/tmp/pages-test)。
const ROOT = process.env.SERVE_ROOT || new URL('./dist', import.meta.url).pathname;
const PORT = Number(process.env.PORT || 8787);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.onnx': 'application/octet-stream',
};

// テキスト系 + wasm + onnx + bin を圧縮する(実CDNの標準構成)。
const COMPRESS_EXT = new Set(
  ['.html', '.js', '.mjs', '.css', '.json', '.wasm', '.onnx', '.bin']);
const NO_BROTLI = process.env.BENCH_NO_BR === '1';

createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let path = normalize(decodeURIComponent(url.pathname));
  if (path === '/' || path === '') path = '/index.html';
  const file = join(ROOT, path);
  if (!file.startsWith(ROOT) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end('not found');
    return;
  }
  const ext = extname(file);
  const headers = {
    'Content-Type': MIME[ext] ?? 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cache-Control': 'no-store',
  };
  const enc = req.headers['accept-encoding'] || '';
  const canBr = !NO_BROTLI && /\bbr\b/.test(enc);
  const canGz = /\bgzip\b/.test(enc);
  if (COMPRESS_EXT.has(ext) && (canBr || canGz)) {
    headers['Content-Encoding'] = canBr ? 'br' : 'gzip';
    headers.Vary = 'Accept-Encoding';
    res.writeHead(200, headers);
    const z = canBr
      ? createBrotliCompress({ params: { [zc.BROTLI_PARAM_QUALITY]: 5 } })
      : createGzip({ level: 6 });
    createReadStream(file).pipe(z).pipe(res);
  } else {
    headers['Content-Length'] = statSync(file).size;
    res.writeHead(200, headers);
    createReadStream(file).pipe(res);
  }
}).listen(PORT, () => {
  console.log(`webbench: http://localhost:${PORT} (root=${ROOT})`);
});
