#!/usr/bin/env node
// bitplane-wasm のマルチスレッド実行が単スレッド実行と
// **整数完全一致** することを実ブラウザで確認する(mtverify.html のドライバ)。
//
// vitest(node)には Worker / SharedArrayBuffer(crossOriginIsolated)が無く、
// スレッド分割の一致は原理的にブラウザでしか確認できない。wasmgpu_verify.html
// が GPU 実行系に対して果たしている役割の CPU 版で、mtverify_main.ts の
// 冒頭コメントが参照しているドライバがこれ。
//
//   node scripts/mt_verify.mjs [--mt=4] [--port=8792]
//
// 前提: `npm run build` 済み(serve.mjs が dist/ を配信する)。COOP/COEP は
// serve.mjs が付けるので crossOriginIsolated=true になる(=実スレッドが立つ)。
// 単スレッド(?mt=1)と ?mt=N を**別ページロード**で取り、全要素を突き合わせる。
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const { chromium } = await import(`${root}/node_modules/playwright/index.mjs`);

const args = process.argv.slice(2);
const arg = (name, dflt) => {
  const m = args.find((a) => a.startsWith(`--${name}=`));
  return m ? Number(m.slice(name.length + 3)) : dflt;
};
const MT = arg('mt', 4);
const PORT = arg('port', 8792);
const TIMEOUT = 600_000;

function startServer(port) {
  return new Promise((res, rej) => {
    const proc = spawn('node', ['serve.mjs'], {
      cwd: root, env: { ...process.env, PORT: String(port) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let done = false;
    proc.stdout.on('data', (b) => {
      const s = b.toString();
      process.stdout.write(`[serve] ${s}`);
      if (!done && s.includes('webbench: http://localhost')) { done = true; res(proc); }
    });
    proc.stderr.on('data', (b) => process.stderr.write(`[serve:err] ${b}`));
    proc.on('error', rej);
    setTimeout(() => { if (!done) rej(new Error('serve起動タイムアウト')); }, 15000);
  });
}

async function collect(browser, mt) {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('[pageerror]', String(e)));
  try {
    await page.goto(`http://localhost:${PORT}/mtverify.html?mt=${mt}`, { waitUntil: 'load' });
    await page.waitForFunction(
      () => document.title === 'MTVERIFY_DONE' || document.title === 'MTVERIFY_ERROR',
      undefined, { timeout: TIMEOUT, polling: 200 },
    );
    const title = await page.title();
    if (title === 'MTVERIFY_ERROR') {
      throw new Error(await page.evaluate(() => document.getElementById('status').textContent));
    }
    return await page.evaluate(() => ({
      scores: window.__mtScores, info: window.__mtInfo,
      coi: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    }));
  } finally {
    await page.close();
  }
}

const server = await startServer(PORT);
const browser = await chromium.launch({ headless: true });
let ok = true;
try {
  const single = await collect(browser, 1);
  const multi = await collect(browser, MT);
  console.log(`crossOriginIsolated: single=${single.coi} multi=${multi.coi}`);
  console.log(`threads: ${JSON.stringify(multi.info)}`);
  let elements = 0;
  for (const task of Object.keys(single.scores)) {
    const a = single.scores[task];
    const b = multi.scores[task];
    let mismatch = 0;
    for (let t = 0; t < a.length; t++) {
      for (let i = 0; i < a[t].length; i++) {
        elements++;
        if (a[t][i] !== b[t][i]) mismatch++;
      }
    }
    console.log(`${task.padEnd(9)} ${a.length}試行×${a[0].length}要素 `
      + `mismatch=${mismatch} ${mismatch === 0 ? 'MATCH' : 'FAIL'}`);
    if (mismatch !== 0) ok = false;
  }
  console.log(`\n>>> MT_VERIFY(mt=1 vs mt=${MT}) = ${ok ? 'ALL MATCH' : 'MISMATCH'} `
    + `(${elements}要素)`);
} catch (e) {
  console.error('失敗:', e);
  ok = false;
} finally {
  await browser.close();
  server.kill();
}
process.exitCode = ok ? 0 : 1;
