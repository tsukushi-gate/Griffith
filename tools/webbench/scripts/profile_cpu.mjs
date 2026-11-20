#!/usr/bin/env node
// プロファイル取得: 各タスクを wasm バックエンドで走らせ、
// window.__cpuProfile.get() のフレーム内訳を集める。
// 使い方: node profile_cpu.mjs [--port=8790] [--frames=60] [--tasks=det,detface,seg]
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

// webbench ルート(このスクリプトの1つ上)。実行場所に依存させない
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const get = (re, d) => { for (const a of args) { const m = re.exec(a); if (m) return m[1]; } return d; };
const port = Number(get(/^--port=(\d+)$/, '8790'));
const frames = Number(get(/^--frames=(\d+)$/, '60'));
const tasks = get(/^--tasks=(.+)$/, 'det,detmono,detface,detreal,seg').split(',');
const backend = get(/^--backend=(\w+)$/, 'wasm');
const extra = get(/^--extra=(.*)$/, '');
const out = get(/^--out=(.*)$/, resolve(root, 'cpu_profile.json'));

function startServer(p) {
  return new Promise((res, rej) => {
    const proc = spawn('node', ['serve.mjs'], { cwd: root, env: { ...process.env, PORT: String(p) }, stdio: ['ignore', 'pipe', 'pipe'] });
    let done = false;
    proc.stdout.on('data', (b) => {
      if (!done && b.toString().includes('webbench: http://localhost')) { done = true; res(proc); }
    });
    proc.on('error', rej);
    setTimeout(() => { if (!done) rej(new Error('serve timeout')); }, 15000);
  });
}

const server = await startServer(port);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const results = [];
try {
  for (const task of tasks) {
    const url = `http://localhost:${port}/${task}.html?autobench=1&backend=${backend}&frames=${frames}&cpuprofile=1${extra ? `&${extra}` : ''}`;
    // 2周目を採用(コールドJIT除去)
    let rec = null;
    for (let round = 1; round <= 2; round++) {
      const page = await browser.newPage();
      const errs = [];
      page.on('pageerror', (e) => errs.push(String(e)));
      try {
        await page.goto(url, { waitUntil: 'load', timeout: 90000 });
        await page.waitForFunction(() => document.title === 'BENCH_DONE', undefined, { timeout: 90000, polling: 100 });
        rec = await page.evaluate(() => ({
          bench: window.__benchResult ?? null,
          cpu: window.__cpuProfile ? window.__cpuProfile.get(8) : null,
          coi: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
          hc: navigator.hardwareConcurrency,
          mt: window.__dlgnMt ?? null,
        }));
        rec.errs = errs;
      } catch (e) {
        rec = { error: String(e), errs };
      } finally { await page.close(); }
    }
    results.push({ task, backend, ...rec });
    const b = rec.bench, c = rec.cpu;
    console.log(`\n=== ${task} (${backend}) median=${b ? b.median.toFixed(3) : '?'}ms actual=${b?.actualBackend} coi=${rec.coi} hc=${rec.hc} mt=${JSON.stringify(rec.mt)}`);
    if (c) {
      console.log(`   frames=${c.frames} frame.total(median)=${c.totalMedianMs.toFixed(3)}ms`);
      for (const l of c.labels) {
        const pct = c.totalMedianMs > 0 ? (100 * l.medianMs / c.totalMedianMs) : 0;
        console.log(`   ${l.label.padEnd(22)} ${l.medianMs.toFixed(4).padStart(9)}ms  ${pct.toFixed(1).padStart(5)}%  calls/frame=${l.callsPerFrame}`);
      }
    } else console.log('   (cpu profile なし)', rec.errs?.slice(0, 3));
  }
} finally {
  await browser.close();
  server.kill();
}
writeFileSync(out, JSON.stringify(results, null, 2));
console.log(`\nwrote ${out}`);
