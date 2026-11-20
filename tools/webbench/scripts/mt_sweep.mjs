#!/usr/bin/env node
// DLGN CPUエンジン(bitplane-wasm)のスレッド数掃引 + 整数一致検証。
//
//   (1) 掃引: 5タスク × ?mt={1,2,4,6,...} の e2e median(autobench の inferMs)。
//       各セルは bench_matrix.mjs と同じく2周ロードして2周目を採る。
//   (2) 一致: mtverify.html?mt=1 と ?mt=N の生スコアを全要素突き合わせる。
//   (3) フォールバック: COOP/COEPヘッダ無しの簡易配信で crossOriginIsolated=false
//       にし、単スレッドへ自動フォールバックすること + 結果が変わらないことを確認。
//
// 使い方: node scripts/mt_sweep.mjs [--frames=50] [--threads=1,2,4,6] [--port=8796]
//         [--skip-build] [--no-fallback]
import { execFileSync, spawn } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const TASKS = ['det', 'detmono', 'detface', 'detreal', 'seg'];
const CELL_TIMEOUT_MS = 120_000;

function cli(argv) {
  const o = {
    frames: 50, port: 8796, threads: [1, 2, 4, 6], skipBuild: false, fallback: true,
    tasks: TASKS, repeat: 1,
  };
  for (const a of argv) {
    let m;
    if ((m = /^--frames=(\d+)$/.exec(a))) o.frames = Number(m[1]);
    if ((m = /^--port=(\d+)$/.exec(a))) o.port = Number(m[1]);
    if ((m = /^--threads=([\d,]+)$/.exec(a))) o.threads = m[1].split(',').map(Number);
    if ((m = /^--tasks=(.+)$/.exec(a))) o.tasks = m[1].split(',');
    if ((m = /^--repeat=(\d+)$/.exec(a))) o.repeat = Number(m[1]);
    if (a === '--skip-build') o.skipBuild = true;
    if (a === '--no-fallback') o.fallback = false;
  }
  return o;
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream', '.onnx': 'application/octet-stream',
};

/** COOP/COEP を **付けない** 簡易静的サーバ(crossOriginIsolated=false の再現用)。 */
function startPlainServer(port) {
  const ROOT = join(root, 'dist');
  return new Promise((res) => {
    const srv = createServer((req, rq) => {
      const url = new URL(req.url, `http://${req.headers.host}`);
      let p = normalize(decodeURIComponent(url.pathname));
      if (p === '/' || p === '') p = '/index.html';
      const file = join(ROOT, p);
      if (!file.startsWith(ROOT) || !existsSync(file) || !statSync(file).isFile()) {
        rq.writeHead(404).end('not found');
        return;
      }
      rq.writeHead(200, {
        'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      createReadStream(file).pipe(rq);
    }).listen(port, () => res(srv));
  });
}

function startServer(port) {
  return new Promise((res, rej) => {
    const proc = spawn('node', ['serve.mjs'], {
      cwd: root, env: { ...process.env, PORT: String(port) }, stdio: ['ignore', 'pipe', 'pipe'],
    });
    let done = false;
    proc.stdout.on('data', (b) => {
      if (!done && b.toString().includes('webbench: http://localhost')) { done = true; res(proc); }
    });
    proc.on('error', rej);
    setTimeout(() => { if (!done) rej(new Error('serve.mjs 起動タイムアウト')); }, 15_000);
  });
}

async function measureCell(browser, port, task, threads, frames) {
  const url = `http://localhost:${port}/${task}.html`
    + `?autobench=1&backend=wasm&frames=${frames}&mt=${threads}`;
  let last = null;
  for (let round = 1; round <= 2; round++) {
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
    try {
      await page.goto(url, { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
      await page.waitForFunction(() => document.title === 'BENCH_DONE', undefined,
        { timeout: CELL_TIMEOUT_MS, polling: 100 });
      last = await page.evaluate(() => ({
        bench: window.__benchResult ?? null,
        mt: window.__dlgnMt ?? null,
        coi: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
      }));
      last.errs = errs;
    } catch (e) {
      last = { error: String(e), errs };
    } finally { await page.close(); }
  }
  return last;
}

async function collectScores(browser, port, threads) {
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  try {
    await page.goto(`http://localhost:${port}/mtverify.html?mt=${threads}`,
      { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
    await page.waitForFunction(
      () => document.title === 'MTVERIFY_DONE' || document.title === 'MTVERIFY_ERROR',
      undefined, { timeout: CELL_TIMEOUT_MS, polling: 100 },
    );
    return await page.evaluate(() => ({
      scores: window.__mtScores ?? null,
      info: window.__mtInfo ?? null,
      coi: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
      title: document.title,
    }));
  } catch (e) {
    return { scores: null, error: String(e), errs };
  } finally { await page.close(); }
}

function compareScores(a, b) {
  if (!a || !b) return { ok: false, detail: 'スコア未取得' };
  const tasks = Object.keys(a);
  for (const t of tasks) {
    const ra = a[t]; const rb = b[t];
    if (!rb || ra.length !== rb.length) return { ok: false, detail: `${t}: 試行数不一致` };
    for (let i = 0; i < ra.length; i++) {
      if (ra[i].length !== rb[i].length) {
        return { ok: false, detail: `${t}#${i}: 長さ ${ra[i].length} vs ${rb[i].length}` };
      }
      for (let k = 0; k < ra[i].length; k++) {
        if (ra[i][k] !== rb[i][k]) {
          return { ok: false, detail: `${t}#${i}: index=${k} ${ra[i][k]} vs ${rb[i][k]}` };
        }
      }
    }
  }
  const n = tasks.reduce((s, t) => s + a[t].reduce((q, r) => q + r.length, 0), 0);
  return { ok: true, detail: `${tasks.length}タスク × 全${n}要素が完全一致` };
}

async function main() {
  const opt = cli(process.argv.slice(2));
  if (!opt.skipBuild) {
    console.log('[mt_sweep] build ...');
    for (const s of ['asbuild:release', 'asbuild:simd', 'asbuild:mt']) {
      execFileSync('npm', ['run', s], { cwd: root, stdio: 'inherit' });
    }
    execFileSync('npx', ['vite', 'build'], { cwd: root, stdio: 'inherit' });
  }

  const serveProc = await startServer(opt.port);
  const browser = await chromium.launch({ headless: true });
  const report = { generatedAt: new Date().toISOString(), frames: opt.frames, cells: [] };
  try {
    const env = await (async () => {
      const p = await browser.newPage();
      await p.goto(`http://localhost:${opt.port}/`, { waitUntil: 'domcontentloaded' });
      const e = await p.evaluate(() => ({
        ua: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        coi: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
      }));
      await p.close();
      return e;
    })();
    report.env = env;
    console.log('[mt_sweep] env:', JSON.stringify(env));

    // (1) 掃引。マシンが他エージェントのビルドで負荷変動するため、
    // 全セルを1周してから次の周へ行く(=遅い時間帯が特定セルに偏らない)
    // 交互測定を repeat 回まわし、セルごとの **最小 median** を採る
    // (最小 = 干渉が最も少なかった回。平均だと外乱に引きずられる)。
    console.log('\n=== スレッド数掃引(wasmバックエンド, e2e median ms)===');
    const best = new Map();
    const all = new Map();
    for (let rep = 0; rep < opt.repeat; rep++) {
      for (const task of opt.tasks) {
        for (const t of opt.threads) {
          const c = await measureCell(browser, opt.port, task, t, opt.frames);
          const med = c?.bench?.median ?? null;
          const key = `${task}|${t}`;
          report.cells.push({ rep, task, threads: t, median: med, mt: c?.mt ?? null,
            coi: c?.coi ?? null, errs: c?.errs ?? [], error: c?.error ?? null });
          if (med !== null) {
            if (!best.has(key) || med < best.get(key)) best.set(key, med);
            all.set(key, [...(all.get(key) ?? []), med]);
          }
        }
      }
      process.stdout.write(`  (pass ${rep + 1}/${opt.repeat} 完了)\n`);
    }
    console.log(['task'.padEnd(9), ...opt.threads.map((t) => `mt=${t}`.padStart(9))].join(' '));
    for (const task of opt.tasks) {
      const row = [task.padEnd(9)];
      for (const t of opt.threads) {
        const v = best.get(`${task}|${t}`);
        row.push((v === undefined ? 'ERR' : v.toFixed(3)).padStart(9));
      }
      console.log(row.join(' '));
    }
    report.bestMedian = Object.fromEntries(best);
    report.allMedians = Object.fromEntries(all);

    // (2) 整数一致(mt=1 基準)
    console.log('\n=== 整数一致検証(mtverify.html)===');
    const base = await collectScores(browser, opt.port, 1);
    console.log(`  mt=1: coi=${base.coi} ${JSON.stringify(base.info)}`);
    report.equality = [];
    for (const t of opt.threads.filter((x) => x !== 1)) {
      const got = await collectScores(browser, opt.port, t);
      const r = compareScores(base.scores, got.scores);
      const actual = got.info ? Object.values(got.info).map((v) => v.actual).join(',') : '?';
      console.log(`  mt=${t} (実効スレッド ${actual}): ${r.ok ? 'MATCH' : 'MISMATCH'} — ${r.detail}`);
      report.equality.push({ threads: t, ok: r.ok, detail: r.detail, info: got.info });
    }
    // mt=0(=単スレッド指定の別表記)も一致することを確認
    const zero = await collectScores(browser, opt.port, 0);
    const rz = compareScores(base.scores, zero.scores);
    console.log(`  mt=0 : ${rz.ok ? 'MATCH' : 'MISMATCH'} — ${rz.detail}`);
    report.equality.push({ threads: 0, ok: rz.ok, detail: rz.detail, info: zero.info });

    // (3) COOP/COEP無しでのフォールバック
    if (opt.fallback) {
      console.log('\n=== フォールバック(COOP/COEPヘッダ無し配信)===');
      const plainPort = opt.port + 1;
      const plain = await startPlainServer(plainPort);
      try {
        const got = await collectScores(browser, plainPort, 4);
        const actual = got.info ? Object.values(got.info).map((v) => v.actual) : [];
        const note = got.info ? Object.values(got.info)[0]?.note : null;
        const r = compareScores(base.scores, got.scores);
        console.log(`  crossOriginIsolated=${got.coi} / mt=4要求 → 実効スレッド ${actual.join(',')}`);
        console.log(`  note: ${note}`);
        console.log(`  スコア一致(COOP/COEPあり mt=1 と): ${r.ok ? 'MATCH' : 'MISMATCH'} — ${r.detail}`);
        report.fallback = { coi: got.coi, actual, note, ok: r.ok, detail: r.detail };
      } finally { plain.close(); }
    }
  } finally {
    await browser.close();
    serveProc.kill();
  }
  const outDir = resolve(root, 'bench-results');
  mkdirSync(outDir, { recursive: true });
  const p = resolve(outDir, `${new Date().toISOString().replace(/[:.]/g, '-')}_mt_sweep.json`);
  writeFileSync(p, JSON.stringify(report, null, 2));
  console.log(`\n[mt_sweep] wrote ${p}`);
}

main().catch((e) => { console.error('[mt_sweep] 失敗:', e); process.exitCode = 1; });
