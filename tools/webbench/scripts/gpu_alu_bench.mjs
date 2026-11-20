#!/usr/bin/env node
// deploykit-integrationキャンペーン(GPU効率化)のWebGPU側計測基盤。
// bench_matrix.mjs と同じ「build → serve.mjs起動 → Playwright(chromium、webgpu
// フォールバック込み)」の流儀で、以下3種の計測を1回にまとめて実行し
// results/webbench/gpu_alu_baseline_breakdown.json へ書き出す:
//
//   (1) seg/det系5モデルのノード別GPU時間ベースライン(webgpuバックエンド、
//       window.__DLGN_GPU_PROFILE_DETAIL__=true でノード別内訳を強制)。
//   (2) 純ALU較正(gpualubench.html): (総命令数÷GPU時間)=このマシンの
//       WebGPU実測ALUピーク(instr/s)。scripts/gpu_calib/metal_calib.mm
//       (Metal直計測)との比較列になる。
//   (3) gate2 switch/mask A/Bマイクロベンチ(gpualubench.html、同ページ):
//       出力の完全一致検査込み。scripts/gpu_calib/gate2_ab.mm の比較列。
//
// 各ページは(知見どおり)2回ロードし2周目を採用する(vite devではなく
// serve.mjsの本番同等配信を使う。1周目はモジュール変換/シェーダコンパイルの
// コールド影響が残るため)。
//
// 使い方: node scripts/gpu_alu_bench.mjs [--frames=30] [--port=8787] [--skip-build]
//         [--tasks=det,detmono,detface,detreal,seg]
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..'); // tools/webbench
const repoRoot = resolve(root, '../..');

const TASKS = ['det', 'detmono', 'detface', 'detreal', 'seg'];
const CELL_TIMEOUT_MS = 180_000;
const SERVER_START_TIMEOUT_MS = 15_000;

function parseCliArgs(argv) {
  const out = { frames: 30, port: 8787, skipBuild: false, tasks: null };
  for (const a of argv) {
    const f = /^--frames=(\d+)$/.exec(a);
    if (f) out.frames = Number(f[1]);
    const p = /^--port=(\d+)$/.exec(a);
    if (p) out.port = Number(p[1]);
    if (a === '--skip-build') out.skipBuild = true;
    const t = /^--tasks=(.+)$/.exec(a);
    if (t) out.tasks = t[1].split(',');
  }
  return out;
}

const REQUIRED_MODELS = {
  det: 'det_multimnist_lut_k256.json',
  detmono: 'det_multimnist_mono_lut_k256.json',
  detface: 'det_face_lut_k256.json',
  detreal: 'det_fewclass_lut_k64.json',
  seg: 'seg_convdlgn_k32.json',
};

function checkRequiredModels(tasks) {
  const missing = [];
  for (const task of tasks) {
    const file = REQUIRED_MODELS[task];
    const p = resolve(root, 'public/models', file);
    if (!existsSync(p)) missing.push({ task, file, path: p });
  }
  return missing;
}

// bench_matrix.mjs と同じビルド手順(asbuild:simd/mtを含めて明示実行する理由も同じ:
// package.jsonのbuildスクリプトはasbuild:simdを含まないため)。
function runBuild() {
  const opts = { cwd: root, stdio: 'inherit' };
  console.log('[gpu_alu_bench] tsc --noEmit ...');
  execFileSync('npx', ['tsc', '--noEmit'], opts);
  console.log('[gpu_alu_bench] asbuild:release ...');
  execFileSync('npm', ['run', 'asbuild:release'], opts);
  console.log('[gpu_alu_bench] asbuild:simd ...');
  execFileSync('npm', ['run', 'asbuild:simd'], opts);
  console.log('[gpu_alu_bench] asbuild:mt ...');
  execFileSync('npm', ['run', 'asbuild:mt'], opts);
  console.log('[gpu_alu_bench] vite build ...');
  execFileSync('npx', ['vite', 'build'], opts);
}

function startServer(port) {
  return new Promise((resolvePromise, reject) => {
    const proc = spawn('node', ['serve.mjs'], {
      cwd: root,
      env: { ...process.env, PORT: String(port) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let resolved = false;
    const onData = (buf) => {
      const s = buf.toString();
      process.stdout.write(`[serve] ${s}`);
      if (!resolved && s.includes('webbench: http://localhost')) {
        resolved = true;
        resolvePromise(proc);
      }
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', (buf) => process.stderr.write(`[serve:err] ${buf}`));
    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (!resolved) reject(new Error(`serve.mjs が起動前に終了した(code=${code})`));
    });
    setTimeout(() => {
      if (!resolved) reject(new Error('serve.mjs 起動待ちタイムアウト'));
    }, SERVER_START_TIMEOUT_MS);
  });
}

async function probeWebGpuAdapter(browser, port) {
  const page = await browser.newPage();
  try {
    await page.goto(`http://localhost:${port}/`, { waitUntil: 'domcontentloaded' });
    return await page.evaluate(async () => {
      if (!('gpu' in navigator)) return false;
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
      } catch {
        return false;
      }
    });
  } finally {
    await page.close();
  }
}

async function launchBrowserWithWebGpuFallback(port) {
  const args = ['--enable-unsafe-webgpu', '--use-angle=metal'];
  let browser = await chromium.launch({ headless: true, args });
  let webgpuOk = await probeWebGpuAdapter(browser, port);
  let headless = true;
  if (!webgpuOk) {
    console.warn(
      '[gpu_alu_bench] headless では navigator.gpu からアダプタが取れなかった。'
      + 'headless:false へフォールバックする。',
    );
    await browser.close();
    browser = await chromium.launch({ headless: false, args });
    webgpuOk = await probeWebGpuAdapter(browser, port);
    headless = false;
  }
  return { browser, headless, args, webgpuOk };
}

async function collectEnvInfo(browser, port) {
  const page = await browser.newPage();
  try {
    await page.goto(`http://localhost:${port}/`, { waitUntil: 'domcontentloaded' });
    return await page.evaluate(async () => {
      let webgpuAdapter = null;
      if ('gpu' in navigator) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            const info = adapter.info
              ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
            webgpuAdapter = info
              ? {
                vendor: info.vendor ?? null, architecture: info.architecture ?? null,
                description: info.description ?? null,
              }
              : 'unknown';
          }
        } catch {
          webgpuAdapter = null;
        }
      }
      return {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        webgpuApi: 'gpu' in navigator,
        webgpuAdapter,
      };
    });
  } finally {
    await page.close();
  }
}

/** 1URLを2回ロードし(コールド/ウォーム)、2周目の指定フィールドを採用する
    (知見: vite devではなくserve.mjs配信でも、初回ページ実行はモジュール
    変換/シェーダコンパイルのコールド影響が残るため)。 */
async function loadTwiceAndExtract(browser, url, doneTitles, extractor) {
  let last = null;
  for (let round = 1; round <= 2; round++) {
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    try {
      await page.goto(url, { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
      await page.waitForFunction(
        (titles) => titles.includes(document.title),
        doneTitles,
        { timeout: CELL_TIMEOUT_MS, polling: 100 },
      );
      const title = await page.title();
      const value = await page.evaluate(extractor);
      last = { round, title, value, consoleErrors };
    } catch (e) {
      last = { round, error: String(e), consoleErrors };
    } finally {
      await page.close();
    }
  }
  return last;
}

/** gpualubench.html を2周ロードし、純ALU較正+gate2 A/Bの結果を取る。*/
async function measureGpuAluBench(browser, port) {
  const url = `http://localhost:${port}/gpualubench.html`;
  const r = await loadTwiceAndExtract(
    browser, url, ['GPUALU_DONE', 'GPUALU_ERROR'],
    // eslint-disable-next-line no-undef -- ブラウザコンテキストで評価される
    () => window.__gpuAluBenchResult ?? null,
  );
  return r;
}

/** 1タスクのノード別GPU内訳を2周ロードして取る
    (`window.__DLGN_GPU_PROFILE_DETAIL__=true` をaddInitScriptで注入)。
    bench_matrix.mjs::measureBreakdownDetail と同じ狙い(調査用のpass分割経路を
    自動ベンチから叩く)だが、ここではmedianへの影響を気にせず内訳だけを見る
    独立実行にする(baseline記録が目的のため)。 */
async function measureNodeBreakdown(browser, port, task, frames) {
  const url = `http://localhost:${port}/${task}.html?autobench=1&backend=webgpu&frames=${frames}`;
  let last = null;
  for (let round = 1; round <= 2; round++) {
    const page = await browser.newPage();
    try {
      await page.addInitScript(() => {
        // eslint-disable-next-line no-underscore-dangle -- 既存フックの命名規約に合わせる
        window.__DLGN_GPU_PROFILE_DETAIL__ = true;
      });
      await page.goto(url, { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
      await page.waitForFunction(
        () => document.title === 'BENCH_DONE', undefined,
        { timeout: CELL_TIMEOUT_MS, polling: 100 },
      );
      const result = await page.evaluate(() => window.__benchResult ?? null);
      last = { round, result };
    } catch (e) {
      last = { round, error: String(e) };
    } finally {
      await page.close();
    }
  }
  return last;
}

async function main() {
  const { frames, port, skipBuild, tasks } = parseCliArgs(process.argv.slice(2));
  const taskList = tasks ?? TASKS;

  const missing = checkRequiredModels(taskList);
  if (missing.length > 0) {
    console.error('[gpu_alu_bench] 必須モデルJSONが public/models/ に見つからない。実行を中止する:');
    for (const m of missing) console.error(`  - ${m.task}: ${m.file} (${m.path})`);
    process.exitCode = 1;
    return;
  }

  if (!skipBuild) {
    runBuild();
  } else {
    console.log('[gpu_alu_bench] --skip-build 指定のためビルドをスキップ');
  }

  console.log('[gpu_alu_bench] serve.mjs 起動中...');
  const serveProc = await startServer(port);

  let browser;
  try {
    const launch = await launchBrowserWithWebGpuFallback(port);
    browser = launch.browser;
    console.log(
      `[gpu_alu_bench] chromium起動: headless=${launch.headless} `
      + `webgpuOk=${launch.webgpuOk} args=${JSON.stringify(launch.args)}`,
    );
    const env = await collectEnvInfo(browser, port);
    console.log('[gpu_alu_bench] env:', JSON.stringify(env));

    console.log('[gpu_alu_bench] (2) 純ALU較正 + (3) gate2 A/B ...');
    const aluBench = await measureGpuAluBench(browser, port);
    if (aluBench?.value) {
      console.log(
        `[gpu_alu_bench]   ALU calib mean instr/s = `
        + `${aluBench.value.aluCalibration?.meanInstrPerSec?.toExponential(4)}`,
      );
      for (const c of aluBench.value.gate2Ab?.configs ?? []) {
        console.log(
          `[gpu_alu_bench]   gate2 threads=${c.threads} speedup=${c.speedup.toFixed(3)}x `
          + `match=${c.outputsMatch} 判定=${c.verdict}`,
        );
      }
    } else {
      console.error('[gpu_alu_bench]   !! ALU/gate2計測に失敗:', aluBench?.error);
    }

    console.log('[gpu_alu_bench] (1) ノード別GPU内訳ベースライン ...');
    const nodeBreakdown = {};
    for (const task of taskList) {
      process.stdout.write(`[gpu_alu_bench]   ${task} ... `);
      // eslint-disable-next-line no-await-in-loop -- 行列は意図的に逐次実行(計測干渉を避ける)
      const r = await measureNodeBreakdown(browser, port, task, frames);
      nodeBreakdown[task] = r;
      if (r?.result) {
        const top = r.result.breakdown?.topNodesGpu ?? [];
        console.log(`ok top=${top.length}件 fallback=${r.result.actualBackend}`);
      } else {
        console.log(`ERROR: ${r?.error ?? '不明'}`);
      }
    }

    const report = {
      generatedAt: new Date().toISOString(),
      frames,
      port,
      headless: launch.headless,
      chromiumArgs: launch.args,
      webgpuHeadlessOk: launch.webgpuOk,
      env,
      aluCalibration: aluBench?.value?.aluCalibration ?? null,
      gate2Ab: aluBench?.value?.gate2Ab ?? null,
      aluBenchError: aluBench?.value?.error ?? aluBench?.error ?? null,
      nodeBreakdown: Object.fromEntries(
        Object.entries(nodeBreakdown).map(([task, r]) => [
          task,
          r?.result
            ? {
              actualBackend: r.result.actualBackend,
              median: r.result.median,
              breakdown: r.result.breakdown ?? null,
            }
            : { error: r?.error ?? '不明' },
        ]),
      ),
    };

    const outDir = resolve(repoRoot, 'results/webbench');
    mkdirSync(outDir, { recursive: true });
    const outPath = resolve(outDir, 'gpu_alu_baseline_breakdown.json');
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`[gpu_alu_bench] wrote ${outPath}`);
  } finally {
    if (browser) await browser.close();
    serveProc.kill();
  }
}

main().catch((e) => {
  console.error('[gpu_alu_bench] 失敗:', e);
  process.exitCode = 1;
});
