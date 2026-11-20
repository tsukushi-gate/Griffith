#!/usr/bin/env node
// DLGNブラウザ推論の自動ベンチマーク行列(5タスク×3バックエンド=15セル)。
//
// 手順:
//   (a) npm run build 相当(tsc --noEmit → asbuild:release → asbuild:simd → vite build)。
//       package.json の "build" スクリプトは asbuild:simd を含まないため、ここで
//       明示的に両方実行し release/release-simd 両方の wasm を build/ に用意する
//       (asbuild:simd を省くと wasm バックエンドが SIMD 版を読めず scalar版へ
// 黙ってフォールバックし、実測値と食い違う)。
//   (b) node serve.mjs をサブプロセス起動(gzip配信+COOP/COEP、vite devではなく
// 本番同等の配信で計測する。知見: vite devのコールドモジュール変換は
//       初回ページ実行に4〜8倍のかさ上げを混ぜるため計測には使わない)。
//   (c) Playwright(chromium)で 5タスク×3バックエンド を順に計測。各セルは
//       `<task>.html?autobench=1&backend=<js|wasm|webgpu>&frames=50` を **2回**
//       ロードし、2周目の window.__benchResult を採用する(1周目はコールド
// ロード/初回JITの影響が残るため、知見と同じ理由で捨てる)。
//       タイトルが 'BENCH_DONE' になるのを待つ(タイムアウト90s)。失敗セルは
//       status:'error' で記録し、行列全体は継続する。
//       WebGPU は headless でも `--enable-unsafe-webgpu --use-angle=metal`
//       (macOS/Metal)で実アダプタが取れることを事前確認した上で使う。取れない
//       環境では headless:false へフォールバックする。
//
// 出力: bench-results/<ISO日時>_matrix.json
//   使い方: node scripts/bench_matrix.mjs [--frames=50] [--port=8787]
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

const TASKS = ['det', 'detmono', 'detface', 'detreal', 'seg'];
const BACKENDS = ['js', 'wasm', 'webgpu'];
const CELL_TIMEOUT_MS = 90_000;
const SERVER_START_TIMEOUT_MS = 15_000;

function parseCliArgs(argv) {
  const out = { frames: 50, port: 8787, skipBuild: false, extra: '', backends: null, tasks: null };
  for (const a of argv) {
    const m = /^--frames=(\d+)$/.exec(a);
    if (m) out.frames = Number(m[1]);
    const p = /^--port=(\d+)$/.exec(a);
    if (p) out.port = Number(p[1]);
    if (a === '--skip-build') out.skipBuild = true;
    // 追加のURLクエリ(例 --extra=prog=0 で conv2d命令列化を無効化した
    // 旧経路を計測する)。空なら従来どおり。
    // `--extra=model=_ll` / `--extra=model=_dc` でA3派生モデルへ差し替え
    // られる(basePath.ts::modelAsset。サフィックス形式はタスク非依存なので
    // 5タスク一括のまま比較できる)。`--extra=mt=1`(単スレッド)、
    // `--extra=codegen=1`(js codegen)、`--extra=gpupre=0`(CPU前処理)も同様。
    const e = /^--extra=(.*)$/.exec(a);
    if (e) out.extra = e[1];
    // 部分計測用(A/B比較で時間を節約する)。省略時は全タスク×全バックエンド。
    const b = /^--backends=(.+)$/.exec(a);
    if (b) out.backends = b[1].split(',');
    const t = /^--tasks=(.+)$/.exec(a);
    if (t) out.tasks = t[1].split(',');
  }
  return out;
}

/** 必須モデルJSON(5タスク分)が public/models/ に無ければ実行せず、欠落を報告して
    終了する(要件4: DLGN model docなしでは計測不能)。*/
const REQUIRED_MODELS = {
  det: 'det_multimnist_lut_k256.json',
  detmono: 'det_multimnist_mono_lut_k256.json',
  detface: 'det_face_lut_k256.json',
  detreal: 'det_fewclass_lut_k64.json',
  seg: 'seg_convdlgn_k32.json',
};

function checkRequiredModels() {
  const missing = [];
  for (const [task, file] of Object.entries(REQUIRED_MODELS)) {
    const p = resolve(root, 'public/models', file);
    if (!existsSync(p)) missing.push({ task, file, path: p });
  }
  return missing;
}

function runBuild() {
  const opts = { cwd: root, stdio: 'inherit' };
  console.log('[bench_matrix] tsc --noEmit ...');
  execFileSync('npx', ['tsc', '--noEmit'], opts);
  console.log('[bench_matrix] asbuild:release ...');
  execFileSync('npm', ['run', 'asbuild:release'], opts);
  console.log('[bench_matrix] asbuild:simd ...');
  execFileSync('npm', ['run', 'asbuild:simd'], opts);
  // C3 で追加された shared-memory 版(release-simd-mt)も必ず作り直す。
  // ここを省くと build/ に古い .wasm が残り、`?mt=N` の計測だけが別ソース由来に
  // なる(= スレッド寄与の比較が無効になる)。package.json の "build" と同内容。
  console.log('[bench_matrix] asbuild:mt ...');
  execFileSync('npm', ['run', 'asbuild:mt'], opts);
  console.log('[bench_matrix] vite build ...');
  execFileSync('npx', ['vite', 'build'], opts);
}

/** node serve.mjs をサブプロセスで起動し、リスニングログが出るまで待つ。 */
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

/** navigator.gpu からアダプタが実際に取得できるかを、配信中オリジンで確認する
    (about:blank / file:// はセキュアコンテキスト外で navigator.gpu 自体が
    露出しないため、必ず http(s) 配信オリジンで確認する)。 */
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

/** headless で実アダプタが取れるか確認し、ダメなら headed(headless:false)へ
    フォールバックする(macOS/Metal: --enable-unsafe-webgpu --use-angle=metal)。 */
async function launchBrowserWithWebGpuFallback(port) {
  const args = ['--enable-unsafe-webgpu', '--use-angle=metal'];
  let browser = await chromium.launch({ headless: true, args });
  let webgpuOk = await probeWebGpuAdapter(browser, port);
  let headless = true;
  if (!webgpuOk) {
    console.warn(
      '[bench_matrix] headless では navigator.gpu からアダプタが取れなかった。'
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
              ? { vendor: info.vendor ?? null, architecture: info.architecture ?? null,
                description: info.description ?? null }
              : 'unknown';
          }
        } catch {
          webgpuAdapter = null;
        }
      }
      return {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated:
          typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
        webgpuApi: 'gpu' in navigator,
        webgpuAdapter,
      };
    });
  } finally {
    await page.close();
  }
}

/** 1セル(task×backend)を計測する。同一URLを2回ロードし2周目を採用する
    (知見: 初回ページ実行はコールドモジュール変換/JITの影響が残る)。
    webgpuセルは `&gpuprofile=0` で計測する。timestamp-query の
    プロファイリングは resolveQuerySet と「2本目のmapAsync」を毎フレーム足すため、
    submit→mapAsync の往復がそのまま床になる webgpu では median を 0.5〜1.5ms
    かさ上げしてしまう(=計測が計測対象を変えてしまう)。内訳(breakdown)は
    別途 measureBreakdown() が profiling 有効の追加1周で取り、結果へ合流させる。 */
async function measureCell(browser, port, task, backend, frames, extra = '') {
  const profileOff = backend === 'webgpu' ? '&gpuprofile=0' : '';
  const extraQ = extra ? `&${extra}` : '';
  const url = `http://localhost:${port}/${task}.html`
    + `?autobench=1&backend=${backend}&frames=${frames}${profileOff}${extraQ}`;
  const rounds = [];
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
        () => document.title === 'BENCH_DONE',
        undefined,
        { timeout: CELL_TIMEOUT_MS, polling: 100 },
      );
      const result = await page.evaluate(() => window.__benchResult ?? null);
      rounds.push({ round, result, consoleErrors });
    } catch (e) {
      rounds.push({ round, error: String(e), consoleErrors });
    } finally {
      await page.close();
    }
  }
  const round2 = rounds[1];
  const round1 = rounds[0];
  if (round2?.result) {
    return {
      task, backend, status: 'ok',
      result: round2.result,
      round1Median: round1?.result?.median ?? null,
      round2Median: round2.result.median,
      notes: [...(round1?.consoleErrors ?? []), ...(round2?.consoleErrors ?? [])],
    };
  }
  return {
    task, backend, status: 'error',
    error: round2?.error ?? round1?.error ?? '不明なエラー(__benchResult未設定)',
    round1, round2,
  };
}

/** webgpuセルの内訳(ノード別GPU時間 / CPU側分解)だけを追加1周で取る。
    median はプロファイリング無しの measureCell 側を採用し、ここでは breakdown
    のみを拾う(計測が計測対象を変えないようにするための分離)。 */
async function measureBreakdown(browser, port, task, frames, extra = '') {
  const url = `http://localhost:${port}/${task}.html`
    + `?autobench=1&backend=webgpu&frames=${frames}${extra ? `&${extra}` : ''}`;
  const page = await browser.newPage();
  try {
    await page.goto(url, { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
    await page.waitForFunction(
      () => document.title === 'BENCH_DONE', undefined,
      { timeout: CELL_TIMEOUT_MS, polling: 100 },
    );
    const r = await page.evaluate(() => window.__benchResult ?? null);
    return r
      ? { breakdown: r.breakdown ?? null, medianWithProfiling: r.median }
      : { breakdown: null, medianWithProfiling: null };
  } catch (e) {
    return { breakdown: null, medianWithProfiling: null, error: String(e) };
  } finally {
    await page.close();
  }
}

/** measureBreakdown のノード別詳細版。webgpuBitplane.ts は既定で
    全ノードを単一compute passへ融合する(=getBreakdown()は '(fused-pass)' の
    1エントリにまとまる)ため、ノード別GPU時間の内訳を見るには
    `window.__DLGN_GPU_PROFILE_DETAIL__ = true` を立てて調査用のpass分割経路
    (webgpuBitplane.ts::detail)を有効化する必要がある(手動でdevtoolsから
    立てないと使えなかったフックを自動ベンチから叩けるようにする)。
    既存の measureBreakdown / cell.result.breakdown の意味は一切変えず、
    別フィールド(cell.breakdownDetail)として追加するだけ(スキーマ後方互換)。
    webgpuGraph.ts(packed-webgpuフォールバック時)はもともと常にノード別
    timestampを取るため、detailフラグの有無に関わらず既に内訳が出る。 */
async function measureBreakdownDetail(browser, port, task, frames, extra = '') {
  const url = `http://localhost:${port}/${task}.html`
    + `?autobench=1&backend=webgpu&frames=${frames}${extra ? `&${extra}` : ''}`;
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
    const r = await page.evaluate(() => window.__benchResult ?? null);
    return r
      ? { breakdownDetail: r.breakdown ?? null, medianWithProfilingDetail: r.median }
      : { breakdownDetail: null, medianWithProfilingDetail: null };
  } catch (e) {
    return { breakdownDetail: null, medianWithProfilingDetail: null, error: String(e) };
  } finally {
    await page.close();
  }
}

/** webgpuセルのスループット(2フレームin-flightパイプライン)を
    追加1周で取る。既存セル(逐次レイテンシ)の計測条件は一切変えず、列だけ足す。
    `&pipeline=1` を付けると autobench が逐次計測の**後**に同じフレーム数を
    back-to-backで流し、pipelineMs(全体時間/N)と wallMs(逐次の1フレーム
    実測wall)を __benchResult に載せる。gpuprofile=0 は measureCell と同条件。 */
async function measurePipeline(browser, port, task, frames, extra = '') {
  const url = `http://localhost:${port}/${task}.html`
    + `?autobench=1&backend=webgpu&frames=${frames}&gpuprofile=0&pipeline=1`
    + `${extra ? `&${extra}` : ''}`;
  const page = await browser.newPage();
  try {
    await page.goto(url, { waitUntil: 'load', timeout: CELL_TIMEOUT_MS });
    await page.waitForFunction(
      () => document.title === 'BENCH_DONE', undefined,
      { timeout: CELL_TIMEOUT_MS, polling: 100 },
    );
    const r = await page.evaluate(() => window.__benchResult ?? null);
    if (!r) return { pipelineMs: null, wallMs: null, seqMedian: null };
    return {
      // エンドツーエンド(前処理+推論+decode): wallMs と比較する
      pipelineMs: r.pipelineMs ?? null,
      wallMs: r.wallMs ?? null,
      // 推論のみ(逐次inferMsと同一区間): meanInferMs / seqMedian と比較する
      pipelineInferMs: r.pipelineInferMs ?? null,
      meanInferMs: r.meanInferMs ?? null,
      seqMedian: r.median ?? null,
      pipelineDepth: r.pipelineDepth ?? null,
    };
  } catch (e) {
    return { pipelineMs: null, wallMs: null, seqMedian: null, error: String(e) };
  } finally {
    await page.close();
  }
}

async function main() {
  const { frames, port, skipBuild, extra, backends, tasks } = parseCliArgs(process.argv.slice(2));
  const taskList = tasks ?? TASKS;
  const backendList = backends ?? BACKENDS;

  const missing = checkRequiredModels();
  if (missing.length > 0) {
    console.error('[bench_matrix] 必須モデルJSONが public/models/ に見つからない。実行を中止する:');
    for (const m of missing) console.error(`  - ${m.task}: ${m.file} (${m.path})`);
    process.exitCode = 1;
    return;
  }

  if (!skipBuild) {
    runBuild();
  } else {
    console.log('[bench_matrix] --skip-build 指定のためビルドをスキップ');
  }

  console.log('[bench_matrix] serve.mjs 起動中...');
  const serveProc = await startServer(port);

  let browser;
  try {
    const launch = await launchBrowserWithWebGpuFallback(port);
    browser = launch.browser;
    console.log(
      `[bench_matrix] chromium起動: headless=${launch.headless} `
      + `webgpuOk=${launch.webgpuOk} args=${JSON.stringify(launch.args)}`,
    );

    const env = await collectEnvInfo(browser, port);
    console.log('[bench_matrix] env:', JSON.stringify(env));

    const cells = [];
    for (const task of taskList) {
      for (const backend of backendList) {
        const t0 = Date.now();
        process.stdout.write(`[bench_matrix] ${task} x ${backend} ... `);
        // eslint-disable-next-line no-await-in-loop -- 行列は意図的に逐次実行(計測干渉を避ける)
        const cell = await measureCell(browser, port, task, backend, frames, extra);
        if (cell.status === 'ok' && backend === 'webgpu'
          && cell.result.actualBackend === 'webgpu') {
          // eslint-disable-next-line no-await-in-loop -- 内訳取得も逐次(計測干渉を避ける)
          const bd = await measureBreakdown(browser, port, task, frames, extra);
          cell.result.breakdown = bd.breakdown ?? undefined;
          cell.medianWithProfiling = bd.medianWithProfiling;
          // ノード別GPU内訳(追加フィールド。既存breakdownの意味は変えない)。
          // eslint-disable-next-line no-await-in-loop -- 内訳(詳細)取得も逐次(計測干渉を避ける)
          const bdDetail = await measureBreakdownDetail(browser, port, task, frames, extra);
          cell.breakdownDetail = bdDetail.breakdownDetail ?? undefined;
          cell.medianWithProfilingDetail = bdDetail.medianWithProfilingDetail;
          // eslint-disable-next-line no-await-in-loop -- パイプライン計測も逐次(計測干渉を避ける)
          cell.pipeline = await measurePipeline(browser, port, task, frames, extra);
        }
        const dt = Date.now() - t0;
        if (cell.status === 'ok') {
          const pl = cell.pipeline;
          const pipeNote = pl?.pipelineMs != null
            ? ` | e2e ${(pl.wallMs ?? NaN).toFixed(2)}→${pl.pipelineMs.toFixed(2)}ms`
              + ` / infer ${(pl.meanInferMs ?? NaN).toFixed(2)}`
              + `→${(pl.pipelineInferMs ?? NaN).toFixed(2)}ms`
            : '';
          console.log(
            `ok median=${cell.result.median.toFixed(2)}ms `
            + `actualBackend=${cell.result.actualBackend}${pipeNote} (${dt}ms)`,
          );
        } else {
          console.log(`ERROR: ${cell.error} (${dt}ms)`);
        }
        cells.push(cell);
      }
    }

    const report = {
      generatedAt: new Date().toISOString(),
      frames,
      extraQuery: extra,
      port,
      headless: launch.headless,
      chromiumArgs: launch.args,
      webgpuHeadlessOk: launch.webgpuOk,
      env,
      cells,
    };

    const outDir = resolve(root, 'bench-results');
    mkdirSync(outDir, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outPath = resolve(outDir, `${stamp}_matrix.json`);
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`[bench_matrix] wrote ${outPath}`);
  } finally {
    if (browser) await browser.close();
    serveProc.kill();
  }
}

main().catch((e) => {
  console.error('[bench_matrix] 失敗:', e);
  process.exitCode = 1;
});
