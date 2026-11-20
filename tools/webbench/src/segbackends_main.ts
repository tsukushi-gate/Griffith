// seg「ランタイム×バックエンド×6指標」マトリクスベンチ。
// DLGN(packed, JS)/ ONNX Runtime Web(wasm/webgpu)/ TF.js(webgl/wasm/webgpu) の
// 6組み合わせを同一64×64入力で順に計測する(harness.tsと同じ方法論:
// warmup→単発レイテンシp50/p95→複数試行スループット中央値→メモリ差分→
// ペイロード gzip transfer/decoded)。DLGN以外はバックエンド非対応/失敗時に
// graceful degrade(該当行のみ「利用不可」表示、他は継続)。
import type { BackendMeasurement, EnvInfo } from './compare';
import { DEFAULT_MEASURE_CONFIG, collectEnvInfo, measureRuntime } from './compare';
import {
  SegDlgnRuntime,
  SegOnnxRuntime,
  SegTfjsRuntime,
} from './compareSeg';
import type { SegRuntime } from './compareSeg';
import { asset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { applyReferences } from './references';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const DLGN_URL = asset('/models/seg_convdlgn_k128.json');
const ONNX_URL = asset('/models/seg_unet.onnx');
const TFJS_DIR = asset('/models/seg_unet_tfjs');
const S = 64;

/** 決定的な擬似ランダム 64×64 RGBA(内容は速度・精度メタに無関係、全ランタイム共通)。 */
function makeInput(): Uint8ClampedArray {
  let seed = 20260714;
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const rgba = new Uint8ClampedArray(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    rgba[i * 4] = (rnd() * 255) | 0;
    rgba[i * 4 + 1] = (rnd() * 255) | 0;
    rgba[i * 4 + 2] = (rnd() * 255) | 0;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function fmtBytes(b: number | null): string {
  if (b == null) return '—';
  return b >= 1024 * 1024
    ? `${(b / (1024 * 1024)).toFixed(2)} MB`
    : `${(b / 1024).toFixed(1)} KB`;
}
function fmtMs(v: number | null | undefined): string {
  return v == null ? '—' : `${v.toFixed(2)} ms`;
}

function renderTable(rows: BackendMeasurement[]): void {
  const cols = [
    'ランタイム', 'backend', '状態',
    'モデル(生/gzip)', 'ランタイム資産',
    'ロード ms', 'p50 ms', 'p95 ms', 'min/max ms',
    'スループット samples/s', '精度', 'メモリΔ MB', '備考',
  ];
  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join('')}</tr>`;
  const tbody = rows.map((r) => {
    if (r.status === 'error') {
      return `<tr><td>${r.runtime}</td><td>${r.backend}</td>`
        + `<td class="err">利用不可</td><td colspan="9" class="note">${r.error ?? ''}</td></tr>`;
    }
    const lat = r.warmLatency;
    const thr = r.throughput;
    return `<tr>`
      + `<td>${r.runtime}</td><td>${r.backend}</td><td class="ok">OK</td>`
      + `<td>${fmtBytes(r.modelBytes)} / ${fmtBytes(r.gzipBytes)}</td>`
      + `<td>${fmtBytes(r.runtimeBytes)}</td>`
      + `<td>${fmtMs(r.loadMs)}</td>`
      + `<td>${fmtMs(lat?.median)}</td>`
      + `<td>${fmtMs(lat?.p95)}</td>`
      + `<td>${fmtMs(lat?.min)} / ${fmtMs(lat?.max)}</td>`
      + `<td>${thr ? thr.samplesPerSec.toFixed(1) : '—'}</td>`
      + `<td>${r.metricLabel ?? ''} ${r.metricValue != null ? r.metricValue.toFixed(4) : '—'}</td>`
      + `<td>${r.memDeltaMB != null ? r.memDeltaMB.toFixed(2) : '—'}</td>`
      + `<td class="note">${r.notes.join(' / ')}</td>`
      + `</tr>`;
  }).join('');
  $('table').innerHTML = `<table>${thead}${tbody}</table>`;
}

async function main(): Promise<void> {
  const log = (s: string): void => { $('status').textContent = s; };
  try {
    log('環境情報を取得中…');
    const env: EnvInfo = await collectEnvInfo();
    $('env').textContent =
      `UA: ${env.userAgent} / cores: ${env.hardwareConcurrency} / `
      + `crossOriginIsolated: ${env.crossOriginIsolated} / `
      + `WebGPU API: ${env.webgpuApi} / adapter: ${env.webgpuAdapter ?? 'なし'}`;

    // 説明文の数値をロード済みモデルJSON/参照値テーブルから描画する。
    // `cache: 'no-store'` は必須: このページは各ランタイムの**コールドロード**を
    // 計測するので、ここでHTTPキャッシュを温めると測定値が変わってしまう。
    try {
      const doc = await fetch(DLGN_URL, { cache: 'no-store' }).then((r) => r.json());
      applyModelMeta(doc);
    } catch (e) {
      console.warn('[segbackends] モデルメタの取得に失敗(説明文は「—」表示のまま)', e);
    }
    applyReferences();

    const rgba = makeInput();
    const cfg = DEFAULT_MEASURE_CONFIG;
    const targets: { backend: string; rt: SegRuntime }[] = [
      { backend: 'js', rt: new SegDlgnRuntime(DLGN_URL, 'js') },
      { backend: 'wasm', rt: new SegDlgnRuntime(DLGN_URL, 'wasm') },
      { backend: 'webgpu', rt: new SegDlgnRuntime(DLGN_URL, 'webgpu') },
      { backend: 'wasm', rt: new SegOnnxRuntime(ONNX_URL, 'wasm') },
      { backend: 'webgpu', rt: new SegOnnxRuntime(ONNX_URL, 'webgpu') },
      { backend: 'webgl', rt: new SegTfjsRuntime(TFJS_DIR, 'webgl') },
      { backend: 'wasm', rt: new SegTfjsRuntime(TFJS_DIR, 'wasm') },
      { backend: 'webgpu', rt: new SegTfjsRuntime(TFJS_DIR, 'webgpu') },
    ];

    const results: BackendMeasurement[] = [];
    // 順次計測(GPU/wasmスレッド間の資源競合を避け、harness.tsと同じ
    // 「同一ページで順に測る」方法論。JIT/GC状態は完全独立ではない旨は
    // notesではなくこのコメントに明記)。
    for (const t of targets) {
      results.push(await measureRuntime(t.backend, t.rt, rgba, cfg, log));
    }
    renderTable(results);

    const report = {
      task: 'W7d seg backend matrix (DLGN / ONNX wasm+webgpu / TF.js webgl+wasm+webgpu)',
      timestamp: new Date().toISOString(),
      env,
      config: cfg,
      input: { height: S, width: S },
      engines: results,
    };
    log('完了');
    $('out').textContent = JSON.stringify(report, null, 2);
    (window as unknown as Record<string, unknown>).__BENCH__ = report;
  } catch (e) {
    log(`失敗: ${String(e)}`);
    $('out').textContent = String((e as Error).stack ?? e);
  }
}

void main();
