// det「ランタイム×バックエンド×6指標」マトリクスベンチ。
// DLGN(packed, JS)/ ONNX Runtime Web(wasm/webgpu)/ TF.js(webgl/wasm/webgpu) の
// 6組み合わせを同一 カラーMulti-MNIST(CIFAR-10背景+色付き数字)サンプルで
// 順に計測する(方法論は segbackends_main.ts と同一)。
//
// DLGN側の説明値はロード済みモデルJSON(modelMeta.ts)、teacher側は
// references.ts の参照値から実行時に描画する。ここにモデル固有の数値を書かない
// (以前はヘッダとHTMLに k64 / 旧mAP が残り、実際に配信していた k256 のモデルと
// 食い違っていた)。DLGN行の精度・box出力は実測値。
import type { BackendMeasurement, EnvInfo } from './compare';
import { DEFAULT_MEASURE_CONFIG, collectEnvInfo, measureRuntime } from './compare';
import {
  DetDlgnRuntime,
  DetOnnxRuntime,
  DetTfjsRuntime,
} from './compareDet';
import type { DetRuntime, DetThermoCtx } from './compareDet';
import { grayFromB64, rgbFromB64 } from './detdemo';
import type { DetModel } from './detdemo';
import { loadDetModel } from './detdemo';
import { asset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { REFERENCES, applyReferences } from './references';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const DLGN_URL = asset('/models/det_multimnist_lut_k256.json');
const ONNX_URL = asset('/models/det_cnn_teacher.onnx');
const TFJS_DIR = asset('/models/det_cnn_teacher_tfjs');

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

    log('DLGN doc からラベル/サンプルを取得…');
    const refModel: DetModel = await loadDetModel(DLGN_URL);
    // 説明文の数値をロード済みモデルJSON/参照値テーブルから描画する。
    const pageMeta = applyModelMeta(refModel.doc);
    applyReferences();
    const labelCtx = (): DetThermoCtx => ({ labels: refModel.labels });
    const S = refModel.w;
    let rgba: Uint8ClampedArray;
    if (refModel.samples.length > 0) {
      const s = refModel.samples[0];
      rgba = new Uint8ClampedArray(S * S * 4);
      if (s.rgb_b64) {
        const rgb = rgbFromB64(s.rgb_b64, s.height * s.width);
        for (let i = 0; i < S * S; i++) {
          rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1];
          rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
        }
      } else {
        const gray = grayFromB64(s.gray_b64 ?? '', s.height * s.width);
        for (let i = 0; i < S * S; i++) {
          const v = gray[i] ?? 0;
          rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
        }
      }
    } else {
      // サンプル未同梱時のフォールバック(決定的擬似ランダム)
      let seed = 20260714;
      const rnd = (): number => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };
      rgba = new Uint8ClampedArray(S * S * 4);
      for (let i = 0; i < S * S; i++) {
        const v = (rnd() * 255) | 0;
        rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
      }
    }

    const cfg = DEFAULT_MEASURE_CONFIG;
    const targets: { backend: string; rt: DetRuntime }[] = [
      { backend: 'js', rt: new DetDlgnRuntime(DLGN_URL, 'js') },
      { backend: 'wasm', rt: new DetDlgnRuntime(DLGN_URL, 'wasm') },
      { backend: 'webgpu', rt: new DetDlgnRuntime(DLGN_URL, 'webgpu') },
      { backend: 'wasm', rt: new DetOnnxRuntime(ONNX_URL, labelCtx, 'wasm') },
      { backend: 'webgpu', rt: new DetOnnxRuntime(ONNX_URL, labelCtx, 'webgpu') },
      { backend: 'webgl', rt: new DetTfjsRuntime(TFJS_DIR, labelCtx, 'webgl') },
      { backend: 'wasm', rt: new DetTfjsRuntime(TFJS_DIR, labelCtx, 'wasm') },
      { backend: 'webgpu', rt: new DetTfjsRuntime(TFJS_DIR, labelCtx, 'webgpu') },
    ];

    const results: BackendMeasurement[] = [];
    for (const t of targets) {
      const m = await measureRuntime(t.backend, t.rt, rgba, cfg, log);
      results.push(m);
    }
    renderTable(results);

    const report = {
      task: 'W7e det backend matrix, color MultiMNIST '
        + '(DLGN / ONNX wasm+webgpu / TF.js webgl+wasm+webgpu)',
      timestamp: new Date().toISOString(),
      env,
      config: cfg,
      input: { height: S, width: S },
      // レポートJSONの説明もメタデータ/参照値から組み立てる(手書きしない)。
      note: `DLGN=${pageMeta.name ?? '?'}(k=${pageMeta.k ?? '?'}) / `
        + `ONNX,TF.js=CenterNetCNN teacher(wandb ${REFERENCES.det_color_teacher.wandb ?? '?'})`,
      dlgnMeta: pageMeta,
      teacherRef: REFERENCES.det_color_teacher,
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
