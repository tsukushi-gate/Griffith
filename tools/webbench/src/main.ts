// webbench UI(WB-6/7): エンジン選択→計測→結果表+JSONダウンロード+デモ。
import { fetchTestData } from './data';
import { startDemo, stopDemo } from './demo';
import {
  DEFAULT_CONFIG,
  makeReport,
  runBench,
  type BenchReport,
  type EngineResult,
  type EngineSpec,
} from './harness';
import { LITERATURE } from './literature';
import type { TestData } from './types';
import { asset } from './basePath';

// convはseg/webcamデモページ(seg.html/webcam.html)が使う(denseベンチ対象外)
interface Manifest { dlgn: string[]; conv?: string[]; onnx: string[] }

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

function fmtBytes(b: number | null): string {
  if (b === null) return '—';
  if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(2)} MB`;
  return `${(b / 1024).toFixed(1)} KB`;
}

function fmtMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function buildSpecs(manifest: Manifest): EngineSpec[] {
  const specs: EngineSpec[] = [];
  const hw = navigator.hardwareConcurrency || 4;
  for (const f of manifest.dlgn) {
    const label = f.replace(/^(har_)?dlgn_/, '').replace(/\.json$/, '');
    specs.push(
      { name: `dlgn-interp (${label})`, kind: 'dlgn', variant: 'interp', modelUrl: `/models/${f}`, maxSamples: 512 },
      { name: `dlgn-packed32 (${label})`, kind: 'dlgn', variant: 'packed32', modelUrl: `/models/${f}` },
      { name: `dlgn-packed64 (${label})`, kind: 'dlgn', variant: 'packed64', modelUrl: `/models/${f}` },
      { name: `dlgn-codegen (${label})`, kind: 'dlgn', variant: 'codegen', modelUrl: `/models/${f}` },
    );
  }
  for (const f of manifest.onnx) {
    const label = f.replace(/\.onnx$/, '');
    specs.push(
      { name: `ort-wasm×1 (${label})`, kind: 'onnx', threads: 1, modelUrl: `/models/${f}` },
      { name: `ort-wasm×${hw} (${label})`, kind: 'onnx', threads: hw, modelUrl: `/models/${f}` },
    );
  }
  return specs;
}

function renderPicker(specs: EngineSpec[]): void {
  const picker = $('engine-picker');
  picker.classList.remove('muted');
  picker.innerHTML = '';
  specs.forEach((s, i) => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.dataset.idx = String(i);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${s.name}`));
    picker.appendChild(label);
  });
}

function selectedSpecs(specs: EngineSpec[]): EngineSpec[] {
  const boxes = $('engine-picker').querySelectorAll<HTMLInputElement>('input');
  const out: EngineSpec[] = [];
  boxes.forEach((cb) => {
    if (cb.checked) out.push(specs[Number(cb.dataset.idx)]);
  });
  return out;
}

function renderRow(tbody: HTMLElement, r: EngineResult, best: {
  cold: number; lat: number; thr: number;
}): void {
  const tr = document.createElement('tr');
  const cells = [
    r.name,
    fmtBytes(r.payload.modelTransferBytes ?? r.payload.modelDecodedBytes),
    fmtBytes(r.payload.runtimeTransferBytes ?? r.payload.runtimeDecodedBytes),
    fmtMs(r.coldLoad.totalMs),
    fmtMs(r.warmLatency.median),
    fmtMs(r.warmLatency.p95),
    `${Math.round(r.throughput.samplesPerSec).toLocaleString()} /s`,
    `${(r.accuracy.acc * 100).toFixed(2)}%`,
    r.memDeltaMB === null ? '—' : `${r.memDeltaMB.toFixed(1)} MB`,
    r.notes.join(' / '),
  ];
  const bestFlags = [
    false, false, false,
    r.coldLoad.totalMs === best.cold,
    r.warmLatency.median === best.lat,
    false,
    r.throughput.samplesPerSec === best.thr,
    false, false, false,
  ];
  cells.forEach((c, i) => {
    const td = document.createElement('td');
    td.textContent = c;
    if (bestFlags[i]) td.classList.add('best');
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
}

function renderResults(results: EngineResult[]): void {
  const table = $('results');
  table.hidden = false;
  const tbody = table.querySelector('tbody')!;
  tbody.innerHTML = '';
  const best = {
    cold: Math.min(...results.map((r) => r.coldLoad.totalMs)),
    lat: Math.min(...results.map((r) => r.warmLatency.median)),
    thr: Math.max(...results.map((r) => r.throughput.samplesPerSec)),
  };
  for (const r of results) renderRow(tbody, r, best);
}

function renderLiterature(): void {
  const tbody = $('literature').querySelector('tbody')!;
  for (const row of LITERATURE) {
    const tr = document.createElement('tr');
    for (const v of [
      row.runtime, row.runtimeSizeNote, row.startupNote, row.throughputNote,
    ]) {
      const td = document.createElement('td');
      td.textContent = v;
      td.style.textAlign = 'left';
      td.style.whiteSpace = 'normal';
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const a = document.createElement('a');
    a.href = row.source;
    a.textContent = 'link';
    a.target = '_blank';
    td.appendChild(a);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

async function init(): Promise<void> {
  $('env').textContent =
    `${navigator.userAgent} / ${navigator.hardwareConcurrency}コア / ` +
    `crossOriginIsolated=${typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated}`;
  renderLiterature();

  let manifest: Manifest;
  let data: TestData;
  try {
    [manifest, data] = await Promise.all([
      fetch(asset('/models/index.json')).then((r) => {
        if (!r.ok) throw new Error(`index.json: ${r.status}`);
        return r.json() as Promise<Manifest>;
      }),
      fetchTestData(),
    ]);
  } catch (e) {
    $('engine-picker').textContent =
      `モデル/データの読み込みに失敗: ${String(e)}` +
      '(scripts/export_webbench.py を先に実行してください)';
    return;
  }

  const specs = buildSpecs(manifest);
  renderPicker(specs);

  const runBtn = $<HTMLButtonElement>('run');
  const dlBtn = $<HTMLButtonElement>('download');
  const progress = $('progress');
  let report: BenchReport | null = null;

  runBtn.disabled = false;
  runBtn.onclick = async () => {
    runBtn.disabled = true;
    const results: EngineResult[] = [];
    const failures: string[] = [];
    for (const spec of selectedSpecs(specs)) {
      try {
        const r = await runBench(spec, data, DEFAULT_CONFIG, (msg) => {
          progress.textContent = msg;
        });
        results.push(r);
        renderResults(results);
      } catch (e) {
        // 1エンジンの失敗(未実装op等)は記録して続行する
        failures.push(`${spec.name}: ${String(e)}`);
        console.error(spec.name, e);
      }
    }
    progress.textContent =
      `完了(${results.length}エンジン` +
      (failures.length ? `、失敗${failures.length}: ${failures.join(' / ')}` : '') +
      ')';
    report = makeReport(results, DEFAULT_CONFIG);
    (report as BenchReport & { failures?: string[] }).failures = failures;
    // 自動計測(ヘッドレス回収)用に結果を公開する
    (window as unknown as { __report?: BenchReport }).__report = report;
    dlBtn.disabled = false;
    runBtn.disabled = false;
  };

  dlBtn.onclick = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 1)], {
      type: 'application/json',
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `webbench_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // デモ
  const demoBtn = $<HTMLButtonElement>('demo-toggle');
  demoBtn.disabled = false;
  let running = false;
  demoBtn.onclick = async () => {
    if (running) {
      stopDemo();
      demoBtn.textContent = 'デモ開始';
      running = false;
      return;
    }
    demoBtn.textContent = 'デモ停止';
    running = true;
    await startDemo(manifest, data);
  };
}

void init();
