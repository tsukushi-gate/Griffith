// 配布バンドルの単一ページデモ(dist/index.html + demo.js)。
// `./model.bundle.json` を既定でfetchし(`?model=<url>`で上書き)、lib.ts::
// loadBundle だけで完結する最小構成: meta表 / programs簡約統計 / notes /
// engine選択(auto/js/wasm、median×20の推論時間) / demo_samples描画
// (seg=マスクオーバレイ+GT輪郭、detection=箱+GT箱、それ以外=スコア先頭値)。
//
// 描画ヘルパーは意図的にこのファイル内に閉じ込める(segdemo.ts/detdemo.tsは
// ページ固有のDOM前提を伴うため、依存を増やさず自己完結にする方針。仕様)。
import { loadBundle } from './lib';
import type { WebBundleRuntime } from './lib';
import { metaFields } from '../modelMeta';

type EngineChoice = 'auto' | 'js' | 'wasm';

interface DemoSampleDoc {
  height: number;
  width: number;
  rgb_b64?: string;
  mask_b64?: string;
  gt_boxes?: number[][];
  gt_labels?: number[];
}

/** decodeDetections() の戻り値の形(detdecode.ts::DetBoxPx と同型)。 */
interface DetBoxLike {
  x1: number; y1: number; x2: number; y2: number; cls: number; score: number;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// base64 / canvas ヘルパー(自己完結。conv.test.ts::b64ToBits / segdemo.ts::
// rgbFromB64 と同じ atob 直読み流儀)
// ---------------------------------------------------------------------------

function bytesFromB64(b64: string, n: number): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = raw.charCodeAt(i);
  return out;
}

function rgbaFromRgbHwc(rgb: Uint8Array, h: number, w: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(h * w * 4);
  for (let i = 0; i < h * w; i++) {
    out[i * 4] = rgb[i * 3];
    out[i * 4 + 1] = rgb[i * 3 + 1];
    out[i * 4 + 2] = rgb[i * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

function drawRgbHwc(canvas: HTMLCanvasElement, rgb: Uint8Array, h: number, w: number): void {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    img.data[i * 4] = rgb[i * 3];
    img.data[i * 4 + 1] = rgb[i * 3 + 1];
    img.data[i * 4 + 2] = rgb[i * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** クラス別の半透明カラー(class0=背景は透明)。segdemo.ts::MASK_COLORS と同型。 */
const MASK_COLORS: [number, number, number][] = [
  [0, 0, 0], [230, 25, 75], [60, 180, 75], [67, 99, 216], [245, 130, 48],
  [145, 30, 180], [70, 240, 240], [240, 50, 230], [188, 246, 12],
];

/** predラベル([outH*outW])をoverlayへnearest拡大で描く。 */
function overlayLabels(
  overlay: HTMLCanvasElement, labels: Int32Array, outH: number, outW: number, alpha = 140,
): void {
  const tmp = document.createElement('canvas');
  tmp.width = outW;
  tmp.height = outH;
  const tctx = tmp.getContext('2d')!;
  const img = tctx.createImageData(outW, outH);
  for (let i = 0; i < labels.length; i++) {
    const col = MASK_COLORS[labels[i] % MASK_COLORS.length];
    img.data[i * 4] = col[0];
    img.data[i * 4 + 1] = col[1];
    img.data[i * 4 + 2] = col[2];
    img.data[i * 4 + 3] = labels[i] === 0 ? 0 : alpha;
  }
  tctx.putImageData(img, 0, 0);
  const ctx = overlay.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.drawImage(tmp, 0, 0, overlay.width, overlay.height);
}

/** GTマスクの輪郭(4近傍と値が違う画素)を重ね描く。overlay解像度=マスク解像度前提。 */
function drawGtContour(overlay: HTMLCanvasElement, mask: Uint8Array, h: number, w: number): void {
  const ctx = overlay.getContext('2d')!;
  ctx.fillStyle = '#ffe94a';
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = mask[i];
      const edge = (x > 0 && mask[i - 1] !== v) || (x < w - 1 && mask[i + 1] !== v)
        || (y > 0 && mask[i - w] !== v) || (y < h - 1 && mask[i + w] !== v);
      if (edge) ctx.fillRect(x, y, 1, 1);
    }
  }
}

function drawBoxes(
  overlay: HTMLCanvasElement,
  boxes: DetBoxLike[],
  labels: string[],
): void {
  const ctx = overlay.getContext('2d')!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 1.5;
  ctx.font = '10px ui-monospace, monospace';
  for (const b of boxes) {
    ctx.strokeStyle = '#e63946';
    ctx.fillStyle = '#e63946';
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillText(
      `${labels[b.cls] ?? b.cls}:${(b.score * 100).toFixed(0)}%`, b.x1 + 1, Math.max(9, b.y1 - 2),
    );
  }
}

function drawGtBoxes(overlay: HTMLCanvasElement, boxes: number[][]): void {
  const ctx = overlay.getContext('2d')!;
  ctx.save();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = 'rgba(160,160,160,0.9)';
  ctx.lineWidth = 1;
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.restore();
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// ---------------------------------------------------------------------------
// レンダリング
// ---------------------------------------------------------------------------

function modelUrl(): string {
  const q = new URL(window.location.href).searchParams.get('model');
  return q ?? './model.bundle.json';
}

async function fetchDoc(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function renderMeta(bundle: WebBundleRuntime): void {
  const fields = metaFields(bundle.meta);
  const rows: [string, string][] = [
    ['モデル名', fields.modelName],
    ['判別kind', bundle.kind],
    ['bundle_version', String((bundle.doc as { bundle_version?: string }).bundle_version ?? '—')],
    ['精度', fields.accuracy],
    ['ゲート数(重み共有)', fields.gatesShared],
    ['ゲート数(展開後)', fields.gatesUnrolled],
    ['論理深さ', fields.logicDepth],
    ['入力解像度', fields.inputRes],
    ['入力チャネル数', fields.inputChannels],
    ['入力ビット数', fields.inputBits],
    ['配信サイズ(json)', fields.jsonBytes],
    ['配信サイズ(gzip)', fields.gzipBytes],
    ['クラス数', fields.nClasses],
  ];

  const programs = (bundle.doc as {
    programs?: Record<string, { stats?: { gates_before?: number; gates_after?: number } }>;
  }).programs;
  if (programs) {
    let before = 0;
    let after = 0;
    for (const p of Object.values(programs)) {
      before += p.stats?.gates_before ?? 0;
      after += p.stats?.gates_after ?? 0;
    }
    const pct = before > 0 ? `(${(100 * (before - after) / before).toFixed(1)}% 減)` : '';
    rows.push(['conv2dゲート(命令列化後)', before > 0 ? `${before.toLocaleString()} → ${after.toLocaleString()} ${pct}` : '—']);
  }

  const tbody = $('meta-table').querySelector('tbody')!;
  tbody.innerHTML = '';
  for (const [k, v] of rows) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = k;
    const td = document.createElement('td');
    td.textContent = v;
    tr.append(th, td);
    tbody.appendChild(tr);
  }
}

function renderNotes(notes: string[]): void {
  const ul = $('notes');
  ul.innerHTML = '';
  const list = notes.length > 0 ? notes : ['(注記なし)'];
  for (const n of list) {
    const li = document.createElement('li');
    li.textContent = n;
    ul.appendChild(li);
  }
}

/** engineベンチ(warmup1回+median×20)。ランダムbit入力。 */
function benchmarkEngine(bundle: WebBundleRuntime): number {
  const inWidth = bundle.runtime.inWidth;
  const bits = new Uint8Array(inWidth);
  for (let i = 0; i < inWidth; i++) bits[i] = Math.random() < 0.5 ? 1 : 0;
  bundle.scores(bits, 1); // warmup
  const times: number[] = [];
  for (let i = 0; i < 20; i++) {
    const t0 = performance.now();
    bundle.scores(bits, 1);
    times.push(performance.now() - t0);
  }
  return median(times);
}

function renderRandomSamples(bundle: WebBundleRuntime, container: HTMLElement): void {
  const n = 5;
  const inWidth = bundle.runtime.inWidth;
  const heading = document.createElement('p');
  heading.style.width = '100%';
  heading.textContent = `demo_samplesが無い(またはdense kind)ため、乱数入力${n}件でスコアと推論時間を計測する。`;
  container.appendChild(heading);
  for (let s = 0; s < n; s++) {
    const bits = new Uint8Array(inWidth);
    for (let i = 0; i < inWidth; i++) bits[i] = Math.random() < 0.5 ? 1 : 0;
    const t0 = performance.now();
    const { data } = bundle.scores(bits, 1);
    const ms = performance.now() - t0;
    const preview = Array.from(data).slice(0, 12).join(', ');
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.textContent = `#${s}: ${ms.toFixed(3)} ms / len=${data.length} / [${preview}${data.length > 12 ? ', …' : ''}]`;
    container.appendChild(card);
  }
}

function renderSample(
  bundle: WebBundleRuntime, s: DemoSampleDoc, container: HTMLElement,
): void {
  const card = document.createElement('div');
  card.className = 'sample-card';
  const holder = document.createElement('div');
  holder.className = 'canvas-holder';
  const base = document.createElement('canvas');
  const overlay = document.createElement('canvas');
  overlay.width = s.width;
  overlay.height = s.height;
  holder.append(base, overlay);
  const info = document.createElement('div');
  info.className = 'sample-info';
  card.append(holder, info);
  container.appendChild(card);

  if (!s.rgb_b64) {
    info.textContent = '(rgb_b64が無い)';
    return;
  }
  const rgb = bytesFromB64(s.rgb_b64, s.height * s.width * 3);
  drawRgbHwc(base, rgb, s.height, s.width);
  const rgba = rgbaFromRgbHwc(rgb, s.height, s.width);

  const t0 = performance.now();
  if (bundle.kind === 'seg' && bundle.segLabels) {
    const spatial = (bundle.doc.readout as {
      spatial?: { n_out: number; height: number; width: number };
    } | undefined)?.spatial;
    const predLabels = bundle.segLabels(rgba);
    const ms = performance.now() - t0;
    if (spatial) overlayLabels(overlay, predLabels, spatial.height, spatial.width);
    if (s.mask_b64) drawGtContour(overlay, bytesFromB64(s.mask_b64, s.height * s.width), s.height, s.width);
    info.textContent = `推論 ${ms.toFixed(2)} ms`;
  } else if (bundle.scoresRgba) {
    const { data } = bundle.scoresRgba(rgba);
    const ms = performance.now() - t0;
    const preview = Array.from(data).slice(0, 8).join(', ');
    info.textContent = `推論 ${ms.toFixed(2)} ms / スコア[${preview}${data.length > 8 ? ', …' : ''}]`;
  }
}

// ---------------------------------------------------------------------------
// detection: 表示閾値スライダー(readout.thresholdとは別物)
// ---------------------------------------------------------------------------

/** decode済みbox一覧を1サンプル分キャッシュしておく入れ物。スライダー操作
    では推論(bundle.detect)を再実行せず、この配列をフィルタして再描画
    するだけにする。 */
interface DetSampleEntry {
  overlay: HTMLCanvasElement;
  boxes: DetBoxLike[];
  gtBoxes?: number[][];
  info: HTMLElement;
  inferMs: number;
}

/** detection kind専用のサンプル描画。detreal_main.ts の 'thr' スライダーと
    同じ設計思想: readout.threshold(decode側の緩い閾値、モデルによっては
    0.1程度)とは別に「表示するboxを絞る」ための表示専用フィルタを1つ
    用意する。既定値0.5はモデル較正値ではない汎用初期値(nostale違反ではない)。
    全サンプルのdecode結果は最初に1回だけ計算してentriesへキャッシュし、
    スライダーのinputイベントでは再描画のみ行う(推論を再実行しない)。 */
function renderDetectionSamples(
  bundle: WebBundleRuntime, samples: DemoSampleDoc[], labels: string[], container: HTMLElement,
): void {
  const controls = document.createElement('div');
  controls.className = 'thr-control';
  const label = document.createElement('label');
  label.textContent = '表示閾値: ';
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '1';
  slider.step = '0.05';
  slider.value = '0.5';
  const valueSpan = document.createElement('span');
  valueSpan.textContent = Number(slider.value).toFixed(2);
  label.append(slider, valueSpan);
  controls.appendChild(label);
  const caption = document.createElement('p');
  caption.className = 'sample-info';
  caption.textContent = '表示閾値(表示用フィルタ。decode閾値=readout.thresholdとは別)';
  controls.appendChild(caption);
  container.appendChild(controls);

  const entries: DetSampleEntry[] = [];
  for (const s of samples) {
    const card = document.createElement('div');
    card.className = 'sample-card';
    const holder = document.createElement('div');
    holder.className = 'canvas-holder';
    const base = document.createElement('canvas');
    const overlay = document.createElement('canvas');
    overlay.width = s.width;
    overlay.height = s.height;
    holder.append(base, overlay);
    const info = document.createElement('div');
    info.className = 'sample-info';
    card.append(holder, info);
    container.appendChild(card);

    if (!s.rgb_b64) {
      info.textContent = '(rgb_b64が無い)';
      continue;
    }
    const rgb = bytesFromB64(s.rgb_b64, s.height * s.width * 3);
    drawRgbHwc(base, rgb, s.height, s.width);
    const rgba = rgbaFromRgbHwc(rgb, s.height, s.width);

    const t0 = performance.now();
    const boxes = bundle.detect!(rgba);
    const inferMs = performance.now() - t0;
    entries.push({
      overlay, boxes, gtBoxes: s.gt_boxes, info, inferMs,
    });
  }

  const redraw = (threshold: number): void => {
    for (const e of entries) {
      const shown = e.boxes.filter((b) => b.score >= threshold);
      drawBoxes(e.overlay, shown, labels);
      if (e.gtBoxes) drawGtBoxes(e.overlay, e.gtBoxes);
      e.info.textContent =
        `推論 ${e.inferMs.toFixed(2)} ms / 検出${e.boxes.length}件 / 表示${shown.length}件`;
    }
  };

  slider.addEventListener('input', () => {
    const v = Number(slider.value);
    valueSpan.textContent = v.toFixed(2);
    redraw(v);
  });
  redraw(Number(slider.value));
}

function renderDemoSamples(bundle: WebBundleRuntime): void {
  const container = $('samples');
  container.innerHTML = '';
  const doc = bundle.doc as unknown as { demo_samples?: DemoSampleDoc[]; labels?: string[] };
  const samples = doc.demo_samples ?? [];
  const labels = doc.labels ?? [];
  if (bundle.kind === 'dense' || samples.length === 0) {
    renderRandomSamples(bundle, container);
    return;
  }
  if (bundle.kind === 'detection' && bundle.detect) {
    renderDetectionSamples(bundle, samples, labels, container);
    return;
  }
  for (const s of samples) renderSample(bundle, s, container);
}

// ---------------------------------------------------------------------------
// エントリポイント
// ---------------------------------------------------------------------------

async function loadAndRender(engine: EngineChoice): Promise<void> {
  const status = $('status');
  const url = modelUrl();
  status.textContent = `${url} を読み込み中(engine=${engine})…`;
  try {
    const doc = await fetchDoc(url);
    const t0 = performance.now();
    const bundle = await loadBundle(doc, { engine });
    const loadMs = performance.now() - t0;
    const benchMs = benchmarkEngine(bundle);

    renderMeta(bundle);
    renderNotes(bundle.notes);
    $('engine-info').textContent =
      `engine=${bundle.engine} / ロード ${loadMs.toFixed(1)} ms / `
      + `推論(median×20) ${benchMs.toFixed(3)} ms`;
    renderDemoSamples(bundle);
    status.textContent = `完了(${url})`;
  } catch (e) {
    status.textContent = `エラー: ${String(e)}`;
    console.error(e);
  }
}

async function main(): Promise<void> {
  const sel = $<HTMLSelectElement>('engine-select');
  sel.addEventListener('change', () => {
    void loadAndRender(sel.value as EngineChoice);
  });
  await loadAndRender(sel.value as EngineChoice);
}

void main();
