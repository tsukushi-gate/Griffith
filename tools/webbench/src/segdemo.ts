// seg/detデモ共通パイプライン(W4骨格)。
//   canvas画像 → RGBA→NCHW → 画像thermometer → ConvRuntime → per-pixelマスク
// 実モデル(SegConvDLGN/CenterNet-LUT)接続は次ラウンド。現状は
// scripts/export_webbench.py demo のランダム重み小IRで配管を検証する。
import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import type { ImagePreprocDoc } from './convpreproc';
import { ConvRuntime } from './engines/conv';
import type { ConvEngine, FrameInput } from './engines/conv';
import type { BundleProgramDict } from './engines/bundleProgram';
import type { ConvModelDoc } from './types';

/** doc に埋め込まれた実Petテスト画像 + GTバイナリマスク。
    rgb_b64=HWC uint8×3、mask_b64=uint8×1(値∈{0,1}、1=pet)。 */
export interface SegDemoSample {
  rgb_b64: string;
  mask_b64: string;
  height: number;
  width: number;
}

export interface SegModel {
  doc: ConvModelDoc & { demo_samples?: SegDemoSample[] };
  runtime: ConvRuntime;
  thermo: ImagePreprocDoc;
  nOut: number; // クラス数
  h: number; // モデル入力(=出力)解像度
  w: number;
  outH: number;
  outW: number;
  samples: SegDemoSample[];
  sizeBytes: number; // ダウンロードしたモデルJSONの実バイト数(非圧縮)
  loadMs: number; // fetch開始→ランタイム構築完了までの実測ms
}

/** engine省略時は既定の'packed'(純JS)。packed-wasm/packed-webgpuは
    ConvRuntime.create()の非同期初期化が要るためこちらもawaitする。 */
export async function loadSegModel(
  url: string,
  engine: ConvEngine = 'packed',
): Promise<SegModel> {
  const t0 = performance.now();
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  const doc = JSON.parse(new TextDecoder().decode(buf)) as SegModel['doc'];
  const p = doc.preproc;
  if (p.kind !== 'image_thermometer') {
    throw new Error(`convモデルではありません(preproc.kind=${p.kind})`);
  }
  const spatial = doc.readout?.spatial;
  if (!spatial) throw new Error('readout.spatial がありません(seg readout必須)');
  // doc.preproc を渡すと bitplane-webgpu が RGBA→ビット符号化の
  // WGSLカーネルを組む(組めなければ runtime.gpuPreprocNote に理由が残る)。
  // doc.programs があれば conv2d 命令列をブラウザ側で再構築せず採用する。
  const runtime = await ConvRuntime.create(doc.ir, {
    engine, preproc: p,
    programs: (doc as { programs?: Record<string, BundleProgramDict> }).programs ?? null,
  });
  return {
    doc,
    runtime,
    // doc.preproc をそのまま保持(kind と低レベル特徴パラメータを落とさない)
    thermo: p,
    nOut: spatial.n_out,
    h: p.height,
    w: p.width,
    outH: spatial.height,
    outW: spatial.width,
    samples: doc.demo_samples ?? [],
    sizeBytes: buf.byteLength,
    loadMs: performance.now() - t0,
  };
}

export interface SegResult {
  labels: Int32Array; // [outH*outW] クラスid
  inferMs: number; // 推論のみ
  totalMs: number; // 前処理込み
}

/** 1フレーム実行。rgba はモデル解像度(model.h×model.w)のImageData.data。
    engine='packed-webgpu'では使用不可(runSegFrameAsyncを使うこと)。 */
export function runSegFrame(
  model: SegModel,
  rgba: Uint8ClampedArray,
): SegResult {
  const t0 = performance.now();
  const nchw = rgbaToNchw(rgba, model.h, model.w);
  const bits = encodeImageAuto(nchw, model.thermo);
  const t1 = performance.now();
  const { data } = model.runtime.scores(bits, 1);
  const labels = model.runtime.argmaxPerPixel(
    data, 1, model.nOut, model.outH * model.outW,
  );
  const t2 = performance.now();
  return { labels, inferMs: t2 - t1, totalMs: t2 - t0 };
}

/** 1フレームぶんの入力を用意する(detdemo.ts::prepareDetFrame と同型)。
    GPU前処理が使えるならRGBAをそのまま渡し、使えないときだけCPU符号化する。 */
export function prepareSegFrame(
  model: SegModel,
  rgba: Uint8ClampedArray,
): FrameInput {
  return model.runtime.prepareFrame(
    rgba,
    () => encodeImageAuto(rgbaToNchw(rgba, model.h, model.w), model.thermo),
  );
}

/** runSegFrame の非同期版。packed-webgpu を含む全エンジンで動く
    (packed/irflat/packed-wasmは内部で同期評価するだけ)。 */
export async function runSegFrameAsync(
  model: SegModel,
  rgba: Uint8ClampedArray,
): Promise<SegResult> {
  const t0 = performance.now();
  const input = prepareSegFrame(model, rgba);
  const t1 = performance.now();
  const { data } = await model.runtime.scoresFrameAsync(input, 1);
  const labels = model.runtime.argmaxPerPixel(
    data, 1, model.nOut, model.outH * model.outW,
  );
  const t2 = performance.now();
  return { labels, inferMs: t2 - t1, totalMs: t2 - t0 };
}

/** runSegFrameAsync のパイプライン版(detdemo.ts::submitDetFrame と同型)。
    前処理→submit までを同期に済ませ、argmax済みラベルのPromiseを返す。
    pipeline非対応エンジンでは submitScoresAsync が scoresAsync へ素通しするため
    動作は runSegFrameAsync と同一(逐次)。 */
export function submitSegFrame(
  model: SegModel,
  rgba: Uint8ClampedArray,
): Promise<SegResult> {
  const t0 = performance.now();
  const input = prepareSegFrame(model, rgba);
  const t1 = performance.now();
  const pending = model.runtime.submitScoresFrameAsync(input, 1);
  return pending.then(({ data }) => {
    const labels = model.runtime.argmaxPerPixel(
      data, 1, model.nOut, model.outH * model.outW,
    );
    const t2 = performance.now();
    return { labels, inferMs: t2 - t1, totalMs: t2 - t0 };
  });
}

/** [計測専用] 1フレームぶんの入力をループ外で用意する
    (detdemo.ts::encodeDetInput と同型)。 */
export function encodeSegInput(
  model: SegModel,
  rgba: Uint8ClampedArray,
): FrameInput {
  return prepareSegFrame(model, rgba);
}

/** [計測専用] 準備済み入力で1フレームをパイプライン発行する。
    計測区間は runSegFrameAsync の inferMs と完全に同じ(submit→argmax完了)。 */
export function submitSegPrepared(
  model: SegModel,
  input: FrameInput,
): Promise<SegResult> {
  const t1 = performance.now();
  return model.runtime.submitScoresFrameAsync(input, 1).then(({ data }) => {
    const labels = model.runtime.argmaxPerPixel(
      data, 1, model.nOut, model.outH * model.outW,
    );
    const t2 = performance.now();
    return { labels, inferMs: t2 - t1, totalMs: t2 - t1 };
  });
}

/** クラス別の半透明カラー(class 0=背景は透明)。 */
const MASK_COLORS: [number, number, number][] = [
  [0, 0, 0], // class0: 透明(背景)
  [230, 25, 75], [60, 180, 75], [67, 99, 216], [245, 130, 48],
  [145, 30, 180], [70, 240, 240], [240, 50, 230], [188, 246, 12],
];

/** マスクをoverlayキャンバスへ描く(nearest拡大、alpha=不透明度0-255)。 */
export function drawMask(
  overlay: HTMLCanvasElement,
  labels: Int32Array,
  outH: number,
  outW: number,
  alpha = 110,
): void {
  const ctx = overlay.getContext('2d')!;
  const img = ctx.createImageData(outW, outH);
  for (let i = 0; i < labels.length; i++) {
    const cls = labels[i];
    const col = MASK_COLORS[cls % MASK_COLORS.length];
    img.data[i * 4] = col[0];
    img.data[i * 4 + 1] = col[1];
    img.data[i * 4 + 2] = col[2];
    img.data[i * 4 + 3] = cls === 0 ? 0 : alpha;
  }
  // 小解像度→表示解像度へnearest拡大
  const tmp = document.createElement('canvas');
  tmp.width = outW;
  tmp.height = outH;
  tmp.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.drawImage(tmp, 0, 0, overlay.width, overlay.height);
}

/** 検出box(CenterNetデコード後)のオーバーレイ描画(次ラウンドの受け口)。
    boxは正規化座標 [0,1]。 */
export interface DetBox {
  x1: number; y1: number; x2: number; y2: number;
  cls: number; score: number;
}

export function drawBoxes(
  overlay: HTMLCanvasElement,
  boxes: DetBox[],
  labels: string[],
): void {
  const ctx = overlay.getContext('2d')!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 2;
  ctx.font = '12px ui-monospace, monospace';
  for (const b of boxes) {
    const col = MASK_COLORS[(b.cls + 1) % MASK_COLORS.length];
    ctx.strokeStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.fillStyle = ctx.strokeStyle;
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    ctx.strokeRect(x, y, (b.x2 - b.x1) * overlay.width, (b.y2 - b.y1) * overlay.height);
    ctx.fillText(
      `${labels[b.cls] ?? b.cls} ${(b.score * 100).toFixed(0)}%`, x + 2, y - 3,
    );
  }
}

// ---------------------------------------------------------------------------
// 埋め込みPetテストサンプル(det.htmlの実サンプルボタンに倣う)
// ---------------------------------------------------------------------------

/** base64 RGB(HWC、3*n バイト)→ Uint8。detdemo.ts::rgbFromB64 と同型。 */
export function rgbFromB64(b64: string, n: number): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(3 * n);
  for (let i = 0; i < 3 * n; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** base64 バイナリマスク(uint8×1、値∈{0,1})→ Uint8。 */
export function maskFromB64(b64: string, n: number): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** RGB(HWC uint8)を canvas に RGBA で描く。detdemo.ts::drawRgbToCanvas と同型。 */
export function drawRgbToCanvas(
  canvas: HTMLCanvasElement,
  rgb: Uint8Array,
  h: number,
  w: number,
): void {
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d')!;
  const img = tctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    img.data[i * 4] = rgb[i * 3];
    img.data[i * 4 + 1] = rgb[i * 3 + 1];
    img.data[i * 4 + 2] = rgb[i * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(img, 0, 0);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

/** GTマスクの輪郭(4近傍のいずれかと値が異なる画素)を overlay に重ね描く。
    drawMask() 呼び出し後の同一overlayに追加で描く想定(clearRectしない)。 */
export function drawGtMask(
  overlay: HTMLCanvasElement,
  gtMask: Uint8Array,
  h: number,
  w: number,
  color = '#ffe94a',
): void {
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d')!;
  const img = tctx.createImageData(w, h);
  const rgb = hexToRgb(color);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = gtMask[i];
      const isEdge =
        (x > 0 && gtMask[i - 1] !== v) ||
        (x < w - 1 && gtMask[i + 1] !== v) ||
        (y > 0 && gtMask[i - w] !== v) ||
        (y < h - 1 && gtMask[i + w] !== v);
      if (isEdge) {
        img.data[i * 4] = rgb[0];
        img.data[i * 4 + 1] = rgb[1];
        img.data[i * 4 + 2] = rgb[2];
        img.data[i * 4 + 3] = 255;
      }
    }
  }
  tctx.putImageData(img, 0, 0);
  const ctx = overlay.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, overlay.width, overlay.height);
}

function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace('#', '');
  return [
    parseInt(v.slice(0, 2), 16),
    parseInt(v.slice(2, 4), 16),
    parseInt(v.slice(4, 6), 16),
  ];
}

/** 予測ラベルとGTバイナリマスク(class1=pet)から IoU / pixel accuracy を計算。
    予測とGTの解像度(H*W長)が一致している前提。 */
export function computeSegMetrics(
  pred: Int32Array,
  gtMask: Uint8Array,
): { iou: number; pixelAcc: number } {
  const n = pred.length;
  let inter = 0, union = 0, correct = 0;
  for (let i = 0; i < n; i++) {
    const p = pred[i] !== 0 ? 1 : 0;
    const g = gtMask[i] !== 0 ? 1 : 0;
    if (p === 1 || g === 1) union++;
    if (p === 1 && g === 1) inter++;
    if (p === g) correct++;
  }
  return {
    iou: union > 0 ? inter / union : 1,
    pixelAcc: n > 0 ? correct / n : 0,
  };
}
