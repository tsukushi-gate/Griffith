// 検出デモ パイプライン(カラー対応)。
//   canvas画像(カラーMulti-MNIST: CIFAR-10背景+色付き数字) → RGBA→NCHW →
//   画像thermometer → ConvRuntime(packed) → CenterNet decode → box描画。
// 前処理契約は seg と共通(convpreproc.ts)。検出モデルは rgb_channels=3
// (RGB)で、thermometer 出力ch = rgb*T+thr(convpreproc.ts のrgb-major。
// 学習時の thr-major との差は export_webbench.py 側の permute ノードで吸収)。
import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import type { ImagePreprocDoc } from './convpreproc';
import { decodeDetections } from './detdecode';
import type { DetBoxPx } from './detdecode';
import { ConvRuntime } from './engines/conv';
import type { ConvEngine, FrameInput } from './engines/conv';
import type { BundleProgramDict } from './engines/bundleProgram';
import type { ConvModelDoc } from './types';
import type { DetReadout } from './types';

/** doc に埋め込まれた実 Multi-MNIST サンプル + GT。
    color=Trueモデルは rgb_b64(HWC uint8×3)、グレースケール旧モデルは
    gray_b64(uint8×1)のどちらかを持つ(両対応、後方互換)。 */
export interface DetDemoSample {
  gray_b64?: string;
  rgb_b64?: string;
  height: number;
  width: number;
  gt_boxes: number[][]; // [[x1,y1,x2,y2], ...]
  gt_labels: number[];
}

export interface DetModel {
  doc: Omit<ConvModelDoc, 'readout'> & {
    readout: DetReadout;
    demo_samples?: DetDemoSample[];
  };
  runtime: ConvRuntime;
  /** doc.preproc そのまま(kind と低レベル特徴パラメータを含む)。 */
  thermo: ImagePreprocDoc;
  readout: DetReadout;
  labels: string[];
  h: number; // モデル入力解像度(=canvas)
  w: number;
  samples: DetDemoSample[];
  sizeBytes: number; // ダウンロードしたモデルJSONの実バイト数(非圧縮)
  loadMs: number; // fetch開始→ランタイム構築完了までの実測ms
}

/** engine省略時は既定の'packed'(純JS)。packed-wasm/packed-webgpuは
    ConvRuntime.create()の非同期初期化が要るためこちらもawaitする。 */
export async function loadDetModel(
  url: string,
  engine: ConvEngine = 'packed',
): Promise<DetModel> {
  const t0 = performance.now();
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  const doc = JSON.parse(new TextDecoder().decode(buf)) as DetModel['doc'];
  const p = doc.preproc;
  // 低レベル特徴検出器つき(image_lowlevel)も受け付ける。
  // エンコーダの選択は convpreproc.encodeImageAuto に一本化。
  if (p.kind !== 'image_thermometer' && p.kind !== 'image_lowlevel') {
    throw new Error(`検出モデルではありません(preproc.kind=${p.kind})`);
  }
  if (doc.readout?.kind !== 'detection') {
    throw new Error('readout.kind=detection がありません(検出 readout 必須)');
  }
  // doc.preproc を渡すと bitplane-webgpu が RGBA→ビット符号化の
  // WGSLカーネルを組む(組めなければ runtime.gpuPreprocNote に理由が残り、
  // 以降 prepareFrame() が自動でCPU前処理を選ぶ)。
  // doc.programs があれば conv2d 命令列をブラウザ側で再構築せず採用する。
  const runtime = await ConvRuntime.create(doc.ir, {
    engine, preproc: p,
    programs: (doc as { programs?: Record<string, BundleProgramDict> }).programs ?? null,
  });
  return {
    doc,
    runtime,
    // kind と低レベル特徴パラメータを落とさずそのまま渡す
    // (以前は mean/std/thresholds/height/width だけ抜き出しており、
    //  image_lowlevel のカーネル方向・閾値が欠落していた)。
    thermo: p,
    readout: doc.readout,
    labels: doc.labels,
    h: p.height,
    w: p.width,
    samples: doc.demo_samples ?? [],
    sizeBytes: buf.byteLength,
    loadMs: performance.now() - t0,
  };
}

export interface DetResult {
  boxes: DetBoxPx[]; // ピクセル座標(モデル canvas 座標系)
  inferMs: number; // 推論のみ(前処理除く)
  totalMs: number; // 前処理込み
}

/** 1フレーム実行。rgba はモデル解像度(h×w)の ImageData.data。
    engine='packed-webgpu'では使用不可(runDetFrameAsyncを使うこと)。 */
export function runDetFrame(
  model: DetModel,
  rgba: Uint8ClampedArray,
): DetResult {
  const t0 = performance.now();
  const nchw = rgbaToNchw(rgba, model.h, model.w);
  const bits = encodeImageAuto(nchw, model.thermo);
  const t1 = performance.now();
  const { data } = model.runtime.scores(bits, 1);
  const boxes = decodeDetections(data, model.readout);
  const t2 = performance.now();
  return { boxes, inferMs: t2 - t1, totalMs: t2 - t0 };
}

/** 1フレームぶんの入力を用意する。GPU前処理が使えるならRGBAをそのまま
    渡し(前処理はGPU側の先頭カーネル)、使えないときだけCPU符号化する。
    前処理経路の分岐はこの1関数に閉じる。 */
export function prepareDetFrame(
  model: DetModel,
  rgba: Uint8ClampedArray,
): FrameInput {
  return model.runtime.prepareFrame(
    rgba,
    () => encodeImageAuto(rgbaToNchw(rgba, model.h, model.w), model.thermo),
  );
}

/** runDetFrame の非同期版。packed-webgpu を含む全エンジンで動く。
    bitplane-webgpu では前処理もGPUで行うため inferMs にGPU前処理が
    含まれる(totalMs との差は decode ぶんだけになる)。 */
export async function runDetFrameAsync(
  model: DetModel,
  rgba: Uint8ClampedArray,
): Promise<DetResult> {
  const t0 = performance.now();
  const input = prepareDetFrame(model, rgba);
  const t1 = performance.now();
  const { data } = await model.runtime.scoresFrameAsync(input, 1);
  const boxes = decodeDetections(data, model.readout);
  const t2 = performance.now();
  return { boxes, inferMs: t2 - t1, totalMs: t2 - t0 };
}

/** runDetFrameAsync のパイプライン版。前処理→submit までを同期に
    済ませ、結果(decode済みbox)のPromiseを返す。呼び出し側は前フレームの
    結果をawaitする前にこれを呼ぶことで、submit→mapAsyncの往復をGPU実行へ
    重ねられる。pipeline非対応エンジンでは submitScoresAsync が scoresAsync へ
    素通しするため、動作は runDetFrameAsync と同一(逐次)になる。
    inferMs は「submit開始→自フレームdecode完了」の実測で、パイプライン中は
    前フレームの待ちを含むため latency 指標としては使わないこと
    (スループット指標は呼び出し側が全体時間/Nで取る)。 */
export function submitDetFrame(
  model: DetModel,
  rgba: Uint8ClampedArray,
): Promise<DetResult> {
  const t0 = performance.now();
  const input = prepareDetFrame(model, rgba);
  const t1 = performance.now();
  const pending = model.runtime.submitScoresFrameAsync(input, 1);
  return pending.then(({ data }) => {
    const boxes = decodeDetections(data, model.readout);
    const t2 = performance.now();
    return { boxes, inferMs: t2 - t1, totalMs: t2 - t0 };
  });
}

/** [計測専用] 1フレームぶんの入力をループ外で用意する。
    以降、GPU前処理が使える構成ではここは実質ノーコピー(RGBAを包むだけ)で、
    前処理そのものは submit したフレームのGPU時間に入る。 */
export function encodeDetInput(
  model: DetModel,
  rgba: Uint8ClampedArray,
): FrameInput {
  return prepareDetFrame(model, rgba);
}

/** [計測専用] 準備済み入力で1フレームをパイプライン発行する。
    計測区間は runDetFrameAsync の inferMs と完全に同じ(submit→decode完了)。 */
export function submitDetPrepared(
  model: DetModel,
  input: FrameInput,
): Promise<DetResult> {
  const t1 = performance.now();
  return model.runtime.submitScoresFrameAsync(input, 1).then(({ data }) => {
    const boxes = decodeDetections(data, model.readout);
    const t2 = performance.now();
    return { boxes, inferMs: t2 - t1, totalMs: t2 - t1 };
  });
}

/** グレースケール(uint8 h*w)を canvas に RGBA で描く。 */
export function drawGrayToCanvas(
  canvas: HTMLCanvasElement,
  gray: Uint8Array,
  h: number,
  w: number,
): void {
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d')!;
  const img = tctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const v = gray[i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(img, 0, 0);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

const COLORS: string[] = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#46f0f0', '#f032e6', '#bcf60c', '#008080', '#9a6324',
];

/** 「box 0」の原因を判別できる注記を作る。
 *
 * DLGN は SpatialGroupSum(popcount/τ)の量子化で hm スコアが低い値域に圧縮される
 * ため、閾値が少し高いだけで表示0件になる。「閾値が原因なのか、本当に何も検出
 * されていないのか」がユーザーに分からないと不具合と区別できないので、
 * 閾値通過前の最良スコアを必ず併記する。
 */
export function boxCountNote(
  boxes: readonly { score: number }[],
  shown: number,
  scoreThreshold: number,
): string {
  const head = `検出枠 ${shown}/${boxes.length}`;
  if (boxes.length === 0) return `${head}(候補なし)`;
  let best = -Infinity;
  for (const b of boxes) if (b.score > best) best = b.score;
  const bestTxt = best.toFixed(3);
  const thrTxt = scoreThreshold.toFixed(3);
  return shown === 0
    ? `${head}(最大スコア ${bestTxt} < しきい値 ${thrTxt})`
    : `${head}(最大スコア ${bestTxt} / しきい値 ${thrTxt})`;
}

/** 検出box(ピクセル座標)を overlay に描く。modelSize→表示canvasへ拡大。
    scoreThreshold 未満は描画しない。 */
export function drawDetBoxes(
  overlay: HTMLCanvasElement,
  boxes: DetBoxPx[],
  labels: string[],
  modelSize: number,
  scoreThreshold: number,
): number {
  const ctx = overlay.getContext('2d')!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const s = overlay.width / modelSize;
  ctx.lineWidth = 2;
  ctx.font = '13px ui-monospace, monospace';
  ctx.textBaseline = 'bottom';
  let shown = 0;
  for (const b of boxes) {
    if (b.score < scoreThreshold) continue;
    shown++;
    const col = COLORS[b.cls % COLORS.length];
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    const x = b.x1 * s;
    const y = b.y1 * s;
    ctx.strokeRect(x, y, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s);
    const tag = `${labels[b.cls] ?? b.cls}:${(b.score * 100).toFixed(0)}%`;
    ctx.fillText(tag, x, Math.max(11, y - 1));
  }
  return shown;
}

/** GTボックス(破線・グレー)を描く(任意表示)。 */
export function drawGtBoxes(
  overlay: HTMLCanvasElement,
  sample: DetDemoSample,
  modelSize: number,
): void {
  const ctx = overlay.getContext('2d')!;
  const s = overlay.width / modelSize;
  ctx.save();
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(160,160,160,0.9)';
  for (const box of sample.gt_boxes) {
    const [x1, y1, x2, y2] = box;
    ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
  }
  ctx.restore();
}

/** base64 グレースケール → Uint8。 */
export function grayFromB64(b64: string, n: number): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** base64 RGB(HWC、3*n バイト)→ Uint8。 */
export function rgbFromB64(b64: string, n: number): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(3 * n);
  for (let i = 0; i < 3 * n; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** RGB(HWC uint8)を canvas に RGBA で描く(カラーMulti-MNIST用)。 */
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
