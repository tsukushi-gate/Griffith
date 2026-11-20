// セグメンテーションデモ(Oxford-IIIT Pet の写真をペットと背景に塗り分ける)。
// 同じ入力を DLGN / ONNX Runtime Web / TensorFlow.js に流し、マスクと4指標を並べる。
import {
  computeSegMetrics, drawGtMask, drawMask, drawRgbToCanvas, loadSegModel,
  maskFromB64, rgbFromB64,
} from './segdemo';
import type { SegDemoSample, SegModel } from './segdemo';
import { SegDlgnRuntime, SegOnnxRuntime, SegTfjsRuntime } from './compareSeg';
import type { DlgnBackend, OnnxBackend, SegRuntime } from './compareSeg';
import { asset, modelAsset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { applyReferences } from './references';
import { parseAutobenchParams, runAutobench } from './autobench';
import { runSequential } from './sequentialRun';
import { RuntimeGrid } from './ui/runtimeCard';
import type { SlotSpec } from './ui/runtimeCard';
import { renderShell } from './ui/shell';
import { DEFAULT_DLGN_BACKEND, PAGES } from './ui/pages';
import { drawCoverCrop, wireImageUpload } from './ui/upload';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

// `?model=_ll` / `?model=<file>.json` でDLGNモデルを差し替えられる。
const DLGN_URL = modelAsset('/models/seg_convdlgn_k128.json');
const ONNX_URL = asset('/models/seg_unet.onnx');
const TFJS_DIR = asset('/models/seg_unet_tfjs');
const DISP = 320;

async function init(): Promise<void> {
  const bench = parseAutobenchParams('seg');
  if (bench) {
    const rt = new SegDlgnRuntime(DLGN_URL, bench.backend);
    await runAutobench(bench, rt, () => ({ w: rt.inW, h: rt.inH }));
    return;
  }

  renderShell(PAGES.seg);

  // DLGNモデルのJSONが埋め込みサンプル画像と正解マスクの真実源(比較列も共有)。
  const refModel: SegModel = await loadSegModel(DLGN_URL);
  applyModelMeta(refModel.doc);
  applyReferences();

  let curSample = -1;
  let curImage: HTMLImageElement | null = null;
  let showGt = true;

  const dlgnBackend = RuntimeGrid.initialBackend('dlgnBackend', DEFAULT_DLGN_BACKEND) as DlgnBackend;
  const onnxBackend = RuntimeGrid.initialBackend('onnxBackend', 'wasm');

  const slots: SlotSpec<SegRuntime>[] = [
    {
      kind: 'dlgn',
      label: 'DLGN(論理ゲート回路)',
      backend: dlgnBackend,
      create: (b) => new SegDlgnRuntime(DLGN_URL, b as DlgnBackend),
    },
    {
      kind: 'onnx',
      label: '比較用のCNN(ONNX Runtime Web)',
      backend: onnxBackend,
      create: (b) => new SegOnnxRuntime(ONNX_URL, b as OnnxBackend),
    },
    {
      kind: 'other',
      label: '比較用のCNN(TensorFlow.js)',
      create: () => new SegTfjsRuntime(TFJS_DIR),
    },
  ];

  const grid = new RuntimeGrid<SegRuntime>({
    gridEl: $('grid'),
    display: DISP,
    slots,
    onRender: () => renderAll(),
  });

  function drawSample(canvas: HTMLCanvasElement, s: SegDemoSample): void {
    drawRgbToCanvas(canvas, rgbFromB64(s.rgb_b64, s.height * s.width), s.height, s.width);
  }

  async function renderAll(): Promise<void> {
    const sample = curSample >= 0 ? refModel.samples[curSample] : null;
    if (!sample && !curImage) return;
    const rt0 = grid.cols[0].rt;
    const small = document.createElement('canvas');
    small.width = rt0.inW; small.height = rt0.inH;
    const sctx = small.getContext('2d')!;
    if (sample) drawSample(small, sample);
    else drawCoverCrop(sctx, curImage!, rt0.inW, rt0.inH);
    const rgba = sctx.getImageData(0, 0, rt0.inW, rt0.inH).data;
    const alpha = Number($<HTMLInputElement>('alpha').value);
    const gtMask = sample
      ? maskFromB64(sample.mask_b64, sample.height * sample.width) : null;

    // 列ごとに厳密に逐次実行する(CPU/GPU資源の競合を避け、真の推論速度に
    // 近づける。複数ランタイムを Promise.all で同時に走らせると、表示される
    // 「推論時間」が資源の奪い合いで実際より遅く出てしまうため)。
    await runSequential(grid.cols, async (c) => {
      if (c.error || !c.info) return;
      const bctx = c.base.getContext('2d')!;
      bctx.clearRect(0, 0, DISP, DISP);
      if (sample) drawSample(c.base, sample);
      else drawCoverCrop(bctx, curImage!, DISP, DISP);
      try {
        const { output, ms } = await RuntimeGrid.timedRun(c.rt, rgba);
        drawMask(c.overlay, output, c.rt.outH, c.rt.outW, alpha);
        let note = '';
        if (showGt && gtMask && output.length === gtMask.length) {
          const { iou, pixelAcc } = computeSegMetrics(output, gtMask);
          drawGtMask(c.overlay, gtMask, c.rt.outH, c.rt.outW);
          note = `正解との一致 IoU ${iou.toFixed(3)} / 画素一致率 ${pixelAcc.toFixed(3)}`;
        }
        grid.setInference(c, ms, note);
      } catch (e) {
        grid.setRunError(c, String(e));
      }
      // 次の列へ移る前に1フレーム分yieldし、描画を落ち着かせてから次の計測に入る。
      await new Promise((r) => requestAnimationFrame(r));
    });
  }

  function buildSampleButtons(): void {
    const box = $('samples');
    box.textContent = '';
    box.className = 'mt-4 flex flex-wrap gap-2';
    refModel.samples.forEach((_s, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'rounded-md border border-line bg-card px-2.5 py-1 text-xs';
      btn.textContent = `#${i + 1}`;
      btn.onclick = () => { curSample = i; curImage = null; void renderAll(); };
      box.append(btn);
    });
  }

  grid.build();
  buildSampleButtons();
  grid.bindBackendSelect('dlgn', 'dlgnBackend', DEFAULT_DLGN_BACKEND);
  grid.bindBackendSelect('onnx', 'onnxBackend', 'wasm');
  await grid.loadAll();

  const fileInput = $<HTMLInputElement>('file');
  fileInput.disabled = false;
  const fileError = document.createElement('p');
  fileError.className = 'mt-1 text-xs text-danger';
  fileInput.insertAdjacentElement('afterend', fileError);
  $<HTMLInputElement>('alpha').oninput = () => void renderAll();
  $<HTMLInputElement>('gt').onchange = (ev) => {
    showGt = (ev.target as HTMLInputElement).checked;
    void renderAll();
  };
  wireImageUpload({
    fileInput,
    onImage: (img) => {
      fileError.textContent = '';
      curImage = img; curSample = -1;
      void renderAll();
    },
    onError: (msg) => { fileError.textContent = msg; },
  });
  if (refModel.samples.length > 0) { curSample = 0; void renderAll(); }
}

void init();
