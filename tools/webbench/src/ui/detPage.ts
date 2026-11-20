// 検出デモ4画面(カラー数字 / モノクロ数字 / 顔 / 実物体)の共通実装。
//
// 4ページは「モデルのURL・teacher設定・比較列の顔ぶれ・表示サイズ」しか違わない。
// 従来は約330行のファイルを4つコピーして保守していたので、ここへ集約する。
import { DetDlgnRuntime, DetOnnxRuntime, DetTfjsRuntime } from '../compareDet';
import type { DetRuntime, DetTeacherCfg, DetThermoCtx } from '../compareDet';
import { DetMediaPipeRuntime } from '../mediapipeDet';
import {
  boxCountNote, drawDetBoxes, drawGrayToCanvas, drawGtBoxes, drawRgbToCanvas,
  grayFromB64, loadDetModel, rgbFromB64,
} from '../detdemo';
import type { DetDemoSample, DetModel } from '../detdemo';
import { applyModelMeta } from '../modelMeta';
import { applyReferences } from '../references';
import { parseAutobenchParams, runAutobench } from '../autobench';
import type { DlgnBackend } from '../compareSeg';
import { runSequential } from '../sequentialRun';
import { RuntimeGrid } from './runtimeCard';
import type { SlotSpec } from './runtimeCard';
import { renderShell } from './shell';
import type { PageConfig } from './shell';
import { DEFAULT_DLGN_BACKEND } from './pages';
import { drawCoverCrop, toGrayRgba, wireImageUpload } from './upload';

export interface DetPageOptions {
  page: PageConfig;
  /** DLGNモデル(?model= 適用済みURL)。 */
  dlgnUrl: string;
  /** 比較用CNNのONNXファイル。 */
  onnxUrl: string;
  /** TensorFlow.js 版を並べる場合のディレクトリ。 */
  tfjsDir?: string;
  /** teacher(比較用CNN)の解像度・クラス数・参照値。省略時はカラー数字検出の既定。 */
  teacher?: DetTeacherCfg;
  /** 顔検出のみ: 第3列に MediaPipe を並べる(必ず末尾)。 */
  mediapipe?: boolean;
  /** 表示canvasの一辺(px)。 */
  display: number;
}

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

export async function startDetPage(opts: DetPageOptions): Promise<void> {
  // 計測モード(?autobench=1)ではUIを一切作らず、DLGN単体を測って終了する。
  const bench = parseAutobenchParams(opts.page.id);
  if (bench) {
    const rt = new DetDlgnRuntime(opts.dlgnUrl, bench.backend);
    await runAutobench(bench, rt, () => ({ w: rt.modelSize, h: rt.modelSize }));
    return;
  }

  renderShell(opts.page);

  // DLGNモデルのJSONがサンプル画像・ラベル・前処理の真実源(比較列も共有する)。
  const refModel: DetModel = await loadDetModel(opts.dlgnUrl);
  applyModelMeta(refModel.doc);
  applyReferences();

  // detmono: 単チャンネルモデルにカラー写真をそのまま流すとR成分だけを見ることに
  // なる(convpreproc.ts::encodeImageThermometerはNCHWのch0しか読まない)。
  // アップロード画像だけグレースケール化して揃える(埋め込みサンプルは既にr=g=bで
  // 変換不要、かつ変換対象から外してビット同一性を保つ)。
  const isMono = refModel.doc.preproc.mean.length === 1;

  const labelCtx = (): DetThermoCtx => ({ labels: refModel.labels });

  let curSample = -1;
  let curImage: HTMLImageElement | null = null;
  let showGt = true;

  const dlgnBackend = RuntimeGrid.initialBackend('dlgnBackend', DEFAULT_DLGN_BACKEND) as DlgnBackend;
  const onnxBackend = RuntimeGrid.initialBackend('onnxBackend', 'wasm');

  const slots: SlotSpec<DetRuntime>[] = [
    {
      kind: 'dlgn',
      label: 'DLGN(論理ゲート回路)',
      backend: dlgnBackend,
      create: (b) => new DetDlgnRuntime(opts.dlgnUrl, b as DlgnBackend),
    },
    {
      kind: 'onnx',
      label: '比較用のCNN(ONNX Runtime Web)',
      backend: onnxBackend,
      create: (b) => new DetOnnxRuntime(
        opts.onnxUrl, labelCtx, b as 'wasm' | 'webgpu', opts.teacher,
      ),
    },
  ];
  if (opts.tfjsDir) {
    const dir = opts.tfjsDir;
    slots.push({
      kind: 'other',
      label: '比較用のCNN(TensorFlow.js)',
      // TF.js も CPU(wasm)に揃える。既定が webgl のままだと ONNX/DLGN だけ
      // CPU で TF.js だけ GPU という不揃いな比較になる。
      create: () => new DetTfjsRuntime(dir, labelCtx, 'wasm', opts.teacher),
    });
  }
  // MediaPipe は cols[0]=DLGN / cols[1]=ONNX の前提を壊さないよう必ず末尾。
  if (opts.mediapipe) {
    slots.push({
      kind: 'other',
      label: 'MediaPipe(顔検出)',
      create: () => new DetMediaPipeRuntime(),
    });
  }

  const grid = new RuntimeGrid<DetRuntime>({
    gridEl: $('grid'),
    display: opts.display,
    slots,
    onRender: () => renderAll(),
  });

  function drawSample(canvas: HTMLCanvasElement, s: DetDemoSample): void {
    if (s.rgb_b64) {
      drawRgbToCanvas(canvas, rgbFromB64(s.rgb_b64, s.height * s.width), s.height, s.width);
    } else if (s.gray_b64) {
      drawGrayToCanvas(canvas, grayFromB64(s.gray_b64, s.height * s.width), s.height, s.width);
    }
  }

  /** 現在の入力をモデル解像度の RGBA にする(全ランタイム共通入力)。
      アップロード画像は中央正方形に切り出してから縮小する(webcamのライブ入力と
      同じ扱い。歪めて引き伸ばさない)。 */
  function currentRgba(size: number): Uint8ClampedArray {
    const small = document.createElement('canvas');
    small.width = size; small.height = size;
    const sctx = small.getContext('2d')!;
    if (curSample >= 0) drawSample(small, refModel.samples[curSample]);
    else if (curImage) drawCoverCrop(sctx, curImage, size, size);
    const data = sctx.getImageData(0, 0, size, size).data;
    if (curImage && curSample < 0 && isMono) toGrayRgba(data);
    return data;
  }

  /** 列ごとに厳密に逐次実行する(CPU/GPU資源の競合を避け、真の推論速度に
      近づける。複数ランタイムを Promise.all で同時に走らせると、表示される
      「推論時間」が資源の奪い合いで実際より遅く出てしまうため)。 */
  async function renderAll(): Promise<void> {
    if (curSample < 0 && !curImage) return;
    const thr = Number($<HTMLInputElement>('thr').value) / 100;
    await runSequential(grid.cols, async (c) => {
      if (c.error || !c.info) return;
      const bctx = c.base.getContext('2d')!;
      bctx.imageSmoothingEnabled = false;
      bctx.clearRect(0, 0, opts.display, opts.display);
      if (curSample >= 0) {
        drawSample(c.base, refModel.samples[curSample]);
      } else if (curImage) {
        drawCoverCrop(bctx, curImage, opts.display, opts.display);
        if (isMono) {
          const img = bctx.getImageData(0, 0, opts.display, opts.display);
          toGrayRgba(img.data);
          bctx.putImageData(img, 0, 0);
        }
      }
      try {
        const { output, ms } = await RuntimeGrid.timedRun(c.rt, currentRgba(c.rt.modelSize));
        // drawDetBoxes → drawGtBoxes の順序厳守(前者が先頭で overlay を消す)。
        const shown = drawDetBoxes(c.overlay, output, c.rt.labels, c.rt.modelSize, thr);
        if (showGt && curSample >= 0) {
          drawGtBoxes(c.overlay, refModel.samples[curSample], c.rt.modelSize);
        }
        grid.setInference(c, ms, boxCountNote(output, shown, thr));
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
    const removed = (refModel.doc as {
      demo_samples_removed?: { count: number; reason: string };
    }).demo_samples_removed;
    if (refModel.samples.length === 0 && removed) {
      // 公開ビルドでは再配布できないデータセット画像を取り除いている。
      // 空のままだと「壊れている」と誤解されるので理由を書く。
      const note = document.createElement('span');
      note.className = 'text-xs text-muted';
      note.textContent = `サンプル画像は公開版には含めていません(${removed.reason})。`
        + '左の「手元の画像を試す」から読み込んでください。';
      box.append(note);
      return;
    }
    refModel.samples.forEach((s, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'rounded-md border border-line bg-card px-2.5 py-1 text-xs';
      const names = s.gt_labels.map((l) => refModel.labels[l] ?? l).join(', ');
      btn.textContent = names ? `#${i + 1}(${names})` : `#${i + 1}`;
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
  $<HTMLInputElement>('thr').oninput = () => void renderAll();
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
