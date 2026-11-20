// カメラでライブ実行。1フレームを取り込み、「手法」セレクタで選んだ1つの
// ランタイムだけを実行する(表示は常に1枚、稼働エンジンも常に1つ)。
//
// 以前は MediaPipe が「計算方法」の選択肢に混ざっていた。MediaPipe は
// DLGN の計算方法ではなく別実装の手法なので、独立した選択肢にした。
// 一時期は比較手法を横に並べる列(チェックボックスで増減)にしたが、
// 「1つだけ選んで動かす」方が単純で分かりやすいという要望により、
// 比較列ではなく単一の「手法」セレクタ方式に変更した。
// 手法は dlgn(DLGN) / onnx(ONNX Runtime Web) / tfjs(TensorFlow.js、seg限定) /
// mediapipe の4択。DLGN/ONNXは選んだときだけ計算方法(js/wasm/webgpu, wasm/webgpu)
// の第2セレクタが現れる(操作は「手法→計算方法」の最大2段)。
// タスク/手法/計算方法いずれの切替でも「in-flight の推論が完了するのを待つ→
// 旧ランタイムを dispose→参照をnullに戻す→新ランタイムをロード」の排他手順を
// 踏む(ONNX/TF.js/MediaPipeのセッションリークを防ぐ)。
//
// 推論は非同期で行い、rAFループとは inflight フラグで疎結合にする(前フレームの
// 推論中は次を開始しない=最新フレーム優先でフレーム落とし)。
import { SegDlgnRuntime, SegOnnxRuntime, SegTfjsRuntime } from './compareSeg';
import type { DlgnBackend, OnnxBackend, SegRuntime } from './compareSeg';
import { DetDlgnRuntime, DetOnnxRuntime, teacherCfgFromRef } from './compareDet';
import type { DetRuntime, DetTeacherCfg } from './compareDet';
import { DetMediaPipeRuntime } from './mediapipeDet';
import { SegMediaPipeRuntime } from './mediapipeSeg';
import { boxCountNote, drawDetBoxes } from './detdemo';
import { drawMask } from './segdemo';
import type { DetBoxPx } from './detdecode';
import { isDet, resolveModelFile, tfjsAvailable } from './webcam_task';
import type { Manifest, Task } from './webcam_task';
import { asset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { applyReferences } from './references';
import { runSequential } from './sequentialRun';
import { RuntimeGrid } from './ui/runtimeCard';
import type { SlotSpec } from './ui/runtimeCard';
import { renderShell } from './ui/shell';
import { DEFAULT_DLGN_BACKEND, PAGES } from './ui/pages';

/** webcamの「手法」セレクタの値。 */
type Method = 'dlgn' | 'onnx' | 'tfjs' | 'mediapipe';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

/** カメラ用ランタイムの最小契約(seg/det どちらの実装も満たす)。 */
type LiveRuntime = SegRuntime | DetRuntime | DetMediaPipeRuntime;

/** タスクごとの比較用CNN(ONNX)。DLGN側モデルと同じ解像度・クラス数で揃える。
    ライブ実行のタスクは seg / face の2つだけ。*/
const ONNX_FOR_TASK: Record<string, { url: string; teacher?: DetTeacherCfg }> = {
  seg: { url: asset('/models/seg_unet.onnx') },
  face: {
    url: asset('/models/det_face_cnn_teacher.onnx'),
    teacher: teacherCfgFromRef('det_face_teacher', 128, 1),
  },
};

/** タスクごとの比較用CNN(TensorFlow.js)。onnx2tf 変換は seg_unet のみ
    済んでいる(det_face_cnn_teacher は未変換)ため seg しかエントリが無い
    (tfjsAvailable(task) と整合させる)。 */
const TFJS_FOR_TASK: Record<string, { dir: string }> = {
  seg: { dir: asset('/models/seg_unet_tfjs') },
};

/** 入力解像度(ランタイムごとに違うのでフレームごとに切り出し直す)。 */
function inputSizeOf(rt: LiveRuntime): { w: number; h: number } {
  if ('modelSize' in rt) return { w: rt.modelSize, h: rt.modelSize };
  const seg = rt as SegRuntime;
  return { w: seg.inW, h: seg.inH };
}

async function init(): Promise<void> {
  renderShell(PAGES.webcam);

  let task: Task = 'seg';
  let running = false;
  let grid: RuntimeGrid<LiveRuntime> | null = null;
  let manifest: Manifest | null = null;
  // 比較用CNNはラベル一覧を DLGN 側のモデルJSONから受け取る(同一デコードを使う)。
  let dlgnLabels: string[] = [];
  const emaMs = new WeakMap<object, number>();

  const video = $<HTMLVideoElement>('video');
  const capture = document.createElement('canvas');
  const cctx = capture.getContext('2d', { willReadFrequently: true })!;

  async function loadManifest(): Promise<Manifest | null> {
    if (manifest) return manifest;
    try {
      manifest = await fetch(asset('/models/index.json')).then((r) => {
        if (!r.ok) throw new Error(`index.json: ${r.status}`);
        return r.json() as Promise<Manifest>;
      });
      return manifest;
    } catch {
      return null;
    }
  }

  /** カメラ映像の中央正方形を切り出してモデル解像度の RGBA にする。 */
  function frameRgba(w: number, h: number): Uint8ClampedArray | null {
    if (video.videoWidth === 0) return null;
    capture.width = w; capture.height = h;
    const side = Math.min(video.videoWidth, video.videoHeight);
    cctx.drawImage(
      video,
      (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side,
      0, 0, w, h,
    );
    return cctx.getImageData(0, 0, w, h).data;
  }

  function drawFrame(canvas: HTMLCanvasElement): void {
    if (video.videoWidth === 0) return;
    const ctx = canvas.getContext('2d')!;
    const side = Math.min(video.videoWidth, video.videoHeight);
    ctx.drawImage(
      video,
      (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side,
      0, 0, canvas.width, canvas.height,
    );
  }

  /** 1フレームぶんを選択中の手法1つだけで実行して描く。
      稼働するランタイムは常に1つ(grid.cols.length===1)なので、
      det/seg/har の比較デモと違って複数列を競合させる問題自体が無い。
      runSequential は「1件だけの配列」として使い、既存の逐次実行ヘルパーを
      そのまま流用する。 */
  async function step(): Promise<void> {
    if (!grid) return;
    const infer = $<HTMLInputElement>('infer').checked;
    const thr = Number($<HTMLInputElement>('thr').value) / 100;
    await runSequential(grid.cols, async (c) => {
      if (c.error || !c.info) return;
      drawFrame(c.base);
      if (!infer) {
        c.overlay.getContext('2d')!.clearRect(0, 0, c.overlay.width, c.overlay.height);
        return;
      }
      const size = inputSizeOf(c.rt);
      const rgba = frameRgba(size.w, size.h);
      if (!rgba) return;
      try {
        const res = await c.rt.run(rgba);
        // 毎フレーム値が踊らないよう指数移動平均で均す。
        const prev = emaMs.get(c.rt as object);
        const ms = prev === undefined ? res.inferMs : prev * 0.8 + res.inferMs * 0.2;
        emaMs.set(c.rt as object, ms);
        const fpsNote = `${(ms > 0 ? 1000 / ms : 0).toFixed(1)} FPS相当`;
        if (isDet(task)) {
          const det = c.rt as DetRuntime | DetMediaPipeRuntime;
          const boxes = res.output as DetBoxPx[];
          const shown = drawDetBoxes(c.overlay, boxes, det.labels, size.w, thr);
          (grid as RuntimeGrid<LiveRuntime>).setInference(
            c, ms, `${boxCountNote(boxes, shown, thr)} / ${fpsNote}`, '直近フレームの平均',
          );
        } else {
          const seg = c.rt as SegRuntime;
          drawMask(c.overlay, res.output as Int32Array, seg.outH, seg.outW);
          (grid as RuntimeGrid<LiveRuntime>).setInference(c, ms, fpsNote, '直近フレームの平均');
        }
      } catch (e) {
        (grid as RuntimeGrid<LiveRuntime>).setRunError(c, String(e));
      }
    });
  }

  let inflight = false;
  // 直前の step() 呼び出し(未解決なら)。タスク/比較手法の切替で
  // ランタイムを排他的に差し替える前に、これが解決するのを待ってから
  // dispose する(切替中の旧ランタイムへ推論が飛び込むのを防ぐ)。
  let currentStep: Promise<void> | null = null;
  function loop(): void {
    if (!running) return;
    if (!inflight) {
      inflight = true;
      currentStep = step().finally(() => { inflight = false; });
    }
    requestAnimationFrame(loop);
  }

  /** in-flight の step() が完了するのを待つ(webcamの排他切替用フック。
      RuntimeGrid.switchBackend にも渡す)。 */
  async function waitInFlight(): Promise<void> {
    if (currentStep) await currentStep.catch(() => {});
  }

  /** タスク or 手法の構成が変わったらカードを組み直す(常に1枚)。
      排他切替: in-flight の推論(step())が完全に終わるのを待ってから、
      旧ランタイムを dispose し、新しい構成をロードし直す。 */
  async function rebuild(): Promise<void> {
    const m = await loadManifest();
    const file = m ? resolveModelFile(m, task) : null;
    if (!file) {
      $('status').textContent = 'このタスクのモデルが見つかりません。';
      return;
    }
    const dlgnUrl = asset(`/models/${file}`);
    // 比較用CNNは DLGN と同じラベル体系でデコードするので、先にモデルJSONを
    // 読んでラベル・メタデータを確定させる(2度目の fetch は HTTP キャッシュ)。
    const doc = await fetch(dlgnUrl).then((r) => r.json() as Promise<{
      labels?: string[];
    }>);
    dlgnLabels = doc.labels ?? [];
    applyModelMeta(doc);
    applyReferences();
    const dlgnBackend = RuntimeGrid.initialBackend('backend', DEFAULT_DLGN_BACKEND) as DlgnBackend;
    const onnxBackend = RuntimeGrid.initialBackend('onnxBackend', 'wasm');
    const onnxCfg = ONNX_FOR_TASK[task];
    const tfjsCfg = TFJS_FOR_TASK[task];
    let method = $<HTMLSelectElement>('method').value as Method;
    // 防御的フォールバック: tfjs はタスク依存(現状seg限定)。selectの選択肢は
    // applyTaskUi() が出し分けるが、万一 tfjsCfg が無いタスクで tfjs が選ばれて
    // いたら DLGN へ落とす(既定のDLGN計算方法を使う)。
    if (method === 'tfjs' && !tfjsCfg) method = 'dlgn';

    let slot: SlotSpec<LiveRuntime>;
    if (method === 'onnx') {
      slot = {
        kind: 'onnx',
        label: '比較用のCNN(ONNX Runtime Web)',
        backend: onnxBackend,
        create: (b) => (isDet(task)
          ? new DetOnnxRuntime(
            onnxCfg.url, () => ({ labels: dlgnLabels }), b as OnnxBackend, onnxCfg.teacher,
          )
          : new SegOnnxRuntime(onnxCfg.url, b as OnnxBackend)),
      };
    } else if (method === 'tfjs' && tfjsCfg) {
      // TensorFlow.js は onnx2tf 変換が seg_unet のみ済んでいるので seg のみ。
      slot = {
        kind: 'other',
        label: '比較用のCNN(TensorFlow.js)',
        create: () => new SegTfjsRuntime(tfjsCfg.dir, 'wasm'),
      };
    } else if (method === 'mediapipe') {
      // MediaPipe は DLGN の計算方法ではなく別実装の手法。タスクに応じて
      // 顔検出モデル / 人物切り抜きモデルを使う(どちらもブラウザの定番)。
      slot = isDet(task)
        ? { kind: 'other', label: 'MediaPipe(顔検出)', create: () => new DetMediaPipeRuntime() }
        : {
          kind: 'other',
          label: 'MediaPipe(人物切り抜き)',
          create: () => new SegMediaPipeRuntime(),
        };
    } else {
      slot = {
        kind: 'dlgn',
        label: 'DLGN(論理ゲート回路)',
        backend: dlgnBackend,
        create: (b) => (isDet(task)
          ? new DetDlgnRuntime(dlgnUrl, b as DlgnBackend)
          : new SegDlgnRuntime(dlgnUrl, b as DlgnBackend)),
      };
    }

    // 排他切替: 新しいグリッドを組む前に、直前のフレームが完全に終わるのを
    // 待ってから旧ランタイムを解放する(ONNX/TF.jsセッションのリーク防止。
    // DLGN/MediaPipeなど dispose 未実装のランタイムは無視される)。
    await waitInFlight();
    const oldGrid = grid;
    grid = null;
    if (oldGrid) {
      await Promise.all(oldGrid.cols.map(async (c) => {
        try {
          await c.rt.dispose?.();
        } catch {
          // 解放失敗は致命的でない(ページ寿命で最終的に回収される)ため無視する。
        }
      }));
    }

    grid = new RuntimeGrid<LiveRuntime>({
      gridEl: $('grid'),
      display: 320,
      slots: [slot],
      // 常に1枚のライブ表示。'live'指定でdet/seg等の比較ページと
      // 同じグリッド列幅を使うため、カード1枚の大きさが比較ページと揃う
      // (詳細は runtimeCard.ts の RuntimeGridOptions.aspect)。
      aspect: 'live',
      onRender: () => {},
      waitInFlight,
    });
    grid.build();
    grid.bindBackendSelect('dlgn', 'backend', DEFAULT_DLGN_BACKEND);
    grid.bindBackendSelect('onnx', 'onnxBackend', 'wasm');
    $('status').textContent = '読み込み中…';
    await grid.loadAll();
    $('status').textContent = running
      ? '' : 'カメラを開始すると推論が始まります。';
  }

  /** 手法(method)セレクタの値に応じて、第2段の計算方法セレクタを出し分ける
      (DLGN選択時は #dlgnBackendWrap、ONNX選択時は #onnxBackendWrap のみ表示)。 */
  function applyMethodUi(): void {
    const method = $<HTMLSelectElement>('method').value as Method;
    $('dlgnBackendWrap').style.display = method === 'dlgn' ? '' : 'none';
    $('onnxBackendWrap').style.display = method === 'onnx' ? '' : 'none';
  }

  /** タスク切替時に、タスク依存の選択肢(検出しきい値・tfjsの可否)を揃える。 */
  function applyTaskUi(): void {
    $('thrWrap').style.display = isDet(task) ? '' : 'none';
    const methodSel = $<HTMLSelectElement>('method');
    const tfjsOpt = methodSel.querySelector('option[value="tfjs"]') as HTMLOptionElement | null;
    const tfjsOk = tfjsAvailable(task);
    if (tfjsOpt) {
      tfjsOpt.disabled = !tfjsOk;
      tfjsOpt.textContent = tfjsOk
        ? '比較用のCNN(TensorFlow.js)'
        : '比較用のCNN(TensorFlow.js)(このタスクでは未対応)';
    }
    // 選択中の手法がこのタスクでは使えなくなった場合はDLGNへ戻す。
    if (!tfjsOk && methodSel.value === 'tfjs') methodSel.value = 'dlgn';
    // MediaPipeの選択肢名をタスクに応じて分かりやすくする。
    const mpOpt = methodSel.querySelector('option[value="mediapipe"]') as HTMLOptionElement | null;
    if (mpOpt) mpOpt.textContent = isDet(task) ? 'MediaPipe(顔検出)' : 'MediaPipe(人物切り抜き)';
    applyMethodUi();
  }

  applyTaskUi();
  await rebuild();

  $<HTMLSelectElement>('task').onchange = (ev) => {
    task = (ev.target as HTMLSelectElement).value as Task;
    applyTaskUi();
    void rebuild();
  };
  $<HTMLSelectElement>('method').onchange = () => {
    applyMethodUi();
    void rebuild();
  };

  const btn = $<HTMLButtonElement>('start');
  btn.onclick = async () => {
    if (running) {
      running = false;
      (video.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      btn.textContent = 'カメラ開始';
      $('status').textContent = 'カメラを停止しました。';
      return;
    }
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }, audio: false,
      });
      await video.play();
      running = true;
      btn.textContent = 'カメラ停止';
      $('status').textContent = '';
      requestAnimationFrame(loop);
    } catch (e) {
      $('status').textContent = `カメラを開始できませんでした: ${String(e)}`;
    }
  };
}

void init();
