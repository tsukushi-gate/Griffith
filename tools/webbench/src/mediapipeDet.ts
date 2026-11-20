// MediaPipe Face Detection(BlazeFace short-range)を detface.html の
// 第3列 / webcam.html の顔検出バックエンド選択肢として追加する。
//
// BlazeFace short-range は Google がWIDER FACEを大幅に上回る規模の実データで
// 事前学習した実運用モデル(Apache License 2.0)。DLGN/ONNX teacherの hard mAP
// (WIDER FACE val実測)とは学習データも学習規模も異なるため単純な精度比較には
// ならない前提でデモに表示する(公平なのは配信サイズ・ロード時間・推論速度)。
//
// DetRuntime契約(name/load/run/modelSize/labels)に合わせる。入力解像度は
// 128×128(detface.htmlの表示解像度・DLGN/ONNX列と完全一致、前処理リサイズに
// よる不公平が生じない)。CenterNet decodeは通さず、MediaPipeの
// `detections[].boundingBox{originX,originY,width,height}` + `categories[0].score`
// をそのまま DetBoxPx[] へ詰め替えるだけ。
//
// アセット配信: FilesetResolver.forVisionTasks(basePath) は basePath を
// 「ディレクトリURL」として要求し、内部で既知ファイル名を文字列連結して fetch する
// (Viteの `?url` 個別ハッシュ付きURLでは成立しない)。そのため
// node_modules/@mediapipe/tasks-vision/wasm/ を public/mediapipe/wasm/ へ
// scripts/copy_mediapipe_assets.mjs でコピーし(predev/prebuildで自動実行)、
// asset('/mediapipe/wasm') をbasePathとして渡す。`.task` モデルは
// scripts/fetch_mediapipe_facemodel.mjs で public/models/ に取得する
// (いずれも .gitignore 済み、手動配置不要)。
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

import { gzipSize, resourceBytes, resourceGzipBytes } from './compare';
import type { LoadInfo, RunInfo } from './compare';
import type { DetRuntime } from './compareDet';
import type { DetBoxPx } from './detdecode';
import { asset } from './basePath';

const WASM_BASE = asset('/mediapipe/wasm');
// 注記: MediaPipe公式ドキュメントが例示する `.../float16/1/blaze_face_short_range.task`
// は404(該当バージョンパスに.taskは存在しない)。実際に配布されているのは
// `float16/latest/blaze_face_short_range.tflite`(約224KB、メタデータ埋め込み済みで
// modelAssetPathへそのまま渡せる)。scripts/fetch_mediapipe_facemodel.mjs が取得する
// ファイル名と一致させる。
const MODEL_URL = asset('/models/blaze_face_short_range.tflite');
/** BlazeFace short-range の想定入力解像度(=detface.htmlの表示解像度と一致)。 */
const INPUT_SIZE = 128;

export class DetMediaPipeRuntime implements DetRuntime {
  readonly name = 'MediaPipe 顔検出(BlazeFace short-range)';
  modelSize = INPUT_SIZE;
  labels: string[] = ['face'];
  private detector: FaceDetector | null = null;

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    // FilesetResolver/FaceDetector は内部でも modelAssetPath を fetch するが、
    // モデルバイト数(サイズ表示・gzip計測)を得るためここでも一度取得する
    // (compareDet.ts の DetOnnxRuntime/DetTfjsRuntime と同じパターン)。
    const r = await fetch(MODEL_URL);
    if (!r.ok) throw new Error(`MediaPipeモデル未配置(${r.status}): ${MODEL_URL}`);
    const buf = await r.arrayBuffer();
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    this.detector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'IMAGE',
      // 閾値フィルタはデモ側スコアスライダが担う(detface_main.ts/webcam_main.ts)ため
      // ここでは低め(MediaPipe既定0.5より緩める)にして箱を絞り込みすぎないようにする。
      minDetectionConfidence: 0.1,
    });
    const warm = new Uint8ClampedArray(INPUT_SIZE * INPUT_SIZE * 4).fill(255);
    await this.run(warm);
    const loadMs = performance.now() - t0;
    return {
      modelBytes: buf.byteLength,
      gzipBytes: await gzipSize(new Uint8Array(buf)),
      runtimeBytes: resourceBytes('vision_wasm'),
      runtimeGzipBytes: await resourceGzipBytes('vision_wasm'),
      loadMs,
      paramsLabel: 'BlazeFace short-range(Apache 2.0、Google公式配布)',
      note: '実運用モデル(膨大な実データで事前学習済み)。'
        + 'DLGN/ONNX teacherとは学習データ・規模が異なるため精度は参考値。'
        + '入力は128×128 RGB(前処理リサイズなし、他列と同一解像度)。',
      metricLabel: 'mAP',
      metricValue: null, // このデモではオフラインmAPを実測しない(決定事項)
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<DetBoxPx[]>> {
    if (!this.detector) throw new Error('not loaded');
    // TSのUint8ClampedArrayは既定でArrayBufferLike(ArrayBuffer|SharedArrayBuffer)
    // ジェネリックだが、ImageDataコンストラクタはArrayBuffer specializationしか
    // 受け付けない。実行時は常にgetImageData().data由来(ArrayBuffer裏付け)なので
    // 型アサーションのみで安全。
    const img = new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, INPUT_SIZE, INPUT_SIZE);
    const t0 = performance.now();
    const res = this.detector.detect(img);
    const inferMs = performance.now() - t0;
    const output: DetBoxPx[] = res.detections.map((d) => {
      const bb = d.boundingBox;
      const score = d.categories[0]?.score ?? 0;
      if (!bb) return { x1: 0, y1: 0, x2: 0, y2: 0, cls: 0, score };
      return {
        x1: bb.originX,
        y1: bb.originY,
        x2: bb.originX + bb.width,
        y2: bb.originY + bb.height,
        cls: 0,
        score,
      };
    });
    return { output, inferMs };
  }

  /** webcamの排他切替用。MediaPipe タスクを明示的に解放する。*/
  dispose(): void {
    this.detector?.close();
    this.detector = null;
  }
}
