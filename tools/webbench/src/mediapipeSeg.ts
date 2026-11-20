// MediaPipe Image Segmenter(Selfie Segmenter)。カメラのライブ実行で
// セグメンテーションの比較対象として並べる。
//
// ブラウザ上の人物セグメンテーションでは事実上の標準として広く使われている
// モデル(Google公式配布、Apache License 2.0、入力256×256)。
// **切り分ける対象が違う点に注意**: こちらは「人物と背景」、DLGN側は
// Oxford-IIIT Pet で学習した「ペットと背景」。速度・配信サイズは同条件で
// 比べられるが、精度の直接比較にはならない(UI側にも明記する)。
//
// アセット配信の事情は mediapipeDet.ts と同じ(wasm はディレクトリごとコピー、
// モデルは scripts/fetch_mediapipe_models.mjs で public/models/ へ取得)。
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

import { gzipSize, resourceBytes, resourceGzipBytes } from './compare';
import type { LoadInfo, RunInfo } from './compare';
import type { SegRuntime } from './compareSeg';
import { asset } from './basePath';

const WASM_BASE = asset('/mediapipe/wasm');
const MODEL_URL = asset('/models/selfie_segmenter.tflite');
/** Selfie Segmenter の想定入力解像度。 */
const INPUT_SIZE = 256;
/** 人物確率のしきい値(MediaPipe のサンプル実装と同じ 0.5)。 */
const PERSON_THRESHOLD = 0.5;

export class SegMediaPipeRuntime implements SegRuntime {
  readonly name = 'MediaPipe 人物切り抜き(Selfie Segmenter)';
  outH = INPUT_SIZE;
  outW = INPUT_SIZE;
  inW = INPUT_SIZE;
  inH = INPUT_SIZE;
  private segmenter: ImageSegmenter | null = null;

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    // サイズ表示・gzip 実測のためモデル実体もここで一度取得する
    // (compareSeg.ts の各ランタイムと同じ手順)。
    const r = await fetch(MODEL_URL);
    if (!r.ok) throw new Error(`MediaPipeモデル未配置(${r.status}): ${MODEL_URL}`);
    const buf = await r.arrayBuffer();
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    this.segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'IMAGE',
      // Selfie Segmenter はカテゴリが1つ("selfie")だけのモデルで、
      // categoryMask は全画素 255 を返して使い物にならない(実測で確認)。
      // 正しくは confidenceMasks[0] の人物確率をしきい値で切る。
      outputCategoryMask: false,
      outputConfidenceMasks: true,
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
      paramsLabel: 'Selfie Segmenter(Apache License 2.0、Google公式配布)',
      note: '人物と背景を切り分ける実運用モデル。DLGN側はペットと背景を切り分ける'
        + 'モデルなので、精度の直接比較にはならない(速度・配信サイズは同条件)。',
      metricLabel: 'mIoU',
      metricValue: null, // 対象クラスが違うのでこのデモでは精度を実測しない
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<Int32Array>> {
    if (!this.segmenter) throw new Error('not loaded');
    // ImageData は ArrayBuffer 裏付けの Uint8ClampedArray しか受けない
    // (実行時は常に getImageData().data 由来なのでアサーションで安全)。
    const img = new ImageData(
      rgba as Uint8ClampedArray<ArrayBuffer>, INPUT_SIZE, INPUT_SIZE,
    );
    const t0 = performance.now();
    const res = this.segmenter.segment(img);
    const inferMs = performance.now() - t0;
    const mask = res.confidenceMasks?.[0];
    if (!mask) {
      res.close();
      throw new Error('confidenceMask を取得できませんでした');
    }
    this.outH = mask.height;
    this.outW = mask.width;
    const conf = mask.getAsFloat32Array();
    const labels = new Int32Array(conf.length);
    // 人物確率がしきい値以上の画素をクラス1にする(他のモデルと同じ塗り分け=赤)。
    for (let i = 0; i < conf.length; i++) labels[i] = conf[i] >= PERSON_THRESHOLD ? 1 : 0;
    res.close();
    return { output: labels, inferMs };
  }

  /** webcamの排他切替用。MediaPipe セグメンタを明示的に解放する。*/
  dispose(): void {
    this.segmenter?.close();
    this.segmenter = null;
  }
}
