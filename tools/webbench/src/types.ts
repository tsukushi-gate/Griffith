// webbenchの共通型(WB-4/W4)。scripts/export_webbench.py の出力スキーマと対応。
import type { IRGraphJson } from './shared';

/** dense(時系列)モデルの前処理パラメータ。
    Python側 export_webbench.dense_thermometer_params と同一の値 */
export interface Preproc {
  kind: 'thermometer';
  input_kind: 'raw' | 'features';
  mean: number[]; // [C]
  std: number[]; // [C]
  thresholds: number[][]; // [C][T](標準化後の等分位点、チャネル内単調増加)
  bit_order: 'ch,t,thr';
}

/** conv(画像)モデルの前処理パラメータ(convpreproc.ts)。
    Python側 export_webbench.image_thermometer_params と同一の値 */
export interface ImagePreproc {
  kind: 'image_thermometer';
  rgb_channels: number;
  n_thresholds: number;
  height: number;
  width: number;
  mean: number[]; // [rgb]
  std: number[]; // [rgb]
  thresholds: number[][]; // [rgb][T]
  bit_order: 'rgb,thr,y,x';
}

/** 出力デコード情報(エクスポート時にPython側が埋める) */
export interface Readout {
  kind: 'class_scores' | 'spatial_scores';
  k: number | null; // group_sumのk(spatialならn_out*H*W)
  spatial: { n_out: number; height: number; width: number } | null;
}

/** CenterNet検出ヘッド1本の分割メタ。offset=連結スコア列内の開始位置、
    scale=SpatialGroupSumのscale(生popcount→実数値、hm確率・size正規化・off)。 */
export interface DetHead {
  offset: number;
  n_out: number;
  h: number;
  w: number;
  scale: number;
}

/** CenterNet(多ヘッド)検出モデルのデコード情報。 */
export interface DetReadout {
  kind: 'detection';
  heads: { hm: DetHead; size: DetHead; off: DetHead };
  output_stride: number;
  top_k: number;
  threshold: number;
  n_classes: number;
}

export interface ModelDoc {
  ir: IRGraphJson;
  preproc: Preproc;
  labels: string[];
  readout?: Readout;
  resources: {
    total_luts: number;
    logic_depth: number;
    json_bytes?: number;
    gzip_bytes?: number;
    [k: string]: unknown;
  };
  eval: {
    hard_acc?: number;
    hard_acc_i16?: number; // int16往復後(ブラウザ一致基準)
    n_test?: number;
    confusion?: number[][];
  };
}

/** conv系モデルJSON(demo_seg.json等、preprocが画像thermometer) */
export interface ConvModelDoc extends Omit<ModelDoc, 'preproc'> {
  preproc: ImagePreproc;
}

/** 共有テストデータのメタ(public/data/<stem>_meta.json) */
export interface TestMeta {
  shape: number[]; // [N, C, L]
  dtype: 'int16le';
  scale: number;
  labels: string[];
  y_test: number[];
  mean: number[]; // ONNX側前処理用
  std: number[];
}

/** int16から復元済みのテストデータ(生値、標準化前) */
export interface TestData {
  meta: TestMeta;
  n: number;
  chLen: number; // C*L(1サンプルのfloat数)
  x: Float32Array; // [N * C * L](int16 * scale で復元済み)
}
