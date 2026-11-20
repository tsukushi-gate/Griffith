// 「参照値」の単一集約点。
//
// ここに置いてよいのは **モデルJSONから導出できない値だけ**:
//   - ONNX / TF.js teacher(CenterNetCNN, SmallUNet)の精度・params。
//     ONNXファイルにはメタデータが無いため、wandb 実測値をここに1箇所だけ書く。
//   - MediaPipe(BlazeFace short-range)の素性。
//
// DLGN側の数値は**絶対にここへ書かない**(modelMeta.ts でロード済みJSONから
// 導出する)。ページ本文からはこの表を参照するだけにして、同じ数字が複数の
// HTML/TSに散らばる状態(= 差し替え時に片方だけ古くなる)を無くす。
//
// nostale.test.ts はこのファイルだけを数値リテラル検査の対象外にしている。
// 値を更新するときは出典(wandb run id)も必ず一緒に更新すること。
import { MISSING, fillPlaceholders, fmtInt } from './modelMeta';

export interface ReferenceEntry {
  /** 表示名。 */
  label: string;
  /** 精度指標名('mAP' / 'mIoU')。持たない参照(MediaPipe)は null。 */
  metricLabel: string | null;
  metricValue: number | null;
  /** パラメータ数(float重み)。 */
  params: number | null;
  /** 出典 wandb run id。 */
  wandb: string | null;
  /** 補足(BN再較正の経緯など)。 */
  note?: string;
  /** 追加の数値(BN再較正前後など。キーは data-ref="<id>.extra.<k>" で参照)。 */
  extra?: Record<string, number>;
}

export const REFERENCES: Record<string, ReferenceEntry> = {
  // カラーMulti-MNISTのteacher。det.html / detbench.html / detbackends.html が共有。
  det_color_teacher: {
    label: '比較用のCNN(カラー数字検出)',
    metricLabel: 'mAP',
    metricValue: 0.6209,
    params: 1589966,
    wandb: 't9lbxjnm',
  },
  // モノクロMulti-MNISTのteacher。detmono.html。
  det_mono_teacher: {
    label: '比較用のCNN(モノクロ数字検出)',
    metricLabel: 'mAP',
    metricValue: 0.8817,
    params: 1589966,
    wandb: null,
    note: 'W2a 実測。det_mono_cnn_teacher.onnx と同一重み。',
  },
  // 少クラス実物体(person/car/dog、VOC)teacher。detreal.html。
  det_fewclass_teacher: {
    label: '比較用のCNN(実物体検出、Pascal VOC)',
    metricLabel: 'mAP',
    metricValue: 0.2764,
    params: 1589511,
    wandb: 'e3rm3118',
  },
  // 顔(WIDER FACE)teacher。detface.html。
  // ★BN再較正後の実測値(§3 と同手法: reset_running_stats + momentum=None +
  //   round入力で訓練データ300バッチ順伝播 → val全体で再実測)。訓練時のwandb記録
  //   (extra.trainLog)は best.pt から再現できない(train_detection.py が hard eval の
  //   後に soft eval(train mode)を走らせてから checkpoint を保存するため、train mode の
  //   BatchNorm が running統計を val バッチ統計で上書きする)。素の best.pt は extra.raw。
  det_face_teacher: {
    label: '比較用のCNN(顔検出、WIDER FACE)',
    metricLabel: 'mAP',
    metricValue: 0.4462,
    params: 1589381,
    wandb: 'yngfrawj',
    note: 'BN running統計を再較正した重みを配信(表示値と実体が一致)。',
    extra: { trainLog: 0.2146, raw: 0.1558 },
  },
  // SmallUNet teacher(Oxford-IIIT Pet、convbench 実測)。seg.html / segbackends.html。
  seg_unet_teacher: {
    label: '比較用のCNN(セグメンテーション、Oxford-IIIT Pet)',
    metricLabel: 'mIoU',
    metricValue: 0.5912,
    params: 1060850,
    wandb: null,
    note: 'convbench 実測。',
  },
  // 行動認識(UCI-HAR)の比較用CNN。ONNXにはメタデータが無いので、
  // 同じテストデータ(int16量子化後、2947窓)でブラウザ上で実測した値を置く。
  har_mlp_teacher: {
    label: '比較用のMLP(UCI-HAR)',
    metricLabel: '正解率',
    metricValue: 0.9169,
    params: null,
    wandb: null,
    note: 'webbench 実測(int16量子化後のテストデータ2947窓で2702正解)。',
  },
  har_cnn1d_teacher: {
    label: '比較用の1D CNN(UCI-HAR)',
    metricLabel: '正解率',
    metricValue: 0.9348,
    params: null,
    wandb: null,
    note: 'webbench 実測(int16量子化後のテストデータ2947窓で2755正解)。',
  },
  // MediaPipe FaceDetector(BlazeFace short-range)。精度はこのデモでは実測しない。
  mediapipe_blazeface: {
    label: 'MediaPipe 顔検出(BlazeFace short-range、Apache License 2.0)',
    metricLabel: null,
    metricValue: null,
    params: null,
    wandb: null,
    note: 'Google が大規模実顔画像で事前学習した実運用モデル。'
      + 'このデモではオフライン精度を実測しない(配信サイズ・ロード時間・推論速度のみ比較)。',
  },
};

/** LoadInfo.paramsLabel 用("パラメータ 1,589,966 個" / 欠損時は "—")。 */
export function teacherParamsLabel(r: ReferenceEntry): string {
  return r.params !== null ? `パラメータ ${fmtInt(r.params)} 個` : MISSING;
}

/** `data-ref="<id>.<field>"` を REFERENCES から埋める。
    field: accuracy(指標名+値) / accuracyValue / metricLabel / params / wandb /
    note / label / extra.<key> */
export function refField(spec: string): string | null {
  const dot = spec.indexOf('.');
  if (dot < 0) return null;
  const id = spec.slice(0, dot);
  const field = spec.slice(dot + 1);
  const r = REFERENCES[id];
  if (!r) return null;
  if (field.startsWith('extra.')) {
    const v = r.extra?.[field.slice('extra.'.length)];
    return v === undefined ? MISSING : v.toFixed(4);
  }
  switch (field) {
    case 'label': return r.label;
    case 'accuracy':
      return r.metricLabel !== null && r.metricValue !== null
        ? `${r.metricLabel} ${r.metricValue.toFixed(4)}` : MISSING;
    case 'accuracyValue':
      return r.metricValue !== null ? r.metricValue.toFixed(4) : MISSING;
    case 'metricLabel': return r.metricLabel ?? MISSING;
    case 'params': return r.params !== null ? `パラメータ ${fmtInt(r.params)} 個` : MISSING;
    case 'wandb': return r.wandb ?? MISSING;
    case 'note': return r.note ?? MISSING;
    default: return MISSING;
  }
}

/** ページ内の `[data-ref]` を全て埋める。 */
export function applyReferences(
  root: ParentNode | null = typeof document === 'undefined' ? null : document,
): number {
  return fillPlaceholders('ref', refField, root);
}
