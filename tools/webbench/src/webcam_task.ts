// webcam_main.ts のタスク判定/モデル選択ロジック(副作用なしの純関数群)。
// webcam_main.ts はモジュール読み込み時に init() を実行し DOM(document)に触れる
// ため、node環境のvitestから直接importできない。ユニットテスト対象の純関数だけを
// ここへ切り出す。
export interface Manifest {
  dlgn: string[];
  conv?: string[];
  det?: string[];
  onnx: string[];
}

/** webcamで回せるタスク。seg=SegConvDLGN(Pet 2クラス)、
    det=CenterNetLUT(少クラス実物体 person/car/dog、カラーRGB thermometer)、
    face=CenterNetLUT(WIDER FACE顔検出、1クラス)。*/
export type Task = 'seg' | 'det' | 'face';

/** det/face はどちらも DetModel(CenterNetLUT検出ヘッド)を使う。
    task==='det' の直値比較を分岐漏れなく置換するための共通判定
    (seg以外は全て検出タスク)。 */
export const isDet = (t: Task): boolean => t !== 'seg';

/** webcamの「手法」セレクタ(1つだけ選ぶ方式)でタスクごとに選べる
    手法を判定する。DLGN・ONNX Runtime Web・MediaPipe はseg/faceどちらにも
    比較用モデルがあるので常に選べる。TensorFlow.js への変換は seg_unet のみ
    済んでいる(det/face の CenterNetCNN teacher は未変換)ため、タスクで
    出し分ける必要があるのは tfjs だけ。 */
export function tfjsAvailable(task: Task): boolean {
  return !isDet(task);
}

/** manifest から task に対応するモデルファイル名を選ぶ純関数。
    face → 'det_face_' 接頭、det(少クラス実物体) → 'det_fewclass_' 接頭
    (無ければ manifest.det の先頭へフォールバック)、seg → 従来通り
    manifest.conv からdemoでない先頭(無ければconvの先頭)。 */
export function resolveModelFile(manifest: Manifest, task: Task): string | null {
  if (task === 'face') {
    return manifest.det?.find((f) => f.startsWith('det_face_')) ?? null;
  }
  if (task === 'det') {
    return manifest.det?.find((f) => f.startsWith('det_fewclass_'))
      ?? manifest.det?.[0] ?? null;
  }
  return manifest.conv?.find((f) => !f.startsWith('demo'))
    ?? manifest.conv?.[0] ?? null;
}
