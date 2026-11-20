// カラー数字検出デモ(CIFAR-10 の風景写真に色付き数字を重ねた合成画像)。
// 画面の組み立ては ui/detPage.ts が共通で行う。ここはモデルの所在だけを持つ。
import { asset, modelAsset } from './basePath';
import { startDetPage } from './ui/detPage';
import { PAGES } from './ui/pages';

// `?model=_ll` / `?model=<file>.json` でDLGNモデルを差し替えられる。
void startDetPage({
  page: PAGES.det,
  dlgnUrl: modelAsset('/models/det_multimnist_lut_k256.json'),
  onnxUrl: asset('/models/det_cnn_teacher.onnx'),
  tfjsDir: asset('/models/det_cnn_teacher_tfjs'),
  display: 300,
});
