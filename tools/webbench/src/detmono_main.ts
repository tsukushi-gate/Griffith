// モノクロ数字検出デモ(手書き数字 MNIST を1枚に複数並べた画像)。
// 比較用CNNの入力は student と同じ 3ch。Multi-MNIST は実質2値画像なので、
// 生グレー3ch複製と 0/1 変換後の3chが数値的に一致する。
import { asset, modelAsset } from './basePath';
import { teacherCfgFromRef } from './compareDet';
import { startDetPage } from './ui/detPage';
import { PAGES } from './ui/pages';

void startDetPage({
  page: PAGES.detmono,
  dlgnUrl: modelAsset('/models/det_multimnist_mono_lut_k256.json'),
  onnxUrl: asset('/models/det_mono_cnn_teacher.onnx'),
  teacher: teacherCfgFromRef('det_mono_teacher', 96, 10),
  display: 300,
});
