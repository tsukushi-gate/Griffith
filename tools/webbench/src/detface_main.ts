// 顔検出デモ(WIDER FACE)。第3列に MediaPipe の顔検出を並べる。
import { asset, modelAsset } from './basePath';
import { teacherCfgFromRef } from './compareDet';
import { startDetPage } from './ui/detPage';
import { PAGES } from './ui/pages';

void startDetPage({
  page: PAGES.detface,
  dlgnUrl: modelAsset('/models/det_face_lut_k256.json'),
  onnxUrl: asset('/models/det_face_cnn_teacher.onnx'),
  teacher: teacherCfgFromRef('det_face_teacher', 128, 1),
  mediapipe: true,
  display: 340,
});
