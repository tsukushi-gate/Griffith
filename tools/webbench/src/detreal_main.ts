// 実物体検出デモ(Pascal VOC の写真から人・車・犬を検出する)。
import { asset, modelAsset } from './basePath';
import { teacherCfgFromRef } from './compareDet';
import { startDetPage } from './ui/detPage';
import { PAGES } from './ui/pages';

void startDetPage({
  page: PAGES.detreal,
  dlgnUrl: modelAsset('/models/det_fewclass_lut_k64.json'),
  onnxUrl: asset('/models/det_fewclass_cnn_teacher.onnx'),
  teacher: teacherCfgFromRef('det_fewclass_teacher', 128, 3),
  display: 340,
});
