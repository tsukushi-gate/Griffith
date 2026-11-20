// webcam_task.ts の resolveModelFile(純関数)のユニットテスト。
// task='det'/'face'/'seg' それぞれで manifest から正しいモデルファイルを
// 選ぶこと(face→det_face_接頭、det→det_fewclass_接頭、seg→conv側)を検証する。
import { describe, expect, it } from 'vitest';

import { resolveModelFile, tfjsAvailable } from './webcam_task';

const manifest = {
  dlgn: [],
  conv: ['demo_seg.json', 'seg_convdlgn.json'],
  det: [
    'det_fewclass_lut_k64.json',
    'det_face_lut_k256.json',
    'det_multimnist_lut_k64.json',
  ],
  onnx: [],
};

describe('resolveModelFile', () => {
  it("'det' は det_fewclass_ 接頭のファイルを返す", () => {
    expect(resolveModelFile(manifest, 'det')).toBe('det_fewclass_lut_k64.json');
  });

  it("'face' は det_face_ 接頭のファイルを返す", () => {
    expect(resolveModelFile(manifest, 'face')).toBe('det_face_lut_k256.json');
  });

  it("'seg' は conv側(demoでない先頭)を返す", () => {
    expect(resolveModelFile(manifest, 'seg')).toBe('seg_convdlgn.json');
  });

  it("'det' 該当ファイルが無ければ manifest.det の先頭にフォールバックする", () => {
    const m = { ...manifest, det: ['det_multimnist_lut_k64.json'] };
    expect(resolveModelFile(m, 'det')).toBe('det_multimnist_lut_k64.json');
  });

  it("'face' 該当ファイルが無ければ null を返す(det側へフォールバックしない)", () => {
    const m = { ...manifest, det: ['det_fewclass_lut_k64.json'] };
    expect(resolveModelFile(m, 'face')).toBeNull();
  });

  it("conv が空なら 'seg' は null を返す", () => {
    const m = { ...manifest, conv: [] };
    expect(resolveModelFile(m, 'seg')).toBeNull();
  });
});

describe('tfjsAvailable', () => {
  it("'seg' は TF.js が選べる", () => {
    expect(tfjsAvailable('seg')).toBe(true);
  });

  it("'face' は TF.js 変換モデルが無いので選べない", () => {
    expect(tfjsAvailable('face')).toBe(false);
  });

  it("'det' も検出系(TF.js未変換)なので選べない", () => {
    expect(tfjsAvailable('det')).toBe(false);
  });
});
