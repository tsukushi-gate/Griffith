// 低レベル特徴前処理(edge/curvature/chroma)の Python⇔TS ビット一致契約。
//
// Conv-DLGN(Petersen+ NeurIPS'24) §3.5 の "edge and curvature detector kernels
// with thresholds" に対応する固定前処理。真実源は Python の
// griffith_logic/encoding/image_features.py::encode_image_features で、
// ゴールデンは scripts/export_webbench.py::build_lowlevel_golden が生成する。
//
// なぜこのテストが必要か: export(IR preproc)と TS(encodeImageLowLevel)は
// **別々に書かれた同一仕様の実装**であり、片方だけ直すと「精度は出ているのに
// ブラウザでは別の入力を食わせている」壊れたモデルを静かに配信してしまう。
// conv 抽出器の配線バグが TS⇔simulate の自己整合テストを素通りした
// のと同型の穴なので、Python 側から作ったゴールデンで縛る。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { encodeImageLowLevel, type ImageLowLevelParams } from './convpreproc';

const here = dirname(fileURLToPath(import.meta.url));

function b64ToBits(b64: string, nBits: number): Uint8Array {
  const raw = atob(b64);
  const bits = new Uint8Array(nBits);
  for (let i = 0; i < nBits; i++) {
    bits[i] = (raw.charCodeAt(i >> 3) >> (i & 7)) & 1;
  }
  return bits;
}

const goldenPath = join(here, 'fixtures/lowlevel_golden.json');
const hasGolden = existsSync(goldenPath);

interface Golden {
  shape: [number, number, number, number];
  n_thresholds: number;
  edge_thresholds: number[];
  curv_thresholds: number[];
  chroma_thresholds: number[];
  edge_offsets: number[][];
  curv_offsets: number[][];
  chroma_pairs: number[][];
  mean: number[];
  std: number[];
  thresholds: number[][];
  x: number[][];
  out_channels: number;
  out_width: number;
  bits_b64: string[];
}

describe.skipIf(!hasGolden)('convpreproc: 低レベル特徴 goldenとのビット一致', () => {
  const golden = JSON.parse(readFileSync(goldenPath, 'utf8')) as Golden;
  const [n, , h, w] = golden.shape;
  const params: ImageLowLevelParams = {
    mean: golden.mean,
    std: golden.std,
    thresholds: golden.thresholds,
    edge_thresholds: golden.edge_thresholds,
    curv_thresholds: golden.curv_thresholds,
    chroma_thresholds: golden.chroma_thresholds,
    edge_offsets: golden.edge_offsets,
    curv_offsets: golden.curv_offsets,
    chroma_pairs: golden.chroma_pairs,
    height: h,
    width: w,
  };

  it(`全${n}サンプルでビット列完全一致(${golden.out_channels}ch)`, () => {
    for (let i = 0; i < n; i++) {
      const row = new Float32Array(golden.x[i].length);
      for (let j = 0; j < row.length; j++) row[j] = Math.fround(golden.x[i][j]);
      const bits = encodeImageLowLevel(row, params);
      const expected = b64ToBits(golden.bits_b64[i], golden.out_width);
      expect(Array.from(bits), `sample ${i}`).toEqual(Array.from(expected));
    }
  });

  it('出力チャネル数が Python の n_output_channels と一致', () => {
    const bits = encodeImageLowLevel(
      new Float32Array(golden.x[0].map((v) => Math.fround(v))),
      params,
    );
    expect(bits.length).toBe(golden.out_width);
    expect(golden.out_width / (h * w)).toBe(golden.out_channels);
  });
});
