// 行動認識(UCI-HAR)のテストデータ読み込み。
//
// スマートフォンの加速度・ジャイロ 9ch を 50Hz で 2.56 秒(=128点)切り出した窓が
// 1サンプル。窓は 50% 重なりで連続しているので、順に送るとセンサの実ストリーム
// 再生に相当する。scripts/export_webbench.py が int16 に量子化して書き出す。
import { decodeI16 } from './preproc';
import type { TestData, TestMeta } from './types';
import { asset } from './basePath';

export async function fetchHarTestData(
  base = asset('/data'), stem = 'har_test',
): Promise<TestData> {
  const [meta, buf] = await Promise.all([
    fetch(`${base}/${stem}_meta.json`).then((r) => {
      if (!r.ok) throw new Error(`${stem}_meta.json: ${r.status}`);
      return r.json() as Promise<TestMeta>;
    }),
    fetch(`${base}/${stem}.bin`).then((r) => {
      if (!r.ok) throw new Error(`${stem}.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  return decodeI16(buf, meta);
}

/** 1窓ぶんの波形を描く(9ch を色分けした折れ線)。 */
export function drawWindow(
  canvas: HTMLCanvasElement, data: TestData, sample: number,
): void {
  const ctx = canvas.getContext('2d')!;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const c = data.meta.shape[1];
  const l = data.meta.shape[2];
  const row = data.x.subarray(sample * data.chLen, (sample + 1) * data.chLen);
  // 加速度(体)3ch / 角速度3ch / 加速度(重力込み)3ch を系統ごとに色分けする。
  const colors = [
    '#e6194b', '#f58231', '#ffb000',
    '#3cb44b', '#20b2aa', '#46f0f0',
    '#4363d8', '#911eb4', '#f032e6',
  ];
  ctx.lineWidth = 1;
  for (let ch = 0; ch < c; ch++) {
    ctx.beginPath();
    ctx.strokeStyle = colors[ch % colors.length];
    for (let t = 0; t < l; t++) {
      const x = (t / (l - 1)) * width;
      // 体動成分は ±0.3g 程度なので、見やすいように大きめに拡大する。
      const y = height / 2 - row[ch * l + t] * (height / 3);
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
