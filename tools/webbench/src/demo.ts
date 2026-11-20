// ストリーミングデモ(WB-7): テスト系列を50Hzで再生し、1.28秒ごとに
// 直近2.56秒窓(=1テストサンプル)をDLGN(codegen)とONNXで並走判定する。
// UCI-HARのテスト窓は50%オーバーラップの連続系列なので、サンプルを順に
// 送るだけで実センサストリームの再生に相当する。
import { fetchModel } from './data';
import { CodegenEngine } from './engines/codegen';
import { OnnxEngine } from './engines/onnx';
import type { TestData } from './types';

interface Manifest { dlgn: string[]; onnx: string[] }

let timer: ReturnType<typeof setInterval> | null = null;

export function stopDemo(): void {
  if (timer !== null) clearInterval(timer);
  timer = null;
}

const $ = (id: string): HTMLElement => document.getElementById(id)!;

function drawWave(
  canvas: HTMLCanvasElement, data: TestData, sample: number,
): void {
  const ctx = canvas.getContext('2d')!;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const c = data.meta.shape[1];
  const l = data.meta.shape[2];
  const row = data.x.subarray(sample * data.chLen, (sample + 1) * data.chLen);
  const colors = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#008080',
  ];
  for (let ch = 0; ch < c; ch++) {
    ctx.beginPath();
    ctx.strokeStyle = colors[ch % colors.length];
    ctx.lineWidth = 1;
    for (let t = 0; t < l; t++) {
      const v = row[ch * l + t];
      const x = (t / (l - 1)) * width;
      const y = height / 2 - v * (height / 8);
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

export async function startDemo(
  manifest: Manifest, data: TestData,
): Promise<void> {
  stopDemo();
  if (manifest.dlgn.length === 0) {
    $('demo-dlgn-label').textContent = 'モデルなし';
    return;
  }
  const model = await fetchModel(`/models/${manifest.dlgn[0]}`);
  const dlgn = new CodegenEngine(model);
  dlgn.warmup(data);
  const onnxFile = manifest.onnx.find((f) => f.includes('fp32')) ?? manifest.onnx[0];
  const onnx = onnxFile
    ? new OnnxEngine('demo-onnx', `/models/${onnxFile}`, 1)
    : null;
  if (onnx) await onnx.init();

  const canvas = $('demo-canvas') as HTMLCanvasElement;
  const labels = model.labels;
  let sample = 0;

  const step = async (): Promise<void> => {
    drawWave(canvas, data, sample);

    const t0 = performance.now();
    const dPred = (await dlgn.predict(data, sample, 1))[0];
    const dMs = performance.now() - t0;
    $('demo-dlgn-label').textContent = labels[dPred] ?? String(dPred);
    $('demo-dlgn-time').textContent = `${(dMs * 1000).toFixed(0)} µs`;

    if (onnx) {
      const t1 = performance.now();
      const oPred = (await onnx.predict(data, sample, 1))[0];
      const oMs = performance.now() - t1;
      $('demo-onnx-label').textContent = labels[oPred] ?? String(oPred);
      $('demo-onnx-time').textContent = `${(oMs * 1000).toFixed(0)} µs`;
      $('demo-onnx-label').className =
        `label ${oPred === data.meta.y_test[sample] ? 'ok' : 'ng'}`;
    }

    const truth = data.meta.y_test[sample];
    $('demo-truth').textContent = labels[truth] ?? String(truth);
    $('demo-pos').textContent = `窓 ${sample + 1}/${data.n}`;
    $('demo-dlgn-label').className =
      `label ${dPred === truth ? 'ok' : 'ng'}`;

    sample = (sample + 1) % data.n;
  };

  await step();
  // 実時間換算: 窓は50%オーバーラップ(64step=1.28秒)ごと
  timer = setInterval(() => void step(), 1280);
}
