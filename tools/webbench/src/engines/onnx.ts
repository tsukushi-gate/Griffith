// ONNX Runtime Web(WASM EP)エンジン(WB-6)。
// onnxruntime-webの既定exportはbundle版(JSローダ内蔵)で、.wasm実体は
// import.meta.url基準で自己解決される(dev/build両対応)。CDNは使わない
// (セルフホスト、ペイロードはresource timingの'ort-wasm'で計測)。
// 前処理はDLGN側と同一のチャネル別標準化(標準化のみ、thermometerなし)。
import * as ort from 'onnxruntime-web';

import type { TestData } from '../types';
import { argmax } from '../preproc';
import type { Engine } from './types';

const MAX_BATCH = 256; // 1回のsession.runに投げる最大サンプル数

export class OnnxEngine implements Engine {
  readonly name: string;
  private url: string;
  private threads: number;
  private session: ort.InferenceSession | null = null;

  constructor(name: string, modelUrl: string, threads = 1) {
    this.name = name;
    this.url = modelUrl;
    this.threads = threads;
  }

  async init(): Promise<void> {
    ort.env.wasm.numThreads = this.threads;
    this.session = await ort.InferenceSession.create(this.url, {
      executionProviders: ['wasm'],
    });
  }

  /** 実効スレッド数(crossOriginIsolated非対応時はortが1にフォールバック) */
  effectiveThreads(): number {
    return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
      ? this.threads
      : 1;
  }

  async predict(data: TestData, from: number, count: number): Promise<Int32Array> {
    if (!this.session) await this.init();
    const session = this.session!;
    const c = data.meta.mean.length;
    const l = data.chLen / c;
    const mean = data.meta.mean;
    const std = data.meta.std;
    const preds = new Int32Array(count);

    for (let b = 0; b < count; b += MAX_BATCH) {
      const n = Math.min(MAX_BATCH, count - b);
      const input = new Float32Array(n * data.chLen);
      for (let i = 0; i < n; i++) {
        const row = data.x.subarray(
          (from + b + i) * data.chLen, (from + b + i + 1) * data.chLen,
        );
        for (let ch = 0; ch < c; ch++) {
          const m = mean[ch];
          const invStd = 1 / std[ch];
          for (let ti = 0; ti < l; ti++) {
            input[i * data.chLen + ch * l + ti] = (row[ch * l + ti] - m) * invStd;
          }
        }
      }
      const tensor = new ort.Tensor('float32', input, [n, c, l]);
      const out = await session.run({ input: tensor });
      const logits = out.logits.data as Float32Array;
      const nClasses = logits.length / n;
      for (let i = 0; i < n; i++) {
        preds[b + i] = argmax(logits.subarray(i * nClasses, (i + 1) * nClasses));
      }
    }
    return preds;
  }
}
