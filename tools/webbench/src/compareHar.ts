// 行動認識デモのランタイム実装(DLGN / ONNX Runtime Web)。
//
// 入力は画像ではなく「センサ窓1つ」なので、他ページの CompareRuntime とは
// run() の引数だけが違う(load() が返す LoadInfo は共通で、指標カードは同じ形)。
import * as ort from 'onnxruntime-web';

import { gzipSize, hasWebGpuApi, resourceBytes, resourceGzipBytes } from './compare';
import type { LoadInfo, RunInfo } from './compare';
import type { OnnxBackend } from './compareSeg';
import { Packed32Engine } from './engines/packed';
import { argmax } from './preproc';
import { deriveModelMeta, fmtInt } from './modelMeta';
import { REFERENCES, teacherParamsLabel } from './references';
import type { ModelDoc, TestData } from './types';

/** ONNX Runtime Web のセッションは同一ページ内で並行実行できない
 *  (2セッションを同時に走らせると "Session mismatch" / "Session already started"
 *  で落ちる)。比較用の MLP と 1D CNN を同じ画面に並べるため、ORT の呼び出しだけ
 *  直列化する。DLGN 側は別実装なので待たされない。 */
let ortQueue: Promise<unknown> = Promise.resolve();
function withOrtLock<T>(fn: () => Promise<T>): Promise<T> {
  const next = ortQueue.then(fn, fn);
  ortQueue = next.catch(() => undefined);
  return next;
}

/** 推論対象の1窓。 */
export interface HarWindow {
  data: TestData;
  index: number;
}

/** 行動認識の共通ランタイム。出力は予測クラス番号。 */
export interface HarRuntime {
  readonly name: string;
  load(): Promise<LoadInfo>;
  run(win: HarWindow): Promise<RunInfo<number>>;
}

/** DLGN(dense)。ビット並列(32窓/ワード)の純JS実行系で1窓ぶんを推論する。
    画像系と違い WebAssembly / WebGPU カーネルは持たないので計算方法の選択はない。 */
export class HarDlgnRuntime implements HarRuntime {
  readonly name = 'DLGN(論理ゲート回路)';
  private engine: Packed32Engine | null = null;
  /** ロード済みモデルJSON(ページ本文の data-meta 描画に使う)。 */
  doc: ModelDoc | null = null;

  constructor(private url: string) {}

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    const r = await fetch(this.url);
    if (!r.ok) throw new Error(`${this.url}: ${r.status}`);
    const buf = await r.arrayBuffer();
    const doc = JSON.parse(new TextDecoder().decode(buf)) as ModelDoc;
    this.doc = doc;
    this.engine = new Packed32Engine(doc);
    const loadMs = performance.now() - t0;
    const ev = doc.eval;
    return {
      modelBytes: buf.byteLength,
      gzipBytes: await gzipSize(new Uint8Array(buf)),
      runtimeBytes: null,
      runtimeGzipBytes: null,
      loadMs,
      paramsLabel: `論理ゲート ${fmtInt(deriveModelMeta(doc).gatesReduced)} 個`,
      // 空間展開後のLUT総数(total_luts)ではなく**簡約後**を出す。
      // 展開後の木は一度も実体化されず、実行されるのは命令列だけなので、
      // total_luts を出すと実態の数十倍の数を掲げることになる。
      note: '入力は学習時と同じ 0/1 変換(センサ値をチャネルごとのしきい値で二値化)',
      metricLabel: '正解率',
      // int16 に量子化したテストデータでの実測(ブラウザ配信と同じ条件)。
      metricValue: ev.hard_acc_i16 ?? ev.hard_acc ?? null,
    };
  }

  get labels(): string[] {
    return this.doc?.labels ?? [];
  }

  async run(win: HarWindow): Promise<RunInfo<number>> {
    if (!this.engine) throw new Error('not loaded');
    const t0 = performance.now();
    const pred = await this.engine.predict(win.data, win.index, 1);
    return { output: pred[0], inferMs: performance.now() - t0 };
  }

}

/** ONNX Runtime Web(比較用の MLP / 1D CNN)。前処理は DLGN と同じ
    チャネル別標準化まで(0/1 変換はしない=CNN 側の学習時と同じ入力)。 */
export class HarOnnxRuntime implements HarRuntime {
  readonly name: string;
  constructor(
    private url: string,
    private refId: string,
    label: string,
    private backend: OnnxBackend = 'wasm',
  ) {
    this.name = label;
  }

  private session: ort.InferenceSession | null = null;
  private input: Float32Array | null = null;

  async load(): Promise<LoadInfo> {
    return withOrtLock(() => this.loadInner());
  }

  private async loadInner(): Promise<LoadInfo> {
    const t0 = performance.now();
    ort.env.wasm.numThreads = Math.max(
      1, Math.min(8, (navigator.hardwareConcurrency ?? 4) - 1));
    let backendNote = '';
    if (this.backend === 'webgpu' && !hasWebGpuApi()) {
      backendNote = 'WebGPU が使えない環境のため WebAssembly で実行します';
      this.session = await ort.InferenceSession.create(
        this.url, { executionProviders: ['wasm'] });
    } else if (this.backend === 'webgpu') {
      try {
        this.session = await ort.InferenceSession.create(
          this.url, { executionProviders: ['webgpu', 'wasm'] });
        backendNote = 'WebGPU 優先(未対応の演算は ONNX Runtime が自動で '
          + 'WebAssembly に切り替えます)';
      } catch (e) {
        backendNote = `WebGPU で初期化できなかったため WebAssembly に切り替えました: ${String(e)}`;
        this.session = await ort.InferenceSession.create(
          this.url, { executionProviders: ['wasm'] });
      }
    } else {
      this.session = await ort.InferenceSession.create(
        this.url, { executionProviders: ['wasm'] });
    }
    const loadMs = performance.now() - t0;
    const r = await fetch(this.url);
    const buf = await r.arrayBuffer();
    const ref = REFERENCES[this.refId];
    return {
      modelBytes: buf.byteLength,
      gzipBytes: await gzipSize(new Uint8Array(buf)),
      runtimeBytes: resourceBytes('ort-wasm'),
      runtimeGzipBytes: await resourceGzipBytes('ort-wasm'),
      loadMs,
      paramsLabel: ref ? teacherParamsLabel(ref) : '—',
      note: `入力は学習時と同じ生のセンサ値(平均・標準偏差で正規化のみ)${
        backendNote ? ` / ${backendNote}` : ''}`,
      metricLabel: ref?.metricLabel ?? '正解率',
      metricValue: ref?.metricValue ?? null,
    };
  }

  async run(win: HarWindow): Promise<RunInfo<number>> {
    if (!this.session) throw new Error('not loaded');
    const { data, index } = win;
    const c = data.meta.mean.length;
    const l = data.chLen / c;
    if (!this.input || this.input.length !== data.chLen) {
      this.input = new Float32Array(data.chLen);
    }
    const input = this.input;
    const row = data.x.subarray(index * data.chLen, (index + 1) * data.chLen);
    for (let ch = 0; ch < c; ch++) {
      const m = data.meta.mean[ch];
      const invStd = 1 / data.meta.std[ch];
      for (let t = 0; t < l; t++) input[ch * l + t] = (row[ch * l + t] - m) * invStd;
    }
    return withOrtLock(async () => {
      const t0 = performance.now();
      const out = await this.session!.run({
        input: new ort.Tensor('float32', input, [1, c, l]),
      });
      const logits = out.logits.data as Float32Array;
      return { output: argmax(logits), inferMs: performance.now() - t0 };
    });
  }
}
