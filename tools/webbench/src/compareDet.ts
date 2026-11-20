// 検出デモの 3 ランタイム実装(DLGN / ONNX Runtime Web / TensorFlow.js)。
// 入力=モデル解像度 96×96 の RGBA、出力=DetBoxPx[](ピクセル座標)。
// カラーMulti-MNIST(CIFAR-10背景+色付き数字)。
// DLGN(CenterNetLUT、配線学習採用)は thermometer 二値化(RGB 3ch×3閾値=9ch)、
// ONNX/TF.js teacher(CenterNetCNN)は学習時と同一の生RGB 3ch([0,1]正規化のみ、
// 二値化なし)を消費する(seg の SegOnnxRuntime/SegTfjsRuntime と同型)。
import * as ort from 'onnxruntime-web';

import {
  TFJS_WASM_PATHS, gzipSize, hasWebGpuApi, resourceBytes, resourceGzipBytes,
} from './compare';
import type { CompareRuntime, LoadInfo, RunInfo } from './compare';
import { nchwToNhwc, nhwcToNchwPlane } from './compareLayout';
import { dlgnEngineOf, tfjsModelSize } from './compareSeg';
import type { DlgnBackend, OnnxBackend, TfjsBackend } from './compareSeg';
import { rgbaToNchw } from './convpreproc';
import { decodeDetections } from './detdecode';
import type { DetBoxPx } from './detdecode';
import {
  encodeDetInput, loadDetModel, runDetFrameAsync, submitDetFrame, submitDetPrepared,
} from './detdemo';
import type { DetModel } from './detdemo';
import { deriveModelMeta, fmtInt } from './modelMeta';
import { REFERENCES, teacherParamsLabel } from './references';
import type { FrameInput } from './engines/conv';
import type { WebgpuBreakdown } from './engines/webgpuGraph';
import type { DetReadout } from './types';

/** det 共通: 表示解像度とラベルを持つランタイム。 */
export interface DetRuntime extends CompareRuntime<DetBoxPx[]> {
  modelSize: number;
  labels: string[];
}

const S = 96;
const C = 10; // クラス数
// ONNX/TF.js teacher(CenterNetCNN、生RGB)の参照値は references.ts に
// 集約した(ONNXにはメタデータが無いので唯一の集約点。出典 wandb run id つき)。
const COLOR_TEACHER = REFERENCES.det_color_teacher;

/** ONNX/TF.js teacher(CenterNetCNN)の解像度・クラス数・指標(W12: fewclassでも
    再利用できるよう定数からconstructor引数へ切り出した設定)。省略時はカラー
    Multi-MNISTの既定値(det_main.ts/detbackends_main.tsの既存挙動を維持)。*/
export interface DetTeacherCfg {
  size: number;
  nClasses: number;
  /** 参照値(references.ts)。持たない比較対象は null(UIは「—」表示)。 */
  teacherMap: number | null;
  paramsLabel: string;
}
const DEFAULT_TEACHER_CFG: DetTeacherCfg = {
  size: S,
  nClasses: C,
  teacherMap: COLOR_TEACHER.metricValue,
  paramsLabel: teacherParamsLabel(COLOR_TEACHER),
};

/** references.ts のエントリから DetTeacherCfg を作る(ページ側の重複定義を無くす)。 */
export function teacherCfgFromRef(
  id: keyof typeof REFERENCES | string,
  size: number,
  nClasses: number,
): DetTeacherCfg {
  const r = REFERENCES[id];
  if (!r) throw new Error(`unknown reference: ${id}`);
  return {
    size,
    nClasses,
    teacherMap: r.metricValue,
    paramsLabel: teacherParamsLabel(r),
  };
}

/** ORT/TF.js の hm/size/off(各 [1,C,H/4,W/4] 相当)を decode 用連結列メタに。 */
function ortReadout(cfg: DetTeacherCfg): DetReadout {
  const hw4 = cfg.size / 4;
  const hw = hw4 * hw4;
  const nc = cfg.nClasses;
  return {
    kind: 'detection',
    heads: {
      hm: { offset: 0, n_out: nc, h: hw4, w: hw4, scale: 1 },
      size: { offset: nc * hw, n_out: 2, h: hw4, w: hw4, scale: 1 },
      off: { offset: (nc + 2) * hw, n_out: 2, h: hw4, w: hw4, scale: 1 },
    },
    output_stride: 4,
    top_k: 100,
    threshold: 0.1,
    n_classes: nc,
  };
}

/** DLGN(ConvRuntime)ランタイム。backend='js'(既定, packed純JS)/
    'wasm'(packed-wasm)/'webgpu'(packed-webgpu)。backend='webgpu'指定時に
    WebGPUが使えない場合は'js'(packed純JS)へ自動フォールバックする
    (W11、compareSeg.ts::SegDlgnRuntime と同じパターン)。 */
export class DetDlgnRuntime implements DetRuntime {
  // dispose() は実装しない: compareSeg.ts::SegDlgnRuntime と同じ理由
  // (wasmヒープは単純バンプアロケータで個別解放不可)。CompareRuntime.dispose?
  // は任意なので未実装のまま(webcamの排他切替でも「解放不要」として扱われる)。
  readonly name: string;
  modelSize = S;
  labels: string[] = [];
  private model: DetModel | null = null;
  /** [計測専用] prepareBenchInput() で用意した1フレームぶんの入力
      (GPU前処理時はRGBAをそのまま包んだもの、CPU前処理時は符号化済みビット列)。 */
  private benchBits: FrameInput | null = null;
  /** load() 後に確定する実際の実行バックエンド(フォールバック後の値)。 */
  actualBackend: DlgnBackend;
  constructor(private url: string, private backend: DlgnBackend = 'js') {
    // 以前は "k64" を決め打ちしていたが、この runtime は det 系4ページ
    // (det/detmono/detreal/detface)で共有され、実際の k は 64/256/640 と異なる。
    // 表示できない値を騙るより出さない(実規模はゲート数表示で分かる)。
    this.name = `DLGN(論理ゲート回路、${backend})`;
    this.actualBackend = backend;
  }

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    const r = await fetch(this.url);
    const buf = await r.arrayBuffer();
    this.actualBackend = this.backend;
    let backendNote = '';
    if (this.backend === 'webgpu' && !hasWebGpuApi()) {
      this.actualBackend = 'js';
      backendNote = 'WebGPU が使えない環境のため CPU(JavaScript)に切り替えました';
    }
    try {
      this.model = await loadDetModel(this.url, dlgnEngineOf(this.actualBackend));
    } catch (e) {
      if (this.actualBackend !== 'webgpu') throw e;
      backendNote = `WebGPU の初期化に失敗したため CPU(JavaScript)に切り替えました: ${String(e)}`;
      this.actualBackend = 'js';
      this.model = await loadDetModel(this.url, dlgnEngineOf(this.actualBackend));
    }
    this.modelSize = this.model.w;
    this.labels = this.model.labels;
    const warm = new Uint8ClampedArray(this.model.w * this.model.h * 4);
    await runDetFrameAsync(this.model, warm);
    const loadMs = performance.now() - t0;
    const doc = this.model.doc;
    const ev = doc.eval as { best_hard_map?: number };
    return {
      modelBytes: buf.byteLength,
      gzipBytes: await gzipSize(new Uint8Array(buf)),
      runtimeBytes: this.actualBackend === 'wasm' ? resourceBytes('release') : null,
      runtimeGzipBytes: this.actualBackend === 'wasm'
        ? await resourceGzipBytes('release') : null,
      loadMs,
      paramsLabel: `論理ゲート ${fmtInt(deriveModelMeta(doc).gatesReduced)} 個`,
      // 空間展開後のLUT総数(total_luts)ではなく**簡約後**を出す。
      // 展開後の木は一度も実体化されず、実行されるのは命令列だけなので、
      // total_luts を出すと実態の数十倍の数を掲げることになる。
      // どの前処理経路で走っているかを必ず可視化する(黙って
      // 従来のCPU前処理へ落ちていると e2e 計測が数ms変わるため)。
      // gpu = WGSLカーネル / cpu-wasm = 融合前処理(WASM) /
      // cpu-js = 融合前処理(JS) / cpu-encode = 従来の encodeImageAuto+packPlanes。
      note: `入力は学習時と同じ 0/1 変換 / 計算方法: ${this.actualBackend}`
        + ` / 前処理: ${this.model?.runtime.preprocMode ?? 'cpu-encode'}`
        + (this.model?.runtime.gpuPreprocNote ? `(${this.model.runtime.gpuPreprocNote})` : '')
        + (this.model?.runtime.cpuPreprocNote ? `(${this.model.runtime.cpuPreprocNote})` : '')
        + (backendNote ? ` (${backendNote})` : ''),
      metricLabel: 'mAP',
      metricValue: ev.best_hard_map ?? null,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<DetBoxPx[]>> {
    if (!this.model) throw new Error('not loaded');
    const res = await runDetFrameAsync(this.model, rgba);
    return { output: res.boxes, inferMs: res.inferMs };
  }

  /** フレームパイプライン(2フレームin-flight)対応か。
      bitplane-webgpu(=backend 'webgpu' が実際に使えた場合)のみ true。 */
  supportsPipeline(): boolean {
    return this.model?.runtime.supportsPipeline ?? false;
  }

  /** run() のパイプライン版。前フレームの結果を待たずに呼べる
      (呼び出し規約は WebgpuBitplaneRuntime.submitFrame を参照)。 */
  submitFrame(rgba: Uint8ClampedArray): Promise<RunInfo<DetBoxPx[]>> {
    if (!this.model) throw new Error('not loaded');
    return submitDetFrame(this.model, rgba)
      .then((res) => ({ output: res.boxes, inferMs: res.inferMs }));
  }

  async flushPipeline(): Promise<void> {
    await this.model?.runtime.flushPipeline();
  }

  /** [計測専用] 前処理をループ外へ追い出すための入力準備。*/
  prepareBenchInput(rgba: Uint8ClampedArray): void {
    if (!this.model) throw new Error('not loaded');
    this.benchBits = encodeDetInput(this.model, rgba);
  }

  /** [計測専用] 準備済み入力で1フレーム発行(区間は run() の inferMs と同一)。*/
  submitPreparedFrame(): Promise<{ inferMs: number }> {
    if (!this.model || !this.benchBits) throw new Error('prepareBenchInput() が未実行');
    return submitDetPrepared(this.model, this.benchBits);
  }

  /** webgpuバックエンドのフレーム時間内訳計測(autobench.tsから
      オプトインで呼ばれる、他バックエンドでは model.runtime側がno-op)。 */
  setWebgpuProfiling(enabled: boolean): void {
    this.model?.runtime.setWebgpuProfiling(enabled);
  }

  resetWebgpuProfilingStats(): void {
    this.model?.runtime.resetWebgpuProfilingStats();
  }

  getWebgpuBreakdown(): WebgpuBreakdown | null {
    return this.model?.runtime.getWebgpuBreakdown() ?? null;
  }
}

/** ラベル(det モデル doc から取得。ONNX/TF.js が共有)。teacher は
    生RGBを直接消費するため thermometer パラメータは不要になった。 */
export interface DetThermoCtx {
  labels: string[];
}

/** ONNX Runtime Web ランタイム。CenterNetCNN teacher(色版): 生RGB
    [1,3,96,96]([0,1]正規化のみ、二値化なし。学習時 use_rgb=True と同一契約)。
    backend='webgpu' 時は ['webgpu','wasm'] 優先度リストでセッションを作る
    (詳細は compareSeg.ts の SegOnnxRuntime コメント参照)。 */
export class DetOnnxRuntime implements DetRuntime {
  readonly name: string;
  modelSize: number;
  labels: string[] = [];
  private session: ort.InferenceSession | null = null;
  private ro: DetReadout;
  constructor(
    private url: string,
    private ctx: () => DetThermoCtx,
    private backend: OnnxBackend = 'wasm',
    private cfg: DetTeacherCfg = DEFAULT_TEACHER_CFG,
  ) {
    this.name = `比較用のCNN(ONNX Runtime Web、${backend})`;
    this.modelSize = cfg.size;
    this.ro = ortReadout(cfg);
  }

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    // 公平比較のためONNX wasmもマルチスレッド化(従来は1固定=ONNX側の
    // ハンデだった)。COOP/COEP設定済み(crossOriginIsolated)なのでthreads可。
    ort.env.wasm.numThreads = Math.max(
      1, Math.min(8, (navigator.hardwareConcurrency ?? 4) - 1));
    let backendNote = '';
    if (this.backend === 'webgpu' && !hasWebGpuApi()) {
      backendNote = 'WebGPU が使えない環境のため WebAssembly で実行します';
      this.session = await ort.InferenceSession.create(this.url, {
        executionProviders: ['wasm'],
      });
    } else if (this.backend === 'webgpu') {
      try {
        this.session = await ort.InferenceSession.create(this.url, {
          executionProviders: ['webgpu', 'wasm'],
        });
        backendNote = 'WebGPU 優先(未対応の演算は ONNX Runtime が自動で '
          + 'WebAssembly に切り替えます)';
      } catch (e) {
        backendNote = `WebGPU で初期化できなかったため WebAssembly に切り替えました: ${String(e)}`;
        this.session = await ort.InferenceSession.create(this.url, {
          executionProviders: ['wasm'],
        });
      }
    } else {
      this.session = await ort.InferenceSession.create(this.url, {
        executionProviders: ['wasm'],
      });
    }
    this.labels = this.ctx().labels;
    const s = this.cfg.size;
    const warm = new Uint8ClampedArray(s * s * 4);
    await this.run(warm);
    const loadMs = performance.now() - t0;
    const r = await fetch(this.url);
    const buf = await r.arrayBuffer();
    return {
      modelBytes: buf.byteLength,
      gzipBytes: await gzipSize(new Uint8Array(buf)),
      runtimeBytes: resourceBytes('ort-wasm'),
      runtimeGzipBytes: await resourceGzipBytes('ort-wasm'),
      loadMs,
      paramsLabel: this.cfg.paramsLabel,
      note: `入力は学習時と同じ RGB のまま(0/1 変換なし)${backendNote ? ` / ${backendNote}` : ''}`,
      metricLabel: 'mAP',
      metricValue: this.cfg.teacherMap,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<DetBoxPx[]>> {
    if (!this.session) throw new Error('not loaded');
    const s = this.cfg.size;
    const nchw = rgbaToNchw(rgba, s, s); // [0,1]正規化RGB、二値化なし
    const t0 = performance.now();
    const tensor = new ort.Tensor('float32', nchw, [1, 3, s, s]);
    const out = await this.session.run({ input: tensor });
    const scores = concatHeadsNchw(
      out.hm.data as Float32Array,
      out.size.data as Float32Array,
      out.off.data as Float32Array,
      this.cfg,
    );
    return { output: decodeDetections(scores, this.ro), inferMs: performance.now() - t0 };
  }

  /** webcamの排他切替用。ONNX Runtime Web のセッションを明示的に解放する。*/
  async dispose(): Promise<void> {
    await this.session?.release();
    this.session = null;
  }
}

/** TensorFlow.js ランタイム。onnx2tf graph model は NHWC。
    生RGB(二値化なし)を NHWC[1,96,96,3] に、出力 hm/size/off(NHWC)を NCHW 連結へ。
    backend の意味は compareSeg.ts の SegTfjsRuntime と同じ。 */
export class DetTfjsRuntime implements DetRuntime {
  readonly name: string;
  modelSize: number;
  labels: string[] = [];
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private model: import('@tensorflow/tfjs').GraphModel | null = null;
  private ro: DetReadout;
  constructor(
    private dir: string,
    private ctx: () => DetThermoCtx,
    private backend: TfjsBackend = 'webgl',
    private cfg: DetTeacherCfg = DEFAULT_TEACHER_CFG,
  ) {
    this.name = `比較用のCNN(TensorFlow.js、${backend})`;
    this.modelSize = cfg.size;
    this.ro = ortReadout(cfg);
  }

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    if (this.backend === 'webgpu' && !hasWebGpuApi()) {
      throw new Error('WebGPU が使えない環境です');
    }
    const tf = await import('@tensorflow/tfjs');
    this.tf = tf;
    if (this.backend === 'wasm') {
      const wasmBackend = await import('@tensorflow/tfjs-backend-wasm');
      wasmBackend.setWasmPaths(TFJS_WASM_PATHS);
    } else if (this.backend === 'webgpu') {
      await import('@tensorflow/tfjs-backend-webgpu');
    }
    const ok = await tf.setBackend(this.backend);
    await tf.ready();
    let backendNote = '';
    if (!ok || tf.getBackend() !== this.backend) {
      backendNote = `${this.backend} が使えないため ${tf.getBackend()} で実行しています`;
    }
    const modelUrl = `${this.dir}/model.json`;
    const head = await fetch(modelUrl, { method: 'GET' });
    if (!head.ok) throw new Error(`TF.js モデル未配置(${head.status}): ${modelUrl}`);
    this.model = await tf.loadGraphModel(modelUrl);
    this.labels = this.ctx().labels;
    const s = this.cfg.size;
    const warm = new Uint8ClampedArray(s * s * 4);
    await this.run(warm);
    const loadMs = performance.now() - t0;
    const size = await tfjsModelSize(this.dir);
    return {
      modelBytes: size.bytes,
      gzipBytes: size.gzip,
      runtimeBytes: this.backend === 'wasm' ? resourceBytes('tfjs-backend-wasm') : null,
      runtimeGzipBytes: this.backend === 'wasm'
        ? await resourceGzipBytes('tfjs-backend-wasm') : null,
      loadMs,
      paramsLabel: this.cfg.paramsLabel,
      note: `入力は学習時と同じ RGB のまま(0/1 変換なし)${backendNote ? ` / ${backendNote}` : ''}`,
      metricLabel: 'mAP',
      metricValue: this.cfg.teacherMap,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<DetBoxPx[]>> {
    if (!this.tf || !this.model) throw new Error('not loaded');
    const tf = this.tf;
    const s = this.cfg.size;
    const nchw = rgbaToNchw(rgba, s, s); // [0,1]正規化RGB、二値化なし
    // NCHW feat[c*hw + p] → NHWC feat[p*3 + c]
    const hwc = nchwToNhwc(nchw, s, s, 3);
    const t0 = performance.now();
    const input = tf.tensor4d(hwc, [1, s, s, 3]);
    const outMap = this.model.execute(input) as
      | import('@tensorflow/tfjs').Tensor
      | import('@tensorflow/tfjs').Tensor[]
      | import('@tensorflow/tfjs').NamedTensorMap;
    const planes = await detHeadsFromTfjs(outMap, this.cfg);
    input.dispose();
    disposeTfOut(outMap);
    return { output: decodeDetections(planes, this.ro), inferMs: performance.now() - t0 };
  }

  /** webcamの排他切替用。TF.js の GraphModel を明示的に解放する。*/
  dispose(): void {
    this.model?.dispose();
    this.model = null;
  }
}

/** hm/size/off(各 NCHW Float32)を decode 用の連結 float64 列へ。 */
function concatHeadsNchw(
  hm: Float32Array,
  sz: Float32Array,
  off: Float32Array,
  cfg: DetTeacherCfg,
): Float64Array {
  const hw4 = cfg.size / 4;
  const hw = hw4 * hw4;
  const nc = cfg.nClasses;
  const scores = new Float64Array((nc + 4) * hw);
  scores.set(hm, 0);
  scores.set(sz, nc * hw);
  scores.set(off, (nc + 2) * hw);
  return scores;
}

/** TF.js 出力(NHWC の hm/size/off)を identify し、NCHW 連結 float64 列にする。
    3 本の出力を「チャネル数」で hm(cfg.nClasses)/size,off(2)を弁別し、size/off は
    ネットワークのヘッド順(size→off)を tensor 名の辞書順で保つ。 */
async function detHeadsFromTfjs(
  outMap:
    | import('@tensorflow/tfjs').Tensor
    | import('@tensorflow/tfjs').Tensor[]
    | import('@tensorflow/tfjs').NamedTensorMap,
  cfg: DetTeacherCfg,
): Promise<Float64Array> {
  type T = import('@tensorflow/tfjs').Tensor;
  let tensors: { name: string; t: T }[];
  if (Array.isArray(outMap)) {
    tensors = outMap.map((t, i) => ({ name: String(i), t }));
  } else if ((outMap as T).shape !== undefined) {
    tensors = [{ name: '0', t: outMap as T }];
  } else {
    tensors = Object.entries(outMap as Record<string, T>).map(([name, t]) => ({ name, t }));
  }
  // 各出力を NHWC 前提で読み、NCHW 平面へ並べ替え
  const byName = new Map<string, { chans: number; plane: Float64Array }>();
  for (const { name, t } of tensors) {
    const data = (await t.data()) as Float32Array;
    const [, H = 0, W = 0, Cx = 0] = t.shape; // [1,H,W,Cx]
    byName.set(name, { chans: Cx, plane: nhwcToNchwPlane(data, H, W, Cx) });
  }
  const nc = cfg.nClasses;
  const hw4 = cfg.size / 4;
  const hw = hw4 * hw4;
  const scores = new Float64Array((nc + 4) * hw);
  // hm = チャネル数 nc、size/off = チャネル数 2。size/off の順は名前昇順。
  const twoChan = [...byName.entries()]
    .filter(([, v]) => v.chans === 2)
    .sort((a, b) => a[0].localeCompare(b[0]));
  const hmEntry = [...byName.values()].find((v) => v.chans === nc);
  if (hmEntry) scores.set(hmEntry.plane, 0);
  if (twoChan[0]) scores.set(twoChan[0][1].plane, nc * hw);
  if (twoChan[1]) scores.set(twoChan[1][1].plane, (nc + 2) * hw);
  return scores;
}

function disposeTfOut(
  outMap:
    | import('@tensorflow/tfjs').Tensor
    | import('@tensorflow/tfjs').Tensor[]
    | import('@tensorflow/tfjs').NamedTensorMap,
): void {
  type T = import('@tensorflow/tfjs').Tensor;
  if (Array.isArray(outMap)) outMap.forEach((t) => t.dispose());
  else if ((outMap as T).dispose) (outMap as T).dispose();
  else Object.values(outMap as Record<string, T>).forEach((t) => t.dispose());
}
