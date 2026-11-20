// seg デモの 3 ランタイム実装(DLGN / ONNX Runtime Web / TensorFlow.js)。
// いずれも入力=モデル解像度 64×64 の RGBA、出力=per-pixel クラスラベル
// Int32Array[outH*outW]。geometry(outH/outW/inW/inH)は load 後に確定する。
import * as ort from 'onnxruntime-web';

import {
  TFJS_WASM_PATHS, gzipSize, hasWebGpuApi, resourceBytes, resourceGzipBytes,
} from './compare';
import type { CompareRuntime, LoadInfo, RunInfo } from './compare';
import { argmaxSpatial } from './compareLayout';
import { rgbaToNchw } from './convpreproc';
import type { ConvEngine, FrameInput } from './engines/conv';
import type { WebgpuBreakdown } from './engines/webgpuGraph';
import {
  encodeSegInput, loadSegModel, runSegFrameAsync, submitSegFrame, submitSegPrepared,
} from './segdemo';
import type { SegModel } from './segdemo';
import { deriveModelMeta, fmtInt } from './modelMeta';
import { REFERENCES, teacherParamsLabel } from './references';

/** DLGN(ConvRuntime)のバックエンド軸('js'/'wasm'の実体を
    bitplane系エンジンへ差し替え)。'js'=CPU実行(既定。旧版はpacked、
    以降はbitplane=行頭ワード境界ビットプレーンでノード間を貫通させる実行系。
    UI上の表示値/選択肢は不変)、'wasm'=AssemblyScript製WASMカーネル
    (旧版はpacked-wasm、以降はbitplane-wasm=同じビットプレーン
    レイアウトのconv2dカーネルをWASM化、タップキャッシュのoc横断dedupは
    構築時にTS側で1回だけ計算しWASMへ渡す。SIMD(v128)化は計画書のPhase2
    (2a)相当だが、このスカラー版で既にk256=10.8ms/k640=22.4ms
    (packed-wasm比17.7倍/20.9倍、ONNX wasm実測82.9ms比7.7倍)と目標を
    大きく超過したため、追加のビルド系統(release-simd二重管理・機能検出)
    のリスクに見合わないと判断し今回は見送った。将来の課題として
    docs/plan側に引き継ぐこと)、'webgpu'=WGSL compute shader。いずれも
    conv_golden/seg_e2e_golden との整数完全一致を取った実行系
    (conv.test.ts/seg_e2e.test.ts/detdecode.test.tsのENGINES行列に
    'bitplane'/'bitplane-wasm'として含まれる、webgpuはBrowser paneでの
    確認)。 */
export type DlgnBackend = 'js' | 'wasm' | 'webgpu';

/** 'webgpu' の実体を bitplane-webgpu(ワードパック+単一compute pass)
    へ切り替えた。未対応ノード構成では ConvRuntime.initWebgpu 内で従来の
    packed-webgpu へ自動フォールバックする(理由は webgpuFallbackNote に残る)。 */
export function dlgnEngineOf(backend: DlgnBackend): ConvEngine {
  if (backend === 'wasm') return 'bitplane-wasm';
  if (backend === 'webgpu') return 'bitplane-webgpu';
  return 'bitplane';
}

/** ONNX Runtime Web の実行プロバイダ軸。webgpu は
    ['webgpu','wasm'] の優先度リストで作成し、op非対応時はORT側がノード単位で
    wasmに自動フォールバックする(公開APIではノード単位の割当を検査できない
    既知の制約があるため、フォールバックそのものはnotesに一般的注記として残す)。 */
export type OnnxBackend = 'wasm' | 'webgpu';
/** TF.js のバックエンド軸。webgl=既定(側で暗黙登録済み)、
    wasm/webgpu は動的importで登録+切替する。 */
export type TfjsBackend = 'webgl' | 'wasm' | 'webgpu';

/** seg 共通: 出力解像度を持つランタイム。 */
export interface SegRuntime extends CompareRuntime<Int32Array> {
  outH: number;
  outW: number;
  inW: number;
  inH: number;
}

const S = 64; // seg 入力=出力解像度(SmallUNet / SegConvDLGN 共通)
// ONNX/TF.js teacher(SmallUNet)の参照値は references.ts に集約
// (ONNXにはメタデータが無いため。出典は同ファイルのコメント)。
const UNET_TEACHER = REFERENCES.seg_unet_teacher;

/** DLGN(ConvRuntime)ランタイム。backend='js'(既定, packed純JS)/
    'wasm'(packed-wasm)/'webgpu'(packed-webgpu)。backend='webgpu'指定時に
    WebGPUが使えない(navigator.gpu無し/アダプタ取得失敗/グラフ構築失敗)場合は
    'js'(packed純JS)へ自動フォールバックする(W11、ONNX/TF.jsランタイムと同じ
    graceful degradeパターン)。実際に使われたバックエンドは actualBackend /
    LoadInfo.note で確認できる(nameはUI構築時に確定した「要求」バックエンドの
    ままなので、フォールバックの有無はnoteを見ること)。 */
export class SegDlgnRuntime implements SegRuntime {
  // dispose() は実装しない: wasm(AssemblyScript)ヒープは単純バンプアロケータ
  // で個別解放不可(モデルごとに確保した領域を狙って返す仕組みが無い)、かつ
  // ConvRuntime側に明示的な破棄経路が無い。CompareRuntime.dispose?は任意なので
  // 未実装のまま(webcamの排他切替でも「解放不要」として扱われる)。
  readonly name: string;
  outH = S;
  outW = S;
  inW = S;
  inH = S;
  private model: SegModel | null = null;
  /** [計測専用] prepareBenchInput() で用意した1フレームぶんの入力
      (GPU前処理時はRGBAをそのまま包んだもの、CPU前処理時は符号化済みビット列)。 */
  private benchBits: FrameInput | null = null;
  /** load() 後に確定する実際の実行バックエンド(フォールバック後の値)。 */
  actualBackend: DlgnBackend;
  constructor(private url: string, private backend: DlgnBackend = 'js') {
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
      this.model = await loadSegModel(this.url, dlgnEngineOf(this.actualBackend)); // 2 度目は HTTP キャッシュ
    } catch (e) {
      if (this.actualBackend !== 'webgpu') throw e;
      backendNote = `WebGPU の初期化に失敗したため CPU(JavaScript)に切り替えました: ${String(e)}`;
      this.actualBackend = 'js';
      this.model = await loadSegModel(this.url, dlgnEngineOf(this.actualBackend));
    }
    this.outH = this.model.outH;
    this.outW = this.model.outW;
    this.inW = this.model.w;
    this.inH = this.model.h;
    // ウォームアップ(lower 後の初回実行を load に計上)
    const warm = new Uint8ClampedArray(this.inW * this.inH * 4);
    await runSegFrameAsync(this.model, warm);
    const loadMs = performance.now() - t0;
    const doc = this.model.doc;
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
      metricLabel: 'mIoU',
      metricValue: (doc.eval as { best_hard_miou?: number }).best_hard_miou ?? null,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<Int32Array>> {
    if (!this.model) throw new Error('not loaded');
    const res = await runSegFrameAsync(this.model, rgba);
    return { output: res.labels, inferMs: res.inferMs };
  }

  /** フレームパイプライン(2フレームin-flight)対応か。*/
  supportsPipeline(): boolean {
    return this.model?.runtime.supportsPipeline ?? false;
  }

  /** run() のパイプライン版(呼び出し規約は
      WebgpuBitplaneRuntime.submitFrame を参照)。 */
  submitFrame(rgba: Uint8ClampedArray): Promise<RunInfo<Int32Array>> {
    if (!this.model) throw new Error('not loaded');
    return submitSegFrame(this.model, rgba)
      .then((res) => ({ output: res.labels, inferMs: res.inferMs }));
  }

  async flushPipeline(): Promise<void> {
    await this.model?.runtime.flushPipeline();
  }

  /** [計測専用] 前処理をループ外へ追い出すための入力準備。*/
  prepareBenchInput(rgba: Uint8ClampedArray): void {
    if (!this.model) throw new Error('not loaded');
    this.benchBits = encodeSegInput(this.model, rgba);
  }

  /** [計測専用] 準備済み入力で1フレーム発行(区間は run() の inferMs と同一)。*/
  submitPreparedFrame(): Promise<{ inferMs: number }> {
    if (!this.model || !this.benchBits) throw new Error('prepareBenchInput() が未実行');
    return submitSegPrepared(this.model, this.benchBits);
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

/** ONNX Runtime Web ランタイム。SmallUNet: 生 float[0,1] NCHW 入力。
    backend='webgpu' 時は ['webgpu','wasm'] 優先度リストでセッションを作る。
    navigator.gpu が無い環境では最初から wasm のみで作成しnotesに記録する
    (webgpu 非対応環境の graceful degrade)。 */
export class SegOnnxRuntime implements SegRuntime {
  readonly name: string;
  outH = S;
  outW = S;
  inW = S;
  inH = S;
  private session: ort.InferenceSession | null = null;
  constructor(private url: string, private backend: OnnxBackend = 'wasm') {
    this.name = `比較用のCNN(ONNX Runtime Web、${backend})`;
  }

  async load(): Promise<LoadInfo> {
    const t0 = performance.now();
    ort.env.wasm.numThreads = 1;
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
    // ウォームアップ
    const warm = new Uint8ClampedArray(S * S * 4);
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
      paramsLabel: teacherParamsLabel(UNET_TEACHER),
      note: `入力は学習時と同じ RGB のまま(0/1 変換なし)${backendNote ? ` / ${backendNote}` : ''}`,
      metricLabel: 'mIoU',
      metricValue: UNET_TEACHER.metricValue,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<Int32Array>> {
    if (!this.session) throw new Error('not loaded');
    const nchw = rgbaToNchw(rgba, S, S);
    const t0 = performance.now();
    const tensor = new ort.Tensor('float32', Float32Array.from(nchw), [1, 3, S, S]);
    const out = await this.session.run({ input: tensor });
    const logits = out.logits.data as Float32Array; // [1,2,H,W]
    const hw = S * S;
    const labels = new Int32Array(hw);
    for (let px = 0; px < hw; px++) labels[px] = logits[hw + px] > logits[px] ? 1 : 0;
    return { output: labels, inferMs: performance.now() - t0 };
  }

  /** webcamの排他切替用。ONNX Runtime Web のセッションを明示的に解放する。*/
  async dispose(): Promise<void> {
    await this.session?.release();
    this.session = null;
  }
}

/** TensorFlow.js ランタイム。onnx2tf 変換の graph model は NHWC。
    生 float[0,1] NHWC[1,H,W,3] 入力、出力 logits(NHWC or NCHW)を argmax。
    backend: 'webgl'(既定, tfjs-backend-webgl は @tensorflow/tfjs に同梱)/
    'wasm'(tfjs-backend-wasm を動的import+セルフホストwasmパス)/
    'webgpu'(tfjs-backend-webgpu を動的import。navigator.gpu 無しは
    ロード時エラーにしてUI側でgraceful degradeさせる)。 */
export class SegTfjsRuntime implements SegRuntime {
  readonly name: string;
  outH = S;
  outW = S;
  inW = S;
  inH = S;
  // 動的 import(未変換でも seg ページ全体は壊さない)
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private model: import('@tensorflow/tfjs').GraphModel | null = null;
  constructor(private dir: string, private backend: TfjsBackend = 'webgl') {
    this.name = `比較用のCNN(TensorFlow.js、${backend})`;
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
    // 存在確認(未変換なら明示エラー→UI が graceful degrade)
    const head = await fetch(modelUrl, { method: 'GET' });
    if (!head.ok) throw new Error(`TF.js モデル未配置(${head.status}): ${modelUrl}`);
    this.model = await tf.loadGraphModel(modelUrl);
    const warm = new Uint8ClampedArray(S * S * 4);
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
      paramsLabel: teacherParamsLabel(UNET_TEACHER),
      note: `入力は学習時と同じ RGB のまま(0/1 変換なし)${backendNote ? ` / ${backendNote}` : ''}`,
      metricLabel: 'mIoU',
      metricValue: UNET_TEACHER.metricValue,
    };
  }

  async run(rgba: Uint8ClampedArray): Promise<RunInfo<Int32Array>> {
    if (!this.tf || !this.model) throw new Error('not loaded');
    const tf = this.tf;
    const hw = S * S;
    // NHWC 生 float[0,1]
    const hwc = new Float32Array(hw * 3);
    for (let i = 0; i < hw; i++) {
      hwc[i * 3] = rgba[i * 4] / 255;
      hwc[i * 3 + 1] = rgba[i * 4 + 1] / 255;
      hwc[i * 3 + 2] = rgba[i * 4 + 2] / 255;
    }
    const t0 = performance.now();
    const input = tf.tensor4d(hwc, [1, S, S, 3]);
    const outT = this.model.execute(input) as import('@tensorflow/tfjs').Tensor;
    const data = (await outT.data()) as Float32Array;
    const shape = outT.shape;
    input.dispose();
    outT.dispose();
    const labels = argmaxSpatial(data, shape);
    return { output: labels, inferMs: performance.now() - t0 };
  }

  /** webcamの排他切替用。TF.js の GraphModel を明示的に解放する。*/
  dispose(): void {
    this.model?.dispose();
    this.model = null;
  }
}

/** tfjs graph model のサイズ(model.json + weight bins)を合算し gzip 実測も返す。 */
export async function tfjsModelSize(
  dir: string,
): Promise<{ bytes: number | null; gzip: number | null }> {
  try {
    const manifestUrl = `${dir}/model.json`;
    const mr = await fetch(manifestUrl);
    const mbuf = new Uint8Array(await mr.arrayBuffer());
    const manifest = JSON.parse(new TextDecoder().decode(mbuf)) as {
      weightsManifest?: { paths: string[] }[];
    };
    const parts: Uint8Array[] = [mbuf];
    let total = mbuf.byteLength;
    for (const group of manifest.weightsManifest ?? []) {
      for (const path of group.paths) {
        const wr = await fetch(`${dir}/${path}`);
        const wbuf = new Uint8Array(await wr.arrayBuffer());
        parts.push(wbuf);
        total += wbuf.byteLength;
      }
    }
    // 連結して gzip
    const cat = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { cat.set(p, off); off += p.byteLength; }
    return { bytes: total, gzip: await gzipSize(cat) };
  } catch {
    return { bytes: null, gzip: null };
  }
}
