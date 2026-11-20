// conv対応DLGN実行系(wasm/webgpu対応)。ネイティブconv IR を
// 受け取り実行する。エンジン4種(正しさは全て conv_golden.json /
// seg_e2e_golden.json との整数完全一致で固定):
//   - 'packed'(既定): conv2d を展開せず evalConv2dPacked で重み共有
//     カーネルスイープ(空間位置を32ビット横パック)。lower時間ほぼゼロ・省メモリ。
//   - 'irflat'(テスト用ベースライン): conv2d/or_pool2d を lut_layer へ全展開して
//     irflat で実行(未最適化interp)。packed の整数一致アンカー。
//   - 'packed-wasm': 'packed'と同一の幾何(タップ分解+木のping-pong
//     評価)を AssemblyScript製 WASM カーネル(wasmConv.ts)で実行する。
//     scores()は同期(WASM呼び出し自体は同期。モジュール初期化のみ非同期
//     なので ConvRuntime.create() で先に済ませる)。
//   - 'packed-webgpu'(グラフ常駐化済み): 同じ幾何を WGSL compute
//     shader で実行する。GPU読み戻し(mapAsync)が本質的に非同期なため
//     scores()は使えず scoresAsync()のみ対応。W10より前は conv2d ノードごとに
//     dispatch+mapAsync読み戻しを行っていた(1フレーム5〜6往復)が、
//     webgpuGraph.ts::WebgpuGraphRuntime が IR 全体(conv2d/upsample2d/
//     permute/lut_layer/group_sum/concat/add/binarize)を1つの
//     commandEncoder/1回queue.submitへ載せ、最終出力のみ1回読み戻す
//     「グラフ常駐」実行に置き換えた。
import { lowerToPrimitives } from './lowerConv';
import { evalIrFlat } from './irflat';
import type { ConvNodeEval } from './irflat';
import { evalIrBitplane } from './bitplaneGraph';
import type { BitplaneConvEval, BitplaneNativeOps } from './bitplaneGraph';
import { WasmBitplaneOps } from './wasmBitplaneOps';
import { WasmLutFlatOps } from './wasmLutFlat';
import { JsPreprocPacker, buildCpuPreprocPlan, planMatchesInputWidth } from './cpuPreproc';
import type { CpuPreprocPlan } from './cpuPreproc';
import { WasmPreprocOps } from './wasmPreproc';
import {
  WasmThreadPool, defaultThreadCount, instantiateSharedWasm, wasmThreadsSupported,
} from './wasmThreadPool';
import type { MtWasmBundle } from './wasmThreadPool';
import { BitplaneProgramConvKernel, convTreeSpecOf, packPlanes } from './bitplane';
import type { BitplaneValue } from './bitplane';
import { resolveConvProgram } from './bundleProgram';
import type { BundleProgramDict, ProgramProvenance } from './bundleProgram';
import type { ConvProgram } from './convProgram';
import { instantiateWasmModule, WasmConvKernel } from './wasmConv';
import type { WasmModule } from './wasmConv';
import { WasmBitplaneConvKernel } from './wasmBitplaneConv';
import { requestWebgpuDevice } from './webgpuConv';
import { WebgpuGraphRuntime } from './webgpuGraph';
import type { WebgpuBreakdown } from './webgpuGraph';
import { WebgpuBitplaneRuntime } from './webgpuBitplane';
import { wasmSimdSupported } from './wasmFeature';
import type { Conv2dParams, IRGraphJson } from '../shared';
import type { ImagePreprocDoc } from '../convpreproc';

// 'bitplane': 'packed'と同じ幾何(タップ分解+木のping-pong評価)だが、
// ノード間の値表現を行頭ワード境界のビットプレーン(bitplane.ts::
// BitplaneValue)へ変え、conv2d/or_pool2d(2x2)/upsample2d(x2)/permute
// (チャネルのみ)をパック↔アンパックの往復無しで連結する
// (bitplaneGraph.ts::evalIrBitplane)。それ以外のノード型は境界で
// アンパックしirflat.tsへ委譲する。
// 'bitplane-wasm': 'bitplane'と同一のグラフ走査
// (bitplaneGraph.ts::evalIrBitplane)を使い、conv2dノードの計算だけを
// AssemblyScript製WASMカーネル(wasmBitplaneConv.ts)へ差し替える。
// 'bitplane-webgpu': 'packed-webgpu'(webgpuGraph.ts、1bit=1個のi32・
// 1スレッド=1出力ビット)を、CPU側bitplaneと同じワードパック表現
// ([C][H][wpr]、1スレッド=1出力ワード=32空間位置)へ置き換えたWebGPU executor
// (webgpuBitplane.ts)。さらに全ノードのdispatchを単一のcompute passへまとめ、
// 配線ノード(入力pack+先頭chanperm、permute+group_sum、concat)を隣接カーネルへ
// 融合してパス数を削る。扱えないノード構成では構築時にthrowするので、
// 呼び出し側は 'packed-webgpu' へフォールバックすること。
export type ConvEngine =
  | 'packed' | 'irflat' | 'packed-wasm' | 'packed-webgpu' | 'bitplane' | 'bitplane-wasm'
  | 'bitplane-webgpu';

/** 1フレームぶんの入力。GPU前処理が使えるなら 'rgba'(モデル解像度の
    ImageData.data をそのままGPUへ)、使えないなら 'bits'(CPUで符号化済みの
    0/1バイト列)。ConvRuntime.prepareFrame() で作る。 */
export type FrameInput =
  | { kind: 'rgba'; rgba: Uint8ClampedArray }
  | { kind: 'bits'; bits: Uint8Array | Int32Array };

export interface ConvRuntimeOptions {
  engine?: ConvEngine;
  /** conv2d の重み共有木を lossless に命令列化(convProgram.ts)して
      実行するか('bitplane' / 'bitplane-wasm' のみ有効、既定 true)。
      false で従来の「全ゲートをバランス木として評価する」旧経路。
      整数出力は完全不変なので、性能比較・切り分け以外で false にする理由は無い
      (デモ/ベンチのURLクエリ ?prog=0 からもここへ落ちてくる)。 */
  convProgram?: boolean;
  /** conv2d の命令列を直線JSコードへ生成(convProgramCodegen.ts)して
      実行するか('bitplane' のみ有効、既定 false = 命令列インタプリタ)。
      整数出力は完全不変。CSP(unsafe-eval禁止)環境では `new Function` が
      throw するのでカーネル単位で自動的にインタプリタへ戻る
      (理由は convCodegenNote に残す)。URLクエリ `?codegen=1` からも来る。
      wasm/webgpu エンジンは対象外(既にAOTコンパイル済みのコードを持つ)。 */
  convCodegen?: boolean;
  /** codegen の1関数あたり最大生成ステートメント数
      (既定 DEFAULT_CHUNK=256)。URLクエリ `?cgchunk=N`。感度確認・掃引用。 */
  convCodegenChunk?: number;
  /** 配布バンドルの `programs` 節({node_id: BundleProgramDict})。
      与えられると conv2d ノードごとに、ブラウザ側で buildConvProgram を
      再実行せず、Python側(deploy/programs.py)で焼き込み済みの命令列を
      デシリアライズ+構造検証+verifyConvProgram(全数真理値表照合)して
      採用する(bundleProgram.ts::resolveConvProgram)。'bitplane' /
      'bitplane-wasm' のみ有効(packed/irflat/webgpu系エンジンは無視する)。
      未指定/該当ノード無し/検証失敗のいずれでもそのノードだけ
      buildConvProgram へフォールバックする(黙殺しない: bundleProgramNote に
      理由を残す)。URLクエリ `?prog=build` を付けると、このオプションが
      与えられていても常にブラウザ側で再構築する(A/B比較・切り分け用。
      `?prog=0` は既存どおり convProgram 自体を丸ごと無効化する)。 */
  programs?: Record<string, BundleProgramDict> | null;
  /** packed-wasm/bitplane-wasm: テスト等で事前インスタンス化したWASM
      モジュールを注入する(未指定時はブラウザで ../../build/release.wasm
      を fetch する)。 */
  wasmModule?: WasmModule;
  /** packed-webgpu: 事前取得したGPUDeviceを注入する(未指定時は
      navigator.gpu から自動取得する)。 */
  gpuDevice?: GPUDevice;
  /** doc.preproc(画像前処理の契約)。'bitplane-webgpu' のとき、
      RGBA→ビット符号化をGPUで行う前処理カーネルを組むのに使う。
      未指定/未対応kindならCPU前処理(convpreproc.ts)のまま動く。 */
  preproc?: ImagePreprocDoc | null;
  /** GPU前処理を使うか(既定はURLクエリ `?gpupre=0` でオフ、他はオン)。*/
  gpuPreproc?: boolean;
  /** CPU側の融合前処理(RGBA→ビットプレーンを1パス、cpuPreproc.ts /
      assembly の preprocPackAsm)を使うか。既定はURLクエリ `?cpupre=0` で
      オフ、他はオン。オフにすると従来どおり convpreproc.ts::encodeImageAuto で
      0/1バイト列を作ってから packPlanes する経路になる(出力ビットは同一)。 */
  cpuPreproc?: boolean;
  /** 'bitplane-wasm' のスレッド数(1 = 従来どおり単スレッド)。
      未指定時は URLクエリ `?mt=N`、それも無ければ min(4, ceil(cores/2))
      (ORT/MediaPipe と同じ保守既定)。エンジン名は増やさず、この
      オプションだけで単/マルチを切り替える。crossOriginIsolated でない
      環境では要求値によらず自動的に単スレッドへ落ちる。 */
  threads?: number;
}

/** conv2d命令列化の既定値。ConvRuntimeOptions.convProgram が未指定の
    ときはURLクエリ `?prog=0` で旧経路(全ゲート評価)へ落とせるようにする
    (デモ/ベンチページを個別に改造せず A/B 比較できるようにするため。
    autobench.ts の `?gpuprofile=0` と同じ流儀)。ブラウザ外(vitest/node)では
    常に true。 */
function convProgramDefaultFromUrl(): boolean {
  try {
    if (typeof window === 'undefined') return true;
    return new URL(window.location.href).searchParams.get('prog') !== '0';
  } catch {
    return true;
  }
}

/** `?prog=build` が付いていれば ConvRuntimeOptions.programs を無視し、
    常にブラウザ側で buildConvProgram を実行する(バンドル命令列とその場
    再構築のA/B比較・切り分け用。`?prog=0` は既存どおり convProgram 自体を
    丸ごと無効化する側なので、ここでは 'build' のみを見る)。 */
function bundleProgramsAllowedFromUrl(): boolean {
  try {
    if (typeof window === 'undefined') return true;
    return new URL(window.location.href).searchParams.get('prog') !== 'build';
  } catch {
    return true;
  }
}

/** conv2d codegen の既定値。URLクエリ `?codegen=1` でのみ有効になる
    オプトイン(CSP unsafe-eval禁止環境では使えないため既定は従来経路)。
    `?cgchunk=N` で1関数あたりの命令数を上書きできる(未指定なら DEFAULT_CHUNK)。
    ブラウザ外(vitest/node)では常に無効(テストは明示指定で有効化する)。 */
function urlFlag(name: string): string | null {
  try {
    if (typeof window === 'undefined') return null;
    return new URL(window.location.href).searchParams.get(name);
  } catch {
    return null;
  }
}

/** 'bitplane-wasm' のスレッド数の既定値。URLクエリ `?mt=N`
    (`?mt=0` / `?mt=1` で単スレッド)で上書きでき、未指定なら
    defaultThreadCount()=min(4, ceil(cores/2))。ブラウザ外(vitest/node)は
    Worker/SAB が無いので常に1(=既存の単スレッド経路と完全に同一)。 */
function threadsDefaultFromUrl(): number {
  try {
    if (typeof window === 'undefined') return 1;
    const raw = new URL(window.location.href).searchParams.get('mt');
    if (raw === null) return defaultThreadCount();
    const n = Number(raw);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 1;
  } catch {
    return 1;
  }
}

/** GPU前処理の既定値。`?gpupre=0` でCPU前処理へ戻せる(A/B比較用)。
    ブラウザ外(vitest/node)では常に true(WebGPU自体が無いので無害)。 */
function gpuPreprocDefaultFromUrl(): boolean {
  try {
    if (typeof window === 'undefined') return true;
    return new URL(window.location.href).searchParams.get('gpupre') !== '0';
  } catch {
    return true;
  }
}

/** CPU融合前処理の既定値。`?cpupre=0` で従来経路
    (encodeImageAuto + packPlanes)へ戻せる(A/B比較・切り分け用)。
    ブラウザ外(vitest/node)では常に true。 */
function cpuPreprocDefaultFromUrl(): boolean {
  try {
    if (typeof window === 'undefined') return true;
    return new URL(window.location.href).searchParams.get('cpupre') !== '0';
  } catch {
    return true;
  }
}

const isNativeSpatialEngine = (e: ConvEngine): boolean => e !== 'irflat';
/** or_pool2d(2x2/pad0)を lut_layer へ展開せずネイティブのまま残すエンジン群。
    bitplane-webgpu も orPool2x2 相当のWGSLカーネルを持つのでここに含める。 */
const isBitplaneEngine = (e: ConvEngine): boolean =>
  e === 'bitplane' || e === 'bitplane-wasm' || e === 'bitplane-webgpu';
const isWebgpuEngine = (e: ConvEngine): boolean =>
  e === 'packed-webgpu' || e === 'bitplane-webgpu';

export class ConvRuntime {
  readonly ir: IRGraphJson; // 元(ネイティブ)IR
  /** packed系: conv2dは残置 / irflat: lut_layerのみ。bitplane-webgpuが未対応
      ノード構成でpacked-webgpuへ落ちた場合のみ、初期化中に作り直される
      (or_pool2dのネイティブ残置↔lut_layer展開の違いを吸収するため)。 */
  lowered: IRGraphJson;
  readonly inWidth: number;
  readonly engine: ConvEngine;
  /** bitplane-webgpu → packed-webgpu へフォールバックした理由(黙殺しない)。 */
  webgpuFallbackNote: string | null = null;

  private wasmKernels: Map<string, WasmConvKernel> | null = null;
  private bitplaneWasmKernels: Map<string, WasmBitplaneConvKernel> | null = null;
  private bitplaneProgKernels: Map<string, BitplaneProgramConvKernel> | null = null;
  private webgpuGraph: WebgpuGraphRuntime | null = null;
  private webgpuBitplane: WebgpuBitplaneRuntime | null = null;
  /** 命令列化の構築/セルフチェックに失敗して旧経路へ落ちた理由。*/
  convProgramFallbackNote: string | null = null;
  private readonly useConvProgram: boolean;
  /** 配布バンドルの programs 節(`?prog=build` のときは常に null に
      落とし、opts.programsが与えられていてもブラウザ側で強制再構築する)。 */
  private readonly programs: Record<string, BundleProgramDict> | null;
  /** conv2dノードごとの命令列の出所('bundle'=バンドル採用/
      'built'=ブラウザ側で再構築)。convProgram無効時・conv2dノード無しなら
      null。 */
  programProvenance: Record<string, ProgramProvenance> | null = null;
  /** バンドル命令列を採用できず再構築した最初の理由(黙殺しない。
      全ノードとも採用できていれば null)。 */
  bundleProgramNote: string | null = null;
  /** 直線JSコード生成を要求したか。*/
  readonly useConvCodegen: boolean;
  private readonly convCodegenChunk: number | undefined;
  /** codegenを要求したのに使えなかった理由(CSP等)。使えていればnull。*/
  convCodegenNote: string | null = null;
  /** 生成コードの総文字数/コンパイルms(ロード時コストの実測用)。*/
  convCodegenStats: { sourceChars: number; compileMs: number; nChunks: number } | null = null;
  /** GPU前処理に使う doc.preproc(null ならCPU前処理のみ)。*/
  private readonly preproc: ImagePreprocDoc | null;
  /** GPU前処理を組めなかった/使わない理由(使えているときは null)。*/
  gpuPreprocNote: string | null = null;
  /** 要求スレッド数(?mt=N / opts.threads)。*/
  readonly threads: number;
  /** マルチスレッド化できなかった理由(できていれば null)。*/
  threadsNote: string | null = null;
  private threadPool: WasmThreadPool | null = null;
  /** pack/or_pool の WASM 実行器(bitplane-wasm のみ。null ならJS)。*/
  private bitplaneOps: WasmBitplaneOps | null = null;
  private wasmLutFlat: WasmLutFlatOps | null = null;
  /** CPU融合前処理の実行計画(null なら従来のCPU前処理のまま)。*/
  private cpuPreprocPlan: CpuPreprocPlan | null = null;
  private jsPreproc: JsPreprocPacker | null = null;
  private wasmPreproc: WasmPreprocOps | null = null;
  /** CPU融合前処理を組めなかった/使わない理由(使えていれば null)。*/
  cpuPreprocNote: string | null = null;

  constructor(ir: IRGraphJson, opts: ConvRuntimeOptions = {}) {
    this.ir = ir;
    this.engine = opts.engine ?? 'packed';
    this.useConvProgram = opts.convProgram ?? convProgramDefaultFromUrl();
    this.programs = bundleProgramsAllowedFromUrl() ? opts.programs ?? null : null;
    this.useConvCodegen = opts.convCodegen ?? urlFlag('codegen') === '1';
    const cgChunk = opts.convCodegenChunk ?? Number(urlFlag('cgchunk') ?? NaN);
    this.convCodegenChunk = Number.isFinite(cgChunk) && cgChunk > 0
      ? Math.floor(cgChunk) : undefined;
    this.threads = Math.max(1, Math.floor(opts.threads ?? threadsDefaultFromUrl()));
    const wantGpuPreproc = opts.gpuPreproc ?? gpuPreprocDefaultFromUrl();
    this.preproc = wantGpuPreproc ? opts.preproc ?? null : null;
    if (!wantGpuPreproc) this.gpuPreprocNote = 'GPU前処理は無効(?gpupre=0)';
    else if (!opts.preproc) this.gpuPreprocNote = 'GPU前処理は無効(preproc未指定)';
    this.lowered = lowerToPrimitives(ir, {
      nativeConv: isNativeSpatialEngine(this.engine),
      nativeOrPool: isBitplaneEngine(this.engine),
    });
    this.inWidth = ir.inputs[0].width;
    this.initCpuPreproc(opts);
    if (this.engine === 'bitplane' && this.useConvProgram) this.initBitplaneProgram();
  }

  /** CPU融合前処理の実行計画を組む(bitplane / bitplane-wasm のみ。
      未対応 kind・幾何不一致・`?cpupre=0` のいずれでも「組まない」だけで、
      その場合は呼び出し側が従来のCPU前処理(encodeImageAuto)を通す)。
      黙って別物を食わせないよう、IR入力ポートの幅との整合をここで必ず検査する。 */
  private initCpuPreproc(opts: ConvRuntimeOptions): void {
    if (!isBitplaneEngine(this.engine) || isWebgpuEngine(this.engine)) return;
    const want = opts.cpuPreproc ?? cpuPreprocDefaultFromUrl();
    if (!want) {
      this.cpuPreprocNote = 'CPU融合前処理は無効(?cpupre=0)';
      return;
    }
    if (!opts.preproc) {
      this.cpuPreprocNote = 'CPU融合前処理は無効(preproc未指定)';
      return;
    }
    try {
      const plan = buildCpuPreprocPlan(opts.preproc);
      if (!planMatchesInputWidth(plan, this.inWidth)) {
        throw new Error(
          `幾何不一致(IR入力幅 ${this.inWidth} vs 記述子 `
          + `${plan.channels}x${plan.height}x${plan.width})`,
        );
      }
      this.cpuPreprocPlan = plan;
    } catch (e) {
      this.cpuPreprocNote = `CPU融合前処理を組めず従来経路へ: ${String(e)}`;
    }
  }

  /** 'bitplane'(純JS)エンジン用に、conv2dノードごとの命令列カーネルを
      構築する。構築時の全数真理値表セルフチェック(verifyConvProgram)に
      失敗した場合は黙殺せず記録し、ノード全体で旧経路(evalConv2dBitplane)へ
      フォールバックする。
      ノードごとの命令列は resolveConvProgram(bundleProgram.ts)を通す:
      this.programs にそのノードの辞書があればデシリアライズ+検証して採用し
      (provenance='bundle')、無い/採用できないときだけそのノード単体で
      buildConvProgram を実行する(provenance='built'、黙殺せず
      bundleProgramNote に理由を残す)。 */
  private initBitplaneProgram(): void {
    const kernels = new Map<string, BitplaneProgramConvKernel>();
    const provenance: Record<string, ProgramProvenance> = {};
    try {
      for (const node of this.lowered.nodes) {
        if (node.type === 'conv2d') {
          const params = node.params as unknown as Conv2dParams;
          const spec = convTreeSpecOf(params);
          const resolved = resolveConvProgram(spec, this.programs?.[node.id] ?? null);
          provenance[node.id] = resolved.provenance;
          if (resolved.note && !this.bundleProgramNote) {
            this.bundleProgramNote = resolved.note;
            console.error(`[conv] ${this.bundleProgramNote}`);
          }
          kernels.set(
            node.id,
            new BitplaneProgramConvKernel(params, {
              codegen: this.useConvCodegen,
              chunkSize: this.convCodegenChunk,
              program: resolved.program,
            }),
          );
        }
      }
    } catch (e) {
      this.convProgramFallbackNote =
        `conv2d命令列化に失敗→旧バランス木経路へフォールバック: ${String(e)}`;
      console.error(`[conv] ${this.convProgramFallbackNote}`);
      return;
    }
    this.bitplaneProgKernels = kernels;
    if (Object.keys(provenance).length > 0) this.programProvenance = provenance;
    if (this.useConvCodegen) {
      // ロード時コスト(生成ソース量 + new Function)と、CSP等で
      // 使えなかったカーネルの理由を集計して公開する(黙殺しない)。
      let sourceChars = 0; let compileMs = 0; let nChunks = 0;
      const notes: string[] = [];
      for (const k of kernels.values()) {
        if (k.compiled) {
          sourceChars += k.compiled.stats.sourceChars;
          compileMs += k.compiled.stats.compileMs;
          nChunks += k.compiled.stats.nChunks;
        } else if (k.codegenNote) notes.push(k.codegenNote);
      }
      this.convCodegenStats = { sourceChars, compileMs, nChunks };
      if (notes.length > 0) {
        this.convCodegenNote = notes[0];
        console.warn(`[conv] ${this.convCodegenNote}`);
      }
    }
  }

  /** 非同期初期化が要るエンジン(packed-wasm/packed-webgpu/bitplane-wasm)
      込みの生成口。'packed'/'irflat'/'bitplane' は同期コンストラクタと
      等価(即resolve)。 */
  static async create(ir: IRGraphJson, opts: ConvRuntimeOptions = {}): Promise<ConvRuntime> {
    const rt = new ConvRuntime(ir, opts);
    if (rt.engine === 'packed-wasm') {
      await rt.initWasm(opts.wasmModule);
    } else if (rt.engine === 'bitplane-wasm') {
      await rt.initBitplaneWasm(opts.wasmModule);
    } else if (isWebgpuEngine(rt.engine)) {
      await rt.initWebgpu(opts.gpuDevice);
    }
    return rt;
  }

  private async initWasm(mod?: WasmModule): Promise<void> {
    const wasmMod = mod ?? await loadWasmModuleForBrowser();
    const kernels = new Map<string, WasmConvKernel>();
    for (const node of this.lowered.nodes) {
      if (node.type === 'conv2d') {
        kernels.set(node.id, new WasmConvKernel(wasmMod, node.params as unknown as Conv2dParams));
      }
    }
    this.wasmKernels = kernels;
  }

  /** bitplane-wasm: conv2dノードごとにWasmBitplaneConvKernelを
      構築する(WasmConvKernelのビットプレーン版。同じWASMモジュール
      インスタンスを共有、alloc()のヒープも共通)。 */
  private async initBitplaneWasm(mod?: WasmModule): Promise<void> {
    // threads>1 が要求され、かつ crossOriginIsolated + SAB + SIMD が
    // 揃っていれば共有メモリ版WASM(release-simd-mt.wasm)を読み、Workerプールを
    // 立てる。1つでも欠けたら **単スレッドの従来経路のまま** 何も変えない。
    let wasmMod = mod;
    if (!wasmMod && this.threads > 1 && wasmSimdSupported() && wasmThreadsSupported()) {
      try {
        const bundle: MtWasmBundle = await loadMtWasmBundle();
        wasmMod = bundle.instance.exports as unknown as WasmModule;
        this.threadPool = await sharedThreadPool(bundle, this.threads);
        if (!this.threadPool) {
          this.threadsNote = 'Workerプール生成に失敗→共有メモリ版WASMの単スレッド実行';
        }
      } catch (e) {
        this.threadsNote = `共有メモリ版WASMの読み込みに失敗→通常版の単スレッドへ: ${String(e)}`;
        console.warn(`[conv] ${this.threadsNote}`);
        wasmMod = undefined;
      }
    } else if (this.threads > 1) {
      this.threadsNote = wasmSimdSupported()
        ? 'マルチスレッド非対応環境(crossOriginIsolated / SharedArrayBuffer)→単スレッド'
        : 'WASM SIMD非対応環境→単スレッド';
    }
    wasmMod ??= await loadWasmModuleForBrowser();
    this.bitplaneOps = new WasmBitplaneOps(wasmMod, this.threadPool);
    // lut_flat → group_sum 融合の WASM カーネル(無い .wasm なら JS 版)
    this.wasmLutFlat = new WasmLutFlatOps(wasmMod);
    // 融合前処理をWASMへ載せる(カーネルが無い古い .wasm では
    // JS版の融合前処理のまま。どちらも出力ビットは同一)。
    if (this.cpuPreprocPlan) {
      if (WasmPreprocOps.supported(wasmMod)) {
        try {
          this.wasmPreproc = new WasmPreprocOps(wasmMod, this.cpuPreprocPlan, this.threadPool);
        } catch (e) {
          this.cpuPreprocNote = `融合前処理のWASM初期化に失敗→JS版へ: ${String(e)}`;
          console.warn(`[conv] ${this.cpuPreprocNote}`);
        }
      } else {
        this.cpuPreprocNote = '読み込んだ .wasm に C6 前処理カーネルが無い→JS版の融合前処理';
      }
    }
    if (!this.bitplaneOps.available && !this.threadsNote) {
      this.threadsNote = '読み込んだ .wasm に C3 カーネル(packPlanesAsm/orPool2x2Asm)が無い→JS実装';
    }
    // 計測/デモ表示用に実効値を露出する(Playwright はここを読んで
    // 「mt=N が本当に N スレッドで走ったか」を確認する)。
    if (typeof window !== 'undefined') {
      (window as unknown as { __dlgnMt?: unknown }).__dlgnMt = {
        requested: this.threads,
        actual: this.actualThreads,
        wasmOps: this.bitplaneOps.available,
        note: this.threadsNote,
      };
    }
    // bitplaneと同じくノードごとに resolveConvProgram を通す。
    // useConvProgram=false のときは呼ぶだけ無駄(WasmBitplaneConvKernel側が
    // useProgram=falseで即座に旧経路を選ぶ)なので呼ばない。
    const kernels = new Map<string, WasmBitplaneConvKernel>();
    const provenance: Record<string, ProgramProvenance> = {};
    for (const node of this.lowered.nodes) {
      if (node.type === 'conv2d') {
        const params = node.params as unknown as Conv2dParams;
        let prebuilt: ConvProgram | undefined;
        if (this.useConvProgram) {
          const spec = convTreeSpecOf(params);
          const resolved = resolveConvProgram(spec, this.programs?.[node.id] ?? null);
          prebuilt = resolved.program;
          provenance[node.id] = resolved.provenance;
          if (resolved.note && !this.bundleProgramNote) {
            this.bundleProgramNote = resolved.note;
            console.error(`[conv] ${this.bundleProgramNote}`);
          }
        }
        const k = new WasmBitplaneConvKernel(wasmMod, params, this.useConvProgram, prebuilt);
        if (k.fallbackNote && !this.convProgramFallbackNote) {
          this.convProgramFallbackNote = `${node.id}: ${k.fallbackNote}`;
        }
        kernels.set(node.id, k);
      }
    }
    this.bitplaneWasmKernels = kernels;
    if (Object.keys(provenance).length > 0) this.programProvenance = provenance;
  }

  /** IR全体(this.lowered、nativeConv=trueでconv2d/upsample2d/permuteは
      ネイティブのまま・or_pool2dはlut_layer展開済み)を1つのGPU常駐グラフへ
      構築する(ノード単位カーネルのMapではなく、単一のWebgpuGraphRuntime)。 */
  private async initWebgpu(dev?: GPUDevice): Promise<void> {
    const device = dev ?? await requestWebgpuDevice();
    if (!device) throw new Error('WebGPUデバイスを取得できません(navigator.gpu無しまたはアダプタ無し)');
    if (this.engine === 'bitplane-webgpu') {
      try {
        this.webgpuBitplane = new WebgpuBitplaneRuntime(device, this.lowered, {
          preproc: this.preproc,
        });
        if (this.preproc) {
          this.gpuPreprocNote = this.webgpuBitplane.gpuPreprocNote;
          if (this.gpuPreprocNote) console.warn(`[conv] ${this.gpuPreprocNote}`);
        }
        return;
      } catch (e) {
        // 未対応ノード構成(lut_layer/add/stride>1のconv2d等)は黙殺せず記録し、
        // 従来の packed-webgpu(1bit=1個のi32)へ落とす。webgpuGraph は
        // or_pool2d ネイティブノードを扱えないので lowered を作り直す。
        this.webgpuFallbackNote =
          `bitplane-webgpu構築失敗→packed-webgpuへフォールバック: ${String(e)}`;
        this.gpuPreprocNote = 'GPU前処理は無効(packed-webgpuへフォールバック)';
        console.warn(`[conv] ${this.webgpuFallbackNote}`);
        this.lowered = lowerToPrimitives(this.ir, { nativeConv: true, nativeOrPool: false });
      }
    }
    this.webgpuGraph = new WebgpuGraphRuntime(device, this.lowered);
  }

  /** [batch*inWidth] の0/1入力 → [batch*outWidth] の整数スコア。
      packed-webgpu では使用不可(scoresAsyncを使うこと)。 */
  scores(bits: ArrayLike<number>, batch: number): {
    data: Int32Array;
    width: number;
  } {
    return this.scoresInner(bits, null, batch);
  }

  /** 融合前処理でパック済みのビットプレーンをそのまま実行する
      (bits を渡す経路と同じグラフ走査。入力表現だけが違う)。 */
  private scoresInner(
    bits: ArrayLike<number> | null, packed: BitplaneValue | null, batch: number,
  ): { data: Int32Array; width: number } {
    if (isWebgpuEngine(this.engine)) {
      throw new Error(`${this.engine}エンジンはscores()非対応(scoresAsync()を使用)`);
    }
    const inputId = this.lowered.inputs[0].id;
    if (isBitplaneEngine(this.engine)) {
      let bpConvEval: BitplaneConvEval | undefined;
      if (this.engine === 'bitplane-wasm') {
        bpConvEval = (node, x, b) => this.bitplaneWasmKernels!.get(node.id)!.evalAll(x, b);
      } else if (this.bitplaneProgKernels) {
        bpConvEval = (node, x, b) => this.bitplaneProgKernels!.get(node.id)!.evalAll(x, b);
      }
      // pack/or_pool を WASM(必要ならワーカープール)へ回す差し替え口。
      // ops 自体が無い / 個々の呼びが null を返した場合は JS 実装のまま。
      const ops: BitplaneNativeOps | undefined = this.bitplaneOps
        ? {
          pack: (x, b, C, H, W) => this.bitplaneOps!.packPlanes(x, b, C, H, W),
          orPool: (bp, b) => this.bitplaneOps!.orPool2x2(bp, b),
          lutFlatGroupSum: (p, x, b, k) => this.wasmLutFlat?.groupSum(p, x, b, k) ?? null,
        }
        : undefined;
      const { output } = evalIrBitplane(
        this.lowered, bits ? { [inputId]: bits } : {}, batch, bpConvEval, undefined, ops,
        packed ? { [inputId]: packed } : undefined,
      );
      return output;
    }
    if (!bits) throw new Error('pack済み入力は bitplane 系エンジン専用');
    const convEval: ConvNodeEval | undefined = this.engine === 'packed-wasm'
      ? (node, x, b) => this.wasmKernels!.get(node.id)!.evalAll(x, b)
      : undefined;
    const { output } = evalIrFlat(this.lowered, { [inputId]: bits }, batch, convEval);
    return output;
  }

  /** RGBA(モデル解像度、batch枚ぶんの連結)→ ビットプレーン。
      WASMカーネルがあればそれを、無ければJS版の融合実装を使う
      (どちらも encodeImageAuto + packPlanes とビット完全一致)。 */
  packRgbaCpu(rgba: ArrayLike<number>, batch: number): BitplaneValue {
    const plan = this.cpuPreprocPlan;
    if (!plan) throw new Error('CPU融合前処理が有効でない(preproc未指定/未対応kind)');
    if (this.wasmPreproc) return this.wasmPreproc.pack(rgba, batch);
    this.jsPreproc ??= new JsPreprocPacker(plan);
    return this.jsPreproc.pack(rgba, batch);
  }

  /** scores()の非同期版。全エンジンで使える(packed/irflat/packed-wasmは
      内部的に同期評価をそのままawaitするだけ)。packed-webgpuの唯一の
      正式な実行経路(IR全体をWebgpuGraphRuntimeへ丸ごと渡し、
      グラフ常駐実行させる。evalIrFlatAsyncのノード単位switchは経由しない)。 */
  async scoresAsync(bits: ArrayLike<number>, batch: number): Promise<{
    data: Int32Array;
    width: number;
  }> {
    if (this.webgpuBitplane) {
      // 生ビット列(0/1)をそのまま渡す: パックはGPU側の先頭カーネルが行う
      // (Int32Array.from の2.8M要素コピーもCPU側packPlanesも挟まない)。
      return this.webgpuBitplane.evalAll(bits, batch);
    }
    if (!isWebgpuEngine(this.engine)) {
      return this.scores(bits, batch);
    }
    const data = bits instanceof Int32Array ? bits : Int32Array.from(bits);
    return this.webgpuGraph!.evalAll({ data, width: this.inWidth }, batch);
  }

  /** フレームパイプライン(2フレームin-flight)が使えるか。
      bitplane-webgpu で走っているときだけ true(それ以外は逐次のみ)。 */
  get supportsPipeline(): boolean {
    return this.webgpuBitplane !== null;
  }

  /** RGBAをそのまま渡せるか(=画像前処理をGPUで行えるか)。
      false のときは呼び出し側がCPU前処理(convpreproc.ts)を通すこと。 */
  get supportsGpuPreproc(): boolean {
    return this.webgpuBitplane?.supportsGpuPreproc ?? false;
  }

  /** CPU側で融合前処理(RGBA→ビットプレーンを1パス)が使えるか。*/
  get supportsCpuPreproc(): boolean {
    return this.cpuPreprocPlan !== null;
  }

  /** 実際に使われている前処理経路(表示・計測メモ用)。
      'cpu-encode' は従来の encodeImageAuto + packPlanes。 */
  get preprocMode(): 'gpu' | 'cpu-wasm' | 'cpu-js' | 'cpu-encode' {
    if (this.supportsGpuPreproc) return 'gpu';
    if (!this.cpuPreprocPlan) return 'cpu-encode';
    return this.wasmPreproc ? 'cpu-wasm' : 'cpu-js';
  }

  /** 1フレームぶんの入力を作る。融合前処理(GPU または 
      CPU融合)が使えるならRGBAをそのまま、使えないなら渡された encoder で
      CPU符号化したビット列を包む。
      **前処理経路の分岐をここ1箇所に閉じ込める**(デモ/ベンチ/比較ページが
      それぞれ分岐を書くと、片方だけ融合前処理になって計測がずれる)。 */
  prepareFrame(
    rgba: Uint8ClampedArray, encodeCpu: () => Uint8Array | Int32Array,
  ): FrameInput {
    if (this.supportsGpuPreproc || this.supportsCpuPreproc) return { kind: 'rgba', rgba };
    return { kind: 'bits', bits: encodeCpu() };
  }

  /** scores() のフレーム入力版(同期。webgpu では使えない)。*/
  scoresFrame(input: FrameInput, batch: number): { data: Int32Array; width: number } {
    if (input.kind === 'bits') return this.scores(input.bits, batch);
    if (!this.cpuPreprocPlan) throw new Error('RGBA直投入は融合前処理が有効なときだけ');
    return this.scoresInner(null, this.packRgbaCpu(input.rgba, batch), batch);
  }

  /** scoresAsync のフレーム入力版(RGBA直投入に対応)。*/
  async scoresFrameAsync(input: FrameInput, batch: number): Promise<{
    data: Int32Array;
    width: number;
  }> {
    if (input.kind === 'bits') return this.scoresAsync(input.bits, batch);
    if (this.webgpuBitplane) return this.webgpuBitplane.evalAllRgba(input.rgba, batch);
    return this.scoresFrame(input, batch);
  }

  /** submitScoresAsync のフレーム入力版。*/
  async submitScoresFrameAsync(input: FrameInput, batch: number): Promise<{
    data: Int32Array;
    width: number;
  }> {
    if (input.kind === 'bits') return this.submitScoresAsync(input.bits, batch);
    if (this.webgpuBitplane) return this.webgpuBitplane.submitFrameRgba(input.rgba, batch);
    return this.scoresFrame(input, batch);
  }

  /** [検証用] CPU側の「入力packまで」を実行して packed planes を返す。
      - kind='rgba': 融合前処理(WASM/JS)の出力そのもの
      - kind='bits': 従来経路(encodeImageAuto の出力)を packPlanes したもの
      両者が全ワード一致することが融合前処理の契約(wasmgpu_verify.html)。
      GPU側の debugPackedPlanes と違い**先頭permuteは融合していない**
      (エンコーダのチャネル順そのまま)ので、CPU経路同士でのみ比較すること。 */
  debugCpuPackedPlanes(input: FrameInput, batch: number): Uint32Array {
    const plan = this.cpuPreprocPlan;
    if (!plan) throw new Error('debugCpuPackedPlanes はCPU融合前処理が有効なときだけ');
    const bp = input.kind === 'rgba'
      ? this.packRgbaCpu(input.rgba, batch)
      : packPlanes(
        { data: input.bits, width: this.inWidth }, batch,
        plan.channels, plan.height, plan.width,
      );
    return new Uint32Array(bp.words.buffer, bp.words.byteOffset, bp.words.length);
  }

  /** [検証用] 入力packまでを実行して packed planes を読み戻す
      (CPU前処理経路とGPU前処理経路のビット完全一致を直接確認する)。 */
  async debugPackedPlanes(input: FrameInput, batch: number): Promise<Uint32Array> {
    if (!this.webgpuBitplane) throw new Error('debugPackedPlanes は bitplane-webgpu 専用');
    return this.webgpuBitplane.debugPackedPlanes(
      input.kind === 'rgba'
        ? { kind: 'rgba', rgba: input.rgba }
        : { kind: 'bits', bits: input.bits },
      batch,
    );
  }

  /** scoresAsync のパイプライン版。前フレームの結果を待たずに
      writeBuffer+encode+submit まで進める(submit→mapAsyncの往復をGPU実行に
      重ねてスループット側で隠す)。pipeline非対応エンジンでは scoresAsync に
      素通しするので、呼び出し側は分岐なしで使える(その場合は逐次と同じ)。
      返り値は逐次モードと完全に同一の整数スコア。 */
  async submitScoresAsync(bits: ArrayLike<number>, batch: number): Promise<{
    data: Int32Array;
    width: number;
  }> {
    if (this.webgpuBitplane) return this.webgpuBitplane.submitFrame(bits, batch);
    return this.scoresAsync(bits, batch);
  }

  /** 発行済みフレームを全て回収する(計測終了時/デモ停止時)。*/
  async flushPipeline(): Promise<void> {
    await this.webgpuBitplane?.flushPipeline();
  }

  /** webgpuGraph.tsのフレーム時間内訳計測をオプトインで有効化する
      (packed-webgpu以外では何もしない no-op)。 */
  setWebgpuProfiling(enabled: boolean): void {
    this.webgpuGraph?.setProfiling(enabled);
    this.webgpuBitplane?.setProfiling(enabled);
  }

  /** 蓄積した内訳統計をクリアする(ウォームアップ後・本計測開始前に
      呼ぶことを想定)。 */
  resetWebgpuProfilingStats(): void {
    this.webgpuGraph?.resetProfilingStats();
    this.webgpuBitplane?.resetProfilingStats();
  }

  /** 蓄積した内訳統計のスナップショットを返す(packed-webgpu以外や
      未計測時はnull)。 */
  getWebgpuBreakdown(): WebgpuBreakdown | null {
    return this.webgpuGraph?.getBreakdown() ?? this.webgpuBitplane?.getBreakdown() ?? null;
  }

  /** bitplane-webgpu の単一compute pass内 dispatch 数(融合効果の確認用。
      他エンジンでは null)。 */
  get webgpuDispatchCount(): number | null {
    return this.webgpuBitplane?.dispatchCount ?? null;
  }

  /** 実際に走っているスレッド数(Workerプールが立たなければ1)。*/
  get actualThreads(): number {
    return this.bitplaneOps?.threads ?? 1;
  }


  /** seg per-pixel argmax。scores [batch*(nOut*H*W)] を [batch*(H*W)] のラベルへ。
      group_sumの並びは (n_out, H, W) の NCHW(SpatialGroupSum lowering規約)。 */
  argmaxPerPixel(
    scores: Int32Array,
    batch: number,
    nOut: number,
    hw: number,
  ): Int32Array {
    const out = new Int32Array(batch * hw);
    const outW = nOut * hw;
    for (let b = 0; b < batch; b++) {
      for (let px = 0; px < hw; px++) {
        let best = 0;
        let bestv = scores[b * outW + px];
        for (let c = 1; c < nOut; c++) {
          const v = scores[b * outW + c * hw + px];
          if (v > bestv) {
            bestv = v;
            best = c;
          }
        }
        out[b * hw + px] = best;
      }
    }
    return out;
  }
}

let browserWasmModulePromise: Promise<WasmModule> | null = null;
let mtWasmBundlePromise: Promise<MtWasmBundle> | null = null;

/** 共有メモリ版WASM(build/release-simd-mt.wasm)を1回だけ読み、
    コンパイル済み Module と共有 Memory を持つバンドルとして返す。
    Module/Memory はそのまま Worker へ postMessage できる(再fetch不要)。
    ABI は release-simd.wasm と完全同一なので、プールが立たなくても
    このインスタンス単体で単スレッド実行できる。 */
async function loadMtWasmBundle(): Promise<MtWasmBundle> {
  if (!mtWasmBundlePromise) {
    mtWasmBundlePromise = (async () => {
      const mod = await import('../../build/release-simd-mt.wasm?url');
      const res = await fetch(mod.default);
      return instantiateSharedWasm(await res.arrayBuffer());
    })();
  }
  return mtWasmBundlePromise;
}

const sharedPools = new Map<number, Promise<WasmThreadPool | null>>();

/** Workerプールは **ページ全体で1組**(スレッド数ごとに1つ)にする。
    比較ページ(detbackends / wasmgpu_verify / mtverify)は1ページで5本以上の
    ConvRuntime を作るため、ランタイムごとにプールを立てると Worker が
    コア数の何倍にも増えて互いに食い合う。共有メモリと同じく、プールも
    プロセス単位の資源として扱う(ジョブは同期バリアで直列化されるので、
    複数ランタイムが同じプールを使っても競合しない: scores() は同期実行で、
    バリアが完了するまで次のジョブは発行されない)。 */
function sharedThreadPool(
  bundle: MtWasmBundle, threads: number,
): Promise<WasmThreadPool | null> {
  let p = sharedPools.get(threads);
  if (!p) {
    p = WasmThreadPool.create(bundle, threads);
    sharedPools.set(threads, p);
  }
  return p;
}

async function fetchAndInstantiate(url: string): Promise<WasmModule> {
  const res = await fetch(url);
  const bytes = await res.arrayBuffer();
  return instantiateWasmModule(bytes);
}

/** ブラウザ実行時に build/release.wasm(scalar)または
    build/release-simd.wasm(v128)を fetch してインスタンス化する
    (テストは ConvRuntimeOptions.wasmModule で node:fs 読み込み分を注入する
    ため、この関数はブラウザ経路でのみ呼ばれる)。WasmModuleのABIは両者で
    完全同一(alloc/evalConv2dBatch/evalConv2dBitplaneBatch)なので、
    呼び出し側(WasmConvKernel/WasmBitplaneConvKernel)はどちらが返っても
    無改造で動く。SIMD対応環境ではsimd版を優先し、fetch/instantiateが
    何らかの理由で失敗した場合(黙殺しない: console.warnで理由を残す)は
    scalar版へフォールバックする。SIMD非対応環境(Safari旧版等)では最初から
    scalar版のみを試す。 */
async function loadWasmModuleForBrowser(): Promise<WasmModule> {
  if (!browserWasmModulePromise) {
    browserWasmModulePromise = (async () => {
      // vite `?url` はビルド/dev両方で正しい配信URLに解決される。
      if (wasmSimdSupported()) {
        try {
          const simdMod = await import('../../build/release-simd.wasm?url');
          return await fetchAndInstantiate(simdMod.default);
        } catch (err) {
          console.warn(
            '[wasm] release-simd.wasm の読み込みに失敗、scalar版(release.wasm)へフォールバックします:',
            err,
          );
        }
      }
      const scalarMod = await import('../../build/release.wasm?url');
      return fetchAndInstantiate(scalarMod.default);
    })();
  }
  return browserWasmModulePromise;
}
