// 配布バンドル(magicdelta-bundle)を読み込んで推論するWebランタイムの
// 公開API。`md deploy codegen-web` が配る `runtime/web/dist/mdl_web.js` の実体。
//
// 既存の webbench(tools/webbench/src)は「比較・計測ツール」であり、
// モデルごとに main.ts/segdemo.ts/detdemo.ts が個別に配線を書いていた
// (bundle programs節を採用する経路もConvRuntimeOptionsへ直接渡す形)。
// このファイルはその配線を1つの関数(loadBundle)へ集約し、**バンドルの
// kind(dense/画像分類/seg/検出)を自動判別して適切な実行メソッドを生やす**
// (Track Kの他ターゲット: C/CUDA/Metalランタイムの「バンドル1本を読めば
// 動く」体験をWebでも揃える)。
//
// 正しさの契約: 出力整数は既存の ConvRuntime(engines/conv.ts)と完全に同一
// (このファイルは薄いラッパーであり、evalは一切自前実装しない)。
// strictPrograms=true(検証経路、verifyCore.ts が常にこれを使う)では、
// バンドルの programs 節をロード**前**に全数検証し、1つでも壊れていれば
// フォールバックせず例外を投げる(黙って再構築すると、古い/改竄された
// programsを検出できずに「出荷ゲート」の意味が無くなるため)。
import { ConvRuntime } from '../engines/conv';
import type { ConvEngine } from '../engines/conv';
import { convProgramFromDict } from '../engines/bundleProgram';
import type { BundleProgramDict } from '../engines/bundleProgram';
import { verifyConvProgram } from '../engines/convProgram';
import { convTreeSpecOf } from '../engines/bitplane';
import { instantiateWasmModule } from '../engines/wasmConv';
import type { WasmModule } from '../engines/wasmConv';
import { decodeDetections } from '../detdecode';
import type { DetBoxPx } from '../detdecode';
import { encodeImageAuto, rgbaToNchw } from '../convpreproc';
import type { ImagePreprocDoc } from '../convpreproc';
import { deriveModelMeta } from '../modelMeta';
import type { ModelMeta } from '../modelMeta';
import type { Conv2dParams } from '../shared';
import type { DetReadout, Readout } from '../types';
import { checkBundleDoc } from './bundleDoc';
import type { BundleDoc } from './bundleDoc';
import { decodeMdlb } from './mdlb';

export interface LoadBundleOptions {
  /** 'auto': wasmエンジンの初期化を試み、失敗したら notes に理由を記録して
      'js' へ落ちる(既定)。'js'/'wasm' は明示指定(失敗時は例外)。 */
  engine?: 'auto' | 'js' | 'wasm';
  /** wasmバイナリのURL(既定 `new URL('./release-simd.wasm', import.meta.url)`、
      すなわちこのファイル(mdl_web.js)と同じディレクトリ)。 */
  wasmUrl?: string | URL;
  /** node/テスト用: 事前にインスタンス化したWasmModuleを注入する
      (指定時は wasmUrl を無視し、fetchを一切行わない)。 */
  wasmModule?: WasmModule;
  /** bitplane-wasmのスレッド数(既定1、COOP/COEP不要)。 */
  threads?: number;
  /** true: バンドルの programs 節を無視し、常にブラウザ側で
      buildConvProgram を実行する(A/B比較用)。 */
  forceRebuildPrograms?: boolean;
  /** true: programsをロード前に全数検証し、失敗したら例外(フォールバック
      禁止)。verifyCore.ts(検証CLI)は常にこれを true にする。 */
  strictPrograms?: boolean;
}

export type BundleKind = 'seg' | 'detection' | 'classify-image' | 'dense';

export interface WebBundleRuntime {
  kind: BundleKind;
  doc: BundleDoc;
  meta: ModelMeta;
  /** 実際に使われたエンジン('js' | 'wasm'。'auto'指定でも解決後の実値)。 */
  engine: string;
  /** programの出所・バージョン警告・エンジンフォールバック等、黙殺しない注記。 */
  notes: string[];
  runtime: ConvRuntime;
  /** [batch*inWidth]の0/1入力 → 整数スコア。全kind共通の最下層口。 */
  scores(bits: ArrayLike<number>, batch: number): { data: Int32Array; width: number };
  /** モデル解像度のRGBA(ImageData.data)→整数スコア(batch=1)。
      画像入力(seg/detection/classify-image)のkindだけに生える。 */
  scoresRgba?(rgba: Uint8ClampedArray): { data: Int32Array; width: number };
  /** seg専用: per-pixel argmaxラベル。 */
  segLabels?(rgba: Uint8ClampedArray): Int32Array;
  /** detection専用: CenterNetデコード後の箱列(ピクセル座標)。 */
  detect?(rgba: Uint8ClampedArray): DetBoxPx[];
}

// ---------------------------------------------------------------------------
// kind判別
// ---------------------------------------------------------------------------

/** readout/preproc から表示・実行方式(kind)を決める唯一の場所。
    優先順位: detection → seg(spatial_scores) → 画像preprocならclassify-image
    → それ以外はdense(時系列/特徴量入力、class_scores)。 */
function bundleKindOf(doc: BundleDoc): BundleKind {
  const readout = doc.readout;
  if (readout && (readout as DetReadout).kind === 'detection') return 'detection';
  if (readout && (readout as Readout).kind === 'spatial_scores') return 'seg';
  const preprocKind = (doc.preproc as { kind?: string } | undefined)?.kind;
  if (preprocKind === 'image_thermometer' || preprocKind === 'image_lowlevel') {
    return 'classify-image';
  }
  return 'dense';
}

// ---------------------------------------------------------------------------
// programs の事前検証(strictPrograms用)
// ---------------------------------------------------------------------------

/** strictPrograms=true のときに ConvRuntime 構築**前**に全プログラムを検証する。
    conv.ts::resolveConvProgram は非strict(壊れていれば黙って再構築)なので、
    「フォールバック禁止」を保証するにはここで先に落とす必要がある
    (仕様: 検証経路はフォールバックがprogramsの陳腐化を隠してしまうため
    禁止)。 */
function preValidatePrograms(
  doc: BundleDoc,
  programs: Record<string, BundleProgramDict>,
): void {
  const convNodes = new Map(
    doc.ir.nodes.filter((n) => n.type === 'conv2d').map((n) => [n.id, n]),
  );
  for (const [nodeId, dict] of Object.entries(programs)) {
    const node = convNodes.get(nodeId);
    if (!node) {
      throw new Error(
        `strictPrograms: programsのノードid ${nodeId} が IR の conv2d ノードに無い`,
      );
    }
    const spec = convTreeSpecOf(node.params as unknown as Conv2dParams);
    const program = convProgramFromDict(dict, spec.outChannels);
    verifyConvProgram(spec, program);
  }
}

// ---------------------------------------------------------------------------
// wasmエンジンの解決
// ---------------------------------------------------------------------------

async function resolveWasmModule(opts: LoadBundleOptions): Promise<WasmModule> {
  if (opts.wasmModule) return opts.wasmModule;
  const url = opts.wasmUrl instanceof URL
    ? opts.wasmUrl
    : (opts.wasmUrl ? new URL(opts.wasmUrl, import.meta.url)
      : new URL('./release-simd.wasm', import.meta.url));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`wasm取得に失敗: HTTP ${res.status} (${String(url)})`);
  const bytes = await res.arrayBuffer();
  return instantiateWasmModule(bytes);
}

/** engineオプションを 'js' | 'wasm' の実行モードへ解決する。'auto'は
    wasmを試み、失敗したら notes に理由を残して 'js' へ(黙殺しない)。 */
async function resolveEngine(
  opts: LoadBundleOptions,
  notes: string[],
): Promise<{ mode: 'js' | 'wasm'; wasmModule?: WasmModule }> {
  const want = opts.engine ?? 'auto';
  if (want === 'js') return { mode: 'js' };
  if (want === 'wasm') {
    return { mode: 'wasm', wasmModule: await resolveWasmModule(opts) };
  }
  try {
    return { mode: 'wasm', wasmModule: await resolveWasmModule(opts) };
  } catch (e) {
    notes.push(`wasmエンジンの初期化に失敗→jsへフォールバック: ${String(e)}`);
    return { mode: 'js' };
  }
}

// ---------------------------------------------------------------------------
// loadBundle
// ---------------------------------------------------------------------------

/** 配布バンドル(またはbundle_formatを持たない素のモデルJSON)を読み込み、
    実行可能な WebBundleRuntime を返す。conv2dエンジンは 'js'→'bitplane',
    'wasm'→'bitplane-wasm'(bitplane系のみがconv2d命令列化を扱えるため)。

    `json` は通常 `JSON.parse()` 済みのdocだが、`.mdlb` バイナリ形式
    (`md deploy bundle -o *.mdlb` が生成)の `ArrayBuffer` /
    `Uint8Array`(`fetch(url).then(r => r.arrayBuffer())` の結果)も直接
    渡せる — その場合は `decodeMdlb`(./mdlb.ts、Python側 deploy/mdlb.py
    のバイト単位ミラー)でデコードしてから以降は通常のバンドルと同じ経路
    を通る。 */
export async function loadBundle(
  json: unknown | ArrayBuffer | Uint8Array,
  opts: LoadBundleOptions = {},
): Promise<WebBundleRuntime> {
  const decoded: unknown = (json instanceof ArrayBuffer || json instanceof Uint8Array)
    ? decodeMdlb(json)
    : json;
  const { doc, warnings } = checkBundleDoc(decoded);
  const notes: string[] = [...warnings];
  const meta = deriveModelMeta(doc);
  const kind = bundleKindOf(doc);

  const threads = Math.max(1, Math.floor(opts.threads ?? 1));
  const forceRebuild = opts.forceRebuildPrograms ?? false;
  const strict = opts.strictPrograms ?? false;
  const rawPrograms = doc.programs ?? null;
  if (forceRebuild && rawPrograms) {
    notes.push('forceRebuildPrograms=true: バンドルのprograms節を無視してブラウザ側で再構築する');
  }
  const programs = forceRebuild ? null : rawPrograms;
  if (strict && programs) preValidatePrograms(doc, programs);

  const { mode, wasmModule } = await resolveEngine(opts, notes);
  const convEngine: ConvEngine = mode === 'wasm' ? 'bitplane-wasm' : 'bitplane';

  const runtime = await ConvRuntime.create(doc.ir, {
    engine: convEngine,
    programs,
    wasmModule,
    threads,
  });

  if (runtime.bundleProgramNote) notes.push(`programs採用: ${runtime.bundleProgramNote}`);
  if (runtime.convProgramFallbackNote) {
    notes.push(`命令列化フォールバック: ${runtime.convProgramFallbackNote}`);
  }
  if (runtime.threadsNote) notes.push(`スレッド: ${runtime.threadsNote}`);
  if (runtime.programProvenance) {
    const entries = Object.entries(runtime.programProvenance);
    const nBundle = entries.filter(([, p]) => p === 'bundle').length;
    notes.push(
      `conv2dプログラム出所: bundle=${nBundle} / built=${entries.length - nBundle}`
      + `(全${entries.length}ノード)`,
    );
  }

  const result: WebBundleRuntime = {
    kind,
    doc,
    meta,
    engine: mode,
    notes,
    runtime,
    scores: (bits, batch) => runtime.scores(bits, batch),
  };

  if (kind !== 'dense') {
    const preprocDoc = doc.preproc as unknown as ImagePreprocDoc;
    result.scoresRgba = (rgba: Uint8ClampedArray) => {
      const nchw = rgbaToNchw(rgba, preprocDoc.height, preprocDoc.width);
      const bits = encodeImageAuto(nchw, preprocDoc);
      return runtime.scores(bits, 1);
    };
  }
  if (kind === 'seg') {
    // readout.spatial は export_webbench.py::_readout_meta が形状を推定
    // できないとき null を返す正当な出力(実segエクスポートのみ埋まる)。
    // ロード時に落とすと scores() しか使わない検証経路(md deploy
    // verify-web)まで壊れるので、ここでは segLabels を生やさないだけに
    // 留め、黙殺せず notes に理由を残す(スコア自体は取れるので実害は
    // 「per-pixel argmaxができない」だけ)。
    const spatial = (doc.readout as Readout).spatial;
    if (spatial) {
      result.segLabels = (rgba: Uint8ClampedArray) => {
        const { data } = result.scoresRgba!(rgba);
        return runtime.argmaxPerPixel(data, 1, spatial.n_out, spatial.height * spatial.width);
      };
    } else {
      notes.push('readout.spatial 無し: segLabels は提供しない(scoresのみ)');
    }
  }
  if (kind === 'detection') {
    const ro = doc.readout as DetReadout;
    result.detect = (rgba: Uint8ClampedArray) => {
      const { data } = result.scoresRgba!(rgba);
      return decodeDetections(data, ro);
    };
  }

  return result;
}
