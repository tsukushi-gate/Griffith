// モデルJSON(ModelDoc / ConvModelDoc)から「ページに出す値」を導出する
// 単一の場所。
//
// 背景(実在した陳腐化): 各ページのHTML/TSに hard mAP・k値・ゲート数・入力ch数・
// 配信サイズがハードコードされており、モデルを差し替えると説明文だけが古いまま
// 残っていた(例: detbackends.html の説明文は旧世代の k と mAP を指したままで、
// 同ページの表がモデルJSONから出していた値と食い違っていた)。
//
// 方針:
//   - モデル固有の数値は**必ず**ロード済みJSONから導出する(この1ファイル経由)。
//   - 欠損フィールドは null を返す。呼び出し側は「—」で描画し、**嘘の既定値を
//     出さない**(古い値のフォールバックは陳腐化そのものなので持たない)。
//   - 比較対象(ONNX/TF.js teacher, MediaPipe)はモデルJSONを持たないため、
//     references.ts に「参照値」として1箇所だけ集約する(出典wandb run id 付き)。
import { convTreeSpecOf } from './engines/bitplane';
import { buildConvProgram } from './engines/convProgram';

// readout は検出(DetReadout)/ spatial・class_scores(Readout)の判別共用体で、
// ここでは「あれば読む」だけなので緩い形で受ける。
type AnyReadout = Partial<{
  kind: string;
  n_classes: number;
  output_stride: number;
  threshold: number;
  top_k: number;
}>;

// ---------------------------------------------------------------------------
// 型
// ---------------------------------------------------------------------------

/** export_webbench.py が書き出す機械可読な背景説明ブロック(追加)。
    既存モデルJSONには存在しないため、**全フィールド optional**。無い場合は
    ページ側の静的散文にフォールバックする。 */
export interface AboutBlock {
  dataset?: string;
  dataset_label?: string;
  task?: string;
  task_label?: string;
  recipe?: Record<string, unknown>;
  teacher?: Record<string, unknown>;
  notes?: string[];
  wandb_run_id?: string;
  source_ckpt?: string;
  exported_at?: string;
}

/** 精度指標(モデルJSONの eval キーから判定した指標名+値)。 */
export interface AccuracyMeta {
  /** eval 内のキー名(例 'best_hard_map')。 */
  key: string;
  /** 表示名(例 'hard mAP')。 */
  label: string;
  value: number;
}

/** ページ描画に必要な値の導出結果。**欠損は null**。 */
export interface ModelMeta {
  name: string | null;
  createdAt: string | null;
  /** モデル名から読み取った group_sum の k(例 det_..._lut_k256 → 256)。
      about.recipe.k があればそちらを優先する。無ければ null。 */
  k: number | null;
  accuracy: AccuracyMeta | null;
  /** 重み共有ゲート数(木の本数ベース。eval.num_gates_weight_shared)。 */
  gatesWeightShared: number | null;
  /** 空間展開後のLUT総数(resources.total_luts)。実際には一度も実体化しない
      仮想的な値なので、画面には出さない(誤解を招くため)。 */
  gatesUnrolled: number | null;
  /** 配布バンドルのフォーマット版(bundle_version)。生モデルdocなら null。 */
  bundleVersion: string | null;
  /** バンドルに焼き込まれた命令列のゲート数(programs.*.stats.gates_after
      の総和)。バンドルでなければ null。conv2dが無いバンドル(dense)は 0。 */
  gatesProgram: number | null;
  /** 簡約後に実際に実行される2入力ゲート数。
      バンドルなら gatesProgram をそのまま使い、生モデルdocなら
      ランタイムと同じ buildConvProgram で数え直す(表示値と実行実体を一致させる)。
      どちらも取れなければ null。 */
  gatesReduced: number | null;
  logicDepth: number | null;
  wireCount: number | null;
  nodeCounts: Record<string, number> | null;
  /** ir.inputs[0].width(=入力ビット総数)。 */
  inputBits: number | null;
  inputHeight: number | null;
  inputWidth: number | null;
  preprocKind: string | null;
  nThresholds: number | null;
  /** 総入力チャネル数(inputBits /(H*W)。割り切れないときは preproc から推定)。 */
  inputChannels: number | null;
  jsonBytes: number | null;
  gzipBytes: number | null;
  nClasses: number | null;
  labels: string[];
  outputStride: number | null;
  /** readout.threshold(検出のスコア既定閾値)。 */
  scoreThreshold: number | null;
  topK: number | null;
  about: AboutBlock | null;
}

/** eval のキー → 指標表示名。ここに無いキーは精度として扱わない
    (num_gates_weight_shared/n_test 等が誤って精度になるのを防ぐ)。
    先に並んでいるものほど優先(同一docに複数あるとき)。 */
export const METRIC_LABELS: readonly (readonly [string, string])[] = [
  ['best_hard_map', 'mAP'],
  ['best_hard_miou', 'mIoU'],
  ['hard_map', 'mAP'],
  ['hard_miou', 'mIoU'],
  ['hard_acc_i16', '正解率'],
  ['hard_acc', '正解率'],
];

/** preproc.kind → 人間向けの説明(モデル固有の**数値は含めない**)。 */
export const PREPROC_LABELS: Record<string, string> = {
  image_thermometer: '明るさのしきい値による0/1変換(学習しない固定処理)',
  image_lowlevel:
    '0/1変換 + 輪郭・曲がり・色の違いの検出(いずれも学習しない固定処理)',
  thermometer: '数値のしきい値による0/1変換(学習しない固定処理)',
};

// ---------------------------------------------------------------------------
// 導出(純関数)
// ---------------------------------------------------------------------------

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function rec(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

/** モデル名末尾の `_k<digits>`(例 seg_convdlgn_k32、det_face_lut_k256_w18)から
    k を読む。命名規約に依存するので**あくまで about が無いときの補助**。 */
export function kFromModelName(name: string | null): number | null {
  if (!name) return null;
  const m = /_k(\d+)(?:[_.]|$)/.exec(name);
  return m ? Number(m[1]) : null;
}

/** eval ブロックから精度(指標名+値)を1つ選ぶ。該当キーが無ければ null。 */
export function pickAccuracy(evalBlock: unknown): AccuracyMeta | null {
  const ev = rec(evalBlock);
  if (!ev) return null;
  for (const [key, label] of METRIC_LABELS) {
    const v = num(ev[key]);
    if (v !== null) return { key, label, value: v };
  }
  return null;
}

/** 配布バンドルの programs から命令列ゲート数の総和を求める。
 *
 * バンドルは `md deploy bundle` が生成時に簡約(ゲート木→命令列)を済ませており、
 * 各 conv2d ノードの `stats.gates_after` がその結果。ブラウザは二度と簡約しない
 * ので、ここを読むのが「実際に実行される命令数」の正しい出どころになる。
 *
 * conv2d を持たないバンドル(dense等)は 0 を返す。バンドルでない生モデルdocは
 * null(=未計測)を返して呼び出し側のフォールバックに委ねる。
 */
function sumProgramGates(d: Record<string, unknown>): number | null {
  const programs = rec(d.programs);
  if (!programs) return null;
  let total = 0;
  for (const v of Object.values(programs)) {
    const g = num(rec(rec(v)?.stats)?.gates_after);
    if (g !== null) total += g;
  }
  return total;
}

/** 簡約後に実際に実行される2入力ゲート数を数える。
 *
 * ランタイム(bitplane / packed / webgpu)がロード時に走らせるのと**同じ**
 * `buildConvProgram` を使うので、画面に出る数と実行される命令数が必ず一致する。
 * モデルJSONに焼き込まれた値を読むのではなく毎回数え直すのは、モデルを
 * 差し替えたときに表示だけ古いまま残る事故(このファイル冒頭の背景を参照)を
 * 構造的に防ぐため。
 */
const REDUCED_CACHE = new WeakMap<object, number | null>();

function countReducedGates(d: Record<string, unknown>): number | null {
  const hit = REDUCED_CACHE.get(d);
  if (hit !== undefined) return hit;
  const v = computeReducedGates(d);
  REDUCED_CACHE.set(d, v);
  return v;
}

function computeReducedGates(d: Record<string, unknown>): number | null {
  const ir = rec(d.ir);
  const nodes = Array.isArray(ir?.nodes) ? ir!.nodes : null;
  if (!nodes) return null;
  let total = 0;
  let seen = false;
  for (const n of nodes) {
    const node = rec(n);
    if (!node || node.type !== 'conv2d') continue;
    try {
      const prog = buildConvProgram(convTreeSpecOf(node.params as never));
      total += prog.instrs.length;
      seen = true;
    } catch {
      return null; // 想定外の形は黙って壊さず「不明」にする
    }
  }
  return seen ? total : null;
}

/** ModelDoc / ConvModelDoc → 表示用メタ。**欠損は null**(嘘の既定値は出さない)。 */
export function deriveModelMeta(doc: unknown): ModelMeta {
  const d = rec(doc) ?? {};
  const ir = rec(d.ir) ?? {};
  const meta = rec(ir.metadata) ?? {};
  const resources = rec(d.resources) ?? {};
  const preproc = rec(d.preproc) ?? {};
  const readout = rec(d.readout) as AnyReadout | null;
  const about = rec(d.about) as AboutBlock | null;

  const name = typeof meta.name === 'string' ? meta.name : null;
  const inputs = Array.isArray(ir.inputs) ? ir.inputs : [];
  const inputBits = num(rec(inputs[0])?.width);
  const h = num(preproc.height);
  const w = num(preproc.width);

  // 総入力ch数: 入力ビット総数 /(H*W)。IRが真実源なので最優先。
  let inputChannels: number | null = null;
  if (inputBits !== null && h !== null && w !== null && h * w > 0
      && inputBits % (h * w) === 0) {
    inputChannels = inputBits / (h * w);
  } else {
    // フォールバック: thermometer のみ(低レベル特徴つきでは足りないので
    // kind を見て thermometer 系だけに限る)。
    const rgb = num(preproc.rgb_channels);
    const nthr = num(preproc.n_thresholds);
    if (preproc.kind === 'image_thermometer' && rgb !== null && nthr !== null) {
      inputChannels = rgb * nthr;
    }
  }

  const labels = Array.isArray(d.labels)
    ? (d.labels as unknown[]).filter((x): x is string => typeof x === 'string')
    : [];

  const aboutK = num(rec(about?.recipe)?.k);

  return {
    name,
    createdAt: typeof meta.created_at === 'string' ? meta.created_at : null,
    k: aboutK ?? kFromModelName(name),
    accuracy: pickAccuracy(d.eval),
    // export_webbench.py は eval 側に置く(resources 側に置く実装もありうるので両対応)。
    gatesWeightShared: num(rec(d.eval)?.num_gates_weight_shared)
      ?? num(resources.num_gates_weight_shared),
    gatesUnrolled: num(resources.total_luts),
    bundleVersion: typeof d.bundle_version === 'string' ? d.bundle_version : null,
    gatesProgram: sumProgramGates(d),
    // バンドルなら焼き込み済みの値をそのまま使う(再計算しない)。
    // 生モデルdocのときだけ、読まれた時点で数える。
    get gatesReduced() { return sumProgramGates(d) ?? countReducedGates(d); },
    logicDepth: num(resources.logic_depth),
    wireCount: num(resources.wire_count),
    nodeCounts: (rec(resources.node_counts)
      ? Object.fromEntries(
        Object.entries(rec(resources.node_counts)!)
          .filter(([, v]) => typeof v === 'number'),
      ) as Record<string, number>
      : null),
    inputBits,
    inputHeight: h,
    inputWidth: w,
    preprocKind: typeof preproc.kind === 'string' ? preproc.kind : null,
    nThresholds: num(preproc.n_thresholds),
    inputChannels,
    // 配布バンドルとして書き出された場合、実際に配るのは bundle_*_bytes の方。
    jsonBytes: num(resources.bundle_json_bytes) ?? num(resources.json_bytes),
    gzipBytes: num(resources.bundle_gzip_bytes) ?? num(resources.gzip_bytes),
    nClasses: num(readout?.n_classes) ?? (labels.length > 0 ? labels.length : null),
    labels,
    outputStride: num(readout?.output_stride),
    scoreThreshold: num(readout?.threshold),
    topK: num(readout?.top_k),
    about,
  };
}

// ---------------------------------------------------------------------------
// 整形
// ---------------------------------------------------------------------------

/** 欠損表示。ページ本文で「空欄」「undefined」を出さないための単一の記号。 */
export const MISSING = '—';

/** 桁区切り(ロケール非依存。テストの決定性のため toLocaleString は使わない)。 */
export function fmtInt(v: number | null): string {
  if (v === null) return MISSING;
  const neg = v < 0;
  const s = Math.abs(Math.round(v)).toString();
  const grouped = s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return neg ? `-${grouped}` : grouped;
}

/** 1,234,567 → "1.2M" 相当の短縮表記(散文中の「約N Mゲート」用)。 */
export function fmtCompact(v: number | null): string {
  if (v === null) return MISSING;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
}

export function fmtBytes(v: number | null): string {
  if (v === null) return MISSING;
  if (v >= 1024 * 1024) return `${(v / (1024 * 1024)).toFixed(2)} MB`;
  if (v >= 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${v} B`;
}

/** 精度の値。桁数はモデルJSONの実値をそのまま4桁で出す(閾値ではない)。 */
export function fmtMetric(v: number | null): string {
  return v === null ? MISSING : v.toFixed(4);
}

/** about.recipe → 1行サマリ。about が無ければ null(=ページの静的散文を残す)。 */
export function formatRecipe(about: AboutBlock | null): string | null {
  const r = about?.recipe;
  if (!r) return null;
  const parts: string[] = [];
  const push = (k: string, label = k): void => {
    const v = r[k];
    if (v === undefined || v === null || v === '') return;
    parts.push(`${label}=${String(v)}`);
  };
  push('k');
  push('wiring', '配線');
  push('selection', '選択');
  push('input_channels_per_tree', 'ICT');
  push('n_thresholds', '閾値数');
  push('tree_depth', '木深さ');
  push('n_layers', '層数');
  return parts.length > 0 ? parts.join(' / ') : null;
}

/** `data-meta="<key>"` に流し込む表示文字列。欠損は全て MISSING。 */
export function metaFields(meta: ModelMeta): Record<string, string> {
  const acc = meta.accuracy;
  return {
    modelName: meta.name ?? MISSING,
    createdAt: meta.createdAt ? meta.createdAt.slice(0, 10) : MISSING,
    k: meta.k === null ? MISSING : String(meta.k),
    kLabel: meta.k === null ? MISSING : `k${meta.k}`,
    accuracy: acc ? `${acc.label} ${fmtMetric(acc.value)}` : MISSING,
    accuracyLabel: acc ? acc.label : MISSING,
    accuracyValue: acc ? fmtMetric(acc.value) : MISSING,
    gatesShared: fmtInt(meta.gatesWeightShared),
    gatesUnrolled: fmtInt(meta.gatesUnrolled),
    gatesUnrolledCompact: fmtCompact(meta.gatesUnrolled),
    // 実行実体。画面にはこれを出す(参照された時だけ数える)。
    get gatesReduced() { return fmtInt(meta.gatesReduced); },
    logicDepth: meta.logicDepth === null ? MISSING : String(meta.logicDepth),
    wireCount: fmtInt(meta.wireCount),
    nodeCounts: meta.nodeCounts
      ? Object.entries(meta.nodeCounts).map(([k, v]) => `${k}×${v}`).join(' / ')
      : MISSING,
    inputRes: meta.inputHeight !== null && meta.inputWidth !== null
      ? `${meta.inputHeight}×${meta.inputWidth}` : MISSING,
    inputChannels: meta.inputChannels === null ? MISSING : String(meta.inputChannels),
    inputBits: fmtInt(meta.inputBits),
    nThresholds: meta.nThresholds === null ? MISSING : String(meta.nThresholds),
    preprocKind: meta.preprocKind ?? MISSING,
    preprocLabel: meta.preprocKind
      ? (PREPROC_LABELS[meta.preprocKind] ?? meta.preprocKind) : MISSING,
    jsonBytes: fmtBytes(meta.jsonBytes),
    gzipBytes: fmtBytes(meta.gzipBytes),
    nClasses: meta.nClasses === null ? MISSING : String(meta.nClasses),
    labels: meta.labels.length > 0 ? meta.labels.join(', ') : MISSING,
    outputStride: meta.outputStride === null ? MISSING : String(meta.outputStride),
    // readout.threshold(decode側の閾値)。UIスライダーの既定値ではない。
    scoreThreshold: meta.scoreThreshold === null
      ? MISSING : String(meta.scoreThreshold),
    topK: meta.topK === null ? MISSING : String(meta.topK),
    datasetLabel: meta.about?.dataset_label ?? meta.about?.dataset ?? MISSING,
    taskLabel: meta.about?.task_label ?? meta.about?.task ?? MISSING,
    recipe: formatRecipe(meta.about) ?? MISSING,
    wandbRun: meta.about?.wandb_run_id ?? MISSING,
  };
}

// ---------------------------------------------------------------------------
// DOM への流し込み
// ---------------------------------------------------------------------------

/** `[data-<attr>]` を持つ要素を lookup の戻り値で埋める共通処理。
    lookup が null を返した要素は**触らない**(HTML側の静的フォールバックを残す)。 */
export function fillPlaceholders(
  attr: string,
  lookup: (key: string) => string | null,
  root: ParentNode | null = typeof document === 'undefined' ? null : document,
): number {
  if (!root) return 0;
  let n = 0;
  root.querySelectorAll(`[data-${attr}]`).forEach((el) => {
    const key = (el as HTMLElement).dataset[attr];
    if (!key) return;
    const v = lookup(key);
    if (v === null) return;
    el.textContent = v;
    n++;
  });
  return n;
}

/** モデルdoc → `[data-meta]` を全て埋める。未知キーは MISSING で埋める
    (綴り間違いを空欄で見逃さないため)。戻り値は導出済みメタ。 */
export function applyModelMeta(
  doc: unknown,
  root: ParentNode | null = typeof document === 'undefined' ? null : document,
): ModelMeta {
  const meta = deriveModelMeta(doc);
  const fields = metaFields(meta);
  fillPlaceholders('meta', (k) => fields[k] ?? MISSING, root);
  // about があるときだけ上書きする(無ければHTMLの静的散文をそのまま残す)。
  applyAbout(meta, root);
  return meta;
}

/** `[data-about]` を about ブロックから埋める。**about が無いページ/モデルでは
    何もしない**(既存モデルJSONは再生成しないのでフォールバック必須)。 */
export function applyAbout(
  meta: ModelMeta,
  root: ParentNode | null = typeof document === 'undefined' ? null : document,
): number {
  const about = meta.about;
  if (!about) return 0;
  return fillPlaceholders('about', (k) => {
    if (k === 'dataset') return about.dataset_label ?? about.dataset ?? null;
    if (k === 'task') return about.task_label ?? about.task ?? null;
    if (k === 'recipe') return formatRecipe(about);
    if (k === 'notes') return about.notes && about.notes.length > 0
      ? about.notes.join(' ') : null;
    if (k === 'wandb') return about.wandb_run_id ?? null;
    return null;
  }, root);
}

// 注: 「スコア閾値スライダーの既定値」は **モデルの readout.threshold とは
// 別物** なので、ここから供給しない。
//   - readout.threshold = CenterNet decode 側の閾値(モデルJSONのメタデータ)
//   - スライダー = デモ画面に box をいくつ描くかの表示用フィルタ。DLGNのhmスコアが
//     低い値域に圧縮される性質に合わせて**実測で較正した**値であり、各ページのHTMLに
//     value= として置いてある(そこが真実源)。
// 両者を同一視して readout.threshold をスライダーに流し込むと過検出になる。
