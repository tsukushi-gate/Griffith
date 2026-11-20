/* eslint-disable */
// このファイルは自動生成(scripts/gen_editor_types.py)。手で編集しないこと。
// 再生成: cd tools/editor/frontend && npm run gen-types
// 真実源: tools/editor/backend/schemas.py

/**
 * griffith-logicエディタのWSフレーム/RESTレスポンス契約(tools/editor/backend/schemas.py から自動生成)
 */
export interface EditorApiSchemas {
  ActivationsFrame?: ActivationsFrame;
  ApDeltaFrame?: ApDeltaFrame;
  CapabilityReason?: CapabilityReason;
  CaptureConfigFrame?: CaptureConfigFrame;
  CaptureInfo?: CaptureInfo;
  ConnectionsConfig?: ConnectionsConfig;
  ConvDLGNModelConfig?: ConvDLGNModelConfig;
  ConvertPresetInfo?: ConvertPresetInfo;
  ConvertStartRequest?: ConvertStartRequest;
  ConvertStatus?: ConvertStatus;
  CreateProjectRequest?: CreateProjectRequest;
  DataConfig?: DataConfig;
  DatasetConfigUpdate?: DatasetConfigUpdate;
  DatasetInfo?: DatasetInfo;
  DatasetSampleItem?: DatasetSampleItem;
  DatasetSamplesResponse?: DatasetSamplesResponse;
  DistillConfig?: DistillConfig;
  EncodingPreviewRequest?: EncodingPreviewRequest;
  EncodingPreviewResponse?: EncodingPreviewResponse;
  EvalItem?: EvalItem;
  EvalSource?: EvalSource;
  EvaluateRequest?: EvaluateRequest;
  EvaluateResponse?: EvaluateResponse;
  GateDistributionFrame?: GateDistributionFrame;
  GradFlowFrame?: GradFlowFrame;
  GradFlowLayerTrace?: GradFlowLayerTrace;
  HdlExportResponse?: HdlExportResponse;
  HdlVerifyResponse?: HdlVerifyResponse;
  HierarchyNode?: HierarchyNode;
  HierarchyResponse?: HierarchyResponse;
  HistoryCounts?: HistoryCounts;
  HistoryIndex?: HistoryIndex;
  HistoryIndexFrame?: HistoryIndexFrame;
  HistoryLayerView?: HistoryLayerView;
  HistoryStepView?: HistoryStepView;
  LayerConnectionsConfig?: LayerConnectionsConfig;
  LayerSummaryFrame?: LayerSummaryFrame;
  LoweringView?: LoweringView;
  MetricsFrame?: MetricsFrame;
  ModelConfig?: ModelConfig;
  NasIrResponse?: NasIrResponse;
  NasStartRequest?: NasStartRequest;
  NasStartResponse?: NasStartResponse;
  NasStatusResponse?: NasStatusResponse;
  NasTrialRow?: NasTrialRow;
  NasTrialsResponse?: NasTrialsResponse;
  NodeCapabilities?: NodeCapabilities;
  OptimConfig?: OptimConfig;
  PreflightBlocker?: PreflightBlocker;
  PreflightResponse?: PreflightResponse;
  ProjectSchema?: ProjectSchema;
  PythonModelImportResponse?: PythonModelImportResponse;
  RegConfig?: RegConfig;
  ResourceEstimate?: ResourceEstimate;
  RunConfig?: RunConfig;
  RunSummary?: RunSummary;
  SessionEndFrame?: SessionEndFrame;
  SkippedItem?: SkippedItem;
  StatsResponse?: StatsResponse;
  StepMetrics?: StepMetrics;
  TrainConfig?: TrainConfig;
  TrainStatus?: TrainStatus;
  TransferRequest?: TransferRequest;
  TransferResponse?: TransferResponse;
  U8Envelope?: U8Envelope;
  UncertainGate?: UncertainGate;
  UncertainGatesResponse?: UncertainGatesResponse;
  UpdateProjectRequest?: UpdateProjectRequest;
  VerifyAvailability?: VerifyAvailability;
}
/**
 * 中間活性フレーム(monitor.build_activations、fullキャプチャ時)。
 */
export interface ActivationsFrame {
  type: "activations";
  layer: string;
  step: number;
  mean: U8Envelope;
  saturation: number;
}
/**
 * u8量子化テンソルエンベロープ(WS/REST共通、monitor.encode_u8 と対応)。
 */
export interface U8Envelope {
  encoding: "u8";
  shape: number[];
  scale: number;
  data: string;
}
/**
 * AP法更新量フレーム(monitor.build_ap_delta)。
 */
export interface ApDeltaFrame {
  type: "ap_delta";
  layer: string;
  step: number;
  delta_norm: number;
  delta?: U8Envelope | null;
}
/**
 * capabilityの可否とUI表示用理由。
 */
export interface CapabilityReason {
  capability: string;
  enabled: boolean;
  reason?: string | null;
}
/**
 * WS接続直後に送るcapture設定(monitor.build_capture_config)。
 */
export interface CaptureConfigFrame {
  type: "capture_config";
  level: string;
  stride: number;
  method: string;
  snapshot_bytes: number;
  detail_max_elems: number;
}
/**
 * capture level/stride のペア(TrainStatus.capture / HistoryIndex.capture)。
 */
export interface CaptureInfo {
  level: string;
  stride: number;
}
/**
 * 規則配線の設定(先行実験)。文字列 'random' / 'unique' の代わりに指定する。
 *
 * kind:
 *   rot  — idx_a[g]=(g+s_a) mod in_dim, idx_b[g]=(g+s_b) mod in_dim。
 *          s_a / s_b は**層数と同じ長さ**のリスト(層ごとのずらし量)。
 *   perm — 2·out_dim スロットを前層出力が等回数使う均衡全射(層ごとに
 *          run.seed から導いた generator で生成)。**s_a / s_b は無視される**
 *          (レシピを `model.connections.kind=perm` の1行上書きで
 *          置換配線へ切り替えられるようにするため)。
 *   random_seeded  — 'random' と同じ生成器を run.seed 由来の generator で
 * 引く(先行実験)。s_a / s_b は無視される
 *   random_uniform — 各スロット独立一様の無制約乱択(先行実験)。同上
 *   per_layer — 層ごとに配線種を変える(先行実験)。**layers 必須**で、
 *          長さは model.layers と一致していなければならない。s_a / s_b
 *          (リスト)は使わず、各層の LayerConnectionsConfig が持つ
 *
 * 構造化固定配線(全層を閉形式で決める。1 層ずつ指定したいときは
 * per_layer の rot_n / butterfly / local1d / local2d を使う):
 *   butterfly — 基数 = model.n_inputs。層 0 = gather(idx[m] = (g·n + m) mod in_dim、
 *          n·k ≥ in_dim なら全入力を被覆)、層 l ≥ 1 = (g + m·stride_l) mod k。
 *          strides(長さ layers−1)を省略すると n^(l−1)、order='desc' で逆順。
 *          内部調査リポジトリの先行実験の BF6(n=6)/ BF4(n=4)
 *   local1d — 層ごとの strides(長さ layers、**必須**)で (c_g + m·stride_l) mod in_dim。
 * 先行実験候補 B。`core.connections.local1d_strides` が幾何級数列を作る
 *   local2d — 層 0 は画像格子 grid(H, W)× bits_per_cell(サーモメータ z)を
 *          layer0_offsets で読む(省略時: n_inputs == bits_per_cell なら自セル [(0,0)]、
 *          n_inputs == 2·bits_per_cell なら右隣・下隣 [(1,0),(0,1)])。隠れ層は
 *          ceil(√k) の正方トーラス格子に ring 列を assign(contig / stride)で割り当てる。
 *          先行実験の L2D6 / L2D4 / 候補 C
 * rot は step(層ごとの歩進のリスト、省略 = 全層 1。先行実験 rot2)を取る。
 * dedupe(既定 false)は構造化配線の重複ピンを +1 ずらしで潰す(先行実験の build() 相当)。
 */
export interface ConnectionsConfig {
  kind: "rot" | "perm" | "random_seeded" | "random_uniform" | "per_layer" | "butterfly" | "local1d" | "local2d";
  s_a?: number[] | null;
  s_b?: number[] | null;
  step?: number[] | null;
  layers?: LayerConnectionsConfig[] | null;
  strides?: number[] | null;
  order?: "asc" | "desc";
  grid?: [unknown, unknown] | null;
  bits_per_cell?: number;
  layer0_offsets?: [unknown, unknown][] | null;
  assign?: "contig" | "stride";
  dedupe?: boolean;
}
/**
 * 1層分の配線指定(先行実験。`ConnectionsConfig.kind='per_layer'` の要素)。
 *
 * kind:
 *   rot            — (g+s_a) mod in_dim / (g+s_b) mod in_dim。**s_a / s_b 必須**
 *                    (層ごとの単一値。リストではない)
 *   stretched_rot  — 基準位置を round(g·in_dim/out_dim) に引き伸ばした rot。
 *                    **s_a / s_b 必須**。in_dim == out_dim では rot と一致
 *   perm           — 均衡全射(配線用 generator を消費)。s_a / s_b は無視
 *   random_seeded  — get_random_connections を配線用 generator で引く。同上
 *                    (**均衡する**。無制約の乱択ではない。生成器の docstring 参照)
 *   random_uniform — 各スロット独立一様の無制約乱択(配線用 generator)。同上
 *
 * 構造化固定配線(乱数を引かない閉形式。移植元は内部調査リポジトリ
 * 先行実験):
 *   rot_n          — n 入力 rot: idx[m] = (g·step + shifts[m]) mod in_dim。
 *                    **shifts 必須**(長さ = model.n_inputs)
 *   butterfly      — 循環 radix-n バタフライ 1 段: (g + m·stride) mod in_dim。
 *                    **stride 必須**
 *   local1d        — 1D 局所窓: (c_g + m·stride) mod in_dim(c_g = 比例写像)。
 *                    **stride 必須**
 *   local2d        — 2D 局所窓: 格子 grid(None = 正方)× bits_per_cell ビットのセルを
 *                    offsets [(dx, dy), …] で読む。**offsets 必須**、
 *                    len(offsets) · bits_per_cell == model.n_inputs
 * rot / rot_n は step(歩進、既定 1。先行実験 rot2 の r)を取る。dedupe(既定 false)は
 * rot_n / butterfly / local1d / local2d の重複ピンを +1 ずらしで潰す
 * (rot 系・乱択系には適用しない = 既存レシピは不変)。
 */
export interface LayerConnectionsConfig {
  kind:
    | "rot"
    | "stretched_rot"
    | "perm"
    | "random_seeded"
    | "random_uniform"
    | "rot_n"
    | "butterfly"
    | "local1d"
    | "local2d";
  s_a?: number | null;
  s_b?: number | null;
  step?: number;
  shifts?: number[] | null;
  stride?: number | null;
  offsets?: [unknown, unknown][] | null;
  grid?: [unknown, unknown] | null;
  bits_per_cell?: number;
  dedupe?: boolean;
}
/**
 * `model.kind='conv_dlgn'` の conv 部と頭部の形(内部調査リポジトリ先行実験
 * ConvLogicNet / cdlgn)。ゲートのパラメータ化・推定器・RI・GroupSum の τ と割当は
 * ModelConfig 側(parameterization / estimator / residual_init / ri_mu / ri_sigma /
 * selection / tau / group_assign)を conv 木と頭部に共通で使う。
 *
 *   in_shape                — 入力 (C, H, W)。要素数はデータの入力幅と一致すること
 *                             (CIFAR-10 3 閾値は閾値 major で [9, 32, 32])
 *   channels                — conv 木ブロックの出力チャネル列(cdlgn k=32: [32, 128, 512, 1024])
 *   sampler                 — 葉の読み座標。random(従来)/ random8(cdlgn 忠実)/
 *                             cross5 / dilated8 / checker13(座標が (チャネル, 層) の計算可能関数)
 *   kernel_size / tree_depth / padding — sampler=random のときだけ(他は腕ごとに固定)
 *   input_channels_per_tree — 木 1 本が読む入力チャネル数(cdlgn num_chn)
 *   downsample              — orpool(stride-1 conv + OR プール 2×2)/ stride(stride-2 conv)
 *   head_dims               — 頭部の隠れ幅列(入力 flat を含まない)。省略で cdlgn 頭部則
 *                             [10·flat, 5·flat, 2.5·flat] × head_scale
 *   head_wiring — rot2(歩進付き rot、先行実験 rot2_schedule)/ random / unique
 */
export interface ConvDLGNModelConfig {
  /**
   * @minItems 3
   * @maxItems 3
   */
  in_shape: [unknown, unknown, unknown];
  channels?: number[];
  sampler?: "random" | "random8" | "cross5" | "dilated8" | "checker13";
  kernel_size?: number;
  tree_depth?: number;
  padding?: number;
  input_channels_per_tree?: number;
  downsample?: "orpool" | "stride";
  head_dims?: number[] | null;
  head_wiring?: "rot2" | "random" | "unique";
  head_scale?: number;
}
/**
 * 変換プリセット1件(GET /api/convert/presets、convert_presets.list_presets)。
 */
export interface ConvertPresetInfo {
  name: string;
  label: string;
  description: string;
  dataset: string;
  teacher_hint: string;
  spec: {
    [k: string]: unknown;
  };
}
/**
 * 変換開始リクエスト(POST /api/convert/start)。
 *
 * source: teacher指定('ckpt:<path>' 等、convert.teacher.resolve_teacherが解釈)。
 * dataset: 較正/評価データセットキー(省略時はpreset既定)。空文字はpreset既定。
 * preset: convert_presets のプリセット名(例 'mnist-intlower-v1')。
 * overrides: ConvertPipelineSpecフィールドの部分上書き(空dict=上書きなし)。
 */
export interface ConvertStartRequest {
  source: string;
  preset: string;
  dataset?: string;
  overrides?: {
    [k: string]: unknown;
  };
}
/**
 * 変換セッション状態(GET /api/convert/{cid}/status、convert.ConversionSession)。
 *
 * phase: preparing/pruning/quantizing/lowering/verifying/done/error/cancelled。
 * progress: 現phase内の進捗0-1。metrics: 各段acc・回路統計・学習曲線配列。
 * has_ir: done時にTrue(生成IRはSTATEへ反映済み、GET /api/models/current/irで取得)。
 */
export interface ConvertStatus {
  convert_id: string;
  phase: string;
  progress: number;
  metrics: {
    [k: string]: unknown;
  };
  error: string | null;
  has_ir: boolean;
  source: string;
  dataset: string;
}
export interface CreateProjectRequest {
  name: string;
  description?: string;
}
/**
 * データセット設定。
 *
 * name: 'mnist' 等(VISION_DATASETS)または 'logic:xor' 等(LOGIC_DATASETS)
 * params: ローダへの追加引数(binarize_threshold / repeat / limit / root 等)
 */
export interface DataConfig {
  name?: string;
  batch_size?: number;
  params?: {
    [k: string]: unknown;
  };
}
export interface DatasetConfigUpdate {
  split?: {
    [k: string]: unknown;
  } | null;
  seed?: number | null;
  augmentation?: {
    [k: string]: unknown;
  } | null;
  encoding?: {
    [k: string]: unknown;
  } | null;
}
/**
 * データセットカタログの1件(GET /api/datasets、datasets.list_datasets)。
 */
export interface DatasetInfo {
  name: string;
  label: string;
  in_width: number;
  out_width: number;
  kind: string;
  input_spec?: {
    [k: string]: unknown;
  } | null;
  output_spec?: {
    [k: string]: unknown;
  } | null;
  imported?: boolean;
}
/**
 * 1サンプル分の表示用データ。
 */
export interface DatasetSampleItem {
  index: number;
  image?: string | null;
  label?: number | null;
  inputs?: number[] | null;
  outputs?: number[] | null;
}
/**
 * GET /api/datasets/{id}/samples のレスポンス。
 */
export interface DatasetSamplesResponse {
  dataset_id: string;
  kind: string;
  num_data: number;
  num_columns: number;
  page: number;
  page_size: number;
  num_pages: number;
  samples: DatasetSampleItem[];
  class_distribution: {
    [k: string]: number;
  };
  class_distribution_approx?: boolean;
}
/**
 * 知識蒸留(C4)設定。
 *
 * TrainConfig.distill に設定されると、runner が classification タスクの学習を
 * KD(convert/distill.make_distill_task)へ差し替える。省略(None)時は従来の
 * 素の分類学習(後方互換)。
 *
 * teacher: resolve_teacher(convert/teacher.py)の source 文字列
 *     (例 'torchvision:resnet18' / 'ckpt:path.pt')
 * alpha: hard(CE)項の重み(0=純KD/1=純CE)
 * temperature: KD温度 T(二重温度 τ_gs×T の T 成分。τ_gs はモデル側 GroupSum.tau)
 * standardize: KD項の両ロジットを Z-score 標準化するか
 * kind: 'kl'(既定)または 'mse'(KL fallback)
 * teacher_transform: TEACHER_TRANSFORMS のキー('none'/'mnist'/'cifar10'/'binarize05')。
 *     teacher に渡す入力への正規化。teacher解決後に上書き適用する
 * cache: teacherロジットをキャッシュ(data-teacher-cache/)するか。
 *     False なら毎回再計算(fingerprint/params が同じでもロードしない)
 */
export interface DistillConfig {
  teacher: string;
  alpha?: number;
  temperature?: number;
  standardize?: boolean;
  kind?: "kl" | "mse";
  teacher_transform?: string;
  cache?: boolean;
  pseudo_labels?: boolean;
}
export interface EncodingPreviewRequest {
  values: number[];
  mode?: string;
  threshold?: number;
  bits?: number;
}
export interface EncodingPreviewResponse {
  mode: string;
  bits_per_value: number;
  grid: number[][];
}
export interface EvalItem {
  index: number;
  label?: number | null;
  pred?: number | null;
  label_bits?: number[] | null;
  pred_bits?: number[] | null;
  correct: boolean;
  soft_confidence: number;
  soft_margin: number;
  hard_soft_mismatch: boolean;
  input_bits?: number[] | null;
  image?: string | null;
}
export interface EvalSource {
  kind: string;
  run_id?: string | null;
}
export interface EvaluateRequest {
  source: EvalSource;
  dataset: string;
  executor?: string | null;
  dataset_params?: {
    [k: string]: unknown;
  } | null;
  page?: number;
  page_size?: number;
}
export interface EvaluateResponse {
  source_kind: string;
  dataset: string;
  executor: string | null;
  kind: string;
  num_classes: number | null;
  num_samples: number;
  page: number;
  page_size: number;
  num_pages: number;
  cached: boolean;
  on_demand: boolean;
  cap: number;
  accuracy: number | null;
  num_correct: number | null;
  num_errors: number | null;
  num_mismatch: number | null;
  confusion: number[][] | null;
  fingerprint: string;
  items: EvalItem[];
}
/**
 * ゲート確率分布フレーム(monitor.build_gate_distribution)。
 *
 * dense: probs(u8エンベロープ)あり / aggregate: hist(argmaxヒスト)あり。
 */
export interface GateDistributionFrame {
  type: "gate_distribution";
  layer: string;
  step: number;
  n_inputs: number;
  parameterization: string;
  mode: "dense" | "aggregate";
  probs?: U8Envelope | null;
  hist?: number[] | null;
  entropy_mean: number | null;
}
/**
 * 誤差逆伝播トレースフレーム(gradcap.build_grad_flow)。
 *
 * 履歴スナップショットのtrace(HistoryStepView.trace)も同形。
 */
export interface GradFlowFrame {
  type: "grad_flow";
  step: number;
  loss: number;
  layers: GradFlowLayerTrace[];
  layer_grad_norm: {
    [k: string]: number;
  };
}
/**
 * grad_flowの1層分トレース(gradcap.build_grad_flow)。order昇順=損失側→入力側。
 */
export interface GradFlowLayerTrace {
  layer: string;
  order: number;
  grad_out?: U8Envelope | null;
  grad_w?: U8Envelope | null;
  edge_grad?: U8Envelope | null;
}
/**
 * HDLエクスポート(POST /api/export/hdl、hdl.export_hdl)。
 */
export interface HdlExportResponse {
  source: string;
  filename: string;
  lang: string;
  estimate: ResourceEstimate;
}
/**
 * IRのリソース見積り(export.estimate.estimate_resources)。
 *
 * lut_counts/node_countsはJSON化でキーが文字列になる(dict[str, int])。
 */
export interface ResourceEstimate {
  total_luts: number;
  lut_counts: {
    [k: string]: number;
  };
  logic_depth: number;
  wire_count: number;
  group_sums: number;
  node_counts: {
    [k: string]: number;
  };
  lowered: boolean;
}
/**
 * HDL等価性検証(POST /api/export/hdl/verify、§4-3)。
 *
 * availableが偽(処理系未導入)のときpassedはNone。
 */
export interface HdlVerifyResponse {
  available: boolean;
  passed: boolean | null;
  n_vectors: number;
  stdout: string;
  lang: string;
}
/**
 * Python由来モデルの階層表示ノード(L3/L2/L1/L0)。
 */
export interface HierarchyNode {
  id: string;
  label: string;
  level: string;
  kind: string;
  parent?: string | null;
  detail?: string | null;
  shape?: {
    [k: string]: unknown;
  } | null;
  capabilities?: {
    [k: string]: boolean;
  } | null;
  lowering?: {
    [k: string]: unknown;
  } | null;
  paper_note?: string | null;
}
/**
 * GET /api/models/current/hierarchy のレスポンス。
 */
export interface HierarchyResponse {
  levels: string[];
  mode: string;
  nodes: HierarchyNode[];
  paper_ref?: {
    [k: string]: unknown;
  } | null;
  dataset_presets?: {
    [k: string]: {
      [k: string]: unknown;
    };
  } | null;
}
/**
 * TrainStatus.history のカウンタ(min/max_step, count)。
 */
export interface HistoryCounts {
  min_step: number | null;
  max_step: number | null;
  count: number;
}
/**
 * 履歴インデックス(GET /api/train/{sid}/history、history.HistoryStore.index)。
 */
export interface HistoryIndex {
  min_step: number | null;
  max_step: number | null;
  stride: number;
  keyframes: number[];
  count: number;
  ram_bytes?: number | null;
  disk_bytes?: number | null;
  capture?: CaptureInfo | null;
}
/**
 * 履歴インデックスのWSプッシュフレーム(RESTのHistoryIndexと同形+type)。
 */
export interface HistoryIndexFrame {
  min_step: number | null;
  max_step: number | null;
  stride: number;
  keyframes: number[];
  count: number;
  ram_bytes?: number | null;
  disk_bytes?: number | null;
  capture?: CaptureInfo | null;
  type: "history_index";
}
/**
 * 履歴ビューの1層分(main.py train_history_step。fieldsクエリで疎に載る)。
 */
export interface HistoryLayerView {
  gate_ids?: number[] | null;
  confidence?: number[] | null;
  n_inputs?: number | null;
  gate_distribution?: U8Envelope | null;
  grad_norm?: U8Envelope | null;
  act_mean?: U8Envelope | null;
}
/**
 * 任意stepのスナップショットビュー(GET /api/train/{sid}/history/{step})。
 */
export interface HistoryStepView {
  resolved_step: number;
  method: string;
  metrics: StepMetrics;
  layers: {
    [k: string]: HistoryLayerView;
  };
  trace?: GradFlowFrame | null;
}
/**
 * HistoryStepView.metrics(スナップショットのstep/epoch/loss/acc)。
 */
export interface StepMetrics {
  step: number;
  epoch: number;
  loss: number;
  acc: number;
}
/**
 * レイヤ要約フレーム(extract.layer_summary、最大5Hz差分送信)。
 */
export interface LayerSummaryFrame {
  type: "layer_summary";
  layer: string;
  step: number;
  n_inputs: number;
  gate_ids: number[];
  confidence: number[];
  entropy_mean: number | null;
  entropy_max: number | null;
}
/**
 * GET /api/models/current/lower のレスポンス。
 */
export interface LoweringView {
  node_id: string;
  level: string;
  description?: string | null;
  capabilities: {
    [k: string]: boolean;
  };
  shape?: {
    [k: string]: unknown;
  } | null;
  children: HierarchyNode[];
  lowering?: {
    [k: string]: unknown;
  } | null;
  virtual_ir_node?: {
    [k: string]: unknown;
  } | null;
  summary?: {
    [k: string]: unknown;
  } | null;
  actions: string[];
}
/**
 * 毎stepのmetricsフレーム(training.py _put_metrics)。
 *
 * val_loss/val_acc はhard eval直後のstepのみ載るスパースなキー。
 * stepperの追加メトリクス(exact_match等)が動的に載るためextraを許可する
 * (未知キーはフロントが無視する後方互換契約)。
 */
export interface MetricsFrame {
  type: "metrics";
  step: number;
  epoch: number;
  total_steps?: number | null;
  loss: number;
  acc: number;
  weight_delta: {
    [k: string]: number;
  };
  val_loss?: number | null;
  val_acc?: number | null;
  exact_match?: number | null;
  val_acc_soft?: number | null;
  selection_gap?: number | null;
  [k: string]: unknown;
}
/**
 * モデル設定。
 *
 * kind:
 *   dlgn      — Flatten + LutLayer×layers + GroupSum(classification用)
 *   logic_mlp — Flatten + LutLayer×layers(GroupSumなし、binary_vector用)
 *   conv_dlgn — LutConv2d(+OrPooling2d)×段 + Flatten + LutLayer×段 + GroupSum
 *               (classification用。形は conv: ConvDLGNModelConfig)
 *
 * 先行実験で追加(いずれも既定は従来動作):
 *   selection        — 訓練時のゲート選択('soft'/'hard_st'/'gumbel_st')
 *   init_mode        — 16ゲートモードのロジット初期化(None=従来のrandn)
 *   init_scale       — 残差初期化('ri'/'ri_randn'/'rri')の加算量(既定5.0。
 * 社内先行実験 Phase 3 の OP 腕は 1.0)
 *   grad_factor      — 勾配スケーリング係数
 *   gate_mask_hidden — 隠れ層で**除外する**ゲートIDの集合
 *   gate_mask_final  — 最終LutLayerで除外するゲートIDの集合
 *   connections      — 'random' / 'unique' / ConnectionsConfig
 *                      (rot / perm / 先行実験で random_seeded / random_uniform /
 *                       per_layer=層ごとの指定を追加)
 */
export interface ModelConfig {
  kind?: "dlgn" | "logic_mlp" | "conv_dlgn";
  conv?: ConvDLGNModelConfig | null;
  k?: number;
  layers?: number;
  n_inputs?: 2 | 4 | 6;
  parameterization?: "softmax" | "independent" | "iwp" | "raw" | "warp";
  connections?: ("random" | "unique") | ConnectionsConfig;
  tau?: number;
  group_assign?: "contig" | "stride";
  estimator?: "sigmoid" | "sin";
  residual_init?: boolean;
  ri_mu?: number;
  ri_sigma?: number;
  selection?: "soft" | "hard_st" | "gumbel_st";
  init_mode?: ("randn" | "ri" | "ri_randn" | "rri" | "uniform") | null;
  init_scale?: number;
  grad_factor?: number;
  gate_mask_hidden?: number[] | null;
  gate_mask_final?: number[] | null;
}
export interface NasIrResponse {
  trial_id: number;
  ir: {
    [k: string]: unknown;
  };
}
export interface NasStartRequest {
  search_id?: string;
  dataset?: string;
  model_kind?: string;
  max_trials?: number;
  epochs?: number;
  seed?: number;
  n_random?: number;
  time_budget_sec?: number | null;
  out_root?: string | null;
  layers_choices?: number[] | null;
  width_choices?: number[] | null;
  n_inputs_choices?: number[] | null;
  strategy?: string;
}
export interface NasStartResponse {
  search_id: string;
  state: string;
}
export interface NasStatusResponse {
  search_id: string | null;
  state: string;
  n_trials: number;
  max_trials: number | null;
  pareto: number[];
  error: string | null;
}
export interface NasTrialRow {
  trial_id: number;
  acc: number;
  best_acc: number;
  total_luts: number;
  logic_depth: number;
  gates: number;
  num_params: number;
  partial: boolean;
  state: string;
  parent_id: number | null;
  genome: {
    [k: string]: unknown;
  };
}
export interface NasTrialsResponse {
  search_id: string | null;
  trials: NasTrialRow[];
  pareto: number[];
}
/**
 * IR node meta.capabilities の標準形(v0.3相当)。
 */
export interface NodeCapabilities {
  python_executable: boolean;
  trainable: boolean;
  ir_buildable: boolean;
  gate_expandable: boolean;
  hdl_exportable: boolean;
}
/**
 * オプティマイザ設定
 */
export interface OptimConfig {
  name?: "adam" | "adamw" | "sgd";
  lr?: number;
}
/**
 * HDL出力を妨げる要因1件(§4-3)。
 *
 * kind: 'blackbox' | 'add_sum' | 'structure'。
 * node_idは該当ノード(構造エラー等で特定不能ならNone)。
 */
export interface PreflightBlocker {
  node_id: string | null;
  kind: string;
  message: string;
}
/**
 * HDL出荷プリフライト(GET /api/models/current/preflight、§4-3)。
 *
 * hdl_exportableはblockersが空のときのみ真。estimateはHDL不可の構造
 * エラー時はNone(見積り計算が破綻するため)。
 */
export interface PreflightResponse {
  hdl_exportable: boolean;
  blockers: PreflightBlocker[];
  warnings: string[];
  estimate: ResourceEstimate | null;
  draft_node_count: number;
  verify_available: VerifyAvailability;
}
/**
 * HDL等価性検証ツールの利用可否(verify.iverilog_available/ghdl_available)。
 */
export interface VerifyAvailability {
  iverilog: boolean;
  ghdl: boolean;
}
/**
 * プロジェクトのルートスキーマ。
 *
 * networks/train_configsの値自体はdictのまま保持し、
 * 妥当性検証(IR/TrainConfigのパース)は保存/読込時に本モジュールの
 * validate_network/validate_train_configで別途行う。
 */
export interface ProjectSchema {
  id: string;
  name: string;
  description?: string;
  networks?: {
    [k: string]: {
      [k: string]: unknown;
    };
  };
  train_configs?: {
    [k: string]: {
      [k: string]: unknown;
    };
  };
  dataset_refs?: string[];
  run_refs?: string[];
  optimizers?: {
    [k: string]: unknown;
  };
  monitors?: {
    [k: string]: unknown;
  };
  executors?: {
    [k: string]: unknown;
  };
  created_at: number;
  updated_at: number;
  deleted?: boolean;
}
/**
 * POST /api/models/import/python のレスポンス。
 */
export interface PythonModelImportResponse {
  ir: {
    [k: string]: unknown;
  };
  input_width: number;
  model_ref?: {
    [k: string]: unknown;
  } | null;
  spec: {
    [k: string]: unknown;
  };
}
/**
 * 正則化設定(先行実験)。省略(None)時は正則化なし=従来動作。
 *
 * kind:
 *   irreducibility — 縮約されやすいゲート種(定数・pass-through)へ置かれた
 *     softmax確率質量のペナルティ R = (1/N_h) Σ_n Σ_i p_{n,i}·c_i。
 *     隠れ層は {0,15,3,5,10,12}、最終LutLayerは {3,5,10,12} に c=1。
 * lambda: 正則化係数(YAMLキーは `lambda`。Python予約語のためフィールド名は
 *     lambda_)。0 なら損失に加算しない(**学習は正則化なしと bitwise 不変**)が、
 *     指標 reg_penalty は記録する。
 * ramp_frac: 学習ステップ全体に対する線形ランプ区間の割合。
 *     step ∈ [0, ramp_frac·total_steps] で 0 → lambda に線形、以降は lambda 固定。
 *     0 なら最初から定数 lambda。
 */
export interface RegConfig {
  kind?: "irreducibility";
  lambda?: number;
  ramp_frac?: number;
}
/**
 * 実行設定。
 *
 * eval_interval: 0=エポック末のみhard eval / N>0=Nステップごと
 * loss: None=タスク既定(classification→ce / binary_vector→bce)
 * device: None=自動検出(cuda→mps→cpu)
 *
 * wandb_* はレシピからwandbのtaxonomy(experiments/wandb_utils.py の
 * `init_wandb`)を指定するためのもの。**既定値は従来のランナーの挙動と同一**
 * (group='md-train' / job_type='train' / 追加タグなし / notes・hypothesis なし)
 * なので、既存レシピの記録内容は変わらない。
 *   wandb_group      — runのグループ名。**モデル族**を指定する(例 'dlgn-ffn')
 *   wandb_job_type   — ジョブ種別。本番学習と補助runを区別する
 *                      (語彙は training-script skill の train / ablation / sweep /
 *                      eval / smoke。較正アンカーの再走は 'train' + tag 'calibration')
 *   wandb_tags       — 横断ラベル。ランナー既定のタグへ**追加**する
 *                      (既定タグ: 'md-train' / model.kind / データ種別)
 *   wandb_notes      — この実験で何をなぜ変えたか(wandbのnotesへ)
 *   wandb_hypothesis — 検証したい仮説(configの experiment/hypothesis へ)
 */
export interface RunConfig {
  seed?: number;
  epochs?: number;
  eval_interval?: number;
  wandb?: boolean;
  wandb_group?: string;
  wandb_job_type?: "train" | "ablation" | "sweep" | "eval" | "smoke";
  wandb_tags?: string[];
  wandb_notes?: string | null;
  wandb_hypothesis?: string | null;
  run_name?: string | null;
  method?: string;
  loss?: string | null;
  device?: string | null;
  export_netlist?: boolean;
  save_eval_history?: boolean;
}
/**
 * runサマリ(GET /api/runs、runs.SUMMARY_KEYS)。壊れたrunはキー欠落=None。
 */
export interface RunSummary {
  run_id: string;
  started_at: number | null;
  ended_at: number | null;
  state: string | null;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  optimizer: string | null;
  lr: number | null;
  batch_size: number | null;
  epochs: number | null;
  final_step: number | null;
  final_loss: number | null;
  final_acc: number | null;
  model_name: string | null;
  wandb_url: string | null;
}
/**
 * セッション終了フレーム(main.py ws_train。TrainStatusと同形+type)。
 */
export interface SessionEndFrame {
  session_id: string;
  state: string;
  started_at?: number | null;
  step: number;
  epoch: number;
  total_steps?: number | null;
  loss: string | null;
  train_loss?: number | null;
  acc: number | null;
  error: string | null;
  step_budget?: number | null;
  method?: string | null;
  capture?: CaptureInfo | null;
  history?: HistoryCounts | null;
  dataset?: string | null;
  task?: string | null;
  last_val_acc?: number | null;
  last_val_acc_soft?: number | null;
  last_selection_gap?: number | null;
  type: "session_end";
}
export interface SkippedItem {
  id: string;
  reason: string;
}
/**
 * 統計APIレスポンス(extra='forbid'でスキーマドリフトを検出)。
 */
export interface StatsResponse {
  total_luts: number;
  lut_counts: {
    [k: string]: number;
  };
  logic_depth: number;
  wire_count: number;
  group_sums: number;
  node_counts: {
    [k: string]: number;
  };
  lowered: boolean;
  num_params: number;
  partial: boolean;
}
/**
 * 学習設定のルート(data/model/optim/run/expected_metrics)。
 *
 * expected_metrics: レシピの期待メトリクス照合。
 * キーは '<metric>_min' / '<metric>_max'(metricはrun_trainingの返り値キー。
 * 例 hard_acc_min / final_loss_max)。runner終了時に照合結果をprintし、
 * 返り値の 'expected_metrics' キーへ格納する(未達でも例外にはしない。
 * `griffith train --strict` のみ未達で非0 exit)。
 */
export interface TrainConfig {
  data?: DataConfig;
  model?: ModelConfig;
  optim?: OptimConfig;
  run?: RunConfig;
  distill?: DistillConfig | null;
  regularization?: RegConfig | null;
  expected_metrics?: {
    [k: string]: number;
  };
}
/**
 * 訓練セッション状態(GET /api/train/{sid}/status 等、training.py status)。
 *
 * lossは現在タスクの損失名(例 'ce'/'bce')。数値の直近train lossは
 * train_loss(旧'loss'。dictキー重複で損失名に上書きされていたのを分離)。
 * 現在の追加キーは旧バックエンド互換のためoptional。
 */
export interface TrainStatus {
  session_id: string;
  state: string;
  started_at?: number | null;
  step: number;
  epoch: number;
  total_steps?: number | null;
  loss: string | null;
  train_loss?: number | null;
  acc: number | null;
  error: string | null;
  step_budget?: number | null;
  method?: string | null;
  capture?: CaptureInfo | null;
  history?: HistoryCounts | null;
  dataset?: string | null;
  task?: string | null;
  last_val_acc?: number | null;
  last_val_acc_soft?: number | null;
  last_selection_gap?: number | null;
}
export interface TransferRequest {
  source_ir: {
    [k: string]: unknown;
  };
  target_ir: {
    [k: string]: unknown;
  };
}
export interface TransferResponse {
  target_ir: {
    [k: string]: unknown;
  };
  transferred: string[];
  skipped: SkippedItem[];
}
/**
 * 確信度が低いゲート1件(GET /api/train/{sid}/uncertain_gates、§4-1)。
 *
 * entropyは正規化迷い[0,1](層間・パラメータ化間で比較可能)。
 */
export interface UncertainGate {
  layer_id: string;
  neuron_index: number;
  entropy: number;
  gate_id: number;
}
/**
 * 低確信ゲートTop-Nレスポンス(§4-1)。gatesはentropy降順。
 */
export interface UncertainGatesResponse {
  step: number;
  gates: UncertainGate[];
}
/**
 * 部分更新的なPUTボディ。省略項目は既存値を維持する。
 */
export interface UpdateProjectRequest {
  name?: string | null;
  description?: string | null;
  networks?: {
    [k: string]: {
      [k: string]: unknown;
    };
  } | null;
  train_configs?: {
    [k: string]: {
      [k: string]: unknown;
    };
  } | null;
  dataset_refs?: string[] | null;
  run_refs?: string[] | null;
  optimizers?: {
    [k: string]: unknown;
  } | null;
  monitors?: {
    [k: string]: unknown;
  } | null;
  executors?: {
    [k: string]: unknown;
  } | null;
}
