// バックエンドREST APIクライアント(vite dev proxyで /api → localhost:8765)

import type {
  HdlExportResponse,
  HdlVerifyResponse,
  PreflightResponse,
} from './generated/types';
import type { IRGraphJson } from '../ir/schema';
import type {
  ConvertPresetInfo,
  ConvertStartRequest,
  ConvertStatus,
} from './generated/types';

export type { ConvertPresetInfo, ConvertStartRequest, ConvertStatus };

export type {
  HdlExportResponse,
  HdlVerifyResponse,
  PreflightBlocker,
  PreflightResponse,
  ResourceEstimate,
} from './generated/types';

export interface ModelResponse {
  ir: IRGraphJson;
  input_width: number;
}

export interface PythonModelResponse extends ModelResponse {
  model_ref?: Record<string, unknown> | null;
  spec: Record<string, unknown>;
}

export interface TrainStatus {
  session_id: string;
  state: string;
  step: number;
  epoch: number;
  total_steps?: number | null;
  loss: string | null; // 現在はタスクの損失名(例: 'bce')
  train_loss?: number | null; // 数値の直近train loss(旧'loss'キーから分離)
  acc: number | null;
  error: string | null;
  // 追加キー(旧バックエンドでは欠落しうるためoptional)
  step_budget?: number;
  method?: string;
  capture?: { level: string; stride: number };
  history?: { min_step: number | null; max_step: number | null; count: number };
  // 追加キー
  dataset?: string;
  task?: string;
  last_val_acc?: number | null;
  // 並行run一覧(GET /api/train/sessions)で開始時刻順に並べるためのキー
  started_at?: number | null;
}

/** u8量子化テンソルエンベロープ(WS/REST共通、backend monitor.py と対応) */
export interface U8Envelope {
  encoding: 'u8';
  shape: number[];
  scale: number;
  data: string; // base64
}

export interface HistoryIndex {
  min_step: number | null;
  max_step: number | null;
  stride: number;
  keyframes: number[];
  count: number;
  ram_bytes?: number;
  disk_bytes?: number;
  capture?: { level: string; stride: number };
}

export interface HistoryLayerView {
  gate_ids?: number[];
  confidence?: number[];
  n_inputs?: number;
  gate_distribution?: U8Envelope | null;
  grad_norm?: U8Envelope;
  act_mean?: U8Envelope;
}

export interface HistoryStepView {
  resolved_step: number;
  method: string;
  metrics: { step: number; epoch: number; loss: number; acc: number };
  layers: Record<string, HistoryLayerView>;
  trace?: unknown;
}

export interface RestoreResult {
  restored_step: number;
  state: string;
  optimizer_restored: boolean;
  history_truncated_from: number;
}

export interface TrainStartParams {
  dataset: string;
  lr: number;
  logic_lr?: number;
  ann_lr?: number;
  batch_size: number;
  epochs: number;
  // 学習方式・損失・optimizer(省略時はバックエンド既定)
  method?: 'bp' | 'ap' | 'bb';
  loss?: string;
  optimizer?: 'adam' | 'sgd';
  dataset_params?: Record<string, unknown>;
  // wandb連携(opt-in。省略時はバックエンド既定=false)
  wandb?: boolean;
}

/** データセットカタログの1件(GET /api/datasets)。*/
export interface DatasetInfo {
  name: string;
  label: string;
  in_width: number;
  out_width: number;
  kind: string; // 'classification' | 'binary_vector'
  input_spec?: Record<string, unknown> | null;
  output_spec?: Record<string, unknown> | null;
  imported?: boolean; // 取込データセット(GET /api/datasets/imported 由来)
}

/** データセットサンプル1件(GET .../samples)。*/
export interface DatasetSampleItem {
  index: number;
  image?: string | null; // 画像系: PNG data URL
  label?: number | null; // 画像系/分類: クラスindex
  inputs?: number[] | null; // 論理系: 入力ビット列
  outputs?: number[] | null; // 論理系: 出力ビット列
}

/** サンプル閲覧レスポンス。*/
export interface DatasetSamplesResponse {
  dataset_id: string;
  kind: string; // 'image' | 'logic'
  num_data: number;
  num_columns: number;
  page: number;
  page_size: number;
  num_pages: number;
  samples: DatasetSampleItem[];
  class_distribution: Record<string, number>;
  class_distribution_approx?: boolean;
}

/** 入力エンコーディングプレビュー。*/
export interface EncodingPreviewResponse {
  mode: string;
  bits_per_value: number;
  grid: number[][];
}

/** 取込データセットの設定(split/augmentation/encoding)。*/
export interface ImportedDatasetConfig {
  split?: Record<string, number>;
  seed?: number;
  augmentation?: { flip?: boolean; shift?: number; bit_flip_noise?: number };
  encoding?: { mode?: string; threshold?: number; bits?: number };
}

export interface CapabilityReason {
  capability: string;
  enabled: boolean;
  reason?: string | null;
}

export interface HierarchyNode {
  id: string;
  label: string;
  level: string;
  kind: string;
  parent?: string | null;
  detail?: string | null;
  shape?: Record<string, unknown> | null;
  capabilities?: Record<string, boolean> | null;
  lowering?: Record<string, unknown> | null;
  paper_note?: string | null;
}

export interface HierarchyResponse {
  levels: string[];
  mode: string;
  nodes: HierarchyNode[];
  paper_ref?: Record<string, unknown> | null;
  dataset_presets?: Record<string, Record<string, unknown>> | null;
}

export interface LoweringView {
  node_id: string;
  level: string;
  title: string;
  description?: string | null;
  capabilities: Record<string, boolean>;
  shape?: Record<string, unknown> | null;
  children: HierarchyNode[];
  lowering?: Record<string, unknown> | null;
  virtual_ir_node?: IRGraphJson['nodes'][number] | null;
  summary?: {
    type: 'layer_summary';
    layer: string;
    step: number;
    n_inputs: number;
    gate_ids: number[];
    confidence: number[];
  } | null;
  actions: string[];
}

/** HTTPステータス付きAPIエラー(404によるセッション消失検出などに使う)。*/
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body.detail) detail = String(body.detail);
    } catch {
      // JSONでないエラーボディは無視
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function put<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** 統計API GET /api/stats のレスポンス(backend/stats.py の StatsResponse と対応)。*/
export interface StatsResponse {
  total_luts: number;
  lut_counts: Record<string, number>; // 入力数n → LUT数(JSONキーは文字列化)
  logic_depth: number;
  wire_count: number;
  group_sums: number;
  node_counts: Record<string, number>; // ノード型 → 個数
  lowered: boolean;
  num_params: number;
  partial: boolean; // blackbox含有で部分値
}

/** 現在のIRのリソース見積り+学習パラメータ数を取得する。IR未設定は404 → ApiError。*/
export function getStats(): Promise<StatsResponse> {
  return request<StatsResponse>('/api/stats');
}

export function createDemoModel(k: number, layers: number): Promise<ModelResponse> {
  return post('/api/models/demo', { k, layers });
}

export function importModel(path: string): Promise<ModelResponse> {
  return post('/api/models/import', { path });
}

export function importPythonModel(
  path: string,
  opts?: { symbol?: string | null; mode?: 'auto' | 'display' },
): Promise<PythonModelResponse> {
  return post('/api/models/import/python', {
    path,
    symbol: opts?.symbol ?? null,
    mode: opts?.mode ?? 'auto',
  });
}

export function getCurrentIR(): Promise<{ ir: IRGraphJson; dirty: boolean }> {
  return request('/api/models/current/ir');
}

export function getModelHierarchy(): Promise<HierarchyResponse> {
  return request('/api/models/current/hierarchy');
}

export function getLowering(nodeId: string, level: 'L0' | 'L1' | 'L2' = 'L0'): Promise<LoweringView> {
  const qs = new URLSearchParams({ node_id: nodeId, level });
  return request(`/api/models/current/lower?${qs.toString()}`);
}

export function putCurrentIR(ir: IRGraphJson): Promise<{ ok: boolean; warnings: string[] }> {
  return put('/api/models/current/ir', { ir });
}

export interface NeuronEditResponse {
  ok: boolean;
  model_updated: boolean; // ビルド済みモデルへin-place反映されたか
}

export function postNodeTruthTable(
  nodeId: string,
  neuron: number,
  ttHex: string,
): Promise<NeuronEditResponse & { tt_hex: string }> {
  return post(`/api/models/current/nodes/${encodeURIComponent(nodeId)}/truth_table`, {
    neuron,
    tt_hex: ttHex,
  });
}

export function postNodeIndices(
  nodeId: string,
  neuron: number,
  slots: number[],
): Promise<NeuronEditResponse> {
  return post(`/api/models/current/nodes/${encodeURIComponent(nodeId)}/indices`, {
    neuron,
    slots,
  });
}

/** 配線生成。バックエンドの make_connections で indices を作る純関数 API */
export interface WiringGenerateRequest {
  kind: string;
  in_width: number;
  out_width: number;
  n_inputs: number;
  spec: Record<string, unknown>;
  dedupe: boolean;
}

export interface WiringGenerateResponse {
  kind: string;
  indices: number[][]; // (n_inputs, out_width)
  stats: {
    coverage: number;
    usage_hist: Record<string, number>;
    same_input: number;
    n_gates: number;
  };
}

export function postWiringGenerate(
  req: WiringGenerateRequest,
): Promise<WiringGenerateResponse> {
  return post('/api/wiring/generate', req);
}

export function buildModel(): Promise<ModelResponse> {
  return post('/api/models/current/build');
}

export function generateCode(): Promise<{ code: string; filename: string }> {
  return post('/api/models/current/codegen');
}

export function saveIRFile(path: string): Promise<{ path: string }> {
  return post('/api/models/current/save', { path });
}

export function loadIRFile(path: string): Promise<ModelResponse> {
  return post('/api/models/ir/load', { path });
}

export function startTraining(params: TrainStartParams): Promise<{ session_id: string; state: string }> {
  return post('/api/train/start', params);
}

export function trainControl(sid: string, action: 'pause' | 'resume' | 'stop'): Promise<TrainStatus> {
  return post(`/api/train/${sid}/${action}`);
}

export function getTrainStatus(sid: string): Promise<TrainStatus> {
  return request(`/api/train/${sid}/status`);
}

/** 全ライブセッション(アクティブ+直近終了)を新しい順に取得する(並行run一覧用)。 */
export function getSessions(): Promise<{ sessions: TrainStatus[] }> {
  return request('/api/train/sessions');
}

/** 指定runの学習済み重みを編集中モデルへ採用し、再エクスポート後のIRを返す。 */
export function applyRun(sid: string): Promise<{ ok: boolean; ir: IRGraphJson }> {
  return post(`/api/train/${sid}/apply`);
}

// --- 学習監視基盤 ---

export function trainStepN(sid: string, n: number): Promise<TrainStatus> {
  return post(`/api/train/${sid}/step`, { n });
}

export function getTrainHistoryIndex(sid: string): Promise<HistoryIndex> {
  return request(`/api/train/${sid}/history`);
}

export function getTrainHistoryStep(
  sid: string,
  step: number,
  opts?: { layers?: string[]; fields?: string[] },
): Promise<HistoryStepView> {
  const params = new URLSearchParams();
  if (opts?.layers?.length) params.set('layers', opts.layers.join(','));
  if (opts?.fields?.length) params.set('fields', opts.fields.join(','));
  const qs = params.toString();
  return request(`/api/train/${sid}/history/${step}${qs ? `?${qs}` : ''}`);
}

export function trainRestore(sid: string, step: number): Promise<RestoreResult> {
  return post(`/api/train/${sid}/restore`, { step, mode: 'branch' });
}

// --- 学習の一般化(B2)+再起動検出 ---

export function getHealth(): Promise<{ status: string; server_epoch: number }> {
  return request('/api/health');
}

/** 取込データセットのみ(GET /api/datasets/imported)。失敗時は空配列。*/
export function getImportedDatasets(): Promise<DatasetInfo[]> {
  return request<DatasetInfo[]>('/api/datasets/imported').catch(() => []);
}

/**
 * データセットカタログ(組込 + 取込)を合成して返す。
 * main.py が GET /api/datasets へ取込分を合成済みでも重複しないよう name で重複排除する。
 * 取込一覧の取得失敗は無害(組込カタログのみ返す)。
 */
export async function getDatasets(): Promise<DatasetInfo[]> {
  const [base, imported] = await Promise.all([
    request<DatasetInfo[]>('/api/datasets'),
    getImportedDatasets(),
  ]);
  const byName = new Map<string, DatasetInfo>();
  for (const d of base) byName.set(d.name, d);
  for (const d of imported) byName.set(d.name, d);
  return [...byName.values()];
}

/** slug部(imported: プレフィックス除去)。取込用エンドポイントのパスに使う。 */
function importedSlug(datasetId: string): string {
  return datasetId.startsWith('imported:') ? datasetId.slice('imported:'.length) : datasetId;
}

/**
 * サンプル閲覧。取込データセット(imported:)は取込用エンドポイントへ、
 * それ以外は既存 /api/datasets/{id}/samples へ振り分ける。
 */
export function getDatasetSamples(
  datasetId: string,
  opts?: { page?: number; pageSize?: number; split?: string },
): Promise<DatasetSamplesResponse> {
  const params = new URLSearchParams();
  if (opts?.page != null) params.set('page', String(opts.page));
  if (opts?.pageSize != null) params.set('page_size', String(opts.pageSize));
  const qs = params.toString();
  if (datasetId.startsWith('imported:')) {
    if (opts?.split) params.set('split', opts.split);
    const q2 = params.toString();
    return request(
      `/api/datasets/imported/${importedSlug(datasetId)}/samples${q2 ? `?${q2}` : ''}`,
    );
  }
  return request(`/api/datasets/${datasetId}/samples${qs ? `?${qs}` : ''}`);
}

/** CSV取込。multipartでアップロードし登録済みDatasetInfoを返す。*/
export function importCsvDataset(
  file: File,
  name?: string,
): Promise<{ dataset: DatasetInfo }> {
  const form = new FormData();
  form.append('file', file);
  if (name) form.append('name', name);
  return request('/api/datasets/import/csv', { method: 'POST', body: form });
}

/** 画像フォルダ(zip)取込。前処理パラメータ付き。*/
export function importImagesDataset(
  file: File,
  opts?: { name?: string; resize?: number; grayscale?: boolean; binarize?: number | null },
): Promise<{ dataset: DatasetInfo; classes: string[] }> {
  const form = new FormData();
  form.append('file', file);
  if (opts?.name) form.append('name', opts.name);
  if (opts?.resize != null) form.append('resize', String(opts.resize));
  if (opts?.grayscale != null) form.append('grayscale', String(opts.grayscale));
  if (opts?.binarize != null) form.append('binarize', String(opts.binarize));
  return request('/api/datasets/import/images', { method: 'POST', body: form });
}

/** 取込データセットのsplit/augmentation/encoding更新。*/
export function updateImportedDatasetConfig(
  datasetId: string,
  config: ImportedDatasetConfig,
): Promise<DatasetInfo & ImportedDatasetConfig> {
  return put(`/api/datasets/imported/${importedSlug(datasetId)}/config`, config);
}

/** 入力エンコーディングプレビュー。実数値列→0/1グリッド。*/
export function encodingPreview(
  values: number[],
  opts?: { mode?: string; threshold?: number; bits?: number },
): Promise<EncodingPreviewResponse> {
  return post('/api/datasets/encoding-preview', {
    values,
    mode: opts?.mode ?? 'binarize',
    threshold: opts?.threshold ?? 0.5,
    bits: opts?.bits ?? 4,
  });
}

/** 最新hard evalの混同行列(classificationのみ。未評価/非分類は404)。*/
export interface ConfusionResponse {
  step: number | null;
  classes: number;
  matrix: number[][];
  val_acc: number | null;
}

export function getTrainConfusion(sid: string): Promise<ConfusionResponse> {
  return request(`/api/train/${sid}/confusion`);
}

/** 確信度が低いゲート1件(選択ギャップ・ダッシュボード §4-1)。 */
export interface UncertainGate {
  layer_id: string;
  neuron_index: number;
  entropy: number;
  gate_id: number;
}

export interface UncertainGatesResponse {
  step: number;
  gates: UncertainGate[];
}

/** 低確信ゲートTop-N(entropy降順、§4-1)。 */
export function getUncertainGates(
  sid: string,
  topN = 20,
): Promise<UncertainGatesResponse> {
  return request(`/api/train/${sid}/uncertain_gates?top_n=${topN}`);
}

/**
 * データセットとモデルI/O幅の不一致理由(一致ならnull)。
 * classificationは出力=クラス数のGroupSum終端、binary_vectorは出力=ビット幅を想定。
 */
export function datasetMismatchReason(
  ds: DatasetInfo,
  inputWidth: number | undefined,
  outputWidth: number | undefined,
): string | null {
  if (inputWidth == null || outputWidth == null) return 'モデルを読み込んでください';
  const parts: string[] = [];
  if (ds.in_width !== inputWidth) {
    parts.push(`入力幅${ds.in_width}が必要(現在${inputWidth})`);
  }
  if (ds.out_width !== outputWidth) {
    parts.push(`出力幅${ds.out_width}が必要(現在${outputWidth})`);
  }
  return parts.length > 0 ? parts.join('・') : null;
}

/** タスク種別に応じた損失の選択肢(backend tasks.py と対応)。*/
export function lossOptionsForKind(kind: string): string[] {
  return kind === 'classification' ? ['ce'] : ['bce', 'mse'];
}

/** 学習手法が選択不能な理由(選択可ならnull)。apはclassification専用。*/
export function methodDisabledReason(method: string, kind: string): string | null {
  if (method === 'ap' && kind !== 'classification') {
    return 'AP法はclassification(GroupSum終端)専用です';
  }
  return null;
}

// --- runレジストリ ---

export interface RunSummary {
  run_id: string;
  started_at: number | null;
  ended_at: number | null;
  state: string;
  dataset: string;
  method: string;
  loss: string;
  optimizer: string;
  lr: number;
  batch_size: number;
  epochs: number;
  final_step: number;
  final_loss: number | null;
  final_acc: number | null;
  model_name: string;
  wandb_url: string | null;
}

export interface RunRecord extends RunSummary {
  series: { steps: number[]; loss: number[]; acc: number[] };
  ir: IRGraphJson | null;
}

export function getRuns(): Promise<{ runs: RunSummary[] }> {
  return request('/api/runs');
}

export function getRun(runId: string): Promise<RunRecord> {
  return request(`/api/runs/${encodeURIComponent(runId)}`);
}

// --- Shipレンズ(HDL出荷プリフライト/エクスポート/検証、§4-3) ---

export type HdlLang = 'verilog' | 'vhdl';

/** 現IRのHDL出荷プリフライト(可否・ブロッカー・見積り・draft数)。 */
export function getPreflight(): Promise<PreflightResponse> {
  return request('/api/models/current/preflight');
}

/** 現IRからHDLソースと見積りを生成する。 */
export function exportHdl(lang: HdlLang, lower = true): Promise<HdlExportResponse> {
  return post('/api/export/hdl', { lang, lower });
}

/** 生成HDLとgolden modelの等価性を(処理系があれば)検証する。 */
export function verifyHdl(lang: HdlLang): Promise<HdlVerifyResponse> {
  return post('/api/export/hdl/verify', { lang });
}

// --- 学習設定スキーマ(CONFIGタブのスキーマ駆動フォーム用) ---

/** TrainConfigのJSON Schema(pydantic model_json_schema形式、$defs+$ref含む)。 */
export function getConfigSchema(): Promise<Record<string, unknown>> {
  return request('/api/config/schema');
}

// --- プロジェクトCRUD(CONFIGタブのプロジェクト管理用) ---

/** プロジェクト一覧の1件分サマリ(GET /api/projects)。 */
export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
}

/** プロジェクト全体(GET/POST/PUT /api/projects/...)。 */
export interface ProjectResponse {
  id: string;
  name: string;
  description: string;
  networks: Record<string, unknown>;
  train_configs: Record<string, unknown>;
  dataset_refs: string[];
  run_refs: string[];
  optimizers: Record<string, unknown>;
  monitors: Record<string, unknown>;
  executors: Record<string, unknown>;
  created_at: number;
  updated_at: number;
  deleted: boolean;
}

/** プロジェクトのPUT時の部分更新パッチ(省略項目は既存値を維持)。 */
export interface ProjectUpdatePatch {
  name?: string;
  description?: string;
  networks?: Record<string, unknown>;
  train_configs?: Record<string, unknown>;
  dataset_refs?: string[];
  run_refs?: string[];
  optimizers?: Record<string, unknown>;
  monitors?: Record<string, unknown>;
  executors?: Record<string, unknown>;
}

export function listProjects(): Promise<ProjectSummary[]> {
  return request('/api/projects');
}

export function createProject(name: string, description = ''): Promise<ProjectResponse> {
  return post('/api/projects', { name, description });
}

export function getProject(id: string): Promise<ProjectResponse> {
  return request(`/api/projects/${encodeURIComponent(id)}`);
}

export function updateProject(id: string, patch: ProjectUpdatePatch): Promise<ProjectResponse> {
  return put(`/api/projects/${encodeURIComponent(id)}`, patch);
}

export function deleteProject(id: string): Promise<{ ok: boolean; id: string }> {
  return request(`/api/projects/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

/** モデルテンプレート1件のサマリ(GET /api/templates)。*/
export interface TemplateSummary {
  id: string;
  name: string;
  description: string;
  category: string;
  /** 腐敗防止CIの検証水準(smoke/build/transfer/validate)。*/
  verification: 'smoke' | 'build' | 'transfer' | 'validate';
  recommended_dataset: string | null;
}

/** テンプレ一覧を取得する。*/
export function getTemplates(): Promise<{ templates: TemplateSummary[] }> {
  return request('/api/templates');
}

/** テンプレをプロジェクトとして生成・保存し、その project payload を返す。*/
export function instantiateTemplate(id: string, name?: string): Promise<ProjectResponse> {
  return post(`/api/templates/${encodeURIComponent(id)}/instantiate`, { name: name ?? null });
}

// --- 事前学習モデルのインポート&変換ウィザード ---

/** 変換プリセット一覧を取得(GET /api/convert/presets)。 */
export function listConvertPresets(): Promise<{ presets: ConvertPresetInfo[] }> {
  return request('/api/convert/presets');
}

/**
 * 変換を開始(POST /api/convert/start)。
 * 409=学習中(排他)、400/422=不正リクエストはApiErrorで投げられる。
 */
export function startConversion(params: ConvertStartRequest): Promise<{ convert_id: string }> {
  return post('/api/convert/start', params);
}

/** 変換セッションの状態を取得(GET /api/convert/{cid}/status)。 */
export function getConversionStatus(cid: string): Promise<ConvertStatus> {
  return request(`/api/convert/${encodeURIComponent(cid)}/status`);
}

/** 変換のキャンセルを要求(POST /api/convert/{cid}/cancel)。 */
export function cancelConversion(cid: string): Promise<{ ok: boolean }> {
  return post(`/api/convert/${encodeURIComponent(cid)}/cancel`);
}
