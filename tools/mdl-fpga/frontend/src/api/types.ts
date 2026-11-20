// バックエンド契約(tools/mdl-fpga/mdl_fpga/viewer/{session,probes,persistence}.py と一致)

export interface MetaNode {
  id: string;
  wire?: string;
  width: number;
  type: string;
  alias?: boolean;
  score_bits?: number | null;
  stage?: number | null;
  path?: string | null;   // DUT相対の信号パス(システムモード)
  label?: string | null;  // 表示名(システムノード)
}

export interface MetaPort {
  id: string;
  width: number;
  shape?: number[] | null;
  semantics?: string | null;
}

export interface MetaEdge {
  src: string;
  dst: string;
  slot: number;
}

export interface Meta {
  name: string;
  in_width: number;
  out_width: number;
  score_bits: number | null;
  k: number | null;
  input_semantics: string | null;
  input_shape: number[] | null;
  nodes: MetaNode[];
  inputs: MetaPort[];
  outputs: MetaPort[];
  edges: MetaEdge[];
  cycle: number;
  workdir: string;
  mode?: 'core' | 'system';
  latency?: number;
  pipelined?: boolean;
  allow_set_x?: boolean;
  n_images?: number | null;
  epoch?: number; // セッション世代(プロジェクト切替ごとに+1)
}

export type ProbePayload =
  | { kind: 'error'; message: string }
  | { kind: 'text'; text: string }
  | { kind: 'bits'; hex: string; width: number }
  | { kind: 'bars'; values: number[]; labels: string[] | null; argmax: number | null }
  | { kind: 'image'; data_url: string; w: number; h: number }
  | { kind: 'queue'; depth: number;
      items: { cycle: number; payload: ProbePayload }[] };

export interface ProbeArgs extends Record<string, unknown> {
  cols?: number;            // bits/image の横幅(bit数)
  order?: 'msb' | 'lsb';    // bits の並び(既定 msb = [msb:lsb])
}

export interface ProbeSpec {
  id: string;
  target: string; // lowered ノード id | 'x' | 'y' | 'y_comb'
  slice: [number, number] | null; // [msb, lsb]
  renderer: string;
  args: ProbeArgs;
  pos: { x: number; y: number };
  pinned: boolean;
  label: string | null;
  depth: number;   // 表示キュー段数(既定1)
  scale: number;   // カード個別の拡大率(既定1)
}

export interface ViewerConfig {
  version: number;
  probes: ProbeSpec[];
  positions: Record<string, { x: number; y: number }>;
  expanded: string[];
  viewport: { x: number; y: number; zoom: number } | null;
  board?: string | null; // 選択ボード key(サーバ側マージで保持される)
}

// --- プロジェクト(viewer/appstate.py と一致) ---

export interface RecentEntry {
  model: string;
  name?: string | null;
  system?: string | null;
  pipelined?: boolean;
  opened_at?: string;
}

export interface ProjectInfo {
  model: string;
  name: string | null;
  mode: string;
  system_dir: string | null;
  pipelined: boolean;
  epoch: number;
  recent: RecentEntry[];
}

export interface ExampleMode {
  mode: string;
  ready: boolean;
  system_dir?: string;
  hint?: string;
}

export interface ExampleEntry {
  key: string;
  name: string;
  path: string;
  modes: ExampleMode[];
}

export interface OpenRequest {
  example?: string;
  mode?: string;
  model?: string;
  system?: string;
  pipelined?: boolean;
}

export interface OpenResponse {
  ok: boolean;
  name: string | null;
  mode: string;
  epoch: number;
}

// --- ボード(board_db v2 / advisor / detect と一致。防御的に緩い型) ---

export interface BoardDoc {
  key: string;
  name: string;
  tier: number;
  unverified?: boolean;
  device: string;
  family: string;
  detection?: {
    silk_hint?: string | null;
    [k: string]: unknown;
  } | null;
  resources: { unit?: string; [k: string]: unknown };
  [k: string]: unknown;
}

export interface Advice {
  gates: number;
  spatial_ok: boolean;
  overlay_ok: boolean | null;
  spatial_alm_needed: number;
  messages: string[];
  unit: string;
  calibrated: boolean;
}

export interface BoardState {
  key: string | null;
  advice: Advice | null;
  advice_error?: string;
}

export interface DetectCandidate {
  key: string;
  name: string;
  tier: number;
  confidence: number;
  reasons: string[];
  silk_hint?: string | null;
  unverified?: boolean;
}

export interface DetectResultDoc {
  observation: {
    cables: string[];
    jtag: { idcode: string; part_text: string; via: string }[];
    serial: { device: string; description: string }[];
    notes: string[];
    [k: string]: unknown;
  };
  candidates: DetectCandidate[];
  best: DetectCandidate | null;
}

export interface SimFrame {
  type: 'values';
  ok: boolean;
  cycle: number;
  values: Record<string, string>; // node id / x / y / y_comb → hex(非パディング)
  xz: string[];
  probes: Record<string, ProbePayload>;
  latency_ms?: number;
  error?: string;
  steps?: number;      // step_until が進めたクロック数
  changed?: boolean;   // step_until: 値変化で停止したか(false=上限打ち切り)
  watch?: string;      // step_until の監視対象
}

export interface RendererListing {
  builtins: string[];
  sidecar: string[];
  sidecar_path: string;
  sidecar_error: string | null;
}

export interface NodeDetail {
  id: string;
  type: string;
  params: Record<string, unknown>;
}
