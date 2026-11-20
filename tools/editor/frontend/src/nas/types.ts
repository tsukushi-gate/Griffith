// NAS(構造自動探索)API型。
// 真実源は tools/editor/backend/nas.py の pydantic モデル。
// backend schemas.py には載せない(nas.py内定義=parent指示)ため、
// 生成 types.ts ではなくこのファイルでフロント側の型を持つ。

import type { IRGraphJson } from '../ir/schema';

export interface NasGenome {
  layers: number;
  width: number;
  n_inputs: number;
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
  genome: NasGenome;
}

export interface NasStatus {
  search_id: string | null;
  state: string; // 'idle'|'running'|'done'|'stopped'|'failed'
  n_trials: number;
  max_trials: number | null;
  pareto: number[];
  error: string | null;
}

export interface NasTrials {
  search_id: string | null;
  trials: NasTrialRow[];
  pareto: number[];
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
  layers_choices?: number[];
  width_choices?: number[];
  n_inputs_choices?: number[];
}

export interface NasStartResponse {
  search_id: string;
  state: string;
}

export interface NasIrResponse {
  trial_id: number;
  ir: IRGraphJson;
}
