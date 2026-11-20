// 生成型(api/generated/)と手書き型の互換チェック(段階1)。
// 代表3型(TrainStatus / MetricsFrame / DatasetInfo)について、
//   - コンパイル時: 同一の代表値が手書き型・生成型の双方に代入可能(構造互換)。
//     vitestはesbuildで型検査しないため、これは `npx tsc --noEmit`(CI)で効く
//   - 実行時: schema.jsonの必須キー・type判別子が期待どおり
// を検証する。手書き型の置換(段階2)はまだ行わない。

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { MetricsFrame } from '../store/trainingStore';
import type {
  DatasetInfo as GenDatasetInfo,
  MetricsFrame as GenMetricsFrame,
  TrainStatus as GenTrainStatus,
} from './generated/types';
import type { DatasetInfo, TrainStatus } from './rest';

// --- コンパイル時チェック: 代表値の二重代入(型が非互換ならtscで落ちる) ---

const trainStatusSample = {
  session_id: 'abc123def456',
  state: 'running',
  step: 10,
  epoch: 0,
  loss: 'ce', // 現在は損失名が入る(number | string | null)
  acc: 0.5,
  error: null,
  step_budget: -1,
  method: 'bp',
  capture: { level: 'full', stride: 1 },
  history: { min_step: 1, max_step: 10, count: 10 },
  dataset: 'logic:xor',
  task: 'binary_vector',
  last_val_acc: null,
};
const trainStatusHand: TrainStatus = trainStatusSample;
const trainStatusGen: GenTrainStatus = trainStatusSample;

const metricsSample = {
  type: 'metrics' as const,
  step: 1,
  epoch: 0,
  loss: 0.6931,
  acc: 0.5,
  weight_delta: { n1_lutlayer: 0.01 },
  val_loss: 0.7,
  val_acc: 0.5,
  exact_match: 0.25,
};
const metricsHand: MetricsFrame = metricsSample;
const metricsGen: GenMetricsFrame = metricsSample;

const datasetSample = {
  name: 'logic:xor',
  label: 'XOR',
  in_width: 2,
  out_width: 1,
  kind: 'binary_vector',
};
const datasetHand: DatasetInfo = datasetSample;
const datasetGen: GenDatasetInfo = datasetSample;

// --- 実行時チェック: schema.jsonの契約 ---

interface DefSchema {
  required?: string[];
  additionalProperties?: boolean;
  properties?: Record<string, { const?: string }>;
}

const schema = JSON.parse(
  readFileSync(new URL('./generated/schema.json', import.meta.url), 'utf-8'),
) as { $defs: Record<string, DefSchema> };

describe('生成スキーマと手書き型の整合', () => {
  it('TrainStatus: 必須キーが実装(training.py status)の基本キーと一致', () => {
    expect(schema.$defs.TrainStatus.required).toEqual([
      'session_id',
      'state',
      'step',
      'epoch',
      'loss',
      'acc',
      'error',
    ]);
    expect(schema.$defs.TrainStatus.additionalProperties).toBe(false);
  });

  it('MetricsFrame: type判別子と必須キー(追加メトリクスはextra許可)', () => {
    expect(schema.$defs.MetricsFrame.properties?.type?.const).toBe('metrics');
    expect(schema.$defs.MetricsFrame.required).toEqual([
      'type',
      'step',
      'epoch',
      'loss',
      'acc',
      'weight_delta',
    ]);
    expect(schema.$defs.MetricsFrame.additionalProperties).toBe(true);
  });

  it('DatasetInfo: 全キー必須', () => {
    expect(schema.$defs.DatasetInfo.required).toEqual([
      'name',
      'label',
      'in_width',
      'out_width',
      'kind',
    ]);
  });

  it('代表値が手書き型・生成型の双方を満たす(コンパイル時チェックのアンカー)', () => {
    expect(trainStatusGen).toEqual(trainStatusHand);
    expect(metricsGen).toEqual(metricsHand);
    expect(datasetGen).toEqual(datasetHand);
  });
});
