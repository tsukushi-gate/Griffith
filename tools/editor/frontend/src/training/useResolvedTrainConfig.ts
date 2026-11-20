// 学習設定の解決フック。
// ヘッダの実行コントロール(TrainControls)と下部ドックの常設フォーム
// (TrainSettingsForm)の両方で、モデルI/O幅と整合するdataset/loss/method、
// および無効理由(disabledReason)を共有するために切り出した。
// datasetカタログはモジュールレベルのpromiseで1度だけ取得して共有する。

import { useEffect, useState } from 'react';
import {
  datasetMismatchReason,
  getDatasets,
  lossOptionsForKind,
  methodDisabledReason,
  type DatasetInfo,
} from '../api/rest';
import { hasShapeErrors } from '../ir/shapeInference';
import type { IRGraphJson } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore, type TrainingMethod } from '../store/trainingStore';

// datasetカタログは全消費側で共有(重複GETを避ける)
let datasetsPromise: Promise<DatasetInfo[]> | null = null;
function loadDatasets(): Promise<DatasetInfo[]> {
  if (!datasetsPromise) datasetsPromise = getDatasets();
  return datasetsPromise;
}

/**
 * カタログキャッシュを破棄する。DATASETタブでの取込後に呼ぶと、
 * 次に学習フォームが再マウントされた際に取込データセットを含めて再取得する。
 */
export function invalidateDatasetsCache(): void {
  datasetsPromise = null;
}

export function modelIO(ir: IRGraphJson | null): {
  inWidth?: number;
  outWidth?: number;
} {
  return { inWidth: ir?.inputs[0]?.width, outWidth: ir?.outputs[0]?.width };
}

function localTrainDisabledReason(ir: IRGraphJson): string | null {
  const editor = ir.metadata?.editor as
    | {
        model_ref?: {
          spec?: {
            capabilities?: Record<string, boolean>;
            paper_ref?: { url?: string };
          };
        };
      }
    | undefined;
  const caps = editor?.model_ref?.spec?.capabilities;
  if (caps?.trainable === false) {
    return 'このモデルはエディタ内の軽量学習ではなく、Modal/experiments/train_llvit.pyで再現学習します';
  }
  return null;
}

export function trainingDisabledReason(
  ir: IRGraphJson | null,
  shapeErrors: boolean,
  dataset: DatasetInfo | undefined,
): string | null {
  if (!ir) return 'モデルを読み込んでください';
  const localReason = localTrainDisabledReason(ir);
  if (localReason) return localReason;
  if (shapeErrors) return '形状エラーを解消してください';
  if (!dataset) return 'データセットを選択してください';
  const { inWidth, outWidth } = modelIO(ir);
  const mismatch = datasetMismatchReason(dataset, inWidth, outWidth);
  if (mismatch) return `${dataset.label}: ${mismatch}`;
  return null;
}

export interface ResolvedTrainConfig {
  datasets: DatasetInfo[];
  inWidth?: number;
  outWidth?: number;
  shapeErrors: boolean;
  /** I/O幅と整合する実効データセット(明示選択が不整合なら最初の整合するもの) */
  selectedDataset: DatasetInfo | undefined;
  kind: DatasetInfo['kind'];
  lossOptions: readonly string[];
  /** タスク種別に整合させた実効loss */
  loss: string;
  /** タスク種別に整合させた実効method */
  method: TrainingMethod;
  /** 学習開始が不可能な理由(nullなら可能) */
  disabledReason: string | null;
}

/** 学習設定を解決して返す(datasets取得+I/O整合の解決を含む) */
export function useResolvedTrainConfig(): ResolvedTrainConfig {
  const ir = useGraphStore((s) => s.ir);
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const trainConfig = useTrainingStore((s) => s.trainConfig);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  useEffect(() => {
    let disposed = false;
    loadDatasets()
      .then((items) => {
        if (!disposed) setDatasets(items);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, []);

  const shapeErrors = hasShapeErrors(diagnostics);
  const { inWidth, outWidth } = modelIO(ir);

  const chosen = datasets.find((d) => d.name === trainConfig.datasetChoice);
  const selectedDataset =
    chosen && datasetMismatchReason(chosen, inWidth, outWidth) === null
      ? chosen
      : datasets.find(
          (d) => datasetMismatchReason(d, inWidth, outWidth) === null,
        );
  const kind = selectedDataset?.kind ?? 'classification';
  const lossOptions = lossOptionsForKind(kind);
  const loss =
    trainConfig.lossChoice && lossOptions.includes(trainConfig.lossChoice)
      ? trainConfig.lossChoice
      : lossOptions[0];
  const method: TrainingMethod =
    methodDisabledReason(trainConfig.methodChoice, kind) === null
      ? trainConfig.methodChoice
      : 'bp';
  const disabledReason = trainingDisabledReason(
    ir,
    shapeErrors,
    selectedDataset,
  );

  return {
    datasets,
    inWidth,
    outWidth,
    shapeErrors,
    selectedDataset,
    kind,
    lossOptions,
    loss,
    method,
    disabledReason,
  };
}
