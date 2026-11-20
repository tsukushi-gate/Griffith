// 混同行列からのメトリクス算出。純関数のみ(RTL非依存でユニットテスト可)。
// 規約: confusion[label][pred] = 件数(バックエンド evaluate.py と一致)。

export interface ConfusionMetrics {
  /** クラス数 */
  n: number;
  /** クラス別 precision(予測iのうち正解の割合)。予測0件は0 */
  precision: number[];
  /** クラス別 recall(実際iのうち正解の割合)。実際0件は0 */
  recall: number[];
  /** クラス別 F1 */
  f1: number[];
  /** 全体正解率 */
  accuracy: number;
  /** マクロ平均 precision */
  avgPrecision: number;
  /** マクロ平均 recall */
  avgRecall: number;
  /** クラス別の実際件数(行和) */
  support: number[];
}

/** confusion[label][pred] から precision/recall/F1/accuracy を計算する。 */
export function computeConfusionMetrics(confusion: number[][]): ConfusionMetrics {
  const n = confusion.length;
  const precision = new Array<number>(n).fill(0);
  const recall = new Array<number>(n).fill(0);
  const f1 = new Array<number>(n).fill(0);
  const support = new Array<number>(n).fill(0);
  let correct = 0;
  let total = 0;

  const colSum = new Array<number>(n).fill(0);
  for (let label = 0; label < n; label++) {
    for (let pred = 0; pred < n; pred++) {
      const c = confusion[label][pred];
      colSum[pred] += c;
      support[label] += c;
      total += c;
      if (label === pred) correct += c;
    }
  }

  for (let i = 0; i < n; i++) {
    const tp = confusion[i][i];
    precision[i] = colSum[i] > 0 ? tp / colSum[i] : 0;
    recall[i] = support[i] > 0 ? tp / support[i] : 0;
    const p = precision[i];
    const r = recall[i];
    f1[i] = p + r > 0 ? (2 * p * r) / (p + r) : 0;
  }

  const avgPrecision = n > 0 ? precision.reduce((a, b) => a + b, 0) / n : 0;
  const avgRecall = n > 0 ? recall.reduce((a, b) => a + b, 0) / n : 0;
  const accuracy = total > 0 ? correct / total : 0;

  return { n, precision, recall, f1, accuracy, avgPrecision, avgRecall, support };
}

/** 評価行から混同行列を組み立てる(classificationのみ、フロント集計フォールバック)。 */
export function buildConfusion(
  items: { label: number | null; pred: number | null }[],
  numClasses: number,
): number[][] {
  const m = Array.from({ length: numClasses }, () => new Array<number>(numClasses).fill(0));
  for (const it of items) {
    if (it.label == null || it.pred == null) continue;
    if (it.label < numClasses && it.pred < numClasses) m[it.label][it.pred] += 1;
  }
  return m;
}
