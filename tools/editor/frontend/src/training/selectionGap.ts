// 選択ギャップ・ダッシュボード(§4-1)の純粋ヘルパ。
// soft(連続緩和)精度とhard(離散化)精度の乖離=選択ギャップ、
// ゲート分布エントロピーの正規化(迷いヒートストリップの色強度)を扱う。

/**
 * バックエンドが配信する正規化迷い(entropy_mean等)を[0,1]へクランプする。
 * 正規化自体はパラメータ化と分布サイズを知るbackend(extract.neuron_uncertainty)
 * の責務。1に近い=迷いが大きい(確信度が低い)。
 */
export function clampUncertainty(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export interface GapSummary {
  soft: number;
  hard: number;
  gap: number;
}

/**
 * soft/hardの検証精度スパース系列から最新の選択ギャップを求める。
 * どちらかが空ならnull。gap = soft - hard(hard化で失う精度)。
 */
export function latestGap(
  valAccSoft: ReadonlyArray<readonly [number, number]>,
  valAcc: ReadonlyArray<readonly [number, number]>,
): GapSummary | null {
  if (valAccSoft.length === 0 || valAcc.length === 0) return null;
  const soft = valAccSoft[valAccSoft.length - 1][1];
  const hard = valAcc[valAcc.length - 1][1];
  return { soft, hard, gap: soft - hard };
}
