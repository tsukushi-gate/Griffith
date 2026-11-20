// エッジ基底スタイル(バス幅→線幅、信号種→色)。
// 出典: tools/editor/frontend/src/editor/edgeStyle.ts の純関数部を移植
// (buildEdgeBaseStyles は IRGraphJson 依存のため meta ベースで再実装)。
import type { CSSProperties } from 'react';

export const MIN_EDGE_WIDTH = 1;
export const MAX_EDGE_WIDTH = 4;

/** ビット幅→線の太さ。1bit=1px、2倍ごとに+1px、上限 MAX_EDGE_WIDTH。 */
export function bitWidthToStrokeWidth(width: number | null | undefined): number {
  if (width == null || !Number.isFinite(width) || width <= 1) return MIN_EDGE_WIDTH;
  return Math.min(MIN_EDGE_WIDTH + Math.log2(width), MAX_EDGE_WIDTH);
}

export function edgeBaseStyle(width: number | null | undefined): CSSProperties {
  return {
    stroke: 'var(--edge-default)',
    strokeWidth: bitWidthToStrokeWidth(width),
  };
}

/** 1bit エッジの実行値スタイル(RTL 値で点灯)。 */
export function edgeValueStyle(bit: 0 | 1): CSSProperties {
  return {
    stroke: bit ? 'var(--edge-on)' : 'var(--edge-off)',
    strokeWidth: MIN_EDGE_WIDTH + 0.5,
  };
}
