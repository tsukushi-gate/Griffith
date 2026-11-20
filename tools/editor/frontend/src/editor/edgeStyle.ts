// エッジ型の色分け(§5、ComfyUI批判からの教訓)。
// ビット幅→線の太さ、信号種→色相を「基底スタイル」として供給する。
// 動的着色(診断エラー/実行値/逆伝播/反例トレース)が無いエッジにのみ適用する。

import type { CSSProperties } from 'react';
import type { IRGraphJson, IRNodeType } from '../ir/schema';
import { edgeSlot } from '../ir/schema';
import { nodeWidths } from '../ir/shapeInference';

/** 信号種: 論理(通常)/実数域(blackbox・ToLogic境界の実数側) */
export type SignalKind = 'logic' | 'real';

/** 太さの下限/上限(px)。1bit=下限、log2スケールで上限へ漸近する。 */
export const MIN_EDGE_WIDTH = 1;
export const MAX_EDGE_WIDTH = 4;

/** ビット幅→線の太さ。1bit=1px、2倍ごとに+1px、上限MAX_EDGE_WIDTH。 */
export function bitWidthToStrokeWidth(width: number | null | undefined): number {
  if (width == null || !Number.isFinite(width) || width <= 1) return MIN_EDGE_WIDTH;
  return Math.min(MIN_EDGE_WIDTH + Math.log2(width), MAX_EDGE_WIDTH);
}

/** ノード出力の信号種。blackboxとadd(sum)は実数域、他は論理。 */
export function nodeOutputKind(type: string | undefined, addMode?: string): SignalKind {
  if (type === 'blackbox') return 'real';
  if (type === 'add' && addMode === 'sum') return 'real';
  return 'logic';
}

/** ノード入力の信号種。blackbox入力とbinarize入力(ToLogic境界の実数側)は実数域。 */
export function nodeInputKind(type: string | undefined): SignalKind {
  if (type === 'blackbox' || type === 'binarize') return 'real';
  return 'logic';
}

/** エッジの信号種。ソース出力が実数、またはbinarizeへの入力(ToLogic境界)なら実数域。 */
export function edgeSignalKind(
  srcType: string | undefined,
  dstType: string | undefined,
  srcAddMode?: string,
): SignalKind {
  if (nodeOutputKind(srcType, srcAddMode) === 'real') return 'real';
  if (nodeInputKind(dstType) === 'real') return 'real';
  return 'logic';
}

/** 基底エッジスタイル(信号種→色相、ビット幅→太さ)。 */
export function edgeBaseStyle(
  width: number | null | undefined,
  kind: SignalKind,
): CSSProperties {
  return {
    stroke: kind === 'real' ? 'var(--edge-real)' : 'var(--edge-default)',
    strokeWidth: bitWidthToStrokeWidth(width),
  };
}

/** 実数域ハンドル(ポート)の装飾。論理ハンドルは既定表示のままにする。 */
export function realHandleStyle(): CSSProperties {
  return { background: 'var(--edge-real)', borderColor: 'var(--edge-real)' };
}

/**
 * IR全エッジのトップレベル基底スタイルマップ(edgeId→CSSProperties)。
 * edgeIdは irToGraph.buildEditorEdge と同じ規約(slot>0で `#s{slot}` 付与)。
 */
export function buildEdgeBaseStyles(ir: IRGraphJson | null): Map<string, CSSProperties> {
  const map = new Map<string, CSSProperties>();
  if (!ir) return map;
  const portWidth = new Map<string, number>();
  for (const p of ir.inputs) portWidth.set(p.id, p.width);
  const outWidth = new Map<string, number | null>();
  const typeById = new Map<string, IRNodeType>();
  const addModeById = new Map<string, string | undefined>();
  for (const n of ir.nodes) {
    outWidth.set(n.id, nodeWidths(n).out);
    typeById.set(n.id, n.type);
    if (n.type === 'add') addModeById.set(n.id, n.params.mode as string | undefined);
  }
  const srcWidth = (id: string): number | null =>
    portWidth.get(id) ?? outWidth.get(id) ?? null;
  // 入力ポートは 'input' 扱い(信号種は論理)、それ以外はノード型を引く
  const srcType = (id: string): string | undefined =>
    portWidth.has(id) ? 'input' : typeById.get(id);
  for (const e of ir.edges) {
    const slot = edgeSlot(e);
    const id = slot > 0 ? `${e.src}->${e.dst}#s${slot}` : `${e.src}->${e.dst}`;
    const kind = edgeSignalKind(srcType(e.src), typeById.get(e.dst), addModeById.get(e.src));
    map.set(id, edgeBaseStyle(srcWidth(e.src), kind));
  }
  return map;
}
