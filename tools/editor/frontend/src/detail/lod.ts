// セマンティックズームのLOD(Level of Detail)判定(§3)。
// 「ノード数しきい値」ではなく「画面上の見かけサイズ」で L2(俯瞰)/L1(タイル)/
// L0(ゲート)を決める純関数群。ズームとゲート実寸から段階を、可視領域と
// 折りたたみ状態から自動展開対象を算出する。副作用・React依存を持たない
// (vitestで固定するため)。予算(DETAIL_TOTAL_MAX)の最終判定は
// composeDetailGraph 側に委ねる。

import { GATE_W } from './detailConfig';

/** セマンティックズームの3段階。俯瞰→タイル→ゲートの順にズームインで深くなる。 */
export type LodStage = 'overview' | 'tile' | 'gate';

/** ゲートカード実寸(GATE_W flow単位)が画面上でこのpx以上ならL0(ゲート展開)。
    設計の「個別ゲートが視認可能サイズ」を、詳細ビューのゲートカード幅基準で較正した値
    (カードは122px幅のリッチカードのため、内容が読める~48px以上を展開の目安とする)。 */
export const GATE_VISIBLE_PX = 48;

/** ゲートカードの見かけサイズがこのpx以上ならL1(タイル表示)。俯瞰→タイルの境界。 */
export const TILE_VISIBLE_PX = 20;

/** ズーム時のゲートカードの見かけサイズ(px)。 */
export function apparentGatePx(zoom: number, gateWidth: number = GATE_W): number {
  return zoom * gateWidth;
}

/** ズームレベル(とゲート実寸)から現在のLOD段階を求める。 */
export function lodStageForZoom(zoom: number, gateWidth: number = GATE_W): LodStage {
  const px = apparentGatePx(zoom, gateWidth);
  if (px >= GATE_VISIBLE_PX) return 'gate';
  if (px >= TILE_VISIBLE_PX) return 'tile';
  return 'overview';
}

/** flow座標系の矩形。 */
export interface FlowRect {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** LOD判定に必要な1レイヤ分の情報(flow座標の矩形+展開可否+折りたたみ状態)。 */
export interface LodLayerInfo {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** out_width <= DETAIL_MANUAL_MAX(手動展開の上限内) */
  expandable: boolean;
  /** 折りたたみグループのメンバー(自動展開の対象外にする) */
  collapsed: boolean;
}

/** viewport({x,y,zoom})とビューポートのCSSピクセル寸法からflow座標の可視矩形を求める。
    flow座標→画面: screen = flow*zoom + offset。逆変換で可視範囲の四隅をflow座標へ戻す。 */
export function viewportToFlowRect(
  viewport: { x: number; y: number; zoom: number },
  widthPx: number,
  heightPx: number,
): FlowRect {
  const { x, y, zoom } = viewport;
  const z = zoom || 1;
  return {
    minX: (0 - x) / z,
    minY: (0 - y) / z,
    maxX: (widthPx - x) / z,
    maxY: (heightPx - y) / z,
  };
}

/** レイヤ矩形が可視矩形と交差するか(AABB判定)。 */
export function layerIntersectsView(layer: LodLayerInfo, view: FlowRect): boolean {
  return (
    layer.x < view.maxX &&
    layer.x + layer.w > view.minX &&
    layer.y < view.maxY &&
    layer.y + layer.h > view.minY
  );
}

/** LODが自動展開すべきレイヤidの集合を求める。
    条件: (1)ズームがゲート段階 (2)展開可能 (3)折りたたみ外 (4)可視領域と交差。
    全レイヤ一括ではなく可視領域ベースで絞る(§3)。総ノード数上限の最終判定は
    composeDetailGraph が行うため、ここでは予算を考慮しない。 */
export function selectAutoExpandLayers(
  layers: LodLayerInfo[],
  zoom: number,
  view: FlowRect,
  gateWidth: number = GATE_W,
): Set<string> {
  if (lodStageForZoom(zoom, gateWidth) !== 'gate') return new Set();
  const out = new Set<string>();
  for (const l of layers) {
    if (!l.expandable || l.collapsed) continue;
    if (!layerIntersectsView(l, view)) continue;
    out.add(l.id);
  }
  return out;
}

/** 2つの集合が同一内容か(要素順不問)。onMove毎の再合成を抑えるための安定化に使う。 */
export function sameIdSet(a: Set<string>, b: Set<string>): boolean {
  if (a === b) return true;
  if (a.size !== b.size) return false;
  for (const id of a) if (!b.has(id)) return false;
  return true;
}
