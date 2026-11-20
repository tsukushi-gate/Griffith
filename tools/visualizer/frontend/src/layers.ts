/**
 * deck.glレイヤー定義。
 *
 * ノード（ScatterplotLayer）、エッジ（LineLayer）、テキスト（TextLayer）の
 * 3レイヤーを構成。ズームレベルに応じたLODで表示/非表示を切り替える。
 */

import { ScatterplotLayer, LineLayer, TextLayer } from '@deck.gl/layers'
import type { NodeAccessor, EdgeAccessor } from './layout'
import { GATE_NAMES } from './layout'

/** ノードの描画半径 */
const NODE_RADIUS = 1.2

/** ズームしきい値 */
const EDGE_ZOOM_THRESHOLD = -3    // これ以上ズームインするとエッジ表示
const TEXT_ZOOM_THRESHOLD = 3     // これ以上ズームインするとテキスト表示

export function createNodeLayer(
  nodes: NodeAccessor[],
  onHover?: (info: { object?: NodeAccessor }) => void,
): ScatterplotLayer<NodeAccessor> {
  return new ScatterplotLayer<NodeAccessor>({
    id: 'nodes',
    data: nodes,
    getPosition: d => d.position,
    getFillColor: d => d.color,
    getLineColor: [40, 40, 60, 200],
    getRadius: NODE_RADIUS,
    radiusUnits: 'common',
    lineWidthMinPixels: 0.5,
    filled: true,
    stroked: true,
    pickable: true,
    onHover,
    updateTriggers: {
      getFillColor: [nodes],
    },
  })
}

export function createEdgeLayer(
  edges: EdgeAccessor[],
  zoom: number,
): LineLayer<EdgeAccessor> {
  return new LineLayer<EdgeAccessor>({
    id: 'edges',
    data: edges,
    getSourcePosition: d => d.source,
    getTargetPosition: d => d.target,
    getColor: [80, 80, 140, 40],
    getWidth: 0.2,
    widthUnits: 'common',
    visible: zoom > EDGE_ZOOM_THRESHOLD,
  })
}

export function createTextLayer(
  nodes: NodeAccessor[],
  zoom: number,
): TextLayer<NodeAccessor> {
  return new TextLayer<NodeAccessor>({
    id: 'labels',
    data: nodes,
    getText: d => GATE_NAMES[d.gateId] ?? `G${d.gateId}`,
    getPosition: d => d.position,
    getSize: 10,
    sizeUnits: 'common',
    getColor: [240, 240, 255, 220],
    getTextAnchor: 'middle',
    getAlignmentBaseline: 'center',
    fontFamily: 'SF Mono, Fira Code, Consolas, monospace',
    fontWeight: 'bold',
    visible: zoom > TEXT_ZOOM_THRESHOLD,
    billboard: false,
  })
}
