// deck.glレイヤー定義。tools/visualizer/frontend/src/layers.ts の移植。
// ノード(Scatterplot)・エッジ(Line)・ラベル(Text)の3層をズームLODで切り替える。

import { LineLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { GATE_NAMES } from '../ir/gates';
import { cssVarRGB, cssVarRGBA, type RGBA } from '../theme/theme';
import type { L0Edge, L0Node } from './l0Layout';

/** ノードの描画半径(common単位) */
const NODE_RADIUS = 1.2;

/** ズームしきい値(visualizer流用) */
export const EDGE_ZOOM_THRESHOLD = -3; // これ以上ズームインするとエッジ表示
export const TEXT_ZOOM_THRESHOLD = 3;  // これ以上ズームインするとテキスト表示

function nodeLabel(d: L0Node, nInputs: number): string {
  if (d.kind === 'input') return `x${d.index}`;
  if (nInputs === 2) return GATE_NAMES[d.gateId] ?? `G${d.gateId}`;
  return `G${d.gateId}`;
}

export function createNodeLayer(
  nodes: L0Node[],
  colorVersion: number,
  onHover?: (info: { object?: L0Node }) => void,
): ScatterplotLayer<L0Node> {
  return new ScatterplotLayer<L0Node>({
    id: 'l0-nodes',
    data: nodes,
    getPosition: (d) => d.position,
    getFillColor: (d) => d.color,
    getLineColor: cssVarRGBA('--edge-default', 200),
    getRadius: NODE_RADIUS,
    radiusUnits: 'common',
    lineWidthMinPixels: 0.5,
    filled: true,
    stroked: true,
    pickable: true,
    onHover,
    updateTriggers: {
      getFillColor: [colorVersion],
    },
  });
}

/**
 * エッジ層。逆伝播リプレイでアクティブ層を表示中は、
 * gradIntensity(edge_gradの層内max正規化、L0Edge.index = neuron×n_inputs+slot と
 * 同じ平坦index規約)によるカラースケール着色(淡グレー→--chart-loss)へ切り替える。
 * gradVersionはupdateTriggersのキー(再生開始/step変化で色を再評価)。
 */
export function createEdgeLayer(
  edges: L0Edge[],
  zoom: number,
  gradIntensity: Float32Array | null = null,
  gradVersion = -1,
): LineLayer<L0Edge> {
  const base = cssVarRGBA('--edge-default', 45);
  const hot = cssVarRGB('--chart-loss', [235, 104, 52]);
  const gradColor = (d: L0Edge): RGBA => {
    const t = Math.min(Math.max(gradIntensity?.[d.index] ?? 0, 0), 1);
    return [
      Math.round(base[0] + (hot[0] - base[0]) * t),
      Math.round(base[1] + (hot[1] - base[1]) * t),
      Math.round(base[2] + (hot[2] - base[2]) * t),
      Math.round(40 + 215 * t),
    ];
  };
  return new LineLayer<L0Edge>({
    id: 'l0-edges',
    data: edges,
    getSourcePosition: (d) => d.source,
    getTargetPosition: (d) => d.target,
    getColor: gradIntensity ? gradColor : base,
    getWidth: gradIntensity
      ? (d) => 0.2 + 0.5 * Math.min(Math.max(gradIntensity[d.index] ?? 0, 0), 1)
      : 0.2,
    widthUnits: 'common',
    visible: zoom > EDGE_ZOOM_THRESHOLD,
    updateTriggers: {
      getColor: [gradVersion],
      getWidth: [gradVersion],
    },
  });
}

export function createTextLayer(
  nodes: L0Node[],
  zoom: number,
  nInputs: number,
  colorVersion: number,
): TextLayer<L0Node> {
  return new TextLayer<L0Node>({
    id: 'l0-labels',
    data: nodes,
    getText: (d) => nodeLabel(d, nInputs),
    getPosition: (d) => d.position,
    getSize: 10,
    sizeUnits: 'common',
    getColor: cssVarRGBA('--foreground', 220),
    getTextAnchor: 'middle',
    getAlignmentBaseline: 'center',
    fontFamily: 'SF Mono, Fira Code, Consolas, monospace',
    fontWeight: 'bold',
    visible: zoom > TEXT_ZOOM_THRESHOLD,
    billboard: false,
    updateTriggers: {
      getText: [colorVersion],
    },
  });
}
