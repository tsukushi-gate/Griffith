// elkjsによる自動レイアウト。
// layered+RIGHTで左→右のデータフロー配置を計算する。ノード実寸は
// コンポーネントのCSS(min-width/padding/ミニバー)から見積もった定数を使う。

import type { ElkNode } from 'elkjs/lib/elk-api';
import type { Edge } from '@xyflow/react';
import type { EditorNode } from './irToGraph';

// ノード種別ごとの実寸ヒント(px)。lutLayerはGateHistogram(176x40)を含む
const LUT_NODE_SIZE = { width: 200, height: 96 };
const DEFAULT_NODE_SIZE = { width: 176, height: 60 };

const ELK_OPTIONS: Record<string, string> = {
  'elk.algorithm': 'layered',
  'elk.direction': 'RIGHT',
  'elk.layered.spacing.nodeNodeBetweenLayers': '80',
  'elk.spacing.nodeNode': '40',
};

// elkjs(約1.4MB)は初期バンドルから外し、初回レイアウト時に遅延ロードする。
// layout()自体がPromiseベースなので呼び出し側の契約は変わらない。
let elkPromise: Promise<InstanceType<typeof import('elkjs/lib/elk.bundled.js').default>> | null =
  null;

function getElk() {
  elkPromise ??= import('elkjs/lib/elk.bundled.js').then((m) => new m.default());
  return elkPromise;
}

function nodeSize(
  node: EditorNode,
  sizeOverrides?: Map<string, { width: number; height: number }>,
): { width: number; height: number } {
  // 詳細ゲートビューの展開グループは実寸overrideを最優先
  const ov = sizeOverrides?.get(node.id);
  if (ov) return ov;
  // React Flowが計測済みの実寸があれば優先
  const m = node.measured;
  if (m?.width && m?.height) return { width: m.width, height: m.height };
  return node.type === 'lutLayer' ? LUT_NODE_SIZE : DEFAULT_NODE_SIZE;
}

/** 全ノードのレイアウト座標(左上基準、React Flow座標系)を計算する。 */
export async function layoutWithElk(
  nodes: EditorNode[],
  edges: Edge[],
  sizeOverrides?: Map<string, { width: number; height: number }>,
): Promise<Map<string, { x: number; y: number }>> {
  const positions = new Map<string, { x: number; y: number }>();
  if (nodes.length === 0) return positions;

  const ids = new Set(nodes.map((n) => n.id));
  const graph: ElkNode = {
    id: 'root',
    layoutOptions: ELK_OPTIONS,
    children: nodes.map((n) => ({ id: n.id, ...nodeSize(n, sizeOverrides) })),
    edges: edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };

  const elk = await getElk();
  const result = await elk.layout(graph);
  for (const child of result.children ?? []) {
    if (typeof child.x === 'number' && typeof child.y === 'number') {
      positions.set(child.id, { x: Math.round(child.x), y: Math.round(child.y) });
    }
  }
  return positions;
}
