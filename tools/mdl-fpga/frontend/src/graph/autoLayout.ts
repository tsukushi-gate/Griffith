// elkjs による自動レイアウト(layered + RIGHT = 左→右データフロー)。
// 出典: tools/editor/frontend/src/ir/autoLayout.ts を適応コピー
// (EditorNode 型依存を汎用 Node へ変更。ELK オプションは同一)。
import type { Edge, Node } from '@xyflow/react';
import type { ElkNode } from 'elkjs/lib/elk-api';

const BLOCK_NODE_SIZE = { width: 190, height: 80 };
const PORT_NODE_SIZE = { width: 150, height: 60 };

const ELK_OPTIONS: Record<string, string> = {
  'elk.algorithm': 'layered',
  'elk.direction': 'RIGHT',
  'elk.layered.spacing.nodeNodeBetweenLayers': '80',
  'elk.spacing.nodeNode': '40',
};

// elkjs(約1.4MB)は初期バンドルから外し、初回レイアウト時に遅延ロードする
let elkPromise: Promise<
  InstanceType<typeof import('elkjs/lib/elk.bundled.js').default>
> | null = null;

function getElk() {
  elkPromise ??= import('elkjs/lib/elk.bundled.js').then((m) => new m.default());
  return elkPromise;
}

function nodeSize(node: Node): { width: number; height: number } {
  const m = node.measured;
  if (m?.width && m?.height) return { width: m.width, height: m.height };
  return node.type === 'port' ? PORT_NODE_SIZE : BLOCK_NODE_SIZE;
}

/** 全ノードのレイアウト座標(左上基準、React Flow 座標系)を計算する。 */
export async function layoutWithElk(
  nodes: Node[],
  edges: Edge[],
): Promise<Map<string, { x: number; y: number }>> {
  const positions = new Map<string, { x: number; y: number }>();
  if (nodes.length === 0) return positions;

  const ids = new Set(nodes.map((n) => n.id));
  const graph: ElkNode = {
    id: 'root',
    layoutOptions: ELK_OPTIONS,
    children: nodes.map((n) => ({ id: n.id, ...nodeSize(n) })),
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
