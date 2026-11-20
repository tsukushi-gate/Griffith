// elkjsによる自動レイアウト(tools/editor/frontend/src/ir/autoLayout.ts の遅延ロード方式を流用)。
//
// 200+ノードを素直に layered(RIGHT) へ流すと「カテゴリ層に200枚が縦一列」となり、
// 高さ2万px超の髪の毛のようなグラフになって俯瞰できない。そこで2段のパッキングにする:
//   1段目: カテゴリごとに配下ノードを rectpacking で矩形ブロックへ詰める
//   2段目: 「カテゴリカード + ブロック」の塊(スーパーノード)を rectpacking で盤面に敷き詰める
// これによりノード数が増えても縦横比がおおむね一定に保たれる。

import type { ElkNode } from 'elkjs/lib/elk-api';
import type { TreeEdge, TreeNode } from '../types';

export const NODE_SIZE: Record<TreeNode['kind'], { width: number; height: number }> = {
  root: { width: 160, height: 56 },
  category: { width: 200, height: 56 },
  paper: { width: 240, height: 84 },
  report: { width: 240, height: 68 },
};

/** カテゴリカードとブロックの間隔 */
const CARD_GAP = 48;
/** rootとカテゴリ盤面の間隔 */
const ROOT_GAP = 160;
/** ブロック内のノード間隔 */
const CELL_SPACING = 18;
/** カテゴリブロック同士の間隔 */
const BLOCK_SPACING = 72;

const PACK_CHILDREN: Record<string, string> = {
  'elk.algorithm': 'rectpacking',
  'elk.aspectRatio': '2.4',
  'elk.spacing.nodeNode': String(CELL_SPACING),
  'elk.rectpacking.rowHeight.targetWidth': '1400',
};

const PACK_BLOCKS: Record<string, string> = {
  'elk.algorithm': 'rectpacking',
  'elk.aspectRatio': '1.7',
  'elk.spacing.nodeNode': String(BLOCK_SPACING),
};

// elkjs(約1.4MB)は初期バンドルから外し、初回レイアウト時に遅延ロードする
let elkPromise: Promise<InstanceType<typeof import('elkjs/lib/elk.bundled.js').default>> | null =
  null;

function getElk() {
  elkPromise ??= import('elkjs/lib/elk.bundled.js').then((m) => new m.default());
  return elkPromise;
}

type Pos = { x: number; y: number };

/** 各ノードの所属カテゴリ(=描画上のブロック)を決める。 */
function assignBlocks(nodes: TreeNode[], edges: TreeEdge[]): Map<string, string[]> {
  const categoryIds = new Set(nodes.filter((n) => n.kind === 'category').map((n) => n.id));
  const parentsOf = new Map<string, string[]>();
  for (const e of edges) {
    if (e.kind !== 'parent') continue;
    const list = parentsOf.get(e.target);
    if (list) list.push(e.source);
    else parentsOf.set(e.target, [e.source]);
  }

  /** ノードから親を遡って最初に見つかるカテゴリ(見つからなければnull) */
  const resolve = (id: string, seen = new Set<string>()): string | null => {
    if (seen.has(id)) return null;
    seen.add(id);
    for (const p of parentsOf.get(id) ?? []) {
      if (categoryIds.has(p)) return p;
      const up = resolve(p, seen);
      if (up) return up;
    }
    return null;
  };

  const blocks = new Map<string, string[]>();
  for (const id of categoryIds) blocks.set(id, []);
  for (const n of nodes) {
    if (n.kind === 'root' || n.kind === 'category') continue;
    const cat = resolve(n.id);
    // カテゴリに辿り着かない孤立ノードは専用ブロックへ退避する
    const key = cat ?? '__orphan__';
    const list = blocks.get(key);
    if (list) list.push(n.id);
    else blocks.set(key, [n.id]);
  }
  return blocks;
}

/**
 * 可視ノード/エッジからレイアウト座標(左上基準・React Flow座標系)を計算する。
 * 失敗しても描画を止めないよう、elkが例外を投げたら素朴なグリッドへフォールバックする。
 */
export async function layoutTree(
  nodes: TreeNode[],
  edges: TreeEdge[],
): Promise<Map<string, Pos>> {
  const positions = new Map<string, Pos>();
  if (nodes.length === 0) return positions;

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const blocks = assignBlocks(nodes, edges);
  const elk = await getElk();

  try {
    // --- 1段目: カテゴリごとに配下を詰める ---
    const packed = await Promise.all(
      [...blocks.entries()].map(async ([catId, childIds]) => {
        if (childIds.length === 0) return { catId, childIds, local: new Map<string, Pos>(), w: 0, h: 0 };
        const graph: ElkNode = {
          id: `pack:${catId}`,
          layoutOptions: PACK_CHILDREN,
          children: childIds.map((id) => ({ id, ...NODE_SIZE[byId.get(id)!.kind] })),
          edges: [],
        };
        const res = await elk.layout(graph);
        const local = new Map<string, Pos>();
        for (const c of res.children ?? []) {
          local.set(c.id, { x: Math.round(c.x ?? 0), y: Math.round(c.y ?? 0) });
        }
        return {
          catId,
          childIds,
          local,
          w: Math.round(res.width ?? 0),
          h: Math.round(res.height ?? 0),
        };
      }),
    );

    // --- 2段目: 「カテゴリカード + ブロック」を盤面へ詰める ---
    const cardW = NODE_SIZE.category.width;
    const cardH = NODE_SIZE.category.height;
    const supers = packed.map((p) => ({
      ...p,
      width: cardW + (p.w > 0 ? CARD_GAP + p.w : 0),
      height: Math.max(cardH, p.h),
    }));
    const boardGraph: ElkNode = {
      id: 'board',
      layoutOptions: PACK_BLOCKS,
      children: supers.map((s) => ({ id: s.catId, width: s.width, height: s.height })),
      edges: [],
    };
    const board = await elk.layout(boardGraph);
    const boardPos = new Map<string, Pos>();
    for (const c of board.children ?? []) {
      boardPos.set(c.id, { x: Math.round(c.x ?? 0), y: Math.round(c.y ?? 0) });
    }

    const originX = NODE_SIZE.root.width + ROOT_GAP;
    for (const s of supers) {
      const base = boardPos.get(s.catId) ?? { x: 0, y: 0 };
      if (byId.has(s.catId)) {
        positions.set(s.catId, {
          x: originX + base.x,
          y: base.y + Math.round((s.height - cardH) / 2),
        });
      }
      const blockX = originX + base.x + cardW + CARD_GAP;
      const blockY = base.y + Math.round((s.height - s.h) / 2);
      for (const id of s.childIds) {
        const l = s.local.get(id) ?? { x: 0, y: 0 };
        positions.set(id, { x: blockX + l.x, y: blockY + l.y });
      }
    }

    // rootは盤面の左・縦中央へ
    const boardH = Math.round(board.height ?? 0);
    const rootNode = nodes.find((n) => n.kind === 'root');
    if (rootNode) {
      positions.set(rootNode.id, { x: 0, y: Math.round((boardH - NODE_SIZE.root.height) / 2) });
    }
    return positions;
  } catch (e) {
    console.warn('[layout] elkに失敗したためグリッド配置へフォールバック', e);
    return gridFallback(nodes);
  }
}

/** elkが使えないときの素朴なグリッド(表示が止まらないことを優先)。 */
function gridFallback(nodes: TreeNode[]): Map<string, Pos> {
  const positions = new Map<string, Pos>();
  const cols = Math.max(1, Math.ceil(Math.sqrt(nodes.length * 0.6)));
  nodes.forEach((n, i) => {
    positions.set(n.id, {
      x: (i % cols) * (NODE_SIZE.paper.width + CELL_SPACING),
      y: Math.floor(i / cols) * (NODE_SIZE.paper.height + CELL_SPACING),
    });
  });
  return positions;
}
