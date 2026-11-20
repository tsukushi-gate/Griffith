// サブサーキット化: 複数ノードを表示上のグループへまとめる純関数群。
// 【重要】グループは表示・編集の概念であってIRのノード構造を変えない。
// グループ定義は ir.metadata.editor.groups に永続化され(座標と同じ機構)、
// flowToIrの `editor: { ...baseMetaEditor, ports }` スプレッドで往復保全される。
//
// 折りたたみ(collapsed=true): メンバーノードを隠し、単一の合成グループノード
//   (id `grp:{groupId}`)へ差し替える。境界をまたぐエッジはグループノードの
//   単一入出力ハンドルへ再配線する(グループ内部の外部入出力だけを見せる)。
// 展開(collapsed=false): メンバーはそのまま表示し、外接矩形の左上へ操作用の
//   ヘッダノード(id `grph:{groupId}`)を浮かせる。エッジは再配線しない。
//
// 子ノードidの区切り '::' や詳細エッジ 'd:' と衝突しない専用プレフィックスを使う。

import type { Edge, Node } from '@xyflow/react';
import type { EditorNode } from '../ir/irToGraph';
import type { IRGraphJson } from '../ir/schema';

/** 折りたたみグループノードのid接頭辞 */
export const GROUP_NODE_PREFIX = 'grp:';
/** 展開グループのヘッダノードのid接頭辞 */
export const GROUP_HEADER_PREFIX = 'grph:';
/** グループ再配線エッジのid接頭辞 */
export const GROUP_EDGE_PREFIX = 'grpE:';

/** グループの合成ノード(折りたたみ/ヘッダ)のidか */
export function isGroupNodeId(id: string): boolean {
  return id.startsWith(GROUP_NODE_PREFIX) || id.startsWith(GROUP_HEADER_PREFIX);
}

/** グループ再配線エッジのidか */
export function isGroupEdgeId(id: string): boolean {
  return id.startsWith(GROUP_EDGE_PREFIX);
}

/** 合成グループノードidからgroupIdを取り出す(グループノードでなければnull) */
export function groupIdFromNodeId(id: string): string | null {
  if (id.startsWith(GROUP_HEADER_PREFIX)) return id.slice(GROUP_HEADER_PREFIX.length);
  if (id.startsWith(GROUP_NODE_PREFIX)) return id.slice(GROUP_NODE_PREFIX.length);
  return null;
}

/** meta.editor.groups に永続化されるグループ定義 */
export interface EditorGroup {
  id: string;
  name: string;
  nodeIds: string[];
  collapsed: boolean;
  position: { x: number; y: number };
}

export type GroupCollapsedData = {
  groupId: string;
  name: string;
  memberCount: number;
  inputCount: number;
  outputCount: number;
  [key: string]: unknown;
};
export type GroupCollapsedNode = Node<GroupCollapsedData, 'group'>;

export type GroupHeaderData = {
  groupId: string;
  name: string;
  memberCount: number;
  [key: string]: unknown;
};
export type GroupHeaderNode = Node<GroupHeaderData, 'groupHeader'>;

/** 折りたたみグループノードの実寸(onlyRenderVisibleElements可視判定用に明示する) */
export const GROUP_NODE_W = 176;
export const GROUP_NODE_H = 64;
/** 展開ヘッダの実寸とメンバー外接矩形からの浮かせ量 */
export const GROUP_HEADER_W = 200;
export const GROUP_HEADER_H = 32;
const HEADER_OFFSET_Y = 44;

/** ir.metadata.editor.groups を EditorGroup[] へ読み出す(不正エントリは除外)。 */
export function readGroups(ir: IRGraphJson | null): EditorGroup[] {
  if (!ir) return [];
  const editor = ir.metadata?.editor as
    | { groups?: Record<string, unknown> }
    | undefined;
  const raw = editor?.groups;
  if (!raw || typeof raw !== 'object') return [];
  const out: EditorGroup[] = [];
  for (const [id, v] of Object.entries(raw)) {
    if (!v || typeof v !== 'object') continue;
    const g = v as Partial<EditorGroup>;
    if (!Array.isArray(g.nodeIds)) continue;
    out.push({
      id,
      name: typeof g.name === 'string' ? g.name : id,
      nodeIds: g.nodeIds.filter((x): x is string => typeof x === 'string'),
      collapsed: g.collapsed === true,
      position:
        g.position && typeof g.position.x === 'number'
          ? { x: g.position.x, y: g.position.y }
          : { x: 0, y: 0 },
    });
  }
  return out;
}

/** メンバーノードの外接矩形左上(座標のみ、実寸は考慮しない簡易版)。 */
function boundingTopLeft(
  memberIds: string[],
  nodeById: Map<string, EditorNode>,
): { x: number; y: number } {
  let minX = Infinity;
  let minY = Infinity;
  for (const id of memberIds) {
    const n = nodeById.get(id);
    if (!n) continue;
    minX = Math.min(minX, n.position.x);
    minY = Math.min(minY, n.position.y);
  }
  if (!Number.isFinite(minX)) return { x: 0, y: 0 };
  return { x: minX, y: minY };
}

/**
 * グループ定義を表示用ノード/エッジへ適用する(IRは不変、描画専用の合成)。
 * 折りたたみグループのメンバーは隠され、合成ノードと再配線エッジに置き換わる。
 * 返すノード配列にはEditorNode以外(グループ合成ノード)が混ざるためunion型。
 */
export function applyGrouping(
  nodes: EditorNode[],
  edges: Edge[],
  groups: EditorGroup[],
): { nodes: (EditorNode | GroupCollapsedNode | GroupHeaderNode)[]; edges: Edge[] } {
  if (groups.length === 0) return { nodes, edges };

  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  // メンバーid → 折りたたみgroupId(展開グループはここには入れない)
  const collapsedMemberToGid = new Map<string, string>();
  const collapsedGroups: { group: EditorGroup; members: string[] }[] = [];
  const expandedGroups: { group: EditorGroup; members: string[] }[] = [];
  for (const g of groups) {
    const members = g.nodeIds.filter((id) => nodeById.has(id));
    if (members.length === 0) continue;
    if (g.collapsed) {
      collapsedGroups.push({ group: g, members });
      for (const id of members) collapsedMemberToGid.set(id, g.id);
    } else {
      expandedGroups.push({ group: g, members });
    }
  }
  if (collapsedGroups.length === 0 && expandedGroups.length === 0) {
    return { nodes, edges };
  }

  const outNodes: (EditorNode | GroupCollapsedNode | GroupHeaderNode)[] = [];
  // 折りたたみメンバー以外のノードは素通し(参照維持で再描画コストを抑える)
  for (const n of nodes) {
    if (collapsedMemberToGid.has(n.id)) continue;
    outNodes.push(n);
  }

  // 折りたたみグループごとの外部入出力数を数える(表示ラベル用)
  const inSrcByGid = new Map<string, Set<string>>();
  const outDstByGid = new Map<string, Set<string>>();
  for (const { group } of collapsedGroups) {
    inSrcByGid.set(group.id, new Set());
    outDstByGid.set(group.id, new Set());
  }
  for (const e of edges) {
    const sg = collapsedMemberToGid.get(e.source);
    const tg = collapsedMemberToGid.get(e.target);
    if (tg && sg !== tg) inSrcByGid.get(tg)?.add(sg ? `grp:${sg}` : e.source);
    if (sg && sg !== tg) outDstByGid.get(sg)?.add(tg ? `grp:${tg}` : e.target);
  }

  for (const { group, members } of collapsedGroups) {
    const node: GroupCollapsedNode = {
      id: GROUP_NODE_PREFIX + group.id,
      type: 'group',
      position: group.position,
      width: GROUP_NODE_W,
      height: GROUP_NODE_H,
      data: {
        groupId: group.id,
        name: group.name,
        memberCount: members.length,
        inputCount: inSrcByGid.get(group.id)?.size ?? 0,
        outputCount: outDstByGid.get(group.id)?.size ?? 0,
      },
    };
    outNodes.push(node);
  }

  for (const { group, members } of expandedGroups) {
    const tl = boundingTopLeft(members, nodeById);
    const node: GroupHeaderNode = {
      id: GROUP_HEADER_PREFIX + group.id,
      type: 'groupHeader',
      position: { x: tl.x, y: tl.y - HEADER_OFFSET_Y },
      width: GROUP_HEADER_W,
      height: GROUP_HEADER_H,
      draggable: false,
      selectable: false,
      deletable: false,
      data: {
        groupId: group.id,
        name: group.name,
        memberCount: members.length,
      },
    };
    outNodes.push(node);
  }

  // エッジの再配線(折りたたみ境界のみ変換、それ以外は素通し)
  const outEdges: Edge[] = [];
  const seen = new Set<string>();
  const push = (e: Edge) => {
    if (seen.has(e.id)) return;
    seen.add(e.id);
    outEdges.push(e);
  };
  for (const e of edges) {
    const sg = collapsedMemberToGid.get(e.source);
    const tg = collapsedMemberToGid.get(e.target);
    if (!sg && !tg) {
      outEdges.push(e);
      continue;
    }
    if (sg && tg && sg === tg) continue; // グループ内部エッジは隠す
    if (sg && tg) {
      // グループ間エッジ
      push({
        id: `${GROUP_EDGE_PREFIX}${sg}=>${tg}`,
        source: GROUP_NODE_PREFIX + sg,
        target: GROUP_NODE_PREFIX + tg,
        deletable: false,
        selectable: false,
      });
      continue;
    }
    if (sg) {
      // メンバー → 外部(ソースをグループノードへ、外部側のハンドルは維持)
      push({
        id: `${GROUP_EDGE_PREFIX}${sg}->${e.target}`,
        source: GROUP_NODE_PREFIX + sg,
        target: e.target,
        targetHandle: e.targetHandle,
        deletable: false,
        selectable: false,
      });
      continue;
    }
    // 外部 → メンバー(ターゲットをグループノードへ、内部slotハンドルは落とす)
    push({
      id: `${GROUP_EDGE_PREFIX}${e.source}->${tg}`,
      source: e.source,
      target: GROUP_NODE_PREFIX + tg,
      sourceHandle: e.sourceHandle,
      deletable: false,
      selectable: false,
    });
  }

  return { nodes: outNodes, edges: outEdges };
}
