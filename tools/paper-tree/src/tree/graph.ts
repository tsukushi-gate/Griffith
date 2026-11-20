// ツリーの可視ノード/エッジを決める純関数群(Reactに依存しない)。
// フィルタ(カテゴリ/年/status)とカテゴリ折りたたみを合成して可視集合を作る。

import {
  isCategory,
  isPaper,
  isReport,
  nodeYear,
  type TreeData,
  type TreeEdge,
  type TreeNode,
} from '../types';

export interface Filters {
  /** 空 = 全カテゴリ。要素は `cat:` 付きのカテゴリid */
  categories: Set<string>;
  status: 'all' | 'summarized' | 'stub';
  yearMin: number | null;
  yearMax: number | null;
  /** レポートノードを表示するか */
  showReports: boolean;
}

export const EMPTY_FILTERS: Filters = {
  categories: new Set(),
  status: 'all',
  yearMin: null,
  yearMax: null,
  showReports: true,
};

/** 論文/レポート1件がフィルタを通過するか。 */
export function matchesFilters(node: TreeNode, f: Filters): boolean {
  if (isReport(node)) {
    if (!f.showReports) return false;
  } else if (isPaper(node)) {
    if (f.status !== 'all' && node.status !== f.status) return false;
    if (f.categories.size > 0) {
      const hit = node.categories.some((c) => f.categories.has(`cat:${c}`));
      if (!hit) return false;
    }
  } else {
    return true; // root / category はフィルタ対象外
  }
  const y = nodeYear(node);
  if (f.yearMin != null && (y == null || y < f.yearMin)) return false;
  if (f.yearMax != null && (y == null || y > f.yearMax)) return false;
  return true;
}

export interface VisibleGraph {
  nodes: TreeNode[];
  edges: TreeEdge[];
  /** カテゴリid -> そのカテゴリ配下でフィルタを通過した件数(折りたたみ中も数える) */
  childCounts: Map<string, number>;
}

/** 親(カテゴリ or 論文 or root)のid配列を id -> parents で引ける形にする。 */
function buildParentIndex(edges: TreeEdge[]): Map<string, string[]> {
  const index = new Map<string, string[]>();
  for (const e of edges) {
    if (e.kind !== 'parent') continue;
    const list = index.get(e.target);
    if (list) list.push(e.source);
    else index.set(e.target, [e.source]);
  }
  return index;
}

/**
 * 折りたたみで隠れるノード集合を求める。
 * 親が論文になり得る(`parents: [<paper-id>]`)ため、折りたたみは部分木全体へ伝播させる。
 * 複数親のノードは「全ての親が隠れている」ときだけ隠す。不動点まで反復する
 * (親子エッジに循環があっても、隠れる側が増えるだけなので停止する)。
 */
function computeHidden(
  nodes: TreeNode[],
  parentIndex: Map<string, string[]>,
  collapsed: ReadonlySet<string>,
): Set<string> {
  const hidden = new Set<string>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const n of nodes) {
      if (n.kind === 'root' || hidden.has(n.id)) continue;
      const parents = parentIndex.get(n.id);
      if (!parents || parents.length === 0) continue;
      if (parents.every((p) => collapsed.has(p) || hidden.has(p))) {
        hidden.add(n.id);
        changed = true;
      }
    }
  }
  return hidden;
}

/**
 * フィルタ + 折りたたみを適用した可視グラフを作る。
 * - root と category は常に可視(カテゴリはフィルタで0件でも「空」として残す)
 * - collapsed に含まれるカテゴリの子(論文/レポート)は隠す
 * - エッジは両端が可視のときのみ残す
 */
export function computeVisibleGraph(
  tree: TreeData,
  filters: Filters,
  collapsed: ReadonlySet<string>,
): VisibleGraph {
  const parentIndex = buildParentIndex(tree.edges);
  const hidden = computeHidden(tree.nodes, parentIndex, collapsed);
  const childCounts = new Map<string, number>();
  const visibleIds = new Set<string>();

  for (const n of tree.nodes) {
    if (n.kind === 'root' || isCategory(n)) {
      visibleIds.add(n.id);
      continue;
    }
    if (!matchesFilters(n, filters)) continue;
    for (const p of parentIndex.get(n.id) ?? []) {
      childCounts.set(p, (childCounts.get(p) ?? 0) + 1);
    }
    if (!hidden.has(n.id)) visibleIds.add(n.id);
  }

  const nodes = tree.nodes.filter((n) => visibleIds.has(n.id));
  const edges = tree.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target));
  return { nodes, edges, childCounts };
}

/** カテゴリ一覧(id昇順)。 */
export function categoryList(tree: TreeData) {
  return tree.nodes.filter(isCategory).sort((a, b) => a.id.localeCompare(b.id));
}

/** 検索対象テキスト(タイトル+id+カテゴリ+本文)。 */
export function searchText(node: TreeNode): string {
  const parts = [node.title, node.id];
  if (isPaper(node)) parts.push(node.categories.join(' '), node.body);
  if (isReport(node)) parts.push(node.branch ?? '', node.path ?? '', node.body);
  return parts.join('\n').toLowerCase();
}
