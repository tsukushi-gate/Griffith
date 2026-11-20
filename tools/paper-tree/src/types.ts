// tree.json のスキーマ(scripts/build-tree.mjs の出力と1対1)。

export type NodeKind = 'root' | 'category' | 'paper' | 'report';
export type EdgeKind = 'parent' | 'next' | 'report';
export type PaperStatus = 'summarized' | 'stub';

export interface RootNode {
  kind: 'root';
  id: string;
  title: string;
  label: string;
}

export interface CategoryNode {
  kind: 'category';
  id: string;
  title: string;
  label: string;
  parent: string;
}

export interface PaperNode {
  kind: 'paper';
  id: string;
  file: string;
  title: string;
  date: string | null;
  summarized: string | null;
  status: PaperStatus;
  categories: string[];
  arxiv: string | null;
  url: string | null;
  pdf: string | null;
  parents: string[];
  next: string[];
  reports: string[];
  extra?: Record<string, unknown>;
  body: string;
}

export interface ReportNode {
  kind: 'report';
  id: string;
  file: string;
  title: string;
  date: string | null;
  branch: string | null;
  path: string | null;
  artifactUrl: string | null;
  type: string;
  papers: string[];
  extra?: Record<string, unknown>;
  body: string;
}

export type TreeNode = RootNode | CategoryNode | PaperNode | ReportNode;

export interface TreeEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
}

export interface TreeStats {
  papers: number;
  reports: number;
  categories: number;
  nodes: number;
  edges: number;
  stubs: number;
  /** superseded_by で正本へ畳んだ重複論文の件数(ノードにはならない) */
  superseded: number;
  stubRatio: number;
  yearMin: number | null;
  yearMax: number | null;
  edgesByKind: Partial<Record<EdgeKind, number>>;
}

export interface TreeData {
  generatedAt: string;
  stats: TreeStats;
  nodes: TreeNode[];
  edges: TreeEdge[];
}

export const isPaper = (n: TreeNode): n is PaperNode => n.kind === 'paper';
export const isReport = (n: TreeNode): n is ReportNode => n.kind === 'report';
export const isCategory = (n: TreeNode): n is CategoryNode => n.kind === 'category';

/** 論文の発表年(不明はnull)。 */
export function nodeYear(n: TreeNode): number | null {
  const d = (n as PaperNode | ReportNode).date;
  if (!d) return null;
  const y = Number(String(d).slice(0, 4));
  return Number.isFinite(y) ? y : null;
}
