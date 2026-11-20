// アプリ状態(zustand)。選択・折りたたみ・フィルタ・LOD・テーマを1ストアに集約する。

import { create } from 'zustand';
import treeJson from './data/tree.json';
import { EMPTY_FILTERS, type Filters } from './tree/graph';
import type { TreeData, TreeNode } from './types';

export const TREE = treeJson as unknown as TreeData;

const NODE_BY_ID = new Map<string, TreeNode>(TREE.nodes.map((n) => [n.id, n]));
export const getNode = (id: string | null | undefined): TreeNode | null =>
  (id ? (NODE_BY_ID.get(id) ?? null) : null);

export type LodStage = 'overview' | 'tile' | 'full';
export type Theme = 'light' | 'dark';

const THEME_KEY = 'paper-tree-theme';

function initialTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY);
    if (v === 'light' || v === 'dark') return v;
  } catch {
    /* localStorage不可(プライベートウィンドウ等)は既定値 */
  }
  return 'dark';
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark');
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* 保存できなくても表示は続行 */
  }
}

interface AppState {
  selectedId: string | null;
  collapsed: Set<string>;
  filters: Filters;
  searchOpen: boolean;
  lod: LodStage;
  theme: Theme;
  /** 検索/リンクからのフォーカス要求(nonceで同一idの再要求も伝える) */
  focus: { id: string; nonce: number } | null;

  select: (id: string | null) => void;
  focusNode: (id: string) => void;
  toggleCollapse: (id: string) => void;
  setAllCollapsed: (ids: string[], collapsed: boolean) => void;
  setFilters: (patch: Partial<Filters>) => void;
  resetFilters: () => void;
  setSearchOpen: (open: boolean) => void;
  setLod: (lod: LodStage) => void;
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedId: null,
  collapsed: new Set<string>(),
  filters: { ...EMPTY_FILTERS, categories: new Set<string>() },
  searchOpen: false,
  lod: 'full',
  theme: initialTheme(),
  focus: null,

  select: (id) => set({ selectedId: id }),
  focusNode: (id) =>
    set((s) => ({
      selectedId: id,
      // 折りたたみ中のカテゴリ配下でも見えるように親を開く
      collapsed: openParentsOf(id, s.collapsed),
      focus: { id, nonce: (s.focus?.nonce ?? 0) + 1 },
    })),
  toggleCollapse: (id) =>
    set((s) => {
      const next = new Set(s.collapsed);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { collapsed: next };
    }),
  setAllCollapsed: (ids, collapsed) =>
    set(() => ({ collapsed: collapsed ? new Set(ids) : new Set<string>() })),
  setFilters: (patch) => set((s) => ({ filters: { ...s.filters, ...patch } })),
  resetFilters: () => set({ filters: { ...EMPTY_FILTERS, categories: new Set<string>() } }),
  setSearchOpen: (open) => set({ searchOpen: open }),
  setLod: (lod) => set((s) => (s.lod === lod ? s : { lod })),
  toggleTheme: () =>
    set((s) => {
      const theme: Theme = s.theme === 'dark' ? 'light' : 'dark';
      applyTheme(theme);
      return { theme };
    }),
}));

/** id -> 親id[](親子エッジ)。折りたたみを遡って開くために使う。 */
const PARENTS = new Map<string, string[]>();
for (const e of TREE.edges) {
  if (e.kind !== 'parent') continue;
  const list = PARENTS.get(e.target);
  if (list) list.push(e.source);
  else PARENTS.set(e.target, [e.source]);
}

/** ノードの祖先(親が論文の場合も遡る)を折りたたみ集合から外す。 */
function openParentsOf(id: string, collapsed: Set<string>): Set<string> {
  if (collapsed.size === 0) return collapsed;
  const next = new Set(collapsed);
  const seen = new Set<string>([id]);
  const queue = [...(PARENTS.get(id) ?? [])];
  while (queue.length > 0) {
    const cur = queue.pop()!;
    if (seen.has(cur)) continue;
    seen.add(cur);
    next.delete(cur);
    queue.push(...(PARENTS.get(cur) ?? []));
  }
  return next;
}
