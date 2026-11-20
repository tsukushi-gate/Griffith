// マイブロック: グループを名前付きテンプレートとしてlocalStorageに保存し、
// Paletteから再配置できるようにする。保存されるのはIR断片(ノード+内部エッジ)で、
// 配置時にid採番コールバックで振り直して addSubgraph へ渡す(blocks/templatesの
// 複合ブロック生成器と同じ「id振り直しで展開」パターン)。

import type { IdAllocator } from './templates';
import type { IREdgeJson, IRNodeJson } from '../ir/schema';
import type { EditorUnit } from './unitTypes';

const STORAGE_KEY = 'magicdelta.myBlocks.v1';
// Unit定義のマイブロック棚(localStorage)。IR断片のマイブロックとは別キー。
const UNITS_STORAGE_KEY = 'magicdelta.myUnits.v1';

/** localStorageに保存するテンプレート。座標は左上原点への正規化済み(相対配置)。 */
export interface MyBlock {
  id: string;
  name: string;
  nodes: IRNodeJson[];
  edges: IREdgeJson[];
  createdAt: number;
}

/** DnDでマイブロックidを運ぶMIME */
export const MYBLOCK_DND_MIME = 'application/x-magicdelta-myblock';

/** 保存/削除でPaletteを再描画させるための変更通知イベント名 */
export const MYBLOCKS_CHANGED_EVENT = 'magicdelta:myblocks-changed';

function safeParse(raw: string | null): MyBlock[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? (v as MyBlock[]) : [];
  } catch {
    return [];
  }
}

/** 保存済みマイブロック一覧(新しい順)。 */
export function listMyBlocks(): MyBlock[] {
  const list = safeParse(localStorage.getItem(STORAGE_KEY));
  return [...list].sort((a, b) => b.createdAt - a.createdAt);
}

function writeAll(list: MyBlock[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  window.dispatchEvent(new Event(MYBLOCKS_CHANGED_EVENT));
}

/**
 * IR断片をマイブロックとして保存する。ノード座標はmeta.editor.positionを
 * 左上原点へ平行移動して正規化する(配置時にドロップ位置基準で読めるようにする)。
 */
export function saveMyBlock(
  name: string,
  nodes: IRNodeJson[],
  edges: IREdgeJson[],
): MyBlock {
  let minX = Infinity;
  let minY = Infinity;
  for (const n of nodes) {
    const p = (n.meta?.editor as { position?: { x: number; y: number } })?.position;
    if (p) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
    }
  }
  if (!Number.isFinite(minX)) {
    minX = 0;
    minY = 0;
  }
  const normNodes = nodes.map((n) => {
    const editor = (n.meta?.editor ?? {}) as Record<string, unknown>;
    const p = (editor.position ?? { x: 0, y: 0 }) as { x: number; y: number };
    return {
      ...n,
      meta: {
        ...n.meta,
        editor: { ...editor, position: { x: p.x - minX, y: p.y - minY } },
      },
    };
  });
  const block: MyBlock = {
    id: `mb_${Date.now().toString(36)}`,
    name,
    nodes: normNodes,
    edges,
    createdAt: Date.now(),
  };
  writeAll([...safeParse(localStorage.getItem(STORAGE_KEY)), block]);
  return block;
}

/** マイブロックを削除する。 */
export function deleteMyBlock(id: string): void {
  writeAll(safeParse(localStorage.getItem(STORAGE_KEY)).filter((b) => b.id !== id));
}

/** idからマイブロックを取得する。 */
export function getMyBlock(id: string): MyBlock | undefined {
  return safeParse(localStorage.getItem(STORAGE_KEY)).find((b) => b.id === id);
}

// --- Unit定義のマイブロック/ギャラリー統合(localStorage保存) ---

/** 保存/削除でパレット等を再描画させるための変更通知イベント名。 */
export const MYUNITS_CHANGED_EVENT = 'magicdelta:myunits-changed';

function safeParseUnits(raw: string | null): EditorUnit[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? (v as EditorUnit[]) : [];
  } catch {
    return [];
  }
}

/** 保存済みマイUnit一覧。 */
export function listMyUnits(): EditorUnit[] {
  return safeParseUnits(localStorage.getItem(UNITS_STORAGE_KEY));
}

/** Unit定義をマイUnitとして保存する(同一idは上書き)。 */
export function saveMyUnit(unit: EditorUnit): void {
  const list = listMyUnits().filter((u) => u.id !== unit.id);
  localStorage.setItem(UNITS_STORAGE_KEY, JSON.stringify([...list, unit]));
  window.dispatchEvent(new Event(MYUNITS_CHANGED_EVENT));
}

/** マイUnitを削除する。 */
export function deleteMyUnit(id: string): void {
  localStorage.setItem(
    UNITS_STORAGE_KEY,
    JSON.stringify(listMyUnits().filter((u) => u.id !== id)),
  );
  window.dispatchEvent(new Event(MYUNITS_CHANGED_EVENT));
}

/** 元idの型サフィックス(`u3_lutlayer`→`lutlayer` / `foo`→`node`) */
function suffixOf(id: string): string {
  const m = /^u\d+_(.+)$/.exec(id);
  if (m) return m[1];
  return id.replace(/[^a-zA-Z0-9]/g, '') || 'node';
}

/**
 * 保存テンプレートを配置用にid振り直しして {nodes, edges} を返す。
 * addSubgraph が座標オフセット(ドロップ位置)を適用するため座標はそのまま。
 */
export function instantiateMyBlock(
  block: MyBlock,
  alloc: IdAllocator,
): { nodes: IRNodeJson[]; edges: IREdgeJson[] } {
  const idMap = new Map<string, string>();
  for (const n of block.nodes) idMap.set(n.id, alloc(suffixOf(n.id)));
  const nodes = block.nodes.map((n) => ({ ...n, id: idMap.get(n.id)! }));
  const edges = block.edges.map((e) => ({
    ...e,
    src: idMap.get(e.src) ?? e.src,
    dst: idMap.get(e.dst) ?? e.dst,
  }));
  return { nodes, edges };
}
