// /api/meta → React Flow のブロック図(ノード+バスエッジ)。
// 表示グラフ = lowered IR(RTL 配線と 1:1)。ID 空間は name_map と一致する。
import type { Edge, Node } from '@xyflow/react';
import type { Meta } from '../api/types';
import { edgeBaseStyle } from './edgeStyle';

export const MULTI_INPUT_TYPES = new Set(['concat', 'add']);

export interface BlockData extends Record<string, unknown> {
  nodeId: string;
  nodeType: string;
  width: number;
  alias: boolean;
  scoreBits: number | null;
  k: number | null;
  inSlots: number;
  label: string | null;
}

export interface PortData extends Record<string, unknown> {
  portId: string;
  width: number;
  direction: 'in' | 'out';
}

/** エッジ id 規約は editor の buildEditorEdge と同一(slot>0 で #s{slot})。 */
export function edgeId(src: string, dst: string, slot: number): string {
  return slot > 0 ? `${src}->${dst}#s${slot}` : `${src}->${dst}`;
}

export function sourceWidth(meta: Meta, id: string): number | null {
  const port = meta.inputs.find((p) => p.id === id);
  if (port) return port.width;
  const node = meta.nodes.find((n) => n.id === id);
  return node ? node.width : null;
}

export function metaToFlow(
  meta: Meta,
  positions: Record<string, { x: number; y: number }>,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const typeById = new Map(meta.nodes.map((n) => [n.id, n.type]));

  // 多入力ノード(concat/add)の入力スロット数(エッジの slot 最大値+1)
  const inSlots = new Map<string, number>();
  for (const e of meta.edges) {
    inSlots.set(e.dst, Math.max(inSlots.get(e.dst) ?? 1, e.slot + 1));
  }

  const pos = (id: string, fallback: { x: number; y: number }) =>
    positions[id] ?? fallback;

  meta.inputs.forEach((p, i) => {
    nodes.push({
      id: p.id,
      type: 'port',
      position: pos(p.id, { x: 0, y: 120 * i }),
      data: { portId: p.id, width: p.width, direction: 'in' } as PortData,
    });
  });
  meta.nodes.forEach((n, i) => {
    nodes.push({
      id: n.id,
      type: 'block',
      position: pos(n.id, { x: 260 * (i + 1), y: 0 }),
      data: {
        nodeId: n.id,
        nodeType: n.type,
        width: n.width,
        alias: n.alias ?? false,
        scoreBits: n.score_bits ?? null,
        k: meta.k,
        inSlots: MULTI_INPUT_TYPES.has(n.type)
          ? (inSlots.get(n.id) ?? 1)
          : 1,
        label: n.label ?? null,
      } as BlockData,
    });
  });
  meta.outputs.forEach((p, i) => {
    nodes.push({
      id: p.id,
      type: 'port',
      position: pos(p.id, { x: 260 * (meta.nodes.length + 1), y: 120 * i }),
      data: { portId: p.id, width: p.width, direction: 'out' } as PortData,
    });
  });

  for (const e of meta.edges) {
    const multi = MULTI_INPUT_TYPES.has(typeById.get(e.dst) ?? '');
    edges.push({
      id: edgeId(e.src, e.dst, e.slot),
      source: e.src,
      target: e.dst,
      targetHandle: multi ? `s${e.slot}` : undefined,
      style: edgeBaseStyle(sourceWidth(meta, e.src)),
    });
  }
  return { nodes, edges };
}
