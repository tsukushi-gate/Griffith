// ブロック展開: lut_layer をゲート格子グループ+ビット単位配線へ差し替える。
// 幅 > EXPAND_MAX_WIDTH は拒否(MNIST 級対策。プローブでの観測へ誘導)。
import type { Edge, Node } from '@xyflow/react';
import type { Meta } from '../api/types';
import type { LutParams } from '../store/detailStore';

export const EXPAND_MAX_WIDTH = 128;

export const GATE_W = 66;
export const GATE_H = 40;
export const GATE_GAP = 8;
export const GROUP_PAD = 12;
export const GROUP_HEAD = 30;

export function gateGridCols(outWidth: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(outWidth / 2)));
}

export function groupSize(outWidth: number): { width: number; height: number } {
  const cols = gateGridCols(outWidth);
  const rows = Math.ceil(outWidth / cols);
  return {
    width: GROUP_PAD * 2 + cols * (GATE_W + GATE_GAP) - GATE_GAP,
    height: GROUP_HEAD + GROUP_PAD + rows * (GATE_H + GATE_GAP) - GATE_GAP + GROUP_PAD,
  };
}

export interface GateData extends Record<string, unknown> {
  layerId: string;
  gateIndex: number;
  tt: number;
  nInputs: number;
  srcId: string;      // 入力バスの供給元ノード id(値購読用)
  srcBits: number[];  // 各スロットが参照する入力バスの bit 番号
}

/** 展開対象レイヤの入力供給元(lut_layer は単一入力)。 */
export function layerSourceId(meta: Meta, layerId: string): string | null {
  const e = meta.edges.find((e) => e.dst === layerId);
  return e ? e.src : null;
}

/**
 * ブロックノード列にゲート展開を適用する。
 * - 展開レイヤのノードを group 化し、子ゲートノードを格子配置
 * - 供給元 → レイヤのバスエッジを、供給元 → 各ゲートのビットエッジ群へ差し替え
 */
export function applyExpansion(
  meta: Meta,
  nodes: Node[],
  edges: Edge[],
  expanded: string[],
  paramsById: Record<string, LutParams>,
): { nodes: Node[]; edges: Edge[] } {
  let outNodes = [...nodes];
  let outEdges = [...edges];

  for (const layerId of expanded) {
    const params = paramsById[layerId];
    const idx = outNodes.findIndex((n) => n.id === layerId);
    if (!params || idx < 0) continue;
    if (params.out_width > EXPAND_MAX_WIDTH) continue;
    const srcId = layerSourceId(meta, layerId);
    if (!srcId) continue;

    const size = groupSize(params.out_width);
    const block = outNodes[idx]!;
    outNodes[idx] = {
      ...block,
      type: 'lutGroup',
      style: { width: size.width, height: size.height },
      data: { ...block.data, expanded: true },
    };

    const cols = gateGridCols(params.out_width);
    const gates: Node[] = [];
    for (let o = 0; o < params.out_width; o++) {
      const col = o % cols;
      const row = Math.floor(o / cols);
      gates.push({
        id: `${layerId}::g${o}`,
        type: 'gate',
        parentId: layerId,
        extent: 'parent',
        draggable: false,
        selectable: false,
        position: {
          x: GROUP_PAD + col * (GATE_W + GATE_GAP),
          y: GROUP_HEAD + GROUP_PAD + row * (GATE_H + GATE_GAP),
        },
        data: {
          layerId,
          gateIndex: o,
          tt: typeof params.truth_tables[o] === 'number'
            ? (params.truth_tables[o] as number)
            : parseInt(params.truth_tables[o] as string, 16),
          nInputs: params.n_inputs,
          srcId,
          srcBits: params.indices.map((slot) => slot[o]!),
        } as GateData,
      });
    }
    // 親(group)の直後に子を挿入(React Flow v12: 親が子より先に必要)
    outNodes = [
      ...outNodes.slice(0, idx + 1),
      ...gates,
      ...outNodes.slice(idx + 1),
    ];

    // バスエッジ(src→layer)をビットエッジへ差し替え
    outEdges = outEdges.filter(
      (e) => !(e.source === srcId && e.target === layerId),
    );
    for (let o = 0; o < params.out_width; o++) {
      for (let j = 0; j < params.n_inputs; j++) {
        outEdges.push({
          id: `bit:${layerId}:${o}:${j}`,
          source: srcId,
          target: `${layerId}::g${o}`,
          data: { srcBit: paramsBit(params, j, o) },
          style: { strokeWidth: 0.8, stroke: 'var(--edge-off)' },
        });
      }
    }
  }
  return { nodes: outNodes, edges: outEdges };
}

function paramsBit(params: LutParams, slot: number, gate: number): number {
  return params.indices[slot]![gate]!;
}
