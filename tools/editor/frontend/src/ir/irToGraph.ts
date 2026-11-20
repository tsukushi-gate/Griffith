// IR JSON → React Flowノード/エッジ変換。
// meta.editor.position(ノード) / metadata.editor.ports(ポート)があれば
// その座標を優先し、なければ左→右の等間隔配置にフォールバックする。
// v0.2: dst_slotをtargetHandle 's{slot}'へ写像し、多入力ノード
// (concat/add)は専用UIノードへ変換する。addのslot1以降は残差エッジ描画。

import type { Edge, Node } from '@xyflow/react';
import type { CSSProperties } from 'react';
import { ttToGateIdN2 } from './gates';
import type {
  AddMode,
  AddParams,
  ConcatParams,
  Conv2dParams,
  GroupSumParams,
  IREdgeJson,
  IRGraphJson,
  IRNodeJson,
  IRNodeType,
  LutLayerParams,
  OrPool2dParams,
} from './schema';
import { edgeSlot, isMultiInputType } from './schema';
import { conv2dOutShape, orPool2dOutShape } from './shapeInference';

const X_SPACING = 260;
const BASE_Y = 120;

export type LutLayerNodeData = {
  label: string;
  inWidth: number;
  outWidth: number;
  nInputs: number;
  initialGateIds: number[];
  parameterization: string;
  connections: string;
  /** 構造化配線の引数(train_info.wiring)。random / unique では undefined */
  wiring?: Record<string, unknown>;
  selection: string;
  initialized: boolean;
  [key: string]: unknown;
};

export type GroupSumNodeData = {
  label: string;
  inWidth: number;
  k: number;
  tau: number;
  /** クラスごとの整数の判定バイアス(params.bias)。無ければ undefined */
  bias?: number[];
  [key: string]: unknown;
};

export type GenericNodeData = {
  label: string;
  subtitle: string;
  kind: string; // 'input' | 'output' | IRノード型
  width?: number; // binarize/flatten用
  threshold?: number; // binarize用
  portWidth?: number; // 入出力ポート用
  [key: string]: unknown;
};

export type ConcatNodeData = {
  label: string;
  inWidths: number[]; // slot順(接続から導出)
  outWidth: number;
  [key: string]: unknown;
};

export type AddNodeData = {
  label: string;
  width: number;
  mode: AddMode;
  [key: string]: unknown;
};

export type ConvNodeData = {
  label: string;
  kind: 'conv2d' | 'or_pool2d';
  inShape: number[]; // [C, H, W]
  outShape: number[] | null; // [C', H', W'](params不正時null)
  inWidth: number;
  outWidth: number | null;
  kernel: string; // 表示用 "3x3" 等
  stride: string;
  padding: string;
  [key: string]: unknown;
};

export type LutLayerFlowNode = Node<LutLayerNodeData, 'lutLayer'>;
export type GroupSumFlowNode = Node<GroupSumNodeData, 'groupSum'>;
export type GenericFlowNode = Node<GenericNodeData, 'generic'>;
export type ConcatFlowNode = Node<ConcatNodeData, 'concat'>;
export type AddFlowNode = Node<AddNodeData, 'add'>;
export type ConvFlowNode = Node<ConvNodeData, 'conv'>;
export type EditorNode =
  | LutLayerFlowNode
  | GroupSumFlowNode
  | GenericFlowNode
  | ConcatFlowNode
  | AddFlowNode
  | ConvFlowNode;

/** 残差エッジ(addのslot1以降)の描画スタイル: 破線+slot-2色 */
export const RESIDUAL_EDGE_STYLE: CSSProperties = {
  stroke: 'var(--slot-2)',
  strokeWidth: 1.5,
  strokeDasharray: '6 3',
};

/**
 * IRエッジ → React Flowエッジ(slot対応、残差装飾つき)。
 * - slot 0: 既存規約のid `${src}->${dst}`(診断edgeIdと一致)
 * - slot 1以降: `${src}->${dst}#s{slot}`(同一ペア多重エッジのid衝突回避)
 * - 多入力ノード行きはtargetHandle 's{slot}' を明示(s0含む)
 */
export function buildEditorEdge(
  src: string,
  dst: string,
  slot: number,
  dstType: IRNodeType | undefined,
): Edge {
  const edge: Edge = {
    id: slot > 0 ? `${src}->${dst}#s${slot}` : `${src}->${dst}`,
    source: src,
    target: dst,
  };
  if (dstType !== undefined && isMultiInputType(dstType)) {
    edge.targetHandle = `s${slot}`;
  }
  if (dstType === 'add' && slot >= 1) {
    edge.style = RESIDUAL_EDGE_STYLE;
  }
  return edge;
}

/** IRノードのmeta.editor.positionがあれば優先する(DnD編集の保存先規約)。 */
function metaPosition(n: IRNodeJson): { x: number; y: number } | null {
  const editor = n.meta?.editor as { position?: { x: number; y: number } } | undefined;
  if (editor?.position && typeof editor.position.x === 'number') {
    return editor.position;
  }
  return null;
}

/** ポート座標は graph.metadata.editor.ports[portId].position に保存される。 */
function portPosition(ir: IRGraphJson, portId: string): { x: number; y: number } | null {
  const editor = ir.metadata?.editor as
    | { ports?: Record<string, { position?: { x: number; y: number } }> }
    | undefined;
  const pos = editor?.ports?.[portId]?.position;
  if (pos && typeof pos.x === 'number' && typeof pos.y === 'number') {
    return pos;
  }
  return null;
}

/** N=2のtruth_tablesからノードミニバー用の初期ゲートIDを導出する。 */
export function gateIdsFromParams(p: LutLayerParams): number[] {
  if (p.n_inputs !== 2) return [];
  return (p.truth_tables ?? []).map((t) => ttToGateIdN2(parseInt(t, 16)));
}

/** 保存済み座標を持たないノード/ポートのid集合(自動レイアウト対象の判定用)。 */
export function idsWithoutStoredPosition(ir: IRGraphJson): Set<string> {
  const missing = new Set<string>();
  for (const n of ir.nodes) {
    if (metaPosition(n) === null) missing.add(n.id);
  }
  for (const p of [...ir.inputs, ...ir.outputs]) {
    if (portPosition(ir, p.id) === null) missing.add(p.id);
  }
  return missing;
}

/** kernel/stride/padding の表示文字列("3x3" / "1x2" 等) */
function pairLabel(v: unknown): string {
  if (Array.isArray(v)) return `${v[0]}x${v[1]}`;
  return `${v}x${v}`;
}

export function irNodeToFlow(n: IRNodeJson, x: number): EditorNode {
  const position = metaPosition(n) ?? { x, y: BASE_Y };
  const editorMeta = n.meta?.editor as { label?: string; detail?: string } | undefined;

  if (n.type === 'lut_layer') {
    const p = n.params as unknown as LutLayerParams;
    return {
      id: n.id,
      type: 'lutLayer',
      position,
      data: {
        label: n.id,
        inWidth: p.in_width,
        outWidth: p.out_width,
        nInputs: p.n_inputs,
        initialGateIds: gateIdsFromParams(p),
        parameterization: (n.train_info.parameterization as string) ?? 'softmax',
        connections: (n.train_info.connections as string) ?? 'random',
        ...(n.train_info.wiring && typeof n.train_info.wiring === 'object'
          ? { wiring: { ...(n.train_info.wiring as Record<string, unknown>) } }
          : {}),
        selection: (n.train_info.selection as string) ?? 'soft',
        initialized: n.train_info.initialized !== false,
      },
    };
  }

  if (n.type === 'group_sum') {
    const p = n.params as unknown as GroupSumParams;
    return {
      id: n.id,
      type: 'groupSum',
      position,
      data: {
        label: n.id,
        inWidth: p.in_width,
        k: p.k,
        tau: p.tau,
        ...(Array.isArray(p.bias) ? { bias: p.bias.map(Number) } : {}),
      },
    };
  }

  if (n.type === 'concat') {
    const p = n.params as unknown as ConcatParams;
    const inWidths = Array.isArray(p.in_widths) ? p.in_widths.map(Number) : [1, 1];
    return {
      id: n.id,
      type: 'concat',
      position,
      data: {
        label: n.id,
        inWidths,
        outWidth: inWidths.reduce((a, w) => a + w, 0),
      },
    };
  }

  if (n.type === 'add') {
    const p = n.params as unknown as AddParams;
    return {
      id: n.id,
      type: 'add',
      position,
      data: {
        label: n.id,
        width: p.width ?? 1,
        mode: p.mode ?? 'or',
      },
    };
  }

  if (n.type === 'conv2d' || n.type === 'or_pool2d') {
    const p = n.params as Record<string, unknown>;
    const inShapeArr = Array.isArray(p.in_shape) ? p.in_shape.map(Number) : [1, 1, 1];
    const outShape =
      n.type === 'conv2d' ? conv2dOutShape(p) : orPool2dOutShape(p);
    const conv = n.params as unknown as Conv2dParams & OrPool2dParams;
    return {
      id: n.id,
      type: 'conv',
      position,
      data: {
        label: n.id,
        kind: n.type,
        inShape: inShapeArr,
        outShape,
        inWidth: inShapeArr.reduce((a, v) => a * v, 1),
        outWidth: outShape ? outShape[0] * outShape[1] * outShape[2] : null,
        kernel: pairLabel(conv.kernel_size),
        stride: pairLabel(conv.stride),
        padding: pairLabel(conv.padding ?? 0),
      },
    };
  }

  const width = (n.params.width ?? n.params.out_width ?? '?') as number | string;
  const subtitle =
    n.type === 'constant'
      ? `constant / width=${width} / bits=${Array.isArray(n.params.bits) ? n.params.bits.join('') : '?'}`
      : n.type === 'blackbox' && editorMeta?.detail
        ? `${n.type} / ${editorMeta.detail}`
      : `${n.type} / width=${width}`;
  return {
    id: n.id,
    type: 'generic',
    position,
    data: {
      label: editorMeta?.label ?? n.id,
      subtitle,
      kind: n.type,
      width: typeof width === 'number' ? width : undefined,
      bits: Array.isArray(n.params.bits) ? n.params.bits.map(Number) : undefined,
      threshold:
        typeof n.params.threshold === 'number' ? n.params.threshold : undefined,
    },
  };
}

/** IRエッジ列 → React Flowエッジ列(dst_slot→targetHandle写像) */
export function irEdgesToFlow(
  edges: IREdgeJson[],
  nodes: IRNodeJson[],
): Edge[] {
  const typeById = new Map<string, IRNodeType>(nodes.map((n) => [n.id, n.type]));
  return edges.map((e) =>
    buildEditorEdge(e.src, e.dst, edgeSlot(e), typeById.get(e.dst)),
  );
}

export function irToFlow(ir: IRGraphJson): { nodes: EditorNode[]; edges: Edge[] } {
  const nodes: EditorNode[] = [];
  let col = 0;

  for (const port of ir.inputs) {
    nodes.push({
      id: port.id,
      type: 'generic',
      position: portPosition(ir, port.id) ?? { x: col * X_SPACING, y: BASE_Y },
      deletable: false, // 入出力ポートは削除不可(保護)
      data: {
        label: port.id,
        subtitle: `入力 / width=${port.width}`,
        kind: 'input',
        portWidth: port.width,
      },
    });
    col += 1;
  }

  for (const n of ir.nodes) {
    nodes.push(irNodeToFlow(n, col * X_SPACING));
    col += 1;
  }

  for (const port of ir.outputs) {
    nodes.push({
      id: port.id,
      type: 'generic',
      position: portPosition(ir, port.id) ?? { x: col * X_SPACING, y: BASE_Y },
      deletable: false,
      data: {
        label: port.id,
        subtitle: `出力 / width=${port.width}${port.semantics ? ` (${port.semantics})` : ''}`,
        kind: 'output',
        portWidth: port.width,
      },
    });
    col += 1;
  }

  return { nodes, edges: irEdgesToFlow(ir.edges, ir.nodes) };
}
