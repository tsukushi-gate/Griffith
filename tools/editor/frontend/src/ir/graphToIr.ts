// React Flowグラフ → IR JSON合成(v0.2多入力DAG対応)。
// フロントは常にvalidな形のIRを合成する(draftノード規約):
//   - placeholder indices: indices[j][o] = (o*n + j) % in_width
//   - placeholder tt: pass-through 'A' = '0x' + 'a' × (2^N / 4)
//   - train_info = {parameterization, connections, selection, initialized: false}
// 構造不変ノードはbaseIrのindices/truth_tables/train_infoを保全し、
// 構造変更ノード(in/out幅・n_inputs・parameterization変更)はplaceholderへ
// 再合成する(=weights破棄)。
// dst_slot規約: targetHandle 's{slot}'→dst_slot。slot 0はJSONキーを出力しない
// (バックエンドschema.pyと同一規約)。concatのin_widthsは接続slot順に上流幅から導出。

import type { Edge } from '@xyflow/react';
import type {
  AddNodeData,
  ConcatNodeData,
  ConvNodeData,
  EditorNode,
  GenericNodeData,
  GroupSumNodeData,
  LutLayerNodeData,
} from './irToGraph';
import type {
  IREdgeJson,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
} from './schema';

/** draft規約のplaceholder真理値表(pass-through 'A')。 */
export function passThroughTT(nInputs: number): string {
  return '0x' + 'a'.repeat((1 << nInputs) / 4);
}

/** draft規約のplaceholder接続: indices[j][o] = (o*n + j) % in_width。 */
export function placeholderIndices(
  inWidth: number,
  outWidth: number,
  nInputs: number,
): number[][] {
  const rows: number[][] = [];
  for (let j = 0; j < nInputs; j++) {
    const row: number[] = new Array<number>(outWidth);
    for (let o = 0; o < outWidth; o++) {
      row[o] = (o * nInputs + j) % inWidth;
    }
    rows.push(row);
  }
  return rows;
}

/** targetHandle('s{slot}')→slot番号。ハンドルなし/不正は0(slot 0規約) */
export function slotFromHandle(handle: string | null | undefined): number {
  if (!handle) return 0;
  const m = /^s(\d+)$/.exec(handle);
  return m ? Number(m[1]) : 0;
}

function roundPos(p: { x: number; y: number }): { x: number; y: number } {
  return { x: Math.round(p.x), y: Math.round(p.y) };
}

/** slot別の導出入力幅(キー=slot番号) */
type SlotWidths = Map<number, number>;

/**
 * concatのin_widths導出: 接続slot順に上流幅を使い、未解決slotは
 * baseIr/表示データへフォールバックする(最低2slot)。
 */
function concatInWidths(
  fn: EditorNode,
  slots: SlotWidths,
  base: IRNodeJson | undefined,
): number[] {
  const baseWidths = Array.isArray(base?.params.in_widths)
    ? (base.params.in_widths as number[])
    : undefined;
  const dataWidths = (fn.data as ConcatNodeData).inWidths;
  const maxConnected = slots.size > 0 ? Math.max(...slots.keys()) : -1;
  const n = Math.max(maxConnected + 1, 2);
  const result: number[] = [];
  for (let s = 0; s < n; s++) {
    result.push(slots.get(s) ?? baseWidths?.[s] ?? dataWidths?.[s] ?? 1);
  }
  return result;
}

/** addのwidth導出: slot0優先で上流幅、なければbase/表示データ */
function addWidth(
  fn: EditorNode,
  slots: SlotWidths,
  base: IRNodeJson | undefined,
): number {
  const minSlot = slots.size > 0 ? Math.min(...slots.keys()) : null;
  const derived = minSlot !== null ? slots.get(minSlot) : undefined;
  const baseWidth =
    typeof base?.params.width === 'number' ? base.params.width : undefined;
  return derived ?? baseWidth ?? (fn.data as AddNodeData).width ?? 1;
}

/** flowノードの出力幅(接続追従の導出元)。 */
function flowOutWidth(
  node: EditorNode,
  derivedIn: Map<string, SlotWidths>,
  baseById: Map<string, IRNodeJson>,
): number | null {
  const slots = derivedIn.get(node.id) ?? new Map<number, number>();
  if (node.type === 'lutLayer') {
    return (node.data as LutLayerNodeData).outWidth;
  }
  if (node.type === 'groupSum') {
    return (node.data as GroupSumNodeData).k;
  }
  if (node.type === 'concat') {
    return concatInWidths(node, slots, baseById.get(node.id)).reduce(
      (a, w) => a + w,
      0,
    );
  }
  if (node.type === 'add') {
    return addWidth(node, slots, baseById.get(node.id));
  }
  if (node.type === 'conv') {
    // conv/poolの出力幅はparams(in_shape等)のみで決まる(表示データに計算済み)
    return (node.data as ConvNodeData).outWidth;
  }
  const d = node.data as GenericNodeData;
  if (d.kind === 'input') return d.portWidth ?? null;
  if (d.kind === 'constant') return d.width ?? null;
  if (d.kind === 'binarize' || d.kind === 'flatten') {
    // 幅維持ノード: 入力幅に追従
    return slots.get(0) ?? d.width ?? null;
  }
  return null;
}

/**
 * lut_layer の train_info を表示データから組む。構造化配線の引数は
 * data.wiring にあるときだけ train_info.wiring に残し、random / unique へ戻したら消す
 * (古い引数が残ると build 時に draft の再生成が誤る)。
 */
function lutTrainInfo(
  base: Record<string, unknown>,
  d: LutLayerNodeData,
): Record<string, unknown> {
  const info: Record<string, unknown> = {
    ...base,
    connections: d.connections,
    selection: d.selection,
  };
  if (d.wiring && typeof d.wiring === 'object' && d.connections !== 'random'
    && d.connections !== 'unique') {
    info.wiring = { ...d.wiring };
  } else {
    delete info.wiring;
  }
  return info;
}

/**
 * flowグラフ(位置・トポロジ・編集パラメータ)とbaseIr(重み・メタ)から
 * IR JSONを合成する。
 */
export function flowToIr(
  nodes: EditorNode[],
  edges: Edge[],
  baseIr: IRGraphJson,
): IRGraphJson {
  const baseById = new Map(baseIr.nodes.map((n) => [n.id, n]));
  const flowById = new Map(nodes.map((n) => [n.id, n]));
  const inputIds = new Set(baseIr.inputs.map((p) => p.id));
  const outputIds = new Set(baseIr.outputs.map((p) => p.id));

  // dst_slot=0はキー省略(バックエンドschema.pyと同一の表現安定規約)
  const irEdges: IREdgeJson[] = edges.map((e) => {
    const slot = slotFromHandle(e.targetHandle);
    return slot > 0
      ? { src: e.source, dst: e.target, dst_slot: slot }
      : { src: e.source, dst: e.target };
  });

  // 各ノードのslot別入力エッジ(slot昇順)
  const inEdgesOf = new Map<string, { src: string; slot: number }[]>();
  for (const e of irEdges) {
    const list = inEdgesOf.get(e.dst) ?? [];
    list.push({ src: e.src, slot: e.dst_slot ?? 0 });
    inEdgesOf.set(e.dst, list);
  }
  for (const list of inEdgesOf.values()) {
    list.sort((a, b) => a.slot - b.slot);
  }

  // 入力ポート(幅編集を反映)
  const inputs = baseIr.inputs.map((p) => {
    const fn = flowById.get(p.id);
    const w = fn ? (fn.data as GenericNodeData).portWidth : undefined;
    return { ...p, width: typeof w === 'number' ? w : p.width };
  });
  const inputWidthById = new Map(inputs.map((p) => [p.id, p.width]));

  // 各ノードのslot別導出入力幅をトポロジ順に解決(循環はvisitingで打ち切り)
  const derivedIn = new Map<string, SlotWidths>();
  const resolveOut = (id: string, visiting: Set<string>): number | null => {
    if (inputIds.has(id)) return inputWidthById.get(id) ?? null;
    const fn = flowById.get(id);
    if (!fn || visiting.has(id)) return null;
    visiting.add(id);
    if (!derivedIn.has(id)) {
      const slots: SlotWidths = new Map();
      for (const { src, slot } of inEdgesOf.get(id) ?? []) {
        const w = resolveOut(src, visiting);
        if (w !== null && !slots.has(slot)) slots.set(slot, w);
      }
      derivedIn.set(id, slots);
    }
    visiting.delete(id);
    return flowOutWidth(fn, derivedIn, baseById);
  };
  for (const fn of nodes) {
    if (!inputIds.has(fn.id) && !outputIds.has(fn.id)) {
      resolveOut(fn.id, new Set());
    }
  }

  const irNodes: IRNodeJson[] = [];
  for (const fn of nodes) {
    if (inputIds.has(fn.id) || outputIds.has(fn.id)) continue;
    const base = baseById.get(fn.id);
    const position = roundPos(fn.position);
    const baseEditor = (base?.meta?.editor as Record<string, unknown>) ?? {};
    const meta = {
      ...(base?.meta ?? {}),
      editor: { ...baseEditor, position },
    };
    const slots = derivedIn.get(fn.id) ?? new Map<number, number>();
    const dIn = slots.get(0) ?? null;

    if (fn.type === 'lutLayer') {
      const d = fn.data as LutLayerNodeData;
      const bp = base?.params as unknown as LutLayerParams | undefined;
      const inWidth = dIn ?? bp?.in_width ?? Math.max(d.inWidth, 1);
      const baseParameterization =
        (base?.train_info.parameterization as string) ?? 'softmax';
      const unchanged =
        !!bp &&
        bp.in_width === inWidth &&
        bp.out_width === d.outWidth &&
        bp.n_inputs === d.nInputs &&
        Array.isArray(bp.indices) &&
        Array.isArray(bp.truth_tables) &&
        baseParameterization === d.parameterization;

      if (unchanged) {
        // 構造不変: indices/truth_tables/train_infoを保全
        irNodes.push({
          id: fn.id,
          type: 'lut_layer',
          params: { ...(base!.params as Record<string, unknown>) },
          train_info: lutTrainInfo(base!.train_info, d),
          meta,
        });
      } else {
        // 構造変更: placeholder再合成(weights破棄、draft化)
        irNodes.push({
          id: fn.id,
          type: 'lut_layer',
          params: {
            in_width: inWidth,
            out_width: d.outWidth,
            n_inputs: d.nInputs,
            indices: placeholderIndices(inWidth, d.outWidth, d.nInputs),
            truth_tables: new Array<string>(d.outWidth).fill(
              passThroughTT(d.nInputs),
            ),
          },
          train_info: {
            ...lutTrainInfo({ parameterization: d.parameterization }, d),
            initialized: false,
          },
          meta,
        });
      }
    } else if (fn.type === 'groupSum') {
      const d = fn.data as GroupSumNodeData;
      // 判定バイアスは長さが k と一致するときだけ保つ(k 変更で捨てる)
      const bias =
        Array.isArray(d.bias) && d.bias.length === d.k && d.bias.some((b) => b !== 0)
          ? d.bias.map((b) => Math.round(Number(b)))
          : undefined;
      irNodes.push({
        id: fn.id,
        type: 'group_sum',
        params: {
          in_width: dIn ?? (base?.params.in_width as number) ?? d.inWidth,
          k: d.k,
          tau: d.tau,
          ...(bias ? { bias } : {}),
        },
        train_info: base?.train_info ?? {},
        meta,
      });
    } else if (fn.type === 'concat') {
      irNodes.push({
        id: fn.id,
        type: 'concat',
        params: {
          ...(base?.params ?? {}),
          in_widths: concatInWidths(fn, slots, base),
        },
        train_info: base?.train_info ?? {},
        meta,
      });
    } else if (fn.type === 'add') {
      const d = fn.data as AddNodeData;
      irNodes.push({
        id: fn.id,
        type: 'add',
        params: {
          ...(base?.params ?? {}),
          width: addWidth(fn, slots, base),
          mode: d.mode,
        },
        train_info: base?.train_info ?? {},
        meta,
      });
    } else if (fn.type === 'conv') {
      // conv/poolはparams編集非対応(表示・検証・往復のみ): baseを保全
      const d = fn.data as ConvNodeData;
      irNodes.push({
        id: fn.id,
        type: d.kind,
        params: { ...(base?.params ?? {}) },
        train_info: base?.train_info ?? {},
        meta,
      });
    } else {
      const d = fn.data as GenericNodeData;
      let params: Record<string, unknown>;
      if (d.kind === 'binarize') {
        params = { width: dIn ?? d.width ?? 1, threshold: d.threshold ?? 0.5 };
      } else if (d.kind === 'flatten') {
        params = { ...(base?.params ?? {}), width: dIn ?? d.width ?? 1 };
      } else if (d.kind === 'constant') {
        const width = d.width ?? 1;
        const dataBits = Array.isArray(d.bits) ? (d.bits as number[]) : undefined;
        const baseBits = Array.isArray(base?.params.bits)
          ? (base.params.bits as number[])
          : undefined;
        params = {
          width,
          bits:
            dataBits && dataBits.length === width
              ? dataBits.map((b) => (b ? 1 : 0))
              : baseBits && baseBits.length === width
                ? baseBits
                : new Array<number>(width).fill(0),
        };
      } else {
        params = { ...(base?.params ?? {}) };
      }
      irNodes.push({
        id: fn.id,
        type: (d.kind as IRNodeJson['type']) ?? 'blackbox',
        params,
        train_info: base?.train_info ?? {},
        meta,
      });
    }
  }

  // 出力ポート: 幅は接続元から導出(未接続時はbase維持)
  const outWidthOf = (id: string): number | null => {
    if (inputIds.has(id)) return inputWidthById.get(id) ?? null;
    const fn = flowById.get(id);
    return fn ? flowOutWidth(fn, derivedIn, baseById) : null;
  };
  const srcOfOutput = new Map<string, string>();
  for (const e of irEdges) {
    if (outputIds.has(e.dst) && !srcOfOutput.has(e.dst)) {
      srcOfOutput.set(e.dst, e.src);
    }
  }
  const outputs = baseIr.outputs.map((p) => {
    const src = srcOfOutput.get(p.id);
    const w = src !== undefined ? outWidthOf(src) : null;
    return { ...p, width: w ?? p.width };
  });

  // ポート座標は metadata.editor.ports[portId].position へ
  const baseMetaEditor =
    (baseIr.metadata?.editor as Record<string, unknown>) ?? {};
  const ports: Record<string, { position: { x: number; y: number } }> = {};
  for (const p of [...baseIr.inputs, ...baseIr.outputs]) {
    const fn = flowById.get(p.id);
    if (fn) ports[p.id] = { position: roundPos(fn.position) };
  }
  const metadata = {
    ...baseIr.metadata,
    editor: { ...baseMetaEditor, ports },
  };

  return {
    format: baseIr.format,
    version: baseIr.version,
    metadata,
    inputs,
    outputs,
    nodes: irNodes,
    edges: irEdges,
  };
}
