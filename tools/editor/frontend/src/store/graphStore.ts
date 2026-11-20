// IRグラフ+編集状態のzustandストア(E2: DnD編集・形状診断・自動同期)。
// 編集actionは flowToIr → inferShapes を再実行し、800msデバウンスで
// PUT /api/models/current/ir へ同期する(形状エラー中は同期を保留)。
// zundo(temporal)でIR履歴を管理し undoGraph/redoGraph を提供する。
// 履歴の実体はIR参照のみ(equality)なので、選択・ドラッグ途中では積まれない。

import {
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react';
import { temporal } from 'zundo';
import { create } from 'zustand';
import { putCurrentIR } from '../api/rest';
import { saveMyBlock } from '../blocks/myBlocks';
import { readGroups, type EditorGroup } from '../editor/grouping';
import {
  expandUnit,
  markEditedInstances,
  nextInstanceId,
  offsetExpandedNodes,
  reexpandUnit,
  unitInstancesOf,
  unitTagOf,
  withUnitInstances,
  type EditorUnit,
  type ReexpandResult,
  type UnitInstance,
} from '../blocks/unitTypes';
import { topologicalOrder } from '../editor/nnccard/order';
import { layoutWithElk } from '../ir/autoLayout';
import { flowToIr, slotFromHandle } from '../ir/graphToIr';
import {
  buildEditorEdge,
  gateIdsFromParams,
  idsWithoutStoredPosition,
  irEdgesToFlow,
  irNodeToFlow,
  irToFlow,
  type AddNodeData,
  type ConcatNodeData,
  type EditorNode,
  type GenericNodeData,
  type GroupSumNodeData,
  type LutLayerNodeData,
} from '../ir/irToGraph';
import type {
  IREdgeJson,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
} from '../ir/schema';
import {
  hasShapeErrors,
  inferShapes,
  type ShapeDiagnostic,
} from '../ir/shapeInference';

const SYNC_DEBOUNCE_MS = 800;
const DEFAULT_LUT_OUT_WIDTH = 100;
const DEFAULT_GROUP_SUM_K = 10;

export type PaletteNodeType =
  | 'lut_layer'
  | 'group_sum'
  | 'binarize'
  | 'flatten'
  | 'concat'
  | 'add'
  | 'constant';

/** 多入力(slot連番)を取るflowノード型 */
const MULTI_INPUT_FLOW_TYPES = new Set(['concat', 'add']);

export type SyncState = 'idle' | 'pending' | 'synced' | 'error';

/** ir.metadata.editor.groups の各エントリ(キー=groupId)。*/
type StoredGroup = Omit<EditorGroup, 'id'>;

interface GraphState {
  ir: IRGraphJson | null;
  nodes: EditorNode[];
  edges: Edge[];
  selectedNodeId: string | null;
  dirty: boolean; // サーバ未同期の編集がある
  needsBuild: boolean; // 構造編集後、訓練前にPOST buildが必要
  diagnostics: ShapeDiagnostic[];
  syncState: SyncState;
  syncError: string | null;
  // モデル読込(setIR)完了時に増える。キャンバスが再fit-viewするトリガ(編集では増えない)
  fitNonce: number;
  setIR: (ir: IRGraphJson, opts?: { needsBuild?: boolean }) => void;
  onNodesChange: (changes: NodeChange<EditorNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (conn: Connection) => void;
  positionsChanged: () => void;
  addNodeFromPalette: (type: PaletteNodeType, position: { x: number; y: number }) => void;
  addSubgraph: (
    nodes: IRNodeJson[],
    edges: IREdgeJson[],
    position: { x: number; y: number },
  ) => void;
  deleteSelected: () => void;
  updateNodeData: (id: string, patch: Record<string, unknown>) => void;
  updatePortWidth: (id: string, width: number) => void;
  updateLutParams: (
    id: string,
    mutate: (params: LutLayerParams) => void,
    opts?: { sync?: boolean },
  ) => void;
  // 任意ノード(blackbox等)のparamsを構造不変で更新する。
  // in/out幅を変えない編集(torch_class/config等)専用。IRを直接差し替え表示更新する。
  updateNodeParams: (id: string, patch: Record<string, unknown>) => void;
  autoLayoutAll: (
    sizeOverrides?: Map<string, { width: number; height: number }>,
  ) => Promise<void>;
  syncNow: () => Promise<void>;
  setSelected: (id: string | null) => void;
  // サブサーキット化: グループはmeta.editor.groupsのみを更新する
  // 表示・編集メタデータで、IRのノード構造・エッジは変えない。
  createGroup: (name?: string) => string | null;
  ungroup: (groupId: string) => void;
  toggleGroupCollapse: (groupId: string) => void;
  renameGroup: (groupId: string, name: string) => void;
  moveGroup: (groupId: string, position: { x: number; y: number }) => void;
  saveGroupAsTemplate: (groupId: string, name: string) => void;
  // レイヤーコメント: metadata.editor.comments へ非破壊で永続化する。
  // 空文字/空白はキー削除。IR構造・graphStore.nodesは不変。
  setComment: (nodeId: string, text: string) => void;
  // NNC縦整列: トポロジカル順に座標だけを更新する。ノード/エッジ構造は不変。
  alignVerticalNnc: () => void;
  // Unit: Unitを引数で展開してドロップ位置へ配置し、台帳へ記録する。
  addUnitInstance: (
    unit: EditorUnit,
    argValues: Record<string, number | string>,
    position: { x: number; y: number },
  ) => void;
  // Unit定義変更の伝播: 同一unitIdの非editedインスタンスを再展開する(結果を返す)。
  reexpandUnitInstances: (unit: EditorUnit) => ReexpandResult;
  // Unit定義: metadata.editor.units を非破壊で更新して同期予約する。
  setUnits: (units: EditorUnit[]) => void;
}

const UNDO_LIMIT = 100;

let syncTimer: ReturnType<typeof setTimeout> | undefined;

/** デバウンスPUT予約(undo/redo復元からも呼ぶためモジュールスコープ)。 */
function scheduleSyncGlobal() {
  if (syncTimer !== undefined) clearTimeout(syncTimer);
  syncTimer = setTimeout(() => {
    void useGraphStore.getState().syncNow();
  }, SYNC_DEBOUNCE_MS);
}

/** 編集後のIRからflowノードの表示用データ(導出幅・draft状態)を更新する。 */
function refreshNodeData(fn: EditorNode, ir: IRGraphJson): EditorNode {
  const irNode = ir.nodes.find((n) => n.id === fn.id);

  if (fn.type === 'lutLayer' && irNode) {
    const p = irNode.params as unknown as LutLayerParams;
    const d = fn.data as LutLayerNodeData;
    return {
      ...fn,
      data: {
        ...d,
        inWidth: p.in_width,
        initialized: irNode.train_info.initialized !== false,
        initialGateIds: gateIdsFromParams(p),
      },
    } as EditorNode;
  }

  if (fn.type === 'groupSum' && irNode) {
    const d = fn.data as GroupSumNodeData;
    return {
      ...fn,
      data: { ...d, inWidth: (irNode.params.in_width as number) ?? d.inWidth },
    } as EditorNode;
  }

  if (fn.type === 'concat' && irNode) {
    const d = fn.data as ConcatNodeData;
    const inWidths = Array.isArray(irNode.params.in_widths)
      ? (irNode.params.in_widths as number[])
      : d.inWidths;
    return {
      ...fn,
      data: {
        ...d,
        inWidths,
        outWidth: inWidths.reduce((a, w) => a + w, 0),
      },
    } as EditorNode;
  }

  if (fn.type === 'add' && irNode) {
    const d = fn.data as AddNodeData;
    return {
      ...fn,
      data: { ...d, width: (irNode.params.width as number) ?? d.width },
    } as EditorNode;
  }

  const d = fn.data as GenericNodeData;
  if (irNode && (d.kind === 'binarize' || d.kind === 'flatten')) {
    const width = (irNode.params.width as number) ?? d.width;
    return {
      ...fn,
      data: { ...d, width, subtitle: `${d.kind} / width=${width}` },
    } as EditorNode;
  }
  if (irNode && d.kind === 'constant') {
    const width = (irNode.params.width as number) ?? d.width ?? 1;
    const bits = Array.isArray(irNode.params.bits)
      ? (irNode.params.bits as number[]).map((b) => (b ? 1 : 0))
      : new Array<number>(width).fill(0);
    return {
      ...fn,
      data: {
        ...d,
        width,
        bits,
        subtitle: `constant / width=${width} / bits=${bits.join('')}`,
      },
    } as EditorNode;
  }
  if (d.kind === 'output') {
    const port = ir.outputs.find((p) => p.id === fn.id);
    if (port) {
      return {
        ...fn,
        data: {
          ...d,
          portWidth: port.width,
          subtitle: `出力 / width=${port.width}${port.semantics ? ` (${port.semantics})` : ''}`,
        },
      } as EditorNode;
    }
  }
  return fn;
}

function nextDraftSeq(nodes: EditorNode[]): number {
  let max = 0;
  for (const n of nodes) {
    const m = /^u(\d+)_/.exec(n.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return max + 1;
}

/** 既存グループid(`g{n}`)と衝突しない次のgroupId。 */
function nextGroupId(groups: EditorGroup[]): string {
  let max = 0;
  for (const g of groups) {
    const m = /^g(\d+)$/.exec(g.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `g${max + 1}`;
}

/** ノード群の左上座標(折りたたみグループノードの初期位置)。 */
function minCorner(nodes: EditorNode[]): { x: number; y: number } {
  let x = Infinity;
  let y = Infinity;
  for (const n of nodes) {
    x = Math.min(x, n.position.x);
    y = Math.min(y, n.position.y);
  }
  return Number.isFinite(x) ? { x: Math.round(x), y: Math.round(y) } : { x: 0, y: 0 };
}

/** ir.metadata.editor から groups レコードを読み出す(なければ空)。 */
function groupsRecordOf(ir: IRGraphJson): Record<string, StoredGroup> {
  const editor = (ir.metadata?.editor as Record<string, unknown>) ?? {};
  return (editor.groups as Record<string, StoredGroup> | undefined) ?? {};
}

/** ir.metadata.editor.comments からレイヤーコメント(nodeId→本文)を読み出す。*/
export function commentsRecordOf(ir: IRGraphJson): Record<string, string> {
  const editor = (ir.metadata?.editor as Record<string, unknown>) ?? {};
  return (editor.comments as Record<string, string> | undefined) ?? {};
}

/** comments レコードを差し替えた新しいIRを返す(構造は不変。groupsと同じ非破壊metadata方式)。*/
function withCommentsRecord(
  ir: IRGraphJson,
  comments: Record<string, string>,
): IRGraphJson {
  const editor = (ir.metadata?.editor as Record<string, unknown>) ?? {};
  return {
    ...ir,
    metadata: { ...ir.metadata, editor: { ...editor, comments } },
  };
}

/** groups レコードを差し替えた新しいIRを返す(構造は不変)。 */
function withGroupsRecord(
  ir: IRGraphJson,
  groups: Record<string, StoredGroup>,
): IRGraphJson {
  const editor = (ir.metadata?.editor as Record<string, unknown>) ?? {};
  return {
    ...ir,
    metadata: { ...ir.metadata, editor: { ...editor, groups } },
  };
}

/** ノードidの所属Unitインスタンスに edited=true を立てた新IRを返す(所属なしなら同一IR)。 */
function withInstanceEditedForNode(ir: IRGraphJson, nodeId: string): IRGraphJson {
  const node = ir.nodes.find((n) => n.id === nodeId);
  if (!node) return ir;
  const tag = unitTagOf(node);
  if (!tag) return ir;
  const instances = unitInstancesOf(ir);
  let changed = false;
  const next = instances.map((inst) => {
    if (inst.instanceId === tag.instanceId && !inst.edited) {
      changed = true;
      return { ...inst, edited: true };
    }
    return inst;
  });
  return changed ? withUnitInstances(ir, next) : ir;
}

/** 削除済みメンバーをグループから除去し、実質1個以下になったグループは解散する。 */
function pruneGroups(ir: IRGraphJson): IRGraphJson {
  const groups = groupsRecordOf(ir);
  if (Object.keys(groups).length === 0) return ir;
  const valid = new Set(ir.nodes.map((n) => n.id));
  let changed = false;
  const next: Record<string, StoredGroup> = {};
  for (const [gid, g] of Object.entries(groups)) {
    const kept = g.nodeIds.filter((id) => valid.has(id));
    if (kept.length < 2) {
      changed = true;
      continue;
    }
    if (kept.length !== g.nodeIds.length) {
      changed = true;
      next[gid] = { ...g, nodeIds: kept };
    } else {
      next[gid] = g;
    }
  }
  return changed ? withGroupsRecord(ir, next) : ir;
}

export const useGraphStore = create<GraphState>()(
  temporal(
    (set, get) => {
      const scheduleSync = scheduleSyncGlobal;

  /** 編集後の全再計算: flowToIr → inferShapes → 表示更新 → 同期予約。
      baseOverrideはaddSubgraph(テンプレートノードのparams保全)用。 */
  const recompute = (
    nodes: EditorNode[],
    edges: Edge[],
    structural: boolean,
    baseOverride?: IRGraphJson,
  ) => {
    const base = baseOverride ?? get().ir;
    if (!base) {
      set({ nodes, edges });
      return;
    }
    const raw = flowToIr(nodes, edges, base);
    // 構造変更時のみ、削除メンバーをグループから掃除する
    let ir = structural ? pruneGroups(raw) : raw;
    // Unit: 構造編集でインスタンスの展開ノードがタッチされたら edited=true を立てる。
    // baseOverride指定時(addSubgraph/展開)は新規追加なので既存インスタンスは変化しない。
    const prevForEdit = get().ir;
    if (structural && prevForEdit && unitInstancesOf(ir).length > 0) {
      const marked = markEditedInstances(prevForEdit, ir);
      ir = withUnitInstances(ir, marked);
    }
    const { diagnostics } = inferShapes(ir.nodes, ir.edges, ir.inputs);
    set({
      ir,
      nodes: nodes.map((fn) => refreshNodeData(fn, ir)),
      edges,
      diagnostics,
      dirty: true,
      needsBuild: get().needsBuild || structural,
    });
    scheduleSync();
  };

  return {
    ir: null,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    dirty: false,
    needsBuild: false,
    diagnostics: [],
    syncState: 'idle',
    syncError: null,
    fitNonce: 0,

    setIR: (ir, opts) => {
      if (syncTimer !== undefined) {
        clearTimeout(syncTimer);
        syncTimer = undefined;
      }
      // モデル差し替え時はundo履歴をリセット(非同期整列後に実行順序の都合で空になる)
      queueMicrotask(() => useGraphStore.temporal.getState().clear());
      const { nodes, edges } = irToFlow(ir);
      const { diagnostics } = inferShapes(ir.nodes, ir.edges, ir.inputs);
      set({
        ir,
        nodes,
        edges,
        diagnostics,
        selectedNodeId: null,
        dirty: false,
        needsBuild: opts?.needsBuild ?? false,
        syncState: 'idle',
        syncError: null,
      });

      // 座標未保存ノードのみ非同期で自動整列。編集状態は変えない
      const missing = idsWithoutStoredPosition(ir);
      // 座標が揃っているモデルは即座にfit-viewを要求する
      if (missing.size === 0) {
        set({ fitNonce: get().fitNonce + 1 });
        return;
      }
      void layoutWithElk(nodes, edges)
        .then((positions) => {
          if (get().ir !== ir) return; // 別IRへ切り替わっていたら破棄
          // 整列後の座標を反映してからfit-viewを要求する(整列前にfitすると枠がずれる)
          set({
            nodes: get().nodes.map((n) => {
              const pos = missing.has(n.id) ? positions.get(n.id) : undefined;
              return pos ? { ...n, position: pos } : n;
            }),
            fitNonce: get().fitNonce + 1,
          });
        })
        .catch(() => undefined); // レイアウト失敗時はフォールバック配置のまま
    },

    onNodesChange: (changes) => {
      const nodes = applyNodeChanges(changes, get().nodes);
      if (changes.some((c) => c.type === 'remove')) {
        const ids = new Set(nodes.map((n) => n.id));
        const edges = get().edges.filter(
          (e) => ids.has(e.source) && ids.has(e.target),
        );
        recompute(nodes, edges, true);
      } else {
        set({ nodes });
      }
    },

    onEdgesChange: (changes) => {
      const edges = applyEdgeChanges(changes, get().edges);
      if (changes.some((c) => c.type === 'remove')) {
        recompute(get().nodes, edges, true);
      } else {
        set({ edges });
      }
    },

    onConnect: (conn) => {
      if (!isValidConnectionCheck(conn.source, conn.target, conn.targetHandle)) {
        return;
      }
      const state = get();
      const targetNode = state.nodes.find((n) => n.id === conn.target);
      const multi =
        targetNode !== undefined && MULTI_INPUT_FLOW_TYPES.has(targetNode.type ?? '');
      // 多入力ノード: ハンドル指定があればそのslot、なければ空きslotへ自動割当
      let slot = 0;
      if (multi) {
        const used = new Set(
          state.edges
            .filter((e) => e.target === conn.target)
            .map((e) => slotFromHandle(e.targetHandle)),
        );
        slot = conn.targetHandle
          ? slotFromHandle(conn.targetHandle)
          : nextFreeSlot(used);
        if (used.has(slot)) return; // 占有slotへの接続は拒否(二重ガード)
      }
      const dstType = multi
        ? (targetNode.type as 'concat' | 'add')
        : undefined;
      const edge = buildEditorEdge(conn.source, conn.target, slot, dstType);
      recompute(state.nodes, [...state.edges, edge], true);
    },

    // ドラッグ終了時: 座標をmeta.editor.positionへ書き戻す(構造は不変)
    positionsChanged: () => {
      recompute(get().nodes, get().edges, false);
    },

    addNodeFromPalette: (type, position) => {
      const state = get();
      if (!state.ir) return;
      const id = `u${nextDraftSeq(state.nodes)}_${type}`;
      let node: EditorNode;
      if (type === 'lut_layer') {
        node = {
          id,
          type: 'lutLayer',
          position,
          data: {
            label: id,
            inWidth: 1,
            outWidth: DEFAULT_LUT_OUT_WIDTH,
            nInputs: 2,
            initialGateIds: [],
            parameterization: 'softmax',
            connections: 'random',
            selection: 'soft',
            initialized: false,
          },
        };
      } else if (type === 'group_sum') {
        node = {
          id,
          type: 'groupSum',
          position,
          data: {
            label: id,
            inWidth: DEFAULT_GROUP_SUM_K,
            k: DEFAULT_GROUP_SUM_K,
            tau: 1.0,
          },
        };
      } else if (type === 'concat') {
        node = {
          id,
          type: 'concat',
          position,
          data: { label: id, inWidths: [1, 1], outWidth: 2 },
        };
      } else if (type === 'add') {
        node = {
          id,
          type: 'add',
          position,
          data: { label: id, width: 1, mode: 'or' },
        };
      } else if (type === 'constant') {
        node = {
          id,
          type: 'generic',
          position,
          data: {
            label: id,
            subtitle: 'constant / width=1 / bits=0',
            kind: 'constant',
            width: 1,
            bits: [0],
          },
        };
      } else if (type === 'flatten') {
        node = {
          id,
          type: 'generic',
          position,
          data: {
            label: id,
            subtitle: 'flatten / width=?',
            kind: 'flatten',
            width: 1,
          },
        };
      } else {
        node = {
          id,
          type: 'generic',
          position,
          data: {
            label: id,
            subtitle: 'binarize / width=?',
            kind: 'binarize',
            width: 1,
            threshold: 0.5,
          },
        };
      }
      recompute([...state.nodes, node], state.edges, true);
    },

    // 複合ブロック: IR部分グラフをドロップ位置へ展開配置する。
    // baseIrへ先にマージしてからrecomputeすることで、テンプレートの
    // indices/truth_tables等のparamsをflowToIrの構造不変判定で保全する。
    addSubgraph: (subNodes, subEdges, position) => {
      const state = get();
      if (!state.ir) return;
      const placed = subNodes.map((n) => {
        const editor = (n.meta?.editor ?? {}) as Record<string, unknown>;
        const rel = (editor.position ?? { x: 0, y: 0 }) as { x: number; y: number };
        return {
          ...n,
          meta: {
            ...n.meta,
            editor: {
              ...editor,
              position: {
                x: Math.round(position.x + rel.x),
                y: Math.round(position.y + rel.y),
              },
            },
          },
        };
      });
      const mergedIr = { ...state.ir, nodes: [...state.ir.nodes, ...placed] };
      const newFlowNodes = placed.map((n) => irNodeToFlow(n, 0));
      const newFlowEdges = irEdgesToFlow(subEdges, placed);
      recompute(
        [...state.nodes, ...newFlowNodes],
        [...state.edges, ...newFlowEdges],
        true,
        mergedIr,
      );
    },

    deleteSelected: () => {
      const { selectedNodeId, nodes, edges, ir } = get();
      if (!selectedNodeId || !ir) return;
      const isPort =
        ir.inputs.some((p) => p.id === selectedNodeId) ||
        ir.outputs.some((p) => p.id === selectedNodeId);
      if (isPort) return; // 入出力ポートは削除不可
      const nodes2 = nodes.filter((n) => n.id !== selectedNodeId);
      const edges2 = edges.filter(
        (e) => e.source !== selectedNodeId && e.target !== selectedNodeId,
      );
      set({ selectedNodeId: null });
      recompute(nodes2, edges2, true);
    },

    updateNodeData: (id, patch) => {
      const cur = get().ir;
      // Unitインスタンスのノードへのデータ編集は手編集としてマーク(構造不変でも検出する)
      if (cur) {
        const marked = withInstanceEditedForNode(cur, id);
        if (marked !== cur) set({ ir: marked });
      }
      const nodes = get().nodes.map((n) =>
        n.id === id ? ({ ...n, data: { ...n.data, ...patch } } as EditorNode) : n,
      );
      recompute(nodes, get().edges, true);
    },

    updatePortWidth: (id, width) => {
      const nodes = get().nodes.map((n) =>
        n.id === id
          ? ({
              ...n,
              data: {
                ...n.data,
                portWidth: width,
                subtitle: `入力 / width=${width}`,
              },
            } as EditorNode)
          : n,
      );
      recompute(nodes, get().edges, true);
    },

    // lut_layerノードのparamsをin-place系編集(tt/indices)で更新する。
    // 構造は不変なのでflowToIrは通さず、IRを直接差し替えて表示だけ更新する。
    // sync=false はバックエンドの単一ニューロン編集APIで同期済みの場合
    updateLutParams: (id, mutate, opts) => {
      const { ir } = get();
      if (!ir) return;
      const irNodes = ir.nodes.map((n) => {
        if (n.id !== id || n.type !== 'lut_layer') return n;
        const params = {
          ...n.params,
          indices: (n.params.indices as number[][]).map((row) => [...row]),
          truth_tables: [...(n.params.truth_tables as string[])],
        };
        mutate(params as unknown as LutLayerParams);
        return { ...n, params };
      });
      const ir2 = withInstanceEditedForNode({ ...ir, nodes: irNodes }, id);
      set({
        ir: ir2,
        nodes: get().nodes.map((fn) => refreshNodeData(fn, ir2)),
        ...(opts?.sync === false ? {} : { dirty: true }),
      });
      if (opts?.sync !== false) scheduleSync();
    },

    // blackbox等のparamsを構造不変で部分更新する。
    // torch_class/config編集はin/out幅を変えないためflowToIrを通さずIR直接差し替え。
    updateNodeParams: (id, patch) => {
      const { ir } = get();
      if (!ir) return;
      const irNodes = ir.nodes.map((n) =>
        n.id === id ? { ...n, params: { ...n.params, ...patch } } : n,
      );
      const ir2 = withInstanceEditedForNode({ ...ir, nodes: irNodes }, id);
      set({
        ir: ir2,
        nodes: get().nodes.map((fn) => refreshNodeData(fn, ir2)),
        dirty: true,
      });
      scheduleSync();
    },

    // 全ノードをelkjsで再配置し、meta.editor.positionへ書き戻す。
    // sizeOverridesは詳細展開グループの実寸ヒント(FlowCanvasが渡す)
    autoLayoutAll: async (sizeOverrides) => {
      const { nodes, edges } = get();
      if (nodes.length === 0) return;
      const positions = await layoutWithElk(nodes, edges, sizeOverrides);
      const moved = get().nodes.map((n) => {
        const pos = positions.get(n.id);
        return pos ? { ...n, position: pos } : n;
      });
      recompute(moved, get().edges, false);
    },

    syncNow: async () => {
      if (syncTimer !== undefined) {
        clearTimeout(syncTimer);
        syncTimer = undefined;
      }
      const { ir, diagnostics, dirty } = get();
      if (!ir || !dirty) return;
      if (hasShapeErrors(diagnostics)) {
        set({ syncState: 'error', syncError: '形状エラーがあるため未同期' });
        return;
      }
      set({ syncState: 'pending' });
      try {
        await putCurrentIR(ir);
        set({ dirty: false, syncState: 'synced', syncError: null });
      } catch (e) {
        set({
          syncState: 'error',
          syncError: e instanceof Error ? e.message : String(e),
        });
      }
    },

    setSelected: (id) => set({ selectedNodeId: id }),

    // --- サブサーキット化: meta.editor.groups の更新のみ ---

    createGroup: (name) => {
      const { ir, nodes } = get();
      if (!ir) return null;
      const portIds = new Set(
        [...ir.inputs, ...ir.outputs].map((p) => p.id),
      );
      const existing = readGroups(ir);
      const alreadyGrouped = new Set(existing.flatMap((g) => g.nodeIds));
      const selected = nodes.filter(
        (n) => n.selected && !portIds.has(n.id) && !alreadyGrouped.has(n.id),
      );
      if (selected.length < 2) return null;
      const gid = nextGroupId(existing);
      const seq = /^g(\d+)$/.exec(gid)?.[1] ?? '';
      const groups = groupsRecordOf(ir);
      const ir2 = withGroupsRecord(ir, {
        ...groups,
        [gid]: {
          name: name && name.trim() ? name.trim() : `サブサーキット${seq}`,
          nodeIds: selected.map((n) => n.id),
          collapsed: true,
          position: minCorner(selected),
        },
      });
      set({ ir: ir2, dirty: true, selectedNodeId: null });
      scheduleSync();
      return gid;
    },

    ungroup: (groupId) => {
      const { ir } = get();
      if (!ir) return;
      const groups = groupsRecordOf(ir);
      if (!groups[groupId]) return;
      const rest = { ...groups };
      delete rest[groupId];
      set({ ir: withGroupsRecord(ir, rest), dirty: true });
      scheduleSync();
    },

    toggleGroupCollapse: (groupId) => {
      const { ir } = get();
      if (!ir) return;
      const groups = groupsRecordOf(ir);
      const g = groups[groupId];
      if (!g) return;
      set({
        ir: withGroupsRecord(ir, {
          ...groups,
          [groupId]: { ...g, collapsed: !g.collapsed },
        }),
        dirty: true,
      });
      scheduleSync();
    },

    renameGroup: (groupId, name) => {
      const { ir } = get();
      if (!ir) return;
      const groups = groupsRecordOf(ir);
      const g = groups[groupId];
      if (!g || !name.trim()) return;
      set({
        ir: withGroupsRecord(ir, {
          ...groups,
          [groupId]: { ...g, name: name.trim() },
        }),
        dirty: true,
      });
      scheduleSync();
    },

    moveGroup: (groupId, position) => {
      const { ir } = get();
      if (!ir) return;
      const groups = groupsRecordOf(ir);
      const g = groups[groupId];
      if (!g) return;
      set({
        ir: withGroupsRecord(ir, {
          ...groups,
          [groupId]: {
            ...g,
            position: { x: Math.round(position.x), y: Math.round(position.y) },
          },
        }),
        dirty: true,
      });
      scheduleSync();
    },

    saveGroupAsTemplate: (groupId, name) => {
      const { ir } = get();
      if (!ir) return;
      const g = readGroups(ir).find((x) => x.id === groupId);
      if (!g) return;
      const memberSet = new Set(g.nodeIds);
      const memberNodes = ir.nodes.filter((n) => memberSet.has(n.id));
      const internalEdges = ir.edges.filter(
        (e) => memberSet.has(e.src) && memberSet.has(e.dst),
      );
      saveMyBlock(name.trim() || g.name, memberNodes, internalEdges);
    },

    // レイヤーコメント: metadata.editor.comments を更新して同期予約する。
    setComment: (nodeId, text) => {
      const { ir } = get();
      if (!ir) return;
      const comments = { ...commentsRecordOf(ir) };
      const trimmed = text.trim();
      if (trimmed) comments[nodeId] = trimmed;
      else delete comments[nodeId];
      set({ ir: withCommentsRecord(ir, comments), dirty: true });
      scheduleSync();
    },

    // NNC縦整列: トポロジカル順で1列に縦積みし、座標のみ更新する。
    // recompute(structural=false)なので meta.editor.position だけが変わり、
    // IRのノード/エッジ構造は不変(diagnosticsも変化しない)。
    alignVerticalNnc: () => {
      const { ir, nodes, edges } = get();
      if (!ir || nodes.length === 0) return;
      const order = topologicalOrder(ir);
      const rank = new Map(order.map((id, i) => [id, i]));
      const X = 240;
      const Y0 = 80;
      const DY = 120;
      const moved = nodes.map((n) => {
        const r = rank.get(n.id);
        return r === undefined
          ? n
          : { ...n, position: { x: X, y: Y0 + r * DY } };
      });
      recompute(moved, edges, false);
    },

    // Unit: Unitを引数で展開し、ドロップ位置へ配置して台帳へ記録する。
    // addSubgraph同様にbaseOverrideでparamsを保全しつつ、metadata.editor.unitInstancesへ
    // インスタンスを1件追加する(展開ノードには expandUnit が由来タグを付与済み)。
    addUnitInstance: (unit, argValues, position) => {
      const state = get();
      if (!state.ir) return;
      const instances = unitInstancesOf(state.ir);
      const instanceId = nextInstanceId(instances);
      const exp = expandUnit(unit, argValues, instanceId);
      const placed = offsetExpandedNodes(exp.nodes, position);
      const instance: UnitInstance = {
        instanceId,
        unitId: unit.id,
        argValues,
        expandedNodeIds: exp.expandedNodeIds,
        edited: false,
        position: { x: Math.round(position.x), y: Math.round(position.y) },
      };
      const mergedIr = withUnitInstances(
        { ...state.ir, nodes: [...state.ir.nodes, ...placed] },
        [...instances, instance],
      );
      const newFlowNodes = placed.map((n) => irNodeToFlow(n, 0));
      const newFlowEdges = irEdgesToFlow(exp.edges, placed);
      recompute(
        [...state.nodes, ...newFlowNodes],
        [...state.edges, ...newFlowEdges],
        true,
        mergedIr,
      );
    },

    // Unit定義変更の伝播: reexpandUnit(純関数)で非editedインスタンスを再展開し、
    // 得た新IRからflowを作り直す(構造変更なので needsBuild へ倒す)。警告は呼び出し側で表示。
    reexpandUnitInstances: (unit) => {
      const { ir } = get();
      if (!ir) {
        return { ir: ir!, reexpandedInstanceIds: [], skipped: [], droppedEdges: [] };
      }
      const res = reexpandUnit(ir, unit);
      if (res.reexpandedInstanceIds.length === 0) return res;
      const { nodes, edges } = irToFlow(res.ir);
      const { diagnostics } = inferShapes(res.ir.nodes, res.ir.edges, res.ir.inputs);
      set({
        ir: res.ir,
        nodes,
        edges,
        diagnostics,
        dirty: true,
        needsBuild: true,
      });
      scheduleSync();
      return res;
    },

    // Unit定義: metadata.editor.units を差し替えて同期予約する(構造不変)。
    setUnits: (units) => {
      const { ir } = get();
      if (!ir) return;
      const editor = (ir.metadata?.editor as Record<string, unknown>) ?? {};
      const ir2 = { ...ir, metadata: { ...ir.metadata, editor: { ...editor, units } } };
      set({ ir: ir2, dirty: true });
      scheduleSync();
    },
  };
    },
    {
      // 履歴の実体はIR(+needsBuild)のみ。nodes/edgesはIRから再導出する
      partialize: (s) => ({ ir: s.ir, needsBuild: s.needsBuild }),
      limit: UNDO_LIMIT,
      // IR参照が変わった時だけ履歴に積む(選択・ドラッグ途中・同期状態変化は対象外)
      equality: (past, cur) => past.ir === cur.ir,
    },
  ),
);

/** undo/redo後の派生状態再構築: 復元IRからflowを再生成し、診断・同期を更新する。 */
function restoreFromHistory() {
  const { ir } = useGraphStore.getState();
  if (!ir) return;
  const { nodes, edges } = irToFlow(ir);
  const { diagnostics } = inferShapes(ir.nodes, ir.edges, ir.inputs);
  useGraphStore.setState({
    nodes,
    edges,
    diagnostics,
    selectedNodeId: null,
    dirty: true,
    // built済みモデルへのin-place編集(tt/indices)を跨ぐ復元でも整合するよう常にbuildへ倒す
    needsBuild: true,
  });
  scheduleSyncGlobal();
}

/** 1手戻す。戻せなければfalse。 */
export function undoGraph(): boolean {
  const t = useGraphStore.temporal.getState();
  if (t.pastStates.length === 0) return false;
  t.undo();
  restoreFromHistory();
  return true;
}

/** 1手進める。進められなければfalse。 */
export function redoGraph(): boolean {
  const t = useGraphStore.temporal.getState();
  if (t.futureStates.length === 0) return false;
  t.redo();
  restoreFromHistory();
  return true;
}

/**
 * グループ化可能な選択ノードid。入出力ポート・既存グループのメンバーは除外。
 * ≥2でcreateGroupが成立する。
 */
export function groupableSelection(): string[] {
  const { ir, nodes } = useGraphStore.getState();
  if (!ir) return [];
  const portIds = new Set([...ir.inputs, ...ir.outputs].map((p) => p.id));
  const grouped = new Set(readGroups(ir).flatMap((g) => g.nodeIds));
  return nodes
    .filter((n) => n.selected && !portIds.has(n.id) && !grouped.has(n.id))
    .map((n) => n.id);
}

/** キャンバス上の全lut_layerノードid(WS購読対象)。 */
export function lutLayerIds(): string[] {
  return useGraphStore
    .getState()
    .nodes.filter((n) => n.type === 'lutLayer')
    .map((n) => n.id);
}

/** detail購読(gate_distribution)対象: 小規模lut_layerのみ(帯域制御)。*/
export function detailLayerIds(maxOutWidth: number): string[] {
  const { ir } = useGraphStore.getState();
  if (!ir) return [];
  return ir.nodes
    .filter(
      (n) =>
        n.type === 'lut_layer' &&
        Number((n.params as { out_width?: unknown }).out_width ?? Infinity) <=
          maxOutWidth,
    )
    .map((n) => n.id);
}

/** 選択中ノードのIRノード(なければnull)。 */
export function selectedIrNode(): IRNodeJson | null {
  const { ir, selectedNodeId } = useGraphStore.getState();
  if (!ir || !selectedNodeId) return null;
  return ir.nodes.find((n) => n.id === selectedNodeId) ?? null;
}

/** 空きslot番号(0始まりの最小未使用値) */
function nextFreeSlot(used: Set<number>): number {
  let s = 0;
  while (used.has(s)) s += 1;
  return s;
}

/**
 * draft id採番コールバック(`u{seq}_{suffix}`規約)。
 * 複合ブロック生成器(blocks/templates)へ渡し、既存ノードと衝突しない
 * 連番を払い出す(nextDraftSeqと同一の走査基準)。
 */
export function createDraftIdAllocator(): (suffix: string) => string {
  let seq = nextDraftSeq(useGraphStore.getState().nodes);
  return (suffix) => `u${seq++}_${suffix}`;
}

/**
 * 接続可否判定:
 *   - 入力ポートへの接続 / 出力ポートからの接続を拒否(ポート保護)
 *   - 単一入力ノード: 既に入力を持つ場合は拒否
 *   - 多入力ノード(concat/add): 占有済みslotのみ拒否(fanoutは合法)
 *   - 循環(target→…→source経路が既存)を拒否(DFS)
 */
export function isValidConnectionCheck(
  source?: string | null,
  target?: string | null,
  targetHandle?: string | null,
): boolean {
  if (!source || !target || source === target) return false;
  const { ir, edges, nodes } = useGraphStore.getState();
  if (!ir) return false;
  if (ir.inputs.some((p) => p.id === target)) return false;
  if (ir.outputs.some((p) => p.id === source)) return false;

  const targetNode = nodes.find((n) => n.id === target);
  if (targetNode && MULTI_INPUT_FLOW_TYPES.has(targetNode.type ?? '')) {
    // 多入力ノード: 指定slotが占有済みなら拒否(未指定は空きslotへ自動割当)
    if (targetHandle) {
      const slot = slotFromHandle(targetHandle);
      const occupied = edges.some(
        (e) => e.target === target && slotFromHandle(e.targetHandle) === slot,
      );
      if (occupied) return false;
    }
  } else if (edges.some((e) => e.target === target)) {
    return false;
  }

  const succ = new Map<string, string[]>();
  for (const e of edges) {
    const list = succ.get(e.source) ?? [];
    list.push(e.target);
    succ.set(e.source, list);
  }
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length > 0) {
    const cur = stack.pop()!;
    if (cur === source) return false; // 循環
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const nxt of succ.get(cur) ?? []) stack.push(nxt);
  }
  return true;
}
