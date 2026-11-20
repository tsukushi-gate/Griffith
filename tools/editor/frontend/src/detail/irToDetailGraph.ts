// 表示専用の合成レイヤ。
// graphStoreの抽象ノード/エッジ(=flowToIrの入力)には一切手を入れず、
// 展開対象のlut_layerを「グループ+個別ゲート」へ、その上流を「ビット展開入力」へ
// 純関数で変換した描画用ノード/エッジを合成する。
// 子ノードidの区切りは '::' (IRノードidには現れないことを確認済み)。

import type { Edge, Node } from '@xyflow/react';
import type { EditorNode } from '../ir/irToGraph';
import type { IRGraphJson, IRNodeJson, LutLayerParams } from '../ir/schema';
import {
  BITS_HEADER,
  BITS_ROW,
  BITS_W,
  DETAIL_AUTO_MAX,
  DETAIL_MANUAL_MAX,
  DETAIL_TOTAL_MAX,
  GATE_CARD_H,
  GATE_CARD_W,
  GATE_GAP,
  GATE_H,
  GATE_W,
  GROUP_HEADER,
  GROUP_PAD,
  INPUT_BITS_MAX,
} from './detailConfig';

export type GateNodeData = {
  layerId: string;
  neuron: number;
  nInputs: number;
  tt: string; // IR由来の真理値表hex
  slotSources: number[]; // indices[j][neuron]
  showSlotLabels: boolean; // 上流が集約エッジのままの場合にs0=713形式で表示
  parameterization: string;
  [key: string]: unknown;
};

export type LutLayerGroupData = {
  label: string;
  inWidth: number;
  outWidth: number;
  nInputs: number;
  parameterization: string;
  initialized: boolean;
  [key: string]: unknown;
};

export type InputBitsData = {
  label: string;
  width: number;
  [key: string]: unknown;
};

export type GateFlowNode = Node<GateNodeData, 'gate'>;
export type LutLayerGroupFlowNode = Node<LutLayerGroupData, 'lutLayerGroup'>;
export type InputBitsFlowNode = Node<InputBitsData, 'inputBits'>;
export type RenderNode =
  | EditorNode
  | GateFlowNode
  | LutLayerGroupFlowNode
  | InputBitsFlowNode;

export interface DetailGraph {
  nodes: RenderNode[];
  edges: Edge[];
  /** 展開中グループの実寸(自動整列のelkサイズヒント用) */
  groupSizes: Map<string, { width: number; height: number }>;
  /** ノード総数上限で展開を拒否した層id(トースト通知用) */
  overflowIds: string[];
}

/** ゲートカードの格子配置: 列数は16行を超えたら増やす(最大4列)、列優先で並べる。 */
export function gateGridLayout(count: number): {
  cols: number;
  rows: number;
  width: number;
  height: number;
} {
  const cols = Math.min(Math.max(Math.ceil(count / 16), 1), 4);
  const rows = Math.ceil(count / cols);
  return {
    cols,
    rows,
    width: GROUP_PAD * 2 + cols * GATE_W + (cols - 1) * GATE_GAP,
    height: GROUP_HEADER + GROUP_PAD * 2 + rows * GATE_H + (rows - 1) * GATE_GAP,
  };
}

/** InputBitsノードの実寸 */
export function inputBitsSize(width: number): { width: number; height: number } {
  return { width: BITS_W, height: BITS_HEADER + width * BITS_ROW + 8 };
}

/** 層が展開可能か(手動展開ボタンの表示判定にも使う) */
export function isLayerExpandable(p: LutLayerParams): boolean {
  return p.out_width <= DETAIL_MANUAL_MAX;
}

/** 実効展開状態: 手動override > 自動判定(in/out両方がしきい値以下) */
export function isLayerExpanded(
  p: LutLayerParams,
  override: boolean | undefined,
): boolean {
  if (override !== undefined) return override && isLayerExpandable(p);
  return p.out_width <= DETAIL_AUTO_MAX && p.in_width <= DETAIL_AUTO_MAX;
}

const SLOT_COUNT = 6;

/** layerIdごとの子ゲートノード/詳細エッジのキャッシュエントリ。
    依存(params参照・上流・詳細化状態・パラメータ化方式)が全て一致する間は
    前回のオブジェクトをそのまま再利用し、ドラッグ/選択等の非構造変化での
    全再生成(1000ゲート規模でのGC・差分検出コスト)を回避する。 */
interface LayerCacheEntry {
  params: LutLayerParams;
  upstream: string | undefined;
  upstreamDetailed: boolean;
  parameterization: string;
  gateNodes: GateFlowNode[];
  detailEdges: Edge[];
}

export type DetailComposeCache = Map<string, LayerCacheEntry>;

export function createDetailComposeCache(): DetailComposeCache {
  return new Map();
}

export function composeDetailGraph(
  storeNodes: EditorNode[],
  storeEdges: Edge[],
  ir: IRGraphJson | null,
  expandedOverrides: Record<string, boolean>,
  cache?: DetailComposeCache,
  // §3: LODが決めた自動展開層(見かけサイズ×可視領域基準)。未指定なら
  // 従来の幅ベース自動判定(isLayerExpanded)へフォールバックする(後方互換)。
  autoExpand?: Set<string>,
): DetailGraph {
  const groupSizes = new Map<string, { width: number; height: number }>();
  const overflowIds: string[] = [];

  if (!ir) {
    cache?.clear();
    return { nodes: storeNodes, edges: storeEdges, groupSizes, overflowIds };
  }

  // 展開対象の決定(ノード総数上限を超えたら以降の層は展開しない)
  const lutParams = new Map<string, LutLayerParams>();
  const irNodeById = new Map<string, IRNodeJson>();
  for (const n of ir.nodes) {
    irNodeById.set(n.id, n);
    if (n.type === 'lut_layer') {
      lutParams.set(n.id, n.params as unknown as LutLayerParams);
    }
  }
  const expanded = new Set<string>();
  let totalNodes = storeNodes.length;
  for (const fn of storeNodes) {
    const p = lutParams.get(fn.id);
    if (!p) continue;
    const override = expandedOverrides[fn.id];
    // 実効展開: 手動override > (autoExpand指定時)LOD自動 > 幅ベース自動(後方互換)
    let want: boolean;
    if (override !== undefined) {
      want = override && isLayerExpandable(p);
    } else if (autoExpand !== undefined) {
      want = autoExpand.has(fn.id) && isLayerExpandable(p);
    } else {
      want = isLayerExpanded(p, undefined);
    }
    if (!want) continue;
    if (totalNodes + p.out_width > DETAIL_TOTAL_MAX) {
      overflowIds.push(fn.id);
      continue;
    }
    expanded.add(fn.id);
    totalNodes += p.out_width;
  }

  if (expanded.size === 0) {
    cache?.clear();
    return { nodes: storeNodes, edges: storeEdges, groupSizes, overflowIds };
  }

  // 各展開層の直接上流(単一入力制約なので高々1本)
  const upstreamOf = new Map<string, string>();
  for (const e of storeEdges) {
    if (expanded.has(e.target)) upstreamOf.set(e.target, e.source);
  }

  // ビット展開する入力ポート: 展開層の直接上流かつ幅がしきい値以下
  const inputPortWidth = new Map<string, number>();
  for (const port of ir.inputs) inputPortWidth.set(port.id, port.width);
  const bitPorts = new Set<string>();
  for (const [, src] of upstreamOf) {
    const w = inputPortWidth.get(src);
    if (w !== undefined && w <= INPUT_BITS_MAX) bitPorts.add(src);
  }

  // 「ゲート/ビット単位の詳細エッジで置き換え済み」の集約エッジを除去する判定
  const detailEdgeReplaced = (e: Edge): boolean => {
    if (!expanded.has(e.target)) return false;
    return expanded.has(e.source) || bitPorts.has(e.source);
  };

  const nodes: RenderNode[] = [];
  const edges: Edge[] = storeEdges.filter((e) => !detailEdgeReplaced(e));

  // 【重要】差し替えノードは毎回新規オブジェクトになるため、React Flow v12の
  // 可視判定(nodeHasDimensions: measured/width/initialWidthのみ参照、styleは無視)を
  // 満たすようノード直下のwidth/heightを明示し、さらにstoreノードのmeasuredを
  // 引き継いで内部測定状態の破棄→再測定の無限ループ(=ノードが永久にhidden)を防ぐ。
  for (const fn of storeNodes) {
    // 入力ポート → ビット展開ノードへ差し替え(idと座標は維持)
    if (bitPorts.has(fn.id)) {
      const width = inputPortWidth.get(fn.id) ?? 0;
      const size = inputBitsSize(width);
      nodes.push({
        id: fn.id,
        type: 'inputBits',
        position: fn.position,
        selected: fn.selected,
        deletable: false,
        width: size.width,
        height: size.height,
        measured: fn.measured,
        data: { label: fn.id, width },
      });
      continue;
    }

    if (!expanded.has(fn.id)) {
      nodes.push(fn);
      continue;
    }

    // 展開層 → グループノード+子ゲートノード群
    const p = lutParams.get(fn.id)!;
    const irNode = irNodeById.get(fn.id)!;
    const count = p.out_width;
    const grid = gateGridLayout(count);
    groupSizes.set(fn.id, { width: grid.width, height: grid.height });

    const upstream = upstreamOf.get(fn.id);
    const upstreamDetailed =
      upstream !== undefined && (expanded.has(upstream) || bitPorts.has(upstream));
    const parameterization =
      (irNode.train_info.parameterization as string) ?? 'softmax';

    nodes.push({
      id: fn.id,
      type: 'lutLayerGroup',
      position: fn.position,
      selected: fn.selected,
      width: grid.width,
      height: grid.height,
      measured: fn.measured,
      data: {
        label: fn.id,
        inWidth: p.in_width,
        outWidth: p.out_width,
        nInputs: p.n_inputs,
        parameterization,
        initialized: irNode.train_info.initialized !== false,
      },
    });

    // 子ゲート/詳細エッジは依存不変なら前回オブジェクトを再利用。
    // 子はparentId相対座標なので、親グループの移動ではキャッシュが無効化されない
    let entry = cache?.get(fn.id);
    if (
      entry === undefined ||
      entry.params !== p ||
      entry.upstream !== upstream ||
      entry.upstreamDetailed !== upstreamDetailed ||
      entry.parameterization !== parameterization
    ) {
      const gateNodes: GateFlowNode[] = [];
      const detailEdges: Edge[] = [];
      for (let o = 0; o < count; o++) {
        // 列優先配置(連番ニューロンが縦に並ぶ)
        const col = Math.floor(o / grid.rows);
        const row = o % grid.rows;
        const slotSources = Array.from(
          { length: p.n_inputs },
          (_, j) => p.indices[j]?.[o] ?? 0,
        );
        gateNodes.push({
          id: `${fn.id}::g${o}`,
          type: 'gate',
          parentId: fn.id,
          extent: 'parent',
          draggable: false,
          deletable: false,
          connectable: false,
          // 子ノードはdimensions changeをフィルタで捨てるため、実寸を明示して即時可視化する
          width: GATE_CARD_W,
          height: GATE_CARD_H,
          position: {
            x: GROUP_PAD + col * (GATE_W + GATE_GAP),
            y: GROUP_HEADER + GROUP_PAD + row * (GATE_H + GATE_GAP),
          },
          data: {
            layerId: fn.id,
            neuron: o,
            nInputs: p.n_inputs,
            tt: p.truth_tables[o] ?? '0x0',
            slotSources,
            showSlotLabels: !upstreamDetailed,
            parameterization,
          },
        });

        // 上流が詳細化済みならスロット単位の実エッジを張る
        if (upstreamDetailed && upstream !== undefined) {
          for (let j = 0; j < p.n_inputs; j++) {
            const src = slotSources[j];
            const fromGate = expanded.has(upstream);
            detailEdges.push({
              id: `d:${fn.id}:${o}:${j}`,
              source: fromGate ? `${upstream}::g${src}` : upstream,
              sourceHandle: fromGate ? 'out' : `b${src}`,
              target: `${fn.id}::g${o}`,
              targetHandle: `s${j}`,
              selectable: false,
              deletable: false,
              focusable: false,
              style: {
                stroke: `var(--slot-${j % SLOT_COUNT})`,
                strokeWidth: 1,
                opacity: 0.55,
              },
            });
          }
        }
      }
      entry = {
        params: p,
        upstream,
        upstreamDetailed,
        parameterization,
        gateNodes,
        detailEdges,
      };
      cache?.set(fn.id, entry);
    }
    for (const g of entry.gateNodes) nodes.push(g);
    for (const de of entry.detailEdges) edges.push(de);
  }

  // 展開が解除された層のキャッシュは破棄する(モデル差し替え時の残留防止)
  if (cache) {
    for (const key of cache.keys()) {
      if (!expanded.has(key)) cache.delete(key);
    }
  }

  // React Flow v12はparentIdを持つ子より前に親が並んでいる必要がある。
  // 親(lutLayerGroup)はstoreNodes順で子より先にpushされるため既に満たしている。
  return { nodes, edges, groupSizes, overflowIds };
}
