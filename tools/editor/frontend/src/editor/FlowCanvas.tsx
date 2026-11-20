// React Flowキャンバス(E2: 表示・選択+DnD配置・接続編集・削除 /: 詳細ゲートビュー)。
// 描画は useDetailGraph の合成結果(抽象ノード+展開グループ+個別ゲート)を使い、
// 変更イベントは子ノード(idに'::'を含む)をフィルタしてから graphStore へ委譲する。

import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useNodesInitialized,
  useReactFlow,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import { GradParticleOverlay } from '../anim/GradParticleOverlay';
import { BLOCK_DND_MIME, useBlockDialogStore } from '../blocks/blockDialogStore';
import type { BlockKind } from '../blocks/templates';
import type { EditorNode } from '../ir/irToGraph';
import { GateNode } from '../detail/nodes/GateNode';
import { InputBitsNode } from '../detail/nodes/InputBitsNode';
import { LutLayerGroupNode } from '../detail/nodes/LutLayerGroupNode';
import { useDetailGraph } from '../detail/useDetailGraph';
import type { RenderNode } from '../detail/irToDetailGraph';
import { DETAIL_MANUAL_MAX } from '../detail/detailConfig';
import { LodIndicator } from '../detail/LodIndicator';
import {
  lodStageForZoom,
  sameIdSet,
  selectAutoExpandLayers,
  viewportToFlowRect,
  type LodLayerInfo,
} from '../detail/lod';
import type { IRGraphJson } from '../ir/schema';
import type { TraceCone } from '../playground/counterexampleTrace';
import { usePlaygroundStore } from '../playground/playgroundStore';
import {
  createDraftIdAllocator,
  isValidConnectionCheck,
  useGraphStore,
  type PaletteNodeType,
} from '../store/graphStore';
import { useUIStore } from '../store/uiStore';
import { useViewStore } from '../store/viewStore';
import { useTrainingStore } from '../store/trainingStore';
import { cssVar } from '../theme/theme';
import { buildEdgeBaseStyles } from './edgeStyle';
import { buildReplayBanner } from './replayBanner';
import { EdgeLegend } from './EdgeLegend';
import { flowApi } from './flowApi';
import {
  GROUP_EDGE_PREFIX,
  groupIdFromNodeId,
  isGroupNodeId,
  readGroups,
} from './grouping';
import { MYBLOCK_DND_MIME, getMyBlock, instantiateMyBlock } from '../blocks/myBlocks';
import { UNIT_DND_MIME, useUnitDialogStore } from '../blocks/unitDialogStore';
import { NetworkTabBar } from './NetworkTabBar';
import { PALETTE_DND_MIME } from './Palette';
import { QuickAddMenu } from './QuickAddMenu';
import { DisplayModeToggle } from './nnccard/DisplayModeToggle';
import { NncCardCanvas } from './nnccard/NncCardCanvas';
import { AddNode } from './nodes/AddNode';
import { ConcatNode } from './nodes/ConcatNode';
import { ConvNode } from './nodes/ConvNode';
import { GenericNode } from './nodes/GenericNode';
import { GroupHeaderNode } from './nodes/GroupHeaderNode';
import { GroupNode } from './nodes/GroupNode';
import { GroupSumNode } from './nodes/GroupSumNode';
import { LutLayerNode } from './nodes/LutLayerNode';

const NODE_TYPES = {
  lutLayer: LutLayerNode,
  groupSum: GroupSumNode,
  generic: GenericNode,
  concat: ConcatNode,
  add: AddNode,
  conv: ConvNode,
  gate: GateNode,
  lutLayerGroup: LutLayerGroupNode,
  inputBits: InputBitsNode,
  group: GroupNode,
  groupHeader: GroupHeaderNode,
};

// エッジはSVGのinline styleなのでCSS変数参照がそのまま使える
const ERROR_EDGE_STYLE = { stroke: 'var(--edge-error)', strokeWidth: 2 };

/** 合成子ノード(gate等)のid規約 */
const isDetailChildId = (id: string) => id.includes('::');

/** 空の自動展開集合(参照安定用。変更しない) */
const EMPTY_EXPAND: Set<string> = new Set();

/** LOD可視判定用のlut_layerノード実寸フォールバック(measured未確定時) */
const LUT_NODE_W = 190;
const LUT_NODE_H = 130;

/** graphStoreのノード+IRから、LOD判定に必要な各lut_layerの矩形/展開可否/折りたたみを組む(§3)。 */
function buildLodLayerInfos(nodes: EditorNode[], ir: IRGraphJson | null): LodLayerInfo[] {
  if (!ir) return [];
  const outWidth = new Map<string, number>();
  for (const n of ir.nodes) {
    if (n.type === 'lut_layer') {
      outWidth.set(
        n.id,
        Number((n.params as { out_width?: unknown }).out_width ?? Infinity),
      );
    }
  }
  // 折りたたみグループのメンバーは自動展開の対象外
  const collapsed = new Set<string>();
  for (const g of readGroups(ir)) {
    if (g.collapsed) for (const id of g.nodeIds) collapsed.add(id);
  }
  const infos: LodLayerInfo[] = [];
  for (const n of nodes) {
    if (n.type !== 'lutLayer') continue;
    const ow = outWidth.get(n.id) ?? Infinity;
    infos.push({
      id: n.id,
      x: n.position.x,
      y: n.position.y,
      w: n.measured?.width ?? LUT_NODE_W,
      h: n.measured?.height ?? LUT_NODE_H,
      expandable: ow <= DETAIL_MANUAL_MAX,
      collapsed: collapsed.has(n.id),
    });
  }
  return infos;
}

/** 詳細エッジのソース(ゲート出力 or 入力ポートビット)の実行値を引く */
function detailEdgeSourceValue(
  edge: Edge,
  values: Record<string, Float32Array>,
): number | undefined {
  const parts = edge.source.split('::');
  if (parts.length === 2 && parts[1].startsWith('g')) {
    // ソースがゲート: `${layerId}::g${neuron}`
    return values[parts[0]]?.[Number(parts[1].slice(1))];
  }
  if (edge.sourceHandle?.startsWith('b')) {
    // ソースが入力ポートのビットハンドル: `b${index}`
    return values[edge.source]?.[Number(edge.sourceHandle.slice(1))];
  }
  return undefined;
}

/** エッジが反例トレースの到達コーン上にあるか(§4-2)。
    詳細エッジ('d:'始まり)は両端の合成ノードのbase idで、
    トップレベルエッジは`${src}->${dst}`キーで判定する。 */
function edgeOnCone(edge: Edge, cone: TraceCone): boolean {
  if (edge.id.startsWith('d:')) {
    const srcBase = edge.source.split('::')[0];
    const dstBase = edge.target.split('::')[0];
    return cone.nodes.has(srcBase) && cone.nodes.has(dstBase);
  }
  return cone.edges.has(`${edge.source}->${edge.target}`);
}

/** 実行値によるエッジ着色: 0=--border / 1=--chart-acc、softの中間値はopacity補間 */
function playgroundEdgeStyle(value: number): React.CSSProperties {
  const t = Math.min(Math.max(value, 0), 1);
  return {
    stroke: t >= 0.5 ? 'var(--chart-acc)' : 'var(--border)',
    strokeWidth: 1 + t,
    opacity: 0.35 + 0.65 * Math.abs(t * 2 - 1),
  };
}

function FlowCanvasInner() {
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const onNodesChange = useGraphStore((s) => s.onNodesChange);
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange);
  const onConnect = useGraphStore((s) => s.onConnect);
  const positionsChanged = useGraphStore((s) => s.positionsChanged);
  const addNodeFromPalette = useGraphStore((s) => s.addNodeFromPalette);
  const addSubgraph = useGraphStore((s) => s.addSubgraph);
  const moveGroup = useGraphStore((s) => s.moveGroup);
  const setSelected = useGraphStore((s) => s.setSelected);
  const autoLayoutAll = useGraphStore((s) => s.autoLayoutAll);
  const setSelectedNeuron = useUIStore((s) => s.setSelectedNeuron);
  const [layouting, setLayouting] = useState(false);
  const [quickAddAt, setQuickAddAt] = useState<{ x: number; y: number } | null>(null);
  const { screenToFlowPosition, fitView, getViewport } = useReactFlow();

  // §3 セマンティックズーム: 見かけサイズ×可視領域で自動展開する層を決める。
  const rawNodes = useGraphStore((s) => s.nodes);
  const ir = useGraphStore((s) => s.ir);
  const autoLod = useViewStore((s) => s.autoLod);
  const setLodStage = useViewStore((s) => s.setLodStage);
  const setView = useViewStore((s) => s.setView);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [autoExpandIds, setAutoExpandIds] = useState<Set<string>>(EMPTY_EXPAND);
  const layerInfos = useMemo(() => buildLodLayerInfos(rawNodes, ir), [rawNodes, ir]);

  // viewport({x,y,zoom})からLOD段階と自動展開集合を再計算する。段階/集合が
  // 変わった時のみstoreへ反映し、onMove毎の再合成(契約)を抑える。
  // 自動LOD off時は段階・展開集合とも凍結する(手動展開ボタンは別途有効)。
  const recomputeLod = useCallback(
    (vp: { x: number; y: number; zoom: number }) => {
      if (!autoLod) return;
      setLodStage(lodStageForZoom(vp.zoom));
      const el = wrapRef.current;
      if (!el) return;
      const { width, height } = el.getBoundingClientRect();
      const view = viewportToFlowRect(vp, width, height);
      const ids = selectAutoExpandLayers(layerInfos, vp.zoom, view);
      setAutoExpandIds((prev) => (sameIdSet(prev, ids) ? prev : ids));
    },
    [autoLod, layerInfos, setLodStage],
  );

  // fitView後の初期ズーム/依存変化(層追加・自動LOD切替)を反映する
  useEffect(() => {
    recomputeLod(getViewport());
  }, [recomputeLod, getViewport]);

  // モデル読込(setIRでfitNonceが増える)時にグラフ全体をfit-viewし、直後にLODを
  // 再計算する。非同期整列後の座標で枠を合わせないと、小規模モデルが画面外へ外れて
  // onlyRenderVisibleElementsで間引かれ「1ノードしか出ない」状態になるため(編集では発火しない)。
  //
  // 【無限ループ対策】このfitは「読込ごとに1回だけ」発火させる(fitNonceをキーにref記録)。
  // fit→recomputeLodが層を自動展開すると、新ゲートノードでnodesInitializedがfalse↔trueへ、
  // かつrecomputeLodの識別子がchurnし、依存配列経由でこの効果が再発火→再fit→展開/折りたたみが
  // 反転する双安定振動(fitズームがLOD閾値0.393を跨ぐ)に陥り、xyflowのStoreUpdaterが
  // "Maximum update depth exceeded" で落ちる。1回に制限してフィードバック辺を断つ。
  const fitNonce = useGraphStore((s) => s.fitNonce);
  const nodesInitialized = useNodesInitialized();
  const fittedNonceRef = useRef(0);
  useEffect(() => {
    if (fitNonce === 0 || !nodesInitialized) return;
    if (fittedNonceRef.current === fitNonce) return; // 読込ごとに1回だけfitする
    fittedNonceRef.current = fitNonce;
    void fitView({ padding: 0.15 }).then(() => recomputeLod(getViewport()));
  }, [fitNonce, nodesInitialized, fitView, recomputeLod, getViewport]);

  const { nodes, edges, groupSizes, overflowIds } = useDetailGraph(autoExpandIds);

  // 展開拒否(ノード総数上限)の通知は同一集合につき1回だけ
  const warnedRef = useRef('');
  useEffect(() => {
    const key = overflowIds.join(',');
    if (key && key !== warnedRef.current) {
      warnedRef.current = key;
      toast.warning(
        `ノード総数上限のため展開をスキップ: ${overflowIds.join(', ')}(deck.glドリルダウンを使用してください)`,
      );
    }
  }, [overflowIds]);

  const handleAutoLayout = useCallback(async () => {
    setLayouting(true);
    try {
      await autoLayoutAll(groupSizes);
      await fitView({ padding: 0.15 });
    } finally {
      setLayouting(false);
    }
  }, [autoLayoutAll, fitView, groupSizes]);

  // コマンドパレット/ショートカット用の操作APIを登録
  useEffect(() => {
    flowApi.current = {
      addAtCenter: (type) => {
        const el = document.querySelector('.react-flow');
        const rect = el?.getBoundingClientRect();
        const screen = rect
          ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
          : { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        addNodeFromPalette(type, screenToFlowPosition(screen));
      },
      fitView: () => void fitView({ padding: 0.15 }),
      autoLayout: handleAutoLayout,
      focusNode: (nodeId) =>
        void fitView({
          nodes: [{ id: nodeId }],
          padding: 0.5,
          duration: 400,
          maxZoom: 1.2,
        }),
    };
    return () => {
      flowApi.current = null;
    };
  }, [addNodeFromPalette, screenToFlowPosition, fitView, handleAutoLayout]);

  // プレイグラウンド実行値。off時はnullで従来描画と完全同一
  const pgValues = usePlaygroundStore((s) => (s.enabled ? s.values : null));
  // 反例トレースの到達コーン(§4-2)。off/非トレース時はnullで従来描画と同一
  const traceCone = usePlaygroundStore((s) => (s.enabled ? s.traceCone : null));
  // 誤差逆伝播リプレイ状態。frame非null=再生中
  const bpAnim = useViewStore((s) => s.bpAnim);

  // §5 エッジ型の色分け: IRから各トップレベルエッジの基底スタイル(太さ=ビット幅・
  // 色=信号種)を導出する。動的着色が無いエッジにのみ適用する(下記styledEdges)。
  const edgeBaseStyles = useMemo(() => buildEdgeBaseStyles(ir), [ir]);

  // §4-4 リプレイ視認性: 履歴スクラブ中はセピアオーバーレイ+バナーを出す。
  // trainingStoreは読み取りのみ(状態遷移はタイムラインUI/exitReplayが行う)。
  const displayMode = useTrainingStore((s) => s.displayMode);
  const cursorStep = useTrainingStore((s) => s.cursorStep);
  const trainStatus = useTrainingStore((s) => s.status);
  const exitReplay = useTrainingStore((s) => s.exitReplay);
  const replay = buildReplayBanner(displayMode, cursorStep, trainStatus);

  // 反例トレース中: コーン上のノードを強調、コーン外を淡色に沈める(§4-2)。
  // ノードidは合成でlayerId::gN等になるためbase(::前)でコーン所属を判定する。
  const styledNodes = useMemo(() => {
    if (!traceCone) return nodes;
    return nodes.map((n) => {
      const base = n.id.split('::')[0];
      const inCone = traceCone.nodes.has(base);
      return {
        ...n,
        style: {
          ...n.style,
          opacity: inCone ? 1 : 0.2,
          outline: inCone ? '2px solid var(--edge-error)' : undefined,
          outlineOffset: inCone ? '2px' : undefined,
          borderRadius: inCone ? '6px' : undefined,
        },
      };
    });
  }, [nodes, traceCone]);

  // エッジ装飾の優先順位(高→低):
  //   1. 診断エラー(赤。edgeIdは`${src}->${dst}`規約)
  //   2. 反例トレース(§4-2。コーン外を沈める)
  //   3. 逆伝播リプレイ(詳細エッジ太さ=edge_grad強度)
  //   4. プレイグラウンド実行値(詳細エッジをソース値で着色)
  //   5. 基底スタイル(§5。トップレベルエッジのビット幅→太さ・信号種→色)
  // 5は動的着色が無いときの床であり、上位を上書きしない。
  const styledEdges = useMemo(() => {
    const errorKeys = new Set(
      diagnostics
        .filter((d) => d.level === 'error' && d.edgeId)
        .map((d) => d.edgeId as string),
    );
    const bpPlaying = bpAnim.frame !== null;
    // 基底スタイルを床として敷く(既存の明示styleはその上に残す=残差破線等を保護)
    const withBase = (e: Edge): Edge => {
      const base = edgeBaseStyles.get(e.id);
      return base ? { ...e, style: { ...base, ...e.style } } : e;
    };
    return edges.map((e) => {
      if (errorKeys.has(`${e.source}->${e.target}`)) {
        return { ...e, style: ERROR_EDGE_STYLE };
      }
      // 反例トレース: コーン外エッジを沈め、コーン上は実行着色を残す(§4-2)
      if (traceCone && !edgeOnCone(e, traceCone)) {
        const based = withBase(e);
        return { ...based, style: { ...based.style, opacity: 0.08 } };
      }
      if (bpPlaying && e.id.startsWith('d:')) {
        // アクティブ層のエッジ: 太さ・不透明度=勾配強度、他層は淡色に沈める
        if (bpAnim.activeLayer && e.id.startsWith(`d:${bpAnim.activeLayer}:`)) {
          const t = bpAnim.edgeIntensity?.get(e.id) ?? 0;
          return {
            ...e,
            style: {
              ...e.style,
              stroke: 'var(--chart-loss)',
              strokeWidth: 1 + 3 * t,
              opacity: 0.3 + 0.7 * t,
            },
          };
        }
        return { ...e, style: { ...e.style, opacity: 0.12 } };
      }
      if (pgValues && e.id.startsWith('d:')) {
        const value = detailEdgeSourceValue(e, pgValues);
        if (value !== undefined) {
          return { ...e, style: { ...e.style, ...playgroundEdgeStyle(value) } };
        }
      }
      return withBase(e);
    });
  }, [edges, diagnostics, pgValues, bpAnim, traceCone, edgeBaseStyles]);

  // 合成子ノード(gate等)/グループ合成ノード由来の変更はstoreへ流さない。
  // グループノードのドラッグはonNodeDragStopでmeta.editor.groupsへ書き戻す。
  const handleNodesChange = useCallback(
    (changes: NodeChange<RenderNode>[]) => {
      const filtered = changes.filter((c) =>
        'id' in c ? !isDetailChildId(c.id) && !isGroupNodeId(c.id) : true,
      );
      if (filtered.length > 0) {
        onNodesChange(filtered as NodeChange<EditorNode>[]);
      }
    },
    [onNodesChange],
  );

  // 合成詳細エッジ(id 'd:')/グループ再配線エッジの変更もstoreへ流さない
  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const filtered = changes.filter((c) =>
        'id' in c
          ? !c.id.startsWith('d:') && !c.id.startsWith(GROUP_EDGE_PREFIX)
          : true,
      );
      if (filtered.length > 0) onEdgesChange(filtered);
    },
    [onEdgesChange],
  );

  const isValidConnection: IsValidConnection<Edge> = useCallback(
    (conn) => isValidConnectionCheck(conn.source, conn.target, conn.targetHandle),
    [],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      // 複合ブロック: パラメータダイアログを開く(展開はダイアログ側)
      const blockKind = event.dataTransfer.getData(BLOCK_DND_MIME);
      if (blockKind) {
        useBlockDialogStore.getState().open(blockKind as BlockKind, position);
        return;
      }
      // Unit: 引数ダイアログを開く(展開はダイアログ側 addUnitInstance)
      const unitId = event.dataTransfer.getData(UNIT_DND_MIME);
      if (unitId) {
        useUnitDialogStore.getState().open(unitId, position);
        return;
      }
      // マイブロック: 保存テンプレートをid振り直しで展開配置
      const myBlockId = event.dataTransfer.getData(MYBLOCK_DND_MIME);
      if (myBlockId) {
        const block = getMyBlock(myBlockId);
        if (block) {
          const { nodes: bNodes, edges: bEdges } = instantiateMyBlock(
            block,
            createDraftIdAllocator(),
          );
          addSubgraph(bNodes, bEdges, position);
        }
        return;
      }
      const type = event.dataTransfer.getData(PALETTE_DND_MIME);
      if (!type) return;
      addNodeFromPalette(type as PaletteNodeType, position);
    },
    [screenToFlowPosition, addNodeFromPalette, addSubgraph],
  );

  return (
    <div ref={wrapRef} className="relative h-full w-full">
    <NetworkTabBar />
    <ReactFlow<RenderNode>
      nodes={styledNodes}
      edges={styledEdges}
      nodeTypes={NODE_TYPES}
      onInit={() => recomputeLod(getViewport())}
      onMove={(_, vp) => recomputeLod(vp)}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={onConnect}
      onNodeClick={(_, node) => {
        // グループ合成ノードはIRノードではないので選択対象外(操作は自身のボタン)
        if (isGroupNodeId(node.id)) return;
        // ゲートカードのクリックは親層を選択+インスペクタのニューロン連動
        if (isDetailChildId(node.id)) {
          const [layerId, rest] = node.id.split('::');
          if (rest?.startsWith('g')) {
            setSelected(layerId);
            setSelectedNeuron({ nodeId: layerId, index: Number(rest.slice(1)) });
          }
          return;
        }
        setSelected(node.id);
        setSelectedNeuron(null);
      }}
      onNodeDoubleClick={(_, node) => {
        // ダブルクリック→ゲート詳細(L0)へドリルダウン。
        // 合成子ノード/グループ/ポートは対象外。IRノードのみ遷移する。
        if (isDetailChildId(node.id) || isGroupNodeId(node.id)) return;
        const irNow = useGraphStore.getState().ir;
        if (!irNow?.nodes.some((n) => n.id === node.id)) return;
        setSelected(node.id);
        setSelectedNeuron(null);
        setView({ kind: 'l0', nodeId: node.id });
      }}
      onPaneClick={() => {
        setSelected(null);
        setSelectedNeuron(null);
        setQuickAddAt(null);
      }}
      onPaneContextMenu={(e) => {
        // 右クリックでクイック追加メニュー(モデル未読込時は無効)
        e.preventDefault();
        if (!useGraphStore.getState().ir) return;
        const ev = e as React.MouseEvent;
        setQuickAddAt({ x: ev.clientX, y: ev.clientY });
      }}
      onNodeDragStop={(_, node) => {
        // 折りたたみグループノードの移動はmeta.editor.groupsへ書き戻す
        const gid = groupIdFromNodeId(node.id);
        if (gid) moveGroup(gid, node.position);
        else positionsChanged();
      }}
      onDrop={onDrop}
      onDragOver={onDragOver}
      nodesConnectable
      isValidConnection={isValidConnection}
      deleteKeyCode={['Backspace', 'Delete']}
      colorMode="light"
      fitView
      // 画面外ノードのDOMを間引く。合成ノードは全てnode直下width/heightを
      // 持つため可視判定(nodeHasDimensions)を即時に満たす — この契約を壊さないこと
      onlyRenderVisibleElements
      minZoom={0.1}
      defaultEdgeOptions={{ style: { stroke: 'var(--edge-default)', strokeWidth: 1.5 } }}
    >
      <Panel position="top-left">
        <div className="flex items-center gap-2">
          <button disabled={nodes.length === 0 || layouting} onClick={() => void handleAutoLayout()}>
            {layouting ? '整列中…' : '自動整列'}
          </button>
          <DisplayModeToggle />
        </div>
      </Panel>
      <Panel position="top-right">
        <div className="flex flex-col items-end gap-1">
          <LodIndicator />
          <EdgeLegend />
        </div>
      </Panel>
      {/* §4-4 履歴リプレイ中のバナー: ライブへ戻す導線つき(キャンバス内Panel) */}
      {replay.visible && (
        <Panel position="top-center">
          <div className="replay-banner">
            <span>{replay.text}</span>
            <button className="nodrag" type="button" onClick={() => exitReplay()}>
              ライブに戻る
            </button>
          </div>
        </Panel>
      )}
      <Background variant={BackgroundVariant.Dots} gap={24} color={cssVar('--canvas-dot')} />
      <Controls />
      <MiniMap pannable zoomable />
      {/* 逆伝播再生中のみ粒子オーバーレイをマウント */}
      {bpAnim.frame !== null && <GradParticleOverlay />}
      <QuickAddMenu
        at={quickAddAt}
        onSelect={(type, screen) =>
          addNodeFromPalette(type, screenToFlowPosition(screen))
        }
        onClose={() => setQuickAddAt(null)}
      />
    </ReactFlow>
    {/* §4-4 セピア調オーバーレイ(pointer-events透過。ノードの上・パネルの下に敷く) */}
    {replay.visible && <div className="replay-overlay" aria-hidden />}
    </div>
  );
}

export function FlowCanvas() {
  // 表示モード: カードモードは表示専用の別ビュー。React Flowを使わないため
  // Provider外で分岐する(グラフ編集状態はストア保持なので往復で失われない)。
  const displayMode = useViewStore((s) => s.displayMode);
  if (displayMode === 'nnc-card') return <NncCardCanvas />;
  return (
    <ReactFlowProvider>
      <FlowCanvasInner />
    </ReactFlowProvider>
  );
}
