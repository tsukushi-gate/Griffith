// React Flowキャンバス(tools/editor/frontend/src/editor/FlowCanvas.tsx の構成を流用)。
// 可視グラフ -> elkレイアウト -> React Flowノード/エッジ。ズームでLODを切り替える。

import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Viewport,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAppStore, type LodStage } from '../store';
import type { TreeData, TreeEdge, TreeNode } from '../types';
import { computeVisibleGraph } from './graph';
import { layoutTree, NODE_SIZE } from './layout';
import { NODE_TYPES, type TreeFlowNode } from './nodes/TreeNodes';

/** 論文カードの実寸(px)。見かけサイズでLODを決める。 */
const PAPER_NODE_W = NODE_SIZE.paper.width;
/** カード全文が読める見かけ幅 */
const FULL_VISIBLE_PX = 150;
/** タイトルだけ読める見かけ幅 */
const TILE_VISIBLE_PX = 70;

/** ズーム(とカード実寸)からLOD段階を求める純関数。 */
export function lodStageForZoom(zoom: number, cardWidth: number = PAPER_NODE_W): LodStage {
  const px = zoom * cardWidth;
  if (px >= FULL_VISIBLE_PX) return 'full';
  if (px >= TILE_VISIBLE_PX) return 'tile';
  return 'overview';
}

const EDGE_STYLE: Record<TreeEdge['kind'], Partial<Edge>> = {
  // 親子=実線 / 次に読む=矢印つき破線 / レポート関連=点線
  parent: {
    style: { stroke: 'var(--edge-parent)', strokeWidth: 1.5 },
  },
  next: {
    style: { stroke: 'var(--edge-next)', strokeWidth: 1.8, strokeDasharray: '6 4' },
    markerEnd: { type: 'arrowclosed', color: 'var(--edge-next)', width: 16, height: 16 } as never,
    animated: false,
  },
  report: {
    style: { stroke: 'var(--edge-report)', strokeWidth: 1.5, strokeDasharray: '2 4' },
  },
};

const MINIMAP_COLOR: Record<string, string> = {
  root: 'var(--accent-root)',
  category: 'var(--accent-category)',
  paper: 'var(--accent-paper)',
  report: 'var(--accent-report)',
};

function toFlowNodes(
  nodes: TreeNode[],
  positions: Map<string, { x: number; y: number }>,
  childCounts: Map<string, number>,
  collapsed: ReadonlySet<string>,
  selectedId: string | null,
): TreeFlowNode[] {
  return nodes.map((n) => ({
    id: n.id,
    type: n.kind,
    position: positions.get(n.id) ?? { x: 0, y: 0 },
    selected: n.id === selectedId,
    draggable: false,
    connectable: false,
    data: {
      raw: n as TreeFlowNode['data']['raw'],
      childCount: childCounts.get(n.id) ?? 0,
      collapsed: collapsed.has(n.id),
    },
    width: NODE_SIZE[n.kind].width,
    height: NODE_SIZE[n.kind].height,
  }));
}

function toFlowEdges(edges: TreeEdge[], selectedId: string | null): Edge[] {
  return edges.map((e) => {
    const touched = selectedId != null && (e.source === selectedId || e.target === selectedId);
    const base = EDGE_STYLE[e.kind];
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      type: 'default',
      ...base,
      style: {
        ...(base.style ?? {}),
        opacity: selectedId == null || touched ? 1 : 0.35,
        strokeWidth: touched ? 2.6 : ((base.style?.strokeWidth as number) ?? 1.5),
      },
      data: { kind: e.kind },
    } as Edge;
  });
}

function InnerCanvas({ tree }: { tree: TreeData }) {
  const filters = useAppStore((s) => s.filters);
  const collapsed = useAppStore((s) => s.collapsed);
  const selectedId = useAppStore((s) => s.selectedId);
  const focus = useAppStore((s) => s.focus);
  const select = useAppStore((s) => s.select);
  const setLod = useAppStore((s) => s.setLod);
  const { setCenter, fitView } = useReactFlow();

  const visible = useMemo(
    () => computeVisibleGraph(tree, filters, collapsed),
    [tree, filters, collapsed],
  );

  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map());
  const layoutSeq = useRef(0);
  const didFit = useRef(false);

  // 可視集合が変わるたびにelkで再レイアウト(結果は最新の要求のみ採用)
  useEffect(() => {
    const seq = ++layoutSeq.current;
    let cancelled = false;
    layoutTree(visible.nodes, visible.edges).then((pos) => {
      if (cancelled || seq !== layoutSeq.current) return;
      setPositions(pos);
    });
    return () => {
      cancelled = true;
    };
  }, [visible]);

  // 初回レイアウトはノードが全て原点に重なった状態でfitViewされているため、
  // 座標がReact Flowへ反映されたあと(=このeffect)にもう一度フィットさせる。
  // fitView中のonMoveでLODも追従する。
  useEffect(() => {
    if (didFit.current || positions.size === 0) return;
    didFit.current = true;
    const raf = requestAnimationFrame(() => fitView({ padding: 0.08, maxZoom: 1, duration: 400 }));
    return () => cancelAnimationFrame(raf);
  }, [positions, fitView]);

  const nodes = useMemo(
    () => toFlowNodes(visible.nodes, positions, visible.childCounts, collapsed, selectedId),
    [visible, positions, collapsed, selectedId],
  );
  const edges = useMemo(() => toFlowEdges(visible.edges, selectedId), [visible.edges, selectedId]);

  const onMove = useCallback(
    (_: unknown, viewport: Viewport) => setLod(lodStageForZoom(viewport.zoom)),
    [setLod],
  );

  // 検索/リンクからのフォーカス要求で該当ノードへ寄せる
  useEffect(() => {
    if (!focus) return;
    const pos = positions.get(focus.id);
    if (!pos) return;
    const size = NODE_SIZE[(tree.nodes.find((n) => n.id === focus.id)?.kind ?? 'paper') as 'paper'];
    setCenter(pos.x + size.width / 2, pos.y + size.height / 2, { zoom: 1, duration: 500 });
  }, [focus, positions, setCenter, tree.nodes]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => select(node.id)}
      onPaneClick={() => select(null)}
      onMove={onMove}
      onInit={(inst) => setLod(lodStageForZoom(inst.getViewport().zoom))}
      fitView
      fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
      minZoom={0.05}
      maxZoom={2}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      proOptions={{ hideAttribution: true }}
      data-testid="tree-canvas"
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="var(--canvas-dot)" />
      <Controls showInteractive={false} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(n) => MINIMAP_COLOR[n.type ?? 'paper'] ?? 'var(--edge-parent)'}
        maskColor="rgba(100,116,139,0.18)"
        style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}
      />
    </ReactFlow>
  );
}

export function TreeCanvas({ tree }: { tree: TreeData }) {
  return (
    <ReactFlowProvider>
      <InnerCanvas tree={tree} />
    </ReactFlowProvider>
  );
}
