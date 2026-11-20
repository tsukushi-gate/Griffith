// 回路図キャンバス(React Flow)。
// - ブロック/ポート/プローブの 3 ノード型。プローブは回路図と一緒に pan/zoom
// - ノードの source ハンドルから空きキャンバスへドラッグ → その場にプローブ生成
//   (isValidConnection=false で実配線は決して作られない=回路は不変)
// - 1bit エッジは RTL 実値で点灯。viewport は永続化
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type OnConnectEnd,
} from '@xyflow/react';
import { useCallback, useMemo } from 'react';
import { useT } from '../i18n/useT';
import { useDetailStore } from '../store/detailStore';
import { useSimStore } from '../store/simStore';
import { useViewerStore } from '../store/viewerStore';
import { applyExpansion } from '../graph/expandGraph';
import { metaToFlow, sourceWidth } from '../graph/metaToFlow';
import { probeEdges, probeNodes } from '../graph/probeEdges';
import { edgeValueStyle } from '../graph/edgeStyle';
import { bitOf } from '../graph/hexBits';
import { BlockNode } from '../nodes/BlockNode';
import { GateNode } from '../nodes/GateNode';
import { LutGroupNode } from '../nodes/LutGroupNode';
import { PortNode } from '../nodes/PortNode';
import { ProbeNode } from '../probes/ProbeNode';
import { PinnedProbes } from '../probes/PinnedProbes';

const NODE_TYPES = {
  block: BlockNode,
  port: PortNode,
  probe: ProbeNode,
  gate: GateNode,
  lutGroup: LutGroupNode,
};

function CanvasInner() {
  const meta = useSimStore((s) => s.meta);
  const values = useSimStore((s) => s.frame?.values);
  const positions = useViewerStore((s) => s.positions);
  const probes = useViewerStore((s) => s.probes);
  const addProbe = useViewerStore((s) => s.addProbe);
  const updateProbe = useViewerStore((s) => s.updateProbe);
  const setNodePosition = useViewerStore((s) => s.setNodePosition);
  const setViewport = useViewerStore((s) => s.setViewport);
  const savedViewport = useViewerStore((s) => s.viewport);
  const { screenToFlowPosition, getViewport } = useReactFlow();

  const expanded = useViewerStore((s) => s.expanded);
  const paramsById = useDetailStore((s) => s.params);

  const base = useMemo(() => {
    if (!meta) return { nodes: [] as Node[], edges: [] as Edge[] };
    const flow = metaToFlow(meta, positions);
    return applyExpansion(meta, flow.nodes, flow.edges, expanded, paramsById);
  }, [meta, positions, expanded, paramsById]);

  const nodes: Node[] = useMemo(
    () => [...base.nodes, ...probeNodes(probes)],
    [base.nodes, probes],
  );

  const edges: Edge[] = useMemo(() => {
    if (!meta) return [];
    const styled = base.edges.map((e) => {
      // ゲート展開のビット配線: 参照 bit の実値で点灯
      const srcBit = (e.data as { srcBit?: number } | undefined)?.srcBit;
      if (srcBit !== undefined && values) {
        const on = bitOf(values[e.source], srcBit);
        return {
          ...e,
          style: { strokeWidth: 0.8,
                   stroke: on ? 'var(--edge-on)' : 'var(--edge-off)' },
        };
      }
      // 1bit バスは実行値で点灯(多 bit は基底スタイルのまま)
      const w = sourceWidth(meta, e.source);
      if (w === 1 && values) {
        return { ...e, style: edgeValueStyle(bitOf(values[e.source], 0)) };
      }
      return e;
    });
    const known = new Set(nodes.map((n) => n.id));
    return [...styled, ...probeEdges(probes, known)];
  }, [meta, base.edges, values, probes, nodes]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // 位置変更だけをストアへ書き戻す(ノード自体はストア導出のため)
      for (const ch of changes) {
        if (ch.type === 'position' && ch.position) {
          if (ch.id.startsWith('probe:')) {
            updateProbe(ch.id.slice('probe:'.length), { pos: ch.position });
          } else {
            setNodePosition(ch.id, ch.position);
          }
        }
      }
    },
    [updateProbe, setNodePosition],
  );

  const onConnectEnd: OnConnectEnd = useCallback(
    (event, connectionState) => {
      if (connectionState.isValid) return;
      const from = connectionState.fromNode;
      if (!from) return;
      const ev = event as MouseEvent;
      const pos = screenToFlowPosition({ x: ev.clientX, y: ev.clientY });
      // 出力ポートからの引き出しは y_comb(組み合わせ出力)を既定にする
      const target = from.id === 'y' ? 'y_comb' : from.id;
      addProbe(target, pos);
    },
    [screenToFlowPosition, addProbe],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodesChange={onNodesChange}
      onConnectEnd={onConnectEnd}
      isValidConnection={() => false}
      onMoveEnd={() => setViewport(getViewport())}
      defaultViewport={savedViewport ?? undefined}
      fitView={!savedViewport}
      minZoom={0.05}
      onlyRenderVisibleElements
      proOptions={{ hideAttribution: true }}
      deleteKeyCode={null}
      defaultEdgeOptions={{
        style: { stroke: 'var(--edge-default)', strokeWidth: 1.5 },
      }}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1}
                  color="var(--canvas-dot)" />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable />
      <Panel position="top-right">
        <PinnedProbes />
      </Panel>
      <Panel position="bottom-center">
        <CanvasLegend />
      </Panel>
    </ReactFlow>
  );
}

/** エッジ凡例+操作ヒント(エディタ EdgeLegend と同様式のピル)。 */
function CanvasLegend() {
  const t = useT();
  return (
    <div className="edge-legend">
      <span><span className="swatch" /> {t('legend.logic')}</span>
      <span><span className="swatch on" /> {t('legend.on')}</span>
      <span><span className="swatch probe" /> {t('legend.probe')}</span>
      <span>{t('legend.width')}</span>
      <span>·</span>
      <span>{t('canvas.hint')}</span>
    </div>
  );
}

export function SchematicCanvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
