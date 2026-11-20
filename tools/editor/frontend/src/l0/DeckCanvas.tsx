// deck.glドリルダウンキャンバス。単一lut_layerを
// OrthographicViewでパン/ズーム表示する。訓練中はWS layer_summaryの
// gate_ids/confidenceで再着色(updateTriggers経由)。
// 逆伝播リプレイで表示層がアクティブになったらエッジをedge_grad強度で着色。

import { Deck, OrthographicView, type OrthographicViewState } from '@deck.gl/core';
import { useEffect, useMemo, useRef, useState } from 'react';
import { layerEdgeGradIntensity } from '../anim/edgeGradMap';
import { getLowering } from '../api/rest';
import { GATE_NAMES } from '../ir/gates';
import type { IRNodeJson } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useLayerView } from '../store/selectors';
import type { LayerSummaryFrame } from '../store/trainingStore';
import { useViewStore } from '../store/viewStore';
import { createEdgeLayer, createNodeLayer, createTextLayer } from './deckLayers';
import { buildL0Data, type L0Data, type L0Node } from './l0Layout';

const MIN_ZOOM = -8;
const MAX_ZOOM = 10;

interface Props {
  nodeId: string;
}

/** データ全体が収まる初期ビューを概算する(visualizer main.ts流用)。 */
function fitViewState(data: L0Data, width: number, height: number): OrthographicViewState {
  const { xMin, xMax, yMin, yMax } = data.bounds;
  const span = Math.max(xMax - xMin, yMax - yMin, 1);
  const canvasSize = Math.max(Math.min(width, height), 100);
  const zoom = Math.max(MIN_ZOOM, Math.min(Math.log2(canvasSize / span) - 0.5, 5));
  return {
    target: [(xMin + xMax) / 2, (yMin + yMax) / 2, 0],
    zoom,
    minZoom: MIN_ZOOM,
    maxZoom: MAX_ZOOM,
  };
}

function hoverText(d: L0Node, nInputs: number): string {
  if (d.kind === 'input') return `入力 x${d.index}`;
  const gate =
    nInputs === 2 ? (GATE_NAMES[d.gateId] ?? `G${d.gateId}`) : `G${d.gateId}`;
  return `#${d.index} ${gate} ・ 確信度 ${(d.confidence * 100).toFixed(1)}%`;
}

export function DeckCanvas({ nodeId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const deckRef = useRef<Deck<OrthographicView[]> | null>(null);
  const [zoom, setZoom] = useState(0);
  const [hover, setHover] = useState<L0Node | null>(null);
  const [virtualNode, setVirtualNode] = useState<IRNodeJson | null>(null);
  const [virtualSummary, setVirtualSummary] = useState<LayerSummaryFrame | null>(null);
  const [virtualTitle, setVirtualTitle] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const ir = useGraphStore((s) => s.ir);
  // live=layer_summary / replay=履歴ビュー(タイムラインスクラブに追従)
  const view = useLayerView(nodeId);
  const liveSummary = useMemo<LayerSummaryFrame | undefined>(() => {
    if (!view.gateIds || !view.confidence) return undefined;
    return {
      type: 'layer_summary',
      layer: nodeId,
      step: view.step ?? 0,
      n_inputs: view.nInputs ?? 2,
      gate_ids: view.gateIds,
      confidence: view.confidence,
    };
  }, [view, nodeId]);

  const irNode = ir?.nodes.find((n) => n.id === nodeId);

  // nodeId/irNodeの変化時に前回の展開結果をクリアする。effectでのsetStateを避け、
  // 依存変化時のstate調整パターンで描画中に反映する(fetch本体はeffectに残す)
  const [prevKey, setPrevKey] = useState<{ nodeId: string; node: IRNodeJson | undefined }>(
    { nodeId, node: irNode },
  );
  if (prevKey.nodeId !== nodeId || prevKey.node !== irNode) {
    setPrevKey({ nodeId, node: irNode });
    setVirtualNode(null);
    setVirtualSummary(null);
    setVirtualTitle(null);
    setLoadError(null);
  }

  useEffect(() => {
    if (irNode?.type === 'lut_layer') return;
    let alive = true;
    getLowering(nodeId, 'L0')
      .then((res) => {
        if (!alive) return;
        if (!res.virtual_ir_node) {
          setLoadError(`${nodeId} はL0 LUTへ展開できません`);
          return;
        }
        setVirtualNode(res.virtual_ir_node);
        setVirtualSummary(res.summary as LayerSummaryFrame | null);
        setVirtualTitle(res.title);
      })
      .catch((e) => {
        if (!alive) return;
        setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [nodeId, irNode]);

  const displayNode = irNode?.type === 'lut_layer' ? irNode : virtualNode;
  const summary = liveSummary ?? virtualSummary ?? undefined;
  const data = useMemo(
    () => (displayNode && displayNode.type === 'lut_layer' ? buildL0Data(displayNode, summary) : null),
    [displayNode, summary],
  );
  // 再着色のupdateTriggersキー(step変化で色を再評価。replayは+0.5でliveと区別)
  const colorVersion = (summary?.step ?? -1) + (view.source === 'replay' ? 0.5 : 0);

  // 逆伝播リプレイ: 表示中の層がアクティブならedge_grad強度を復元
  const bpAnim = useViewStore((s) => s.bpAnim);
  const setView = useViewStore((s) => s.setView);

  // パンくず: 「全体 > レイヤ名」。全体クリックでグラフ/カードの俯瞰へ復帰する。
  const breadcrumb = (
    <div className="l0-breadcrumb" data-testid="l0-breadcrumb">
      <button
        type="button"
        className="nodrag"
        data-testid="l0-breadcrumb-home"
        onClick={() => setView({ kind: 'graph' })}
      >
        全体
      </button>
      <span aria-hidden> › </span>
      <span>{virtualTitle ?? nodeId}</span>
    </div>
  );
  const gradIntensity = useMemo(() => {
    if (!bpAnim.frame || bpAnim.activeLayer !== nodeId) return null;
    const entry = bpAnim.frame.layers.find((l) => l.layer === nodeId);
    return entry ? layerEdgeGradIntensity(entry) : null;
  }, [bpAnim, nodeId]);
  // エッジ再着色のupdateTriggersキー(非アクティブ=-1で基本色へ戻る)
  const gradVersion = gradIntensity ? bpAnim.frame!.step : -1;

  // Deckの生成/破棄(nodeIdごとに作り直して初期ビューをfitする)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const rect = containerRef.current?.getBoundingClientRect();
    const initial = fitViewState(data, rect?.width ?? 800, rect?.height ?? 600);
    setZoom((initial.zoom as number) ?? 0);
    setHover(null);

    const deck = new Deck({
      canvas,
      views: [new OrthographicView({ id: 'ortho', flipY: false })],
      initialViewState: { ortho: initial },
      controller: true,
      onViewStateChange: (params) => {
        const vs = params.viewState as OrthographicViewState;
        if (typeof vs.zoom === 'number') setZoom(vs.zoom);
      },
      getTooltip: () => null,
      layers: [],
    });
    deckRef.current = deck as unknown as Deck<OrthographicView[]>;
    return () => {
      deck.finalize();
      deckRef.current = null;
    };
    // dataの中身更新はレイヤー差し替えで反映するため、作り直しはnodeIdのみ
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId]);

  // レイヤー更新(データ・ズームLOD・訓練再着色)
  useEffect(() => {
    const deck = deckRef.current;
    if (!deck || !data) return;
    deck.setProps({
      layers: [
        createEdgeLayer(data.edges, zoom, gradIntensity, gradVersion),
        createNodeLayer(data.nodes, colorVersion, (info) =>
          setHover(info.object ?? null),
        ),
        createTextLayer(data.nodes, zoom, data.nInputs, colorVersion),
      ],
    });
  }, [data, zoom, colorVersion, gradIntensity, gradVersion]);

  if (!data) {
    return (
      <div className="l0-canvas">
        {breadcrumb}
        <div className="l0-overlay">
          {loadError ?? `ノード ${nodeId} のL0ビューを読込中`}
        </div>
      </div>
    );
  }

  return (
    <div className="l0-canvas" ref={containerRef}>
      {breadcrumb}
      <canvas ref={canvasRef} />
      <div className="l0-overlay">
        <div>
          {virtualTitle ?? nodeId} ・ 入力 {data.inWidth} ・ ゲート {data.outWidth} ・ N={data.nInputs}
        </div>
        <div>
          エッジ {data.edges.length.toLocaleString()} ・ ズーム {zoom.toFixed(1)}
          {summary && ` ・ step ${summary.step}`}
        </div>
        {hover && <div className="l0-hover">{hoverText(hover, data.nInputs)}</div>}
      </div>
    </div>
  );
}
