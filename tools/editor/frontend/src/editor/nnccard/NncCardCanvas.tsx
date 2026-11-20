// NNC風レイヤーカードキャンバス。表示専用。
// graphStoreのIR/nodesと形状診断を「読むだけ」で、トポロジカル順の縦積み+
// 直交(rail)エッジとして描画する。編集ハンドラ(ドラッグ/接続)は一切持たない
// — 編集はグラフモード(FlowCanvas)で行う、という役割分担を徹底する。

import { useMemo } from 'react';
import { commentsRecordOf, useGraphStore } from '../../store/graphStore';
import { useUIStore } from '../../store/uiStore';
import { useViewStore } from '../../store/viewStore';
import { nodeWidths } from '../../ir/shapeInference';
import type { CardKind } from './CARD_STYLE_BY_TYPE';
import { mainParamsFor } from './CARD_STYLE_BY_TYPE';
import { DisplayModeToggle } from './DisplayModeToggle';
import { NncCardNode } from './NncCardNode';
import { topologicalOrder } from './order';

// レイアウト定数(SVGエッジのy座標算出と共有する単一情報源)
const PAD_TOP = 16;
const CARD_H = 56;
const GAP = 14;
const ROW = CARD_H + GAP;
const CARD_LEFT = 44; // カード左端x(エッジ接続点)
const RIGHT_PAD = 16;

interface CardModel {
  id: string;
  kind: CardKind;
  name: string;
  params: string;
  inWidth: number | null;
  outWidth: number | null;
  drillable: boolean;
  inDegree: number;
  comment?: string;
  hasError: boolean;
}

function rowCenter(rank: number): number {
  return PAD_TOP + rank * ROW + CARD_H / 2;
}

export function NncCardCanvas() {
  const ir = useGraphStore((s) => s.ir);
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const setSelected = useGraphStore((s) => s.setSelected);
  const setSelectedNeuron = useUIStore((s) => s.setSelectedNeuron);
  const setView = useViewStore((s) => s.setView);

  const { cards, edges } = useMemo(() => {
    if (!ir) return { cards: [] as CardModel[], edges: [] as { src: number; dst: number }[] };

    const order = topologicalOrder(ir);
    const rank = new Map(order.map((id, i) => [id, i]));
    const comments = commentsRecordOf(ir);
    const errorIds = new Set(
      diagnostics.filter((d) => d.level === 'error' && d.nodeId).map((d) => d.nodeId as string),
    );
    const inDeg = new Map<string, number>();
    for (const e of ir.edges) inDeg.set(e.dst, (inDeg.get(e.dst) ?? 0) + 1);

    const portWidth = new Map<string, number>();
    for (const p of ir.inputs) portWidth.set(p.id, p.width);
    for (const p of ir.outputs) portWidth.set(p.id, p.width);
    const inputIds = new Set(ir.inputs.map((p) => p.id));
    const nodeById = new Map(ir.nodes.map((n) => [n.id, n]));

    const cards: CardModel[] = order.map((id) => {
      const node = nodeById.get(id);
      if (node) {
        const w = nodeWidths(node);
        const label = ((node.meta?.editor as { label?: string } | undefined)?.label) ?? node.id;
        return {
          id,
          kind: node.type as CardKind,
          name: label,
          params: mainParamsFor(node),
          inWidth: w.in,
          outWidth: w.out,
          drillable: true,
          inDegree: inDeg.get(id) ?? 0,
          comment: comments[id],
          hasError: errorIds.has(id),
        };
      }
      // ポート(必ず入力 or 出力)
      const kind: CardKind = inputIds.has(id) ? 'input' : 'output';
      const w = portWidth.get(id) ?? null;
      return {
        id,
        kind,
        name: id,
        params: '',
        inWidth: kind === 'output' ? w : null,
        outWidth: kind === 'input' ? w : w,
        drillable: false,
        inDegree: inDeg.get(id) ?? 0,
        comment: comments[id],
        hasError: errorIds.has(id),
      };
    });

    const edges = ir.edges
      .map((e) => ({ src: rank.get(e.src), dst: rank.get(e.dst) }))
      .filter((e): e is { src: number; dst: number } => e.src !== undefined && e.dst !== undefined);

    return { cards, edges };
  }, [ir, diagnostics]);

  const maxWidth = useMemo(
    () => cards.reduce((m, c) => Math.max(m, c.outWidth ?? 0), 1),
    [cards],
  );

  const height = PAD_TOP * 2 + cards.length * ROW;

  const onDrill = (nodeId: string) => {
    setSelected(nodeId);
    setSelectedNeuron(null);
    setView({ kind: 'l0', nodeId });
  };

  if (!ir) {
    return (
      <div className="relative h-full w-full">
        <div className="absolute right-2 top-2 z-10">
          <DisplayModeToggle />
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-auto bg-[var(--canvas-bg,transparent)]" data-testid="nnc-card-canvas">
      <div className="sticky right-2 top-2 z-10 float-right mr-2 mt-2">
        <DisplayModeToggle />
      </div>
      <div className="relative mx-auto max-w-2xl" style={{ height, minWidth: 360 }}>
        {/* 直交エッジ(rail配線): カード左端 → 縦rail → 目的カード左端 */}
        <svg
          className="pointer-events-none absolute inset-0"
          width="100%"
          height={height}
          aria-hidden
        >
          {edges.map((e, i) => {
            const y1 = rowCenter(e.src);
            const y2 = rowCenter(e.dst);
            const railX = 6 + (i % 5) * 3; // エッジごとに微オフセットして重なりを緩和
            return (
              <path
                key={i}
                d={`M ${CARD_LEFT} ${y1} H ${railX} V ${y2} H ${CARD_LEFT}`}
                fill="none"
                stroke="var(--edge-default)"
                strokeWidth={1.5}
              />
            );
          })}
        </svg>
        {cards.map((c, i) => (
          <div
            key={c.id}
            className="absolute"
            style={{ top: PAD_TOP + i * ROW, left: CARD_LEFT, right: RIGHT_PAD, height: CARD_H }}
          >
            <NncCardNode
              id={c.id}
              kind={c.kind}
              name={c.name}
              params={c.params}
              inWidth={c.inWidth}
              outWidth={c.outWidth}
              maxWidth={maxWidth}
              hasError={c.hasError}
              inDegree={c.inDegree}
              comment={c.comment}
              drillable={c.drillable}
              selected={selectedNodeId === c.id}
              onSelect={() => {
                setSelected(c.id);
                setSelectedNeuron(null);
              }}
              onDrill={() => onDrill(c.id)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
