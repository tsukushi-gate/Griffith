// 反例トレースの原因候補パネル(§4-2)。
// トレース中(traceK != null)のみ表示し、到達コーン上の原因候補ゲートを
// スコア降順で並べる。クリックで該当ノードを選択+キャンバスへジャンプする。

import { flowApi } from '../editor/flowApi';
import { useGraphStore } from '../store/graphStore';
import { useUIStore } from '../store/uiStore';
import { useViewStore } from '../store/viewStore';
import { usePlaygroundStore } from './playgroundStore';

const CANDIDATE_LIMIT = 8;

/** 候補クリック: グラフビューへ戻し、該当ノード+ニューロンを選択して中央寄せする */
function jumpToGate(nodeId: string, neuron: number): void {
  useViewStore.getState().setView({ kind: 'graph' });
  useGraphStore.getState().setSelected(nodeId);
  useUIStore.getState().setSelectedNeuron({ nodeId, index: neuron });
  flowApi.current?.focusNode(nodeId);
}

export function CounterexampleTracePanel() {
  const traceK = usePlaygroundStore((s) => s.traceK);
  const cone = usePlaygroundStore((s) => s.traceCone);
  const candidates = usePlaygroundStore((s) => s.traceCandidates);
  const clearTrace = usePlaygroundStore((s) => s.clearTrace);

  if (traceK === null || !cone) return null;

  const shown = candidates.slice(0, CANDIDATE_LIMIT);

  return (
    <div className="flex flex-col gap-1 rounded-md border border-[var(--edge-error)] bg-destructive/5 p-2">
      <div className="flex items-center justify-between">
        <div className="text-[11px] font-semibold text-destructive">
          反例トレース(パターン #{traceK})
        </div>
        <button
          type="button"
          className="cursor-pointer rounded border px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted"
          onClick={clearTrace}
        >
          解除
        </button>
      </div>
      <div className="text-[10px] text-muted-foreground">
        到達コーン {cone.nodes.size}ノード・原因候補 {candidates.length}件
        (濃い枠=コーン上、キャンバスで信号伝播を確認)
      </div>
      {shown.length === 0 ? (
        <div className="text-[10px] text-muted-foreground">
          コーン上に迷い・食い違いのあるゲートはありません
        </div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {shown.map((c) => (
            <button
              key={`${c.nodeId}:${c.neuron}`}
              type="button"
              onClick={() => jumpToGate(c.nodeId, c.neuron)}
              className="flex items-center justify-between gap-2 rounded px-2 py-0.5 text-[11px] hover:bg-muted"
              title="クリックで該当ゲートへジャンプ"
            >
              <span className="truncate font-mono">
                {c.nodeId} #{c.neuron}
              </span>
              <span className="flex shrink-0 items-center gap-1 text-muted-foreground">
                {c.disagreement && (
                  <span className="rounded bg-destructive/15 px-1 text-[9px] font-semibold text-destructive">
                    soft≠hard
                  </span>
                )}
                <span>迷い {c.entropy.toFixed(2)}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
