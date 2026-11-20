// 選択ギャップ・ダッシュボード(§4-1、学習タブ内)。
//   1. soft/hard精度と選択ギャップの数値サマリ
//   2. レイヤ別の迷いヒートストリップ(layer_summaryのentropy_mean、色強度=迷い)
//   3. 低確信ゲートTop-N(REST)→クリックで該当ノード選択+キャンバスジャンプ
// hard化で失う精度=このドメイン最大のデバッグ課題を一望できる看板パネル。

import { useEffect, useMemo, useState } from 'react';
import { getUncertainGates, type UncertainGate } from '../api/rest';
import { flowApi } from '../editor/flowApi';
import { GATE_NAMES } from '../ir/gates';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useViewStore } from '../store/viewStore';
import { cssVarRGB } from '../theme/theme';
import { clampUncertainty, latestGap } from './selectionGap';

const TOP_N = 12;

/** 迷い(正規化エントロピー)を赤系の背景色へ。0=透明、1=不透明の--destructive。 */
function heatColor(norm: number): string {
  const [r, g, b] = cssVarRGB('--destructive', [220, 38, 38]);
  const alpha = Math.max(0.06, norm);
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

/** Top-Nクリック: グラフビューへ戻し、該当レイヤノードを選択+中央寄せする。 */
function jumpToLayer(layerId: string): void {
  useViewStore.getState().setView({ kind: 'graph' });
  useGraphStore.getState().setSelected(layerId);
  flowApi.current?.focusNode(layerId);
}

/** レイヤ別の迷いヒートストリップ。セル=レイヤ、色=平均迷い、クリックでジャンプ。 */
function LayerHeatStrip() {
  const version = useTrainingStore((s) => s.version);
  void version;
  const summaries = useTrainingStore.getState().summaries;
  const cells = useMemo(
    () =>
      Object.values(summaries)
        .filter((s) => s.entropy_mean != null)
        .sort((a, b) => a.layer.localeCompare(b.layer)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  );

  if (cells.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-muted-foreground">
        レイヤ別の迷い(正規化エントロピー 0〜1・濃いほど未確定)
      </div>
      <div className="flex flex-wrap gap-1">
        {cells.map((s) => {
          const norm = clampUncertainty(s.entropy_mean ?? 0);
          return (
            <button
              key={s.layer}
              type="button"
              onClick={() => jumpToLayer(s.layer)}
              title={`${s.layer} ・ 平均 ${(s.entropy_mean ?? 0).toFixed(3)} / 最大 ${(s.entropy_max ?? 0).toFixed(3)}(正規化迷い 0〜1)`}
              className="flex h-9 min-w-16 flex-col items-center justify-center rounded border px-2 text-[10px] leading-tight hover:ring-1 hover:ring-primary"
              style={{ backgroundColor: heatColor(norm) }}
            >
              <span className="max-w-20 truncate font-mono">{s.layer}</span>
              <span className="text-muted-foreground">
                {(s.entropy_mean ?? 0).toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** 低確信ゲートTop-N(REST取得)。eval回数の変化を再取得トリガにする。 */
function UncertainGateList() {
  const sessionId = useTrainingStore((s) => s.sessionId);
  const version = useTrainingStore((s) => s.version);
  void version;
  const valCount = useTrainingStore.getState().valAcc.length;
  const [gates, setGates] = useState<UncertainGate[]>([]);

  useEffect(() => {
    if (!sessionId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- セッション切替時のリセット(意図的)
      setGates([]);
      return;
    }
    let disposed = false;
    getUncertainGates(sessionId, TOP_N)
      .then((res) => {
        if (!disposed) setGates(res.gates);
      })
      .catch(() => {
        if (!disposed) setGates([]);
      });
    return () => {
      disposed = true;
    };
  }, [sessionId, valCount]);

  if (gates.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-muted-foreground">
        確信度が低いゲート Top-{TOP_N}(クリックで該当ノードへジャンプ)
      </div>
      <div className="flex flex-col gap-0.5">
        {gates.map((g) => (
          <button
            key={`${g.layer_id}:${g.neuron_index}`}
            type="button"
            onClick={() => jumpToLayer(g.layer_id)}
            className="flex items-center justify-between gap-2 rounded px-2 py-0.5 text-[11px] hover:bg-muted"
          >
            <span className="truncate font-mono">
              {g.layer_id} #{g.neuron_index}
            </span>
            <span className="shrink-0 text-muted-foreground">
              {GATE_NAMES[g.gate_id] ?? `#${g.gate_id}`} ・ 迷い {g.entropy.toFixed(3)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function SelectionGapPanel() {
  const version = useTrainingStore((s) => s.version);
  void version;
  const store = useTrainingStore.getState();
  const gap = latestGap(store.valAccSoft, store.valAcc);

  return (
    <div className="flex flex-col gap-2 border-t pt-2">
      <div className="text-[11px] font-medium">選択ギャップ(§4-1)</div>
      {gap ? (
        <div className="text-[11px] text-muted-foreground">
          soft {(gap.soft * 100).toFixed(1)}% ・ hard {(gap.hard * 100).toFixed(1)}%
          {' ・ '}
          <span className={gap.gap > 0.01 ? 'font-semibold text-destructive' : ''}>
            ギャップ {(gap.gap * 100).toFixed(1)}pt
          </span>
        </div>
      ) : (
        <div className="text-[11px] text-muted-foreground">
          hard eval後に soft/hard 精度の乖離を表示します
        </div>
      )}
      <LayerHeatStrip />
      <UncertainGateList />
    </div>
  );
}
