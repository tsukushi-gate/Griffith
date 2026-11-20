// 統計パネル: NNCの右下Statistics相当。GET /api/statsを購読し、
// IR変更/同期完了のたびに300msデバウンスで再取得する。論理ゲートNN固有の指標
// (ゲート数=total_luts / 学習パラメータ / 論理深度 / 配線数 / GroupSum / ノード構成)を
// 数値+小バーで表示し、選択ノード型の寄与をハイライトする。グラフ/カード両モードで常設。

import { useEffect, useState } from 'react';
import { ApiError, getStats, type StatsResponse } from '../api/rest';
import { getNasTrials } from '../nas/api';
import { paretoCandidates } from '../nas/pareto';
import type { NasTrialRow } from '../nas/types';
import { useGraphStore } from '../store/graphStore';

/** 数値+相対バーの1行。selected時は寄与ハイライト。 */
function StatBar({
  label,
  value,
  max,
  selected = false,
  testid,
}: {
  label: string;
  value: number;
  max: number;
  selected?: boolean;
  testid?: string;
}) {
  const pct = max > 0 ? Math.max(2, (value / max) * 100) : 0;
  return (
    <div
      className={`flex items-center gap-2 rounded px-1 py-0.5 ${selected ? 'bg-primary/10 ring-1 ring-primary/40' : ''}`}
      data-testid={testid}
    >
      <span className="w-20 shrink-0 truncate text-[11px] text-muted-foreground">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-12 shrink-0 text-right text-[11px] tabular-nums font-medium">
        {value.toLocaleString()}
      </span>
    </div>
  );
}

/**
 * 現モデルのパレート面積位置。NAS trialが存在する時のみ表示する。
 * hard accはStatsPanel文脈では未知のため、回路面積(total_luts)次元での
 * 相対位置(探索済み最小面積以下か/より小さい面積のtrialが存在するか)を示す。
 */
function ParetoPositionSection({
  currentLuts,
  trials,
}: {
  currentLuts: number;
  trials: NasTrialRow[];
}) {
  const cands = paretoCandidates(trials);
  if (cands.length === 0) return null;
  const minLut = Math.min(...cands.map((t) => t.total_luts));
  const ahead = currentLuts <= minLut;
  return (
    <div className="flex flex-col gap-0.5" data-testid="pareto-position">
      <div className="text-[10px] font-semibold tracking-wide text-muted-foreground uppercase">
        パレート面積位置
      </div>
      <div className="flex items-center justify-between px-1 text-[11px]">
        <span className="text-muted-foreground">現モデルLUT</span>
        <span className="tabular-nums font-medium">{currentLuts.toLocaleString()}</span>
      </div>
      <div className="flex items-center justify-between px-1 text-[11px]">
        <span className="text-muted-foreground">探索済み最小LUT</span>
        <span className="tabular-nums font-medium">{minLut.toLocaleString()}</span>
      </div>
      <div
        className={`rounded px-1 py-0.5 text-[10px] ${ahead ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground'}`}
      >
        {ahead
          ? '面積は探索済み最小以下(パレート前方候補)'
          : 'より小面積のtrialが存在(面積で劣後)'}
      </div>
    </div>
  );
}

export function StatsPanel() {
  const ir = useGraphStore((s) => s.ir);
  const syncState = useGraphStore((s) => s.syncState);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // NAS trials(現モデルのパレート面積上の相対位置表示。存在時のみ)
  const [nasTrials, setNasTrials] = useState<NasTrialRow[]>([]);

  // IR参照変更 or 同期状態変化(PUT完了=synced)のたびに300msデバウンスで再取得。
  // 追加/削除直後はローカルIRが先行し、800ms後のPUT→syncedで再度取得して確定値に追いつく。
  useEffect(() => {
    // ir未設定時はfetchせず終了(表示は下の早期returnがhintを出す)。
    // 同期setStateはeffect内で避け、staleなstatsは早期returnで隠す。
    if (!ir) return;
    let alive = true;
    const t = setTimeout(() => {
      getStats()
        .then((s) => {
          if (!alive) return;
          setStats(s);
          setError(null);
        })
        .catch((e) => {
          if (!alive) return;
          // IR未設定(404)や不正(422)は数値を消してメッセージ表示
          setError(e instanceof ApiError ? e.message : String(e));
        });
    }, 300);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [ir, syncState]);

  // NAS trialsを購読。存在すれば現モデルのパレート面積位置を出す。
  // 失敗(未探索=trials空)は無視して非表示にする。
  useEffect(() => {
    let alive = true;
    getNasTrials()
      .then((tr) => {
        if (alive) setNasTrials(tr.trials);
      })
      .catch(() => {
        if (alive) setNasTrials([]);
      });
    return () => {
      alive = false;
    };
  }, [ir]);

  const selectedType = ir?.nodes.find((n) => n.id === selectedNodeId)?.type ?? null;

  if (!ir) {
    return <div className="hint-text">モデルを読み込むと統計が表示されます</div>;
  }
  if (!stats) {
    return <div className="hint-text">{error ?? '統計を計算中…'}</div>;
  }

  const scalarMax = Math.max(
    stats.total_luts,
    stats.num_params,
    stats.logic_depth,
    stats.wire_count,
    stats.group_sums,
    1,
  );
  const nodeMax = Math.max(1, ...Object.values(stats.node_counts));
  const lutMax = Math.max(1, ...Object.values(stats.lut_counts));

  return (
    <div className="flex flex-col gap-2" data-testid="stats-panel">
      {stats.partial && (
        <div className="rounded bg-amber-100 px-2 py-0.5 text-[10px] text-amber-800">
          部分値(blackbox含有)
        </div>
      )}
      <div className="flex flex-col gap-0.5">
        <StatBar label="ゲート数" value={stats.total_luts} max={scalarMax} testid="stat-total_luts" />
        <StatBar label="パラメータ" value={stats.num_params} max={scalarMax} testid="stat-num_params" />
        <StatBar label="論理深度" value={stats.logic_depth} max={scalarMax} testid="stat-logic_depth" />
        <StatBar label="配線数" value={stats.wire_count} max={scalarMax} testid="stat-wire_count" />
        <StatBar label="GroupSum" value={stats.group_sums} max={scalarMax} testid="stat-group_sums" />
      </div>

      <ParetoPositionSection currentLuts={stats.total_luts} trials={nasTrials} />

      {Object.keys(stats.node_counts).length > 0 && (
        <div className="flex flex-col gap-0.5">
          <div className="text-[10px] font-semibold tracking-wide text-muted-foreground uppercase">
            ノード構成
          </div>
          {Object.entries(stats.node_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([type, count]) => (
              <StatBar
                key={type}
                label={type}
                value={count}
                max={nodeMax}
                selected={type === selectedType}
                testid={`stat-node-${type}`}
              />
            ))}
          <div
            className="mt-0.5 flex justify-between px-1 text-[10px] text-muted-foreground"
            data-testid="stat-nodes-total"
          >
            <span>合計ノード</span>
            <span className="tabular-nums font-medium">
              {Object.values(stats.node_counts).reduce((a, b) => a + b, 0)}
            </span>
          </div>
        </div>
      )}

      {Object.keys(stats.lut_counts).length > 0 && (
        <div className="flex flex-col gap-0.5">
          <div className="text-[10px] font-semibold tracking-wide text-muted-foreground uppercase">
            LUT内訳(入力数別)
          </div>
          {Object.entries(stats.lut_counts)
            .sort((a, b) => Number(a[0]) - Number(b[0]))
            .map(([n, count]) => (
              <StatBar key={n} label={`${n}入力`} value={count} max={lutMax} testid={`stat-lut-${n}`} />
            ))}
        </div>
      )}
    </div>
  );
}
