// Run一覧パネル。
// セッション横断のrun履歴を一覧表示し、選択したrunのacc曲線を重ね描き比較する。
// データはバックエンドのrunレジストリ(outputs/editor_runs/)から取得する。

import { ExternalLink, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type uPlot from 'uplot';
import { Button } from '@/components/ui/button';
import { getRun, getRuns, type RunRecord, type RunSummary } from '../api/rest';
import { cssVar } from '../theme/theme';
import { UplotChart } from '../training/UplotChart';
import { ActiveRunsPanel } from './ActiveRunsPanel';
import { alignRunSeries } from './runsCompare';

const SERIES_COUNT = 8;
const MAX_COMPARE = 4;

function seriesColor(i: number): string {
  return cssVar(`--chart-series-${i % SERIES_COUNT}`);
}

function fmtTime(ts: number | null): string {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString('ja-JP', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

export function RunsPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [records, setRecords] = useState<Record<string, RunRecord>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await getRuns();
      setRuns(res.runs);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // 初回マウント時のデータ取得(非同期フェッチ起点のsetStateは意図的)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const toggleSelect = async (runId: string) => {
    if (selected.includes(runId)) {
      setSelected(selected.filter((id) => id !== runId));
      return;
    }
    if (selected.length >= MAX_COMPARE) return;
    if (!records[runId]) {
      try {
        const rec = await getRun(runId);
        setRecords((prev) => ({ ...prev, [runId]: rec }));
      } catch {
        return;
      }
    }
    setSelected([...selected, runId]);
  };

  // 選択runのacc系列を共通x軸へ整列
  const compareData = useMemo(() => {
    const recs = selected.map((id) => records[id]).filter(Boolean);
    if (recs.length === 0) return null;
    const aligned = alignRunSeries(
      recs.map((r) => ({ steps: r.series.steps, values: r.series.acc })),
    );
    return {
      data: aligned as uPlot.AlignedData,
      series: recs.map((r, i) => ({
        label: `${r.model_name} (${r.dataset})`,
        stroke: seriesColor(i),
        spanGaps: true,
      })),
    };
  }, [selected, records]);

  return (
    <div className="flex h-full flex-col gap-2 overflow-y-auto p-3">
      {/* 上段: 並行実行中のライブrun。下段: 終了済みrun履歴 */}
      <ActiveRunsPanel />
      <div className="flex items-center gap-2">
        <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
          Run履歴
        </h2>
        <Button variant="ghost" size="sm" onClick={() => void refresh()} title="再読込">
          <RefreshCw className="size-3.5" />
        </Button>
        <span className="text-[11px] text-muted-foreground">
          行クリックで比較対象に追加(最大{MAX_COMPARE}件)
        </span>
      </div>
      {error && <div className="error-text">{error}</div>}
      {runs.length === 0 && !error && (
        <div className="hint-text">
          まだrunがありません(学習セッションが終了すると記録されます)
        </div>
      )}
      {runs.length > 0 && (
        <table className="w-full text-left text-[11px]">
          <thead className="text-muted-foreground">
            <tr className="border-b">
              <th className="py-1 pr-2">比較</th>
              <th className="py-1 pr-2">終了時刻</th>
              <th className="py-1 pr-2">モデル</th>
              <th className="py-1 pr-2">dataset</th>
              <th className="py-1 pr-2">method/loss</th>
              <th className="py-1 pr-2">lr</th>
              <th className="py-1 pr-2">step</th>
              <th className="py-1 pr-2">最終acc</th>
              <th className="py-1 pr-2">状態</th>
              <th className="py-1">wandb</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const sel = selected.includes(r.run_id);
              const selIdx = selected.indexOf(r.run_id);
              return (
                <tr
                  key={r.run_id}
                  className={`cursor-pointer border-b border-border/50 hover:bg-accent ${
                    sel ? 'bg-accent/70' : ''
                  }`}
                  onClick={() => void toggleSelect(r.run_id)}
                >
                  <td className="py-1 pr-2">
                    {sel && (
                      <span
                        className="inline-block h-2.5 w-2.5 rounded-full"
                        style={{ background: seriesColor(selIdx) }}
                      />
                    )}
                  </td>
                  <td className="py-1 pr-2 whitespace-nowrap">{fmtTime(r.ended_at)}</td>
                  <td className="max-w-[140px] truncate py-1 pr-2 font-mono">
                    {r.model_name}
                  </td>
                  <td className="py-1 pr-2">{r.dataset}</td>
                  <td className="py-1 pr-2">{r.method}/{r.loss}</td>
                  <td className="py-1 pr-2 font-mono">{r.lr}</td>
                  <td className="py-1 pr-2 font-mono">{r.final_step}</td>
                  <td className="py-1 pr-2 font-mono">
                    {r.final_acc !== null ? (r.final_acc * 100).toFixed(1) + '%' : '-'}
                  </td>
                  <td className="py-1 pr-2">{r.state}</td>
                  <td className="py-1">
                    {r.wandb_url && (
                      <a
                        href={r.wandb_url}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="text-primary"
                        title="wandbで開く"
                      >
                        <ExternalLink className="size-3.5" />
                      </a>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {compareData && (
        <div className="mt-2">
          <UplotChart
            title="acc比較(選択run)"
            series={compareData.series}
            data={compareData.data}
            height={180}
          />
        </div>
      )}
    </div>
  );
}
