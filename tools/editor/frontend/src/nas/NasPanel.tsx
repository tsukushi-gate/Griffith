// 構造探索パネル: TRAININGタブ内「構造探索」サブタブの本体。
//   - 探索の開始/停止(max_trials/epochs)
//   - Trade-offグラフ(精度×LUT数散布図、パレート強調)
//   - Results History(trial一覧表: acc/LUT/深度/ゲート、パレート/評価済みバッジ)
//   - 点/行クリックでそのtrialのIRをEDITに開く
// 探索中は1秒ごとに status/trials をポーリングする(小規模CPU逐次前提)。

import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { putCurrentIR } from '../api/rest';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';
import {
  getNasStatus,
  getNasTrialIr,
  getNasTrials,
  startNasSearch,
  stopNasSearch,
} from './api';
import { TradeoffChart } from './TradeoffChart';
import type { NasStatus, NasTrialRow } from './types';

const POLL_MS = 1000;

/** trialのIRをサーバへPUTしてEDITに開く(loadSampleModelと同型の経路)。 */
async function openTrialInEdit(trialId: number): Promise<void> {
  const res = await getNasTrialIr(trialId);
  await putCurrentIR(res.ir);
  useGraphStore.getState().setIR(res.ir, { needsBuild: true });
  useTrainingStore.getState().resetAll();
  useUIStore.getState().setTopTab('edit');
  toast.success(`trial ${trialId} の構造をEDITに読み込みました(buildして学習できます)`);
}

export function NasPanel() {
  const [maxTrials, setMaxTrials] = useState(12);
  const [epochs, setEpochs] = useState(5);
  const [status, setStatus] = useState<NasStatus | null>(null);
  const [trials, setTrials] = useState<NasTrialRow[]>([]);
  const [pareto, setPareto] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [st, tr] = await Promise.all([getNasStatus(), getNasTrials()]);
      setStatus(st);
      setTrials(tr.trials);
      setPareto(new Set(tr.pareto));
      return st.state;
    } catch {
      return 'idle';
    }
  }, []);

  // 実行中のみポーリング。完了/停止で自動的に止める。
  useEffect(() => {
    const running = status?.state === 'running';
    if (running && timerRef.current === null) {
      timerRef.current = setInterval(() => {
        void refresh();
      }, POLL_MS);
    }
    if (!running && timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [status?.state, refresh]);

  // 初回に既存台帳を読む(再開・別セッションの結果表示)。
  // effect本体での同期setStateを避けるためマイクロタスクへ逃がす。
  useEffect(() => {
    const t = setTimeout(() => void refresh(), 0);
    return () => clearTimeout(t);
  }, [refresh]);

  const onStart = async () => {
    setBusy(true);
    try {
      await startNasSearch({
        search_id: 'editor',
        dataset: 'logic:xor',
        model_kind: 'logic_mlp',
        max_trials: maxTrials,
        epochs,
        n_inputs_choices: [2, 4],
      });
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onStop = async () => {
    setBusy(true);
    try {
      await stopNasSearch();
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onOpen = async (trialId: number) => {
    try {
      await openTrialInEdit(trialId);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  const running = status?.state === 'running';

  return (
    <div className="flex h-full flex-col gap-2 overflow-auto p-2" data-testid="nas-panel">
      {/* コントロール */}
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-[10px] text-muted-foreground">
          最大trial数
          <Input
            id="nas-max-trials"
            data-testid="nas-max-trials"
            type="number"
            min={1}
            value={maxTrials}
            onChange={(e) => setMaxTrials(Number(e.target.value))}
            className="h-7 w-20 text-xs"
          />
        </label>
        <label className="flex flex-col text-[10px] text-muted-foreground">
          epoch数
          <Input
            id="nas-epochs"
            data-testid="nas-epochs"
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            className="h-7 w-20 text-xs"
          />
        </label>
        {running ? (
          <Button size="sm" variant="destructive" onClick={onStop} disabled={busy} data-testid="nas-stop">
            停止
          </Button>
        ) : (
          <Button size="sm" onClick={onStart} disabled={busy} data-testid="nas-start">
            構造探索を開始
          </Button>
        )}
        {status && (
          <span className="text-[11px] text-muted-foreground" data-testid="nas-status">
            {status.state} · {status.n_trials}
            {status.max_trials ? ` / ${status.max_trials}` : ''} trial
          </span>
        )}
      </div>

      {/* Trade-offグラフ */}
      <div className="rounded border p-1">
        <TradeoffChart trials={trials} onSelectTrial={onOpen} />
      </div>

      {/* Results History(trial一覧) */}
      <div className="min-h-0 flex-1">
        <div className="mb-1 text-[10px] font-semibold tracking-wide text-muted-foreground uppercase">
          Results History
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]" data-testid="nas-trials-table">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="px-1 py-0.5">#</th>
                <th className="px-1 py-0.5">acc</th>
                <th className="px-1 py-0.5">LUT</th>
                <th className="px-1 py-0.5">深度</th>
                <th className="px-1 py-0.5">ゲート</th>
                <th className="px-1 py-0.5">状態</th>
              </tr>
            </thead>
            <tbody>
              {trials.map((t) => (
                <tr
                  key={t.trial_id}
                  className="cursor-pointer border-t hover:bg-muted/50"
                  onClick={() => onOpen(t.trial_id)}
                  data-testid={`nas-trial-row-${t.trial_id}`}
                >
                  <td className="px-1 py-0.5 tabular-nums">{t.trial_id}</td>
                  <td className="px-1 py-0.5 tabular-nums">{t.acc.toFixed(3)}</td>
                  <td className="px-1 py-0.5 tabular-nums">{t.total_luts}</td>
                  <td className="px-1 py-0.5 tabular-nums">{t.logic_depth}</td>
                  <td className="px-1 py-0.5 tabular-nums">{t.gates}</td>
                  <td className="px-1 py-0.5">
                    {pareto.has(t.trial_id) ? (
                      <Badge className="h-4 px-1 text-[9px]">PARETO</Badge>
                    ) : t.partial ? (
                      <Badge variant="outline" className="h-4 px-1 text-[9px]">部分値</Badge>
                    ) : t.state === 'done' ? (
                      <Badge variant="secondary" className="h-4 px-1 text-[9px]">評価済</Badge>
                    ) : (
                      <Badge variant="outline" className="h-4 px-1 text-[9px]">{t.state}</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {trials.length === 0 && (
            <div className="hint-text py-2">まだtrialがありません</div>
          )}
        </div>
      </div>
    </div>
  );
}
