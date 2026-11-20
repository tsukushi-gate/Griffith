// アクティブ学習パネル: 並行実行中の学習run一覧を1秒間隔でポーリング表示する。
// 行クリックで監視対象(store.sessionId)を切替、行ごとに停止/モデルへ適用を行う。
// 系列のバックフィルは行わない: フォーカス切替後は以降の更新のみ表示する(仕様上許容)。

import { CheckCircle2, RefreshCw, Square } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { applyRun, getSessions, trainControl, type TrainStatus } from '../api/rest';
import { useGraphStore } from '../store/graphStore';
import { ACTIVE_STATES, useTrainingStore } from '../store/trainingStore';
import { statusVariant } from '../training/statusVariant';

const POLL_MS = 1000;
const SID_LEN = 8; // 表示用の短縮session id長

export function ActiveRunsPanel() {
  const focusedSid = useTrainingStore((s) => s.sessionId);
  const [sessions, setSessions] = useState<TrainStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // 操作中のsid

  const refresh = useCallback(async () => {
    try {
      const res = await getSessions();
      setSessions(res.sessions);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // 初回取得+1秒間隔ポーリング(非同期フェッチ起点のsetStateは意図的)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const handleStop = async (sid: string) => {
    setBusy(sid);
    try {
      await trainControl(sid, 'stop');
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleApply = async (sid: string) => {
    setBusy(sid);
    try {
      const res = await applyRun(sid);
      useGraphStore.getState().setIR(res.ir, { needsBuild: false });
      toast.success('学習済み重みを編集中のモデルへ適用しました');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-2" data-testid="active-runs-panel">
      <div className="flex items-center gap-2">
        <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
          アクティブ学習
        </h2>
        <Button variant="ghost" size="sm" onClick={() => void refresh()} title="再読込">
          <RefreshCw className="size-3.5" />
        </Button>
        <span className="text-[11px] text-muted-foreground">行クリックで監視対象に切替</span>
      </div>
      {error && <div className="error-text">{error}</div>}
      {sessions.length === 0 && !error && (
        <div className="hint-text">実行中の学習はありません</div>
      )}
      {sessions.length > 0 && (
        <table className="w-full text-left text-[11px]">
          <thead className="text-muted-foreground">
            <tr className="border-b">
              <th className="py-1 pr-2">session</th>
              <th className="py-1 pr-2">状態</th>
              <th className="py-1 pr-2">step</th>
              <th className="py-1 pr-2">acc</th>
              <th className="py-1 pr-2">dataset</th>
              <th className="py-1 pr-2">method</th>
              <th className="py-1">操作</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => {
              const focused = s.session_id === focusedSid;
              const terminal = !ACTIVE_STATES.includes(s.state);
              return (
                <tr
                  key={s.session_id}
                  data-testid="active-run-row"
                  className={`cursor-pointer border-b border-border/50 hover:bg-accent ${
                    focused ? 'bg-accent/70' : ''
                  }`}
                  onClick={() => useTrainingStore.getState().focusSession(s.session_id)}
                >
                  <td className="py-1 pr-2 font-mono">{s.session_id.slice(0, SID_LEN)}</td>
                  <td className="py-1 pr-2">
                    <Badge variant="outline" className={statusVariant(s.state)}>
                      {s.state}
                    </Badge>
                  </td>
                  <td className="py-1 pr-2 font-mono">{s.step}</td>
                  <td className="py-1 pr-2 font-mono">
                    {s.acc !== null ? (s.acc * 100).toFixed(1) + '%' : '-'}
                  </td>
                  <td className="py-1 pr-2">{s.dataset ?? '-'}</td>
                  <td className="py-1 pr-2">{s.method ?? '-'}</td>
                  <td className="py-1">
                    <div className="flex gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={terminal || busy === s.session_id}
                        title="停止"
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleStop(s.session_id);
                        }}
                      >
                        <Square className="size-3.5" />
                        停止
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={busy === s.session_id}
                        title="この学習済み重みを編集中のモデルへ適用"
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleApply(s.session_id);
                        }}
                      >
                        <CheckCircle2 className="size-3.5" />
                        適用
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
