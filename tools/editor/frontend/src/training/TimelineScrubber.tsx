// 学習タイムラインスクラバー。
// step単位シーク・再生/逆再生/コマ送り・live復帰・branch再学習(ここから再学習)。
// スクラブ時は historyClient がスナップショットビューを取得し、
// selectors.useLayerView 経由で全可視化が当該stepへ同期する。
// 履歴インデックスはWSのhistory_indexプッシュが主。ポーリングはWS未接続時のみ。
// 仕様: replayへの進入はこのコンポーネントのUI操作(スライダー/コマ送り/
//   再生ボタン)のみ。liveへの復帰は「LIVEへ戻る」またはresume/step実行時。
// スライダー下にエポック境界目盛とval_accマーカーを表示する。

import {
  ChevronFirst,
  ChevronLast,
  GitBranch,
  Loader2,
  Pause,
  Play,
  Radio,
  StepBack,
  StepForward,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { invalidateHistoryCache, loadReplayStep } from '../api/historyClient';
import { getTrainHistoryIndex, trainRestore } from '../api/rest';
import { ACTIVE_STATES, useTrainingStore } from '../store/trainingStore';

const INDEX_POLL_MS = 2000;   // 履歴インデックスの更新間隔(WS未接続時のみ)
const SCRUB_FETCH_DEBOUNCE_MS = 80;
const SPEED_OPTIONS = [2, 5, 10, 20]; // steps/sec

export function TimelineScrubber() {
  const sessionId = useTrainingStore((s) => s.sessionId);
  const status = useTrainingStore((s) => s.status);
  const displayMode = useTrainingStore((s) => s.displayMode);
  const cursorStep = useTrainingStore((s) => s.cursorStep);
  const playback = useTrainingStore((s) => s.playback);
  const historyMeta = useTrainingStore((s) => s.historyMeta);
  const captureConfig = useTrainingStore((s) => s.captureConfig);
  const replayFrame = useTrainingStore((s) => s.replayFrame);
  const replayLoading = useTrainingStore((s) => s.replayLoading);
  const version = useTrainingStore((s) => s.version);

  const [restoring, setRestoring] = useState(false);
  const fetchTimer = useRef<number | undefined>(undefined);

  const lost = status === 'lost'; // バックエンド再起動でセッション消失
  const stride = Math.max(1, historyMeta?.stride ?? 1);
  const minStep = historyMeta?.min_step ?? null;
  const maxStep = historyMeta?.max_step ?? null;
  const hasHistory = !lost && minStep !== null && maxStep !== null && maxStep >= minStep;

  // 履歴インデックスのポーリングはWS未接続時のみのフォールバック。
  // WS接続中はhistory_indexフレームのプッシュがhistoryMetaを更新する
  useEffect(() => {
    if (!sessionId) return;
    let disposed = false;
    const fetchIndex = async () => {
      try {
        const idx = await getTrainHistoryIndex(sessionId);
        if (!disposed) useTrainingStore.getState().setHistoryMeta(idx);
      } catch {
        // セッション消滅等は無視(lost検出はstatusポーリング/health側が担う)
      }
    };
    if (!useTrainingStore.getState().wsConnected) void fetchIndex();
    const timer = window.setInterval(() => {
      const st = useTrainingStore.getState();
      if (st.wsConnected || st.status === 'lost') return;
      // live中のみ自動更新(replay中はレンジが動くと操作しづらい)
      if (st.displayMode === 'live') void fetchIndex();
    }, INDEX_POLL_MS);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  // カーソル変更時にビューを取得(デバウンス)
  useEffect(() => {
    if (displayMode !== 'replay' || cursorStep === null || lost) return;
    if (fetchTimer.current !== undefined) window.clearTimeout(fetchTimer.current);
    fetchTimer.current = window.setTimeout(() => {
      void loadReplayStep(cursorStep);
    }, SCRUB_FETCH_DEBOUNCE_MS);
    return () => {
      if (fetchTimer.current !== undefined) window.clearTimeout(fetchTimer.current);
    };
  }, [displayMode, cursorStep, lost]);

  // 再生ループ(速度=steps/sec、方向±)
  useEffect(() => {
    if (!playback.playing || displayMode !== 'replay') return;
    const intervalMs = Math.max(1000 / (playback.speed / stride), 30);
    const timer = window.setInterval(() => {
      const st = useTrainingStore.getState();
      const cur = st.cursorStep;
      const meta = st.historyMeta;
      if (cur === null || !meta) return;
      const next = cur + playback.direction * stride;
      const lo = meta.min_step ?? 0;
      const hi = meta.max_step ?? cur;
      if (next < lo || next > hi) {
        st.setPlayback({ playing: false });
        return;
      }
      st.setCursor(next);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playback.playing, playback.direction, playback.speed, displayMode, stride]);

  if (!sessionId) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        学習セッションを開始するとタイムラインが表示されます
      </div>
    );
  }

  const store = useTrainingStore.getState();
  const liveStep = (() => {
    const steps = store.steps;
    return steps.length > 0 ? steps[steps.length - 1] : 0;
  })();
  void version; // liveStepの再計算トリガ

  const sliderValue = displayMode === 'replay' ? (cursorStep ?? liveStep) : liveStep;
  const paused = status === 'paused';
  const canBranch = paused && displayMode === 'replay' && cursorStep !== null;

  const enterAt = (step: number) => {
    store.setCursor(step);
  };

  const stepBy = (delta: number) => {
    const base = displayMode === 'replay' ? (cursorStep ?? liveStep) : liveStep;
    enterAt(base + delta * stride);
  };

  const handleBranch = async () => {
    if (!canBranch || cursorStep === null) return;
    setRestoring(true);
    try {
      const res = await trainRestore(sessionId, cursorStep);
      invalidateHistoryCache();
      store.exitReplay();
      const idx = await getTrainHistoryIndex(sessionId);
      store.setHistoryMeta(idx);
      toast.success(
        `step ${res.restored_step} へ巻き戻しました` +
          (res.optimizer_restored
            ? '(optimizer状態も復元)'
            : '(optimizerはリセット=モメンタム消失)'),
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setRestoring(false);
    }
  };

  return (
    <div className="flex h-full flex-col gap-2 p-3">
      <div className="flex items-center gap-1.5">
        <Button
          variant="ghost" size="sm" title="最初へ"
          disabled={!hasHistory}
          onClick={() => minStep !== null && enterAt(minStep)}
        >
          <ChevronFirst className="size-4" />
        </Button>
        <Button
          variant="ghost" size="sm" title={`${stride}ステップ戻る`}
          disabled={!hasHistory}
          onClick={() => stepBy(-1)}
        >
          <StepBack className="size-4" />
        </Button>
        {playback.playing ? (
          <Button
            variant="outline" size="sm" title="再生停止"
            onClick={() => store.setPlayback({ playing: false })}
          >
            <Pause className="size-4" />
          </Button>
        ) : (
          <>
            <Button
              variant="outline" size="sm" title="逆再生"
              disabled={!hasHistory}
              onClick={() => {
                if (displayMode !== 'replay') enterAt(sliderValue);
                store.setPlayback({ playing: true, direction: -1 });
              }}
            >
              <Play className="size-4 -scale-x-100" />
            </Button>
            <Button
              variant="outline" size="sm" title="再生"
              disabled={!hasHistory}
              onClick={() => {
                if (displayMode !== 'replay') {
                  enterAt(minStep ?? sliderValue);
                }
                store.setPlayback({ playing: true, direction: 1 });
              }}
            >
              <Play className="size-4" />
            </Button>
          </>
        )}
        <Button
          variant="ghost" size="sm" title={`${stride}ステップ進む`}
          disabled={!hasHistory}
          onClick={() => stepBy(1)}
        >
          <StepForward className="size-4" />
        </Button>
        <Button
          variant="ghost" size="sm" title="最後へ"
          disabled={!hasHistory}
          onClick={() => maxStep !== null && enterAt(maxStep)}
        >
          <ChevronLast className="size-4" />
        </Button>

        <select
          className="h-7 rounded border bg-background px-1 text-xs"
          value={playback.speed}
          title="再生速度(steps/sec)"
          onChange={(e) => store.setPlayback({ speed: Number(e.target.value) })}
        >
          {SPEED_OPTIONS.map((v) => (
            <option key={v} value={v}>
              {v} st/s
            </option>
          ))}
        </select>

        <div className="flex-1" />

        {/* スクラブ中でフレーム未着(キャッシュ未ヒット)の間だけ表示 */}
        {displayMode === 'replay' && replayLoading && (
          <Badge variant="outline" className="gap-1 text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            読込中…
          </Badge>
        )}
        {displayMode === 'replay' && (
          <>
            <Button
              size="sm"
              variant="outline"
              disabled={!canBranch || restoring || lost}
              title={
                paused
                  ? 'このstepへ巻き戻して再学習する(以降の履歴はbranches/へ退避)'
                  : 'branch再学習はpause中のみ可能です'
              }
              onClick={() => void handleBranch()}
            >
              <GitBranch className="size-4" />
              ここから再学習
            </Button>
            <Button size="sm" onClick={() => store.exitReplay()} title="最新へ追従復帰">
              <Radio className="size-4" />
              LIVEへ戻る
            </Button>
          </>
        )}
        {displayMode === 'live' && !lost && (
          <Badge variant="outline" className="text-[var(--status-ok)]">
            LIVE
          </Badge>
        )}
        {lost && (
          <Badge variant="outline" className="text-destructive border-destructive/40">
            セッション消失
          </Badge>
        )}
      </div>

      {/* シークバー+keyframe目盛 */}
      <div className="relative px-1">
        <input
          type="range"
          className="w-full accent-[var(--primary)]"
          min={minStep ?? 0}
          max={maxStep ?? 0}
          step={stride}
          value={sliderValue}
          disabled={!hasHistory}
          onChange={(e) => enterAt(Number(e.target.value))}
        />
        {hasHistory &&
          (historyMeta?.keyframes ?? []).map((kf) => {
            const range = (maxStep ?? 1) - (minStep ?? 0);
            if (range <= 0) return null;
            const pct = ((kf - (minStep ?? 0)) / range) * 100;
            return (
              <div
                key={kf}
                className="pointer-events-none absolute top-full h-1.5 w-px bg-[var(--slot-2)]"
                style={{ left: `${pct}%` }}
                title={`keyframe: step ${kf}`}
              />
            );
          })}
        {/* エポック境界目盛。metricsのepoch遷移stepをtrainingStoreが記録する */}
        {hasHistory &&
          store.epochMarks.map((s, i) => {
            const range = (maxStep ?? 1) - (minStep ?? 0);
            if (range <= 0 || s < (minStep ?? 0) || s > (maxStep ?? 0)) return null;
            const pct = ((s - (minStep ?? 0)) / range) * 100;
            return (
              <div
                key={`ep-${s}`}
                className="pointer-events-none absolute top-full h-2 w-px bg-[var(--status-warn)]"
                style={{ left: `${pct}%` }}
                title={`epoch ${i + 1} 開始: step ${s}`}
              />
            );
          })}
        {/* val_accマーカー: hard eval実施stepに小さい点を打つ */}
        {hasHistory &&
          store.valAcc.map(([s, v]) => {
            const range = (maxStep ?? 1) - (minStep ?? 0);
            if (range <= 0 || s < (minStep ?? 0) || s > (maxStep ?? 0)) return null;
            const pct = ((s - (minStep ?? 0)) / range) * 100;
            return (
              <div
                key={`val-${s}`}
                className="pointer-events-none absolute top-full mt-2 size-1.5 -translate-x-1/2 rounded-full bg-[var(--chart-acc)]"
                style={{ left: `${pct}%` }}
                title={`val_acc ${(v * 100).toFixed(1)}% @ step ${s}`}
              />
            );
          })}
      </div>

      <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
        <span>
          step <b className="text-foreground">{sliderValue}</b>
          {maxStep !== null && ` / ${maxStep}`}
          {stride > 1 && ` (間引き 1/${stride})`}
        </span>
        {displayMode === 'replay' && replayFrame && (
          <span>
            loss {replayFrame.metrics.loss.toFixed(4)} ・ acc{' '}
            {(replayFrame.metrics.acc * 100).toFixed(1)}%
          </span>
        )}
        {captureConfig && (
          <span>
            capture: {captureConfig.level}
            {captureConfig.level === 'metrics' &&
              '(モデルが大きいため重み履歴は保存されません)'}
          </span>
        )}
        {lost && (
          <span className="text-destructive">
            バックエンドが再起動されたため、このセッションのタイムラインは利用できません
          </span>
        )}
        {!hasHistory && !lost && <span>履歴がまだありません(学習を開始してください)</span>}
        {ACTIVE_STATES.includes(status) && !paused && displayMode === 'replay' && (
          <span className="text-[var(--status-warn)]">
            学習は進行中です(branchするにはpauseしてください)
          </span>
        )}
      </div>
    </div>
  );
}
