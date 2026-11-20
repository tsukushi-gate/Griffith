// 学習実行コントロール(ヘッダ右に常駐): 開始/一時停止/再開/停止 + ステータス。
// status 404によるセッション消失検出(lost遷移)。
// §2: dataset/loss/optimizer/method等の設定は下部ドック「学習」タブの常設フォーム
//   (TrainSettingsForm)へ移動した。ここには実行操作とステータス表示だけを残す。
// チャート表示は下部ドックの TrainCharts が担当する。

import { Pause, Play, Square, StepForward } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import {
  ApiError,
  buildModel,
  getTrainStatus,
  startTraining,
  trainControl,
  trainStepN,
} from '../api/rest';
import { declareSessionLost } from '../api/useTrainingSocket';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useGraphStore } from '../store/graphStore';
import { ACTIVE_STATES, useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';
import { statusVariant } from './statusVariant';
import {
  trainingDisabledReason,
  useResolvedTrainConfig,
} from './useResolvedTrainConfig';

const STATUS_POLL_MS = 2000;

export function TrainControls() {
  const sessionId = useTrainingStore((s) => s.sessionId);
  const status = useTrainingStore((s) => s.status);
  const wsConnected = useTrainingStore((s) => s.wsConnected);
  const trainConfig = useTrainingStore((s) => s.trainConfig);
  const totalSteps = useTrainingStore((s) => s.totalSteps);
  const metricsVersion = useTrainingStore((s) => s.version);

  const { shapeErrors, selectedDataset, loss, method, disabledReason } =
    useResolvedTrainConfig();

  const [busy, setBusy] = useState(false);

  const active = ACTIVE_STATES.includes(status);
  const lost = status === 'lost';
  const { lr, logicLr, annLr, batchSize, epochs } = trainConfig;

  const liveSteps = useTrainingStore.getState().steps;
  const latestStep =
    metricsVersion > -1 && liveSteps.length > 0 ? liveSteps[liveSteps.length - 1] : 0;
  const progressLabel =
    totalSteps != null
      ? `${Math.min(latestStep, totalSteps)}/${totalSteps} step`
      : latestStep > 0
        ? `${latestStep} step`
        : null;
  const progressValue =
    totalSteps != null && totalSteps > 0
      ? Math.min(100, Math.max(0, (latestStep / totalSteps) * 100))
      : null;

  // WS切断時のフォールバックとして状態をポーリング
  useEffect(() => {
    if (!sessionId || !active) return;
    const timer = window.setInterval(async () => {
      try {
        const st = await getTrainStatus(sessionId);
        const store = useTrainingStore.getState();
        store.setStatus(st.state);
        store.setTotalSteps(st.total_steps ?? null);
      } catch (e) {
        // 404=バックエンド再起動でセッション消失。それ以外はWS再接続に任せる
        if (e instanceof ApiError && e.status === 404) declareSessionLost();
      }
    }, STATUS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [sessionId, active]);

  const handleStart = async () => {
    const reason = trainingDisabledReason(
      useGraphStore.getState().ir,
      shapeErrors,
      selectedDataset,
    );
    if (reason) {
      toast.error(reason);
      return;
    }
    setBusy(true);
    try {
      // 未同期編集をflushし、構造編集後はbuildでdraft確定+ノードid同期
      const graph = useGraphStore.getState();
      await graph.syncNow();
      if (useGraphStore.getState().syncError) {
        throw new Error(useGraphStore.getState().syncError ?? '同期失敗');
      }
      if (useGraphStore.getState().needsBuild) {
        const built = await buildModel();
        graph.setIR(built.ir, { needsBuild: false });
      }
      const logicLrValue = logicLr.trim() ? Number(logicLr) : undefined;
      const annLrValue = annLr.trim() ? Number(annLr) : undefined;
      const res = await startTraining({
        dataset: selectedDataset!.name,
        lr: Number(lr),
        ...(logicLrValue !== undefined ? { logic_lr: logicLrValue } : {}),
        ...(annLrValue !== undefined ? { ann_lr: annLrValue } : {}),
        batch_size: Number(batchSize),
        epochs: Number(epochs),
        method,
        loss,
        optimizer: trainConfig.optimizer,
        ...(trainConfig.wandbEnabled ? { wandb: true } : {}),
      });
      // start成功後にのみ旧stateを消す。失敗時(409/400/network)は既存チャート・状態を
      // 保持し、catchのtoastでエラーを見せる(「押したのに黙って消える」を防ぐ)
      const store = useTrainingStore.getState();
      store.resetAll();
      store.setSession(res.session_id);
      store.setStatus(res.state);
      useUIStore.getState().openBottomTab('train');
    } catch (e) {
      // バリデーションエラー(400)はバックエンドの日本語detailをそのまま表示する
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleControl = async (action: 'pause' | 'resume' | 'stop') => {
    if (!sessionId) return;
    setBusy(true);
    try {
      // 仕様: replay進入はタイムライン操作のみ。resumeはliveへ復帰する
      if (action === 'resume') useTrainingStore.getState().exitReplay();
      const st = await trainControl(sessionId, action);
      useTrainingStore.getState().setStatus(st.state);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // paused中のコマ送り: 1ステップだけ進めて自動pause
  const handleStepOnce = async () => {
    if (!sessionId) return;
    setBusy(true);
    try {
      // 仕様: 学習step実行時はliveへ復帰してstepの進みを見せる
      useTrainingStore.getState().exitReplay();
      const st = await trainStepN(sessionId, 1);
      useTrainingStore.getState().setStatus(st.state);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-1.5">
      {/* 学習開始は常時押下可能: 押すたびに新しいrunを開始しフォーカスする */}
      <Button
        size="sm"
        disabled={busy || disabledReason !== null}
        title={disabledReason ?? `${selectedDataset?.label ?? ''}で学習開始`}
        onClick={() => void handleStart()}
      >
        <Play className="size-4" />
        学習開始
      </Button>
      {/* フォーカス中runがアクティブな間だけ一時停止/再開/コマ送り/停止を出す */}
      {active && (
        <>
          {status === 'running' || status === 'starting' ? (
            <Button
              size="sm"
              variant="outline"
              disabled={busy || status !== 'running'}
              onClick={() => void handleControl('pause')}
              title="一時停止"
            >
              <Pause className="size-4" />
            </Button>
          ) : (
            <>
              <Button
                size="sm"
                variant="outline"
                disabled={busy || status !== 'paused'}
                onClick={() => void handleControl('resume')}
                title="再開"
              >
                <Play className="size-4" />
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy || status !== 'paused'}
                onClick={() => void handleStepOnce()}
                title="1ステップ実行(コマ送り)"
              >
                <StepForward className="size-4" />
              </Button>
            </>
          )}
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => void handleControl('stop')}
            title="停止"
          >
            <Square className="size-4" />
          </Button>
        </>
      )}
      <Badge
        variant="outline"
        className={statusVariant(status)}
        title={
          lost
            ? 'バックエンドが再起動されました。新しい学習セッションを開始してください'
            : (progressLabel ?? undefined)
        }
      >
        {status}
      </Badge>
      {progressLabel && (
        <span className="min-w-16 text-right text-[11px] tabular-nums text-muted-foreground">
          {progressLabel}
        </span>
      )}
      {progressValue !== null && (
        <Progress value={progressValue} className="h-1.5 w-20 bg-muted" />
      )}
      <span
        className={`text-[11px] ${wsConnected ? 'text-primary' : 'text-muted-foreground'}`}
        title={wsConnected ? '学習WebSocket接続中' : '学習WebSocket未接続'}
      >
        WS
      </span>
    </div>
  );
}
