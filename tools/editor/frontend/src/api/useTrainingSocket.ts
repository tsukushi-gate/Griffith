// 訓練WebSocket接続hook。
// metricsはバッファに溜めて10Hzでストアへflushする(毎ステップの再描画を避ける)。
// 訓練がアクティブな間の切断は1秒間隔で再接続する。
// subscribe v2でdetail(gate_distribution/activations)も購読し、
// capture_config/新フレームをストアへdispatchする。
// /api/health の server_epoch 変化(=バックエンド再起動)を検出したら
//   status 'lost' へ遷移し、WS再接続を打ち切る。
// history_indexフレームを受信してhistoryMetaを更新する(ポーリング代替)。
// 層展開(uiStore.expandedLayers)やbuildによるlutLayerIds変化時に
//   接続中WSへsubscribeを再送する。

import { useEffect } from 'react';
import { toast } from 'sonner';
import { DETAIL_MANUAL_MAX } from '../detail/detailConfig';
import { detailLayerIds, lutLayerIds, useGraphStore } from '../store/graphStore';
import { useUIStore } from '../store/uiStore';
import { getHealth, type HistoryIndex } from './rest';
import {
  ACTIVE_STATES,
  useTrainingStore,
  type ActivationsFrame,
  type ApDeltaFrame,
  type CaptureConfigFrame,
  type GateDistributionFrame,
  type GradFlowFrame,
  type HistoryIndexFrame,
  type LayerSummaryFrame,
  type MetricsFrame,
} from '../store/trainingStore';

const RECONNECT_DELAY_MS = 1000;
const FLUSH_INTERVAL_MS = 100;
const INITIAL_CONNECT_DELAY_MS = 0;
const HEALTH_CHECK_INTERVAL_MS = 30_000; // server_epoch定期確認の間隔

function trimTrailingSlash(url: string): string {
  return url.replace(/\/+$/, '');
}

/**
 * 学習WSの接続先URLを決める。
 *
 * 既定は必ず「同一オリジン」へ張る(dev では vite の /ws proxy 経由でバックエンドへ、
 * 本番では配信元と同一ホストへ)。これにより WS の接続先は REST(/api)と同じ
 * バックエンドに追従する。ポートを埋め込んでいた旧実装は、フロントが 5173 以外で
 * 起動した/バックエンドが 8765 以外で動く構成のとき WS だけ別ポート(8765)へ逸れ、
 * 「タイムライン(REST)は進むのに学習曲線(WS)が step 0 のまま・statusが starting のまま」
 * という退行を招いていた。バックエンドを別ホストで直結したい場合のみ、明示的な
 * VITE_EDITOR_WS_BASE_URL でopt-inする。
 */
export function wsUrl(sessionId: string): string {
  const configuredBaseUrl = import.meta.env.VITE_EDITOR_WS_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return `${trimTrailingSlash(configuredBaseUrl)}/ws/train/${sessionId}`;
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/train/${sessionId}`;
}

type WsFrame =
  | MetricsFrame
  | LayerSummaryFrame
  | GateDistributionFrame
  | ActivationsFrame
  | GradFlowFrame
  | ApDeltaFrame
  | CaptureConfigFrame
  | HistoryIndexFrame
  | { type: 'session_end'; state?: string }
  | { type: 'error'; message?: string }
  | { type?: string };

/**
 * バックエンド再起動によるセッション消失を宣言する。
 * TrainControlsのstatus 404検出からも呼ばれる。二重toastは抑止する。
 */
export function declareSessionLost(): void {
  const store = useTrainingStore.getState();
  if (store.status === 'lost') return;
  store.markLost();
  toast.error(
    'バックエンドが再起動されました。新しい学習セッションを開始してください',
  );
}

/**
 * /api/health のserver_epochを照合する。
 * 変化していたらlost宣言してtrue、変化なし/確認不能はfalseを返す。
 */
async function checkServerEpochChanged(): Promise<boolean> {
  try {
    const h = await getHealth();
    const store = useTrainingStore.getState();
    const known = store.serverEpoch;
    store.setServerEpoch(h.server_epoch);
    if (known !== null && h.server_epoch !== known) {
      declareSessionLost();
      return true;
    }
  } catch {
    // バックエンド停止中はエポック不明のまま(WS再接続に委ねる)
  }
  return false;
}

/** 現行subscribe v2形式の購読内容。*/
function buildSubscribeMessage(): string {
  const method = useTrainingStore.getState().captureConfig?.method;
  return JSON.stringify({
    type: 'subscribe',
    layers: lutLayerIds(),
    detail: {
      gate_distribution: detailLayerIds(DETAIL_MANUAL_MAX),
      grad_flow: true,
      activations: true,
      // AP法のみ更新量delta(.grad非経由)を購読する
      ap_delta: method === 'ap',
    },
    detail_interval_ms: 500,
  });
}

export function useTrainingSocket(): void {
  const sessionId = useTrainingStore((s) => s.sessionId);

  useEffect(() => {
    if (!sessionId) return;

    let ws: WebSocket | null = null;
    let disposed = false;
    let retryTimer: number | undefined;
    let lastSubscribeSent = '';
    const buffer: MetricsFrame[] = [];

    const flushTimer = window.setInterval(() => {
      if (buffer.length > 0) {
        useTrainingStore.getState().appendMetrics(buffer.splice(0, buffer.length));
      }
    }, FLUSH_INTERVAL_MS);

    const scheduleConnect = (delayMs: number) => {
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      retryTimer = window.setTimeout(() => {
        retryTimer = undefined;
        if (!disposed) connect();
      }, delayMs);
    };

    // 内容が前回送信時から変わった時だけsubscribeを再送する
    const resubscribe = () => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const msg = buildSubscribeMessage();
      if (msg === lastSubscribeSent) return;
      ws.send(msg);
      lastSubscribeSent = msg;
    };

    const connect = () => {
      const socket = new WebSocket(wsUrl(sessionId));
      ws = socket;

      socket.onopen = () => {
        useTrainingStore.getState().setWsConnected(true);
        // キャンバス上の全lut_layerを購読(ノードのミニバー+ヒートマップ用)。
        // detail(確率分布/活性)は帯域制御のため小規模層のみ購読する
        lastSubscribeSent = buildSubscribeMessage();
        socket.send(lastSubscribeSent);
      };

      socket.onmessage = (ev) => {
        let msg: WsFrame;
        try {
          msg = JSON.parse(ev.data as string) as WsFrame;
        } catch {
          return;
        }
        const store = useTrainingStore.getState();
        if (msg.type === 'metrics') {
          buffer.push(msg as MetricsFrame);
        } else if (msg.type === 'layer_summary') {
          store.setSummary(msg as LayerSummaryFrame);
        } else if (
          msg.type === 'gate_distribution' ||
          msg.type === 'activations' ||
          msg.type === 'grad_flow' ||
          msg.type === 'ap_delta'
        ) {
          store.applyDetailFrame(
            msg as
              | GateDistributionFrame
              | ActivationsFrame
              | GradFlowFrame
              | ApDeltaFrame,
          );
        } else if (msg.type === 'capture_config') {
          store.setCaptureConfig(msg as CaptureConfigFrame);
          // methodが確定したらap_delta購読を反映するため再送
          resubscribe();
        } else if (msg.type === 'history_index') {
          // タイムラインのポーリング代替。replay操作中はレンジを
          // 動かさない(TimelineScrubberのポーリングと同じ方針)
          if (store.displayMode === 'live') {
            const f = msg as HistoryIndexFrame;
            const meta: HistoryIndex = {
              min_step: f.min_step,
              max_step: f.max_step,
              stride: f.stride,
              keyframes: f.keyframes,
              count: f.count,
              ram_bytes: f.ram_bytes,
              disk_bytes: f.disk_bytes,
              capture: f.capture,
            };
            store.setHistoryMeta(meta);
          }
        } else if (msg.type === 'session_end') {
          store.setStatus((msg as { state?: string }).state ?? 'finished');
        } else if (msg.type === 'error') {
          store.setStatus('error');
          toast.error((msg as { message?: string }).message ?? '学習中にエラーが発生しました');
        }
      };

      socket.onclose = () => {
        useTrainingStore.getState().setWsConnected(false);
        const active = ACTIVE_STATES.includes(useTrainingStore.getState().status);
        if (disposed || !active) return;
        // 切断時はまずserver_epochを照合し、再起動なら再接続を打ち切る
        void checkServerEpochChanged().then((lost) => {
          if (!lost && !disposed) scheduleConnect(RECONNECT_DELAY_MS);
        });
      };
    };

    // server_epochの基準値を取得(以降の変化=再起動として検出する)
    void checkServerEpochChanged();
    const healthTimer = window.setInterval(() => {
      if (!ACTIVE_STATES.includes(useTrainingStore.getState().status)) return;
      void checkServerEpochChanged().then((lost) => {
        if (lost) ws?.close();
      });
    }, HEALTH_CHECK_INTERVAL_MS);

    // 層展開・build等による購読対象の変化を監視して再購読する
    const unsubUI = useUIStore.subscribe((s, prev) => {
      if (s.expandedLayers !== prev.expandedLayers) resubscribe();
    });
    const unsubGraph = useGraphStore.subscribe((s, prev) => {
      if (s.ir !== prev.ir || s.nodes !== prev.nodes) resubscribe();
    });

    scheduleConnect(INITIAL_CONNECT_DELAY_MS);

    return () => {
      disposed = true;
      window.clearInterval(flushTimer);
      window.clearInterval(healthTimer);
      unsubUI();
      unsubGraph();
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      ws?.close();
    };
  }, [sessionId]);
}
