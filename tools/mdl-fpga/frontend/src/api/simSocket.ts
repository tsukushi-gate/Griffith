// /ws/sim クライアント。latest-wins・1 インフライト(editor の学習 WS の教訓):
// 応答待ち中の新コマンドは上書き保留し、フレーム到着時に 1 件だけ送る。
// RUN はフレーム到着駆動(応答が返ってから次の step を送る)ので、
// プローブ評価や PNG 生成が遅くても WS が詰まらない。
import { useSimStore, type SimCommand } from '../store/simStore';
import type { SimFrame } from './types';

let ws: WebSocket | null = null;
let inFlight = false;
let pending: SimCommand | null = null;
let silentClose = false;

function rawSend(cmd: SimCommand) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  inFlight = true;
  ws.send(JSON.stringify(cmd));
}

function send(cmd: SimCommand) {
  if (inFlight) {
    pending = cmd; // latest-wins(古い保留は捨てる)
  } else {
    rawSend(cmd);
  }
}

export function closeSimSocket(opts?: { silent?: boolean }): void {
  // プロジェクト切替時の意図的クローズ(silent=切断トーストを出さない)
  if (opts?.silent) silentClose = true;
  const cur = ws;
  ws = null;
  inFlight = false;
  pending = null;
  try {
    cur?.close();
  } catch {
    /* 既に閉じている */
  }
}

export function connectSimSocket(): void {
  useSimStore.setState({ send });

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/sim`);

  ws.onopen = () => useSimStore.getState().setConnected(true);
  ws.onclose = () => {
    const wasConnected = useSimStore.getState().connected;
    useSimStore.getState().setConnected(false);
    inFlight = false;
    pending = null;
    if (silentClose) {
      silentClose = false;
      return;
    }
    if (wasConnected) {
      void (async () => {
        const { toast } = await import('sonner');
        const { translate } = await import('../i18n/messages');
        const { useLanguageStore } = await import('../i18n/useT');
        toast.error(translate('toast.disconnected',
                              useLanguageStore.getState().language));
      })();
    }
  };
  ws.onmessage = (ev) => {
    const frame = JSON.parse(ev.data as string) as
      SimFrame | { type: 'session_changed'; epoch: number };
    if (frame.type === 'session_changed') {
      // 他タブがプロジェクトを切り替えた → こちらも全再取得(バックストップ)
      silentClose = true;
      void import('../boot/bootApp').then((m) => m.rehydrateApp());
      return;
    }
    if (frame.type !== 'values') return;
    inFlight = false;
    useSimStore.getState().setFrame(frame);
    if (pending) {
      const next = pending;
      pending = null;
      rawSend(next);
    } else if (useSimStore.getState().running) {
      rawSend({ cmd: 'step', n: 1 });
    }
  };
}
