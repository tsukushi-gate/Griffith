// @vitest-environment jsdom
// 学習WSの接続先URL導出の回帰テスト。
//
// 退行の再発防止(root cause): 旧実装はフロントが 5173 のとき WS を
// ws://127.0.0.1:8765 へ「ポート埋め込み直結」していた。バックエンドが 8765 以外で
// 動く/フロントが 5173 以外で動く構成だと WS だけ別バックエンドへ逸れ、REST(/api,
// タイムライン)は進むのに学習曲線(WS)が step 0 のまま・statusが starting のまま、
// という症状になっていた。修正後は常に「同一オリジン」(dev では vite の /ws proxy
// 経由でRESTと同じバックエンドへ)に張る。ここではその不変条件を固定する。

import { afterEach, describe, expect, it, vi } from 'vitest';
import { wsUrl } from './useTrainingSocket';

/** window.location を任意のorigin/portへ差し替える。 */
function setLocation(href: string): void {
  const url = new URL(href);
  vi.stubGlobal('window', {
    ...window,
    location: {
      href: url.href,
      protocol: url.protocol,
      host: url.host,
      hostname: url.hostname,
      port: url.port,
    },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('wsUrl', () => {
  it('5173でもポートを埋め込まず、同一オリジン(=RESTと同じproxy)へ張る', () => {
    setLocation('http://localhost:5173/');
    // 旧実装の退行: ws://127.0.0.1:8765 に逸れていた。
    expect(wsUrl('abc123')).toBe('ws://localhost:5173/ws/train/abc123');
  });

  it('5173以外のポートでも同一オリジンへ張る', () => {
    setLocation('http://localhost:5174/');
    expect(wsUrl('sid')).toBe('ws://localhost:5174/ws/train/sid');
  });

  it('https配信ではwssへ切り替える', () => {
    setLocation('https://editor.example.com/');
    expect(wsUrl('s1')).toBe('wss://editor.example.com/ws/train/s1');
  });

  it('VITE_EDITOR_WS_BASE_URL指定時のみ明示的に直結する(opt-in)', () => {
    setLocation('http://localhost:5173/');
    vi.stubEnv('VITE_EDITOR_WS_BASE_URL', 'ws://10.0.0.5:9000/');
    expect(wsUrl('s2')).toBe('ws://10.0.0.5:9000/ws/train/s2');
  });
});
