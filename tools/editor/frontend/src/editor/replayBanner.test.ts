// リプレイバナー(§4-4)の純関数テスト: 表示可否と文言。

import { describe, expect, it } from 'vitest';
import { buildReplayBanner } from './replayBanner';

describe('buildReplayBanner', () => {
  it('liveモードは非表示', () => {
    expect(buildReplayBanner('live', 100, 'running')).toEqual({
      visible: false,
      text: '',
    });
  });

  it('replayでもカーソル未設定なら非表示', () => {
    expect(buildReplayBanner('replay', null, 'paused').visible).toBe(false);
  });

  it('running中は「学習は継続中」注記', () => {
    const b = buildReplayBanner('replay', 500, 'running');
    expect(b.visible).toBe(true);
    expect(b.text).toBe('⏪ step 500 を表示中(学習は継続中)');
  });

  it('running以外は「一時停止中」注記', () => {
    expect(buildReplayBanner('replay', 42, 'paused').text).toBe(
      '⏪ step 42 を表示中(一時停止中)',
    );
    expect(buildReplayBanner('replay', 42, 'stopped').text).toBe(
      '⏪ step 42 を表示中(一時停止中)',
    );
  });
});
