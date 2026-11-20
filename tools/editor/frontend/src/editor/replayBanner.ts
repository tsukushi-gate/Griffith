// 履歴リプレイ(タイムマシン)視認性(§4-4)。
// バナー表示可否と文言の組み立てを純関数化して、FlowCanvasは表示に専念する。

export interface ReplayBanner {
  visible: boolean;
  text: string;
}

/** 学習が進行中(running)か否かで注記を切替える。 */
function trainingNote(status: string): string {
  return status === 'running' ? '学習は継続中' : '一時停止中';
}

/** replayモード時かつカーソルstepありのときのみvisible。 */
export function buildReplayBanner(
  displayMode: 'live' | 'replay',
  cursorStep: number | null,
  status: string,
): ReplayBanner {
  if (displayMode !== 'replay' || cursorStep == null) {
    return { visible: false, text: '' };
  }
  return {
    visible: true,
    text: `⏪ step ${cursorStep} を表示中(${trainingNote(status)})`,
  };
}
