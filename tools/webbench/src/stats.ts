// 計測統計ユーティリティ(WB-6)。
export function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function summarize(samples: number[]): {
  median: number; p95: number; min: number; max: number; n: number;
} {
  const s = [...samples].sort((a, b) => a - b);
  return {
    median: quantile(s, 0.5),
    p95: quantile(s, 0.95),
    min: s[0] ?? NaN,
    max: s[s.length - 1] ?? NaN,
    n: s.length,
  };
}

/** イベントループへ制御を返す(GC/描画の割込み点)。
    setTimeoutは非表示タブでスロットリングされる(分オーダーまで遅延)ため、
    スロットリング対象外のMessageChannelでyieldする。 */
export function tick(): Promise<void> {
  if (typeof MessageChannel === 'undefined') {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => {
      ch.port1.close();
      resolve();
    };
    ch.port2.postMessage(null);
  });
}
