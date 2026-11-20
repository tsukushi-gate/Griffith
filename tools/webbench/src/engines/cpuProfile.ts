// CPUエンジン(bitplane / bitplane-wasm)のフレーム内訳プロファイラ。
//
// webgpuGraph.ts の setProfiling/getBreakdownに対応する CPU 側の
// 計測口。C5完了時点で conv2d は wasm フレームの1割弱しか占めておらず、
// 残り9割の内訳(非convノード / JS↔wasmヒープ往復 / 入力パック / アンパック)
// を数値で確定させないと、どのレバー(区間化 / threads)に投資すべきか
// 判断できない。そのための計測専用モジュール。
//
// 設計:
//   - **既定は完全に無効**(`enabled` が false のとき addCpuSample は即return、
//     呼び出し側は `cpuProfilingEnabled()` の1分岐だけを踏む)。
//   - フレーム単位でラベル別の累積msを溜め、フレーム配列として保持する。
//     ウォームアップの影響を捨てるため getCpuBreakdown() は先頭 skipFrames 本を
//     除いた **中央値** を返す(autobench の median と同じ流儀)。
//   - autobench.ts / compareDet.ts を改造せずに使えるよう、ブラウザでは
//     `?cpuprofile=1` で自動的に有効化し `window.__cpuProfile` を生やす
//     (Playwright 側は BENCH_DONE 後に window.__cpuProfile.get() を読むだけ)。
//
// ラベルの入れ子(合計しても frame.total にはならない点に注意):
//   frame.total                 evalIrBitplane 全体
//   ├ node.<type>               ノード種別ごとの実行時間(全ノードの合計)
//   │  └ conv.copyIn/wasm/copyOut   node.conv2d の内訳(JS↔wasmヒープ往復)
//   ├ pack.input                グラフ入力ポートの packPlanes
//   ├ pack.boundary             非ネイティブノード出力の再 packPlanes
//   ├ unpack                    ネイティブ→非ネイティブ境界の unpackPlanes
//   └ out.toNodeValue           最終出力の実体化

export interface CpuLabelStat {
  label: string;
  /** フレーム中央値(ms)。 */
  medianMs: number;
  /** フレーム平均(ms)。 */
  meanMs: number;
  /** 1フレームあたりの呼び出し回数(中央値フレームではなく全体平均)。 */
  callsPerFrame: number;
}

export interface CpuBreakdown {
  /** 集計に使ったフレーム数(skipFrames 除外後)。 */
  frames: number;
  /** frame.total のフレーム中央値(ms)。 */
  totalMedianMs: number;
  /** 降順(medianMs)のラベル別内訳。 */
  labels: CpuLabelStat[];
}

interface FrameRecord {
  ms: Map<string, number>;
  calls: Map<string, number>;
}

let enabled = false;
let cur: FrameRecord | null = null;
const records: FrameRecord[] = [];

/** 1フレームの上限(暴走防止)。50〜100フレーム計測を想定。 */
const MAX_FRAMES = 4096;

export function setCpuProfiling(on: boolean): void {
  enabled = on;
  cur = null;
}

export function cpuProfilingEnabled(): boolean {
  return enabled;
}

export function resetCpuProfile(): void {
  records.length = 0;
  cur = null;
}

/** evalIrBitplane の入口で呼ぶ。 */
export function beginCpuFrame(): void {
  if (!enabled) return;
  cur = { ms: new Map(), calls: new Map() };
}

/** evalIrBitplane の出口で呼ぶ(totalMs は frame.total として記録)。 */
export function endCpuFrame(totalMs: number): void {
  if (!enabled || !cur) return;
  cur.ms.set('frame.total', totalMs);
  cur.calls.set('frame.total', 1);
  if (records.length < MAX_FRAMES) records.push(cur);
  cur = null;
}

export function addCpuSample(label: string, ms: number): void {
  if (!enabled || !cur) return;
  cur.ms.set(label, (cur.ms.get(label) ?? 0) + ms);
  cur.calls.set(label, (cur.calls.get(label) ?? 0) + 1);
}

/** 計測対象を関数で包む糖衣(profiling無効時は分岐1回+直接呼び出し)。 */
export function timed<T>(label: string, fn: () => T): T {
  if (!enabled || !cur) return fn();
  const t0 = performance.now();
  const r = fn();
  addCpuSample(label, performance.now() - t0);
  return r;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** 先頭 skipFrames 本(ウォームアップ)を捨てたフレーム中央値の内訳を返す。
    1フレームも記録が無ければ null。 */
export function getCpuBreakdown(skipFrames = 8): CpuBreakdown | null {
  const used = records.length > skipFrames ? records.slice(skipFrames) : records;
  if (used.length === 0) return null;
  const labels = new Set<string>();
  for (const r of used) for (const k of r.ms.keys()) labels.add(k);
  const stats: CpuLabelStat[] = [];
  for (const label of labels) {
    const per = used.map((r) => r.ms.get(label) ?? 0);
    const calls = used.reduce((a, r) => a + (r.calls.get(label) ?? 0), 0);
    stats.push({
      label,
      medianMs: median(per),
      meanMs: per.reduce((a, b) => a + b, 0) / per.length,
      callsPerFrame: calls / used.length,
    });
  }
  stats.sort((a, b) => b.medianMs - a.medianMs);
  return {
    frames: used.length,
    totalMedianMs: stats.find((s) => s.label === 'frame.total')?.medianMs ?? 0,
    labels: stats,
  };
}

/** ブラウザで `?cpuprofile=1` が付いていれば自動的に有効化し、
    Playwright から読める窓口 `window.__cpuProfile` を生やす
    (autobench.ts / 各デモの main を一切改造しないための仕掛け)。 */
function initFromUrl(): void {
  if (typeof window === 'undefined') return;
  try {
    if (new URL(window.location.href).searchParams.get('cpuprofile') !== '1') return;
  } catch {
    return;
  }
  enabled = true;
  (window as unknown as { __cpuProfile: unknown }).__cpuProfile = {
    get: (skip = 8) => getCpuBreakdown(skip),
    reset: resetCpuProfile,
    setEnabled: setCpuProfiling,
  };
}

initFromUrl();
