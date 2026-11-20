// 履歴取得クライアント。
// LRUキャッシュ+in-flight重複排除+前後プリフェッチで、スクラブ操作を
// キャッシュヒット時ほぼ即応にする。
// フレーム未着時も直前のreplayFrameは保持し(クリアしない)、
// キャッシュ未ヒットの間だけ replayLoading を立てて「読込中…」を表示する。

import {
  getTrainHistoryStep,
  type HistoryStepView,
} from './rest';
import { lutLayerIds } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';

const CACHE_MAX = 256;
const PREFETCH_RADIUS = 8; // カーソル前後にプリフェッチするcaptured step数

const FIELDS = ['metrics', 'summary', 'gate_distribution', 'trace'];

const cache = new Map<string, HistoryStepView>(); // 挿入順でLRU管理
const inflight = new Map<string, Promise<HistoryStepView | null>>();

function cacheKey(sid: string, step: number): string {
  return `${sid}:${step}`;
}

function cachePut(key: string, view: HistoryStepView): void {
  if (cache.has(key)) cache.delete(key);
  cache.set(key, view);
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

async function fetchStep(sid: string, step: number): Promise<HistoryStepView | null> {
  const key = cacheKey(sid, step);
  const hit = cache.get(key);
  if (hit) return hit;
  const pending = inflight.get(key);
  if (pending) return pending;

  const p = getTrainHistoryStep(sid, step, { layers: lutLayerIds(), fields: FIELDS })
    .then((view) => {
      cachePut(key, view);
      // 丸め込み先のstepでも引けるように二重登録
      if (view.resolved_step !== step) {
        cachePut(cacheKey(sid, view.resolved_step), view);
      }
      return view;
    })
    .catch(() => null)
    .finally(() => {
      inflight.delete(key);
    });
  inflight.set(key, p);
  return p;
}

/** 指定stepのビューがキャッシュ済みか(読込中インジケータの判定用)。*/
export function isStepCached(sid: string, step: number): boolean {
  return cache.has(cacheKey(sid, step));
}

/**
 * カーソルstepのビューを取得してストアへ反映し、前後をプリフェッチする。
 * 呼び出し時点よりカーソルが移動していたら反映をスキップする(古い応答の上書き防止)。
 * 未着の間は直前のreplayFrameを保持する(キャンバスを空にしない)。
 */
export async function loadReplayStep(step: number): Promise<void> {
  const st = useTrainingStore.getState();
  const sid = st.sessionId;
  if (!sid) return;

  // キャッシュ未ヒット時のみ「読込中…」を立てる(ヒット時はほぼ即応)
  if (!isStepCached(sid, step)) st.setReplayLoading(true);

  const view = await fetchStep(sid, step);
  const now = useTrainingStore.getState();
  if (now.displayMode === 'replay' && now.cursorStep === step) {
    if (view) now.applyReplayFrame(view);
    // 取得完了(失敗含む)でカーソルが動いていなければ読込中を解除する。
    // カーソルが先へ動いていた場合は、最新カーソルのloadReplayStepが管理する
    now.setReplayLoading(false);
  }

  // 前後プリフェッチ(fire-and-forget、captured stride間隔)
  const stride = Math.max(1, now.historyMeta?.stride ?? 1);
  const meta = now.historyMeta;
  for (let i = 1; i <= PREFETCH_RADIUS; i++) {
    for (const dir of [1, -1] as const) {
      const s = step + dir * i * stride;
      if (meta?.min_step != null && s < meta.min_step) continue;
      if (meta?.max_step != null && s > meta.max_step) continue;
      void fetchStep(sid, s);
    }
  }
}

/** branch復元などで履歴が変わった時にキャッシュを破棄する。 */
export function invalidateHistoryCache(): void {
  cache.clear();
}
