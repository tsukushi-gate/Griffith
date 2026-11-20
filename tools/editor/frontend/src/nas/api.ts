// NAS API フェッチャ。rest.ts は編集不可のため、その request/post
// ラッパを模した最小実装をここに持つ(ApiError は rest.ts から再利用)。
// vite dev の /api プロキシ経由なので相対パスでよい(ホスト/ポートは書かない)。

import { ApiError } from '../api/rest';
import type {
  NasIrResponse,
  NasStartRequest,
  NasStartResponse,
  NasStatus,
  NasTrials,
} from './types';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
    } catch {
      // JSONでない場合はstatusTextのまま
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export function startNasSearch(req: NasStartRequest): Promise<NasStartResponse> {
  return post<NasStartResponse>('/api/nas/start', req);
}

export function getNasStatus(): Promise<NasStatus> {
  return request<NasStatus>('/api/nas/status');
}

export function getNasTrials(): Promise<NasTrials> {
  return request<NasTrials>('/api/nas/trials');
}

export function stopNasSearch(): Promise<NasStatus> {
  return post<NasStatus>('/api/nas/stop');
}

export function getNasTrialIr(trialId: number): Promise<NasIrResponse> {
  return request<NasIrResponse>(`/api/nas/trials/${trialId}/ir`);
}
