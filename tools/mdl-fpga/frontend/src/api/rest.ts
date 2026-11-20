import type {
  BoardDoc, BoardState, DetectResultDoc, ExampleEntry, Meta, NodeDetail,
  OpenRequest, OpenResponse, ProjectInfo, RendererListing, ViewerConfig,
} from './types';

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json() as Promise<T>;
}

async function sendJson<T>(url: string, method: string,
                           body: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // サーバの detail(原因+次の一手)をそのままユーザーへ届ける
    let detail = `${method} ${url}: ${res.status}`;
    try {
      const doc = (await res.json()) as { detail?: string };
      if (doc.detail) detail = doc.detail;
    } catch {
      /* JSON でないエラー応答はステータスのまま */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getMeta = () => getJson<Meta>('/api/meta');
export const getRenderers = () => getJson<RendererListing>('/api/renderers');
export const getLayout = () => getJson<ViewerConfig>('/api/layout');
export const getNodeDetail = (id: string) =>
  getJson<NodeDetail>(`/api/node/${encodeURIComponent(id)}`);

export async function putLayout(cfg: ViewerConfig,
                                epoch?: number): Promise<void> {
  const url = epoch != null ? `/api/layout?epoch=${epoch}` : '/api/layout';
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  if (!res.ok) throw new Error(`PUT ${url}: ${res.status}`);
}

// --- プロジェクト ---
export const getProject = () => getJson<ProjectInfo>('/api/project');
export const getExamples = () =>
  getJson<{ examples: ExampleEntry[] }>('/api/examples');
export const openProject = (req: OpenRequest) =>
  sendJson<OpenResponse>('/api/project/open', 'POST', req);

// --- ボード ---
export const getBoards = () => getJson<{ boards: BoardDoc[] }>('/api/boards');
export const getBoard = () => getJson<BoardState>('/api/board');
export const putBoard = (key: string | null) =>
  sendJson<BoardState & { ok: boolean }>('/api/board', 'PUT', { key });
export const postDetect = () =>
  sendJson<DetectResultDoc>('/api/board/detect', 'POST', {});
