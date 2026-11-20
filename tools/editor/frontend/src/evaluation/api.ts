// EVALUATIONタブ専用のREST APIクライアント。
// 並行競合回避のため src/api/rest.ts には手を入れず、この配下に自前で書く。
// バックエンド tools/editor/backend/evaluate.py の EvaluateResponse/EvalItem と対応。

/** HTTPステータス付きAPIエラー。 */
export class EvalApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'EvalApiError';
    this.status = status;
  }
}

/** 評価対象のソース(現IR or 学習runのスナップショット)。 */
export interface EvalSource {
  kind: 'current' | 'run';
  run_id?: string | null;
}

/** POST /api/evaluate のリクエスト。 */
export interface EvaluateParams {
  source: EvalSource;
  dataset: string;
  executor?: string | null;
  dataset_params?: Record<string, unknown> | null;
  page?: number;
  page_size?: number;
}

/** 1サンプルの評価行。 */
export interface EvalItem {
  index: number;
  label: number | null;
  pred: number | null;
  label_bits: number[] | null;
  pred_bits: number[] | null;
  correct: boolean;
  soft_confidence: number;
  soft_margin: number;
  hard_soft_mismatch: boolean;
  input_bits: number[] | null;
  image: string | null;
}

/** POST /api/evaluate のレスポンス。 */
export interface EvaluateResponse {
  source_kind: string;
  dataset: string;
  executor: string | null;
  kind: 'classification' | 'binary_vector';
  num_classes: number | null;
  num_samples: number;
  page: number;
  page_size: number;
  num_pages: number;
  cached: boolean;
  on_demand: boolean;
  cap: number;
  accuracy: number | null;
  num_correct: number | null;
  num_errors: number | null;
  num_mismatch: number | null;
  confusion: number[][] | null;
  fingerprint: string;
  items: EvalItem[];
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body.detail) detail = String(body.detail);
    } catch {
      // JSONでないエラーボディは無視
    }
    throw new EvalApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

function body(params: EvaluateParams): string {
  return JSON.stringify({
    source: params.source,
    dataset: params.dataset,
    executor: params.executor ?? null,
    dataset_params: params.dataset_params ?? null,
    page: params.page ?? 0,
    page_size: params.page_size ?? 200,
  });
}

/** 一括評価(ページング)。 */
export function evaluate(params: EvaluateParams): Promise<EvaluateResponse> {
  return request<EvaluateResponse>('/api/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body(params),
  });
}

/** 上限内なら全ページを順に取得して1つの配列へまとめる(全件をフロント集計するため)。 */
export async function evaluateAll(
  params: Omit<EvaluateParams, 'page'>,
): Promise<{ meta: EvaluateResponse; items: EvalItem[] }> {
  const pageSize = params.page_size ?? 500;
  const first = await evaluate({ ...params, page: 0, page_size: pageSize });
  const items = [...first.items];
  if (first.on_demand) {
    // 大規模: 全件ロードはしない(呼び出し側がページUIで扱う)
    return { meta: first, items };
  }
  for (let page = 1; page < first.num_pages; page++) {
    const next = await evaluate({ ...params, page, page_size: pageSize });
    items.push(...next.items);
  }
  return { meta: first, items };
}

/** CSVエクスポート。テキストを返す。*/
export async function exportCsv(params: EvaluateParams): Promise<string> {
  const res = await fetch('/api/evaluate/export.csv', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body(params),
  });
  if (!res.ok) {
    throw new EvalApiError(`${res.status} ${res.statusText}`, res.status);
  }
  return res.text();
}
