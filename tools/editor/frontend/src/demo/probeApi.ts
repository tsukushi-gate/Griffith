// 手描きデモ専用の probe API 薄クライアント。
// 既存 POST /api/probe(backend/probe.py)を叩き、出力ノードの中間値を取り出す。
// 共有の api/rest.ts を触らずにデモ内で完結させる(並行所有の衝突回避)。

/** probe応答(必要部分のみ)。node_id → [batch][width] の中間値トレース。 */
interface ProbeResponse {
  input_width: number;
  batch: number;
  mode: string;
  hard?: Record<string, number[][]>;
  soft?: Record<string, number[][]>;
}

export type ProbeMode = 'hard' | 'soft';

/**
 * 1サンプル(平坦ビット列)を probe API で推論し、出力ノードのスコア列を返す。
 *
 * @param inputBits 0/1 の平坦ビット配列(長さ=IR入力幅)
 * @param outputNodeId 出力ポートへ接続されたソースノードid(drawUtils.outputSourceNodeId)
 * @param mode 'soft'(連続スコア) or 'hard'(整数スコア)
 * @returns 出力ノードのスコア列(batch先頭)
 */
export async function probeInference(
  inputBits: number[],
  outputNodeId: string,
  mode: ProbeMode = 'soft',
): Promise<number[]> {
  const res = await fetch('/api/probe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: [inputBits], mode }),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body.detail) detail = String(body.detail);
    } catch {
      /* JSONでないエラーボディは無視 */
    }
    throw new Error(detail);
  }
  const data = (await res.json()) as ProbeResponse;
  const trace = mode === 'hard' ? data.hard : data.soft;
  const row = trace?.[outputNodeId]?.[0];
  if (!row) {
    throw new Error(`probe応答に出力ノード(${outputNodeId})の値がありません`);
  }
  return row;
}
