// Shipレンズ(HDL出荷プリフライト、§4-3)の純粋ヘルパ。
// プリフライト応答の表示整形(バッジ・見積り表・ブロッカーラベル)を副作用なしで扱う。

import type {
  PreflightResponse,
  ResourceEstimate,
} from '../api/rest';

/** ブロッカー種別の日本語ラベル。 */
export function blockerKindLabel(kind: string): string {
  switch (kind) {
    case 'blackbox':
      return 'blackbox(非論理)';
    case 'add_sum':
      return 'add(整数和)';
    case 'structure':
      return '構造エラー';
    default:
      return kind;
  }
}

export interface PreflightBadge {
  tone: 'ok' | 'blocked';
  text: string;
}

/** HDL出力可否バッジの状態。 */
export function preflightBadge(res: PreflightResponse): PreflightBadge {
  if (res.hdl_exportable) {
    return { tone: 'ok', text: 'HDL出力可' };
  }
  return { tone: 'blocked', text: `HDL出力不可(${res.blockers.length}件)` };
}

/** lut_counts(n_inputs別個数)を "2入力×1, 3入力×2" の形へ整形する。 */
export function formatLutCounts(lutCounts: Record<string, number>): string {
  const keys = Object.keys(lutCounts).sort((a, b) => Number(a) - Number(b));
  if (keys.length === 0) return '-';
  return keys.map((k) => `${k}入力×${lutCounts[k]}`).join(', ');
}

export interface EstimateRow {
  label: string;
  value: string;
}

/** リソース見積りをラベル/値の表行へ整形する。 */
export function estimateRows(est: ResourceEstimate): EstimateRow[] {
  return [
    { label: 'LUT総数', value: String(est.total_luts) },
    { label: 'LUT内訳(入力数別)', value: formatLutCounts(est.lut_counts) },
    { label: '論理深さ', value: String(est.logic_depth) },
    { label: '配線本数', value: String(est.wire_count) },
    { label: 'group_sum数', value: String(est.group_sums) },
    {
      label: '展開',
      value: est.lowered ? '展開済(conv/pool→LUT列)' : '不要',
    },
  ];
}
