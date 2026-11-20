// 構造化配線のフロント側定義。
// 種と引数の一覧はライブラリ core/connections.py の WIRING_SPEC_KEYS と対応する
// (生成はバックエンド POST /api/wiring/generate = make_connections。ここでは
// インスペクタの入力欄・既定値・文字列 ⇄ 値の変換だけを持つ)。
// lut_layer の train_info.connections に種、train_info.wiring に引数を残し、
// draft ノードは build 時に同じ関数で再生成される。

/** LutLayer が文字列のまま受ける配線種 */
export const BASIC_CONNECTIONS = ['random', 'unique'] as const;

/** 乱数を引かず引数だけで決まる配線種(core.connections.STRUCTURED_WIRING_KINDS) */
export const STRUCTURED_CONNECTIONS = [
  'rot',
  'rot_n',
  'butterfly',
  'local1d',
  'local2d',
] as const;

export type StructuredConnection = (typeof STRUCTURED_CONNECTIONS)[number];

export const CONNECTION_CHOICES: readonly string[] = [
  ...BASIC_CONNECTIONS,
  ...STRUCTURED_CONNECTIONS,
];

export function isStructuredConnection(kind: string): kind is StructuredConnection {
  return (STRUCTURED_CONNECTIONS as readonly string[]).includes(kind);
}

/** 入力欄 1 つの定義。value は文字列で持ち、toSpec で型付きの値へ変換する */
export interface WiringField {
  key: string;
  label: string;
  hint: string;
  /** 'int' = 整数 1 つ、'ints' = 整数のカンマ区切り、'pairs' = "dy,dx;dy,dx"、'grid' = "H,W"(省略可) */
  kind: 'int' | 'ints' | 'pairs' | 'grid';
  optional?: boolean;
}

export const WIRING_FIELDS: Record<StructuredConnection, WiringField[]> = {
  rot: [
    { key: 's_a', label: 's_a', hint: 'スロット0のずらし量', kind: 'int' },
    { key: 's_b', label: 's_b', hint: 'スロット1のずらし量', kind: 'int' },
    { key: 'step', label: 'step', hint: 'ゲート番号の歩進(既定1)', kind: 'int' },
  ],
  rot_n: [
    { key: 'shifts', label: 'shifts', hint: 'スロットごとのずらし量(n_inputs個、カンマ区切り)', kind: 'ints' },
    { key: 'step', label: 'step', hint: 'ゲート番号の歩進(既定1)', kind: 'int' },
  ],
  butterfly: [
    { key: 'stride', label: 'stride', hint: '(g + m·stride) mod in_width', kind: 'int' },
  ],
  local1d: [
    { key: 'stride', label: 'stride', hint: '比例写像の中心からの間隔', kind: 'int' },
  ],
  local2d: [
    { key: 'offsets', label: 'offsets', hint: '格子オフセット "dy,dx;dy,dx"(n_inputs組)', kind: 'pairs' },
    { key: 'grid', label: 'grid', hint: '画像格子 "H,W"(省略=正方トーラス)', kind: 'grid', optional: true },
    { key: 'bits_per_cell', label: 'bits/cell', hint: 'セル当たりのビット数(既定1)', kind: 'int', optional: true },
  ],
};

/** 種を選んだ直後の既定引数(n_inputs に応じて埋める) */
export function defaultWiringSpec(
  kind: StructuredConnection,
  nInputs: number,
): Record<string, unknown> {
  switch (kind) {
    case 'rot':
      return { s_a: 0, s_b: 1, step: 1 };
    case 'rot_n':
      return { shifts: Array.from({ length: nInputs }, (_, j) => j), step: nInputs };
    case 'butterfly':
      return { stride: 1 };
    case 'local1d':
      return { stride: 1 };
    case 'local2d':
      return {
        offsets: Array.from({ length: nInputs }, (_, j) => [0, j]),
        bits_per_cell: 1,
      };
  }
}

/** spec の値を入力欄用の文字列にする */
export function specValueToText(value: unknown, field: WiringField): string {
  if (value === undefined || value === null) return '';
  switch (field.kind) {
    case 'int':
      return String(value);
    case 'ints':
      return Array.isArray(value) ? value.map(String).join(',') : String(value);
    case 'pairs':
      return Array.isArray(value)
        ? value.map((p) => (Array.isArray(p) ? p.map(String).join(',') : String(p))).join(';')
        : String(value);
    case 'grid':
      return Array.isArray(value) ? value.map(String).join(',') : String(value);
  }
}

function parseInts(text: string): number[] | null {
  const parts = text.split(',').map((s) => s.trim()).filter((s) => s.length > 0);
  if (parts.length === 0) return null;
  const nums = parts.map(Number);
  return nums.every((v) => Number.isInteger(v)) ? nums : null;
}

/**
 * 入力欄の文字列を spec の値へ変換する。不正なら null(呼び出し側がエラー表示)。
 * 空文字は optional なら「未指定」(undefined)、必須なら null。
 */
export function textToSpecValue(
  text: string,
  field: WiringField,
): { ok: true; value: unknown } | { ok: false } {
  const t = text.trim();
  if (t === '') return field.optional ? { ok: true, value: undefined } : { ok: false };
  switch (field.kind) {
    case 'int': {
      const v = Number(t);
      return Number.isInteger(v) ? { ok: true, value: v } : { ok: false };
    }
    case 'ints': {
      const v = parseInts(t);
      return v ? { ok: true, value: v } : { ok: false };
    }
    case 'pairs': {
      const pairs = t.split(';').map((s) => parseInts(s));
      return pairs.every((p) => p !== null && p.length === 2)
        ? { ok: true, value: pairs as number[][] }
        : { ok: false };
    }
    case 'grid': {
      const v = parseInts(t);
      return v && v.length === 2 ? { ok: true, value: v } : { ok: false };
    }
  }
}

/** 被覆率などの統計(バックエンド connection_stats の形) */
export interface WiringStats {
  coverage: number;
  usage_hist: Record<string, number>;
  same_input: number;
  n_gates: number;
}

export function describeWiringStats(stats: WiringStats, inWidth: number): string {
  const read = Math.round(stats.coverage * inWidth);
  const pct = (stats.coverage * 100).toFixed(1);
  return `被覆 ${read}/${inWidth}(${pct}%)・未使用入力 ${stats.usage_hist['0'] ?? 0}・同一入力ゲート ${stats.same_input}`;
}
