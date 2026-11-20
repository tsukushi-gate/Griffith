// 真理値表検証テーブルの行列生成ロジック(純関数)。
//
// パターン規約(butterfly基底順): パターンkの入力bit j = (k >> j) & 1。
// 複数入力ポートはIR宣言順に連結した1本のビット列として扱う。
// expectedTable(パターンk昇順の出力bits配列)は将来「この真理値表を学習」へ
// そのまま渡せるデータ形式として維持する。

import type { IRGraphJson } from '../ir/schema';
import { evalCircuit, type LayerProbs } from './evalCircuit';

/** 全列挙を許可する入力総ビット幅の上限(2^10=1024行) */
export const TRUTH_TABLE_MAX_BITS = 10;

export interface TruthRow {
  /** 入力パターン(bit j = (k >> j) & 1) */
  k: number;
  /** 全入力ポート連結のビット列(下位ビットから) */
  inputBits: number[];
  /** hard評価の回路出力 */
  outputBits: number[];
}

export interface MatchInfo {
  perRow: boolean[];
  matched: number;
  total: number;
}

/** 入力ポート幅の合計 */
export function totalInputWidth(ir: IRGraphJson): number {
  return ir.inputs.reduce((acc, p) => acc + p.width, 0);
}

/** パターンkをポート別ビット配列へ分配する(ポート宣言順に下位ビットから割当) */
export function patternToInputBits(ir: IRGraphJson, k: number): Record<string, number[]> {
  const result: Record<string, number[]> = {};
  let offset = 0;
  for (const port of ir.inputs) {
    result[port.id] = Array.from({ length: port.width }, (_, j) => (k >> (offset + j)) & 1);
    offset += port.width;
  }
  return result;
}

/** ポート別ビット配列からパターンkを復元する(patternToInputBitsの逆) */
export function inputBitsToPattern(
  ir: IRGraphJson,
  inputBits: Record<string, number[]>,
): number {
  let k = 0;
  let offset = 0;
  for (const port of ir.inputs) {
    const bits = inputBits[port.id] ?? [];
    for (let j = 0; j < port.width; j++) {
      if (bits[j]) k |= 1 << (offset + j);
    }
    offset += port.width;
  }
  return k;
}

/** 2^n全パターンをhard評価して行列を生成する(入力幅>上限はエラー) */
export function enumerateTruthTable(
  ir: IRGraphJson,
  layerProbs?: Record<string, LayerProbs>,
): TruthRow[] {
  const width = totalInputWidth(ir);
  if (width > TRUTH_TABLE_MAX_BITS) {
    throw new Error(`入力幅${width}bitは全列挙上限(${TRUTH_TABLE_MAX_BITS}bit)を超えています`);
  }
  const rows: TruthRow[] = [];
  const count = 1 << width;
  for (let k = 0; k < count; k++) {
    const bits = patternToInputBits(ir, k);
    const { output } = evalCircuit(ir, bits, 'hard', layerProbs);
    rows.push({
      k,
      inputBits: Array.from({ length: width }, (_, j) => (k >> j) & 1),
      outputBits: Array.from(output),
    });
  }
  return rows;
}

/** 期待値の初期値(現在の回路出力のコピー) */
export function rowsToExpected(rows: TruthRow[]): Record<number, number[]> {
  const expected: Record<number, number[]> = {};
  for (const row of rows) {
    expected[row.k] = [...row.outputBits];
  }
  return expected;
}

/** 現在の回路出力と期待値の一致情報(行別+一致率) */
export function matchInfo(
  rows: TruthRow[],
  expected: Record<number, number[]> | null,
): MatchInfo {
  const perRow = rows.map((row) => {
    const exp = expected?.[row.k];
    if (!exp || exp.length !== row.outputBits.length) return false;
    return row.outputBits.every((b, j) => b === exp[j]);
  });
  return {
    perRow,
    matched: perRow.filter(Boolean).length,
    total: rows.length,
  };
}

/** 期待値が現在の行列と同じ次元か(IR構造変更時の再初期化判定) */
export function expectedMatchesShape(
  expected: Record<number, number[]> | null,
  rows: TruthRow[] | null,
): boolean {
  if (!expected || !rows) return false;
  if (Object.keys(expected).length !== rows.length) return false;
  return rows.every((row) => expected[row.k]?.length === row.outputBits.length);
}
