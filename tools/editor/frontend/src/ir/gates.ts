// N=2の16ゲート名テーブルとゲートID変換(名前/ID規約は tools/editor/backend/extract.py と同一)。
// 表示色はフロントテーマの所管として theme/gateColors.ts に定義し、ここからre-exportする。
// ゲートID: 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数

import type { RGB } from '../theme/theme';

export { gateColor } from '../theme/gateColors';
export type { RGB };

export const GATE_NAMES: Record<number, string> = {
  0: 'FALSE', 1: 'AND', 2: 'A>B', 3: 'A',
  4: 'B>A', 5: 'B', 6: 'XOR', 7: 'OR',
  8: 'NOR', 9: 'XNOR', 10: '!B', 11: 'A>=B',
  12: '!A', 13: 'B>=A', 14: 'NAND', 15: 'TRUE',
};

/**
 * N=2真理値表(bit k=butterfly基底順、bit j of k=スロットjの値)を16ゲートIDへ変換する。
 * core/basis.py の _build_gate_table_n2 と同一規約。
 */
export function ttToGateIdN2(tt: number): number {
  const b00 = tt & 1;
  const b10 = (tt >> 1) & 1;
  const b01 = (tt >> 2) & 1;
  const b11 = (tt >> 3) & 1;
  return (b00 << 3) | (b01 << 2) | (b10 << 1) | b11;
}

export function rgbCss(c: RGB, alpha = 1): string {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

/** 確信度に応じて背景色へブレンドする(白背景では確信度が低いほど淡く沈む)。 */
export function blendToBg(c: RGB, bg: RGB, t: number): RGB {
  const k = Math.min(Math.max(t, 0), 1);
  return [
    Math.round(c[0] * k + bg[0] * (1 - k)),
    Math.round(c[1] * k + bg[1] * (1 - k)),
    Math.round(c[2] * k + bg[2] * (1 - k)),
  ];
}
