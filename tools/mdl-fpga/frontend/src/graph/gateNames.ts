// N=2 の 16 ゲート名(difflogic 系譜の真理値表順)。
// tt の bit k = 入力パターン k(k = B<<1 | A、A=slot0)の出力。
export const GATE_NAMES_N2: Record<number, string> = {
  0x0: 'FALSE',
  0x1: 'NOR',
  0x2: 'A∧¬B',
  0x3: '¬B',
  0x4: '¬A∧B',
  0x5: '¬A',
  0x6: 'XOR',
  0x7: 'NAND',
  0x8: 'AND',
  0x9: 'XNOR',
  0xa: 'A',
  0xb: 'A∨¬B',
  0xc: 'B',
  0xd: '¬A∨B',
  0xe: 'OR',
  0xf: 'TRUE',
};

export function gateLabel(tt: number, nInputs: number): string {
  if (nInputs === 2 && tt in GATE_NAMES_N2) return GATE_NAMES_N2[tt]!;
  return `LUT${nInputs} 0x${tt.toString(16)}`;
}
