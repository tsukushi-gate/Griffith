// スライス文字列の解釈: '' → null(全体)/ '5' → [5,5] / '7:0' → [7,0]。
// '[7:0]' の角括弧も許容する。不正は Error(UI 側でエラー表示)。
export function parseSlice(
  text: string,
  width: number,
): [number, number] | null {
  const t = text.trim().replace(/^\[|\]$/g, '');
  if (t === '') return null;
  const m = /^(\d+)(?::(\d+))?$/.exec(t);
  if (!m) throw new Error(`スライスの書式が不正: "${text}"(例: 5 / 7:0)`);
  const msb = Number(m[1]);
  const lsb = m[2] === undefined ? msb : Number(m[2]);
  if (lsb > msb) throw new Error(`msb >= lsb が必要: [${msb}:${lsb}]`);
  if (msb >= width) throw new Error(`バス幅 ${width} を超えている: [${msb}:${lsb}]`);
  return [msb, lsb];
}

export function sliceLabel(sl: [number, number] | null): string {
  if (sl === null) return '';
  return sl[0] === sl[1] ? `[${sl[0]}]` : `[${sl[0]}:${sl[1]}]`;
}
