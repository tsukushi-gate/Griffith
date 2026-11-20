// 真理値表hex文字列のビット操作。
// N=6の真理値表は64bitでNumberの安全整数を超えるためBigIntで扱う。
// ビット規約: bit k = butterfly基底順の入力パターンk(bit j of k = スロットjの値)

/** hex文字列('0x…'または裸のhex)をBigIntへ。不正なら0n。 */
export function ttParse(hex: string): bigint {
  try {
    const s = hex.trim().toLowerCase();
    return BigInt(s.startsWith('0x') ? s : `0x${s}`);
  } catch {
    return 0n;
  }
}

/** 2^Nビットの真理値表としてhex文字列('0x…'小文字、ゼロ詰めなし)へ。 */
export function ttToHex(value: bigint): string {
  return `0x${value.toString(16)}`;
}

/** ビットkの値(0/1)。 */
export function ttBit(hex: string, k: number): 0 | 1 {
  return (ttParse(hex) >> BigInt(k)) & 1n ? 1 : 0;
}

/** ビットkをトグルした新しいhex文字列を返す。 */
export function ttToggleBit(hex: string, k: number): string {
  return ttToHex(ttParse(hex) ^ (1n << BigInt(k)));
}

/** 立っているビット数。 */
export function popcount(hex: string): number {
  let v = ttParse(hex);
  let count = 0;
  while (v > 0n) {
    count += Number(v & 1n);
    v >>= 1n;
  }
  return count;
}
