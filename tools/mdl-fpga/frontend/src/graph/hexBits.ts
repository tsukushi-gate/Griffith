// hex(非パディング)⇔ bit 配列。幅は常に meta 基準で渡す(ブリッジの hex は
// ゼロパディングされない=実測仕様)。
export function hexToBigInt(hex: string | undefined): bigint {
  if (!hex) return 0n;
  return BigInt(`0x${hex}`);
}

export function hexToBits(hex: string | undefined, width: number): Uint8Array {
  const v = hexToBigInt(hex);
  const bits = new Uint8Array(width);
  for (let i = 0; i < width; i++) bits[i] = Number((v >> BigInt(i)) & 1n);
  return bits;
}

export function bitOf(hex: string | undefined, index: number): 0 | 1 {
  return Number((hexToBigInt(hex) >> BigInt(index)) & 1n) as 0 | 1;
}

export function formatHex(hex: string | undefined, width: number): string {
  const digits = Math.ceil(width / 4);
  return '0x' + (hex ?? '0').padStart(digits, '0');
}
