import { describe, expect, it } from 'vitest';
import { bitCells } from './PayloadView';

describe('bitCells', () => {
  it('既定 [msb:lsb] = 左端が最上位ビット', () => {
    // 0xd = 1101: bit3=1 bit2=1 bit1=0 bit0=1
    const cells = bitCells('d', 4, 'msb');
    expect(cells.map((c) => c.index)).toEqual([3, 2, 1, 0]);
    expect(cells.map((c) => c.on)).toEqual([true, true, false, true]);
  });
  it('[lsb:msb] 切替', () => {
    const cells = bitCells('d', 4, 'lsb');
    expect(cells.map((c) => c.index)).toEqual([0, 1, 2, 3]);
    expect(cells.map((c) => c.on)).toEqual([true, false, true, true]);
  });
  it('非パディングhexも幅基準で展開', () => {
    const cells = bitCells('1', 8, 'msb');
    expect(cells).toHaveLength(8);
    expect(cells[7]).toEqual({ index: 0, on: true });
  });
});
