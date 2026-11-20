"""位置エンコーディング部品。

位置→ビット列テーブル(buffer)を入力へ連結する。学習パラメータなし。
blocks_to_ir では constantノード + concatノードへ展開される
(IR側は定数線のハードワイヤ。HDL生成可能)。
"""

import torch
import torch.nn as nn

POSITIONAL_CODES = ('binary', 'gray', 'thermometer')


def position_codes(seq_len: int, width: int, code: str) -> torch.Tensor:
    """位置→ビット列テーブルを生成する。

    code:
        'binary'      — 位置pの2進表現(bit j = (p >> j) & 1)。
                        seq_len ≦ 2^width が必要
        'gray'        — グレイコード(隣接位置が1bit差)。同上
        'thermometer' — bit j = 1 if p > j(単調な埋め込み)。
                        width ≧ seq_len-1 が必要(popcount == p)

    Returns:
        (seq_len, width) の{0,1} float32テンソル
    """
    assert code in POSITIONAL_CODES, f'code は {POSITIONAL_CODES} のいずれか: {code}'
    if code in ('binary', 'gray'):
        assert seq_len <= (1 << width), (
            f'binary/grayは seq_len({seq_len}) ≦ 2^width({1 << width}) が必要'
        )
        table = []
        for p in range(seq_len):
            g = p ^ (p >> 1) if code == 'gray' else p
            table.append([(g >> j) & 1 for j in range(width)])
    else:  # thermometer
        assert width >= seq_len - 1, (
            f'thermometerは width({width}) ≧ seq_len-1({seq_len - 1}) が必要'
        )
        table = [[1 if p > j else 0 for j in range(width)]
                 for p in range(seq_len)]
    return torch.tensor(table, dtype=torch.float32)


class LogicPositionalEncoding(nn.Module):
    """位置エンコーディングの定数ビット列を入力へ連結する部品。

    forward(x: (B, in_width)) → (B, in_width + seq_len*width)。
    末尾に位置テーブルのflatten(位置優先: bit(p, j) → index p*width+j)を
    連結する。blocks_to_ir では constantノード(bits=bits_flat) +
    concatノード(slot 0=入力、slot 1=定数)へ展開される
    (IRのconcatビット順規約: slot 0が出力ビット列の先頭)。
    """

    def __init__(self, seq_len: int, width: int, code: str = 'binary'):
        super().__init__()
        self.seq_len = seq_len
        self.width = width
        self.code = code
        self.register_buffer('table', position_codes(seq_len, width, code))

    @property
    def bits_flat(self) -> list:
        """flatten済みビット列(IR constantノードのbits)"""
        return [int(b) for b in self.table.reshape(-1).tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits = self.table.reshape(1, -1).to(x.dtype).expand(x.shape[0], -1)
        return torch.cat([x, bits], dim=-1)

    def extra_repr(self):
        return f'seq_len={self.seq_len}, width={self.width}, code={self.code}'
