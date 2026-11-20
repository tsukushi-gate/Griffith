"""UART フレーミング(唯一の正= mdl-fpga の hdl/frame_rx.v)。

mdl_fpga/hw_check.py の build_request/parse_response と**バイト一致**する
複製(独立性のため import しない。一致はリポジトリ側テストで機械固定):

- 要求: [0xA5][seq][payload = ceil(w/8) バイト little-endian][checksum]
- 応答: [0x5A][seq エコー][payload 同上][checksum]
- checksum = (seq + payload バイト総和) & 0xFF。受信は SOF 探索で resync
"""

from __future__ import annotations

import math

REQ_SOF = 0xA5
RSP_SOF = 0x5A
FRAMING = 'mdl-frame-v1'


def payload_bytes(width: int) -> int:
    return math.ceil(width / 8)


def build_request(seq: int, x_value: int, n_bytes: int) -> bytes:
    payload = x_value.to_bytes(n_bytes, 'little')
    body = bytes([seq & 0xFF]) + payload
    chk = sum(body) & 0xFF
    return bytes([REQ_SOF]) + body + bytes([chk])


def parse_response(buf: bytes, n_bytes: int) -> tuple[int, int] | None:
    """(seq, payload値) を返す。フレーム不成立なら None(SOF 探索で resync)。"""
    total = n_bytes + 3
    i = buf.find(bytes([RSP_SOF]))
    if i < 0 or len(buf) - i < total:
        return None
    frame = buf[i:i + total]
    chk = sum(frame[1:-1]) & 0xFF
    if frame[-1] != chk:
        return None
    payload = int.from_bytes(frame[2:2 + n_bytes], 'little')
    return frame[1], payload


def decode_scores(y_value: int, k: int, score_bits: int) -> list[int]:
    """class_scores 出力バスの分解(LSB 側= class0。core.py パッキング規約)。"""
    mask = (1 << score_bits) - 1
    return [(y_value >> (c * score_bits)) & mask for c in range(k)]


def bits_of(value: int, width: int) -> list[int]:
    """整数 → bit 列(index i = bit i。pack_bits の逆)。"""
    return [(value >> i) & 1 for i in range(width)]


def value_of(bits) -> int:
    """bit 列 → 整数(index i = bit i)。"""
    return sum((1 if b else 0) << i for i, b in enumerate(bits))
