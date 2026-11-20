"""byte-level 語彙(翻訳DLGN)の共有プリミティブ = 単一情報源。

学生(DLGNAttentionTranslator)・データセットが共有する byte 語彙をここに集約する。
生UTF-8バイト 0..255 + 特殊トークン3個で vocab=259。

系列KD方針(準拠): teacher(opus-mt-ja-en 等の subword MT)は**訳文テキスト**を
返すだけで、logit-level の soft KD は使わない。よって teacher の出力英文を
`encode_bytes` で byte 化して学生の CE ターゲットにする。teacher と学生で語彙を
共有する必要はなく、byte 語彙は学生側だけの規約となる。
"""

from __future__ import annotations

from collections.abc import Sequence

# --- 学生 byte 語彙 -------------------------------------------------------
BYTE_VALUES = 256          # 生UTF-8バイト 0..255
BYTE_PAD_ID = 256
BYTE_BOS_ID = 257          # decoder開始トークン
BYTE_EOS_ID = 258
BYTE_VOCAB_SIZE = 259

__all__ = [
    'BYTE_VALUES', 'BYTE_PAD_ID', 'BYTE_BOS_ID', 'BYTE_EOS_ID', 'BYTE_VOCAB_SIZE',
    'encode_bytes', 'decode_bytes',
]


def encode_bytes(text: str) -> list[int]:
    """文字列を UTF-8 バイト列(値 0..255 のID列)へ変換する。"""
    return list(text.encode('utf-8'))


def decode_bytes(ids: Sequence[int]) -> str:
    """byte ID列を UTF-8 文字列へ復号する(特殊トークンは無視、不正バイトは置換)。"""
    raw = bytes(int(i) for i in ids if 0 <= int(i) < BYTE_VALUES)
    return raw.decode('utf-8', errors='replace')
