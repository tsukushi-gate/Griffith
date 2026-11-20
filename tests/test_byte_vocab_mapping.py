"""byte語彙(翻訳DLGN)の encode/decode 検証(オフライン)。"""

from griffith_logic.convert.byte_vocab import (
    BYTE_BOS_ID,
    BYTE_EOS_ID,
    BYTE_PAD_ID,
    BYTE_VALUES,
    BYTE_VOCAB_SIZE,
    decode_bytes,
    encode_bytes,
)


def test_vocab_constants():
    assert BYTE_VALUES == 256
    assert (BYTE_PAD_ID, BYTE_BOS_ID, BYTE_EOS_ID) == (256, 257, 258)
    assert BYTE_VOCAB_SIZE == 259


def test_encode_decode_roundtrip_ascii_and_japanese():
    for text in ['Hello, world!', 'あいうえお', 'ja→en 変換', '']:
        ids = encode_bytes(text)
        assert all(0 <= i < BYTE_VALUES for i in ids)
        assert decode_bytes(ids) == text


def test_encode_japanese_is_multibyte():
    # 日本語1文字=UTF-8で3バイト。
    assert len(encode_bytes('あ')) == 3
    assert encode_bytes('あ') == [227, 129, 130]


def test_decode_ignores_special_tokens():
    ids = encode_bytes('hi') + [BYTE_PAD_ID, BYTE_BOS_ID, BYTE_EOS_ID]
    assert decode_bytes(ids) == 'hi'
