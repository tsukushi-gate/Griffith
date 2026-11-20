"""griffith_logic/convert/text_teacher.py のテスト。

デフォルト実行はネットワークアクセス無し(offline-safe)。HFからの実DL検証は
@pytest.mark.slow を付与し、高速スイート(make test)からは除外する。
"""

import pytest
import torch

from griffith_logic.convert.text_teacher import (
    SHAKESPEARE_CHAR_VOCAB,
    _MinGPT,
    decode_shakespeare_char,
    encode_shakespeare_char,
    resolve_text_teacher,
)


# ----------------------------------------------------------------------
# resolve_text_teacher: 未知スキーム
# ----------------------------------------------------------------------
def test_resolve_unknown_scheme_raises_value_error():
    with pytest.raises(ValueError):
        resolve_text_teacher('bogus-scheme:whatever')


def test_resolve_missing_colon_raises_value_error():
    with pytest.raises(ValueError):
        resolve_text_teacher('no-colon-here')


# ----------------------------------------------------------------------
# vocab定数
# ----------------------------------------------------------------------
def test_vocab_length_is_65():
    assert len(SHAKESPEARE_CHAR_VOCAB) == 65


def test_vocab_roundtrip():
    text = "ROMEO:\nWhat, art thou hurt?"
    # プロンプトの全文字がvocabに含まれることを確認してからencode/decode往復
    assert all(ch in SHAKESPEARE_CHAR_VOCAB for ch in text)
    tokens = encode_shakespeare_char(text)
    assert decode_shakespeare_char(tokens) == text


# ----------------------------------------------------------------------
# _MinGPT: 乱数初期化の極小configでforward形状
# ----------------------------------------------------------------------
def _tiny_gpt_cfg():
    return dict(n_layer=1, n_head=2, n_embd=8, block_size=16, vocab_size=65, bias=False)


def test_min_gpt_forward_shape():
    cfg = _tiny_gpt_cfg()
    model = _MinGPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg['vocab_size'], (3, 10))
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (3, 10, cfg['vocab_size'])


def test_min_gpt_weight_tying():
    cfg = _tiny_gpt_cfg()
    model = _MinGPT(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_min_gpt_block_size_assertion():
    cfg = _tiny_gpt_cfg()
    model = _MinGPT(cfg)
    idx = torch.randint(0, cfg['vocab_size'], (1, cfg['block_size'] + 1))
    with pytest.raises(AssertionError):
        model(idx)


# ----------------------------------------------------------------------
# NanoGPTCharTeacher.generate相当のロジック: _MinGPTを直接使い自己回帰生成を検証
# (HF DL無しでgenerateの決定性・形状・block_sizeクロップを確認するため、
#  NanoGPTCharTeacher.__init__を経由せず_MinGPTに対して同じ手順を手動実行する)
# ----------------------------------------------------------------------
@torch.no_grad()
def _generate_with_min_gpt(model, n_seqs, length, block_size, *, temperature=1.0, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    idx = torch.zeros(n_seqs, 1, dtype=torch.long)
    for _ in range(length - 1):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        if generator is not None:
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def test_generate_determinism_shape_and_values():
    cfg = _tiny_gpt_cfg()
    torch.manual_seed(0)
    model = _MinGPT(cfg)
    model.eval()

    out1 = _generate_with_min_gpt(model, n_seqs=2, length=12, block_size=cfg['block_size'], seed=42)
    out2 = _generate_with_min_gpt(model, n_seqs=2, length=12, block_size=cfg['block_size'], seed=42)

    assert out1.shape == (2, 12)
    assert torch.equal(out1, out2)  # 同一seedで決定的
    assert (out1 >= 0).all()
    assert (out1 < cfg['vocab_size']).all()


def test_generate_respects_block_size_cropping():
    """block_sizeを超える文脈でもクラッシュせず、直近block_size分のみ使われる。"""
    cfg = _tiny_gpt_cfg()
    torch.manual_seed(1)
    model = _MinGPT(cfg)
    model.eval()

    length = cfg['block_size'] + 5  # block_sizeを超えた生成長
    out = _generate_with_min_gpt(model, n_seqs=1, length=length, block_size=cfg['block_size'], seed=7)
    assert out.shape == (1, length)
    assert (out < cfg['vocab_size']).all()


# ----------------------------------------------------------------------
# HFダウンロードを伴う実ネットワークテスト(遅い/オフライン環境ではskip対象)
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_hf_sequence_classifier_teacher_real_download():
    from griffith_logic.convert.text_teacher import resolve_text_teacher

    teacher = resolve_text_teacher('hf-seqcls:distilbert-base-uncased-finetuned-sst-2-english')
    logits = teacher.predict_logits(['this movie is great', 'this movie is terrible'])
    assert logits.shape == (2, teacher.num_classes)
    assert logits.dtype == torch.float32
    assert logits.device.type == 'cpu'


@pytest.mark.slow
def test_nanogpt_char_teacher_real_download():
    from griffith_logic.convert.text_teacher import resolve_text_teacher

    teacher = resolve_text_teacher('hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')
    assert teacher.vocab_size == 65

    prompt = torch.tensor([[teacher.stoi[c] for c in 'ROMEO:']], dtype=torch.long)
    logits = teacher.next_token_logits(prompt)
    assert logits.shape == (1, 6, 65)

    out = teacher.generate(2, 20, seed=1337, prompt=prompt)
    assert out.shape == (2, 20)
