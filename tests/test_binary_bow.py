"""experiments/data/binary_bow.py の単体テスト(SST-2 binary-BoW特徴量)。

`build_bow_vocab`/`texts_to_binary_bow` は
`convert_sst2_to_native_dlgn.py` の binary-BoW STE-MLP teacher入力を作る
再利用可能ヘルパ。CONTRIBUTING.mdのテスト規約に基づき、素朴な性質
(値域{0,1}・語彙頻度順・出現有無の意味論)を専用テストで固定する。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from data.binary_bow import build_bow_vocab, texts_to_binary_bow


class TestBuildBowVocab:
    def test_frequency_order(self):
        texts = ['the cat sat', 'the cat ran', 'the dog sat']
        vocab = build_bow_vocab(texts, vocab_size=10)
        # 'the'=3回, 'cat'=2回, 'sat'=2回, 'ran'=1回, 'dog'=1回
        assert vocab[0] == 'the'
        assert set(vocab[1:3]) == {'cat', 'sat'}

    def test_truncates_to_vocab_size(self):
        texts = ['a b c d e f']
        vocab = build_bow_vocab(texts, vocab_size=3)
        assert len(vocab) == 3

    def test_fewer_unique_words_than_vocab_size(self):
        texts = ['a a a', 'b b']
        vocab = build_bow_vocab(texts, vocab_size=100)
        assert set(vocab) == {'a', 'b'}

    def test_deterministic_tiebreak(self):
        """同頻度語はアルファベット順に決定的にタイブレークする(再現性)。"""
        texts = ['zebra apple mango']  # 全語頻度1
        vocab1 = build_bow_vocab(texts, vocab_size=10)
        vocab2 = build_bow_vocab(texts, vocab_size=10)
        assert vocab1 == vocab2
        assert vocab1 == sorted(vocab1)

    def test_rejects_nonpositive_vocab_size(self):
        import pytest
        with pytest.raises(ValueError):
            build_bow_vocab(['x'], vocab_size=0)


class TestTextsToBinaryBow:
    def test_binary_values_only(self):
        vocab = ['cat', 'dog', 'bird']
        texts = ['the cat sat on the cat mat', 'a dog ran']
        x = texts_to_binary_bow(texts, vocab)
        assert set(np.unique(x)).issubset({0, 1})

    def test_shape(self):
        vocab = ['a', 'b', 'c', 'd']
        texts = ['a b', 'c', 'a b c d', '']
        x = texts_to_binary_bow(texts, vocab)
        assert x.shape == (4, 4)

    def test_presence_semantics_ignores_count(self):
        """出現回数でなく出現有無のみを見る(binary、TF-IDFではない)。"""
        vocab = ['cat']
        x = texts_to_binary_bow(['cat cat cat'], vocab)
        assert x[0, 0] == 1

    def test_out_of_vocab_words_ignored(self):
        vocab = ['cat']
        x = texts_to_binary_bow(['dog bird fish'], vocab)
        assert x.tolist() == [[0]]

    def test_empty_text_all_zero(self):
        vocab = ['cat', 'dog']
        x = texts_to_binary_bow([''], vocab)
        assert x.tolist() == [[0, 0]]

    def test_dtype_int64(self):
        vocab = ['cat']
        x = texts_to_binary_bow(['cat'], vocab)
        assert x.dtype == np.int64

    def test_case_insensitive(self):
        """_regex_tokenizeは小文字化するため、大文字小文字を区別しない。"""
        vocab = ['cat']
        x = texts_to_binary_bow(['CAT'], vocab)
        assert x.tolist() == [[1]]


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
