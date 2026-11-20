"""固定長二値bag-of-words特徴量（テキスト→{0,1}^V、SST-2 native DLGN獲得用）。

`convert_sst2_to_native_dlgn.py` のbinary-BoW STE-MLP teacherの入力表現。
上位V語の語彙に対する出現有無を{0,1}^Vベクトルへ写す最も単純な特徴抽出で、
`griffith_logic/convert/transcribe.py` が要求する「0/1入力のMLP」という
転写前提（`TranscribeSpec`のdocstring参照）を最初から満たすように設計する。

トークナイズは `experiments/data/tokenizer.py::_regex_tokenize`
（RDDLGN論文準拠のregex word-level splitting）を再利用し、独自のトークン化則を
持ち込まない（既存WordTokenizerと同じ語割りで語彙を作る）。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .tokenizer import _regex_tokenize

__all__ = ['build_bow_vocab', 'texts_to_binary_bow']


def build_bow_vocab(texts: list[str], vocab_size: int) -> list[str]:
    """テキスト集合から出現頻度上位 `vocab_size` 語の語彙リストを構築する。

    特殊トークン（PAD/UNK等）は持たない: 二値BoWは「語彙内の語が出現したか」
    だけを見る集合特徴量であり、位置情報も未知語プレースホルダーも不要
    （出現しない語のビットは単に0のまま）。

    Args:
        texts: 語彙構築に使う文のリスト（通常は訓練split）
        vocab_size: 採用する語の数（vocab_size以上の異なり語があれば頻度順に
            上位のみ採用。異なり語数がvocab_size未満ならその全語）

    Returns:
        頻度降順（同頻度はアルファベット順で決定的にタイブレーク）の語リスト。
        `texts_to_binary_bow` の `vocab` 引数へそのまま渡す。
    """
    if vocab_size < 1:
        raise ValueError(f'vocab_sizeは1以上: {vocab_size}')
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_regex_tokenize(text))
    # 頻度降順、同頻度は語のアルファベット順（Counter.most_commonは同頻度の
    # 順序が挿入順依存で非決定的になり得るため、明示的にソートし直す）
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:vocab_size]]


def texts_to_binary_bow(texts: list[str], vocab: list[str]) -> np.ndarray:
    """テキストのリストを二値bag-of-words行列へ変換する。

    各サンプルは{0,1}^V（V=len(vocab)）: vocab[j]がその文に1回以上出現すれば
    1、そうでなければ0（出現回数は捨てる=binary、TF-IDFではない）。

    Args:
        texts: 変換対象の文のリスト
        vocab: 語彙リスト（`build_bow_vocab` の出力、または再利用の既存語彙）

    Returns:
        (len(texts), len(vocab)) int64 ndarray、値は{0,1}のみ
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    out = np.zeros((len(texts), v), dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = set(_regex_tokenize(text))
        for tok in tokens:
            idx = word2idx.get(tok)
            if idx is not None:
                out[row, idx] = 1
    return out
