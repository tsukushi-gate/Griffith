"""Word-levelトークナイザー（RDDLGN論文準拠）。

論文: "tokenized at the word level using regex-based splitting"
句読点を単語から分離し、小文字化してトークン化する。
"""

import json
import re
from collections import Counter
from pathlib import Path

# 論文準拠のregex-based splitting:
# 句読点・記号を単語から分離し、空白で分割
_TOKENIZE_RE = re.compile(r"""
    (?:[A-Za-zÀ-ÖØ-öø-ÿß]+        # 英語・ドイツ語の単語（ウムラウト含む）
    |[0-9]+                          # 数字列
    |[.,!?;:\"'()\[\]{}\-/…–—])     # 句読点・記号を個別トークンに
""", re.VERBOSE)


def _regex_tokenize(text: str) -> list[str]:
    """regex-based word-level tokenization（論文準拠）。"""
    return _TOKENIZE_RE.findall(text.lower())


class WordTokenizer:
    """共有語彙のword-levelトークナイザー。

    論文の仕様:
    - word-level tokenization using regex-based splitting
    - 英独共有語彙（16,000トークン）
    - 頻度順でカットオフ
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    SPECIAL_TOKENS = 4
    SPECIAL_WORDS = ['<pad>', '<bos>', '<eos>', '<unk>']

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.word2id: dict[str, int] = {}
        self.id2word: list[str] = []
        self._built = False

    def build_vocab(self, sentences: list[str]) -> None:
        """文のリストから頻度順に語彙を構築。

        Args:
            sentences: 全文のリスト（英語 + ドイツ語を混ぜて渡す）
        """
        counter: Counter[str] = Counter()
        for sent in sentences:
            counter.update(_regex_tokenize(sent))

        # 頻度上位 (vocab_size - SPECIAL_TOKENS) 語を採用
        max_words = self.vocab_size - self.SPECIAL_TOKENS
        most_common = counter.most_common(max_words)

        # 語彙構築: 特殊トークン → 頻出語
        self.id2word = list(self.SPECIAL_WORDS)
        for word, _ in most_common:
            self.id2word.append(word)

        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.vocab_size = len(self.id2word)
        self._built = True

    def encode(self, text: str, max_len: int) -> list[int]:
        """テキスト → トークンID列。

        Args:
            text: 入力テキスト
            max_len: 最大トークン数（truncation）

        Returns:
            トークンIDのリスト（長さ ≤ max_len）
        """
        assert self._built, 'build_vocab()を先に呼んでください'
        tokens = _regex_tokenize(text)[:max_len]
        return [self.word2id.get(w, self.UNK) for w in tokens]

    def decode(self, ids: list[int]) -> str:
        """トークンID列 → テキスト。特殊トークンは除去。"""
        words = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            if 0 <= i < len(self.id2word):
                words.append(self.id2word[i])
            else:
                words.append('<unk>')
        return ' '.join(words)

    def save(self, path: str) -> None:
        """語彙をJSONで保存。"""
        data = {
            'vocab_size': self.vocab_size,
            'id2word': self.id2word,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: str) -> 'WordTokenizer':
        """JSONから語彙を読み込み。"""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        tok = cls(vocab_size=data['vocab_size'])
        tok.id2word = data['id2word']
        tok.word2id = {w: i for i, w in enumerate(tok.id2word)}
        tok.vocab_size = len(tok.id2word)
        tok._built = True
        return tok

    def __len__(self) -> int:
        return self.vocab_size
