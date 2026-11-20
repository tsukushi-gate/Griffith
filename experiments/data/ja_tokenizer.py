"""日本語形態素解析トークナイザー。

MeCab（fugashi）による形態素分割でword-levelトークナイズを行う。
WordTokenizerと同じインターフェースだが、BOS/EOSは不要（分類タスク用）。

依存: fugashi, unidic-lite
    uv sync --extra sentiment でインストール
"""

import json
import re
from collections import Counter
from pathlib import Path

# 数字・記号の正規化用
_NORMALIZE_RE = re.compile(r'[0-9]+')


class JapaneseTokenizer:
    """形態素解析ベースのword-levelトークナイザー。

    fugashi（MeCab wrapper）を使用して日本語テキストを形態素に分割する。
    """

    PAD = 0
    UNK = 1
    SPECIAL_TOKENS = 2
    SPECIAL_WORDS = ['<pad>', '<unk>']

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word2id: dict[str, int] = {}
        self.id2word: list[str] = []
        self._built = False
        self._tagger = None

    @property
    def tagger(self):
        """fugashi.Taggerの遅延初期化。"""
        if self._tagger is None:
            try:
                import fugashi
            except ImportError:
                raise ImportError(
                    'fugashi が必要です。\n'
                    '  pip install fugashi unidic-lite\n'
                    '  または uv sync --extra sentiment でインストールしてください。'
                )
            self._tagger = fugashi.Tagger()
        return self._tagger

    def tokenize(self, text: str) -> list[str]:
        """テキストを形態素に分割する。

        Args:
            text: 入力テキスト

        Returns:
            形態素（表層形）のリスト
        """
        # 数字を正規化（語彙の効率化）
        text = _NORMALIZE_RE.sub('0', text)
        return [word.surface for word in self.tagger(text) if word.surface.strip()]

    def build_vocab(self, sentences: list[str]) -> None:
        """文のリストから頻度順に語彙を構築する。

        Args:
            sentences: 全文のリスト
        """
        counter: Counter[str] = Counter()
        for sent in sentences:
            counter.update(self.tokenize(sent))

        max_words = self.vocab_size - self.SPECIAL_TOKENS
        most_common = counter.most_common(max_words)

        self.id2word = list(self.SPECIAL_WORDS)
        for word, _ in most_common:
            self.id2word.append(word)

        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.vocab_size = len(self.id2word)
        self._built = True

    def encode(self, text: str, max_len: int) -> list[int]:
        """テキストをトークンID列に変換する。

        Args:
            text: 入力テキスト
            max_len: 最大トークン数（truncation）

        Returns:
            トークンIDのリスト（長さ <= max_len）
        """
        assert self._built, 'build_vocab()を先に呼んでください'
        morphemes = self.tokenize(text)[:max_len]
        return [self.word2id.get(w, self.UNK) for w in morphemes]

    def decode(self, ids: list[int]) -> str:
        """トークンID列をテキストに復元する。特殊トークンは除去。"""
        words = []
        for i in ids:
            if i == self.PAD:
                continue
            if 0 <= i < len(self.id2word):
                words.append(self.id2word[i])
            else:
                words.append('<unk>')
        return ''.join(words)

    def save(self, path: str) -> None:
        """語彙をJSONで保存する。"""
        data = {
            'vocab_size': self.vocab_size,
            'id2word': self.id2word,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: str) -> 'JapaneseTokenizer':
        """JSONから語彙を読み込む。"""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        tok = cls(vocab_size=data['vocab_size'])
        tok.id2word = data['id2word']
        tok.word2id = {w: i for i, w in enumerate(tok.id2word)}
        tok.vocab_size = len(tok.id2word)
        tok._built = True
        return tok

    def __len__(self) -> int:
        return self.vocab_size
