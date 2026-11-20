"""SST-2感情極性データセット（英語映画レビュー、二値分類）。

SST-2 (Stanford Sentiment Treebank v2): 英語映画レビュー文の感情極性データセット。
`stanfordnlp/sst2`（GLUE版と同一ソース、trainは67,349文・validationは872文）を使用。

- ソース: stanfordnlp/sst2 (HuggingFace, Parquet形式)
- ラベル: 0=negative, 1=positive
- Split: train(67,349), validation(872)（testはラベル非公開のため未使用）

依存: datasets
    uv sync --extra text でインストール

(ROADMAP-v5 Track D7): テキストKD(hf-seqcls: teacher)のstudent側データセット。
teacherロジットのインデックス整列キャッシュ(convert/cache.py)と組み合わせて使うため、
`SST2Dataset`/`SST2RawTextDataset` は共に決定的な順序(shuffleなし)で同一splitの
同一インデックスが同一サンプルを指すことを保証する。

注意: トークナイザーを変更した場合は data-sst2/ のキャッシュを削除して再生成すること。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer import WordTokenizer

__all__ = ['SST2Dataset', 'SST2RawTextDataset', 'download_and_tokenize']


class SST2Dataset(Dataset):
    """SST-2感情極性データセット（トークンID版）。

    各サンプルは (token_ids, label) のタプル。token_idsはseq_lenに
    パディング/truncation済み。`.texts`で元テキストへインデックス整列で
    アクセスできる（teacherロジットキャッシュとの突き合わせ用）。
    """

    def __init__(self, token_ids: torch.Tensor, labels: torch.Tensor, texts: list[str]):
        """
        Args:
            token_ids: (num_samples, seq_len) パディング済みトークンID
            labels: (num_samples,) 0=negative, 1=positive
            texts: 元テキスト（token_ids/labelsとインデックス整列）
        """
        assert token_ids.shape[0] == labels.shape[0] == len(texts)
        self.token_ids = token_ids
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


class SST2RawTextDataset(Dataset):
    """SST-2の生テキスト版（teacherロジット計算用）。

    `(text_str, label)` を返す。convert/cache.py の `get_or_compute_logits` は
    DataLoaderのデフォルトcollateで束ねた `x`(文字列のtuple)をそのまま
    `teacher.predict_logits(x)` に渡すため、テキストをトークナイズせずに扱う。

    **不変条件**: SST2Dataset と同一split・同一インデックスで同一サンプルを
    指すこと(convert/cache.pyのインデックス整列前提)。これは
    `download_and_tokenize` がフィルタ(空文除外)を一切行わないことで保証される
    (WRIME/WMT14と異なり、SST-2は空文が存在しないためフィルタ不要)。
    """

    def __init__(self, split: str, data_dir: str = './data-sst2', train_limit: int | None = None):
        """
        Args:
            split: 'train' または 'validation'
            data_dir: データ保存先（download_and_tokenizeと同じディレクトリ）
            train_limit: split='train'のとき先頭N件に決定的にサブサンプルする
                （teacherロジットキャッシュキーにも反映させること）
        """
        texts, labels = _load_raw_split(split, data_dir)
        if train_limit is not None and split == 'train':
            texts = texts[:train_limit]
            labels = labels[:train_limit]
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def _raw_cache_path(data_dir: str) -> Path:
    return Path(data_dir) / 'raw_splits.pt'


def _load_raw_split(split: str, data_dir: str) -> tuple[list[str], list[int]]:
    """生テキスト+ラベルをロードする（キャッシュがあればそれを使う）。"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    raw_cache = _raw_cache_path(data_dir)

    if raw_cache.exists():
        raw = torch.load(raw_cache, weights_only=True)
    else:
        raw = _download_raw_splits()
        torch.save(raw, raw_cache)

    if split not in raw:
        raise ValueError(f"未知のsplit: {split!r} (対応: {sorted(raw.keys())})")
    return list(raw[split]['texts']), list(raw[split]['labels'])


def _download_raw_splits() -> dict:
    """HuggingFaceからSST-2の train/validation を生テキストのままダウンロードする。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            'SST2データセットには datasets パッケージが必要です。\n'
            '  uv sync --extra text でインストールしてください。'
        )

    print('SST-2感情極性データセットをダウンロード中...')
    ds = load_dataset('stanfordnlp/sst2')

    raw = {}
    for split, key in (('train', 'train'), ('validation', 'validation')):
        texts = [item['sentence'] for item in ds[key]]
        labels = [int(item['label']) for item in ds[key]]
        raw[split] = {'texts': texts, 'labels': labels}
        print(f'  {split}: {len(texts):,} サンプル')

    return raw


def _tokenize_samples(
    texts: list[str],
    labels: list[int],
    tokenizer: WordTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """テキストをトークナイズしてテンソルに変換する（パディングのみ、フィルタなし）。

    student(SST2Dataset)とraw(SST2RawTextDataset)のインデックス整列を保つため、
    サンプルの除外は一切行わない(WRIME/WMT14と異なりSST-2は空文が無いため安全)。
    """
    token_list = []
    for text in tqdm(texts, desc='トークナイズ中'):
        encoded = tokenizer.encode(text, seq_len)
        padded = encoded + [WordTokenizer.PAD] * (seq_len - len(encoded))
        token_list.append(padded)

    return (
        torch.tensor(token_list, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        list(texts),
    )


def download_and_tokenize(
    vocab_size: int = 8000,
    seq_len: int = 64,
    data_dir: str = './data-sst2',
    train_limit: int | None = None,
) -> tuple[SST2Dataset, SST2Dataset, WordTokenizer]:
    """SST-2データのダウンロード・トークナイズ・キャッシュを一括実行する。

    初回: HuggingFaceからダウンロード → 語彙構築 → トークナイズ → .pt保存
    2回目以降: .ptキャッシュから直接ロード

    Args:
        vocab_size: 語彙サイズ（デフォルト8000）
        seq_len: 最大系列長（デフォルト64）
        data_dir: データ保存先ディレクトリ
        train_limit: 訓練データを先頭N件に決定的にサブサンプルする（デバッグ・
            スモークテスト用。Noneなら全件）

    Returns:
        (train_dataset, validation_dataset, tokenizer)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    limit_suffix = f'_lim{train_limit}' if train_limit is not None else ''
    cache_train = data_path / f'train_v{vocab_size}_s{seq_len}{limit_suffix}.pt'
    cache_val = data_path / f'validation_v{vocab_size}_s{seq_len}.pt'
    vocab_path = data_path / f'vocab_v{vocab_size}.json'

    if cache_train.exists() and cache_val.exists() and vocab_path.exists():
        print(f'キャッシュからロード: {data_dir}')
        tokenizer = WordTokenizer.load(str(vocab_path))
        train_data = torch.load(cache_train, weights_only=True)
        val_data = torch.load(cache_val, weights_only=True)
        train_set = SST2Dataset(train_data['token_ids'], train_data['labels'], train_data['texts'])
        val_set = SST2Dataset(val_data['token_ids'], val_data['labels'], val_data['texts'])
        print(f'  訓練: {len(train_set):,} サンプル, 検証: {len(val_set):,} サンプル')
        print(f'  語彙: {tokenizer.vocab_size:,} トークン')
        return train_set, val_set, tokenizer

    train_texts, train_labels = _load_raw_split('train', data_dir)
    val_texts, val_labels = _load_raw_split('validation', data_dir)

    if train_limit is not None:
        train_texts = train_texts[:train_limit]
        train_labels = train_labels[:train_limit]

    print(f'  訓練: {len(train_texts):,} サンプル')
    print(f'  検証: {len(val_texts):,} サンプル')

    # クラス分布を表示
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    print(f'  訓練クラス分布: negative={train_neg:,} positive={train_pos:,}')

    # 語彙構築（訓練データのみ。vocab_pathはtrain_limitに依存させない: 全件語彙で
    # 固定し、サブサンプル間でも語彙の再現性を保つ）
    if vocab_path.exists():
        print('語彙をロード中...')
        tokenizer = WordTokenizer.load(str(vocab_path))
    else:
        print(f'語彙を構築中（上位 {vocab_size:,} 語）...')
        full_train_texts, _ = _load_raw_split('train', data_dir)
        tokenizer = WordTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(full_train_texts)
        tokenizer.save(str(vocab_path))
    print(f'  語彙サイズ: {tokenizer.vocab_size:,}')

    print('訓練データをトークナイズ中...')
    train_ids, train_lbls, train_texts_out = _tokenize_samples(
        train_texts, train_labels, tokenizer, seq_len,
    )
    print(f'  訓練: {train_ids.shape[0]:,} サンプル')

    print('検証データをトークナイズ中...')
    val_ids, val_lbls, val_texts_out = _tokenize_samples(
        val_texts, val_labels, tokenizer, seq_len,
    )
    print(f'  検証: {val_ids.shape[0]:,} サンプル')

    torch.save(
        {'token_ids': train_ids, 'labels': train_lbls, 'texts': train_texts_out},
        cache_train,
    )
    torch.save(
        {'token_ids': val_ids, 'labels': val_lbls, 'texts': val_texts_out},
        cache_val,
    )
    print(f'キャッシュを保存: {data_dir}')

    train_set = SST2Dataset(train_ids, train_lbls, train_texts_out)
    val_set = SST2Dataset(val_ids, val_lbls, val_texts_out)
    return train_set, val_set, tokenizer
