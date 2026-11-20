"""WRIME感情極性データセット（日本語ネガポジ分類）。

WRIME: 日本語SNS投稿の感情極性データセット。
tetsuro731/wrime-sentiment（中立除外済みのバイナリ版）を使用。

- ソース: tetsuro731/wrime-sentiment (HuggingFace, Parquet形式)
- ラベル: 0=Positive, 1=Negative（既に二値化・中立除外済み）
- Split: train(~20K), validation(~1.6K), test(~1.8K)

依存: datasets, fugashi, unidic-lite
    uv sync --extra sentiment でインストール

注意: トークナイザーを変更した場合は data-wrime/ のキャッシュを削除して再生成すること。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .ja_tokenizer import JapaneseTokenizer


class WRIMEDataset(Dataset):
    """WRIME感情極性データセット。各サンプルは (token_ids, label) のタプル。"""

    def __init__(self, token_ids: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            token_ids: (num_samples, seq_len) パディング済みトークンID
            labels: (num_samples,) 0=positive, 1=negative
        """
        assert token_ids.shape[0] == labels.shape[0]
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def _tokenize_samples(
    texts: list[str],
    labels: list[int],
    tokenizer: JapaneseTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """テキストをトークナイズしてテンソルに変換する。"""
    token_list = []
    label_list = []

    for text, label in tqdm(zip(texts, labels), desc='トークナイズ中', total=len(texts)):
        encoded = tokenizer.encode(text, seq_len)

        # 空文を除外
        if len(encoded) == 0:
            continue

        # パディング
        padded = encoded + [JapaneseTokenizer.PAD] * (seq_len - len(encoded))

        # 全PADサンプルを除外
        if all(t == JapaneseTokenizer.PAD for t in padded):
            continue

        token_list.append(padded)
        label_list.append(label)

    return (
        torch.tensor(token_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.long),
    )


def download_and_tokenize(
    vocab_size: int = 8000,
    seq_len: int = 128,
    data_dir: str = './data-wrime',
) -> tuple[WRIMEDataset, WRIMEDataset, JapaneseTokenizer]:
    """WRIMEデータのダウンロード・トークナイズ・キャッシュを一括実行する。

    初回: HuggingFaceからダウンロード → トークナイズ → .pt保存
    2回目以降: .ptキャッシュから直接ロード

    Args:
        vocab_size: 語彙サイズ（デフォルト8000）
        seq_len: 最大系列長（デフォルト128）
        data_dir: データ保存ディレクトリ

    Returns:
        (train_dataset, test_dataset, tokenizer)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    cache_train = data_path / f'train_v{vocab_size}_s{seq_len}.pt'
    cache_test = data_path / f'test_v{vocab_size}_s{seq_len}.pt'
    vocab_path = data_path / f'vocab_v{vocab_size}.json'

    # キャッシュが存在すればロード
    if cache_train.exists() and cache_test.exists() and vocab_path.exists():
        print(f'キャッシュからロード: {data_dir}')
        tokenizer = JapaneseTokenizer.load(str(vocab_path))
        train_data = torch.load(cache_train, weights_only=True)
        test_data = torch.load(cache_test, weights_only=True)
        train_set = WRIMEDataset(train_data['token_ids'], train_data['labels'])
        test_set = WRIMEDataset(test_data['token_ids'], test_data['labels'])
        print(f'  訓練: {len(train_set):,} サンプル, テスト: {len(test_set):,} サンプル')
        print(f'  語彙: {tokenizer.vocab_size:,} トークン')
        return train_set, test_set, tokenizer

    # HuggingFace datasetsからダウンロード
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            'WRIMEデータセットには datasets パッケージが必要です。\n'
            '  uv sync --extra sentiment でインストールしてください。'
        )

    print('WRIME感情極性データセットをダウンロード中...')
    ds = load_dataset('tetsuro731/wrime-sentiment')

    # trainとtest splitを取得（validationをtestに統合）
    train_texts = [item['sentence'] for item in ds['train']]
    train_labels = [item['label'] for item in ds['train']]
    # validation + test を統合してテストデータとする
    test_texts = [item['sentence'] for item in ds['validation']]
    test_labels = [item['label'] for item in ds['validation']]
    test_texts += [item['sentence'] for item in ds['test']]
    test_labels += [item['label'] for item in ds['test']]

    print(f'  訓練: {len(train_texts):,} サンプル')
    print(f'  テスト: {len(test_texts):,} サンプル (validation + test)')

    # クラス分布を表示
    train_neg = sum(train_labels)
    train_pos = len(train_labels) - train_neg
    print(f'  訓練クラス分布: Positive={train_pos:,} Negative={train_neg:,}')

    # 語彙構築（訓練データのみ）
    print(f'語彙を構築中（上位 {vocab_size:,} 語）...')
    tokenizer = JapaneseTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(train_texts)
    print(f'  語彙サイズ: {tokenizer.vocab_size:,}')

    # トークナイズ
    print('訓練データをトークナイズ中...')
    train_ids, train_lbls = _tokenize_samples(
        train_texts, train_labels, tokenizer, seq_len,
    )
    print(f'  訓練: {train_ids.shape[0]:,} サンプル (空文除外後)')

    print('テストデータをトークナイズ中...')
    test_ids, test_lbls = _tokenize_samples(
        test_texts, test_labels, tokenizer, seq_len,
    )
    print(f'  テスト: {test_ids.shape[0]:,} サンプル')

    # キャッシュ保存
    tokenizer.save(str(vocab_path))
    torch.save({'token_ids': train_ids, 'labels': train_lbls}, cache_train)
    torch.save({'token_ids': test_ids, 'labels': test_lbls}, cache_test)
    print(f'キャッシュを保存: {data_dir}')

    train_set = WRIMEDataset(train_ids, train_lbls)
    test_set = WRIMEDataset(test_ids, test_lbls)
    return train_set, test_set, tokenizer
