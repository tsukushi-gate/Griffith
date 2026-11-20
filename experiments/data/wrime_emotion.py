"""WRIME 8感情2値分類データセット。

WRIME ver1（43,200件）の読者アノテーション（Avg. Readers）を使用し、
8感情（Joy, Sadness, Anticipation, Surprise, Anger, Fear, Disgust, Trust）
それぞれの有無を二値分類するためのデータローダー。

arXiv 2505.00013 の実験条件に準拠:
- アノテーション: Avg. Readers（読者3名の平均強度）
- 二値化: 強度 ≥1 → 陽性(1), 0 → 陰性(0)
- 分割: 全43,200件を80/20でランダム分割（train/test）
- 評価: 8感情のマクロ平均accuracy・マクロ平均F1

ソース: https://github.com/ids-cv/wrime (wrime-ver1.tsv)

依存: fugashi, unidic-lite, pandas
    uv sync --extra sentiment でインストール
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .ja_tokenizer import JapaneseTokenizer

# 8感情の定義（arXiv 2505.00013 と同じ順序）
EMOTIONS = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

# WRIME ver1 の URL
WRIME_V1_URL = 'https://raw.githubusercontent.com/ids-cv/wrime/refs/heads/master/wrime-ver1.tsv'


class WRIMEEmotionDataset(Dataset):
    """WRIME 8感情2値分類データセット。各サンプルは (token_ids, labels) のタプル。"""

    def __init__(self, token_ids: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            token_ids: (num_samples, seq_len) パディング済みトークンID
            labels: (num_samples, 8) 各感情の二値ラベル (0 or 1)
        """
        assert token_ids.shape[0] == labels.shape[0]
        assert labels.shape[1] == len(EMOTIONS)
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def _tokenize_samples(
    texts: list[str],
    labels: list[list[int]],
    tokenizer: JapaneseTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """テキストをトークナイズしてテンソルに変換する。"""
    token_list = []
    label_list = []

    for text, label in tqdm(zip(texts, labels), desc='トークナイズ中', total=len(texts)):
        encoded = tokenizer.encode(text, seq_len)

        if len(encoded) == 0:
            continue

        padded = encoded + [JapaneseTokenizer.PAD] * (seq_len - len(encoded))

        if all(t == JapaneseTokenizer.PAD for t in padded):
            continue

        token_list.append(padded)
        label_list.append(label)

    return (
        torch.tensor(token_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.long),
    )


def _print_emotion_distribution(labels: list[list[int]], split_name: str):
    """各感情の陽性/陰性分布を表示する。"""
    n = len(labels)
    print(f'  {split_name} クラス分布 (n={n:,}):')
    for i, emo in enumerate(EMOTIONS):
        pos = sum(1 for lbl in labels if lbl[i] == 1)
        neg = n - pos
        print(f'    {emo:>13s}: pos={pos:>5,} ({pos / n * 100:5.1f}%)  neg={neg:>5,} ({neg / n * 100:5.1f}%)')


def download_and_tokenize_emotion(
    vocab_size: int = 8000,
    seq_len: int = 128,
    data_dir: str = './data-wrime-emotion',
    annotation: str = 'reader',
    threshold: int = 1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[WRIMEEmotionDataset, WRIMEEmotionDataset, JapaneseTokenizer]:
    """WRIME 8感情データのダウンロード・トークナイズ・キャッシュを一括実行する。

    arXiv 2505.00013 の実験条件:
    - annotation='reader' (Avg. Readers)
    - threshold=1 (強度≥1で陽性)
    - test_ratio=0.2 (80/20分割)

    Args:
        vocab_size: 語彙サイズ
        seq_len: 最大系列長
        data_dir: データ保存ディレクトリ
        annotation: 'reader' (Avg. Readers) or 'writer'
        threshold: 二値化閾値（≥threshold で陽性）
        test_ratio: テスト分割比率
        seed: ランダム分割のシード

    Returns:
        (train_dataset, test_dataset, tokenizer)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    suffix = f'v{vocab_size}_s{seq_len}_{annotation}_t{threshold}_r{test_ratio}_seed{seed}'
    cache_train = data_path / f'train_{suffix}.pt'
    cache_test = data_path / f'test_{suffix}.pt'
    vocab_path = data_path / f'vocab_v{vocab_size}_{annotation}_seed{seed}.json'

    # キャッシュが存在すればロード
    if cache_train.exists() and cache_test.exists() and vocab_path.exists():
        print(f'キャッシュからロード: {data_dir}')
        tokenizer = JapaneseTokenizer.load(str(vocab_path))
        train_data = torch.load(cache_train, weights_only=True)
        test_data = torch.load(cache_test, weights_only=True)
        train_set = WRIMEEmotionDataset(train_data['token_ids'], train_data['labels'])
        test_set = WRIMEEmotionDataset(test_data['token_ids'], test_data['labels'])
        print(f'  訓練: {len(train_set):,} サンプル, テスト: {len(test_set):,} サンプル')
        print(f'  語彙: {tokenizer.vocab_size:,} トークン')
        print(f'  アノテーション: {annotation}, 閾値: ≥{threshold}, 分割: {1 - test_ratio:.0%}/{test_ratio:.0%}')
        return train_set, test_set, tokenizer

    # TSVをダウンロード
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'pandasが必要です。\n'
            '  uv sync --extra sentiment でインストールしてください。'
        )

    tsv_path = data_path / 'wrime-ver1.tsv'
    if tsv_path.exists():
        print(f'ローカルTSVからロード: {tsv_path}')
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        print('WRIME ver1 をダウンロード中...')
        df = pd.read_csv(WRIME_V1_URL, sep='\t')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f'  保存: {tsv_path}')

    print(f'  全データ: {len(df):,} サンプル')

    # アノテーション列の選択
    if annotation == 'reader':
        prefix = 'Avg. Readers'
    elif annotation == 'writer':
        prefix = 'Writer'
    else:
        raise ValueError(f"annotation は 'reader' または 'writer': {annotation}")

    # 8感情ラベルの抽出・二値化
    texts = df['Sentence'].tolist()
    labels = []
    for _, row in df.iterrows():
        label = [1 if row[f'{prefix}_{emo}'] >= threshold else 0 for emo in EMOTIONS]
        labels.append(label)

    print(f'  アノテーション: {prefix}, 二値化閾値: ≥{threshold}')

    # 80/20 ランダム分割（arXiv 2505.00013準拠）
    import random
    indices = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_ratio))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f'  分割: 訓練={len(train_texts):,} テスト={len(test_texts):,} (seed={seed})')
    _print_emotion_distribution(train_labels, '訓練')
    _print_emotion_distribution(test_labels, 'テスト')

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

    train_set = WRIMEEmotionDataset(train_ids, train_lbls)
    test_set = WRIMEEmotionDataset(test_ids, test_lbls)
    return train_set, test_set, tokenizer
