"""WMT'14 English-German翻訳データセット（RDDLGN論文準拠）。

論文の仕様:
- WMT'14 English-German（約450万文ペア）
- word-level regex-based splitting、共有16K語彙
- 系列長16に固定（truncation + padding）
- 空文・全PADサンプルをフィルタ

依存: datasets (HuggingFace), sacrebleu
    uv sync --extra wmt でインストール

注意: トークナイザーを変更した場合は data-wmt14/ のキャッシュを削除して再生成すること。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer import WordTokenizer


class WMT14Dataset(Dataset):
    """WMT'14 English-German翻訳データセット。

    各サンプルは (src_ids, tgt_ids) のタプル。
    BOS/EOSは付加しない（train_rddlgn.pyのprepare_batchで付加）。
    """

    def __init__(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        """
        Args:
            src_ids: (num_samples, seq_len) パディング済みソーストークンID
            tgt_ids: (num_samples, seq_len) パディング済みターゲットトークンID
        """
        assert src_ids.shape == tgt_ids.shape
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self):
        return self.src_ids.shape[0]

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]


def _tokenize_pairs(
    pairs: list[tuple[str, str]],
    tokenizer: WordTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """文ペアをトークナイズしてテンソルに変換。"""
    src_list = []
    tgt_list = []

    for src_text, tgt_text in tqdm(pairs, desc='トークナイズ中'):
        src_enc = tokenizer.encode(src_text, seq_len)
        tgt_enc = tokenizer.encode(tgt_text, seq_len)

        # 論文準拠: 空文をフィルタ
        if len(src_enc) == 0 or len(tgt_enc) == 0:
            continue

        # パディング（PAD=0）
        src_padded = src_enc + [WordTokenizer.PAD] * (seq_len - len(src_enc))
        tgt_padded = tgt_enc + [WordTokenizer.PAD] * (seq_len - len(tgt_enc))

        # 論文準拠: 全PADサンプルをフィルタ
        if all(t == WordTokenizer.PAD for t in src_padded) or \
           all(t == WordTokenizer.PAD for t in tgt_padded):
            continue

        src_list.append(src_padded)
        tgt_list.append(tgt_padded)

    return (
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(tgt_list, dtype=torch.long),
    )


def download_and_tokenize(
    vocab_size: int = 16000,
    seq_len: int = 16,
    data_dir: str = './data-wmt14',
) -> tuple[WMT14Dataset, WMT14Dataset, WordTokenizer]:
    """WMT'14データのダウンロード・トークナイズ・キャッシュを一括実行。

    初回: HuggingFaceからダウンロード → トークナイズ → .pt保存
    2回目以降: .ptキャッシュから直接ロード

    Args:
        vocab_size: 語彙サイズ（デフォルト16000、論文準拠）
        seq_len: 最大系列長（デフォルト16、論文準拠）
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
        tokenizer = WordTokenizer.load(str(vocab_path))
        train_data = torch.load(cache_train, weights_only=True)
        test_data = torch.load(cache_test, weights_only=True)
        train_set = WMT14Dataset(train_data['src'], train_data['tgt'])
        test_set = WMT14Dataset(test_data['src'], test_data['tgt'])
        print(f'  訓練: {len(train_set):,} ペア, テスト: {len(test_set):,} ペア')
        print(f'  語彙: {tokenizer.vocab_size:,} トークン')
        return train_set, test_set, tokenizer

    # HuggingFace datasetsからダウンロード
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            'WMT14データセットには datasets パッケージが必要です。\n'
            '  uv sync --extra wmt でインストールしてください。'
        )

    print('WMT\'14 En-De データセットをダウンロード中...')
    ds = load_dataset('wmt14', 'de-en', trust_remote_code=True)

    # 訓練データとテストデータを取得
    train_raw = ds['train']
    test_raw = ds['test']

    # テキストペアを抽出
    def extract_pairs(split):
        pairs = []
        for item in tqdm(split, desc='テキスト抽出'):
            trans = item['translation']
            pairs.append((trans['en'], trans['de']))
        return pairs

    print('テキストペアを抽出中...')
    train_pairs = extract_pairs(train_raw)
    test_pairs = extract_pairs(test_raw)
    print(f'  訓練: {len(train_pairs):,} ペア, テスト: {len(test_pairs):,} ペア')

    # 語彙構築（英独全文から共有語彙）
    print(f'語彙を構築中（上位 {vocab_size:,} 語）...')
    all_sentences = [en for en, de in train_pairs] + [de for en, de in train_pairs]
    tokenizer = WordTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(all_sentences)
    print(f'  語彙サイズ: {tokenizer.vocab_size:,}')

    # トークナイズ
    print('訓練データをトークナイズ中...')
    train_src, train_tgt = _tokenize_pairs(train_pairs, tokenizer, seq_len)
    print(f'  訓練: {train_src.shape[0]:,} ペア (空文除外後)')

    print('テストデータをトークナイズ中...')
    test_src, test_tgt = _tokenize_pairs(test_pairs, tokenizer, seq_len)
    print(f'  テスト: {test_src.shape[0]:,} ペア')

    # キャッシュ保存
    tokenizer.save(str(vocab_path))
    torch.save({'src': train_src, 'tgt': train_tgt}, cache_train)
    torch.save({'src': test_src, 'tgt': test_tgt}, cache_test)
    print(f'キャッシュを保存: {data_dir}')

    train_set = WMT14Dataset(train_src, train_tgt)
    test_set = WMT14Dataset(test_src, test_tgt)
    return train_set, test_set, tokenizer
