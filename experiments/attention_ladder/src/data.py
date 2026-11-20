"""attention梯子のデータローダ(L0基準用)。

2種類のタスクを提供する:
  - sMNIST(sequential MNIST): 28x28画像を行単位で系列化(T=28ステップ×28bit)した分類タスク
  - Tiny Shakespeare: 文字レベル言語モデル(char-level LM)

コア実装(griffith_logic/)は読み取りのみ。sMNISTは
`griffith_logic.datasets.vision.load_mnist` の薄いreshapeアダプタ、
Tiny Shakespeareは完全自作(DL・語彙構築・窓生成)。
"""

from __future__ import annotations

import os
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset

from griffith_logic.datasets.vision import load_mnist

# ------------------------------------------------------------
# 定数
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LADDER_ROOT = os.path.dirname(_SRC_DIR)  # 絶対: リポジトリ内の実体(バンドル済みデータ)
# cwd相対の既定ベース(Modal規約: os.chdir('/workspace') 後に永続Volumeへ載る)
_DEFAULT_DATA_BASE = os.path.join('experiments', 'attention_ladder')

# sMNIST: 行単位系列化(各行28bit×28行)
SMNIST_SEQ_LEN = 28
SMNIST_INPUT_DIM = 28
SMNIST_NUM_CLASSES = 10

# Tiny Shakespeare
SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
_SHAKESPEARE_SUBPATH = os.path.join('data', 'tinyshakespeare', 'input.txt')
_DEFAULT_CONTEXT_LEN = 64
_SHAKESPEARE_TRAIN_RATIO = 0.9


# ============================================================
# sMNIST(行単位系列化)
# ============================================================

class _RowSequenceMNIST(Dataset):
    """(1,28,28)二値画像を(28,28)の行系列に整形するラッパ。

    各行28bitが1ステップの入力、系列長T=28。値域{0,1} float32。
    """

    def __init__(self, base_dataset: Dataset) -> None:
        self._base = base_dataset

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self._base[idx]
        # img: (1,28,28) {0,1} float32 → (28,28)
        x = img.reshape(SMNIST_SEQ_LEN, SMNIST_INPUT_DIM).float()
        return x, int(label)


def load_smnist(
    batch_size: int,
    root: str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """sMNIST(行単位系列化)の(train_loader, val_loader, meta)を返す。

    `load_mnist(download=True)` の二値化済み(1,28,28){0,1}を行系列(28,28)に
    整形する。testセットをvalとして流用する。

    Returns:
        train_loader: x:(B,28,28) float32 {0,1}, y:(B,) long
        val_loader:   同上(testセット)
        meta: {'task','seq_len','input_dim','num_classes'}
    """
    train_base, test_base = load_mnist(root=root, download=True)
    train_set = _RowSequenceMNIST(train_base)
    val_set = _RowSequenceMNIST(test_base)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    meta = {
        'task': 'classification',
        'seq_len': SMNIST_SEQ_LEN,
        'input_dim': SMNIST_INPUT_DIM,
        'num_classes': SMNIST_NUM_CLASSES,
    }
    return train_loader, val_loader, meta


# ============================================================
# Tiny Shakespeare(char-level LM)
# ============================================================

def _shakespeare_candidates(root: str | None = None) -> list[str]:
    """input.txt の探索候補パス(読み取り優先順)を返す。

    root指定時はそれのみ。既定は cwd相対(Modalでは/workspace Volume) →
    リポジトリ絶対(バンドル済み) の順。DL先は先頭候補(cwd相対)。
    """
    if root is not None:
        return [os.path.join(root, _SHAKESPEARE_SUBPATH)]
    return [
        os.path.join(_DEFAULT_DATA_BASE, _SHAKESPEARE_SUBPATH),
        os.path.join(_LADDER_ROOT, _SHAKESPEARE_SUBPATH),
    ]


def prepare_shakespeare(root: str | None = None) -> str:
    """Tiny Shakespeare の input.txt を用意しパスを返す。

    既存ファイル(cwd相対 or リポジトリ絶対)があれば再利用する。無ければ
    urllibでDLする(DL先は永続Volumeに載る cwd相対の先頭候補)。DL失敗時は
    明確に例外を投げる(捏造データは作らない)。
    """
    candidates = _shakespeare_candidates(root)
    for candidate in candidates:
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate

    path = candidates[0]  # DL先(cwd相対=Modalでは/workspace Volume)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001 (DL失敗は原因を明示して再送出)
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロードに失敗しました: {SHAKESPEARE_URL} '
            f'(原因: {e})'
        ) from e
    if not data:
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロード内容が空です: {SHAKESPEARE_URL}'
        )
    with open(path, 'wb') as f:
        f.write(data)
    return path


class _CharWindowDataset(Dataset):
    """スライディング窓でchar-LMの(x, y)ペアを生成する。

    x = tokens[i : i+context_len]、y = tokens[i+1 : i+context_len+1]。
    """

    def __init__(self, data: torch.Tensor, context_len: int) -> None:
        self._data = data
        self._context_len = context_len

    def __len__(self) -> int:
        return max(0, self._data.numel() - self._context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._data[idx: idx + self._context_len + 1]
        return chunk[:-1], chunk[1:]


def load_shakespeare(
    batch_size: int,
    context_len: int = _DEFAULT_CONTEXT_LEN,
    root: str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Tiny Shakespeare(char-level LM)の(train_loader, val_loader, meta)を返す。

    テキストからchar語彙(stoi/itos, 語彙数V〜65)を構築し、90/10でtrain/val分割。
    スライディング窓(context_len)で次トークン予測データを作る。

    Returns:
        train_loader: x:(B,T) long, y:(B,T) long(次トークンid)
        val_loader:   同上(末尾10%)
        meta: {'task','seq_len','vocab_size','stoi','itos'}
    """
    path = prepare_shakespeare(root)
    with open(path, encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(ids.numel() * _SHAKESPEARE_TRAIN_RATIO)
    train_data = ids[:split]
    val_data = ids[split:]

    train_set = _CharWindowDataset(train_data, context_len)
    val_set = _CharWindowDataset(val_data, context_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    meta = {
        'task': 'lm',
        'seq_len': context_len,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    return train_loader, val_loader, meta


if __name__ == '__main__':
    # 簡易セルフチェック(スモーク用): Shakespeareの取得とローダのshape確認
    p = prepare_shakespeare()
    size = os.path.getsize(p)
    print(f'shakespeare input.txt: {p} ({size:,} bytes)')

    tr, va, m = load_shakespeare(batch_size=8)
    xb, yb = next(iter(tr))
    print(f'vocab_size={m["vocab_size"]} seq_len={m["seq_len"]}')
    print(f'x shape={tuple(xb.shape)} dtype={xb.dtype} '
          f'y shape={tuple(yb.shape)} dtype={yb.dtype}')

    tr2, va2, m2 = load_smnist(batch_size=8)
    xb2, yb2 = next(iter(tr2))
    print(f'smnist x shape={tuple(xb2.shape)} dtype={xb2.dtype} '
          f'y shape={tuple(yb2.shape)} dtype={yb2.dtype} meta={m2}')
