"""時間的信用割当(temporal credit assignment)を要する合成タスク。

copyタスクだけでは「時間方向の依存が必要」であることを保証できないため、
先頭情報の長期保持(delayed recall)と2時点間の非線形結合(temporal XOR)を
強制する2タスクを追加する。いずれもseq2seq形式・単一トークン出力で、
src系列長 seq_len が「遅延長T」に対応する。

CopyDataset と同じ (vocab_size, seq_len, num_samples) 引数形式に合わせる。

トークン割当(SPECIAL_TOKENS=3 のPAD/BOS/EOSの後):
  delayed_recall: filler=3、想起シンボル ∈ [4, 4+n_symbols)
  temporal_xor:   BIT0=3, BIT1=4, FILLER=5、出力 ∈ {BIT0, BIT1}
"""

import torch
from torch.utils.data import Dataset

from .synthetic import SPECIAL_TOKENS

# delayed_recall
_RECALL_FILLER = SPECIAL_TOKENS          # 3
_RECALL_SYMBOL_BASE = SPECIAL_TOKENS + 1  # 4
_DEFAULT_N_SYMBOLS = 4                     # 想起シンボル数(chance=1/n_symbols)

# temporal_xor
_XOR_BIT0 = SPECIAL_TOKENS       # 3 → bit 0
_XOR_BIT1 = SPECIAL_TOKENS + 1   # 4 → bit 1
_XOR_FILLER = SPECIAL_TOKENS + 2  # 5


class DelayedRecallDataset(Dataset):
    """系列先頭のシンボルを最終ステップで答えるタスク。

    src = [symbol, filler, filler, ..., filler]  (長さ seq_len)
    tgt = [symbol]                                (長さ1)

    先頭symbolをT-1個のfillerを跨いでcontextに保持することを強制する。
    filler固定なので難易度の交絡(distractor識別)を排し、遅延長Tのみを変数化する。
    seq_len=1 は遅延ゼロ(自明)なので、遅延を要求するには seq_len>=2 を想定。
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        n_symbols: int = _DEFAULT_N_SYMBOLS,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.n_symbols = min(n_symbols, vocab_size - _RECALL_SYMBOL_BASE)
        if self.n_symbols < 2:
            raise ValueError(
                f'vocab_size={vocab_size}が小さすぎる(想起シンボル数<2)'
            )
        symbols = torch.randint(
            _RECALL_SYMBOL_BASE,
            _RECALL_SYMBOL_BASE + self.n_symbols,
            (num_samples,),
        )
        src = torch.full((num_samples, seq_len), _RECALL_FILLER, dtype=torch.long)
        src[:, 0] = symbols
        self.src = src
        self.tgt = symbols.unsqueeze(1)  # (num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


class TemporalXORDataset(Dataset):
    """先頭と末尾の2ビットのXORを最終ステップで答えるタスク。

    src = [bit_a, filler, ..., filler, bit_b]  (長さ seq_len, seq_len>=2)
    tgt = [XOR(bit_a, bit_b)]                   (長さ1、BIT0/BIT1トークン)

    bit_a を系列全体に跨いで保持し、末尾 bit_b と非線形結合(XOR)することを強制する。
    2時点間の依存を必須とするため、単純な最終step依存では chance(50%) を超えない。
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
    ):
        if vocab_size <= _XOR_FILLER:
            raise ValueError(
                f'temporal_xorは vocab_size>{_XOR_FILLER} が必要: {vocab_size}'
            )
        if seq_len < 2:
            raise ValueError(f'temporal_xorは seq_len>=2 が必要: {seq_len}')
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        bit_a = torch.randint(0, 2, (num_samples,))
        bit_b = torch.randint(0, 2, (num_samples,))
        xor = bit_a ^ bit_b

        src = torch.full(
            (num_samples, seq_len), _XOR_FILLER, dtype=torch.long
        )
        src[:, 0] = _XOR_BIT0 + bit_a
        src[:, -1] = _XOR_BIT0 + bit_b
        self.src = src
        self.tgt = (_XOR_BIT0 + xor).unsqueeze(1)  # (num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
