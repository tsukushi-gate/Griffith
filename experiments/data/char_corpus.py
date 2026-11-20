"""teacherコーパス読み込み + tiny_shakespeare実データ検証セット。

data-free文字レベルKDの2つのデータソースを提供する:

    TeacherCorpusDataset    — gen_teacher_corpus.py が生成したteacherサンプル
                              (訓練データ。実テキストは一切含まない)。
    load_tiny_shakespeare_val — 実テキスト(tiny_shakespeare)の末尾を検証専用に
                              読み込む(訓練には使わない。BPC評価専用)。

**data-freeの主張が成立する理由**: 訓練ループが読むのは TeacherCorpusDataset
（teacherの自己回帰サンプル）のみ。tiny_shakespeareは `load_tiny_shakespeare_val`
経由でBPC計算のためだけに読み込まれ、勾配は一切流れない(training/train_kd_charlm_rddlgn.py
のeval関数内でのみ使用する)。
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

from griffith_logic.convert.text_teacher import SHAKESPEARE_CHAR_VOCAB, encode_shakespeare_char

__all__ = [
    'RealCharDataset',
    'TeacherCorpusDataset',
    'load_tiny_shakespeare_train',
    'load_tiny_shakespeare_val',
    'split_src_tgt',
]

# サンプルレイアウト(PROGRESS/設計ノート参照):
#   teacherは長さ64の文字列を生成する。
#     src        = chars[0:32]                       (エンコーダ文脈, 32文字)
#     tgt_input  = [BOS] + chars[32:63]               (デコーダ入力, 32トークン)
#     tgt_target = chars[32:64]                       (デコーダ正解, 32文字)
#   studentトークン空間 = teacher文字vocab(65) + PAD(65) + BOS(66) → vocab_size=67。
_WINDOW = 64
_SPLIT = 32  # src/tgt境界
PAD_ID = 65
BOS_ID = 66
STUDENT_VOCAB_SIZE = 67


def split_src_tgt(tokens_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """長さ64の1シーケンスを (src[32], tgt_input[32], tgt_target[32]) に分割する。

    tgt_inputはBOSを先頭に付加してtgt_targetと同じ長さ32を保つ
    (最後の1文字 chars[63] はtgt_inputに含めない = teacher forcingの入力ズレ)。

    Args:
        tokens_row: (64,) long tensor。

    Returns:
        (src[32], tgt_input[32], tgt_target[32])
    """
    assert tokens_row.shape[-1] == _WINDOW, f'長さ{_WINDOW}のシーケンスが必要: {tokens_row.shape}'
    tokens_row = tokens_row.long()
    src = tokens_row[:_SPLIT]
    tgt_target = tokens_row[_SPLIT:_WINDOW]
    bos = torch.tensor([BOS_ID], dtype=torch.long)
    tgt_input = torch.cat([bos, tokens_row[_SPLIT:_WINDOW - 1]], dim=0)
    return src, tgt_input, tgt_target


_DATASET_MODES = ('seq2seq', 'decoder_only')


class TeacherCorpusDataset(Dataset):
    """gen_teacher_corpus.py が生成した .pt コーパスを読み込むDataset。

    `__getitem__` は決定的(シャッフルなし。DataLoader側のshuffle=Trueで
    エポックごとのシャッフルを行う想定)に (src, tgt_input, tgt_target) を返す
    (mode='seq2seq'、既定・既存呼び出し元は無改造)。
    生トークン列(64文字)はteacher蒸留のためのteacherロジット計算にも必要なので
    `.raw_tokens(idx)` で別途取得できる。

    mode='decoder_only'(LogicGPT用)では `__getitem__` が
    (64,) の生トークン列をそのまま返す(src/tgt_input/tgt_target分割をしない。
    decoder-onlyモデルはinput=tokens[:-1], target=tokens[1:]を自前でスライスする)。
    """

    def __init__(self, path: str, mode: str = 'seq2seq'):
        assert mode in _DATASET_MODES, f'mode は {_DATASET_MODES} のいずれか: {mode}'
        self.mode = mode
        data = torch.load(path, weights_only=True)
        self.tokens = data['tokens'].long()  # (N, 64) int16 → long
        self.meta = data['meta']
        if self.tokens.shape[1] != _WINDOW:
            raise ValueError(
                f'想定シーケンス長{_WINDOW}と不一致: {self.tokens.shape}'
                '(gen_teacher_corpus.pyの--lengthを確認してください)'
            )

    @property
    def vocab_size(self) -> int:
        """teacher側の文字vocabサイズ(65。studentのPAD/BOSを含まない)。"""
        return 65

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int):
        if self.mode == 'decoder_only':
            return self.tokens[idx]
        return split_src_tgt(self.tokens[idx])

    def raw_tokens(self, idx: int) -> torch.Tensor:
        """teacherロジット計算用の生64トークン列を返す(BOS/PAD無し、teacher vocab)。"""
        return self.tokens[idx]


_TINY_SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
)


def _download_tiny_shakespeare(cache_path: Path) -> str:
    """tiny_shakespeareの生テキストを取得する(複数ソースをフォールバック)。

    優先順:
        1. datasets.load_dataset('karpathy/tiny_shakespeare')
           (2024年以降、HF datasetsライブラリがdataset scriptsを廃止したため
           このrepoはロードエラーになる場合がある。実測でRuntimeErrorを確認済み)
        2. huggingface_hub.hf_hub_download による既知ミラーからの取得
        3. urllib.request で karpathy/char-rnn の生input.txtを直接取得
           (実測で動作確認済み。1,115,394バイト)
    """
    if cache_path.exists():
        return cache_path.read_text(encoding='utf-8')

    text = None
    try:
        from datasets import load_dataset
        ds = load_dataset('karpathy/tiny_shakespeare')
        # スキーマはrepoによって異なりうるため、'text'列を素朴に結合する。
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        text = '\n'.join(ds[split]['text'])
        print('tiny_shakespeare: datasets.load_dataset経由で取得')
    except Exception as e:
        print(f'tiny_shakespeare: datasets経由の取得失敗({e!r})、フォールバックへ')

    if text is None:
        try:
            with urllib.request.urlopen(_TINY_SHAKESPEARE_URL, timeout=30) as response:
                text = response.read().decode('utf-8')
            print('tiny_shakespeare: urllib(karpathy/char-rnn生ファイル)経由で取得')
        except Exception as e:
            raise RuntimeError(
                'tiny_shakespeareの取得にすべての方式が失敗しました'
                f'(最終手段のurllibも失敗: {e!r})'
            ) from e

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding='utf-8')
    return text


def load_tiny_shakespeare_val(
    max_chars: int = 100_000,
    cache_dir: str = './data-charlm',
) -> torch.Tensor:
    """tiny_shakespeareの末尾max_chars文字を検証用ウィンドウ列として返す(EVAL専用)。

    **data-freeの主張はこの関数を訓練に使わないことで成立する**: ここで返す
    テンソルはBPC計算(teacher-forced NLL)専用であり、student訓練ループの
    勾配計算には一切使わない。訓練データはTeacherCorpusDataset(teacherの
    自己回帰サンプル)のみ。

    末尾max_chars文字を取る理由: 先頭側(train prefix)には触れないことで、
    仮にteacherや将来のstudentの訓練データがtiny_shakespeare先頭部分と
    重複していたとしても、この検証セットは独立性を保ちやすい
    (本パイプラインはdata-freeなので厳密な重複はそもそも起きないが、
    将来の拡張に備えた安全側の選択)。

    Args:
        max_chars: 検証に使う末尾文字数。
        cache_dir: ダウンロードキャッシュ先(相対パス、Modal無改造規約)。

    Returns:
        (N, 64) long tensor。65文字vocabに含まれない文字は除去済み
        (フィルタで落ちた文字数はログに出力)。
    """
    cache_path = Path(cache_dir) / 'tiny_shakespeare_input.txt'
    full_text = _download_tiny_shakespeare(cache_path)

    val_text = full_text[-max_chars:] if len(full_text) > max_chars else full_text

    # 65文字vocabに無い文字(タブ等の稀な制御文字)を除去する。
    vocab_set = set(SHAKESPEARE_CHAR_VOCAB)
    filtered_chars = [ch for ch in val_text if ch in vocab_set]
    n_dropped = len(val_text) - len(filtered_chars)
    if n_dropped > 0:
        print(f'tiny_shakespeare検証データ: vocab外文字を{n_dropped}文字除去')

    ids = encode_shakespeare_char(''.join(filtered_chars))

    n_windows = len(ids) // _WINDOW
    if n_windows == 0:
        raise ValueError(
            f'max_chars={max_chars}が短すぎてウィンドウ長{_WINDOW}を1つも作れません'
        )
    ids = ids[:n_windows * _WINDOW]
    tensor = torch.tensor(ids, dtype=torch.long).reshape(n_windows, _WINDOW)
    print(f'tiny_shakespeare検証データ: {n_windows}ウィンドウ({n_windows * _WINDOW}文字)')
    return tensor


def load_tiny_shakespeare_train(
    fraction: float,
    val_reserve_chars: int = 100_000,
    cache_dir: str = './data-charlm',
) -> torch.Tensor:
    """tiny_shakespeare実データの訓練用サブセットを1次元idテンソルとして返す。

    data-free路線(TeacherCorpusDataset)とは別の「実データKD/scratch」アーム用。
    `load_tiny_shakespeare_val`が末尾`val_reserve_chars`文字を検証専用に使うのに
    対し、こちらは**先頭側**のtrain pool(`full_text[:-val_reserve_chars]`)から
    さらに先頭`fraction`割合をprefixとして切り出す。これにより:

        - train poolとvalは常にdisjoint(重複ゼロ、末尾/先頭で完全分離)。
        - fraction=1%/5%/25%/100%は決定的にnested(1%は5%のprefix、5%は25%の
          prefix、...)。データ効率実験でfractionを変えても学習データの前方
          互換性が保たれる(小さいfractionのデータは常に大きいfractionに含まれる)。

    Args:
        fraction: train poolに対する取得割合(0 < fraction <= 1)。
        val_reserve_chars: 末尾を検証用に予約する文字数
            (load_tiny_shakespeare_valのmax_charsと揃えること。ずれるとdisjoint
            性が崩れる)。
        cache_dir: ダウンロードキャッシュ先(load_tiny_shakespeare_valと同じ規約)。

    Returns:
        (N,) long tensor(1次元、フラットなid列)。65文字vocabに含まれない文字は
        除去済み。ウィンドウ分割はRealCharDataset側で行う。
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f'fractionは(0, 1]の範囲: {fraction}')

    cache_path = Path(cache_dir) / 'tiny_shakespeare_input.txt'
    full_text = _download_tiny_shakespeare(cache_path)

    if val_reserve_chars >= len(full_text):
        raise ValueError(
            f'val_reserve_chars={val_reserve_chars}が全文長{len(full_text)}以上です'
        )
    pool_text = full_text[:-val_reserve_chars]

    n_chars = max(1, int(len(pool_text) * fraction))
    subset_text = pool_text[:n_chars]

    vocab_set = set(SHAKESPEARE_CHAR_VOCAB)
    filtered_chars = [ch for ch in subset_text if ch in vocab_set]
    n_dropped = len(subset_text) - len(filtered_chars)
    if n_dropped > 0:
        print(f'tiny_shakespeare訓練データ: vocab外文字を{n_dropped}文字除去')

    ids = encode_shakespeare_char(''.join(filtered_chars))
    tensor = torch.tensor(ids, dtype=torch.long)
    print(
        f'tiny_shakespeare訓練データ: fraction={fraction} → {len(tensor):,}文字'
        f'(pool={len(pool_text):,}文字, val_reserve={val_reserve_chars:,}文字)'
    )
    return tensor


class RealCharDataset(Dataset):
    """load_tiny_shakespeare_trainが返すidテンソルからstride=1のスライディング窓を切る Dataset。

    `TeacherCorpusDataset`と同一の`__getitem__`契約に合わせてあるため、
    train_kd_charlm_rddlgn.pyの訓練ループ側は無改造で動く:
        - mode='seq2seq'(既定): `split_src_tgt(ids[i:i+64])`を返す
          (src, tgt_input, tgt_target)の3-tuple。
        - mode='decoder_only': (64,)の生トークン窓をそのまま返す。

    TeacherCorpusDatasetとの違いはサンプルの切り出し方のみ:
    TeacherCorpusDatasetは事前生成された非重複サンプル列(.pt)を読むのに対し、
    RealCharDatasetは1本のid列からstride=1でオーバーラップする窓を動的に
    切り出す(単純な言語モデリング窓、`__len__ = len(ids) - 64 + 1`)。
    """

    def __init__(self, ids: torch.Tensor, mode: str = 'seq2seq'):
        assert mode in _DATASET_MODES, f'mode は {_DATASET_MODES} のいずれか: {mode}'
        if ids.dim() != 1:
            raise ValueError(f'idsは1次元テンソルが必要: shape={tuple(ids.shape)}')
        if len(ids) < _WINDOW:
            raise ValueError(
                f'idsの長さ{len(ids)}がウィンドウ長{_WINDOW}未満です'
                '(fractionが小さすぎる可能性があります)'
            )
        self.mode = mode
        self.ids = ids.long()

    @property
    def vocab_size(self) -> int:
        """teacher側の文字vocabサイズ(65。studentのPAD/BOSを含まない)。"""
        return 65

    def __len__(self) -> int:
        return len(self.ids) - _WINDOW + 1

    def __getitem__(self, idx: int):
        window = self.ids[idx:idx + _WINDOW]
        if self.mode == 'decoder_only':
            return window
        return split_src_tgt(window)

    def raw_tokens(self, idx: int) -> torch.Tensor:
        """teacherロジット計算用の生64トークン列を返す(BOS/PAD無し、teacher vocab)。"""
        return self.ids[idx:idx + _WINDOW]
