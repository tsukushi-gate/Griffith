"""画像用の固定(非学習)低レベル特徴の二値符号化。

Conv-DLGN(Petersen et al., NeurIPS 2024) §3.5 Input Processing に対応する前処理:

    "For our larger CIFAR-10 models (B, L, G), we use 5 bit precision inputs, and
     process them with low-level feature detectors, in particular, we use edge and
     curvature detector kernels with thresholds, converting them into binary
     encodings, which are converted into LGNs and not learned."

論文の小モデル(S/M、CIFAR-10 60.38%/71.01%)は 3閾値サーモメータ(=2bit精度)のみ、
大モデル(B以降、80.17%〜)は 5bit精度 + edge/curvature 検出器を併用する。
本モジュールは後者に相当する特徴バンクを提供する。検出器は**学習しない固定カーネル**で、
出力は全て {0,1}(将来 IR/TS へそのまま写せるよう、素朴な差分とスカラ比較のみで構成)。

カラー検出タスク向けに、論文の edge/curvature に加えて chroma(チャネル間比較)ブロックを
持つ。数字が鮮やか色・背景が自然画像という設定では「絶対輝度のしきい値」より
「チャネル間の大小関係」の方が前景を分離するため。

チャネル順の契約(export/TS ミラー時に厳守すること):

    [thermo] [edge] [curv] [chroma]

各ブロック内の順序は下の `*_channel_names` が返すリストが唯一の真実源。
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

# 1階差分(edge)カーネルの方向。(dy, dx) = 比較先ピクセルの相対位置。
# 差分は x[p + (dy,dx)] - x[p] で定義する。
EDGE_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),   # 水平(右隣との差)
    (1, 0),   # 垂直(下隣との差)
    (1, 1),   # 斜め ↘
    (1, -1),  # 斜め ↙
)

# 2階差分(curvature)カーネルの方向。x[p-(dy,dx)] - 2·x[p] + x[p+(dy,dx)]
CURV_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),   # 水平
    (1, 0),   # 垂直
    (1, 1),   # 斜め ↘
)

# chroma のチャネル対(RGB前提)。(i, j) は x_i - x_j を意味する。
CHROMA_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """(..., H, W) を (dy, dx) だけずらす。境界は replicate。

    戻り値の位置 p には元テンソルの p + (dy, dx) の値が入る。
    """
    left, right = max(-dx, 0), max(dx, 0)
    top, bottom = max(-dy, 0), max(dy, 0)
    shape = x.shape
    h, w = shape[-2], shape[-1]
    flat = x.reshape(-1, 1, h, w)
    padded = F.pad(flat, (left, right, top, bottom), mode='replicate')
    # padded[k] は元の k-left に対応するので、out[p]=x[p+d] の読み出し開始は d+left
    y0, x0 = dy + top, dx + left
    out = padded[..., y0:y0 + h, x0:x0 + w]
    return out.reshape(shape)


def _signed_bits(diff: torch.Tensor, thresholds: Sequence[float]) -> list[torch.Tensor]:
    """符号付き差分を (diff > t) と (diff < -t) の対に符号化する。"""
    bits = []
    for t in thresholds:
        bits.append((diff > t).to(diff.dtype))
        bits.append((diff < -t).to(diff.dtype))
    return bits


def thermometer_bits(x: torch.Tensor, n_thresholds: int) -> list[torch.Tensor]:
    """均一しきい値サーモメータ。t_i = (i+1)/(n+1)。

    difflogic 公式実装の 'cifar-10-{3,31}-thresholds' と同一定義
    (n=3 → 0.25/0.5/0.75、n=31 → (i+1)/32)。既存の
    ``experiments/data/detection.py::ThresholdBinarize`` と同じ **thr-major** 順で返す。
    """
    return [
        (x > (i + 1) / (n_thresholds + 1)).to(x.dtype)
        for i in range(n_thresholds)
    ]


def edge_bits(x: torch.Tensor, thresholds: Sequence[float]) -> list[torch.Tensor]:
    """1階差分(edge)検出器。方向 × しきい値 × 符号 の順に並べる。"""
    bits: list[torch.Tensor] = []
    for dy, dx in EDGE_OFFSETS:
        diff = _shift(x, dy, dx) - x
        bits.extend(_signed_bits(diff, thresholds))
    return bits


def curvature_bits(x: torch.Tensor, thresholds: Sequence[float]) -> list[torch.Tensor]:
    """2階差分(curvature)検出器。方向 × しきい値 × 符号 の順に並べる。"""
    bits: list[torch.Tensor] = []
    for dy, dx in CURV_OFFSETS:
        diff = _shift(x, -dy, -dx) - 2.0 * x + _shift(x, dy, dx)
        bits.extend(_signed_bits(diff, thresholds))
    return bits


def chroma_bits(x: torch.Tensor, thresholds: Sequence[float]) -> list[torch.Tensor]:
    """チャネル間比較(色相の粗い符号化)+ 彩度(max−min)。

    x は (..., C, H, W) でチャネル次元が dim=-3 であること。C=3(RGB)を前提とする。
    """
    if x.shape[-3] != 3:
        raise ValueError(f'chroma は3チャネル(RGB)専用: C={x.shape[-3]}')
    bits: list[torch.Tensor] = []
    for i, j in CHROMA_PAIRS:
        diff = x[..., i, :, :] - x[..., j, :, :]
        # チャネル方向を潰した (..., H, W) なので、後段で unsqueeze して揃える
        bits.extend(b.unsqueeze(-3) for b in _signed_bits(diff, thresholds))
    saturation = x.max(dim=-3).values - x.min(dim=-3).values
    for t in thresholds:
        bits.append((saturation > t).to(x.dtype).unsqueeze(-3))
    return bits


def encode_image_features(
    x: torch.Tensor,
    *,
    n_thresholds: int = 3,
    edge_thresholds: Sequence[float] = (),
    curv_thresholds: Sequence[float] = (),
    chroma_thresholds: Sequence[float] = (),
) -> torch.Tensor:
    """固定低レベル特徴を二値チャネルとして連結する。

    Args:
        x: (C, H, W) または (B, C, H, W) の [0,1] 実数画像
        n_thresholds: サーモメータのしきい値数(0で無効)
        edge_thresholds: 1階差分のしきい値(空で無効)
        curv_thresholds: 2階差分のしきい値(空で無効)
        chroma_thresholds: チャネル間比較・彩度のしきい値(空で無効、C=3限定)

    Returns:
        x と同じ dtype/device の {0,1} テンソル。チャネル数は
        ``n_output_channels()`` と一致する。
    """
    blocks: list[torch.Tensor] = []
    if n_thresholds > 0:
        blocks.extend(thermometer_bits(x, n_thresholds))
    if edge_thresholds:
        blocks.extend(edge_bits(x, edge_thresholds))
    if curv_thresholds:
        blocks.extend(curvature_bits(x, curv_thresholds))
    if chroma_thresholds:
        blocks.extend(chroma_bits(x, chroma_thresholds))
    if not blocks:
        raise ValueError('少なくとも1ブロックを有効にすること')
    return torch.cat(blocks, dim=-3)


def n_output_channels(
    n_rgb: int,
    *,
    n_thresholds: int = 3,
    edge_thresholds: Sequence[float] = (),
    curv_thresholds: Sequence[float] = (),
    chroma_thresholds: Sequence[float] = (),
) -> int:
    """encode_image_features の出力チャネル数(テンソルを作らずに求める)。"""
    n = n_rgb * n_thresholds
    n += n_rgb * len(EDGE_OFFSETS) * len(edge_thresholds) * 2
    n += n_rgb * len(CURV_OFFSETS) * len(curv_thresholds) * 2
    n += len(CHROMA_PAIRS) * len(chroma_thresholds) * 2 + len(chroma_thresholds)
    return n


def channel_names(
    n_rgb: int,
    *,
    n_thresholds: int = 3,
    edge_thresholds: Sequence[float] = (),
    curv_thresholds: Sequence[float] = (),
    chroma_thresholds: Sequence[float] = (),
) -> list[str]:
    """出力チャネルの意味を並び順どおりに返す(export/TS ミラーの契約)。"""
    names: list[str] = []
    for i in range(n_thresholds):
        names.extend(f'thermo[t={(i + 1) / (n_thresholds + 1):.4g}]c{c}' for c in range(n_rgb))
    for dy, dx in EDGE_OFFSETS:
        for t in edge_thresholds:
            for sign in ('+', '-'):
                names.extend(f'edge[{dy},{dx}][{sign}{t:g}]c{c}' for c in range(n_rgb))
    for dy, dx in CURV_OFFSETS:
        for t in curv_thresholds:
            for sign in ('+', '-'):
                names.extend(f'curv[{dy},{dx}][{sign}{t:g}]c{c}' for c in range(n_rgb))
    for i, j in CHROMA_PAIRS:
        for t in chroma_thresholds:
            names.extend([f'chroma[c{i}-c{j}>{t:g}]', f'chroma[c{i}-c{j}<-{t:g}]'])
    names.extend(f'sat[>{t:g}]' for t in chroma_thresholds)
    return names
