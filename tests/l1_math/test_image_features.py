"""L1: 固定低レベル特徴の二値符号化(Conv-DLGN §3.5 相当)の数式性質。

素朴なループ実装を ground truth として、ベクトル化実装と照合する。
"""

import pytest
import torch

from griffith_logic.encoding.image_features import (
    CHROMA_PAIRS,
    CURV_OFFSETS,
    EDGE_OFFSETS,
    channel_names,
    encode_image_features,
    n_output_channels,
    thermometer_bits,
)

FEATURES = dict(
    n_thresholds=3,
    edge_thresholds=(0.15, 0.35),
    curv_thresholds=(0.2,),
    chroma_thresholds=(0.1, 0.25),
)


def _naive_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """out[c,h,w] = x[c, clamp(h+dy), clamp(w+dx)] の素朴実装(replicate境界)。"""
    c, h, w = x.shape
    out = torch.empty_like(x)
    for i in range(h):
        for j in range(w):
            si = min(max(i + dy, 0), h - 1)
            sj = min(max(j + dx, 0), w - 1)
            out[:, i, j] = x[:, si, sj]
    return out


def _sample(seed: int = 0, c: int = 3, h: int = 7, w: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, w, generator=g)


def test_thermometer_matches_difflogic_definition():
    """しきい値は difflogic 公式の (i+1)/(n+1) と一致し、比較は厳密 > である。"""
    x = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0]]])
    bits = torch.cat(thermometer_bits(x, 3), dim=-3)
    # t = 0.25 / 0.5 / 0.75、境界値ちょうどは 0(> であって >= ではない)
    expected = torch.tensor([
        [[0.0, 0.0, 1.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 1.0]],
    ])
    assert torch.equal(bits, expected)


def test_output_is_binary_and_shape_matches_contract():
    x = _sample()
    out = encode_image_features(x, **FEATURES)
    assert out.shape == (n_output_channels(3, **FEATURES), 7, 5)
    assert out.shape[0] == len(channel_names(3, **FEATURES))
    assert torch.all((out == 0) | (out == 1))
    assert out.dtype == x.dtype


def test_batched_matches_unbatched():
    xs = torch.stack([_sample(0), _sample(1)])
    out = encode_image_features(xs, **FEATURES)
    for b in range(2):
        assert torch.equal(out[b], encode_image_features(xs[b], **FEATURES))


def test_edge_block_matches_naive_shift():
    """edge ブロックが x[p+d] - x[p] の符号付きしきい値と一致する。"""
    x = _sample()
    out = encode_image_features(x, n_thresholds=0, edge_thresholds=(0.15, 0.35))
    idx = 0
    for dy, dx in EDGE_OFFSETS:
        diff = _naive_shift(x, dy, dx) - x
        for t in (0.15, 0.35):
            assert torch.equal(out[idx:idx + 3], (diff > t).float())
            idx += 3
            assert torch.equal(out[idx:idx + 3], (diff < -t).float())
            idx += 3
    assert idx == out.shape[0]


def test_curvature_block_matches_naive_second_difference():
    x = _sample()
    out = encode_image_features(x, n_thresholds=0, curv_thresholds=(0.2,))
    idx = 0
    for dy, dx in CURV_OFFSETS:
        diff = _naive_shift(x, -dy, -dx) - 2.0 * x + _naive_shift(x, dy, dx)
        assert torch.equal(out[idx:idx + 3], (diff > 0.2).float())
        idx += 3
        assert torch.equal(out[idx:idx + 3], (diff < -0.2).float())
        idx += 3
    assert idx == out.shape[0]


def test_chroma_block_matches_channel_differences():
    x = _sample()
    thr = (0.1, 0.25)
    out = encode_image_features(x, n_thresholds=0, chroma_thresholds=thr)
    idx = 0
    for i, j in CHROMA_PAIRS:
        diff = x[i] - x[j]
        for t in thr:
            assert torch.equal(out[idx], (diff > t).float())
            idx += 1
            assert torch.equal(out[idx], (diff < -t).float())
            idx += 1
    sat = x.max(dim=0).values - x.min(dim=0).values
    for t in thr:
        assert torch.equal(out[idx], (sat > t).float())
        idx += 1
    assert idx == out.shape[0]


def test_edge_and_curvature_are_invariant_to_global_brightness_shift():
    """★論文が edge/curvature を入れる理由: 差分特徴は一様な明度変化に不変。

    サーモメータ(絶対輝度)は同じシフトで壊れることも同時に示す。
    """
    x = _sample() * 0.5 + 0.2
    shifted = x + 0.15
    kw = dict(n_thresholds=0, edge_thresholds=(0.15,), curv_thresholds=(0.2,))
    assert torch.equal(encode_image_features(x, **kw), encode_image_features(shifted, **kw))

    thermo = dict(n_thresholds=3)
    assert not torch.equal(
        encode_image_features(x, **thermo), encode_image_features(shifted, **thermo)
    )


def test_chroma_is_invariant_to_global_brightness_shift():
    """チャネル間比較も加法的な明度変化に不変(彩度も同様)。"""
    x = _sample() * 0.5 + 0.2
    kw = dict(n_thresholds=0, chroma_thresholds=(0.1, 0.25))
    assert torch.equal(encode_image_features(x, **kw), encode_image_features(x + 0.15, **kw))


def test_thermometer_only_equals_existing_threshold_binarize():
    """既存 ThresholdBinarize(experiments 側)と thr-major 順まで含めて一致する。"""
    from experiments.data.detection import ThresholdBinarize

    x = _sample()
    assert torch.equal(
        encode_image_features(x, n_thresholds=3), ThresholdBinarize(3)(x)
    )


def test_all_blocks_disabled_raises():
    with pytest.raises(ValueError):
        encode_image_features(_sample(), n_thresholds=0)


def test_chroma_requires_three_channels():
    with pytest.raises(ValueError, match='3チャネル'):
        encode_image_features(_sample(c=4), n_thresholds=0, chroma_thresholds=(0.1,))
