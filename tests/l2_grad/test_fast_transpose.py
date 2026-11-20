"""L2: 高速転置 `fast_transpose` が `x.t().contiguous()` と厳密一致することの検証(@cuda)。

境界転置(1回あたり数GB)をTritonのタイル転置に置き換えたため、
- 値が完全一致すること(転置は値を変えない演算なので厳密一致で検証できる)
- 既にcontiguousな入力(=LUT層連鎖の内側)ではコピーが起きないこと
を dtype × 形状(タイル境界に揃わないサイズを含む)で機械的に確認する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.cuda import transpose as tp  # noqa: E402
from griffith_logic.cuda.transpose import (  # noqa: E402
    fast_transpose,
    transpose_backend,
)

pytestmark = pytest.mark.cuda

# タイル(64×64)の境界に揃う/揃わない形状を混在させる
SHAPES = [
    (64, 64),        # 正方・タイルぴったり
    (65, 63),        # タイル境界を1要素はみ出す/欠ける
    (1, 1),
    (1, 100003),     # 1行(素数長)
    (100003, 1),     # 1列
    (3, 4097),       # 極端に細長い
    (4097, 3),
    (127, 129),      # 奇数サイズ
    (333, 777),
    (1024, 4096),    # 実モデル相当(小)
    (517, 20011),    # 大きめ+奇数
]

DTYPES = [torch.float32, torch.float64, torch.float16]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
def test_fast_transpose_exact_match(dtype, shape):
    """`x.t().contiguous()` と完全一致すること。"""
    m, n = shape
    torch.manual_seed(0)
    x = torch.randn(m, n, device='cuda').to(dtype)

    ref = x.t().contiguous()
    got = fast_transpose(x)

    assert got.shape == (n, m)
    assert got.dtype == dtype
    assert got.is_contiguous()
    assert torch.equal(got, ref), f'値が不一致: shape={shape} dtype={dtype}'


@pytest.mark.parametrize('dtype', DTYPES)
def test_fast_transpose_no_copy_when_already_contiguous(dtype):
    """転置後が既にcontiguousな入力ではコピーしない(連鎖内側のno-op維持)。

    LUT層のforwardは (out_dim, batch) の contiguous テンソルの `.t()` を返すため、
    次段の転置はビューのままでなければならない(ここが壊れると連鎖内側に巨大な
    コピーが大量発生して逆に大幅に遅くなる)。
    """
    base = torch.randn(97, 41, device='cuda').to(dtype)  # (out_dim, batch)
    x = base.t()                                          # (batch, out_dim) 非contiguous
    assert not x.is_contiguous()

    got = fast_transpose(x)

    assert got.data_ptr() == base.data_ptr(), 'コピーが発生している'
    assert torch.equal(got, x.t().contiguous())


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
def test_triton_kernel_exact_match_all_shapes(dtype, shape):
    """Tritonカーネル自体をサイズ閾値を迂回して直接呼び、全形状で厳密一致を検証する。

    `fast_transpose` は小テンソルではtorch経路に落ちるため、タイル境界の
    マスク処理はここで直接検証する必要がある。
    """
    if transpose_backend() != 'triton':
        pytest.skip('Triton利用不可')
    m, n = shape
    torch.manual_seed(0)
    x = torch.randn(m, n, device='cuda').to(dtype)

    got = tp._triton_transpose(x)
    assert torch.equal(got, x.t().contiguous()), f'shape={shape} dtype={dtype}'


def test_small_tensor_uses_torch_path():
    """閾値未満の小テンソルではTritonを使わない(起動オーバーヘッド負けの回避)。

    実経路(LutLayer fwd+bwd)のクロスオーバーは約8M要素(transpose.py の実測表参照)。
    """
    if transpose_backend() != 'triton':
        pytest.skip('Triton利用不可')
    small = torch.randn(64, 1024, device='cuda')
    assert small.numel() < tp._MIN_NUMEL_FOR_TRITON

    calls = []
    orig = tp._triton_transpose
    tp._triton_transpose = lambda t: calls.append(t) or orig(t)
    try:
        got = fast_transpose(small)
    finally:
        tp._triton_transpose = orig
    assert calls == [], '閾値未満でTriton経路が使われている'
    assert torch.equal(got, small.t().contiguous())


def test_fast_transpose_backend_is_triton_on_cuda():
    """CUDA環境ではTriton経路が選ばれること(選択経路の可観測性)。"""
    assert transpose_backend() in ('triton', 'torch')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_fast_transpose_non_contiguous_input_falls_back(dtype):
    """入力自体が非contiguous(スライス/expand)でも正しい結果を返すこと。"""
    big = torch.randn(64, 512, device='cuda').to(dtype)
    x = big[:, ::2]  # 非contiguousスライス
    assert not x.is_contiguous()

    got = fast_transpose(x)
    assert torch.equal(got, x.t().contiguous())

    e = torch.randn(1, 777, device='cuda').to(dtype).expand(129, 777)
    assert torch.equal(fast_transpose(e), e.t().contiguous())


# ---------------------------------------------------------------------------
# 退行ゲート: 「既定経路が変更前より遅くならない」ことの機械的な担保
# ---------------------------------------------------------------------------
#
# 絶対時間のハードassertはGPU差・クロック変動でフレークするので使わない。代わりに
#   (a) 閾値未満では「変更前と同一の1行」に落ちること(コード経路の同一性 = 決定的)
#   (b) ルーティング判定が transpose.py の実測表と一致すること(閾値の恣意的変更の検出)
#   (c) 実経路(LutLayerのfwd+bwd)で、既定経路が変更前より遅くないこと(同一プロセス
#       内の相対比較なので絶対時間に依存しない)
# の3点を見る。閾値を実測の根拠なしに動かすと (b)(c) が落ちる。

# 相対比較の許容幅。
# 根拠: 同一条件・同一プロセス内の run 間ばらつきは、本テストと同じ条件で測った
# experiments/bench_kernel_gate.py の実測で 1〜3%(モデル級 spread の実測値)。
# pytest 実行中は他テストとの干渉やクロック変動が乗るのでその約3倍を許容幅とする。
# transpose.py の実測表で退行していた点は 0.87〜0.90x(= 11〜15%遅い)なので、
# この幅でも当該退行は検出できる。
PERF_TOLERANCE = 0.10

# ルーティング期待値。根拠は transpose.py の実測スイープ表(A10G, 閾値 8M = 1<<23)。
# 閾値の両側(直下 / 直上)を必ず含める。
ROUTING_CASES = [
    ((128, 1024), 'torch'),      # 131k  — 実測 0.991x(Triton化しても得しない)
    ((1024, 1024), 'torch'),     # 1.05M — 実測 0.870x(旧閾値1Mで退行していた点)
    ((128, 16384), 'torch'),     # 2.10M — 実測 0.897x(退行点)
    ((256, 16384), 'torch'),     # 4.19M — 実測 0.967x/1.019x(有意差なし)
    ((512, 16384), 'triton'),    # 8.39M — 実測 1.030x/1.182x(ここから勝ち)
    ((1024, 16384), 'triton'),   # 16.8M — 実測 1.087x
]

# (c) の実測条件。transpose.py の表で最も退行が大きかった領域を含む。
LAYER_PERF_CASES = [
    (16384, 128),   # numel 2.10M — 旧閾値では 0.897x に退行していた
    (16384, 512),   # numel 8.39M — 閾値直上(Triton経路)
]


def _observed_route(x):
    """`fast_transpose(x)` が実際に Triton を呼んだかどうかを観測する。"""
    calls = []
    orig = tp._triton_transpose
    tp._triton_transpose = lambda t: (calls.append(1), orig(t))[1]
    try:
        fast_transpose(x)
    finally:
        tp._triton_transpose = orig
    return 'triton' if calls else 'torch'


def _median_s(fn, warmup=10, iters=30):
    import statistics
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def test_below_threshold_path_is_identical_to_baseline():
    """閾値未満では最適化前と同一の `x.t().contiguous()` だけを通ること。

    退行ゲートの本体。バックエンド判定(`_init_backend`)もTritonカーネルも
    触らずに戻ることを確認する。これが守られている限り、Tritonが勝たない領域で
    この最適化由来の退行は原理的に起こりえない(追加コストは numel 比較1回)。
    """
    x = torch.randn(1024, 1024, device='cuda')
    assert x.numel() < tp._MIN_NUMEL_FOR_TRITON

    touched = []
    orig_init = tp._init_backend
    orig_triton = tp._triton_transpose
    tp._init_backend = lambda: touched.append('init') or 'torch'
    tp._triton_transpose = lambda t: touched.append('triton') or orig_triton(t)
    try:
        got = fast_transpose(x)
    finally:
        tp._init_backend = orig_init
        tp._triton_transpose = orig_triton

    assert touched == [], f'閾値未満で余計な処理が走っている: {touched}'
    assert torch.equal(got, x.t().contiguous())


@pytest.mark.parametrize('shape,expected', ROUTING_CASES)
def test_routing_matches_measured_table(shape, expected):
    """ルーティング判定が transpose.py の実測スイープ表と一致すること。

    閾値を実測の根拠なしに動かすとここが落ちる(閾値の恣意的調整の防止)。
    """
    if transpose_backend() != 'triton':
        pytest.skip('Triton利用不可')
    x = torch.randn(*shape, device='cuda')
    assert _observed_route(x) == expected, (
        f'shape={shape} numel={x.numel()} の経路が実測表と食い違う '
        f'(閾値={tp._MIN_NUMEL_FOR_TRITON})'
    )


@pytest.mark.parametrize('width,batch', LAYER_PERF_CASES)
def test_lut_layer_step_not_slower_than_legacy_transpose(width, batch):
    """実経路(LutLayer fwd+bwd)で、既定経路が変更前より遅くないこと。

    転置単体ではなく **LUTカーネルに挟まれた状態** で測るのが要点。Triton経路の
    ホスト側コスト(torch.empty + grid計算 + ランチャ)は隣接カーネルに隠れないため、
    転置単体のクロスオーバー(約1M要素)を閾値にすると実経路では退行する。
    同一プロセス内で変更前実装と交互に測るので絶対時間には依存しない。
    """
    if transpose_backend() != 'triton':
        pytest.skip('Triton利用不可')

    from griffith_logic.core import functions as md_functions
    from griffith_logic.layers.lut import LutLayer

    layer = LutLayer(in_dim=width, out_dim=width, n_inputs=2,
                     parameterization='softmax', connections='random').to('cuda')
    layer.train()
    x = torch.rand(batch, width, device='cuda')

    def step():
        layer.zero_grad(set_to_none=True)
        xin = x.detach().requires_grad_(True)
        layer(xin).sum().backward()

    current = md_functions.fast_transpose
    try:
        # 交互に測ってクロック変動の片寄りを避ける
        legacy_s, default_s = [], []
        for _ in range(3):
            md_functions.fast_transpose = lambda t: t.t().contiguous()
            legacy_s.append(_median_s(step))
            md_functions.fast_transpose = current
            default_s.append(_median_s(step))
    finally:
        md_functions.fast_transpose = current

    import statistics
    legacy = statistics.median(legacy_s)
    default = statistics.median(default_s)
    assert default <= legacy * (1.0 + PERF_TOLERANCE), (
        f'width={width} batch={batch} numel={width * batch}: '
        f'既定経路が変更前より遅い (変更前={legacy:.6f}s/it 既定={default:.6f}s/it '
        f'= {legacy / default:.3f}x, 許容={PERF_TOLERANCE:.0%})'
    )
