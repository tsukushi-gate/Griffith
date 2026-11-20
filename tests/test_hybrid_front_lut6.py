"""HYBRID-3: 後段LUTのN入力一般化(N=2/4/6)の配線・構造テスト。

`experiments/train_hybrid_front.py` の expand-fill 配線は元々 N=2 前提の
「ポート数=2k」で書かれていた。HYBRID-3 で N·k に一般化したので、
その不変条件(全被覆・fan-out均等・重複ポート解消・プール範囲)が
N=4/6 でも保たれることを確認する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


def _mod():
    pytest.importorskip('torchvision')
    import train_hybrid_front

    return train_hybrid_front


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_fanout2_covers_all_and_is_balanced(n_inputs):
    """fanout2: 全前層出力が最低1回使われ、fan-outが最大均等になる。"""
    m = _mod()
    prev_width, k = 64, 80
    rng = torch.Generator().manual_seed(0)
    conn = m.build_fanout2_connections(prev_width, k, rng, n_inputs)

    ports = n_inputs * k
    assert conn.shape == (n_inputs, k)
    cnt = torch.bincount(conn.flatten(), minlength=prev_width)
    assert int(cnt.min()) >= 1, '未使用の前層出力がある'
    # 最大均等: floor(ports/P) 〜 ceil(ports/P) の範囲に収まる
    assert int(cnt.min()) >= ports // prev_width
    assert int(cnt.max()) <= -(-ports // prev_width)
    assert int(cnt.sum()) == ports


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_bypass_covers_prev_and_stays_in_pool(n_inputs):
    """bypass: 前層を全被覆し、余りは補助プール内のインデックスに収まる。"""
    m = _mod()
    prev_width, aux_width, k = 80, 64, 160
    rng = torch.Generator().manual_seed(0)
    conn = m.build_bypass_connections(prev_width, aux_width, k, rng, n_inputs)

    ports = n_inputs * k
    assert conn.shape == (n_inputs, k)
    assert int(conn.min()) >= 0
    assert int(conn.max()) < prev_width + aux_width, 'プール範囲外の配線'
    cnt = torch.bincount(conn.flatten(), minlength=prev_width + aux_width)
    assert int(cnt[:prev_width].min()) >= 1, '未使用の前層出力がある'
    assert int(cnt[prev_width:].sum()) == ports - prev_width


@pytest.mark.parametrize('n_inputs', [4, 6])
def test_fix_self_pairs_removes_duplicate_ports(n_inputs):
    """N>2 でも同一ゲート内の重複ポートが解消される(実質fan-in低下の防止)。"""
    m = _mod()
    prev_width, k = 64, 80
    rng = torch.Generator().manual_seed(0)
    conn = m.build_fanout2_connections(prev_width, k, rng, n_inputs)
    # 各ゲートの入力元はすべて相異なる
    n_unique = torch.tensor([len(set(conn[:, g].tolist())) for g in range(k)])
    assert int(n_unique.min()) == n_inputs, '重複ポートが残っている'


def test_fix_self_pairs_n2_is_backward_compatible():
    """N=2 は従来どおり slot1 のみを見る挙動(乱数消費も含め)を保つ。

    従来実装と同じ手順を素朴に再現し、同じシードから同じ配線が出ることで
    HYBRID-1 の基準値との比較可能性を担保する。
    """
    m = _mod()

    def legacy_fix(conn, rng, max_tries=200):
        k = conn.shape[1]
        for _ in range(max_tries):
            dup = (conn[0] == conn[1]).nonzero().flatten()
            if dup.numel() == 0:
                return conn
            partner = torch.randint(0, k, (dup.numel(),), generator=rng,
                                    dtype=torch.int64)
            tmp = conn[1, dup].clone()
            conn[1, dup] = conn[1, partner]
            conn[1, partner] = tmp
        return conn

    base = torch.arange(64, dtype=torch.int64).repeat(3)[:160].reshape(2, 80)
    got = m._fix_self_pairs(base.clone(), torch.Generator().manual_seed(7))
    want = legacy_fix(base.clone(), torch.Generator().manual_seed(7))
    assert torch.equal(got, want)


@pytest.mark.parametrize('n_inputs', [2, 6])
def test_lut6_table_size_scales_as_2_pow_n(n_inputs):
    """LUT1個あたりのパラメータ数が 2^N (N=2はdifflogic互換の16) になる。"""
    m = _mod()
    from griffith_logic import LutLayer

    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 8],
        dlgn_widths=[20, 40], n_inputs=n_inputs, expand_fill='bypass', seed=0,
    )
    luts = [x for x in model.modules() if isinstance(x, LutLayer)]
    total_gates = sum(x.num_neurons for x in luts)
    lut_params = sum(x.weights.numel() for x in luts)
    expected_entries = 16 if n_inputs == 2 else 2 ** n_inputs
    assert lut_params == total_gates * expected_entries


def test_lut6_forward_shape_and_binary_interface():
    """N=6 でも順伝播が通り、界面が厳密に0/1になる。"""
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 8],
        dlgn_widths=[20, 40], n_inputs=6, expand_fill='bypass', seed=0,
    )
    x = torch.rand(4, 64)
    assert model(x).shape == (4, 10)
    bits = model.binarize(model.front(model.flatten(x)))
    assert bool(((bits == 0) | (bits == 1)).all())


# ============================================================
# HYBRID-4: 均等被覆配線 / 層間再二値化 / 第1層配線学習
# ============================================================


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
@pytest.mark.parametrize(('prev_width', 'k'), [(120, 120), (120, 240), (240, 480)])
def test_balanced_is_distinct_and_uniform(n_inputs, prev_width, k):
    """均等被覆: ゲート内は全相異、前層出力の使用回数は max-min <= 1。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    conn = m.build_balanced_connections(prev_width, k, rng, n_inputs)

    assert conn.shape == (n_inputs, k)
    # 制約1: 各ゲートのN本が相異なる
    n_unique = torch.tensor([len(set(conn[:, g].tolist())) for g in range(k)])
    assert int(n_unique.min()) == n_inputs, 'ゲート内に重複入力がある'
    # 制約2: fan-outが均等
    cnt = torch.bincount(conn.flatten(), minlength=prev_width)
    assert int(cnt.min()) >= 1, '使われない前層出力がある'
    assert int(cnt.max()) - int(cnt.min()) <= 1, 'fan-outが均等でない'
    assert int(cnt.sum()) == n_inputs * k


def test_balanced_is_exactly_uniform_when_divisible():
    """N·k が前層幅の倍数(6の倍数系)なら全出力が厳密に同じ回数使われる。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    conn = m.build_balanced_connections(120, 240, rng, 6)  # 1440 = 120 * 12
    cnt = torch.bincount(conn.flatten(), minlength=120)
    assert int(cnt.min()) == int(cnt.max()) == 12


def test_balanced_rejects_too_narrow_prev_layer():
    """前層幅 < N ではゲート内を相異なる入力で埋められないので弾く。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(AssertionError):
        m.build_balanced_connections(4, 100, rng, 6)


def test_interlayer_binarize_outputs_are_binary_in_both_modes():
    """層間再二値化: 各LUT層出力が train/eval とも厳密に0/1。"""
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 12],
        dlgn_widths=[12, 24, 60], n_inputs=6, connections_mode='balanced',
        interlayer_binarize=True, seed=0,
    )
    x = torch.rand(8, 64)
    for mode in (True, False):
        model.train(mode=mode)
        with torch.no_grad():
            bits = model.binarize(model.front(model.flatten(x)))
            h = bits
            for i, layer in enumerate(model.dlgn):
                h = model.inter_binarize(model.inter_bn[i](layer(h)))
                uniq = torch.unique(h)
                assert bool(((uniq == 0) | (uniq == 1)).all()), (mode, i, uniq)
    model.train(mode=False)
    assert model(x).shape == (8, 10)


def test_interlayer_binarize_bn_init_follows_binarybrain():
    """BN初期値は BinaryBrain 既定の gamma=0.3 / beta=0.5。"""
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 12],
        dlgn_widths=[12, 60], n_inputs=6, connections_mode='balanced',
        interlayer_binarize=True, seed=0,
    )
    for bn in model.inter_bn:
        assert torch.allclose(bn.weight, torch.full_like(bn.weight, 0.3))
        assert torch.allclose(bn.bias, torch.full_like(bn.bias, 0.5))


def test_layer0_learnable_wiring_only_affects_first_layer():
    """DWN流配線学習: 第1層だけ learnable、隠れ層は fixed のまま。"""
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 12],
        dlgn_widths=[12, 24, 60], n_inputs=6, connections_mode='balanced',
        dlgn_layer0_wiring='learnable', seed=0,
    )
    assert model.dlgn[0].wiring == 'learnable'
    assert hasattr(model.dlgn[0], 'learnable_mapping')
    for layer in list(model.dlgn)[1:]:
        assert layer.wiring == 'fixed'
        assert not hasattr(layer, 'learnable_mapping')

    x = torch.rand(4, 64)
    out = model(x)
    assert out.shape == (4, 10)
    # 配線パラメータに勾配が流れる(学習対象になっている)
    out.sum().backward()
    lm = model.dlgn[0].learnable_mapping
    grads = [p.grad for p in lm.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any()) for g in grads), (
        '配線学習パラメータに勾配が流れていない'
    )


def test_balanced_without_bypass_ignores_expand_fill():
    """balanced は bypass 以外の expand_fill を無視して前層のみを見る。

    HYBRID-4 では balanced が bypass も無視していたが、HYBRID-4b で
    **bypassとは併用できる**ように変更した(要因分解で効果量が最大だった
    ショートカットと、次点の均等被覆を合成するため)。bypass併用時の挙動は
    `test_balanced_bypass_*` を参照。
    """
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 12],
        dlgn_widths=[12, 24, 60], n_inputs=6, connections_mode='balanced',
        expand_fill='fanout2', seed=0,
    )
    assert model.aux_widths == [0, 0, 0]
    for i, layer in enumerate(model.dlgn):
        prev = 12 if i == 0 else [12, 24][i - 1]
        assert layer.in_dim == prev


# ============================================================
# HYBRID-4b: balanced + bypass の併用
# ============================================================


def test_balanced_bypass_covers_whole_pool_evenly():
    """balanced+bypass: [前層+界面+既出中間層]のプール全体が均等被覆される。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    prev_width, aux_width, k, n_inputs = 120, 120, 240, 6
    conn, pool = m.build_layer_connections(
        'bypass', prev_width, aux_width, k, rng, n_inputs,
        connections_mode='balanced',
    )
    assert pool == prev_width + aux_width, '補助プールが使われていない'
    assert conn.shape == (n_inputs, k)
    # ゲート内は全相異
    n_unique = torch.tensor([len(set(conn[:, g].tolist())) for g in range(k)])
    assert int(n_unique.min()) == n_inputs
    # プール全体(前層も補助プールも)が均等
    cnt = torch.bincount(conn.flatten(), minlength=pool)
    assert int(cnt.min()) >= 1, 'プールに未使用要素がある'
    assert int(cnt.max()) - int(cnt.min()) <= 1, 'プール全体が均等でない'
    # 6·240 = 1440 = 240 * 6 なので厳密均等
    assert int(cnt.min()) == int(cnt.max()) == 6
    # 前層部分も確実に覆われている
    assert int(cnt[:prev_width].min()) >= 1


def test_balanced_without_bypass_stays_prev_only():
    """balanced 単体(expand_fill='none')は従来どおり前層のみを見る。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    conn, pool = m.build_layer_connections(
        'none', 120, 120, 240, rng, 6, connections_mode='balanced',
    )
    assert pool == 120, 'balanced単体で補助プールを使ってはいけない'
    assert int(conn.max()) < 120


def test_balanced_bypass_falls_back_when_ports_too_few():
    """ポート数がプール全体に足りないときは前層のみの均等被覆へ退避する。"""
    m = _mod()
    rng = torch.Generator().manual_seed(0)
    # n=2, k=60 -> ports=120 < pool(120+120)=240 なので退避
    conn, pool = m.build_layer_connections(
        'bypass', 120, 120, 60, rng, 2, connections_mode='balanced',
    )
    assert pool == 120
    assert int(conn.max()) < 120


def test_balanced_bypass_model_uses_aux_pool():
    """モデル全体で balanced+bypass が補助プールを使う(aux_widths>0)。"""
    m = _mod()
    model = m.HybridFrontDLGN(
        in_dim=64, num_classes=10, front_widths=[32, 12],
        dlgn_widths=[12, 24, 60], n_inputs=6, connections_mode='balanced',
        expand_fill='bypass', seed=0,
    )
    # 層0は補助プール無し(前層=界面)、層1以降は補助プールを持つ
    assert model.aux_widths[0] == 0
    assert all(a > 0 for a in model.aux_widths[1:]), model.aux_widths
    x = torch.rand(4, 64)
    assert model(x).shape == (4, 10)
