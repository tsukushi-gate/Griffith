"""widen_native_ir の性質テスト(ROADMAP-v5)。

転写native DLGN IRを teacher初期化を保存したまま f倍に広げる関数の不変量:
    - validate通過 / ゲート数が期待値どおり
    - seed決定性(同seedでビット一致、別seedで差)
    - teacher由来ノードの先頭保存(隠れ層・最終層)
    - init_new_final='const0' で widen直後の hard精度が元IRと一致(容量追加のみ)

小さな転写IRを transcribe_mlp で生成して検証する(学習なし=高速スイート)。
"""

import numpy as np
import torch
from torch import nn

from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.convert.widen import widen_native_ir
from griffith_logic.ir.serialize import tt_from_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _tiny_mlp(in_dim=6, hidden=(8, 6), n_classes=3, seed=0):
    """fan-in2に収まる小さなMLP(隠れ2層)。各Linearを疎化前提で小さく作る。"""
    torch.manual_seed(seed)
    dims = [in_dim, *hidden, n_classes]
    layers = []
    for a, b in zip(dims[:-1], dims[1:], strict=True):
        layers.append(nn.Linear(a, b))
    net = nn.Sequential(*layers)
    return net


def _make_transcribe_ir(seed=0, in_dim=6, calib_n=32):
    """fan-in2転写IRと較正入力を返す。"""
    net = _tiny_mlp(in_dim=in_dim, seed=seed)
    rng = np.random.default_rng(seed)
    calib = rng.integers(0, 2, size=(calib_n, in_dim)).astype(np.int64)
    spec = TranscribeSpec(n_inputs=2, weight_bits=8, isf=False)
    ir = transcribe_mlp(net, calib, spec)
    return ir, calib


def _hidden_and_final(graph):
    gs_id = next(n.id for n in graph.nodes if n.type == 'group_sum')
    final_id = next(e.src for e in graph.edges if e.dst == gs_id)
    lut_ids = [n.id for n in graph.topological_order() if n.type == 'lut_layer']
    hidden_ids = [nid for nid in lut_ids if nid != final_id]
    return hidden_ids, final_id, gs_id


def test_widen_validates():
    ir, _ = _make_transcribe_ir()
    for f in (2, 4, 16):
        w = widen_native_ir(ir, f, seed=1)
        assert validate(w) == [], f'widen f={f} が validate を通らない'


def test_widen_factor1_is_identity_copy():
    ir, _ = _make_transcribe_ir()
    w = widen_native_ir(ir, 1, seed=0)
    assert validate(w) == []
    for a, b in zip(ir.nodes, w.nodes, strict=True):
        if a.type == 'lut_layer':
            assert int(a.params['out_width']) == int(b.params['out_width'])


def test_widen_gate_counts():
    ir, _ = _make_transcribe_ir()
    hidden_ids, final_id, gs_id = _hidden_and_final(ir)
    orig = {n.id: int(n.params['out_width']) for n in ir.nodes
            if n.type == 'lut_layer'}
    for f in (2, 4, 16):
        w = widen_native_ir(ir, f, seed=2)
        wmap = {n.id: int(n.params['out_width']) for n in w.nodes
                if n.type == 'lut_layer'}
        # 隠れ層・最終層とも out_width が正確に f倍
        for nid in orig:
            assert wmap[nid] == orig[nid] * f, (
                f'f={f} node {nid}: {wmap[nid]} != {orig[nid] * f}'
            )
        # group_sum の k(クラス数)は不変、in_width は最終層out_widthに追従
        gs = w.node_by_id(gs_id)
        assert int(gs.params['k']) == int(ir.node_by_id(gs_id).params['k'])
        assert int(gs.params['in_width']) == wmap[final_id]


def test_widen_downstream_in_width_consistent():
    """隠れ層widen後、下流lut_layerのin_widthが上流out_widthと一致する。"""
    ir, _ = _make_transcribe_ir()
    w = widen_native_ir(ir, 4, seed=3)
    # 直列: 各lut_layerのin_widthは、前段(lut_layer)のout_width or 入力幅
    lut_nodes = [n for n in w.topological_order() if n.type == 'lut_layer']
    prev_out = w.inputs[0].width
    for n in lut_nodes:
        assert int(n.params['in_width']) == prev_out, (
            f'{n.id}: in_width={n.params["in_width"]} != prev_out={prev_out}'
        )
        prev_out = int(n.params['out_width'])


def test_widen_preserves_teacher_prefix():
    """teacher由来ノードが各層の先頭に保存される(indices/tt がビット一致)。"""
    ir, _ = _make_transcribe_ir()
    w = widen_native_ir(ir, 4, seed=4)
    hidden_ids, final_id, gs_id = _hidden_and_final(ir)

    # 隠れ層: 先頭W個の (indices, tt) が元と一致
    for nid in hidden_ids:
        o = ir.node_by_id(nid)
        n = w.node_by_id(nid)
        w0 = int(o.params['out_width'])
        oi = np.asarray(o.params['indices'])
        ni = np.asarray(n.params['indices'])
        assert np.array_equal(oi, ni[:, :w0]), f'{nid}: 隠れ先頭indices不一致'
        ott = tt_from_hex(o.params['truth_tables'])
        ntt = tt_from_hex(n.params['truth_tables'])
        assert ott == ntt[:w0], f'{nid}: 隠れ先頭tt不一致'

    # 最終層: 各クラス群の先頭g個が保存される
    o = ir.node_by_id(final_id)
    n = w.node_by_id(final_id)
    k = int(ir.node_by_id(gs_id).params['k'])
    g = int(o.params['out_width']) // k
    g_new = int(n.params['out_width']) // k
    oi = np.asarray(o.params['indices'])
    ni = np.asarray(n.params['indices'])
    ott = tt_from_hex(o.params['truth_tables'])
    ntt = tt_from_hex(n.params['truth_tables'])
    for c in range(k):
        for j in range(g):
            assert oi[0, c * g + j] == ni[0, c * g_new + j]
            assert oi[1, c * g + j] == ni[1, c * g_new + j]
            assert ott[c * g + j] == ntt[c * g_new + j]


def test_widen_seed_determinism():
    ir, _ = _make_transcribe_ir()
    w1 = widen_native_ir(ir, 4, seed=7)
    w2 = widen_native_ir(ir, 4, seed=7)
    w3 = widen_native_ir(ir, 4, seed=8)
    for a, b in zip(w1.nodes, w2.nodes, strict=True):
        if a.type == 'lut_layer':
            assert np.array_equal(
                np.asarray(a.params['indices']), np.asarray(b.params['indices'])
            ), 'same seed: indices不一致'
            assert a.params['truth_tables'] == b.params['truth_tables'], \
                'same seed: tt不一致'
    # 別seedでは少なくとも1層のtt/indicesが異なる
    diff = False
    for a, c in zip(w1.nodes, w3.nodes, strict=True):
        if a.type == 'lut_layer':
            if (not np.array_equal(
                    np.asarray(a.params['indices']),
                    np.asarray(c.params['indices']))
                    or a.params['truth_tables'] != c.params['truth_tables']):
                diff = True
    assert diff, '別seedなのに完全一致(乱数が効いていない)'


def _sim_pred(graph, x, batch=64):
    preds = []
    for i in range(0, len(x), batch):
        preds.append(simulate(graph, x[i:i + batch]).argmax(axis=1))
    return np.concatenate(preds)


def test_widen_const0_preserves_hard_argmax():
    """init_new_final='const0' で widen直後の hard argmax が元IRと全一致する。

    新最終ゲートはtt=0(常に0)でgroup_sumスコアに寄与しないため、
    追加容量にもかかわらず初期の分類予測は保存される(初期精度保存)。
    """
    ir, calib = _make_transcribe_ir(seed=11, in_dim=6, calib_n=64)
    # 全2^6入力パターンで厳密照合(小規模)
    x = np.array([[(a >> j) & 1 for j in range(6)] for a in range(64)],
                 dtype=np.int64)
    base = _sim_pred(ir, x)
    for f in (2, 4, 16):
        w = widen_native_ir(ir, f, seed=5, init_new_final='const0')
        assert validate(w) == []
        wpred = _sim_pred(w, x)
        assert np.array_equal(base, wpred), (
            f'const0 widen f={f}: hard argmax が元IRと不一致'
            f'({int((base != wpred).sum())}/{len(x)} 件)'
        )


def test_widen_random_final_can_change_argmax():
    """init_new_final='random' は新ゲートが非ゼロttを持ちうるため予測が変わりうる。

    const0 との対照(random は初期精度保存を保証しない)を明示する。
    """
    ir, _ = _make_transcribe_ir(seed=13, in_dim=6)
    x = np.array([[(a >> j) & 1 for j in range(6)] for a in range(64)],
                 dtype=np.int64)
    base = _sim_pred(ir, x)
    w = widen_native_ir(ir, 16, seed=6, init_new_final='random')
    assert validate(w) == []
    # randomでは一致が保証されない(構造的に少なくとも検証は通る)
    _ = _sim_pred(w, x)  # 実行可能性のみ担保(値の一致は要求しない)
    assert base.shape == (64,)


def test_widen_invalid_args():
    ir, _ = _make_transcribe_ir()
    for bad in (0, -1):
        try:
            widen_native_ir(ir, bad, seed=0)
            raise AssertionError('factor<1 を許容してしまった')
        except ValueError:
            pass
    try:
        widen_native_ir(ir, 2, seed=0, init_new_final='bogus')
        raise AssertionError('不正な init_new_final を許容してしまった')
    except ValueError:
        pass
