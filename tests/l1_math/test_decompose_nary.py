"""n入力ゲートの2入力分解が元と厳密に等価かを確かめる。

ground truth は素朴な n分木評価。分解後は 2分木評価。
ビット単位で完全一致すること(近似ではない)。
"""

import random

import numpy as np
import pytest
import torch

from griffith_logic.conv import LutConv2d
from griffith_logic.export.decompose import (
    MAX_BINARY_DEPTH,
    binary_depth_for,
    decompose_conv_params,
    decompose_gate,
)
from griffith_logic.export.discretize import truth_tables_from_conv
from griffith_logic.export.lowering import _effective_indices_flat
from griffith_logic.ir.serialize import tt_from_hex


def _eval_gate(table_bits: int, xs) -> int:
    """n入力表の素朴評価。アドレスは Σ x_i << i。"""
    addr = 0
    for i, x in enumerate(xs):
        addr |= (x & 1) << i
    return (table_bits >> addr) & 1


def _eval_binary_subtree(gtt: dict, leafslot: dict, depth: int, xs) -> int:
    """分解結果(相対heap id)を素朴に評価する。"""
    def go(hid: int, lv: int) -> int:
        if lv == depth:
            return xs[leafslot[hid]] & 1
        a = go(2 * hid, lv + 1)
        b = go(2 * hid + 1, lv + 1)
        return (gtt[hid] >> ((b << 1) | a)) & 1
    return go(1, 0)


def _eval_tree(tts_row, idx_row, leaves_val, n: int, d: int) -> int:
    """完全n分木の素朴評価。level0=葉の直上、子は g*n+j、表は level 昇順。"""
    level_offset, acc = [], 0
    for lv in range(d):
        level_offset.append(acc)
        acc += n ** (d - 1 - lv)

    cur = [leaves_val[int(idx_row[i])] for i in range(n ** d)]
    for lv in range(d):
        nxt = []
        for g in range(len(cur) // n):
            xs = [cur[g * n + j] for j in range(n)]
            nxt.append(_eval_gate(int(tts_row[level_offset[lv] + g]), xs))
        cur = nxt
    assert len(cur) == 1
    return cur[0]


@pytest.mark.parametrize('n', [2, 4, 6])
def test_binary_depth_matches_recurrence(n):
    """g(2)=1, g(n)=g(n-1)+2。"""
    expected = {2: 1, 4: 5, 6: 9}
    assert binary_depth_for(n) == expected[n]


@pytest.mark.parametrize('n', [1, 2, 4, 6])
def test_single_gate_all_inputs(n):
    """1ゲートの分解が全入力パターンで元の表と一致する。"""
    # n=6 の表は 2^64 通りあり np.random.Generator の上限を超えるので getrandbits
    rng = random.Random(0)
    depth = binary_depth_for(n)
    for _ in range(8):
        table = rng.getrandbits(1 << n)
        gtt, leafslot = decompose_gate(table, n)
        for addr in range(1 << n):
            xs = [(addr >> i) & 1 for i in range(n)]
            assert _eval_binary_subtree(gtt, leafslot, depth, xs) == \
                _eval_gate(table, xs), f'n={n} table={table:#x} addr={addr}'


@pytest.mark.parametrize('n', [4, 6])
def test_constant_and_degenerate_tables(n):
    """定数表・単一変数依存の表でも一致する(Shannon再帰の端の確認)。"""
    depth = binary_depth_for(n)
    tables = [0, (1 << (1 << n)) - 1]
    for i in range(n):  # x_i そのもの
        tables.append(sum(((a >> i) & 1) << a for a in range(1 << n)))
    for table in tables:
        gtt, leafslot = decompose_gate(table, n)
        for addr in range(1 << n):
            xs = [(addr >> j) & 1 for j in range(n)]
            assert _eval_binary_subtree(gtt, leafslot, depth, xs) == \
                _eval_gate(table, xs), f'n={n} table={table:#x}'


@pytest.mark.parametrize(('n', 'd'), [(4, 2), (6, 1), (4, 1)])
def test_conv_params_tree_equivalence(n, d):
    """conv2d params ごと分解して、木の出力が全入力で一致する。

    受容野の入力ビットを総当たりできる小さい設定で確認する。
    """
    rng = np.random.default_rng(1)
    prng = random.Random(1)  # 2^64 通りの表は getrandbits で作る
    oc, n_leaves = 3, n ** d
    n_internal = (n_leaves - 1) // (n - 1)
    n_taps = 8  # 受容野内のタップ数(2^8=256通りを総当たり)

    params = {
        'in_shape': [n_taps, 1, 1], 'out_channels': oc,
        'kernel_size': [1, 1], 'stride': [1, 1], 'padding': [0, 0],
        'tree_depth': d, 'n_inputs': n,
        'indices_flat': rng.integers(0, n_taps, size=(oc, n_leaves)),
        'truth_tables': [prng.getrandbits(1 << n)
                         for _ in range(oc * n_internal)],
    }
    new = decompose_conv_params(params)
    assert new['n_inputs'] == 2
    assert new['tree_depth'] == d * binary_depth_for(n)

    old_tts = np.array(params['truth_tables'], dtype=object).reshape(oc, n_internal)
    nd = new['tree_depth']
    new_tts = np.array(new['truth_tables'], dtype=object).reshape(oc, (1 << nd) - 1)

    for pattern in range(1 << n_taps):
        vals = [(pattern >> i) & 1 for i in range(n_taps)]
        for c in range(oc):
            got = _eval_tree(new_tts[c], new['indices_flat'][c], vals, 2, nd)
            want = _eval_tree(old_tts[c], params['indices_flat'][c], vals, n, d)
            assert got == want, f'n={n} d={d} oc={c} pattern={pattern:#x}'


def test_n2_is_passthrough():
    """n_inputs=2 は書き換えない。"""
    params = {
        'in_shape': [4, 1, 1], 'out_channels': 2,
        'kernel_size': [1, 1], 'stride': [1, 1], 'padding': [0, 0],
        'tree_depth': 2, 'n_inputs': 2,
        'indices_flat': np.zeros((2, 4), dtype=np.int64),
        'truth_tables': [1, 2, 3, 4, 5, 6],
    }
    out = decompose_conv_params(params)
    assert out['n_inputs'] == 2
    assert out['tree_depth'] == 2
    assert out['truth_tables'] == params['truth_tables']


def test_depth_limit_is_refused():
    """深すぎる組み合わせは黙って巨大化させず、明示的に断る。"""
    params = {
        'in_shape': [4, 1, 1], 'out_channels': 1,
        'kernel_size': [1, 1], 'stride': [1, 1], 'padding': [0, 0],
        'tree_depth': 3, 'n_inputs': 6,   # 3 × 9 = 27
        'indices_flat': np.zeros((1, 6 ** 3), dtype=np.int64),
        'truth_tables': [0] * ((6 ** 3 - 1) // 5),
    }
    with pytest.raises(ValueError, match='深すぎる'):
        decompose_conv_params(params)
    assert 3 * binary_depth_for(6) > MAX_BINARY_DEPTH


@pytest.mark.parametrize('parameterization',
                         ['independent', 'independent_ste'])
def test_export_pipeline_end_to_end(parameterization):
    """n_inputs=4 の検出モデルを export→分解 しても hard eval と整数一致する。

    DWN(n入力RAMノード)は export 経路で4回続けて詰まった(state_dict不一致 /
    fx TraceError / getattr非対応 / 真理値表64bit超え)。木の n_inputs=4 は
    conv2d ノードのままなので通るはず、というのをここで固定する。
    実験で実際に使うのは independent_ste（independent は離散化で崩壊する）。
    """
    from experiments.detection_model import CenterNetLUT
    from griffith_logic.export.dag import export_dag_ir
    from griffith_logic.export.decompose import decompose_graph
    from griffith_logic.ir.simulate import simulate
    from griffith_logic.spatial_group_sum import SpatialGroupSum

    torch.manual_seed(0)
    model = CenterNetLUT(
        in_channels=12, n_classes=3, k=8, tree_depth=2, head_depth=2,
        n_inputs=4, selection='soft', parameterization=parameterization,
        input_channels_per_tree=8, hm_group=8, n_layers=3,
    )
    model.eval()
    # simulate は SpatialGroupSum.scale を無視して生 popcount を返すため、
    # 整数同士で比べられるよう scale を 1.0 に落とす。
    for m in model.modules():
        if isinstance(m, SpatialGroupSum):
            m.scale = 1.0

    in_shape = (12, 16, 16)
    graph = export_dag_ir(model, in_shape, name='nin4')
    assert {int(n.params.get('n_inputs', 2))
            for n in graph.nodes if n.type == 'conv2d'} == {4}

    decomposed = decompose_graph(graph)
    assert {int(n.params.get('n_inputs', 2))
            for n in decomposed.nodes if n.type == 'conv2d'} == {2}
    assert {int(n.params['tree_depth'])
            for n in decomposed.nodes if n.type == 'conv2d'} == {2 * 5}

    gen = torch.Generator().manual_seed(0)
    n_samples = 2
    x = torch.randint(0, 2, (n_samples, *in_shape), generator=gen).float()
    with torch.no_grad():
        out = model(x)
    # 複数ヘッドは metadata の output_heads 順に連結されて1本の出力になる
    heads = graph.metadata.get('output_heads') or []
    want = np.rint(torch.cat(
        [out[h['name']].reshape(n_samples, -1) for h in heads], dim=1).numpy()
    ).astype(np.int64)
    flat = x.flatten(1).numpy().astype(np.int64)

    for graph_under_test, label in ((graph, '生IR'), (decomposed, '分解後IR')):
        got = np.asarray(simulate(graph_under_test, flat)).astype(np.int64)
        np.testing.assert_array_equal(
            got, want, err_msg=f'{label} が PyTorch hard eval と不一致')


def test_real_lutconv2d_roundtrip():
    """実の LutConv2d(n_inputs=4) を分解して hard 出力が一致する。"""
    torch.manual_seed(0)
    conv = LutConv2d(
        in_channels=8, out_channels=5, kernel_size=1, tree_depth=2,
        n_inputs=4, parameterization='independent', input_channels_per_tree=8,
    )
    params = {
        'in_shape': [8, 1, 1], 'out_channels': 5,
        'kernel_size': [1, 1], 'stride': [1, 1], 'padding': [0, 0],
        'tree_depth': conv.tree_depth, 'n_inputs': conv.n_inputs,
        'indices_flat': _effective_indices_flat(conv).cpu().numpy(),
        'truth_tables': [int(t) for t in
                         truth_tables_from_conv(conv).reshape(-1)],
    }
    new = decompose_conv_params(params)

    x = (torch.rand(4, 8, 1, 1) > 0.5).float()
    with torch.no_grad():
        conv.eval()
        want = conv(x).round().long().numpy()[:, :, 0, 0]

    nd = new['tree_depth']
    oc = int(params['out_channels'])
    tts = np.array(tt_from_hex(new['truth_tables'])
                   if isinstance(new['truth_tables'][0], str)
                   else new['truth_tables'], dtype=object).reshape(oc, (1 << nd) - 1)
    for b in range(x.shape[0]):
        vals = x[b, :, 0, 0].long().tolist()
        for c in range(oc):
            got = _eval_tree(tts[c], new['indices_flat'][c], vals, 2, nd)
            assert got == want[b, c], f'batch={b} oc={c}'
