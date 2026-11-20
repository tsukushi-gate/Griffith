"""dense 平坦化(`flatten_dense` / lut_flat / `bundle --dense-flat`)のテスト。

等価性の鎖(ゴールデンは常に**元IRの simulate**で、変換後IR自身ではない):

    元IR(lut_layer列+group_sum) simulate
        ⇔ flatten_dense 後IR(lut_flat+group_sum) simulate
        ⇔ netlist_reduce.ReduceResult.class_scores
        ⇔ 生成C(scalar/表駆動)

(a) 定数・パススルー・NOT・重複を意図的に混ぜた合成IRで、変換前後の simulate が
    一致しゲート数が減る  (b) lut_flat simulate ⇔ netlist_reduce の class_scores
(c) バンドル→codegen-c→C実行 ⇔ 元IRの simulate(cc無しはskip)
(d) 不一致の注入で ValueError  (e) 既定オフでは従来出力と同一
(f) hypothesis による小規模ランダムIR  (g) u32添字・未対応バックエンドの拒否 ほか
(h) Web ターゲット: codegen-web / verify-web の既定 auto(閾値 0)と
    TS 側 lut_flat 実行器 ⇔ 元IRの simulate(node + runtime/web/dist が無ければskip)
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from griffith_logic.deploy import build_bundle, load_bundle, save_bundle
from griffith_logic.deploy import flatten_dense as fd
from griffith_logic.deploy.backends import can_run
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.flatten_dense import (
    find_dense_chains,
    flatten_dense_doc,
    flatten_dense_ir,
)
from griffith_logic.export.netlist_reduce import (
    class_scores,
    flat_netlist_from_ir,
    reduce_netlist,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate, simulate_trace
from griffith_logic.ir.validate import IRValidationError, validate

from .conftest import CC


def _dense_ir(layers, in_width, k, name='t'):
    """layers: [(indices (2,w), tts[int])...] から lut_layer列+group_sum の IR を作る。"""
    nodes, edges = [], [IREdge(src='x', dst='n1')]
    prev = in_width
    for li, (idx, tts) in enumerate(layers):
        w = len(tts)
        nodes.append(IRNode(id=f'n{li + 1}', type='lut_layer', params={
            'in_width': prev, 'out_width': w, 'n_inputs': 2,
            'indices': np.asarray(idx, dtype=np.int64),
            'truth_tables': [f'0x{int(t):x}' for t in tts]}))
        edges.append(IREdge(src=f'n{li + 1}', dst=f'n{li + 2}'))
        prev = w
    gs = f'n{len(layers) + 1}'
    nodes.append(IRNode(id=gs, type='group_sum',
                        params={'in_width': prev, 'k': k, 'tau': 1.0}))
    edges.append(IREdge(src=gs, dst='y'))
    return IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width)],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=nodes, edges=edges)


def _random_layers(rng, in_width, widths, p_trivial=0.5):
    """自明な tt(定数・パススルー・NOT)を多めに混ぜたランダム層。"""
    trivial = [0x0, 0xf, 0xa, 0xc, 0x3, 0x5]
    layers, prev = [], in_width
    for w in widths:
        idx = rng.integers(0, prev, size=(2, w))
        tts = np.where(rng.random(w) < p_trivial,
                       rng.choice(trivial, size=w), rng.integers(0, 16, size=w))
        layers.append((idx, tts))
        prev = w
    return layers


def _mixed_ir():
    """定数・パススルー・NOT・重複(完全一致/可換入替)・同一入力を意図的に含むIR。"""
    # 層1: in_width=6, 12ゲート
    idx1 = np.array([
        [0, 0, 0, 2, 4, 0, 0, 1, 2, 2, 2, 4],     # a
        [1, 1, 1, 3, 5, 5, 1, 0, 3, 3, 2, 5],     # b
    ])
    tt1 = [0x0,   # 定数0
           0xf,   # 定数1
           0xa,   # パススルー a
           0xc,   # パススルー b
           0x5,   # NOT a
           0x3,   # NOT b
           0x6,   # xor(x0,x1)
           0x6,   # xor(x1,x0) = 可換重複
           0x8,   # and(x2,x3)
           0x8,   # and(x2,x3) = 完全重複
           0x6,   # xor(x2,x2) = 同一入力 → 定数0
           0xe]   # or(NOT x4, NOT x5 の元) 通常ゲート
    rng = np.random.default_rng(7)
    idx2 = rng.integers(0, 12, size=(2, 8))
    tt2 = [0x6, 0x8, 0xe, 0x7, 0x9, 0x2, 0x4, 0xb]
    return _dense_ir([(idx1, tt1), (idx2, tt2)], in_width=6, k=2, name='mixed')


def _bits(width, n, seed=0):
    return np.random.default_rng(seed).integers(
        0, 2, size=(n, width), dtype=np.uint8)


# ---------------------------------------------------------------------------
# (a) 変換前後の simulate 一致とゲート数減少
# ---------------------------------------------------------------------------
class TestFlattenPreservesSimulate:
    def test_mixed_ir_matches_and_shrinks(self):
        g = _mixed_ir()
        validate(g)
        new, rep = flatten_dense_ir(g)
        assert [n.type for n in new.nodes] == ['lut_flat', 'group_sum']
        x = _bits(6, 64, seed=1)
        assert np.array_equal(simulate(g, x), simulate(new, x))
        assert rep['gates_after'] < rep['gates_before']
        # 意図的に混ぜた各種が全部効いている(内訳が空でない)
        assert rep['constants'] > 0
        assert rep['passthrough'] > 0
        assert rep['cse'] > 0
        assert rep['gates_before'] == 20
        assert rep['verify_random'] == 16

    def test_mixed_ir_exhaustive_inputs(self):
        """入力6bit = 64通りを全数で突き合わせる(ランダムの取りこぼしを塞ぐ)。"""
        g = _mixed_ir()
        new, _ = flatten_dense_ir(g)
        x = ((np.arange(64)[:, None] >> np.arange(6)[None, :]) & 1).astype(
            np.uint8)
        assert np.array_equal(simulate(g, x), simulate(new, x))

    def test_original_graph_is_not_modified(self):
        g = _mixed_ir()
        before = json.dumps(ir_to_dict(g), sort_keys=True)
        flatten_dense_ir(g)
        assert json.dumps(ir_to_dict(g), sort_keys=True) == before

    def test_lut_flat_keeps_last_layer_node_id_and_edges(self):
        """lut_flat は最後の層のidを引き継ぎ、後段・入出力の配線は無改造。"""
        g = _mixed_ir()
        new, _ = flatten_dense_ir(g)
        flat = next(n for n in new.nodes if n.type == 'lut_flat')
        assert flat.id == 'n2'
        assert flat.train_info['flattened_from'] == ['n1', 'n2']
        assert {(e.src, e.dst) for e in new.edges} == {
            ('x', 'n2'), ('n2', 'n3'), ('n3', 'y')}
        assert validate(new) == []

    def test_final_layer_bits_match_trace(self):
        """lut_flat の出力ビット列 = 元の最終 lut_layer の出力ビット列。"""
        g = _mixed_ir()
        new, _ = flatten_dense_ir(g)
        x = _bits(6, 40, seed=2)
        want = simulate_trace(g, x)['n2']
        got = simulate_trace(new, x)['n2']
        assert np.array_equal(want, got)

    @pytest.mark.parametrize('seed', range(6))
    def test_random_deep_ir(self, seed):
        rng = np.random.default_rng(seed)
        layers = _random_layers(rng, 40, [64, 64, 48, 48, 30], p_trivial=0.4)
        g = _dense_ir(layers, in_width=40, k=6)
        new, rep = flatten_dense_ir(g, n_verify=64, seed=seed)
        x = _bits(40, 128, seed=seed + 100)
        assert np.array_equal(simulate(g, x), simulate(new, x))
        assert rep['gates_after'] <= rep['gates_before']

    def test_no_chain_returns_same_graph(self):
        """lut_layer が無い(n_inputs=2 でない)IR は何もしない。"""
        g = IRGraph(
            inputs=[IRPort(id='x', width=4)], outputs=[IRPort(id='y', width=2)],
            nodes=[IRNode(id='g', type='group_sum',
                          params={'in_width': 4, 'k': 2, 'tau': 1.0})],
            edges=[IREdge(src='x', dst='g'), IREdge(src='g', dst='y')])
        new, rep = flatten_dense_ir(g)
        assert new is g and rep['chains'] == []

    def test_fanout_in_the_middle_breaks_the_chain(self):
        """途中の層を複数が読む(出力ビットが外から見える)ならそこで鎖を切る。"""
        rng = np.random.default_rng(3)
        n1 = IRNode(id='n1', type='lut_layer', params={
            'in_width': 8, 'out_width': 8, 'n_inputs': 2,
            'indices': rng.integers(0, 8, size=(2, 8)),
            'truth_tables': ['0x6'] * 8})
        n2 = IRNode(id='n2', type='lut_layer', params={
            'in_width': 8, 'out_width': 6, 'n_inputs': 2,
            'indices': rng.integers(0, 8, size=(2, 6)),
            'truth_tables': [f'0x{t:x}' for t in rng.integers(1, 15, size=6)]})
        n3 = IRNode(id='n3', type='lut_layer', params={
            'in_width': 8, 'out_width': 6, 'n_inputs': 2,
            'indices': rng.integers(0, 8, size=(2, 6)),
            'truth_tables': [f'0x{t:x}' for t in rng.integers(1, 15, size=6)]})
        cat = IRNode(id='cat', type='concat', params={'in_widths': [6, 6]})
        gs = IRNode(id='gs', type='group_sum',
                    params={'in_width': 12, 'k': 3, 'tau': 1.0})
        g = IRGraph(
            inputs=[IRPort(id='x', width=8)], outputs=[IRPort(id='y', width=3)],
            nodes=[n1, n2, n3, cat, gs],
            edges=[IREdge('x', 'n1'), IREdge('n1', 'n2'), IREdge('n1', 'n3'),
                   IREdge('n2', 'cat', 0), IREdge('n3', 'cat', 1),
                   IREdge('cat', 'gs'), IREdge('gs', 'y')])
        validate(g)
        chains = find_dense_chains(g)
        assert sorted(len(c) for c in chains) == [1, 1, 1]
        new, rep = flatten_dense_ir(g)
        assert [n.type for n in new.nodes].count('lut_flat') == 3
        x = _bits(8, 64, seed=4)
        assert np.array_equal(simulate(g, x), simulate(new, x))
        assert len(rep['chains']) == 3

    def test_n_inputs_not_2_is_left_alone(self):
        rng = np.random.default_rng(5)
        n1 = IRNode(id='n1', type='lut_layer', params={
            'in_width': 8, 'out_width': 6, 'n_inputs': 3,
            'indices': rng.integers(0, 8, size=(3, 6)),
            'truth_tables': [f'0x{t:x}' for t in rng.integers(0, 256, size=6)]})
        gs = IRNode(id='gs', type='group_sum',
                    params={'in_width': 6, 'k': 2, 'tau': 1.0})
        g = IRGraph(
            inputs=[IRPort(id='x', width=8)], outputs=[IRPort(id='y', width=2)],
            nodes=[n1, gs],
            edges=[IREdge('x', 'n1'), IREdge('n1', 'gs'), IREdge('gs', 'y')])
        new, rep = flatten_dense_ir(g)
        assert new is g and rep['chains'] == []


# ---------------------------------------------------------------------------
# (b) lut_flat simulate ⇔ netlist_reduce の class_scores
# ---------------------------------------------------------------------------
class TestAgainstNetlistReduce:
    @pytest.mark.parametrize('seed', range(4))
    def test_group_sum_of_lut_flat_equals_class_scores(self, seed):
        rng = np.random.default_rng(seed)
        layers = _random_layers(rng, 48, [60, 60, 60, 36], p_trivial=0.45)
        g = _dense_ir(layers, in_width=48, k=6)
        x = _bits(48, 200, seed=seed + 50)
        netlist = flat_netlist_from_ir(g)
        result = reduce_netlist(netlist)
        new, _ = flatten_dense_ir(g)
        flat_scores = simulate(new, x).astype(np.int64)
        assert np.array_equal(flat_scores, result.class_scores(x))
        # 原ネットのスコア(縮約器の真実源)とも一致
        assert np.array_equal(flat_scores, class_scores(netlist, x))
        assert np.array_equal(flat_scores, simulate(g, x).astype(np.int64))

    def test_effective_gate_count_matches_reducer(self):
        g = _mixed_ir()
        result = reduce_netlist(flat_netlist_from_ir(g))
        _, rep = flatten_dense_ir(g)
        assert rep['gates_after'] == result.effective_gates


# ---------------------------------------------------------------------------
# (c) バンドル → codegen-c → C 実行 ⇔ 元IRの simulate
# ---------------------------------------------------------------------------
@pytest.mark.skipif(CC is None, reason='Cコンパイラが無い環境')
class TestCEndToEnd:
    def _doc(self, graph, **extra):
        return {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': [], **extra}

    def _c_scores(self, bundle, x, tmp_path):
        from griffith_logic.deploy.verify import build_c_model, run_c_model

        exe = build_c_model(bundle, tmp_path)
        n_out = int(bundle['ir']['outputs'][0]['width'])
        return np.stack([run_c_model(exe, row.tobytes(), n_out) for row in x])

    def test_flat_c_matches_original_simulate(self, tmp_path):
        """ゴールデンは元IRの simulate(変換後IRの simulate ではない)。"""
        rng = np.random.default_rng(11)
        layers = _random_layers(rng, 40, [64, 64, 48, 48], p_trivial=0.4)
        g = _dense_ir(layers, in_width=40, k=6)
        doc, rep = flatten_dense_doc(self._doc(g))
        assert rep['gates_after'] < rep['gates_before']
        bundle = build_bundle(doc)
        assert 'lut_flat' in {n['type'] for n in bundle['ir']['nodes']}
        x = _bits(40, 12, seed=12)
        want = simulate(g, x).astype(np.int64)
        assert np.array_equal(self._c_scores(bundle, x, tmp_path), want)

    def test_verify_c_matches_simulate_passes(self, tmp_path):
        from griffith_logic.deploy.verify import verify_c_matches_simulate

        doc, _ = flatten_dense_doc(self._doc(_mixed_ir()))
        res = verify_c_matches_simulate(
            build_bundle(doc), n_samples=8, work_dir=tmp_path, verbose=False)
        assert res['path'] == 'bits' and res['max_abs_diff'] == 0

    def test_mixed_ir_exhaustive_c(self, tmp_path):
        """6bit入力 64通りを全数で C と元IR simulate に突き合わせる。"""
        g = _mixed_ir()
        doc, _ = flatten_dense_doc(self._doc(g))
        x = ((np.arange(64)[:, None] >> np.arange(6)[None, :]) & 1).astype(
            np.uint8)
        want = simulate(g, x).astype(np.int64)
        assert np.array_equal(
            self._c_scores(build_bundle(doc), x, tmp_path), want)

    def test_u32_index_path(self, tmp_path):
        """信号数 > 32768 では添字表が uint32_t になり、値は変わらない。"""
        rng = np.random.default_rng(21)
        in_w = 33000
        idx = rng.integers(0, in_w, size=(2, 64))
        tts = rng.integers(1, 15, size=64)       # 定数を避け、ゲートを残す
        g = _dense_ir([(idx, tts)], in_width=in_w, k=4)
        doc, rep = flatten_dense_doc(self._doc(g))
        bundle = build_bundle(doc)
        src = generate_c(bundle, name='m', validate=False)['m.c']
        assert 'mdl_lut_flat_u32' in src and 'mdl_lut_flat_u16' not in src
        x = _bits(in_w, 4, seed=22)
        want = simulate(g, x).astype(np.int64)
        assert np.array_equal(self._c_scores(bundle, x, tmp_path), want)

    def test_u16_index_path_in_source(self):
        doc, _ = flatten_dense_doc(self._doc(_mixed_ir()))
        src = generate_c(build_bundle(doc), name='m', validate=False)['m.c']
        assert 'mdl_lut_flat_u16' in src
        assert 'static const uint8_t flat' in src          # tt 表(flash)

    def test_ctx_bytes_accounts_signal_table(self):
        """信号表(1+in+n_gates バイト)が MDL_CTX_BYTES に算入される。"""
        g = _mixed_ir()
        doc, rep = flatten_dense_doc(self._doc(g))
        hdr = generate_c(build_bundle(doc), name='m', validate=False)['m.h']
        line = next(ln for ln in hdr.splitlines()
                    if ln.startswith('#define MDL_CTX_BYTES'))
        ctx = int(line.split()[-1].rstrip('u'))
        assert ctx >= 1 + 6 + rep['gates_after']

    def test_xor_example_end_to_end(self, tmp_path):
        """examples/06_deploy_c の XOR(学習→IR→flat→C)が元IRと一致する。"""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from griffith_logic import GroupSum, LutLayer
        from griffith_logic.export import export_ir

        torch.manual_seed(0)
        xs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        ys = torch.tensor([0, 1, 1, 0])
        model = nn.Sequential(LutLayer(2, 16, n_inputs=2),
                              LutLayer(16, 16, n_inputs=2),
                              GroupSum(k=2, tau=2.0))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            F.cross_entropy(model(xs), ys).backward()
            opt.step()
        model.eval()
        ir = export_ir(model, input_width=2, name='xor')
        doc, rep = flatten_dense_doc({'ir': ir_to_dict(ir)})
        assert rep['gates_after'] < rep['gates_before']
        x = xs.numpy().astype(np.uint8)
        want = simulate(ir, x).astype(np.int64)
        assert np.array_equal(
            self._c_scores(build_bundle(doc), x, tmp_path), want)


# ---------------------------------------------------------------------------
# (d) 不一致の注入で ValueError(多重防御)
# ---------------------------------------------------------------------------
class TestMismatchInjection:
    def _corrupt(self, monkeypatch):
        """縮約結果の最終出力 0 番の極性を反転する(全入力で出力が変わる)。"""
        real = fd.reduce_netlist

        def broken(netlist):
            res = real(netlist)
            kind, idx, pol = res.final_refs[0]
            res.final_refs[0] = (kind, idx, 1 - pol) if kind != 'c' else (
                'c', 1 - idx, 0)
            return res

        monkeypatch.setattr(fd, 'reduce_netlist', broken)

    def test_random_mismatch_raises(self, monkeypatch):
        self._corrupt(monkeypatch)
        with pytest.raises(ValueError, match='一致しない'):
            flatten_dense_ir(_mixed_ir())

    def test_demo_data_mismatch_raises(self, monkeypatch):
        """ランダム検証を省いても、実データ検証が不一致を捕まえる。"""
        self._corrupt(monkeypatch)
        with pytest.raises(ValueError, match='実データ'):
            flatten_dense_ir(_mixed_ir(), n_verify=0,
                             demo_bits=_bits(6, 8, seed=9))

    def test_doc_path_raises_too(self, monkeypatch):
        self._corrupt(monkeypatch)
        with pytest.raises(ValueError, match='一致しない'):
            flatten_dense_doc({'ir': ir_to_dict(_mixed_ir())})

    def test_verify_counts_in_report(self):
        _, rep = flatten_dense_ir(_mixed_ir(), n_verify=5,
                                  demo_bits=_bits(6, 7, seed=1))
        assert rep['verify_random'] == 5 and rep['verify_demo'] == 7


# ---------------------------------------------------------------------------
# (e) 既定オフでは従来出力と同一 / CLI
# ---------------------------------------------------------------------------
def _cli_args(src, out, **kw):
    # 既定の --compress は fast(sweep を自動適用)。ここでは dense-flat
    # 単体の挙動(ゲート数 20 など)を見たいので none を明示する。
    base = dict(source=str(src), output=str(out), gzip=False, format=None,
                no_verify=False, resynth=None, resynth_max_cost=5,
                resynth_verify_samples=8, compress='none')
    base.update(kw)
    return SimpleNamespace(**base)


class TestCliAndDefaultOff:
    def _write_doc(self, tmp_path):
        doc = {'ir': ir_to_dict(_mixed_ir()), 'preproc': {}, 'labels': []}
        src = tmp_path / 'm.json'
        src.write_text(json.dumps(doc), encoding='utf-8')
        return src

    def test_default_off_output_is_byte_identical(self, tmp_path):
        """--dense-flat 無し(属性が無い/False)かつ --compress none の出力は、
        機能追加前と同じ経路の出力(build_bundle + attach_bundle_sizes +
        save_bundle)とバイト同一。"""
        from griffith_logic.deploy.bundle import attach_bundle_sizes
        from griffith_logic.deploy.cli import _cmd_bundle

        src = self._write_doc(tmp_path)
        out_absent = tmp_path / 'absent.bundle.json'
        out_false = tmp_path / 'false.bundle.json'
        _cmd_bundle(_cli_args(src, out_absent))
        _cmd_bundle(_cli_args(src, out_false, dense_flat=False))
        ref = build_bundle(json.loads(src.read_text(encoding='utf-8')))
        attach_bundle_sizes(ref)
        ref_path = save_bundle(ref, tmp_path / 'ref.bundle.json')
        assert out_absent.read_bytes() == ref_path.read_bytes()
        assert out_false.read_bytes() == ref_path.read_bytes()
        bundle = json.loads(out_false.read_text(encoding='utf-8'))
        assert 'dense_flat' not in bundle['ir']['metadata']
        assert {n['type'] for n in bundle['ir']['nodes']} == {
            'lut_layer', 'group_sum'}

    def test_default_off_generated_c_has_no_flat(self, dense_doc):
        files = generate_c(build_bundle(dense_doc), name='m', validate=False)
        assert 'lut_flat' not in files['m.c']

    def test_bundle_dense_flat_writes_report(self, tmp_path, capsys):
        from griffith_logic.deploy.cli import _cmd_bundle

        src = self._write_doc(tmp_path)
        out = tmp_path / 'flat.bundle.json'
        assert _cmd_bundle(_cli_args(src, out, dense_flat=True)) == 0
        bundle = load_bundle(out, strict=True)
        rep = bundle['ir']['metadata']['dense_flat']
        assert rep['gates_before'] == 20 and rep['gates_after'] < 20
        assert {n['type'] for n in bundle['ir']['nodes']} == {
            'lut_flat', 'group_sum'}
        assert 'dense-flat(lossless)' in capsys.readouterr().out

    def test_info_prints_dense_flat(self, tmp_path, capsys):
        from griffith_logic.deploy.cli import _cmd_bundle, _cmd_info

        src = self._write_doc(tmp_path)
        out = tmp_path / 'flat.bundle.json'
        _cmd_bundle(_cli_args(src, out, dense_flat=True))
        capsys.readouterr()
        _cmd_info(SimpleNamespace(source=str(out), strict=True))
        assert 'dense-flat(lossless)' in capsys.readouterr().out

    def test_dense_flat_on_graph_without_chain_is_a_noop(self, tmp_path,
                                                         capsys):
        from griffith_logic.deploy.cli import _cmd_bundle

        g = IRGraph(
            inputs=[IRPort(id='x', width=4)], outputs=[IRPort(id='y', width=2)],
            nodes=[IRNode(id='g', type='group_sum',
                          params={'in_width': 4, 'k': 2, 'tau': 1.0})],
            edges=[IREdge(src='x', dst='g'), IREdge(src='g', dst='y')])
        src = tmp_path / 'g.json'
        src.write_text(json.dumps({'ir': ir_to_dict(g)}), encoding='utf-8')
        out = tmp_path / 'g.bundle.json'
        _cmd_bundle(_cli_args(src, out, dense_flat=True))
        assert '変換なし' in capsys.readouterr().out

    def test_parser_accepts_dense_flat(self):
        from griffith_logic.cli import _build_parser

        args = _build_parser().parse_args(
            ['deploy', 'bundle', 'm.json', '--dense-flat'])
        assert args.dense_flat is True
        assert args.dense_flat_verify_samples == 16
        args = _build_parser().parse_args(['deploy', 'bundle', 'm.json'])
        assert args.dense_flat is False

    def test_orthogonal_to_resynth_flag(self, tmp_path):
        """--dense-flat は --resynth と独立(併用してもエラーにならない)。
        dense のみのIRなら resynth は何もせず、flat だけが効く。"""
        from griffith_logic.deploy.cli import _cmd_bundle

        src = self._write_doc(tmp_path)
        out = tmp_path / 'both.bundle.json'
        assert _cmd_bundle(_cli_args(src, out, dense_flat=True,
                                     resynth='lossless')) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert 'lut_flat' in {n['type'] for n in bundle['ir']['nodes']}


# ---------------------------------------------------------------------------
# (g) IR/検証/バックエンド宣言
# ---------------------------------------------------------------------------
class TestIrAndBackends:
    def _flat_graph(self):
        return flatten_dense_ir(_mixed_ir())[0]

    def test_ir_json_roundtrip(self):
        g = self._flat_graph()
        g2 = ir_from_dict(json.loads(json.dumps(ir_to_dict(g))))
        x = _bits(6, 16)
        assert np.array_equal(simulate(g, x), simulate(g2, x))
        validate(g2)

    def test_mdlb_roundtrip(self, tmp_path):
        doc, _ = flatten_dense_doc(
            {'ir': ir_to_dict(_mixed_ir()), 'preproc': {}, 'labels': []})
        bundle = build_bundle(doc)
        p = save_bundle(bundle, tmp_path / 'f.mdlb')
        back = load_bundle(p)
        assert back['ir'] == json.loads(json.dumps(bundle['ir']))

    @pytest.mark.parametrize('mutate, msg', [
        (lambda p: p['gates'].__setitem__(0, [0xf + 1, 1, 2]), 'ttが0..15'),
        (lambda p: p['gates'].__setitem__(0, [6, 99, 2]), '前方参照'),
        (lambda p: p['gates'].__setitem__(0, [6, -1, 2]), '前方参照'),
        (lambda p: p['outputs'].pop(), 'outputsは長さ'),
        (lambda p: p['outputs'].__setitem__(0, 10 ** 6), '範囲外'),
    ])
    def test_validate_rejects_malformed_lut_flat(self, mutate, msg):
        g = self._flat_graph()
        node = next(n for n in g.nodes if n.type == 'lut_flat')
        node.params = json.loads(json.dumps(node.params))
        mutate(node.params)
        with pytest.raises(IRValidationError, match=msg):
            validate(g)

    def test_validate_rejects_width_mismatch(self):
        g = self._flat_graph()
        node = next(n for n in g.nodes if n.type == 'lut_flat')
        node.params['in_width'] = 7
        with pytest.raises(IRValidationError):
            validate(g)

    def test_hdl_target_rejects_lut_flat(self):
        with pytest.raises(IRValidationError, match='HDL'):
            validate(self._flat_graph(), target='hdl')

    def test_build_torch_refuses_with_reason(self):
        from griffith_logic.ir.build import build_torch

        with pytest.raises(NotImplementedError, match='再構築不可'):
            build_torch(self._flat_graph())

    def test_all_backends_accept_lut_flat(self):
        """で Web と GPU(CUDA/Metal)も lut_flat を実行できるようになった。"""
        doc, _ = flatten_dense_doc(
            {'ir': ir_to_dict(_mixed_ir()), 'preproc': {}, 'labels': []})
        bundle = build_bundle(doc)
        for backend in ('c', 'web', 'cuda', 'metal'):
            assert can_run(bundle, backend) == (True, ''), backend

    def test_resource_estimate_counts_flat_gates(self):
        from griffith_logic.export.estimate import estimate_resources

        g = self._flat_graph()
        flat = next(n for n in g.nodes if n.type == 'lut_flat')
        res = estimate_resources(g)
        assert res['lut_counts'][2] == len(flat.params['gates'])

    def test_resource_estimate_depth_counts_flat_levels(self):
        """論理深さは lut_flat の中の段数を数える(0 段扱いにしない)。段数は
        GPU の段分け(flatlevels)と一致し、置換前の lut_layer 層数を超えない。"""
        from griffith_logic.deploy.flatlevels import levelize_lut_flat
        from griffith_logic.export.estimate import estimate_resources

        orig = _mixed_ir()
        g = self._flat_graph()
        flat = next(n for n in g.nodes if n.type == 'lut_flat')
        depth = estimate_resources(g)['logic_depth']
        assert depth == levelize_lut_flat(flat.params).n_levels
        assert 0 < depth <= estimate_resources(orig)['logic_depth']

    def test_resource_estimate_depth_of_handmade_flat(self):
        """鎖 3 段・NOT は段に数えない・入力の素通しは 0 段。"""
        from griffith_logic.export.estimate import _lut_flat_depth

        # 信号: 0=定数0, 1..2=入力, 3=g0(1,2), 4=g1(3,1), 5=g2(4,3)
        params = {'in_width': 2, 'out_width': 2,
                  'gates': [[0x8, 1, 2], [0x6, 3, 1], [0xe, 4, 3]],
                  'outputs': [2 * 5 + 1, 2 * 3]}
        assert _lut_flat_depth(params) == 3
        assert _lut_flat_depth({**params, 'outputs': [2 * 3 + 1]}) == 1
        assert _lut_flat_depth({**params, 'outputs': [2 * 1, 2 * 2 + 1]}) == 0


# ---------------------------------------------------------------------------
# (f) hypothesis: 小規模ランダムIRの property
# ---------------------------------------------------------------------------
@st.composite
def _small_dense_ir(draw):
    in_w = draw(st.integers(2, 10))
    depth = draw(st.integers(1, 4))
    widths = [draw(st.integers(2, 12)) for _ in range(depth)]
    k = draw(st.sampled_from([d for d in range(1, widths[-1] + 1)
                              if widths[-1] % d == 0]))
    layers, prev = [], in_w
    for w in widths:
        idx = np.array([[draw(st.integers(0, prev - 1)) for _ in range(w)]
                        for _ in range(2)])
        tts = [draw(st.integers(0, 15)) for _ in range(w)]
        layers.append((idx, tts))
        prev = w
    return _dense_ir(layers, in_width=in_w, k=k), in_w


class TestHypothesis:
    @settings(max_examples=60, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(case=_small_dense_ir(), seed=st.integers(0, 2 ** 16))
    def test_flatten_preserves_simulate(self, case, seed):
        g, in_w = case
        new, rep = flatten_dense_ir(g, n_verify=4, seed=seed)
        validate(new)
        # 入力が小さいので全数で比較する
        n = 1 << in_w
        x = ((np.arange(n)[:, None] >> np.arange(in_w)[None, :]) & 1).astype(
            np.uint8)
        assert np.array_equal(simulate(g, x), simulate(new, x))
        assert rep['gates_after'] <= rep['gates_before']
        netlist = flat_netlist_from_ir(g)
        assert np.array_equal(simulate(new, x).astype(np.int64),
                              reduce_netlist(netlist).class_scores(x))


# ---------------------------------------------------------------------------
# (h) Web ターゲット
# ---------------------------------------------------------------------------
def _nontrivial_doc(seed=3):
    """自明ゲートが無く平坦化してもほぼ減らない doc(削減率 < C の閾値 25%)。"""
    rng = np.random.default_rng(seed)
    layers = _random_layers(rng, 40, [64, 64, 48], p_trivial=0.0)
    nontriv = np.array([0x1, 0x2, 0x4, 0x6, 0x7, 0x8, 0x9, 0xb, 0xd, 0xe])
    layers = [(idx, rng.choice(nontriv, size=len(tts))) for idx, tts in layers]
    return {'ir': ir_to_dict(_dense_ir(layers, in_width=40, k=6)),
            'preproc': {}, 'labels': []}


def _web_skip():
    import shutil

    from griffith_logic.deploy.webverify import RUNTIME_WEB
    return pytest.mark.skipif(
        shutil.which('node') is None
        or not (RUNTIME_WEB / 'dist' / 'verify.mjs').is_file(),
        reason='node と runtime/web/dist(make web-runtime)のある環境でのみ実行')


class TestWebTarget:
    def test_web_threshold_is_zero_c_threshold_unchanged(self):
        """Web は lut_flat が1ゲートあたりも速いので、削減率が C の閾値未満でも採用。"""
        from griffith_logic.deploy.plan import DENSE_FLAT_AUTO_MIN_REDUCTION

        bundle = build_bundle(_nontrivial_doc())
        _, dec_c = fd.resolve_dense_flat(bundle, mode='auto', n_verify=4)
        assert dec_c['reduction'] < DENSE_FLAT_AUTO_MIN_REDUCTION
        assert dec_c['applied'] is False
        out, dec_w = fd.resolve_dense_flat(bundle, mode='auto', n_verify=4,
                                           target='web')
        assert dec_w['applied'] is True and dec_w['min_reduction'] == 0.0
        assert 'lut_flat' in {n['type'] for n in out['ir']['nodes']}
        # 元バンドルは変えない
        assert 'lut_flat' not in {n['type'] for n in bundle['ir']['nodes']}

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match='target'):
            fd.resolve_dense_flat(build_bundle(_nontrivial_doc()),
                                  mode='auto', target='cuda')

    @pytest.mark.parametrize('mode,expect_flat', [('auto', True), ('off', False)])
    def test_codegen_web_applies_dense_flat_by_default(self, tmp_path, mode,
                                                       expect_flat, capsys):
        from griffith_logic.cli import main
        from griffith_logic.deploy.bundle import validate_bundle

        src = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(
            {'ir': ir_to_dict(_mixed_ir()), 'preproc': {}, 'labels': []}), src)
        out = tmp_path / 'out'
        argv = ['deploy', 'codegen-web', str(src), '-o', str(out), '--no-runtime']
        if mode != 'auto':
            argv += ['--dense-flat', mode]
        assert main(argv) == 0
        assert f'dense-flat({mode})' in capsys.readouterr().out
        shipped = json.loads((out / 'model.bundle.json').read_text(encoding='utf-8'))
        validate_bundle(shipped)
        types = {n['type'] for n in shipped['ir']['nodes']}
        assert ('lut_flat' in types) is expect_flat
        assert ('lut_layer' in types) is not expect_flat

    @_web_skip()
    def test_verify_web_flat_matches_original_simulate(self, tmp_path):
        """TS の lut_flat(1枚ずつ・バッチ軸パック)⇔ 元IRの simulate。"""
        from griffith_logic.deploy.webverify import verify_web_matches_simulate

        rng = np.random.default_rng(21)
        g = _dense_ir(_random_layers(rng, 40, [64, 64, 48, 48], p_trivial=0.4),
                      in_width=40, k=6)
        bundle = build_bundle({'ir': ir_to_dict(g), 'preproc': {}, 'labels': []})
        for engine in ('js', 'wasm'):
            res = verify_web_matches_simulate(
                bundle, n_samples=40, engine=engine, work_dir=tmp_path / engine,
                verbose=False, dense_flat='auto')
            assert res['dense_flat']['applied'] is True
            assert res['max_abs_diff'] == 0

    @_web_skip()
    def test_verify_web_detects_broken_flat_runtime_input(self, tmp_path,
                                                          monkeypatch):
        """ゴールデンは元IR: Web に渡す平坦化バンドルを壊すと検出される。"""
        from griffith_logic.deploy import webverify

        bundle = build_bundle(
            {'ir': ir_to_dict(_mixed_ir()), 'preproc': {}, 'labels': []})
        real = fd.resolve_dense_flat

        def broken(*a, **kw):
            out, dec = real(*a, **kw)
            out = json.loads(json.dumps(out))
            node = next(n for n in out['ir']['nodes'] if n['type'] == 'lut_flat')
            node['params']['outputs'][0] ^= 1          # 出力0の極性を反転
            return out, dec

        monkeypatch.setattr(fd, 'resolve_dense_flat', broken)
        with pytest.raises(ValueError, match='一致しない'):
            webverify.verify_web_matches_simulate(
                bundle, n_samples=16, work_dir=tmp_path, verbose=False,
                dense_flat='on')
