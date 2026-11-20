"""M1: cocotb 対話ブリッジの実証(リスク R2 の回帰テスト)。

SimSession 経由で RTL を 1 クロックずつ進め、全ノードの値が
ライブラリ simulate_trace(golden)と bit 一致することを確認する。
= 「クリック=1clk で値が読める」ことの機械的証明。
"""

import importlib.util

import pytest
from conftest import _tc, requires_iverilog

from mdl_fpga.ir_io import load_graph
from mdl_fpga.vectors import gen_inputs, pack_bits, pack_scores

requires_cocotb = pytest.mark.skipif(
    importlib.util.find_spec('cocotb') is None
    or importlib.util.find_spec('cocotb_tools') is None,
    reason='cocotb 未導入(uv sync --extra fpga)',
)


def _expected_node_hex(trace, node, row_idx):
    row = trace[node['id']][row_idx]
    if node['type'] == 'group_sum':
        return pack_scores(row.reshape(1, -1), node['score_bits'])[0]
    return pack_bits(row.reshape(1, -1))[0]


def _run_session_and_compare(model_path, tmp_path, n_check=8):
    from griffith_logic.export.lowering import lower_to_primitives
    from griffith_logic.ir.simulate import simulate_trace

    from mdl_fpga.viewer.session import SimSession

    graph = lower_to_primitives(load_graph(model_path))
    x = gen_inputs(graph.inputs[0].width, 256)[:n_check]
    trace = simulate_trace(graph, x)
    x_vals = pack_bits(x)

    session = SimSession(model=str(model_path), workdir=tmp_path / 'view')
    try:
        meta = session.start()
        assert meta['in_width'] == graph.inputs[0].width
        session.reset()

        prev_y_comb = None
        for i, xv in enumerate(x_vals):
            session.set_x(f'{xv:x}')
            r = session.step(1)   # xin 確定 → 組み合わせ確定
            assert r['ok'], r
            assert r['xz'] == [], f'X/Z 混入: {r["xz"]}'
            assert int(r['values']['x'], 16) == xv

            for node in meta['nodes']:
                exp = _expected_node_hex(trace, node, i)
                got = int(r['values'][node['id']], 16)
                assert got == exp, \
                    f'vec{i} node {node["id"]}: got={got:x} exp={exp:x}'

            final = _expected_node_hex(trace, meta['nodes'][-1], i)
            assert int(r['values']['y_comb'], 16) == final
            if prev_y_comb is not None:   # y_reg は 1 クロック遅れ
                assert int(r['values']['y'], 16) == prev_y_comb
            prev_y_comb = final

        # ステップ数の一貫性(reset 後 n_check 回)
        assert session.cycle == n_check
    finally:
        session.close()


@requires_iverilog
@requires_cocotb
def test_bridge_parity_matches_trace(parity4_path, tmp_path):
    _run_session_and_compare(parity4_path, tmp_path, n_check=8)


@requires_iverilog
@requires_cocotb
def test_bridge_groupsum_matches_trace(groupsum_path, tmp_path):
    _run_session_and_compare(groupsum_path, tmp_path, n_check=8)


@requires_iverilog
@requires_cocotb
def test_bridge_reset_and_read(parity4_path, tmp_path):
    from mdl_fpga.viewer.session import SimSession
    session = SimSession(model=str(parity4_path), workdir=tmp_path / 'view')
    try:
        session.start()
        session.set_x('f')
        session.step(2)
        r = session.reset()
        assert r['cycle'] == 0
        assert int(r['values']['x'], 16) == 0   # rst で xin クリア
        r2 = session.read()                     # read は時間を進めない
        assert r2['cycle'] == 0
    finally:
        session.close()


# conftest の toolchain 参照(未使用警告の回避を兼ねた存在確認)
def test_toolchain_available_for_bridge():
    assert _tc is not None
