"""M2: オンチップ自己検証(selftest)のシミュレーション検証。"""

from conftest import requires_iverilog

from mdl_fpga.check import expected_packed
from mdl_fpga.hdlgen.core import generate_core
from mdl_fpga.hdlgen.selftest import generate_selftest, generate_selftest_tb
from mdl_fpga.ir_io import load_graph
from mdl_fpga.vectors import gen_inputs, pack_bits


def _write_selftest(workdir, graph, tamper=False):
    gen = generate_core(graph)
    x = gen_inputs(gen.in_width, 256)
    exp_packed, _ = expected_packed(graph, x)
    if tamper:
        exp_packed = list(exp_packed)
        exp_packed[3] ^= 1  # 期待値を1bit破壊 → fail が立つはず
    st = generate_selftest(gen, pack_bits(x), exp_packed)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
    (workdir / 'selftest.v').write_text(st.verilog, encoding='utf-8')
    (workdir / 'tb.v').write_text(generate_selftest_tb(st.n_vectors),
                                  encoding='utf-8')
    return st


def _run(toolchain, workdir):
    comp = toolchain.run_sim_tool(
        [toolchain.iverilog, '-g2012', '-o', 'sim.vvp',
         'tb.v', 'selftest.v', 'mdl_core.v'], cwd=workdir)
    assert comp.returncode == 0, comp.stderr
    return toolchain.run_sim_tool([toolchain.vvp, 'sim.vvp'], cwd=workdir).stdout


@requires_iverilog
def test_selftest_pass(parity4_path, tmp_path, toolchain):
    graph = load_graph(parity4_path)
    st = _write_selftest(tmp_path, graph)
    assert st.n_vectors == 16
    out = _run(toolchain, tmp_path)
    assert 'ALLPASS' in out, out


@requires_iverilog
def test_selftest_detects_tamper(parity4_path, tmp_path, toolchain):
    graph = load_graph(parity4_path)
    _write_selftest(tmp_path, graph, tamper=True)
    out = _run(toolchain, tmp_path)
    assert 'FAILED' in out, out
