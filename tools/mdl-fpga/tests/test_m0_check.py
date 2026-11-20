"""M0: check(参照シム ⇔ RTL bit 一致+生成器二重化)の統合テスト。

iverilog が無い環境では skip(CI では oss-cad-suite 導入ジョブでのみ実行)。
"""

import numpy as np
from conftest import requires_iverilog

from mdl_fpga.check import (
    _write_check_files,
    expected_packed,
    run_check,
    run_sim_only,
)
from mdl_fpga.hdlgen.core import generate_core
from mdl_fpga.ir_io import load_graph
from mdl_fpga.vectors import gen_inputs


@requires_iverilog
def test_check_parity_allpass(parity4_path, tmp_path):
    res = run_check(parity4_path, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.edges['ref_vs_rtl(a)'] == 'PASS'
    assert res.edges['generator_cross(c)'] == 'PASS'
    assert res.n_vectors == 16  # 4bit は全数
    assert (tmp_path / 'run' / 'check_report.json').exists()
    assert (tmp_path / 'run' / 'name_map.json').exists()


@requires_iverilog
def test_check_groupsum_allpass(groupsum_path, tmp_path):
    res = run_check(groupsum_path, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout


@requires_iverilog
def test_check_detects_mismatch(parity4_path, tmp_path, toolchain):
    """壊れたコアを check が本当に FAIL にするか(検証器自体の検証)。"""
    graph = load_graph(parity4_path)
    # 生成器に渡すグラフの真理値表を故意に破壊(XOR → AND)
    broken = load_graph(parity4_path)
    broken.nodes[1].params['truth_tables'] = ['0x8']
    gen = generate_core(broken)
    x = gen_inputs(4, 16)
    wd = tmp_path / 'run'
    _write_check_files(wd, gen, graph, x, cross_check=True)
    comp = toolchain.run_sim_tool(
        [toolchain.iverilog, '-g2012', '-o', 'sim.vvp',
         'tb.v', 'mdl_core.v', 'lib_core.v'], cwd=wd)
    assert comp.returncode == 0, comp.stderr
    run = toolchain.run_sim_tool([toolchain.vvp, 'sim.vvp'], cwd=wd)
    assert 'ALLPASS' not in run.stdout
    assert 'MISMATCH_TOOL' in run.stdout
    assert 'MISMATCH_CROSS' in run.stdout


@requires_iverilog
def test_sim_only_writes_outputs(parity4_path, tmp_path):
    res = run_sim_only(parity4_path, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed
    lines = (tmp_path / 'run' / 'y_out.hex').read_text().split()
    assert len(lines) == 16
    # 先頭ベクトル x=0000 → パリティ0
    assert int(lines[0], 16) == 0


def test_expected_packed_groupsum(groupsum_path):
    graph = load_graph(groupsum_path)
    x = gen_inputs(4, 16)
    packed, out_width = expected_packed(graph, x)
    assert out_width == 4
    # x=1111: AND=1,OR=1,XOR=0,XNOR=1 → class0(g0,g1)=2, class1(g2,g3)=1
    x_all1 = np.ones((1, 4), dtype=np.int64)
    p, _ = expected_packed(graph, x_all1)
    assert p[0] == (1 << 2) | 2  # class1=1 が上位2bit、class0=2 が下位2bit
