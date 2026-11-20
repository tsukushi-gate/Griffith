"""オーバーレイの RTL シミュレーション検証(iverilog)。

検証の辺:
  (a) ライブラリ ir.simulate(正解オラクル)⇔ overlay RTL の全ベクトル一致
  (b) RTL が発行する配線アドレス ⇔ Python 参照列の全数一致(計算型配線の要)
  (c) 破損した .mem を本当に FAIL にするか(検証器自体の検証)
  (d) サイクルカウンタ = 設計式(先行実験と同じサイクルモデル)

tiny ネット(W=32, DEPTH=3, IN=24/22, CLASSES=2)で全経路を通す。48K / 96K の
実ネットは build --mode overlay の presim(少数枚)と実機で見る。
"""

import re

import pytest
from conftest import build_rot_tiny, build_stretched_tiny, requires_iverilog

from griffith_logic.ir.serialize import ir_from_dict
from mdl_fpga.check import (
    prepare_overlay_workdir,
    run_addrgen_check,
    run_overlay_check,
    run_overlay_sim,
)
from mdl_fpga.hdlgen.overlay import plan_overlay

TINY_CYCLES = 34   # ceil(24/8) + 3*(32/8) + 4*3 + 1 + 6


def _cycle_from(stdout: str) -> int:
    m = re.search(r'cycle=(\d+)', stdout)
    assert m, stdout
    return int(m.group(1))


@requires_iverilog
@pytest.mark.parametrize('builder', [build_rot_tiny, build_stretched_tiny])
def test_overlay_matches_golden(builder, tmp_path):
    """全ベクトルで overlay RTL の予測が ir.simulate と一致する。"""
    graph = ir_from_dict(builder())
    res = run_overlay_check(graph, tmp_path / 'ov', n_vectors=64, v=8)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.n_vectors == 64
    assert res.edges['overlay_vs_golden(a)'] == 'PASS'
    assert (tmp_path / 'ov' / 'overlay_check_report.json').exists()


@requires_iverilog
def test_cycle_counter_matches_formula(tmp_path):
    """サイクルカウンタが N x 設計式と厳密一致する(±0)。"""
    graph = ir_from_dict(build_rot_tiny())
    plan = plan_overlay(graph, v=8)
    assert plan.cycles_per_image == TINY_CYCLES
    res = run_overlay_check(graph, tmp_path / 'ov', n_vectors=16, v=8)
    assert res.passed, res.stdout
    assert _cycle_from(res.stdout) == 16 * TINY_CYCLES
    assert f'cyc/image = {TINY_CYCLES}' in res.stdout


@requires_iverilog
@pytest.mark.parametrize('builder', [build_rot_tiny, build_stretched_tiny])
def test_addrgen_full_match(builder, tmp_path):
    """RTL が発行する全アドレスが Python 参照列と一致する。"""
    graph = ir_from_dict(builder())
    res = run_addrgen_check(graph, tmp_path / 'addr', v=8)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.edges['addrgen_vs_python'] == 'PASS'
    # 3 層 x 4 語 x 8 レーン x {a,b} x {p1, RAM 提示}
    assert 'checks=384' in res.stdout


@requires_iverilog
def test_top_system_sim(tmp_path):
    """トップ(ISSP スタブ + /2 分周 + 7セグ BCD)まで含めて一致する。"""
    graph = ir_from_dict(build_rot_tiny())
    res = run_overlay_check(graph, tmp_path / 'top', n_vectors=8, v=8,
                            clock_div=2, use_top=True)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.edges['overlay_top_vs_golden(a)'] == 'PASS'
    assert 'bcd match=0008' in res.stdout


@requires_iverilog
def test_opmem_corruption_detected(tmp_path):
    """opmem を 1 語壊したら FAIL になるか(検証器自体の検証)。"""
    graph = ir_from_dict(build_rot_tiny())
    wd = tmp_path / 'ov'
    prepare_overlay_workdir(graph, wd, n_vectors=64, v=8)
    ok, _ = run_overlay_sim(wd)
    assert ok, '前提: 破損前は PASS'

    mem = wd / 'opmem0.mem'
    lines = mem.read_text(encoding='utf-8').split('\n')
    lines[0] = f'{int(lines[0], 16) ^ 0xFFFFFFFF:08x}'    # 語 0 の全 op を反転
    mem.write_text('\n'.join(lines), encoding='utf-8')
    ok, out = run_overlay_sim(wd)
    assert not ok, out
    assert 'FAILED' in out


@requires_iverilog
def test_addrinit_corruption_detected(tmp_path):
    """配線初期値を 1 行壊したらアドレス照合が FAIL になる。"""
    graph = ir_from_dict(build_stretched_tiny())
    wd = tmp_path / 'addr'
    prepare_overlay_workdir(graph, wd, n_vectors=1, v=8)
    mem = wd / 'addrinit_a.mem'
    lines = mem.read_text(encoding='utf-8').split('\n')
    lines[0] = f'{(int(lines[0], 16) + 1) % 22:04x}'
    mem.write_text('\n'.join(lines), encoding='utf-8')

    from mdl_fpga.check import _OVERLAY_CORE_SOURCES
    from mdl_fpga.toolchain import discover
    tc = discover()
    comp = tc.run_sim_tool(
        [tc.iverilog, '-g2012', '-o', 'sim_addrgen.vvp', 'tb_addrgen.v',
         *_OVERLAY_CORE_SOURCES], cwd=wd)
    assert comp.returncode == 0, comp.stderr
    run = tc.run_sim_tool([tc.vvp, 'sim_addrgen.vvp'], cwd=wd, timeout=600)
    assert 'ALLPASS' not in run.stdout
    assert 'MISMATCH_ADDR' in run.stdout
