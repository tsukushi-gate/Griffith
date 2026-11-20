"""保存型(自由配線)オーバーレイの RTL シミュレーション検証(iverilog)。

検証の辺:
  (a) ライブラリ ir.simulate(正解オラクル)⇔ 保存型 overlay RTL の全ベクトル一致
  (b) .mem に落ちた記録 ⇔ IR の (op, a, b) の全数照合(run_record_check。
      計算型の run_addrgen_check に対応する辺で、EDA を使わない)
  (c) 記録を壊すと本当に FAIL になるか(検証器自体の検証。RTL はレーン 1 本、
      Python の記録照合は 1 語の破損で落ちる)
  (d) サイクルカウンタ = 設計式(計算型と同じ値になること)

tiny ネット(W=32, DEPTH=3, IN=24, CLASSES=2)の**ランダム配線**で全経路を
通す。48K の実ネットは build --mode overlay の presim(少数枚)と実機で見る。
"""

import re

import pytest
from conftest import build_random_tiny, build_rot_tiny, requires_iverilog

from griffith_logic.ir.serialize import ir_from_dict
from mdl_fpga.check import (
    prepare_overlay_workdir,
    run_overlay_check,
    run_overlay_sim,
    run_record_check,
)
from mdl_fpga.hdlgen.overlay import plan_overlay

TINY_CYCLES = 34   # ceil(24/8) + 3*(32/8) + 4*3 + 1 + 6(計算型 tiny と同じ)


def _cycle_from(stdout: str) -> int:
    m = re.search(r'cycle=(\d+)', stdout)
    assert m, stdout
    return int(m.group(1))


@requires_iverilog
def test_stored_overlay_matches_golden(tmp_path):
    """ランダム配線 tiny の 64 ベクトルで RTL の予測が ir.simulate と一致。"""
    graph = ir_from_dict(build_random_tiny())
    res = run_overlay_check(graph, tmp_path / 'ov', n_vectors=64, v=8)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.n_vectors == 64
    assert res.edges['overlay_vs_golden(a)'] == 'PASS'
    assert any('stored' in m for m in res.messages)
    assert (tmp_path / 'ov' / 'lgn_overlay_stored.v').exists()
    assert (tmp_path / 'ov' / 'gatemem_lane7.mem').exists()
    assert not (tmp_path / 'ov' / 'opmem0.mem').exists()


@requires_iverilog
def test_stored_cycle_counter_matches_formula(tmp_path):
    """サイクルカウンタが N x 設計式と厳密一致し、計算型と同値になる。"""
    graph = ir_from_dict(build_random_tiny())
    plan = plan_overlay(graph, v=8, wiring='stored')
    assert plan.cycles_per_image == TINY_CYCLES
    assert plan_overlay(ir_from_dict(build_rot_tiny()),
                        v=8).cycles_per_image == TINY_CYCLES
    res = run_overlay_check(graph, tmp_path / 'ov', n_vectors=16, v=8)
    assert res.passed, res.stdout
    assert _cycle_from(res.stdout) == 16 * TINY_CYCLES
    assert f'cyc/image = {TINY_CYCLES}' in res.stdout


@requires_iverilog
def test_stored_top_system_sim(tmp_path):
    """トップ(ISSP スタブ + /2 分周 + 7セグ BCD)まで含めて一致する。"""
    graph = ir_from_dict(build_random_tiny())
    res = run_overlay_check(graph, tmp_path / 'top', n_vectors=8, v=8,
                            clock_div=2, use_top=True)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.edges['overlay_top_vs_golden(a)'] == 'PASS'
    assert 'bcd match=0008' in res.stdout


def _corrupt_lane(path, mask: int) -> None:
    """レーン 1 本の記録を全語 XOR で壊す。

    RTL シムでは **1 語だけ**の破損が必ず出力に出るとは限らない(そのゲートが
    後段で読まれない・クラススコアが動いても argmax が変わらないことがある)
    ので、観測可能性を保証するためにレーン 1 本を壊す。1 語の破損は
    run_record_check が .mem の復号で確実に捕まえる(下のテスト)。
    """
    lines = [ln for ln in path.read_text(encoding='utf-8').split('\n') if ln]
    width = len(lines[0])
    path.write_text('\n'.join(f'{int(ln, 16) ^ mask:0{width}x}'
                              for ln in lines) + '\n', encoding='utf-8')


@requires_iverilog
def test_stored_op_corruption_detected_in_sim(tmp_path):
    """記録の op フィールドを壊したら RTL シムが FAIL になる。"""
    graph = ir_from_dict(build_random_tiny())
    wd = tmp_path / 'ov'
    prepare_overlay_workdir(graph, wd, n_vectors=64, v=8, wiring='stored')
    ok, _ = run_overlay_sim(wd)
    assert ok, '前提: 破損前は PASS'

    _corrupt_lane(wd / 'gatemem_lane0.mem', 0xF0000000)   # op を全反転
    ok, out = run_overlay_sim(wd)
    assert not ok, out
    assert 'FAILED' in out


@requires_iverilog
def test_stored_address_corruption_detected_in_sim(tmp_path):
    """記録のアドレスフィールドを壊しても FAIL になる(配線の辺)。"""
    graph = ir_from_dict(build_random_tiny())
    wd = tmp_path / 'ov'
    prepare_overlay_workdir(graph, wd, n_vectors=64, v=8, wiring='stored')
    _corrupt_lane(wd / 'gatemem_lane3.mem', 0x00004000)   # a の最下位ビット
    ok, out = run_overlay_sim(wd)
    assert not ok, out


def test_record_check_full_match(tmp_path):
    """.mem の全語を復号して IR と一致する(EDA 不要の辺)。"""
    graph = ir_from_dict(build_random_tiny())
    res = run_record_check(graph, tmp_path / 'rec', v=8)
    assert res.passed, res.render()
    assert res.edges['record_vs_ir'] == 'PASS'
    assert res.n_vectors == 96                      # 32 x 3 ゲート
    assert (tmp_path / 'rec' / 'gatemem_lane0.mem').exists()


def test_record_check_detects_corruption(tmp_path):
    """1 語壊した .mem を run_record_check が FAIL にする。"""
    graph = ir_from_dict(build_random_tiny())
    wd = tmp_path / 'rec'
    assert run_record_check(graph, wd, v=8).passed
    mem = wd / 'gatemem_lane5.mem'
    lines = mem.read_text(encoding='utf-8').split('\n')
    lines[2] = f'{int(lines[2], 16) ^ 0x1:08x}'
    mem.write_text('\n'.join(lines), encoding='utf-8')
    res = run_record_check(graph, wd, v=8, regenerate=False)
    assert not res.passed
    assert res.edges['record_vs_ir'] == 'FAIL'
    assert any('一致しない' in m for m in res.messages)


def test_record_check_reports_missing_mem(tmp_path):
    graph = ir_from_dict(build_random_tiny())
    res = run_record_check(graph, tmp_path / 'empty', v=8, regenerate=False)
    assert not res.passed
    assert any('が無い' in m for m in res.messages)


@requires_iverilog
@pytest.mark.parametrize('kind', ['random', 'perm'])
def test_stored_matches_golden_for_wiring_kinds(kind, tmp_path):
    """ランダム配線・置換配線のどちらでも RTL が golden と一致する。"""
    from conftest import build_free_ffn
    graph = ir_from_dict(build_free_ffn(32, 3, 32, 2, kind=kind, seed=1))
    res = run_overlay_check(graph, tmp_path / kind, n_vectors=32, v=8)
    assert res.passed, res.render() + '\n' + res.stdout
