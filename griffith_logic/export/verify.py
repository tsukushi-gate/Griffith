"""HDLシミュレーションによる等価性検証(チェーンB)。

生成Verilogを iverilog + 自己検証テストベンチ($readmemh方式)で
シミュレートし、IR simulate(golden model)との出力一致を確認する。

cocotbではなく素のiverilogを使う理由: 依存が少なく(apt/brewの
iverilogのみ)、組み合わせ回路の等価性検証には十分な厳密さのため。
"""

import math
import os
import shutil
import subprocess

import numpy as np

from ..ir.schema import IRGraph
from ..ir.simulate import simulate
from .verilog import write_verilog

_TB_TEMPLATE = """`timescale 1ns/1ps
module tb;
    reg  [{in_msb}:0] x;
    wire [{out_msb}:0] y;
    logic_core dut(.x(x), .y(y));

    reg [{in_msb}:0]  in_vecs  [0:{last}];
    reg [{out_msb}:0] exp_vecs [0:{last}];
    integer i, errors;

    initial begin
        $readmemh("inputs.hex", in_vecs);
        $readmemh("expected.hex", exp_vecs);
        errors = 0;
        for (i = 0; i < {n_vecs}; i = i + 1) begin
            x = in_vecs[i]; #1;
            if (y !== exp_vecs[i]) begin
                errors = errors + 1;
                $display("MISMATCH %0d: got=%h exp=%h", i, y, exp_vecs[i]);
            end
        end
        if (errors == 0) $display("ALLPASS");
        else $display("FAILED %0d/%0d", errors, {n_vecs});
        $finish;
    end
endmodule
"""


_VHDL_TB_TEMPLATE = """library ieee;
use ieee.std_logic_1164.all;
use std.textio.all;

entity tb is end entity;

architecture sim of tb is
    signal x : std_logic_vector({in_msb} downto 0);
    signal y : std_logic_vector({out_msb} downto 0);
begin
    dut: entity work.logic_core port map (x => x, y => y);

    process
        file fin  : text open read_mode is "inputs.hex";
        file fexp : text open read_mode is "expected.hex";
        variable lin, lexp : line;
        variable vbuf : std_logic_vector({in_buf_msb} downto 0);
        variable ebuf : std_logic_vector({out_buf_msb} downto 0);
        variable errors : integer := 0;
    begin
        for i in 0 to {last} loop
            readline(fin, lin);
            hread(lin, vbuf);
            readline(fexp, lexp);
            hread(lexp, ebuf);
            x <= vbuf({in_msb} downto 0);
            wait for 1 ns;
            if y /= ebuf({out_msb} downto 0) then
                errors := errors + 1;
                report "MISMATCH " & integer'image(i);
            end if;
        end loop;
        if errors = 0 then
            report "ALLPASS";
        else
            report "FAILED " & integer'image(errors);
        end if;
        wait;
    end process;
end architecture sim;
"""


def iverilog_available() -> bool:
    return shutil.which('iverilog') is not None and shutil.which('vvp') is not None


def ghdl_available() -> bool:
    return shutil.which('ghdl') is not None


def _final_node(graph: IRGraph):
    """simulate/HDL生成が最終出力とみなすノード。

    出力ポートに接続されたノードを優先し、未接続なら最終トポノード
    (simulate()・to_verilog()と同一規則)。
    """
    out_src = graph.output_source()
    if out_src is not None and any(n.id == out_src for n in graph.nodes):
        return graph.node_by_id(out_src)
    return graph.topological_order()[-1]


def _pack_bits(rows: np.ndarray) -> list[int]:
    """(batch, width)の0/1配列 → bit i を i桁目に詰めた整数列"""
    return [
        sum(int(b) << i for i, b in enumerate(row)) for row in rows
    ]


def _pack_scores(scores: np.ndarray, score_bits: int) -> list[int]:
    """(batch, k)の整数スコア → score_bitsごとに連結した整数列"""
    return [
        sum(int(s) << (c * score_bits) for c, s in enumerate(row))
        for row in scores
    ]


def _hex_lines(values: list[int], width_bits: int) -> str:
    digits = (width_bits + 3) // 4
    return '\n'.join(f'{v:0{digits}x}' for v in values) + '\n'


def run_iverilog_equivalence(
    graph: IRGraph, x: np.ndarray, workdir: str,
) -> dict:
    """生成Verilogとgolden model(simulate)の一致をiverilogで検証する。

    Args:
        graph: 検証済みIR(group_sum終端を想定)
        x: (batch, in_width) の0/1入力
        workdir: 作業ディレクトリ(生成物を残す)

    Returns:
        {'passed': bool, 'n_vectors': int, 'stdout': str}
    """
    from .lowering import lower_to_primitives

    assert iverilog_available(), 'iverilogが見つからない'
    os.makedirs(workdir, exist_ok=True)

    # conv2d/or_pool2dネイティブノードはHDL生成前に展開する
    graph = lower_to_primitives(graph)

    # golden model
    expected = simulate(graph, x)

    # 出力幅の決定(verilog.pyのgroup_sum規約と一致させる)
    last = _final_node(graph)
    if last.type == 'group_sum':
        k = last.params['k']
        group = last.params['in_width'] // k
        score_bits = max(1, math.ceil(math.log2(group + 1)))
        out_width = k * score_bits
        exp_packed = _pack_scores(expected, score_bits)
    else:
        out_width = expected.shape[1]
        exp_packed = _pack_bits(expected)

    in_width = graph.inputs[0].width
    assert x.shape[1] == in_width, (x.shape[1], in_width)
    n_vecs = x.shape[0]

    write_verilog(graph, os.path.join(workdir, 'logic_core.v'))
    with open(os.path.join(workdir, 'tb.v'), 'w', encoding='utf-8') as f:
        f.write(_TB_TEMPLATE.format(
            in_msb=in_width - 1, out_msb=out_width - 1,
            last=n_vecs - 1, n_vecs=n_vecs,
        ))
    with open(os.path.join(workdir, 'inputs.hex'), 'w', encoding='utf-8') as f:
        f.write(_hex_lines(_pack_bits(x), in_width))
    with open(os.path.join(workdir, 'expected.hex'), 'w', encoding='utf-8') as f:
        f.write(_hex_lines(exp_packed, out_width))

    subprocess.run(
        ['iverilog', '-g2012', '-o', 'sim.vvp', 'tb.v', 'logic_core.v'],
        cwd=workdir, check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        ['vvp', 'sim.vvp'], cwd=workdir, check=True,
        capture_output=True, text=True,
    )
    return {
        'passed': 'ALLPASS' in result.stdout,
        'n_vectors': n_vecs,
        'stdout': result.stdout,
    }


def run_ghdl_equivalence(
    graph: IRGraph, x: np.ndarray, workdir: str,
) -> dict:
    """生成VHDLとgolden model(simulate)の一致をGHDLで検証する。

    run_iverilog_equivalence のVHDL版。同じhexベクトル形式を
    VHDL-2008のhreadで読む自己検証テストベンチを生成する。
    """
    from .lowering import lower_to_primitives
    from .vhdl import write_vhdl

    assert ghdl_available(), 'ghdlが見つからない'
    os.makedirs(workdir, exist_ok=True)

    # conv2d/or_pool2dネイティブノードはHDL生成前に展開する
    graph = lower_to_primitives(graph)

    expected = simulate(graph, x)

    last = _final_node(graph)
    if last.type == 'group_sum':
        k = last.params['k']
        group = last.params['in_width'] // k
        score_bits = max(1, math.ceil(math.log2(group + 1)))
        out_width = k * score_bits
        exp_packed = _pack_scores(expected, score_bits)
    else:
        out_width = expected.shape[1]
        exp_packed = _pack_bits(expected)

    in_width = graph.inputs[0].width
    assert x.shape[1] == in_width, (x.shape[1], in_width)
    n_vecs = x.shape[0]

    # hreadはnibble単位で読むため、バッファ幅を4の倍数に切り上げる
    in_buf = ((in_width + 3) // 4) * 4
    out_buf = ((out_width + 3) // 4) * 4

    write_vhdl(graph, os.path.join(workdir, 'logic_core.vhdl'))
    with open(os.path.join(workdir, 'tb.vhdl'), 'w', encoding='utf-8') as f:
        f.write(_VHDL_TB_TEMPLATE.format(
            in_msb=in_width - 1, out_msb=out_width - 1,
            in_buf_msb=in_buf - 1, out_buf_msb=out_buf - 1,
            last=n_vecs - 1,
        ))
    with open(os.path.join(workdir, 'inputs.hex'), 'w', encoding='utf-8') as f:
        f.write(_hex_lines(_pack_bits(x), in_width))
    with open(os.path.join(workdir, 'expected.hex'), 'w', encoding='utf-8') as f:
        f.write(_hex_lines(exp_packed, out_width))

    subprocess.run(
        ['ghdl', '-a', '--std=08', 'logic_core.vhdl', 'tb.vhdl'],
        cwd=workdir, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ['ghdl', '-e', '--std=08', 'tb'],
        cwd=workdir, check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        ['ghdl', '-r', '--std=08', 'tb'],
        cwd=workdir, check=True, capture_output=True, text=True,
    )
    out = result.stdout + result.stderr
    return {
        'passed': 'ALLPASS' in out,
        'n_vectors': n_vecs,
        'stdout': out,
    }
