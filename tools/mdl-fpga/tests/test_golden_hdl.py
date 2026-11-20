"""生成物ゴールデン照合(Stage0 でスキーマ改修前の出力を契約として固定)。

qsf/sdc/生成トップ/TB を goldens/de0cv/ の捕獲値と完全一致で照合する。
F2(synth uart 引数化)・F3(TopPorts データ駆動化)のリファクタは
このテストが green のまま通ることが合格条件(de0-cv では出力不変が契約)。

比較は改行正規化付きテキスト(read_text の universal newlines)で行う:
生成側の Path.write_text は Windows で CRLF になるため、byte 比較だと
OS 依存になる。改行種以外の 1 文字の差も検出する点で byte 一致と等価。
ゴールデンの更新は禁止(スキーマ改定で必要になった場合は進捗記録に
記録の上で意図的に行う)。
"""

import json
from pathlib import Path

from conftest import build_groupsum_model
from mdl_fpga.board_db import load_board
from mdl_fpga.hdlgen.system import generate_system, generate_system_view_wrapper
from mdl_fpga.hdlgen.wrapper import generate_wrapper, generate_wrapper_tb
from mdl_fpga.io_link import resolve_uart_link
from mdl_fpga.ir_io import load_graph
from mdl_fpga.synth import write_project_files

GOLD = Path(__file__).resolve().parent / 'goldens' / 'de0cv'


def _golden(name: str) -> str:
    return (GOLD / name).read_text(encoding='utf-8')


def _assert_matches(actual: str, name: str) -> None:
    expected = _golden(name)
    assert actual == expected, (
        f'ゴールデン不一致: {name}。'
        '生成器の変更が de0-cv の出力を変えている(契約違反)。'
        '意図的なスキーマ改定の場合のみ進捗記録に記録の上で更新する')


def test_qsf_sdc_golden(tmp_path):
    # uart は F2 以降 board ではなく io.link から解決して渡す。
    # 既定リンク(GPIO_0/0/1)= 従来の board['uart'] と同値なので出力は不変
    board = load_board('de0-cv')
    uart = resolve_uart_link(board, None).to_uart_dict()
    write_project_files(tmp_path, name='top', board=board,
                        sources=['top.v', 'mdl_core.v', 'uart_rx.v',
                                 'uart_tx.v', 'frame_rx.v', 'frame_tx.v'],
                        uart=uart)
    _assert_matches((tmp_path / 'top.qsf').read_text(encoding='utf-8'),
                    'top.qsf')
    _assert_matches((tmp_path / 'top.sdc').read_text(encoding='utf-8'),
                    'top.sdc')


def test_wrapper_golden():
    _assert_matches(generate_wrapper(4, 1, clks_per_bit=16).top_verilog,
                    'top_wrapper_4x1_cpb16.v')
    _assert_matches(generate_wrapper(3, 2).top_verilog,
                    'top_wrapper_3x2.v')
    _assert_matches(generate_wrapper(4, 1, with_selftest=True).top_verilog,
                    'top_wrapper_4x1_selftest.v')


def test_wrapper_tb_golden():
    gen = generate_wrapper(4, 1, clks_per_bit=16)
    _assert_matches(generate_wrapper_tb(gen, 4), 'tb_wrapper_4x1.v')


def test_system_golden(tmp_path):
    p = tmp_path / 'gs_demo.ir.json'
    p.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    sysgen = generate_system(load_graph(p), n_images=6)
    _assert_matches(sysgen.top_verilog, 'top_system_gs6.v')
    _assert_matches(generate_system_view_wrapper('top'), 'top_view_system.v')
