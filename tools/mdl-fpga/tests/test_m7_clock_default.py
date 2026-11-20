"""M7: clock_hz の解決規約(明示値 → board の clock.hz → 50MHz)。"""

from mdl_fpga.board_db import load_board
from mdl_fpga.build import resolve_clock_hz


def test_explicit_wins():
    tn = load_board('tang-nano-20k')
    assert resolve_clock_hz(12_000_000, tn) == 12_000_000


def test_board_hz_when_unspecified():
    tn = load_board('tang-nano-20k')
    assert resolve_clock_hz(None, tn) == 27_000_000
    de = load_board('de0-cv')
    assert resolve_clock_hz(None, de) == 50_000_000


def test_fallback_50mhz_without_board_hz():
    assert resolve_clock_hz(None, {'clock': {}}) == 50_000_000
    assert resolve_clock_hz(None, {}) == 50_000_000


def test_zero_and_none_treated_as_unspecified():
    tn = load_board('tang-nano-20k')
    assert resolve_clock_hz(0, tn) == 27_000_000


def test_cli_default_is_none():
    from mdl_fpga.cli import build_parser
    args = build_parser().parse_args(['build', 'x.ir.json'])
    assert args.clock_hz is None
