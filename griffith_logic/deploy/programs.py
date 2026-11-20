"""conv2d命令列(ConvProgram)のバンドル用シリアライズ。

`export/optimize.py` が作る ConvProgram を、配布バンドルの `programs` 節へ
JSON化する。ターゲット側(C/CUDA/…)はここで焼かれた命令列をそのまま
実行するだけでよく、縮約パスを実装する必要がない。

instrs はC/GPU側で扱いやすいよう **4要素/命令のフラット配列**
(tt, a, b, slot)にする。ref のエンコードは optimize.py と同一
((idx << 2) | kind、kind∈{CONST=0, TAP=1, INSTR=2})で、変換しない。
"""

from dataclasses import asdict, fields

from ..export.optimize import (
    ConvInstr,
    ConvProgram,
    ConvProgramStats,
    build_conv_program,
    spec_from_node_params,
    verify_conv_program,
)
from ..ir.schema import IRGraph

_STATS_FIELDS = tuple(f.name for f in fields(ConvProgramStats))


def program_to_dict(prog: ConvProgram) -> dict:
    """ConvProgram → JSON化可能なdict。"""
    instrs: list[int] = []
    for ins in prog.instrs:
        instrs.extend((int(ins.tt), int(ins.a), int(ins.b), int(ins.slot)))
    return {
        'taps': [int(v) for v in prog.taps],
        'n_instrs': len(prog.instrs),
        'instrs': instrs,
        'n_slots': int(prog.n_slots),
        'oc_out': [int(v) for v in prog.oc_out],
        'write_start': [int(v) for v in prog.write_start],
        'write_oc': [int(v) for v in prog.write_oc],
        'stats': asdict(prog.stats),
    }


def program_from_dict(d: dict) -> ConvProgram:
    """program_to_dict の逆変換(往復同一性テスト・C検証で使う)。"""
    flat = list(d['instrs'])
    n = int(d['n_instrs'])
    if len(flat) != 4 * n:
        raise ValueError(
            f"instrs長が不正: {len(flat)} (期待 4*n_instrs={4 * n})"
        )
    instrs = [
        ConvInstr(tt=int(flat[4 * j]), a=int(flat[4 * j + 1]),
                  b=int(flat[4 * j + 2]), slot=int(flat[4 * j + 3]))
        for j in range(n)
    ]
    stats_raw = d.get('stats', {})
    stats = ConvProgramStats(**{name: int(stats_raw.get(name, 0))
                                for name in _STATS_FIELDS})
    return ConvProgram(
        taps=[int(v) for v in d['taps']],
        instrs=instrs,
        n_slots=int(d['n_slots']),
        oc_out=[int(v) for v in d['oc_out']],
        write_start=[int(v) for v in d['write_start']],
        write_oc=[int(v) for v in d['write_oc']],
        stats=stats,
    )


#: ゲート木を持つIRノード型 → 命令列化の3点セット。
#: 新しいゲート木ノード(RDDLGN など)は**ここに1行登録するだけ**で、
#: 命令列縮約・バンドル・C/CUDA/Metal の全経路に載る。
GATE_PROGRAM_BUILDERS: dict[str, tuple] = {}


def register_gate_program(node_type: str, *, spec_of, build, verify):
    """ゲート木ノード型を命令列化の対象として登録する。

    Args:
        node_type: IRノード型(`ir/build.py::register_builder` と同じキー)
        spec_of: `node.params` → 命令列ビルダのspec
        build: spec → ConvProgram 相当(`program_to_dict` が扱える形)
        verify: (spec, prog) → 縮約前後の全数照合。不一致なら例外
    """
    GATE_PROGRAM_BUILDERS[node_type] = (spec_of, build, verify)


register_gate_program('conv2d', spec_of=spec_from_node_params,
                      build=build_conv_program, verify=verify_conv_program)


def build_programs(graph: IRGraph, *, verify: bool = True) -> dict:
    """IR中のゲート木ノードを命令列化し、{node_id: program_dict} を返す。

    対象は `GATE_PROGRAM_BUILDERS` に登録された型(現状は conv2d のみ)。

    Args:
        graph: 対象IR
        verify: 縮約前後の全数真理値表照合(verify_conv_program)を行う。
            既定True。**Falseにするのは検証テストで故障注入する時だけ**で、
            通常のバンドル生成では必ずTrueのまま使うこと(焼き込んだ命令列が
            元の木と等価であることを、書き出し前に保証するのがこの節の前提)
    """
    programs: dict[str, dict] = {}
    for node in graph.nodes:
        entry = GATE_PROGRAM_BUILDERS.get(node.type)
        if entry is None:
            continue
        spec_of, build, verify_fn = entry
        spec = spec_of(node.params)
        prog = build(spec)
        if verify:
            verify_fn(spec, prog)
        programs[node.id] = program_to_dict(prog)
    return programs
