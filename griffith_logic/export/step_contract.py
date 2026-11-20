"""1 ステップ IR の契約。仕様は docs/reference/step-ir-contract.md。

再帰モデル(RDDLGN の encoder / decoder、文字 LM、制御器、収束型セル網など)を FPGA の
再帰ステップラッパ(状態レジスタ + 帰還)で動かすため、1 ステップ分を**組み合わせ回路の IR**
として切り出す。IR 自体はループ構造を持たない(状態の帰還・初期値・ステップ数はラッパ側の
宣言)。この契約は FPGA 側の `RecurrentSpec`(manifest の `recurrent`: state_in / state_out /
init / steps)が受け取れる形を学習側で先に検査するためのもの。

    入力ポート x = concat([外部入力ビット | 状態ビット])       幅 in_width
    出力ポート y = concat([次状態ビット | 読み出しビット])     幅 out_width
    state_in  = [in_width − state_width, in_width)    状態は入力の末尾の連続 1 区間
    state_out = [0, state_width)                      次状態は出力の先頭の連続 1 区間

禁止: 出力ノードが group_sum(整数スコアからは状態ビットが取れない。popcount / argmax は
FPGA 側の終端部品)、recurrent / scan_var / blackbox ノード(組み合わせ回路でない)、
add(mode='sum')(整数出力)、外部入力 0 ビット(state_width ≥ in_width)。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.schema import IRGraph
from ..ir.validate import IRValidationError, validate

__all__ = ['StepIRContractError', 'StepIRSpec', 'check_step_ir', 'step_ir_manifest']

_FORBIDDEN_NODE_TYPES = ('recurrent', 'scan_var', 'blackbox')


class StepIRContractError(ValueError):
    """1 ステップ IR の契約違反。"""


@dataclass(frozen=True)
class StepIRSpec:
    """契約検査に通った 1 ステップ IR の形。"""

    in_width: int
    out_width: int
    state_width: int
    state_in: tuple[int, int]
    state_out: tuple[int, int]

    @property
    def ext_width(self) -> int:
        """外部入力(状態以外)のビット数"""
        return self.in_width - self.state_width

    @property
    def readout_width(self) -> int:
        """読み出し(次状態以外)のビット数"""
        return self.out_width - self.state_width


def check_step_ir(graph: IRGraph, state_width: int) -> StepIRSpec:
    """1 ステップ IR が契約を満たすか検査し、状態ビットの区間を返す。

    Args:
        graph: 1 ステップ分の組み合わせ IR
        state_width: 状態ビット数(入力の末尾 / 出力の先頭)

    Raises:
        StepIRContractError: 契約違反(理由を文で持つ)
    """
    state_width = int(state_width)
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise StepIRContractError(
            f'入力ポート 1 本・出力ポート 1 本が必要: inputs={len(graph.inputs)}, '
            f'outputs={len(graph.outputs)}')
    try:
        validate(graph)
    except IRValidationError as e:
        raise StepIRContractError(f'IR が validate を通らない: {e}') from e
    bad = [n.id for n in graph.nodes if n.type in _FORBIDDEN_NODE_TYPES]
    if bad:
        raise StepIRContractError(
            f'組み合わせ回路でないノードがある(recurrent / scan_var / blackbox): {bad}')
    sums = [n.id for n in graph.nodes if n.type == 'add' and n.params.get('mode') == 'sum']
    if sums:
        raise StepIRContractError(
            f"add(mode='sum') は整数出力なので 1 ステップ IR に置けない: {sums}")
    src = graph.output_source()
    out_node = next((n for n in graph.nodes if n.id == src), None)
    if out_node is not None and out_node.type == 'group_sum':
        raise StepIRContractError(
            '出力が group_sum(整数スコア)なので状態ビットが取れない。'
            '状態と読み出しは lut_layer / concat の出力ビットにし、popcount / argmax は'
            ' FPGA 側の終端部品に任せる')
    in_width = int(graph.inputs[0].width)
    out_width = int(graph.outputs[0].width)
    if state_width < 1:
        raise StepIRContractError(f'state_width は 1 以上: {state_width}')
    if state_width >= in_width:
        raise StepIRContractError(
            f'state_width({state_width}) は入力幅({in_width})未満が必要(外部入力 1 bit 以上)')
    if state_width > out_width:
        raise StepIRContractError(
            f'state_width({state_width}) が出力幅({out_width})を超える')
    return StepIRSpec(
        in_width=in_width, out_width=out_width, state_width=state_width,
        state_in=(in_width - state_width, in_width), state_out=(0, state_width),
    )


def step_ir_manifest(spec: StepIRSpec, init_bits, steps: int | None = None) -> dict:
    """FPGA 側 manifest の `recurrent` 欄(state_in / state_out / init / steps)の形にする。

    init は '0x…' の 16 進で、bit i = 状態ビット i(state_in の lo から数える)。
    """
    bits = [int(v) for v in init_bits]
    if len(bits) != spec.state_width or any(v not in (0, 1) for v in bits):
        raise StepIRContractError(
            f'init_bits は 0/1 の {spec.state_width} 個が必要: {len(bits)} 個')
    init = 0
    for i, v in enumerate(bits):
        init |= v << i
    out = {
        'state_in': list(spec.state_in),
        'state_out': list(spec.state_out),
        'init': f'0x{init:x}',
    }
    if steps is not None:
        if int(steps) < 1:
            raise StepIRContractError(f'steps は 1 以上: {steps}')
        out['steps'] = int(steps)
    return out
