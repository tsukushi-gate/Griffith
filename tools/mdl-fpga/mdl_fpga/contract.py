"""io_contract.json 生成(STANDALONE.md の生成源(a))。

スタンドアローンランタイム(runtime/mdl_runtime、mdl 非依存)がボード DB・
IR 無しで動くための全情報を、IR/manifest/board から機械生成する:
入出力の幅と意味づけ・UART フレーミング参照・ポート自動検出用 USB 署名・
書込コマンド・golden ベクトル(参照シム由来= selftest は hw-check の配布版)。

フレーミング仕様の唯一の正は hdl/frame_rx.v(hw_check.py と同一)。
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

from griffith_logic.export.lowering import lower_to_primitives

from .check import expected_packed
from .hw_check import serial_signatures
from .ir_io import load_graph
from .manifest import is_manifest_file, load_manifest
from .vectors import gen_inputs, pack_bits

CONTRACT_VERSION = 1
FRAMING = 'mdl-frame-v1'


def _terminal_scores(graph) -> tuple[int, int] | None:
    """終端が group_sum なら (k, score_bits)。check.expected_packed と同規約。"""
    out_src = graph.output_source()
    nodes = {n.id: n for n in graph.nodes}
    last = nodes.get(out_src) or graph.topological_order()[-1]
    if last.type == 'group_sum':
        k = last.params['k']
        group = last.params['in_width'] // k
        return k, max(1, math.ceil(math.log2(group + 1)))
    return None


def flash_spec(board: dict, bitstream_arc: str) -> dict:
    """openFPGALoader の書込コマンド仕様(board DB 不要で実行できる形)。"""
    det = board.get('detection') or {}
    if det.get('ofl_board'):
        args = ['-b', det['ofl_board']]
    else:
        args = ['-c', det.get('cable', 'usb-blaster')]
    spec = {'tool': 'openFPGALoader', 'args': args, 'bitstream': bitstream_arc}
    note = (board.get('serial') or {}).get('note')
    if note:
        spec['driver_note'] = note
    return spec


def build_io_contract(model: str | Path, board: dict, *,
                      bitstream_arc: str, baud: int = 115200,
                      n_goldens: int = 32, seed: int = 0) -> tuple[dict, dict]:
    """(io_contract, goldens) を返す。model は IR か manifest のパス。"""
    if is_manifest_file(model):
        mf = load_manifest(model)
        baud = mf.effective_baud()
        model = mf.ir_path()
    graph = lower_to_primitives(load_graph(model))

    in_width = graph.inputs[0].width
    x = gen_inputs(in_width, n_goldens, seed)
    exp, out_width = expected_packed(graph, x)
    x_vals = pack_bits(x)

    scores = _terminal_scores(graph)
    out_spec = {
        'name': getattr(graph.outputs[0], 'id', 'y'),
        'width': out_width,
        'semantics': getattr(graph.outputs[0], 'semantics', None),
    }
    if scores is not None:
        k, score_bits = scores
        out_spec['semantics'] = out_spec['semantics'] or 'class_scores'
        out_spec['k'] = k
        out_spec['score_bits'] = score_bits

    meta = getattr(graph, 'metadata', None) or {}
    name = (meta.get('name') if isinstance(meta, dict) else None) \
        or Path(str(model)).stem
    contract = {
        'contract_version': CONTRACT_VERSION,
        'name': name,
        'generated': _dt.date.today().isoformat(),
        'inputs': [{
            'name': getattr(graph.inputs[0], 'id', 'x'),
            'width': in_width,
            'semantics': getattr(graph.inputs[0], 'semantics', None),
        }],
        'outputs': [out_spec],
        'in_width': in_width,
        'out_width': out_width,
        'uart': {
            'baud': baud,
            'framing': FRAMING,
            'req_sof': '0xA5',
            'rsp_sof': '0x5A',
            'req_payload_bytes': math.ceil(in_width / 8),
            'rsp_payload_bytes': math.ceil(out_width / 8),
            'note': '唯一の正は hdl/frame_rx.v: '
                    '[SOF][seq][payload LE][checksum=(seq+payload)&0xFF]',
        },
        'serial': {'usb': serial_signatures(board)},
        'flash': flash_spec(board, bitstream_arc),
        'board': {'key': board.get('key'), 'name': board.get('name')},
        'goldens': {'file': 'goldens.json', 'n': len(x_vals), 'seed': seed},
    }
    goldens = {
        'in_width': in_width,
        'out_width': out_width,
        'seed': seed,
        'vectors': [{'x': f'0x{xv:x}', 'y': f'0x{ev:x}'}
                    for xv, ev in zip(x_vals, exp)],
    }
    return contract, goldens
