"""実機 bit 一致検証(mdl-fpga hw-check)。

書込済みの DE0-CV に UART でベクトルを送り、応答を参照シム(golden)と
bit 照合する。フレーム仕様は hdl/frame_rx.v ヘッダの定義が唯一の正。
demo2 hw_check.py / serial_util.py の実証手順(CP210x 自動検出・resync)を
汎用プロトコルへ移植。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from griffith_logic.export.lowering import lower_to_primitives

from .check import expected_packed
from .ir_io import load_graph
from .vectors import gen_inputs, pack_bits

REQ_SOF = 0xA5
RSP_SOF = 0x5A
CP210X_VID, CP210X_PID = 0x10C4, 0xEA60


@dataclass
class HwCheckResult:
    passed: bool
    n_vectors: int = 0
    n_ok: int = 0
    port: str = ''
    messages: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f'hw-check: {"PASS" if self.passed else "FAIL"}'
                 f'({self.n_ok}/{self.n_vectors} bit一致, port={self.port})']
        lines.extend(f'  {m}' for m in self.messages)
        return '\n'.join(lines)


def serial_signatures(board_def: dict | None) -> list[dict]:
    """ポート自動検出に使う USB 署名の一覧(board 由来 → 既定 CP210x の順)。

    board JSON の serial.usb: [{vid, pid, desc_regex}](実機採取値のみ。
    Tang Nano 20K の BL616 内蔵 USB-UART は Phase4 で採取して記録する)。
    """
    sigs: list[dict] = []
    if board_def:
        sigs.extend((board_def.get('serial') or {}).get('usb', []))
    sigs.append({'vid': CP210X_VID, 'pid': CP210X_PID,
                 'desc_regex': 'cp210'})
    return sigs


def _to_int(v) -> int | None:
    if v is None:
        return None
    return int(v, 16) if isinstance(v, str) else int(v)


def find_port(board_def: dict | None = None) -> str | None:
    import re
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for sig in serial_signatures(board_def):
        vid, pid = _to_int(sig.get('vid')), _to_int(sig.get('pid'))
        rx = sig.get('desc_regex')
        hits = []
        for p in ports:
            if vid is not None and p.vid != vid:
                continue
            if pid is not None and p.pid != pid:
                continue
            if (vid, pid) == (None, None) and rx is None:
                continue
            if (vid, pid) == (None, None) and rx is not None and \
                    not re.search(rx, f'{p.description} {p.manufacturer}',
                                  re.I):
                continue
            hits.append(p)
        if not hits and rx:
            hits = [p for p in ports
                    if re.search(rx, f'{p.description} {p.manufacturer}',
                                 re.I)]
        if hits:
            return sorted(p.device for p in hits)[0]
    return None


def build_request(seq: int, x_value: int, payload_bytes: int) -> bytes:
    payload = x_value.to_bytes(payload_bytes, 'little')
    body = bytes([seq & 0xFF]) + payload
    chk = sum(body) & 0xFF
    return bytes([REQ_SOF]) + body + bytes([chk])


def parse_response(buf: bytes, payload_bytes: int) -> tuple[int, int] | None:
    """(seq, payload値) を返す。フレーム不成立なら None。"""
    total = payload_bytes + 3
    i = buf.find(bytes([RSP_SOF]))
    if i < 0 or len(buf) - i < total:
        return None
    frame = buf[i:i + total]
    chk = sum(frame[1:-1]) & 0xFF
    if frame[-1] != chk:
        return None
    payload = int.from_bytes(frame[2:2 + payload_bytes], 'little')
    return frame[1], payload


def run_hw_check(model: str | Path, port: str | None = None,
                 n_vectors: int = 100, seed: int = 0,
                 baud: int = 115200, timeout_s: float = 1.0,
                 board: str | None = None) -> HwCheckResult:
    res = HwCheckResult(passed=False)
    try:
        import serial  # noqa: F401
    except ImportError:
        res.messages.append(
            'pyserial 未導入。`uv sync --extra dev --extra cpu --extra fpga`')
        return res
    import serial

    board_def: dict | None = None
    if board:
        from .board_db import resolve_board
        board_def = resolve_board(board)

    if port is None:
        port = find_port(board_def)
        if port is None:
            res.messages.append(
                '対象のシリアルポートが見つからない。USB-UART の接続と'
                'ドライバを確認(外付け CP210x ブリッジ、または Tang Nano 20K '
                '等の内蔵 USB-UART。既定配線はボード最初のヘッダの '
                'index 0=RX / 1=TX。manifest io.link で変更した場合はその配線)。'
                '--port COMx で明示指定も可能')
            return res
    res.port = port

    graph = lower_to_primitives(load_graph(model))
    in_width = graph.inputs[0].width
    x = gen_inputs(in_width, n_vectors, seed)
    exp_packed, out_width = expected_packed(graph, x)
    req_p = math.ceil(in_width / 8)
    rsp_q = math.ceil(out_width / 8)
    x_vals = pack_bits(x)
    res.n_vectors = len(x)

    with serial.Serial(port, baudrate=baud, bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                       timeout=0.05, write_timeout=1.0) as ser:
        n_ok = 0
        first_fail = None
        for i, (xv, ev) in enumerate(zip(x_vals, exp_packed)):
            ser.reset_input_buffer()
            ser.write(build_request(i, xv, req_p))
            ser.flush()
            deadline = time.time() + timeout_s
            buf = b''
            got = None
            while time.time() < deadline:
                buf += ser.read(256)
                parsed = parse_response(buf, rsp_q)
                if parsed is not None:
                    got = parsed
                    break
            if got is None:
                if first_fail is None:
                    first_fail = f'vec {i}: 応答タイムアウト'
                continue
            seq, payload = got
            if seq == (i & 0xFF) and payload == ev:
                n_ok += 1
            elif first_fail is None:
                first_fail = (f'vec {i}: got=0x{payload:x} exp=0x{ev:x} '
                              f'(seq {seq} vs {i & 0xFF})')
        res.n_ok = n_ok
        res.passed = n_ok == len(x)
        if first_fail:
            res.messages.append(f'最初の不一致: {first_fail}')
        if not res.passed and n_ok == 0:
            res.messages.append(
                '全滅の場合の次の一手: ①flash 済みか ②SW[9](リセット)が下'
                'か ③配線(TX/RX クロス)か ④baud 一致かを確認')
    return res
