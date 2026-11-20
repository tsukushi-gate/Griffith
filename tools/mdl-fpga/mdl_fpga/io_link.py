"""io.link(ホスト⇔FPGA リンク)の解決。

board JSON(物理事実)と manifest の io.link(プロジェクト側の配線判断)から、
合成に使う UART ピン割当(ResolvedUart)を決める。

ポート名 UART_RXD / UART_TXD はボードの事実ではなく生成 HDL 側の規約なので、
ここで定数として持つ(de0-cv の従来出力と byte 一致が自明に成立する根拠)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .board_db import PIN_FORMATS, pin_format_of, resolve_header_io
from .manifest import IoLink, RawAttach

UART_RX_PORT = 'UART_RXD'
UART_TX_PORT = 'UART_TXD'

# 既定リンク = GPIO_0 / rx=0 / tx=1 / 115200(従来の DE0-CV 実配線と同値)。
# link=None のときはボードの最初のヘッダへ同じ index 0/1 で読み替える
# (_default_link_for)。ヘッダ名がボードごとに違っても既定で動くようにする
DEFAULT_UART_LINK = IoLink()


def _default_link_for(board: dict) -> IoLink:
    headers = board.get('headers') or []
    if headers:
        at = DEFAULT_UART_LINK.attach
        return IoLink(baud=DEFAULT_UART_LINK.baud,
                      attach=type(at)(header=headers[0]['name'],
                                      rx_index=at.rx_index,
                                      tx_index=at.tx_index))
    return DEFAULT_UART_LINK


@dataclass(frozen=True)
class ResolvedUart:
    rx_port: str
    rx_pin: str
    tx_port: str
    tx_pin: str
    baud: int
    source: str                       # 'manifest' | 'default'
    rx_physical: int | None = None    # コネクタ物理ピン番号(bundle の配線表用)
    tx_physical: int | None = None

    def to_uart_dict(self) -> dict:
        """synth.write_project_files の uart 引数形(旧 board['uart'] と同形)。"""
        return {'rx': {'port': self.rx_port, 'pin': self.rx_pin},
                'tx': {'port': self.tx_port, 'pin': self.tx_pin}}


def resolve_uart_link(board: dict, link: IoLink | None) -> ResolvedUart:
    """io.link → ResolvedUart。link=None は既定リンク(従来配線)を使う。"""
    source = 'manifest' if link is not None else 'default'
    link = link if link is not None else _default_link_for(board)
    if link.type != 'uart':
        raise ValueError(
            f'io.link.type={link.type!r} は未対応(v1 は uart のみ。'
            'spi / ftdi-fifo 等は将来拡張)')
    at = link.attach
    if isinstance(at, RawAttach):
        # manifest 側の緩い共用体パターンを、ボードの pin_format で厳密化する
        fmt = pin_format_of(board)
        pat = re.compile(PIN_FORMATS[fmt])
        for label, pin in (('rx_pin', at.rx_pin), ('tx_pin', at.tx_pin)):
            if not pat.match(pin):
                raise ValueError(
                    f'io.link の {label}={pin!r} がボード {board.get("key")} の'
                    f'ピン形式 {fmt}({PIN_FORMATS[fmt]})に不一致。'
                    'ボードのピン表記(例は boards/*.json)に合わせる')
        return ResolvedUart(UART_RX_PORT, at.rx_pin, UART_TX_PORT, at.tx_pin,
                            link.baud, source)
    rx = resolve_header_io(board, at.header, at.rx_index)
    tx = resolve_header_io(board, at.header, at.tx_index)
    return ResolvedUart(UART_RX_PORT, rx['pin'], UART_TX_PORT, tx['pin'],
                        link.baud, source,
                        rx_physical=rx.get('physical'),
                        tx_physical=tx.get('physical'))
