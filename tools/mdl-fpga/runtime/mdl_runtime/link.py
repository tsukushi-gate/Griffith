"""UART リンク(送受信ループ。serial_factory 注入でテスト可能)。"""

from __future__ import annotations

import time

from . import ports as _ports
from . import protocol


def _pyserial_factory(port: str, baud: int):
    import serial
    return serial.Serial(port, baudrate=baud, bytesize=serial.EIGHTBITS,
                         parity=serial.PARITY_NONE,
                         stopbits=serial.STOPBITS_ONE,
                         timeout=0.05, write_timeout=1.0)


class UartLink:
    """半二重の要求/応答リンク(1インフライト、SOF resync、タイムアウト)。"""

    def __init__(self, port: str, baud: int, *, in_width: int, out_width: int,
                 timeout_s: float = 1.0, serial_factory=None):
        self.port = port
        self.timeout_s = timeout_s
        self._req_p = protocol.payload_bytes(in_width)
        self._rsp_q = protocol.payload_bytes(out_width)
        self._ser = (serial_factory or _pyserial_factory)(port, baud)

    @classmethod
    def from_contract(cls, contract: dict, port: str | None = None,
                      *, timeout_s: float = 1.0, serial_factory=None
                      ) -> UartLink:
        port = port or _ports.find_port(contract)
        if port is None:
            raise ConnectionError(
                'シリアルポートが見つからない(契約の USB 署名に一致なし)。'
                'ボードの USB 接続とドライバを確認するか --port で明示指定')
        return cls(port, contract['uart']['baud'],
                   in_width=contract['in_width'],
                   out_width=contract['out_width'],
                   timeout_s=timeout_s, serial_factory=serial_factory)

    def query(self, seq: int, x_value: int) -> int:
        """1ベクトル送信→応答ペイロード値。タイムアウトは TimeoutError。"""
        self._ser.reset_input_buffer()
        self._ser.write(protocol.build_request(seq, x_value, self._req_p))
        self._ser.flush()
        deadline = time.time() + self.timeout_s
        buf = b''
        while time.time() < deadline:
            buf += self._ser.read(256)
            got = protocol.parse_response(buf, self._rsp_q)
            if got is not None:
                rsp_seq, value = got
                if rsp_seq != (seq & 0xFF):
                    buf = b''      # 古い応答 → 読み直し(resync)
                    continue
                return value
        raise TimeoutError(
            f'応答タイムアウト(port={self.port})。①書込済みか ②リセット'
            'スイッチ位置 ③配線/baud を確認')

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> UartLink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
