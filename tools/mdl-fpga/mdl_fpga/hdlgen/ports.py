"""生成トップのポート名/幅(TopPorts)。

board JSON(物理事実)から導出する唯一の口。既定値は DE0-CV の従来リテラルと
同値で、既定 TopPorts での生成物は Stage0 ゴールデンと byte 一致する(契約)。

wrapper.py / system.py はポート名をリテラルで持たず、必ずここを参照する
(board JSON の port 名と生成 HDL の暗黙一致という二重管理の解消、F3)。

M7: リセット元(switch/button)と極性・LED 極性を追加。スライドスイッチの
無いボード(Tang Nano 20K 等)は button 最上位ビットをリセットに使う
(SW[9] 規約のミラー)。既定値は全て DE0-CV 従来動作と同値。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopPorts:
    clock: str = 'CLOCK_50'
    switch: str = 'SW'
    switch_width: int = 10
    led: str = 'LEDR'
    led_width: int = 10
    button: str = 'KEY'
    button_width: int = 4
    hex_digits: tuple[str, ...] = ('HEX0', 'HEX1', 'HEX2',
                                   'HEX3', 'HEX4', 'HEX5')
    uart_rx: str = 'UART_RXD'
    uart_tx: str = 'UART_TXD'
    # リセット元: 'switch'(既定=SW 最上位、DE0-CV 従来)| 'button'
    reset_group: str = 'switch'
    reset_index: int | None = None    # None = 群の最上位ビット
    reset_active_low: bool = False    # button リセットは board の極性に従う
    led_active_low: bool = False      # Tang Nano 20K 等の負論理 LED

    @property
    def reset_switch_index(self) -> int:
        """リセットに使うスイッチ(最上位ビット)。旧名互換(switch 前提の呼び手用)。"""
        return self.switch_width - 1

    @property
    def reset_port(self) -> str:
        return self.switch if self.reset_group == 'switch' else self.button

    @property
    def reset_width(self) -> int:
        return (self.switch_width if self.reset_group == 'switch'
                else self.button_width)

    @property
    def reset_bit(self) -> int:
        return self.reset_index if self.reset_index is not None \
            else self.reset_width - 1

    @classmethod
    def from_board(cls, board: dict, uart=None) -> TopPorts:
        """board JSON から構築(uart は io_link.ResolvedUart か None)。

        board に無い資源は既定値のまま(存在チェックは synth の groups 検証と
        生成器側の NotImplementedError が担う)。
        """
        kw: dict = {}
        clock = board.get('clock', {}).get('port')
        if clock:
            kw['clock'] = clock
        for key, (name_kw, width_kw) in {
            'switch': ('switch', 'switch_width'),
            'led': ('led', 'led_width'),
            'button': ('button', 'button_width'),
        }.items():
            grp = board.get(key)
            if grp:
                kw[name_kw] = grp['port']
                kw[width_kw] = len(grp['pins'])
        hx = board.get('hex')
        if hx:
            kw['hex_digits'] = tuple(d['port'] for d in hx['digits'])
        if uart is not None:
            kw['uart_rx'] = uart.rx_port
            kw['uart_tx'] = uart.tx_port
        led = board.get('led')
        if led and led.get('active_low'):
            kw['led_active_low'] = True
        # スイッチ非搭載ボードはボタン最上位をリセットに(SW[9] 規約のミラー)
        if not board.get('switch') and board.get('button'):
            kw['reset_group'] = 'button'
            if board['button'].get('active_low'):
                kw['reset_active_low'] = True
        return cls(**kw)
