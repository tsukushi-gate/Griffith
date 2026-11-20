"""システムマニフェスト(IR を汚さない上位ファイル、構想書 §3.3)。

ボード・I/O・観測・表示は IR に足さず、IR を参照するツール専用 JSON に置く。
manifest_version を必ず持つ(IR の version 検査と独立にスキーマ進化させる)。

0.2: io.link を追加。UART をどの GPIO ピンに配線するかは「ボードの物理事実」
ではなく「プロジェクト側の配線判断」なので、board JSON(v2)ではなくここが持つ。

0.3: overlay ブロックを追加(mode=overlay のときだけ意味を持つ)。レーン数 v・
検証ベクトル枚数 n_vectors・コアクロックの分周比 clock_div・配線の持ち方
wiring は「モデルの性質」ではなく「実装の選び方」なので IR ではなくここが
持つ。0.1 / 0.2 は後方互換(overlay 省略時は既定値)。wiring は 0.3 の追加
フィールドで、省略時は auto(既存の 0.3 manifest はそのまま読める)。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

SUPPORTED_MANIFEST_VERSIONS = ('0.1', '0.2', '0.3')


def _protocol_baud(protocol: str) -> int | None:
    # 'uart-115200-8n1' → 115200(解釈できなければ None)
    try:
        return int(protocol.split('-')[1])
    except (IndexError, ValueError):
        return None


class HeaderAttach(BaseModel):
    """GPIO ヘッダ経由の配線指定(index は board headers[].io の論理 index)。"""
    header: str
    rx_index: int
    tx_index: int


class RawAttach(BaseModel):
    """FPGA ピン直接指定(ヘッダ表が未収録の場合の逃げ道)。

    パターンはベンダ共用体(Intel PIN_xx / Gowin 素番号 / 英数字 BGA)。
    ボード別の厳密検証は io_link.resolve_uart_link が board の pin_format で行う。
    """
    rx_pin: str = Field(pattern=r'^(PIN_[A-Z]+\d+|[A-Z]{1,3}\d{1,3}|\d{1,3})$')
    tx_pin: str = Field(pattern=r'^(PIN_[A-Z]+\d+|[A-Z]{1,3}\d{1,3}|\d{1,3})$')


class IoLink(BaseModel):
    """ホスト⇔FPGA リンク(v1 は uart のみ。type は将来 spi/ftdi-fifo 等へ拡張)。"""
    type: str = 'uart'
    baud: int = 115200
    attach: HeaderAttach | RawAttach = Field(
        default_factory=lambda: HeaderAttach(header='GPIO_0',
                                             rx_index=0, tx_index=1))


class IoConfig(BaseModel):
    protocol: str = 'uart-115200-8n1'
    link: IoLink | None = None
    mapping: dict[str, str] = Field(default_factory=dict)


class DebugConfig(BaseModel):
    step_enable: bool = False
    observation_points: list[str] = Field(default_factory=list)


class ViewConfig(BaseModel):
    port: str
    renderer: str = 'bits'
    model_config = {'extra': 'allow'}  # shape/k 等レンダラ固有パラメータを許容


class OverlayConfig(BaseModel):
    """オーバーレイ実装の選択(manifest 0.3)。

    v          レーン数(2 の冪・8 の倍数)。速度は V に比例し、活性複製の
               M10K が V に比例して増える
    n_vectors  オンチップ検証 ROM に焼くベクトル枚数
    clock_div  コアクロックの分周比(1 / 2 / 4。2 = 50MHz を 25MHz へ)
    wiring     配線の持ち方。auto(既定)= 全層が閉形式(rot 系)なら計算型、
               外れれば保存型(自由配線)/ computed = 計算型のみ(外れたら
               拒否)/ stored = 保存型を強制
    """
    v: int = 8
    n_vectors: int = 1000
    clock_div: int = 2
    wiring: str = 'auto'


class Manifest(BaseModel):
    manifest_version: str = '0.3'
    ir: str
    board: str = 'de0-cv'
    mode: str = 'spatial'          # 'overlay' | 'spatial'(overlay 生成は M3)
    # None = 未指定(build が board の clock.hz → 50MHz の順で解決する)
    clock_hz: int | None = None
    io: IoConfig = Field(default_factory=IoConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    views: list[ViewConfig] = Field(default_factory=list)
    example: dict = Field(default_factory=dict)

    _base_dir: Path | None = None

    def ir_path(self) -> Path:
        p = Path(self.ir)
        if not p.is_absolute() and self._base_dir is not None:
            p = self._base_dir / p
        return p

    def effective_baud(self) -> int:
        """実効 baud(io.link 優先、無ければ protocol 文字列から)。"""
        if self.io.link is not None:
            return self.io.link.baud
        return _protocol_baud(self.io.protocol) or 115200


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    ver = doc.get('manifest_version')
    if ver not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(
            f'manifest_version が未対応: {ver}(対応: '
            f'{", ".join(SUPPORTED_MANIFEST_VERSIONS)})。'
            'ツールを更新するか manifest を移行する')
    m = Manifest(**doc)
    m._base_dir = path.parent
    if m.mode not in ('overlay', 'spatial'):
        raise ValueError(f'mode が不正: {m.mode}(overlay | spatial)')
    if m.overlay.clock_div not in (1, 2, 4):
        raise ValueError(
            f'overlay.clock_div が不正: {m.overlay.clock_div}(1 | 2 | 4)')
    if m.overlay.wiring not in ('auto', 'computed', 'stored'):
        raise ValueError(
            f'overlay.wiring が不正: {m.overlay.wiring}'
            '(auto | computed | stored)')
    if 'overlay' in doc and ver in ('0.1', '0.2'):
        raise ValueError(
            f'manifest_version={ver} に overlay ブロックがある。'
            'overlay は 0.3 で追加された(manifest_version を 0.3 にする)')
    if m.io.link is not None:
        pb = _protocol_baud(m.io.protocol)
        if pb is not None and pb != m.io.link.baud:
            raise ValueError(
                f'io.protocol の baud({pb})と io.link.baud({m.io.link.baud})'
                'が不一致。どちらかに揃える(io.link.baud が実効値)')
    return m


def is_manifest_file(path: str | Path) -> bool:
    """JSON が manifest か(IR かの判別。build 動詞が両方受けるため)。"""
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        return 'manifest_version' in doc
    except Exception:  # noqa: BLE001
        return False
