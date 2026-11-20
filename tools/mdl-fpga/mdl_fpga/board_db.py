"""ボード DB(mdl_fpga/boards/*.json)。

最初から複数ボード前提の設計(v1 の Tier1 は DE0-CV のみ)。
「対応」と約束するのは Tier1 だけ(3 Tier 表記、構想書 §2)。

schema_version=2: board JSON は「ボードに固定された物理事実」のみを持つ。
プロジェクト側の割当(UART をどの GPIO ピンに配線するか等)は manifest の
io.link が持ち、ここには置かない。**ピン割当の創作は絶対禁止**
(headers[].io には確認済みのピンのみ収録し、未確認 index は載せない。
detection の IDCODE/USB 署名も実機採取値のみ)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

BOARDS_DIR = Path(__file__).resolve().parent / 'boards'
BOARD_SCHEMA_VERSION = 2

# ピン表記のベンダ別フォーマット。ボードは pin_format(無ければ toolchain 由来、
# どちらも無ければ intel)で全ピンが検証される。ここの緩い共用体は
# 「どのベンダにも該当しない文字列」を弾く一次フィルタ(厳密検証はボード単位)。
PIN_FORMATS = {
    'intel': r'^PIN_[A-Z]+\d+$',     # 例: PIN_N16(Quartus)
    'gowin': r'^\d{1,3}$',           # 例: 76(QFN パッケージの素の番号)
    'alnum': r'^[A-Z]{1,3}\d{1,3}$',  # 例: E3 / A11(Xilinx/Lattice BGA)
}
PIN_UNION_RE = r'^(PIN_[A-Z]+\d+|[A-Z]{1,3}\d{1,3}|\d{1,3})$'

# toolchain 名 → 既定 pin_format(pin_format 未指定時の導出)
_TOOLCHAIN_PIN_FORMAT = {
    'quartus': 'intel',
    'gowin-oss': 'gowin',
    # 将来: 'vivado'/'openxc7' → 'alnum', 'lattice-oss'/'radiant' → 'alnum'
}


class HeaderIo(BaseModel):
    """GPIO ヘッダ 1 ピン(論理 index → FPGA ピン+コネクタ物理ピン番号)。"""
    pin: str = Field(pattern=PIN_UNION_RE)
    physical: int | None = None
    model_config = {'extra': 'allow'}


class Header(BaseModel):
    name: str
    connector: str | None = None
    pin_count: int | None = None
    io: dict[str, HeaderIo] = Field(default_factory=dict)
    power: list[dict] = Field(default_factory=list)
    gnd: list[dict] = Field(default_factory=list)
    model_config = {'extra': 'allow'}

    @model_validator(mode='after')
    def _io_keys_are_indices(self):
        for k in self.io:
            if not k.isdigit():
                raise ValueError(
                    f'headers[].io のキーは論理 index の数値文字列: {k!r}')
        return self


class Detection(BaseModel):
    """ボード検出署名(detect 動詞が照合する。値は実機採取のみ)。"""
    cable: str | None = None
    ofl_board: str | None = None   # openFPGALoader -b のボード名(--list-boards から転記)
    jtag_idcodes: list[str] = Field(default_factory=list)
    jtag_part_regex: str | None = None
    usb: list[dict] = Field(default_factory=list)
    silk_hint: str | None = None
    model_config = {'extra': 'allow'}


def pin_format_of(board: dict) -> str:
    """ボードの実効ピンフォーマット名(pin_format → toolchain 由来 → intel)。"""
    fmt = board.get('pin_format')
    if fmt:
        return fmt
    return _TOOLCHAIN_PIN_FORMAT.get(board.get('toolchain') or '', 'intel')


class BoardDef(BaseModel):
    """検証専用モデル。load_board の戻り値は従来どおり生 dict(消費箇所互換)。"""
    schema_version: int
    key: str
    name: str
    tier: int
    device: str
    family: str
    clock: dict
    resources: dict
    limits: dict
    toolchain: str | None = None    # 合成バックエンド名(backends/ のレジストリキー)
    pin_format: str | None = None   # PIN_FORMATS のキー(無ければ toolchain から導出)
    detection: Detection | None = None
    headers: list[Header] = Field(default_factory=list)
    model_config = {'extra': 'allow'}

    @model_validator(mode='after')
    def _resources_unit(self):
        unit = self.resources.get('unit')
        if not unit:
            raise ValueError('resources.unit が必要(例: "alm" / "le")')
        if not isinstance(self.resources.get(unit), int):
            raise ValueError(
                f'resources.{unit} が整数で必要(unit={unit!r} の総数)')
        return self

    @model_validator(mode='after')
    def _pins_match_format(self):
        fmt = pin_format_of(
            {'pin_format': self.pin_format, 'toolchain': self.toolchain})
        if fmt not in PIN_FORMATS:
            raise ValueError(
                f'未知の pin_format: {fmt!r}(利用可能: {", ".join(PIN_FORMATS)})')
        pat = re.compile(PIN_FORMATS[fmt])
        for where, pin in self._iter_pins():
            if not pat.match(pin):
                raise ValueError(
                    f'{where} のピン {pin!r} が pin_format={fmt!r}'
                    f'({PIN_FORMATS[fmt]})に不一致。ボードの pin_format / '
                    'toolchain 指定と転記元を確認する(創作禁止)')
        return self

    def _iter_pins(self):
        """検証対象の全ピン (位置ラベル, ピン文字列) を列挙する。"""
        extra = self.model_extra or {}
        if isinstance(self.clock.get('pin'), str):
            yield 'clock.pin', self.clock['pin']
        for group in ('led', 'switch', 'button'):
            g = extra.get(group)
            if isinstance(g, dict):
                for i, p in enumerate(g.get('pins', [])):
                    yield f'{group}.pins[{i}]', p
        hx = extra.get('hex')
        if isinstance(hx, dict):
            for d in hx.get('digits', []):
                for i, p in enumerate(d.get('pins', [])):
                    yield f'hex.{d.get("port", "?")}.pins[{i}]', p
        for h in self.headers:
            for k, io in h.io.items():
                yield f'headers[{h.name}].io[{k}]', io.pin


def list_boards() -> list[str]:
    return sorted(p.stem for p in BOARDS_DIR.glob('*.json'))


def load_board(key: str) -> dict:
    path = BOARDS_DIR / f'{key}.json'
    if not path.exists():
        raise KeyError(
            f'未知のボード: {key}。利用可能: {", ".join(list_boards())}。'
            f'追加は {BOARDS_DIR} に JSON を置く(スキーマは de0-cv.json 参照)')
    with open(path, encoding='utf-8') as f:
        board = json.load(f)
    assert board['key'] == key, f'boards/{key}.json の key 不一致'
    if board.get('schema_version') != BOARD_SCHEMA_VERSION:
        raise ValueError(
            f'boards/{key}.json は schema_version={BOARD_SCHEMA_VERSION} が必要'
            f'(現在: {board.get("schema_version")!r})。de0-cv.json を参考に移行する')
    BoardDef.model_validate(board)   # 検証のみ(戻りは生 dict のまま)
    return board


def resolve_board(board: str | Path | dict) -> dict:
    """ボード指定(key / board JSON パス / dict)を検証済み dict へ解決する。

    boards/ へ昇格前の draft(board-def fetch の生成物)をパスで直接使う
    ための共通入口(AUTOFLOW 工程5)。検証水準は load_board と同一
    (schema_version+BoardDef。信頼性第一: 経路によらず同じ検査を通す)。
    """
    if isinstance(board, dict):
        if board.get('schema_version') != BOARD_SCHEMA_VERSION:
            raise ValueError(
                f'board dict は schema_version={BOARD_SCHEMA_VERSION} が必要'
                f'(現在: {board.get("schema_version")!r})')
        BoardDef.model_validate(board)
        return board
    p = Path(board)
    if p.suffix == '.json':
        if not p.exists():
            raise FileNotFoundError(
                f'board JSON が見つからない: {p}。key 指定なら拡張子を外す'
                f'(利用可能: {", ".join(list_boards())})')
        with open(p, encoding='utf-8') as f:
            doc = json.load(f)
        if doc.get('schema_version') != BOARD_SCHEMA_VERSION:
            raise ValueError(
                f'{p} は schema_version={BOARD_SCHEMA_VERSION} が必要'
                f'(現在: {doc.get("schema_version")!r})')
        BoardDef.model_validate(doc)
        return doc
    return load_board(str(board))


def get_header(board: dict, name: str) -> dict:
    for h in board.get('headers', []):
        if h.get('name') == name:
            return h
    avail = ', '.join(h.get('name', '?') for h in board.get('headers', []))
    raise KeyError(
        f'ボード {board.get("key")} にヘッダ {name} が未定義。'
        f'定義済み: {avail or "(なし)"}')


def resolve_header_io(board: dict, header: str, index: int) -> dict:
    """ヘッダの論理 index → {'pin','physical',...}。未収録は明示エラー(創作禁止)。"""
    h = get_header(board, header)
    io = h.get('io', {})
    entry = io.get(str(index))
    if entry is None:
        known = ', '.join(sorted(io, key=int))
        raise KeyError(
            f'{board.get("key")} の {header}[{index}] はピン未収録(創作禁止の'
            f'ため解決不可。収録済み index: {known or "(なし)"})。'
            'manifest の io.link で raw pin を指定するか、確認済みピン表を '
            'boards/*.json の headers に追記する')
    return entry
