"""外部ソースのパーサ群(board-def fetch のプロバイダ層、AUTOFLOW 工程3)。

創作禁止ポリシーの機械化: ここは「ソースに書いてある事実の抽出」だけを行い、
ピン値の推定・補完は一切しない(取れないものは呼び出し側で明示エラーにする)。

- litex:    litex-boards platform .py を AST で構文解析(取得コードは実行しない)
- cst:      Gowin 物理制約(IO_LOC / IO_PORT)+ヘッダコメントの Part Number
- qsf:      Quartus 設定(set_location_assignment)
- part-hpp: openFPGALoader src/part.hpp の IDCODE 表
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- litex ---


@dataclass
class LitexIo:
    """litex `_io` の 1 エントリ(("name", index, Pins(...), ...))。"""

    name: str
    index: int
    pins: list[str] = field(default_factory=list)
    width: int | None = None      # Pins(4) 形(位置なし幅指定。SIP 内蔵配線等)
    subsignals: dict[str, list[str]] = field(default_factory=dict)
    iostandard: str | None = None


@dataclass
class LitexPlatform:
    """litex platform ファイルから抽出した事実(実行せず AST のみ)。"""

    io: list[LitexIo] = field(default_factory=list)
    connectors: dict[str, object] = field(default_factory=dict)
    device: str | None = None      # *Platform.__init__ の第1文字列引数
    devicename: str | None = None  # 同 devicename= keyword
    cable: str | None = None       # OpenFPGALoader(cable=...)
    as_gpio: list[str] = field(default_factory=list)   # use_<k>_as_gpio = 1

    def entries(self, name: str) -> list[LitexIo]:
        return sorted((e for e in self.io if e.name == name),
                      key=lambda e: e.index)


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _parse_pins(call: ast.Call) -> tuple[list[str], int | None]:
    pins: list[str] = []
    width: int | None = None
    for a in call.args:
        if isinstance(a, ast.Constant):
            if isinstance(a.value, str):
                pins += a.value.split()
            elif isinstance(a.value, int):
                width = a.value
    return pins, width


def _parse_io_entry(t: ast.Tuple) -> LitexIo | None:
    if len(t.elts) < 2:
        return None
    name = _const_str(t.elts[0])
    idx = t.elts[1]
    if name is None or not (isinstance(idx, ast.Constant)
                            and isinstance(idx.value, int)):
        return None
    entry = LitexIo(name=name, index=idx.value)
    for e in t.elts[2:]:
        if not isinstance(e, ast.Call):
            continue
        cn = _call_name(e)
        if cn == 'Pins':
            entry.pins, entry.width = _parse_pins(e)
        elif cn == 'IOStandard' and e.args:
            entry.iostandard = _const_str(e.args[0])
        elif cn == 'Subsignal' and e.args:
            sub = _const_str(e.args[0])
            if sub is None:
                continue
            for sa in e.args[1:]:
                if isinstance(sa, ast.Call) and _call_name(sa) == 'Pins':
                    pins, _w = _parse_pins(sa)
                    entry.subsignals[sub] = pins
    return entry


def _visit_assign(node: ast.Assign, plat: LitexPlatform) -> None:
    if len(node.targets) != 1:
        return
    tgt = node.targets[0]
    if isinstance(tgt, ast.Name) and tgt.id == '_io' \
            and isinstance(node.value, ast.List):
        for elt in node.value.elts:
            if isinstance(elt, ast.Tuple):
                e = _parse_io_entry(elt)
                if e is not None:
                    plat.io.append(e)
    elif isinstance(tgt, ast.Name) and tgt.id == '_connectors' \
            and isinstance(node.value, ast.List):
        for elt in node.value.elts:
            if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
                continue
            cname = _const_str(elt.elts[0])
            if cname is None:
                continue
            v = elt.elts[1]
            s = _const_str(v)
            if s is not None:
                plat.connectors[cname] = s.split()
            elif isinstance(v, ast.Dict):
                d: dict[str, str] = {}
                for k, vv in zip(v.keys, v.values):
                    if isinstance(k, ast.Constant) and isinstance(vv, ast.Constant):
                        d[str(k.value)] = str(vv.value)
                plat.connectors[cname] = d
    elif isinstance(tgt, ast.Subscript):
        # self.toolchain.options["use_<k>_as_gpio"] = 1
        base, key = tgt.value, tgt.slice
        if (isinstance(base, ast.Attribute) and base.attr == 'options'
                and isinstance(key, ast.Constant) and isinstance(key.value, str)
                and isinstance(node.value, ast.Constant)
                and node.value.value == 1):
            m = re.fullmatch(r'use_(\w+)_as_gpio', key.value)
            if m:
                plat.as_gpio.append(m.group(1))


def _visit_call(node: ast.Call, plat: LitexPlatform) -> None:
    fn = node.func
    if (isinstance(fn, ast.Attribute) and fn.attr == '__init__'
            and isinstance(fn.value, ast.Name)
            and fn.value.id.endswith('Platform') and len(node.args) >= 2):
        dev = _const_str(node.args[1])
        if dev is not None and plat.device is None:
            plat.device = dev
        for kw in node.keywords:
            if kw.arg == 'devicename':
                plat.devicename = _const_str(kw.value)
    elif isinstance(fn, ast.Name) and fn.id == 'OpenFPGALoader':
        for kw in node.keywords:
            if kw.arg == 'cable':
                plat.cable = _const_str(kw.value)


def parse_litex_platform(text: str) -> LitexPlatform:
    """litex-boards platform .py から事実を抽出する(コードは実行しない)。"""
    tree = ast.parse(text)
    plat = LitexPlatform()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            _visit_assign(node, plat)
        elif isinstance(node, ast.Call):
            _visit_call(node, plat)
    return plat


# ------------------------------------------------------------------ cst ---


@dataclass
class CstFile:
    """Gowin .cst(物理制約)から抽出した事実。"""

    locs: dict[str, str] = field(default_factory=dict)   # net → ピン(差動は "33,34")
    ports: dict[str, dict[str, str]] = field(default_factory=dict)
    part_number: str | None = None    # ヘッダコメント //Part Number: 行


_IO_LOC_RE = re.compile(r'^\s*IO_LOC\s+"([^"]+)"\s+([A-Za-z0-9_,]+)\s*;', re.M)
_IO_PORT_RE = re.compile(r'^\s*IO_PORT\s+"([^"]+)"\s+([^;]+);', re.M)
_PART_NUMBER_RE = re.compile(r'^//\s*Part Number:\s*(\S+)', re.M)


def parse_cst(text: str) -> CstFile:
    f = CstFile()
    for net, pin in _IO_LOC_RE.findall(text):
        f.locs[net] = pin
    for net, attrs in _IO_PORT_RE.findall(text):
        d: dict[str, str] = {}
        for kv in attrs.split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                d[k] = v
        f.ports[net] = d
    m = _PART_NUMBER_RE.search(text)
    if m:
        f.part_number = m.group(1)
    return f


# ------------------------------------------------------------------ qsf ---

_QSF_LOC_RE = re.compile(
    r'^\s*set_location_assignment\s+(\S+)\s+-to\s+(\S+)\s*$', re.M)


def parse_qsf(text: str) -> dict[str, str]:
    """Quartus .qsf の位置割当(net → ピン。再割当は後勝ち=Quartus と同じ)。"""
    return {net: pin for pin, net in _QSF_LOC_RE.findall(text)}


# ------------------------------------------------------------- part.hpp ---

_PART_HPP_RE = re.compile(
    r'\{\s*0x([0-9a-fA-F]{8})\s*,\s*\{\s*"([^"]*)"\s*,\s*"([^"]*)"')


def parse_part_hpp(text: str) -> list[dict[str, str]]:
    """openFPGALoader src/part.hpp の IDCODE 表({idcode, vendor, model})。"""
    return [{'idcode': f'0x{code.lower()}', 'vendor': vendor, 'model': model}
            for code, vendor, model in _PART_HPP_RE.findall(text)]
