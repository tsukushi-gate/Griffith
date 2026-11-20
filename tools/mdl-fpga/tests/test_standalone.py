"""スタンドアローンアプリ(STANDALONE.md 実装順序1〜4)の検証。

- io_contract 生成(幅・フレーミング・flash 引数・golden=参照シム由来)
- runtime の独立性(griffith_logic / mdl_fpga を import しない)を機械保証
- runtime protocol が mdl_fpga/hw_check.py と**バイト一致**することを固定
- fake serial での link/selftest E2E(resync 込み)
- CLI(info/send/serve-stdio)と bundle への app/flasher 同梱
"""

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import TOOL_DIR, build_groupsum_model, build_parity4

from mdl_fpga import hw_check as _hw
from mdl_fpga.board_db import load_board
from mdl_fpga.contract import build_io_contract
from mdl_runtime import cli as rt_cli
from mdl_runtime import protocol as rt_proto
from mdl_runtime.contract import load_contract, load_goldens
from mdl_runtime.link import UartLink
from mdl_runtime.selftest import run_selftest

RUNTIME_DIR = TOOL_DIR / 'runtime' / 'mdl_runtime'
EXAMPLES = TOOL_DIR / 'examples'


@pytest.fixture(autouse=True)
def _isolate_flasher_default(monkeypatch):
    """開発機の flasher 既定(env/local.toml)に依存しない(明示指定のみ検証)。"""
    monkeypatch.delenv('MDL_FPGA_FLASHER_PACK', raising=False)
    monkeypatch.setattr('mdl_fpga.bundle._local_flasher_pack', lambda: None)


def _parity4_ir(tmp_path) -> Path:
    p = tmp_path / 'parity4.ir.json'
    p.write_text(json.dumps(build_parity4()), encoding='utf-8')
    return p


def _make_app(tmp_path) -> Path:
    """parity4 + tang-nano-20k の app/ ディレクトリを組み立てる。"""
    ir = _parity4_ir(tmp_path)
    contract, goldens = build_io_contract(
        ir, load_board('tang-nano-20k'), bitstream_arc='bitstream/top.fs')
    app = tmp_path / 'app'
    app.mkdir(exist_ok=True)
    (app / 'io_contract.json').write_text(
        json.dumps(contract, ensure_ascii=False), encoding='utf-8')
    (app / 'goldens.json').write_text(
        json.dumps(goldens, ensure_ascii=False), encoding='utf-8')
    return app


# ------------------------------------------------------------ contract ---


def test_contract_parity4_tang(tmp_path):
    ir = _parity4_ir(tmp_path)
    c, g = build_io_contract(ir, load_board('tang-nano-20k'),
                             bitstream_arc='bitstream/top.fs')
    assert c['contract_version'] == 1 and c['name'] == 'parity4'
    assert c['in_width'] == 4 and c['out_width'] == 1
    assert c['uart']['req_payload_bytes'] == 1
    assert c['uart']['rsp_payload_bytes'] == 1
    assert c['uart']['baud'] == 115200
    assert c['flash'] == {'tool': 'openFPGALoader',
                          'args': ['-b', 'tangnano20k'],
                          'bitstream': 'bitstream/top.fs',
                          'driver_note': c['flash']['driver_note']}
    assert c['serial']['usb'][0]['vid'] == '0x0403'   # 契約だけでポート検出可
    # 4bit は全数16ベクトル。y はパリティ(参照シム由来)
    assert len(g['vectors']) == 16
    for v in g['vectors']:
        x, y = int(v['x'], 16), int(v['y'], 16)
        assert y == bin(x).count('1') % 2


def test_contract_groupsum_class_scores(tmp_path):
    p = tmp_path / 'gs.ir.json'
    p.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    c, _g = build_io_contract(p, load_board('de0-cv'),
                              bitstream_arc='bitstream/top.sof')
    out = c['outputs'][0]
    assert out['semantics'] == 'class_scores'
    assert out['k'] == 2 and out['score_bits'] == 2
    assert c['out_width'] == 4
    assert c['flash']['args'] == ['-c', 'usb-blaster']


def test_contract_manifest_baud(tmp_path):
    ir = _parity4_ir(tmp_path)
    mf = tmp_path / 'p4.manifest.json'
    mf.write_text(json.dumps({
        'manifest_version': '0.2', 'ir': ir.name, 'board': 'tang-nano-20k',
        'io': {'protocol': 'uart-57600-8n1',
               'link': {'type': 'uart', 'baud': 57600}},
    }), encoding='utf-8')
    c, _ = build_io_contract(mf, load_board('tang-nano-20k'),
                             bitstream_arc='bitstream/top.fs')
    assert c['uart']['baud'] == 57600


# ------------------------------------------------- runtime 独立性/protocol ---


def test_runtime_never_imports_mdl():
    # import 文のみ検査(docstring での仕様参照は許容)
    bad = re.compile(r'^\s*(import|from)\s+(griffith_logic|mdl_fpga)\b',
                     re.M)
    for p in RUNTIME_DIR.rglob('*.py'):
        text = p.read_text(encoding='utf-8')
        m = bad.search(text)
        assert m is None, f'{p} が mdl 側を import している(独立性違反): ' \
                          f'{m.group(0) if m else ""}'


def test_protocol_bytes_identical_to_hw_check():
    for seq, val, nb in ((0, 0, 1), (5, 0x1234, 3), (255, 0xF, 1),
                         (7, (1 << 90) - 3, 12)):
        assert rt_proto.build_request(seq, val, nb) == \
            _hw.build_request(seq, val, nb)
    frame = _hw.build_request(9, 0xAB, 2).replace(
        bytes([_hw.REQ_SOF]), bytes([_hw.RSP_SOF]), 1)
    garbage = b'\x00\xff\x13' + frame
    assert rt_proto.parse_response(garbage, 2) == \
        _hw.parse_response(garbage, 2) == (9, 0xAB)
    # checksum 破壊は両者とも拒否
    bad = frame[:-1] + bytes([frame[-1] ^ 1])
    assert rt_proto.parse_response(bad, 2) is None
    assert _hw.parse_response(bad, 2) is None


def test_decode_scores_matches_pack_scores():
    import numpy as np

    from mdl_fpga.vectors import pack_scores
    scores = [[3, 0, 2, 1]]
    packed = pack_scores(np.array(scores), 2)[0]
    assert rt_proto.decode_scores(packed, 4, 2) == scores[0]


# ------------------------------------------------ fake serial E2E ---


class FakeSerial:
    """golden 写像どおりに応答するループバック(先頭にゴミ=resync 検証)。"""

    def __init__(self, mapping: dict[int, int], req_p: int, rsp_q: int,
                 garbage: bytes = b''):
        self.mapping, self.req_p, self.rsp_q = mapping, req_p, rsp_q
        self.garbage = garbage
        self.rx = b''

    def write(self, data: bytes) -> None:
        assert data[0] == rt_proto.REQ_SOF
        seq = data[1]
        x = int.from_bytes(data[2:2 + self.req_p], 'little')
        y = self.mapping[x]
        body = bytes([seq]) + y.to_bytes(self.rsp_q, 'little')
        chk = sum(body) & 0xFF
        self.rx += self.garbage + bytes([rt_proto.RSP_SOF]) + body + \
            bytes([chk])

    def read(self, n: int) -> bytes:
        out, self.rx = self.rx[:n], self.rx[n:]
        return out

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.rx = b''

    def close(self) -> None:
        pass


def _fake_link(contract, goldens, garbage=b'') -> UartLink:
    mapping = {v['x']: v['y'] for v in goldens}
    fake = FakeSerial(mapping, contract['uart']['req_payload_bytes'],
                      contract['uart']['rsp_payload_bytes'], garbage)
    return UartLink('FAKE', contract['uart']['baud'],
                    in_width=contract['in_width'],
                    out_width=contract['out_width'],
                    serial_factory=lambda port, baud: fake)


def test_selftest_pass_with_fake_serial(tmp_path):
    app = _make_app(tmp_path)
    contract = load_contract(app)
    goldens = load_goldens(app, contract)
    res = run_selftest(_fake_link(contract, goldens), goldens)
    assert res == {'passed': True, 'n_ok': 16, 'n': 16, 'first_fail': None}


def test_selftest_resync_through_garbage(tmp_path):
    app = _make_app(tmp_path)
    contract = load_contract(app)
    goldens = load_goldens(app, contract)
    res = run_selftest(_fake_link(contract, goldens, garbage=b'\xff\x00'),
                       goldens)
    assert res['passed'] is True


def test_selftest_detects_wrong_hw(tmp_path):
    app = _make_app(tmp_path)
    contract = load_contract(app)
    goldens = load_goldens(app, contract)
    broken = [{'x': v['x'], 'y': v['y'] ^ 1} for v in goldens]   # 壊れた実機
    res = run_selftest(_fake_link(contract, broken), goldens)
    assert res['passed'] is False and res['n_ok'] == 0


# ------------------------------------------------------------- CLI ---


def test_cli_info_json(tmp_path, capsys):
    app = _make_app(tmp_path)
    rc = rt_cli.main(['--app', str(tmp_path), 'info', '--json'])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc['name'] == 'parity4'


def test_cli_send_and_serve_stdio(tmp_path, capsys, monkeypatch):
    app = _make_app(tmp_path)
    contract = load_contract(app)
    goldens = load_goldens(app, contract)
    monkeypatch.setattr(rt_cli, '_open_link',
                        lambda c, port: _fake_link(c, goldens))
    rc = rt_cli.main(['--app', str(app), 'send', '0x7', '--json'])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc == {'x': '0x7', 'y': '0x1'}   # parity(0b111)=1

    monkeypatch.setattr(sys, 'stdin', io.StringIO(
        '{"cmd":"send","x":"0x3"}\n{"cmd":"selftest"}\n{"cmd":"quit"}\n'))
    rc = rt_cli.main(['--app', str(app), 'serve-stdio'])
    lines = [json.loads(ln) for ln in
             capsys.readouterr().out.strip().splitlines()]
    assert rc == 0
    assert lines[0] == {'ok': True, 'x': '0x3', 'y': '0x0'}
    assert lines[1]['passed'] is True and lines[1]['n'] == 16
    assert lines[2] == {'ok': True, 'bye': True}


def test_cli_missing_contract(tmp_path, capsys):
    rc = rt_cli.main(['--app', str(tmp_path), 'info'])
    assert rc == 2
    assert 'io_contract.json' in capsys.readouterr().out


# ------------------------------------------------------------ bundle ---


@pytest.mark.skipif(not (EXAMPLES / 'full_adder.ir.json').exists(),
                    reason='examples/full_adder.ir.json が無い')
def test_bundle_includes_app_and_flasher(tmp_path):
    from mdl_fpga.bundle import run_bundle
    wd = tmp_path / 'wd'
    (wd / 'gowin').mkdir(parents=True)
    (wd / 'gowin' / 'top.fs').write_bytes(b'\x00fake-bitstream')
    pack = tmp_path / 'pack'
    (pack / 'windows').mkdir(parents=True)
    (pack / 'windows' / 'openFPGALoader.exe').write_bytes(b'MZfake')
    (pack / 'LICENSE').write_text('Apache-2.0', encoding='utf-8')

    res = run_bundle(wd, manifest=EXAMPLES / 'full_adder.tn20k.manifest.json',
                     board='tang-nano-20k', flasher_pack=pack)
    assert res.ok, res.messages
    with zipfile.ZipFile(res.zip_path) as zf:
        names = set(zf.namelist())
        assert {'bitstream/top.fs', 'board/tang-nano-20k.json',
                'app/io_contract.json', 'app/goldens.json',
                'app/requirements.txt', 'app/run.py',
                'app/mdl_runtime/cli.py',
                'app/mdl_runtime/templates/canvas_app.py',
                'flasher/windows/openFPGALoader.exe',
                'flasher/LICENSE', 'README.md'} <= names
        contract = json.loads(zf.read('app/io_contract.json'))
        assert contract['flash']['args'] == ['-b', 'tangnano20k']
        assert contract['flash']['bitstream'] == 'bitstream/top.fs'
        readme = zf.read('README.md').decode('utf-8')
        assert 'スタンドアローンアプリ' in readme
        assert 'pip install -r app/requirements.txt' in readme
        assert 'python app/run.py flash' in readme
        # 書き込み節: flasher 同梱時は同梱版を主線にする(oss-cad-suite
        # 導入手順ではなく)。macOS 分は未同梱なので brew 誘導のまま
        assert '同梱の flasher を使う' in readme
        assert 'python app\\run.py flash' in readme
        win_sec = readme.split('## 書き込み(Windows)')[1] \
                        .split('## 書き込み(macOS)')[0]
        assert 'oss-cad-suite' not in win_sec
        assert 'brew install openfpgaloader' in readme
        req = zf.read('app/requirements.txt').decode('utf-8')
        assert 'pyserial' in req and 'pygame' in req


@pytest.mark.skipif(not (EXAMPLES / 'full_adder.ir.json').exists(),
                    reason='examples/full_adder.ir.json が無い')
def test_bundle_flasher_default_from_local_toml(tmp_path, monkeypatch):
    """既定同梱: 引数なしでも local.toml の flasher_pack から同梱される。"""
    from mdl_fpga.bundle import run_bundle
    pack = tmp_path / 'pack' / 'windows'
    pack.mkdir(parents=True)
    (pack / 'openFPGALoader.exe').write_bytes(b'MZfake')
    monkeypatch.setattr('mdl_fpga.bundle._local_flasher_pack',
                        lambda: str(tmp_path / 'pack'))
    wd = tmp_path / 'wd'
    (wd / 'gowin').mkdir(parents=True)
    (wd / 'gowin' / 'top.fs').write_bytes(b'\x00fake')
    res = run_bundle(wd, manifest=EXAMPLES / 'full_adder.tn20k.manifest.json',
                     board='tang-nano-20k')
    assert res.ok
    assert 'flasher/windows/openFPGALoader.exe' in res.contents
    # オプトアウト(--no-flasher 相当)
    res2 = run_bundle(wd, manifest=EXAMPLES / 'full_adder.tn20k.manifest.json',
                      board='tang-nano-20k', flasher=False,
                      out=tmp_path / 'no_flasher.zip')
    assert res2.ok
    assert not any(c.startswith('flasher/') for c in res2.contents)


@pytest.mark.skipif(not (EXAMPLES / 'full_adder.ir.json').exists(),
                    reason='examples/full_adder.ir.json が無い')
def test_bundle_no_app_optout(tmp_path):
    from mdl_fpga.bundle import run_bundle
    wd = tmp_path / 'wd'
    (wd / 'gowin').mkdir(parents=True)
    (wd / 'gowin' / 'top.fs').write_bytes(b'\x00fake')
    res = run_bundle(wd, manifest=EXAMPLES / 'full_adder.tn20k.manifest.json',
                     board='tang-nano-20k', app=False)
    assert res.ok
    with zipfile.ZipFile(res.zip_path) as zf:
        assert not any(n.startswith('app/') for n in zf.namelist())


# ------------------------------------------------------------ flash 解決 ---


def test_flash_resolution_and_dry_run(tmp_path, monkeypatch):
    from mdl_runtime import flash as rt_flash
    app = _make_app(tmp_path)
    contract = load_contract(app)
    # 同梱 flasher を最優先で解決する
    osdir = rt_flash._os_dir()
    exe_name = 'openFPGALoader.exe' if sys.platform == 'win32' \
        else 'openFPGALoader'
    fdir = tmp_path / 'flasher' / osdir
    fdir.mkdir(parents=True)
    (fdir / exe_name).write_bytes(b'fake')
    exe, source = rt_flash.resolve_flasher(app)
    assert exe == str(fdir / exe_name) and '同梱' in source
    # ビットストリーム配置+dry-run で argv を確認
    (tmp_path / 'bitstream').mkdir()
    (tmp_path / 'bitstream' / 'top.fs').write_bytes(b'\x00')
    res = rt_flash.run_flash(contract, app, dry_run=True)
    assert res['ok'] and res['argv'][1:3] == ['-b', 'tangnano20k']
    # 見つからない場合は誘導文
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    exe2, hint = rt_flash.resolve_flasher(tmp_path / 'nowhere' / 'app')
    assert exe2 is None and 'brew install openfpgaloader' in hint
    # --persist は SPI フラッシュ永続書込の -f をビットストリーム直前に挿入
    res_p = rt_flash.run_flash(contract, app, dry_run=True, persist=True)
    assert res_p['ok'] and res_p['argv'][1:4] == ['-b', 'tangnano20k', '-f']


# ------------------------------------------------------ flasher pack ---


def test_flasher_pack_build_with_licenses(tmp_path):
    from mdl_fpga.flasher_pack import (
        LICENSE_FILES, WINDOWS_DLLS, WINDOWS_EXES, build_pack)
    from mdl_fpga.toolchain import Toolchain
    oss = tmp_path / 'oss-cad-suite'
    for n in WINDOWS_EXES:
        (oss / 'bin').mkdir(parents=True, exist_ok=True)
        (oss / 'bin' / n).write_bytes(b'MZfake')
    for n in WINDOWS_DLLS:
        (oss / 'lib').mkdir(parents=True, exist_ok=True)
        (oss / 'lib' / n).write_bytes(b'DLL')
    (oss / 'license').mkdir()
    for lf in LICENSE_FILES:
        (oss / 'license' / lf).write_text('license text', encoding='utf-8')

    class FakeProc:
        returncode = 0
        stdout = 'openFPGALoader v9.9-test\n'

    out = tmp_path / 'pack'
    res = build_pack(out, tc=Toolchain(oss_root=oss),
                     runner=lambda argv, **kw: FakeProc())
    assert res.version == 'openFPGALoader v9.9-test'
    assert res.license_missing == []
    assert (out / 'windows' / 'openFPGALoader.exe').exists()
    assert (out / 'LICENSES' / 'LICENSE.openfpgaloader').exists()
    notice = (out / 'NOTICE.md').read_text(encoding='utf-8')
    assert 'Apache-2.0' in notice and 'LGPL' in notice
    assert 'oss-cad-suite-build' in notice          # LGPL ソース入手先
    assert 'v9.9-test' in notice
