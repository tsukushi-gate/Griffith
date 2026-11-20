"""M6: detect 動詞(パーサ/署名照合/確信度)の検証。

fixture の出所:
- jtagconfig: 01signal 掲載の実出力(https://www.01signal.com/vendor-specific/
  intel/fpga-command-line-jtag/)
- openFPGALoader: --detect 成功例は README 掲載例(実機採取で更新予定)、
  失敗例と --scan-usb ヘッダは本環境の v1.1.1 実出力(採取)
実機を要する統合テストは requires_hw(MDL_FPGA_HW_TESTS=1)で既定 skip。
"""

from conftest import requires_hw
from mdl_fpga.board_db import list_boards, load_board
from mdl_fpga.detect import (
    DetectResult, JtagDevice, Observation, UsbSig, match_boards,
    normalize_idcode, parse_jtagconfig, parse_openfpgaloader_detect,
    parse_scan_usb)

JTAGCONFIG_SAMPLE = """1) USB-BlasterII [1-5.1.2]
  02B040DD   5CGTFD9(A5|C5|D5|E5)/..
  020A40DD   5M2210Z/EPM2210
"""

# 実機 DE0-CV の生出力(本環境で採取。boards/NOTICE 参照)
JTAGCONFIG_DE0CV_REAL = """1) USB-Blaster [USB-0]
  02B050DD   5CE(BA4|FA4)
"""

JTAGCONFIG_NO_HW = "No JTAG hardware available"

# 本環境 v1.1.1 実出力(ケーブル未接続)
OFL_NO_CABLE = """No cable or board specified: using direct ft2232 interface
unable to open ftdi device: -3 (device not found)
JTAG init failed with: unable to open ftdi device
"""

# openFPGALoader README 掲載例(タブインデント)
OFL_DETECT_SAMPLE = (
    "index 0:\n"
    "\tidcode 0x218a043\n"
    "\tmanufacturer lattice\n"
    "\tfamily MachXO2\n"
    "\tmodel  LCMXO2-7000HC\n"
    "\tirlength 8\n"
)

# 本環境 v1.1.1 実出力(デバイスなし)
SCAN_USB_EMPTY = "Bus device vid:pid       probe_type manufacturer serial product\n"


def test_normalize_idcode():
    assert normalize_idcode('02B040DD') == '0x02b040dd'
    assert normalize_idcode('0x2b040dd') == '0x02b040dd'
    assert normalize_idcode('  0X02B040DD ') == '0x02b040dd'
    assert normalize_idcode('not-hex') is None


def test_parse_jtagconfig_sample():
    cables, devices = parse_jtagconfig(JTAGCONFIG_SAMPLE)
    assert cables == ['USB-BlasterII [1-5.1.2]']
    assert [d.idcode for d in devices] == ['0x02b040dd', '0x020a40dd']
    assert devices[0].part_text.startswith('5CGTFD9')
    assert devices[0].via == 'jtagconfig'


def test_parse_jtagconfig_no_hardware():
    cables, devices = parse_jtagconfig(JTAGCONFIG_NO_HW)
    assert cables == [] and devices == []


def test_parse_openfpgaloader_detect():
    devices = parse_openfpgaloader_detect(OFL_DETECT_SAMPLE)
    assert len(devices) == 1
    assert devices[0].idcode == '0x0218a043'
    assert devices[0].part_text == 'MachXO2 LCMXO2-7000HC'
    assert devices[0].via == 'openfpgaloader'
    assert parse_openfpgaloader_detect(OFL_NO_CABLE) == []


def test_parse_scan_usb():
    assert parse_scan_usb(SCAN_USB_EMPTY) == []
    # 行の列構成に依存しない(vid:pid+残りテキストのみ拾う)防御的パース
    row = '001 002 0x09fb:0x6001 ftdi Altera USB-Blaster'
    sigs = parse_scan_usb(SCAN_USB_EMPTY + row + '\n')
    assert sigs == [UsbSig(vid=0x09FB, pid=0x6001,
                           text='ftdi Altera USB-Blaster')]


def _obs(**kw) -> Observation:
    obs = Observation()
    for k, v in kw.items():
        setattr(obs, k, v)
    return obs


def test_match_de0cv_from_real_capture():
    """実機採取の jtagconfig 出力 → de0-cv が一意 best(IDCODE+部品名+ケーブル)。"""
    cables, devices = parse_jtagconfig(JTAGCONFIG_DE0CV_REAL)
    assert devices == [JtagDevice(idcode='0x02b050dd',
                                  part_text='5CE(BA4|FA4)',
                                  via='jtagconfig')]
    obs = _obs(cables=cables, jtag=devices)
    boards = [load_board(k) for k in list_boards()]
    res = DetectResult(observation=obs,
                       candidates=match_boards(obs, boards))
    best = res.best()
    assert best is not None and best.key == 'de0-cv'
    assert best.confidence == 1.0    # 0.7(IDCODE)+0.5(部品名)+0.1(ケーブル)→cap
    assert any('IDCODE' in r for r in best.reasons)


def test_match_de0cv_via_part_regex_and_cable():
    """IDCODE 未採取でも part 名+ケーブル種で de0-cv が一意候補になる。"""
    obs = _obs(cables=['USB-Blaster [USB-0]'],
               jtag=[JtagDevice(idcode='0x02b050dd',
                                part_text='5CEBA4(.|E4)/5CEME4/..',
                                via='jtagconfig')])
    boards = [load_board(k) for k in list_boards()]
    res = DetectResult(observation=obs,
                       candidates=match_boards(obs, boards))
    best = res.best()
    assert best is not None and best.key == 'de0-cv'
    assert best.confidence >= 0.5


def test_match_idcode_and_ambiguity():
    board_a = {'key': 'a', 'name': 'A', 'tier': 2,
               'detection': {'jtag_idcodes': ['0x01234567']}}
    board_b = {'key': 'b', 'name': 'B', 'tier': 2,
               'detection': {'jtag_idcodes': ['0x01234567']}}
    obs = _obs(jtag=[JtagDevice(idcode='0x01234567', via='jtagconfig')])
    cands = match_boards(obs, [board_a, board_b])
    assert len(cands) == 2
    assert all(c.confidence == 0.7 for c in cands)
    # 同率トップ複数 → best は None(自動選択せずユーザー確認)
    assert DetectResult(observation=obs, candidates=cands).best() is None


def test_match_usb_signature():
    board = {'key': 'x', 'name': 'X', 'tier': 2,
             'detection': {'usb': [{'vid': '0x09fb', 'pid': '0x6001',
                                    'desc_regex': 'USB-Blaster'}]}}
    obs = _obs(usb=[UsbSig(vid=0x09FB, pid=0x6001,
                           text='ftdi Altera USB-Blaster')])
    cands = match_boards(obs, [board])
    assert len(cands) == 1 and cands[0].confidence == 0.2


def test_match_empty_observation():
    obs = Observation()
    boards = [load_board(k) for k in list_boards()]
    assert match_boards(obs, boards) == []
    res = DetectResult(observation=obs, candidates=[])
    assert res.best() is None
    assert '候補なし' in res.render()


def test_to_dict_json_serializable():
    import json
    obs = _obs(jtag=[JtagDevice(idcode='0x02b050dd', part_text='5CEBA4',
                                via='jtagconfig')])
    boards = [load_board(k) for k in list_boards()]
    res = DetectResult(observation=obs, candidates=match_boards(obs, boards))
    json.dumps(res.to_dict(), ensure_ascii=False)


@requires_hw
def test_run_detect_finds_de0cv_on_hardware():
    """実機統合(DE0-CV 接続+電源 ON が前提。MDL_FPGA_HW_TESTS=1 で実行)。"""
    from mdl_fpga.detect import run_detect
    res = run_detect()
    best = res.best()
    assert best is not None and best.key == 'de0-cv', res.render()
