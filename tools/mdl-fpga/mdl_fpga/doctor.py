"""環境診断(mdl-fpga doctor)。

初心者向けメッセージ規約: 各項目は「状態 / 原因 / 次の一手」の3点で報告する。
ベンダツール本体は EULA により同梱できないため、検出と誘導のみを行う。
"""

from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass, field

from .toolchain import Toolchain, discover

OK, WARN, FAIL = 'OK', 'WARN', 'FAIL'


@dataclass
class Check:
    name: str
    status: str
    detail: str
    next_step: str = ''

    def to_dict(self) -> dict:
        return {'name': self.name, 'status': self.status,
                'detail': self.detail, 'next_step': self.next_step}


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *a, **kw) -> None:
        self.checks.append(Check(*a, **kw))

    @property
    def sim_ready(self) -> bool:
        return all(c.status == OK for c in self.checks
                   if c.name in ('iverilog', 'vvp'))

    def _backend_ready(self, names: tuple[str, ...]) -> bool:
        found = {c.name: c.status for c in self.checks}
        return all(found.get(n) == OK for n in names)

    @property
    def quartus_ready(self) -> bool:
        return self._backend_ready(('quartus',))

    @property
    def gowin_oss_ready(self) -> bool:
        return self._backend_ready(('yosys', 'nextpnr-himbaechel',
                                    'gowin_pack'))

    @property
    def synth_ready(self) -> bool:
        """いずれかの合成バックエンドが使えれば True(バックエンド別 OR)。"""
        return self.quartus_ready or self.gowin_oss_ready

    def to_dict(self) -> dict:
        return {'checks': [c.to_dict() for c in self.checks],
                'sim_ready': self.sim_ready, 'synth_ready': self.synth_ready,
                'backends': {'quartus': self.quartus_ready,
                             'gowin-oss': self.gowin_oss_ready}}

    def render(self) -> str:
        icon = {OK: '[OK]  ', WARN: '[WARN]', FAIL: '[FAIL]'}
        lines = []
        for c in self.checks:
            lines.append(f'{icon[c.status]} {c.name}: {c.detail}')
            if c.next_step and c.status != OK:
                lines.append(f'        次の一手: {c.next_step}')
        lines.append('')
        lines.append(f'シミュレーション経路(convert/sim/check): '
                     f'{"利用可能" if self.sim_ready else "利用不可"}')
        lines.append(f'合成/書込経路(build/flash): '
                     f'{"利用可能" if self.synth_ready else "利用不可"}'
                     f'(quartus: {"可" if self.quartus_ready else "不可"} / '
                     f'gowin-oss: {"可" if self.gowin_oss_ready else "不可"})')
        return '\n'.join(lines)


def _try_run(argv: list[str], env: dict | None = None,
             timeout: int = 60) -> tuple[bool, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           errors='replace', timeout=timeout, env=env)
        out = (r.stdout or '') + (r.stderr or '')
        return r.returncode == 0, out.strip().splitlines()[0] if out.strip() else ''
    except Exception as e:  # noqa: BLE001 - 診断は全例外を報告に変換する
        return False, f'{type(e).__name__}: {e}'


def run_doctor(tc: Toolchain | None = None, probe_jtag: bool = True) -> Report:
    tc = tc or discover()
    rep = Report()

    # --- Python 側 ---
    for mod, why in (('numpy', '参照シミュレーション'),
                     ('griffith_logic', 'IR 読込/検証/simulate')):
        if importlib.util.find_spec(mod) is not None:
            rep.add(f'python:{mod}', OK, 'import 可能')
        else:
            rep.add(f'python:{mod}', FAIL, f'{mod} が見つからない({why}に必須)',
                    'リポジトリ直下で `uv sync --extra dev --extra cpu` を実行')
    if importlib.util.find_spec('serial') is not None:
        rep.add('python:pyserial', OK, 'import 可能(実機 UART check 利用可)')
    else:
        rep.add('python:pyserial', WARN, 'pyserial 未導入(実機 UART check のみ不可)',
                '`uv sync --extra dev --extra cpu --extra fpga` で導入')

    # --- oss-cad-suite(対話シム) ---
    if tc.oss_root is None:
        rep.add('iverilog', FAIL, 'oss-cad-suite が見つからない',
                'https://github.com/YosysHQ/oss-cad-suite-build から展開し、'
                '環境変数 MDL_FPGA_OSS_CAD か tools/mdl-fpga/local.toml の '
                'oss_cad_root にルートパスを設定')
        rep.add('vvp', FAIL, '同上', '同上')
    else:
        src = tc.sources.get('oss_cad', '?')
        env = tc.sim_env()
        for name, exe in (('iverilog', tc.iverilog), ('vvp', tc.vvp)):
            if exe is None:
                rep.add(name, FAIL, f'{tc.oss_root} に {name} 実行体が無い',
                        'oss-cad-suite の展開が壊れていないか確認')
                continue
            ok, ver = _try_run([exe, '-V'], env=env)
            # vvp -V は rc!=0 を返す版があるため出力の有無でも判定する
            if ok or 'Icarus' in ver:
                rep.add(name, OK, f'{ver}({src})')
            else:
                rep.add(name, FAIL, f'{name} 実行失敗: {ver}',
                        'PATH に oss-cad-suite の bin と lib の両方が必要'
                        '(本ツールは自動前置する。壊れている場合は再展開)')
        if tc.verilator:
            rep.add('verilator', OK, f'{tc.verilator}(早送りは開発側プリビルド運用)')
        else:
            rep.add('verilator', WARN, 'verilator_bin が見つからない(早送り不可)',
                    '対話シム(iverilog)のみで M0/check は動作する')
        if tc.openfpgaloader:
            rep.add('openFPGALoader', OK,
                    f'{tc.openfpgaloader}(EDA 無しの detect / svf・rbf・fs 書込)')
        else:
            rep.add('openFPGALoader', WARN,
                    'openFPGALoader が見つからない(detect と EDA 無し書込に使用)',
                    'oss-cad-suite を更新する(近年のビルドには同梱されている)')

        # --- gowin-oss 合成(yosys → nextpnr-himbaechel → gowin_pack) ---
        for name, exe in (('yosys', tc.yosys),
                          ('nextpnr-himbaechel', tc.nextpnr_himbaechel),
                          ('gowin_pack', tc.gowin_pack)):
            if exe:
                rep.add(name, OK, f'{exe}(gowin-oss 合成)')
            else:
                rep.add(name, FAIL,
                        f'{name} が見つからない(gowin-oss 合成に必須)',
                        'oss-cad-suite を最新版に更新する(全て同梱されている)')
        share = tc.oss_share()
        if tc.nextpnr_himbaechel and share is not None:
            from .board_db import list_boards, load_board
            missing_db = []
            for key in list_boards():
                try:
                    b = load_board(key)
                except Exception:  # noqa: BLE001 - 診断中は個別ボード破損を無視
                    continue
                if b.get('toolchain') == 'gowin-oss':
                    db = share / 'nextpnr' / 'himbaechel' / 'gowin' / \
                        f'chipdb-{b["family"]}.bin'
                    if not db.exists():
                        missing_db.append(f'{key}({db.name})')
            if missing_db:
                rep.add('gowin-chipdb', WARN,
                        f'chipdb 不足: {", ".join(missing_db)}',
                        'oss-cad-suite を最新版に更新する')

    # --- Quartus(合成/書込) ---
    if tc.quartus_bin is None:
        rep.add('quartus', FAIL, 'Quartus が見つからない(build/flash に必須)',
                'Quartus Prime Lite を導入し、環境変数 MDL_FPGA_QUARTUS_BIN か '
                'local.toml の quartus_bin に bin64 パスを設定。'
                '書込だけなら Programmer 単体版(約4GB)でも可')
    else:
        exes = ['quartus_map', 'quartus_fit', 'quartus_asm', 'quartus_sta',
                'quartus_pgm', 'quartus_cpf']
        missing = [e for e in exes if tc.quartus(e) is None]
        if missing:
            rep.add('quartus', WARN,
                    f'{tc.quartus_bin} に不足: {", ".join(missing)}',
                    'Programmer 単体版の場合 build 系は使えない(flash は可)')
        else:
            rep.add('quartus', OK,
                    f'{tc.quartus_bin}({tc.sources.get("quartus", "?")})')

        if probe_jtag and tc.quartus('quartus_pgm'):
            ok, out = _try_run([tc.quartus('quartus_pgm'), '-l'], timeout=30)
            if ok and 'USB-Blaster' in out:
                rep.add('jtag', OK, f'ケーブル検出: {out}')
            elif ok:
                rep.add('jtag', WARN, f'JTAG ケーブル未検出({out or "一覧が空"})',
                        'ボードの電源と USB ケーブル(UB2 ポート)を確認。'
                        'ドライバは Quartus 同梱の usb_blaster を Device Manager から導入')
            else:
                rep.add('jtag', WARN, f'quartus_pgm -l 失敗: {out}',
                        'ボード未接続でも build までは可能')

    # --- ベンダ EDA 探索枠(任意。見つかれば表示のみ、バックエンドは将来) ---
    for name, path in (('vivado', tc.vivado_bin),
                       ('gowin-eda', tc.gowin_eda_bin),
                       ('radiant', tc.radiant_bin)):
        if path is not None:
            rep.add(f'eda:{name}', OK,
                    f'{path}({tc.sources.get(name.replace("-", "_") + "_bin", "?")}。'
                    'バックエンド実装は将来対応)')

    # --- シリアル(実機 UART check) ---
    if importlib.util.find_spec('serial') is not None:
        try:
            from serial.tools import list_ports

            from .board_db import list_boards as _lb, load_board as _load
            from .hw_check import find_port
            ports = list(list_ports.comports())
            found: list[str] = []
            hit = find_port(None)      # 既定 CP210x
            if hit:
                found.append(f'{hit}(CP210x)')
            for key in _lb():
                try:
                    b = _load(key)
                except Exception:  # noqa: BLE001 - 診断中はボード破損を無視
                    continue
                if (b.get('serial') or {}).get('usb'):
                    hit = find_port(b)
                    if hit and not any(hit in f for f in found):
                        found.append(f'{hit}({key} 内蔵 UART)')
            if found:
                rep.add('serial', OK, f'UART 検出: {", ".join(found)}')
            else:
                have = ', '.join(p.device for p in ports) or 'なし'
                rep.add('serial', WARN,
                        f'既知の UART 署名に一致なし(存在するポート: {have})',
                        '外付け CP210x ブリッジを接続するか、内蔵 USB-UART の'
                        'ボード(Tang Nano 20K 等)は boards/*.json の serial.usb '
                        '署名を実機採取して記録する。実機 UART check 以外は影響なし')
        except Exception as e:  # noqa: BLE001
            rep.add('serial', WARN, f'ポート列挙失敗: {e}')

    return rep
