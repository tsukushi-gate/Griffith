"""board-def fetch: レシピ駆動のボード定義オンデマンド生成(AUTOFLOW 工程3)。

レシピ(recipes/*.json)は「どのソースのどの信号が schema のどの役割か」という
対応付けの意図だけを持ち、**ピン値そのものは一切書かない**(創作禁止)。
ピンはソース(litex-boards / ベンダ .cst / .qsf)から機械抽出し、役割の各位置が
**独立2ソース以上で一致した場合のみ** 2ソース検証済みとする。不一致は生成拒否。

生成物は draft(unverified、Tier2=全位置2ソース / Tier3=単一ソース位置あり)。
実機受入(up --hw-check)を経て人手で boards/ へ昇格・Tier 判定する。
既存ファイルは上書きしない(--out 先に既存があれば拒否)。

DE10-Lite / Tang Nano 20K で scratch 実行していた機械照合転記
(boards/NOTICE、references/tang-nano-20k/gen_board_json.py)の汎用化。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..board_db import BoardDef
from ..toolchain import TOOL_DIR
from . import providers

RECIPES_DIR = Path(__file__).resolve().parent / 'recipes'
REPO_ROOT = TOOL_DIR.parent.parent            # リポジトリ直下
CACHE_DIR = TOOL_DIR / 'runs' / 'boarddef-cache'

# nextpnr-himbaechel(gowin)に実在する vopt(--vopt help 実出力ベース。
# litex の use_*_as_gpio のうちここに無いものは gowin_pack 側フラグのみに落とす)
NEXTPNR_KNOWN_VOPTS = ('sspi_as_gpio',)

_ROLE_KEYS = ('led', 'switch', 'button')      # ピン列の役割(clock は単ピン)


class CrossCheckError(ValueError):
    """ソース間不一致・参照解決失敗(創作禁止のため生成を拒否する)。"""


# ------------------------------------------------------------ 取得層 ---


@dataclass
class SourceData:
    provider: str
    origin: str                                   # NOTICE 用の取得元説明
    platform: providers.LitexPlatform | None = None
    files: dict[str, object] = field(default_factory=dict)
    parts: list[dict] = field(default_factory=list)


def _http_get(url: str, cache_dir: Path, timeout: int = 30) -> str:
    """URL 取得(キャッシュあり)。fetcher 注入時は使われない。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(url.encode()).hexdigest()[:16]
    cached = cache_dir / f'{tag}-{Path(url).name}'
    if cached.exists():
        return cached.read_text(encoding='utf-8', errors='replace')
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        text = r.read().decode('utf-8', errors='replace')
    cached.write_text(text, encoding='utf-8')
    return text


def _source_text(spec: dict, fname: str | None, *, offline: bool,
                 fetcher, cache_dir: Path) -> tuple[str, str]:
    """(テキスト, 取得元説明)。local(リポジトリ相対の既得コピー)優先。"""
    if fname is None:
        local = spec.get('local')
        url = spec.get('url')
    else:
        local_dir = spec.get('local_dir')
        local = f'{local_dir}/{fname}' if local_dir else None
        url = (spec.get('urls') or {}).get(fname)
    if local:
        p = REPO_ROOT / local
        if p.exists():
            return p.read_text(encoding='utf-8', errors='replace'), local
    if url:
        if offline:
            raise CrossCheckError(
                f'--offline 指定だがローカルコピーが無い: {local or fname}'
                f'(URL: {url})')
        text = fetcher(url) if fetcher else _http_get(url, cache_dir)
        return text, url
    raise CrossCheckError(
        f'ソースの取得先が未定義: local={local!r} / url 無し'
        f'(file={fname!r})。レシピの sources を確認')


def acquire_sources(recipe: dict, *, offline: bool = False, fetcher=None,
                    cache_dir: Path | None = None) -> dict[str, SourceData]:
    cache_dir = cache_dir or CACHE_DIR
    out: dict[str, SourceData] = {}
    for key, spec in recipe['sources'].items():
        provider = spec['provider']
        if provider == 'litex':
            text, origin = _source_text(spec, None, offline=offline,
                                        fetcher=fetcher, cache_dir=cache_dir)
            out[key] = SourceData(provider, origin,
                                  platform=providers.parse_litex_platform(text))
        elif provider in ('cst', 'qsf'):
            sd = SourceData(provider, spec.get('home') or spec.get('local_dir', ''))
            for fname in spec.get('files', []):
                text, _o = _source_text(spec, fname, offline=offline,
                                        fetcher=fetcher, cache_dir=cache_dir)
                sd.files[fname] = (providers.parse_cst(text) if provider == 'cst'
                                   else providers.parse_qsf(text))
            out[key] = sd
        elif provider == 'part-hpp':
            text, origin = _source_text(spec, None, offline=offline,
                                        fetcher=fetcher, cache_dir=cache_dir)
            out[key] = SourceData(provider, origin,
                                  parts=providers.parse_part_hpp(text))
        else:
            raise CrossCheckError(f'未知の provider: {provider!r}(source {key})')
    return out


# ------------------------------------------------------------ 突合層 ---


def _ref_desc(ref: dict) -> str:
    if 'signal' in ref:
        sub = f'.{ref["sub"]}' if ref.get('sub') else ''
        return f'{ref["source"]}:{ref["signal"]}{sub}'
    if 'net' in ref:
        at = f'@{ref["at"]}' if 'at' in ref else ''
        return f'{ref["source"]}:{ref["file"]}:{ref["net"]}{at}'
    if 'field' in ref:
        f = f'{ref["file"]}:' if ref.get('file') else ''
        return f'{ref["source"]}:{f}{ref["field"]}'
    return f'{ref["source"]}:?'


def _pins_from_ref(ref: dict, sources: dict[str, SourceData]) -> list[str]:
    src = sources[ref['source']]
    try:
        if src.provider == 'litex':
            entries = src.platform.entries(ref['signal'])
            if not entries:
                raise KeyError(ref['signal'])
            if ref.get('sub'):
                pins = [p for e in entries for p in e.subsignals[ref['sub']]]
            else:
                pins = [p for e in entries for p in e.pins]
            if not pins:
                raise CrossCheckError(
                    f'{_ref_desc(ref)}: ピン位置なし(Pins(幅) 形式のみ)')
            return pins
        if src.provider in ('cst', 'qsf'):
            f = src.files[ref['file']]
            locs = f.locs if src.provider == 'cst' else f
            net = ref['net']
            if '{i}' in net:
                return [locs[net.format(i=i)] for i in range(ref['count'])]
            return [locs[net]]
    except KeyError as e:
        raise CrossCheckError(
            f'{_ref_desc(ref)}: ソースに存在しない({e})') from e
    raise CrossCheckError(f'{_ref_desc(ref)}: ピン参照に使えない provider')


def _scalar_from_ref(ref: dict, sources: dict[str, SourceData]) -> str:
    src = sources[ref['source']]
    try:
        if src.provider == 'litex':
            val = getattr(src.platform, ref['field'])
        elif src.provider == 'cst':
            val = getattr(src.files[ref['file']], ref['field'])
        else:
            raise CrossCheckError(
                f'{_ref_desc(ref)}: スカラ参照に使えない provider')
    except (AttributeError, KeyError) as e:
        raise CrossCheckError(f'{_ref_desc(ref)}: 取得失敗({e})') from e
    if val is None:
        raise CrossCheckError(f'{_ref_desc(ref)}: 値が無い')
    return str(val)


@dataclass
class RoleResult:
    pins: list[str]
    witnesses: list[str]                 # NOTICE 用の参照記述
    position_sources: list[set]          # 位置 i を照合したソース集合
    @property
    def verified(self) -> bool:
        return all(len(s) >= 2 for s in self.position_sources)

    def single_positions(self) -> list[int]:
        return [i for i, s in enumerate(self.position_sources) if len(s) < 2]


def crosscheck_role(label: str, refs: list[dict],
                    sources: dict[str, SourceData]) -> RoleResult:
    """役割 1 つ分の突合。全体 ref は完全一致、at ref は該当位置一致を要求。"""
    full = [(r, _pins_from_ref(r, sources)) for r in refs if 'at' not in r]
    partial = [(r, _pins_from_ref(r, sources)) for r in refs if 'at' in r]
    if not full:
        raise CrossCheckError(
            f'{label}: 全体を与える ref(at 無し)が最低1つ必要')
    base_ref, base = full[0]
    for r, pins in full[1:]:
        if pins != base:
            raise CrossCheckError(
                f'{label}: ソース間不一致 {_ref_desc(base_ref)}={base} vs '
                f'{_ref_desc(r)}={pins}')
    pos_sources: list[set] = [{base_ref['source']} for _ in base]
    for r, _pins in full[1:]:
        for s in pos_sources:
            s.add(r['source'])
    for r, pins in partial:
        if len(pins) != 1:
            raise CrossCheckError(
                f'{label}: {_ref_desc(r)} は単一ピンでない({pins})')
        at = r['at']
        if not (0 <= at < len(base)):
            raise CrossCheckError(
                f'{label}: {_ref_desc(r)} の at={at} が範囲外(幅 {len(base)})')
        if base[at] != pins[0]:
            raise CrossCheckError(
                f'{label}[{at}]: 不一致 {base[at]}({_ref_desc(base_ref)})vs '
                f'{pins[0]}({_ref_desc(r)})')
        pos_sources[at].add(r['source'])
    witnesses = [f'{_ref_desc(r)}={"/".join(p)}' for r, p in full + partial]
    return RoleResult(pins=base, witnesses=witnesses,
                      position_sources=pos_sources)


def crosscheck_scalar(label: str, refs: list[dict],
                      sources: dict[str, SourceData]) -> tuple[str, bool, list[str]]:
    """(値, 2ソース検証済みか, witnesses)。全 ref の値一致を要求。"""
    vals = [(r, _scalar_from_ref(r, sources)) for r in refs]
    base_ref, base = vals[0]
    for r, v in vals[1:]:
        if v != base:
            raise CrossCheckError(
                f'{label}: 不一致 {base!r}({_ref_desc(base_ref)})vs '
                f'{v!r}({_ref_desc(r)})')
    n_sources = len({r['source'] for r, _ in vals})
    return base, n_sources >= 2, [f'{_ref_desc(r)}={v}' for r, v in vals]


def _verify_io_standard(recipe: dict, sources: dict[str, SourceData]) -> int:
    """使用役割の全 ref で IOStandard / IO_TYPE がレシピ値と矛盾しないか。

    情報を持たない ref はカウントしない。矛盾は生成拒否。戻り値=確認できた ref 数。
    """
    spec = recipe.get('io_standard')
    if not spec:
        return 0
    expected = spec['value']
    checked = 0

    def _check_ref(ref: dict) -> None:
        nonlocal checked
        src = sources[ref['source']]
        if src.provider == 'litex' and 'signal' in ref:
            for e in src.platform.entries(ref['signal']):
                if e.iostandard is not None:
                    if e.iostandard != expected:
                        raise CrossCheckError(
                            f'io_standard 不一致: {_ref_desc(ref)}='
                            f'{e.iostandard} vs レシピ={expected}')
                    checked += 1
        elif src.provider == 'cst' and 'net' in ref:
            f = src.files[ref['file']]
            nets = ([ref['net'].format(i=i) for i in range(ref['count'])]
                    if '{i}' in ref['net'] else [ref['net']])
            for net in nets:
                io_type = (f.ports.get(net) or {}).get('IO_TYPE')
                if io_type is not None:
                    if io_type != expected:
                        raise CrossCheckError(
                            f'io_standard 不一致: {_ref_desc(ref)} '
                            f'IO_TYPE={io_type} vs レシピ={expected}')
                    checked += 1

    for role in _ROLE_KEYS + ('clock',):
        for ref in (recipe.get(role) or {}).get('refs', []):
            _check_ref(ref)
    for h in recipe.get('headers', []):
        for refs in h.get('io', {}).values():
            for ref in refs:
                _check_ref(ref)
    return checked


# ---------------------------------------------------------- 組み立て層 ---


@dataclass
class FetchReport:
    verified_roles: list[str] = field(default_factory=list)
    single_source: list[str] = field(default_factory=list)   # 'role[pos]' 表記
    witness_lines: list[str] = field(default_factory=list)
    io_standard_checked: int = 0


def _note_keys(spec: dict) -> dict:
    """役割 spec の注記系キー(*_note / note)をそのまま通す。"""
    return {k: v for k, v in spec.items() if k == 'note' or k.endswith('_note')}


def build_board(recipe: dict, sources: dict[str, SourceData],
                today: str | None = None) -> tuple[dict, FetchReport]:
    today = today or _dt.date.today().isoformat()
    rep = FetchReport()

    def _record(label: str, rr: RoleResult) -> None:
        rep.witness_lines.append(f'{label}: {", ".join(rr.witnesses)}')
        if rr.verified:
            rep.verified_roles.append(label)
        rep.single_source += [f'{label}[{i}]' for i in rr.single_positions()]

    # device(スカラ突合)
    device, dev_ok, dev_wit = crosscheck_scalar(
        'device', recipe['device_refs'], sources)
    rep.witness_lines.append(f'device: {", ".join(dev_wit)}')
    if dev_ok:
        rep.verified_roles.append('device')
    else:
        rep.single_source.append('device')

    # clock(単ピン)
    clk_spec = recipe['clock']
    clk = crosscheck_role('clock', clk_spec['refs'], sources)
    if len(clk.pins) != 1:
        raise CrossCheckError(f'clock は単ピンのはず: {clk.pins}')
    _record('clock', clk)

    board: dict = {
        'schema_version': 2,
        'key': recipe['key'],
        'name': recipe['name'],
        'tier': 2,
        'unverified': True,
        'tier_note': '',                      # 突合結果確定後に埋める
    }
    if recipe.get('toolchain'):
        board['toolchain'] = recipe['toolchain']
    if recipe.get('pin_format'):
        board['pin_format'] = recipe['pin_format']
    board['device'] = device
    board['family'] = recipe['family']
    if recipe.get('family_note'):
        board['family_note'] = recipe['family_note']

    # detection(静的メタ+cable_ref+part-hpp IDCODE。実機採取値は持たない)
    det_spec = recipe.get('detection') or {}
    det: dict = {}
    if 'cable_ref' in det_spec:
        cable = _scalar_from_ref(det_spec['cable_ref'], sources)
        det['cable'] = cable
        det['cable_note'] = (f'{_ref_desc(det_spec["cable_ref"])} から転記'
                             '(単一ソース。実機で要確認)')
        rep.witness_lines.append(f'detection.cable: '
                                 f'{_ref_desc(det_spec["cable_ref"])}={cable}')
    if 'idcode_refs' in det_spec:
        codes: list[str] = []
        for ref in det_spec['idcode_refs']:
            src = sources[ref['source']]
            if src.provider != 'part-hpp':
                raise CrossCheckError(
                    f'idcode_refs は part-hpp ソースのみ: {_ref_desc(ref)}')
            pat = re.compile(ref['model_regex'])
            codes += [p['idcode'] for p in src.parts
                      if pat.search(p['model']) or pat.search(p['vendor'])]
        if codes:
            det['jtag_idcodes'] = sorted(set(codes))
            det['jtag_idcodes_note'] = ('openFPGALoader part.hpp からの転記。'
                                        '実機採取(detect)で確認後に更新する')
    for k, v in det_spec.items():
        if k in ('cable_ref', 'idcode_refs'):
            continue
        det[k] = v
    if det:
        board['detection'] = det

    board['clock'] = {'port': clk_spec['port'], 'pin': clk.pins[0],
                      'hz': clk_spec['hz']}

    # headers(論理 index → ピン)
    headers_out = []
    for h in recipe.get('headers', []):
        io_out: dict[str, dict] = {}
        for index, refs in sorted(h['io'].items(), key=lambda kv: int(kv[0])):
            rr = crosscheck_role(f'headers[{h["name"]}].io[{index}]',
                                 refs, sources)
            if len(rr.pins) != 1:
                raise CrossCheckError(
                    f'headers[{h["name"]}].io[{index}] は単ピンのはず: {rr.pins}')
            _record(f'headers[{h["name"]}].io[{index}]', rr)
            io_out[index] = {'pin': rr.pins[0]}
        entry = {'name': h['name'],
                 'connector': h.get('connector'),
                 'io': io_out, 'power': [], 'gnd': []}
        if h.get('builtin') is not None:
            entry['builtin'] = h['builtin']
        entry.update(_note_keys(h))
        headers_out.append(entry)
    if headers_out:
        board['headers'] = headers_out

    # led / switch / button(ピン列)
    for role in _ROLE_KEYS:
        spec = recipe.get(role)
        if not spec:
            continue
        rr = crosscheck_role(role, spec['refs'], sources)
        _record(role, rr)
        entry = {'port': spec['port']}
        if 'active_low' in spec:
            entry['active_low'] = spec['active_low']
        entry.update(_note_keys(spec))
        entry['pins'] = rr.pins
        board[role] = entry

    # io_standard(整合確認つき)
    rep.io_standard_checked = _verify_io_standard(recipe, sources)
    if recipe.get('io_standard'):
        board['io_standard'] = recipe['io_standard']['value']
        note = recipe['io_standard'].get('note', '')
        board['io_standard_note'] = (
            f'{note}(fetch が使用役割の {rep.io_standard_checked} 参照で'
            '整合を機械確認)' if note else
            f'fetch が使用役割の {rep.io_standard_checked} 参照で整合を機械確認')

    board['resources'] = recipe['resources']
    board['limits'] = {
        'calibrated': False,
        'note': '未較正(board-def fetch 生成 draft)。advisor は保守的既定で'
                '判定する。実機較正後に実測値を記録する',
    }

    # synth_options(litex の use_*_as_gpio からの機械転記)
    if recipe.get('synth_options_from_litex'):
        lx = next((s for s in sources.values() if s.provider == 'litex'), None)
        if lx is None:
            raise CrossCheckError('synth_options_from_litex には litex ソースが必要')
        vopts = [f'{k}_as_gpio' for k in lx.platform.as_gpio
                 if f'{k}_as_gpio' in NEXTPNR_KNOWN_VOPTS]
        board['synth_options'] = {
            'nextpnr_vopts': vopts,
            'gowin_pack_flags': [f'--{k}_as_gpio' for k in lx.platform.as_gpio],
            'note': 'litex toolchain.options(use_*_as_gpio)からの機械転記。'
                    'nextpnr-himbaechel 側の vopt は既知のもののみ',
        }

    # tier / tier_note の確定
    if rep.single_source:
        board['tier'] = 3
    src_lines = []
    for key, spec in recipe['sources'].items():
        bits = [spec.get('home') or spec.get('local') or key]
        if spec.get('commit'):
            bits.append(spec['commit'][:7])
        if spec.get('license'):
            bits.append(spec['license'])
        src_lines.append(f'{key}({", ".join(bits)})')
    single_txt = (f'。単一ソース位置(要追加確認): '
                  f'{", ".join(rep.single_source)}' if rep.single_source else '')
    board['tier_note'] = (
        f'board-def fetch による機械生成 draft({today}、実機未検証)。'
        f'ピンは {" / ".join(src_lines)} の機械突合による転記(手打ちなし)'
        f'{single_txt}。実機受入(up --hw-check)後に人手で boards/ へ昇格し '
        'Tier を判定する')

    BoardDef.model_validate(board)
    return board, rep


# ------------------------------------------------------------ 入出力層 ---


def load_recipe(recipe: str | Path) -> dict:
    p = Path(recipe)
    if not p.suffix == '.json' or not p.exists():
        p = RECIPES_DIR / f'{recipe}.json'
    if not p.exists():
        avail = ', '.join(sorted(x.stem for x in RECIPES_DIR.glob('*.json')))
        raise FileNotFoundError(
            f'レシピが見つからない: {recipe}(利用可能: {avail or "(なし)"})')
    doc = json.loads(p.read_text(encoding='utf-8'))
    for req in ('key', 'name', 'family', 'sources', 'device_refs',
                'clock', 'resources'):
        if req not in doc:
            raise ValueError(f'レシピに {req} が無い: {p}')
    if not (doc.get('toolchain') or doc.get('pin_format')):
        raise ValueError(f'レシピに toolchain か pin_format が必要: {p}')
    return doc


def render_notice(recipe: dict, rep: FetchReport, today: str) -> str:
    lines = [f'## {recipe["key"]}.json(draft、board-def fetch 生成 {today})',
             '',
             f'- 生成: `mdl-fpga board-def fetch {recipe["key"]}`'
             '(ピン値はレシピに無い=全て下記ソースからの機械抽出)']
    for key, spec in recipe['sources'].items():
        bits = [f'provider={spec["provider"]}']
        if spec.get('home'):
            bits.append(spec['home'])
        if spec.get('commit'):
            bits.append(f'commit {spec["commit"]}')
        if spec.get('license'):
            bits.append(f'SPDX-License-Identifier: {spec["license"]}')
        if spec.get('local') or spec.get('local_dir'):
            bits.append(f'ローカルコピー: {spec.get("local") or spec.get("local_dir")}')
        lines.append(f'- ソース {key}: {"; ".join(bits)}')
    lines.append('- 突合結果(役割: ref=ピン):')
    lines += [f'  - {w}' for w in rep.witness_lines]
    if rep.io_standard_checked:
        lines.append(f'- io_standard 整合確認: {rep.io_standard_checked} 参照')
    if rep.single_source:
        lines.append(f'- **単一ソース位置(要追加確認): '
                     f'{", ".join(rep.single_source)}**')
    lines.append('- 注意: 実機未検証(unverified)。detection 署名'
                 '(IDCODE/USB)は実機採取後に更新する。boards/ へ昇格する際は'
                 '本節を boards/NOTICE へ移す')
    return '\n'.join(lines) + '\n'


@dataclass
class FetchResult:
    ok: bool
    board_path: str = ''
    notice_path: str = ''
    board: dict | None = None
    verified_roles: list[str] = field(default_factory=list)
    single_source: list[str] = field(default_factory=list)
    error: str = ''

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'board_path': self.board_path,
                'notice_path': self.notice_path,
                'verified_roles': self.verified_roles,
                'single_source': self.single_source, 'error': self.error}

    def render(self) -> str:
        if not self.ok:
            return f'[FAIL] board-def fetch: {self.error}'
        lines = [f'[OK] board JSON draft: {self.board_path}',
                 f'     NOTICE draft   : {self.notice_path}',
                 f'     2ソース検証済み役割: {", ".join(self.verified_roles)}']
        if self.single_source:
            lines.append(f'     単一ソース位置(要追加確認): '
                         f'{", ".join(self.single_source)}')
        lines += ['     次の一手: board-def validate でピン実在を EDA/nextpnr 検証し、',
                  '     実機受入(up --hw-check)後に boards/ へ昇格する'
                  '(NOTICE draft を boards/NOTICE へ移す)']
        return '\n'.join(lines)


def run_fetch(recipe: str | Path, out_dir: str | Path | None = None,
              *, offline: bool = False, fetcher=None) -> FetchResult:
    try:
        doc = load_recipe(recipe)
        sources = acquire_sources(doc, offline=offline, fetcher=fetcher)
        today = _dt.date.today().isoformat()
        board, rep = build_board(doc, sources, today=today)
    except (CrossCheckError, FileNotFoundError, ValueError, KeyError,
            OSError) as e:
        return FetchResult(ok=False, error=f'{type(e).__name__}: {e}')

    out = Path(out_dir) if out_dir else (
        TOOL_DIR / 'runs' / 'boarddef'
        / f'{doc["key"]}-{time.strftime("%Y%m%d-%H%M%S")}')
    board_path = out / f'{doc["key"]}.json'
    notice_path = out / 'NOTICE.draft.md'
    if board_path.exists():
        return FetchResult(ok=False, error=(
            f'{board_path} が既に存在する(上書きしない)。--out で別の'
            '出力先を指定するか、既存 draft を確認して手動で整理する'))
    out.mkdir(parents=True, exist_ok=True)
    board_path.write_text(
        json.dumps(board, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    notice_path.write_text(render_notice(doc, rep, today), encoding='utf-8')
    return FetchResult(ok=True, board_path=str(board_path),
                       notice_path=str(notice_path), board=board,
                       verified_roles=rep.verified_roles,
                       single_source=rep.single_source)
