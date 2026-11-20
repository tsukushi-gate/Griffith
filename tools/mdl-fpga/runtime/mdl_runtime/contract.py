"""io_contract.json / goldens.json の読込(契約はランタイムの唯一の情報源)。"""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_VERSIONS = (1,)
_REQUIRED = ('contract_version', 'name', 'in_width', 'out_width',
             'uart', 'flash')


def find_app_dir(path: str | Path) -> Path:
    """app ディレクトリを解決する(app/ 直指定でも zip 展開ルートでも可)。"""
    p = Path(path)
    for cand in (p, p / 'app'):
        if (cand / 'io_contract.json').exists():
            return cand
    raise FileNotFoundError(
        f'{p} に io_contract.json が見つからない(bundle zip の展開先か '
        'その中の app/ を指定する)')


def load_contract(app_dir: str | Path) -> dict:
    app_dir = find_app_dir(app_dir)
    doc = json.loads((app_dir / 'io_contract.json').read_text(encoding='utf-8'))
    missing = [k for k in _REQUIRED if k not in doc]
    if missing:
        raise ValueError(f'io_contract.json に必須キーが無い: {missing}')
    if doc['contract_version'] not in SUPPORTED_VERSIONS:
        raise ValueError(
            f'未対応の contract_version: {doc["contract_version"]}'
            f'(対応: {SUPPORTED_VERSIONS})。ランタイムを更新する')
    return doc


def load_goldens(app_dir: str | Path, contract: dict) -> list[dict]:
    """[{'x': int, 'y': int}, ...](hex 文字列は int へ解決済み)。"""
    app_dir = find_app_dir(app_dir)
    fname = (contract.get('goldens') or {}).get('file', 'goldens.json')
    p = app_dir / fname
    if not p.exists():
        raise FileNotFoundError(f'goldens が無い: {p}(selftest 不可)')
    doc = json.loads(p.read_text(encoding='utf-8'))
    return [{'x': int(v['x'], 16), 'y': int(v['y'], 16)}
            for v in doc['vectors']]
