"""シリアルポート自動検出(契約内の USB 署名のみを使う。board DB 不要)。"""

from __future__ import annotations

import re


def _to_int(v) -> int | None:
    if v is None:
        return None
    return int(v, 16) if isinstance(v, str) else int(v)


def list_candidates(contract: dict) -> list[dict]:
    """存在するポートの一覧(契約署名との一致情報つき)。"""
    from serial.tools import list_ports
    sigs = (contract.get('serial') or {}).get('usb', [])
    out = []
    for p in list_ports.comports():
        matched = None
        for sig in sigs:
            vid, pid = _to_int(sig.get('vid')), _to_int(sig.get('pid'))
            rx = sig.get('desc_regex')
            if vid is not None and p.vid != vid:
                continue
            if pid is not None and p.pid != pid:
                continue
            if rx and not re.search(rx, f'{p.description} {p.manufacturer}',
                                    re.I):
                continue
            matched = sig
            break
        out.append({'device': p.device,
                    'description': p.description,
                    'vid': p.vid, 'pid': p.pid,
                    'matched': matched is not None})
    return out


def find_port(contract: dict) -> str | None:
    """契約署名に一致する最初のポート(名前順)。無ければ None。"""
    hits = [c['device'] for c in list_candidates(contract) if c['matched']]
    return sorted(hits)[0] if hits else None
