#!/usr/bin/env python
"""Modal detachジョブのハング検知(読み取り専用の監視)。

`modal app logs <app_id>` の出力の伸びを追跡し、一定時間新規出力が無ければ
ALERTを出す。**停止操作は一切しない(通知のみ)**。事故防止: JITハング等の
「気づけない30分放置」を防ぐ。

使い方:
    uv run python scripts/modal_watch.py ap-xxxx --interval 60 --stall-min 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

_MODAL_LIST_TIMEOUT_S = 15
_LOGS_FETCH_TIMEOUT_S = 25  # `modal app logs` はfollowするため打ち切って現状を得る
_SEC_PER_MIN = 60

# state値が実行中相当かを判定するトークン(modal_preflight.pyと同基準)
_RUNNING_TOKENS = ('running', 'ephemeral', 'deploying', 'pending', 'starting')
# 最終行がJITコンパイル関連かを判定するトークン
_JIT_TOKENS = ('コンパイル', 'カーネル', 'compiling', 'ninja', 'nvcc', 'cpp_extension', 'building')


def _is_running_state(state: str) -> bool:
    s = (state or '').lower()
    return any(tok in s for tok in _RUNNING_TOKENS)


def _looks_like_jit(line: str) -> bool:
    s = (line or '').lower()
    return any(tok in s for tok in _JIT_TOKENS)


def _last_nonempty_line(text: str) -> str:
    for line in reversed((text or '').splitlines()):
        if line.strip():
            return line
    return ''


def _fetch_logs(app_id: str) -> str | None:
    """現時点のログを取得する(followを打ち切って取得済み分を返す)。

    Returns:
        ログ文字列。取得コマンド自体が起動不能ならNone。
    """
    try:
        proc = subprocess.run(
            ['uv', 'run', 'modal', 'app', 'logs', app_id],
            capture_output=True, text=True, timeout=_LOGS_FETCH_TIMEOUT_S,
        )
        return proc.stdout or ''
    except subprocess.TimeoutExpired as e:
        # followモードで打ち切り。取得済み出力を使う
        out = e.stdout
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return out or ''
    except OSError:
        return None


def _app_state(app_id: str) -> str | None:
    """app_idのstateを返す。

    Returns:
        - None: `modal app list` が失敗・解釈不能(取得失敗)
        - '':   一覧に存在しない(終了して除外された可能性)
        - str:  stateの文字列
    """
    try:
        proc = subprocess.run(
            ['uv', 'run', 'modal', 'app', 'list', '--json'],
            capture_output=True, text=True, timeout=_MODAL_LIST_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # JSONキーはCLIバージョンで大文字/小文字が揺れるため両対応
        aid = entry.get('app_id') or entry.get('App ID') or entry.get('app id') or ''
        if aid == app_id:
            return str(entry.get('state') or entry.get('State') or '')
    return ''  # 一覧に無い


def _notify(msg: str) -> None:
    """macOS通知(ベストエフォート。失敗は無視)。"""
    try:
        subprocess.run(
            ['osascript', '-e',
             f'display notification {json.dumps(msg)} with title "modal_watch"'],
            capture_output=True, timeout=5,
        )
    except Exception:  # noqa: BLE001 — 通知失敗は監視継続に影響させない
        pass


def watch(app_id: str, interval: float, stall_min: float) -> int:
    """app_idを監視する。終了状態で0を返す。停止操作はしない。"""
    stall_sec = stall_min * _SEC_PER_MIN
    print(
        f'[modal_watch] 監視開始: {app_id} '
        f'(interval={interval}s, stall={stall_min}min。停止操作はしません)'
    )
    prev_logs: str | None = None
    last_change = time.monotonic()
    alerted = False

    while True:
        state = _app_state(app_id)
        if state == '':
            print(f'[modal_watch] {app_id} は一覧に見つかりません(終了の可能性)。監視終了')
            return 0
        if state is not None and not _is_running_state(state):
            print(f'[modal_watch] {app_id} は終了状態です(state={state})。監視終了')
            return 0

        logs = _fetch_logs(app_id)
        if logs is not None and logs != prev_logs:
            prev_logs = logs
            last_change = time.monotonic()
            alerted = False
        else:
            idle = time.monotonic() - last_change
            if idle >= stall_sec and not alerted:
                minutes = int(idle // _SEC_PER_MIN)
                msg = f'[modal_watch] ALERT: {app_id} が{minutes}分間無出力(ハングの可能性)'
                if _looks_like_jit(_last_nonempty_line(prev_logs or '')):
                    msg += ' — 既知のCUDA JITハングの可能性(MODAL_KERNEL_ISSUE.md参照)'
                print(msg)
                _notify(msg)
                alerted = True

        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Modal detachジョブのハング検知(監視のみ・停止は絶対にしない)',
    )
    parser.add_argument('app_id', help='監視するModal app ID(例: ap-xxxx)')
    parser.add_argument('--interval', type=float, default=60,
                        help='ポーリング間隔(秒, 既定60)')
    parser.add_argument('--stall-min', type=float, default=15,
                        help='無出力とみなす時間(分, 既定15)')
    args = parser.parse_args(argv)
    try:
        return watch(args.app_id, args.interval, args.stall_min)
    except KeyboardInterrupt:
        print('\n[modal_watch] 中断しました', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
