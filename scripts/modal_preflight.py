#!/usr/bin/env python
"""Modal GPU同時実行数の投入前チェック(preflight)。

事故防止: 同時実行app数がlimit(既定8)を超えた状態で新規ジョブを投入し、
Modalの同時GPU実行limit(10)を突破してしまうのを防ぐ。

- count_running_apps() -> int | None: 実行中app数(取得失敗はNone)
- preflight_check(limit=None) -> bool: 上限超過ならFalse。API失敗・timeoutは
  fail-open(True=チェックスキップ)。limit既定は環境変数 MD_MODAL_MAX_APPS(既定8)。
  MD_SKIP_PREFLIGHT=1 で無条件スキップ(緊急用)。

CLI:
    uv run python scripts/modal_preflight.py   # 読み取りのみ。上限超過で exit 1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Modalの同時GPU実行limitは10。安全マージンを取り既定8。
DEFAULT_MAX_APPS = 8
_MODAL_LIST_TIMEOUT_S = 15

# state値が実行中相当かを判定するトークン(CLIバージョン差・大文字小文字を吸収)。
# 例: "ephemeral (detached)", "running", "deploying"
_RUNNING_TOKENS = ('running', 'ephemeral', 'deploying', 'pending', 'starting')


def _is_running_state(state: str) -> bool:
    """state文字列が実行中相当かを返す。"""
    s = (state or '').lower()
    return any(tok in s for tok in _RUNNING_TOKENS)


def _env_int(name: str, default: int) -> int:
    """環境変数を整数として読む(未設定・不正時はdefault)。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def count_running_apps() -> int | None:
    """`modal app list --json` で実行中相当のapp数を数える。

    Returns:
        実行中app数。CLI失敗・timeout・出力解釈不能時はNone(取得失敗)。
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

    count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # JSONキーはCLIバージョンで大文字/小文字が揺れるため両対応
        state = entry.get('state') or entry.get('State') or ''
        if _is_running_state(str(state)):
            count += 1
    return count


def preflight_check(limit: int | None = None) -> bool:
    """投入前チェック。上限超過ならFalse、それ以外(取得失敗含む)はTrue。"""
    if os.environ.get('MD_SKIP_PREFLIGHT') == '1':
        print('[preflight] MD_SKIP_PREFLIGHT=1 のためチェックをスキップします')
        return True

    if limit is None:
        limit = _env_int('MD_MODAL_MAX_APPS', DEFAULT_MAX_APPS)

    running = count_running_apps()
    if running is None:
        # fail-open: 取得失敗時は投入をブロックしない(警告のみ)
        print(
            '[preflight] 警告: 実行中app数を取得できませんでした(チェックをスキップ)',
            file=sys.stderr,
        )
        return True

    if running >= limit:
        print(
            f'[preflight] 中止: 実行中のModal app数が {running} 件で上限 {limit} 件に'
            '達しています。\n'
            '  Modalの同時GPU実行limitは10です。`modal app list` で確認し、\n'
            '  不要なジョブを1件ずつ停止してから再投入してください。\n'
            '  緊急時は MD_SKIP_PREFLIGHT=1 でスキップできます。',
            file=sys.stderr,
        )
        return False

    print(f'[preflight] OK: 実行中app {running} 件 / 上限 {limit} 件')
    return True


def main() -> int:
    return 0 if preflight_check() else 1


if __name__ == '__main__':
    raise SystemExit(main())
