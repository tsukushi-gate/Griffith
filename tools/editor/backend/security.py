"""エディタ/ビジュアライザ backend の共通セキュリティ機構。

エディタとビジュアライザは**信頼された単一ユーザのローカル環境専用**ツールで、
ログイン認証を持たない。「ローカルだから安全」を成立させるため、本モジュールは
次の3点を提供する。

1. パスの作業ディレクトリ内包チェック(監査B3)
   `resolve_workspace_path` は `python_importer.resolve_python_path` と同じ方式
   (`Path.resolve()` 後に workspace root の配下かを検証)で、`../` 相対脱出・
   絶対パス指定・workspace内シンボリックリンク経由の脱出をまとめて拒否する。

2. ブラウザ経由攻撃の遮断
   - `Host` ヘッダ検証: DNSリバインディング対策。ループバック名義のHostだけを
     通し、攻撃者が用意したDNS名(`evil.example`)経由のアクセスを421で拒否する。
   - `Origin` ヘッダ検証: 状態変更メソッド(POST/PUT/DELETE/PATCH)とWebSocketで、
     ループバック以外のOriginを403で拒否する(古典的CSRF対策)。

3. アップロードサイズ上限

環境変数:
    GRIFFITH_EDITOR_ALLOWED_HOSTS  追加で許可するホスト(カンマ区切り)。
                                   外部バインド(`--host 0.0.0.0`)時に指定する。
    GRIFFITH_EDITOR_MAX_UPLOAD_MB  アップロード上限(MiB、既定512)。
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

ALLOWED_HOSTS_ENV = 'GRIFFITH_EDITOR_ALLOWED_HOSTS'
MAX_UPLOAD_MB_ENV = 'GRIFFITH_EDITOR_MAX_UPLOAD_MB'

DEFAULT_MAX_UPLOAD_MB = 512
DEFAULT_ALLOWED_HOSTS = frozenset({'localhost', '127.0.0.1', '::1', '[::1]'})

# ブラウザからのアクセスを許可するオリジンのスキーム
_ALLOWED_ORIGIN_SCHEMES = frozenset({'http', 'https'})


class WorkspacePathError(ValueError):
    """workspace外・拡張子違反などパス検証に失敗したことを表す。"""


# ---------------------------------------------------------------------------
# 1. パス内包チェック(B3)
# ---------------------------------------------------------------------------

def resolve_workspace_path(
    path: str | os.PathLike[str],
    *,
    suffixes: tuple[str, ...] | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
    must_exist: bool = False,
) -> Path:
    """workspace(リポジトリルート)配下のパスだけを許可して解決する。

    相対パスは workspace root 基準で解決する。`Path.resolve()` はシンボリック
    リンクも辿るため、workspace内のリンクからworkspace外を指すケースも弾ける。

    Args:
        path: APIが受け取った生のパス文字列。
        suffixes: 許可する拡張子(小文字、ドット込み)。Noneなら制限しない。
        workspace_root: 内包判定の基準ディレクトリ。
        must_exist: Trueなら存在しないパスを拒否する。

    Returns:
        解決済みの絶対パス。

    Raises:
        WorkspacePathError: workspace外・拡張子違反・(must_exist時)不存在。
    """
    raw = str(path).strip()
    if not raw:
        raise WorkspacePathError('パスが空です')
    if '\x00' in raw:
        raise WorkspacePathError('パスにNUL文字を含められません')

    root = workspace_root.resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()

    if p != root and root not in p.parents:
        raise WorkspacePathError(f'ワークスペース外のパスは指定できません: {p}')
    if suffixes is not None and p.suffix.lower() not in suffixes:
        allowed = '/'.join(suffixes)
        raise WorkspacePathError(f'許可されていない拡張子です({allowed} のみ): {p.name}')
    if must_exist and not p.exists():
        raise WorkspacePathError(f'ファイルが存在しません: {p}')
    return p


# ---------------------------------------------------------------------------
# 2. Host / Origin 検証(R6)
# ---------------------------------------------------------------------------

def _normalize_host(value: str) -> str:
    """`host:port` からポートを落として小文字化する(IPv6の `[::1]` に対応)。"""
    host = value.strip().lower()
    if host.startswith('['):
        end = host.find(']')
        if end != -1:
            return host[: end + 1]
        return host
    return host.rsplit(':', 1)[0] if host.count(':') == 1 else host


def allowed_hosts(extra: tuple[str, ...] = ()) -> frozenset[str]:
    """許可ホスト集合(ループバック + `--host` 指定 + 環境変数)を返す。"""
    hosts = set(DEFAULT_ALLOWED_HOSTS)
    for source in (extra, tuple(os.environ.get(ALLOWED_HOSTS_ENV, '').split(','))):
        for item in source:
            name = _normalize_host(item)
            # 全インタフェース指定は「特定ホスト名の許可」ではないので無視する
            if name and name not in ('0.0.0.0', '::', '[::]'):
                hosts.add(name)
    return frozenset(hosts)


def is_host_allowed(host_header: str | None, hosts: frozenset[str]) -> bool:
    """`Host` ヘッダが許可集合に含まれるか(ポートは任意)。"""
    if not host_header:
        return False
    return _normalize_host(host_header) in hosts


def is_origin_allowed(origin: str | None, hosts: frozenset[str]) -> bool:
    """`Origin` ヘッダが許可集合に含まれるか。

    Originが無い(非ブラウザ・same-origin GET等)場合は許可する。ブラウザは
    クロスオリジンの状態変更リクエストとWebSocketに必ずOriginを付けるため、
    「不正なOriginを拒否する」だけでCSRF/リバインディングを遮断できる。
    """
    if origin is None or origin == '':
        return True
    if origin == 'null':
        return False
    parts = urlsplit(origin)
    if parts.scheme not in _ALLOWED_ORIGIN_SCHEMES or not parts.netloc:
        return False
    return _normalize_host(parts.netloc) in hosts


# ---------------------------------------------------------------------------
# 3. アップロード上限(R10)
# ---------------------------------------------------------------------------

def max_upload_bytes() -> int:
    """アップロード上限(バイト)。環境変数で上書きできる。"""
    raw = os.environ.get(MAX_UPLOAD_MB_ENV, '')
    try:
        mb = float(raw) if raw else DEFAULT_MAX_UPLOAD_MB
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    if mb <= 0:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(mb * 1024 * 1024)
