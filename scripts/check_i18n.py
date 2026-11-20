#!/usr/bin/env python3
"""日本語ページ(真実源)と英語ページ(en/)の追従状況を検査する。

`docs-site/src/content/docs/` 直下が日本語の真実源で、`en/` 以下が英訳。
どちらか一方だけを直すとサイレントにズレるので、英語ページの末尾に
**翻訳元と、翻訳した時点の日本語ファイルのハッシュ**をマーカーとして埋め、
このスクリプトで突き合わせる。

マーカー書式(英語ページの末尾に1行。`<相対パス>` は docs/ からの相対パス、
`<hash>` は**翻訳元の日本語ファイル**の md5 先頭12桁):

    .mdx:  {/* i18n:source=<相対パス> hash=<hash> */}
    .md:   <!-- i18n:source=<相対パス> hash=<hash> -->

検出する3種のドリフト:

- MISSING — 日本語ページに対応する英語ページが無い(未翻訳)
- STALE   — 英語ページはあるが、マーカーが無い/壊れている/hashが日本語の現在値と不一致
            (= 日本語を更新したのに英訳が追従していない)
- ORPHAN  — 日本語側が消えた/移動したのに残っている英語ページ

torchもnodeも要らない標準ライブラリのみのスクリプト。

    uv run --no-project python scripts/check_i18n.py            # 人間向け出力
    uv run --no-project python scripts/check_i18n.py --json     # 機械可読
    uv run --no-project python scripts/check_i18n.py --allow-missing   # 移行モード

終了コードは、問題があれば1(CIで落とす)。`--allow-missing` を付けると MISSING は
警告扱いになり、STALE/ORPHAN だけで落とす(英訳が全ページ揃うまでの移行モード)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs-site" / "src" / "content" / "docs"
EN_DIR = DOCS / "en"

PAGE_SUFFIXES = (".md", ".mdx")

# 意図的に英訳しないページ(docs/ からの相対パス、または `fnmatch` 形式のグロブ)。
# ここに入れたページは MISSING/ORPHAN/STALE のいずれからも除外される。
# 現時点では空。増やすときは「なぜ訳さないか」を必ずコメントで残すこと。
EXCLUDE: tuple[str, ...] = ()

# マーカーの抽出。`source=`/`hash=` の順は固定、コメント記法(.md/.mdx)は問わない。
MARKER_RE = re.compile(r"i18n:source=(?P<source>\S+)\s+hash=(?P<hash>[0-9a-fA-F]+)")

HASH_LEN = 12


def page_hash(path: Path) -> str:
    """ページの内容から追従用ハッシュ(md5先頭12桁)を求める。"""
    return hashlib.md5(path.read_bytes()).hexdigest()[:HASH_LEN]


def is_excluded(rel: str) -> bool:
    """除外リストに載っているページか。"""
    return any(rel == pattern or fnmatch(rel, pattern) for pattern in EXCLUDE)


def iter_pages(base: Path, *, skip_en: bool) -> list[Path]:
    """`base` 以下のページを列挙する(`skip_en` で en/ 以下を除く)。"""
    pages = []
    for suffix in PAGE_SUFFIXES:
        for path in base.rglob(f"*{suffix}"):
            if skip_en and EN_DIR in path.parents:
                continue
            pages.append(path)
    return sorted(pages)


def read_marker(path: Path) -> tuple[str, str] | None:
    """英語ページ末尾のマーカーを読む。無ければ None。"""
    # 複数あれば最後(=末尾)のものを採用する
    matches = MARKER_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        return None
    source, digest = matches[-1]
    return source, digest.lower()


@dataclass
class Issue:
    """1ページ分の検出結果。"""

    kind: str  # MISSING / STALE / ORPHAN
    page: str  # 日本語ページの相対パス(ORPHANのみ英語ページ)
    reason: str
    expected: str | None = None
    found: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "page": self.page,
            "reason": self.reason,
            "expected": self.expected,
            "found": self.found,
        }


def check() -> list[Issue]:
    """日英の対応を検査して問題の一覧を返す。"""
    if not DOCS.is_dir():
        raise SystemExit(f"ドキュメントディレクトリが見つかりません: {DOCS}")

    issues: list[Issue] = []
    ja_rels: set[str] = set()

    for ja_path in iter_pages(DOCS, skip_en=True):
        rel = ja_path.relative_to(DOCS).as_posix()
        if is_excluded(rel):
            continue
        ja_rels.add(rel)

        en_path = EN_DIR / rel
        if not en_path.is_file():
            issues.append(Issue("MISSING", rel, "英語版がありません"))
            continue

        expected = page_hash(ja_path)
        marker = read_marker(en_path)
        if marker is None:
            issues.append(
                Issue("STALE", rel, "英語版に i18n マーカーがありません", expected=expected)
            )
            continue

        source, found = marker
        if source != rel:
            issues.append(
                Issue(
                    "STALE",
                    rel,
                    "マーカーの source が対応する日本語ページと一致しません",
                    expected=rel,
                    found=source,
                )
            )
            continue
        if found != expected:
            issues.append(
                Issue(
                    "STALE",
                    rel,
                    "日本語版が更新されています(hash不一致)",
                    expected=expected,
                    found=found,
                )
            )

    for en_path in iter_pages(EN_DIR, skip_en=False) if EN_DIR.is_dir() else []:
        rel = en_path.relative_to(EN_DIR).as_posix()
        if is_excluded(rel) or rel in ja_rels:
            continue
        if (DOCS / rel).is_file():
            continue  # 除外された日本語ページに対応する英語版は不問
        issues.append(Issue("ORPHAN", f"en/{rel}", "対応する日本語ページがありません"))

    return issues


def marker_line(rel: str, digest: str, suffix: str) -> str:
    """ページ末尾に貼るマーカー行を組み立てる(手直し時の案内用)。"""
    body = f"i18n:source={rel} hash={digest}"
    return f"{{/* {body} */}}" if suffix == ".mdx" else f"<!-- {body} -->"


def report(issues: list[Issue], *, allow_missing: bool) -> None:
    """人間向けの検査結果を標準出力へ書く。"""
    total_ja = len(iter_pages(DOCS, skip_en=True))
    by_kind = {
        kind: [i for i in issues if i.kind == kind] for kind in ("MISSING", "STALE", "ORPHAN")
    }

    labels = {
        "MISSING": "MISSING(英語版なし)",
        "STALE": "STALE(日本語が更新済み/マーカー不備)",
        "ORPHAN": "ORPHAN(日本語側が消えた英語ページ)",
    }
    for kind, label in labels.items():
        items = by_kind[kind]
        note = " ※--allow-missing により警告" if kind == "MISSING" and allow_missing else ""
        print(f"\n== {label}: {len(items)}件{note}")
        for issue in items:
            print(f"  - {issue.page}: {issue.reason}")
            if kind == "STALE" and issue.expected is not None:
                if re.fullmatch(rf"[0-9a-f]{{{HASH_LEN}}}", issue.expected):
                    line = marker_line(issue.page, issue.expected, Path(issue.page).suffix)
                    print(f"      → 追記/更新: {line}")
                else:
                    print(f"      → source を '{issue.expected}' に直す(現在: '{issue.found}')")

    translated = total_ja - len(by_kind["MISSING"])
    print(
        f"\n日本語ページ {total_ja}件 / 英訳あり {translated}件 / "
        f"MISSING {len(by_kind['MISSING'])} · STALE {len(by_kind['STALE'])} · "
        f"ORPHAN {len(by_kind['ORPHAN'])}"
    )
    if not issues:
        print("OK: 日英のドキュメントは同期しています")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="機械可読な結果を出力する")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="MISSING(未翻訳)を警告に緩める移行モード。STALE/ORPHANでは落ちる",
    )
    args = parser.parse_args(argv)

    issues = check()
    fatal = [i for i in issues if not (args.allow_missing and i.kind == "MISSING")]

    if args.json:
        total_ja = len(iter_pages(DOCS, skip_en=True))
        print(
            json.dumps(
                {
                    "ja_pages": total_ja,
                    "allow_missing": args.allow_missing,
                    "counts": {
                        kind: len([i for i in issues if i.kind == kind])
                        for kind in ("MISSING", "STALE", "ORPHAN")
                    },
                    "issues": [i.to_dict() for i in issues],
                    "ok": not fatal,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        report(issues, allow_missing=args.allow_missing)
        if fatal:
            print(
                "\n::error::日英ドキュメントがズレています。英語ページを更新し、"
                "末尾の i18n マーカーの hash を新しい md5 先頭12桁へ直してください"
                "(手順: docs-site/README.md の「執筆メモ」節)"
            )

    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
