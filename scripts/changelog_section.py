#!/usr/bin/env python3
"""CHANGELOG.md から指定バージョンの節だけを抜き出す。

release-please が生成する CHANGELOG(Keep a Changelog 風の `## [0.1.0] ...` /
`## 0.1.0 ...` 見出し)を想定する。公開スナップショットのコミット本文と
GitHub Release の本文の両方でこの出力を使う。

使い方:
    uv run python scripts/changelog_section.py v0.1.0 [--file CHANGELOG.md]
見つからない場合は空文字を出して exit 0(リリースを止めない)。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^##\s+")
VERSION_IN_HEADING = re.compile(r"(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)")


def extract(text: str, version: str) -> str:
    want = version.lstrip("vV")
    lines = text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if not HEADING.match(line):
            continue
        if start is None:
            m = VERSION_IN_HEADING.search(line)
            if m and m.group(1) == want:
                start = i
            continue
        # 次の見出しで終端
        return "\n".join(lines[start:i]).strip() + "\n"
    if start is None:
        return ""
    return "\n".join(lines[start:]).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CHANGELOG から該当バージョンの節を抽出")
    ap.add_argument("version", help="v0.1.0 または 0.1.0")
    ap.add_argument("--file", type=Path, default=Path("CHANGELOG.md"))
    args = ap.parse_args(argv)

    if not args.file.exists():
        print("", end="")
        print(f"warning: {args.file} がありません", file=sys.stderr)
        return 0
    section = extract(args.file.read_text(encoding="utf-8"), args.version)
    if not section:
        print(f"warning: {args.version} の節が見つかりません", file=sys.stderr)
    sys.stdout.write(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
