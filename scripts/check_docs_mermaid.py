#!/usr/bin/env python3
"""公開ドキュメントサイトの Mermaid 図の構文を機械検査する(DOCS-7)。

Mermaid はクライアント側で描画されるので、**構文が壊れていてもビルドは緑のまま通る**。
壊れた図はページを開いた読者にだけ「Syntax error in text」の赤い箱として見える
(しかも図のあるページでしか mermaid を読み込まないため、他ページの検証では気づけない)。

このスクリプトは ja/en 全ページから ```mermaid フェンスを抜き出し、
**docs-site/node_modules の mermaid を Node から直接呼んで `mermaid.parse()` に通す**。
サイトが実際に描画するのと同じバージョン・同じパーサで検証するので、
「ローカルでは出たが CI では壊れている」というズレが起きない。

    uv run --no-project python scripts/check_docs_mermaid.py
    uv run --no-project python scripts/check_docs_mermaid.py --json
    cd docs-site && npm run check:mermaid

前提: `docs-site/node_modules`(= `npm ci` 済み)と node。torch も Python 依存も要らない。

抽出の規則:

- 対象は `docs-site/src/content/docs/` 配下(`en/` を含む)の `.md` / `.mdx` 全ファイル
- フェンスの入れ子を追跡し、**````markdown の中に書かれた説明用の ```mermaid は検査しない**
  (執筆ガイドの「書き方の例」であって、実際に描画される図ではないため)
- 情報文字列(```mermaid title="…")は許容する

終了コードは、構文エラーが1件でもあれば 1。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs-site" / "src" / "content" / "docs"
WORKER = ROOT / "docs-site" / "scripts" / "mermaid_parse.mjs"
NODE_MODULES = ROOT / "docs-site" / "node_modules"

# フェンスの開始/終了。``` 以上のバッククォート、または ~~~ 以上のチルダ。
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})\s*(?P<info>.*)$")
# mermaid の parse エラーは "Parse error on line 3:" の形で相対行番号を持つ。
PARSE_LINE_RE = re.compile(r"on line (\d+)")


@dataclass
class Block:
    """1つの ```mermaid フェンス。"""

    path: Path
    line: int  # フェンス開始行(1始まり)
    source: str

    @property
    def rel(self) -> str:
        """リポジトリルートからの相対パス。"""
        return str(self.path.relative_to(ROOT))


def extract_blocks(path: Path) -> list[Block]:
    """1ファイルから mermaid フェンスを抜き出す(入れ子フェンスは中身として飛ばす)。"""
    blocks: list[Block] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    open_fence: str | None = None  # 開いているフェンスの文字列(長さで閉じ判定)
    collecting_from: int | None = None
    buffer: list[str] = []

    for index, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if open_fence is not None:
            # フェンスの中: 同種で同じ長さ以上のフェンスだけが閉じる
            if (
                match
                and not match.group("info")
                and match.group("fence")[0] == open_fence[0]
                and len(match.group("fence")) >= len(open_fence)
            ):
                if collecting_from is not None:
                    blocks.append(
                        Block(path=path, line=collecting_from, source="\n".join(buffer) + "\n")
                    )
                    collecting_from, buffer = None, []
                open_fence = None
            elif collecting_from is not None:
                buffer.append(line)
            continue
        if match and match.group("fence"):
            open_fence = match.group("fence")
            info = match.group("info").strip()
            if info == "mermaid" or info.startswith("mermaid "):
                collecting_from, buffer = index, []

    return blocks


def collect_blocks() -> list[Block]:
    """ja/en 全ページの mermaid フェンスを集める。"""
    blocks: list[Block] = []
    for path in sorted(DOCS.rglob("*")):
        if path.suffix in {".md", ".mdx"} and path.is_file():
            blocks.extend(extract_blocks(path))
    return blocks


def parse_with_mermaid(blocks: list[Block]) -> list[dict]:
    """Node 経由で mermaid.parse を呼び、ブロックごとの結果を返す。"""
    payload = json.dumps(
        {"blocks": [{"id": i, "source": b.source} for i, b in enumerate(blocks)]},
        ensure_ascii=False,
    )
    proc = subprocess.run(
        ["node", str(WORKER)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(ROOT / "docs-site"),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mermaid_parse.mjs が異常終了しました(exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)["results"]


def format_failure(block: Block, error: str) -> str:
    """失敗1件を『パス:行 エラー』の形に整える。"""
    message = " ".join(error.split())
    relative = PARSE_LINE_RE.search(error)
    # mermaid の行番号はブロック先頭を1行目とする相対値。フェンス行を足して絶対行にする。
    line = block.line + int(relative.group(1)) if relative else block.line
    return f"{block.rel}:{line}  {message}"


def main() -> int:
    """検査を実行して結果を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="機械可読な出力")
    args = parser.parse_args()

    if not NODE_MODULES.is_dir():
        print(
            f"NG: {NODE_MODULES} がありません。`cd docs-site && npm ci` を実行してください。",
            file=sys.stderr,
        )
        return 1

    blocks = collect_blocks()
    results = parse_with_mermaid(blocks) if blocks else []

    failures: list[str] = []
    diagram_types: dict[str, int] = {}
    for result in results:
        block = blocks[result["id"]]
        if result["ok"]:
            kind = result.get("diagramType") or "unknown"
            diagram_types[kind] = diagram_types.get(kind, 0) + 1
        else:
            failures.append(format_failure(block, result["error"]))

    files = len({b.rel for b in blocks})
    if args.json:
        print(
            json.dumps(
                {
                    "ok": not failures,
                    "failures": failures,
                    "stats": {
                        "blocks": len(blocks),
                        "files": files,
                        "diagram_types": diagram_types,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if failures else 0

    print("docs Mermaid 構文検査(DOCS-7)")
    print(f"  {files} ファイル / {len(blocks)} 図を mermaid.parse で検証")
    if diagram_types:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(diagram_types.items()))
        print(f"  図種: {kinds}")
    for message in failures:
        print(f"  FAIL  {message}")
    if failures:
        print(f"\nNG: {len(failures)} 件の Mermaid 構文エラーがあります")
        return 1
    print("\nOK: すべての Mermaid 図がパースできました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
