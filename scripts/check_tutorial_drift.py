#!/usr/bin/env python3
"""生成済みチュートリアル(docs-site/.../tutorials)のドリフト検出器。

`scripts/build_tutorials.py` は examples/*.py を実行してmd・図を再生成する。
CIでは再生成後に「コミット済みの内容と一致するか」を見たいが、素の `git diff --exit-code` は
**一度も通らなかった**。理由は、学習を伴う出力(損失・精度の浮動小数)と、そこから描かれる図が
CPUアーキテクチャ(Apple Silicon と x86)で変わるため。実測例: logic-gan の終盤の D 精度は
Mac 0.797 / x86 0.876。コミット済みは Mac 生成、CIは x86 なので、生の差分は常に偽陽性になる。

そこで本スクリプトは「ソース由来の内容」だけを比較する。

比較する(1文字でも違えばドリフト):
- 各 `*.md` の frontmatter・見出し・地の文・コードフェンス(```python 等)
- 出力ブロック(```text title="出力")の **個数と順序**
- 画像参照行(`![png](...)`)の **個数と順序**
- `index.mdx`(ギャラリー。TUTORIALS から決定的に生成される)
- PNG ファイルの **集合**(追加・削除・改名はドリフト。同名PNGのバイト差は許容)
- 未追跡の新規 md、基準リビジョンにあって作業ツリーに無い md

比較しない(プラットフォーム差で揺れる):
- 出力ブロックの中身(数値)
- PNG のバイト列

使い方: ビルド後(作業ツリー=再生成結果)に実行し、基準(既定 HEAD)と比べる。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = "docs-site/src/content/docs/tutorials"

# build_tutorials.normalize_markdown / _IMAGE_LINE と揃える
OUTPUT_FENCE_OPEN = re.compile(r'^```text title="出力"\s*$')
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")
OUTPUT_PLACEHOLDER = "<<出力ブロック>>"
IMAGE_PLACEHOLDER = "<<図>>"


def normalize(text: str) -> list[str]:
    """出力ブロックと画像行をプレースホルダへ置換した行リストを返す。"""
    out: list[str] = []
    in_output = False
    for line in text.splitlines():
        if in_output:
            if line.rstrip() == "```":
                in_output = False
            continue
        if OUTPUT_FENCE_OPEN.match(line):
            in_output = True
            out.append(OUTPUT_PLACEHOLDER)
            continue
        if IMAGE_LINE.match(line):
            out.append(IMAGE_PLACEHOLDER)
            continue
        out.append(line)
    return out


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)


def _repo_rel(tdir: Path, cwd: Path) -> str:
    top = _git(["rev-parse", "--show-toplevel"], cwd)
    if top.returncode != 0:
        raise SystemExit("git リポジトリではありません")
    return tdir.resolve().relative_to(Path(top.stdout.decode().strip()).resolve()).as_posix()


def first_diff(a: list[str], b: list[str]) -> str:
    for n in range(max(len(a), len(b))):
        x = a[n] if n < len(a) else "<行なし>"
        y = b[n] if n < len(b) else "<行なし>"
        if x != y:
            return f"{n + 1}行目\n      基準 : {x}\n      現在 : {y}"
    return "(差分位置不明)"


def check(tdir: Path, base: str, verbose: bool = False) -> list[str]:
    """ドリフトの説明文リストを返す(空ならOK)。"""
    tdir = tdir.resolve()
    rel = _repo_rel(tdir, tdir if tdir.exists() else tdir.parent)
    cwd = tdir
    tracked = _git(["ls-tree", "-r", "--name-only", base, "--", "."], cwd)
    if tracked.returncode != 0:
        raise SystemExit(f"基準 {base} を読めません: {tracked.stderr.decode().strip()}")
    # ls-tree は cwd 相対で出力される
    base_files = {p for p in tracked.stdout.decode().splitlines() if p}
    problems: list[str] = []

    disk_docs = {
        p.relative_to(tdir).as_posix() for p in tdir.glob("*") if p.suffix in (".md", ".mdx")
    }
    base_docs = {p for p in base_files if "/" not in p and p.endswith((".md", ".mdx"))}
    for name in sorted(disk_docs - base_docs):
        problems.append(f"{rel}/{name}: 基準に無い新規ファイルです(コミットしてください)")
    for name in sorted(base_docs - disk_docs):
        problems.append(f"{rel}/{name}: 再生成結果から消えています")

    for name in sorted(disk_docs & base_docs):
        cur = (tdir / name).read_text(encoding="utf-8")
        old = _git(["show", f"{base}:./{name}"], cwd).stdout.decode("utf-8")
        if name.endswith(".mdx"):
            a, b = old.splitlines(), cur.splitlines()
            same = old == cur
        else:
            a, b = normalize(old), normalize(cur)
            same = a == b
        if same:
            if verbose:
                print(f"  OK  {name}")
            continue
        problems.append(
            f"{rel}/{name}: ソース由来の内容がずれています。最初の差分は {first_diff(a, b)}"
        )

    disk_png = {p.relative_to(tdir).as_posix() for p in tdir.rglob("*.png")}
    base_png = {p for p in base_files if p.endswith(".png")}
    for name in sorted(disk_png - base_png):
        problems.append(f"{rel}/{name}: 基準に無い図が増えています")
    for name in sorted(base_png - disk_png):
        problems.append(f"{rel}/{name}: 基準にあった図が無くなっています")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tutorials-dir", default=str(ROOT / DEFAULT_DIR))
    ap.add_argument("--base", default="HEAD", help="比較する基準リビジョン(既定 HEAD)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    tdir = Path(args.tutorials_dir)
    problems = check(tdir, args.base, args.verbose)
    if problems:
        print(f"チュートリアルのドリフトを検出しました({len(problems)}件):")
        for p in problems:
            print(f"  - {p}")
        print(
            "\n対処: 'uv run --group docs python scripts/build_tutorials.py' を実行し、"
            "md と図の変更をコミットしてください。\n"
            "(出力ブロックの数値と PNG のバイト差は、環境差で揺れるため無視しています)"
        )
        return 1
    n = len(list(tdir.glob("*.md")))
    print(f"OK: チュートリアル {n} 本にソース由来のドリフトはありません(基準 {args.base})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
