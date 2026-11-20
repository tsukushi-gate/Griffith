#!/usr/bin/env python3
"""公開ドキュメントサイトの図(アセット)の健全性を機械検査する(DOCS-5)。

チュートリアルの図は `scripts/build_tutorials.py` が examples を実行して生成する。
生成は一方向なので、途中で壊れても**ページを開くまで気づかない**類の劣化が起きる:

- 図が低解像度のまま残る(ライトボックスで拡大するとボケる)
- セル分割の事故で**軸も凡例も無い空の図**が出力される
- 図番号がズレて、ページやギャラリーが**存在しないファイル**を指す
- 古い図がディレクトリに残り続ける(孤児。中身は古い設定のまま)

このスクリプトはそれらを CI で落とせる形に固定する。torch も node も要らず、
標準ライブラリ + Pillow だけで動く。

    uv run --no-project --with pillow python scripts/check_docs_assets.py
    uv run --no-project --with pillow python scripts/check_docs_assets.py --json
    cd docs-site && npm run check:assets

検査項目(詳細は各 CHECK_* 定数のコメント):

  resolution — 参照されている PNG の DPI と最小幅(拡大表示に耐えるか)
  ink        — 空図・ほぼ白紙の図の検出
  refs       — ページ(ja/en)から参照される図が実在するか
  gallery    — 全チュートリアルのサムネイルが解決できるか
  orphan     — どのページからも参照されない図(既定で失敗。`--allow-orphans` で警告に落とせる)

終了コードは、失敗が1件でもあれば 1。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 図の生成側と定数を共有する(FIGURE_DPI の単一情報源は build_tutorials.py)。
# build_tutorials は import 時点では標準ライブラリしか使わないので torch 不要で読める。
from build_tutorials import FIGURE_DPI, OUT_DIR, TUTORIALS, _thumb_for  # noqa: E402

DOCS = ROOT / "docs-site" / "src" / "content" / "docs"
JA_TUTORIALS = OUT_DIR
EN_TUTORIALS = DOCS / "en" / "tutorials"
GALLERY = JA_TUTORIALS / "index.mdx"

# --- 閾値(すべて実測分布から較正。根拠は各コメント) ------------------------

# CHECK_DPI: PNG の pHYs チャンクに記録された解像度。matplotlib の inline バックエンドは
# `InlineBackend.print_figure_kwargs` の dpi をそのまま埋めるので、生成時の
# FIGURE_DPI(=200)が素通しで出る。DOCS-3 で 100 → 200 に引き上げた際、
# 再生成されなかった古い図だけが 100dpi のまま残っていた(実測: 66枚中5枚)。
# 「生成器の設定と、実際にコミットされている図」のズレを直接検出できる唯一の指標なので、
# 許容下限は FIGURE_DPI そのもの(丸め誤差ぶんだけ緩める)。
CHECK_DPI_MIN = FIGURE_DPI - 1

# CHECK_WIDTH: ライトボックス拡大に耐える最低ピクセル幅。
# DPI200 で参照済みの図の実測分布は 1280〜3380px(中央値 1779px)で、
# 最小の 1280px は figsize=(6.4, ...) の matplotlib 既定サイズに対応する。
# 「既定 figsize を DPI200 で描けば必ず超える」値として 1000px を下限に置く
# (= figsize 幅 5.0 インチ相当。これを下回る図は figsize か DPI の設定ミスを疑う)。
CHECK_WIDTH_MIN = 1000

# CHECK_INK: 「インク率」= 背景色(最頻色)と目に見えて違う画素の割合。
# セル分割の事故で軸も凡例も無い空の図が出た事例(DOCS-3)を恒久検出する。
# 参照済み図の実測分布は 2.38%〜(最小は logic-gan の生成グリッド)で、
# 空図は 0.5% 未満に落ちる。両者の間の 1.5% を失敗閾値とする。
CHECK_INK_MIN = 0.015

# 背景色と「違う」と見なすチャンネル差(8bit)。JPEG 由来のノイズは無いので小さくてよい。
_INK_TOLERANCE = 12

# Markdown の画像参照。nbconvert 由来は `![png](...)` だが、手書きページの図も拾う。
_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
# ギャラリー(index.mdx)の `import thumbXxx from './slug_files/....png';`
# (en 側は `../../tutorials/...` で ja の実体を指すので、相対パス全般を拾う)
_IMPORT_RE = re.compile(r"^import\s+(\w+)\s+from\s+'(\.[^']+)';", re.MULTILINE)


@dataclass
class Report:
    """検査結果。`failures` が空なら成功。"""

    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def fail(self, check: str, message: str) -> None:
        """失敗を1件記録する。"""
        self.failures.append(f"[{check}] {message}")

    def warn(self, check: str, message: str) -> None:
        """警告を1件記録する(終了コードには影響しない)。"""
        self.warnings.append(f"[{check}] {message}")


def _rel(path: Path) -> str:
    """リポジトリルートからの相対パス文字列。"""
    return str(path.relative_to(ROOT))


def collect_references() -> dict[Path, list[str]]:
    """ja/en の全チュートリアルページから画像参照を集める。

    Returns:
        解決済み絶対パス → それを参照しているページ(相対パス)の一覧。
        解決できなかった参照は `None` キーではなく、呼び出し側で別途検査する。
    """
    refs: dict[Path, list[str]] = {}
    for page_dir in (JA_TUTORIALS, EN_TUTORIALS):
        if not page_dir.is_dir():
            continue
        for page in sorted(page_dir.glob("*.md")) + sorted(page_dir.glob("*.mdx")):
            text = page.read_text(encoding="utf-8")
            for target in _IMG_RE.findall(text):
                if target.startswith(("http://", "https://", "data:", "/")):
                    continue
                resolved = (page.parent / target).resolve()
                refs.setdefault(resolved, []).append(_rel(page))
    return refs


def check_refs(report: Report) -> dict[Path, list[str]]:
    """ページから参照される図が実在するかを検査し、参照表を返す。"""
    refs = collect_references()
    missing = 0
    for target, pages in sorted(refs.items()):
        if not target.exists():
            missing += 1
            for page in pages:
                report.fail("refs", f"{page}: 参照先の図が存在しません → {target.name}")
    report.stats["referenced_images"] = len(refs)
    report.stats["missing_references"] = missing
    return refs


def _image_metrics(path: Path) -> tuple[int, int, float, float]:
    """(幅, 高さ, dpi, インク率) を返す。"""
    from PIL import Image

    with Image.open(path) as im:
        dpi = float(im.info.get("dpi", (0.0, 0.0))[0])
        rgb = im.convert("RGBA")
        # 透過は白背景に合成する(サイトの図はライト/ダーク双方で白地に載る)
        from PIL import Image as _Image

        flat = _Image.new("RGB", rgb.size, (255, 255, 255))
        flat.paste(rgb, mask=rgb.split()[3])
        width, height = flat.size
        pixels = list(flat.getdata())

    background = Counter(pixels).most_common(1)[0][0]
    ink = sum(
        1
        for p in pixels
        if max(abs(p[0] - background[0]), abs(p[1] - background[1]), abs(p[2] - background[2]))
        > _INK_TOLERANCE
    )
    return width, height, dpi, ink / max(len(pixels), 1)


def check_images(report: Report, refs: dict[Path, list[str]]) -> None:
    """参照されている図の解像度・DPI・インク率を検査する。"""
    widths: list[int] = []
    inks: list[float] = []
    for target in sorted(p for p in refs if p.exists() and p.suffix.lower() == ".png"):
        width, height, dpi, ink = _image_metrics(target)
        widths.append(width)
        inks.append(ink)
        name = _rel(target)
        if dpi < CHECK_DPI_MIN:
            report.fail(
                "resolution",
                f"{name}: {dpi:.0f}dpi(下限 {CHECK_DPI_MIN}dpi)。"
                f"build_tutorials.py の FIGURE_DPI={FIGURE_DPI} で再生成してください",
            )
        if width < CHECK_WIDTH_MIN:
            report.fail(
                "resolution",
                f"{name}: 幅 {width}x{height}px(下限 {CHECK_WIDTH_MIN}px)。"
                "figsize が小さすぎるか DPI が効いていません",
            )
        if ink < CHECK_INK_MIN:
            report.fail(
                "ink",
                f"{name}: インク率 {ink:.3%}(下限 {CHECK_INK_MIN:.1%})。"
                "軸も凡例も無い空図の可能性があります"
                "(図の描画ブロック途中のトップレベルコメントでセルが割れていないか確認)",
            )
    report.stats["checked_images"] = len(widths)
    if widths:
        report.stats["width_min"] = min(widths)
        report.stats["width_max"] = max(widths)
        report.stats["ink_min"] = round(min(inks), 5)


def check_gallery(report: Report) -> set[Path]:
    """例ギャラリー(ja/en)が全チュートリアルのサムネイルを解決できるかを検査する。

    Returns:
        ギャラリーが参照しているサムネイルの絶対パス集合(孤児判定に使う)。
    """
    if not GALLERY.exists():
        report.fail("gallery", f"{_rel(GALLERY)} がありません")
        return set()
    thumbs: set[Path] = set()

    # ja/en 両方のギャラリーで import の実体が存在すること
    # (サムネは Markdown の `![]()` ではなく astro:assets の import で出す。
    #  素の相対パス参照だと同一内容 PNG が Astro の重複排除で畳まれ、別の図が出る)
    for gallery in (GALLERY, EN_TUTORIALS / "index.mdx"):
        if not gallery.exists():
            continue
        for _, target in _IMPORT_RE.findall(gallery.read_text(encoding="utf-8")):
            resolved = (gallery.parent / target).resolve()
            thumbs.add(resolved)
            if not resolved.exists():
                report.fail(
                    "gallery", f"{_rel(gallery)}: サムネイルの実体がありません → {target}"
                )

    text = GALLERY.read_text(encoding="utf-8")
    imported = {name: target for name, target in _IMPORT_RE.findall(text)}
    report.stats["gallery_thumbs"] = len(imported)

    for spec in TUTORIALS:
        thumb = _thumb_for(spec)
        if thumb is None:
            report.fail(
                "gallery",
                f"{spec.slug}: サムネイルを解決できません"
                f"({spec.slug}_files/ に png が無いか、thumb={spec.thumb!r} が不正)",
            )
            continue
        rel = f"./{spec.slug}_files/{thumb}"
        if rel not in imported.values():
            report.fail(
                "gallery",
                f"{spec.slug}: 解決結果 {rel} が index.mdx の import にありません"
                "(build_tutorials.py --gallery-only で再生成してください)",
            )
    return thumbs


def check_orphans(
    report: Report, refs: dict[Path, list[str]], thumbs: set[Path], *, strict: bool = True
) -> None:
    """どのページからも(本文・ギャラリーのどちらからも)参照されない図を検出する。

    DOCS-5 の棚卸しで旧 DPI100 の残骸 5 枚を削除して孤児ゼロにしたので、既定で失敗扱いにする
    (孤児は「図番号がズレた」「ページ側の参照を消し忘れた」のシグナルで、放置すると
    古い設定の図が混ざったまま気づけなくなる)。図の一時的な退避などで許容したい場合だけ
    `strict=False`(CLI では `--allow-orphans`)にする。
    """
    referenced = {p.resolve() for p in refs} | thumbs
    orphans = [
        p
        for p in sorted(JA_TUTORIALS.glob("*_files/*"))
        if p.is_file() and p.resolve() not in referenced
    ]
    report.stats["orphans"] = len(orphans)
    record = report.fail if strict else report.warn
    for path in orphans:
        record(
            "orphan",
            f"{_rel(path)}: どのページからも参照されていません"
            "(図番号がズレた際の残骸。ページ側の参照漏れを直すか、不要なら削除してください)",
        )


def main() -> int:
    """検査を実行して結果を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="機械可読な出力")
    parser.add_argument(
        "--allow-orphans",
        action="store_true",
        help="孤児ファイルを失敗でなく警告に落とす(既定は失敗)",
    )
    args = parser.parse_args()

    report = Report()
    refs = check_refs(report)
    check_images(report, refs)
    thumbs = check_gallery(report)
    check_orphans(report, refs, thumbs, strict=not args.allow_orphans)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not report.failures,
                    "failures": report.failures,
                    "warnings": report.warnings,
                    "stats": report.stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if report.failures else 0

    print("docs アセット検査(DOCS-5)")
    print(
        f"  参照された図 {report.stats.get('referenced_images', 0)} 件 / "
        f"検査した PNG {report.stats.get('checked_images', 0)} 枚 / "
        f"ギャラリーのサムネ {report.stats.get('gallery_thumbs', 0)} 枚"
    )
    if "width_min" in report.stats:
        print(
            f"  幅 {report.stats['width_min']}〜{report.stats['width_max']}px"
            f"(下限 {CHECK_WIDTH_MIN}px) / "
            f"インク率 最小 {float(report.stats['ink_min']):.2%}"
            f"(下限 {CHECK_INK_MIN:.1%}) / DPI 下限 {CHECK_DPI_MIN}"
        )
    for message in report.warnings:
        print(f"  WARN  {message}")
    for message in report.failures:
        print(f"  FAIL  {message}")
    if report.failures:
        print(f"\nNG: {len(report.failures)} 件の問題があります")
        return 1
    print("\nOK: すべての図が解像度・インク率・参照の検査を通過しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
