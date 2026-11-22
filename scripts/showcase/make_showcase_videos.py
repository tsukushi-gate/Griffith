"""NEAT で進化させた回路のデモ動画を、公開ドキュメント(Showcase ページ)用に選んで圧縮する。

元動画は実験ブランチ(既定 `experiment/neat-demo`)の `results/neatdemo/videos/` にあり、
チェックアウトせずに `git show <branch>:<path>` で 1 本ずつ取り出す。出力名は docs から
GitHub Release の asset として参照される契約なので変えない(`SHOWCASE_VIDEOS` 参照)。

使い方(リポジトリ内のどこからでも。ffmpeg / ffprobe が要る):

    uv run --no-sync python scripts/showcase/make_showcase_videos.py \
        --branch experiment/neat-demo --out-dir /tmp/showcase-videos

出力: <out-dir>/src/ に元動画、<out-dir>/out/ に <name>.mp4・<name>.jpg・manifest.json・
preview.html(人が全本を確かめる 1 枚のページ)。
既に出力がある動画は飛ばす(`--force` で作り直し)。
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SOURCE_DIR_IN_BRANCH = "results/neatdemo/videos"
MAX_LONG_SIDE_PX = 720
MAX_FPS = 30
MAX_DURATION_S = 60.0
MAX_SIZE_BYTES = 10 * 1024 * 1024
POSTER_TIME_S = 3.0
# 10MB を超えたら CRF を上げて作り直す(画質を少しずつ下げる)
CRF_STEPS = (26, 30, 34, 38)


@dataclass(frozen=True)
class ShowcaseVideo:
    name: str  # 出力名(拡張子なし)。Release の asset 名でもある
    source: str  # SOURCE_DIR_IN_BRANCH からの相対パス
    note: str
    keep_end: bool = False  # True なら長いとき最後の 60 秒を残す(クリアの瞬間が最後にある動画)
    poster_time_s: float = POSTER_TIME_S


SHOWCASE_VIDEOS: tuple[ShowcaseVideo, ...] = (
    ShowcaseVideo(
        "mario_dlgn_clear",
        "mario_dlgn_s0_CLEAR.mp4",
        "1-1クリアの確定成果物として保存された動画(コミット b1989474)。元が10.5秒と短く、"
        "最後のフレームはコースの途中でゴールの旗は映っていない(要確認)",
        keep_end=True,
    ),
    ShowcaseVideo(
        "mario_dlgn_progress",
        "mario_s0_dlgn_progress.mp4",
        "世代ごとのベスト個体をつないだタイムラプス",
    ),
    ShowcaseVideo("bugwalker_dlgn", "bugwalker_dlgn.mp4", "2Dの虫の歩行(DLGN回路)"),
    ShowcaseVideo("bugwalker_nn", "bugwalker_nn.mp4", "2Dの虫の歩行(比較のNN)"),
    ShowcaseVideo("swimmer_dlgn", "swimmer_dlgn_v2_swim.mp4", "MuJoCo Swimmer(DLGN回路、v2)"),
    ShowcaseVideo("swimmer_nn", "swimmer_nn_v2_swim.mp4", "MuJoCo Swimmer(比較のNN、v2)"),
    ShowcaseVideo("snake_dlgn", "snake_dlgn.mp4", "スネークゲーム(DLGN回路)"),
    ShowcaseVideo(
        "snake_dlgn_timelapse", "snake_dlgn_timelapse.mp4", "スネークの世代ごとのタイムラプス"
    ),
    ShowcaseVideo("car2d_dlgn", "car2d_dlgn.mp4", "2Dの車でコースを走る(DLGN回路)"),
    ShowcaseVideo(
        "halfcheetah_dlgn_gallop",
        "halfcheetah_dlgn_v1_gallop.mp4",
        "MuJoCo HalfCheetah のギャロップ(DLGN回路、v1)",
    ),
    ShowcaseVideo(
        "halfcheetah_dlgn_run",
        "halfcheetah_dlgn_v2_run.mp4",
        "MuJoCo HalfCheetah の走り(DLGN回路、v2)",
    ),
    ShowcaseVideo("bipedal_dlgn", "bipedal_dlgn.mp4", "BipedalWalker(DLGN回路、seed0)"),
    ShowcaseVideo(
        "ant_dlgn_champion",
        "ant_f3_dlgn_s0_g250_champion.mp4",
        "MuJoCo Ant(DLGN回路、f3設定・seed0・250世代のチャンピオン)",
    ),
    ShowcaseVideo(
        "hopper_dlgn",
        "hopper_dlgn_latest.mp4",
        "MuJoCo Hopper(DLGN回路)。候補の hopper_dlgn_v2_stand は名前どおり立つだけなので、"
        "最新のチャンピオン(latest)を選んだ",
    ),
    ShowcaseVideo(
        "humanoid_dlgn",
        "humanoid_f3_dlgn_s0_g250_champion.mp4",
        "MuJoCo Humanoid(DLGN回路、f3設定・seed0・250世代のチャンピオン。"
        "Antと同じ条件の回を選んだ)",
    ),
    ShowcaseVideo("reacher_dlgn", "reacher_dlgn.mp4", "MuJoCo Reacher(DLGN回路)"),
    ShowcaseVideo("arm2d_dlgn", "arm2d_dlgn.mp4", "2Dのアームで目標に手先を届ける(DLGN回路)"),
    ShowcaseVideo("flappy_dlgn", "flappy_dlgn.mp4", "Flappy Bird(DLGN回路)"),
    ShowcaseVideo("cartpole_dlgn", "cartpole_dlgn.mp4", "CartPole(DLGN回路)"),
    ShowcaseVideo("lunarlander_dlgn", "lunarlander_dlgn.mp4", "LunarLander(DLGN回路)"),
    ShowcaseVideo("lunarlander_nn", "lunarlander_nn.mp4", "LunarLander(比較のNN)"),
)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def probe(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator or 1),
        "duration_s": float(data["format"]["duration"]),
    }


def extract_source(branch: str, source: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        result = subprocess.run(
            ["git", "show", f"{branch}:{SOURCE_DIR_IN_BRANCH}/{source}"],
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        print(f"[skip] 元ファイルが無い: {source} ({result.stderr.decode().strip()})")
        return False
    return True


def scale_filter(fps: float) -> str:
    # 長辺を 720px 以下に(拡大はしない)、偶数に丸める
    scale = (
        f"scale='if(gte(iw,ih),min(iw,{MAX_LONG_SIDE_PX}),-2)':"
        f"'if(gte(iw,ih),-2,min(ih,{MAX_LONG_SIDE_PX}))'"
    )
    filters = [scale, "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if fps > MAX_FPS:
        filters.append(f"fps={MAX_FPS}")
    return ",".join(filters)


def encode(video: ShowcaseVideo, source_path: Path, output_path: Path) -> None:
    info = probe(source_path)
    trim_args: list[str] = []
    if info["duration_s"] > MAX_DURATION_S:
        start = info["duration_s"] - MAX_DURATION_S if video.keep_end else 0.0
        trim_args = ["-ss", f"{start:.3f}", "-t", f"{MAX_DURATION_S:.3f}"]
    for crf in CRF_STEPS:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *trim_args,
                "-i",
                str(source_path),
                "-an",
                "-vf",
                scale_filter(info["fps"]),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
        )
        if output_path.stat().st_size <= MAX_SIZE_BYTES:
            return
        print(
            f"[retry] {video.name}: CRF {crf} で {output_path.stat().st_size} バイト。CRF を上げる"
        )
    raise RuntimeError(f"{video.name} を {MAX_SIZE_BYTES} バイト以下にできなかった")


def make_poster(video: ShowcaseVideo, output_path: Path, poster_path: Path) -> None:
    duration = probe(output_path)["duration_s"]
    poster_time = video.poster_time_s if duration > video.poster_time_s * 2 else duration / 2
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{poster_time:.3f}",
            "-i",
            str(output_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(poster_path),
        ],
    )


PREVIEW_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f6f7f9; --card: #fff; --text: #1b1f24; --muted: #5b6470; --border: #d9dde3;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111418; --card: #1a1f25; --text: #e6e9ee; --muted: #9aa4b1; --border: #2c333c; }
}
body {
  margin: 0; padding: 24px 16px; background: var(--bg); color: var(--text);
  font-family: system-ui, sans-serif;
}
header { max-width: 1400px; margin: 0 auto 16px; }
h1 { margin: 0 0 4px; font-size: 1.4rem; }
.summary { color: var(--muted); }
.grid {
  max-width: 1400px; margin: 0 auto; display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
}
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
}
video { width: 100%; border-radius: 4px; background: #000; }
h2 { font-size: 1rem; margin: 8px 0 4px; word-break: break-all; }
dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; margin: 0; font-size: 0.85rem; }
dt { color: var(--muted); }
dd { margin: 0; word-break: break-all; }
"""
BYTES_PER_MEGABYTE = 1024 * 1024


def preview_card(entry: dict) -> str:
    name = html.escape(entry["name"])
    poster = html.escape(entry["poster"])
    size_mb = entry["size_bytes"] / BYTES_PER_MEGABYTE
    return (
        '<article class="card">\n'
        f'  <video controls preload="metadata" muted loop playsinline poster="{poster}" '
        f'src="{name}"></video>\n'
        f"  <h2>{name}</h2>\n"
        "  <dl>\n"
        f"    <dt>元ファイル</dt><dd><code>{html.escape(entry['source'])}</code></dd>\n"
        f"    <dt>長さ</dt><dd>{entry['duration_s']:.2f} 秒"
        f"(元 {entry['source_duration_s']:.2f} 秒)</dd>\n"
        f"    <dt>サイズ</dt><dd>{size_mb:.2f} MB・{entry['width']}×{entry['height']}</dd>\n"
        f"    <dt>メモ</dt><dd>{html.escape(entry['note'])}</dd>\n"
        "  </dl>\n"
        "</article>"
    )


def write_preview(manifest: list[dict], output_dir: Path) -> Path:
    """人が全本をブラウザ 1 枚で確かめるための preview.html(外部ライブラリなし)。"""
    total_bytes = sum(entry["size_bytes"] for entry in manifest)
    total_mb = total_bytes / BYTES_PER_MEGABYTE
    cards = "\n".join(preview_card(entry) for entry in manifest)
    page = (
        '<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Showcase videos preview</title>\n<style>{PREVIEW_CSS}</style>\n</head>\n"
        "<body>\n<header>\n  <h1>Showcase videos preview</h1>\n"
        f'  <p class="summary">{len(manifest)} 本・合計 {total_mb:.1f} MB'
        f"({total_bytes:,} バイト)</p>\n</header>\n"
        f'<main class="grid">\n{cards}\n</main>\n</body>\n</html>\n'
    )
    preview_path = output_dir / "preview.html"
    preview_path.write_text(page, encoding="utf-8")
    return preview_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--branch", default="experiment/neat-demo", help="元動画があるブランチ")
    parser.add_argument("--out-dir", type=Path, required=True, help="出力先(src/ と out/ を作る)")
    parser.add_argument("--force", action="store_true", help="出力があっても作り直す")
    parser.add_argument("--only", nargs="*", help="この名前だけ作る")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe", "git"):
        if shutil.which(tool) is None:
            print(f"{tool} が見つからない(macOS は `brew install ffmpeg`)", file=sys.stderr)
            return 1

    source_dir = args.out_dir / "src"
    output_dir = args.out_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    missing: list[str] = []
    for video in SHOWCASE_VIDEOS:
        if args.only and video.name not in args.only:
            continue
        source_path = source_dir / video.source
        output_path = output_dir / f"{video.name}.mp4"
        poster_path = output_dir / f"{video.name}.jpg"
        if not source_path.exists() and not extract_source(args.branch, video.source, source_path):
            missing.append(video.name)
            continue
        if args.force or not output_path.exists():
            print(f"[encode] {video.source} -> {output_path.name}")
            encode(video, source_path, output_path)
        if args.force or not poster_path.exists():
            make_poster(video, output_path, poster_path)
        info = probe(output_path)
        source_info = probe(source_path)
        manifest.append(
            {
                "name": f"{video.name}.mp4",
                "poster": f"{video.name}.jpg",
                "source": f"{args.branch}:{SOURCE_DIR_IN_BRANCH}/{video.source}",
                "source_duration_s": round(source_info["duration_s"], 2),
                "duration_s": round(info["duration_s"], 2),
                "size_bytes": output_path.stat().st_size,
                "width": info["width"],
                "height": info["height"],
                "note": video.note,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    preview_path = write_preview(manifest, output_dir)
    total = sum(entry["size_bytes"] for entry in manifest)
    print(f"[done] {len(manifest)} 本、合計 {total} バイト -> {manifest_path}")
    print(f"[preview] {preview_path}")
    if missing:
        print(f"[missing] {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
