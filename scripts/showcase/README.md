# scripts/showcase

NEAT で進化させた論理ゲート回路(DLGN)のデモ動画を、docs の Showcase ページ用に選んで圧縮する。
動画そのものはリポジトリに入れず、GitHub Release `showcase-videos-v1`(prerelease)の asset に置く。
対応表(元ファイル・長さ・サイズ・URL)は [`docs/reference/showcase-videos.md`](https://docs.griffith.magicdelta.jp/reference/showcase-videos/)。

## 作り方

元動画は実験ブランチ `experiment/neat-demo` の `results/neatdemo/videos/` にある(公開リポジトリには無い)。
ブランチをチェックアウトせず、`git show` で 1 本ずつ取り出す。ffmpeg / ffprobe が要る(macOS は `brew install ffmpeg`)。

```bash
uv run --no-sync python scripts/showcase/make_showcase_videos.py \
    --branch experiment/neat-demo --out-dir /tmp/showcase-videos
```

- `<out-dir>/src/` に元動画、`<out-dir>/out/` に `<name>.mp4`・`<name>.jpg`(ポスター)・`manifest.json`・`preview.html`
- 既に出力がある動画は飛ばす。作り直すときは `--force`、一部だけなら `--only mario_dlgn_clear ...`
- 形式: H.264(libx264・yuv420p・`+faststart`・音声なし)、長辺720px以下、30fps以下、1本10MB以下
  (超えたら CRF を 26→30→34→38 と上げる)。60秒を超える動画は最初の60秒に切る
  (`keep_end=True` の動画だけ最後の60秒)。ポスターは3秒目(短い動画は真ん中)のフレーム
- 元ファイルの選び方とその理由は、スクリプトの `SHOWCASE_VIDEOS` の `note` に書く(manifest と対応表にも出る)

## 契約の名前(Release の asset 名。docs がこの名前で URL を書くので変えない)

`mario_dlgn_clear` `mario_dlgn_progress` `bugwalker_dlgn` `bugwalker_nn` `swimmer_dlgn` `swimmer_nn`
`snake_dlgn` `snake_dlgn_timelapse` `car2d_dlgn` `halfcheetah_dlgn_gallop` `halfcheetah_dlgn_run`
`bipedal_dlgn` `ant_dlgn_champion` `hopper_dlgn` `humanoid_dlgn` `reacher_dlgn` `arm2d_dlgn`
`flappy_dlgn` `cartpole_dlgn` `lunarlander_dlgn` `lunarlander_nn`(それぞれ `.mp4` と `.jpg`)

URL: `https://github.com/tsukushi-gate/Griffith/releases/download/showcase-videos-v1/<name>.mp4`

## Release へ上げる(人が確認した後)

Release へ上げるのは、人が `<out-dir>/out/preview.html` をブラウザで開いて全本を確かめた後。
ソフトウェアのリリースではないので `--prerelease` を付ける。notes は `release-notes.md` を使う。

```bash
OUT=/tmp/showcase-videos/out; gh release create showcase-videos-v1 --repo tsukushi-gate/Griffith --target main --prerelease --title "Showcase videos v1" --notes-file scripts/showcase/release-notes.md && gh release upload showcase-videos-v1 --repo tsukushi-gate/Griffith "$OUT"/*.mp4 "$OUT"/*.jpg
```

上げた後の確認:

```bash
gh release view showcase-videos-v1 --repo tsukushi-gate/Griffith --json assets --jq '.assets | length'   # 42
curl -sI https://github.com/tsukushi-gate/Griffith/releases/download/showcase-videos-v1/snake_dlgn.mp4 | head -1   # 302
```

同じ名前の動画を作り直して差し替えるときだけ、`gh release upload ... --clobber` を使う。
