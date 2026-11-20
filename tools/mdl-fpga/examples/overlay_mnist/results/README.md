# 計算型配線 overlay の実機結果（DE0-CV）

`build_overlay_demo.py --n 1000 --v 8 --clock-div 2 --presim-vectors 2` で合成し、`cli.py flash` と `cli.py issp` で実機実行した記録。
入力 IR は griffith-logic の recipe（`configs/recipes/cwo_mnist_48k_p6.yaml` seed 1、`cwo_mnist_96k_p6.yaml` seed 0、自由配線は `cwo_mnist_48k_p0.yaml` seed 0）が書き出したもの。自由配線（保存型）は `--wiring stored`。画像 1,000 枚は 32 bit の記録では載らない（見積もり 336 > 308）ので N=512。

| ネット | ゲート ROM | 総 M10K | ALM | fit | 実機 |
|---|---|---|---|---|---|
| 48K（w8000×d6） | 32 blk | 127 blk | 1,766 | 585 s | 1000/1000、6,129 cyc/推論、4,079.0 inf/s @25 MHz |
| 96K（w16000×d6） | 48 blk | 159 blk | 1,857 | 585 s | 1000/1000、12,129 cyc/推論、2,061.2 inf/s @25 MHz |
| 48K 自由配線（保存型 32 bit、P0 seed 0、N=512） | 240 blk | 296 blk | 1,253 | 284 s | 512/512、6,129 cyc/推論、4,079.0 inf/s @25 MHz |

注: overlay に保存型（自由配線）モードを足したとき `overlay_params.json` に方式の情報（`WIRING` / `RECORD_BITS` / `GATE_ROM_BLOCKS` ほか）が加わった。同じ IR から出る RTL・`.mem`・qsf・sdc は不変なので上表の fit / 実機の値は変わらないが、いま再生成すると params の**キーが増える**（本ファイル群は再走時の現物）。

各ディレクトリ: `flow_summary.json`（段ごとの所要、ALM、M10K の entity 別内訳、slack、Fmax）、`overlay_params.json`（層ごとの配線と導出定数）、
`overlay_system.json`（probe 割付と期待サイクル）、`golden.json`（参照シムの予測と精度）、`issp_run_out.txt` / `issp_run.tcl`（実機の生ログと実行 Tcl）、`sta_fmax_excerpt.txt`。
fit / sta 報告の全文は容量のため置かない（生成手順は上のコマンド）。
