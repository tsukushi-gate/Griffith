# 較正アンカーの再走結果

`run_calibration.py --device cuda`（RTX 4070、torch 2.14.0+cu126、wandb 有効）で 7 レシピ × seed 0/1 を回した結果。
判定は `aggregate.py` が `anchors.json` の原値と許容幅だけを見て行う（`calibration_table.md` / `.json`）。

- `cwo_*_s<seed>.json`: run ごとの結果（`hard_acc` / `best_hard_acc`、出力先、library commit）
- `calibration_table.md` / `.json`: 腕ごとの原値・再走値・差・判定（C3 全件合格、C4 全件合格）
- `wandb_runs.json`: run 名 → wandb run ID（project magicdelta-logic、group dlgn-ffn、job_type train、tag calibration）
- `cwo_mnist_48k_p6_s1.ir_verification.json` / `cwo_mnist_96k_p6_s0.ir_verification.json`: 実機アンカーの入力 IR の検証（全層が rot の閉形式に一致）

同一マシンで 2 回走らせ（wandb 無効・有効）、14 run すべて同値だった。実機側（DE0-CV の fit と ISSP）の結果は
mdl-fpga の `tools/mdl-fpga/examples/overlay_mnist/results/`（feature/mdl-fpga-overlay）。
