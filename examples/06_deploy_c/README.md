# 06_deploy_c — 学習済みモデル → C実行ファイル(デプロイキット)

XORモデルを学習し、IR → 配布バンドル → 自己完結したC99ソース → 実行ファイル、まで
通す最小サンプル。依存はC99標準ヘッダのみ(組み込み/モバイル/デスクトップで同じ
ソースが動く)。手順の解説は[デプロイキットクイックスタート](https://docs.griffith.magicdelta.jp/guides/deploy-quickstart/)、
対応環境は[対応プラットフォーム](https://docs.griffith.magicdelta.jp/reference/platform-support/)を参照。

## ファイル

| ファイル | 役割 |
|---|---|
| `export_model.py` | XOR学習 → IR抽出 → `out_examples/xor_model.json`(デプロイ用モデルJSON) |
| `main.c` | 生成された `model.h` の `mdl_infer()` を呼ぶだけの最小Cドライバ |

## 実行コマンド

```bash
# 1) 学習 + IR抽出(webbench形式のモデルJSONを書き出す)
uv run python examples/06_deploy_c/export_model.py

# 2) 配布バンドル化(既定で lossless なゲート削減=sweep が自動で効く: 32 → 22 ゲート)
uv run md deploy bundle out_examples/xor_model.json -o out_examples/xor.bundle.json

# 3) Cソース生成(ランタイムのコピーも同梱される)。既定で dense-flat auto:
#    実効ゲート 22 → 3(削減率 86%)で lut_flat を採用し、model.h に MDL_DENSE_FLAT 1 が入る
uv run md deploy codegen-c out_examples/xor.bundle.json -o out_examples/xor_c

# 4) ビルド + 実行
cc -std=c99 -O2 -I out_examples/xor_c \
   out_examples/xor_c/model.c out_examples/xor_c/mdl_kernels.c \
   examples/06_deploy_c/main.c -o out_examples/xor_c/xor_demo
./out_examples/xor_c/xor_demo

# 5) (任意)出荷前ゲート: 生成C(codegen-cと同じ判定)が元バンドルIR simulateと整数厳密一致することを確認
uv run md deploy verify-c out_examples/xor.bundle.json
```

## 期待結果

```
xor(0,0) = 0 (scores=8,1) expected=0 OK
xor(0,1) = 1 (scores=0,8) expected=1 OK
xor(1,0) = 1 (scores=0,8) expected=1 OK
xor(1,1) = 0 (scores=8,1) expected=0 OK
全4パターン一致
```

(圧縮を切って従来どおりの生成にするなら `bundle --compress none` と
`codegen-c --dense-flat off`。分類結果は変わらない。)

(スコアの具体値は学習の乱数シードに依存しないよう `torch.manual_seed(0)` で固定して
あるが、環境のPyTorchバージョンによって多少ずれる場合がある。argmaxの結果=分類が
全4パターンで一致していれば成功。)

## 次に読むもの

- CUDA / Metal / Web向けのコード生成は[デプロイキット クイックスタート](../../docs/guides/deploy-quickstart.md#c以外のターゲット)
- Cランタイムのマルチコア化・スレッド安全性・対応ノード種は[runtime/c/README.md](../../runtime/c/README.md)
- バンドル形式の全仕様は[配布バンドル仕様](https://docs.griffith.magicdelta.jp/reference/deploy-bundle/)
