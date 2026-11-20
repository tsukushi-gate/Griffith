# 32_deploy_web — デプロイキットの Web(ブラウザ)デモ

配布バンドルを `griffith deploy codegen-web` でブラウザ向けの配布物にし、**自前の最小ページ**
(`index.html` + `main.js`)から `mdl_web.js` の `loadBundle` を呼んで推論する例。
同梱サンプルの予測クラス・正解率・µs/推論を表示し、engine(auto / js / wasm)を切り替えられる。
ネットワーク不要(XOR を学習する)。詳しい手順は公開ガイド「Web(ブラウザ)に載せる」(`/guides/deploy-web/`)。

## ファイル

| ファイル | 役割 |
|---|---|
| `gen.py` | XOR → バンドル → `codegen-web` → ページを重ねる → `verify-web`(js/wasm 両方) |
| `index.html` | 最小のページ(engine セレクタ + 結果表) |
| `main.js` | `loadBundle` を呼んでサンプルを推論・計測する本体(日本語コメント付き) |

出力は `out_examples/deploy_web/`(gitignore 済み)に出る。

## 手順

```bash
make web-runtime                                  # 初回のみ: runtime/web/dist をビルド(要 node/npm)
uv run python examples/32_deploy_web/gen.py       # 生成 + verify-web
python -m http.server 8080 -d out_examples/deploy_web
# → http://localhost:8080/ を開く(COOP/COEP ヘッダは不要)
```

別のバンドルを載せるときは `--bundle <x.bundle.json> --samples <samples.json>`。
`samples.json` は `{"input_width": N, "samples": [{"bits": [0,1,...], "label": int|null}]}`。
`?model=<url>` / `?samples=<url>` / `?engine=js` でブラウザ側から差し替えもできる。
