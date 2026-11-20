# griffith-logic Webランタイム

論理ゲートNN(DLGN)の配布バンドル(`magicdelta-bundle`)をブラウザ/Node.jsで
実行する軽量ランタイム。C/CUDA/Metal(`runtime/c` `runtime/cuda` `runtime/metal`)が
「ソース生成」なのに対し、Webは実行体を最初から持っている
(`tools/webbench` がビルドする TypeScript製のWASMエンジン)。

## ビルド

```bash
make web-runtime
# 実体: npm --prefix tools/webbench run build:runtime
#     = tsc --noEmit && asbuild:release && asbuild:simd && asbuild:mt && vite build -c vite.lib.config.ts
```

ソースは `tools/webbench/src/deploy/`(このディレクトリ**ではない**)。
`vite.lib.config.ts` がビルドしてここ(`runtime/web/dist/`)へ書き出す。
`dist/` はビルド成果物のため git 管理しない(`.gitignore` 参照、都度再生成する)。

## dist レイアウト契約

```
runtime/web/dist/
  mdl_web.js         公開API(loadBundle等)。ESM、ブラウザ/Node両対応
  demo.js            単一ページデモの本体(index.htmlが読む)
  verify.mjs         node CLI(griffith deploy verify-web が呼ぶ出荷ゲート)
  index.html         静的デモシェル(src/deploy/demo.htmlのコピー)
  release.wasm       AssemblyScript製カーネル(scalar)
  release-simd.wasm  同(v128 SIMD、既定で優先される)
  lib-*.js, assets/  上記が参照する共有チャンク(rollupが自動生成、名前は変わりうる)
```

`mdl_web.js` / `demo.js` / `release*.wasm` / `index.html` の5点セットが
`griffith deploy codegen-web` の配布物(`{name}.bundle.json`)と一緒に配られる
契約。ファイル名は固定だが、共有チャンク(`lib-*.js` 等)のハッシュ付き
ファイル名はビルドのたびに変わるので、**`dist/` 配下は丸ごとコピーする**
こと(個別ファイル名を決め打ちしない)。`griffith deploy codegen-web` は実際に
`dist/` を再帰的にコピーしている(`griffith_logic/deploy/cli.py`)。

## `griffith deploy codegen-web` からの使われ方

```bash
uv run griffith deploy bundle model.json -o model.bundle.json
uv run griffith deploy codegen-web model.bundle.json -o out/
# out/ = model.bundle.json + dist/ 一式(mdl_web.js/demo.js/verify.mjs/
#        index.html/release*.wasm)
```

`codegen-web` は `runtime/web/dist` が無いと **先に `make web-runtime` を
実行するよう指示して失敗する**(古い/存在しないランタイムを黙って配らない)。

**dense の平坦化(`lut_flat`)。** `codegen-web` / `verify-web` は既定
(`--dense-flat auto`)で、dense の 2 入力 `lut_layer` 列を縮約済みの平坦ネット
リスト(`lut_flat`)へ置き換えてから配る(C の `codegen-c` と同じ変換。Web は
閾値 0 = 列があれば常に採用、`--dense-flat off` で従来の `lut_layer` 列)。
実行器は `tools/webbench/src/engines/lutFlat.ts`(js)と
`wasmLutFlat.ts` + `assembly/index*.ts::lutFlatGsAsm / lutFlatPackedGsAsm`(wasm)。
後段の `group_sum` と融合し、3 枚以上の `scores(bits, n)` は 32 枚/語の
バッチ軸パック経路を使う。HAR k6000(Node・M3、µs/推論): `lut_layer` 238 →
js 45 / wasm 29(1 枚)、224 → 11 / 6.1(32 枚)。

## 配信

`threads=1` が既定(`LoadBundleOptions.threads`)なので、SharedArrayBuffer用の
COOP/COEPヘッダは不要。**素の静的配信で動く**:

```bash
python -m http.server -d out/
```

ブラウザで `index.html` を開くと `./model.bundle.json` を fetch して
meta表・engine選択(auto/js/wasm)・demo_samplesの推論結果を表示する
(`?model=<url>` で読み込み先を差し替え可能)。

## 公開API(`mdl_web.js`)

```ts
import { loadBundle } from './mdl_web.js';

const bundle = await loadBundle(bundleJson, { engine: 'auto' });
// bundle.kind: 'seg' | 'detection' | 'classify-image' | 'dense'
// bundle.scores(bits, batch)            全kind共通
// bundle.scoresRgba(rgba)               画像入力のkindのみ
// bundle.segLabels(rgba)                kind==='seg'のみ
// bundle.detect(rgba)                   kind==='detection'のみ
// bundle.notes: string[]                黙殺しない注記(programの出所・
//                                        エンジンフォールバック・バージョン警告)
```

`engine: 'js'` は `bitplane`(純TS)、`'wasm'` は `bitplane-wasm`
(AssemblyScript製カーネル)。`'auto'`(既定)は wasm 初期化を試み、失敗したら
`notes` に理由を記録して `'js'` へ落ちる。出力整数は両エンジンで完全一致
(`loadBundle.test.ts` で固定)。

## `verify.mjs` の契約(`griffith deploy verify-web` が使う出荷ゲート)

```bash
node dist/verify.mjs <bundle.json> <cases.json> [--engine js|wasm] [--wasm <path>]
```

- `--wasm <path>` を省略すると `verify.mjs` と同じディレクトリの
  `release-simd.wasm` を既定にする(`dist/` 配布物では常に隣にコピーされる
  ので、`--engine wasm` だけで通常は動く。`griffith deploy verify-web` もこの
  既定に依存し `--wasm` は渡さない)。
- `cases.json`: `{"mode": "bits"|"image", "cases": [{"bits_b64"?, "image_b64"?, "expected": number[]}]}`
  - `bits_b64` = 入力幅(`in_width`)ビットの **packed LSB-first** バイト列(base64)
  - `image_b64` = 生 `uint8` **CHW**(`image_channels×H×W`)の base64
- stdout: 1行JSON `{"ok","engine","mode","outputs","maxAbsDiff","error"?}`。
  `ok` は**整数完全一致**のときのみ true(exit 0)。不一致は最初のケース/
  出力インデックス/expected/actualを `error` に積んで exit 1。
- `mode=bits` でケースが 2 件以上あれば、全ケースを **1 回のバッチ**
  (`scores(bits, n)`)でも実行して照合する(3 枚以上で切り替わる
  `lut_flat` のバッチ軸パック経路を出荷ゲートに乗せるため)。
- 検証経路は常に `strictPrograms=true` で走る: バンドルの `programs`
  節が壊れていればフォールバックせず即エラーにする(出荷ゲートなので、
  黙って再構築すると古い/改竄された programs を検出できなくなるため)。

## 詳細

- `docs/reference/deploy-bundle.md` — 配布バンドルの仕様
- `runtime/README.md` — 全ターゲット共通のランタイム一覧
- `tools/webbench/src/deploy/` — このランタイムのソース
