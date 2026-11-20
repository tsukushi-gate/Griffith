# THIRD-PARTY / 出典

## 取り込みコード

| ファイル | 出典 | 備考 |
|---|---|---|
| `mdl_fpga/hdl/uart_rx.v` | 社内先行実験 `demo/hdl/uart_rx.v` | DE0-CV 実機で MNIST 1000/1000 bit 一致を実証済みの物理層。ロジック同一、ヘッダのみ更新 |
| `mdl_fpga/hdl/uart_tx.v` | 社内先行実験 `demo/hdl/uart_tx.v` | 同上 |
| `mdl_fpga/hdl/frame_rx.v` | 社内先行実験 `demo/hdl/frame_rx.v` を原型に PAYLOAD_BYTES で汎用化 | FSM 構造・再同期規約は原型と同一 |
| `mdl_fpga/hdl/frame_tx.v` | 社内先行実験 `demo/hdl/frame_tx.v` を原型に汎用化 | pred バイトを廃し任意幅ペイロードに |
| `mdl_fpga/boards/de0-cv.json` のピン表 | 社内先行実験 `demo/fpga/pins_de0cv.md` | golden `platform_setup.tcl` + Terasic DE0-CV マニュアル Table 3-7 の 2 ソース突合済 |
| `mdl_fpga/boards/de10-lite.json` のピン表 | litex-boards(BSD-2-Clause)+ 実プロジェクト qsf の機械照合 | 出典・commit・照合方法は `mdl_fpga/boards/NOTICE` が正 |
| `mdl_fpga/boards/tang-nano-20k.json` のピン表 | litex-boards(BSD-2-Clause)+ sipeed/TangNano-20K-example 公式 .cst の機械照合 | 出典・commit・照合方法は `mdl_fpga/boards/NOTICE` が正。生成スクリプトとソースコピーは `references/tang-nano-20k/` |
| 実測則(advisor の上限値) | 社内先行実験のFPGA サイジング調査 R1-R14 | DE0-CV 級のみ有効(大型機への外挿は棄却済み) |

社内先行実験は本プロジェクトの調査リポジトリ(未公開)。OSS 公開時は上記の実証記録を
本リポジトリへ移すか、出典注記を維持すること。

## 外部ツール(同梱しない)

| ツール | 用途 | ライセンス上の扱い |
|---|---|---|
| oss-cad-suite(iverilog / vvp / Verilator) | RTL シミュレーション | OSS。ユーザーが導入(doctor が検出・誘導) |
| Intel Quartus Prime Lite | 合成・書込 | **EULA により同梱不可**。doctor が検出・誘導。書込のみなら Programmer 単体版(約4GB)で可 |

## エディタ(tools/editor)からのUI基盤コピー(UI統一契約)

| 対象 | 出典 | 備考 |
|---|---|---|
| `frontend/src/index.css` の `:root`/`@theme inline`/`@layer base` | `tools/editor/frontend/src/index.css` | デザイントークン(名前=統合時の契約)。`tokens.drift.test.ts` が「エディタ ⊆ ビューア」を監視 |
| `frontend/src/index.css` の `.dark` ブロック | ビューア新規較正 | エディタは現状ライト固定。同名トークンのみで構成=将来エディタへ逆移植可能 |
| `frontend/src/components/ui/*`(11部品) | `tools/editor/frontend/src/components/ui/*` | shadcn new-york。sonner のみテーマ追従へ適応(ヘッダに明記) |
| `frontend/src/lib/utils.ts` / `components.json` | 同名ファイル | cn ヘルパ / shadcn 設定 |
| i18n 方式(`frontend/src/i18n/`) | `tools/editor/frontend/src/i18n/` の方式踏襲 | フラット辞書+useT(ライブラリ非依存) |
| レイアウト規約(AppHeader h-12 等)・`.editor-node` 視覚契約 | エディタ `layout/`・`styles.css` | 構造・クラス仕様を同型化(コードは再実装) |

## エディタ(tools/editor)からの方式移植(コードコピーではなくパターン踏襲)

| 対象 | 出典パターン |
|---|---|
| `frontend/src/graph/autoLayout.ts` | `tools/editor/frontend/src/ir/autoLayout.ts` の適応コピー(elk layered 設定同一、型依存を汎用化) |
| `frontend/src/graph/edgeStyle.ts` | `tools/editor/frontend/src/editor/edgeStyle.ts` の純関数部(バス幅→線幅)移植 |
| `viewer/renderers.py` のサイドカー読込 | `tools/editor/backend/python_importer.py` の spec_from_file_location+ユニーク名+sys.path ガード方式 |
| `frontend/vite.config.ts` の WS Cookie 除去 | editor の vite proxy 実証パターン(Cookie 肥大 400 対策) |

## ライブラリ(griffith-logic 本体)からの流用

`ir.serialize` / `ir.validate` / `ir.simulate`(正解オラクル)/ `export.lowering` /
`export.estimate` / `export.verilog`(相互検証リファレンス)。再実装はしない(構想書 §3.1 の契約)。
