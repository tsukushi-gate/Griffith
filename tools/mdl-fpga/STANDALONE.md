# STANDALONE — 配布用スタンドアローンアプリ(設計+実装TODO)

FPGA実装後の利用体験を、mdl-fpga の
ローカルホストビューア(開発用)から切り離し、**bundle zip を移動した先で
「Python 導入+pip 1コマンド」だけで動く独立アプリ**にする。

## 決定事項

| 論点 | 決定 |
|---|---|
| 依存 | ランタイムは griffith-logic / mdl-fpga **非依存**(import-lint テストで機械保証)。コア依存= pyserial のみ(numpy/pygame はテンプレ側) |
| 書込 | **必須機能**。flasher(openFPGALoader、Apache-2.0)は**既定同梱・オプトアウト式**(`--no-flasher`)。同梱バイナリは自前 GitHub Releases の「flasher パック」(3 OS 分+LICENSE/NOTICE)から bundle 時に取得・キャッシュ(=1案) |
| ベンダEDA | **EULA 上再配布不可のため同梱しない**(誘導のみ。DE0-CV 系は Quartus Programmer 単体版を案内)。OSS 公開時のライセンス監査観点で明文化 |
| I/O契約 | `io_contract.json` を共通ハブに。生成源 (a)=bundle が IR/manifest/board から自動生成(生成回路はゼロ手間)、(b)=ユーザ定義 Verilog の `(* mdl_* *)` 属性から抽出(将来)。**フレーミングは共通 UART ラッパ準拠に限定**(契約が保証するのはペイロードの意味づけまで) |
| UI | **CLI が第一級**。`serve-stdio`(stdin コマンド/stdout JSON 行)が言語非依存の公式 API。pygame はテンプレの一形態(他フレームワークは CLI 経由 or Python ライブラリ直 import) |
| 置き場所 | ソースの正= `tools/mdl-fpga/runtime/mdl_runtime/`。配布= bundle zip へ **vendor 同梱**(pip 配布はしない) |
| 環境構築 | **bundle README に手順を明記**: Python 導入 → `pip install -r app/requirements.txt` → 実行例 |
| 社内先行実験/demo | 参考実証(30fps ループ・前処理・半二重+resync の教訓)のみ。コードは本プロジェクト品質で新規 |

## アーキテクチャ

```
tools/mdl-fpga/runtime/mdl_runtime/     ← ソースの正(mdl非依存)
  contract.py   io_contract.json の読込・検証
  protocol.py   UART フレーミング(下記仕様の複製。バイト一致テストで固定)
  ports.py      契約内 USB 署名でのポート自動検出(board DB 不要)
  link.py       送受信ループ(タイムアウト+SOF resync、serial 注入可)
  flash.py      flasher 解決(同梱 flasher/ → PATH → 誘導)+書込実行
  selftest.py   同梱 golden ベクトルの実機再生= bit 一致検証
  cli.py        ports/flash/send/run/selftest/serve-stdio/gui(--json)
  templates/    pygame テンプレ(canvas_app: 手描き/ビットトグル+出力表示)

bundle zip 構成(app 付き):
  bitstream/…  board/…  manifest.json  flow_summary.json  README.md(環境構築込み)
  app/io_contract.json  app/goldens.json  app/requirements.txt  app/run.py
  app/mdl_runtime/…(vendor)   flasher/<os>/…(オプトアウト可)
```

## UART フレーミング(唯一の正= hdl/frame_rx.v。契約はこれを参照)

- 要求: `[0xA5][seq][payload = in_width bit を ceil(w/8) バイト little-endian][checksum = (seq+payload) & 0xFF]`
- 応答: `[0x5A][seq エコー][payload = out_width 同上][checksum]`、受信は SOF 探索で resync
- runtime/protocol.py は mdl_fpga/hw_check.py とバイト一致することをテストで固定する

## io_contract.json v1(bundle が自動生成)

name / inputs・outputs(width, semantics, k)/ in_width・out_width /
uart(baud, framing=mdl-frame-v1, req/rsp payload bytes)/
serial.usb(board 由来の署名+既定 CP210x)/
flash(bitstream 相対パス, tool, args=-b ofl_board 等, driver_note)/
board(key, name)/ goldens(n, file)。
semantics(bitmap28x28 / class_scores 等)はテンプレの表示切替ヒント。

## ユーザ定義 Verilog の注釈規約(将来、生成源 b)

```verilog
(* mdl_role="clock" *)   input clk,
(* mdl_role="uart_rx" *) input uart_rx,   // uart_tx も同様
(* mdl_input="x", mdl_semantics="bitmap28x28" *)  input [783:0] x_in,
(* mdl_output="y", mdl_semantics="class_scores", mdl_k=10 *) output [89:0] y
```

- 幅はポート宣言から取得(二重記載しない)。パーサはポート+属性のみの軽量解析
- 制約: フレーミングは提供する共通 UART ラッパ準拠のみサポート
- hdlgen 生成 Verilog にも同属性を埋め込めば、生成源 (a)(b) は将来1本化できる

## 実装順序と進捗

1. [x] `mdl_fpga/contract.py` — io_contract+goldens 生成(IR/manifest/board から)
2. [x] `runtime/mdl_runtime/` コア — contract/protocol/ports/link/flash/selftest/cli
       (独立性 lint・hw_check とのバイト一致・fake serial E2E をテストで固定。
       serve-stdio 含む。test_standalone.py 15件)
3. [x] bundle 拡張 — app/ 同梱(vendor+requirements+README 環境構築節)+
       flasher パック同梱(`--flasher-pack <dir|zip>` / 環境変数
       MDL_FPGA_FLASHER_PACK、無指定時は README 誘導に降格)。board 書出は
       解決済み dict 化(draft ボードでも bundle 可)
4. [x] pygame テンプレ(canvas_app: bitmapNxM 手描き/ビットトグル+クラス棒/
       ビットランプ)+ `gui` サブコマンド(pygame 不在は誘導)。
       **実機での目視受入は未実施**(pygame 導入+書込済みボードで確認する)
5. [~] flasher パック: **生成は `eda flasher-pack` 動詞化済み**(oss-cad-suite
       から exe+厳選 DLL8 個=31MB を抽出し、Apache-2.0/LGPL 全文を
       LICENSES/ へ、SPDX+ソース入手先を NOTICE.md へ機械生成。PATH 絶縁で
       起動確認済み。local.toml の flasher_pack で既定同梱・`--no-flasher` で
       オプトアウト)。README 書き込み節は同梱時に同梱版を主線に切替。
       残: linux/darwin 分の抽出と GitHub Releases ホスト
6. [ ] (将来)Verilog 注釈パーサ(生成源 b)/ serve-stdio を使う他フレームワーク例

## 検証方針

- goldens は bundle 時に ir.simulate から生成 → selftest は「参照シム⇔実機」の
  bit 一致検証そのもの(hw-check の配布版)
- runtime は fake serial(ループバック応答器)で CI 完結、実機は selftest で受入
- Windows の書込ドライバ(Zadig/USB-Blaster)は同梱可否と無関係に残る制約
  → flash 前チェック+OS 別ガイド表示で吸収
