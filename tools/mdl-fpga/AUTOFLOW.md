# AUTOFLOW — PC接続からボード実装までの自動化設計

目的: **ボードをPCに接続してから実機で動くまでのユーザ操作を、
原理的・法的に消せない3点まで減らす**。既存の `up`(detect→advise→build→flash→
hw-check)を骨格に、未知ボード対応(ボード定義のオンデマンド生成)と
ツールチェーン導入の自動化を差し込む。

関連: [PORTABILITY.md](PORTABILITY.md)(非Win対応=本設計の前提修正)/
`mdl_fpga/boards/NOTICE`(出典規約)/
[README.md](README.md) Tier表。

## 残る人手(これ以上減らせない3点)

| # | 操作 | 理由 |
|---|---|---|
| ① | ボード名の確定(候補から1選択) | JTAG IDCODEで分かるのはチップまで。同一チップ搭載ボードは原理的に区別不能(README:48)。ボード側にピン配置の電子記録は無い |
| ② | GPIO増設等のI/O意図(manifest記述 or 対話) | ユーザの設計意図そのもの。標準構成(LED/SW/UART)なら省略可 |
| ③ | ベンダEDA本体の入手とEULA同意 | ダウンロード認証・同意の主体はユーザ(法的)。**ossファミリ(Gowin/Lattice系)ならゼロになる** |

Tier昇格(実機受入の合否判断)も人だが、これは開発側の作業でエンドユーザ操作ではない。

## パイプライン全体

```
[接続] → 1 detect → 2 ボード確定(人①) → 3 board-def fetch → 4 pin-validate
       → 5 I/O意図(人②/省略可) → 6 eda plan/install(人③/oss系はゼロ)
       → 7 build → 8 flash → 9 hw-check → 10 記録・昇格
```

## 各工程の達成手段

| # | 工程 | 主体 | 実機 | Net | 状態 |
|---|---|---|---|---|---|
| 1 | 接続検出 | スクリプト | 要 | 不要 | **実装済み**(`detect.py`) |
| 2 | ボード確定 | **ユーザ1選択** | — | 不要 | 実装済み(候補+確信度)。未知ボード分岐を追加 |
| 3 | ボード定義取得・生成 | スクリプト | 不要 | 要 | **実装済み**(`boarddef/`、`board-def fetch`) |
| 4 | EDA検証パス | スクリプト | 不要 | 不要 | **実装済み**(`pin_validate.py`、`board-def validate`) |
| 5 | I/O意図 | ユーザ(manifest) | 不要 | 不要 | 実装済み(`io_link.py`/manifest) |
| 6 | ツールチェーン判定・導入 | スクリプト+ユーザ③ | 不要 | 要 | **実装済み**(`eda_plan.py`、`eda plan/install`。qdz自動投入は後続) |
| 7 | 合成 | スクリプト | 不要 | 不要 | **実装済み**(Quartus+gowin-oss=M7で完了) |
| 8 | 書込 | スクリプト | 要 | 不要 | 実装済み(quartus_pgm+openFPGALoader、.fs直接書込) |
| 9 | 実機検証 | スクリプト | 要 | 不要 | **実装済み**(`hw_check.py`) |
| 10 | 記録・昇格 | 半自動 | 要 | 不要 | 運用(NOTICE/Tier規約) |

### 1. 接続検出(実装済み)

`detect.py`: openFPGALoader --scan-usb → jtagconfig → pyserial の多段プローブで
JTAG IDCODE・USB VID/PID/製品文字列・ケーブル種を採取し、boards/*.json の
detection 署名と照合して候補+確信度を返す。

追加実装: **board DB に一致が無い場合**、IDCODE→チップ/ファミリだけ確定して
「チップ既知・ボード未知」として工程2へ渡す(IDCODE→ファミリ表は
openFPGALoader `src/part.hpp` 由来。転記規約はNOTICEに従う)。

### 2. ボード確定(人①・不可避)

- 候補あり: 確信度順リストから1選択(CLI/ビューア両方)。
- 候補なし(未知ボード): ユーザがボード名を入力 → 工程3のプロバイダ検索語にする。
- 選択結果はモデル別サイドカーに記憶(既存挙動)し、2回目以降は無確認。

### 3. board-def fetch(新規・中核)

DE10-Lite追加でscratch実行した「機械照合転記」(NOTICE:14-17)を汎用CLI化する。
**デフォルトでボード定義は同梱しない**(オンデマンド生成。再配布ライセンス問題も消える)。

```
mdl-fpga board-def fetch <board名>
  1. プロバイダ群から取得(下表。信頼できる上流のホワイトリスト制)
  2. 共通正規化 → schema_version=2 JSON(pydantic検証は board_db.py 既存)
  3. 2ソース相互照合: 全ピン完全一致のみ採用(創作禁止ポリシーの機械化)。
     不一致 → 生成拒否+差分レポート
  4. NOTICE 自動追記(プロバイダ名・SPDX・commit/版・取得日)
```

| プロバイダ | 対象 | 形式 | 備考 |
|---|---|---|---|
| litex-boards | 数百ボード横断 | Python platform(機械可読) | BSD-2-Clause。1次ソースの本命 |
| Vivado Board Store | Xilinx系 | board.xml / part0_pins.xml(構造化XML) | 公式。xhub Tcl or GitHub直接。Xilinx系はこれが最良 |
| ベンダgolden qsf | Intel/Terasic系 | qsf(行形式) | Terasic System Builder産・Intel Design Store GHRD。qsfパーサはDE10-Lite照合の型を復元 |
| 公式example .cst | Gowin/Sipeed系 | cst | Tang Nano 20K は litex + Sipeed公式example の2ソース |
| openFPGALoader part.hpp | 全社 | C++ヘッダ | IDCODE転記元(既存実績。Apache-2.0) |

**信頼度モデル(Tier拡張の提案)**:

| 条件 | 扱い |
|---|---|
| 独立2ソース完全一致 | Tier2(unverified)— 現行DE10-Lite同等 |
| 単一ソース+工程4のEDA検証パス通過 | **Tier3(新設)**: 自動生成・要実機確認を明示。advisorは保守的既定 |
| 照合不一致 / ソース無し | 生成拒否 → 従来の手動手順(NOTICE手書き)へフォールバック |

### 4. pin-validate(新規)

EDAは**ボード配線を知らないがチップは完全に知っている**ことを利用し、
「情報源」でなく「検証者」として使う第3層:

- Intel系: 生成JSONからstubトップ+全ピン割当qsfを作り(`synth.py` の
  `write_project_files` 資産を流用)、`quartus_map` に食わせる。
  実在しないピン・I/O不可ピン・電圧規格違反はEDAエラーとして機械検出。
- oss系: nextpnr(apycula)はチップDBを内蔵 → cst を与えたnextpnr実行で同等検証。
- Xilinx系: `get_board_part_pins` / stub+xdc で同様。

該当EDAが未導入なら本工程はスキップ(Tier3の昇格条件が満たせないだけで、
2ソース一致のTier2生成は可能)。

### 5. I/O意図(人②・省略可)

既存のmanifest(io.link)資産をそのまま使う。manifestに事前記述すれば無対話、
無ければ既定(ボード先頭ヘッダのUARTリンク等、`resolve_uart_link` 既存挙動)。
GPIO増設はボード定義の `headers` +`resolve_header_io` で解決(schema v2 既存)。

### 6. eda plan / eda install(新規)

family→toolchain の静的表で分岐し、doctor を「診断」から「導入計画の提示・実行」へ拡張:

| ファミリ | ツールチェーン | 自動化水準 |
|---|---|---|
| Gowin / Lattice(iCE40/ECP5) | oss-cad-suite(yosys+nextpnr+apycula) | **完全自動**: GitHub Releasesから該当OSアーカイブをDL→展開→local.toml記録→doctor検証。ログイン・EULAクリック不要 |
| Intel(Cyclone V / MAX 10 等) | Quartus | **半自動**: 本体=ユーザ導入(planが版・URLを提示=人③)。**デバイスサポートは自動**: 導入済み版を特定→版一致の family qdz をDL→無人投入(`--mode unattended --accept_eula 1`、qdz同梱ディレクトリ方式)。導入済みfamily判定は quartus_sh Tcl(get_family_list系)か devinfo ディレクトリ列挙(実機で1回要確認) |
| Xilinx(7シリーズ等) | Vivado | **半自動**: install_config.txt の `Modules=` 行を生成し `xsetup --agree ... --batch Install --config`。DL認証は `-b AuthTokenGen` で初回1回に圧縮(人③) |

設計原則: `--accept_eula` / `--agree` は**実行前に必ずユーザへy/Nを1回確認**してから
渡す(同意の主体はユーザ)。管理者権限は原則不要(USB-Blasterドライバのみ別、工程8)。

### 7-9. build / flash / hw-check(既存+P2)

- build: Quartusバックエンド実装済み。**yosys+nextpnr バックエンドがP2の新規**
  (未実装)。boards JSON→cst/pcf 生成は synth.py のqsf生成と同型。
- flash: openFPGALoader は合成ツールと独立に書込可(Quartus産 .rbf 実績あり)。
  Windowsでは Quartus のドライバと WinUSB(Zadig)が排他で、USB-Blaster 系のボードは openFPGALoader で書けないことがある(既知の制約。`doctor` が案内する)。Tang Nano 20K は競合なし。
- hw-check: UART自動検出→全ベクトルbit一致(実装済み)。

### 10. 記録・昇格(運用)

実機グリーン後: IDCODE実測値で `jtag_idcodes_note` 更新(NOTICE規約)、
limits較正値の記録、Tier昇格。**値の採取と差分提示はスクリプト、受入判断と
コミットは人**(開発側作業。エンドユーザには関係しない)。

## 実装順序

1. [x] **board-def fetch** — `mdl_fpga/boarddef/`(providers+fetch+recipes)として実装。
   受入基準=tang-nano-20k レシピが references/ のコピーから実機受入済み board JSON と
   同一の物理事実を再導出(test_boarddef_fetch.py)
2. [x] **pin-validate** — `mdl_fpga/pin_validate.py`。実 nextpnr で受理/拒否の統合テスト済み
3. [x] **eda plan + oss-cad-suite自動導入** — `mdl_fpga/eda_plan.py`
4. [x] **yosys+nextpnr バックエンド** — M7 で完了(`backends/gowin_oss.py`、実機受入済み)
5. [x] **up への統合** — `board_db.resolve_board`(key/JSONパス/dict)で draft を全動詞
   (advise/build/flash/hw-check/bundle/eda plan/up)へ貫通。up は未知ボード時の
   board-def fetch 誘導+unverified ボードの合成前 pin-validate ゲート(FAIL は
   build へ進めない)+build 前段のツール充足確認(不足なら eda plan 表示)。
   実E2E: fetch 生成 draft → up(実合成込み)全段グリーン(test_up_autoflow.py)
6. [ ] (後続)Quartus qdz自動投入・Vivado Board Store プロバイダ — Intel/Xilinx系への一般化

前提: [PORTABILITY.md](PORTABILITY.md) の toolchain.py 修正(非Win対応)は
実装済み(test_portability_toolchain.py)。

## 参考リンク(実装時に参照)

- litex-boards: https://github.com/litex-hub/litex-boards (BSD-2-Clause)
- Xilinx Board Store: https://github.com/Xilinx/XilinxBoardStore
  (board.xml/part0_pins.xml。xhub Tcl: `xhub::refresh_catalog`→`xhub::install`)
- get_board_part_pins(UG835): https://docs.amd.com/r/2025.1-English/ug835-vivado-tcl-commands/get_board_part_pins
- Vivado batch install(UG973): https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Running-the-Installer
- Quartus無人インストール(コミュニティ実例): https://community.intel.com/t5/Intel-FPGA-Software-Installation/Quartus-Prime-Standard-20-4-silent-installation-with-SCCM/td-p/1288504
- oss-cad-suite releases: https://github.com/YosysHQ/oss-cad-suite-build/releases
- openFPGALoader: https://github.com/trabucayre/openFPGALoader (Apache-2.0。IDCODE表=src/part.hpp)
- Sipeed Tang Nano 20K example(.cst): https://github.com/sipeed/TangNano-20K-example
