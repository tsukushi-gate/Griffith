# mdl-fpga — FPGA 実装支援ツール

Griffith(griffith-logic)で学習した論理ゲート NN(IR JSON)を、FPGA 未経験者が実機で動かすための
ツール。構想書はmdl-fpga 構想書、
契約面は **magicdelta-ir v0.2.0**(IR は固定入力。HDL 生成から先を本ツールが所有)。

```
学習(ライブラリ) → IR(JSON) → convert → check(bit一致) → build(合成) → flash(書込) → hw-check(実機bit一致)
                                  ↑ detect(ボード自動認識)/ advise(搭載可能性判定)/ bundle(他PC書込用zip)/ up(一気通貫)
```

## クイックスタート

```bash
# 0) 環境診断(最初に必ず)
uv run python tools/mdl-fpga/cli.py doctor

# 1) 学習済みモデルの IR を検証(blackbox 検出・拒否はここ)
uv run python tools/mdl-fpga/cli.py convert model.ir.json

# 2) ボード無しで bit 一致検証(参照シム ⇔ RTL + 生成器二重化)
uv run python tools/mdl-fpga/cli.py check model.ir.json

# 3) 合成(advisor が搭載可能性を事前判定 → presim → Quartus map/fit/asm/sta)
uv run python tools/mdl-fpga/cli.py build model.ir.json --selftest

# 4) JTAG 書込(DE0-CV、USB-Blaster)
uv run python tools/mdl-fpga/cli.py flash <build出力の.sof>

# 5) 実機 bit 一致(UART。CP210x ブリッジ接続時)
uv run python tools/mdl-fpga/cli.py hw-check model.ir.json

# --- 多ボード対応の新動詞 ---
uv run python tools/mdl-fpga/cli.py detect          # 接続ボードの自動認識(EDA不要)
uv run python tools/mdl-fpga/cli.py advise model.ir.json --board de0-cv   # 搭載可能性(EDA不要)
uv run python tools/mdl-fpga/cli.py bundle <buildのworkdir>  # 他PC書込用zip(svf/rbf変換込み)
uv run python tools/mdl-fpga/cli.py up examples/full_adder.manifest.json --hw-check  # 一気通貫
```

一本道の実例(学習 → 実機まで): [examples/train_full_adder.py](examples/train_full_adder.py)
のヘッダ参照。最小 IR サンプルは [examples/parity4.ir.json](examples/parity4.ir.json)。

## CLI 動詞

| 動詞 | 内容 |
|---|---|
| `doctor` | 環境診断(iverilog/Verilator/openFPGALoader/Quartus/JTAG/シリアルの検出と次の一手) |
| `detect` | **接続ボードの自動認識**(EDA 不要)。openFPGALoader → jtagconfig の順で JTAG を叩き、boards/*.json の detection 署名(IDCODE/部品名/USB/ケーブル種)と照合して候補+確信度。同一 FPGA 搭載ボードは原理的に区別できないため、同率複数はユーザー確認へ |
| `advise` | **搭載可能性の事前判定**(EDA 不要)。実測則ベース。未較正ボードは保守的既定+「未較正」明示 |
| `convert` | IR 読込・検証。blackbox(実数ノード)は明示拒否+誘導 |
| `sim` | RTL 全体シミュレーション(TB 自動生成、出力を y_out.hex へ) |
| `check` | **bit 一致 PASS/FAIL**。(a) 参照シム⇔自製RTL、(c) 自製⇔ライブラリ生成器の二重化照合 |
| `view` | **可視化ビューア**(クリック=1clk の対話 RTL シム)。回路図(RTL Viewer ライク)+任意地点プローブ+Python 整形関数。下記「view の使い方」参照 |
| `build` | HDL 生成 → advisor 事前判定 → presim → Quartus 合成 → .sof。`--selftest` でオンチップ自己検証を同梱 |
| `flash` | JTAG 書込。`--via quartus`(.sof)/ `--via openfpgaloader`(.svf/.rbf/.fs、**EDA 不要**)。既定 auto=拡張子で選択。`--board` でボード定義から -b/-c を自動導出(Tang Nano 20K は `-b tangnano20k`) |
| `bundle` | **持ち運び zip**: sof+svf/rbf 変換(quartus_cpf)+ボード定義+golden+配線・書込手順 README(Windows/macOS)。書込 PC は openFPGALoader だけで済む(macOS は Quartus 非対応でもこの経路で書ける) |
| `issp` | **オーバーレイの実機実行**(JTAG の In-System Sources and Probes)。start を叩いて probe を読み、一致数・処理数・サイクル数(32bit)から cyc/推論 と inf/s を出す。UART 不要 |
| `hw-check` | 実機 bit 一致(UART 経由で全ベクトル照合) |
| `up` | **一気通貫**: detect → advise → build → flash →(`--hw-check`)。失敗段で停止し「原因+次の一手」を段別に言い切る |

## 実機での確認方法(2 経路)

1. **UART(hw-check)**: CP210x 系 USB-UART ブリッジを GPIO_0 に接続
   (RX=物理ピン1 / TX=物理ピン2 / GND)。PC からベクトルを送り応答を golden と照合。
2. **オンチップ自己検証(`build --selftest`)**: 全ベクトルと期待値を回路に焼き込み、
   電源投入直後に自動照合。**LEDR[0] 点灯 = 全ベクトル bit 一致**(UART 不要)。
   LEDR[1]=不一致あり、LEDR[2]=ハートビート(生存確認)、SW[9]=リセット(下=通常)。

## 検証の辺(「見る前に合っている」)

```
ライブラリ ir.simulate(numpy bit-exact = 正解オラクル)
   ⇔(a) 自製コアの RTL シミュレーション(check の主判定)
   ⇔(b) 実機(hw-check / selftest)
(c) 自製生成器 ⇔ ライブラリ export/verilog.py の相互 bit 一致(ドリフト検出)
```

## モジュール構成(構想書 §4 の L0-L7)

| モジュール | 責務 |
|---|---|
| `toolchain.py` | oss-cad-suite / Quartus の探索(env → local.toml → PATH → 既知パス)と実行環境 |
| `doctor.py` | 環境診断(原因・次の一手の 3 点固定) |
| `ir_io.py` | IR 読込・検証・blackbox 拒否(convert) |
| `hdlgen/core.py` | IR → コア Verilog。**安定命名**(wire=`w_<ノードid>`)+観測点 `(* keep *)`+名前マップ JSON |
| `hdlgen/overlay.py` | **オーバーレイの設計計算**(IR の配線を全ゲート照合し、計算型 = rot / 引き伸ばし rot なら op 4bit の ROM + アドレス生成器の初期表、保存型なら `{op, a, b}` のレーン別記録 ROM を出す)+サイクル式+M10K 見積り |
| `hdlgen/overlay_system.py` | オーバーレイの検証システム生成(画像 ROM・期待値 ROM・ISSP・サイクルカウンタ・7セグ・TB) |
| `issp.py` | 実機 ISSP 実行(quartus_stp)。probe の割付は生成側の `overlay_system.json` が契約 |
| `hdlgen/wrapper.py` | トップ生成(UART フレーミング+入出力 FF+FSM)。`hdl/` の実証済み UART/フレーム部品を使用 |
| `hdlgen/selftest.py` | オンチップ自己検証(ベクトル焼き込み+LED 表示) |
| `vectors.py` / `check.py` | ベクトル生成・TB 生成・iverilog 駆動・bit 照合 |
| `board_db.py` + `boards/` | ボード定義 **schema_version=2**(物理事実のみ: ピン・デバイス・detection 署名・GPIO ヘッダ。pydantic 検証。**ピン・IDCODE の創作禁止**=出典は boards/NOTICE) |
| `io_link.py` | manifest io.link(uart 配線=プロジェクト側の判断)→ ピン解決。既定はボード最初のヘッダの index 0/1 |
| `hdlgen/ports.py` | TopPorts=生成トップのポート名/幅(board JSON から導出。リテラル二重管理の解消) |
| `detect.py` | ボード検出(openFPGALoader/jtagconfig/pyserial の3プローブ+署名照合+確信度) |
| `advisor.py` | 搭載可能性の事前判定(実測則 R1/R2/R4。resources.unit 一般化+未較正パス) |
| `manifest.py` | システムマニフェスト 0.2(IR を汚さない上位ファイル。io.link で uart 配線を宣言) |
| `bundle.py` | 持ち運び zip 生成(svf/rbf 変換+配線 README) |
| `up.py` | 一気通貫オーケストレーション(各動詞の run_* を薄く連結) |
| `backends/` | **合成バックエンド層**(toolchain 欄でディスパッチ)。base.py=FlowResult/プロトコル、quartus.py=.qsf/.sdc+map/fit/asm/sta、gowin_oss.py=.cst/.ys+yosys→nextpnr-himbaechel→gowin_pack→.fs。結果は共通の flow_summary.json |
| `synth.py` | 互換シム(実体は backends/quartus.py。既存 import 維持) |
| `flash.py` | JTAG 書込(USB-Blaster) |
| `hw_check.py` | 実機 UART bit 照合(CP210x 自動検出) |

## サポートボード(3 Tier 表記)

| Tier | ボード | 合成 | 状態 |
|---|---|---|---|
| **Tier1** | Terasic DE0-CV(Cyclone V 5CEBA4F23C7) | quartus | 実機テスト済み |
| **Tier1** | Sipeed Tang Nano 20K(GW2AR-LV18QN88C8/I7) | **gowin-oss(EDA 完全不要)** | 実機受入済み(detect 1.0 / up 全段 / hw-check bit一致。**必ず hw-check まで**=下記「既知の罠」) |
| Tier2 | Terasic DE10-Lite(MAX 10 10M50DAF484C7G) | quartus | **実機未検証**(定義のみ)。ピンは litex-boards+実プロジェクト qsf の機械照合で転記(boards/NOTICE) |

追加は `mdl_fpga/boards/*.json`(schema_version=2。スキーマは de0-cv.json 参照)。
合成バックエンドは board JSON の `toolchain` 欄でディスパッチされる
(`mdl_fpga/backends/` のレジストリ。quartus / gowin-oss。将来枠:
vivado / openxc7 / lattice-oss / radiant — backends/ に実装を1つ足し
レジストリ登録するだけで build/up は無改修で載る)。

### Tang Nano 20K(gowin-oss)の使い方と既知の罠

```bash
# ケーブル1本(BL616 内蔵 JTAG+UART)で一気通貫。EDA インストール不要
uv run python tools/mdl-fpga/cli.py up tools/mdl-fpga/examples/full_adder.tn20k.manifest.json --hw-check
```

- Windows 初回のみ: BL616(USB 0403:6010)の **Interface 0 を Zadig で
  WinUSB に切替**(Interface 1=UART は COM のまま残す。Altera USB-Blaster
  には触れない)
- クロック 27MHz は board 定義から自動(`clock_hz` は manifest → board →
  50MHz の順で解決)
- **既知の罠(実測)**: nextpnr-himbaechel(gowin)は**配置依存で機能不良
  ビットストリームを生成することがある**(タイミング PASS のまま論理が化ける。
  詳細は `backends/gowin_oss.py` 冒頭)。既定は `-nowidelut` で、実機検証は
  **必ず hw-check(bit 一致)まで**行う。化けた場合は
  `MDL_FPGA_NEXTPNR_SEED` を変えて再ビルド(使用 seed は
  flow_summary.json に記録される)
**ピン・IDCODE は転記元を必ず boards/NOTICE に記録する(創作禁止=実配線事故防止)**。
uart のピンはボード定義には無い(物理事実でないため): manifest 0.2 の `io.link` が
`{header, rx_index, tx_index}`(または raw pin)で宣言する。実例は
[examples/full_adder.manifest.json](examples/full_adder.manifest.json)。

## ローカル設定

ツールパスが PATH に無い場合は `local.toml`(gitignore 済み)を置く:

```toml
oss_cad_root = 'C:\oss-cad-suite'
quartus_bin  = 'C:\altera_lite\24.1std\quartus\bin64'
```

環境変数 `MDL_FPGA_OSS_CAD` / `MDL_FPGA_QUARTUS_BIN` でも指定可。
実機 UART check には `uv sync --extra dev --extra cpu --extra fpga`(pyserial)。

## テスト

```bash
uv run pytest tools/mdl-fpga/tests -q     # iverilog 無しの環境ではシム系は自動 skip
```

## 現状と今後(構想書のマイルストーン対応)

- [x] **M0**: ボード無しで convert → sim → check が bit 一致 PASS
- [x] **M2(spatial)**: DE0-CV で build → flash → 実機検証(selftest LED / UART hw-check)
- [x] **M1(v2)**: クリック=1clk ビューア。最大リスク R2(GUI↔cocotb 橋)は
      TCP JSON 行ブリッジで解消し、全ノード値が simulate_trace と bit 一致することを
      回帰テストで固定。UI は React 回路図(RTL Viewer ライク)+任意地点プローブ+
      ユーザー定義 Python 整形関数+レイアウト永続化(残: 時間スケール UX・実機並走モード・
      deck.gl 大規模描画)
- [x] **M6(多ボード対応)**: board JSON v2(物理事実のみ+detection 署名)/
      manifest 0.2 io.link / TopPorts データ駆動化 / detect・advise・bundle・up 動詞 /
      DE10-Lite 定義(Tier2)/ ビューアのプロジェクト・ボードペイン。
      実機受入(detect の IDCODE 採取・bundle svf 書込検証)は DE0-CV 接続時に実施
- [x] **M7(P2 オープンツールチェーン+マルチベンダ)**: backends/ 層
      (SynthBackend レジストリ、Quartus 移設+gowin-oss 新設)/ ボードスキーマ
      多ベンダ化(pin_format/toolchain)/ ラッパ LED・リセット汎用化 /
      Tang Nano 20K 実機受入完了(EDA 完全不要で
      detect→build→flash→hw-check 全段グリーン、ケーブル1本)。
      ベンダ EDA 探索枠(Vivado/Gowin EDA/Radiant)を toolchain/doctor に整備。
      残: nextpnr 配置依存の機能不良への build 時自動検証(fit 後シム、M3 と共通)
- [x] **M3(オーバーレイ、2 変種)**: IR の配線を全ゲート照合し、**計算型**(rot / 引き伸ばし rot → op 4bit のゲート ROM + アドレス生成器)と**保存型**(自由配線 → `{op, a, b}` 32bit のレーン別記録 ROM)のどちらかで生成 → presim(ir.simulate ⇔ RTL)→ 合成 → 実機 ISSP(一致数・サイクル・inf/s)。上の「オーバーレイ」節を参照。残: Verilator 早送り、.mem 差替だけのモデル切替 UX(同一構成なら再合成不要という性質は成立済み)
- [ ] **M4**: イグザンプル集・配布品質 — 未着手

### view の使い方(v2: 回路図+プローブ+Python 整形関数)

```bash
uv run python tools/mdl-fpga/cli.py view <model>.ir.json    # ブラウザが開く
```

- **回路図**: lowered IR を RTL Viewer ライクに表示(ブロック=IRノード、バス配線、
  elk 自動整列、pan/zoom)。lut_layer は「展開」で**個別ゲート+ビット配線**表示
  (>128 ゲートの層は展開拒否 → プローブで観測)。1bit 配線と各ゲートは実値で点灯
- **プローブ(値表示部)**: ノード右の ● から空きキャンバスへドラッグすると、その場に
  値表示カードが生える。回路図と一緒に拡大縮小/移動し、📍 で**画面固定**もできる。
  カード上でビットスライス(`5` / `7:0`)とレンダラを切替、✕ で削除
- **Python 整形関数**: モデルと同じディレクトリの `renderers.py` に
  `def my_view(bits, ctx) -> dict` を書くとレンダラ一覧に現れる
  (**実行中の書き換えも自動リロード**)。実例は
  [examples/renderers.py](examples/renderers.py)。内蔵: auto/bits/hex/uint/bars/image。
  注意: renderers.py はサーバプロセス内で実行される(自分のローカルコード前提)
- **プローブの表示オプション**(カード2段目):
  - **幅**: bits/image の横幅(bit 数)を任意指定。image は既定=正方形、端数は
    0 パディングでどんな bit 数でも長方形に並ぶ
  - **ビット順**: bits は既定 `[msb:lsb]`(左端=最上位)。ボタンで `[lsb:msb]` に切替
  - **段**: 表示キュー段数。過去 N クロックの値を新しい順に保持
    (例: 3→2→8 と出力されると (8, 2, 3, x, x))。bits/image/uint 等どの表示でも使え、
    レンダラを切り替えると**過去値にも遡って**再整形される
  - **−/+**: カード個別の拡大縮小(回路図全体のズームとは独立、0.5〜3倍)
  - **⏭変化まで**: この信号(スライス適用後)の値が変わるまでクロックを進める
    (上限1000。進んだ clk 数と結果はステータスバーに表示)
  - 表示設定の変更は step を進めなくても即座に再表示される
- **プロジェクトペイン**(左サイドバー最上部): 現在のプロジェクトを表示し、
  「開く…」ダイアログ(イグザンプル/ファイルの2タブ+最近使った一覧)から
  **ビューアを再起動せずにモデルを切替**できる(旧シミュレーションは切替完了まで
  動き続け、失敗時はそのまま継続=ロールバック)。未ビルドの system イグザンプルは
  disabled+ビルド手順のヒント表示
- **ボードペイン**: ボード DB からの選択+**認識ボタン**(JTAG は排他資源のため
  自動実行しない)。一意候補は自動選択、複数候補は基板のシルク印字の案内で確認。
  選択ボードでの搭載可能性(advise、有り/やや有り/無しの3段階)が常時表示され、
  選択はモデルごとに記憶される
- **永続化**: プローブ・位置・展開状態・viewport・選択ボードは `<model>.viewer.json`
  (モデル横のサイドカー)に自動保存され、次回起動で復元される
- **UI基盤はエディタと統一**(将来統合の布石): Tailwind 4 + shadcn(new-york/slate)+
  radix-ui + lucide + sonner。デザイントークンは `frontend/src/index.css`
  (エディタからコピー、**トークン名=統合契約**。`tokens.drift.test.ts` が
  エディタ側との名前一致をCIで監視)。**ライト(既定・エディタ同一)/ダーク切替**を
  ヘッダの🌙/☀️で行う(ダークトークンは将来エディタへ逆移植可能な `.dark` ブロック)
- **フロント開発**: `cd tools/mdl-fpga/frontend && npm install && npm run dev`
  (vite 5175 → 8788 プロキシ)。配布は `npm run build` の dist/ を FastAPI が配信
  (エンドユーザに Node 不要)。dist 未ビルド環境は旧単一 HTML へ自動フォールバック

### view のアーキテクチャ

```
ブラウザ(React+ReactFlow: frontend/dist、無ければ static/index.html)
   ⇕ WebSocket /ws/sim(JSON。フレームに probes:{id: payload} を同梱)
FastAPI サーバ(viewer/server.py)… プローブ=表示専用の後処理。真実は常に RTL 側
   │  probes.py(スライス)/ renderers.py(内蔵+サイドカー)/ persistence.py
   ⇕ TCP 127.0.0.1(JSON 行)
cocotb ブリッジ(viewer/cocotb_bridge.py)… vvp 内で実行、クロック手動駆動
   ⇕ VPI
iverilog(top_view = 境界注入モード: x_in → xin FF → mdl_core → y_reg)
```

## オーバーレイ(M3)

固定回路を 1 度だけ合成し、モデルは **.mem の差替**で載せる方式。ゲートは毎
サイクル V 個ずつ逐次評価するので、載る規模は**メモリ量だけ**で決まる(空間
焼込みの配線天井が無い)。配線の持ち方が 2 通りあり、`--overlay-wiring`
(manifest の `overlay.wiring`。既定 `auto`)で選ぶ。

| 方式 | ゲート記述 | 載せられる配線 | DE0-CV の実測上限 |
|---|---|---|---|
| **計算型**(computed) | op **4bit** のみ | rot / 引き伸ばし rot(閉形式) | **448,000 ゲート**(計算型の先行実験) |
| **保存型**(stored) | `{op 4bit, a, b}` = **32bit**(層幅 > 16,384 なら 40bit) | **自由配線**(置換・ランダム・学習配線) | **54,000 ゲート**(保存型の先行実験) |

計算型は配線 (a, b) を表で持たず層ごとのシフト量から算術で生成するので記述が
8 分の 1 になる(同一 48K ネットのゲート ROM 240 → 32 M10K)。保存型は 1 ゲート
ごとに前層アドレスを 2 本持つ代わりに、**どんな配線でも載る**。上限はどちらも
M10K 308 ブロック律速(保存型・計算型の先行実験の実測。いずれも 308/308
で到達)。`auto` は全層が閉形式に一致すれば計算型、外れれば保存型を選び、その
理由をメッセージに出す。

```bash
# 1) 載るか・どちらの方式になるかを判定(EDA 不要)
uv run python tools/mdl-fpga/cli.py advise model.ir.json

# 2) 生成 → presim(ir.simulate ⇔ overlay RTL)→ Quartus 合成
uv run python tools/mdl-fpga/cli.py build model.ir.json --mode overlay \
    --overlay-vectors 1000 --clock-div 2               # 既定 auto
uv run python tools/mdl-fpga/cli.py build model.ir.json --mode overlay \
    --overlay-wiring stored --overlay-vectors 512 --clock-div 2

# 3) 書込 → 実機で一致数・サイクル・inf/s を読む(JTAG。UART 不要)
uv run python tools/mdl-fpga/cli.py flash <build が示す .sof>
uv run python tools/mdl-fpga/cli.py issp <build の workdir>/quartus
```

### 計算型が対応する配線規則(これ以外は保存型へ回る)

| 規則 | 定義 | 使いどころ |
|---|---|---|
| rot | `idx[g] = (g + s) mod W_in` | 入力幅と出力幅が同じ層 |
| 引き伸ばし rot | `idx[g] = (base[g] + s) mod W_in`、`base[g] = (2*g*W_in + W_out) // (2*W_out)` | 入力幅と層幅が違う層(層 0) |

生成器は **IR の indices を全ゲート照合**し、どちらの閉形式にも一致しなければ
`--overlay-wiring computed` では「rot / 引き伸ばし rot の配線ではないので
overlay 化できない」と言い切って拒否し、`auto` なら保存型へ落とす
(`metadata['wiring']` は信用せず indices で判定する)。アドレス生成は
加算と条件減算だけで、**乗算器を使わない**(引き伸ばし rot はレーンごとの余り
レジスタを持つ Bresenham 型。rot は余り歩幅 0 で同じ回路を通る)。

### 保存型のゲート記録

1 ゲート = `{op[31:28], a[27:14], b[13:0]}`(保存型の先行実験と同一形式)を**レーン別の
ROM** `gatemem_lane<k>.mem` に置く(レーン k の語 w = ゲート w*V+k)。層幅が
16,384 を超えるときだけ記録を 40bit(a/b 各 18bit = M10K の最も広いモードの
自然幅)へ広げる。書き出す前に全語を復号して IR の `(op, a, b)` と全数照合し、
1 語でも違えば `.mem` を書かない(`check` の `record_vs_ir` 辺で後からも
確かめられる)。RTL は `lgn_overlay_stored.v` で、計算型の `lgn_overlay.v` とは
ゲート記録の読み方だけが違い(パイプライン位相・層境界・drain・popcount・FSM
は同一)、**サイクル数は計算型と厳密に同じ**になる(48K で 6,129。保存型・計算型の先行実験と同じ値)。

### 構成の制約

- 全層同じ幅 W(活性バッファの前提)、2 入力ゲート、`group_sum` 終端の分類 IR
- `W % V == 0` / `group_size % V == 0` / `W >= 入力幅`
- V は 2 の冪かつ 8 の倍数(op ROM の 32bit 語詰め)
- **W/V は 2 の冪以下に保つ**(超えると活性バッファが 2^ceil(log2(W/V)) 語を
  確保して複製が倍化する。先行実験の実測から得た設計則。生成器は警告を出す)

### サイクル式と資源見積り

```
cyc/推論 = ceil(IN/V) + DEPTH*(W/V) + 4*DEPTH + 1 + 6
inf/s    = clk_core / (cyc/推論)
```

サイクル式は**方式によらない**(記録の読み方が違うだけでパイプラインが同じ)。

`advise` と `build` が M10K の内訳見積り(ゲート ROM / 活性バッファ / 画像
ROM)を出す。ゲート ROM は構造で決まる(計算型・保存型とも先行実験の
fit 実測に較正済み。48K V=8 で計算型 32 blk / 保存型 240 blk = 30 blk x 8
レーン)、活性と画像は Quartus の割付に依存するため近似(±10% 程度)。fit 後は
`flow_summary.json` の `m10k_by_entity` に**実測の内訳**が入る。

保存型では検証 ROM に回せる M10K が減る。DE0-CV の 48K は画像 1,000 枚だと
超過するので N=512 枚にする(ゲート ROM 240 + 活性 16 + 画像 40 = 296/308)。

### manifest(0.3)

```jsonc
{
  "manifest_version": "0.3",
  "ir": "model.ir.json",
  "board": "de0-cv",
  "mode": "overlay",
  "overlay": { "v": 8, "n_vectors": 1000, "clock_div": 2, "wiring": "auto" }
}
```

`overlay.clock_div` はコアクロックの分周比(2 = 50MHz を 25MHz へ。PLL を使わず
トグル FF + SDC の派生クロック)。`clock_div=1` では生成物が従来と byte 一致
する(ゴールデンで固定)。`overlay.wiring` は `auto`(既定)/ `computed` /
`stored`。省略した 0.3 manifest はそのまま読める。

### 実機の読み方

- **7セグ**: HEX3-0 = golden(`ir.simulate` の予測)との**一致数**、
  HEX5-4 = 処理数の百位。LEDR[0] = 完了、LEDR[9:1] = 一致数の下位
- **`issp`**: probe から done / 処理数 / 一致数 / サイクル数(32bit)を読み、
  設計式との合同(mod 2^32)で厳密照合してから cyc/推論 と inf/s を出す
- 期待値 ROM は**ラベルではなく golden の予測**なので、一致数 == 枚数が
  「実シリコンが参照シムと bit 一致」の判定になる(精度は golden.json 側)

### 検証の辺(オーバーレイ)

```
ライブラリ ir.simulate(正解オラクル)
   ⇔ overlay RTL シム(check の run_overlay_check。サイクル式も同時照合)
   ⇔ 実機(issp: 一致数・サイクル・inf/s)
計算型: RTL が発行する配線アドレス ⇔ Python 参照列(run_addrgen_check、全数照合)
保存型: .mem の記録を復号 ⇔ IR の (op, a, b)(run_record_check、全数照合)
```

## MNIST 実機イグザンプル(ROM→パイプライン推論→7セグ)

学習済み 24K ゲート MNIST モデル(社内先行実験から取込済み、
`examples/mnist/mnist24k.ir.json`、**取込時に既知精度 938/1000 と完全一致を検証済み**)を、
**ROM(テスト1000枚)→ 各層FFのパイプライン推論(latency 7clk)→ argmax → 正解カウント →
7セグ表示** の自走システムとして DE0-CV に焼く。

```bash
# 一本道(生成 → 3枚スモークsim → Quartus合成 → 書込コマンド提示)
uv run python tools/mdl-fpga/examples/mnist/build_mnist_demo.py
uv run python tools/mdl-fpga/cli.py flash <表示された.sofパス>
```

- **実機の見方**: HEX3-0 = 正解数(10進4桁、期待 **0938**)、HEX5-4 = 処理数の百位(10)。
  SW[0] 上 = 低速モード(毎秒~10枚、カウントアップが目視できる)、KEY[0] = リスタート、
  全1000枚処理後はホールド表示。LEDR[9]=完了 / [0]=ハートビート
- **7セグの数字がそのまま精度の物理証明**: golden(ir.simulate)と bit 一致するはずの値が
  実シリコンで再現される(0938 = 93.8%)
- **システム全体の可視化**(ROM/カウンタ/FSM 込み):

```bash
uv run python tools/mdl-fpga/cli.py view tools/mdl-fpga/examples/mnist/mnist24k.ir.json \
    --system tools/mdl-fpga/runs/mnist_demo/quartus
```

  回路図に ROM index / ラベル / argmax / 正解カウンタ / FSM がシステムノードとして現れ、
  STEP(1clk)で画像が層を流れ、カウンタが増える様子をプローブで観測できる
  (入力は ROM 供給のため手描きパネルは無効化。24K ゲートの vvp は ~1s/clk なので
  クリックステップ観賞向き)。コア単体のパイプライン可視化は `view <model> --pipelined`

## THIRD-PARTY

[THIRD-PARTY.md](THIRD-PARTY.md) 参照(UART/フレーム HDL の出典、外部ツールの扱い)。
