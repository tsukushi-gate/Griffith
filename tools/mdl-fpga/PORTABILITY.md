# PORTABILITY — Win以外(Linux/macOS)への移植メモ

**更新: 下記「修正が必要な箇所」1〜4は実装済み**(コミット `06a0de8`、
回帰は tests/test_portability_toolchain.py)。5の表示文言は doctor のみ修正済み
(COMx 表記は残置)。残るのは各 OS での実機検証(「検証手順」参照)。

調査。現状のコードは設計レベルではほぼクロスプラットフォーム
(純Python+pathlib+os.pathsep、subprocessはリストargv、cocotbは公式ランナー、
GUI↔シム橋は127.0.0.1 TCP、UARTはpyserial)。
**Windows決め打ちは `mdl_fpga/toolchain.py` のツール探索レイヤに集中**しており、
下記の小修正+各OSでの実機検証で移植できる見込み。

## 修正が必要な箇所

### 1. `toolchain.py` `_oss_exe()` — `.exe` 決め打ち(toolchain.py:74)

```python
if self.oss_bin and (self.oss_bin / f'{name}.exe').exists():
```

- 非Windowsでは oss_bin 配下の実行体(拡張子なし)を見ずに `shutil.which(name)`
  へフォールバックする。which は**親プロセスのPATH**しか見ないため、
  local.toml / 環境変数で oss_root を指定するスタイルが Linux/macOS では
  実行体解決に繋がらない。
- 修正: `.exe` 付きに加えて拡張子なしパス(`self.oss_bin / name`)もチェックする。

### 2. `toolchain.py` `quartus()` — 同じく `.exe` 決め打ち(toolchain.py:104)

```python
p = self.quartus_bin / f'{exe}.exe'
```

- Linux版Quartusの実行体には `.exe` が無い。現状はPATH経由(which)でのみ解決可能。
- 修正: 拡張子なしパスのチェックを追加(1.と同型)。

### 3. `toolchain.py` 既知パス候補がドライブレター(toolchain.py:22-31)

- `_QUARTUS_CANDIDATES` は `C:\altera_lite\...` 等のWindowsパスのみ。
- `_OSS_CAD_CANDIDATES` も `C:\oss-cad-suite` / `D:\oss-cad-suite`
  (`Path.home() / 'oss-cad-suite'` だけはポータブル)。
- 修正: Linux向け候補を追加(例: `~/oss-cad-suite`, `/opt/oss-cad-suite`,
  `~/intelFPGA_lite/*/quartus/bin` 等)。env/local.toml/PATH 経由は既に
  プラットフォーム中立なので、候補追加は利便性のみ。

### 4. `verilator_bin` という実行体名(toolchain.py:88)

```python
return self._oss_exe('verilator_bin')
```

- Windows版 oss-cad-suite のレイアウト前提。Linux版ではラッパースクリプト
  `verilator` が正で、`verilator_bin` では見つからない可能性あり(**要実機確認**)。
- 修正案: `verilator_bin` → 見つからなければ `verilator` の順で探す。

### 5. 表示文言の微修正(機能には影響なし)

- `doctor.py:107` — FAILメッセージが「`{name}.exe` が無い」とWindows前提の文言。
- `hw_check.py:94` / `cli.py:156,252` — 「COMx」「COM ポート」表記
  (pyserial は Linux では `/dev/ttyUSB*`、macOS では `/dev/cu.*` を返す。
  機能は動くが文言がWindows前提)。

## そのままで問題ない箇所(修正不要)

- `session.py:32-44` `_ensure_sim_path()` の bin+lib PATH前置は
  WindowsのDLL解決(0xC0000139)対策だが、他OSでも無害。
- cocotb橋(`viewer/cocotb_bridge.py`)・TCP JSON行プロトコル・
  cocotb公式ランナー呼び出し(`session.py:195-211`)はポータブル。
- pyserial の CP210x 自動検出(VID/PIDベース)はOS非依存。
- `bundle.py` は既に「EDA無しPC(Windows/macOS)で書込」を想定した
  README同梱設計(bundle.py:3,191)。

## プラットフォーム別の見通し

| OS | 対話シム/ビューア | 合成 | 書込 |
|---|---|---|---|
| Linux | ○(上記1-4の修正後) | ○ Quartus Linux版あり | ○ openFPGALoader(Windowsのドライバ排他問題が消える) |
| macOS | ○(同上) | × Quartusネイティブ版なし → bundle経由で書込専用PC運用 | ○ openFPGALoader |
| (全OS) | — | **P2のyosys+nextpnr(apycula)が実現すればoss-cad-suiteのみで合成まで完結** | — |

## 検証手順(修正後にやること)

1. Linux/macOS で `uv sync --extra dev --extra cpu --extra fpga`
2. oss-cad-suite を導入し、(a) PATH経由 (b) local.toml指定 の両方で
   `python cli.py doctor` が iverilog/vvp/verilator/openFPGALoader を解決すること
3. ビューア起動 → cocotb対話ステップが動くこと(M1相当のスモーク)
4. Linux: Quartus導入環境で build→flash→hw-check(M2相当)
5. Tang Nano 20K の実機で: openFPGALoader 書込経路(yosys+nextpnr バックエンドの実装後)
