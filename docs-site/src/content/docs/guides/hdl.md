---
title: 'HDL等価性検証'
description: 'iverilog / ghdl の導入と3段等価性の回し方は「HDLエクスポートと等価性検証」に統合しました。'
---

:::note[移動しました]
**→ [HDLエクスポートと等価性検証](/guides/hdl-export/)**

シミュレータ(iverilog / ghdl)の導入、Verilog / VHDL の生成、
**3 段等価性(PyTorch hard eval ⇔ `ir.simulate` ⇔ HDL シミュレーション)** の回し方、
つまずきどころは、すべてそのページに 1 本化しています。
同じ手順を 2 か所に書かないため、このページは入口だけを残した誘導ページです。
:::

## 関連ページ

| 目的 | ページ |
|---|---|
| 手を動かして一度通してみる | [チュートリアル 3: HDLへエクスポート](/tutorials/hdl-export/) |
| 生成される回路の元になる IR の仕様を引く | [IRスキーマ](/reference/ir-schema/) |
| 生成した RTL を実機へ焼く | [FPGAに焼く](/guides/fpga/) |
| 等価性をテストとしてどう書くか | [テスト体系(L1〜L5)](/contributing/testing/) |
