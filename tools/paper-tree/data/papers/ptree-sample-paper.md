---
id: ptree-sample-paper
title: "(見本)落合式フル要約の書き方"
date: 2022-10-15
summarized: null
status: summarized
categories: [00_sample]
arxiv: "2210.08277"
url: https://arxiv.org/abs/2210.08277
pdf: null
parents: [cat:00_sample]
next: [ptree-sample-stub-2025]
---

> このファイルは `tools/paper-tree` の**フォーマット見本**(P0の動作確認用)。
> 内容は difflogic (Petersen et al., NeurIPS 2022) を題材にしているが、
> 実在の論文ノートではなく、frontmatter と本文の書式を示す雛形。

### 1. どんなもの？

論理ゲート(AND/OR/XOR など2入力16種)からなる回路そのものを**微分可能に緩和**し、
勾配降下で「どのゲートを置くか」を直接学習する手法。学習後は各ニューロンを
`argmax` で1つのゲートへ崩壊させることで、純粋なブーリアン回路として推論できる。

### 2. 先行研究と比べてどこがすごいの？

従来の二値化ニューラルネット(BNN)や量子化手法が「行列積を保ったまま値を粗くする」
アプローチだったのに対し、本手法は**行列積そのものを捨て、論理ゲートの結線を学習対象にする**。
そのため推論は積和演算ではなくゲート評価だけで済む。

### 3. 技術や手法の"キモ"はどこにある？

各ニューロンが2入力 $a,b \in [0,1]$ に対して16種の実数値緩和ゲートを持ち、
学習可能な重み $W$ の softmax で混合する:

$$y = \sum_{g=0}^{15} \operatorname{softmax}(W)_g\, f_g(a,b)$$

緩和ゲートは実数値論理(例 $f_{AND}=ab$, $f_{OR}=a+b-ab$, $f_{XOR}=a+b-2ab$)。
層間の結線はランダム固定で、学習するのは「各ニューロンがどのゲートになるか」だけ。

### 4. どうやって有効だと検証した？

画像分類ベンチマークで精度を測り、同等精度の既存手法に対する推論スループットを比較する
——というように、**数値は必ず出典のある値だけ**を書く(創作禁止)。

### 5. 議論はあるか？

- 学習時 soft / 推論時 hard の**ギャップ**
- ランダム固定結線に由来する学習の不安定さ

### 6. 次に読むべき論文はあるか？

frontmatter の `next:` に書いたidが、ツリー上の破線矢印エッジになる(1本以上必須)。

### 論文情報・リンク

- [Felix Petersen, Christian Borgelt, Hilde Kuehne, Oliver Deussen, "Deep Differentiable Logic Gate Networks," NeurIPS, 2022](https://arxiv.org/abs/2210.08277)
