# paper-tree — 論文・実験レポートのツリー管理アプリ

論文(落合陽一式要約)と実験レポートを、React Flow のツリーで一元管理・閲覧するための
独立フロントエンド(Vite + React 19 + TS + Tailwind 4)。データは `data/**.md` の frontmatter が真実源で、
`scripts/build-tree.mjs` が `src/data/tree.json` へ変換する。

公開版に同梱するデータはフォーマット見本(`data/papers/ptree-sample-*.md`)だけで、論文ノートは
各自 `data/papers/` に追加する。

## 起動

```bash
cd tools/paper-tree
npm install                  # 初回のみ(Playwrightを使うなら npx playwright install chromium も)
npm run dev                  # http://localhost:5175 (tree.jsonを再生成してからvite起動)
npm run build                # 検査(strict) → tsc → vite build
npm run e2e                  # Playwrightスモーク(専用ポート6175でdevを起動)
node scripts/build-tree.mjs  # tree.jsonの再生成のみ(検査に失敗するとexit 1)
```

`npm run dev` は `--allow-dangling` 付きで tree.json を作る(他レーンが `data/` を書いている途中でも
起動できるようにするため)。`npm run build` は素の strict 検査なので、CI/受入はこちらで判定する。

## 論文・レポートの追加手順

1. `data/papers/<id>.md`(論文)または `data/reports/<id>.md`(レポート)を1枚置く。
   frontmatter のスキーマは `data/papers/ptree-sample-paper.md`(落合式フル)、
   `data/papers/ptree-sample-stub-2025.md`(スタブ)を雛形にする。
   **この見本(`00_sample` カテゴリ)はフォーマット見本であり、
   実在の論文/レポートではない。削除せず、内容の正確さも問わない**(e2eスモークの固定アンカーでもある)。
2. `node scripts/build-tree.mjs` を実行して検査を通す。
3. `npm run dev` で表示を確認する。

id は kebab-case(論文スラッグと揃える)。ファイル名は `<id>.md` を推奨(不一致は警告)。
レポート id は `rep:` 接頭辞を付ける(付け忘れは補完のうえ警告)。

論文mdの本文は落合陽一式7節(1.どんなもの? / 2.先行研究と比べてどこがすごい? / 3.技術のキモ /
4.どうやって有効だと検証した? / 5.議論はあるか? / 6.次に読むべき論文 / 論文情報・リンク)で書く。
**数値は必ず出典のある値だけ**を書く(創作禁止)。「6. 次に読むべき論文」は frontmatter の
`next:` と対応させる(`next` が破線矢印エッジになる)。

### 論文↔レポートの紐付け

論文md の `reports: [rep:xxx]` と レポートmd の `papers: [paper-id]` は**どちらに書いてもよい**
(build時に「論文 → レポート」向きへ正規化して重複を畳む)。片側だけ書けば十分。

### 重複論文の一本化(`superseded_by`)

同じ論文が別idで二重登録されたときは、**ファイルを消さずに** frontmatter で畳む:

```yaml
# data/papers/matmul-free-lm-2024.md(重複側)
id: matmul-free-lm-2024
superseded_by: matmulfree-lm-2024   # 正本のid
```

- 重複側は**ノードを生成しない**(ツリーから消える)。ファイルと本文は残るので履歴を追える
- 他ファイルが持つ重複idへの参照(`parents` / `next` / `reports` / report側の `papers`)は
  **自動で正本idへ付け替わる**(連鎖も辿る。循環・自己参照・参照先不在は exit 1)
- 運用手順: ①正本を決める(内容が充実している方 / 論文スラッグと一致する方を優先)
  ②重複側の固有情報(節・数値・`categories`・`next`・pdf/url)を正本へマージ
  ③重複側に `superseded_by` と「統合済み」の1行引用を追加 ④`node scripts/build-tree.mjs` で検査

## build-tree.mjs の検査仕様

致命(**exit 1**。tree.json は更新しない):

| # | 検査 | 内容 |
|---|------|------|
| 1 | id重複 | papers/reports 横断で同じ id が2枚以上 |
| 2 | 必須フィールド欠落 | paper: `id` / `title` / `status`(summarized\|stub) / `categories`(非空)、report: `id` / `title` |
| 3 | 参照先不在 | `parents` / `next` / `reports`(paper)、`papers`(report)の参照先idが存在しない |
| 4 | frontmatter破損 | YAMLとして解析できない |
| 5 | `superseded_by` 不正 | 参照先が存在しない / 自己参照 / 循環 |

警告(生成は継続): ファイル名と id の不一致 / `rep:` 接頭辞の補完 / `--allow-dangling` で落としたエッジ /
`superseded_by` によるノード除外。

オプション: `--allow-dangling`(3を警告へ降格しエッジを省略) / `--out <path>` / `--data <dir>` / `--quiet`。

並行編集への耐性: 未知のカテゴリは自動でカテゴリノードを生成し、未知の frontmatter キーは `extra` へ退避する。
`parents` が全て不在に落ちた論文は、孤立させずに自分の `categories` へ退避する。

## 画面

- 左: ツリーキャンバス。root → カテゴリ → 論文/レポート。カテゴリカードの `−/+` で配下を折りたたむ
  (親が論文の場合も部分木ごと畳む)
- エッジ3種: 親子=実線(グレー) / 次に読む=**破線+矢印**(青) / レポート関連=点線(紫)
- スタブ(`status: stub`)はグレーの破線カード
- 右: 選択ノードの落合式 Markdown(react-markdown + KaTeX)、arXiv/原文リンク、PDFパス、
  `next` / `reports` のチップ(クリックでそのノードへフォーカス)
- 上: 検索(⌘K / Ctrl+K、cmdk。タイトル/id/カテゴリ/**本文**が対象)、カテゴリ・status・年フィルタ、
  レポート表示切替、全折りたたみ/全展開、テーマ切替(dark既定・localStorage保存)
- MiniMap + LOD(ズームの見かけサイズで overview / tile / full を切り替え、俯瞰時の描画量を落とす)

## レイアウト

200+ノードを `layered` に流すと「1層に200枚が縦一列」で高さ2万px超になるため、elk を2段で使う
(`src/tree/layout.ts`): ①カテゴリごとに配下を `rectpacking` で矩形ブロックへ詰める →
②「カテゴリカード+ブロック」を `rectpacking` で盤面へ敷き詰める。elk が失敗しても素朴なグリッドへ
フォールバックして描画は止めない。

## 構成

```
data/papers/*.md, data/reports/*.md  # 真実源(P1/P2レーンが量産)
scripts/build-tree.mjs               # frontmatter走査 + 検査 → src/data/tree.json(gitignore)
src/types.ts                         # tree.jsonのスキーマ
src/tree/graph.ts                    # フィルタ・折りたたみ → 可視グラフ(純関数)
src/tree/layout.ts                   # elk 2段パッキング
src/tree/TreeCanvas.tsx              # React Flow + LOD + MiniMap
src/panel/DetailPanel.tsx            # 落合式Markdownレンダリング
src/search/SearchPalette.tsx         # cmdk検索
src/Toolbar.tsx, src/store.ts        # フィルタUI / zustandストア
e2e/smoke.spec.ts                    # Playwrightスモーク(描画・クリック・検索・フィルタ・両テーマ)
```
