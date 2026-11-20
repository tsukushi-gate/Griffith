# docs-site — griffith-logic ドキュメントサイト

[Astro](https://astro.build/) + [Starlight](https://starlight.astro.build/) 製の公開ドキュメントサイト。
リポジトリ内の `docs/`(開発者・AIエージェント向けの内部ドキュメント)とは役割が別で、
こちらは **外部読者向けの読み物** を Diátaxis 4分類(tutorials / guides / explanation / reference)で構成する。

## コマンド

```bash
cd docs-site
npm install
npm run dev       # 開発サーバ(http://localhost:4321)
npm run build     # 本番ビルド + Pagefind索引 + リンク検証(リンク切れがあれば失敗)
npm run preview   # ビルド結果のプレビュー(検索はこちらでのみ動く)
npm run gen:api   # APIリファレンスを再生成(dev/build の前段で自動実行される)
npm run check:api # 生成せず網羅検証だけ行う(CI向け。非0で失敗)
npm run check:i18n   # 日本語(真実源)と英語(en/)の追従(MISSING/STALE/ORPHAN)
npm run check:assets # 図の解像度・DPI・空図・参照切れ・ギャラリーのサムネイル
```

## APIリファレンスの自動生成

`src/content/docs/reference/api/` 配下(索引 + カテゴリ14ページ)は
`scripts/gen_api_docs.py` が `griffith_logic/__init__.py` の `_SYMBOL_MODULES` から
生成する**ビルド生成物**で、`.gitignore` 済み。手で編集せず、Pythonのdocstringを直す。

- 抽出は [griffe](https://mkdocstrings.github.io/griffe/) による静的解析。パッケージをimportしないので
  torchのインストールは不要(`uv run --no-project --with griffe` の隔離環境で走る)
- 依存は `pyproject.toml` の `[dependency-groups] docs`(`uv sync --group docs` でも入る)
- サイドバーは `autogenerate: { directory: 'reference/api' }`。並び順は各ページの
  frontmatter `sidebar.order`(= `_SYMBOL_MODULES` のカテゴリ順)で決まる
- `--check` は「シンボル数 = 索引の行数」「ページ数 = カテゴリ数 + 1」
  「後方互換シム経由の参照が0件」を機械検証する

## 配信先(site / base)の切り替え

`astro.config.mjs` の `site` / `base` は**環境変数から読む**。同じソースを
カスタムドメインにもサブパス配信(GitHub Pages のプロジェクトページ)にも出せる。

| 環境変数 | 未設定時 | 例 |
|---|---|---|
| `DOCS_SITE_URL` | `https://tsukushi-gate.github.io` | `https://docs.example.jp` |
| `DOCS_BASE_PATH` | `/Griffith` | `/`(= base なし) |

本文のリンクは全て `/guides/foo/` のようなルート相対で書く。Astro は本文中の
ルート相対リンクに `base` を自動付与しないため、`src/plugins/remark-base-path.mjs`
が AST 段階で前置している(`base` なしでは no-op)。frontmatter の `hero.actions`
だけは remark を通らないので `src/components/Hero.astro` 側で前置する。

サブパス配信を壊していないことは、ローカルで次を通して確認する。

```bash
DOCS_BASE_PATH=/Griffith npm run build   # リンク検証込みで緑になること
```

## GitHub Pages への配備

`.github/workflows/docs.yml` の `deploy` ジョブが配備する。動くのは
**main への push** かつ **リポジトリ変数 `IS_PUBLIC_REPO` が `true`** のときだけで、
private リポではジョブごとスキップされる(Pages は private だと Team/Enterprise が要る)。

人間側で 1 回だけ行う設定:

1. リポジトリを public にする
2. Settings → Pages → Build and deployment → **Source = "GitHub Actions"**
3. Settings → Secrets and variables → Actions → **Variables** に登録する
   - `IS_PUBLIC_REPO` = `true`
   - `DOCS_SITE_URL` = `https://<org>.github.io`(カスタムドメインならそのURL)
   - `DOCS_BASE_PATH` = `/<リポジトリ名>`(カスタムドメインなら**登録しない**)
4. main へ push すると build → 各チェック → deploy が走り、
   Actions の `github-pages` environment に公開URLが出る

## 構成

| パス | 役割 |
|---|---|
| `astro.config.mjs` | i18n(ルート=日本語 / `en` サブロケール)・サイドバーIA・プラグイン設定 |
| `src/content/docs/` | 日本語(ルートロケール)のページ |
| `src/content/docs/en/` | 英語ページ。無い場合は日本語へ自動フォールバックする |
| `src/components/` | Starlight コンポーネント上書き(Hero / Header / Head / ThemeProvider / ThemeSelect)。Header はトップナビ(ドキュメント/チュートリアル/API/DLGNとは)を追加し、72rem 未満では非表示にしてモバイルメニューへ委ねる |
| `src/styles/global.css` | Tailwind v4 + プロジェクトカラー(実数NN=橙 / 論理ゲート=緑)のテーマトークン |

## 執筆メモ

- **数式**: `$...$` / `$$...$$`。remark-math + rehype-katex でビルド時にレンダリングする
- **図**: ` ```mermaid ` コードブロック。クライアントサイドで SVG 化する(ビルドに Chromium 不要)
- **リンク**: `npm run build` の `starlight-links-validator` がリンク切れを検出してビルドを落とす。
  サイドバーへページを足したら必ず実体ページも作る
- **新規ページ**: `src/content/docs/<セクション>/<slug>.md` を作り、`astro.config.mjs` の `sidebar` に登録する
- **図の解像度・Mermaidの配色・拡大表示・翻訳の判断基準**: 規約は `npm run check:assets` /
  `npm run check:mermaid` が機械検査する(図中の文字は**英字のみ**。日本語は豆腐になる)
- **日英の同期**: 日本語(`src/content/docs/` 直下)が真実源。英語ページ(`src/content/docs/en/`)は
  末尾に `<!-- i18n: source=<相対パス> hash=<md5先頭12桁> -->` のマーカーを持つ。日本語を更新したら
  同じ PR で英語も直し、マーカーの hash を新しい md5 先頭 12 桁へ更新する
  (検査は `npm run check:i18n`。MISSING / STALE / ORPHAN のいずれでも落ちる)
- **チュートリアルの章見出し・散文**: 生成元は `examples/*.py`。コードセル先頭の `# === 章名 ===` は
  `## 章名`、`# --- 小見出し ---` は `### 小見出し`、`# > 本文` はコードフェンス外の段落になる
  (`---` と `>` は `# === … ===` を1つ以上持つ example でのみ有効)。生成された `.md` / `.ipynb` は
  手で編集しない(`scripts/build_tutorials.py` が上書きする)。再生成後のドリフトは
  `uv run python scripts/check_tutorial_drift.py` で検査する(出力ブロックの数値とPNGのバイト差は
  CPUアーキ差で揺れるため無視し、地の文・コード・出力/図の個数・図の集合を比較。CIも同じ)
