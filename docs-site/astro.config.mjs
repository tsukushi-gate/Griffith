// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsTxt from 'starlight-llms-txt';
import tailwindcss from '@tailwindcss/vite';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkBasePath } from './src/plugins/remark-base-path.mjs';

// 配信先は環境変数で切り替える(既定値は GitHub Pages の https://tsukushi-gate.github.io/Griffith/)。
//
//   - カスタムドメイン配信 (例 https://docs.griffith.example.jp/)
//       DOCS_SITE_URL=https://docs.griffith.example.jp   (DOCS_BASE_PATH は未設定=`/`)
//   - GitHub Pages のプロジェクトページ (https://<org>.github.io/Griffith/)
//       DOCS_SITE_URL=https://<org>.github.io  DOCS_BASE_PATH=/Griffith
//
// `base` を付けるとサイト内リンク・画像・アセットの先頭に自動でパスが挿さる。
// Markdown 中のルート相対リンク(`/guides/foo/`)にも remarkBasePath で同じ前置を行う。
const SITE_URL = process.env.DOCS_SITE_URL || 'https://tsukushi-gate.github.io';
// 末尾スラッシュを落として正規化する(`/` 指定 = base なしと同義)
const RAW_BASE = process.env.DOCS_BASE_PATH || '/Griffith';
const BASE_PATH = RAW_BASE === '/' ? undefined : RAW_BASE.replace(/\/+$/, '');

// https://astro.build/config
export default defineConfig({
	site: SITE_URL,
	base: BASE_PATH,
	// 数式: remark-math で $...$ / $$...$$ を解析し rehype-katex でHTML化(ビルド時レンダリング)
	markdown: {
		processor: unified({
			// remarkBasePath は本文のルート相対リンクに `base` を前置する(base なしでは no-op)
			remarkPlugins: [remarkMath, [remarkBasePath, { base: BASE_PATH ?? '' }]],
			rehypePlugins: [rehypeKatex],
		}),
	},
	vite: {
		plugins: [tailwindcss()],
		// mermaid は Head.astro の動的 import からしか参照されないため、dev では初回アクセス時に
		// 依存が再最適化され、その回の import が 504 (Outdated Optimize Dep) で落ちる
		// (症状=図がコードブロックのまま。build 出力では起きない)。起動時に事前バンドルしておく。
		optimizeDeps: { include: ['mermaid'] },
	},
	integrations: [
		starlight({
			title: 'Griffith',
			description: '高速・省電力で動くAIチップを、Pythonから開発する。微分可能論理ゲートNNを学習しIR経由でVerilog/VHDLへ。',
			// ルートロケール=日本語、/en/ を英語サブロケールに割り当てる
			defaultLocale: 'root',
			locales: {
				root: { label: '日本語', lang: 'ja' },
				en: { label: 'English', lang: 'en' },
			},
			// 未翻訳ページは日本語へフォールバック(Starlightの既定挙動を明示)
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/tsukushi-gate/Griffith',
				},
			],
			// コードブロックの配色。発色の良いテーマをダーク/ライトの対で指定する
			// (枠・背景はテーマ由来のまま。装飾は足さない)
			expressiveCode: {
				themes: ['one-dark-pro', 'one-light'],
			},
			customCss: ['./src/styles/global.css', 'katex/dist/katex.min.css'],
			components: {
				// ダークモードを既定にする(OS設定より優先。ユーザ選択は localStorage 優先)
				ThemeProvider: './src/components/ThemeProvider.astro',
				ThemeSelect: './src/components/ThemeSelect.astro',
				// Mermaid のクライアントサイドレンダリングを注入する
				Head: './src/components/Head.astro',
				// LP専用のカスタムヒーロー(hero frontmatter を持つページのみ通る)
				Hero: './src/components/Hero.astro',
				// トップナビゲーション付きのヘッダー
				Header: './src/components/Header.astro',
			},
			// Pagefind はビルド時に自動で有効(既定)。明示しておく
			pagefind: true,
			plugins: [
				// LLM がサイト全体をそのまま読めるよう /llms.txt・/llms-full.txt を生成する。
				// 英語ロケールは日本語ページのフォールバックなので、ルート(日本語)のみを対象にする。
				starlightLlmsTxt({
					projectName: 'griffith-logic',
					description:
						'微分可能論理ゲートNN(DLGN)を PyTorch で学習し、IR(JSON)経由で Verilog/VHDL を生成して FPGA へ載せるためのライブラリ。',
					details: [
						'- 学習時は連続緩和(soft)で微分可能、推論時は argmax 離散化(hard)で純粋な Boolean 回路になる。',
						'- IR(JSON)が真実源。PyTorch hard eval ⇔ IR simulate ⇔ HDL シミュレーションの3段一致をテストで保証している。',
						'- ドキュメントの本文は日本語。/en/ 配下は未翻訳ページを日本語へフォールバックする。',
					].join('\n'),
					exclude: ['en/**'],
				}),
				starlightLinksValidator({
					errorOnRelativeLinks: false,
					// hero.actions(frontmatter)のリンクは remark を通らないため、実際の描画では
					// Hero.astro が base を前置している。バリデータは frontmatter の生値を見るので、
					// サブパス配信のときだけこの 2 本を除外する(base なしでは通常どおり検証する)。
					exclude: BASE_PATH
						? ['/getting-started/quickstart/', '/en/getting-started/quickstart/']
						: [],
					// 未翻訳ページは日本語へフォールバックする方針のため、/en/ 配下への
					// リンクをリンク切れ扱いにしない
					errorOnFallbackPages: false,
				}),
			],
			sidebar: [
				{
					label: 'はじめに',
					translations: { en: 'Getting Started' },
					items: [
						{ label: 'インストール', translations: { en: 'Installation' }, slug: 'getting-started/installation' },
						{ label: 'クイックスタート', translations: { en: 'Quickstart' }, slug: 'getting-started/quickstart' },
						{ label: '基本コンセプト', translations: { en: 'Core Concepts' }, slug: 'getting-started/concepts' },
					],
				},
				{
					label: 'チュートリアル',
					translations: { en: 'Tutorials' },
					// 23本あるので4群+基本にサブグループ化する。「基本」を最初に読み、
					// あとはモデル種別ごとに好きな順で辿れる構成(各ページ末尾に「次に読む」あり)。
					items: [
						// 節の入口。実行済み24本のサムネイル一覧から入れる
						{ label: 'ギャラリー', translations: { en: 'Gallery' }, slug: 'tutorials' },
						{
							// NN未経験の読者向けの入口。PyTorch の基礎から入り、同じ問題を
							// DLGN へ置き換えるところまでを3本で通す(「基本」の手前に置く)
							label: '入門コース',
							translations: { en: 'Intro Course' },
							items: [
								{ label: '入門1. NNとPyTorchの基礎', translations: { en: 'Intro 1. NN & PyTorch Basics' }, slug: 'tutorials/nn-basics' },
								{ label: '入門2. PyTorchでMLPを組む', translations: { en: 'Intro 2. Build an MLP in PyTorch' }, slug: 'tutorials/pytorch-mlp' },
								{ label: '入門3. MLPをDLGNへ置き換える', translations: { en: 'Intro 3. Replace the MLP with a DLGN' }, slug: 'tutorials/mlp-to-dlgn' },
							],
						},
						{
							label: '基本',
							translations: { en: 'Basics' },
							items: [
								// 主役は MNIST(実データで「論理ゲートの塊が数字を読む」)。
								// XOR は仕組みを最小で覗く付録として末尾に置く。
								// ラベルには一般的なモデル名(MLP / CNN / RNN / Transformer …)を先頭近くに含める。
									// 読者は一般名で探すため、通称だけのラベルだと「RNNのチュートリアルが無い」と
									// 誤認される(DOCS-6)。対応表は guides/roadmap にある
									{ label: '1. MNISTを学習する(MLP)', translations: { en: '1. Train on MNIST (MLP)' }, slug: 'tutorials/mnist' },
								{ label: '2. ハイブリッドNN(実数+論理)', translations: { en: '2. Hybrid NN (Float + Logic)' }, slug: 'tutorials/hybrid' },
								{ label: '3. HDLへエクスポート', translations: { en: '3. Export to HDL' }, slug: 'tutorials/hdl-export' },
								{ label: '4. デプロイする', translations: { en: '4. Deploy' }, slug: 'tutorials/deploy-mnist' },
								{ label: '5. CNNのHDLエクスポート', translations: { en: '5. Export a CNN to HDL' }, slug: 'tutorials/cnn-hdl-export' },
								{ label: '6. 学習則(BP以外: DFA/AP)', translations: { en: '6. Learning Rules beyond BP (DFA/AP)' }, slug: 'tutorials/learning-rules' },
								{ label: '7. 通しチュートリアル', translations: { en: '7. End-to-End Walkthrough' }, slug: 'tutorials/full-pipeline' },
								{ label: '付録: XORで仕組みを覗く(最小の回路)', translations: { en: 'Appendix: XOR Internals (Smallest Circuit)' }, slug: 'tutorials/xor' },
							],
						},
						{
							label: '系列・言語',
							translations: { en: 'Sequence & Language' },
							items: [
								{ label: 'Transformer(論理ゲートGPT)', translations: { en: 'Transformer (Logic GPT)' }, slug: 'tutorials/logic-gpt' },
								{ label: 'RNNでseq2seq(RDDLGN)', translations: { en: 'RNN seq2seq (RDDLGN)' }, slug: 'tutorials/rddlgn-seq2seq' },
								{ label: 'RNN: LSTM / GRU', translations: { en: 'RNN: LSTM / GRU' }, slug: 'tutorials/logic-lstm' },
								{ label: 'WaveNet(因果CNN・TCN)', translations: { en: 'WaveNet (Causal CNN / TCN)' }, slug: 'tutorials/logic-wavenet' },
								{ label: 'リザバー計算(学習しないRNN)', translations: { en: 'Reservoir Computing (Untrained RNN)' }, slug: 'tutorials/logic-reservoir' },
								{ label: 'Transformerを段階的に論理化', translations: { en: 'Logic Ladder: Transformer Step by Step' }, slug: 'tutorials/logic-ladder' },
							],
						},
						{
							label: '視覚',
							translations: { en: 'Vision' },
							items: [
								{ label: 'U-Net(セグメンテーション)', translations: { en: 'U-Net (Segmentation)' }, slug: 'tutorials/logic-unet' },
								{ label: 'ResNet(残差接続)', translations: { en: 'ResNet (Residual Connections)' }, slug: 'tutorials/logic-resnet' },
								{ label: 'MLP-Mixer(token mixing)', translations: { en: 'MLP-Mixer (Token Mixing)' }, slug: 'tutorials/logic-mixer' },
								{ label: 'DWN(二値NN・温度計符号化)', translations: { en: 'DWN (Weightless NN, Thermometer)' }, slug: 'tutorials/dwn' },
							],
						},
						{
							label: '生成モデル',
							translations: { en: 'Generative Models' },
							items: [
								{ label: '拡散モデル(Diffusion)', translations: { en: 'Diffusion Model' }, slug: 'tutorials/logic-diffusion' },
								{ label: 'VAE(二値潜在オートエンコーダ)', translations: { en: 'VAE (Binary-Latent Autoencoder)' }, slug: 'tutorials/logic-vae' },
								{ label: 'GAN(敵対的生成)', translations: { en: 'GAN (Adversarial Generation)' }, slug: 'tutorials/logic-gan' },
							],
						},
						{
							label: '応用',
							translations: { en: 'Applications' },
							items: [
								{ label: 'Attentionの4変種(Transformer部品)', translations: { en: 'Attention Variants (Transformer Block)' }, slug: 'tutorials/attention-tour' },
								{ label: '物体検出(CenterNet)', translations: { en: 'Object Detection (CenterNet)' }, slug: 'tutorials/detection' },
								{ label: 'BinaryBrain互換API(二値NN)', translations: { en: 'BinaryBrain-compatible API (Binary NN)' }, slug: 'tutorials/binarybrain' },
								{ label: 'TextCNN(テキスト分類)', translations: { en: 'TextCNN (Text Classification)' }, slug: 'tutorials/textcnn' },
								{ label: '行動認識(UCI-HAR、スマホ加速度)', translations: { en: 'Activity Recognition (UCI-HAR, Phone Accelerometer)' }, slug: 'tutorials/har' },
							],
						},
					],
				},
				{
					// 神経進化(NEAT)で論理ゲート回路を進化させたデモ動画集。動画は GitHub Release から配信する
					label: 'ショーケース',
					translations: { en: 'Showcase' },
					items: [
						{ label: '一覧', translations: { en: 'Overview' }, slug: 'showcase' },
						{ label: 'スーパーマリオ 1-1', translations: { en: 'Super Mario 1-1' }, slug: 'showcase/mario' },
						{ label: '生き物とロボット', translations: { en: 'Creatures & Robots' }, slug: 'showcase/creatures' },
						{ label: 'ゲームと制御', translations: { en: 'Games & Control' }, slug: 'showcase/games' },
						{ label: '仕組み', translations: { en: 'How It Works' }, slug: 'showcase/how-it-works' },
					],
				},
				{
					label: 'ガイド',
					translations: { en: 'Guides' },
					items: [
						// 節の入口
						{ label: 'ガイド一覧', translations: { en: 'All Guides' }, slug: 'guides' },
						// 最初の分岐(何を選ぶか)と、PyTorch経験者の橋渡しを手順の前に置く
						{ label: '学習ロードマップ', translations: { en: 'Learning Roadmap' }, slug: 'guides/roadmap' },
						{ label: '何を使えばいいのか', translations: { en: 'What Should I Use?' }, slug: 'guides/choosing' },
						{ label: 'PyTorchからの移行', translations: { en: 'Coming from PyTorch' }, slug: 'guides/from-pytorch' },
						{ label: '学習を回す', translations: { en: 'Training' }, slug: 'guides/training' },
						{ label: '学習曲線の読み方', translations: { en: 'Reading Training Curves' }, slug: 'guides/training-curves' },
						{ label: 'パラメータ化を選ぶ', translations: { en: 'Parameterization' }, slug: 'guides/parameterization' },
						{ label: 'IRを扱う', translations: { en: 'Working with IR' }, slug: 'guides/ir' },
						// guides/hdl は誘導ページ化されたので、サイドバーは hdl-export を正とする
						{ label: 'HDLエクスポート', translations: { en: 'HDL Export' }, slug: 'guides/hdl-export' },
						{
							// 学習済みモデルを実機・ブラウザ・GPU・FPGAへ載せる手順をまとめた群
							label: 'デプロイ',
							translations: { en: 'Deploy' },
							items: [
								{ label: 'デプロイキット クイックスタート', translations: { en: 'Deploy Kit Quickstart' }, slug: 'guides/deploy-quickstart' },
								{ label: '自分のモデルをデプロイする', translations: { en: 'Deploy Your Own Model' }, slug: 'guides/deploy-your-model' },
								{ label: 'デモモデルを用意する', translations: { en: 'Prepare Demo Models' }, slug: 'guides/deploy-models' },
								{ label: 'ESP32に載せる', translations: { en: 'Run on ESP32' }, slug: 'guides/deploy-esp32' },
								{ label: 'iPhoneに載せる', translations: { en: 'Run on iPhone' }, slug: 'guides/deploy-ios' },
								{ label: 'ブラウザで動かす', translations: { en: 'Run in the Browser' }, slug: 'guides/deploy-web' },
								{ label: 'GPU(CUDA / Metal)で動かす', translations: { en: 'Run on GPU (CUDA / Metal)' }, slug: 'guides/deploy-gpu' },
								{ label: 'FPGAに焼く', translations: { en: 'Deploy to FPGA' }, slug: 'guides/fpga' },
								{ label: '圧縮と最適化の一覧', translations: { en: 'Compression and Optimizations' }, slug: 'reference/deploy-optimizations' },
								// 機器ごとの対応状況(リファレンス側の実体へのリンク)
								{ label: '対応プラットフォーム一覧', translations: { en: 'Platform Support' }, slug: 'reference/platform-support' },
							],
						},
						{ label: 'ビジュアルエディタ', translations: { en: 'Visual Editor' }, slug: 'guides/editor' },
						{ label: '回路と配線の可視化', translations: { en: 'Circuit & Wiring Visualization' }, slug: 'guides/wiring-visualization' },
						{ label: 'クラウドGPU', translations: { en: 'Cloud GPU' }, slug: 'guides/cloud-gpu' },
						// 詰まったときの索引なので末尾
						{ label: 'トラブルシューティング', translations: { en: 'Troubleshooting' }, slug: 'guides/troubleshooting' },
					],
				},
				{
					label: '解説',
					translations: { en: 'Explanation' },
					items: [
						// 節の入口
						{ label: '解説の歩き方', translations: { en: 'Overview' }, slug: 'explanation' },
						// 看板記事
						{ label: 'DLGNとは', translations: { en: 'What is DLGN?' }, slug: 'explanation/dlgn' },
						// NN経験ゼロ〜PyTorch経験者の橋渡し(入門コースの読み物版)
						{ label: 'NNからDLGNへ', translations: { en: 'From NN to DLGN' }, slug: 'explanation/nn-to-dlgn' },
						{ label: 'AIの基礎', translations: { en: 'AI Basics' }, slug: 'explanation/ai-basics' },
						{ label: 'DLGNの分類', translations: { en: 'DLGN Taxonomy' }, slug: 'explanation/taxonomy' },
						{ label: 'アーキテクチャ全体像', translations: { en: 'Architecture' }, slug: 'explanation/architecture' },
						{ label: 'LutLayerの数理', translations: { en: 'LutLayer Internals' }, slug: 'explanation/lut-layer' },
						{ label: '学習則(BP/AP/DFA/ES)', translations: { en: 'Learning Rules' }, slug: 'explanation/learning-rules' },
						{ label: '設計判断の記録', translations: { en: 'Design Decisions' }, slug: 'explanation/design-decisions' },
						{
							// 「なぜそう決めたのか」の根拠になる実測。理屈の直後・FAQの前に置く
							label: '実績・実験結果',
							translations: { en: 'Results & Experiments' },
							items: [
								{ label: '論文再現と改善', translations: { en: 'Paper Reproduction' }, slug: 'explanation/results/dlgn-repro' },
								{ label: 'パラメータ化を決めた実験', translations: { en: 'Parameterization Study' }, slug: 'explanation/results/parameterization-study' },
								{ label: '離散化ギャップ', translations: { en: 'Discretization Gap' }, slug: 'explanation/results/discretization-gap' },
								{ label: 'アーキテクチャ横断の到達点', translations: { en: 'Architecture Survey' }, slug: 'explanation/results/architecture-survey' },
								{ label: '世界初主張の監査', translations: { en: 'World-First Claims Audit' }, slug: 'explanation/results/world-firsts' },
								{ label: '実験ブランチのモデル群', translations: { en: 'Experimental Models' }, slug: 'explanation/results/experimental-models' },
							],
						},
						{ label: 'よくある質問', translations: { en: 'FAQ' }, slug: 'explanation/faq' },
						{ label: '用語集', translations: { en: 'Glossary' }, slug: 'explanation/glossary' },
					],
				},
				{
					label: 'リファレンス',
					translations: { en: 'Reference' },
					items: [
						// 節の入口
						{ label: 'リファレンス索引', translations: { en: 'Reference Index' }, slug: 'reference' },
						{
							// scripts/gen_api_docs.py が生成する reference/api/ 配下を自動収集する
							// (frontmatter の sidebar.order が _SYMBOL_MODULES の並び順に対応)
							label: 'Python API',
							items: [{ autogenerate: { directory: 'reference/api' } }],
						},
						{ label: 'ベンチマーク', translations: { en: 'Benchmarks' }, slug: 'reference/benchmarks' },
						{ label: 'モデル一覧(動物園)', translations: { en: 'Model Zoo' }, slug: 'reference/model-zoo' },
						{ label: 'griffith CLI', slug: 'reference/cli' },
						{ label: 'IRスキーマ', translations: { en: 'IR Schema' }, slug: 'reference/ir-schema' },
						{ label: '対応プラットフォーム', translations: { en: 'Platform Support' }, slug: 'reference/platform-support' },
						{ label: '配布バンドル仕様', translations: { en: 'Deploy Bundle Spec' }, slug: 'reference/deploy-bundle' },
						{ label: '圧縮と最適化の一覧', translations: { en: 'Compression and Optimizations' }, slug: 'reference/deploy-optimizations' },
						// reference/glossary は誘導ページ。正本は解説節の用語集(explanation/glossary)なので
						// 同名ラベルが2箇所に並ばないよう、サイドバーからは外す(ページ自体は残す)。
					],
				},
				{
					label: 'コントリビュート',
					translations: { en: 'Contributing' },
					items: [
						{ label: '開発をはじめる', translations: { en: 'Get Started' }, slug: 'contributing' },
						{ label: 'テスト体系(L1〜L5)', translations: { en: 'Test Layers' }, slug: 'contributing/testing' },
						{ label: 'ドキュメントを書く', translations: { en: 'Writing Docs' }, slug: 'contributing/docs' },
						{ label: 'ドキュメント管理', translations: { en: 'Docs Management' }, slug: 'contributing/docs-management' },
					],
				},
			],
		}),
	],
});
