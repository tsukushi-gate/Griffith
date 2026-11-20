# 標準タスクランナー。人間とAIエージェント共通の入口。
# コマンドの正はこのファイル(CONTRIBUTING.md はここを参照する)

.PHONY: sync test test-all cov lint type hdl editor backend front front-test front-lint front-build e2e gen-types modal-test clean web-runtime

# 素の `uv run` は lock の既定(extras 無し)へ再同期するため、`--extra cpu`/`cu128` で入れた torch が
# PyPI 版に入れ替わり torchvision と不整合になる(`operator torchvision::nms does not exist`)。
# 環境は `make sync`(または extras 付き `uv sync`)で作り、各ターゲットは同期せずに実行する。
# CI(.github/workflows/ci.yml)と同じ流儀。
UV_RUN := uv run --no-sync

sync:
	uv sync --extra dev --extra cpu

test:  ## 高速スイート(常時グリーン基準: 2分以内。learning は CI と同じく除外 → test-all)
	$(UV_RUN) pytest -m "not slow and not cuda and not paper and not hdl and not hdl_vhdl and not learning" -q

test-all:  ## slow・learning 含む(hdl/cuda/paperは除外)
	$(UV_RUN) pytest -m "not cuda and not paper and not hdl and not hdl_vhdl" -q

cov:  ## カバレッジ計測(基線は開発記録参照)
	$(UV_RUN) pytest -m "not slow and not cuda and not paper and not hdl and not hdl_vhdl and not learning" \
		--cov=griffith_logic --cov-report=term -q

lint:
	uvx ruff check .

type:  ## 型チェック(core/encodingのみ厳格化の段階導入)
	$(UV_RUN) mypy griffith_logic/core griffith_logic/encoding

train:  ## 統一CLIで学習(例: make train ARGS="configs/xxx.yaml optim.lr=0.02")
	$(UV_RUN) griffith train $(ARGS)

sweep:  ## wandb sweep(例: make sweep ARGS="configs/sweeps/example_mnist.yaml --count 4")
	$(UV_RUN) griffith sweep $(ARGS)

launch:  ## グリッド投入(例: make launch ARGS='--grid "optim.lr=[0.01,0.02]" --dry-run')
	$(UV_RUN) griffith launch $(ARGS)

collect:  ## Modal Volumeから成果物回収
	$(UV_RUN) griffith collect

runs:  ## 最近のrun一覧
	$(UV_RUN) griffith runs list

report:  ## 実験レポート生成(例: make report ARGS="<run_id> --title '...'")
	$(UV_RUN) griffith report $(ARGS)

hdl:  ## HDL等価性(要iverilog/ghdl)
	$(UV_RUN) pytest -m "hdl or hdl_vhdl" -q

backend:  ## エディタバックエンドのみ
	$(UV_RUN) uvicorn tools.editor.backend.main:app --port 8765

front:  ## フロント開発サーバのみ
	cd tools/editor/frontend && npm run dev

editor:  ## バックエンド+フロントを同時起動(Ctrl-Cで両方止まる)
	make -j2 backend front

front-test:
	cd tools/editor/frontend && npm test

front-lint:
	cd tools/editor/frontend && npm run lint

front-build:
	cd tools/editor/frontend && npm run build

webbench:  ## ブラウザ推論比較ツール(W4)。devサーバ起動
	cd tools/webbench && npm run dev

webbench-test:
	cd tools/webbench && npm test

webbench-serve:  ## build+gzip配信(ペイロード実測用、port 8787)
	cd tools/webbench && npm run build && npm run serve

web-runtime:  ## Webランタイム(runtime/web/dist)をビルド(griffith deploy codegen-webが使う)
	npm --prefix tools/webbench run build:runtime

e2e:  ## Playwright E2E(要: uv sync --extra editor / npx playwright install chromium)
	cd tools/editor/frontend && npm run e2e

gen-types:  ## エディタAPI契約のTS型生成(生成物はsrc/api/generated/)
	cd tools/editor/frontend && npm run gen-types

modal-test:  ## GPU必須テストをModalで実行
	$(UV_RUN) modal run scripts/modal_app.py::pytest_cuda

