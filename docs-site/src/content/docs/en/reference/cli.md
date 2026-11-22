---
title: 'griffith CLI'
description: 'Subcommands of the unified `griffith` CLI (train / eval / export-hdl / deploy / editor / sweep / launch / collect / runs / report).'
---

`griffith` is the unified CLI of griffith-logic. Training, evaluation, HDL export, deployment (C/CUDA/Metal/Web), launching the editor
and experiment orchestration (sweep / launch / collect / runs / report) all run from this single
entry point.

It is declared under `[project.scripts]` in `pyproject.toml`
(`md = "griffith_logic.cli:main"`). After `uv sync`, invoke it as `uv run md ...`. The
implementation (`griffith_logic/cli.py`) is nothing but argparse subcommand dispatch; the logic of
each subcommand lives in modules under `griffith_logic/training/`.

```bash
uv run griffith --help
```

## Subcommands

| Subcommand | Description | Implementation |
|---|---|---|
| `train` | Run training (config YAML + dot-notation overrides) | `training/config.py`, `training/runner.py` |
| `eval` | Run hard eval (discretized evaluation) on a checkpoint | `training/runner.py` |
| `export-hdl` | Generate Verilog/VHDL from an IR (JSON) | `export/verilog.py`, `export/vhdl.py` |
| `deploy` | Bundle a trained model, then generate and verify C/CUDA/Metal/Web code (9 subcommands) | `deploy/cli.py` |
| `editor` | Start the visual editor backend (requires a clone) | uvicorn |
| `sweep` | Create a wandb sweep and run an agent | `training/sweep.py` |
| `launch` | Submit a grid of experiments at once (Cartesian product) | `training/launch_cli.py` |
| `collect` | Retrieve run directories under `outputs/` from a Modal Volume | `training/launch_cli.py` |
| `runs list` / `runs compare` | List and compare wandb runs | `training/runs_cli.py` |
| `report` | Generate an experiment report (wandb, with a markdown fallback) | `training/report_cli.py` |
| `deploy` | Build a distribution bundle, ship it to C/CUDA/Metal/Web, verify equivalence | `deploy/cli.py` |

For the full set of flags of any subcommand, run `uv run md <subcommand> --help` (this page lists
only the main ones).

## griffith train

Runs training through the shared runner. Configuration comes from a YAML file plus dot-notation
overrides in `key=value` form.

| Flag | Description |
|---|---|
| `config` (positional, optional) | Path to the config YAML. Omitted means defaults; a `key=value` argument is treated as an override |
| `overrides` (positional, variadic) | Dot-notation overrides (e.g. `model.k=8000 run.epochs=5`) |
| `--from-run DIR` | Re-run using the `config.yaml` inside an existing run directory |
| `--out-dir DIR` | Output directory (default `outputs/<timestamp>_<run name>`) |
| `--strict` | Exit non-zero when `expected_metrics` are not met |

```bash
uv run griffith train configs/recipes/dlgn_mnist_quick.yaml
uv run griffith train configs/recipes/dlgn_mnist_quick.yaml optim.lr=0.02 model.k=4000
uv run griffith train --strict configs/recipes/dlgn_mnist_quick.yaml
```

`expected_metrics` turns success criteria into numbers via the `<metric>_min` / `<metric>_max` keys
of the recipe YAML (see [`TrainConfig`](/en/reference/api/training-config/#trainconfig)). They are
checked automatically when the runner finishes and stored in the result (not meeting them does not
raise). Only with `--strict` does an unmet criterion produce a non-zero exit — useful for CI and
automated flows.

## griffith eval

Loads a checkpoint saved by `griffith train` and reports the accuracy of the discretized circuit (hard
acc). The model structure is restored from the configuration stored in the checkpoint.

| Flag | Description |
|---|---|
| `checkpoint` (positional) | Path to the checkpoint (`.pt`) |
| `--dataset` | Evaluation dataset name (defaults to the training-time setting in the checkpoint) |
| `--batch-size` | Evaluation batch size |
| `--device` | Device (auto-detected when omitted) |

```bash
uv run griffith eval outputs/20260702_dlgn_mnist_quick/best.pt
```

## griffith export-hdl

Validates an IR (intermediate representation) JSON and transpiles it into Verilog/VHDL.

| Flag | Description |
|---|---|
| `source` (positional) | Path to the IR JSON file |
| `--lang {verilog,vhdl}` | HDL language to emit (default: `verilog`) |
| `-o, --output` | Output path (defaults to `.v` / `.vhdl` next to the input) |

```bash
uv run griffith export-hdl outputs/xor/ir.json --lang verilog
uv run griffith export-hdl outputs/xor/ir.json --lang vhdl -o build/xor.vhdl
```

It does not run the simulator comparison (iverilog / ghdl). For the procedure including that check,
see the [HDL equivalence guide](/en/guides/hdl/).

## griffith deploy

Packs a trained model (IR) into a **distribution bundle common to all targets**, generates C / CUDA / Metal / Web code,
and verifies it against `ir/simulate.py` with integer-exact agreement. By design, optimization is finished at bundling time and the target side only executes.
For the end-to-end procedure see the [Deploy Kit Quickstart](/en/guides/deploy-quickstart/), and for the bundle format see the
[Deploy bundle spec](/en/reference/deploy-bundle/).

```bash
uv run griffith deploy --help
```

| Subcommand | Description | Output |
|---|---|---|
| `bundle` | Model JSON (webbench format) → deploy bundle v1 (bakes in the conv2d instruction lists) | `.bundle.json` / `.bundle.mdlb` |
| `codegen-c` | Bundle → self-contained C source (for embedded and mobile) | `model.c` / `model.h` + runtime |
| `codegen-cuda` | Bundle → CUDA source (for NVIDIA GPUs) | `model_cuda.cu` + runtime |
| `codegen-metal` | Bundle → Metal source (for Apple GPUs) | `model_metal.mm` + runtime |
| `codegen-web` | Bundle → Web (browser) artifacts (with a copy of the runtime) | bundle + `runtime/web/dist` |
| `verify-c` | Build the generated C and verify integer-exact agreement with IR simulate | match / mismatch (exit code) |
| `verify-metal` | Build the generated Metal and verify integer-exact agreement with IR simulate (macOS + Apple GPU) | same |
| `verify-web` | Verify the Web runtime (`verify.mjs` on Node) against IR simulate with integer-exact agreement | same |
| `info` | Validate a bundle and print a summary of its contents | stdout |

CUDA has no matching `verify-cuda` (code generation only). For the verification status see [Platform support](/en/reference/platform-support/).

### griffith deploy bundle

| Flag | Description |
|---|---|
| `source` (positional) | Path to the model JSON (webbench format). The minimum is `{"ir": <IR>}` |
| `-o, --output` | Output path (defaults to `.bundle.json` next to the input) |
| `--format {json,mdlb}` | Output format. Defaults to auto-detection from the output path's extension (`.mdlb` means the binary format) |
| `--gzip` | Also write a gzip version to measure the delivery size |
| `--compress {none,fast,max}` | Level of automatic lossless circuit compression (default `fast`). Same meaning for every model kind. `none` = no IR-rewriting optimization; `fast` = every lossless reduction that finishes in seconds (dense `lut_layer` chains: sweep; conv2d trees: resynthesis, search limit 4, verified by exhaustive per-tree truth-table equality and forward computation on real samples); `max` = `fast` + search limit 5 and additional ir/simulate verification (minutes on real models). Instruction-list conversion of conv2d trees happens at every level (IR unchanged). The result is recorded in `about.compress` and shown by `info`. Individual flags take priority over `--compress` |
| `--resynth lossless` | Force resynthesis of the conv2d trees with simulate verification (accuracy unchanged; default search limit 5). Resynthesis also runs automatically with the default `--compress fast` |
| `--resynth-max-cost N` / `--resynth-verify-samples N` | Upper bound on gates searched during resynthesis (default depends on the level: `fast` 4, `max` / explicitly given 5) / number of random inputs used for verification (default 8) |
| `--sweep` | Sweep the dense (`lut_layer` chain) IR to remove constant, duplicate and unreferenced gates (lossless, integer-exact verification included). It is applied automatically by the default `fast`, but `--sweep` forces it even with `--compress none` and rejects IRs containing conv (use `--resynth` for conv) |
| `--no-sweep` | Do not run the automatic sweep (mutually exclusive with `--sweep`) |
| `--sweep-verify-samples N` | Number of random inputs used to verify the sweep (default 8; real data is checked too when `demo_samples` exist) |
| `--dense-flat` | Replaces the 2-input `lut_layer` chain of a dense model with a flat netlist (a `lut_flat` node) and **bakes it into the bundle's IR** (compatibility path). Lossless. The baked `lut_flat` runs on C / Web / CUDA / Metal alike (Web and GPU). By default nothing is baked; `codegen-c` / `codegen-web` (and `verify-c` / `verify-web`) decide automatically in memory at generation time (below; CUDA / Metal default to `off`, opt-in). The order of application is resynth → sweep → dense-flat |
| `--dense-flat-verify-samples N` | Number of random inputs used by the `--dense-flat` equivalence check (default 16; real data is checked too when `demo_samples` exist) |
| `--no-verify` | Skip the exhaustive equivalence check of the instruction lists (for debugging; not normally used) |

```bash
uv run griffith deploy bundle out_examples/xor_model.json -o out_examples/xor.bundle.json
```

The result of the compression is recorded in `about.compress` (`{level, applied, skipped, gates, always}`); the verification mode of resynthesis in `about.resynth.verify` (`fast` / `simulate`); the gate counts before/after of sweep in `about.sweep`; and with `--dense-flat`, the reduction breakdown (gate counts before/after, constants, pass-throughs, CSE)
is recorded in the IR's `metadata.dense_flat`. `griffith deploy info` prints both. For the
`lut_flat` specification see section 2.6 of the [IR schema](/en/reference/ir-schema/).

### griffith deploy info

| Flag | Description |
|---|---|
| `source` (positional) | Path to the bundle (`.json` / `.json.gz` / `.mdlb`) |
| `--strict` | Rebuild the instruction lists from the IR and confirm an exact match |

### griffith deploy codegen-c

| Flag | Description |
|---|---|
| `source` (positional) | Path to the bundle |
| `-o, --output DIR` (required) | Output directory |
| `--name NAME` | Base name for generated files (default: `model`) |
| `--no-runtime` | Do not copy the `runtime/c` kernels (when already linked) |
| `--fuse` | Expand the gate instruction list into straight-line code (conv models are measured 2 to 9% faster, at the cost of generated C 1.2 to 2.3 times larger) |
| `--dense-batch N` | Add the batch-axis-packed path `mdl_infer_batch_packed` to dense-only models (up to N images per call; ignored for models containing conv). Works together with `--dense-flat`, and **defaults to 32 when flattening is applied** (`0` suppresses it). Not emitted by default for dense models that are not flattened. At 32 images it is 1.7 to 5.9 times faster than one-at-a-time `lut_flat` (HAR k600 to k6000, MNIST; Apple M3). A single image is 5 to 12 times slower than `mdl_infer`, so use it for batches only |
| `--dense-flat {auto,on,off}` | Whether to run the dense circuit as a flat netlist (`lut_flat`) at C-generation time (default `auto`). The bundle is unchanged; flattening happens in memory. `auto` = try it and adopt if the effective-gate reduction is at or above the threshold, `on` = always adopt, `off` = the conventional bit-slice kernel. Whether it was adopted and the reduction appear on standard output, in the top comment of `model.h`, and as the `MDL_DENSE_FLAT` macro. Working RAM grows by the signal table, so use `off` if RAM comes first |
| `--dense-flat-min-reduction R` | Adoption threshold of `--dense-flat auto` (effective-gate reduction, default 0.25; the host-measured break-even is about 26%) |

```bash
uv run griffith deploy codegen-c out_examples/xor.bundle.json -o out_examples/xor_c
```

### griffith deploy codegen-cuda / codegen-metal

The two have almost the same flags (only the default of `--name` differs: `model_cuda` / `model_metal`).

| Flag | Description |
|---|---|
| `source` (positional) | Path to the bundle |
| `-o, --output DIR` (required) | Output directory |
| `--name NAME` | Base name for generated files |
| `--no-runtime` | Do not copy the runtime (`runtime/cuda` and `runtime/gpu` for CUDA, `runtime/metal` for Metal) |
| `--batch N` | Maximum number of images processed at once (default 1). On a GPU the batch is directly the parallelism, so larger is faster |
| `--n-streams N` | Pipeline across streams (default 1 = as before). With 2 or more, buffer sets are duplicated and submit / collect APIs are added |
| `--conv-engine {split,combined}` | Specify how conv2d is executed. If omitted, automatic selection decides |
| `--conv-codegen` | Generate a model-specialized kernel per conv2d node (default OFF; code size grows 1.78 to 6.65 times) |
| `--dense-fuse` | Fuse consecutive dense layers and the trailing group_sum into one kernel |
| `--dense-packed` | Repack the batch axis of dense models into bit planes of 32 samples per word |
| `--dense-flat {auto,on,off}` | Whether to generate dense as a flattened netlist (`lut_flat`) (default `off` = do not flatten; opt-in). `auto` flattens when the reduction is at or above the threshold and runs level by level in parallel. No flattening when `--dense-fuse` / `--dense-packed` is explicit. A large model (tens of thousands of gates) run at batch 128 or more was measured at 0.47-0.71x throughput, so compare with the default before choosing |
| `--dense-flat-min-reduction R` | Adoption threshold of `--dense-flat auto` (effective-gate reduction, default 0.25) |
| `--no-auto` | Disable automatic selection of optimizations (a table of measured decisions; default ON) |

```bash
uv run griffith deploy codegen-cuda out_examples/xor.bundle.json -o out_examples/xor_cuda
uv run griffith deploy codegen-metal out_examples/xor.bundle.json -o out_examples/xor_metal
```

### griffith deploy codegen-web

| Flag | Description |
|---|---|
| `source` (positional) | Path to the bundle |
| `-o, --output DIR` (required) | Output directory |
| `--name NAME` | Base name for generated files (default: `model`) |
| `--no-runtime` | Do not copy `runtime/web/dist` (when already being served) |
| `--slim` | Write the webbench model doc itself rather than the bundle (gives up the reduction guarantee, so not normally used) |
| `--gzip` | Also write a gzip version to measure the delivery size |
| `--dense-flat {auto,on,off}` | Whether to ship dense as a flattened netlist (`lut_flat`) (default `auto`). On the Web the threshold is 0: it always flattens when there is a chain of 2-input `lut_layer`s (it is faster per gate than the legacy path too) |
| `--dense-flat-min-reduction R` | Adoption threshold of `--dense-flat auto` (default 0) |

### griffith deploy verify-c / verify-metal / verify-web

| Flag | Description |
|---|---|
| `source` (positional) | Path to the bundle |
| `-n, --samples N` | Number of verification samples (default 3) |
| `--seed SEED` | Random seed (default 0) |
| `--engine {js,wasm}` | `verify-web` only. The Web runtime engine (default `js`; run both before shipping) |
| `--conv-codegen` / `--dense-fuse` | `verify-metal` only. Verifies the corresponding generation option |
| `--dense-batch N` | `verify-c` only. Generates the same batch-axis packed entry `mdl_infer_batch_packed` as `codegen-c` and checks it in addition to the one-at-a-time `mdl_infer` (default: same as `codegen-c` = 32 only when flattening is adopted, `0` = do not generate). The check uses 33 or more samples, crossing the 32-samples-per-word boundary and the split at the entry's limit |
| `--dense-flat {auto,on,off}` / `--dense-flat-min-reduction R` | All three. Generates with the same dense-flat decision as the matching `codegen-*` (default `auto` for `verify-c` / `verify-web` and `off` for `verify-metal`; the `auto` threshold is 0.25 for C and Metal, 0 for the Web) and verifies it. `verify-web` also checks all cases run as one batch. The golden is **`simulate` of the original bundle's IR**, so this demonstrates that the flattening is lossless |

```bash
uv run griffith deploy verify-c out_examples/xor.bundle.json -n 8 --seed 1
uv run griffith deploy verify-web out_examples/xor.bundle.json --engine wasm
```

## griffith editor

:::note[A clone is required]
`griffith editor` imports and starts the editor's backend implementation
(`tools/editor/backend/`). The editor is not shipped inside the PyPI package, so this subcommand
only works in a clone where you have run `uv sync --extra editor`.
:::

Starts the editor backend (FastAPI) with uvicorn. It does not start the frontend, so use
`make editor` when you want backend and frontend running together for development.

| Flag | Description |
|---|---|
| `--host` | Host to bind (default: `127.0.0.1`) |
| `--port` | Port number (default: `8765`) |

```bash
uv run griffith editor
make editor   # starts backend:8765 + frontend:5173 together
```

## griffith sweep

Creates a wandb sweep from a `sweep.yaml` (`{method, metric, parameters, base_config}`) and runs an
agent.

| Flag | Description |
|---|---|
| `spec` (positional) | Path to the sweep YAML |
| `--count COUNT` | Number of trials to run (defaults to the full grid / unlimited) |
| `--local` / `--modal` | Where to run (mutually exclusive, default: `--local`) |
| `--parallel N` | Number of parallel agents with `--modal` (default: 1; `--count` is split N ways) |

```bash
uv run griffith sweep configs/sweeps/example_mnist.yaml --count 4 --local
uv run griffith sweep configs/sweeps/example_mnist.yaml --count 8 --modal --parallel 2
```

## griffith launch

Expands the list values in `--grid` into every combination (Cartesian product) and submits training
for each.

| Flag | Description |
|---|---|
| `--grid SPEC` (required) | Sweep specification (e.g. `"optim.lr=[0.001,0.01] model.k=[64,256]"`; scalar values act as fixed overrides) |
| `config` (positional, optional) | Base config YAML |
| `--local` / `--modal` | Where to run (mutually exclusive, default: `--local`) |
| `--dry-run` | Only print the expanded combinations (do not run) |

```bash
uv run griffith launch --grid "optim.lr=[0.001,0.01] model.k=[64,256]" \
    configs/recipes/dlgn_mnist_quick.yaml --dry-run
uv run griffith launch --grid "optim.lr=[0.001,0.01]" \
    configs/recipes/dlgn_mnist_quick.yaml --modal
```

## griffith collect

Downloads run directories under `outputs/` in the Modal Volume that are not present locally yet.
The volume is selected with the `MD_MODAL_VOLUME` environment variable (a default name is used when
it is unset).

| Flag | Description |
|---|---|
| `--dest DIR` | Destination directory (default: `outputs/collected`) |

```bash
uv run griffith collect
```

## griffith runs list / griffith runs compare

Lists and compares runs using `.wandb_runs.jsonl` and the wandb public API.

`runs list`:

| Flag | Description |
|---|---|
| `-n N` | Number of entries to show (default: 20) |

`runs compare`:

| Flag | Description |
|---|---|
| `run_ids` (positional, variadic) | wandb run IDs to compare |
| `--metrics` | Metrics to compare (comma-separated; all summary keys when omitted) |
| `--format {md,csv}` | Output format |

```bash
uv run griffith runs list -n 10
uv run griffith runs compare <run_id1> <run_id2> --metrics hard_acc,best_hard_acc,loss
```

## griffith report

Generates an experiment report. When `wandb-workspaces` (the `cloud` extra) is installed it creates
a wandb report (training curves + config diff + a conclusions section) and prints its URL.
Otherwise it writes markdown into `reports/auto/` (change the destination with `--dest`).

| Flag | Description |
|---|---|
| `run_ids` (positional, variadic, required) | Target wandb run IDs |
| `--title` | Report title |
| `--dest DIR` | Destination for the markdown fallback (default: `reports/auto`) |

```bash
uv run griffith report <run_id> --title "lr×k掃引(MNIST 3エポック)"
```

## Modal integration

With `--modal`, the Modal functions in `scripts/modal_app.py` are spawned internally.

- `griffith launch --modal` — `.spawn()`s `train_config` for each expanded combination and waits for all
  of them
- `griffith sweep --modal` — spawns `sweep_agent` `--parallel` times in parallel (the trial count from
  `--count` is split N ways)

The GPU type is selected with the `MODAL_GPU` environment variable (default A10G). For
authentication, image setup and artifact retrieval, see the
[cloud GPU guide](/en/guides/cloud-gpu/).

## Relationship to recipes and sweep definitions

- `configs/recipes/*.yaml` — the base configs read by `griffith train` / `griffith launch`. The YAML
  representation of [`TrainConfig`](/en/reference/api/training-config/#trainconfig)
  (`data` / `model` / `optim` / `run` / `expected_metrics`)
- `configs/sweeps/*.yaml` — the grid/distribution definitions read by `griffith sweep`
  (`method` / `metric` / `parameters` / `base_config`). Keys under `parameters` are dot paths into
  `TrainConfig`
- `expected_metrics` (`<metric>_min` / `<metric>_max`) is checked by `check_expected_metrics()`
  (`griffith_logic/training/config.py`), and `griffith train --strict` turns it into a gate for
  automated flows

## Related pages

- [Training](/en/guides/training/) — where this sits within the whole experiment loop
- [Cloud GPU](/en/guides/cloud-gpu/) — running on Modal / RunPod
- [Deploy Kit Quickstart](/en/guides/deploy-quickstart/) — the end-to-end procedure for `griffith deploy`
- [Python API](/en/reference/api/) — the public API the CLI uses internally

<!-- i18n:source=reference/cli.md hash=c6303248678a -->
