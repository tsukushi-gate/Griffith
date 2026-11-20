---
title: 'Design decisions'
description: 'Why the IR is the source of truth, why the reference implementation stays naive, and five more decisions that keep paying off — recorded ADR-style.'
---

This page records the **major design decisions** that shaped this library, written as public ADRs
(Architecture Decision Records). Each entry is built on three lines —
**problem / decision / rationale** — with consequences added where they matter.

"What the structure is" belongs in [Architecture overview](/en/explanation/architecture/), and
"how to write it" belongs in the repository's `CONTRIBUTING.md`. What lives here is only
**why it was decided that way**.

| # | Decision | In one line |
|---|---|---|
| [1](#adr-1-the-ir-is-the-source-of-truth) | The IR is the source of truth | keeps training and output targets orthogonal |
| [2](#adr-2-the-reference-implementation-is-ground-truth-and-is-never-rewritten-for-speed) | The reference implementation is ground truth | its naivety *is* the verification value |
| [3](#adr-3-logic-gate-conversion-is-scoped-and-everything-around-it-stays-real-valued) | Everything around stays real-valued | don't turn the whole model into logic gates |
| [4](#adr-4-n2-softmax-is-the-default-held-as-a-16-gate-table) | N=2 softmax as the default | a known reproduction point as the baseline |
| [5](#adr-5-residual-initialization-is-on-by-default-in-lutconv2d) | Conv defaults to residual init ON | push signal through deep trees |
| [6](#adr-6-dependencies-are-managed-with-uv-extras-and-pytorch-is-pushed-into-one) | Dependencies managed with uv extras | CPU/CUDA/GPU from one definition |
| [7](#adr-7-tests-are-split-into-five-layers-l1-to-l5) | Tests split into L1–L5 | match the kind of change to the kind of test |

---

## ADR-1: The IR is the source of truth

- **Problem** — wiring the training framework (PyTorch) directly to output targets (Verilog / VHDL /
  editor / code generation) means every change to one side breaks the other, and the combinations
  explode as targets are added.
- **Decision** — freeze the meaning of a trained model into an **IR (JSON)**, and have everything
  downstream look at the IR **only**. The IR belongs to neither the training side nor the downstream
  side; it is a shared contract. Node granularity is **per layer**, not expanded to individual gates
  (the gate view is derived from `indices` and `truth_tables`).
- **Rationale** — swap the training framework and everything downstream of the IR is unchanged; add
  an HDL target and the training side is unchanged. Gates are not expanded because even an
  MNIST-scale model reaches tens of thousands of nodes, breaking file size and editability.
- **Consequence** — `export/`, the visual editor, and the `griffith` CLI all became single consumers of the
  same IR. That a model built in the editor and a model written in Python are **saved in the same
  form** follows from this decision.

## ADR-2: The reference implementation is ground truth, and is never rewritten for speed

- **Problem** — `LutLayer`'s backward is optimized with custom autograd and CUDA kernels, so a
  mistake in the algebra still lets the loss go down "plausibly". Something has to define what
  correct means.
- **Decision** — keep a **naive pure-autograd implementation** in `core/reference.py` and treat it
  as ground truth. The optimized version proves itself with float64 `gradcheck` and agreement with
  the reference to 1e-10 (`tests/l2_grad/`). **The reference implementation is never rewritten for
  speed.**
- **Rationale** — the naivety itself is the verification value. The more the reference gets
  optimized, the likelier it is that both sides make the same mistake, and the comparison stops
  meaning anything.
- **Consequence** — when adding a new custom autograd Function, always write the comparison test
  against the reference first (see the "Test layers" section of CONTRIBUTING).

## ADR-3: Logic-gate conversion is scoped, and everything around it stays real-valued

- **Problem** — trying to convert an entire existing PyTorch model into logic gates also breaks the
  parts where being continuous is the point — embeddings, normalization, attention — and training
  never gets off the ground.
- **Decision** — convert **block by block**, and **place an encoder / decoder at each boundary**.
  Surrounding real-valued layers stay floating point. The boundary options are collected in
  `LogicSpec` (encoder = `sigmoid` / `binarize_ste` / `thermometer` / `identity`,
  decoder = `groupsum_centered` / `groupsum` / `layernorm` / `identity`).
- **Rationale** — the benefit of going to hardware lives in the compute-heavy core; forcing the
  periphery to 0/1 gains little and loses much. With the boundary construction gathered into one
  dataclass, "how far to take logic gates" becomes a sweepable experiment parameter.
- **Consequence** — Attention / Transformer are treated as **blackboxes** in the editor (not
  trainable, no HDL generation, the Python model is authoritative for execution). That isn't a
  limitation but the UI reflecting this policy.

## ADR-4: N=2 softmax is the default, held as a 16-gate table

- **Problem** — there are five parameterizations, and whichever becomes the default determines the
  behavior a first-time user sees. N=2 can be written either as "a softmax over $2^N=4$ entries" or
  as "a softmax over 16 gate types".
- **Decision** — default to `parameterization='softmax'`, and **only for N=2 make the internal
  representation 16-gate logits `(out_dim, 16)`** (`_use_16gate`). N>2 uses the $2^N$-entry table.
- **Rationale** — the 16 gates at N=2 are equivalent to difflogic (NeurIPS 2022), which makes them a
  **baseline that reproduces the paper's numbers**. On top of that, a single $\arg\max$ can select
  any of the 16 Boolean functions, so at N=2 no discretization gap can arise in principle.
- **Consequence** — conversely, **softmax at N>2 carries the known weakness that the hard table
  degenerates into "a single minterm or FALSE"**. For hardware deployment at N=4/6, choose
  `independent` / `iwp` / `warp`
  (→ [LutLayer internals §5](/en/explanation/lut-layer/),
  [Choosing a parameterization](/en/guides/parameterization/)).

## ADR-5: Residual initialization is ON by default in LutConv2d

- **Problem** — a logic-gate tree fans out to $N^d$ leaves at depth $d$. With random initialization
  the early tree crushes the input and gradients never reach the far end.
- **Decision** — `LutConv2d` sets `residual_init=True` by **default**, putting a logit of 5.0 on the
  pass-through entry alone ($\mathrm{softmax}([5,0,0,0]) \approx [0.993, \dots]$). `LutLayer`, by
  contrast, defaults to OFF and is enabled explicitly when needed.
- **Rationale** — depth only appears structurally on the Conv side. Biasing the initial state toward
  "pass the first input through" starts the network near the identity map, so signal and gradient
  get through (CDDLGN §3.2). Applying the same default to the fully connected `LutLayer` would mean
  carrying an unnecessary initial bias all the time.
- **Consequence** — the differing defaults between layers are deliberate. `iwp` structurally cannot
  have this initialization, so specifying it is ignored with a warning.

## ADR-6: Dependencies are managed with uv extras, and PyTorch is pushed into one

- **Problem** — PyTorch ships different wheels for CPU, CUDA, and MPS, and no single dependency
  specification satisfies Mac dev machines, CPU CI, and GPU boxes at once. Forcing the editor and
  plotting dependencies on everyone also makes setup heavy.
- **Decision** — standardize on **uv** for package management and split every heavy dependency,
  PyTorch included, into an extra (`cpu` / `cu128` / `dev` / `editor` / `figures`). Setup runs
  through `uv sync --extra ...`; `uv pip install` and activating `.venv` directly are not used.
  Python is always launched via `uv run`.
- **Rationale** — combinations of extras absorb the environment differences, and one lock file
  suffices. `uv run` is mandatory because it structurally prevents the accident of producing
  experimental results in a drifted environment.
- **Consequence** — `cpu` and `cu128` **cannot be specified together** (mutually exclusive). The
  `editor` extra only contains FastAPI/uvicorn, so using the editor still requires `--extra cpu`
  (or `cu128`).

## ADR-7: Tests are split into five layers, L1 to L5

- **Problem** — "is the math right", "is the gradient right", "does the paper reproduce", "does
  training converge", and "does it match the HDL" differ completely in what a failure means and what
  it costs to run. Mixed into one suite, the slow tests stop the fast ones from running.
- **Decision** — split the tests into five layers and **match the kind of change to the test to
  write**. The default fast suite is limited to L1/L2/L4/L5, and the heavy layers are isolated
  behind markers.

  | Layer | Location | Contents | Run |
  |---|---|---|---|
  | L1 | `tests/l1_math/` | mathematical properties (partition of unity, truth tables, discretization spec) | default |
  | L2 | `tests/l2_grad/` | gradients (float64 gradcheck + reference comparison) | default |
  | L3 | `tests/l3_paper/` | paper reproduction (thresholds calibrated from measurements) | `-m paper` |
  | L4 | `tests/l4_learning/` | training convergence (**hard eval 100%** to pass) | default |
  | L5 / HDL | `tests/l5_hybrid/`, `tests/hdl/` | hybrid integration / three-way equivalence | default / `-m hdl` |

- **Rationale** — keeping the fast suite green at all times (under 2 minutes) was the top priority.
  L4 passes on **hard eval 100%** rather than soft because that is what stops the essential failure
  — "solvable under continuous relaxation, unsolvable as a Boolean circuit" — from slipping past.
- **Consequence** — L3 thresholds are **never guessed; they are measured and then fixed**, with the
  calibration history left in the docstring. Loosening a test threshold to match the implementation
  is written down as prohibited.

---

## The stance underneath the decisions

Two things run through all seven decisions.

- **Put the standard of correctness outside the fast thing** — the reference implementation (ADR-2),
  the hard-eval criterion (ADR-7), the three-way equivalence chain. Optimized code can only be
  called correct once it is checked against something unoptimized.
- **Collect boundaries into one place** — the IR (ADR-1), `LogicSpec` (ADR-3), extras (ADR-6).
  Wherever combinations could explode, put a single contract in between so both sides can move
  independently.

## What to read next

- [Architecture overview](/en/explanation/architecture/) — how the decisions show up as actual structure
- [LutLayer internals](/en/explanation/lut-layer/) — the mathematical background of ADR-2 / ADR-4 / ADR-5
- [Test structure (L1–L5)](/en/contributing/testing/) — how ADR-7 is run in practice
- [Getting started with development](/en/contributing/) — the setup procedure from ADR-6

<!-- i18n:source=explanation/design-decisions.md hash=398e13131c91 -->
