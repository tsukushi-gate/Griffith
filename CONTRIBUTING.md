# Contributing to Griffith

Thanks for your interest in Griffith (`griffith-logic`). This page is the entry point for people
who touch the code: set up an environment, write a change, get the tests green, open a PR.

Issues and pull requests may be written in **English or Japanese** — whichever you are more
comfortable with. The same goes for commit messages.

日本語版の解説は
[ドキュメントサイトの「開発をはじめる」](https://tsukushi-gate.github.io/Griffith/contributing/)
にあります。

## License and sign-off

Griffith is licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)). By contributing, you agree that your contribution is licensed under the same
terms.

We use the [Developer Certificate of Origin](DCO) (DCO) instead of a CLA. **Every commit must be
signed off**, which certifies that you have the right to submit the work:

```bash
git commit -s -m "fix: guard against empty IR nodes"
```

`-s` appends a line like `Signed-off-by: Your Name <you@example.com>` using your configured
`user.name` / `user.email`. If you forgot on the last commit, `git commit -s --amend` fixes it; for
several commits, `git rebase --signoff <base>` does the whole range.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[`SECURITY.md`](SECURITY.md), **not** the public issue tracker.

## Development environment

Griffith uses [uv](https://docs.astral.sh/uv/). Always go through `uv run`; never activate
`.venv` directly, and select dependencies with extras plus `uv sync` rather than `uv pip install`.

```bash
uv sync --extra dev --extra cpu   # macOS (MPS) / Linux CPU. On a CUDA machine: --extra cu128
uv run pytest -q                  # fast suite (the default; it must stay green)
uvx ruff check .                  # lint
```

`cpu` and `cu128` are conflicting extras — pick exactly one.

Some suites need external tools: `-m hdl` requires [Icarus Verilog](https://steveicarus.github.io/iverilog/)
(`iverilog`), and `-m hdl_vhdl` requires [GHDL](https://ghdl.github.io/ghdl/). Both are skipped
automatically when the tool is missing.

### pre-commit

Enable the commit-time gate once (ruff, mypy on the core modules, trailing whitespace / newline,
private-key detection):

```bash
uvx pre-commit install
uvx pre-commit run --all-files   # to check everything manually
```

It runs in the same uv environment as `make lint` / `make type`, so a green hook means green CI
static analysis.

## Test layers (write them in this order)

| Layer | Location | What it covers | How to run |
| --- | --- | --- | --- |
| L1 | `tests/l1_math/` | Mathematical properties (partition of unity, truth tables, discretization spec) | default |
| L2 | `tests/l2_grad/` | Gradients (`gradcheck` in float64 + agreement with `core/reference.py`) | default |
| L3 | `tests/l3_paper/` | Paper reproduction, with empirically calibrated thresholds | `-m paper` |
| L4 | `tests/l4_learning/` | Convergence smoke tests (**hard eval 100%** is the bar) | default |
| L5 | `tests/l5_hybrid/` | Hybrid integration (gradient flow, replacement APIs) | default |
| HDL | `tests/hdl/` | Verilog / VHDL simulation agreement | `-m hdl`, `-m hdl_vhdl` |

Principles:

- **New model-level and integration tests go in `tests/integration/`.** The flat files directly
  under `tests/` are historical; do not move them.
- **Every new custom autograd `Function` needs a test against a naive reference implementation**
  in the style of `griffith_logic/core/reference.py`. Do not "optimize" the reference — its
  naivety is what makes it ground truth.
- **Never loosen a threshold, tolerance or golden file to make a test pass.** Calibrate thresholds
  from measurements and record the calibration in the test's docstring.
- HDL work must preserve the three-way equivalence chain: PyTorch hard eval ⇔ `ir.simulate` ⇔ HDL
  simulation.

Tolerances live in `tests/conftest.py`; reference them instead of hard-coding new constants.

Adding a whole new layer or autograd `Function`? Write the reference implementation first, then
L2, then L1, then L4 — in that order, tests before implementation.

## Coverage (ratchet)

Coverage is measured on the fast suite:

```bash
uv run pytest --cov=griffith_logic --cov-report=term
```

CI gates it with `--cov-fail-under`, and the floor is always **"current baseline − 1", rounded to
an integer** (e.g. baseline 74.4% → floor 73). When coverage rises, raise the floor to the new
baseline − 1; it never goes back down. PRs that lower coverage need an explicit justification.

## Commit conventions

Commit messages follow **Conventional Commits**, which is what drives the changelog and release
automation:

```
<type>: <summary>
```

- **Types**: `feat` (feature) / `fix` (bug fix) / `perf` / `refactor` / `docs` / `test` /
  `build` / `ci` / `chore`.
- `feat` and `fix` appear in the changelog; a `!` after the type (`feat!:`) or a
  `BREAKING CHANGE:` footer marks a breaking change.
- The summary may be **English or Japanese**. Keep it in the imperative and under ~72 characters.
- Examples: `feat: add butterfly-basis switch to LutLayer` /
  `fix: LutConv2d の softmax16g 構造バグを修正`

Before committing, the fast suite and ruff should be green.

## Coding conventions

- Comments and docstrings explain the **why**: the formula, the spec, the source of a constant —
  not a line-by-line restatement of the code. They may be written in English or Japanese.
- Identifiers (variables, functions, classes) are in English.
- Follow the existing style: parameterization naming, the butterfly-basis bit convention, and so on.
- New public symbols go into `_SYMBOL_MODULES` in `griffith_logic/__init__.py` (lazy import).
- Backwards compatibility: when a module moves, leave a re-export shim at the old path and add a
  case to `tests/test_backcompat.py`.
- Type safety matters; avoid `Any` where a concrete type will do.

## Opening a pull request

1. Branch from the default branch and keep the change focused — unrelated refactors make review
   harder and are often the reason a PR stalls.
2. Make sure `make test` and `make lint` pass locally.
3. Update the docs alongside the code: docstrings, the guides, and the tutorials on the
   [documentation site](https://tsukushi-gate.github.io/Griffith/) all live in this repository.
4. Fill in [the PR template](.github/PULL_REQUEST_TEMPLATE.md) — it asks for a summary, the type of
   change, what you tested, whether docs changed, and the DCO sign-off.
5. CI runs the fast suite, lint, type checks and the docs build. Green CI plus one maintainer
   approval is the merge bar.

If you are planning something large (a new model family, an IR schema change, a new HDL backend),
please open an issue first so we can agree on the shape before you write it.
