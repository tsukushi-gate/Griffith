# Griffith

**Train neural networks made of logic gates in PyTorch, then compile them to Verilog / VHDL.**

[![PyPI version](https://img.shields.io/pypi/v/griffith-logic.svg)](https://pypi.org/project/griffith-logic/)
[![PyPI downloads](https://static.pepy.tech/badge/griffith-logic/month)](https://pepy.tech/project/griffith-logic)
[![CI](https://github.com/tsukushi-gate/Griffith/actions/workflows/ci.yml/badge.svg)](https://github.com/tsukushi-gate/Griffith/actions/workflows/ci.yml)
[![Docs](https://github.com/tsukushi-gate/Griffith/actions/workflows/docs.yml/badge.svg)](https://tsukushi-gate.github.io/Griffith/en/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Named after Griffith Observatory, whose mission is to bring astronomy to everyone — we want
differentiable logic to be just as accessible.

## What is Griffith?

Griffith is a PyTorch library for **differentiable logic gate networks**. Every neuron is an
N-input lookup table (LUT) instead of a weighted sum: training relaxes the table into a
continuous mixture so gradients flow, and `model.eval()` collapses it with `argmax` into a pure
Boolean circuit that loses no accuracy.

- **N-input LUTs (N = 2, 4, 6)** — generalizes [difflogic](https://github.com/Felix-Petersen/difflogic)'s
  fixed 2-input gates all the way to FPGA-native LUT6, in a single layer type.
- **Train → IR → hardware** — extract the discretized network into a JSON **IR** (the single
  source of truth), then emit synthesizable Verilog or VHDL.
- **Deploy kit** — from one distribution bundle, generate code for C (embedded / mobile / desktop),
  CUDA, Metal and the Web (WASM). Every path is verified bit-exact against `ir/simulate.py`; the C
  runtime is dependency-free C99 at roughly 6 KB (see `griffith deploy`).
- **Pluggable learning rules** — BP, **AP (Akiho Propagation)**, DFA and EGGROLL, annotated per
  layer and mixable within one network.
- **A model zoo of logic-only architectures** — classification, sequence models, generative
  models, detection and NLP, all built out of gates.

## Install

```bash
pip install griffith-logic
```

Griffith requires **Python 3.11+** and PyTorch. `pip install griffith-logic` pulls in whatever
PyTorch wheel your platform resolves to; on a CUDA machine, install the matching PyTorch build
first (see [pytorch.org](https://pytorch.org/get-started/locally/)) and then install Griffith.
CPU and Apple Silicon (MPS) work out of the box — **no CUDA is required**, and the optional CUDA
kernels fall back to a pure-PyTorch path when they are unavailable.

Working on Griffith itself (or running the examples) uses [uv](https://docs.astral.sh/uv/)
instead — see [Development](#development).

## 30-second example

Train XOR, check that the **discretized** circuit is still perfect, then write out Verilog.

```python
import torch
import torch.nn.functional as F

import griffith_logic as gl
from griffith_logic.export import export_ir
from griffith_logic.export.verilog import write_verilog

torch.manual_seed(0)

# XOR: all four input patterns are the whole dataset
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])

# Two LUT layers (2-input gates) + a GroupSum read-out
model = torch.nn.Sequential(
    gl.LutLayer(2, 16, n_inputs=2),
    gl.LutLayer(16, 16, n_inputs=2),
    gl.GroupSum(k=2, tau=2.0),
)

opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(500):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()

# eval() collapses every gate into a single Boolean function (argmax)
model.eval()
with torch.no_grad():
    hard_acc = (model(x).argmax(-1) == y).float().mean().item()
print(f"hard accuracy: {hard_acc:.2f}")   # -> 1.00

# Freeze the discretized circuit into the IR, then emit synthesizable Verilog
ir = export_ir(model, input_width=2, name="xor")
write_verilog(ir, "xor.v")
```

The two numbers to watch are always **soft** accuracy (the relaxed mixture, where gradients live)
and **hard** accuracy (the Boolean circuit you can actually burn to an FPGA). Only the second one
is a result.

Scaling the same idea up to MNIST:

```python
import torch
import griffith_logic as gl

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    gl.LutLayer(in_dim=784, out_dim=16000, n_inputs=4),
    gl.LutLayer(in_dim=16000, out_dim=16000, n_inputs=4),
    gl.LutLayer(in_dim=16000, out_dim=16000, n_inputs=4),
    gl.GroupSum(k=10, tau=30.0),
)
y = model(torch.rand(32, 784))   # (32, 10) class logits
```

## Model zoo

A selection of the architectures shipped with Griffith. Numbers are **hard** (discretized)
results from full runs; the tutorials use CPU-sized reductions and report smaller figures.
Conditions, seeds and caveats live in
[the full model zoo](https://tsukushi-gate.github.io/Griffith/en/reference/model-zoo/) and
[benchmarks](https://tsukushi-gate.github.io/Griffith/en/reference/benchmarks/).

| Model | Task | Headline result (hard) | Docs |
| --- | --- | --- | --- |
| DLGN (baseline) | MNIST digit classification | acc **0.9784** (48,000 gates) | [MNIST](https://tsukushi-gate.github.io/Griffith/en/tutorials/mnist/) |
| Conv-DLGN | CIFAR-10 classification with LUT-tree convolutions | acc **0.5808** | [Benchmarks](https://tsukushi-gate.github.io/Griffith/en/reference/benchmarks/) |
| BinaryBrainModel | MNIST with 6-input LUTs + frame modulation | acc **0.9483** (14,516 gates) | [BinaryBrain](https://tsukushi-gate.github.io/Griffith/en/tutorials/binarybrain/) |
| DWNClassifier | Learnable wiring on toy logic tasks | 100% with learnable wiring | [DWN](https://tsukushi-gate.github.io/Griffith/en/tutorials/dwn/) |
| LogicGPT | Character-level language modelling | Tiny Shakespeare val ppl **13.31** | [LogicGPT](https://tsukushi-gate.github.io/Griffith/en/tutorials/logic-gpt/) |
| LogicLadderGPT | How much of a Transformer can be made of gates | rung L1 ppl **19.24 ± 0.94** (3 seeds) | [Logic ladder](https://tsukushi-gate.github.io/Griffith/en/tutorials/logic-ladder/) |
| RDDLGN | Sequence-to-sequence without attention | data-free distilled char-LM BPC **2.71** | [RDDLGN](https://tsukushi-gate.github.io/Griffith/en/tutorials/rddlgn-seq2seq/) |
| LogicLSTMCell / LogicGRUCell | Delayed copy (T=8), i.e. gated memory | acc **1.00**, all seeds within 200 steps | [Logic LSTM](https://tsukushi-gate.github.io/Griffith/en/tutorials/logic-lstm/) |
| LogicBinaryDiffusion | Discrete diffusion on binary images (XOR noise) | binary MNIST acc@t1 **0.886** | [Diffusion](https://tsukushi-gate.github.io/Griffith/en/tutorials/logic-diffusion/) |
| LogicVAE | Bernoulli-bit latent VAE | binarized MNIST **193.6 nat/image** | [VAE](https://tsukushi-gate.github.io/Griffith/en/tutorials/logic-vae/) |
| CenterNetLUT | Object detection on Multi-MNIST | mAP **0.5425** (13,944 params) | [Detection](https://tsukushi-gate.github.io/Griffith/en/tutorials/detection/) |
| TextCNNModel | Japanese sentiment polarity (WRIME) | acc **82%** | [TextCNN](https://tsukushi-gate.github.io/Griffith/en/tutorials/textcnn/) |

## How Griffith differs from difflogic

Griffith keeps difflogic's core idea — relax a discrete gate choice into a differentiable mixture —
and changes four things. Attribution and license details are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); parts of the CUDA kernels are derived from
difflogic (MIT).

1. **N-input LUTs instead of 2-input gates.** difflogic learns a distribution over the 16 two-input
   Boolean functions. Griffith generalizes the same construction to N-input truth tables
   (N = 2, 4, 6), so one layer type spans "difflogic-compatible" through "FPGA LUT6-native".
2. **CUDA is optional, not required.** The reference implementation is plain PyTorch and runs on
   CPU and Apple Silicon (MPS); the JIT CUDA kernels are an accelerator, and the library falls
   back automatically when they are missing.
3. **An IR sits between training and hardware.** Training results are frozen into a JSON IR that is
   validated, simulated with integer bit operations, and lowered to Verilog / VHDL. Tests assert
   three-way agreement: PyTorch hard eval ⇔ IR simulation ⇔ HDL simulation.
4. **Five parameterizations and four learning rules.** Beyond DLGN-style softmax there are
   BinaryBrain-style independent tables and the LDDLGN-style per-input parameterization (~10×
   fewer parameters), and backprop is one of four selectable credit-assignment rules.

## Learning rules

Choose the credit-assignment rule per layer with `with_learning()` and train with `HybridTrainer`.

| Rule | How the error travels | Needs derivatives? | Use it when |
| --- | --- | --- | --- |
| `bp` | Backprop: transformed layer by layer | Yes | Default; when accuracy matters most |
| `ap` | **Akiho Propagation (our backprop-free rule)** — the output error is handed to every reachable LUT unchanged | No | Demonstrating backprop-free training; requires `parameterization='independent'` and a plain sequential stack |
| `dfa` | Direct Feedback Alignment: the output error is projected by a fixed random matrix | Only locally | Cutting inter-layer backprop; parallelism / biological-plausibility settings |
| `eggroll` | Evolution strategies: low-rank perturbations scored by fitness | No | When the objective has no usable gradient |

```python
from griffith_logic import HybridTrainer, LearningSpec, LutLayer, with_learning

fc1 = with_learning(
    LutLayer(784, 8000, parameterization="independent"),
    LearningSpec(rule="ap"),
)
```

Details and the credit-assignment diagrams:
[Learning rules](https://tsukushi-gate.github.io/Griffith/en/explanation/learning-rules/).

## Hybrid networks

You do not have to make the whole network out of gates. Replace only the parts you want:

```python
from griffith_logic import LogicSpec, replace_mlp_with_lut

hybrid, replaced = replace_mlp_with_lut(vit, spec=LogicSpec(n_inputs=2, depth=2))
```

## Documentation

Full documentation: **<https://tsukushi-gate.github.io/Griffith/en/>**

- [Installation](https://tsukushi-gate.github.io/Griffith/en/getting-started/installation/) ·
  [Quickstart](https://tsukushi-gate.github.io/Griffith/en/getting-started/quickstart/) ·
  [Core concepts](https://tsukushi-gate.github.io/Griffith/en/getting-started/concepts/)
- [Tutorials](https://tsukushi-gate.github.io/Griffith/en/tutorials/) — runnable, CPU-sized versions of
  every model, generated from the scripts in `examples/`
- [What is DLGN?](https://tsukushi-gate.github.io/Griffith/en/explanation/dlgn/) ·
  [LUT layer internals](https://tsukushi-gate.github.io/Griffith/en/explanation/lut-layer/) ·
  [Architecture](https://tsukushi-gate.github.io/Griffith/en/explanation/architecture/)
- [Deploy to FPGA](https://tsukushi-gate.github.io/Griffith/en/guides/fpga/) ·
  [HDL export](https://tsukushi-gate.github.io/Griffith/en/guides/hdl-export/) ·
  [IR](https://tsukushi-gate.github.io/Griffith/en/guides/ir/)
- [API reference](https://tsukushi-gate.github.io/Griffith/en/reference/api/) ·
  [CLI reference](https://tsukushi-gate.github.io/Griffith/en/reference/cli/) ·
  [IR schema](https://tsukushi-gate.github.io/Griffith/en/reference/ir-schema/)

The `griffith` command is the single entry point for training, evaluation and HDL export
(`md` remains as a backwards-compatible alias).

## Tools

Everything under `tools/` ships with the repository (it is not part of the PyPI package, so clone
the repo to use it). Python-side extras come from `uv sync`; each front-end has its own
`package.json`.

| Tool | What it is | Start it with |
| --- | --- | --- |
| `tools/editor` | Visual editor: build a model as a graph, watch training live, evaluate, and export HDL — all against the IR (FastAPI + React). | `uv sync --extra editor` then `make editor` (backend :8765 + frontend :5173) |
| `tools/webbench` | Browser benchmark: runs the exported circuits as WASM/WebGPU in the browser and compares latency and payload size against MediaPipe / TFJS baselines. | `make webbench` (or `cd tools/webbench && npm ci && npm run dev`) |
| `tools/visualizer` | WebGL viewer that draws a DLGN / CDDLGN / RDDLGN gate by gate and streams activations live. | `uv sync --extra visualizer`, then `cd tools/visualizer/frontend && npm ci && npm run dev` |
| `tools/nas` | Architecture search (genome mutation + NSGA-II) driven from the editor's `POST /api/nas/start`, usable as a Python package too. | Used from the editor, or `from tools.nas import search` |
| `tools/paper-tree` | Reading tree for the related literature: renders `data/papers/*.md` front matter as a graph of papers and their successors. | `cd tools/paper-tree && npm ci && npm run dev` |
| `tools/mdl-fpga` | FPGA support commands (`doctor` / `convert` / `build` / `flash` / `hw-check`) for Tang Nano 20K / DE0-CV. **Being merged in.** | see the [FPGA guide](https://tsukushi-gate.github.io/Griffith/en/guides/fpga/) |

## Development

```bash
git clone https://github.com/tsukushi-gate/Griffith.git
cd Griffith
uv sync --extra dev --extra cpu     # use --extra cu128 on a CUDA machine
uv run python examples/01_xor.py    # XOR → hard accuracy 1.00
make test                           # fast suite (must stay green, under 2 minutes)
make lint                           # ruff
```

Always go through `uv run`; never activate `.venv` directly, and use extras with `uv sync` rather
than `uv pip install`. Tests are layered L1 (math) → L2 (gradients) → L3 (paper reproduction) →
L4 (convergence) → L5/HDL (equivalence); which layer your change needs is spelled out in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Contributions are welcome. Every commit must carry a
[Developer Certificate of Origin](DCO) sign-off (`git commit -s`). Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md) — please do not open
a public issue.

**Only load checkpoints (`.pt`) and IR files from sources you trust**: as with any PyTorch project,
deserializing an untrusted file can execute arbitrary code.

## Citation

If you use Griffith in your research, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{griffith_logic,
  title  = {Griffith: Differentiable Logic Gate Networks for PyTorch},
  author = {{Masato Akiho}},
  year   = {2026},
  date   = {},
  note   = {Version 0.1.0},
  url    = {https://github.com/tsukushi-gate/Griffith}
}
```

Please also cite the work Griffith builds on: Petersen, Borgelt, Kuehne and Deussen,
*Deep Differentiable Logic Gate Networks*, NeurIPS 2022
([arXiv:2210.08277](https://arxiv.org/abs/2210.08277)).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Upstream copyright notices
and license texts for third-party material are collected in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Acknowledgements

Griffith stands on:

- [difflogic](https://github.com/Felix-Petersen/difflogic) (Felix Petersen et al., NeurIPS 2022) —
  the differentiable logic gate network formulation, and the CUDA kernel structure our kernels are
  derived from.
- [BinaryBrain](https://github.com/ryuz/BinaryBrain) (Ryuji Fuchikami) — the LUT-network training
  approach behind our `independent` parameterization.
- [DWN](https://github.com/alanbacellar/DWN) (Bacellar et al.) — differentiable weightless neural
  networks and learnable wiring.
- The authors of the papers behind the reproductions in our model zoo, whose published recipes and
  numbers made the comparisons possible.

---

日本語版: [README.ja.md](README.ja.md)
