# Changelog

All notable changes to Griffith (`griffith-logic`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases below 1.0.0 may change the public API in a minor version.

<!--
Section headings use the names configured in release-please-config.json
(Features / Bug Fixes / Performance / Refactoring / Documentation), so entries added by
release-please from Conventional Commits stay consistent with the hand-written ones.
Do not rename them.
-->

## [Unreleased]

### Features

### Bug Fixes

### Documentation

## [0.1.0] - 2026-11-20

Initial public release.

### Features

- **N-input LUT layers.** `LutLayer` generalizes difflogic's fixed 2-input gates to N-input
  lookup tables (N = 2, 4, 6), covering everything from difflogic-compatible 16-gate layers to
  FPGA-native LUT6.
- **Five parameterizations** of the truth table: `softmax` (DLGN), `independent` (BinaryBrain),
  `iwp` (LDDLGN; roughly 10× fewer parameters), `raw` and `warp`, selectable per layer.
- **Logic-gate convolutions.** `LutConv2d` (LUT trees generalizing CDDLGN's binary gate trees) and
  `OrPooling2d`.
- **IR (intermediate representation).** Trained networks are discretized and frozen into a JSON IR
  with schema validation, an integer-bit-operation simulator, and weight editing — the IR is the
  single source of truth for everything downstream.
- **Verilog and VHDL export** from the IR, with a three-way equivalence check
  (PyTorch hard eval ⇔ IR simulation ⇔ HDL simulation via `iverilog` / `ghdl`).
- **Pluggable learning rules.** BP, AP (Akiho Propagation, backprop-free), DFA, and EGGROLL
  (evolution strategies), annotated per layer with `with_learning()` and trained by
  `HybridTrainer`.
- **Hybrid networks.** `replace_mlp_with_lut()` and `LogicLinear` swap only part of an existing
  `nn.Module` (for example the MLP of a Transformer) for logic gates.
- **DWN support.** `DWNLayer` / `DWNClassifier` implement differentiable weightless neural
  networks with learnable wiring.
- **Model zoo.** Logic-only implementations across image classification (DLGN, Conv-DLGN,
  BinaryBrain, LogicMixer, LogicResNet), sequence and language modelling (LogicGPT,
  LogicLadderGPT, RDDLGN, logic attention variants, LogicLSTM / LogicGRU, LogicReservoir),
  generative models (LogicBinaryDiffusion, LogicVAE, LogicGAN, LogicWaveNet), detection and dense
  prediction (CenterNetLUT, LogicUNet), and NLP (TextCNN).
- **Optional CUDA kernels** (JIT-compiled) for N-input LUTs, with automatic fallback to a pure
  PyTorch path on CPU and Apple Silicon (MPS).
- **`griffith` CLI** for training, evaluation, HDL export and experiment orchestration
  (`md` is kept as a backwards-compatible alias).
- **Runnable examples** in `examples/`, which also generate the tutorials on the documentation
  site.

### Documentation

- Public documentation site with tutorials, guides, explanation pages, a model zoo, benchmarks and
  a generated API reference, in Japanese and English.

[Unreleased]: https://github.com/tsukushi-gate/Griffith/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tsukushi-gate/Griffith/releases/tag/v0.1.0
