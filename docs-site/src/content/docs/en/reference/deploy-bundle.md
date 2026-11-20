---
title: 'Deploy Bundle Spec'
description: 'The format of the deploy kit''s distribution unit, the bundle. Top-level structure, the programs / preproc_desc / readout sections, the .mdlb binary format, the version compatibility policy, validation rules, and the Web target.'
---

The **bundle** is the format that packs a trained model into **one file = one distribution unit common to all targets**.
Every runtime — Web / C (embedded, TinyML) / CUDA / Metal / iOS / Android — reads the same bundle and returns **the same integer output**.

The implementation is in `griffith_logic/deploy/` (`bundle.py` / `programs.py`), and the CLI is `griffith deploy`
(all subcommands are in the [griffith CLI](/en/reference/cli/#griffith-deploy)). For the end-to-end procedure, see the
[Deploy Kit Quickstart](/en/guides/deploy-quickstart/).

## Design premise

**Finish all optimization on the exporter side, and let the target side only execute.**

The weight-shared gate trees of conv2d are **reduced to an instruction list** at bundle-generation time and baked into the `programs` section
(NOT absorption → constant folding → degenerate truth tables → identical references → commutative normalization → CSE → dead-code elimination).
Measured, this cuts gates by **59.7 to 95.1%**. The target-side runtime just executes this instruction list in order and implements no reduction passes.
Preprocessing is handled the same way, shipped as data in the form of integer descriptors.

This division of labor was taken from the ORT format for mobile in ONNX Runtime (graph optimization is done offline, and the device runs on a reduced runtime with no optimization code) and
from MediaPipe's `.task` bundle (model + pre/post-processing + metadata bundled into one file, read by the same API on every platform).

`programs` and `preproc_desc` are **redundant data that can be deterministically rebuilt** from `ir` and `preproc`.
**The source of truth is still the IR**, and if the redundant side disagrees, the IR is right (`griffith deploy info --strict` detects this disagreement).

## Top-level structure

| Key | Required | Contents |
|---|---|---|
| `bundle_format` | ✅ | Fixed to `"magicdelta-bundle"` |
| `bundle_version` | ✅ | `"MAJOR.MINOR"` (currently `1.0`) |
| `ir` | ✅ | The IR (JSON). The source of truth. Schema: [IR Schema](/en/reference/ir-schema/) |
| `programs` | if there is conv2d | conv2d node id → instruction list (below) |
| `preproc` | | Declaration of the input encoding (kind / mean / std / thresholds / bit_order, and so on) |
| `preproc_desc` | for image input | The integerized preprocessing descriptors (below) |
| `labels` | | Array of class names |
| `readout` | | How to read the output (`class_scores` / `spatial_scores` / `detection`, and so on) |
| `resources` | | Estimates such as gate count, logic depth, and delivery size |
| `eval` | | Measured values such as accuracy |
| `about` | | Provenance (checkpoint, recipe, wandb run id, export time). The result of the lossless gate reduction that `griffith deploy bundle` applies automatically is kept in `about.compress` (`{level, applied, skipped, gates, always}`). When resynthesis ran, it is kept in `about.resynth`, whose `verify` is either `fast` (per-tree exhaustive equality + real samples) or `simulate` (additional ir/simulate verification) |
| `demo_samples` | | Real samples for demos and end-to-end verification |
| `embed_table` | for RDDLGN autoregressive generation | Token ID → embedding bit string (below) |
| `folded_n_table` / `folded_l_table` | optional | Vocabulary-folding tables (after embedding and passing through N / L-layers, below) |

An existing webbench delivery JSON becomes a bundle just by adding `bundle_format` / `bundle_version` / `programs`.
The extra keys are ignored by the existing TS loader, so models already being served are not broken.

## The `programs` section

There is one entry per conv2d node.

```jsonc
"programs": {
  "<node_id>": {
    "taps":        [int, ...],   // flat positions in the receptive field (deduplicated; the order is the tap ID)
    "n_instrs":    int,
    "instrs":      [tt, a, b, slot, ...],  // flat array, 4 elements per instruction
    "n_slots":     int,          // max number of intermediate values live at once (number of work planes)
    "oc_out":      [ref, ...],   // per output channel, a reference to its final value (length = out_channels)
    "write_start": [int, ...],   // write-out schedule (CSR, length = n_instrs + 2)
    "write_oc":    [int, ...],   // same (a permutation of output channels, length = out_channels)
    "stats":       { ... }       // 16-item breakdown of the gate reduction
  }
}
```

### Reference (ref) encoding

`ref = (idx << 2) | kind`, where `kind` means the following.

| kind | Meaning | idx |
|---|---|---|
| 0 | Constant | 0 = false, 1 = true |
| 1 | Tap | Index into `taps` |
| 2 | Instruction | Index into `instrs`. **Always earlier than itself** (topological order) |

The encoding is identical on the Python side (`export/optimize.py`) and the TS side (`convProgram.ts`), and bundling performs no conversion.

### Bit convention of the truth table `tt`

A 2-input function with `bit index = (b<<1)|a` and value = `f(a,b)`. It uses the same convention as the IR's `truth_tables` (bit k = input pattern k), so it can be used as is, without reordering.

### Write-out schedule

It holds "which output channels to write out right after executing instruction j" in CSR form (`write_start` / `write_oc`).
Bucket 0 is for outputs that do not depend on any instruction (constants and direct tap connections), handled before the instructions run.

```
bucket 0              : write_oc[write_start[0] .. write_start[1])
right after instr j   : write_oc[write_start[j+1] .. write_start[j+2])
```

### When the instruction list is empty

`n_instrs = 0` is a normal state. In an untrained or residual-initialized pass-through convolution, every gate folds into a constant or an alias and disappears,
leaving every output as a constant or a direct tap connection.

## The `preproc_desc` section

The encoding from an image (u8) into binary channels, compiled into **data that contains no floating-point arithmetic**.
The implementation is `deploy/preproc_desc.py`, and the TS-side sources of truth are `gpuPreproc.ts` and `cpuPreproc.ts`.

```jsonc
"preproc_desc": {
  "desc":  [int, ...],  // channels × 16 words
  "qval":  [int; 256],  // I(k) = f32(k/255) * 2^31
  "channels": int, "image_channels": int, "height": int, "width": int,
  "stride": 16,
  "counts": { "thermo_step": n, "thermo_lut": n, "cmp": n }
}
```

One channel is 16 words, and the contents of each word are as follows.

| Word | Contents |
|---|---|
| 0 | op (0 = THERMO, 1 = EDGE, 2 = CURV, 3 = CHROMA, 4 = SAT, 5 = THERMO_STEP) |
| 1 | Image channel a |
| 2 | Image channel b (CHROMA) / threshold k0 (THERMO_STEP) |
| 3, 4 | dy, dx (i32 two's complement) |
| 5, 6 | Low and high 32 bits of cut (int64 two's complement) |
| 7 | Direction of the comparison (0: `D >= cut`, 1: `D <= cut`) |
| 8..15 | The 256-bit LUT for the thermometer |

### Why it can be integerized

The input is u8, so a pixel value takes only 256 possible values `f32(k/255)`, and `I(k) := f32(k/255) × 2^31` is **an exact integer** (`0 ≤ I(k) ≤ 2^31`).
Every quantity the detector computes can be expressed as an integer combination of `I`.

```
edge   : D = I(a) - I(b)
curv   : D = I(a) - 2·I(m) + I(c)     (|D| ≤ 2^32 → needs 34 signed bits)
chroma : D = I(a) - I(b)
saturation : D = I(kmax) - I(kmin)
```

The comparisons can also be integerized, because `t × 2^31` is an exact power-of-two multiple.

```
d >  t  ⟺  D ≥ floor(t·2^31) + 1
d < -t  ⟺  D ≤ ceil(-t·2^31) - 1
```

Only the thermometer cannot be integerized, because `(x - mean)·(1/std) > th` involves arbitrary real coefficients, but since the input has only 256 values,
a **256-bit LUT** makes it bit-exact by definition. Moreover, if `std > 0` it is monotonic in k, so it almost always folds into a
single threshold "`k ≥ k0`" (THERMO_STEP) (whether it folded is decided only after sweeping the whole LUT and checking that it really is a step function).
In real models thermometer channels make up the majority, so this pays off. The border is replicate (the edge value is duplicated).

### Difference from the training-time encoder (float32)

The delivery path and the descriptors both compute at **float64-equivalent**, whereas the training-time `encode_image_features` computes in **torch float32**.
When the f32 rounding of `a - b` or `a - 2m + c` straddles a threshold, bits very rarely disagree at the boundary (measured: 3 out of 26,433 bits = 0.01%).
This difference was already present in the Web delivery path, and the descriptor side is pinned by a test to match the float64 reference exactly, while the divergence from f32 is measured by a separate test with an upper bound set on it.

### Geometric invariant

`channels × height × width` must equal the IR's input width. A bundle that is off here will certainly break on the device, and with a hard-to-diagnose
symptom ("somehow accuracy is poor"), so `build_bundle` and `validate_bundle` **fail it at generation time**.

## The `readout` section

It declares how to read out the output: `class_scores` for classification, `spatial_scores` for segmentation (with the spatial shape of the output), and `detection` for detection.

A detection (`kind: "detection"`) concatenates three heads (`hm` / `size` / `off`) in its output vector, and `readout` supplies the information needed for decoding.

| Key | Meaning |
|---|---|
| `heads` | The three heads `hm` / `size` / `off`. Each has its start position (`offset`), channel count (`n_out`), height, width, and scale (`scale`) |
| `output_stride` | How much smaller the score map is than the input image |
| `top_k` / `threshold` | How many top candidates to take, and at what minimum score |
| `n_classes` | Number of classes |

**The runtime returns up to the score map and does not include decoding into boxes (peak extraction plus NMS)**. Decoding is post-processing on the application side, and
the reference implementation is `tools/webbench/src/detdecode.ts`. For a summary of the steps, see [Prepare demo models](/en/guides/deploy-models/#decoding-boxes-is-the-apps-job).

## The `embed_table` / `folded_n_table` / `folded_l_table` sections

The seq2seq of RDDLGN (`griffith_logic/models/rddlgn.py`) has **the one boundary that does not become IR**, the token embedding
(the IR's input port contracts to receive "embedded, Heaviside-applied bit strings" directly). Autoregressive generation re-embeds the previous step's output token
as the next step's input, so the conversion from token ID to bit string has to happen at run time outside the IR.
These three sections all have the same shape: a "token ID → fixed bit string" table baked into the JSON.

```jsonc
"embed_table":    {"vocab": V, "d_emb": D, "bits": [u32, ...]}
"folded_n_table": {"vocab": V, "width": W, "bits": [u32, ...]}
"folded_l_table": {"vocab": V, "width": W, "bits": [u32, ...]}
```

`bits` is an array packing `ceil(width_or_d_emb / 32)` words per row (token), LSB-first
(column x is bit `x&31` of word `x>>5`; the same layout as the C runtime's bit-plane contract).

- **`embed_table`**: `bits[token]` is that token's embedding bit string itself (`sigmoid(embedding.weight) > 0.5`).
  On the C side it is baked in as `mdl_embed_lookup(token, out_bits)`
- **`folded_n_table` / `folded_l_table` (optional)**: `bits[token]` is the output bit string of that token **after embedding it and passing it through the N-layers (or L-layers)**.
  The N / L-layers are functions of the token ID only (independent of state), so for a finite vocabulary all tokens can be precomputed into a table (vocabulary folding).
  If the table size (`vocab × ceil(width/32) × 4B`) is at least the weight size of the original layers, **it automatically returns `None` and adds no section to the bundle**.
  On the C side they are baked in as `mdl_folded_n_lookup` / `mdl_folded_l_lookup`, and `MDL_HAS_FOLDED_N_TABLE` (`_L_` for the L side) is defined.
  When both N and L are folded, the corresponding weights and calls can be dropped from the output entirely (a real reduction in delivery size; in that case teacher forcing and the non-folded greedy function become unavailable)

## Version compatibility policy

Modeled on ONNX's opset operation.

- **MAJOR mismatch → the loader explicitly refuses.** It is treated as a change of semantics
- **MINOR newer than the reader → warning only.** Additions are treated as backward compatible, and unknown keys are ignored
- The IR version is a separate axis, and any version not in `SUPPORTED_IR_MAJOR_MINOR` in `bundle.py` produces a warning (currently `0.2`)

## Validation

What `validate_bundle(bundle, strict=False)` checks is as follows.

- Required keys and version compatibility
- A one-to-one correspondence between `programs` and the conv2d nodes
- Array-length invariants (`instrs` = 4 × `n_instrs`, `write_start` = `n_instrs` + 2, `oc_out` and `write_oc` = `out_channels`)
- Value ranges of references (constants are 0 / 1, taps are within `taps`, instructions are forward references only)
- Taps lie inside the receptive field (`in_channels × kH × kW`) with no duplicates
- `write_oc` is a permutation of the output channels

With `strict=True`, it **rebuilds the instruction list from the IR and confirms an exact match**. This detects staleness such as having rewritten the IR without rebuilding `programs`
(from the lesson of once missing that "the conv extractor ignored the learned wiring and broke every Web demo", the export path always carries a self-check).

Note that because the instruction list includes dead-code elimination, **rewriting the truth table of a branch that does not affect the output does not change the rebuilt result**.
What strict detects is not "any change to the IR" but "a change to the logic that actually executes."

`build_bundle(doc, verify=True)` runs an exhaustive truth-table comparison before writing. `verify=False` is acceptable only for fault-injection tests.

### A note on JSON round trips

`resources.lut_counts` is a **dict with integer keys**, and a JSON round trip turns the keys into strings (the TS side also reads them as string keys).
The canonical form of a bundle is **the form after JSON decoding**. `ir` and `programs` are completely invariant under a round trip.

## The `.mdlb` binary format

For environments with tens of KB of RAM, and for zero-copy loading via mmap, there is a flat binary format that can be converted deterministically from the JSON.
The implementation is `griffith_logic/deploy/mdlb.py` (`write_mdlb` / `read_mdlb`) plus a read-only decoder on the TS side, `tools/webbench/src/deploy/mdlb.ts`,
and `.mdlb` is always produced by Python. The design principles follow how TFLite operates its FlatBuffers.

- Magic `MDLB` + u32 version, little-endian
- Sections are TLV: `IR_META` / `TT` / `INDICES` / `PROGRAM` / `PREPROC_DESC` / `LABELS` / `READOUT` / `EXTRA`
  (`EXTRA` gathers top-level keys added later, such as `resources` / `eval` / `about` / `demo_samples` / `embed_table` / `folded_*_table`, with a generic encoding)
- Sections are 8-byte aligned, and offsets are relative to the start of the payload region
- Each section has a CRC32 (together with the magic, version, and section boundaries, detection by fault injection is pinned by a test)
- There is two-way conversion with JSON and a round-trip test, and the contract is that **`read_mdlb(write_mdlb(bundle))` matches `json.loads(json.dumps(bundle))` exactly, down to types**

In the CLI you can write it out with `griffith deploy bundle -o m.mdlb` (auto-detected from the extension; `--format mdlb` also works).
`info` / `codegen-c` / `codegen-cuda` / `codegen-metal` / `codegen-web` / `verify-c` / `verify-web` can all read `.mdlb` and `.mdlb.gz` transparently.
On the codegen path (baking the model into C source) `.mdlb` is not needed, and it is only for when you want to swap models dynamically.

## CLI

The essentials of the commands for creating and checking a bundle are below (all flags are in the [griffith CLI](/en/reference/cli/#griffith-deploy)).

```bash
griffith deploy bundle <model.json> -o <out.bundle.json> [--gzip] [--compress {none,fast,max}]
griffith deploy bundle <model.json> -o <out.bundle.mdlb>      # .mdlb binary (auto-detected from the extension)
griffith deploy info <bundle.json|bundle.mdlb> [--strict]
```

`--compress` sets the strength of the lossless circuit compression and means the same for every kind of model. The default `fast` applies every reduction that finishes in seconds (sweep for dense, resynthesis for conv2d). `none` applies no IR-rewriting optimization, and `max` searches deeper (limit 5) with extra verification.
The result is recorded in `about.compress` and can be checked with `info`. Its shape is `{level, applied, skipped, gates: {pass: [before, after]}, always: [{pass: 'conv-program', gates: [before, after]}]}`. `skipped` lists only compressions that had targets but could not be applied (sweep refused because dense and conv2d are mixed, a failure, and so on); inapplicable compressions (resynthesis on a dense-only model, sweep on a conv-only model) are not listed. Instruction-list conversion of conv2d (the `programs` section) happens at every level and is recorded in `always`. The individual `--sweep` / `--no-sweep` / `--resynth lossless` take precedence over `--compress`.

## The Web target

C / CUDA / Metal **generate** "bundle → self-contained source," but the Web already has an executable from the start (`runtime/web`, a family of
TypeScript-built WASM / WebGPU engines). So what the Web backend (`griffith_logic/deploy/webbackend.py`) does is asymmetric:
**it generates not a single line of source code and only writes the bundle itself out as the distribution JSON**.
The TS side parses the bundle's `programs` section (instruction-listed conv2d) itself and runs it as is, never re-running the reduction in the browser.

### `griffith deploy codegen-web`

```bash
uv run griffith deploy codegen-web <bundle.json> -o <out_dir> [--name model] [--slim] [--no-runtime] [--gzip]
```

The default output layout (without `--slim`) is as follows.

```
<out_dir>/
  model.bundle.json          the bundle itself
  mdl_web.js                 public API (loadBundle)
  demo.js / index.html       a single-page demo
  verify.mjs                 node CLI (the core of the ship gate verify-web uses)
  release.wasm / release-simd.wasm   AssemblyScript-built kernels
  lib-*.js, assets/          shared chunks (names change per build, so copy dist whole)
```

- `--slim`: writes out not the bundle but a **plain webbench model doc** that has no `bundle_format` / `bundle_version` / `programs` / `preproc_desc`.
  This presumes the browser rebuilds conv2d's instruction listing from the IR, so it is a choice that **gives up the reduction guarantee the exporter side had completed**.
  It is not used for normal distribution and is limited to debugging and experiments that touch the IR directly
- `--no-runtime`: skips copying `runtime/web/dist` (when it is already being served). If `dist` is missing it fails, telling you to run `make web-runtime` first (it never silently ships an old or missing runtime)

`resources.bundle_json_bytes` and `bundle_gzip_bytes` (referenced by the UI header's delivery-size display) are rebaked on every run of `griffith deploy bundle`.
`bundle_json_bytes` matches the actual file's byte count exactly, and `bundle_gzip_bytes` is a best-effort value within ±1 byte.

Measured results on real models (the effect of instruction listing) are as follows.

| Model | Bundle | conv2d nodes | Gates (before → after instruction listing) | Reduction |
|---|---:|---:|---:|---:|
| seg_convdlgn_k128 | 310,782B | 5 | 8,512 → 2,811 | 67.0% |
| det_fewclass_lut_k64 | 521,320B | 6 | 6,272 → 1,503 | 76.0% |

The whole of `runtime/web/dist` (model-independent) totals about 279KB (measured 278,704B), so the delivery-size estimate is **bundle size + about 279KB**.
Putting `griffith deploy bundle --resynth lossless` in front cuts the gate count after instruction listing further (accuracy unchanged).
For seg_convdlgn_k128 the final instruction list became 8,512 → 2,620 (69.2% fewer, 2.2 pt better than 67.0% without resynthesis).

### `griffith deploy verify-web`

```bash
uv run griffith deploy verify-web <bundle.json> [-n SAMPLES] [--seed SEED] [--engine js|wasm]
```

A pre-ship gate that, taking `ir/simulate.py` (the numpy golden model) as the reference, actually launches `runtime/web/dist/verify.mjs` (TS-built, runs on Node) and compares them integer-exactly.
The chain of verification extends to `PyTorch hard eval ⇔ ir/simulate ⇔ web(js) ⇔ web(wasm)`.

- The **bits path** is always verified (random bit strings are fed to both simulate and web and compared)
- If the bundle has `preproc_desc` (that is, image input), the **image path** is verified as well. The TS side binarizes from the raw image **on its own**, so
  agreement of the two paths verifies the Python-versus-TS cross-language contract of the preprocessing implementation itself. A mismatch here is a signal of a serious bug, so it is never swallowed
- `--engine js|wasm` (default `js`). The internal engines are separate implementations (pure TS and AssemblyScript-built), so run **both** before shipping
- **`strictPrograms=true` is fixed**: the verification path fully verifies the bundle's `programs` **before** loading, and if even one is broken it raises rather than falling back.
  The normal `loadBundle` (demo UI and so on) does not swallow a verification failure; it leaves the reason in `notes` and falls back to rebuilding in the browser.
  Only the ship gate is made asymmetric so that the fallback does not become a loophole that cannot detect "stale or tampered `programs`"
- Node.js is required. If `runtime/web/dist/verify.mjs` is missing, it tells you to run `make web-runtime` first

## Note on public distribution

The JSON of face-detection and VOC few-class detection models may have **images from real datasets** (WIDER FACE / Pascal VOC) embedded as base64 in `demo_samples`.
**You must not distribute such a bundle publicly to third parties as is.** `demo_samples` is an accessory for demo display and end-to-end verification, and
is not something for which redistribution permission from the dataset was obtained. Build bundles for public release without embedding images, as in `export_webbench.py det-ckpt ... --demo-samples 0`
(see [Prepare demo models](/en/guides/deploy-models/#face-detection-wider-face)).

## Related pages

- [Deploy Kit Quickstart](/en/guides/deploy-quickstart/) — the end-to-end procedure from IR through bundling and C generation to running
- [Platform support](/en/reference/platform-support/) — the verification level of each target
- [IR Schema](/en/reference/ir-schema/) — the spec of the bundle's `ir`
- [griffith CLI](/en/reference/cli/#griffith-deploy) — all `griffith deploy` subcommands

<!-- i18n:source=reference/deploy-bundle.md hash=fb588b873dac -->
