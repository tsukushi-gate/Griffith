# Third-Party Notices

Griffith (`griffith-logic`) incorporates, adapts, or reimplements material from
the third-party projects listed below. This file collects the required
copyright notices and license texts.

<!-- TODO(human): 本プロジェクトのライセンスが Apache-2.0 で確定したら、この注記を削除してください。
     MIT に切り替える場合は LICENSE・NOTICE・pyproject.toml の license・CITATION.cff・README の表記を同時に直すこと。 -->

Griffith itself is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Nothing in this file changes the
license of Griffith's own code; it documents the terms of upstream material
only.

## Summary

| Project | License | Relationship to Griffith |
| --- | --- | --- |
| [difflogic](https://github.com/Felix-Petersen/difflogic) | MIT | **Derived code.** Portions of `griffith_logic/cuda/lut_kernels.cu` are derived from difflogic. |
| [BinaryBrain](https://github.com/ryuz/BinaryBrain) | MIT | Reimplemented from the design of BinaryBrain. No code was copied. |
| [DWN (Differentiable Weightless Neural Networks)](https://github.com/alanbacellar/DWN) | MIT | Reimplemented from the design of DWN. No code was copied. |
| [DIST_KD](https://github.com/hunto/DIST_KD) | Apache-2.0 | Follows the formulation of the DIST knowledge-distillation loss. No code was copied. |
| [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | **Runtime asset.** `tools/webbench` loads Google's MediaPipe WASM runtime and models as a reference baseline. |
| [elkjs](https://github.com/kieler/elkjs) | EPL-2.0 | **Runtime dependency** of `tools/editor` and `tools/paper-tree` (graph layout). Notice obligation; no code copied. |
| [AssemblyScript](https://github.com/AssemblyScript/assemblyscript) | Apache-2.0 | Compiler and runtime for `tools/webbench/assembly/*.ts` (Griffith's own source). |
| [litex-boards](https://github.com/litex-hub/litex-boards) | BSD-2-Clause | **Derived data.** FPGA pin assignments in `tools/mdl-fpga/mdl_fpga/boards/*.json` are derived from litex-boards platform files (cross-checked against vendor documentation). See §7. |
| [MNIST database](http://yann.lecun.com/exdb/mnist/) | CC BY-SA 3.0 | **Bundled data (subset).** `tools/mdl-fpga/examples/mnist/mnist_test_1000.npz` holds a binarized subset of the MNIST test set. See §8. |
| [UCI HAR — Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K) | CC BY 4.0 | **Dataset, not bundled.** Downloaded at run time by `griffith_logic.datasets.load_uci_har`; used by the activity-recognition tutorial (`examples/27_har.py`) and the deploy-kit demos. See §9. |
| Other datasets (MNIST, CIFAR, PASCAL VOC, WIDER FACE, WRIME, SST-2, WMT14, OPUS-100, Tatoeba, …) | Varies; **WIDER FACE and WRIME are CC BY-NC-ND 4.0 (non-commercial)** | **Datasets, not bundled.** Downloaded at run time by loaders and experiment scripts. See §9.2. |

---

## 1. difflogic — derived code

Portions of `griffith_logic/cuda/lut_kernels.cu` are derived from difflogic
(revision `469702c0`), specifically the CUDA argument-checking macros
(`CHECK_CUDA`, `CHECK_CONTIGUOUS`, `CHECK_INPUT`), the error-handling helpers
(`gpuErrchk`, `gpuAssert`), the `ceil_div` helper, and the overall kernel
structure.

Upstream note: difflogic states that its `gpuErrchk` / `gpuAssert` idiom was
adapted from a Stack Overflow answer
(<https://stackoverflow.com/questions/14038589/>), which is available under the
Creative Commons license applicable to Stack Overflow contributions.

- Project: difflogic
- Source: <https://github.com/Felix-Petersen/difflogic>
- License: MIT

```
MIT License

Copyright (c) 2021-2023 Dr. Felix Petersen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. BinaryBrain — independent reimplementation

Parts of Griffith were reimplemented from the design of BinaryBrain
(in particular the LUT-network training approach). No source code from
BinaryBrain was copied into this repository; the attribution below is given as
a courtesy and to document the provenance of the ideas.

- Project: BinaryBrain
- Source: <https://github.com/ryuz/BinaryBrain>
- License: MIT (`license.txt`)

```
MIT License

Copyright 2018 by Ryuji Fuchikami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. DWN — independent reimplementation

Parts of Griffith were reimplemented from the design of DWN (Differentiable
Weightless Neural Networks). No source code from DWN was copied into this
repository.

- Project: DWN
- Source: <https://github.com/alanbacellar/DWN>
- License: MIT

```
MIT License

Copyright (c) 2024 Alan T. L. Bacellar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. DIST_KD — formulation

Griffith's DIST knowledge-distillation loss follows the formulation of
DIST_KD. No source code from DIST_KD was copied into this repository.
DIST_KD is distributed under the Apache License, Version 2.0; the upstream
repository ships no `NOTICE` file, so no additional attribution notice is
required beyond the copyright statement below.

- Project: DIST_KD
- Source: <https://github.com/hunto/DIST_KD>
- License: Apache-2.0 (full text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))

```
Copyright 2022 DIST_KD Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 5. Runtime and development dependencies

Griffith depends on the following third-party packages. They are **not**
vendored into this repository; they are installed from PyPI at install time.
Their licenses apply to those packages, not to Griffith's own code. The list
below highlights dependencies whose licensing warrants attention when
redistributing Griffith (for example, in a bundled or containerized product).

| Package | License (SPDX) | Notes |
| --- | --- | --- |
| `tqdm` | `MPL-2.0 AND MIT` | Dual-component licensing. MPL-2.0 is a file-level copyleft: redistributing modified `tqdm` sources requires publishing those modifications. Unmodified redistribution is unaffected. |
| `hypothesis` (dev only) | `MPL-2.0` | Test-time dependency only; not shipped with the distributed package. |
| `python-sat` (`pysat`) | `MIT` | The Python wrapper is MIT, but the **bundled SAT solvers are individually licensed**. Some solver versions (e.g. certain Glucose and Lingeling releases) carry non-commercial or research-only clauses. Review before redistributing binaries. |
| `fugashi` | `MIT AND BSD-3-Clause` | MeCab bindings; the underlying MeCab code is BSD-3-Clause. |
| `unidic-lite` | `MIT` / `WTFPL` (package) + `BSD-3-Clause` (dictionary data) | The dictionary data has its own license; check terms before redistributing the dictionary itself. |

<!-- TODO(human): 依存の完全な一覧とライセンスは、リリース前に `uv pip list` ベースの
     自動 SBOM（例: `pip-licenses` / `cyclonedx-py`）で再生成し、この表と突き合わせてください。 -->

A full dependency license inventory (including transitive dependencies such as
PyTorch, NumPy, and their own third-party notices) should be generated from the
resolved lock file at release time.

---

## 6. Web tooling assets (`tools/`)

The visual editor, the browser benchmark (`tools/webbench`), the visualizer, the
paper tree, and the documentation site are shipped as source. Their npm
dependencies are **not** vendored into this repository; they are installed from
the npm registry via each `package-lock.json`. The notes below cover (a) assets
that are fetched or copied into a build output and therefore redistributed when
you deploy a built bundle, and (b) dependencies whose licenses warrant attention.

### 6.1 MediaPipe Tasks Vision — runtime asset (Apache-2.0)

`tools/webbench` compares Griffith's logic-gate vision models against Google's
MediaPipe baselines. The MediaPipe WASM runtime
(`vision_wasm_internal.{js,wasm}`, `vision_wasm_module_internal.{js,wasm}`,
`vision_wasm_nosimd_internal.{js,wasm}`, roughly 32 MB in total) is copied out of
`node_modules/@mediapipe/tasks-vision/wasm/` into `tools/webbench/public/mediapipe/wasm/`
by `npm run mediapipe:setup` (wired into `predev` / `prebuild`), and the model
files (`*.tflite` / `*.task`) are downloaded at setup time from
<https://storage.googleapis.com/mediapipe-models>. Neither the runtime nor the
models are part of this repository's published tree; both are generated
artifacts. **If you redistribute a built `tools/webbench` bundle, it contains
Google's MediaPipe binaries and you must carry the Apache-2.0 license and the
notice below with it.**

- Project: MediaPipe / `@mediapipe/tasks-vision` (0.10.x)
- Source: <https://github.com/google-ai-edge/mediapipe>
- License: Apache-2.0 (full text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))

```
Copyright 2019-2024 The MediaPipe Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The MediaPipe model files carry their own model cards and terms of use; see
<https://ai.google.dev/edge/mediapipe/solutions/vision> before redistributing
them.

### 6.2 AssemblyScript — compiler for Griffith's WASM kernels

`tools/webbench/assembly/index.ts` and `index-simd.ts` are **Griffith's own
source code** (the packed conv2d gate-tree kernels), written in AssemblyScript
and compiled to `build/release.wasm` / `build/release-simd.wasm`. No third-party
kernel source is vendored. The compiler and its bundled runtime
(`assemblyscript`, 0.28.x) are Apache-2.0; a compiled `.wasm` embeds a small
AssemblyScript runtime, so a redistributed bundle carries that Apache-2.0
material.

- Project: AssemblyScript
- Source: <https://github.com/AssemblyScript/assemblyscript>
- License: Apache-2.0 (full text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))

### 6.3 elkjs — EPL-2.0 (notice obligation)

`elkjs` (the Eclipse Layout Kernel compiled to JavaScript) lays out graphs in the
visual editor and in the paper tree. It is used unmodified, as an npm
dependency. EPL-2.0 requires that recipients of a distribution containing the
program are told where to obtain the source and under which terms.

- Project: elkjs (Eclipse Layout Kernel)
- Source: <https://github.com/kieler/elkjs>
- License: EPL-2.0 (full text: <https://www.eclipse.org/legal/epl-2.0/>)
- Modifications by Griffith: none.

### 6.4 Front-end dependency license inventory

Generated with `npx license-checker --production --summary` in each workspace
(2026-09-24, on the versions pinned by the respective `package-lock.json`).
**No GPL, AGPL, SSPL, or non-commercial (CC-BY-NC) licensed dependency is
present in any workspace**, so no replacement is required.

| Workspace | Licenses (count) |
| --- | --- |
| `tools/editor/frontend` | MIT 172, ISC 12, Apache-2.0 5, BSD-3-Clause 2, MPL-2.0 2, BSD-2-Clause 1, EPL-2.0 1, 0BSD 1, UNLICENSED 1 (this package itself) |
| `tools/webbench` | MIT 55, Apache-2.0 12, BSD-3-Clause 12, ISC 5, "Apache-2.0 AND MIT" 1, BSD-2-Clause 1, UNLICENSED 1 (this package itself) |
| `tools/visualizer/frontend` | MIT 44, Apache-2.0 3, BSD-2-Clause 1, ISC 1, 0BSD 1, UNLICENSED 1 (this package itself) |
| `tools/paper-tree` | MIT 190, ISC 12, BSD-3-Clause 2, MPL-2.0 2, Apache-2.0 1, EPL-2.0 1, BSD-2-Clause 1, 0BSD 1, UNLICENSED 1 (this package itself) |
| `docs-site` | MIT 429, ISC 44, BSD-3-Clause 10, BSD-2-Clause 9, Apache-2.0 6, MPL-2.0 4, BlueOak-1.0.0 3, CC0-1.0 2, LGPL-3.0-or-later 1, Python-2.0 1, Unlicense 1, "MPL-2.0 OR Apache-2.0" 1, MIT* 1, Custom (astro.build) 1 (this package itself) |

Packages that are not permissive-by-default, and what they mean in practice:

| Package | License (SPDX) | Where | Notes |
| --- | --- | --- | --- |
| `elkjs` | `EPL-2.0` | editor, paper-tree | Weak copyleft with a notice obligation. Used unmodified; see §6.3. |
| `lightningcss` (+ platform binary) | `MPL-2.0` | editor, paper-tree, docs-site | File-level copyleft. Build-time CSS transformer, used unmodified; publishing modifications of its own sources would require releasing them. |
| `dompurify` | `MPL-2.0 OR Apache-2.0` | docs-site | Dual-licensed; Griffith relies on the Apache-2.0 option. |
| `@img/sharp-libvips-darwin-arm64` | `LGPL-3.0-or-later` | docs-site | Prebuilt libvips binary pulled in by `sharp` (Astro image optimization). Installed from npm at build time and not redistributed by this repository. Shipping it inside a binary bundle triggers LGPL relinking obligations. |
| `argparse` | `Python-2.0` | docs-site | Permissive; PSF license. |
| `mdn-data` | `CC0-1.0` | docs-site | Public-domain dedication (data, not code). |
| `khroma` | `MIT*` (license text inferred, no SPDX field) | docs-site | Upstream ships MIT terms in its README rather than a `LICENSE` file. |
| `robust-predicates` | `Unlicense` | docs-site | Public-domain dedication. |
| `common-ancestor-path`, `lru-cache`, `sax` | `BlueOak-1.0.0` | docs-site | Permissive, OSI-approved. |
| each workspace's own `package.json` | `UNLICENSED` / `Custom: astro.build` | all | These entries are Griffith's **own** `private: true` packages, not third-party code. They are covered by this repository's Apache-2.0 [`LICENSE`](LICENSE). |

<!-- TODO(human): リリース前に各ワークスペースで
     `npx license-checker --production --summary` を再実行し、この表を更新してください。 -->

---

## 7. FPGA board definitions (`tools/mdl-fpga`)

`tools/mdl-fpga` ships board description files (`mdl_fpga/boards/*.json`,
`mdl_fpga/boarddef/recipes/*.json`) that record physical facts about development
boards: pin assignments, device part numbers and resource counts. The pin tables
were **derived from** third-party platform definitions and cross-checked against
the vendor documentation of each board. No third-party source file is bundled in
this repository; only the extracted physical facts are.

- Project: litex-boards (`litex_boards/platforms/sipeed_tang_nano_20k.py`,
  `litex_boards/platforms/terasic_de10lite.py`)
- Source: <https://github.com/litex-hub/litex-boards>
- License: BSD-2-Clause (full text: [`LICENSES/BSD-2-Clause.txt`](LICENSES/BSD-2-Clause.txt))
- Modifications by Griffith: pin names and device identifiers were transcribed
  into Griffith's own JSON schema; no code was copied.

The second, independent source used to cross-check the Tang Nano 20K pin table is
Sipeed's official example constraints (`.cst`) files:

- Project: sipeed/TangNano-20K-example
- Source: <https://github.com/sipeed/TangNano-20K-example>

Per-board provenance (upstream commit, checking method, and which fact came from
which source) is recorded in
[`tools/mdl-fpga/mdl_fpga/boards/NOTICE`](tools/mdl-fpga/mdl_fpga/boards/NOTICE),
which is the authoritative record. Vendor toolchains (Intel Quartus Prime,
Gowin EDA) are **not** bundled: `griffith`'s FPGA flow detects a locally
installed toolchain and never redistributes it.

## 8. MNIST test-set subset (`tools/mdl-fpga`)

`tools/mdl-fpga/examples/mnist/mnist_test_1000.npz` contains a subset of the
MNIST test set (its first 1000 images; the count is set by `--n`), binarized at
threshold 0.5. It is used as the shared source for the FPGA demo's ROM image,
golden outputs and accuracy gate. The file is regenerated from the original
dataset by `tools/mdl-fpga/examples/mnist/gen_data.py`, which downloads MNIST via
`torchvision.datasets.MNIST`.

- Dataset: The MNIST database of handwritten digits
- Authors: Yann LeCun, Corinna Cortes, Christopher J. C. Burges
- Source: <http://yann.lecun.com/exdb/mnist/>
- License: Creative Commons Attribution-Share Alike 3.0
  (<https://creativecommons.org/licenses/by-sa/3.0/>)
- Modifications by Griffith: a subset of the test images was selected and the
  pixel values were binarized (threshold 0.5) and stored in NumPy `.npz` format.
  The modified data is distributed under the same license (CC BY-SA 3.0).

## 9. Datasets downloaded at run time

Griffith does not redistribute these datasets. The loaders download them from
the original publisher when first used, and the attribution below applies to
the data and to anything derived from it (for example, trained models and the
sample inputs that the deploy-kit demo scripts generate from the test split).

### 9.1 UCI HAR — Human Activity Recognition Using Smartphones (CC BY 4.0)

- Dataset: Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto,
  Xavier Parra. "Human Activity Recognition Using Smartphones."
  UCI Machine Learning Repository, 2013. <https://doi.org/10.24432/C54S4K>
- Source: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>
- Used by: `griffith_logic/datasets/har.py` (`load_uci_har`), which downloads the
  archive, checks its SHA-256 (`UCI_HAR_SHA256`) and caches the raw inertial
  signals; `examples/27_har.py`; `scripts/fetch_demo_models.py` and
  `scripts/make_device_samples.py` (deploy-kit demos).
- Modifications by Griffith: the raw signals are standardized and
  thermometer-encoded into bits. The dataset files themselves are not stored in
  this repository; the tutorial page includes figures plotted from the data.

### 9.2 Other datasets fetched at run time

The loaders and experiment scripts below download these datasets on first use
(through `torchvision.datasets`, Hugging Face `datasets.load_dataset`, or a
direct URL). None of them is stored in this repository. **Check each dataset's
terms yourself before using it, and especially before any commercial use or
before redistributing derived data or trained models.** Entries marked
"to be confirmed" could not be verified on the publisher's own page when this
file was written.

| Dataset | Source | License / terms | Commercial use | Cite |
| --- | --- | --- | --- | --- |
| MNIST (`torchvision.datasets.MNIST`) | <http://yann.lecun.com/exdb/mnist/> | CC BY-SA 3.0 as commonly redistributed (see §8); publisher page to be confirmed | To be confirmed | Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-Based Learning Applied to Document Recognition," Proc. IEEE, 1998 |
| Fashion-MNIST (`torchvision.datasets.FashionMNIST`) | <https://github.com/zalandoresearch/fashion-mnist> | MIT (Copyright 2017 Zalando SE) | Allowed | H. Xiao, K. Rasul, R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," arXiv:1708.07747, 2017 |
| CIFAR-10 / CIFAR-100 (`torchvision.datasets.CIFAR10/100`) | <https://www.cs.toronto.edu/~kriz/cifar.html> | No license stated; the page asks users to cite the report | To be confirmed | A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Report, University of Toronto, 2009 |
| Oxford 102 Flowers (`torchvision.datasets.Flowers102`) | <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/> | No license stated on the page; to be confirmed | To be confirmed | M.-E. Nilsback, A. Zisserman, "Automated Flower Classification over a Large Number of Classes," ICVGIP, 2008 |
| Oxford-IIIT Pet (`torchvision.datasets.OxfordIIITPet`, used by `griffith_logic/datasets/segmentation.py`) | <https://www.robots.ox.ac.uk/~vgg/data/pets/> | CC BY-SA 4.0; copyright of the images remains with their original owners | Allowed under CC BY-SA 4.0 (image rights stay with the owners) | O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," CVPR, 2012 |
| PASCAL VOC 2012 (`torchvision.datasets.VOCSegmentation/VOCDetection`, used by `experiments/data/`) | <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> | No dataset license. The images come from Flickr and their use must follow the Flickr terms of use | Depends on each image's rights; to be confirmed | M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman, "The PASCAL Visual Object Classes (VOC) Challenge," IJCV, 2010 |
| **WIDER FACE** (`scripts/fetch_widerface.py`, Hugging Face `CUHK-CSE/wider_face`) | <http://shuoyang1213.me/WIDERFACE/>, <https://huggingface.co/datasets/CUHK-CSE/wider_face> | **CC BY-NC-ND 4.0** (per the Hugging Face dataset card; the publisher page is to be confirmed) | **Not allowed** (non-commercial, no derivatives). Models trained on it, such as the face-detection demo, should be treated as non-commercial | S. Yang, P. Luo, C. C. Loy, X. Tang, "WIDER FACE: A Face Detection Benchmark," CVPR, 2016 |
| Tiny Shakespeare (char-rnn `input.txt`, Hugging Face `karpathy/tiny_shakespeare`) | <https://github.com/karpathy/char-rnn>, <https://huggingface.co/datasets/karpathy/tiny_shakespeare> | No license stated for the data (the char-rnn code is MIT); to be confirmed | To be confirmed | A. Karpathy, "char-rnn," GitHub, 2015 |
| SST-2 (Hugging Face `stanfordnlp/sst2`) | <https://nlp.stanford.edu/sentiment/>, <https://huggingface.co/datasets/stanfordnlp/sst2> | Listed as "unknown" on the dataset card; to be confirmed | To be confirmed | R. Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank," EMNLP, 2013 |
| **WRIME** (`experiments/data/wrime_emotion.py`; Hugging Face copy `tetsuro731/wrime-sentiment` in `experiments/data/wrime.py`) | <https://github.com/ids-cv/wrime> | **CC BY-NC-ND 4.0** (publisher repository). The Hugging Face copy states no license of its own | **Not allowed** (non-commercial, no derivatives) | T. Kajiwara, C. Chu, N. Takemura, Y. Nakashima, H. Nagahara, "WRIME: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations," NAACL, 2021 |
| WMT14 de-en (Hugging Face `wmt14`) | <https://www.statmt.org/wmt14/translation-task.html> | Mixed sources (Europarl, Common Crawl, News Commentary); no single license, listed as "unknown" on the dataset card; to be confirmed | To be confirmed | O. Bojar et al., "Findings of the 2014 Workshop on Statistical Machine Translation," WMT, 2014 |
| OPUS-100 en-ja (Hugging Face `Helsinki-NLP/opus-100`) | <https://opus.nlpl.eu/legacy/opus-100.php> | No license stated; the terms of each underlying OPUS corpus apply; to be confirmed | To be confirmed | B. Zhang, P. Williams, I. Titov, R. Sennrich, "Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation," ACL, 2020; J. Tiedemann, "Parallel Data, Tools and Interfaces in OPUS," LREC, 2012 |
| Tatoeba (Hugging Face `Helsinki-NLP/tatoeba`) | <https://tatoeba.org/en/terms_of_use> | Sentences default to CC BY 2.0 FR (attribution required); some sentences carry other licenses | Generally allowed with attribution, but check per sentence | J. Tiedemann, "Parallel Data, Tools and Interfaces in OPUS," LREC, 2012 (to be confirmed) |
