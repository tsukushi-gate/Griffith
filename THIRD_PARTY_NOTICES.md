# Third-Party Notices

Griffith (`griffith-logic`) incorporates, adapts, or reimplements material from
the third-party projects listed below. This file collects the required
copyright notices and license texts.

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
| [elkjs](https://github.com/kieler/elkjs) | EPL-2.0 | **Runtime dependency** of `tools/editor`, `tools/mdl-fpga` (viewer) and `tools/paper-tree` (graph layout). Notice obligation; no code copied. |
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

Griffith depends on the third-party Python packages listed below. They are
**not** vendored into this repository; pip or uv installs them from PyPI (and,
for the `cpu` / `cu128` extras on Linux and Windows, from the PyTorch package
index) at install time. Their licenses apply to those packages, not to
Griffith's own code. Several of them bundle further third-party components that
their own metadata and license files describe (for example, `numpy` declares
`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`).

### 5.1 Dependencies whose licensing warrants attention

Review these before redistributing Griffith together with its dependencies (for
example, in a bundled or containerized product).

| Package | License (as declared) | Pulled in by | Notes |
| --- | --- | --- | --- |
| `tqdm` | `MPL-2.0 AND MIT` | core | Dual-component licensing. MPL-2.0 is a file-level copyleft: redistributing modified `tqdm` sources requires publishing those modifications. Unmodified redistribution is unaffected. |
| `certifi` | `MPL-2.0` | core (via `requests`, `httpx`) | The Mozilla CA bundle. File-level copyleft as above; unmodified redistribution is unaffected. |
| NVIDIA CUDA components (`nvidia-*`, `cuda-bindings`, `cuda-toolkit`) | `LicenseRef-NVIDIA-Proprietary`, `NVIDIA Proprietary Software`, `LicenseRef-NVIDIA-SOFTWARE-LICENSE`, or not declared | core on Linux (via PyPI `torch`), `cu128` | **Not open source.** pip/uv download them from PyPI or the PyTorch index; this repository does not redistribute them. A container or bundle that includes them is subject to NVIDIA's license terms. See §5.4. |
| `paramiko` | `LGPL-2.1` | `cloud` (via `runpod`) | Weak copyleft. Used unmodified, as a dependency of the RunPod client. |
| `hypothesis` | `MPL-2.0` | `dev` | Test-time dependency only; not shipped with the distributed package. |
| `pathspec` | `Mozilla Public License 2.0 (MPL 2.0)` | `dev` (via `mypy`) | Type-check-time dependency only. |
| `python-sat` (`pysat`) | `MIT` | `sat` | The Python wrapper is MIT, but the **bundled SAT solvers are individually licensed**. Some solver versions (e.g. certain Glucose and Lingeling releases) carry non-commercial or research-only clauses. Review before redistributing binaries. |
| `fugashi` | `MIT AND BSD-3-Clause` | `sentiment` | MeCab bindings; the underlying MeCab code is BSD-3-Clause. |
| `unidic-lite` | `MIT` / `WTFPL` (package) + `BSD-3-Clause` (dictionary data) | `sentiment` | The dictionary data has its own license; check terms before redistributing the dictionary itself. |

### 5.2 Core runtime dependencies

What `pip install griffith-logic` installs: the `dependencies` of
`pyproject.toml` and everything they require. The Linux-only CUDA components
that PyPI's `torch` pulls in are listed in §5.4.

| Package | Version | License (as declared) |
| --- | --- | --- |
| `annotated-types` | 0.7.0 | MIT License |
| `anyio` | 4.13.0 | MIT |
| `certifi` | 2026.2.25 | MPL-2.0 |
| `charset-normalizer` | 3.4.7 | MIT |
| `click` | 8.5.0 | BSD-3-Clause |
| `colorama` | 0.4.6 | BSD License |
| `cwevals` | 0.0.1b1 | Apache-2.0 |
| `distro` | 1.9.0 | Apache License, Version 2.0 |
| `filelock` | 3.25.2 | MIT |
| `fsspec` | 2026.2.0 | BSD-3-Clause |
| `googleapis-common-protos` | 1.76.0 | Apache-2.0 |
| `h11` | 0.16.0 | MIT |
| `httpcore` | 1.0.9 | BSD-3-Clause |
| `httpx` | 0.28.1 | BSD-3-Clause |
| `idna` | 3.18 | BSD-3-Clause |
| `Jinja2` | 3.1.6 | BSD License |
| `MarkupSafe` | 3.0.3 | BSD-3-Clause |
| `mpmath` | 1.3.0 | BSD |
| `networkx` | 3.6.1 | BSD-3-Clause |
| `numpy` | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `opentelemetry-api` | 1.45.2 | Apache-2.0 |
| `opentelemetry-exporter-http-transport` | 0.66b2 | Apache-2.0 |
| `opentelemetry-exporter-otlp-common` | 0.66b2 | Apache-2.0 |
| `opentelemetry-exporter-otlp-proto-common` | 1.45.2 | Apache-2.0 |
| `opentelemetry-exporter-otlp-proto-http` | 1.45.2 | Apache-2.0 |
| `opentelemetry-proto` | 1.45.2 | Apache-2.0 |
| `opentelemetry-sdk` | 1.45.2 | Apache-2.0 |
| `opentelemetry-semantic-conventions` | 0.66b2 | Apache-2.0 |
| `packaging` | 26.0 | Apache-2.0 OR BSD-2-Clause |
| `platformdirs` | 4.9.6 | MIT |
| `protobuf` | 6.33.6 | 3-Clause BSD License |
| `pydantic` | 2.14.3 | MIT |
| `pydantic_core` | 2.50.4 | MIT |
| `PyYAML` | 6.0.3 | MIT |
| `requests` | 2.34.2 | Apache-2.0 |
| `setuptools` | 81.0.0 | MIT |
| `sniffio` | 1.3.1 | MIT OR Apache-2.0 |
| `sympy` | 1.14.0 | BSD |
| `torch` ‡ | 2.11.0 | BSD-3-Clause |
| `tqdm` | 4.70.1 | MPL-2.0 AND MIT |
| `triton` † | 3.6.0 | MIT License |
| `typing-inspection` | 0.4.5 | MIT |
| `typing_extensions` | 4.16.0 | PSF-2.0 |
| `urllib3` | 2.7.0 | MIT |
| `wandb` | 0.31.0 | MIT License |
| `xxhash` | 3.6.0 | BSD |

### 5.3 Optional extras

Packages that an extra adds on top of §5.2
(`pip install "griffith-logic[<extra>]"`). `cu128` resolves the same packages
as `cpu`, with CUDA builds of `torch` / `torchvision`.

| Package | Version | License (as declared) | Extras |
| --- | --- | --- | --- |
| `accelerate` | 1.15.0 | Apache | `text` |
| `aiodns` | 4.0.4 | MIT | `cloud` |
| `aiohappyeyeballs` | 2.6.1 | PSF-2.0 | `sentiment`, `wmt`, `text`, `cloud` |
| `aiohttp` | 3.14.3 | Apache-2.0 AND MIT | `sentiment`, `wmt`, `text`, `cloud` |
| `aiohttp-retry` | 2.9.1 | MIT | `cloud` |
| `aiosignal` | 1.4.0 | Apache 2.0 | `sentiment`, `wmt`, `text`, `cloud` |
| `annotated-doc` | 0.0.4 | MIT | `editor`, `fpga`, `text`, `cloud` |
| `ast_serialize` | 0.13.1 | MIT | `dev` |
| `attrs` | 26.1.0 | MIT | `sentiment`, `wmt`, `text`, `cloud` |
| `backoff` | 2.2.1 | MIT | `cloud` |
| `backports.zstd` | 1.6.0 | PSF-2.0 | `cloud` |
| `bcrypt` | 5.0.0 | Apache-2.0 | `cloud` |
| `boto3` | 1.43.46 | Apache-2.0 | `cloud` |
| `botocore` | 1.43.46 | Apache-2.0 | `cloud` |
| `brotli` | 1.2.0 | MIT | `cloud` |
| `brotlicffi` † | 1.2.0.1 | MIT | `cloud` |
| `cbor2` | 6.1.2 | MIT | `cloud` |
| `cffi` | 2.1.0 | MIT-0 | `cloud` |
| `cocotb` | 2.1.0 | BSD-3-Clause | `fpga` |
| `contourpy` | 1.3.3 | BSD License | `figures` |
| `coverage` | 7.14.3 | Apache-2.0 | `dev` |
| `cryptography` | 50.0.1 | Apache-2.0 OR BSD-3-Clause | `cloud` |
| `cycler` | 0.12.1 | BSD License | `figures` |
| `datasets` | 5.1.0 | Apache 2.0 | `sentiment`, `wmt`, `text` |
| `detect-installer` | 0.1.0 | 0BSD | `cloud` |
| `dill` | 0.4.1 | BSD-3-Clause | `sentiment`, `wmt`, `text` |
| `dnspython` | 2.8.0 | ISC | `cloud` |
| `email-validator` | 2.3.0 | Unlicense | `cloud` |
| `fastapi` | 0.162.0 | MIT | `editor`, `fpga`, `cloud` |
| `fastapi-cli` | 0.0.32 | MIT | `cloud` |
| `fastapi-cloud-cli` | 0.22.1 | MIT | `cloud` |
| `fastar` | 0.11.0 | MIT | `cloud` |
| `find_libpython` | 0.5.1 | MIT | `fpga` |
| `flatbuffers` | 25.12.19 | Apache 2.0 | `teachers` |
| `fonttools` | 4.63.0 | MIT | `figures` |
| `frozenlist` | 1.8.0 | Apache-2.0 | `sentiment`, `wmt`, `text`, `cloud` |
| `fugashi` | 1.5.2 | MIT AND BSD-3-Clause | `sentiment` |
| `grpclib` | 0.4.9 | BSD-3-Clause | `cloud` |
| `h2` | 4.4.1 | MIT | `cloud` |
| `hf-xet` | 1.8.0 | Apache-2.0 | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cu128` |
| `hpack` | 4.2.0 | MIT | `cloud` |
| `httpcore2` | 2.16.0 | BSD-3-Clause | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cloud`, `cu128` |
| `httptools` | 0.8.0 | MIT | `editor`, `cloud` |
| `httpx2` | 2.16.0 | BSD-3-Clause | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cloud`, `cu128` |
| `httpx2-jsfetch` † | 1.0 | BSD-3-Clause | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cloud`, `cu128` |
| `huggingface_hub` | 2.4.0 | Apache-2.0 | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cu128` |
| `hyperframe` | 6.1.0 | MIT License | `cloud` |
| `hypothesis` | 6.170.15 | MPL-2.0 | `dev` |
| `iniconfig` | 2.3.0 | MIT | `dev`, `fpga` |
| `inquirerpy` | 0.3.4 | MIT | `cloud` |
| `invoke` | 3.0.3 | BSD-2-Clause | `cloud` |
| `itsdangerous` | 2.2.0 | BSD License | `cloud` |
| `jmespath` | 1.1.0 | MIT | `cloud` |
| `joblib` | 1.5.3 | BSD-3-Clause | `text` |
| `kiwisolver` | 1.5.0 | BSD License | `figures` |
| `librt` | 0.16.0 | MIT | `dev` |
| `lxml` | 6.1.3 | BSD-3-Clause | `wmt` |
| `markdown-it-py` | 4.0.0 | MIT License | `text`, `cloud` |
| `matplotlib` | 3.11.2 | Python Software Foundation License | `figures` |
| `mdurl` | 0.1.2 | MIT License | `text`, `cloud` |
| `ml_dtypes` | 0.6.0 | Apache-2.0 | `teachers` |
| `modal` | 1.6.3 | Apache-2.0 | `cloud` |
| `multidict` | 6.7.1 | Apache License 2.0 | `sentiment`, `wmt`, `text`, `cloud` |
| `multiprocess` | 0.70.19 | BSD-3-Clause | `sentiment`, `wmt`, `text` |
| `mypy` | 2.4.1 | MIT | `dev` |
| `mypy_extensions` | 1.1.0 | MIT | `dev` |
| `ninja` | 1.13.2 | Apache-2.0 | `cuda` |
| `onnx` | 1.23.2 | Apache-2.0 | `teachers` |
| `onnxruntime` | 1.31.1 | MIT License | `teachers` |
| `pandas` | 3.1.1 | BSD-3-Clause | `sentiment`, `wmt`, `text` |
| `paramiko` | 5.0.0 | LGPL-2.1 | `cloud` |
| `pathspec` | 1.1.1 | Mozilla Public License 2.0 (MPL 2.0) | `dev` |
| `pfzy` | 0.3.4 | MIT | `cloud` |
| `pillow` | 12.3.0 | MIT-CMU | `cpu`, `figures`, `fpga`, `cu128` |
| `pluggy` | 1.6.0 | MIT | `dev`, `fpga` |
| `portalocker` | 3.2.0 | BSD-3-Clause | `wmt` |
| `prettytable` | 3.18.0 | BSD-3-Clause | `cloud` |
| `prompt_toolkit` | 3.0.52 | BSD License | `cloud` |
| `propcache` | 0.4.1 | Apache-2.0 | `sentiment`, `wmt`, `text`, `cloud` |
| `psutil` | 7.2.2 | BSD-3-Clause | `text`, `cloud` |
| `py-cpuinfo` | 9.0.0 | MIT | `cloud` |
| `pyarrow` | 26.0.1 | Apache-2.0 | `sentiment`, `wmt`, `text` |
| `pycares` | 5.0.1 | MIT | `cloud` |
| `pycparser` | 3.0 | BSD-3-Clause | `cloud` |
| `pydantic-extra-types` | 2.11.1 | MIT | `cloud` |
| `pydantic-settings` | 2.14.2 | MIT | `cloud` |
| `Pygments` | 2.20.0 | BSD-2-Clause | `dev`, `fpga`, `text`, `cloud` |
| `PyNaCl` | 1.6.2 | Apache-2.0 | `cloud` |
| `pyparsing` | 3.3.2 | MIT | `figures` |
| `pyserial` | 3.5 | BSD | `fpga` |
| `pytest` | 9.2.0 | MIT | `dev`, `fpga` |
| `pytest-cov` | 7.1.0 | MIT | `dev` |
| `python-dateutil` | 2.9.0.post0 | BSD License / Apache Software License | `figures`, `sentiment`, `wmt`, `text`, `cloud` |
| `python-dotenv` | 1.2.2 | BSD-3-Clause | `editor`, `cloud` |
| `python-multipart` | 0.0.33 | Apache-2.0 | `editor`, `cloud` |
| `python-sat` | 1.9.dev21 | MIT | `sat` |
| `pywin32` † | 311 | PSF | `wmt` |
| `regex` | 2026.4.4 | Apache-2.0 AND CNRI-Python | `wmt`, `text` |
| `rich` | 15.0.0 | MIT | `text`, `cloud` |
| `rich-toolkit` | 0.20.1 | MIT | `cloud` |
| `rignore` | 0.7.6 | MIT | `cloud` |
| `runpod` | 1.10.1 | MIT License | `cloud` |
| `s3transfer` | 0.19.1 | Apache License 2.0 | `cloud` |
| `sacrebleu` | 2.6.0 | Apache-2.0 | `wmt` |
| `sacremoses` | 0.2.0 | MIT License | `text` |
| `safetensors` | 0.8.0 | Apache Software License | `cpu`, `teachers`, `text`, `cu128` |
| `sentencepiece` | 0.2.3 | Apache-2.0 | `text` |
| `sentry-sdk` | 2.57.0 | MIT | `cloud` |
| `shellingham` | 1.5.4 | ISC License | `text`, `cloud` |
| `six` | 1.17.0 | MIT | `figures`, `sat`, `sentiment`, `wmt`, `text`, `cloud` |
| `sortedcontainers` | 2.4.0 | Apache 2.0 | `dev` |
| `starlette` | 1.3.1 | BSD-3-Clause | `editor`, `fpga`, `cloud` |
| `synchronicity` | 0.12.7 | Apache Software License | `cloud` |
| `tabulate` | 0.10.0 | MIT | `wmt` |
| `timm` | 1.0.31 | Apache-2.0 | `cpu`, `cu128` |
| `tokenizers` | 0.23.3 | Apache Software License | `text` |
| `toml` | 0.10.2 | MIT | `cloud` |
| `tomli` | 2.4.1 | MIT | `dev`, `cloud` |
| `tomlkit` | 0.15.0 | MIT | `cloud` |
| `torchvision` ‡ | 0.26.0 | BSD | `cpu`, `cu128` |
| `tqdm-loggable` | 0.4.1 | MIT | `cloud` |
| `transformers` | 5.22.2 | Apache 2.0 License | `text` |
| `truststore` | 0.10.5 | MIT | `cpu`, `sentiment`, `wmt`, `teachers`, `text`, `cloud`, `cu128` |
| `typer` | 0.24.1 | MIT | `text`, `cloud` |
| `types-certifi` | 2021.10.8.3 | Apache-2.0 license | `cloud` |
| `types-toml` | 0.10.8.20260518 | Apache-2.0 | `cloud` |
| `tzdata` † | 2026.1 | Apache-2.0 | `sentiment`, `wmt`, `text` |
| `unidic-lite` | 1.0.8 | MIT License | `sentiment` |
| `uvicorn` | 0.54.0 | BSD-3-Clause | `editor`, `fpga`, `cloud` |
| `uvloop` | 0.22.1 | MIT License | `editor`, `cloud` |
| `wandb-workspaces` | 0.4.13 | Apache-2.0 | `cloud` |
| `watchdog` | 6.0.0 | Apache-2.0 | `cloud` |
| `watchfiles` | 1.2.0 | MIT | `editor`, `cloud` |
| `wcwidth` | 0.8.2 | MIT | `cloud` |
| `websockets` | 17.4 | BSD-3-Clause | `editor`, `visualizer`, `cloud` |
| `yarl` | 1.23.0 | Apache-2.0 | `sentiment`, `wmt`, `text`, `cloud` |

† Not installable on the platform where this list was generated (macOS,
arm64); the license was read from the PyPI metadata of the locked version.

‡ On Linux and Windows the `cpu` extra resolves `torch` 2.12.1+cpu /
`torchvision` 0.27.1+cpu, and `cu128` resolves `torch` 2.11.0+cu128 /
`torchvision` 0.26.0+cu128, from the PyTorch package index. These local builds
are not on PyPI; the license shown is that of the macOS wheel of the same
project.

### 5.4 Linux-only CUDA components (pulled in by `torch`)

On Linux, PyPI's `torch` wheel depends on NVIDIA's CUDA packages, and the
`cu128` extra pulls in their CUDA 12.8 counterparts. They are not installed on
macOS or Windows. The license column is what PyPI's metadata for the locked
version declares; "(not declared)" means that metadata carries neither a
license field nor a license classifier, so consult NVIDIA's terms for that
package before redistributing it. The exact versions are pinned in `uv.lock`
(`uv export --frozen --no-dev` without extras, or with `--extra cu128`).

| License (as declared on PyPI) | Linux, default PyPI `torch` (CUDA 13) | Linux, `cu128` extra (CUDA 12.8) |
| --- | --- | --- |
| (not declared) | `cuda-toolkit`, `nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13` | `cuda-toolkit`, `nvidia-cudnn-cu12`, `nvidia-nccl-cu12`, `nvidia-nvshmem-cu12` |
| Apache 2.0 | `nvidia-nvtx` | `nvidia-nvtx-cu12` |
| Apache-2.0 | `cuda-pathfinder` | — |
| LicenseRef-NVIDIA-Proprietary | `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-nvjitlink` | — |
| LicenseRef-NVIDIA-SOFTWARE-LICENSE | `cuda-bindings` | `cuda-bindings` |
| NVIDIA Proprietary Software | `nvidia-cusparselt-cu13` | `nvidia-cublas-cu12`, `nvidia-cuda-cupti-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cufft-cu12`, `nvidia-cufile-cu12`, `nvidia-curand-cu12`, `nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusparselt-cu12`, `nvidia-nvjitlink-cu12` |

### 5.5 How this inventory was generated

Generated on 2026-11-22 from `uv.lock` at griffith-logic 0.1.0, on macOS
(arm64):

1. Build a throw-away environment with every extra except `cu128` (it
   conflicts with `cpu`):
   `UV_PROJECT_ENVIRONMENT=<tmp>/venv uv sync --frozen --no-dev --extra cpu --extra cuda --extra dev --extra editor --extra figures --extra sat --extra sentiment --extra visualizer --extra fpga --extra wmt --extra teachers --extra text --extra cloud`
2. Membership: `uv export --frozen --no-dev --no-emit-project --no-hashes`
   without extras (§5.2), with each `--extra <name>` (§5.3), and with
   `--extra cu128` (§5.4). An extra is credited with the packages it adds
   beyond §5.2.
3. License: read with `importlib.metadata` inside that environment
   (`UV_PROJECT_ENVIRONMENT=<tmp>/venv uv run --no-sync python`), taking
   `License-Expression`, else a one-line `License` field, else the
   `License ::` Trove classifiers (also used when the `License` field names no
   license, e.g. `python-dateutil`'s "Dual License"). Packages that cannot be
   installed on the generating platform were read from
   `https://pypi.org/pypi/<name>/<version>/json`.

Values are copied verbatim and not normalised to SPDX identifiers. The `docs`
dependency group (used only to build the documentation site) is not included.

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
visual editor, the mdl-fpga viewer and the paper tree; `docs-site` also receives
it as a dependency of `mermaid`. It is used unmodified, as an npm
dependency. EPL-2.0 requires that recipients of a distribution containing the
program are told where to obtain the source and under which terms.

- Project: elkjs (Eclipse Layout Kernel)
- Source: <https://github.com/kieler/elkjs>
- License: EPL-2.0 (full text: <https://www.eclipse.org/legal/epl-2.0/>)
- Modifications by Griffith: none.

### 6.4 Front-end dependency license inventory

Read on 2026-11-22 from the `license` field that npm records for each package
in each workspace's `package-lock.json` (no install and no network access).
"Production" is every entry without `"dev": true`. The root entry of each lock
file is the workspace itself (a `private` package covered by this repository's
Apache-2.0 [`LICENSE`](LICENSE)) and is not counted. A lock file lists the
optional platform binaries for every OS (`lightningcss-*`, `@img/sharp-*`, …),
so the counts are larger than one installed tree.

| Workspace | Production (count) | Dev only (count) |
| --- | --- | --- |
| `docs-site` | MIT 485, ISC 44, MPL-2.0 24, Apache-2.0 22, BSD-3-Clause 11, LGPL-3.0-or-later 10, BSD-2-Clause 9, "Apache-2.0 AND LGPL-3.0-or-later" 3, BlueOak-1.0.0 3, CC0-1.0 2, "Apache-2.0 AND LGPL-3.0-or-later AND MIT" 1, Python-2.0 1, "(MPL-2.0 OR Apache-2.0)" 1, "EPL-2.0 OR GPL-3.0-or-later" 1, Unlicense 1, 0BSD 1, no license field 1 | — (every dependency is a production dependency) |
| `tools/editor/frontend` | MIT 231, ISC 12, MPL-2.0 12, Apache-2.0 5, BSD-3-Clause 2, BSD-2-Clause 1, EPL-2.0 1, 0BSD 1, no license field 1 | MIT 179, Apache-2.0 20, ISC 11, BSD-2-Clause 8, MIT-0 2, BSD-3-Clause 2, BlueOak-1.0.0 2, Python-2.0 1, CC-BY-4.0 1, CC0-1.0 1 |
| `tools/mdl-fpga/frontend` ¹ | MIT 138, MPL-2.0 12, ISC 10, Apache-2.0 2, BSD-3-Clause 2, EPL-2.0 1, 0BSD 1, **unknown 47** | MIT 98, ISC 5, Apache-2.0 4, BSD-2-Clause 1, **unknown 31** |
| `tools/paper-tree` | MIT 253, ISC 12, MPL-2.0 12, BSD-3-Clause 2, Apache-2.0 1, EPL-2.0 1, BSD-2-Clause 1, 0BSD 1 | MIT 43, Apache-2.0 5, ISC 4, CC-BY-4.0 1, BSD-2-Clause 1, BSD-3-Clause 1 |
| `tools/visualizer/frontend` | MIT 43, Apache-2.0 3, BSD-2-Clause 1, ISC 1, 0BSD 1, no license field 1 | MIT 61, ISC 1, BSD-3-Clause 1, Apache-2.0 1 |
| `tools/webbench` | MIT 55, Apache-2.0 12, BSD-3-Clause 12, ISC 5, "Apache-2.0 AND MIT" 1, BSD-2-Clause 1 | MIT 111, MPL-2.0 12, Apache-2.0 7, ISC 3, BSD-3-Clause 2 |

¹ `tools/mdl-fpga/frontend/package-lock.json` records no `license` field at
all. 274 of its 352 entries were filled in from another workspace's lock entry
with the same `integrity` hash (that is, the same tarball). The other 78 are
**unknown** here — production: 36 `@radix-ui/*` packages, `radix-ui`,
`@floating-ui/{core,dom,react-dom,utils}`, `@xyflow/{react,system}`,
`enhanced-resolve`, `lucide-react`, `nanoid`, `postcss`; dev: `jsdom`,
`@asamuzakjp/css-color`, `@csstools/{color-helpers,css-calc,css-color-parser,css-parser-algorithms,css-tokenizer}`,
`agent-base`, `baseline-browser-mapping`, `browserslist`, `caniuse-lite`,
`cssstyle`, `data-urls`, `electron-to-chromium`, `html-encoding-sniffer`,
`http-proxy-agent`, `https-proxy-agent`, `lru-cache`, `node-releases`,
`nwsapi`, `rrweb-cssom` (2 versions), `tldts`, `tldts-core`, `tough-cookie`,
`tr46`, `webidl-conversions`, `whatwg-encoding`, `whatwg-mimetype`,
`whatwg-url`, `ws`. Resolve them (for example with
`npm ci && npx license-checker --production --summary` in that workspace)
before redistributing a built mdl-fpga viewer.

Among the entries whose license is recorded, **no dependency is licensed only
under GPL, AGPL, SSPL, or a non-commercial (CC-BY-NC) license**. The only GPL
option is `elkjs` 0.12.1 in `docs-site`, which is dual-licensed and used under
EPL-2.0. LGPL appears only in the `sharp` / libvips image binaries of
`docs-site` (below).

Packages that are not permissive-by-default, or that have no recorded license,
and what they mean in practice:

| Package | License (from the lock file) | Where | Notes |
| --- | --- | --- | --- |
| `elkjs` | `EPL-2.0` (0.11.1); `EPL-2.0 OR GPL-3.0-or-later` (0.12.1) | editor, mdl-fpga, paper-tree (0.11.1); docs-site (0.12.1, via `mermaid`) | Weak copyleft with a notice obligation. Used unmodified; see §6.3. For the dual-licensed 0.12.1, Griffith relies on the EPL-2.0 option. |
| `lightningcss` (+ platform binaries) | `MPL-2.0` | docs-site, editor, mdl-fpga, paper-tree; webbench (dev only) | File-level copyleft. Build-time CSS transformer, used unmodified; publishing modifications of its own sources would require releasing them. |
| `dompurify` | `(MPL-2.0 OR Apache-2.0)` | docs-site | Dual-licensed; Griffith relies on the Apache-2.0 option. |
| `@img/sharp-libvips-*` (10 platform binaries) | `LGPL-3.0-or-later` | docs-site | Prebuilt libvips binaries pulled in by `sharp` (Astro image optimization). Installed from npm at build time and not redistributed by this repository. Shipping one inside a binary bundle triggers LGPL relinking obligations. |
| `@img/sharp-win32-*`, `@img/sharp-wasm32` | `Apache-2.0 AND LGPL-3.0-or-later` (wasm32: `… AND MIT`) | docs-site | `sharp` builds that embed libvips; same consideration as above. |
| `argparse` | `Python-2.0` | docs-site; editor (dev only) | Permissive; PSF license. |
| `mdn-data` | `CC0-1.0` | docs-site; editor (dev only) | Public-domain dedication (data, not code). |
| `caniuse-lite` | `CC-BY-4.0` | editor, paper-tree (dev only) | Browser-support data read by `browserslist` at build time. Attribution license. |
| `robust-predicates` | `Unlicense` | docs-site | Public-domain dedication. |
| `common-ancestor-path`, `lru-cache`, `sax`, `minimatch` | `BlueOak-1.0.0` | docs-site; editor (dev only: `lru-cache`, `minimatch`) | Permissive, OSI-approved. |
| `@csstools/color-helpers`, `@csstools/css-syntax-patches-for-csstree` | `MIT-0` | editor (dev only) | Permissive (MIT without the attribution condition). |
| `khroma` | no `license` field | docs-site | The lock records none. The 2026-09-24 `license-checker` run inferred MIT from its README (upstream ships MIT terms there rather than in a `LICENSE` file). |
| `json-bignum` | no `license` field | editor, visualizer | The lock records none. Check upstream before redistributing a built bundle. |

To regenerate a workspace's counts:

```bash
python3 -c 'import json,sys,collections; p=json.load(open(sys.argv[1]))["packages"]; print(collections.Counter(("dev" if v.get("dev") else "prod", v.get("license")) for k, v in p.items() if k))' <workspace>/package-lock.json
```

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
