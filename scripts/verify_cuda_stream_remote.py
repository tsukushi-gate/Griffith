#!/usr/bin/env python
"""CUDAストリームAPI(ミラー)+dense-fuse 48KB化の実機検証・A/B。

このMacにCUDA実機/nvccが無いため、リモートGPU機(ssh)へソースをrsyncし、
nvccでビルド・実行して結果を回収する(`scripts/verify_cuda_remote.py`と
同じ設計、`scripts/bench_dense_gpu.py`のMetalベンチドライバ(_BARE_DRIVER_TMPL/
_STREAM_DRIVER_TMPL)と対称なCUDA版ドライバを使う)。

フェーズ:
    verify       ストリーム経路(submit/collect)⇔逐次経路が全枚整数厳密一致
                 することを確認する(dense bits経路 + 画像経路)。
    bench-stream ストリーム(--n-streams)ありなしのスループットA/B。
    bench-fuse   dense-fuse(48KB)ありなしのスループットA/B + 整数一致。
    calib        mdl_gpu_pick_rのtarget_threads/MDL_BLOCKのA/B掃引。

使い方例:
    uv run python scripts/verify_cuda_stream_remote.py --phase verify \\
        --host <user>@<gpu-host>

    uv run python scripts/verify_cuda_stream_remote.py --phase bench-stream \\
        --host <user>@<gpu-host> \\
        --models har_dlgn_k600_gumbel:128 har_dlgn_k6000_gumbel:128 \\
        --sets 5 --out results/deploy/cuda_streams_k55_ab.json
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cudabackend import (  # noqa: E402
    CUDA_DENSE_FUSE_MAX_SHARED_BYTES,
    generate_cuda,
)
from griffith_logic.deploy.plan import build_plan  # noqa: E402
from griffith_logic.ir.serialize import ir_to_dict, load_ir  # noqa: E402

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
RUNTIME = REPO / 'runtime'
COMPILE_PIPELINE = REPO / 'results' / 'compile-pipeline'

DEFAULT_NVCC_PATH = '$HOME/.local/micromamba/envs/llamabuild/bin'


def sh(host: str, cmd: str, *, check: bool = True, timeout: int = 280) -> str:
    proc = subprocess.run(
        ['ssh', '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=10',
         '-o', 'BatchMode=yes', host, cmd],
        capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f'リモート実行に失敗:\n{cmd}\n{proc.stderr}')
    return proc.stdout


def rsync_to(host: str, local_paths: list[Path], remote_dir: str) -> None:
    subprocess.run(
        ['rsync', '-az', *[str(p) for p in local_paths],
         f'{host}:{remote_dir}/'], check=True, timeout=280)


def rsync_from(host: str, remote_path: str, local_path: Path) -> None:
    subprocess.run(['rsync', '-az', f'{host}:{remote_path}', str(local_path)],
                   check=True, timeout=280)


# ---------------------------------------------------------------------------
# モデル読み込み(webbenchのmodels/直下 + mnist_adder特例、同じ手順)
# ---------------------------------------------------------------------------

def load_model_bundle(spec: str) -> dict:
    """`spec`がwebbenchモデル名ならそのまま、'mnist_adder'系なら
    `results/compile-pipeline/*_ir_final.json`をload_ir経由で読む
    (タスク指示・results/deploy/cuda_dense_packed_k53_ab.jsonの
    methodologyと同じ手順。mnist_adderはwebbenchのmodels/に無いため)。
    """
    if spec.startswith('mnist_adder'):
        path = COMPILE_PIPELINE / 'mnist_adder_s0_20260712_092520_ir_final.json'
        graph = load_ir(str(path))
        return build_bundle({'ir': ir_to_dict(graph)})
    cand = MODELS_DIR / f'{spec}.json'
    if not cand.is_file():
        raise SystemExit(f'モデルが見つからない: {spec!r}')
    doc = json.loads(cand.read_text(encoding='utf-8'))
    return build_bundle(doc)


# ---------------------------------------------------------------------------
# CUDAドライバ(scripts/bench_dense_gpu.pyのMetal版と対称)
# ---------------------------------------------------------------------------

_BARE_DRIVER_TMPL = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {{
int mdl_cuda_infer_batch(const unsigned char*, int*, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_init(void);
}}

static double now_ms(void) {{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}}

static int cmp_d(const void *a, const void *b) {{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}}

int main(int argc, char **argv) {{
  const int reps = argc > 1 ? atoi(argv[1]) : {reps};
  const int warmup = argc > 2 ? atoi(argv[2]) : {warmup};
  const int batch = {batch};
  const int in_w = mdl_cuda_input_width();
  const int out_w = mdl_cuda_output_width();
  unsigned char *bits = (unsigned char *)malloc((size_t)batch * (size_t)in_w);
  int *scores = (int *)malloc((size_t)batch * (size_t)out_w * sizeof(int));
  double *t = (double *)malloc((size_t)reps * sizeof(double));
  size_t k;
  int r;

  for (k = 0; k < (size_t)batch * (size_t)in_w; k++) {{
    bits[k] = (unsigned char)((k * 2654435761u) & 1);
  }}
  if (mdl_cuda_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  for (r = 0; r < warmup; r++) {{
    if (mdl_cuda_infer_batch(bits, scores, batch) != 0) {{
      fprintf(stderr, "warmup infer failed\\n");
      return 2;
    }}
  }}

  for (r = 0; r < reps; r++) {{
    const double t0 = now_ms();
    if (mdl_cuda_infer_batch(bits, scores, batch) != 0) {{
      fprintf(stderr, "infer failed\\n");
      return 2;
    }}
    t[r] = now_ms() - t0;
  }}
  qsort(t, (size_t)reps, sizeof(double), cmp_d);
  {{
    double sum = 0.0;
    for (r = 0; r < reps; r++) sum += t[r];
    printf("e2e_ms_median %.5f\\n", t[reps / 2]);
    printf("e2e_ms_best %.5f\\n", t[0]);
    printf("e2e_ms_mean %.5f\\n", sum / reps);
    printf("batch %d\\n", batch);
    printf("reps %d\\n", reps);
    printf("throughput_median_per_s %.2f\\n", (double)batch * 1000.0 / t[reps / 2]);
    printf("throughput_mean_per_s %.2f\\n", (double)batch * (double)reps * 1000.0 / sum);
  }}
  return 0;
}}
'''

# ストリーム・パイプライン化(Metalミラー)版。chunk枚ずつNSLOT本を
# 先行投入して後ろから回収する定常パイプラインをreps回ぶん回す
# (scripts/bench_dense_gpu.py::_STREAM_DRIVER_TMPLと同じ考え方)。
_STREAM_DRIVER_TMPL = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {{
int mdl_cuda_infer_batch_submit(const unsigned char*, int, int);
int mdl_cuda_infer_batch_collect(int*, int, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_init(void);
}}

#define NSLOT {n_streams}

static double now_ms(void) {{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}}

static unsigned char *g_bits;
static int *g_scores;
static int g_batch;

static void run_n(int n) {{
  int pending = 0, next_submit = 0, next_collect = 0;
  while (next_collect < n) {{
    while (next_submit < n && pending < NSLOT) {{
      const int slot = next_submit % NSLOT;
      if (mdl_cuda_infer_batch_submit(g_bits, g_batch, slot) != 0) {{
        fprintf(stderr, "submit failed at k=%d\\n", next_submit);
        exit(2);
      }}
      next_submit++;
      pending++;
    }}
    {{
      const int slot = next_collect % NSLOT;
      if (mdl_cuda_infer_batch_collect(g_scores, g_batch, slot) != 0) {{
        fprintf(stderr, "collect failed at k=%d\\n", next_collect);
        exit(2);
      }}
      next_collect++;
      pending--;
    }}
  }}
}}

int main(int argc, char **argv) {{
  const int reps = argc > 1 ? atoi(argv[1]) : {reps};
  const int warmup = argc > 2 ? atoi(argv[2]) : {warmup};
  const int in_w = mdl_cuda_input_width();
  const int out_w = mdl_cuda_output_width();
  size_t k;
  g_batch = {batch};
  g_bits = (unsigned char *)malloc((size_t)g_batch * (size_t)in_w);
  g_scores = (int *)malloc((size_t)g_batch * (size_t)out_w * sizeof(int));
  for (k = 0; k < (size_t)g_batch * (size_t)in_w; k++) {{
    g_bits[k] = (unsigned char)((k * 2654435761u) & 1);
  }}
  if (mdl_cuda_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  run_n(warmup);

  {{
    const double t0 = now_ms();
    run_n(reps);
    const double total_ms = now_ms() - t0;
    printf("total_ms %.5f\\n", total_ms);
    printf("chunks %d\\n", reps);
    printf("batch %d\\n", g_batch);
    printf("n_streams %d\\n", NSLOT);
    printf("throughput_per_s %.2f\\n",
          (double)reps * (double)g_batch * 1000.0 / total_ms);
    printf("e2e_ms_per_chunk_mean %.5f\\n", total_ms / (double)reps);
  }}
  return 0;
}}
'''

# 全枚整数厳密一致(逐次 vs ストリーム)。tests/deploy/test_cuda_stream.pyの
# _BITS_DRIVER_TMPLと同型(Metal版test_metal_stream.pyのCUDAミラー)。
_VERIFY_DRIVER_TMPL = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {{
int mdl_cuda_infer_batch(const unsigned char*, int*, int);
int mdl_cuda_infer_batch_submit(const unsigned char*, int, int);
int mdl_cuda_infer_batch_collect(int*, int, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_max_batch(void);
int mdl_cuda_init(void);
}}

#define K {k}
#define NSLOT {nslot}

static const int NS[K] = {{{ns_csv}}};

int main(void) {{
  const int in_w = mdl_cuda_input_width();
  const int out_w = mdl_cuda_output_width();
  unsigned char *inputs[K];
  int *seq[K];
  int *strm[K];
  int k;
  size_t i;

  if (mdl_cuda_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  for (k = 0; k < K; k++) {{
    const int n = NS[k];
    inputs[k] = (unsigned char*)malloc((size_t)n * in_w);
    seq[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    strm[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    for (i = 0; i < (size_t)n * in_w; i++) {{
      inputs[k][i] = (unsigned char)(((i + 1) * 2654435761u * (k + 1)) & 1);
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (mdl_cuda_infer_batch(inputs[k], seq[k], NS[k]) != 0) {{
      fprintf(stderr, "seq infer failed k=%d\\n", k); return 2;
    }}
  }}

  {{
    int pending = 0, next_submit = 0, next_collect = 0;
    while (next_collect < K) {{
      while (next_submit < K && pending < NSLOT) {{
        const int slot = next_submit % NSLOT;
        if (mdl_cuda_infer_batch_submit(inputs[next_submit], NS[next_submit],
                                        slot) != 0) {{
          fprintf(stderr, "submit failed k=%d\\n", next_submit); return 2;
        }}
        next_submit++; pending++;
      }}
      {{
        const int slot = next_collect % NSLOT;
        if (mdl_cuda_infer_batch_collect(strm[next_collect], NS[next_collect],
                                         slot) != 0) {{
          fprintf(stderr, "collect failed k=%d\\n", next_collect); return 2;
        }}
        next_collect++; pending--;
      }}
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (memcmp(seq[k], strm[k], (size_t)NS[k] * out_w * sizeof(int)) != 0) {{
      fprintf(stderr, "MISMATCH at k=%d (n=%d)\\n", k, NS[k]);
      return 1;
    }}
  }}
  printf("OK\\n");
  return 0;
}}
'''

_VERIFY_IMAGE_DRIVER_TMPL = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {{
int mdl_cuda_infer_image_batch(const unsigned char*, int*, int);
int mdl_cuda_infer_image_batch_submit(const unsigned char*, int, int);
int mdl_cuda_infer_image_batch_collect(int*, int, int);
int mdl_cuda_output_width(void);
int mdl_cuda_init(void);
}}

#define K {k}
#define NSLOT {nslot}
static const int n_img = {n_img};
static const int NS[K] = {{{ns_csv}}};

int main(void) {{
  const int out_w = mdl_cuda_output_width();
  unsigned char *images[K];
  int *seq[K];
  int *strm[K];
  int k;
  size_t i;

  if (mdl_cuda_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  for (k = 0; k < K; k++) {{
    const int n = NS[k];
    images[k] = (unsigned char*)malloc((size_t)n * n_img);
    seq[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    strm[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    for (i = 0; i < (size_t)n * n_img; i++) {{
      images[k][i] = (unsigned char)(((i + 7) * 2654435761u * (k + 3)) >> 24);
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (mdl_cuda_infer_image_batch(images[k], seq[k], NS[k]) != 0) {{
      fprintf(stderr, "seq infer failed k=%d\\n", k); return 2;
    }}
  }}

  {{
    int pending = 0, next_submit = 0, next_collect = 0;
    while (next_collect < K) {{
      while (next_submit < K && pending < NSLOT) {{
        const int slot = next_submit % NSLOT;
        if (mdl_cuda_infer_image_batch_submit(images[next_submit],
                                              NS[next_submit], slot) != 0) {{
          fprintf(stderr, "submit failed k=%d\\n", next_submit); return 2;
        }}
        next_submit++; pending++;
      }}
      {{
        const int slot = next_collect % NSLOT;
        if (mdl_cuda_infer_image_batch_collect(strm[next_collect],
                                               NS[next_collect], slot) != 0) {{
          fprintf(stderr, "collect failed k=%d\\n", next_collect); return 2;
        }}
        next_collect++; pending--;
      }}
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (memcmp(seq[k], strm[k], (size_t)NS[k] * out_w * sizeof(int)) != 0) {{
      fprintf(stderr, "MISMATCH at k=%d (n=%d)\\n", k, NS[k]);
      return 1;
    }}
  }}
  printf("OK\\n");
  return 0;
}}
'''


# ---------------------------------------------------------------------------
# ビルド・実行(リモート)
# ---------------------------------------------------------------------------

def stage_runtime(work: Path) -> None:
    for rel in ('cuda/mdl_cuda.h', 'cuda/mdl_cuda.cu', 'gpu/mdl_gpu_kernels.h'):
        (work / Path(rel).name).write_text(
            (RUNTIME / rel).read_text(encoding='utf-8'), encoding='utf-8')


def nvcc_build(host: str, remote_dir: str, env: str, arch: str, *,
              sources: list[str], out: str, extra_flags: str = '') -> None:
    srcs = ' '.join(sources)
    sh(host, f'cd {remote_dir} && {env} && '
             f'nvcc -O2 -arch={arch} -I. {extra_flags} {srcs} -o {out}',
       timeout=280)


def run_remote(host: str, remote_dir: str, cmd: str, *,
               timeout: int = 280) -> str:
    return sh(host, f'cd {remote_dir} && {cmd}', timeout=timeout)


def parse_kv_lines(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r'(\w+) ([-\d.]+)', line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        out[key] = float(val) if ('.' in val or 'e' in val.lower()) else int(val)
    return out


def median(xs: list[float]) -> float:
    return float(np.median(xs))


def write_merged(out_path: Path, payload: dict) -> None:
    """既存ファイルがあれば `results` キーをモデル単位でマージしてから書く
    (このスクリプトを--modelsを変えて複数回呼んでも、以前の結果を
    上書きで消さない)。"""
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            existing = {}
        merged_results = dict(existing.get('results', {}))
        for name, val in payload.get('results', {}).items():
            if isinstance(val, dict) and isinstance(
                    merged_results.get(name), dict):
                merged_results[name] = {**merged_results[name], **val}
            else:
                merged_results[name] = val
        payload = {**existing, **payload, 'results': merged_results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding='utf-8')


# ---------------------------------------------------------------------------
# フェーズ: verify(全枚整数一致、逐次⇔ストリーム)
# ---------------------------------------------------------------------------

def phase_verify(args) -> int:
    host = args.host
    remote_dir = args.remote_dir
    env = f'export PATH={args.nvcc_path}:$PATH'
    work = Path(tempfile.mkdtemp(prefix='mdl_cuda_verify_'))
    sh(host, f'mkdir -p {remote_dir}')
    stage_runtime(work)

    failures = 0
    # 1) dense bits経路(mnist_adderで、submit時のpackも含めて確認)
    bundle = load_model_bundle('mnist_adder')
    for filename, text in generate_cuda(
            bundle, name='model_cuda', batch=8, n_streams=3).items():
        (work / filename).write_text(text, encoding='utf-8')
    rng = np.random.default_rng(0)
    k, nslot, batch = 20, 3, 8
    ns = rng.integers(1, batch + 1, size=k).tolist()
    (work / 'verify_bits.cu').write_text(
        _VERIFY_DRIVER_TMPL.format(k=k, nslot=nslot,
                                   ns_csv=','.join(str(n) for n in ns)),
        encoding='utf-8')
    rsync_to(host, sorted(work.glob('*')), remote_dir)
    try:
        nvcc_build(host, remote_dir, env, args.arch,
                  sources=['model_cuda.cu', 'mdl_cuda.cu', 'verify_bits.cu'],
                  out='verify_bits_test')
        out = run_remote(host, remote_dir, './verify_bits_test')
        print('[verify] mnist_adder bits(streaming submit時のpack含む):',
              out.strip())
        if 'OK' not in out:
            failures += 1
    except RuntimeError as e:
        print('[verify] mnist_adder bits: FAIL\n', e)
        failures += 1

    # 2) 画像経路(seg_convdlgn_k32)
    bundle2 = load_model_bundle('seg_convdlgn_k32')
    p = bundle2['preproc']
    n_img = p['rgb_channels'] * p['height'] * p['width']
    for filename, text in generate_cuda(
            bundle2, name='model_cuda', batch=6, n_streams=2).items():
        (work / filename).write_text(text, encoding='utf-8')
    k2, nslot2, batch2 = 16, 2, 6
    ns2 = rng.integers(1, batch2 + 1, size=k2).tolist()
    (work / 'verify_image.cu').write_text(
        _VERIFY_IMAGE_DRIVER_TMPL.format(
            k=k2, nslot=nslot2, n_img=n_img,
            ns_csv=','.join(str(n) for n in ns2)),
        encoding='utf-8')
    rsync_to(host, [work / 'model_cuda.cu', work / 'verify_image.cu'],
             remote_dir)
    try:
        nvcc_build(host, remote_dir, env, args.arch,
                  sources=['model_cuda.cu', 'mdl_cuda.cu', 'verify_image.cu'],
                  out='verify_image_test')
        out = run_remote(host, remote_dir, './verify_image_test')
        print('[verify] seg_convdlgn_k32 image:', out.strip())
        if 'OK' not in out:
            failures += 1
    except RuntimeError as e:
        print('[verify] seg_convdlgn_k32 image: FAIL\n', e)
        failures += 1

    print(f'\nverifyフェーズ: {"全て一致" if failures == 0 else f"{failures}件失敗"}')
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# フェーズ: bench-stream(ストリームA/B)
# ---------------------------------------------------------------------------

def phase_bench_stream(args) -> int:
    host = args.host
    remote_dir = args.remote_dir
    env = f'export PATH={args.nvcc_path}:$PATH'
    sh(host, f'mkdir -p {remote_dir}')
    results: dict = {}

    for spec in args.models:
        if ':' in spec:
            name, batches = spec.split(':', 1)
            batch_list = [int(b) for b in batches.split(',')]
        else:
            name, batch_list = spec, [args.default_batch]
        bundle = load_model_bundle(name)
        results.setdefault(name, {})
        for batch in batch_list:
            work = Path(tempfile.mkdtemp(prefix=f'mdl_cuda_bs_{name}_'))
            stage_runtime(work)
            for filename, text in generate_cuda(
                    bundle, name='model_cuda', batch=batch,
                    n_streams=args.n_streams).items():
                (work / filename).write_text(text, encoding='utf-8')
            (work / 'bare_driver.cu').write_text(
                _BARE_DRIVER_TMPL.format(batch=batch, reps=args.reps,
                                         warmup=args.warmup),
                encoding='utf-8')
            (work / 'stream_driver.cu').write_text(
                _STREAM_DRIVER_TMPL.format(
                    n_streams=args.n_streams, batch=batch, reps=args.reps,
                    warmup=args.warmup),
                encoding='utf-8')
            rsync_to(host, sorted(work.glob('*')), remote_dir)
            tag = f'{name}_b{batch}'
            nvcc_build(host, remote_dir, env, args.arch,
                      sources=['model_cuda.cu', 'mdl_cuda.cu',
                              'bare_driver.cu'],
                      out=f'{tag}_bare')
            nvcc_build(host, remote_dir, env, args.arch,
                      sources=['model_cuda.cu', 'mdl_cuda.cu',
                              'stream_driver.cu'],
                      out=f'{tag}_stream')

            bare_sets = []
            stream_sets = []
            for _ in range(args.sets):
                out_bare = run_remote(host, remote_dir, f'./{tag}_bare')
                bare_sets.append(parse_kv_lines(out_bare)
                                 ['throughput_median_per_s'])
                out_stream = run_remote(host, remote_dir, f'./{tag}_stream')
                stream_sets.append(parse_kv_lines(out_stream)
                                   ['throughput_per_s'])
            bare_med = median(bare_sets)
            stream_med = median(stream_sets)
            speedup = stream_med / bare_med
            print(f'{name} batch={batch} n_streams={args.n_streams}: '
                 f'bare={bare_med:.1f}/s stream={stream_med:.1f}/s '
                 f'speedup={speedup:.3f}x')
            results[name][f'batch{batch}'] = {
                'n_streams': args.n_streams,
                'bare': {'sets': bare_sets, 'median': bare_med,
                        'n': len(bare_sets)},
                'stream': {'sets': stream_sets, 'median': stream_med,
                          'n': len(stream_sets)},
                'speedup_stream_over_bare': speedup,
            }

    if args.out:
        out_path = REPO / args.out
        payload = {
            'task': 'streams(ミラー) A/B',
            'host': f'{host} (RTX 4070 Ti, sm_89, CUDA 12.8)',
            'remote_dir': remote_dir,
            'reps': args.reps, 'warmup': args.warmup, 'sets': args.sets,
            'results': results,
        }
        write_merged(out_path, payload)
        print(f'\n結果: {out_path}')
    return 0


# ---------------------------------------------------------------------------
# フェーズ: bench-fuse(dense-fuse 48KB A/B + 整数一致)
# ---------------------------------------------------------------------------

def phase_bench_fuse(args) -> int:
    host = args.host
    remote_dir = args.remote_dir
    env = f'export PATH={args.nvcc_path}:$PATH'
    sh(host, f'mkdir -p {remote_dir}')
    results: dict = {}

    for name in args.models:
        bundle = load_model_bundle(name)
        # 融合対象があるか事前にローカルで判定(mnist_adderは対象外という
        # の実測訂正を反映 ─ 対象外モデルはA/Bをスキップし、その旨を
        # 記録するだけにする)。
        plan48 = build_plan(bundle, batch=1, dense_fuse=True,
                            dense_fuse_max_shared_bytes=
                            CUDA_DENSE_FUSE_MAX_SHARED_BYTES)
        n_fused = sum(1 for op in plan48.ops if op.kind == 'dense_fuse')
        if n_fused == 0:
            print(f'{name}: dense_fuseの対象鎖なし(48KBでも0件)。A/Bをskip')
            results[name] = {'n_fused_chains_at_48kb': 0, 'skipped': True}
            continue

        for batch in args.batches:
            work = Path(tempfile.mkdtemp(prefix=f'mdl_cuda_fuse_{name}_'))
            stage_runtime(work)
            bare_files = generate_cuda(bundle, name='model_cuda_bare',
                                       batch=batch, dense_fuse=False)
            fuse_files = generate_cuda(bundle, name='model_cuda_fuse',
                                       batch=batch, dense_fuse=True)
            for filename, text in {**bare_files, **fuse_files}.items():
                (work / filename).write_text(text, encoding='utf-8')
            (work / 'bare_driver.cu').write_text(
                _bare_driver_named('model_cuda_bare', batch, args.reps,
                                  args.warmup), encoding='utf-8')
            (work / 'fuse_driver.cu').write_text(
                _bare_driver_named('model_cuda_fuse', batch, args.reps,
                                  args.warmup), encoding='utf-8')
            rsync_to(host, sorted(work.glob('*')), remote_dir)
            tag = f'{name}_b{batch}'
            nvcc_build(host, remote_dir, env, args.arch,
                      sources=['model_cuda_bare.cu', 'mdl_cuda.cu',
                              'bare_driver.cu'],
                      out=f'{tag}_bare')
            nvcc_build(host, remote_dir, env, args.arch,
                      sources=['model_cuda_fuse.cu', 'mdl_cuda.cu',
                              'fuse_driver.cu'],
                      out=f'{tag}_fuse')
            bare_sets = []
            fuse_sets = []
            for _ in range(args.sets):
                out_bare = run_remote(host, remote_dir, f'./{tag}_bare')
                bare_sets.append(parse_kv_lines(out_bare)
                                 ['throughput_median_per_s'])
                out_fuse = run_remote(host, remote_dir, f'./{tag}_fuse')
                fuse_sets.append(parse_kv_lines(out_fuse)
                                 ['throughput_median_per_s'])
            bare_med = median(bare_sets)
            fuse_med = median(fuse_sets)
            speedup = fuse_med / bare_med
            print(f'{name} batch={batch}: bare={bare_med:.1f}/s '
                 f'fuse(48KB)={fuse_med:.1f}/s speedup={speedup:.3f}x')
            results.setdefault(name, {})[f'batch{batch}'] = {
                'n_fused_chains_at_48kb': n_fused,
                'bare': {'sets': bare_sets, 'median': bare_med},
                'fuse_48kb': {'sets': fuse_sets, 'median': fuse_med},
                'speedup_fuse_over_bare': speedup,
            }

    if args.out:
        out_path = REPO / args.out
        payload = {
            'task': 'dense-fuse 48KB(Metal 32KBの代わり) A/B',
            'host': f'{host} (RTX 4070 Ti, sm_89, CUDA 12.8)',
            'note_correction': (
                '当初「mnist_adder(層幅45.9KB)は32KB縛りだけで融合対象外'
                'になっている」という前提だったが、実測では対象の鎖'
                '(n0_pb_pc_ge→n1_final→group_sum)が2*act_bytes=91,800B'
                '(約89.6KB)を要求し、48KBでも収まらない(48KB化では'
                'mnist_adderは解禁されない)。48KB化の配線自体は合成モデル'
                '(2*act_bytes=40,000B、32KB<x<=48KB)で機能確認済み'
                '(tests/deploy/test_cuda_stream.py::'
                'test_dense_fuse_32kb_48kb_gap_is_backend_specific)。'),
            'reps': args.reps, 'warmup': args.warmup, 'sets': args.sets,
            'results': results,
        }
        write_merged(out_path, payload)
        print(f'\n結果: {out_path}')
    return 0


def _bare_driver_named(model_name: str, batch: int, reps: int,
                       warmup: int) -> str:
    """_BARE_DRIVER_TMPLはmdl_cuda_infer_batch等を直接呼ぶので、
    bare/fuseの2バイナリを同一のシンボル名で別々にリンクする(model_cuda_bare.cu
    と model_cuda_fuse.cu はどちらも `mdl_cuda_infer_batch` 等
    同じCシンボルを持つが、各driver.cuは常に対応するmodel_cuda_*.cuと
    1本ずつでリンクするので衝突しない)。"""
    return _BARE_DRIVER_TMPL.format(batch=batch, reps=reps, warmup=warmup)


# ---------------------------------------------------------------------------
# フェーズ: calib(target_threads/MDL_BLOCK掃引)
# ---------------------------------------------------------------------------

def phase_calib(args) -> int:
    host = args.host
    remote_dir = args.remote_dir
    env = f'export PATH={args.nvcc_path}:$PATH'
    sh(host, f'mkdir -p {remote_dir}')
    bundle = load_model_bundle(args.calib_model)
    results: dict = {}

    for batch in args.batches:
        work = Path(tempfile.mkdtemp(prefix='mdl_cuda_calib_'))
        for filename, text in generate_cuda(
                bundle, name='model_cuda', batch=batch).items():
            (work / filename).write_text(text, encoding='utf-8')
        (work / 'bare_driver.cu').write_text(
            _BARE_DRIVER_TMPL.format(batch=batch, reps=args.reps,
                                     warmup=args.warmup), encoding='utf-8')
        rsync_to(host, sorted(work.glob('*.cu')) + sorted(work.glob('*.h')),
                 remote_dir)

        param = args.param
        for value in args.values:
            mdl_cuda_text = (RUNTIME / 'cuda' / 'mdl_cuda.cu').read_text(
                encoding='utf-8')
            if param == 'target_threads':
                needle = '#define MDL_GS_TARGET_THREADS 1024'
                patched = mdl_cuda_text.replace(
                    needle, f'#define MDL_GS_TARGET_THREADS {value}')
            elif param == 'block':
                needle = '#define MDL_BLOCK 256'
                patched = mdl_cuda_text.replace(
                    needle, f'#define MDL_BLOCK {value}')
            else:
                raise SystemExit(f'未対応のparam: {param}')
            assert needle in mdl_cuda_text, f'置換対象が見つからない({param})'
            (work / 'mdl_cuda_patched.cu').write_text(patched,
                                                       encoding='utf-8')
            rsync_to(host, [work / 'mdl_cuda_patched.cu'], remote_dir)
            tag = f'calib_{param}_{value}_b{batch}'
            nvcc_build(host, remote_dir, env, args.arch,
                      sources=['model_cuda.cu', 'mdl_cuda_patched.cu',
                              'bare_driver.cu'],
                      out=tag)
            sets = []
            for _ in range(args.sets):
                out = run_remote(host, remote_dir, f'./{tag}')
                sets.append(parse_kv_lines(out)['throughput_median_per_s'])
            med = median(sets)
            print(f'{args.calib_model} batch={batch} {param}={value}: '
                 f'{med:.1f}/s (sets={sets})')
            results.setdefault(f'batch{batch}', {})[str(value)] = {
                'sets': sets, 'median': med}

    if args.out:
        out_path = REPO / args.out
        payload = {
            'task': f'{args.param}較正({args.calib_model})',
            'host': f'{host} (RTX 4070 Ti, sm_89, CUDA 12.8)',
            'param': args.param, 'values': args.values,
            'reps': args.reps, 'warmup': args.warmup, 'sets': args.sets,
            'results': results,
        }
        write_merged(out_path, payload)
        print(f'\n結果: {out_path}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', required=True,
                    choices=['verify', 'bench-stream', 'bench-fuse', 'calib'])
    ap.add_argument('--host', required=True)
    ap.add_argument('--remote-dir', default='mdl_cuda_k55_20260911')
    ap.add_argument('--arch', default='sm_89')
    ap.add_argument('--nvcc-path', default=DEFAULT_NVCC_PATH)
    ap.add_argument('--models', nargs='+',
                    default=['har_dlgn_k600_gumbel', 'har_dlgn_k6000_gumbel'])
    ap.add_argument('--n-streams', type=int, default=4)
    ap.add_argument('--default-batch', type=int, default=128)
    ap.add_argument('--batches', nargs='+', type=int, default=[128])
    ap.add_argument('--sets', type=int, default=5)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--out', default=None)
    ap.add_argument('--param', choices=['target_threads', 'block'],
                    default='target_threads')
    ap.add_argument('--values', nargs='+', type=int, default=[1024])
    ap.add_argument('--calib-model', default='har_dlgn_k6000_gumbel')
    args = ap.parse_args()

    if not shutil.which('rsync'):
        raise SystemExit('rsync が見つからない')

    phases = {
        'verify': phase_verify,
        'bench-stream': phase_bench_stream,
        'bench-fuse': phase_bench_fuse,
        'calib': phase_calib,
    }
    return phases[args.phase](args)


if __name__ == '__main__':
    sys.exit(main())
