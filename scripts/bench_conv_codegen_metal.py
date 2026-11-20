#!/usr/bin/env python
"""conv2dのモデル特化MSL生成(--conv-codegen) vs 現行インタプリタの
Metal実効果測定(batch=1)。`scripts/bench_conv_split_metal.py`と
同じ構造・同じ計測手口(mdl_metal_gpu_ms/mdl_metal_gpu_breakdown)を使う。

conv_engine は両方とも'split'固定(codegenはsplitレイアウト前提)。
違いは conv_codegen=False(インタプリタ、既定)/True(コード生成)だけ。

使い方:
    uv run python scripts/bench_conv_codegen_metal.py
    uv run python scripts/bench_conv_codegen_metal.py --models seg_convdlgn_k32
    uv run python scripts/bench_conv_codegen_metal.py --reps 50 --sets 5
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.metalbackend import generate_metal  # noqa: E402
from griffith_logic.deploy.preproc_desc import (  # noqa: E402
    SUPPORTED_KINDS,
    build_preproc_descriptors,
)

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
RUNTIME_METAL = REPO / 'runtime' / 'metal'

DEFAULT_MODELS = [
    'seg_convdlgn_k32', 'seg_convdlgn_k128',
    'det_face_lut_k256', 'det_multimnist_mono_lut_k640',
]

DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {{
int mdl_metal_infer_image(const unsigned char*, int*);
int mdl_metal_output_width(void);
int mdl_model_init(void);
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
  const int reps = argc > 1 ? atoi(argv[1]) : 30;
  const int warmup = argc > 2 ? atoi(argv[2]) : 8;
  static const int n_img = {n_img};
  int nout = 0;
  unsigned char *image = (unsigned char *)malloc((size_t)n_img);
  int *scores = NULL;
  double *t = (double *)malloc((size_t)reps * sizeof(double));
  size_t k;
  int r;

  for (k = 0; k < (size_t)n_img; k++) {{
    image[k] = (unsigned char)((k * 2654435761u) >> 24);
  }}
  if (mdl_model_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}
  nout = mdl_metal_output_width();
  scores = (int *)malloc((size_t)nout * sizeof(int));

  mdl_metal_set_gpu_timing(1);

  for (r = 0; r < warmup; r++) {{
    if (mdl_metal_infer_image(image, scores) != 0) {{
      fprintf(stderr, "warmup infer failed\\n");
      return 2;
    }}
  }}

  mdl_metal_reset_gpu_stats();
  for (r = 0; r < reps; r++) {{
    const double t0 = now_ms();
    if (mdl_metal_infer_image(image, scores) != 0) {{
      fprintf(stderr, "infer failed\\n");
      return 2;
    }}
    t[r] = now_ms() - t0;
  }}
  qsort(t, (size_t)reps, sizeof(double), cmp_d);

  printf("e2e_ms_median %.5f\\n", t[reps / 2]);
  printf("e2e_ms_best %.5f\\n", t[0]);
  printf("gpu_ms_total %.5f\\n", mdl_metal_gpu_ms());
  printf("gpu_ms_avg %.5f\\n", mdl_metal_gpu_ms() / (double)reps);
  printf("reps %d\\n", reps);
  printf("BREAKDOWN_BEGIN\\n");
  printf("%s", mdl_metal_gpu_breakdown());
  printf("BREAKDOWN_END\\n");
  return 0;
}}
'''


def find_clangxx() -> str:
    path = shutil.which('clang++')
    if path is None:
        raise SystemExit('clang++ が見つからない(macOS + Xcode CLTが必要)')
    return path


def build_and_run(bundle: dict, work_dir: Path, *, conv_codegen: bool,
                  reps: int, warmup: int, clangxx: str) -> dict:
    preproc = bundle.get('preproc') or {}
    if preproc.get('kind') not in SUPPORTED_KINDS:
        raise SystemExit(f'画像入力以外のモデルは未対応(preproc={preproc})')
    plan = build_preproc_descriptors(preproc)
    n_img = plan['image_channels'] * plan['height'] * plan['width']

    tag = 'codegen' if conv_codegen else 'interp'
    work = work_dir / tag
    work.mkdir(parents=True, exist_ok=True)
    for filename, text in generate_metal(
            bundle, name='model_metal', batch=1,
            conv_codegen=conv_codegen).items():
        (work / filename).write_text(text, encoding='utf-8')
    (work / 'driver.mm').write_text(DRIVER_TMPL.format(n_img=n_img),
                                    encoding='utf-8')

    exe = work / 'app'
    build = subprocess.run(
        [clangxx, '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{RUNTIME_METAL}', '-framework', 'Metal', '-framework', 'Foundation',
         str(work / 'model_metal.mm'), str(RUNTIME_METAL / 'mdl_metal.mm'),
         str(work / 'driver.mm'), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit(f'ビルド失敗({tag}):\n{build.stderr}')

    proc = subprocess.run([str(exe), str(reps), str(warmup)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f'実行失敗({tag}, exit={proc.returncode}):\n{proc.stderr}')

    out = proc.stdout
    result: dict = {}
    for key in ('e2e_ms_median', 'e2e_ms_best', 'gpu_ms_total', 'gpu_ms_avg'):
        m = re.search(rf'{key} ([-\d.]+)', out)
        result[key] = float(m.group(1)) if m else None
    m = re.search(r'BREAKDOWN_BEGIN\n(.*?)BREAKDOWN_END', out, re.S)
    result['breakdown'] = m.group(1).strip() if m else ''
    return result


def conv_kernel_ms(breakdown: str, *, exec_only: bool) -> float:
    """内訳から conv 関連カーネルの合計msを足し合わせる。

    Args:
        exec_only: Trueなら命令実行カーネルだけ
            (interp=mdl_k_conv_exec_split、codegen=mdl_k_conv_codegen_*)。
            Falseなら mdl_k_conv_planes/mdl_k_conv_const も含めた
            「conv系全部」(タップ平面構築はinterp/codegenで共通なので
            ここを混ぜると差分が薄まる — 判定はexec_only=Trueの方を使う)。
    """
    total = 0.0
    for line in breakdown.splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split()[0]
        is_exec = name.startswith('mdl_k_conv_exec') or \
            name.startswith('mdl_k_conv_codegen_')
        is_conv_family = name.startswith('mdl_k_conv_')
        if (exec_only and is_exec) or (not exec_only and is_conv_family):
            m = re.search(r'([\d.]+)\s*ms', line)
            if m:
                total += float(m.group(1))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--sets', type=int, default=5)
    ap.add_argument('--work-dir', default=None)
    args = ap.parse_args()

    clangxx = find_clangxx()
    import tempfile
    work_root = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix='mdl_bench_codegen_'))

    for name in args.models:
        path = MODELS_DIR / f'{name}.json'
        if not path.is_file():
            print(f'=== {name}: モデルJSONが無い、skip ===', flush=True)
            continue
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)

        print(f'\n=== {name} ===', flush=True)
        rows = {'interp': [], 'codegen': []}
        for set_idx in range(args.sets):
            print(f'  -- セット{set_idx + 1}/{args.sets} --', flush=True)
            for conv_codegen in (False, True):
                tag = 'codegen' if conv_codegen else 'interp'
                r = build_and_run(bundle, work_root / name,
                                  conv_codegen=conv_codegen,
                                  reps=args.reps, warmup=args.warmup,
                                  clangxx=clangxx)
                r['conv_ms'] = conv_kernel_ms(r['breakdown'], exec_only=False)
                r['exec_ms'] = conv_kernel_ms(r['breakdown'], exec_only=True)
                rows[tag].append(r)
                print(f'    {tag:8s}: e2e median={r["e2e_ms_median"]:.4f}ms '
                      f'gpu_avg={r["gpu_ms_avg"]:.4f}ms '
                      f'conv(全)={r["conv_ms"]:.4f}ms '
                      f'exec(命令実行のみ)={r["exec_ms"]:.4f}ms', flush=True)

        def med(key, tag, rows=rows):
            vals = sorted(row[key] for row in rows[tag])
            return vals[len(vals) // 2]

        e2e_i, e2e_c = med('e2e_ms_median', 'interp'), med('e2e_ms_median', 'codegen')
        gpu_i, gpu_c = med('gpu_ms_avg', 'interp'), med('gpu_ms_avg', 'codegen')
        conv_i, conv_c = med('conv_ms', 'interp'), med('conv_ms', 'codegen')
        exec_i, exec_c = med('exec_ms', 'interp'), med('exec_ms', 'codegen')
        print(f'  == {name} 中央値(sets={args.sets}) ==', flush=True)
        print(f'    e2e:            interp={e2e_i:.4f}ms codegen={e2e_c:.4f}ms '
              f'speedup={e2e_i / e2e_c:.3f}x', flush=True)
        print(f'    gpu合計:        interp={gpu_i:.4f}ms codegen={gpu_c:.4f}ms '
              f'speedup={gpu_i / gpu_c:.3f}x', flush=True)
        print(f'    conv系全体:     interp={conv_i:.4f}ms codegen={conv_c:.4f}ms '
              f'speedup={conv_i / conv_c:.3f}x '
              f'(planes/const込み、interp/codegen共通部分で薄まる)', flush=True)
        print(f'    exec(命令実行): interp={exec_i:.4f}ms codegen={exec_c:.4f}ms '
              f'speedup={exec_i / exec_c:.3f}x '
              f'(★判定はこちら — tt即値+スロット展開の純効果)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
