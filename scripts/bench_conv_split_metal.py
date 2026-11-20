#!/usr/bin/env python
"""conv2dの「ocごとの命令列分割」(split) vs 旧経路(combined)の
Metal実効果測定(batch=1)。

成功指標はbatch=1レイテンシ。旧経路(mdl_k_conv_exec、並列度=(batch,ワード
位置))と新経路(mdl_k_conv_exec_split、並列度=(batch,oc,ワード位置))を
それぞれ生成・ビルドし、同じ入力で:

    (a) e2eレイテンシ(ウォームアップ8回+30回median、CPU側時計)
    (b) mdl_metal_gpu_ms() の実GPU時間(累積を推論回数で割った1回あたり平均)
    (c) mdl_metal_gpu_breakdown() のconv_exec(_split)行

を測る。3セット走らせて安定性を見る。

使い方:
    uv run python scripts/bench_conv_split_metal.py
    uv run python scripts/bench_conv_split_metal.py --models seg_convdlgn_k32
    uv run python scripts/bench_conv_split_metal.py --reps 50 --warmup 10 --sets 3
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
from griffith_logic.deploy.layout import (  # noqa: E402
    build_conv_layout_split,
    conv_layout_split_stats,
)
from griffith_logic.deploy.metalbackend import generate_metal  # noqa: E402
from griffith_logic.deploy.preproc_desc import (  # noqa: E402
    SUPPORTED_KINDS,
    build_preproc_descriptors,
)
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
RUNTIME_METAL = REPO / 'runtime' / 'metal'

DEFAULT_MODELS = [
    'seg_convdlgn_k128', 'seg_convdlgn_k32',
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

  /* ウォームアップ(シェーダのJIT・パイプライン生成コストを外す)。 */
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


def build_and_run(bundle: dict, work_dir: Path, *, conv_engine: str,
                  reps: int, warmup: int, clangxx: str) -> dict:
    preproc = bundle.get('preproc') or {}
    if preproc.get('kind') not in SUPPORTED_KINDS:
        raise SystemExit(f'画像入力以外のモデルは未対応(preproc={preproc})')
    plan = build_preproc_descriptors(preproc)
    n_img = plan['image_channels'] * plan['height'] * plan['width']

    work = work_dir / conv_engine
    work.mkdir(parents=True, exist_ok=True)
    for filename, text in generate_metal(
            bundle, name='model_metal', batch=1,
            conv_engine=conv_engine).items():
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
        raise SystemExit(f'ビルド失敗({conv_engine}):\n{build.stderr}')

    proc = subprocess.run([str(exe), str(reps), str(warmup)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f'実行失敗({conv_engine}, exit={proc.returncode}):\n{proc.stderr}')

    out = proc.stdout
    result: dict = {}
    for key in ('e2e_ms_median', 'e2e_ms_best', 'gpu_ms_total', 'gpu_ms_avg'):
        m = re.search(rf'{key} ([-\d.]+)', out)
        result[key] = float(m.group(1)) if m else None
    m = re.search(r'BREAKDOWN_BEGIN\n(.*?)BREAKDOWN_END', out, re.S)
    result['breakdown'] = m.group(1).strip() if m else ''
    return result


def conv_exec_line(breakdown: str) -> str:
    for line in breakdown.splitlines():
        if 'mdl_k_conv_exec' in line:
            return line.strip()
    return '(conv_exec行なし)'


def split_stats_for_bundle(bundle: dict) -> list[dict]:
    graph = ir_from_dict(bundle['ir'])
    out = []
    for node in graph.nodes:
        if node.type != 'conv2d':
            continue
        prog_dict = bundle['programs'][node.id]
        layout_split = build_conv_layout_split(node.params, prog_dict)
        stats = conv_layout_split_stats(prog_dict, layout_split)
        stats['node_id'] = node.id
        out.append(stats)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--sets', type=int, default=3)
    ap.add_argument('--work-dir', default=None)
    args = ap.parse_args()

    clangxx = find_clangxx()
    import tempfile
    work_root = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix='mdl_bench_split_'))

    for name in args.models:
        path = MODELS_DIR / f'{name}.json'
        if not path.is_file():
            print(f'=== {name}: モデルJSONが無い、skip ===')
            continue
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)

        print(f'\n=== {name} ===')
        for s in split_stats_for_bundle(bundle):
            print(f"  分割統計 {s['node_id']}: "
                  f"combined={s['n_instrs_combined']} "
                  f"split_total={s['n_instrs_split_total']} "
                  f"dup={s['duplication_pct']:.2f}% "
                  f"max_slots={s['max_slots_per_oc']} "
                  f"oc={s['out_channels']} "
                  f"zero_instr_oc={s['zero_instr_oc_count']}")

        for set_idx in range(args.sets):
            print(f'  -- セット{set_idx + 1}/{args.sets} --')
            for engine in ('combined', 'split'):
                r = build_and_run(bundle, work_root / name, conv_engine=engine,
                                  reps=args.reps, warmup=args.warmup,
                                  clangxx=clangxx)
                print(f'    {engine:9s}: e2e median={r["e2e_ms_median"]:.4f}ms '
                      f'best={r["e2e_ms_best"]:.4f}ms  '
                      f'gpu_avg={r["gpu_ms_avg"]:.4f}ms  '
                      f'[{conv_exec_line(r["breakdown"])}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
