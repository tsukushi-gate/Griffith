#!/usr/bin/env python
"""DLGN大規模化のスケーリング検証(恒久ハーネス)。

百万ゲート級のDLGNをデプロイキット(バンドル→プラン→Metal生成→
clang++コンパイル→実行)へ通したとき、どの段がどう伸びるか(線形/超線形/
その場で失敗)を合成denseモデルで測る。実モデル(det_multimnist_lut_k256、
1.6Mゲート)でのe2e計測はこのスクリプトの外(タスクの一次報告)で
別途行っている——ここは「幅」「深さ」を独立に振れる合成モデルで、
実モデルでは混ざってしまう要因(conv層のoc分割制約など)を切り分ける。

やること(各構成につき):
    1. `export_webbench.build_dense_ir`でdense層(lut_layer直列)+group_sum
       のIRを合成する(conv2dは含まない、幅/深さだけを振れる最小系)
    2. `build_bundle`→`build_plan`→`generate_metal`の各段を計測
    3. u16添字が使えるか(`plan.py::lut_dense_indices_fit_u16`)を記録
    4. clang++でコンパイル(タイムアウトは`--compile-timeout`、既定280秒。
       超えたら打ち切って「compile timeout」として記録する——それ自体が
       壁の実測)
    5. Metalで実行1回(整数一致はここでは検証しない。実モデル側の
       verify-metal流儀の検証で別途行っている)

幅掃引 (--suite width): depth=4固定、width in {1000, 16000, 64000, 131072}
    (131072 は u16添字の上限65536を超える境界)。batch=32で1回実行。
深さ掃引 (--suite depth): width=1000固定、depth in {16, 64, 128, 256}。
    dispatch数(層あたり1〜3)が線形に効くはずなので、batch=1のe2eを
    3回測って中央値を取る(dense_fuse/dense_packedは両方OFFの素経路)。

使い方:
    uv run python scripts/bench_scaling.py --suite width
    uv run python scripts/bench_scaling.py --suite depth
    uv run python scripts/bench_scaling.py --suite all \
        --out results/deploy/scaling_k59.json
"""

import argparse
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy.bundle import build_bundle  # noqa: E402
from griffith_logic.deploy.metalbackend import generate_metal  # noqa: E402
from griffith_logic.deploy.metalverify import find_clangxx, has_metal  # noqa: E402
from griffith_logic.deploy.plan import (  # noqa: E402
    build_plan,
    lut_dense_indices_fit_u16,
)
from griffith_logic.ir.serialize import ir_to_dict  # noqa: E402

RUNTIME_METAL = REPO / 'runtime' / 'metal'

WIDTH_SWEEP = [1_000, 16_000, 64_000, 131_072]
DEPTH_SWEEP = [16, 64, 128, 256]
DENSE_DEPTH_FOR_WIDTH_SWEEP = 4
DENSE_WIDTH_FOR_DEPTH_SWEEP = 1_000

_DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C" {{
int mdl_metal_infer_batch(const unsigned char*, int*, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}}
static double now_ms(void) {{
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}}
int main(void) {{
  static const int n = {batch};
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* buf = (unsigned char*)malloc((size_t)nin * n);
  int* scores = (int*)malloc((size_t)nout * n * sizeof(int));
  if (fread(buf, 1, (size_t)nin * n, stdin) != (size_t)nin * n) return 1;
  double t0 = now_ms();
  int rc_init = mdl_model_init();
  double t1 = now_ms();
  fprintf(stderr, "INIT_MS %f\\n", t1 - t0);
  if (rc_init != 0) return 3;
  double t2 = now_ms();
  int rc_infer = mdl_metal_infer_batch(buf, scores, n);
  double t3 = now_ms();
  fprintf(stderr, "INFER_MS %f\\n", t3 - t2);
  if (rc_infer != 0) return 2;
  fwrite(scores, sizeof(int), (size_t)nout * n, stdout);
  return 0;
}}
'''


def _load_export_webbench():
    """scripts/export_webbench.py はパッケージ外なのでファイルパスから
    直接import する(tests/deploy/conftest.py::_load_export_webbench と
    同じ方式)。"""
    if 'export_webbench' in sys.modules:
        return sys.modules['export_webbench']
    spec = importlib.util.spec_from_file_location(
        'export_webbench', REPO / 'scripts' / 'export_webbench.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['export_webbench'] = module
    spec.loader.exec_module(module)
    return module


def build_dense_doc(webbench, *, seed: int, in_width: int,
                    widths: list[int], k: int = 4,
                    name: str = 'scaling') -> dict:
    """幅/深さだけを振れる最小のdense IR doc(conv2dなし)。

    `preproc={'kind': 'thermometer', 'input_kind': 'bits'}` はビット列を
    そのまま入力として扱う経路(tests/deploy/conftest.py::dense_doc と
    同じ)。webbenchのbuild_model_docは通さない(resources/json_bytesの
    二重シリアライズが幅131072級だと無視できないコストになるため、
    このハーネスでは不要)。
    """
    graph = webbench.build_dense_ir(seed=seed, in_width=in_width,
                                    widths=widths, k=k, name=name)
    return {'ir': ir_to_dict(graph),
           'preproc': {'kind': 'thermometer', 'input_kind': 'bits'},
           'labels': [f'c{i}' for i in range(k)]}


def timed(fn):
    t0 = time.time()
    val = fn()
    return val, time.time() - t0


def compile_metal(files: dict[str, str], work_dir: Path, *, batch: int,
                  compile_timeout: float) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    for fn, text in files.items():
        (work_dir / fn).write_text(text, encoding='utf-8')
    for rel in ('mdl_metal.h', 'mdl_metal.mm'):
        shutil.copy2(RUNTIME_METAL / rel, work_dir / rel)
    (work_dir / 'driver.mm').write_text(
        _DRIVER_TMPL.format(batch=batch), encoding='utf-8')
    exe = work_dir / 'app'
    clangxx = find_clangxx()
    t0 = time.time()
    try:
        proc = subprocess.run(
            [clangxx, '-std=c++17', '-O2', '-fobjc-arc', f'-I{work_dir}',
             '-framework', 'Metal', '-framework', 'Foundation',
             str(work_dir / 'model_metal.mm'), str(work_dir / 'mdl_metal.mm'),
             str(work_dir / 'driver.mm'), '-o', str(exe)],
            capture_output=True, text=True, timeout=compile_timeout)
        dt = time.time() - t0
        ok = proc.returncode == 0
        err = None if ok else proc.stderr[-4000:]
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        ok, err = False, f'compile timeout after {compile_timeout:.0f}s'
    size = exe.stat().st_size if exe.exists() else None
    return {'ok': ok, 'seconds': dt, 'error': err, 'binary_bytes': size,
           'exe': str(exe) if exe.exists() else None}


def run_metal(exe: Path, *, in_width: int, out_width: int, batch: int,
             reps: int = 1) -> dict:
    import numpy as np
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=in_width * batch, dtype=np.uint8)
    init_ms_list, infer_ms_list = [], []
    ok = True
    err = None
    for _ in range(reps):
        proc = subprocess.run([str(exe)], input=bits.tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            ok = False
            err = proc.stderr.decode(errors='replace')
            break
        stderr = proc.stderr.decode(errors='replace')
        for line in stderr.splitlines():
            if line.startswith('INIT_MS'):
                init_ms_list.append(float(line.split()[1]))
            elif line.startswith('INFER_MS'):
                infer_ms_list.append(float(line.split()[1]))
        out = np.frombuffer(proc.stdout, dtype='<i4')
        if out.size != out_width * batch:
            ok = False
            err = f'output size {out.size} != expected {out_width * batch}'
            break
    return {'ok': ok, 'error': err, 'init_ms': init_ms_list,
           'infer_ms': infer_ms_list,
           'init_ms_median': statistics.median(init_ms_list) if init_ms_list else None,
           'infer_ms_median': statistics.median(infer_ms_list) if infer_ms_list else None}


def run_one_config(webbench, *, tag: str, in_width: int, widths: list[int],
                   batch: int, e2e_reps: int, compile_timeout: float,
                   work_root: Path) -> dict:
    print(f'=== {tag}: in_width={in_width} widths={widths} batch={batch} ===',
         flush=True)
    cfg = {'tag': tag, 'in_width': in_width, 'widths': widths,
          'depth': len(widths), 'batch': batch}
    layer_u16 = [lut_dense_indices_fit_u16(w) for w in
                ([in_width] + widths[:-1])]
    cfg['layer_u16'] = layer_u16
    cfg['all_u16'] = all(layer_u16)

    doc, t_doc = timed(lambda: build_dense_doc(
        webbench, seed=0, in_width=in_width, widths=widths, name=tag))
    cfg['build_ir_seconds'] = t_doc

    bundle, t_bundle = timed(lambda: build_bundle(doc, verify=True))
    cfg['build_bundle_seconds'] = t_bundle

    plan, t_plan = timed(lambda: build_plan(bundle, batch=batch))
    cfg['build_plan_seconds'] = t_plan
    cfg['n_ops'] = len(plan.ops)
    cfg['n_values'] = len(plan.values)

    try:
        files, t_gen = timed(lambda: generate_metal(
            bundle, name='model_metal', batch=batch))
        cfg['metal_generate_seconds'] = t_gen
        cfg['metal_generate_ok'] = True
        cfg['metal_src_bytes'] = sum(len(v) for v in files.values())
    except Exception as e:  # noqa: BLE001 -- record the wall, don't hide it
        cfg['metal_generate_ok'] = False
        cfg['metal_generate_error'] = repr(e)
        return cfg

    work = work_root / tag
    comp = compile_metal(files, work, batch=batch,
                         compile_timeout=compile_timeout)
    cfg['compile'] = comp
    if not comp['ok']:
        return cfg

    in_w = int(plan.input_width)
    out_w = int(plan.output_width)
    run = run_metal(Path(comp['exe']), in_width=in_w, out_width=out_w,
                    batch=batch, reps=e2e_reps)
    cfg['run'] = run
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', choices=['width', 'depth', 'all'], default='all')
    ap.add_argument('--out', default=str(REPO / 'results/deploy/scaling_k59.json'))
    ap.add_argument('--work-dir', default=None,
                    help='生成/コンパイル物(.mm/.cu/実行ファイル、数十MB級)の'
                         '置き場。既定はシステムのtempディレクトリ配下'
                         '(mdl_bench_scaling_build)——resultsツリーの下に'
                         '生成物を置きたい場合だけ明示的に指定すること')
    ap.add_argument('--compile-timeout', type=float, default=280.0)
    ap.add_argument('--widths', default=None,
                    help='カンマ区切りで幅掃引の値を上書き(既定: '
                         '1000,16000,64000,131072)')
    ap.add_argument('--depths', default=None,
                    help='カンマ区切りで深さ掃引の値を上書き(既定: '
                         '16,64,128,256)')
    args = ap.parse_args()

    if not has_metal():
        print('Metal(clang++ + runtime/metal)が無い環境。'
              'ソース生成段までは測れるが実行検証はできない。')

    webbench = _load_export_webbench()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work_root = Path(args.work_dir)
    else:
        import tempfile
        work_root = Path(tempfile.gettempdir()) / 'mdl_bench_scaling_build'

    results = json.loads(out_path.read_text()) if out_path.exists() else {
        'width_sweep': [], 'depth_sweep': []}

    widths = ([int(x) for x in args.widths.split(',')] if args.widths
             else WIDTH_SWEEP)
    depths = ([int(x) for x in args.depths.split(',')] if args.depths
             else DEPTH_SWEEP)

    if args.suite in ('width', 'all'):
        for w in widths:
            tag = f'width_{w}'
            if any(c.get('tag') == tag for c in results['width_sweep']):
                print(f'[skip] {tag} already in {out_path}')
                continue
            cfg = run_one_config(
                webbench, tag=tag, in_width=w,
                widths=[w] * DENSE_DEPTH_FOR_WIDTH_SWEEP, batch=32,
                e2e_reps=1, compile_timeout=args.compile_timeout,
                work_root=work_root)
            results['width_sweep'].append(cfg)
            out_path.write_text(json.dumps(results, indent=2, default=str))

    if args.suite in ('depth', 'all'):
        for d in depths:
            tag = f'depth_{d}'
            if any(c.get('tag') == tag for c in results['depth_sweep']):
                print(f'[skip] {tag} already in {out_path}')
                continue
            cfg = run_one_config(
                webbench, tag=tag, in_width=DENSE_WIDTH_FOR_DEPTH_SWEEP,
                widths=[DENSE_WIDTH_FOR_DEPTH_SWEEP] * d, batch=1,
                e2e_reps=3, compile_timeout=args.compile_timeout,
                work_root=work_root)
            results['depth_sweep'].append(cfg)
            out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
