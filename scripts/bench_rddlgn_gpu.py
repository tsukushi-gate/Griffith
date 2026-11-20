#!/usr/bin/env python
"""RDDLGN(教師強制)のCUDA/Metal実測。

`scripts/verify_cuda_remote.py`(K9)と同じ手法をRDDLGNへ適用する。
GPUバックエンド(`deploy/cudabackend.py`/`deploy/metalbackend.py`)は
recurrentノードを汎用の`mdl_*_infer_batch(bits, scores, n)`
契約でサポートしているが、(packed teacher forcing)(語彙折り畳み)/
(退化ゲート除去)(u16添字はCUDA/Metalにも適用されるが自動)のうち
C専用拡張(cbackend.py)でGPU側には実装されていない。
つまりGPUで測れるのは**教師強制の「ベース経路」**(自己回帰生成はC専用の
`mdl_rddlgn_generate_greedy`ドライバのみで、GPU版ドライバは存在しない
——本タスクのスコープ外)のみ。(添字u16化)は`in_width<=65536`なら
自動的に適用される(無効化のフラグは無い)。

やること(モデルごと):
    1. RDDLGNモデル読み込み → `export_rddlgn_ir`(teacher forcing、
       optimize_degenerate=False)→ バンドル
    2. CUDA: リモート(ssh)へrsync、nvccビルド、`--batch`で単系列+バッチ実測
       Metal: ローカルでclang++ビルド、実行
    3. `ir/simulate.py`と整数厳密一致を確認(batch=1、乱数ビット列)
    4. `mdl_*_infer_batch`をreps回叩き中央値/枚を記録(batch=1と
       --bench-batchesで指定した各batch)

使い方:
    # Metal(ローカル)
    uv run python scripts/bench_rddlgn_gpu.py --backend metal \
        --models copy_tiny,reverse_deep,charlm --bench

    # CUDA(リモート)
    uv run python scripts/bench_rddlgn_gpu.py --backend cuda \
        --host <gpu-host> --models copy_tiny,reverse_deep,charlm --bench
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from griffith_logic.deploy.bundle import build_bundle  # noqa: E402
from griffith_logic.export.rddlgn_ir import export_rddlgn_ir  # noqa: E402
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig  # noqa: E402

RUNTIME = REPO / 'runtime'
METAL_RUNTIME = RUNTIME / 'metal'


def load_ckpt_model(path: Path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = RDDLGNConfig(**ckpt['metadata']['config'])
    model = RDDLGN(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def load_paper_model(scale: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    cfg = RDDLGNConfig.from_paper(scale=scale)
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


MODEL_SPECS = {
    'copy_tiny': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_copy_tiny_20260815_074318/best.pt'),
    'reverse_deep': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_reverse_custom_20260815_081056/best.pt'),
    'charlm': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_charlm_scale005/'
               '2wsppkmu_alpha1.0_pureCE_best.pt'),
    'paper_scale1': lambda: load_paper_model(1.0),
}

# ---------------------------------------------------------------------------
# CUDAドライバ(scripts/verify_cuda_remote.pyのDRIVERとほぼ同一。画像経路なし)
# ---------------------------------------------------------------------------
CUDA_DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" {
int mdl_cuda_infer_batch(const unsigned char*, int*, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_max_batch(void);
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
/* 引数: [枚数] [反復]。反復>0 なら結果ではなく ms/枚 を stderr に出す。 */
int main(int argc, char** argv) {
  int nin = mdl_cuda_input_width(), nout = mdl_cuda_output_width();
  int n = (argc > 1) ? atoi(argv[1]) : 1;
  int reps = (argc > 2) ? atoi(argv[2]) : 0;
  int rc, r;
  if (n < 1 || n > mdl_cuda_max_batch()) {
    fprintf(stderr, "batch out of range\n"); return 4;
  }
  {
  unsigned char* buf = (unsigned char*)malloc((size_t)n * nin);
  int* scores = (int*)malloc((size_t)n * nout * sizeof(int));
  if (fread(buf, 1, (size_t)n * nin, stdin) != (size_t)n * nin) {
    fprintf(stderr, "short read\n"); return 1;
  }
  rc = mdl_cuda_infer_batch(buf, scores, n);
  if (rc != 0) { fprintf(stderr, "infer failed\n"); return 2; }
  if (reps > 0) {
    double best = 1e18, sum = 0.0;
    for (r = 0; r < reps; r++) {
      double t0 = now_ms();
      if (mdl_cuda_infer_batch(buf, scores, n) != 0) return 2;
      { double dt = now_ms() - t0; if (dt < best) best = dt; sum += dt; }
    }
    fprintf(stderr, "BENCH %d %.4f %.1f %.4f\n", n, best / n,
            n * 1000.0 / best, sum / reps);
    return 0;
  }
  fwrite(scores, sizeof(int), (size_t)n * nout, stdout);
  }
  return 0;
}
'''

METAL_DRIVER = r'''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C" {
int mdl_metal_infer_batch(const unsigned char*, int*, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_metal_max_batch(void);
int mdl_model_init(void);
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
int main(int argc, char** argv) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  int n = (argc > 1) ? atoi(argv[1]) : 1;
  int reps = (argc > 2) ? atoi(argv[2]) : 0;
  if (n < 1 || n > mdl_metal_max_batch()) {
    fprintf(stderr, "batch out of range\n"); return 4;
  }
  if (mdl_model_init() != 0) return 3;
  {
  unsigned char* buf = (unsigned char*)malloc((size_t)n * nin);
  int* scores = (int*)malloc((size_t)n * nout * sizeof(int));
  if (fread(buf, 1, (size_t)n * nin, stdin) != (size_t)n * nin) {
    fprintf(stderr, "short read\n"); return 1;
  }
  if (mdl_metal_infer_batch(buf, scores, n) != 0) { fprintf(stderr, "infer failed\n"); return 2; }
  if (reps > 0) {
    double best = 1e18, sum = 0.0;
    int r;
    for (r = 0; r < reps; r++) {
      double t0 = now_ms();
      if (mdl_metal_infer_batch(buf, scores, n) != 0) return 2;
      { double dt = now_ms() - t0; if (dt < best) best = dt; sum += dt; }
    }
    fprintf(stderr, "BENCH %d %.4f %.1f %.4f\n", n, best / n,
            n * 1000.0 / best, sum / reps);
    return 0;
  }
  fwrite(scores, sizeof(int), (size_t)n * nout, stdout);
  }
  return 0;
}
'''


def sh(host: str, cmd: str, *, check: bool = True) -> str:
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f'リモート実行に失敗:\n{cmd}\n{proc.stdout}\n{proc.stderr}')
    return proc.stdout


def build_case(name: str, seed: int, batches: list[int]) -> dict:
    model, cfg = MODEL_SPECS[name]()
    graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                             optimize_degenerate=False)
    doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
    bundle = build_bundle(doc)
    max_batch = max(batches)
    rng = np.random.default_rng(seed)
    in_width = int(graph.inputs[0].width)
    bits = rng.integers(0, 2, size=(max_batch, in_width), dtype=np.uint8)
    want = simulate(ir_from_dict(bundle['ir']), bits).astype(np.int64)
    return {'name': name, 'bundle': bundle, 'bits': bits, 'want': want,
           'max_batch': max_batch, 'in_width': in_width,
           'out_width': int(want[0].size),
           'gate_total': None}


def run_cuda(args, cases: list[dict]) -> dict:
    from griffith_logic.deploy.cudabackend import generate_cuda
    work = Path(tempfile.mkdtemp(prefix='mdl_cuda_rddlgn_'))
    print(f'作業ディレクトリ: {work}')
    for case in cases:
        for filename, text in generate_cuda(
                case['bundle'], batch=case['max_batch']).items():
            (work / f"{case['name']}_{filename}").write_text(
                text, encoding='utf-8')
        (work / f"{case['name']}_bits.bin").write_bytes(
            case['bits'].tobytes())
        np.save(work / f"{case['name']}_want.npy", case['want'])
    (work / 'driver.cu').write_text(CUDA_DRIVER, encoding='utf-8')
    for rel in ('cuda/mdl_cuda.h', 'cuda/mdl_cuda.cu', 'gpu/mdl_gpu_kernels.h'):
        (work / Path(rel).name).write_text(
            (RUNTIME / rel).read_text(encoding='utf-8'), encoding='utf-8')

    sh(args.host, f'mkdir -p {args.remote_dir}')
    subprocess.run(['rsync', '-az', '--delete',
                    *[str(p) for p in sorted(work.glob('*'))],
                    f'{args.host}:{args.remote_dir}/'], check=True)
    print('転送完了')
    env = f'export PATH={args.nvcc_path}:$PATH'
    sh(args.host, f'cd {args.remote_dir} && {env} && '
                  f'nvcc -O2 -arch={args.arch} -I. -c mdl_cuda.cu -o mdl_cuda.o')
    print('ランタイムのビルドOK')

    gpu_name = sh(args.host, 'nvidia-smi --query-gpu=name --format=csv,noheader').strip()
    results = {}
    for case in cases:
        name = case['name']
        sh(args.host, f'cd {args.remote_dir} && {env} && '
                      f'nvcc -O2 -arch={args.arch} -I. '
                      f'{name}_model_cuda.cu driver.cu mdl_cuda.o -o {name}_test')
        # 整合性(batch=1)
        sh(args.host, f'cd {args.remote_dir} && head -c {case["in_width"]} '
                      f'{name}_bits.bin | ./{name}_test 1 0 > {name}_out1.bin')
        subprocess.run(['rsync', '-az',
                        f'{args.host}:{args.remote_dir}/{name}_out1.bin',
                        str(work)], check=True)
        got1 = np.frombuffer((work / f'{name}_out1.bin').read_bytes(),
                             dtype='<i4').astype(np.int64)
        ok = np.array_equal(got1, case['want'][0])
        print(f'  {name}: batch=1 整合性 {"OK" if ok else "NG"}')

        batches = [int(b) for b in args.bench_batches.split(',')]
        perf = {}
        for nb in batches:
            if nb > case['max_batch']:
                continue
            out = sh(args.host, f'cd {args.remote_dir} && '
                                f'head -c {nb * case["in_width"]} {name}_bits.bin | '
                                f'./{name}_test {nb} {args.reps} 2>&1 >/dev/null | grep BENCH')
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] == 'BENCH':
                    perf[nb] = {'ms_per_item': float(parts[2]),
                               'items_per_s': float(parts[3]),
                               'ms_mean': float(parts[4])}
                    print(f'    batch={nb:4d}  ms/item(best)={float(parts[2]):.4f}  '
                          f'items/s={float(parts[3]):.1f}')
        results[name] = {'match_batch1': bool(ok), 'perf': perf}
    return {'gpu': gpu_name, 'results': results}


def run_metal(args, cases: list[dict]) -> dict:
    from griffith_logic.deploy.metalbackend import generate_metal
    work = Path(tempfile.mkdtemp(prefix='mdl_metal_rddlgn_'))
    print(f'作業ディレクトリ: {work}')
    results = {}
    for case in cases:
        name = case['name']
        cwork = work / name
        cwork.mkdir(parents=True)
        for filename, text in generate_metal(
                case['bundle'], batch=case['max_batch']).items():
            (cwork / filename).write_text(text, encoding='utf-8')
        (cwork / 'driver.mm').write_text(METAL_DRIVER, encoding='utf-8')
        build = subprocess.run(
            ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
             f'-I{METAL_RUNTIME}', '-framework', 'Metal',
             '-framework', 'Foundation',
             str(cwork / 'model_metal.mm'), str(METAL_RUNTIME / 'mdl_metal.mm'),
             str(cwork / 'driver.mm'), '-o', str(cwork / 'app')],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(f'  {name}: Metalビルド不可: {build.stderr[:400]}')
            results[name] = {'error': 'build_failed'}
            continue

        b1 = case['bits'][:1].tobytes()
        proc = subprocess.run([str(cwork / 'app'), '1', '0'], input=b1,
                              capture_output=True)
        if proc.returncode != 0:
            print(f'  {name}: Metal実行不可: '
                  f'{proc.stderr.decode(errors="replace")[:300]}')
            results[name] = {'error': 'run_failed'}
            continue
        got1 = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        ok = np.array_equal(got1, case['want'][0])
        print(f'  {name}: batch=1 整合性 {"OK" if ok else "NG"}')

        batches = [int(b) for b in args.bench_batches.split(',')]
        perf = {}
        for nb in batches:
            if nb > case['max_batch']:
                continue
            buf = case['bits'][:nb].tobytes()
            proc = subprocess.run(
                [str(cwork / 'app'), str(nb), str(args.reps)],
                input=buf, capture_output=True)
            for line in proc.stderr.decode(errors='replace').splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] == 'BENCH':
                    perf[nb] = {'ms_per_item': float(parts[2]),
                               'items_per_s': float(parts[3]),
                               'ms_mean': float(parts[4])}
                    print(f'    batch={nb:4d}  ms/item(best)={float(parts[2]):.4f}  '
                          f'items/s={float(parts[3]):.1f}')
        results[name] = {'match_batch1': bool(ok), 'perf': perf}
    return {'results': results}


# ---------------------------------------------------------------------------
# 非推奨の薄いラッパ。計測本体(build_case/run_cuda/run_metal、上の
# すべての関数)はscripts/bench_deploy.pyが`--backend metal|cuda`の
# 「教師強制の系列一括スループット」枝からそのままimportして呼ぶ
# 「計測エンジン」として残る。このmain()自体は旧CLI引数を
# `bench_deploy.py`の統一引数へ変換して呼ぶだけ(挙動互換: `--out`未指定
# なら従来通りファイルは書かない)。
# ---------------------------------------------------------------------------
def _load_bench_deploy():
    import importlib.util
    if 'bench_deploy' in sys.modules:
        return sys.modules['bench_deploy']
    spec = importlib.util.spec_from_file_location(
        'bench_deploy', Path(__file__).resolve().parent / 'bench_deploy.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['bench_deploy'] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['cuda', 'metal'], required=True)
    ap.add_argument('--models', default='copy_tiny,reverse_deep,charlm')
    ap.add_argument('--host', default=None,
                    help='CUDA: ssh先(user@hostまたは~/.ssh/configエイリアス)')
    ap.add_argument('--remote-dir', default='mdl_cuda_rddlgn_verify')
    ap.add_argument('--arch', default='sm_89')
    ap.add_argument('--nvcc-path',
                    default='$HOME/.local/micromamba/envs/llamabuild/bin')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--bench-batches', default='1,8,32')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print('[非推奨] scripts/bench_rddlgn_gpu.py は scripts/bench_deploy.py '
          '--backend metal|cuda --rddlgn-only(教師強制の系列一括スループット'
          'を測る枝)へ統一された。今後は '
          '`uv run python scripts/bench_deploy.py --backend metal '
          '--rddlgn-only ...` を使うこと。このラッパは統一実装をそのまま'
          '呼び出すだけ。', file=sys.stderr)

    if args.backend == 'cuda' and not args.host:
        print('--backend cuda には --host が必要', file=sys.stderr)
        return 1

    deploy = _load_bench_deploy()
    forward = ['--backend', args.backend, '--rddlgn-only',
              '--rddlgn-gpu-metrics', 'tf',
              '--rddlgn-models', args.models,
              '--rddlgn-seed', str(args.seed),
              '--rddlgn-reps', str(args.reps),
              '--rddlgn-tf-batches', args.bench_batches,
              '--rddlgn-tf-remote-dir', args.remote_dir,
              '--arch', args.arch, '--nvcc-path', args.nvcc_path]
    if args.host:
        forward += ['--host', args.host]
    if args.out:
        forward += ['--rddlgn-out', args.out]
    return deploy.main(forward)


if __name__ == '__main__':
    sys.exit(main())
