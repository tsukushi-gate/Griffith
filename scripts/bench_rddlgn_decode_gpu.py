#!/usr/bin/env python
"""RDDLGN自己回帰生成(decode)のGPU(CUDA/Metal)実測。

`scripts/bench_rddlgn_gpu.py`(教師強制)と同じ手法を、`deploy/
rddlgn_decode_gpu.py`が生成するdecodeドライバ(`mdl_{metal,cuda}_
rddlgn_decode`)へ適用する。

やること(モデルごと):
    1. RDDLGNモデル読み込み → `export_rddlgn_ir` → バンドル +
       `extract_embed_table`
    2. レーン数(=batch容量)を掃引しつつ生成・ビルド
       Metal: ローカルでclang++ビルド、実行
       CUDA: リモート(ssh)へrsync、nvccビルド、実行
    3. `export/rddlgn_ir.py::simulate_rddlgn_greedy_bits`(CPU golden)と
       整数厳密一致を確認(全レーン、goldenの長さぶんのプレフィックス)
    4. decodeドライバをreps回叩き、中央値/枚(tok/s)を記録
       (1呼び出し = 1レーンぶんの`steps`トークン生成、なので
       tok/s = レーン数 * steps * 1000 / ms_per_call)

使い方:
    # Metal(ローカル)
    uv run python scripts/bench_rddlgn_decode_gpu.py --backend metal \
        --models copy_tiny --lanes 1,32,256,1024

    # CUDA(リモート)
    uv run python scripts/bench_rddlgn_decode_gpu.py --backend cuda \
        --host <user>@<gpu-host> --models copy_tiny --lanes 1,32,256
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
from griffith_logic.export.rddlgn_ir import (  # noqa: E402
    export_rddlgn_ir,
    extract_embed_table,
    simulate_rddlgn_greedy_bits,
)
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


def load_tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = RDDLGNConfig.tiny()
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


def load_paper_model(scale: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    cfg = RDDLGNConfig.from_paper(scale=scale)
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


MODEL_SPECS = {
    'tiny': lambda: load_tiny_model(),
    'copy_tiny': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_copy_tiny_20260815_074318/best.pt'),
    'reverse_deep': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_reverse_custom_20260815_081056/best.pt'),
    'charlm': lambda: load_ckpt_model(
        REPO / 'checkpoints/rddlgn_charlm_scale005/'
               '2wsppkmu_alpha1.0_pureCE_best.pt'),
    'paper_scale005': lambda: load_paper_model(0.005),
    'paper_scale02': lambda: load_paper_model(0.02),
}

# ---------------------------------------------------------------------------
CUDA_DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" {
int mdl_cuda_rddlgn_decode(const unsigned char*, int, int*, int);
int mdl_cuda_rddlgn_decode_steps(void);
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
/* 引数: [レーン数n] [src_width] [bos] [batch容量] [反復]。
   反復>0ならms/呼び出し・レーン*トークン/秒をstderrへBENCH行で出す。 */
int main(int argc, char** argv) {
  int steps = mdl_cuda_rddlgn_decode_steps();
  int n = atoi(argv[1]);
  int src_width = atoi(argv[2]);
  int bos = atoi(argv[3]);
  int batch_cap = atoi(argv[4]);
  int reps = (argc > 5) ? atoi(argv[5]) : 0;
  unsigned char* src = (unsigned char*)malloc((size_t)n * src_width);
  int* out = (int*)malloc((size_t)steps * batch_cap * sizeof(int));
  if (fread(src, 1, (size_t)n * src_width, stdin) != (size_t)n * src_width) {
    fprintf(stderr, "short read\n"); return 1;
  }
  if (mdl_cuda_rddlgn_decode(src, bos, out, n) != 0) {
    fprintf(stderr, "decode failed\n"); return 2;
  }
  if (reps > 0) {
    double best = 1e18, sum = 0.0;
    int r;
    for (r = 0; r < reps; r++) {
      double t0 = now_ms();
      if (mdl_cuda_rddlgn_decode(src, bos, out, n) != 0) return 2;
      { double dt = now_ms() - t0; if (dt < best) best = dt; sum += dt; }
    }
    fprintf(stderr, "BENCH %d %.4f %.1f %.4f\n", n, best,
            (double)n * steps * 1000.0 / best, sum / reps);
    return 0;
  }
  fwrite(out, sizeof(int), (size_t)steps * batch_cap, stdout);
  return 0;
}
'''

METAL_DRIVER = r'''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C" {
int mdl_metal_rddlgn_decode(const unsigned char*, int, int*, int);
int mdl_metal_rddlgn_decode_steps(void);
int mdl_model_init(void);
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
int main(int argc, char** argv) {
  int steps = mdl_metal_rddlgn_decode_steps();
  int n = atoi(argv[1]);
  int src_width = atoi(argv[2]);
  int bos = atoi(argv[3]);
  int batch_cap = atoi(argv[4]);
  int reps = (argc > 5) ? atoi(argv[5]) : 0;
  unsigned char* src = (unsigned char*)malloc((size_t)n * src_width);
  int* out = (int*)malloc((size_t)steps * batch_cap * sizeof(int));
  if (mdl_model_init() != 0) return 3;
  if (fread(src, 1, (size_t)n * src_width, stdin) != (size_t)n * src_width) {
    fprintf(stderr, "short read\n"); return 1;
  }
  if (mdl_metal_rddlgn_decode(src, bos, out, n) != 0) {
    fprintf(stderr, "decode failed\n"); return 2;
  }
  if (reps > 0) {
    double best = 1e18, sum = 0.0;
    int r;
    for (r = 0; r < reps; r++) {
      double t0 = now_ms();
      if (mdl_metal_rddlgn_decode(src, bos, out, n) != 0) return 2;
      { double dt = now_ms() - t0; if (dt < best) best = dt; sum += dt; }
    }
    fprintf(stderr, "BENCH %d %.4f %.1f %.4f\n", n, best,
            (double)n * steps * 1000.0 / best, sum / reps);
    return 0;
  }
  fwrite(out, sizeof(int), (size_t)steps * batch_cap, stdout);
  return 0;
}
'''


def sh(host: str, cmd: str, *, check: bool = True) -> str:
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f'リモート実行に失敗:\n{cmd}\n{proc.stdout}\n{proc.stderr}')
    return proc.stdout


def build_case(name: str, seed: int) -> dict:
    model, cfg = MODEL_SPECS[name]()
    graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                             optimize_degenerate=False)
    from griffith_logic.ir.serialize import ir_to_dict
    doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    bundle = build_bundle(doc)

    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()

    # goldenは1レーンぶんだけ計算すれば十分(perfループは全レーン同一srcを
    # 複製して使う)。
    rng = np.random.default_rng(seed)
    src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,))
    bos_token = int(rng.integers(0, cfg.vocab_size))
    src_bits = embed[src_tokens].reshape(-1).astype(np.uint8)
    golden = simulate_rddlgn_greedy_bits(
        model, src_bits, bos_token=bos_token, eos_token=999999,
        max_new_tokens=cfg.seq_len)
    return {'name': name, 'bundle': bundle, 'src_bits': src_bits,
           'bos_token': bos_token, 'golden': golden, 'steps': cfg.seq_len,
           'src_width': int(src_bits.shape[0])}


def run_cuda(args, cases: list[dict]) -> dict:
    from griffith_logic.deploy.rddlgn_decode_gpu import generate_rddlgn_decode_cuda
    work = Path(tempfile.mkdtemp(prefix='mdl_cuda_rddlgn_decode_'))
    print(f'作業ディレクトリ: {work}')
    lanes = [int(x) for x in args.lanes.split(',')]
    max_lane = max(lanes)
    results = {}
    markers = {}
    for case in cases:
        name = case['name']
        for filename, text in generate_rddlgn_decode_cuda(
                case['bundle'], batch=max_lane,
                megakernel=args.megakernel).items():
            (work / f'{name}_{filename}').write_text(text, encoding='utf-8')
            if args.megakernel and filename.endswith('.cu'):
                markers[name] = ('megakernel: enabled' in text)
                print(f'  {name}: megakernel='
                     f'{"enabled" if markers[name] else "fallback"}')
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

    gpu_name = sh(args.host,
                 'nvidia-smi --query-gpu=name --format=csv,noheader').strip()
    for case in cases:
        name = case['name']
        sh(args.host, f'cd {args.remote_dir} && {env} && '
                      f'nvcc -O2 -arch={args.arch} -I. '
                      f'{name}_model_cuda.cu driver.cu mdl_cuda.o -o {name}_test')
        # max_laneぶん複製したファイルを1本作り、全レーン数の実行で使い回す
        # (freadはn*src_widthバイトしか読まないので、末尾の余りは無害)。
        in_path = work / f'{name}_in.bin'
        in_path.write_bytes(case['src_bits'].tobytes() * max_lane)
        subprocess.run(['rsync', '-az', str(in_path),
                        f'{args.host}:{args.remote_dir}/'], check=True)

        # 整合性(lane=1)
        sh(args.host, f'cd {args.remote_dir} && ./{name}_test 1 '
                      f'{case["src_width"]} {case["bos_token"]} {max_lane} 0 '
                      f'< {name}_in.bin > {name}_out1.bin')
        subprocess.run(['rsync', '-az',
                        f'{args.host}:{args.remote_dir}/{name}_out1.bin',
                        str(work)], check=True)
        raw = (work / f'{name}_out1.bin').read_bytes()
        got1 = np.frombuffer(raw, dtype='<i4').reshape(case['steps'], max_lane)
        got_lane0 = got1[:len(case['golden']), 0].tolist()
        ok = got_lane0 == case['golden']
        print(f'  {name}: lane=1 整合性 {"OK" if ok else "NG"} '
             f'(got={got_lane0} want={case["golden"]})')

        perf = {}
        for nb in lanes:
            out = sh(args.host, f'cd {args.remote_dir} && '
                                f'./{name}_test {nb} {case["src_width"]} '
                                f'{case["bos_token"]} {max_lane} {args.reps} '
                                f'< {name}_in.bin 2>&1 >/dev/null | grep BENCH')
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] == 'BENCH':
                    perf[nb] = {'ms_per_call_best': float(parts[2]),
                               'tok_per_s': float(parts[3]),
                               'ms_mean': float(parts[4])}
                    print(f'    lanes={nb:5d}  ms/call(best)={float(parts[2]):.4f}  '
                          f'tok/s={float(parts[3]):.1f}')
        results[name] = {'match': bool(ok), 'perf': perf,
                         'megakernel': markers.get(name)}
    return {'gpu': gpu_name, 'results': results}


def run_metal(args, cases: list[dict]) -> dict:
    from griffith_logic.deploy.rddlgn_decode_gpu import generate_rddlgn_decode_metal
    work = Path(tempfile.mkdtemp(prefix='mdl_metal_rddlgn_decode_'))
    print(f'作業ディレクトリ: {work}')
    lanes = [int(x) for x in args.lanes.split(',')]
    max_lane = max(lanes)
    results = {}
    for case in cases:
        name = case['name']
        cwork = work / name
        cwork.mkdir(parents=True)
        megakernel_marker = None
        for filename, text in generate_rddlgn_decode_metal(
                case['bundle'], batch=max_lane,
                megakernel=args.megakernel).items():
            (cwork / filename).write_text(text, encoding='utf-8')
            if args.megakernel and filename.endswith('.mm'):
                megakernel_marker = 'megakernel: enabled' in text
        if args.megakernel:
            print(f'  {name}: megakernel='
                 f'{"enabled" if megakernel_marker else "fallback"}')
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

        src_bytes = case['src_bits'].tobytes()

        proc = subprocess.run(
            [str(cwork / 'app'), '1', str(case['src_width']),
             str(case['bos_token']), str(max_lane)],
            input=src_bytes, capture_output=True)
        if proc.returncode != 0:
            print(f'  {name}: Metal実行不可: '
                 f'{proc.stderr.decode(errors="replace")[:300]}')
            results[name] = {'error': 'run_failed'}
            continue
        got1 = np.frombuffer(proc.stdout, dtype='<i4').reshape(
            case['steps'], max_lane)
        got_lane0 = got1[:len(case['golden']), 0].tolist()
        ok = got_lane0 == case['golden']
        print(f'  {name}: lane=1 整合性 {"OK" if ok else "NG"} '
             f'(got={got_lane0} want={case["golden"]})')

        perf = {}
        for nb in lanes:
            proc = subprocess.run(
                [str(cwork / 'app'), str(nb), str(case['src_width']),
                 str(case['bos_token']), str(max_lane), str(args.reps)],
                input=src_bytes * nb, capture_output=True)
            for line in proc.stderr.decode(errors='replace').splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] == 'BENCH':
                    perf[nb] = {'ms_per_call_best': float(parts[2]),
                               'tok_per_s': float(parts[3]),
                               'ms_mean': float(parts[4])}
                    print(f'    lanes={nb:5d}  ms/call(best)={float(parts[2]):.4f}  '
                          f'tok/s={float(parts[3]):.1f}')
        results[name] = {'match': bool(ok), 'perf': perf,
                         'megakernel': megakernel_marker}
    return {'results': results}


# ---------------------------------------------------------------------------
# 非推奨の薄いラッパ。計測本体(build_case/run_cuda/run_metal、上の
# すべての関数)はscripts/bench_deploy.pyが`--backend metal|cuda`から
# そのままimportして呼ぶ「計測エンジン」として残る。このmain()自体は
# 旧CLI引数を`bench_deploy.py`の統一引数へ変換して呼ぶだけ(挙動互換:
# `--out`未指定なら従来通りファイルは書かない)。
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
    ap.add_argument('--models', default='tiny')
    ap.add_argument('--host', default=None)
    ap.add_argument('--remote-dir', default='mdl_cuda_rddlgn_decode_verify')
    ap.add_argument('--arch', default='sm_89')
    ap.add_argument('--nvcc-path',
                    default='$HOME/.local/micromamba/envs/llamabuild/bin')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--lanes', default='1,32,256')
    ap.add_argument('--megakernel', action='store_true',
                    help='decodeのtickループ全体を1スレッドグループ'
                         '(Metal)/1スレッドブロック(CUDA)=1レーンの単一'
                         'カーネルへ畳む経路を使う(既定は従来のK51'
                         'tickループ経路)。32KB超/Metalバッファ引数上限'
                         '超のモデルは生成時に自動フォールバックする'
                         '(results中の"megakernel"キーで判別できる)。')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print('[非推奨] scripts/bench_rddlgn_decode_gpu.py は '
          'scripts/bench_deploy.py --backend metal|cuda --rddlgn-only '
          'へ統一された。今後は `uv run python scripts/bench_deploy.py '
          '--backend metal --rddlgn-only ...` を使うこと。このラッパは'
          '統一実装をそのまま呼び出すだけ。', file=sys.stderr)

    if args.backend == 'cuda' and not args.host:
        print('--backend cuda には --host が必要', file=sys.stderr)
        return 1

    deploy = _load_bench_deploy()
    forward = ['--backend', args.backend, '--rddlgn-only',
              '--rddlgn-gpu-metrics', 'decode',
              '--rddlgn-models', args.models,
              '--rddlgn-seed', str(args.seed),
              '--rddlgn-reps', str(args.reps),
              '--rddlgn-lanes', args.lanes,
              '--remote-dir', args.remote_dir, '--arch', args.arch,
              '--nvcc-path', args.nvcc_path]
    if args.host:
        forward += ['--host', args.host]
    if args.megakernel:
        forward += ['--rddlgn-megakernel']
    if args.out:
        forward += ['--rddlgn-out', args.out]
    return deploy.main(forward)


if __name__ == '__main__':
    sys.exit(main())
