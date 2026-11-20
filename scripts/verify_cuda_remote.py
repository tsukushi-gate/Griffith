#!/usr/bin/env python
"""CUDAバックエンドをリモートGPU機で検証する(K9)。

手元にNVIDIA GPUが無いので、ssh先でビルド・実行し、結果を
`ir/simulate.py` と**整数厳密**に突き合わせる。Cバックエンドの
`griffith deploy verify-c` と同じ役割をCUDAに対して果たす。

やること:
    1. モデル → バンドル → CUDAソースを生成
    2. ランタイム(runtime/cuda, runtime/gpu)ごと rsync
    3. リモートで nvcc ビルド
    4. ランダム入力(ビット列 / 生画像の両経路)で実行
    5. simulate の結果と比較

使い方:
    uv run python scripts/verify_cuda_remote.py \
        --host <user>@<gpu-host> \
        --models seg_convdlgn_k32,det_face_lut_k256,har_dlgn_k600_gumbel
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cudabackend import generate_cuda  # noqa: E402
from griffith_logic.deploy.preproc_desc import (  # noqa: E402
    build_preproc_descriptors,
    encode_bits,
)
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
RUNTIME = REPO / 'runtime'

# リモートで nvcc を見つけるためのPATH候補(見つからなければそのまま nvcc)
DEFAULT_NVCC_PATH = '$HOME/.local/micromamba/envs/llamabuild/bin'

DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" {
int mdl_cuda_infer_batch(const unsigned char*, int*, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_max_batch(void);
#ifdef HAS_IMAGE
int mdl_cuda_infer_image_batch(const unsigned char*, int*, int);
#endif
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
/* 引数: <b|i> [枚数] [反復]。反復>0 なら結果ではなく ms/枚 を stderr に出す。 */
int main(int argc, char** argv) {
  int nin = mdl_cuda_input_width(), nout = mdl_cuda_output_width();
  int use_image = (argc > 1 && argv[1][0] == 'i');
  int n = (argc > 2) ? atoi(argv[2]) : 1;
  int reps = (argc > 3) ? atoi(argv[3]) : 0;
  int per = nin, rc, r;
#ifdef HAS_IMAGE
  if (use_image) per = IMAGE_BYTES;
#endif
  if (n < 1 || n > mdl_cuda_max_batch()) {
    fprintf(stderr, "batch out of range\n"); return 4;
  }
  {
  unsigned char* buf = (unsigned char*)malloc((size_t)n * per);
  int* scores = (int*)malloc((size_t)n * nout * sizeof(int));
  if (fread(buf, 1, (size_t)n * per, stdin) != (size_t)n * per) {
    fprintf(stderr, "short read\n"); return 1;
  }
#ifdef HAS_IMAGE
#define MDL_RUN() (use_image ? mdl_cuda_infer_image_batch(buf, scores, n) \
                             : mdl_cuda_infer_batch(buf, scores, n))
#else
#define MDL_RUN() ((void)use_image, mdl_cuda_infer_batch(buf, scores, n))
#endif
  rc = MDL_RUN();
  if (rc != 0) { fprintf(stderr, "infer failed\n"); return 2; }
  if (reps > 0) {
    double best = 1e18;
    for (r = 0; r < reps; r++) {
      double t0 = now_ms();
      if (MDL_RUN() != 0) return 2;
      { double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    fprintf(stderr, "BENCH %d %.4f %.1f\n", n, best / n, n * 1000.0 / best);
    return 0;
  }
  fwrite(scores, sizeof(int), (size_t)n * nout, stdout);
  }
  return 0;
}
'''


def sh(host: str, cmd: str, *, check: bool = True) -> str:
    proc = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', host, cmd],
        capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f'リモート実行に失敗:\n{cmd}\n{proc.stderr}')
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', required=True, help='ssh先(user@host)')
    ap.add_argument('--models', default='seg_convdlgn_k32,har_dlgn_k600_gumbel',
                    help='検証するモデル名のカンマ区切り')
    ap.add_argument('--remote-dir', default=None,
                    help='既定: mdl_cuda_verify(--conv-codegen指定時は'
                         'mdl_cuda_verify_codegen、ONとOFFの成果物を'
                         '混ぜないため)')
    ap.add_argument('--arch', default='sm_89', help='nvcc の -arch(default: sm_89)')
    ap.add_argument('--nvcc-path', default=DEFAULT_NVCC_PATH,
                    help='リモートでPATHに足すディレクトリ')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bench', action='store_true', help='性能も測る')
    ap.add_argument('--batch', type=int, default=1,
                    help='同時に流す枚数(既定1)。等価性は「batch=N が '
                         'batch=1 を N 回回したものと一致する」で見る')
    ap.add_argument('--bench-batches', default='1,4,16,64',
                    help='--bench のときに掃く枚数(カンマ区切り)')
    ap.add_argument('--conv-codegen', action='store_true', dest='conv_codegen',
                    help='conv2dノードごとのモデル特化CUDAカーネル生成'
                         '(--conv-codegen)を検証・計測する(既定OFF='
                         'インタプリタ mdl_k_conv_exec_split のまま)')
    ap.add_argument('--dense-flat', choices=['auto', 'on', 'off'], default='off',
                    help='codegen-cuda と同じ平坦化判定(既定 off = codegen-cuda の既定と'
                         '同じ。lut_flat の CUDA 経路を検証するときは on)。ゴールデンは'
                         '元バンドルIRの simulate のまま')
    args = ap.parse_args()
    if args.remote_dir is None:
        args.remote_dir = ('mdl_cuda_verify_codegen' if args.conv_codegen
                           else 'mdl_cuda_verify')

    names = [n for n in args.models.split(',') if n]
    work = Path(tempfile.mkdtemp(prefix='mdl_cuda_'))
    print(f'作業ディレクトリ: {work}')

    cases = []
    for name in names:
        path = MODELS_DIR / f'{name}.json'
        if not path.exists():
            print(f'  {name}: モデルが無いのでskip')
            continue
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)
        max_batch = max([args.batch] + (
            [int(v) for v in args.bench_batches.split(',') if v.strip()]
            if args.bench else []))
        from griffith_logic.deploy.flatten_dense import (
            describe_dense_flat,
            resolve_dense_flat,
        )
        gen_bundle, flat_dec = resolve_dense_flat(
            bundle, mode=args.dense_flat, target='gpu')
        print(f'  {name}: ' + describe_dense_flat(flat_dec))
        for filename, text in generate_cuda(
                gen_bundle, batch=max_batch,
                conv_codegen=args.conv_codegen).items():
            (work / f'{name}_{filename}').write_text(text, encoding='utf-8')

        graph = ir_from_dict(bundle['ir'])
        rng = np.random.default_rng(args.seed)
        nb = args.batch
        plan = (build_preproc_descriptors(bundle['preproc'])
                if bundle.get('preproc_desc') is not None else None)
        if plan is not None:
            images = rng.integers(
                0, 256,
                size=(nb, plan['image_channels'], plan['height'],
                      plan['width']),
                dtype=np.uint8)
            bits = encode_bits(plan, images)
            (work / f'{name}_img.bin').write_bytes(images.tobytes())
            image_bytes = images[0].size
        else:
            bits = rng.integers(0, 2, size=(nb, int(graph.inputs[0].width)),
                                dtype=np.uint8)
            image_bytes = 0
        (work / f'{name}_bits.bin').write_bytes(bits.tobytes())
        # 基準は「1枚ずつ simulate したもの」。バッチで枚をまたいだ参照が
        # あれば、ここと食い違う。
        want = simulate(graph, bits).astype(np.int64)
        np.save(work / f'{name}_want.npy', want)
        cases.append({'name': name, 'image_bytes': image_bytes,
                      'n_out': int(want[0].size), 'batch': nb})
        print(f'  {name}: 生成OK(出力{want[0].size}要素×{nb}枚、'
              f'画像経路={"あり" if image_bytes else "なし"}、'
              f'MDL_MAX_BATCH={max_batch})')

    if not cases:
        print('検証対象が無い', file=sys.stderr)
        return 1

    (work / 'driver.cu').write_text(DRIVER, encoding='utf-8')
    for rel in ('cuda/mdl_cuda.h', 'cuda/mdl_cuda.cu', 'gpu/mdl_gpu_kernels.h'):
        (work / Path(rel).name).write_text(
            (RUNTIME / rel).read_text(encoding='utf-8'), encoding='utf-8')

    sh(args.host, f'mkdir -p {args.remote_dir}')
    subprocess.run(
        ['rsync', '-az', '--delete',
         *[str(p) for p in sorted(work.glob('*'))],
         f'{args.host}:{args.remote_dir}/'],
        check=True)
    print('転送完了')

    env = f'export PATH={args.nvcc_path}:$PATH'
    sh(args.host,
       f'cd {args.remote_dir} && {env} && '
       f'nvcc -O2 -arch={args.arch} -I. -c mdl_cuda.cu -o mdl_cuda.o')
    print('ランタイムのビルドOK')

    failures = 0
    for case in cases:
        name = case['name']
        flags = ''
        if case['image_bytes']:
            flags = f"-DHAS_IMAGE -DIMAGE_BYTES={case['image_bytes']}"
        sh(args.host,
           f'cd {args.remote_dir} && {env} && '
           f'nvcc -O2 -arch={args.arch} -I. {flags} '
           f'{name}_model_cuda.cu driver.cu mdl_cuda.o -o {name}_test')

        want = np.load(work / f'{name}_want.npy')
        paths = ['b'] + (['i'] if case['image_bytes'] else [])
        nb = case['batch']
        for mode in paths:
            src = f'{name}_img.bin' if mode == 'i' else f'{name}_bits.bin'
            sh(args.host,
               f'cd {args.remote_dir} && ./{name}_test {mode} {nb} 0 < {src} '
               f'> {name}_{mode}_out.bin')
            subprocess.run(
                ['rsync', '-az',
                 f'{args.host}:{args.remote_dir}/{name}_{mode}_out.bin',
                 str(work)], check=True)
            got = np.frombuffer(
                (work / f'{name}_{mode}_out.bin').read_bytes(),
                dtype='<i4').astype(np.int64)
            label = '画像経路' if mode == 'i' else 'ビット経路'
            suffix = f'×{nb}枚' if nb > 1 else ''
            if got.size == want.size and np.array_equal(
                    got.reshape(want.shape), want):
                print(f'  {name:26s} {label}: 一致'
                      f'({want[0].size}要素{suffix}、整数厳密)')
            else:
                bad = int((got != want.ravel()).sum()) \
                    if got.size == want.size else -1
                print(f'  {name:26s} {label}: ★不一致 {bad}/{want.size}')
                failures += 1
        if args.bench:
            batches = [int(v) for v in args.bench_batches.split(',')
                       if v.strip()]
            mode = 'i' if case['image_bytes'] else 'b'
            src = f'{name}_img.bin' if mode == 'i' else f'{name}_bits.bin'
            print(f'  {name} 性能(経路={"画像" if mode == "i" else "ビット"}):')
            print('      n   ms/枚      枚/秒')
            for nb2 in batches:
                # 入力は先頭を繰り返して n 枚ぶんに水増しする
                out = sh(args.host,
                         f'cd {args.remote_dir} && '
                         f'(for i in $(seq 1 {nb2}); do cat {src}; done) | '
                         f'./{name}_test {mode} {nb2} 20 2>&1 >/dev/null '
                         '| grep BENCH')
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) == 4 and parts[0] == 'BENCH':
                        print(f'    {int(parts[1]):3d}  {float(parts[2]):7.3f}'
                              f'  {float(parts[3]):9.1f}')

    if args.bench:
        print('\n性能(median、ホスト↔デバイス転送込み):')
        print(sh(args.host, f'cd {args.remote_dir} && nvidia-smi '
                            '--query-gpu=name --format=csv,noheader').strip())

    print(f'\n結果: {"全て一致" if failures == 0 else f"{failures}件が不一致"}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
