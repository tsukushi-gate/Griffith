#!/usr/bin/env python
"""dense 平坦化(`--dense-flat`)のホスト計測: 従来 C と flat C の µs/推論。

合成の dense IR(HAR 相当の 6 層 × 6000 幅など)を作り、同じバンドルから

    従来 C  — lut_layer 列をそのまま(全ゲートを bit-slice カーネルで実行)
    flat C  — `flatten_dense` で縮約した lut_flat(実効ゲートだけを表駆動で実行)

を生成して `cc -O2`(任意で `-DMDL_NO_SIMD` = スカラー)でビルドし、
µs/推論を比べる。**数値は参考**(ホストの SIMD 経路と、ESP32 のようなスカラー
Xtensa では従来 C の 1 ゲートあたりのコストが違う。実機は別途計測すること)。
スコアの XOR チェックサムを両者で比べ、不一致なら失敗にする。

使い方:
    uv run python scripts/bench_dense_flat.py                 # 6層×6000、in=4608
    uv run python scripts/bench_dense_flat.py --widths 400 400 --in-width 64
    uv run python scripts/bench_dense_flat.py --no-simd       # スカラー経路で比較
    uv run python scripts/bench_dense_flat.py --p-trivial 0.2  # 自明ttを減らす

tt の分布: 既定は 16 通りを一様に引く(自明な tt = 定数 2 + パススルー/NOT 4
= 6/16 = 0.375)。ランダム配線は上流の定数が下流へ波及して実効ゲートが
大きく減る(6層×6000 で約 75%)。学習済み HAR は 58% 削減なので、
`--p-trivial` を下げると実モデルに近い削減率になる。
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cbackend import generate_c  # noqa: E402
from griffith_logic.deploy.flatten_dense import flatten_dense_doc  # noqa: E402
from griffith_logic.deploy.verify import RUNTIME_C, find_cc  # noqa: E402
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort  # noqa: E402
from griffith_logic.ir.serialize import ir_to_dict  # noqa: E402

_DRIVER = r'''#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
int main(int argc, char **argv) {
  static uint8_t bits[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  int reps = argc > 2 ? atoi(argv[2]) : 1;
  int i, j, r;
  unsigned long long chk = 0ull;
  double best = 1e30;
  unsigned s = 12345u;
  for (j = 0; j < MDL_INPUT_WIDTH; j++) { s = s * 1103515245u + 12345u; bits[j] = (s >> 16) & 1u; }
  for (i = 0; i < 5; i++) mdl_infer(bits, scores);
  for (i = 0; i < iters; i++) {
    double t0 = now_us(), dt;
    for (r = 0; r < reps; r++) mdl_infer(bits, scores);
    dt = (now_us() - t0) / reps;
    if (dt < best) best = dt;
  }
  for (j = 0; j < MDL_OUTPUT_WIDTH; j++) chk = chk * 1000003ull + (unsigned)scores[j];
  printf("%.3f %llu\n", best, chk);
  return 0;
}
'''


def build_synthetic_doc(in_width: int, widths: list[int], k: int, seed: int,
                        p_trivial: float) -> dict:
    """ランダム dense IR(lut_layer 列 + group_sum)のモデル doc。

    p_trivial: 自明な tt(定数 0/15・パススルー/NOT 3/5/10/12、計 6 通り)を
    選ぶ確率。残りは自明でない 10 通りから一様に引く(0.375 で一様ランダム)。
    """
    rng = np.random.default_rng(seed)
    trivial = [0x0, 0xf, 0xa, 0xc, 0x3, 0x5]
    nodes, edges = [], [IREdge(src='x', dst='n1')]
    prev_w = in_width
    for li, w in enumerate(widths):
        idx = rng.integers(0, prev_w, size=(2, w)).astype(np.int64)
        nontrivial = [t for t in range(16) if t not in trivial]
        tts = np.where(rng.random(w) < p_trivial,
                       rng.choice(trivial, size=w),
                       rng.choice(nontrivial, size=w))
        nodes.append(IRNode(id=f'n{li + 1}', type='lut_layer', params={
            'in_width': prev_w, 'out_width': w, 'n_inputs': 2,
            'indices': idx, 'truth_tables': [f'0x{int(t):x}' for t in tts]}))
        edges.append(IREdge(src=f'n{li + 1}', dst=f'n{li + 2}'))
        prev_w = w
    gs = f'n{len(widths) + 1}'
    nodes.append(IRNode(id=gs, type='group_sum',
                        params={'in_width': prev_w, 'k': k, 'tau': 1.0}))
    edges.append(IREdge(src=gs, dst='y'))
    graph = IRGraph(
        metadata={'name': 'synthetic_dense'},
        inputs=[IRPort(id='x', width=in_width, encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=nodes, edges=edges)
    return {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}


def time_bundle(bundle: dict, work: Path, cc: str, *, no_simd: bool,
                iters: int, reps: int) -> tuple[float, str, dict]:
    """バンドルから C を生成・ビルド・実行し (µs/推論 best, チェックサム, 生成物情報)。"""
    work.mkdir(parents=True, exist_ok=True)
    files = generate_c(bundle, name='model', validate=False)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(_DRIVER, encoding='utf-8')
    exe = work / 'app'
    cmd = [cc, '-std=gnu99', '-O2', f'-I{work}', f'-I{RUNTIME_C / "include"}']
    if no_simd:
        cmd.append('-DMDL_NO_SIMD')
    cmd += [str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
            str(work / 'driver.c'), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f'ビルド失敗:\n{build.stderr}')
    run = subprocess.run([str(exe), str(iters), str(reps)], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f'実行失敗:\n{run.stderr}')
    us, chk = run.stdout.split()
    header = (work / 'model.h').read_text(encoding='utf-8')
    ctx_bytes = int(next(
        line.split()[-1] for line in header.splitlines()
        if line.startswith('#define MDL_CTX_BYTES')).rstrip('u'))
    src_bytes = len((work / 'model.c').read_text(encoding='utf-8'))
    return float(us), chk, {'ctx_bytes': ctx_bytes, 'model_c_bytes': src_bytes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--in-width', type=int, default=4608)
    ap.add_argument('--widths', type=int, nargs='+', default=[6000] * 6)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--p-trivial', type=float, default=0.375)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--reps', type=int, default=0,
                    help='1回の計時で連続実行する推論回数(0なら規模から自動)')
    ap.add_argument('--no-simd', action='store_true',
                    help='-DMDL_NO_SIMD(スカラー経路。ESP32 の目安)')
    args = ap.parse_args()

    cc = find_cc()
    if cc is None:
        print('Cコンパイラが無い')
        return 2
    doc = build_synthetic_doc(args.in_width, args.widths, args.k, args.seed,
                              args.p_trivial)
    flat_doc, report = flatten_dense_doc(doc)
    print(f'ゲート数: {report["gates_before"]:,} → {report["gates_after"]:,} '
          f'({100.0 * (1 - report["gates_after"] / report["gates_before"]):.1f}%削減、'
          f'定数{report["constants"]:,} / パススルー{report["passthrough"]:,} / '
          f'CSE{report["cse"]:,})')
    reps = args.reps or max(1, 400_000 // max(1, report['gates_before']))
    with tempfile.TemporaryDirectory(prefix='dense_flat_bench_') as tmp:
        base_us, base_chk, base_info = time_bundle(
            build_bundle(doc), Path(tmp) / 'base', cc,
            no_simd=args.no_simd, iters=args.iters, reps=reps)
        flat_us, flat_chk, flat_info = time_bundle(
            build_bundle(flat_doc), Path(tmp) / 'flat', cc,
            no_simd=args.no_simd, iters=args.iters, reps=reps)
    if base_chk != flat_chk:
        print(f'スコアが一致しない: 従来C={base_chk} flatC={flat_chk}')
        return 1
    mode = 'MDL_NO_SIMD(スカラー)' if args.no_simd else 'SIMD'
    print(f'[{mode}] 従来C {base_us:,.1f} µs/推論  →  flat C {flat_us:,.1f} µs/推論 '
          f'({base_us / flat_us:.2f}x)')
    print(f'  作業RAM(MDL_CTX_BYTES): {base_info["ctx_bytes"]:,}B → '
          f'{flat_info["ctx_bytes"]:,}B   model.c: '
          f'{base_info["model_c_bytes"]:,}B → {flat_info["model_c_bytes"]:,}B')
    return 0


if __name__ == '__main__':
    sys.exit(main())
