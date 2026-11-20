#!/usr/bin/env python
"""dense の 4 経路を生成 C で比べる(レイテンシ batch=1 とスループット)。

同じバンドルから次の 4 本を `generate_c` で生成し、`cc -O2` でビルドして測る。

    layer-1   従来の bit-slice(lut_layer、--dense-flat off)を 1 枚ずつ
    flat-1 平坦化(lut_flat)を 1 枚ずつ ← 既定
    layer-pk  従来の lut_layer をバッチ軸パック(--dense-batch)
    flat-pk 平坦化 + バッチ軸パック(で新設)

加えて `--old-kernels <rev>` を渡すと、layer-pk を**その版の** mdl_kernels.c
でもビルドして測る(の mdl_pack_batch 高速化の前後比較用)。

全経路のスコアを layer-1 と全要素比較し、1 つでも違えば失敗にする。
時間は「N 枚を処理する 1 回」を reps 回まわした平均の、iters 回中の最良値を
1 枚あたり µs に直したもの。N=1 の行がレイテンシ、N が大きい行がスループット。

使い方:
    uv run python scripts/bench_flat_variants.py --models <model.json|bundle.json> ...
    uv run python scripts/bench_flat_variants.py --models a.json --no-simd   # ESP32 の目安
    uv run python scripts/bench_flat_variants.py --models a.json --log-wandb
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cbackend import generate_c  # noqa: E402
from griffith_logic.deploy.flatten_dense import resolve_dense_flat  # noqa: E402
from griffith_logic.deploy.verify import RUNTIME_C, find_cc  # noqa: E402
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402

DEFAULT_BATCHES = (1, 32, 128, 1024)

_DRIVER = r'''#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mdl_stdio.h"
static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
/* argv: mode(0=1枚ずつ, 1=packed) n iters reps。入力は stdin から n 枚。 */
int main(int argc, char **argv) {
  const int mode = atoi(argv[1]), n = atoi(argv[2]);
  const int iters = atoi(argv[3]), reps = atoi(argv[4]);
  uint8_t *in = malloc((size_t)n * MDL_INPUT_WIDTH);
  int32_t *sc = malloc(sizeof(int32_t) * (size_t)n * MDL_OUTPUT_WIDTH);
  double best = 1e30;
  int it, r, i;
  MDL_STDIO_BINARY();
#if defined(_WIN32)
  _setmode(_fileno(stderr), _O_BINARY); /* スコアは stderr にバイナリで書く */
#endif
  if (fread(in, 1, (size_t)n * MDL_INPUT_WIDTH, stdin) != (size_t)n * MDL_INPUT_WIDTH) return 1;
#ifdef MDL_DENSE_BATCH
  uint32_t *dw = malloc(MDL_DENSE_WORK_BYTES);
#endif
  for (it = 0; it < iters; it++) {
    double t0 = now_us();
    for (r = 0; r < reps; r++) {
      if (mode == 0) {
        for (i = 0; i < n; i++)
          mdl_infer(in + (size_t)i * MDL_INPUT_WIDTH, sc + (size_t)i * MDL_OUTPUT_WIDTH);
      } else {
#ifdef MDL_DENSE_BATCH
        if (mdl_infer_batch_packed(dw, MDL_DENSE_WORK_BYTES, in, sc, n) != 0) return 2;
#else
        return 3;
#endif
      }
    }
    {
      const double dt = (now_us() - t0) / reps / n;
      if (dt < best) best = dt;
    }
  }
  printf("%.5f\n", best);
  fwrite(sc, sizeof(int32_t), (size_t)n * MDL_OUTPUT_WIDTH, stderr);
  return 0;
}
'''


def _load_bundle(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('format') == 'magicdelta-bundle' or 'plan_hints' in doc:
        return doc
    return build_bundle(doc)


def _build(bundle: dict, work: Path, cc: str, *, flat: bool, batch: int,
           no_simd: bool, kernels_c: Path) -> tuple[Path, dict]:
    work.mkdir(parents=True, exist_ok=True)
    b, dec = resolve_dense_flat(bundle, mode='on' if flat else 'off')
    files = generate_c(b, name='model', validate=False, dense_batch=batch)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(_DRIVER, encoding='utf-8')
    exe = work / 'app'
    cmd = [cc, '-std=gnu99', '-O2', f'-I{work}', f'-I{RUNTIME_C / "include"}']
    if no_simd:
        cmd.append('-DMDL_NO_SIMD')
    cmd += [str(work / 'model.c'), str(kernels_c), str(work / 'driver.c'), '-o', str(exe)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f'ビルド失敗 {work}:\n{res.stderr[-2000:]}')
    h = files['model.h']

    def macro(name):
        for line in h.splitlines():
            if line.startswith(f'#define {name} '):
                return int(line.split()[-1].rstrip('u'))
        return None
    info = {'ctx_bytes': macro('MDL_CTX_BYTES'),
            'dense_work_bytes': macro('MDL_DENSE_WORK_BYTES'),
            'model_c_bytes': len(files['model.c'].encode()),
            'gates_after': dec.get('gates_after'), 'gates_before': dec.get('gates_before')}
    return exe, info


def _run(exe: Path, mode: int, x: np.ndarray, iters: int, reps: int):
    p = subprocess.run([str(exe), str(mode), str(len(x)), str(iters), str(reps)],
                       input=x.tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(f'実行失敗 {exe} mode={mode}: rc={p.returncode}')
    us = float(p.stdout.decode().strip())
    scores = np.frombuffer(p.stderr, dtype='<i4')
    return us, scores


def bench_model(path: Path, cc: str, *, batches, no_simd: bool, old_kernels: Path | None,
                budget_gate_evals: float) -> dict:
    bundle = _load_bundle(path)
    graph = ir_from_dict(bundle['ir'])
    in_w = int(graph.inputs[0].width)
    nmax = max(batches)
    x_all = np.random.default_rng(7).integers(0, 2, size=(nmax, in_w), dtype=np.uint8)
    gates = sum(int(n.params.get('out_width', 0)) for n in graph.nodes
                if n.type == 'lut_layer')
    out = {'model': path.name, 'in_width': in_w, 'gates_lut_layer': gates,
           'simd': not no_simd, 'rows': []}
    kc = RUNTIME_C / 'src' / 'mdl_kernels.c'
    with tempfile.TemporaryDirectory(prefix='flatvar_') as tmp:
        t = Path(tmp)
        arms = {
            'layer': _build(bundle, t / 'layer', cc, flat=False, batch=nmax,
                            no_simd=no_simd, kernels_c=kc),
            'flat': _build(bundle, t / 'flat', cc, flat=True, batch=nmax,
                           no_simd=no_simd, kernels_c=kc),
        }
        if old_kernels is not None:
            arms['layer_oldpack'] = _build(bundle, t / 'old', cc, flat=False, batch=nmax,
                                           no_simd=no_simd, kernels_c=old_kernels)
        out['info'] = {k: v[1] for k, v in arms.items()}
        for n in batches:
            x = x_all[:n]
            # 1 回の計時が十分長くなるよう reps を規模から決める
            reps = max(1, int(budget_gate_evals / max(1, gates * n)))
            row = {'n': n}
            ref = None
            for arm, (exe, _) in arms.items():
                for mode, label in ((0, '1'), (1, 'pk')):
                    if arm == 'layer_oldpack' and mode == 0:
                        continue
                    us, sc = _run(exe, mode, x, iters=9, reps=reps)
                    if ref is None:
                        ref = sc
                    elif not np.array_equal(sc, ref):
                        raise RuntimeError(f'{path.name} n={n} {arm}-{label} のスコアが不一致')
                    row[f'{arm}-{label}_us'] = us
            out['rows'].append(row)
            print(f'  n={n:5d} ' + '  '.join(
                f'{k[:-3]}={v:8.3f}' for k, v in row.items() if k != 'n'), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', nargs='+', required=True, type=Path,
                    help='モデル doc(webbench 形式)またはバンドル JSON')
    ap.add_argument('--batches', nargs='+', type=int, default=list(DEFAULT_BATCHES))
    ap.add_argument('--no-simd', action='store_true', help='-DMDL_NO_SIMD(ESP32 の目安)')
    ap.add_argument('--old-kernels', default=None,
                    help='layer-pk をこの git 版の mdl_kernels.c でも測る(例 a276db43^)')
    ap.add_argument('--budget', type=float, default=3e7,
                    help='1 回の計時で評価するゲート数の目安(大きいほど安定・遅い)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--log-wandb', action='store_true')
    args = ap.parse_args()

    cc = find_cc()
    if cc is None:
        print('Cコンパイラが無い')
        return 2
    old = None
    tmpdir = tempfile.TemporaryDirectory(prefix='oldk_')
    if args.old_kernels:
        old = Path(tmpdir.name) / 'mdl_kernels_old.c'
        src = subprocess.run(['git', 'show', f'{args.old_kernels}:runtime/c/src/mdl_kernels.c'],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
        old.write_text(src, encoding='utf-8')
    results = []
    for m in args.models:
        print(f'== {m.name} ({"スカラー" if args.no_simd else "SIMD"})', flush=True)
        results.append(bench_model(m, cc, batches=args.batches, no_simd=args.no_simd,
                                   old_kernels=old, budget_gate_evals=args.budget))
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M')
    tag = 'scalar' if args.no_simd else 'simd'
    out = args.out or REPO / 'results' / 'deploy' / f'flat_bitpar_{tag}_{stamp}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'generated_at': datetime.now().astimezone().isoformat(), 'cc': cc,
               'simd': not args.no_simd, 'old_kernels': args.old_kernels, 'models': results}
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')
    if args.log_wandb:
        import wandb

        from griffith_logic.training.wandb_utils import init_wandb, log_artifact

        ok = init_wandb(f'flat-bitpar-{tag}',
                        {'simd': not args.no_simd, 'batches': args.batches,
                         'models': [m.name for m in args.models],
                         'old_kernels': args.old_kernels},
                        tags=['K77', tag], group='deploykit-flat-bitpar',
                        job_type='bench',
                        hypothesis='lut_flat をバッチ軸パックすればスループットが上がる。'
                                   'ゲート方向の32並列はレイテンシに効かない(試作で0.85〜0.98x)')
        if ok:
            cols = ['model', 'n', 'arm', 'us_per_sample']
            table = wandb.Table(columns=cols)
            for r in results:
                for row in r['rows']:
                    for k, v in row.items():
                        if k.endswith('_us'):
                            table.add_data(r['model'], row['n'], k[:-3], v)
            wandb.log({'flat_bitpar': table})
            log_artifact(str(out), name=f'flat_bitpar_{tag}', type='results')
            print(f'WANDB_RUN_ID={wandb.run.id}')
            wandb.finish()
    return 0


if __name__ == '__main__':
    sys.exit(main())
