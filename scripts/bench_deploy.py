#!/usr/bin/env python
"""デプロイキットの総合評価(全種別/全バックエンドの単一入口に統一)。

レイテンシだけ見ていると判断を誤る。**スループット・配信サイズ・メモリ**まで
同じ表に並べる:

    レイテンシ    1枚の推論(median と best)。前処理込み/なしの両方
    スループット  1コンテキスト / N コンテキスト並列 の 枚/秒
    配信サイズ    バンドル(gzip)/ 生成ソース / strip後バイナリ / gzip後
    メモリ        `__const`(重み)/ MDL_CTX_BYTES(作業RAM)/ ピークRSS

「作業RAMが重みの13倍」のような事実はレイテンシ表には出てこないが、
MCUに載るかどうかはそれで決まる。

このスクリプトが唯一の入口。モデル種別(dense/conv/RDDLGN)はIRから
自動判定し(`detect_kind`)、RDDLGNは同じ4指標(decode tok/s・1トークンあたり
ms/教師強制の系列一括スループット/配信サイズ/メモリ)で測る(計測ロジック自体は
`scripts/bench_rddlgn.py`のものをそのまま呼ぶ、測り方は無改修)。
`--backend metal|cuda`でGPU計測へ委譲する(`scripts/bench_dense_gpu.py`
/ `scripts/bench_rddlgn_decode_gpu.py`をimportして呼ぶ、subprocessは使わない)。
旧来の`scripts/bench_rddlgn.py`・`scripts/bench_rddlgn_decode_gpu.py`は非推奨の
薄いラッパになり、内部でこのスクリプトを呼ぶ。

使い方:
    uv run python scripts/bench_deploy.py                # 全dense/convモデル(既定、backend=c)
    uv run python scripts/bench_deploy.py --models seg_convdlgn_k32
    uv run python scripts/bench_deploy.py --fuse-compare         # 融合の効果
    uv run python scripts/bench_deploy.py --rddlgn               # RDDLGN既定4モデルも追加
    uv run python scripts/bench_deploy.py --rddlgn-only --rddlgn-models copy_tiny
    uv run python scripts/bench_deploy.py --backend metal --models har_dlgn_k600_gumbel
    uv run python scripts/bench_deploy.py --backend metal --rddlgn-only --rddlgn-models tiny
"""

import argparse
import gzip
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cbackend import generate_c  # noqa: E402
from griffith_logic.deploy.verify import RUNTIME_C, find_cc  # noqa: E402

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
OUT_DIR = REPO / 'results' / 'deploy'


# ---------------------------------------------------------------------------
# 兄弟スクリプトの遅延import。scripts/ はパッケージ外なので
# tests/deploy/conftest.py::_load_export_webbench と同じ流儀でファイルパスから
# 直接importする(計測ロジックは移植・複製せず、既存関数をそのまま呼ぶ)。
# ---------------------------------------------------------------------------
def _load_sibling(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bench_rddlgn():
    """scripts/bench_rddlgn.py: RDDLGN Cバックエンドの計測本体。"""
    return _load_sibling('bench_rddlgn')


def _bench_dense_gpu():
    """scripts/bench_dense_gpu.py: dense/conv系Metal計測。"""
    return _load_sibling('bench_dense_gpu')


def _bench_rddlgn_decode_gpu():
    """scripts/bench_rddlgn_decode_gpu.py: RDDLGN自己回帰生成のMetal/CUDA。"""
    return _load_sibling('bench_rddlgn_decode_gpu')


def _bench_rddlgn_gpu():
    """scripts/bench_rddlgn_gpu.py: RDDLGN教師強制のMetal/CUDA。"""
    return _load_sibling('bench_rddlgn_gpu')


def detect_kind(doc: dict) -> str:
    """モデル種別を自動判定する('dense' / 'conv' / 'rddlgn')。

    IRにrecurrentノードがあれば'rddlgn'(recurrent bodyは入れ子のIRGraphの
    dict表現なので再帰的に探す、`griffith_logic/ir/serialize.py`参照)。
    そうでなく`preproc.kind`が'image'始まりなら'conv'(image_thermometer/
    image_lowlevel)、それ以外は'dense'(thermometer等の非画像ビット列入力)。
    """
    def has_recurrent(ir_doc: dict) -> bool:
        for node in ir_doc.get('nodes', []):
            if node.get('type') == 'recurrent':
                return True
            body = (node.get('params') or {}).get('body')
            if isinstance(body, dict) and has_recurrent(body):
                return True
        return False

    if has_recurrent(doc.get('ir') or {}):
        return 'rddlgn'
    preproc_kind = (doc.get('preproc') or {}).get('kind') or ''
    if preproc_kind.startswith('image'):
        return 'conv'
    return 'dense'


DRIVER = '''#include "model.h"
#include "mdl_threads.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CTX 16
#define N_ITEMS 32

static uint32_t work[MAX_CTX][MDL_CTX_BYTES / 4];
static mdl_ctx ctx[MAX_CTX];
static void *ctxp[MAX_CTX];
static uint8_t bits[MDL_INPUT_WIDTH];
static uint8_t bits_n[N_ITEMS * MDL_INPUT_WIDTH];
static int32_t scores[N_ITEMS * MDL_OUTPUT_WIDTH];
#ifdef HAS_IMAGE
static uint8_t img[MDL_IMAGE_CHANNELS * MDL_INPUT_HEIGHT * MDL_INPUT_WIDTH_PX];
#endif

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 200;
  const int max_thr = argc > 2 ? atoi(argv[2]) : 8;
  double *t = (double *)malloc((size_t)reps * sizeof(double));
  int i, r, n;
  size_t k;

  for (k = 0; k < sizeof(bits); k++) bits[k] = (uint8_t)((k * 2654435761u) & 1);
  for (k = 0; k < sizeof(bits_n); k++)
    bits_n[k] = (uint8_t)((k * 2654435761u) & 1);
#ifdef HAS_IMAGE
  for (k = 0; k < sizeof(img); k++) img[k] = (uint8_t)((k * 2654435761u) >> 24);
#endif
  for (i = 0; i < MAX_CTX; i++) {
    if (mdl_ctx_init(&ctx[i], work[i], sizeof(work[i])) != 0) return 3;
    ctxp[i] = &ctx[i];
  }

  /* --- レイテンシ(推論のみ) --- */
  mdl_infer_ctx(&ctx[0], bits, scores);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_infer_ctx(&ctx[0], bits, scores);
    t[r] = now_ms() - t0;
  }
  qsort(t, (size_t)reps, sizeof(double), cmp_d);
  printf("infer_ms %.5f %.5f\\n", t[reps / 2], t[0]);

  /* --- レイテンシ(前処理込み) --- */
#ifdef HAS_IMAGE
  mdl_infer_image_ctx(&ctx[0], img, scores);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_infer_image_ctx(&ctx[0], img, scores);
    t[r] = now_ms() - t0;
  }
  qsort(t, (size_t)reps, sizeof(double), cmp_d);
  printf("e2e_ms %.5f %.5f\\n", t[reps / 2], t[0]);

  /* --- 前処理をコアで割ったときのレイテンシ --- */
  for (n = 1; n <= max_thr; n *= 2) {
    double best = 1e18;
    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_preproc_parallel(mdl_preproc_range_v, &ctx[0], img,
                           MDL_PREPROC_CHANNELS, n);
      mdl_run_ctx(&ctx[0], scores);
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("e2e_mt %d %.5f\\n", n, best);
  }
#endif

  /* --- スループット(Nコンテキスト並列) --- */
  for (n = 1; n <= max_thr && n <= MAX_CTX; n *= 2) {
    double best = 1e18;
    for (r = 0; r < 7; r++) {
      const double t0 = now_ms();
      if (mdl_run_pool(mdl_infer_ctx_v, ctxp, n, bits_n,
                       (size_t)MDL_INPUT_WIDTH, scores,
                       sizeof(int32_t) * MDL_OUTPUT_WIDTH, N_ITEMS) != 0)
        return 2;
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("thr %d %.1f\\n", n, N_ITEMS * 1000.0 / best);
  }
  printf("ctx_bytes %d\\n", MDL_CTX_BYTES);
  free(t);
  return 0;
}
'''


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def section_sizes(obj: Path) -> dict:
    """`size -m` の出力から `__const`(重み)と `__bss`(作業RAM)を拾う。"""
    out = sh(['size', '-m', str(obj)]).stdout
    got = {}
    for name, key in (('__TEXT, __const', 'const'), ('__DATA, __bss', 'bss'),
                      ('__TEXT, __text', 'text')):
        m = re.search(re.escape(name) + r'\): (\d+)', out)
        if m:
            got[key] = int(m.group(1))
    return got


def build_one(work: Path, bundle: dict, cc: str, *, fuse: bool,
              threads: bool = True):
    work.mkdir(parents=True, exist_ok=True)
    files = generate_c(bundle, name='model', validate=False, fuse=fuse)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h',
                'src/mdl_kernels.c', 'include/mdl_threads.h',
                'src/mdl_threads.c'):
        shutil.copy2(RUNTIME_C / rel, work / Path(rel).name)
    (work / 'driver.c').write_text(DRIVER, encoding='utf-8')
    has_image = 'mdl_infer_image_ctx' in files['model.h']
    flags = ['-DHAS_IMAGE'] if has_image else []
    if not threads:
        flags.append('-DMDL_NO_THREADS')
    cmd = [cc, '-std=c99', '-O2', *flags, f'-I{work}', '-pthread',
           str(work / 'model.c'), str(work / 'mdl_kernels.c'),
           str(work / 'mdl_threads.c'), str(work / 'driver.c'),
           '-o', str(work / 'app')]
    r = sh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f'ビルド失敗:\n{r.stderr[:800]}')
    sh([cc, '-std=c99', '-O2', *flags, f'-I{work}', '-c',
        str(work / 'model.c'), '-o', str(work / 'model.o')])
    return work / 'app', has_image


def measure(app: Path, reps: int, max_thr: int) -> dict:
    r = sh([str(app), str(reps), str(max_thr)])
    if r.returncode != 0:
        raise RuntimeError(f'実行失敗: {r.stderr[:400]}')
    out = {'e2e_mt': {}, 'thr': {}}
    for line in r.stdout.splitlines():
        f = line.split()
        if f[0] in ('infer_ms', 'e2e_ms'):
            out[f[0]] = (float(f[1]), float(f[2]))
        elif f[0] == 'e2e_mt':
            out['e2e_mt'][int(f[1])] = float(f[2])
        elif f[0] == 'thr':
            out['thr'][int(f[1])] = float(f[2])
        elif f[0] == 'ctx_bytes':
            out['ctx_bytes'] = int(f[1])
    return out


MIN_DRIVER = '''#include "model.h"
#include <stdlib.h>
/* ピークRSS計測用の最小ドライバ。コンテキスト1本・入力1枚だけを持つ
   (総合ベンチのドライバは並列度ぶんのバッファを抱えるので、
   そのRSSを「モデルのメモリ」と読むと桁を間違える)。 */
static uint32_t work[MDL_CTX_BYTES / 4];
static uint8_t bits[MDL_INPUT_WIDTH];
static int32_t scores[MDL_OUTPUT_WIDTH];
int main(void) {
  mdl_ctx ctx; size_t k; int r;
  for (k = 0; k < sizeof(bits); k++) bits[k] = (uint8_t)((k * 2654435761u) & 1);
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 3;
  for (r = 0; r < 20; r++) mdl_infer_ctx(&ctx, bits, scores);
  return scores[0] & 1;
}
'''


def peak_rss_kb(work: Path, cc: str) -> int:
    """モデル単体のピークRSS(KB)。最小ドライバで測る。"""
    (work / 'minimal.c').write_text(MIN_DRIVER, encoding='utf-8')
    r = sh([cc, '-std=c99', '-O2', f'-I{work}', str(work / 'model.c'),
            str(work / 'mdl_kernels.c'), str(work / 'minimal.c'),
            '-o', str(work / 'minimal')])
    if r.returncode != 0:
        return 0
    r = sh(['/usr/bin/time', '-l', str(work / 'minimal')])
    m = re.search(r'(\d+)\s+maximum resident set size', r.stderr)
    return int(m.group(1)) // 1024 if m else 0


def delivery_sizes(work: Path, bundle: dict) -> dict:
    src = sum((work / n).stat().st_size for n in ('model.c', 'model.h'))
    raw = json.dumps(bundle, separators=(',', ':')).encode()
    stripped = work / 'app.stripped'
    shutil.copy2(work / 'app', stripped)
    sh(['strip', '-x', str(stripped)])
    binz = stripped.stat().st_size
    return {
        'bundle_gz': len(gzip.compress(raw, 9)),
        'src': src,
        'bin': binz,
        'bin_gz': len(gzip.compress(stripped.read_bytes(), 9)),
    }


def kb(n: int) -> str:
    return f'{n / 1024:,.0f}KB'


def print_rddlgn_summary(results: list) -> None:
    """RDDLGN(backend=c)の4指標(decode tok/s・1トークンあたりms・
    教師強制の系列一括スループット・配信サイズ・メモリ)。値は
    `scripts/bench_rddlgn.py::bench_one`が計測したものをそのまま使う
    (計測ロジックは無改修、表示だけをdense/convの2表と対になる形にした)。
    詳細診断(全表)が要る場合は `--rddlgn-out` のJSONを見るか
    `scripts/bench_rddlgn.py`(非推奨だが動く)を直接呼ぶこと。"""
    print('\n## RDDLGN: 4指標(C, -O2)\n')
    print('| モデル | decode/tok ms | tokens/s | tf 1ctx 系列/s | '
          'tf 8ctx 系列/s | バンドル gz | バイナリ gz | 作業RAM | ピークRSS |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        if 'error' in r:
            print(f"| {r['name']} | 失敗: {r['error']} | | | | | | | |")
            continue
        p, s = r['perf'], r['sizes']
        print(f"| {r['name']} | {p['decode_step_ms']:.4f} | "
              f"{p['tokens_per_s']:.1f} | {p['tf_thr'].get(1, 0):.1f} | "
              f"{p['tf_thr'].get(8, 0):.1f} | {kb(s['bundle_gz'])} | "
              f"{kb(s['bin_gz'])} | {kb(p['ctx_bytes'])} | {r['rss_kb']:,}KB |")


def print_rddlgn_gpu_summary(backend: str, rows: list) -> None:
    """RDDLGNのMetal/CUDA実測を2本立てで出す。

    「生成(decode)tok/s」は`scripts/bench_rddlgn_decode_gpu.py::run_metal/
    run_cuda`を、「教師強制の系列一括スループット」は
    `scripts/bench_rddlgn_gpu.py::run_metal/run_cuda`を、それぞれ
    そのまま呼んだ結果を整形するだけ(計測ロジックは無改修)。
    `--rddlgn-gpu-metrics`で片方だけに絞った場合はそちらの表だけ出る。
    """
    decode_rows = [r for r in rows if 'perf' in r]
    if decode_rows:
        lanes_all = sorted({n for row in decode_rows for n in row['perf']})
        print(f'\n## RDDLGN decode: 生成tok/s({backend})\n')
        print('| モデル | 整合性 | '
              + ' | '.join(f'lanes={n}' for n in lanes_all) + ' |')
        print('|---|---|' + '---:|' * len(lanes_all))
        for row in decode_rows:
            lanes = sorted(row['perf'])
            cells = ' | '.join(
                f"{row['perf'][n]['tok_per_s']:.1f}" for n in lanes)
            match = row.get('match')
            match_s = '—' if match is None else ('OK' if match else 'NG!!')
            print(f"| {row['model']} | {match_s} | {cells} |")

    tf_rows = [r for r in rows if 'teacher_forcing' in r]
    if tf_rows:
        print(f'\n## RDDLGN 教師強制: 系列一括処理のスループット({backend})\n')
        print('| モデル | 整合性(batch=1) | ' + ' | '.join(
            f'batch={n}' for n in sorted({n for r in tf_rows
                                          for n in r['teacher_forcing']
                                          .get('perf', {})})) + ' |')
        for r in tf_rows:
            tf = r['teacher_forcing']
            batches = sorted(tf.get('perf', {}))
            cells = ' | '.join(
                f"{tf['perf'][n]['items_per_s']:.1f}/s" for n in batches)
            match_key = 'match' if 'match' in tf else 'match_batch1'
            match_s = '—' if match_key not in tf else (
                'OK' if tf[match_key] else 'NG!!')
            print(f"| {r['model']} | {match_s} | {cells} |")


def main(argv: list[str] | None = None) -> int:
    """`argv`(既定None=sys.argv[1:])。: 旧scripts/bench_rddlgn.py等の薄い
    ラッパから統一引数へ変換して呼べるよう、argvを直接渡せるようにしてある。"""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=None,
                    help='カンマ区切り(既定: dense/convの全部)。webbenchの'
                         'JSONモデル名を指定する')
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--threads', type=int, default=8, help='並列度の上限')
    ap.add_argument('--fuse', action='store_true', help='融合経路で測る')
    ap.add_argument('--fuse-compare', action='store_true',
                    help='融合あり/なしを並べる')
    ap.add_argument('--out', default=None,
                    help='統一結果JSON(dense/conv/RDDLGN全部を1本のrowsへ)')
    ap.add_argument('--backend', choices=['c', 'metal', 'cuda'], default='c',
                    help='計測バックエンド(既定c。metal/cudaは既存GPU'
                         'ハーネスへ関数呼び出しで委譲する)')
    # --- RDDLGN(旧scripts/bench_rddlgn.py・bench_rddlgn_decode_gpu.py) ---
    ap.add_argument('--rddlgn', action='store_true',
                    help='RDDLGN既定モデル一式もdense/convに加えて測る')
    ap.add_argument('--rddlgn-only', action='store_true',
                    help='dense/convをスキップし、RDDLGNだけ測る')
    ap.add_argument('--rddlgn-models', default=None,
                    help='RDDLGN対象をカンマ区切りで指定(既定: 全部。'
                         'backend=cならscripts/bench_rddlgn.MODEL_SPECS、'
                         'metal/cudaならbench_rddlgn_decode_gpu.MODEL_SPECS'
                         'のキー)')
    ap.add_argument('--rddlgn-opt', type=int, default=None,
                    help='(backend=c)最適化レベル固定。既定はpaper系モデルの'
                         'コンパイル時間から自動決定(bench_rddlgn.py同等)')
    ap.add_argument('--rddlgn-skip-verify', action='store_true')
    ap.add_argument('--rddlgn-max-thr', type=int, default=8)
    ap.add_argument('--rddlgn-optimize-degenerate', action='store_true')
    ap.add_argument('--rddlgn-out', default=None,
                    help='(backend=c)旧rddlgn_bench.json互換の出力先'
                         '(既定: results/deploy/rddlgn_bench.json。'
                         'backend=metal/cudaでは指定時のみ書く、旧'
                         'bench_rddlgn_decode_gpu.py --out と同じ既定)')
    ap.add_argument('--rddlgn-lanes', default='1,32,256',
                    help='(backend=metal/cuda)decode生成のレーン数掃引')
    ap.add_argument('--rddlgn-reps', type=int, default=30)
    ap.add_argument('--rddlgn-megakernel', action='store_true')
    ap.add_argument('--rddlgn-seed', type=int, default=0)
    ap.add_argument('--rddlgn-tf-batches', default='1,8,32',
                    help='(backend=metal/cuda)教師強制の系列一括スループット'
                         '(4指標のうち「スループット」)のバッチ数掃引。'
                         '旧scripts/bench_rddlgn_gpu.py --bench-batches相当'
                         '(旧scriptのMODEL_SPECSにあるモデルのみ対応、無い'
                         'ものはスキップしてその旨を出す)')
    ap.add_argument('--rddlgn-tf-remote-dir', default='mdl_cuda_rddlgn_verify',
                    help='(backend=cuda)教師強制スループット計測の'
                         'リモート作業ディレクトリ(decode用--remote-dirとは'
                         '別。同時実行時にファイルが衝突しないよう分離)')
    ap.add_argument('--rddlgn-gpu-metrics', choices=['both', 'decode', 'tf'],
                    default='both',
                    help='(backend=metal/cuda)どちらの指標を測るか。'
                         'decode=生成tok/s(旧bench_rddlgn_decode_gpu.py)、'
                         'tf=教師強制の系列一括スループット'
                         '(旧bench_rddlgn_gpu.py)、既定bothは両方(4指標'
                         'そろえるならこちら)')
    ap.add_argument('--host', default=None, help='(backend=cuda)SSHホスト')
    ap.add_argument('--remote-dir', default='mdl_cuda_rddlgn_decode_verify')
    ap.add_argument('--arch', default='sm_89')
    ap.add_argument('--nvcc-path',
                    default='$HOME/.local/micromamba/envs/llamabuild/bin')
    # --- dense/conv GPU(metal)委譲用(scripts/bench_dense_gpu.py) ---
    ap.add_argument('--gpu-batch', type=int, nargs='+', default=[1],
                    help='(backend=metal, dense/conv)推論1回あたりの枚数')
    ap.add_argument('--gpu-n-streams', type=int, default=1)
    ap.add_argument('--gpu-dense-fuse', action='store_true')
    ap.add_argument('--gpu-dense-packed', action='store_true')
    ap.add_argument('--gpu-sets', type=int, default=5)
    ap.add_argument('--gpu-reps', type=int, default=30)
    ap.add_argument('--gpu-warmup', type=int, default=8)
    args = ap.parse_args(argv)

    wanted = set(args.models.split(',')) if args.models else None

    def is_model(path: Path) -> bool:
        """モデルJSONか(teacher設定などの別JSONが混ざっている)。"""
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        return isinstance(doc, dict) and 'ir' in doc

    paths = []
    if not args.rddlgn_only:
        paths = [p for p in sorted(MODELS_DIR.glob('*.json'))
                 if not p.stem.endswith(('_dc', '_ll'))
                 and (wanted is None or p.stem in wanted) and is_model(p)]

    # --- RDDLGN対象の解決(backend/測る指標ごとに別のspec辞書を参照)。
    # metal/cudaでは「生成(decode)」と「教師強制の系列一括スループット(tf)」
    # が別スクリプト・別MODEL_SPECS辞書(旧bench_rddlgn_decode_gpu.py /
    # 旧bench_rddlgn_gpu.py)なので、--rddlgn-gpu-metricsに応じて対象名の
    # 全体集合(all_rddlgn、未知モデル判定用)を選ぶ。各枝の実行時はさらに
    # 「そのモデルがその枝のMODEL_SPECSに実在するか」で個別にフィルタする
    # (decode/tf_supported)。 ---
    rddlgn_names: list[str] = []
    if args.backend == 'c':
        rddlgn_module = _bench_rddlgn()
        all_rddlgn = set(rddlgn_module.MODEL_SPECS)
    else:
        decode_specs = set(_bench_rddlgn_decode_gpu().MODEL_SPECS)
        tf_specs = set(_bench_rddlgn_gpu().MODEL_SPECS)
        if args.rddlgn_gpu_metrics == 'decode':
            all_rddlgn = decode_specs
        elif args.rddlgn_gpu_metrics == 'tf':
            all_rddlgn = tf_specs
        else:
            all_rddlgn = decode_specs | tf_specs
    if args.rddlgn_models:
        rddlgn_names = [n for n in args.rddlgn_models.split(',') if n]
        unknown = [n for n in rddlgn_names if n not in all_rddlgn]
        if unknown:
            print(f'未知のRDDLGNモデル: {unknown}'
                  f'(backend={args.backend}の候補: {sorted(all_rddlgn)})',
                  file=sys.stderr)
            return 1
    elif wanted is not None:
        # --models にRDDLGNのspecキーが混ざっていれば自動判定で拾う
        rddlgn_names = sorted(wanted & all_rddlgn)
    elif args.rddlgn or args.rddlgn_only:
        rddlgn_names = sorted(all_rddlgn)

    if not paths and not rddlgn_names:
        print('対象モデルが無い', file=sys.stderr)
        return 1

    dense_conv_rows: list = []
    rddlgn_c_results: list = []
    rddlgn_gpu_rows: list = []

    if args.backend == 'c':
        cc = find_cc()
        if cc is None:
            print('Cコンパイラが見つからない', file=sys.stderr)
            return 1

        root = OUT_DIR / 'build_bench'
        for path in paths:
            doc = json.loads(path.read_text(encoding='utf-8'))
            kind = detect_kind(doc)
            bundle = build_bundle(doc)
            variants = ([(False, ''), (True, '+fuse')] if args.fuse_compare
                        else [(args.fuse, '+fuse' if args.fuse else '')])
            for fuse, tag in variants:
                work = root / (path.stem + ('_fuse' if fuse else ''))
                if work.exists():
                    shutil.rmtree(work)
                app, has_image = build_one(work, bundle, cc, fuse=fuse)
                m = measure(app, args.reps, args.threads)
                sizes = delivery_sizes(work, bundle)
                secs = section_sizes(work / 'model.o')
                dense_conv_rows.append({
                    'model': path.stem + tag, 'fuse': fuse,
                    'kind': kind, 'backend': 'c',
                    'infer_ms': m.get('infer_ms', (0, 0)),
                    'e2e_ms': m.get('e2e_ms'),
                    'e2e_mt': m.get('e2e_mt', {}),
                    'thr': m.get('thr', {}),
                    'ctx_bytes': m.get('ctx_bytes', 0),
                    'weights_bytes': secs.get('const', 0),
                    'rss_kb': peak_rss_kb(work, cc),
                    **sizes,
                })
                print(f'  {path.stem + tag}: 計測OK', flush=True)

        if rddlgn_names:
            rddlgn = rddlgn_module
            rddlgn.BUILD_ROOT.mkdir(parents=True, exist_ok=True)
            if args.rddlgn_opt is not None:
                opt = args.rddlgn_opt
            elif 'paper_scale1' in rddlgn_names:
                model, cfg = rddlgn.MODEL_SPECS['paper_scale1']['loader']()
                doc = rddlgn.build_doc(model, cfg)
                bundle = build_bundle(doc)
                preflight = rddlgn.BUILD_ROOT / '_preflight_paper_scale1'
                if preflight.exists():
                    shutil.rmtree(preflight)
                rddlgn.emit_c_files(bundle, preflight)
                opt, _dt = rddlgn.decide_opt_level(cc, preflight, None)
                del model, doc, bundle
            else:
                opt = 2
            print(f'RDDLGN最適化レベル: -O{opt}')

            for name in rddlgn_names:
                spec = rddlgn.MODEL_SPECS[name]
                try:
                    r = rddlgn.rddlgn_bench_row(
                        name, spec, cc, opt,
                        max_thr=args.rddlgn_max_thr,
                        skip_verify=args.rddlgn_skip_verify,
                        optimize_degenerate=args.rddlgn_optimize_degenerate)
                except Exception as exc:  # noqa: BLE001 1件の失敗で全体を止めない
                    print(f'!!! rddlgn:{name} が失敗: {exc}', file=sys.stderr)
                    r = {'name': name, 'kind': 'rddlgn', 'backend': 'c',
                         'error': str(exc)}
                rddlgn_c_results.append(r)

            ok = [r for r in rddlgn_c_results if 'error' not in r]
            if ok:
                print_rddlgn_summary(ok)
                # 旧scripts/bench_rddlgn.pyの詳細診断(全表)。
                # 挙動互換のため、統一後もそのまま出す(計算済みの結果を
                # 整形するだけで再計測はしない)。
                rddlgn.print_markdown(ok)
            rddlgn_out = (Path(args.rddlgn_out) if args.rddlgn_out
                         else OUT_DIR / 'rddlgn_bench.json')
            rddlgn_out.parent.mkdir(parents=True, exist_ok=True)
            rddlgn_out.write_text(json.dumps(
                {'opt_level': opt, 'results': rddlgn_c_results,
                 'when': time.strftime('%Y-%m-%d %H:%M:%S')},
                ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'\nRDDLGN結果: {rddlgn_out}')

    else:
        # --- metal / cuda。既存GPUハーネスをimportして呼ぶ(subprocessは
        # 使わない)。元のCLI(bench_dense_gpu.py / bench_rddlgn_decode_gpu.py)
        # はそのまま残っており、挙動はここから呼んでも変わらない。 ---
        if paths:
            if args.backend != 'metal':
                print(f'--backend={args.backend} はdense/convモデル未対応'
                      '(現状metalのみ。scripts/bench_dense_gpu.py参照)',
                      file=sys.stderr)
                return 1
            dense_gpu = _bench_dense_gpu()
            if not dense_gpu.has_metal():
                print('macOS + Metal(clang++)のある環境が必要', file=sys.stderr)
                return 1
            clangxx = dense_gpu.find_clangxx()
            gpu_root = OUT_DIR / 'build_bench_gpu'
            for path in paths:
                doc = json.loads(path.read_text(encoding='utf-8'))
                kind = detect_kind(doc)
                bundle = build_bundle(doc)
                for batch in args.gpu_batch:
                    row = dense_gpu.measure_metal(
                        bundle, gpu_root / path.stem, batch=batch,
                        n_streams=args.gpu_n_streams,
                        dense_fuse=args.gpu_dense_fuse,
                        dense_packed=args.gpu_dense_packed,
                        sets=args.gpu_sets, reps=args.gpu_reps,
                        warmup=args.gpu_warmup, clangxx=clangxx)
                    row = {'model': path.stem, 'kind': kind,
                          'backend': 'metal', **row}
                    dense_conv_rows.append(row)
                    print(f"  {path.stem} batch={batch}: throughput="
                          f"{row['throughput_per_s_median']:,.0f}/s")

        if rddlgn_names:
            if args.backend == 'cuda' and not args.host:
                print('--backend cuda には --host が必要', file=sys.stderr)
                return 1

            gpu_result = None
            tf_result = None
            want_decode = args.rddlgn_gpu_metrics in ('both', 'decode')
            want_tf = args.rddlgn_gpu_metrics in ('both', 'tf')

            # --- 4指標のうち「生成(decode)tok/s・1トークンあたりms」。
            # scripts/bench_rddlgn_decode_gpu.pyへ委譲する。---
            if want_decode:
                decode_gpu = _bench_rddlgn_decode_gpu()
                decode_supported = [n for n in rddlgn_names
                                    if n in decode_gpu.MODEL_SPECS]
                decode_unsupported = [n for n in rddlgn_names
                                      if n not in decode_gpu.MODEL_SPECS]
                if decode_unsupported:
                    print(f'  (生成tok/sは bench_rddlgn_decode_gpu.MODEL_SPECS'
                          f'に無いモデルのためスキップ: {decode_unsupported})',
                          file=sys.stderr)
                if decode_supported:
                    gpu_args = types.SimpleNamespace(
                        backend=args.backend, models=','.join(decode_supported),
                        host=args.host, remote_dir=args.remote_dir,
                        arch=args.arch, nvcc_path=args.nvcc_path,
                        seed=args.rddlgn_seed, reps=args.rddlgn_reps,
                        lanes=args.rddlgn_lanes,
                        megakernel=args.rddlgn_megakernel, out=None)
                    cases = [decode_gpu.build_case(n, args.rddlgn_seed)
                            for n in decode_supported]
                    gpu_result = (decode_gpu.run_cuda(gpu_args, cases)
                                 if args.backend == 'cuda'
                                 else decode_gpu.run_metal(gpu_args, cases))
                    for name, res in gpu_result.get('results', {}).items():
                        rddlgn_gpu_rows.append({'model': name, 'kind': 'rddlgn',
                                                'backend': args.backend, **res})

            # --- 4指標のうち「系列一括処理のスループット」(教師強制、生成では
            # ない)。scripts/bench_rddlgn_gpu.pyへ委譲する(decodeとは
            # 別のMODEL_SPECS辞書を使うので、対応外のモデルはスキップして
            # 理由を出す)。 ---
            if want_tf:
                tf_gpu = _bench_rddlgn_gpu()
                tf_supported = [n for n in rddlgn_names
                                if n in tf_gpu.MODEL_SPECS]
                tf_unsupported = [n for n in rddlgn_names
                                  if n not in tf_gpu.MODEL_SPECS]
                if tf_unsupported:
                    print(f'  (教師強制スループットは bench_rddlgn_gpu.'
                          f'MODEL_SPECSに無いモデルのためスキップ: '
                          f'{tf_unsupported})', file=sys.stderr)
                if tf_supported:
                    tf_batches = [int(b)
                                 for b in args.rddlgn_tf_batches.split(',')]
                    tf_args = types.SimpleNamespace(
                        backend=args.backend, host=args.host,
                        remote_dir=args.rddlgn_tf_remote_dir, arch=args.arch,
                        nvcc_path=args.nvcc_path, reps=args.rddlgn_reps,
                        bench_batches=args.rddlgn_tf_batches)
                    tf_cases = [tf_gpu.build_case(n, args.rddlgn_seed,
                                                  tf_batches)
                               for n in tf_supported]
                    tf_result = (tf_gpu.run_cuda(tf_args, tf_cases)
                                if args.backend == 'cuda'
                                else tf_gpu.run_metal(tf_args, tf_cases))
                    tf_by_name = tf_result.get('results', {})
                    existing = {r['model'] for r in rddlgn_gpu_rows}
                    for name in tf_supported:
                        if name in existing:
                            continue
                        rddlgn_gpu_rows.append({'model': name, 'kind': 'rddlgn',
                                                'backend': args.backend})
                    for row in rddlgn_gpu_rows:
                        if row['model'] in tf_by_name:
                            row['teacher_forcing'] = tf_by_name[row['model']]

            if rddlgn_gpu_rows:
                print_rddlgn_gpu_summary(args.backend, rddlgn_gpu_rows)
            if args.rddlgn_out and (gpu_result is not None
                                    or tf_result is not None):
                rddlgn_out = Path(args.rddlgn_out)
                rddlgn_out.parent.mkdir(parents=True, exist_ok=True)
                combined = {}
                if gpu_result is not None:
                    combined['decode'] = gpu_result
                if tf_result is not None:
                    combined['teacher_forcing'] = tf_result
                # 単一指標のみ測った場合は旧スクリプトと同じ最小スキーマ
                # (挙動互換)で書く。両方測った場合のみ{'decode':...,
                # 'teacher_forcing':...}のネスト形にする。
                payload = (gpu_result if (gpu_result is not None
                                          and tf_result is None)
                          else tf_result if (tf_result is not None
                                             and gpu_result is None)
                          else combined)
                rddlgn_out.write_text(json.dumps(
                    payload, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'\nRDDLGN GPU({args.backend})結果: {rddlgn_out}')

    c_backend_dc_rows = [r for r in dense_conv_rows if r.get('backend') == 'c']
    if c_backend_dc_rows:
        print('\n## レイテンシとスループット(Apple系, -O2)\n')
        print('| モデル | 推論 ms | e2e ms | e2e ms(前処理を4コアへ) | '
              '枚/秒 1ctx | 枚/秒 4ctx | 枚/秒 8ctx |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        for r in c_backend_dc_rows:
            e2e = f"{r['e2e_ms'][0]:.3f}" if r['e2e_ms'] else '—'
            mt4 = f"{r['e2e_mt'].get(4, 0):.3f}" if r['e2e_mt'] else '—'
            print(f"| {r['model']} | {r['infer_ms'][0]:.3f} | {e2e} | {mt4} | "
                  f"{r['thr'].get(1, 0):,.0f} | {r['thr'].get(4, 0):,.0f} | "
                  f"{r['thr'].get(8, 0):,.0f} |")

        print('\n## 配信サイズとメモリ\n')
        print('| モデル | バンドル gz | 生成C | バイナリ | バイナリ gz | '
              '重み(.rodata) | 作業RAM | ピークRSS |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in c_backend_dc_rows:
            print(f"| {r['model']} | {kb(r['bundle_gz'])} | {kb(r['src'])} | "
                  f"{kb(r['bin'])} | {kb(r['bin_gz'])} | "
                  f"{kb(r['weights_bytes'])} | {kb(r['ctx_bytes'])} | "
                  f"{r['rss_kb']:,}KB |")

    gpu_dc_rows = [r for r in dense_conv_rows if r.get('backend') != 'c']
    if gpu_dc_rows:
        print(f'\n## dense/conv GPU({args.backend})スループット\n')
        print('| モデル | batch | n_streams | throughput/s(中央値) |')
        print('|---|---:|---:|---:|')
        for r in gpu_dc_rows:
            print(f"| {r['model']} | {r['batch']} | {r['n_streams']} | "
                  f"{r['throughput_per_s_median']:,.0f} |")

    rows = dense_conv_rows + rddlgn_c_results + rddlgn_gpu_rows
    out = Path(args.out) if args.out else OUT_DIR / 'deploy_bench.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'rows': rows, 'when': time.strftime('%Y-%m-%d %H:%M:%S')},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n結果: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
