#!/usr/bin/env python
"""dense系モデル(thermometer等のビットベクタ入力)向けGPUスループット
ベンチハーネス(恒久版)。

既存の `scripts/bench_conv_split_metal.py` / `scripts/bench_conv_codegen_metal.py`
は画像入力モデル専用(`preproc.kind` が `image_thermometer`/`image_lowlevel`
以外だと `画像入力以外のモデルは未対応` でSystemExitする)。har_dlgn系
(HAR、`preproc.kind='thermometer'`、前処理済みビット列を直接IR入力へ渡す)
はこの経路に乗らないため、(`--n-streams`、ストリーム・パイプライン化)/
(`--dense-fuse`、dense多層融合カーネル)のスループットをA/Bする恒久
ハーネスが無かった(過去の計測はいずれもscratchpad限定の使い捨てベンチ)。

このスクリプトは非画像入力(ビット列を`mdl_metal_infer_batch`/
`mdl_metal_infer_batch_submit`+`_collect`へ直接注入)専用のドライバを
生成してビルド・実行する。`griffith deploy codegen-metal` 相当のソース生成は
`griffith_logic.deploy.metalbackend.generate_metal` をPython APIとして
直接呼ぶ(`griffith_logic/deploy/cli.py::_cmd_codegen_metal` と同じ流儀)。
ビルドコマンド・ドライバの型は `griffith_logic/deploy/metalverify.py`
(整数一致ガード)と `tests/deploy/test_metal_stream.py`(ストリームAPIの
ドライバ雛形)を踏襲している。

使い方:
    # 素の計測(batch∈{1,32,128,256}をこの1回の呼び出しで掃引)
    uv run python scripts/bench_dense_gpu.py \\
        --models har_dlgn_k600_gumbel --batch 1 32 128 256

    # 最良構成の再現(n_streams=4、chunk=128)
    uv run python scripts/bench_dense_gpu.py \\
        --models har_dlgn_k600_gumbel --n-streams 4 --batch 128

    # 最良構成の再現(dense_fuse、batch=128)
    uv run python scripts/bench_dense_gpu.py \\
        --models har_dlgn_k6000_gumbel --dense-fuse --batch 128

    #(lut_flat、融合版 / 段ごと版)
    uv run python scripts/bench_dense_gpu.py \\
        --models har_dlgn_k6000_gumbel --dense-flat fused --batch 1 32 128 256

    #(dense_packed、batch=32/128/256をこの1回の呼び出しで掃引)
    uv run python scripts/bench_dense_gpu.py \\
        --models har_dlgn_k6000_gumbel --dense-packed --batch 32 128 256

    # ハーネス自身のスモーク(ir/simulateとの整数厳密一致を1回だけ確認)
    uv run python scripts/bench_dense_gpu.py --models har_dlgn_k600_gumbel --verify

    # 結果をJSONへ(既存ファイルがあれば model+config キーでマージする)
    uv run python scripts/bench_dense_gpu.py --models har_dlgn_k600_gumbel \\
        --batch 1 32 128 256 --out results/deploy/local_baseline_20260911_metal_dense.json

将来cuda backendを足すときは `--backend` の分岐(`_BACKEND_RUNNERS`)へ
追加する(今回はmetalのみ実装)。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.metalbackend import generate_metal  # noqa: E402
from griffith_logic.deploy.metalverify import (  # noqa: E402
    has_metal,
    verify_metal_matches_simulate,
)

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
RUNTIME_METAL = REPO / 'runtime' / 'metal'
OUT_DIR = REPO / 'results' / 'deploy'

DEFAULT_MODELS = [
    'har_dlgn_k600_gumbel', 'har_dlgn_k2400_gumbel',
    'har_dlgn_k6000_gumbel', 'har_dlgn_dwn',
]

# 低電力モード等、計測結果を読むときに必要な前提条件(K0-Rの既存ベースライン
# results/deploy/local_baseline_20260911_metal.json と同じ形にしてある)。
PRECONDITIONS_NOTE = (
    'pmset -g lowpowermode=1 (low power mode) だった。絶対値は下振れしうる '
    '(電源状態遷移でGPUが「速い状態」「遅い状態」の二峰を取ることが'
    '既知)。'
)

# ---- ドライバ(非画像=ビット列入力専用。tests/deploy/test_metal_stream.py の
#      雛形を計測用途へ簡略化したもの) ----------------------------------

_BARE_DRIVER_TMPL = '''#include "mdl_metal.h"
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
  const int in_w = mdl_metal_input_width();
  const int out_w = mdl_metal_output_width();
  unsigned char *bits = (unsigned char *)malloc((size_t)batch * (size_t)in_w);
  int *scores = (int *)malloc((size_t)batch * (size_t)out_w * sizeof(int));
  double *t = (double *)malloc((size_t)reps * sizeof(double));
  size_t k;
  int r;

  for (k = 0; k < (size_t)batch * (size_t)in_w; k++) {{
    bits[k] = (unsigned char)((k * 2654435761u) & 1);
  }}
  if (mdl_model_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  /* ウォームアップ(シェーダのJIT・パイプライン生成コストを外す)。 */
  for (r = 0; r < warmup; r++) {{
    if (mdl_metal_infer_batch(bits, scores, batch) != 0) {{
      fprintf(stderr, "warmup infer failed\\n");
      return 2;
    }}
  }}

  for (r = 0; r < reps; r++) {{
    const double t0 = now_ms();
    if (mdl_metal_infer_batch(bits, scores, batch) != 0) {{
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

# ストリーム・パイプライン化版。「chunk枚ずつNSLOTスロットへ先行投入して
# 後ろから回収する」定常パイプラインを reps 回ぶん回し、その総壁時計時間から
# スループットを出す(開発記録記述「1000枚換算」と同じ考え方 —
# 個々のchunkのレイテンシ内訳ではなく定常状態のスループットが指標)。
_STREAM_DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {{
int mdl_metal_infer_batch_submit(const unsigned char*, int, int);
int mdl_metal_infer_batch_collect(int*, int, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
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

/* 定常パイプライン: NSLOT本先行してsubmitし、以降は1個collectするたびに
 * 1個submitする(tests/deploy/test_metal_stream.pyと同じ
 * submit/collect交互パターン)。collectせずに同一スロットを二重submitしない
 * ことがmdl_metal_commit_asyncのガード対象なので、この順序を守る。*/
static void run_n(int n) {{
  int pending = 0, next_submit = 0, next_collect = 0;
  while (next_collect < n) {{
    while (next_submit < n && pending < NSLOT) {{
      const int slot = next_submit % NSLOT;
      if (mdl_metal_infer_batch_submit(g_bits, g_batch, slot) != 0) {{
        fprintf(stderr, "submit failed at k=%d\\n", next_submit);
        exit(2);
      }}
      next_submit++;
      pending++;
    }}
    {{
      const int slot = next_collect % NSLOT;
      if (mdl_metal_infer_batch_collect(g_scores, g_batch, slot) != 0) {{
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
  const int in_w = mdl_metal_input_width();
  const int out_w = mdl_metal_output_width();
  size_t k;
  g_batch = {batch};
  g_bits = (unsigned char *)malloc((size_t)g_batch * (size_t)in_w);
  g_scores = (int *)malloc((size_t)g_batch * (size_t)out_w * sizeof(int));
  for (k = 0; k < (size_t)g_batch * (size_t)in_w; k++) {{
    g_bits[k] = (unsigned char)((k * 2654435761u) & 1);
  }}
  if (mdl_model_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

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


def find_clangxx() -> str:
    path = shutil.which('clang++')
    if path is None:
        raise SystemExit('clang++ が見つからない(macOS + Xcode CLTが必要)')
    return path


def resolve_model_path(spec: str) -> Path:
    """バンドルJSONへの直接パス、または `tools/webbench/public/models/` の
    モデル名(拡張子無し)を解決する。"""
    p = Path(spec)
    if p.is_file():
        return p
    cand = MODELS_DIR / f'{spec}.json'
    if cand.is_file():
        return cand
    raise SystemExit(f'モデルが見つからない: {spec!r}(パスにもモデル名にも該当しない)')


def load_model_bundle(spec: str) -> dict:
    """`bench_deploy.py`/`bench_conv_split_metal.py` と同じ流儀:
    生JSONを読んで `build_bundle` へ通す(既にバンドル済みのファイルでも
    `ir` からprograms等を再構築するだけなので安全に冪等)。"""
    path = resolve_model_path(spec)
    doc = json.loads(path.read_text(encoding='utf-8'))
    return build_bundle(doc)


def _build_metal(bundle: dict, work_dir: Path, *, batch: int, n_streams: int,
                 dense_fuse: bool, dense_packed: bool, clangxx: str,
                 lut_flat_mode: str = 'auto') -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in generate_metal(
            bundle, name='model_metal', batch=batch, n_streams=n_streams,
            dense_fuse=dense_fuse, dense_packed=dense_packed,
            lut_flat_mode=lut_flat_mode).items():
        (work_dir / filename).write_text(text, encoding='utf-8')

    if n_streams > 1:
        driver = _STREAM_DRIVER_TMPL.format(
            n_streams=n_streams, batch=batch, reps=30, warmup=8)
    else:
        driver = _BARE_DRIVER_TMPL.format(batch=batch, reps=30, warmup=8)
    (work_dir / 'driver.mm').write_text(driver, encoding='utf-8')

    exe = work_dir / 'app'
    build = subprocess.run(
        [clangxx, '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{RUNTIME_METAL}', '-framework', 'Metal', '-framework', 'Foundation',
         str(work_dir / 'model_metal.mm'), str(RUNTIME_METAL / 'mdl_metal.mm'),
         str(work_dir / 'driver.mm'), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit(f'ビルド失敗(batch={batch}, n_streams={n_streams}, '
                         f'dense_fuse={dense_fuse}, '
                         f'dense_packed={dense_packed}):\n{build.stderr}')
    return exe


def _run_set(exe: Path, *, reps: int, warmup: int) -> dict:
    proc = subprocess.run([str(exe), str(reps), str(warmup)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'実行失敗(exit={proc.returncode}):\n{proc.stderr}')
    out: dict = {}
    for line in proc.stdout.splitlines():
        m = re.match(r'(\w+) ([-\d.]+)', line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        out[key] = float(val) if ('.' in val or 'e' in val.lower()) else int(val)
    return out


def measure_metal(bundle: dict, work_dir: Path, *, batch: int, n_streams: int,
                  dense_fuse: bool, dense_packed: bool, sets: int, reps: int,
                  warmup: int, clangxx: str, dense_flat: str = 'off') -> dict:
    """1つの(model, batch, n_streams, dense_fuse, dense_packed)組を
    setsセット計測する。"""
    exe = _build_metal(bundle, work_dir, batch=batch, n_streams=n_streams,
                       dense_fuse=dense_fuse, dense_packed=dense_packed,
                       clangxx=clangxx,
                       lut_flat_mode='auto' if dense_flat == 'off' else dense_flat)
    set_results = [_run_set(exe, reps=reps, warmup=warmup) for _ in range(sets)]

    if n_streams > 1:
        thr_key = 'throughput_per_s'
    else:
        thr_key = 'throughput_median_per_s'
    thr_values = sorted(r[thr_key] for r in set_results)
    median_thr = thr_values[len(thr_values) // 2]

    row = {
        'batch': batch,
        'n_streams': n_streams,
        'dense_fuse': dense_fuse,
        'dense_packed': dense_packed,
        'dense_flat': dense_flat,
        'reps': reps,
        'warmup': warmup,
        'sets': sets,
        'set_results': set_results,
        'throughput_per_s_sets': thr_values,
        'throughput_per_s_median': median_thr,
    }
    if n_streams <= 1:
        e2e_meds = sorted(r['e2e_ms_median'] for r in set_results)
        row['e2e_ms_median_of_sets'] = e2e_meds[len(e2e_meds) // 2]
    else:
        e2e_meds = sorted(r['e2e_ms_per_chunk_mean'] for r in set_results)
        row['e2e_ms_per_chunk_median_of_sets'] = e2e_meds[len(e2e_meds) // 2]
    return row


_BACKEND_RUNNERS = {
    'metal': measure_metal,
}


def config_key(row: dict) -> str:
    mode = 'bare'
    if row['n_streams'] > 1:
        mode = f"n_streams{row['n_streams']}"
    elif row['dense_fuse']:
        mode = 'dense_fuse'
    elif row.get('dense_packed'):
        mode = 'dense_packed'
    elif row.get('dense_flat', 'off') != 'off':
        mode = f"dense_flat_{row['dense_flat']}"
    return f"{mode}_batch{row['batch']}"


def merge_rows(out_path: Path, model: str, new_rows: list[dict],
               meta: dict) -> dict:
    """既存JSON(あれば)へ `model + config_key` をキーにマージする(同じ組を
    再計測したら上書き、それ以外は温存)。"""
    doc: dict = {}
    if out_path.is_file():
        try:
            doc = json.loads(out_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            doc = {}
    doc.setdefault('rows', {})
    doc['meta'] = meta
    for row in new_rows:
        key = f'{model}::{config_key(row)}'
        doc['rows'][key] = {'model': model, **row}
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                    help='バンドルJSONパス、または models/ 配下のモデル名')
    ap.add_argument('--backend', default='metal', choices=list(_BACKEND_RUNNERS),
                    help='計測バックエンド(現状metalのみ実装)')
    ap.add_argument('--batch', type=int, nargs='+', default=[1],
                    help='1回の推論呼び出しあたりの枚数(複数指定で掃引、'
                         'n_streams>1時はチャンクサイズとして使う)')
    ap.add_argument('--n-streams', type=int, default=1,
                    help='ストリーム・パイプライン化のスロット数(既定1=無効)')
    ap.add_argument('--dense-fuse', action='store_true',
                    help='dense多層融合カーネルを使う')
    ap.add_argument('--dense-packed', action='store_true',
                    help='denseのバッチ軸ビットプレーン化カーネルを使う'
                         '(batch>=32必須。--dense-fuseとの併用不可)')
    ap.add_argument('--dense-flat', choices=['off', 'fused', 'levels', 'packed'],
                    default='off',
                    help='dense を lut_flat(縮約済み平坦ネットリスト)へ'
                         '平坦化してから生成する(fused=1スレッドグループ=1枚の'
                         '単一起動、levels=段ごとに1起動)')
    ap.add_argument('--sets', type=int, default=5, help='セット数(既定5、中央値を取る)')
    ap.add_argument('--reps', type=int, default=30, help='1セット内の反復回数')
    ap.add_argument('--warmup', type=int, default=8, help='ウォームアップ回数')
    ap.add_argument('--out', default=None, help='結果JSONの出力先(既存があればマージ)')
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--verify', action='store_true',
                    help='計測せず、batch=1でMetal実行 ⇔ ir/simulate の整数厳密'
                         '一致だけを確認する(ハーネス自身のスモーク)')
    args = ap.parse_args()

    if args.backend != 'metal':
        raise SystemExit(f'--backend={args.backend} は未実装(現状metalのみ)')
    if not has_metal():
        raise SystemExit('macOS + Metal(clang++)のある環境が必要')
    clangxx = find_clangxx()

    if args.dense_packed and args.dense_fuse:
        raise SystemExit('--dense-packed と --dense-fuse は併用不可'
                         '(v1スコープ外、タスク指示)')

    if args.verify:
        if args.dense_packed:
            # verify_metal_matches_simulate はbatch=1固定でビルドするため
            # (`build_metal_model`が`batch`引数を明示的に渡していない)、
            # batch>=32必須のdense_packedとは組み合わせられない。
            # 整数一致は tests/deploy/test_metal_dense_packed.py が
            # batch∈{32,33,64,100,128}で既に固定している(こちらのスモークは
            # bare/dense_fuse専用のまま)。
            raise SystemExit(
                '--verify は --dense-packed 未対応(batch=1固定のため)。'
                'dense_packedの整数一致は '
                'uv run pytest tests/deploy/test_metal_dense_packed.py '
                '-m slow で確認すること')
        ok = True
        for name in args.models:
            bundle = load_model_bundle(name)
            print(f'=== verify: {name} ===')
            try:
                result = verify_metal_matches_simulate(
                    bundle, n_samples=1, dense_fuse=args.dense_fuse)
            except Exception as exc:  # noqa: BLE001 スモークなので何でも報告する
                ok = False
                print(f'  NG: {exc}')
                continue
            print(f"  OK: 経路={result['path']} 出力{result['n_out']}要素 × "
                  f"{result['n_samples']}サンプル(整数厳密)")
        return 0 if ok else 1

    import tempfile
    work_root = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix='mdl_bench_dense_'))

    out_path = Path(args.out) if args.out else OUT_DIR / 'bench_dense_gpu.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = _BACKEND_RUNNERS[args.backend]
    meta = {
        'when': time.strftime('%Y-%m-%d %H:%M:%S'),
        'backend': args.backend,
        'preconditions_note': PRECONDITIONS_NOTE,
    }

    for name in args.models:
        try:
            bundle = load_model_bundle(name)
        except SystemExit as exc:
            print(f'=== {name}: skip ({exc}) ===')
            continue
        if args.dense_flat != 'off':
            from griffith_logic.deploy.flatten_dense import (
                describe_dense_flat,
                resolve_dense_flat,
            )
            bundle, flat_dec = resolve_dense_flat(bundle, mode='on')
            print('  ' + describe_dense_flat(flat_dec))

        print(f'\n=== {name} ===')
        rows = []
        for batch in args.batch:
            print(f'  -- batch={batch} n_streams={args.n_streams} '
                  f'dense_fuse={args.dense_fuse} '
                  f'dense_packed={args.dense_packed} --')
            row = runner(
                bundle, work_root / name, batch=batch, n_streams=args.n_streams,
                dense_fuse=args.dense_fuse, dense_packed=args.dense_packed,
                sets=args.sets, reps=args.reps,
                warmup=args.warmup, clangxx=clangxx,
                dense_flat=args.dense_flat)
            rows.append(row)
            sets_str = ', '.join(f'{v:.0f}' for v in row['throughput_per_s_sets'])
            if args.n_streams > 1:
                print(f"    throughput median({args.sets}set)="
                      f"{row['throughput_per_s_median']:,.0f}/s  sets=[{sets_str}]")
            else:
                print(f"    throughput median({args.sets}set)="
                      f"{row['throughput_per_s_median']:,.0f}/s  "
                      f"e2e_ms median={row['e2e_ms_median_of_sets']:.4f}  "
                      f"sets=[{sets_str}]")

        doc = merge_rows(out_path, name, rows, meta)
        out_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  -> {out_path} へ保存(マージ)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
