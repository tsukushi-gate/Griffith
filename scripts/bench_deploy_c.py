#!/usr/bin/env python
"""Cランタイムのスループット計測(K6)。

各モデルについて生成CをビルドしO2で実行し、
    infer     符号化済みビット → スコア(推論のみ)
    e2e       生画像 → 前処理 → 推論(mdl_infer_image)
を計測する。Web 実測(wasm / webgpu の推論のみ inferMs)を同じ表に
並べて出す。**別マシン・別ランタイムの数字なので厳密な比較ではない**
(Webの数字はブラウザのウォームページ2周目中央値)。目的は桁の把握。

使い方:
    uv run python scripts/bench_deploy_c.py [--models m1,m2] [--iters 50]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.preproc_desc import encode_bits  # noqa: E402
from griffith_logic.deploy.verify import (  # noqa: E402
    RUNTIME_C,
    find_cc,
    generate_c,
)

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'
W19_UNIFIED = (REPO / 'tools' / 'webbench' / 'bench-results'
               / '2026-08-04_w19v2_unified.json')

# webbench のデモ名 → 配信モデル(各 *_main.ts の DLGN_URL と同じ対応)
TASK_MODEL = {
    'det': 'det_multimnist_lut_k256',
    'detmono': 'det_multimnist_mono_lut_k256',
    'detface': 'det_face_lut_k256',
    'detreal': 'det_fewclass_lut_k64',
    'seg': 'seg_convdlgn_k32',
}

BENCH_DRIVER = '''#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint8_t bits[MDL_INPUT_WIDTH];
static int32_t scores[MDL_OUTPUT_WIDTH];
#ifdef MDL_IMAGE_CHANNELS
static uint8_t image[MDL_IMAGE_CHANNELS * MDL_INPUT_HEIGHT
                     * MDL_INPUT_WIDTH_PX];
#endif

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20;
  int i;
  double t0;
  if (fread(bits, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH)
    return 1;
#ifdef MDL_IMAGE_CHANNELS
  if (fread(image, 1, sizeof(image), stdin) != sizeof(image)) return 1;
#endif
  /* ウォームアップ(static バッファの初回ページフォールトを計測から外す) */
  for (i = 0; i < 20; i++) mdl_infer(bits, scores);
  for (i = 0; i < iters; i++) {
    t0 = now_ms();
    mdl_infer(bits, scores);
    printf("infer %.6f\\n", now_ms() - t0);
  }
#ifdef MDL_IMAGE_CHANNELS
  for (i = 0; i < 20; i++) mdl_infer_image(image, scores);
  for (i = 0; i < iters; i++) {
    t0 = now_ms();
    mdl_infer_image(image, scores);
    printf("e2e %.6f\\n", now_ms() - t0);
  }
#endif
  return 0;
}
'''


def w19_reference() -> dict:
    """統一ベンチから wasm/webgpu の推論のみ中央値(ms)を取り出す。"""
    if not W19_UNIFIED.exists():
        return {}
    doc = json.loads(W19_UNIFIED.read_text(encoding='utf-8'))
    out: dict[str, dict[str, float]] = {}
    for config in doc.get('configs', []):
        if config.get('id') not in ('wasm_default', 'webgpu_default'):
            continue
        key = 'wasm' if config['id'] == 'wasm_default' else 'webgpu'
        for task, cell in (config.get('cells') or {}).items():
            passes = cell.get('passes') or []
            values = [p['median'] for p in passes if p.get('median') is not None]
            if values:
                out.setdefault(task, {})[key] = round(min(values), 3)
    return out


def bench_model(name: str, path: Path, cc: str, iters: int,
                work_root: Path) -> dict | None:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if 'ir' not in doc:
        return None
    bundle = build_bundle(doc)
    files = generate_c(bundle, name='model', validate=False)
    work = work_root / name
    work.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (work / filename).write_text(text, encoding='utf-8')
    (work / 'bench.c').write_text(BENCH_DRIVER, encoding='utf-8')

    build = subprocess.run(
        [cc, '-std=c99', '-O2', f'-I{work}', f'-I{RUNTIME_C / "include"}',
         str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
         str(work / 'bench.c'), '-o', str(work / 'bench')],
        capture_output=True, text=True)
    if build.returncode != 0:
        print(f'  {name}: ビルド失敗\n{build.stderr[:800]}')
        return None

    plan = bundle.get('preproc_desc')
    rng = np.random.default_rng(0)
    if plan is not None:
        image = rng.integers(
            0, 256,
            size=(plan['image_channels'], plan['height'], plan['width']),
            dtype=np.uint8)
        bits = encode_bits(plan, image[None])[0]
        payload = bits.tobytes() + image.tobytes()
    else:
        in_width = int(bundle['ir']['inputs'][0]['width'])
        bits = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        payload = bits.tobytes()

    started = time.time()
    proc = subprocess.run([str(work / 'bench'), str(iters)],
                          input=payload, capture_output=True)
    if proc.returncode != 0:
        print(f'  {name}: 実行失敗 {proc.stderr.decode(errors="replace")[:300]}')
        return None

    samples: dict[str, list[float]] = {}
    for line in proc.stdout.decode().splitlines():
        kind, value = line.split()
        samples.setdefault(kind, []).append(float(value))

    result = {
        'model': name,
        'source_bytes': sum(len(t) for t in files.values()),
        'input_width': int(bundle['ir']['inputs'][0]['width']),
        'output_width': int(bundle['ir']['outputs'][0]['width']),
        'wall_s': round(time.time() - started, 2),
    }
    for kind, values in samples.items():
        result[f'{kind}_ms'] = round(statistics.median(values), 4)
        result[f'{kind}_min_ms'] = round(min(values), 4)
    programs = bundle.get('programs') or {}
    if programs:
        result['gates_before'] = sum(p['stats']['gates_before']
                                     for p in programs.values())
        result['gates_after'] = sum(p['stats']['gates_after']
                                    for p in programs.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=None,
                        help='対象モデル名のカンマ区切り(既定: 全部)')
    parser.add_argument('--iters', type=int, default=20, help='計測回数')
    parser.add_argument('--out', default=None, help='結果JSONの出力先')
    args = parser.parse_args()

    cc = find_cc()
    if cc is None:
        print('Cコンパイラが見つからない', file=sys.stderr)
        return 1
    if not MODELS_DIR.is_dir():
        print(f'モデルディレクトリが無い: {MODELS_DIR}', file=sys.stderr)
        return 1

    wanted = set(args.models.split(',')) if args.models else None
    paths = [p for p in sorted(MODELS_DIR.glob('*.json'))
             if not p.stem.endswith(('_dc', '_ll'))
             and (wanted is None or p.stem in wanted)]

    work_root = REPO / 'results' / 'deploy' / 'build'
    rows = []
    print(f'Cランタイム計測(cc={cc}, iters={args.iters})')
    for path in paths:
        row = bench_model(path.stem, path, cc, args.iters, work_root)
        if row is not None:
            rows.append(row)
            print(f"  {row['model']:38s} infer={row.get('infer_ms', '-')}ms "
                  f"e2e={row.get('e2e_ms', '-')}ms")

    ref = w19_reference()
    model_to_task = {v: k for k, v in TASK_MODEL.items()}

    print('\n## C ランタイム vs Web実測(推論のみ、ms/フレーム)\n')
    print('| モデル | C infer | C e2e | Web wasm | Web webgpu | ゲート | Cソース |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        task = model_to_task.get(row['model'])
        web = ref.get(task, {}) if task else {}
        gates = (f"{row['gates_after']:,}" if 'gates_after' in row else '-')
        print(f"| {row['model']} | {row.get('infer_ms', '-')} "
              f"| {row.get('e2e_ms', '-')} "
              f"| {web.get('wasm', '-')} | {web.get('webgpu', '-')} "
              f"| {gates} | {row['source_bytes'] // 1024}KB |")
    print('\n注: Webの数字は別マシン・ブラウザ上のウォームページ2周目中央値'
          '(tools/webbench/bench-results/2026-08-04_w19v2_unified.json)。'
          '同一条件の比較ではなく桁の把握用。')

    out_path = Path(args.out) if args.out else (
        REPO / 'results' / 'deploy' / 'c_runtime_bench.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {'iters': args.iters, 'cc': cc, 'rows': rows, 'w19_reference': ref},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n結果: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
