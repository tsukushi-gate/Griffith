#!/usr/bin/env python
"""デプロイキットの実機・シミュレータ実測を wandb へ記録する。

`results/deploy/devices/` に残してある実測の集約 JSON(`esp32_summary_<日付>.json` /
`ios_summary_<日付>.json`)を読み、**モデル × プラットフォーム**の表と主要指標を作る。
`--log-wandb` を付けると次を 1 本の run に記録する(付けなければ表の表示だけ)。

    - 指標(wandb.log): `<platform>/<model>/mean_us` `acc` `score_mismatch` ほか
    - 表(wandb.Table): モデル × プラットフォームの全指標(`device_bench`)
    - 画像(log_images): iOS シミュレータのスクリーンショット
    - Artifact:
        deploykit-devices-results      集約 JSON・個別 JSON・UART/ホストのログ・idf.py size
        deploykit-devices-screenshots  スクリーンショット PNG
        deploykit-demo-models          out_examples/demo_models/summary.json と bundle 3 本

注意: iOS の時間は **シミュレータ = Apple Silicon ホスト CPU 上のネイティブ実行値**で、
実機(iPhone)の値ではない。表・指標名の `ios_sim` はそれを意味する(実機は `ios_device`)。

使い方:
    uv run python scripts/bench_deploy_devices.py                      # 表を表示するだけ
    uv run python scripts/bench_deploy_devices.py --log-wandb          # wandb へ記録
    uv run python scripts/bench_deploy_devices.py --date 20261006 --log-wandb
    uv run python scripts/bench_deploy_devices.py --date 20261007 --log-wandb  #

wandb の認証は環境変数 WANDB_API_KEY(または --env-file。既定はリポジトリ直下の .env)。
鍵の値は出力しない。run 名は init_wandb がタイムスタンプを付加する。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEVICES_DIR = REPO / 'results' / 'deploy' / 'devices'
DEMO_DIR = REPO / 'out_examples' / 'demo_models'
DEMO_MODELS = ('har_dlgn_k600', 'har_dlgn_k2400', 'har_dlgn_k6000', 'det_face_lut_k256',
               'det_face_lut_k256_resynth')
MNIST_BUNDLE = REPO / 'out_examples' / 'deploy_mnist' / 'mnist.bundle.json'

RUN_NAME_PREFIX = 'deploykit-devices'   # 日付を付けて deploykit-devices-<date> にする
GROUP = 'deploykit-devices'
TAGS = ['K70', 'K71', 'K72', 'K73', 'K74']

# 表に載せる列(順序固定)
TABLE_COLUMNS = [
    'platform', 'model', 'variant', 'status', 'gates', 'gates_exec', 'input_width', 'ctx_bytes',
    'n', 'acc',
    'score_checked', 'score_mismatch', 'mean_us', 'median_us', 'min_us', 'max_us',
    'app_bin_bytes', 'dram_used', 'flash_rodata', 'note',
]


def _load_env(env_file: Path) -> None:
    """`.env` の KEY=VALUE を未設定の環境変数へ読み込む(値は表示しない)。"""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _find_summary(prefix: str, date: str | None) -> Path:
    """`<prefix>_summary_<date>.json`。date 省略時は最新(ファイル名の辞書順)。"""
    if date:
        p = DEVICES_DIR / f'{prefix}_summary_{date}.json'
        if not p.exists():
            raise SystemExit(f'見つからない: {p}')
        return p
    cands = sorted(DEVICES_DIR.glob(f'{prefix}_summary_*.json'))
    if not cands:
        raise SystemExit(f'{prefix}_summary_*.json が {DEVICES_DIR} に無い')
    return cands[-1]


def collect_rows(esp32_path: Path, ios_path: Path) -> list[dict]:
    """ESP32 と iOS の集約 JSON から、モデル × プラットフォームの行を作る。"""
    rows: list[dict] = []
    esp = json.loads(esp32_path.read_text(encoding='utf-8'))
    for m in esp['models']:
        rows.append({
            'platform': 'esp32',
            'model': m['model'],
            'variant': m.get('variant', 'raw'),
            'status': m.get('status'),
            'gates': m.get('gates'),
            'gates_exec': m.get('gates_exec'),
            'input_width': m.get('input_width'),
            'ctx_bytes': m.get('ctx_bytes'),
            'n': m.get('n'),
            'acc': m.get('acc'),
            'score_checked': m.get('score_checked'),
            'score_mismatch': m.get('score_mismatch'),
            'mean_us': m.get('mean_us'),
            'median_us': m.get('median_us'),
            'min_us': m.get('min_us'),
            'max_us': m.get('max_us'),
            'app_bin_bytes': m.get('app_bin_bytes'),
            'dram_used': m.get('dram_used'),
            'flash_rodata': m.get('flash_rodata'),
            'note': m.get('note') or m.get('link_error'),
        })
    ios = json.loads(ios_path.read_text(encoding='utf-8'))
    for r in ios['results']:
        # 同じモデルが複数行(XOR の Debug / Release)ある。Release を優先して採用する。
        rows.append({
            'platform': r['platform'].replace('-', '_'),   # ios_simulator → 下で ios_sim へ
            'model': r['model'],
            'variant': r.get('variant', 'raw'),
            'status': 'ok',
            'gates': r.get('gates'),
            'gates_exec': r.get('gates'),   # iOS の gates は実行ゲート数
            'input_width': r.get('input_width'),
            'ctx_bytes': r.get('ctx_bytes'),
            'n': r.get('n'),
            'acc': r.get('acc'),
            'score_checked': r.get('score_total'),
            'score_mismatch': r.get('score_mismatch'),
            'mean_us': r.get('mean_us'),
            'median_us': r.get('median_us'),
            'min_us': r.get('min_us'),
            'max_us': r.get('max_us'),
            'app_bin_bytes': r.get('app_bytes'),
            'dram_used': None,
            'flash_rodata': None,
            'note': f"{r.get('device')} / {r.get('build')}(時間はホスト CPU 値)",
            '_build': r.get('build'),
        })
    for row in rows:
        if row['platform'] == 'ios_simulator':
            row['platform'] = 'ios_sim'
    return rows


def _metric_base(row: dict) -> str:
    """指標キーの接頭辞。ESP32 の sweep/flat は `<model>_<variant>`(iOS は名前に付与済み)。"""
    name = row['model']
    variant = row.get('variant', 'raw')
    if variant not in ('raw', 'resynth') and not name.endswith(f'_{variant}'):
        name = f'{name}_{variant}'
    return f"{row['platform']}/{name}"


def metric_dict(rows: list[dict]) -> dict:
    """wandb.log 用の指標(`<platform>/<model>[_<variant>]/<key>`)。

    同名の複数行(iOS の素の回路の2回の計測、XOR の Debug)は、Debug を除き後の行が勝つ。
    """
    out: dict = {}
    for row in rows:
        # XOR の iOS は Debug(前任者値)と Release の 2 行。Release のみ指標へ載せる。
        if row['platform'] == 'ios_sim' and str(row.get('_build', '')).startswith('Debug'):
            continue
        base = _metric_base(row)
        for key in ('mean_us', 'median_us', 'min_us', 'max_us', 'acc', 'score_mismatch',
                    'score_checked', 'ctx_bytes', 'gates', 'gates_exec', 'app_bin_bytes',
                    'dram_used', 'flash_rodata'):
            v = row.get(key)
            if isinstance(v, int | float):
                out[f'{base}/{key}'] = v
        out[f'{base}/fits'] = 1 if row.get('status') == 'ok' else 0
    return out


def print_table(rows: list[dict]) -> None:
    """表を標準出力へ(時間は µs)。"""
    print(f"{'platform':8s} {'model':26s} {'variant':8s} {'status':8s} {'gates':>7s} {'exec':>7s} "
          f"{'n':>3s} {'acc':>6s} {'mism':>4s} {'mean_us':>12s}")
    for r in rows:
        def f(v, fmt):
            return '-' if v is None else format(v, fmt)
        print(f"{r['platform']:8s} {r['model']:26s} {str(r.get('variant')):8s} "
              f"{str(r['status']):8s} "
              f"{f(r['gates'], '7d'):>7s} {f(r.get('gates_exec'), '7d'):>7s} "
              f"{f(r['n'], '3d'):>3s} {f(r['acc'], '.3f'):>6s} "
              f"{f(r['score_mismatch'], '4d'):>4s} {f(r['mean_us'], '12.3f'):>12s}")


def _device_files(date: str | None) -> dict[str, list[Path]]:
    """Artifact へ載せるファイル群(results / screenshots / demo_models)。"""
    allf = sorted(p for p in DEVICES_DIR.iterdir() if p.is_file())
    if date:
        allf = [p for p in allf if date in p.name]
    results = [p for p in allf if p.suffix != '.png']
    shots = [p for p in allf if p.suffix == '.png']
    # summary.json と、demo_models 直下の全バンドル(raw / .sweep / .flat / resynth)+ MNIST
    demo = [DEMO_DIR / 'summary.json'] + sorted(DEMO_DIR.glob('*.bundle.json'))
    if MNIST_BUNDLE.exists():
        demo.append(MNIST_BUNDLE)
    missing = [p for p in demo if not p.exists()] + [
        DEMO_DIR / f'{m}.bundle.json' for m in DEMO_MODELS
        if not (DEMO_DIR / f'{m}.bundle.json').exists()]
    if missing:
        raise SystemExit('デモモデルの成果物が無い(scripts/fetch_demo_models.py で作る): '
                         + ', '.join(str(p.relative_to(REPO)) for p in missing))
    return {'results': results, 'screenshots': shots, 'demo': demo}


def log_to_wandb(rows: list[dict], files: dict[str, list[Path]], esp32_path: Path,
                 ios_path: Path, date_tag: str) -> None:
    """1 本の run に指標・表・画像・Artifact を記録する。"""
    import wandb
    from PIL import Image

    from griffith_logic.training.wandb_utils import (
        init_wandb,
        log_artifact,
        log_images,
    )

    esp = json.loads(esp32_path.read_text(encoding='utf-8'))
    ios = json.loads(ios_path.read_text(encoding='utf-8'))
    config = {
        'date': esp.get('date'),
        'esp32/board': esp.get('board'),
        'esp32/idf': esp.get('idf'),
        'esp32/optimization': esp.get('optimization'),
        'esp32/timing': esp.get('timing'),
        'ios/platform_note': ios.get('platform_note'),
        'ios/method': ios.get('method'),
        'ios/app_size': ios.get('app_size'),
        'ios/app_size_15models': ios.get('app_size_15models'),
        'esp32/variants': esp.get('variants'),
        'esp32/sweep_bundle_stats': esp.get('sweep_bundle_stats'),
        'models': list(DEMO_MODELS),
    }
    ok = init_wandb(
        run_name=f"{RUN_NAME_PREFIX}-{date_tag}", config_dict=config, tags=TAGS,
        group=GROUP, job_type='bench',
        notes='デプロイキットの実機(ESP32)・シミュレータ(iOS、ホスト CPU 値)実測の集約記録',
    )
    if not ok:
        raise SystemExit('wandb が無効(WANDB_API_KEY 未設定または wandb 未インストール)')

    # 指標
    metrics = metric_dict(rows)
    # iOS の 12 回中央値(raw / sweep / flat の比較。ホスト CPU 値)
    for model, c in (ios.get('comparisons_flat_12runs') or {}).items():
        for key, v in c.items():
            metrics[f'ios_sim_12runs/{model}/{key}'] = v
    wandb.log(metrics)
    for k, v in metrics.items():
        wandb.run.summary[k] = v

    # 表(モデル × プラットフォーム)
    table = wandb.Table(columns=TABLE_COLUMNS)
    for r in rows:
        table.add_data(*[r.get(c) for c in TABLE_COLUMNS])
    wandb.log({'device_bench': table})

    # スクリーンショット(iOS シミュレータ)
    pngs = files['screenshots']
    log_images('ios_sim/screenshots',
               [Image.open(p) for p in pngs],
               captions=[p.stem for p in pngs])

    common = {'date': esp.get('date')}  # コミットは init_wandb が run の config に残す
    log_artifact([str(p) for p in files['results']], 'deploykit-devices-results',
                 type='benchmark', aliases=['latest'], metadata=common)
    log_artifact([str(p) for p in pngs], 'deploykit-devices-screenshots',
                 type='images', aliases=['latest'],
                 metadata={**common, 'note': 'iOS シミュレータ(時間はホスト CPU 値)'})
    log_artifact([str(p) for p in files['demo']], 'deploykit-demo-models',
                 type='model', aliases=['latest'],
                 metadata={**common, 'models': list(DEMO_MODELS)})

    run_id = wandb.run.id
    url = wandb.run.get_url()
    wandb.finish()
    print(f'WANDB_RUN_ID={run_id}')
    print(f'WANDB_RUN_URL={url}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--date', help='集約 JSON の日付(例 20261006。省略時は最新)')
    ap.add_argument('--log-wandb', action='store_true', help='wandb へ記録する')
    ap.add_argument('--env-file', default=str(REPO / '.env'),
                    help='WANDB_API_KEY を含む .env(既定: リポジトリ直下の .env)')
    args = ap.parse_args()

    esp32_path = _find_summary('esp32', args.date)
    ios_path = _find_summary('ios', args.date)
    rows = collect_rows(esp32_path, ios_path)
    print(f'ESP32: {esp32_path.relative_to(REPO)}\niOS  : {ios_path.relative_to(REPO)}')
    print_table(rows)

    if args.log_wandb:
        _load_env(Path(args.env_file))
        files = _device_files(args.date)
        date_tag = esp32_path.stem.rsplit('_', 1)[-1]
        log_to_wandb(rows, files, esp32_path, ios_path, date_tag)


if __name__ == '__main__':
    main()
