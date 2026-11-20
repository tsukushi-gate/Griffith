"""デプロイキットのデモ用モデル(行動認識 UCI-HAR / 顔検出 WIDER FACE)を取得して配布物まで作る。

wandb Artifacts に残っている学習済みモデルを**読み取り専用**で取得し、
デプロイキット(`griffith deploy ...`)の各段を通して、バンドル・C ソース・サイズ指標を
`out_examples/demo_models/` に再現可能な形で書き出す。新しい wandb run の作成や
アップロードは一切しない(`wandb.Api().artifact(...).download()` のみ)。

対象モデル(取得元の wandb entity / project は `--wandb-entity` / `--wandb-project` または
環境変数 WANDB_ENTITY / WANDB_PROJECT で指定する。既定値は持たない。現状これらの学習済み
モデルは非公開の wandb プロジェクトにのみあり、公開配布先は未定):

    har_dlgn_k6000   行動認識 UCI-HAR  36,000 ゲート(IR artifact uci-har-4t_k6000_l6_n2_e50000-ir)
    har_dlgn_k2400   行動認識 UCI-HAR  9,600 ゲートの中型版(同 k2400_l4_n2_e50000-ir、run g10xuhzw)
    har_dlgn_k600    行動認識 UCI-HAR  2,400 ゲートの小型版(同 k600_l4_n2_e50000-ir)
    det_face_lut_k256  顔検出 WIDER FACE CenterNetLUT(ckpt det_face_lut_k256_d3_L3_obj3-ckpt)

パイプライン(モデルごと、各段は出力が既にあればスキップ。`--force` で作り直し):

    1. wandb 取得     out_examples/demo_models/wandb_cache/<artifact>/ にキャッシュ
    2. モデル JSON    <name>.model.json  (webbench 形式 {ir, preproc, labels, readout, ...})
    3. バンドル <name>.bundle.json (= `griffith deploy bundle` 相当。: build_bundle の前に
                      lossless な回路圧縮 `--compress none|fast|max`(既定 fast = dense に sweep)を
                      CLI と同じ順で適用し、結果を bundle の about.compress に残す)
    4. verify-c       生成 C を cc でビルドして ir/simulate と整数厳密一致を確認
    5. codegen-c      c/<name>/ に model.c/model.h/ランタイム
    6. 指標           summary.json(ゲート数・各種サイズ・MDL_CTX_BYTES・入力幅・精度・ESP32 判定)

行動認識の前処理(IR 入力 4,608 ビット = 9ch × 128 step × 4 閾値)は次のとおり確定している
(wandb run 9gbzoj6i の学習時 test acc 0.89931 = 2608/2900 を ir/simulate で厳密再現して確認済み):

    - 入力は UCI-HAR の生 Inertial Signals 9ch×128step(561 特徴量ではない)
    - チャネル順: body_acc_{x,y,z}, body_gyro_{x,y,z}, total_acc_{x,y,z}
    - train の各チャネル mean/std で標準化 → チャネル別の等分位 thermometer
      (閾値 = 標準化後 train の分位点 (i+1)/(T+1), i=0..T-1、T=4)。判定は z > thr(strict)
    - ビット順: index = ((ch*128) + t)*T + thr
    - 学習時の test 精度は batch_size=100 のループで評価されるため、先頭 2,900 窓
      (29 バッチ。端数 47 窓は drop_last で除外)での値。全 2,947 窓では 0.90092

顔検出の出力は 3 ヘッド連結のスコアマップ(hm/size/off)までで、CenterNet のピーク抽出+デコードは
モデルに含まれない。仕様は experiments/detection_utils.py::decode_detections と
tools/webbench/src/detdecode.ts にあり、バンドルの readout 節(ヘッドの offset/scale)に
必要なメタデータが入っている。顔検出モデルの入力は生 RGB u8 画像(3×128×128)で、
前処理(thermometer + edge/curv/chroma)は整数記述子としてバンドルに焼かれている
(C では mdl_infer_image())。

使い方:

    uv run python scripts/fetch_demo_models.py all            # 3 モデルすべて
    uv run python scripts/fetch_demo_models.py har            # HAR 3 モデルだけ
    uv run python scripts/fetch_demo_models.py har_dlgn_k2400 # モデル名で 1 本だけ
    uv run python scripts/fetch_demo_models.py face           # 顔検出だけ(約 5〜6 分)
    uv run python scripts/fetch_demo_models.py all --offline  # キャッシュ済みなら wandb 不要
    uv run python scripts/fetch_demo_models.py har --compress none
        # 回路圧縮なしのバンドル(既定は --compress fast。max は conv2d 木の再合成も行うため
        # 顔検出では分オーダー。CLI の `griffith deploy bundle --compress` と同じ意味)
    uv run python scripts/fetch_demo_models.py face --resynth lossless
        # 顔検出に lossless 再合成を適用した版を det_face_lut_k256_resynth.* として別名出力
        # (既存の非 resynth 版は触らない。--resynth-max-cost N で探索ゲート数上限を指定)

wandb の認証は環境変数 WANDB_API_KEY(または --env-file / リポジトリ直下の .env)。
取得元は `--wandb-entity <entity> --wandb-project <project>`(または環境変数 WANDB_ENTITY /
WANDB_PROJECT。.env に書いてもよい)。どちらも未指定で wandb への取得が必要になると
エラーで止まる(キャッシュ済みなら不要)。
UCI-HAR が data-uci-har/ に無ければ UCI リポジトリから約 60MB をダウンロードして展開する
(`--no-download` で抑止)。HAR ローダは本スクリプト内の private 関数
(experiments/train_har_lowbit_baseline.py::load_har_raw の移植)。
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy.bundle import (  # noqa: E402
    attach_bundle_sizes,
    build_bundle,
    load_bundle,
    save_bundle,
)
from griffith_logic.deploy.cli import run as deploy_cli_run  # noqa: E402
from griffith_logic.ir.serialize import ir_from_dict, load_ir  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402

#: 取得元の wandb entity / project の既定値(空 = 未指定)。公開配布先が決まるまでは
#: 既定を持たず、`--wandb-entity` / `--wandb-project` か環境変数 WANDB_ENTITY /
#: WANDB_PROJECT で明示させる(`resolve_wandb_source`)。
WANDB_ENTITY = ''
WANDB_PROJECT = ''

RESYNTH_SUFFIX = '_resynth'
DEFAULT_OUT_DIR = REPO / 'out_examples' / 'demo_models'
DEFAULT_HAR_ROOT = REPO / 'data-uci-har'
UCI_HAR_URL = ('https://archive.ics.uci.edu/static/public/240/'
               'human+activity+recognition+using+smartphones.zip')

# ESP32(無印)の目安。DRAM は 320KB 級、flash は 4MB。
ESP32_DRAM_BYTES = 320 * 1024
ESP32_FLASH_BYTES = 4 * 1024 * 1024
# 無印 ESP32 は静的 .bss/.data に載せられる DRAM が約 160KB に制限される
# (残りは heap。mdl_ctx の作業メモリは呼び出し側が渡すので heap_caps_malloc でも可)。
ESP32_STATIC_DRAM_BYTES = 160 * 1024

# ---------------------------------------------------------------------------
# UCI-HAR(定数・ローダ・前処理)
# ---------------------------------------------------------------------------
HAR_CHANNELS = (
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
)
HAR_WINDOW = 128
HAR_N_CHANNELS = 9
HAR_LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
              'SITTING', 'STANDING', 'LAYING']
# 学習時(batch_size=100, drop_last)の評価対象 = 先頭 29 バッチ
HAR_TRAIN_EVAL_WINDOWS = 2900


def _load_inertial_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    sig_dir = dataset_dir / split / 'Inertial Signals'
    channels = [
        np.loadtxt(sig_dir / f'{ch}_{split}.txt', dtype=np.float32)
        for ch in HAR_CHANNELS
    ]
    x = np.stack(channels, axis=1)  # [N, 9, 128]
    y = np.loadtxt(dataset_dir / split / f'y_{split}.txt', dtype=np.int64) - 1
    return x, y


def download_uci_har(root: Path) -> None:
    """UCI リポジトリから UCI-HAR(約 60MB)を取得して root に展開する。

    配布 zip は「UCI HAR Dataset.zip」を内包する二重 zip なので 2 回展開する。
    """
    import urllib.request

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uci_har_') as tmp:
        outer = Path(tmp) / 'har.zip'
        print(f'  UCI-HAR をダウンロード: {UCI_HAR_URL}', flush=True)
        urllib.request.urlretrieve(UCI_HAR_URL, outer)
        with zipfile.ZipFile(outer) as zf:
            zf.extractall(root)
    inner = root / 'UCI HAR Dataset.zip'
    if inner.exists():
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(root)


def load_har_raw(root: Path = DEFAULT_HAR_ROOT, *, download: bool = True):
    """生信号 (x_train[N,9,128], y_train, x_test[M,9,128], y_test) を返す。

    experiments/train_har_lowbit_baseline.py::load_har_raw の移植(自己完結)。
    root/har_raw.npz があればキャッシュとして使い、無ければ 'UCI HAR Dataset' から
    読み込んで作る。展開済みデータも無ければ download=True で取得する。
    """
    root = Path(root)
    cache = root / 'har_raw.npz'
    if cache.exists():
        z = np.load(cache)
        return z['x_train'], z['y_train'], z['x_test'], z['y_test']
    dataset_dir = root / 'UCI HAR Dataset'
    if not dataset_dir.is_dir():
        if not download:
            raise FileNotFoundError(f'{dataset_dir} が無い(--no-download 指定)')
        download_uci_har(root)
    x_train, y_train = _load_inertial_split(dataset_dir, 'train')
    x_test, y_test = _load_inertial_split(dataset_dir, 'test')
    np.savez_compressed(cache, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def har_thermometer_params(x_train: np.ndarray, n_thresholds: int) -> dict:
    """[N,9,128] からチャネル別 mean/std と等分位 thermometer 閾値 [9,T] を作る。

    標準化は train 全窓のチャネル別 mean/std(float64、std<1e-8 は 1)。閾値は標準化後の
    分位点 (i+1)/(T+1)。これが学習時(uci-har-4t)の符号化と一致することは
    ir/simulate の test 精度再現(0.89931)で確認済み。
    """
    assert x_train.ndim == 3 and x_train.shape[1] == HAR_N_CHANNELS
    flat = x_train.transpose(1, 0, 2).reshape(HAR_N_CHANNELS, -1).astype(np.float64)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    std = np.where(std < 1e-8, 1.0, std)
    z = (flat - mean[:, None]) / std[:, None]
    qs = np.array([(i + 1) / (n_thresholds + 1) for i in range(n_thresholds)])
    thresholds = np.quantile(z, qs, axis=1).T  # [9, T]
    return {'mean': mean, 'std': std, 'thresholds': thresholds}


def encode_thermometer(x: np.ndarray, mean, std, thresholds) -> np.ndarray:
    """[N,C,L] → [N, C*L*T] の {0,1}(uint8)。

    bit index = ((ch*L) + t)*T + thr、判定は strict greater(z > thr)。
    thresholds は [C,T]。mean/std は [C]。
    """
    assert x.ndim == 3
    n, _, _ = x.shape
    mean = np.asarray(mean, dtype=np.float64)[None, :, None]
    std = np.asarray(std, dtype=np.float64)[None, :, None]
    thr = np.asarray(thresholds, dtype=np.float64)       # [C, T]
    z = (x.astype(np.float64) - mean) / std               # [N, C, L]
    bits = z[..., None] > thr[None, :, None, :]           # [N, C, L, T]
    return bits.reshape(n, -1).astype(np.uint8)


def har_accuracy(graph, x_test: np.ndarray, y_test: np.ndarray, params: dict) -> dict:
    """IR を ir/simulate で実行して test 精度を測る(先頭 2,900 窓と全 2,947 窓)。"""
    bits = encode_thermometer(x_test, params['mean'], params['std'], params['thresholds'])
    pred = simulate(graph, bits).argmax(axis=1)
    ok = pred == y_test
    n = HAR_TRAIN_EVAL_WINDOWS
    return {
        'sim_acc_first2900': float(ok[:n].mean()),
        'sim_acc_full': float(ok.mean()),
        'n_full': int(len(ok)),
        'n_first': int(min(n, len(ok))),
    }


def har_preproc(params: dict, n_thresholds: int) -> dict:
    """モデル JSON の preproc 節(dense thermometer。TS の dense エンジン契約と同形)。"""
    return {
        'kind': 'thermometer', 'input_kind': 'raw',
        'mean': params['mean'].tolist(), 'std': params['std'].tolist(),
        'thresholds': params['thresholds'].tolist(),
        'bit_order': 'ch,t,thr',
        'channels': list(HAR_CHANNELS), 'window': HAR_WINDOW,
        'n_thresholds': int(n_thresholds),
        'threshold_rule': 'quantile (i+1)/(T+1) of standardized train, strict greater',
    }


# ---------------------------------------------------------------------------
# wandb(読み取り専用)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                # 'har' | 'face'
    artifact: str            # wandb artifact(IR または ckpt)
    filename: str            # artifact 内のファイル名
    title: str


SPECS = {
    'har_dlgn_k6000': ModelSpec(
        'har_dlgn_k6000', 'har', 'uci-har-4t_k6000_l6_n2_e50000-ir:latest',
        'uci-har-4t_k6000_l6_n2_e50000.ir.json', '行動認識 UCI-HAR (k6000, 6層, 36,000ゲート)'),
    'har_dlgn_k2400': ModelSpec(
        # :latest(v3, run t030lozy)は test_acc 0.8352 の別 run。学習時 test acc 0.8903 の
        # run g10xuhzw は v0。同名 artifact に 4 run 分(v0..v3)あるので版を固定する
        'har_dlgn_k2400', 'har', 'uci-har-4t_k2400_l4_n2_e50000-ir:v0',
        'uci-har-4t_k2400_l4_n2_e50000.ir.json', '行動認識 UCI-HAR (k2400, 4層, 9,600ゲート)'),
    'har_dlgn_k600': ModelSpec(
        'har_dlgn_k600', 'har', 'uci-har-4t_k600_l4_n2_e50000-ir:latest',
        'uci-har-4t_k600_l4_n2_e50000.ir.json', '行動認識 UCI-HAR (k600, 4層, 2,400ゲート)'),
    'det_face_lut_k256': ModelSpec(
        'det_face_lut_k256', 'face', 'det_face_lut_k256_d3_L3_obj3-ckpt:latest',
        'best.pt', '顔検出 WIDER FACE CenterNetLUT (k256)'),
}
KIND_MODELS = {
    'har': ['har_dlgn_k6000', 'har_dlgn_k2400', 'har_dlgn_k600'],
    'face': ['det_face_lut_k256'],
    'all': ['har_dlgn_k6000', 'har_dlgn_k2400', 'har_dlgn_k600', 'det_face_lut_k256'],
}


def load_env_file(path: Path) -> None:
    """KEY=VALUE 形式の .env を環境変数へ読み込む(既存の環境変数は上書きしない。値は出力しない)。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().removeprefix('export ').strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def artifact_name(ref: str) -> str:
    """artifact 参照から版を除いた名前。"""
    return ref.split(':')[0]


def artifact_dir_name(ref: str) -> str:
    """キャッシュ用ディレクトリ名。:latest は版なしの名前、版固定(:v0 等)は `<名前>-v0`。

    同名 artifact の別版を同じキャッシュに混ぜない(:latest のキャッシュが版固定の取得を
    素通りさせてしまうのを防ぐ)。
    """
    name, _, version = ref.partition(':')
    return name if version in ('', 'latest') else f'{name}-{version}'


def resolve_wandb_source(entity: str | None, project: str | None) -> tuple[str, str]:
    """取得元の (entity, project) を引数 → 環境変数 → 既定値の順で決める。

    既定値は空なので、どれでも決まらなければ「何を指定すればよいか」を添えて止める
    (誤った entity へ黙って問い合わせない)。
    """
    entity = entity or os.environ.get('WANDB_ENTITY') or WANDB_ENTITY
    project = project or os.environ.get('WANDB_PROJECT') or WANDB_PROJECT
    missing = [name for name, v in (('--wandb-entity / WANDB_ENTITY', entity),
                                    ('--wandb-project / WANDB_PROJECT', project)) if not v]
    if missing:
        raise SystemExit(
            'wandb の取得元が未指定: ' + '、'.join(missing) + ' を指定してください'
            '(デモモデルの学習済み artifact は現状非公開の wandb プロジェクトにのみあり、'
            '公開配布先は未定。キャッシュ済みなら --offline で wandb なしに動く)')
    return entity, project


def fetch_artifact(spec: ModelSpec, cache_dir: Path, *, offline: bool,
                   entity: str | None = None,
                   project: str | None = None) -> tuple[Path, dict]:
    """wandb から artifact を取得(キャッシュ済みならスキップ)し、(ファイルパス, 来歴) を返す。

    来歴(provenance)は <artifact>/provenance.json にキャッシュし、再実行では wandb に
    触れない(--offline でなくても、キャッシュが揃っていれば API を呼ばない)。
    取得元は `resolve_wandb_source(entity, project)` で決める(キャッシュ時は不要)。
    """
    dest = cache_dir / artifact_dir_name(spec.artifact)
    path = dest / spec.filename
    prov_path = dest / 'provenance.json'
    if path.exists() and prov_path.exists():
        print(f'  [skip] キャッシュ済み: {path}')
        return path, json.loads(prov_path.read_text(encoding='utf-8'))
    if offline:
        raise SystemExit(f'--offline だがキャッシュが無い: {path}')
    entity, project = resolve_wandb_source(entity, project)
    import wandb

    api = wandb.Api(timeout=60)
    ref = f'{entity}/{project}/{spec.artifact}'
    art = api.artifact(ref)
    art.download(root=str(dest))
    run = art.logged_by()
    prov = {
        'artifact': (f'{entity}/{project}/'
                     f'{artifact_name(spec.artifact)}:{art.version}'),
        'artifact_size': int(art.size),
        'artifact_metadata': dict(art.metadata or {}),
    }
    if run is not None:
        meta = {}
        try:
            meta = run.metadata or {}
        except Exception:  # noqa: BLE001 - 来歴の欠損は致命的でない
            meta = {}
        prov.update({
            'run_id': run.id, 'run_name': run.name,
            'run_summary': {k: v for k, v in dict(run.summary).items()
                            if not k.startswith('_') and isinstance(v, int | float | str)},
            'run_cli_args': meta.get('args'),
        })
    prov_path.write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding='utf-8')
    if not path.exists():
        raise SystemExit(f'artifact にファイルが無い: {path}')
    return path, prov


# ---------------------------------------------------------------------------
# モデル JSON / バンドル / C 生成 / 指標
# ---------------------------------------------------------------------------
def load_export_webbench():
    """scripts/export_webbench.py をファイルパスから import する(パッケージ外のため)。"""
    if 'export_webbench' in sys.modules:
        return sys.modules['export_webbench']
    spec = importlib.util.spec_from_file_location(
        'export_webbench', REPO / 'scripts' / 'export_webbench.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['export_webbench'] = module
    spec.loader.exec_module(module)
    return module


def apply_resynth(doc: dict, mode: str | None, max_cost: int) -> tuple[dict, dict | None]:
    """`--resynth lossless` のとき conv2d 木を再合成した doc と統計を返す(それ以外は素通し)。

    `griffith deploy bundle --resynth lossless` と同じ関数(resynth_doc_lossless)を、
    build_bundle の**前**に適用する。conv2d が無いモデル(HAR 等)では何も変わらない。
    統計には所要時間(秒)を足す。
    """
    if mode is None:
        return doc, None
    if mode != 'lossless':
        raise SystemExit(f'未対応の --resynth: {mode}')
    from griffith_logic.deploy.resynth import resynth_doc_lossless

    t0 = time.time()
    new_doc, st = resynth_doc_lossless(doc, max_cost=max_cost)
    st = {**st, 'mode': mode, 'max_cost': max_cost, 'seconds': round(time.time() - t0, 1)}
    print(f"  resynth({mode}, max_cost={max_cost}): ゲート数 {st['gates_before']:,} → "
          f"{st['gates_after']:,} (木{st['n_trees']}本中{st['n_trees_changed']}本変更) "
          f"{st['seconds']}s")
    return new_doc, st


def write_model_and_bundle(doc: dict, out_dir: Path, name: str, *, verify: bool = True,
                           resynth: str | None = None, resynth_max_cost: int = 5,
                           compress: str = 'fast') -> dict:
    """モデル doc を <name>.model.json と <name>.bundle.json に書き出し、バンドルを返す。

    `griffith deploy bundle`(apply_compress → attach_compress_info → build_bundle(verify=True)
    → attach_bundle_sizes → save_bundle。griffith_logic/deploy/cli.py::_cmd_bundle)と
    同じ順序の手順。model.json は webbench 形式で、**圧縮前**の素の doc(= CLI の入力。
    resynth 指定時は再合成後)。resynth の統計は out_dir/<name>.resynth.json に保存する
    (summary が拾う)。

    `compress`('none'|'fast'|'max'、既定 fast)で lossless な回路圧縮を build_bundle の
    前に適用し、結果を about.compress に残す。`--resynth lossless` 済みなら再合成を二重に
    走らせない(apply_compress に resynth=False を渡し、info['applied'] の先頭に 'resynth' を足す)。
    Python API の build_bundle 単体には既定圧縮が入らない(CLI/本関数のみ)。
    """
    from griffith_logic.deploy.compress import (
        apply_compress,
        attach_compress_info,
        describe_compress,
    )

    doc, rs_stats = apply_resynth(doc, resynth, resynth_max_cost)
    if rs_stats is not None:
        (out_dir / f'{name}.resynth.json').write_text(
            json.dumps(rs_stats, indent=1, ensure_ascii=False), encoding='utf-8')
    model_path = out_dir / f'{name}.model.json'
    bundle_path = out_dir / f'{name}.bundle.json'
    model_path.write_text(json.dumps(doc, separators=(',', ':')), encoding='utf-8')
    print(f'  wrote {model_path} ({model_path.stat().st_size:,}B)')
    # 再合成済みなら apply_compress の自動再合成(現在は fast でも走る)を止める
    try:
        doc, compress_info, summaries = apply_compress(
            doc, level=compress,
            resynth=False if rs_stats is not None else None)
    except ValueError as exc:
        raise SystemExit(f'--compress: {exc}') from exc
    if rs_stats is not None:
        compress_info['applied'].insert(0, 'resynth')
        compress_info.setdefault('gates', {})['resynth'] = [
            int(rs_stats['gates_before']), int(rs_stats['gates_after'])]
    sweep_summary = summaries.get('sweep')
    if sweep_summary is not None:
        print(f"  sweep(lossless): ゲート数 {sweep_summary['gates_before']:,} → "
              f"{sweep_summary['gates_after']:,}")
    doc = attach_compress_info(doc, compress_info)
    if doc.get('about', {}).get('compress'):
        print('  ' + describe_compress(compress_info))
    bundle = build_bundle(doc, verify=verify)
    attach_bundle_sizes(bundle)
    save_bundle(bundle, bundle_path)
    print(f'  wrote {bundle_path} ({bundle_path.stat().st_size:,}B)')
    return bundle


def build_har_doc(spec: ModelSpec, ir_path: Path, prov: dict, har_root: Path,
                  *, download: bool) -> dict:
    """HAR の IR → モデル doc。ir/simulate で test 精度を再現して eval に記録する。"""
    eb = load_export_webbench()
    graph = load_ir(str(ir_path))
    in_width = int(graph.inputs[0].width)
    per_thr = HAR_N_CHANNELS * HAR_WINDOW
    if in_width % per_thr:
        raise SystemExit(f'IR入力幅 {in_width} が 9×128 の倍数でない(HAR 以外のIR?)')
    n_thr = in_width // per_thr
    x_tr, _, x_te, y_te = load_har_raw(har_root, download=download)
    params = har_thermometer_params(x_tr, n_thr)
    acc = har_accuracy(graph, x_te, y_te, params)
    # IR は学習の最終イテレーションから書き出される(best ではない)ので、再現の比較対象は
    # run の最終 test_acc。best_test_acc は参考(k600 では最終より 0.14pt 高い)
    rs = prov.get('run_summary') or {}
    reported, reported_best = rs.get('test_acc'), rs.get('best_test_acc')
    print(f'  simulate 再現: 先頭2900窓 acc={acc["sim_acc_first2900"]:.5f} '
          f'(学習時 最終test_acc={reported} / best_test_acc={reported_best})  '
          f'全{acc["n_full"]}窓 acc={acc["sim_acc_full"]:.5f}')
    cli = prov.get('run_cli_args') or []
    about = {
        'task': 'har', 'task_label': '行動認識(UCI-HAR 6クラス)',
        'dataset': 'uci-har-4t',
        'dataset_label': 'UCI HAR 生 Inertial Signals 9ch×128 step、4 閾値 thermometer',
        'recipe': {'cli_args': ' '.join(cli)} if cli else {},
        'wandb_run_id': prov.get('run_id'),
        'source_artifact': prov.get('artifact'),
        'exported_at': datetime.now().astimezone().isoformat(),
    }
    eval_info = {
        'train_reported_test_acc': reported,
        'train_reported_best_test_acc': reported_best,
        'train_eval_protocol': 'test 先頭2900窓(batch_size=100, drop_last)',
        **acc,
    }
    doc = eb.build_model_doc(
        graph, preproc=har_preproc(params, n_thr), labels=list(HAR_LABELS),
        eval_info=eval_info, readout={'kind': 'class_scores', 'k': len(HAR_LABELS),
                                      'spatial': None},
        about=about)
    return doc


def build_face_bundle(spec: ModelSpec, ckpt_path: Path, prov: dict, out_dir: Path,
                      name: str, *, verify: bool = True, resynth: str | None = None,
                      resynth_max_cost: int = 5, compress: str = 'fast') -> dict:
    """顔検出 ckpt → モデル doc + バンドル(export_webbench.cmd_det_ckpt を再利用)。

    cmd_det_ckpt は tools/webbench/public/models/ へ書く仕様なので、書き出し関数だけを
    差し替えて out_dir に流す(tools/ 配下は一切変更しない)。WIDER FACE の実データは
    不要(--demo-samples 0)。PyTorch hard eval ⇔ IR simulate の整数一致ガード
    (assert_ir_matches_torch)は cmd_det_ckpt 内蔵。
    """
    eb = load_export_webbench()
    captured: dict = {}

    def write_here(_path, doc):
        about = doc.get('about') or {}
        about['source_ckpt'] = prov.get('artifact', str(ckpt_path.name))
        if prov.get('run_id'):
            about['wandb_run_id'] = prov['run_id']
        doc['about'] = about
        eb_refresh_json_sizes(doc)
        captured['bundle'] = write_model_and_bundle(
            doc, out_dir, name, verify=verify, resynth=resynth,
            resynth_max_cost=resynth_max_cost, compress=compress)
        # resynth 指定時は再合成後の別 doc からバンドルを作るので、cmd_det_ckpt が最後に
        # 表示する resources(total_luts/サイズ)を実際に書き出したバンドルのものへ揃える
        doc['resources'] = captured['bundle']['resources']

    saved = (eb._write_model_bundle, eb.write_manifest)
    eb._write_model_bundle = write_here
    eb.write_manifest = lambda: None
    try:
        ckpt_arg = os.path.relpath(ckpt_path, Path.cwd())
        eb.cmd_det_ckpt(argparse.Namespace(
            ckpt=ckpt_arg, label=name, top_k=100, threshold=0.1,
            demo_samples=0, sample_seed=7, dry_run=False))
    finally:
        eb._write_model_bundle, eb.write_manifest = saved
    return captured['bundle']


def eb_refresh_json_sizes(doc: dict) -> None:
    """about 書き換え後の resources.json_bytes/gzip_bytes を測り直す(build_model_doc と同じ式)。"""
    import gzip

    # build_model_doc は json_bytes/gzip_bytes がまだ resources に無い状態で測っている
    doc['resources'].pop('json_bytes', None)
    doc['resources'].pop('gzip_bytes', None)
    text = json.dumps(doc, separators=(',', ':')).encode()
    doc['resources']['json_bytes'] = len(text)
    doc['resources']['gzip_bytes'] = len(gzip.compress(text, 6))


def run_verify_c(bundle_path: Path, n_samples: int, sidecar: Path, *, force: bool) -> dict:
    """verify-c(生成 C ⇔ ir/simulate の整数厳密一致)。結果は sidecar JSON に保存して再利用する。"""
    if sidecar.exists() and not force:
        print(f'  [skip] verify-c 済み: {sidecar}')
        return json.loads(sidecar.read_text(encoding='utf-8'))
    from griffith_logic.deploy.verify import verify_c_matches_simulate

    bundle = load_bundle(str(bundle_path), validate=True)
    t0 = time.time()
    res = verify_c_matches_simulate(bundle, n_samples=n_samples, seed=0, verbose=True)
    out = {'ok': True, 'path': res['path'], 'n_samples': res['n_samples'],
           'n_out': res['n_out'], 'seconds': round(time.time() - t0, 1)}
    sidecar.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'  verify-c OK: {out}')
    return out


def run_codegen_c(bundle_path: Path, c_dir: Path, *, force: bool) -> None:
    """`griffith deploy codegen-c` を実行する(model.c/model.h + ランタイムのコピー)。"""
    if (c_dir / 'model.h').exists() and not force:
        print(f'  [skip] C 生成済み: {c_dir}')
        return
    rc = deploy_cli_run(argparse.Namespace(
        deploy_command='codegen-c', source=str(bundle_path), output=str(c_dir),
        name='model', no_runtime=False, fuse=False, dense_batch=0))
    if rc != 0:
        raise SystemExit(f'codegen-c が失敗(rc={rc})')


_MACRO_RE = re.compile(r'^\s*#\s*define\s+(MDL_[A-Z0-9_]+)\s+(-?\d+)\b', re.MULTILINE)


def parse_model_h_macros(text: str) -> dict[str, int]:
    """model.h の `#define MDL_XXX <整数>` を辞書にする(MDL_CTX_BYTES/MDL_INPUT_WIDTH 等)。"""
    return {k: int(v) for k, v in _MACRO_RE.findall(text)}


def elf32_section_sizes(data: bytes) -> dict[str, int]:
    """ELF32(リトルエンディアン)オブジェクトの割り当てセクション合計サイズを分類する。

    flash = ALLOC かつ非 WRITE かつ非 NOBITS(.text/.rodata 等)、
    data = ALLOC かつ WRITE かつ非 NOBITS(初期値付き RAM。flash にも像が載る)、
    bss = ALLOC かつ NOBITS。
    """
    if data[:4] != b'\x7fELF' or data[4] != 1 or data[5] != 1:
        raise ValueError('ELF32 リトルエンディアンではない')
    shoff = struct.unpack_from('<I', data, 0x20)[0]
    shentsize, shnum = struct.unpack_from('<HH', data, 0x2E)
    sizes = {'flash': 0, 'data': 0, 'bss': 0}
    for i in range(shnum):
        _, sh_type, sh_flags, _, _, sh_size = struct.unpack_from(
            '<IIIIII', data, shoff + i * shentsize)
        if not sh_flags & 2:  # SHF_ALLOC
            continue
        if sh_type == 8:      # SHT_NOBITS
            sizes['bss'] += sh_size
        elif sh_flags & 1:    # SHF_WRITE
            sizes['data'] += sh_size
        else:
            sizes['flash'] += sh_size
    return sizes


def flash_estimate(c_dir: Path) -> dict | None:
    """clang の thumbv7em(Cortex-M4相当、-Os、libc無し)クロスビルドでコード+定数サイズを見積もる。

    ESP32(Xtensa)用ツールチェーンが手元に無いための代替見積り。Thumb-2 と Xtensa で
    コード密度は近いが同一ではないので、目安として扱うこと。mdl_kernels.o は未使用関数を
    含んだままの数字なので上限寄り(実リンクでは gc-sections で減る)。
    """
    clang = shutil.which('clang')
    if clang is None:
        return None
    try:
        total = {'flash': 0, 'data': 0, 'bss': 0}
        with tempfile.TemporaryDirectory(prefix='mdl_xc_') as tmp:
            for src in ('model.c', 'mdl_kernels.c'):
                obj = Path(tmp) / (src[:-2] + '.o')
                cmd = [clang, '--target=thumbv7em-none-eabi', '-Os', '-std=c99',
                       '-ffreestanding', '-DMDL_NO_LIBC', '-DMDL_NO_DEFAULT_CTX',
                       f'-I{c_dir}', '-c', str(c_dir / src), '-o', str(obj)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    return None
                for k, v in elf32_section_sizes(obj.read_bytes()).items():
                    total[k] += v
    except (OSError, ValueError):
        return None
    return {'target': 'thumbv7em-none-eabi -Os -DMDL_NO_LIBC', 'approximate': True, **total}


def gate_counts(bundle: dict) -> dict:
    """IR/バンドルからゲート数の指標を集める。

    gates_ir: IR 上のゲート数(dense=lut_layer の出力幅合計、conv=重み共有の木の数)。
    gates_after_programs: conv2d 命令列化後の命令数(conv が無ければ None)。
    total_luts_unrolled: conv/pool を空間全位置へ展開したときの LUT 総数(resources 由来)。
    """
    ir = bundle['ir']
    lut = sum(int(n['params']['out_width']) for n in ir['nodes'] if n['type'] == 'lut_layer')
    programs = bundle.get('programs') or {}
    before = sum(p['stats']['gates_before'] for p in programs.values())
    after = sum(p['stats']['gates_after'] for p in programs.values())
    res = bundle.get('resources') or {}
    return {
        'gates_lut_layer': lut,
        'gates_conv_weight_shared': before or None,
        'gates_after_programs': after if programs else None,
        'total_luts_unrolled': res.get('total_luts'),
    }


def esp32_verdict(macros: dict, input_kind: str, flash: dict | None,
                  model_c_bytes: int) -> dict:
    """ESP32(DRAM 約 320KB / flash 4MB)に載るかの機械的な見積り。

    作業 RAM = MDL_CTX_BYTES + 入力バッファ + スコア出力(int32)。入力バッファは bits 入力なら
    1 ビット 1 バイトに展開した MDL_INPUT_WIDTH バイト、画像入力なら u8 画像そのもの。
    flash は thumbv7em 見積り(無ければ model.c のソースサイズを粗い上限として使う)。
    """
    ctx = macros.get('MDL_CTX_BYTES', 0)
    out = macros.get('MDL_OUTPUT_WIDTH', 0) * 4
    if input_kind == 'image_u8':
        inp = (macros.get('MDL_IMAGE_CHANNELS', 0) * macros.get('MDL_INPUT_HEIGHT', 0)
               * macros.get('MDL_INPUT_WIDTH_PX', 0))
    else:
        inp = macros.get('MDL_INPUT_WIDTH', 0)
    ram = ctx + inp + out
    flash_bytes = (flash['flash'] + flash['data']) if flash else model_c_bytes
    return {
        'ram_working_bytes': ram,
        'ram_breakdown': {'ctx': ctx, 'input_buffer': inp, 'scores': out},
        'fits_dram_320KB': ram <= ESP32_DRAM_BYTES,
        'fits_static_dram_160KB': ram <= ESP32_STATIC_DRAM_BYTES,
        'flash_estimate_bytes': flash_bytes,
        'fits_flash_4MB': flash_bytes <= ESP32_FLASH_BYTES,
        'note': ('ctx は mdl_ctx_init() に渡す領域なので heap_caps_malloc でも可。'
                 '画像/ビット入力の u8 展開バッファを含めて見積もっている'),
    }


def summarize_model(spec: ModelSpec, out_dir: Path, prov: dict, verify: dict | None) -> dict:
    """バンドル・C 生成物・来歴から summary.json の 1 モデル分を組み立てる。"""
    bundle_path = out_dir / f'{spec.name}.bundle.json'
    bundle = load_bundle(str(bundle_path), validate=False)
    c_dir = out_dir / 'c' / spec.name
    h_text = (c_dir / 'model.h').read_text(encoding='utf-8')
    macros = parse_model_h_macros(h_text)
    model_c_bytes = (c_dir / 'model.c').stat().st_size
    graph = ir_from_dict(bundle['ir'])
    input_kind = 'image_u8' if bundle.get('preproc_desc') is not None else 'bits'
    in_port = graph.inputs[0]
    res = bundle.get('resources') or {}
    flash = flash_estimate(c_dir)
    ev = bundle.get('eval') or {}
    node_types: dict[str, int] = {}
    for n in bundle['ir']['nodes']:
        node_types[n['type']] = node_types.get(n['type'], 0) + 1
    entry = {
        'name': spec.name,
        'title': spec.title,
        'task': spec.kind,
        'source': {k: prov.get(k) for k in ('artifact', 'run_id', 'run_name')},
        'input': {
            'kind': input_kind,
            'width_bits': int(in_port.width),
            'image_shape': ([macros.get('MDL_IMAGE_CHANNELS'), macros.get('MDL_INPUT_HEIGHT'),
                             macros.get('MDL_INPUT_WIDTH_PX')]
                            if input_kind == 'image_u8' else None),
            'window_shape': [HAR_N_CHANNELS, HAR_WINDOW] if spec.kind == 'har' else None,
        },
        'labels': bundle.get('labels'),
        'output_width': int(graph.outputs[0].width),
        'node_types': node_types,
        'gates': gate_counts(bundle),
        'sizes': {
            'bundle_json_bytes': bundle_path.stat().st_size,
            'bundle_gzip_bytes': res.get('bundle_gzip_bytes'),
            'model_json_bytes': (out_dir / f'{spec.name}.model.json').stat().st_size,
            'model_c_bytes': model_c_bytes,
            'model_h_bytes': (c_dir / 'model.h').stat().st_size,
            'flash_estimate_thumbv7em': flash,
        },
        'macros': {k: macros[k] for k in (
            'MDL_CTX_BYTES', 'MDL_INPUT_WIDTH', 'MDL_OUTPUT_WIDTH', 'MDL_NUM_LABELS',
            *(('MDL_IMAGE_CHANNELS', 'MDL_INPUT_HEIGHT', 'MDL_INPUT_WIDTH_PX')
              if input_kind == 'image_u8' else ()))
            if k in macros},
        'accuracy': {
            'train_reported': ev.get('train_reported_test_acc', ev.get('best_hard_map')),
            'train_reported_best': ev.get('train_reported_best_test_acc'),
            'train_reported_metric': ('final test_acc' if spec.kind == 'har'
                                      else 'best_hard_mAP'),
            'sim_first2900': ev.get('sim_acc_first2900'),
            'sim_full': ev.get('sim_acc_full'),
            'n_full': ev.get('n_full'),
        },
        'compress': (bundle.get('about') or {}).get('compress'),
        'verify_c': verify,
        'esp32': esp32_verdict(macros, input_kind, flash, model_c_bytes),
    }
    rs_path = out_dir / f'{spec.name}.resynth.json'
    if rs_path.exists():
        entry['resynth'] = json.loads(rs_path.read_text(encoding='utf-8'))
    return entry


def format_summary_table(summary: dict) -> str:
    """summary.json の models を人間向けの Markdown 表にする。"""
    head = ('| モデル | ゲート数 | bundle(B) | model.c(B) | MDL_CTX_BYTES | 入力幅 | '
            'simulate再現精度 | ESP32 |')
    sep = '|---|---:|---:|---:|---:|---:|---:|---|'
    rows = [head, sep]
    for m in summary['models'].values():
        g = m['gates']
        gates = g['gates_lut_layer'] or g['gates_conv_weight_shared'] or 0
        acc = m['accuracy']
        acc_s = (f"{acc['sim_first2900']:.5f}" if acc.get('sim_first2900') is not None
                 else '-')
        e = m['esp32']
        verdict = ('DRAM/flash とも可' if e['fits_dram_320KB'] and e['fits_flash_4MB']
                   else ('flash可・DRAM不可' if e['fits_flash_4MB'] else '不可'))
        rows.append(
            f"| {m['name']} | {gates:,} | {m['sizes']['bundle_json_bytes']:,} | "
            f"{m['sizes']['model_c_bytes']:,} | {m['macros'].get('MDL_CTX_BYTES', 0):,} | "
            f"{m['macros'].get('MDL_INPUT_WIDTH', 0):,} | {acc_s} | {verdict} |")
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def resynth_variant(spec: ModelSpec, mode: str | None) -> ModelSpec:
    """resynth 指定時は出力名に `_resynth` を付けた別 spec にする(非 resynth 版を上書きしない)。"""
    if mode is None:
        return spec
    return replace(spec, name=f'{spec.name}{RESYNTH_SUFFIX}',
                   title=f'{spec.title} [resynth {mode}]')


def process_model(spec: ModelSpec, args, out_dir: Path) -> dict:
    t0 = time.time()
    print(f'\n=== {spec.name}: {spec.title} ===', flush=True)
    cache_dir = out_dir / 'wandb_cache'
    bundle_path = out_dir / f'{spec.name}.bundle.json'

    print('[1/5] wandb 取得')
    src_path, prov = fetch_artifact(spec, cache_dir, offline=args.offline,
                                    entity=args.wandb_entity, project=args.wandb_project)

    print('[2-3/5] モデル JSON → バンドル')
    if bundle_path.exists() and (out_dir / f'{spec.name}.model.json').exists() \
            and not args.force:
        print(f'  [skip] バンドル済み: {bundle_path}')
    elif spec.kind == 'har':
        doc = build_har_doc(spec, src_path, prov, Path(args.har_root),
                            download=not args.no_download)
        write_model_and_bundle(doc, out_dir, spec.name, resynth=args.resynth,
                               resynth_max_cost=args.resynth_max_cost,
                               compress=args.compress)
    else:
        build_face_bundle(spec, src_path, prov, out_dir, spec.name, resynth=args.resynth,
                          resynth_max_cost=args.resynth_max_cost, compress=args.compress)

    print('[4/5] verify-c')
    verify = None
    if not args.no_verify_c:
        n = args.verify_samples or (5 if spec.kind == 'har' else 1)
        verify = run_verify_c(bundle_path, n, out_dir / f'{spec.name}.verify_c.json',
                              force=args.force)

    print('[5/5] codegen-c')
    run_codegen_c(bundle_path, out_dir / 'c' / spec.name, force=args.force)

    entry = summarize_model(spec, out_dir, prov, verify)
    entry['elapsed_seconds'] = round(time.time() - t0, 1)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='デプロイキットのデモ用モデル(行動認識 UCI-HAR / 顔検出 WIDER FACE)を '
                    'wandb から取得し、バンドル・C ソース・サイズ指標まで作る',
        epilog='詳細はこのファイルの docstring を参照(前処理の確定内容・出力一覧)。')
    ap.add_argument('target', nargs='?', default='all',
                    choices=['har', 'face', 'all', *SPECS],
                    help='処理対象: har=HAR 3モデル / face=顔検出 / all=全部(既定) / '
                         'またはモデル名(例 har_dlgn_k2400)で 1 本だけ')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR),
                    help='出力先(既定: out_examples/demo_models)')
    ap.add_argument('--har-root', default=str(DEFAULT_HAR_ROOT),
                    help='UCI-HAR のディレクトリ(既定: data-uci-har)')
    ap.add_argument('--no-download', action='store_true',
                    help='UCI-HAR が無くてもダウンロードしない(無ければエラー)')
    ap.add_argument('--offline', action='store_true',
                    help='wandb に接続しない(キャッシュが無ければエラー)')
    ap.add_argument('--wandb-entity', default=None,
                    help='取得元の wandb entity(既定: 環境変数 WANDB_ENTITY。既定値なし)')
    ap.add_argument('--wandb-project', default=None,
                    help='取得元の wandb project(既定: 環境変数 WANDB_PROJECT。既定値なし)')
    ap.add_argument('--env-file', default=None,
                    help='WANDB_API_KEY を含む .env(既定: リポジトリ直下の .env)')
    ap.add_argument('--force', action='store_true',
                    help='既存バンドル/verify-c/C 生成物を作り直す(wandb キャッシュは再取得しない)')
    ap.add_argument('--no-verify-c', action='store_true',
                    help='verify-c を省略する(顔検出は 1 サンプルでも約 3 分かかる)')
    ap.add_argument('--verify-samples', type=int, default=0,
                    help='verify-c のサンプル数(既定: HAR 5 / 顔検出 1)')
    ap.add_argument('--resynth', choices=['lossless'], default=None,
                    help='バンドル化の前に conv2d 木を lossless 再合成する(`griffith deploy '
                         'bundle --resynth` と同じ)。出力は <name>_resynth.* の別名で、'
                         '非 resynth 版は触らない。実モデルは分オーダー')
    ap.add_argument('--resynth-max-cost', type=int, default=5,
                    help='再合成の探索ゲート数上限(既定: 5。--resynth 指定時のみ有効)')
    ap.add_argument('--compress', choices=['none', 'fast', 'max'], default='fast',
                    help='バンドル化の前に適用する lossless 回路圧縮(`griffith deploy '
                         'bundle --compress` と同じ。既定: fast = dense に sweep、max は '
                         'conv2d 木の再合成も=顔検出は分オーダー、none = 圧縮なし)')
    args = ap.parse_args(argv)

    load_env_file(Path(args.env_file) if args.env_file else REPO / '.env')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / 'summary.json'
    summary = {'models': {}}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary.setdefault('models', {})
    for name in KIND_MODELS.get(args.target, [args.target]):
        spec = resynth_variant(SPECS[name], args.resynth)
        summary['models'][spec.name] = process_model(spec, args, out_dir)
        summary['generated_at'] = datetime.now().astimezone().isoformat()
        summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                encoding='utf-8')
    print(f'\nsummary: {summary_path}\n')
    print(format_summary_table(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
