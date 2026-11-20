"""データセット取込API。

CSV / 画像フォルダ(zip)を取り込み、split・augmentation・入力エンコーディング
プレビューを備えた「動的データセット」として登録する独立モジュール。

設計裁定9(本体昇格はファイル移動のみで済む構造):
  - 取込ロジックは本モジュール内に閉じ、`griffith_logic/` 本体・`datasets.py` は
    読み取り流用のみ(いずれも非改変)。`ResolvedDataset` 互換の loader を返すため、
    将来の本体昇格時は本ファイルを `griffith_logic/datasets/` へ移すだけで済む。
  - 永続化は `outputs/editor_datasets/<slug>/`(gitignore配下)。manifest.json + data.npz。
    tensorへ正規化済みで保存するため、resolve は npz を読むだけで再現的に解決できる。

main.py への統合(統合はメインが実施。本ファイルは新規追加のみ):
    from . import dataset_import
    app.include_router(dataset_import.router)
  加えて、取込データセットを既存カタログ/学習resolveへ露出するため以下2点を追記する
  (詳細は最終報告の「main.py統合手順」を参照):
    1. GET /api/datasets(main.py datasets_catalog):
         return list_datasets() + dataset_import.list_imported_datasets()
    2. 学習resolve(main.py _validate_trainable / train本体の resolve_dataset 呼び出し):
         try: resolved = resolve_dataset(name, ...)
         except ValueError:
             resolved = dataset_import.resolve_imported(name, batch_size=..., params=...)
  フロントは GET /api/datasets/imported を併せて取得しカタログ合成するため、
  上記1が未統合でも DATASET タブと学習セレクタには取込分が現れる。
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from .datasets import ResolvedDataset
from .security import max_upload_bytes

router = APIRouter()

# zip展開後の合計サイズ上限。圧縮率の高いzip(zip bomb)で
# メモリを食い潰されないよう、アップロード上限と同じ値で頭を押さえる。
UPLOAD_CHUNK_BYTES = 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORE_ROOT = PROJECT_ROOT / 'outputs' / 'editor_datasets'

IMPORTED_PREFIX = 'imported:'
MAX_PREVIEW_SAMPLES = 200
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200

async def _read_upload_limited(file: UploadFile, what: str) -> bytes:
    """アップロードを上限付きで読み込む。

    `await file.read()` は本文全体を無制限にメモリへ載せるため、巨大ファイルで
    プロセスを落とせる(ローカル専用ツールでもブラウザ経由で踏める)。
    1MiBずつ読み、`max_upload_bytes()` を超えた時点で **413** を返して打ち切る。
    """
    limit = max_upload_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f'{what}が上限({limit / 1024 / 1024:.0f}MiB)を超えています。'
                       'GRIFFITH_EDITOR_MAX_UPLOAD_MB で上限を変更できます',
            )
        chunks.append(chunk)
    return b''.join(chunks)


def _check_zip_uncompressed_size(zf: zipfile.ZipFile) -> None:
    """zip展開後の合計サイズ上限を検査する(zip bomb対策)。"""
    limit = max_upload_bytes()
    total = sum(info.file_size for info in zf.infolist())
    if total > limit:
        raise HTTPException(
            status_code=413,
            detail=f'zip展開後の合計サイズ({total / 1024 / 1024:.0f}MiB)が'
                   f'上限({limit / 1024 / 1024:.0f}MiB)を超えています。'
                   'GRIFFITH_EDITOR_MAX_UPLOAD_MB で上限を変更できます',
        )


# ヘッダ列名 name[__dimIndex][:labelName] のパーサ
_HEADER_RE = re.compile(r'^(?P<base>[^:]+?)(?:__(?P<dim>\d+))?(?::(?P<disp>.*))?$')
_INPUT_BASE_RE = re.compile(r'^x$')
_OUTPUT_BASE_RE = re.compile(r'^y$')


# ---------------------------------------------------------------------------
# データモデル
# ---------------------------------------------------------------------------
@dataclass
class ImportedDataset:
    """取込済みデータセット(manifest.json と対応)。"""

    id: str            # 'imported:<slug>'
    label: str
    kind: str          # 'classification' | 'binary_vector'
    in_width: int
    out_width: int
    num_data: int
    input_shape: list[int] | None   # 画像取込時 [C,H,W]。数値ベクトルはNone
    split: dict          # {'train','val','test'} 比率(合計1.0)
    seed: int
    augmentation: dict   # {'flip','shift','bit_flip_noise'}
    encoding: dict       # {'mode','threshold','bits'}
    source: str          # 'csv' | 'images'
    created_at: float

    def catalog_entry(self) -> dict:
        """GET /api/datasets 互換のカタログエントリ(DatasetInfo)。"""
        if self.kind == 'classification':
            output_spec: dict = {'width': self.out_width, 'semantics': 'class_scores'}
        else:
            output_spec = {'width': self.out_width, 'semantics': 'binary_vector'}
        input_spec: dict = {'width': self.in_width, 'encoding': 'binary'}
        if self.input_shape is not None:
            input_spec['shape'] = self.input_shape
        return {
            'name': self.id,
            'label': self.label,
            'in_width': self.in_width,
            'out_width': self.out_width,
            'kind': self.kind,
            'input_spec': input_spec,
            'output_spec': output_spec,
            'imported': True,
        }

    def manifest(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'kind': self.kind,
            'in_width': self.in_width,
            'out_width': self.out_width,
            'num_data': self.num_data,
            'input_shape': self.input_shape,
            'split': self.split,
            'seed': self.seed,
            'augmentation': self.augmentation,
            'encoding': self.encoding,
            'source': self.source,
            'created_at': self.created_at,
        }


DEFAULT_SPLIT = {'train': 0.8, 'val': 0.2, 'test': 0.0}
DEFAULT_AUGMENTATION = {'flip': False, 'shift': 0, 'bit_flip_noise': 0.0}
DEFAULT_ENCODING = {'mode': 'none', 'threshold': 0.5, 'bits': 4}


# ---------------------------------------------------------------------------
# レジストリ(ディスク永続 + メモリキャッシュ)
# ---------------------------------------------------------------------------
_registry: dict[str, ImportedDataset] = {}
_loaded = False


def _slug_dir(dataset_id: str) -> Path:
    slug = dataset_id[len(IMPORTED_PREFIX):] if dataset_id.startswith(IMPORTED_PREFIX) else dataset_id
    return STORE_ROOT / slug


def _load_registry() -> None:
    """起動時に outputs/editor_datasets/*/manifest.json を走査してレジストリを復元する。"""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not STORE_ROOT.exists():
        return
    for manifest_path in sorted(STORE_ROOT.glob('*/manifest.json')):
        try:
            data = json.loads(manifest_path.read_text(encoding='utf-8'))
            ds = ImportedDataset(**data)
            _registry[ds.id] = ds
        except Exception:  # noqa: BLE001 - 壊れたmanifestは無視して継続
            continue


def _slugify(name: str) -> str:
    base = re.sub(r'[^0-9A-Za-z_-]+', '_', name).strip('_').lower() or 'dataset'
    return base[:40]


def _unique_slug(name: str) -> str:
    base = _slugify(name)
    slug = base
    suffix = int(time.time() * 1000) % 100000
    while (STORE_ROOT / slug).exists() or f'{IMPORTED_PREFIX}{slug}' in _registry:
        slug = f'{base}_{suffix}'
        suffix += 1
    return slug


def _persist(ds: ImportedDataset, x: np.ndarray, y: np.ndarray) -> None:
    """manifest.json と data.npz を書き出し、レジストリへ登録する。"""
    d = _slug_dir(ds.id)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / 'data.npz', x=x, y=y)
    (d / 'manifest.json').write_text(
        json.dumps(ds.manifest(), ensure_ascii=False, indent=2), encoding='utf-8'
    )
    _registry[ds.id] = ds


def _load_arrays(dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    npz = _slug_dir(dataset_id) / 'data.npz'
    if not npz.exists():
        raise HTTPException(status_code=404, detail=f'取込データが見つかりません: {dataset_id}')
    with np.load(npz) as z:
        return z['x'], z['y']


# ---------------------------------------------------------------------------
# 公開API(main.py / dataset_samples から読み取り流用される)
# ---------------------------------------------------------------------------
def list_imported_datasets() -> list[dict]:
    """取込データセットのカタログエントリ一覧(DatasetInfo互換)。"""
    _load_registry()
    return [ds.catalog_entry() for ds in _registry.values()]


def get_imported(dataset_id: str) -> ImportedDataset | None:
    _load_registry()
    return _registry.get(dataset_id)


def _split_indices(num_data: int, split: dict, seed: int) -> dict[str, list[int]]:
    """seed固定の決定的分割。train/val/testのindexリストを返す。"""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_data, generator=g).tolist()
    train_r = float(split.get('train', 0.0))
    val_r = float(split.get('val', 0.0))
    n_train = int(round(num_data * train_r))
    n_val = int(round(num_data * val_r))
    n_train = min(n_train, num_data)
    n_val = min(n_val, num_data - n_train)
    return {
        'train': perm[:n_train],
        'val': perm[n_train:n_train + n_val],
        'test': perm[n_train + n_val:],
    }


def _apply_augmentation(
    x: torch.Tensor, aug: dict, input_shape: list[int] | None, seed: int,
) -> torch.Tensor:
    """train split向けaugmentation。バッチ(N,D)テンソルへ適用する。

    - flip: 画像形状がある場合のみ水平反転(左右)
    - shift: 画像形状がある場合のみ最大±shift画素の巡回シフト
    - bit_flip_noise: 各ビットを確率pで反転({0,1}入力整合)
    """
    g = torch.Generator().manual_seed(int(seed))
    out = x.clone()
    flip = bool(aug.get('flip'))
    shift = int(aug.get('shift', 0))
    if input_shape is not None and (flip or shift):
        c, h, w = input_shape
        img = out.reshape(-1, c, h, w)
        if flip:
            mask = torch.rand(img.shape[0], generator=g) < 0.5
            img[mask] = torch.flip(img[mask], dims=[3])
        if shift > 0:
            dx = int(torch.randint(-shift, shift + 1, (1,), generator=g).item())
            dy = int(torch.randint(-shift, shift + 1, (1,), generator=g).item())
            img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
        out = img.reshape(out.shape)
    noise = float(aug.get('bit_flip_noise', 0.0))
    if noise > 0.0:
        flip_mask = torch.rand(out.shape, generator=g) < noise
        out = torch.where(flip_mask, 1.0 - out, out)
    return out


def resolve_imported(
    name: str, *, batch_size: int, params: dict | None = None,
) -> ResolvedDataset:
    """取込データセットを ResolvedDataset へ解決する(学習resolve経路)。

    train splitのみを学習に用い、augmentationはtrain splitへ適用する。
    未知IDは ValueError(main.py側で400へ)。
    """
    _load_registry()
    ds = _registry.get(name)
    if ds is None:
        raise ValueError(f'取込データセットが見つかりません: {name}')
    x_np, y_np = _load_arrays(name)
    idx = _split_indices(ds.num_data, ds.split, ds.seed)
    train_idx = idx['train'] or list(range(ds.num_data))  # trainが空なら全件で学習
    x_all = torch.from_numpy(x_np).float()
    if ds.kind == 'classification':
        y_all = torch.from_numpy(y_np).long().reshape(-1)
    else:
        y_all = torch.from_numpy(y_np).float()

    def factory(train_idx=train_idx):
        xs = x_all[train_idx]
        ys = y_all[train_idx]
        xs = _apply_augmentation(xs, ds.augmentation, ds.input_shape, ds.seed)
        dataset = torch.utils.data.TensorDataset(xs, ys)
        bs = min(batch_size, len(dataset)) or 1
        return torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=True, drop_last=False,
        )

    return ResolvedDataset(
        name=name, kind=ds.kind,
        in_width=ds.in_width, out_width=ds.out_width,
        loader_factory=factory,
    )


# ---------------------------------------------------------------------------
# CSV取込
# ---------------------------------------------------------------------------
def _parse_header(col: str) -> tuple[str, int | None, str | None]:
    m = _HEADER_RE.match(col.strip())
    if not m:
        return col.strip(), None, None
    return m.group('base').strip(), (
        int(m.group('dim')) if m.group('dim') is not None else None
    ), (m.group('disp').strip() if m.group('disp') is not None else None)


def _parse_csv_bytes(raw: bytes) -> tuple[list[str], list[list[str]]]:
    import csv

    text = raw.decode('utf-8-sig')
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise HTTPException(status_code=400, detail='CSVにはヘッダ行と1行以上のデータが必要です')
    return rows[0], rows[1:]


def _build_numeric_dataset(
    header: list[str], data_rows: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, str, int, list[int], list[int]]:
    """NNC形式ヘッダ(x__k / y / y:label)+汎用CSVを数値配列へ変換する。

    戻り値: (X, Y, kind, out_width, input_cols, output_cols)。
    """
    parsed = [_parse_header(c) for c in header]
    input_cols: list[int] = []
    output_cols: list[int] = []
    label_disp: list[str | None] = []
    for i, (base, _dim, disp) in enumerate(parsed):
        if _INPUT_BASE_RE.match(base):
            input_cols.append(i)
        elif _OUTPUT_BASE_RE.match(base):
            output_cols.append(i)
            label_disp.append(disp)
    # x/y列が無い汎用CSV: 末尾列をターゲット、残りを特徴とみなす
    if not input_cols and not output_cols:
        input_cols = list(range(len(header) - 1))
        output_cols = [len(header) - 1]
        label_disp = [None]
    if not input_cols or not output_cols:
        raise HTTPException(
            status_code=400,
            detail='入力列(x/x__k)と出力列(y/y__k)を判別できませんでした',
        )

    def to_floats(cols: list[int]) -> np.ndarray:
        out = []
        for r in data_rows:
            try:
                out.append([float(r[c]) for c in cols])
            except (ValueError, IndexError) as e:
                raise HTTPException(
                    status_code=400, detail=f'CSVの数値変換に失敗しました: {e}',
                )
        return np.asarray(out, dtype=np.float32)

    x = to_floats(input_cols)
    y_raw = to_floats(output_cols)

    # kind判定: y:label は分類(クラスindex)、単一y列でも{0,1}以外なら分類、
    # それ以外(y/y__k が全て{0,1})は binary_vector
    is_label = any(d == 'label' for d in label_disp)
    y_is_binary = bool(np.all(np.isin(y_raw, [0.0, 1.0])))
    if is_label or (y_raw.shape[1] == 1 and not y_is_binary):
        classes = y_raw[:, 0].astype(np.int64)
        out_width = int(classes.max()) + 1 if len(classes) else 1
        return x, classes, 'classification', out_width, input_cols, output_cols
    if not y_is_binary:
        raise HTTPException(
            status_code=400,
            detail='binary_vector出力の値は0/1のみ許可されます(分類は y:label を使ってください)',
        )
    return x, y_raw.astype(np.float32), 'binary_vector', y_raw.shape[1], input_cols, output_cols


@router.post('/api/datasets/import/csv')
async def import_csv(
    file: UploadFile = File(...),  # noqa: B008 — FastAPIのDI正規パターン
    name: str | None = Form(None),  # noqa: B008
):
    """CSVを取込み動的カタログへ登録する。"""
    _load_registry()
    raw = await _read_upload_limited(file, 'CSVファイル')
    header, data_rows = _parse_csv_bytes(raw)
    x, y, kind, out_width, _in_cols, _out_cols = _build_numeric_dataset(header, data_rows)

    label = (name or (file.filename or 'csv').rsplit('.', 1)[0])
    slug = _unique_slug(label)
    ds = ImportedDataset(
        id=f'{IMPORTED_PREFIX}{slug}',
        label=f'{label}(取込)',
        kind=kind,
        in_width=int(x.shape[1]),
        out_width=int(out_width),
        num_data=int(x.shape[0]),
        input_shape=None,
        split=dict(DEFAULT_SPLIT),
        seed=0,
        augmentation=dict(DEFAULT_AUGMENTATION),
        encoding=dict(DEFAULT_ENCODING),
        source='csv',
        created_at=time.time(),
    )
    _persist(ds, x, y)
    return {'dataset': ds.catalog_entry()}


# ---------------------------------------------------------------------------
# 画像フォルダ取込
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}


def _preprocess_image(img, resize: int, grayscale: bool, binarize: float | None):
    """PIL画像 → (C,H,W) float32 テンソル。resize/grayscale/binarizeパイプ。"""
    from PIL import Image

    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    t = torch.from_numpy(arr)
    if binarize is not None:
        t = (t > binarize).float()
    return t


def _safe_extract_members(zf: zipfile.ZipFile):
    for info in zf.infolist():
        n = info.filename
        if n.startswith('/') or '..' in Path(n).parts:
            raise HTTPException(status_code=400, detail='zip内に不正なパスが含まれています')
        yield info


@router.post('/api/datasets/import/images')
async def import_images(
    file: UploadFile = File(...),  # noqa: B008 — FastAPIのDI正規パターン
    name: str | None = Form(None),
    resize: int = Form(16),
    grayscale: bool = Form(True),
    binarize: float | None = Form(0.5),
):
    """zip(トップ階層フォルダ名=クラス)から画像データセットを作成する。"""
    _load_registry()
    from PIL import Image

    raw = await _read_upload_limited(file, 'zipファイル')
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail='zipファイルを読み込めませんでした')
    _check_zip_uncompressed_size(zf)

    # クラス → 画像テンソル
    class_names: list[str] = []
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    with zf:
        for info in _safe_extract_members(zf):
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                continue  # トップ直下ファイルはクラス不明として無視
            cls = parts[0]
            if Path(info.filename).suffix.lower() not in _IMAGE_EXTS:
                continue
            if cls not in class_names:
                class_names.append(cls)
            try:
                img = Image.open(io.BytesIO(zf.read(info)))
                t = _preprocess_image(img, resize, grayscale, binarize)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f'画像の読み込みに失敗しました: {e}')
            xs.append(t.reshape(-1))
            ys.append(class_names.index(cls))

    if not xs:
        raise HTTPException(status_code=400, detail='zip内にクラス別画像が見つかりませんでした')

    x = torch.stack(xs).numpy().astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    c = 1 if grayscale else 3
    label = name or (file.filename or 'images').rsplit('.', 1)[0]
    slug = _unique_slug(label)
    ds = ImportedDataset(
        id=f'{IMPORTED_PREFIX}{slug}',
        label=f'{label}(取込)',
        kind='classification',
        in_width=int(x.shape[1]),
        out_width=len(class_names),
        num_data=int(x.shape[0]),
        input_shape=[c, resize, resize],
        split=dict(DEFAULT_SPLIT),
        seed=0,
        augmentation=dict(DEFAULT_AUGMENTATION),
        encoding={'mode': 'binarize' if binarize is not None else 'none',
                  'threshold': binarize if binarize is not None else 0.5, 'bits': 4},
        source='images',
        created_at=time.time(),
    )
    _persist(ds, x, y)
    return {'dataset': ds.catalog_entry(), 'classes': class_names}


# ---------------------------------------------------------------------------
# 設定更新(split / augmentation / encoding)
# ---------------------------------------------------------------------------
class DatasetConfigUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    split: dict | None = None
    seed: int | None = None
    augmentation: dict | None = None
    encoding: dict | None = None


@router.put('/api/datasets/imported/{dataset_id_slug}/config')
def update_config(dataset_id_slug: str, body: DatasetConfigUpdate):
    """取込データセットのsplit/seed/augmentation/encodingを更新する。"""
    _load_registry()
    dataset_id = f'{IMPORTED_PREFIX}{dataset_id_slug}'
    ds = _registry.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f'取込データセットが見つかりません: {dataset_id}')
    if body.split is not None:
        total = sum(float(v) for v in body.split.values())
        if total <= 0:
            raise HTTPException(status_code=422, detail='split比率の合計は正である必要があります')
        ds.split = {k: float(v) / total for k, v in body.split.items()}
    if body.seed is not None:
        ds.seed = int(body.seed)
    if body.augmentation is not None:
        ds.augmentation = {**DEFAULT_AUGMENTATION, **body.augmentation}
    if body.encoding is not None:
        ds.encoding = {**DEFAULT_ENCODING, **body.encoding}
    (_slug_dir(dataset_id) / 'manifest.json').write_text(
        json.dumps(ds.manifest(), ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return ds.catalog_entry() | {
        'split': ds.split, 'seed': ds.seed,
        'augmentation': ds.augmentation, 'encoding': ds.encoding,
    }


# ---------------------------------------------------------------------------
# サンプル閲覧(取込データ)
# ---------------------------------------------------------------------------
def _split_counts(ds: ImportedDataset) -> dict[str, int]:
    idx = _split_indices(ds.num_data, ds.split, ds.seed)
    return {k: len(v) for k, v in idx.items()}


@router.get('/api/datasets/imported/{dataset_id_slug}/samples')
def imported_samples(
    dataset_id_slug: str, page: int = 0, page_size: int = DEFAULT_PAGE_SIZE,
    split: str = 'all',
):
    """取込データセットのサンプルをページングして返す(dataset_samples互換の形)。"""
    from .dataset_samples import DatasetSampleItem, DatasetSamplesResponse, _tensor_to_png_data_url

    if page < 0:
        raise HTTPException(status_code=422, detail='pageは0以上を指定してください')
    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise HTTPException(status_code=422, detail=f'page_sizeは1〜{MAX_PAGE_SIZE}で指定してください')
    _load_registry()
    dataset_id = f'{IMPORTED_PREFIX}{dataset_id_slug}'
    ds = _registry.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f'取込データセットが見つかりません: {dataset_id}')
    x_np, y_np = _load_arrays(dataset_id)

    if split != 'all':
        sel = _split_indices(ds.num_data, ds.split, ds.seed).get(split)
        if sel is None:
            raise HTTPException(status_code=422, detail=f'不明なsplit: {split}')
        order = sel
    else:
        order = list(range(ds.num_data))

    num_data = len(order)
    num_pages = max(1, (num_data + page_size - 1) // page_size)
    start = page * page_size
    end = min(start + page_size, num_data)
    is_image = ds.input_shape is not None

    samples = []
    for pos in range(start, end):
        i = order[pos]
        if is_image:
            t = torch.from_numpy(x_np[i]).float().reshape(*ds.input_shape)
            samples.append(DatasetSampleItem(
                index=i, image=_tensor_to_png_data_url(t), label=int(y_np[i]),
            ))
        elif ds.kind == 'classification':
            samples.append(DatasetSampleItem(
                index=i, inputs=[int(round(v)) for v in x_np[i].tolist()], label=int(y_np[i]),
            ))
        else:
            samples.append(DatasetSampleItem(
                index=i,
                inputs=[int(round(v)) for v in x_np[i].tolist()],
                outputs=[int(round(v)) for v in np.atleast_1d(y_np[i]).tolist()],
            ))

    # クラス分布
    if ds.kind == 'classification':
        dist: dict[str, int] = {}
        for i in order:
            k = str(int(y_np[i]))
            dist[k] = dist.get(k, 0) + 1
    else:
        dist = {}
        for i in order:
            k = ','.join(str(int(round(v))) for v in np.atleast_1d(y_np[i]).tolist())
            dist[k] = dist.get(k, 0) + 1

    return DatasetSamplesResponse(
        dataset_id=dataset_id,
        kind='image' if is_image else 'logic',
        num_data=num_data,
        num_columns=ds.in_width,
        page=page,
        page_size=page_size,
        num_pages=num_pages,
        samples=samples,
        class_distribution=dist,
        class_distribution_approx=False,
    )


# ---------------------------------------------------------------------------
# 入力エンコーディングプレビュー
# ---------------------------------------------------------------------------
def encode_values(
    values: list[float], mode: str, threshold: float = 0.5, bits: int = 4,
) -> list[list[int]]:
    """実数値列 → 0/1ビット列(vision.py の BinarizeTransform/ThermometerTransform 規約準拠)。

    - binarize : bit = 1 if v > threshold
    - thermometer: bit_j = 1 if v > (j+1)/(bits+1)  for j in range(bits)
    - none     : v をそのまま四捨五入した1ビット
    """
    out: list[list[int]] = []
    for v in values:
        if mode == 'thermometer':
            out.append([1 if v > (j + 1) / (bits + 1) else 0 for j in range(bits)])
        elif mode == 'binarize':
            out.append([1 if v > threshold else 0])
        else:
            out.append([int(round(v))])
    return out


class EncodingPreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    values: list[float]
    mode: str = 'binarize'
    threshold: float = 0.5
    bits: int = 4


class EncodingPreviewResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: str
    bits_per_value: int
    grid: list[list[int]]


@router.post('/api/datasets/encoding-preview', response_model=EncodingPreviewResponse)
def encoding_preview(body: EncodingPreviewRequest):
    """実数値列のエンコード結果(0/1グリッド)を返す。"""
    if body.mode not in ('binarize', 'thermometer', 'none'):
        raise HTTPException(status_code=422, detail=f'不明なエンコードmode: {body.mode}')
    if body.mode == 'thermometer' and body.bits < 1:
        raise HTTPException(status_code=422, detail='thermometerのbitsは1以上を指定してください')
    grid = encode_values(body.values, body.mode, body.threshold, body.bits)
    bits_per = len(grid[0]) if grid else (body.bits if body.mode == 'thermometer' else 1)
    return EncodingPreviewResponse(mode=body.mode, bits_per_value=bits_per, grid=grid)


# ---------------------------------------------------------------------------
# テスト補助
# ---------------------------------------------------------------------------
def _reset_for_test() -> None:
    """テスト専用: メモリレジストリを空にする(ディスクは触らない)。

    テストは STORE_ROOT をtmp_pathへ monkeypatch した上で本関数を呼び、
    レジストリを再走査させる(実データ outputs/editor_datasets には触れない)。
    """
    global _loaded
    _registry.clear()
    _loaded = False
