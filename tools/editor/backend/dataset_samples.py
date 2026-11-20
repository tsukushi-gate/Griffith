"""データセットサンプル閲覧API。

REST:
    GET /api/datasets/{dataset_id}/samples?page=&page_size=
        → ページ分のサンプル(画像=PNG data URL、論理=入力/出力ビット列)
          + num_data / num_columns / class_distribution

main.py への組み込み(循環importを避けるためconfigure不要、STATE非依存):
    from . import dataset_samples
    app.include_router(dataset_samples.router)

設計:
  - `datasets.py` の `list_datasets`/`resolve_dataset`/`LOGIC_DATASETS` を読み取り流用する
    (datasets.py・griffith_logic本体は非改変)。
  - 論理データセット(binary_vector)は `LOGIC_DATASETS[key]()` を repeat=1 で直接呼び、
    resolve_dataset の学習用repeat水増し(LOGIC_TARGET_ROWS)を経由しない
    (閲覧は真理値表そのものの行数で見せたいため)。
  - 画像系(classification)は resolve_dataset の loader_factory 経由でDatasetを取得する
    (mnist/llvit共通。未ダウンロード等の例外はそのまま503へ変換)。
  - class_distribution はデータ件数が大きい場合、上限件数のサンプリングで近似する
    (`approx: true` を添えて示す)。
"""

from __future__ import annotations

import base64
import io

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# class_distributionを全件走査する上限(超えたら先頭からこの件数だけ見て近似する)
CLASS_DIST_SCAN_LIMIT = 5000

# ページングの既定値・上限
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200


class DatasetSampleItem(BaseModel):
    """1サンプル分の表示用データ。"""

    model_config = ConfigDict(extra='forbid')

    index: int
    image: str | None = None      # 画像系: PNG data URL
    label: int | None = None      # 画像系: クラスindex
    inputs: list[int] | None = None   # 論理系: 入力ビット列
    outputs: list[int] | None = None  # 論理系: 出力ビット列


class DatasetSamplesResponse(BaseModel):
    """GET /api/datasets/{id}/samples のレスポンス。"""

    model_config = ConfigDict(extra='forbid')

    dataset_id: str
    kind: str  # 'image' | 'logic'
    num_data: int
    num_columns: int
    page: int
    page_size: int
    num_pages: int
    samples: list[DatasetSampleItem]
    class_distribution: dict[str, int]
    class_distribution_approx: bool = False


def _catalog_entry(dataset_id: str) -> dict:
    """list_datasets()から該当エントリを引く。未知IDは404(日本語)。"""
    from .datasets import list_datasets

    for entry in list_datasets():
        if entry['name'] == dataset_id:
            return entry
    raise HTTPException(
        status_code=404, detail=f'不明なデータセット: {dataset_id}'
    )


def _tensor_to_png_data_url(tensor: torch.Tensor) -> str:
    """(C,H,W) or (H,W) の0..1floatテンソルをPNG data URLへ変換する。"""
    from PIL import Image

    t = tensor.detach().to(torch.float32).clamp(0, 1)
    if t.dim() == 3:
        c = t.shape[0]
        if c == 1:
            arr = (t[0] * 255).round().byte().numpy()
            img = Image.fromarray(arr, mode='L')
        else:
            arr = (t[:3].permute(1, 2, 0) * 255).round().byte().numpy()
            img = Image.fromarray(arr, mode='RGB')
    elif t.dim() == 2:
        arr = (t * 255).round().byte().numpy()
        img = Image.fromarray(arr, mode='L')
    else:
        raise ValueError(f'画像化できないテンソル形状: {tuple(t.shape)}')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _logic_dataset(dataset_id: str, entry: dict):
    """logic:<key> をrepeat=1の生データセットとして解決する。"""
    from griffith_logic.datasets import LOGIC_DATASETS

    key = dataset_id.split(':', 1)[1]
    if key not in LOGIC_DATASETS:
        raise HTTPException(
            status_code=404, detail=f'不明なデータセット: {dataset_id}'
        )
    spec = LOGIC_DATASETS[key]()
    return spec.dataset  # TensorDataset(X, Y)


def _image_dataset(dataset_id: str):
    """classification系(mnist/llvit等)をDatasetとして解決する(resolve_dataset流用)。"""
    from .datasets import resolve_dataset

    try:
        resolved = resolve_dataset(dataset_id, batch_size=1)
        loader = resolved.loader_factory()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - 未ダウンロード等の外部要因を503へ変換
        raise HTTPException(
            status_code=503,
            detail=f'データセットの読み込みに失敗しました({dataset_id}): {e}',
        )
    return loader.dataset


def _class_distribution(labels: list) -> dict[str, int]:
    """ラベル列(既に走査上限で切り詰め済み)からクラス→件数の分布を作る。"""
    dist: dict[str, int] = {}
    for v in labels:
        key = str(v)
        dist[key] = dist.get(key, 0) + 1
    return dist


@router.get('/api/datasets/{dataset_id}/samples', response_model=DatasetSamplesResponse)
def get_dataset_samples(dataset_id: str, page: int = 0, page_size: int = DEFAULT_PAGE_SIZE):
    """データセットのサンプルをページングして返す。"""
    if page < 0:
        raise HTTPException(status_code=422, detail='pageは0以上を指定してください')
    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f'page_sizeは1〜{MAX_PAGE_SIZE}の範囲で指定してください',
        )

    entry = _catalog_entry(dataset_id)
    is_logic = dataset_id.startswith('logic:')

    if is_logic:
        dataset = _logic_dataset(dataset_id, entry)
        num_columns = entry['in_width']
    else:
        dataset = _image_dataset(dataset_id)
        num_columns = entry['in_width']

    num_data = len(dataset)
    num_pages = max(1, (num_data + page_size - 1) // page_size)
    start = page * page_size
    end = min(start + page_size, num_data)

    samples: list[DatasetSampleItem] = []
    for i in range(start, end):
        x, y = dataset[i]
        if is_logic:
            samples.append(DatasetSampleItem(
                index=i,
                inputs=[int(v) for v in x.reshape(-1).tolist()],
                outputs=[int(v) for v in y.reshape(-1).tolist()],
            ))
        else:
            samples.append(DatasetSampleItem(
                index=i,
                image=_tensor_to_png_data_url(x),
                label=int(y),
            ))

    scan_end = min(num_data, CLASS_DIST_SCAN_LIMIT)
    approx = num_data > CLASS_DIST_SCAN_LIMIT
    if is_logic:
        # 出力ビット列の組み合わせ(カンマ区切り文字列)を「クラス」とみなして分布を取る
        labels = [
            ','.join(str(int(v)) for v in dataset[i][1].reshape(-1).tolist())
            for i in range(scan_end)
        ]
    else:
        labels = [int(dataset[i][1]) for i in range(scan_end)]
    class_distribution = _class_distribution(labels)

    return DatasetSamplesResponse(
        dataset_id=dataset_id,
        kind='logic' if is_logic else 'image',
        num_data=num_data,
        num_columns=num_columns,
        page=page,
        page_size=page_size,
        num_pages=num_pages,
        samples=samples,
        class_distribution=class_distribution,
        class_distribution_approx=approx,
    )
