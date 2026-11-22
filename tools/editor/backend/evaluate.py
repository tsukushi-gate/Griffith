"""一括評価API: EVALUATIONタブのバックエンド。

REST:
    POST /api/evaluate
        body={source:{kind:'run'|'current', run_id?}, dataset, executor?,
              page, page_size}
        → サンプル毎の pred/label/correct/soft_confidence/soft_margin/
          hard_soft_mismatch。ページング+セッションスコープの指紋キャッシュ。
    GET /api/evaluate/export.csv?...
        → 評価キャッシュからCSVを生成する。

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入):
    from . import evaluate
    app.include_router(evaluate.configure(STATE))

設計:
  - 評価対象IRは source.kind で解決する。
      'current' → STATE.ir(現在編集中のIR)
      'run'     → runs.load_run(RUNS_ROOT, run_id)['ir'](学習runのスナップショット)
    executorが渡された場合の優先解決は将来拡張。未対応時は上記
    「現IR/runのIR × 指定dataset」の直接指定へフォールバックする(疎結合)。
  - dataset は datasets.resolve_dataset で解決し、ValueError時は
    dataset_import.resolve_imported へフォールバックする(main.pyの既存パターン踏襲)。
  - 推論は本体の ir/probe を流用する: probe(hard=ビット厳密)と、build_torchで
    組んだ独立モデルの soft forward。IR・本体は非改変(読み取り流用のみ)。
  - 評価キャッシュ上限(設計裁定4): 10,000サンプル または 64MB の小さい方。
    超過時は全量をキャッシュせず、ページ単位でオンデマンド評価する(on_demand=true)。
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from griffith_logic.ir import build_torch, ir_from_dict, simulate, validate

router = APIRouter()

_STATE = None       # main.pyのAppState参照(configureで注入)
_RUNS_ROOT = None   # runスナップショットの探索ルート(テスト時に上書き可)

# --- 評価キャッシュの上限(設計裁定4) ---
CACHE_MAX_SAMPLES = 10_000
CACHE_MAX_BYTES = 64 * 1024 * 1024  # 64MB
# 1サンプルの推定バイト数(cap算出用のラフな上限見積り)
_EST_BYTES_IMAGE = 4096
_EST_BYTES_LOGIC = 256

# ページングの既定値・上限(全件を数リクエストで載せられるよう大きめ)
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 2000

# バッチ評価のミニバッチ幅
_EVAL_BATCH = 256

# セッションスコープの指紋キャッシュ(fingerprint → _EvalResult)
_CACHE: dict[str, _EvalResult] = {}


def configure(state, runs_root=None) -> APIRouter:
    """STATE参照(+テスト用のrunsルート)を注入し、includeすべきrouterを返す。"""
    global _STATE, _RUNS_ROOT
    _STATE = state
    _RUNS_ROOT = runs_root
    _CACHE.clear()
    return router


def _runs_root():
    if _RUNS_ROOT is not None:
        return _RUNS_ROOT
    from .training import RUNS_ROOT
    return RUNS_ROOT


# --- リクエスト/レスポンススキーマ ---

class EvalSource(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: str = Field(pattern='^(run|current)$')
    run_id: str | None = None


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source: EvalSource
    dataset: str
    executor: str | None = None  # 連携用(現状は未使用=source直接指定へフォールバック)
    dataset_params: dict | None = None
    page: int = Field(default=0, ge=0)
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra='forbid')

    index: int
    # classification: label/pred はクラスindex。binary_vector: *_bits がビット列
    label: int | None = None
    pred: int | None = None
    label_bits: list[int] | None = None
    pred_bits: list[int] | None = None
    correct: bool
    soft_confidence: float   # [0,1]: 全体的な確信度
    soft_margin: float       # [0,1]: 最も弱い/僅差の判定マージン(小さいほど危うい)
    hard_soft_mismatch: bool  # hard離散化結果 ≠ soft argmax/しきい値
    input_bits: list[int] | None = None  # 論理系: 反例トレース注入用の入力ビット列
    image: str | None = None             # 画像系: PNGサムネイルのdata URL


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_kind: str
    dataset: str
    executor: str | None
    kind: str  # 'classification' | 'binary_vector'
    num_classes: int | None
    num_samples: int
    page: int
    page_size: int
    num_pages: int
    cached: bool       # 指紋キャッシュヒットで返したか
    on_demand: bool    # 上限超過でページ単位オンデマンド評価か
    cap: int           # キャッシュ上限サンプル数
    accuracy: float | None       # 全体正解率(on_demand時はNone)
    num_correct: int | None
    num_errors: int | None
    num_mismatch: int | None
    confusion: list[list[int]] | None  # classification全体の混同行列(on_demand時はNone)
    fingerprint: str
    items: list[EvalItem]


# --- 評価対象(IR + dataset)の解決 ---

def _resolve_ir(source: EvalSource) -> dict:
    """source から評価対象IR(dict)を解決する。"""
    if source.kind == 'current':
        if _STATE is None or _STATE.ir is None:
            raise HTTPException(
                status_code=400, detail='IR未設定。先に /api/models/demo 等を呼ぶこと',
            )
        return _STATE.ir
    # kind == 'run'
    if not source.run_id:
        raise HTTPException(status_code=422, detail="source.kind='run' には run_id が必要")
    from .runs import load_run

    doc = load_run(_runs_root(), source.run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f'runが存在しない: {source.run_id}')
    ir = doc.get('ir')
    if ir is None:
        raise HTTPException(
            status_code=422, detail=f'runにIRスナップショットがない: {source.run_id}',
        )
    return ir


def _current_graph(ir: dict):
    try:
        graph = ir_from_dict(ir)
        validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')
    return graph


def _resolve_dataset(name: str, params: dict | None):
    """datasets.resolve_dataset →(ValueError時)dataset_import.resolve_imported。"""
    from .datasets import resolve_dataset

    try:
        return resolve_dataset(name, batch_size=1, params=params)
    except ValueError:
        from . import dataset_import

        try:
            return dataset_import.resolve_imported(name, batch_size=1, params=params)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


def _fingerprint(ir: dict, req: EvaluateRequest) -> str:
    """source(IR内容)+dataset+params+executor から指紋キーを作る。"""
    payload = {
        'ir': ir,
        'dataset': req.dataset,
        'dataset_params': req.dataset_params,
        'executor': req.executor,
        'source_kind': req.source.kind,
        'run_id': req.source.run_id,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


# --- 推論・集計 ---

class _EvalResult:
    """全件評価の結果(キャッシュ格納単位)。"""

    def __init__(self, kind: str, num_classes: int | None):
        self.kind = kind
        self.num_classes = num_classes
        self.items: list[dict] = []
        self.confusion: list[list[int]] | None = None
        self.accuracy: float | None = None
        self.num_correct = 0
        self.num_errors = 0
        self.num_mismatch = 0


def _classification_item(
    index: int, hard_scores: np.ndarray, soft_row: torch.Tensor, label: int,
) -> dict:
    """classification 1サンプルの評価行を作る。"""
    pred_hard = int(np.argmax(hard_scores))
    probs = torch.softmax(soft_row, dim=-1)
    k = probs.shape[-1]
    top = torch.topk(probs, min(2, k))
    conf = float(top.values[0])
    margin = float(top.values[0] - top.values[1]) if k >= 2 else conf
    pred_soft = int(torch.argmax(soft_row))
    return {
        'index': index,
        'label': label,
        'pred': pred_hard,
        'correct': pred_hard == label,
        'soft_confidence': round(conf, 4),
        'soft_margin': round(margin, 4),
        'hard_soft_mismatch': pred_hard != pred_soft,
    }


def _binary_item(
    index: int, hard_bits: np.ndarray, soft_row: torch.Tensor, label_bits: list[int],
) -> dict:
    """binary_vector 1サンプルの評価行を作る(しきい値0.5=_bit_acc規約)。"""
    pred_hard = [int(v) for v in hard_bits.tolist()]
    soft = soft_row.detach().to(torch.float32)
    pred_soft = [1 if float(v) >= 0.5 else 0 for v in soft]
    # 各ビットの確信度 = |s-0.5|*2 を [0,1] にクランプ
    per_bit = (soft - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
    conf = float(per_bit.mean()) if per_bit.numel() else 1.0
    margin = float(per_bit.min()) if per_bit.numel() else 1.0
    correct = pred_hard == label_bits
    mismatch = pred_hard != pred_soft
    return {
        'index': index,
        'label_bits': label_bits,
        'pred_bits': pred_hard,
        'correct': correct,
        'soft_confidence': round(conf, 4),
        'soft_margin': round(margin, 4),
        'hard_soft_mismatch': mismatch,
    }


def _iter_batches(n: int, batch: int):
    for start in range(0, n, batch):
        yield start, min(start + batch, n)


def _eval_range(graph, model, dataset, kind, in_width, is_image, start, end) -> list[dict]:
    """[start,end) のサンプルを評価して行リストを返す。"""
    xs, ys, raw = [], [], []
    for i in range(start, end):
        x, y = dataset[i]
        raw.append(x)
        xs.append(x.reshape(-1).to(torch.float32))
        ys.append(y)
    x_float = torch.stack(xs)  # (B, in_width)
    x_bits = (x_float.numpy() > 0.5).astype(np.int64)
    hard = simulate(graph, x_bits)  # (B, out_width or num_classes)
    with torch.no_grad():
        soft = model(x_float)
    out: list[dict] = []
    for j in range(end - start):
        index = start + j
        if kind == 'classification':
            label = int(ys[j])
            item = _classification_item(index, np.asarray(hard[j]), soft[j], label)
        else:
            label_bits = [int(v) for v in torch.as_tensor(ys[j]).reshape(-1).tolist()]
            item = _binary_item(index, np.asarray(hard[j]), soft[j], label_bits)
        # 入力表現(サムネイル or ビット列)を付与
        if is_image:
            from .dataset_samples import _tensor_to_png_data_url

            item['image'] = _tensor_to_png_data_url(raw[j])
        else:
            item['input_bits'] = [int(v) for v in x_bits[j].tolist()]
        out.append(item)
    return out


def _cap_for(is_image: bool) -> int:
    per = _EST_BYTES_IMAGE if is_image else _EST_BYTES_LOGIC
    return min(CACHE_MAX_SAMPLES, max(1, CACHE_MAX_BYTES // per))


def _full_evaluate(graph, model, dataset, kind, in_width, is_image, n) -> _EvalResult:
    """全件評価して集計付きの結果を作る。"""
    res = _EvalResult(kind, int(graph.outputs[0].width) if kind == 'classification' else None)
    num_classes = res.num_classes
    if kind == 'classification' and num_classes:
        conf = [[0] * num_classes for _ in range(num_classes)]
    else:
        conf = None
    for start, end in _iter_batches(n, _EVAL_BATCH):
        rows = _eval_range(graph, model, dataset, kind, in_width, is_image, start, end)
        for item in rows:
            if item['correct']:
                res.num_correct += 1
            else:
                res.num_errors += 1
            if item['hard_soft_mismatch']:
                res.num_mismatch += 1
            if conf is not None:
                conf[item['label']][item['pred']] += 1
        res.items.extend(rows)
    res.confusion = conf
    res.accuracy = round(res.num_correct / n, 6) if n else None
    return res


def _page_response(
    req: EvaluateRequest, ir: dict, kind: str, num_classes: int | None,
    n: int, fingerprint: str, cached: bool, on_demand: bool, cap: int,
    items: list[dict], res: _EvalResult | None,
) -> EvaluateResponse:
    num_pages = max(1, (n + req.page_size - 1) // req.page_size)
    return EvaluateResponse(
        source_kind=req.source.kind,
        dataset=req.dataset,
        executor=req.executor,
        kind=kind,
        num_classes=num_classes,
        num_samples=n,
        page=req.page,
        page_size=req.page_size,
        num_pages=num_pages,
        cached=cached,
        on_demand=on_demand,
        cap=cap,
        accuracy=res.accuracy if res else None,
        num_correct=res.num_correct if res else None,
        num_errors=res.num_errors if res else None,
        num_mismatch=res.num_mismatch if res else None,
        confusion=res.confusion if res else None,
        fingerprint=fingerprint,
        items=[EvalItem(**it) for it in items],
    )


@router.post('/api/evaluate', response_model=EvaluateResponse)
def evaluate_endpoint(req: EvaluateRequest):
    """現IR/runのIR × dataset を一括評価し、ページ分の結果を返す。"""
    ir = _resolve_ir(req.source)
    graph = _current_graph(ir)
    resolved = _resolve_dataset(req.dataset, req.dataset_params)

    kind = resolved.kind
    if kind not in ('classification', 'binary_vector'):
        raise HTTPException(status_code=400, detail=f'未対応のタスク種別: {kind}')

    loader = resolved.loader_factory()
    dataset = loader.dataset
    n = len(dataset)
    if n == 0:
        raise HTTPException(status_code=400, detail='データセットが空です')
    # 入力が2次元以上(C,H,W)ならサムネイル対象の画像とみなす(論理系は平坦ビット列)
    is_image = torch.as_tensor(dataset[0][0]).dim() >= 2
    in_width = resolved.in_width
    cap = _cap_for(is_image)
    on_demand = n > cap
    fingerprint = _fingerprint(ir, req)

    try:
        model = build_torch(graph)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=f'モデル構築失敗: {e}')
    model.eval()

    num_classes = int(graph.outputs[0].width) if kind == 'classification' else None

    start = req.page * req.page_size
    end = min(start + req.page_size, n)

    if on_demand:
        # 大規模: キャッシュせずページ分だけ評価(集計・混同行列なし)
        items = _eval_range(graph, model, dataset, kind, in_width, is_image, start, end)
        return _page_response(
            req, ir, kind, num_classes, n, fingerprint, False, True, cap, items, None,
        )

    # 通常: 全件評価をキャッシュし、ページ分を切り出して返す
    cached = fingerprint in _CACHE
    if cached:
        res = _CACHE[fingerprint]
    else:
        res = _full_evaluate(graph, model, dataset, kind, in_width, is_image, n)
        _CACHE[fingerprint] = res
    page_items = res.items[start:end]
    return _page_response(
        req, ir, kind, num_classes, n, fingerprint, cached, False, cap, page_items, res,
    )


# --- CSVエクスポート ---

_CSV_HEADER = (
    'index,correct,label,pred,soft_confidence,soft_margin,hard_soft_mismatch'
)


def _csv_row(it: dict) -> str:
    label = it.get('label')
    pred = it.get('pred')
    if label is None and it.get('label_bits') is not None:
        label = ''.join(str(v) for v in it['label_bits'])
        pred = ''.join(str(v) for v in it['pred_bits'])
    return (
        f"{it['index']},{int(it['correct'])},{label},{pred},"
        f"{it['soft_confidence']},{it['soft_margin']},{int(it['hard_soft_mismatch'])}"
    )


@router.post('/api/evaluate/export.csv', response_class=PlainTextResponse)
def evaluate_export_csv(req: EvaluateRequest):
    """評価結果をCSVで返す。指紋キャッシュがあれば流用する。"""
    ir = _resolve_ir(req.source)
    graph = _current_graph(ir)
    resolved = _resolve_dataset(req.dataset, req.dataset_params)
    kind = resolved.kind
    if kind not in ('classification', 'binary_vector'):
        raise HTTPException(status_code=400, detail=f'未対応のタスク種別: {kind}')
    loader = resolved.loader_factory()
    dataset = loader.dataset
    n = len(dataset)
    if n == 0:
        raise HTTPException(status_code=400, detail='データセットが空です')
    is_image = torch.as_tensor(dataset[0][0]).dim() >= 2
    cap = _cap_for(is_image)
    if n > cap:
        raise HTTPException(
            status_code=400,
            detail=f'CSVエクスポートは{cap}サンプルまで(現在{n})',
        )
    fingerprint = _fingerprint(ir, req)
    if fingerprint in _CACHE:
        res = _CACHE[fingerprint]
    else:
        try:
            model = build_torch(graph)
        except (ValueError, NotImplementedError) as e:
            raise HTTPException(status_code=400, detail=f'モデル構築失敗: {e}')
        model.eval()
        res = _full_evaluate(graph, model, dataset, kind, resolved.in_width, is_image, n)
        _CACHE[fingerprint] = res

    buf = io.StringIO()
    buf.write(_CSV_HEADER + '\n')
    for it in res.items:
        buf.write(_csv_row(it) + '\n')
    return PlainTextResponse(
        content=buf.getvalue(),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename="evaluation.csv"'},
    )
