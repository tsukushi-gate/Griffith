"""推論プローブAPI。

REST:
    POST /api/probe  body: {inputs: [[0,1,...]], mode: 'hard'|'soft'|'both'}
        → ノード別の中間値(hard=ビット厳密int、soft=小数4桁丸めfloat)

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入):
    from . import probe
    app.include_router(probe.configure(STATE))
"""

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from griffith_logic.ir import build_torch, ir_from_dict, validate
from griffith_logic.ir.probe import probe as probe_hard
from griffith_logic.ir.probe import probe_soft

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


class ProbeRequest(BaseModel):
    inputs: list[list[int]]
    mode: str = Field(default='both', pattern='^(hard|soft|both)$')


def _current_graph():
    """STATE.irをIRGraphへ変換して返す(未設定400/不正422)。"""
    if _STATE is None or _STATE.ir is None:
        raise HTTPException(
            status_code=400, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    try:
        graph = ir_from_dict(_STATE.ir)
        validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')
    return graph


def _soft_model(graph) -> tuple[torch.nn.Module, bool]:
    """soft実行に使うモデルと「一時構築か」を返す。

    訓練セッションがrunning中はSTATE.modelに触らない(eval切替が学習の
    連続緩和forwardを壊す)ため、IRから独立モデルを構築する。
    未構築・IR編集済み(ir_dirty)のときも同様にbuild_torchで直接組む
    (入力はすでに平坦なビット列である前提。Flatten前置は不要)。
    """
    sessions = getattr(_STATE, 'sessions', None)
    active = sessions.active_session() if sessions is not None else None
    running = active is not None and active.state in ('starting', 'running')
    dirty = bool(getattr(_STATE, 'ir_dirty', False))
    if _STATE.model is not None and not running and not dirty:
        return _STATE.model, False
    try:
        return build_torch(graph), True
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=f'モデル構築失敗: {e}')


@router.post('/api/probe')
def probe_endpoint(req: ProbeRequest):
    """全ノードの中間値を返す(訓練running中でも安全に呼べる)。"""
    graph = _current_graph()
    if not graph.inputs:
        raise HTTPException(status_code=422, detail='IRに入力ポートがない')
    in_width = graph.inputs[0].width
    if not req.inputs:
        raise HTTPException(status_code=422, detail='inputsが空')
    if any(len(row) != in_width for row in req.inputs):
        raise HTTPException(
            status_code=422,
            detail=f'入力幅が不一致(期待={in_width})',
        )
    if any(v not in (0, 1) for row in req.inputs for v in row):
        raise HTTPException(status_code=422, detail='入力は0/1のみ')

    out: dict = {
        'input_width': in_width,
        'batch': len(req.inputs),
        'mode': req.mode,
    }

    if req.mode in ('hard', 'both'):
        try:
            trace = probe_hard(graph, np.asarray(req.inputs, dtype=np.int64))
        except NotImplementedError as e:
            raise HTTPException(status_code=400, detail=f'hardプローブ不可: {e}')
        out['hard'] = {nid: v.tolist() for nid, v in trace.items()}

    if req.mode in ('soft', 'both'):
        model, temporary = _soft_model(graph)
        was_training = model.training
        model.eval()
        try:
            x = torch.tensor(req.inputs, dtype=torch.float32)
            trace = probe_soft(model, graph, x)
        except (TypeError, ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=f'softプローブ失敗: {e}')
        finally:
            if not temporary:
                model.train(was_training)
        out['soft'] = {
            nid: [[round(float(v), 4) for v in row] for row in t.tolist()]
            for nid, t in trace.items()
        }

    return out
