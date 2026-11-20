"""NAS(構造自動探索)API。

REST(いずれもextra='forbid'スキーマでドリフト検出):
    POST /api/nas/start          — バックグラウンドスレッドで探索を開始/再開
    GET  /api/nas/status         — 探索の状態(running/done/stopped/idle)
    GET  /api/nas/trials         — trial一覧(acc×total_luts×logic_depth×gates)
    POST /api/nas/stop           — 実行中の探索へ停止要求
    GET  /api/nas/trials/{id}/ir — そのtrialのIR JSON(EDITで開く用)

本体・オーケストレータ(tools/nas/search.py)は呼び出すだけ。探索は小規模
(CPU逐次)前提で、1プロセス1探索を単一のバックグラウンドスレッドで回す。

main.py への組み込み(stats.py 等と同じ configure 注入方式):
    from . import nas
    app.include_router(nas.configure(STATE))
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from tools.nas.search import (
    SearchConfig,
    SearchSpace,
    Trial,
    load_ledger,
    load_trial_ir,
    pareto_front,
    run_search,
)

router = APIRouter()

_STATE = None  # main.py の AppState 参照(configureで注入)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


# ------------------------------------------------------------------
# バックグラウンド探索マネージャ(1プロセス1探索)
# ------------------------------------------------------------------


class _NasManager:
    """単一のバックグラウンド探索スレッドを管理する。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cfg: SearchConfig | None = None
        self._state: str = 'idle'  # 'idle'|'running'|'done'|'stopped'|'failed'
        self._error: str | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: SearchConfig) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError('探索は既に実行中です')
            self._cfg = cfg
            self._stop.clear()
            self._state = 'running'
            self._error = None
            self._thread = threading.Thread(
                target=self._run, name=f'nas-{cfg.search_id}', daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            run_search(self._cfg, stop_check=self._stop.is_set)
            self._state = 'stopped' if self._stop.is_set() else 'done'
        except Exception as e:  # 探索スレッドの失敗を状態に残す
            self._state = 'failed'
            self._error = f'{type(e).__name__}: {e}'

    def stop(self) -> None:
        self._stop.set()

    @property
    def cfg(self) -> SearchConfig | None:
        return self._cfg

    @property
    def state(self) -> str:
        # スレッドが終わっていれば running を実状態へ寄せる
        if self._state == 'running' and not self.is_running():
            return 'stopped' if self._stop.is_set() else 'done'
        return self._state

    @property
    def error(self) -> str | None:
        return self._error


_MANAGER = _NasManager()


# ------------------------------------------------------------------
# pydanticスキーマ(extra='forbid')
# ------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class NasStartRequest(_Strict):
    search_id: str = 'default'
    dataset: str = 'logic:xor'
    model_kind: str = 'logic_mlp'
    max_trials: int = Field(default=12, ge=1)
    epochs: int = Field(default=5, ge=1)
    seed: int = 0
    n_random: int = Field(default=4, ge=0)
    time_budget_sec: float | None = None
    out_root: str | None = None
    layers_choices: list[int] | None = None
    width_choices: list[int] | None = None
    n_inputs_choices: list[int] | None = None
    strategy: str = 'evolution'  # 'evolution'(既定) | 'nsga2'


class NasStartResponse(_Strict):
    search_id: str
    state: str


class NasStatusResponse(_Strict):
    search_id: str | None
    state: str
    n_trials: int
    max_trials: int | None
    pareto: list[int]
    error: str | None


class NasTrialRow(_Strict):
    trial_id: int
    acc: float
    best_acc: float
    total_luts: int
    logic_depth: int
    gates: int
    num_params: int
    partial: bool
    state: str
    parent_id: int | None
    genome: dict


class NasTrialsResponse(_Strict):
    search_id: str | None
    trials: list[NasTrialRow]
    pareto: list[int]


class NasIrResponse(_Strict):
    trial_id: int
    ir: dict


# ------------------------------------------------------------------
# エンドポイント
# ------------------------------------------------------------------


def _build_config(req: NasStartRequest) -> SearchConfig:
    space = SearchSpace()
    if req.layers_choices:
        space.layers_choices = tuple(req.layers_choices)
    if req.width_choices:
        space.width_choices = tuple(req.width_choices)
    if req.n_inputs_choices:
        space.n_inputs_choices = tuple(req.n_inputs_choices)
    kwargs = dict(
        search_id=req.search_id,
        dataset=req.dataset,
        model_kind=req.model_kind,
        max_trials=req.max_trials,
        epochs=req.epochs,
        seed=req.seed,
        n_random=req.n_random,
        time_budget_sec=req.time_budget_sec,
        space=space,
        strategy=req.strategy,
    )
    if req.out_root is not None:
        kwargs['out_root'] = req.out_root
    return SearchConfig(**kwargs)


@router.post('/api/nas/start', response_model=NasStartResponse)
def nas_start(req: NasStartRequest):
    """探索を開始/再開する(バックグラウンドスレッド)。"""
    cfg = _build_config(req)
    try:
        _MANAGER.start(cfg)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return NasStartResponse(search_id=cfg.search_id, state=_MANAGER.state)


@router.get('/api/nas/status', response_model=NasStatusResponse)
def nas_status():
    """探索の現在状態を返す。"""
    cfg = _MANAGER.cfg
    n_trials = 0
    pareto: list[int] = []
    if cfg is not None:
        ledger = load_ledger(cfg)
        n_trials = len(ledger)
        pareto = pareto_front(ledger)
    return NasStatusResponse(
        search_id=cfg.search_id if cfg else None,
        state=_MANAGER.state,
        n_trials=n_trials,
        max_trials=cfg.max_trials if cfg else None,
        pareto=pareto,
        error=_MANAGER.error,
    )


@router.post('/api/nas/stop', response_model=NasStatusResponse)
def nas_stop():
    """実行中の探索へ停止要求を出す(現trial完了後に停止)。"""
    _MANAGER.stop()
    return nas_status()


def _row(t: Trial) -> NasTrialRow:
    return NasTrialRow(
        trial_id=t.trial_id, acc=t.acc, best_acc=t.best_acc,
        total_luts=t.total_luts, logic_depth=t.logic_depth, gates=t.gates,
        num_params=t.num_params, partial=t.partial, state=t.state,
        parent_id=t.parent_id, genome=t.genome,
    )


@router.get('/api/nas/trials', response_model=NasTrialsResponse)
def nas_trials():
    """trial一覧を返す(台帳JSONLから)。"""
    cfg = _MANAGER.cfg
    if cfg is None:
        return NasTrialsResponse(search_id=None, trials=[], pareto=[])
    ledger = load_ledger(cfg)
    return NasTrialsResponse(
        search_id=cfg.search_id,
        trials=[_row(t) for t in ledger],
        pareto=pareto_front(ledger),
    )


@router.get('/api/nas/trials/{trial_id}/ir', response_model=NasIrResponse)
def nas_trial_ir(trial_id: int):
    """指定trialのIR JSONを返す(EDITで開く用)。"""
    cfg = _MANAGER.cfg
    if cfg is None:
        raise HTTPException(status_code=404, detail='探索が未開始です')
    ir = load_trial_ir(cfg, trial_id)
    if ir is None:
        raise HTTPException(status_code=404, detail=f'trial {trial_id} のIRが見つかりません')
    return NasIrResponse(trial_id=trial_id, ir=ir)
