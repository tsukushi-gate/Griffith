"""学習の途中停止・再開ハーネス。

設計方針:
    学習を途中で止めても、そこからすぐ再開できるように重みを管理する。
    wandbに重みを定期的に保存し、どの時点からでも再開できるようにする
    (すぐ終わる小さいモデルは対象外)。

提供するもの:
    - ``save_checkpoint`` / ``load_checkpoint``: atomic書き（tmp→rename）+ RNG状態の保存復元
    - ``CheckpointManager``: 一定間隔での定期保存 + wandb Artifactアップロード（latest/best）
    - ``load_latest``: ローカル run_dir → wandb Artifact("latest") の順で解決
    - ``resume_or_init``: ``--resume`` フラグ運用の定型
    - ``install_graceful_save``: SIGTERM/SIGINT受信時に最終ckptを保存してから終了

torch非依存の実験（進化系など）でも ``state=`` に任意の辞書を渡すだけで使える。

使用例（PyTorch学習ループ）::

    from experiments import ckpt_utils

    parser.add_argument('--resume', nargs='?', const='auto', default=None)
    args = parser.parse_args()

    run_dir = 'checkpoints/my_run'
    mgr = ckpt_utils.CheckpointManager(run_dir, name='my_run', interval_sec=300)

    def state_fn():
        return {'model': model, 'optimizer': optimizer, 'extra': {'epoch': epoch}}

    start_step = 0
    resumed = ckpt_utils.load_latest(run_dir) if args.resume else None
    if resumed is not None:
        payload, start_step = resumed
        ckpt_utils.restore_into(payload, model=model, optimizer=optimizer)

    ckpt_utils.install_graceful_save(mgr, state_fn)

    for step in range(start_step, total_steps):
        ...
        mgr.maybe_save(step, state_fn)          # 5分経過時のみ保存+アップロード
        if metric > best:
            mgr.save_best(step, state_fn, metric=metric)

使用例（torchを使わない進化系）::

    mgr.maybe_save(gen, lambda: {'state': {'population': pop, 'rng': rng.bit_generator.state}})

削除に関する方針（プロジェクト規約: rm等の削除は絶対禁止）:
    チェックポイントは ``<run_dir>/ckpt_<step>.pt`` に**連番で追記保存**する。
    ``keep`` を超えた古い世代も**削除しない**（規約により削除操作を一切行わない）。
    最新位置は ``<run_dir>/latest.json`` のポインタで示すため、latest のコピーで
    ディスクを二重消費することもない。ディスク肥大が気になる場合は
    ``interval_sec`` を大きくするか、保存先を ``checkpoints/`` 配下に固定して
    人手で棚卸しすること（自動削除は行わない）。
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

try:  # torchはあれば使う（進化系実験ではなくてもよい）
    import torch
except ImportError:  # pragma: no cover - torch必須環境のため通常到達しない
    torch = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]


DEFAULT_INTERVAL_SEC = 300  # 既定5分間隔（training-scriptスキル §8）
LATEST_POINTER = 'latest.json'
BEST_POINTER = 'best.json'


# ============================================================
# RNG状態
# ============================================================

def capture_rng_state() -> dict:
    """python/numpy/torch のRNG状態をまとめて取得する。"""
    state: dict[str, Any] = {'python': random.getstate()}
    if np is not None:
        state['numpy'] = np.random.get_state()
    if torch is not None:
        state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """``capture_rng_state`` が返した状態を復元する（欠けている項目は無視）。"""
    if not state:
        return
    if 'python' in state:
        py = state['python']
        # torch.save/loadでtupleがlistに落ちることがあるため戻す
        random.setstate(tuple(py) if isinstance(py, list) else py)
    if np is not None and 'numpy' in state:
        np.random.set_state(state['numpy'])
    if torch is not None and 'torch' in state:
        torch.set_rng_state(_as_byte_tensor(state['torch']))
    if torch is not None and 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state['torch_cuda']])


def _as_byte_tensor(value):
    if torch is not None and torch.is_tensor(value):
        return value.cpu().to(torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


# ============================================================
# 保存・読み込み
# ============================================================

def save_checkpoint(
    path: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    step: int,
    state: dict | None = None,
    extra: dict | None = None,
    rng: bool = True,
) -> str:
    """チェックポイントをatomicに保存する（tmp→rename）。

    Args:
        path: 保存先パス（``.pt``）
        model: ``state_dict()`` を持つモデル（省略可）
        optimizer: オプティマイザ（省略可）
        scheduler: スケジューラ（省略可）
        step: 現在のステップ/エポック/世代
        state: torch非依存の実験向け。任意の辞書をそのまま保存する
        extra: 付随情報（args, best_metric, config 等）
        rng: True なら python/numpy/torch のRNG状態も保存する

    Returns:
        保存したパス
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError('save_checkpointにはtorchが必要（torch.saveでシリアライズする）')
    payload: dict[str, Any] = {'step': int(step), 'saved_at': time.time()}
    if model is not None:
        payload['model'] = model.state_dict()
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if state is not None:
        payload['state'] = state
    if extra is not None:
        payload['extra'] = extra
    if rng:
        payload['rng'] = capture_rng_state()

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic rename
    finally:
        if os.path.exists(tmp):
            # 例外時の残骸のみ後始末（自分が今作ったtmpに限定）
            os.unlink(tmp)
    return path


def load_checkpoint(path: str, map_location: str = 'cpu', rng: bool = True) -> dict:
    """チェックポイントを読み込む（RNG状態はrng=Trueなら即復元）。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError('load_checkpointにはtorchが必要')
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if rng:
        restore_rng_state(payload.get('rng'))
    return payload


def restore_into(payload: dict, *, model=None, optimizer=None, scheduler=None,
                 strict: bool = True) -> int:
    """payloadの state_dict をモデル/オプティマイザへ流し込み、stepを返す。"""
    if model is not None and 'model' in payload:
        model.load_state_dict(payload['model'], strict=strict)
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload.get('scheduler') is not None:
        scheduler.load_state_dict(payload['scheduler'])
    return int(payload.get('step', 0))


# ============================================================
# ポインタファイル
# ============================================================

def _write_pointer(run_dir: str, filename: str, path: str, step: int) -> None:
    pointer = os.path.join(run_dir, filename)
    tmp = f'{pointer}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'path': os.path.basename(path), 'step': int(step),
                   'updated_at': time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pointer)


def _read_pointer(run_dir: str, filename: str) -> dict | None:
    pointer = os.path.join(run_dir, filename)
    if not os.path.exists(pointer):
        return None
    try:
        with open(pointer, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    path = os.path.join(run_dir, data.get('path', ''))
    if not os.path.exists(path):
        return None
    data['abs_path'] = path
    return data


def _scan_checkpoints(run_dir: str) -> list[str]:
    """run_dir内の ckpt_*.pt をstep昇順で返す。"""
    if not os.path.isdir(run_dir):
        return []
    names = [n for n in os.listdir(run_dir)
             if n.startswith('ckpt_') and n.endswith('.pt')]
    return [os.path.join(run_dir, n) for n in sorted(names)]


# ============================================================
# CheckpointManager
# ============================================================

class CheckpointManager:
    """一定間隔での定期保存 + wandb Artifactアップロードを行うマネージャ。

    Args:
        run_dir: チェックポイント保存先ディレクトリ
        wandb_run: wandbのrunオブジェクト。Noneなら ``wandb.run`` を自動参照
        interval_sec: 保存間隔（秒）。既定5分
        keep: 「最新」とみなす世代数。**超過分も削除しない**（プロジェクト規約）
        name: Artifact名のベース。Noneならrun名またはrun_dirのbasename
        upload: Falseならwandbアップロードを行わない（ローカルのみ）
        artifact_factory: テスト用フック。既定は ``wandb.Artifact``
    """

    def __init__(
        self,
        run_dir: str,
        wandb_run=None,
        interval_sec: int = DEFAULT_INTERVAL_SEC,
        keep: int = 3,
        name: str | None = None,
        upload: bool = True,
        artifact_factory: Callable[..., Any] | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._explicit_run = wandb_run
        self.interval_sec = interval_sec
        self.keep = keep
        self.upload = upload
        self._artifact_factory = artifact_factory
        self._time = time_fn
        self._last_save_at = self._time()
        self.name = name or self._default_name()
        self.saved_paths: list[str] = _scan_checkpoints(run_dir)

    # -- 内部 -------------------------------------------------
    def _default_name(self) -> str:
        run = self._resolve_run()
        if run is not None and getattr(run, 'name', None):
            return str(run.name)
        return os.path.basename(os.path.abspath(self.run_dir))

    def _resolve_run(self):
        if self._explicit_run is not None:
            return self._explicit_run
        if wandb is not None:
            return getattr(wandb, 'run', None)
        return None

    def _make_artifact(self, artifact_name: str, metadata: dict | None):
        factory = self._artifact_factory
        if factory is None:
            if wandb is None:
                return None
            factory = wandb.Artifact
        return factory(artifact_name, type='model', metadata=metadata or {})

    def _upload(self, path: str, aliases: list[str], metadata: dict | None) -> bool:
        if not self.upload:
            return False
        run = self._resolve_run()
        if run is None:
            return False
        artifact = self._make_artifact(f'{_sanitize(self.name)}-ckpt', metadata)
        if artifact is None:
            return False
        artifact.add_file(path)
        run.log_artifact(artifact, aliases=aliases)
        return True

    def _path_for(self, step: int) -> str:
        return os.path.join(self.run_dir, f'ckpt_{int(step):09d}.pt')

    # -- 公開API ----------------------------------------------
    def save(self, step: int, state_fn: Callable[[], dict],
             aliases: list[str] | None = None, metadata: dict | None = None) -> str:
        """間隔に関係なく即保存する（アップロードも行う）。"""
        kwargs = dict(state_fn())
        path = self._path_for(step)
        save_checkpoint(path, step=step, **kwargs)
        self.saved_paths.append(path)
        _write_pointer(self.run_dir, LATEST_POINTER, path, step)
        self._last_save_at = self._time()
        meta = dict(metadata or {})
        meta.setdefault('step', int(step))
        self._upload(path, aliases or ['latest'], meta)
        return path

    def maybe_save(self, step: int, state_fn: Callable[[], dict],
                   force: bool = False, metadata: dict | None = None) -> str | None:
        """``interval_sec`` 経過時のみ保存する。保存したらパス、しなければNone。"""
        if not force and (self._time() - self._last_save_at) < self.interval_sec:
            return None
        return self.save(step, state_fn, metadata=metadata)

    def save_best(self, step: int, state_fn: Callable[[], dict],
                  metric: float | None = None, metadata: dict | None = None) -> str:
        """best として保存する（aliasは ``best`` + ``latest``）。"""
        meta = dict(metadata or {})
        if metric is not None:
            meta['best_metric'] = float(metric)
        path = os.path.join(self.run_dir, 'best.pt')
        kwargs = dict(state_fn())
        save_checkpoint(path, step=step, **kwargs)
        _write_pointer(self.run_dir, BEST_POINTER, path, step)
        self._upload(path, ['best', 'latest'], meta)
        return path

    def stale_paths(self) -> list[str]:
        """keep世代を超えた古いckpt（**削除はしない**。棚卸し用の情報提供）。"""
        if len(self.saved_paths) <= self.keep:
            return []
        return self.saved_paths[:-self.keep]


def _sanitize(name: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_.\-]', '-', name)


# ============================================================
# 再開
# ============================================================

def load_latest(run_dir_or_artifact: str, map_location: str = 'cpu',
                download_dir: str = 'checkpoints', rng: bool = True
                ) -> tuple[dict, int] | None:
    """最新チェックポイントを解決して ``(payload, step)`` を返す。

    解決順:
        1. ``run_dir_or_artifact`` がディレクトリなら ``latest.json`` ポインタ
        2. ポインタが無ければ ``ckpt_*.pt`` の最大step
        3. それも無ければ wandb Artifact ``<run_dir_or_artifact>:latest``

    見つからなければ None。
    """
    target = run_dir_or_artifact
    if os.path.isdir(target):
        pointer = _read_pointer(target, LATEST_POINTER)
        if pointer is not None:
            return load_checkpoint(pointer['abs_path'], map_location, rng), int(pointer['step'])
        candidates = _scan_checkpoints(target)
        if candidates:
            payload = load_checkpoint(candidates[-1], map_location, rng)
            return payload, int(payload.get('step', 0))
        return None
    if os.path.isfile(target):
        payload = load_checkpoint(target, map_location, rng)
        return payload, int(payload.get('step', 0))
    path = _download_artifact(target, download_dir)
    if path is None:
        return None
    payload = load_checkpoint(path, map_location, rng)
    return payload, int(payload.get('step', 0))


def _download_artifact(ref: str, download_dir: str) -> str | None:
    """wandb Artifactからckptを取得する。取得できなければNone。"""
    if wandb is None:
        return None
    if ':' not in ref:
        ref = f'{ref}:latest'
    try:
        if getattr(wandb, 'run', None) is not None:
            artifact = wandb.run.use_artifact(ref)
        else:
            artifact = wandb.Api().artifact(ref)
        root = artifact.download(root=download_dir)
    except Exception as exc:  # noqa: BLE001 - 再開失敗はfatalにしない
        print(f'ckpt_utils: wandb Artifact取得に失敗 {ref}: {exc}')
        return None
    pts = sorted(f for f in os.listdir(root) if f.endswith('.pt'))
    if not pts:
        return None
    return os.path.join(root, pts[-1])


def resume_or_init(args, run_dir: str, init_fn: Callable[[], Any],
                   map_location: str = 'cpu') -> tuple[Any, int, dict | None]:
    """``--resume`` 運用の定型。``(obj, start_step, payload)`` を返す。

    ``args.resume`` の値:
        None/False          → 常に初期化（init_fn）
        'auto' / True / ''  → run_dir から再開を試み、無ければ初期化
        それ以外の文字列    → パス or wandb Artifact参照として解決

    ``init_fn`` は ``(model, optimizer, ...)`` などの任意オブジェクトを返してよい。
    再開できた場合の state_dict の流し込みは呼び出し側で ``restore_into`` を使う。
    """
    resume = getattr(args, 'resume', None)
    if not resume:
        return init_fn(), 0, None
    target = run_dir if resume in (True, 'auto', '') else str(resume)
    found = load_latest(target, map_location=map_location)
    obj = init_fn()
    if found is None:
        print(f'ckpt_utils: 再開元が見つからないため初期化 target={target}')
        return obj, 0, None
    payload, step = found
    print(f'ckpt_utils: 再開 target={target} step={step}')
    return obj, step, payload


# ============================================================
# graceful shutdown
# ============================================================

def install_graceful_save(manager: CheckpointManager,
                          state_fn: Callable[[], dict],
                          step_fn: Callable[[], int] | None = None,
                          signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
                          exit_code: int = 0) -> Callable[[], None]:
    """SIGTERM/SIGINT受信時に最終ckptを保存してから終了するハンドラを登録する。

    途中打ち切りは ``kill -9`` ではなく ``kill``（SIGTERM）で行うこと。

    Returns:
        手動で最終保存を行うための関数（正常終了時にも呼べる）。
    """
    state = {'saving': False}

    def _final_save() -> None:
        if state['saving']:
            return
        state['saving'] = True
        step = step_fn() if step_fn is not None else _current_step(manager)
        try:
            path = manager.save(step, state_fn, aliases=['latest', 'interrupted'])
            print(f'ckpt_utils: 最終チェックポイントを保存 {path} (step={step})')
        except Exception as exc:  # noqa: BLE001
            print(f'ckpt_utils: 最終保存に失敗: {exc}')
        finally:
            state['saving'] = False

    def _handler(signum, _frame):  # pragma: no cover - シグナル経路
        print(f'ckpt_utils: signal {signum} 受信 → 最終チェックポイントを保存して終了')
        _final_save()
        sys.exit(exit_code)

    for sig in signals:
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # メインスレッド以外では登録できない
            print(f'ckpt_utils: signal {sig} のハンドラ登録をスキップ')
    return _final_save


def _current_step(manager: CheckpointManager) -> int:
    pointer = _read_pointer(manager.run_dir, LATEST_POINTER)
    return int(pointer['step']) if pointer else 0
