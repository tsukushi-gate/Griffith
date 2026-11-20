"""experiments/ckpt_utils.py の往復・atomic性・間隔制御テスト（CKPT-1）。

wandb部分は fake run / fake Artifact でモックし、オフラインで完結させる。
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments import ckpt_utils  # noqa: E402


def _make_model_and_opt(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, opt


def _train_a_bit(model, opt, steps: int = 3):
    for _ in range(steps):
        x = torch.randn(5, 4)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ============================================================
# fake wandb
# ============================================================

class FakeArtifact:
    def __init__(self, name, type='model', metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files: list[str] = []

    def add_file(self, path):
        assert os.path.exists(path), f'存在しないファイルをArtifact化しようとした: {path}'
        self.files.append(path)


class FakeRun:
    name = 'fake-run'

    def __init__(self):
        self.logged: list[tuple[FakeArtifact, list[str]]] = []

    def log_artifact(self, artifact, aliases=None):
        self.logged.append((artifact, list(aliases or [])))


# ============================================================
# save/load 往復
# ============================================================

def test_save_load_roundtrip_model_optimizer(tmp_path):
    model, opt = _make_model_and_opt()
    _train_a_bit(model, opt)
    path = str(tmp_path / 'ckpt.pt')
    ckpt_utils.save_checkpoint(path, model=model, optimizer=opt, step=42,
                               extra={'best_metric': 0.5})

    payload = ckpt_utils.load_checkpoint(path)
    assert payload['step'] == 42
    assert payload['extra']['best_metric'] == 0.5

    fresh, fresh_opt = _make_model_and_opt(seed=99)
    step = ckpt_utils.restore_into(payload, model=fresh, optimizer=fresh_opt)
    assert step == 42
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)
    # optimizer state（Adamのexp_avg等）も復元されている
    assert len(fresh_opt.state_dict()['state']) == len(opt.state_dict()['state'])
    assert len(fresh_opt.state_dict()['state']) > 0


def test_rng_state_reproducible(tmp_path):
    torch.manual_seed(1234)
    random.seed(1234)
    path = str(tmp_path / 'rng.pt')
    ckpt_utils.save_checkpoint(path, step=0)

    expected_torch = torch.randn(3)
    expected_py = [random.random() for _ in range(3)]

    # RNGを撹乱してから復元
    torch.randn(100)
    [random.random() for _ in range(100)]
    ckpt_utils.load_checkpoint(path, rng=True)

    assert torch.equal(torch.randn(3), expected_torch)
    assert [random.random() for _ in range(3)] == expected_py


def test_state_only_checkpoint_without_torch_modules(tmp_path):
    """torch非依存の実験（進化系）向け: state辞書だけの保存。"""
    path = str(tmp_path / 'evo.pt')
    ckpt_utils.save_checkpoint(path, step=7, state={'population': [1, 2, 3], 'fitness': 0.9})
    payload = ckpt_utils.load_checkpoint(path)
    assert payload['state']['population'] == [1, 2, 3]
    assert payload['step'] == 7
    assert 'model' not in payload


# ============================================================
# atomic性
# ============================================================

def test_atomic_write_leaves_no_tmp(tmp_path):
    path = str(tmp_path / 'atomic.pt')
    ckpt_utils.save_checkpoint(path, step=1, state={'a': 1})
    names = os.listdir(tmp_path)
    assert names == ['atomic.pt']
    assert not any('.tmp' in n for n in names)


def test_atomic_write_failure_cleans_tmp_and_keeps_old(tmp_path, monkeypatch):
    path = str(tmp_path / 'atomic.pt')
    ckpt_utils.save_checkpoint(path, step=1, state={'a': 1})

    def boom(*args, **kwargs):
        raise RuntimeError('シリアライズ失敗')

    monkeypatch.setattr(ckpt_utils.torch, 'save', boom)
    with pytest.raises(RuntimeError):
        ckpt_utils.save_checkpoint(path, step=2, state={'a': 2})

    # 失敗しても既存ckptは無傷、tmpは残らない
    assert not any('.tmp' in n for n in os.listdir(tmp_path))
    assert ckpt_utils.load_checkpoint(path)['step'] == 1


# ============================================================
# CheckpointManager
# ============================================================

def test_maybe_save_respects_interval(tmp_path):
    clock = {'t': 1000.0}
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), interval_sec=300, upload=False,
                                       time_fn=lambda: clock['t'])
    state_fn = lambda: {'state': {'v': clock['t']}}  # noqa: E731

    assert mgr.maybe_save(1, state_fn) is None       # 経過0秒
    clock['t'] += 299
    assert mgr.maybe_save(2, state_fn) is None       # まだ間隔未満
    clock['t'] += 1
    saved = mgr.maybe_save(3, state_fn)
    assert saved is not None and os.path.exists(saved)
    # 保存直後はタイマがリセットされる
    assert mgr.maybe_save(4, state_fn) is None
    # forceなら間隔に関係なく保存
    assert mgr.maybe_save(5, state_fn, force=True) is not None


def test_manager_uploads_to_wandb_with_aliases(tmp_path):
    run = FakeRun()
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), wandb_run=run, interval_sec=0,
                                       name='my run', artifact_factory=FakeArtifact)
    state_fn = lambda: {'state': {'v': 1}}  # noqa: E731

    mgr.save(10, state_fn)
    mgr.save_best(10, state_fn, metric=0.75)

    assert len(run.logged) == 2
    latest_art, latest_aliases = run.logged[0]
    best_art, best_aliases = run.logged[1]
    assert latest_aliases == ['latest']
    assert best_aliases == ['best', 'latest']
    assert latest_art.name == 'my-run-ckpt'   # 空白がサニタイズされる
    assert best_art.metadata['best_metric'] == 0.75
    assert len(latest_art.files) == 1


def test_load_latest_from_run_dir(tmp_path):
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), interval_sec=0, upload=False)
    model, opt = _make_model_and_opt()
    _train_a_bit(model, opt)
    mgr.save(5, lambda: {'model': model, 'optimizer': opt})
    _train_a_bit(model, opt)
    mgr.save(11, lambda: {'model': model, 'optimizer': opt})

    found = ckpt_utils.load_latest(str(tmp_path))
    assert found is not None
    payload, step = found
    assert step == 11
    fresh, _ = _make_model_and_opt(seed=7)
    ckpt_utils.restore_into(payload, model=fresh)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)


def test_load_latest_falls_back_to_scan_when_pointer_missing(tmp_path):
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), interval_sec=0, upload=False)
    mgr.save(3, lambda: {'state': {'v': 3}})
    mgr.save(9, lambda: {'state': {'v': 9}})
    # ポインタを壊す（削除はしない: 空文字で無効化）
    with open(tmp_path / ckpt_utils.LATEST_POINTER, 'w', encoding='utf-8') as f:
        f.write('')
    payload, step = ckpt_utils.load_latest(str(tmp_path))
    assert step == 9 and payload['state']['v'] == 9


def test_load_latest_returns_none_for_empty_dir(tmp_path):
    assert ckpt_utils.load_latest(str(tmp_path)) is None


def test_keep_generations_are_not_deleted(tmp_path):
    """規約により古い世代も削除しない（stale_pathsは情報提供のみ）。"""
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), interval_sec=0, upload=False, keep=2)
    for step in range(4):
        mgr.save(step, lambda: {'state': {'v': 0}})
    files = sorted(n for n in os.listdir(tmp_path) if n.endswith('.pt'))
    assert len(files) == 4
    assert len(mgr.stale_paths()) == 2
    assert all(os.path.exists(p) for p in mgr.stale_paths())


# ============================================================
# resume_or_init / graceful save
# ============================================================

class _Args:
    def __init__(self, resume=None):
        self.resume = resume


def test_resume_or_init_initializes_without_flag(tmp_path):
    obj, step, payload = ckpt_utils.resume_or_init(_Args(None), str(tmp_path), lambda: 'fresh')
    assert (obj, step, payload) == ('fresh', 0, None)


def test_resume_or_init_auto_resumes(tmp_path):
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), interval_sec=0, upload=False)
    mgr.save(21, lambda: {'state': {'v': 21}})
    obj, step, payload = ckpt_utils.resume_or_init(_Args('auto'), str(tmp_path), lambda: 'fresh')
    assert obj == 'fresh' and step == 21
    assert payload['state']['v'] == 21


def test_resume_or_init_missing_target_falls_back_to_init(tmp_path):
    obj, step, payload = ckpt_utils.resume_or_init(_Args('auto'), str(tmp_path), lambda: 'fresh')
    assert (obj, step, payload) == ('fresh', 0, None)


def test_install_graceful_save_writes_final_ckpt(tmp_path):
    run = FakeRun()
    mgr = ckpt_utils.CheckpointManager(str(tmp_path), wandb_run=run, interval_sec=10 ** 9,
                                       artifact_factory=FakeArtifact)
    final_save = ckpt_utils.install_graceful_save(
        mgr, lambda: {'state': {'v': 'final'}}, step_fn=lambda: 77)
    final_save()
    payload, step = ckpt_utils.load_latest(str(tmp_path))
    assert step == 77 and payload['state']['v'] == 'final'
    assert run.logged[-1][1] == ['latest', 'interrupted']
