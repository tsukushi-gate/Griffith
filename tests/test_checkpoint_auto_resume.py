"""checkpoint_utils.find_resumable_checkpoint（自動再開の候補探索）のユニットテスト。

Modal の自動リトライでは同じコマンドが再実行され、setup_checkpoint_dir が
**新しいタイムスタンプ付き run_dir** を作るため、run_dir 内だけを見ても
前試行の latest.pt は見つからない（で約$25.3の計算を失った事故）。
prefix 横断の走査・metadata検証・完了runの除外・最大stepの選択を検証する。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from checkpoint_utils import (  # noqa: E402
    checkpoint_paths,
    find_resumable_checkpoint,
    save_checkpoint,
)

_PREFIX = 'unittest_mixer_copy'


def _model():
    return torch.nn.Linear(3, 2)


def _write_ckpt(base_dir, timestamp, step, metadata, best_metric=0.5):
    run_dir = os.path.join(base_dir, f'{_PREFIX}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    latest_path, _ = checkpoint_paths(run_dir)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(latest_path, model, optimizer, None, step, best_metric, metadata)
    return run_dir, latest_path


def _metadata(seed=42, task='copy', d_emb=8):
    return {
        'task': task,
        'config': {'d_emb': d_emb, 'seq_len': 8},
        'args': {'seed': seed, 'num_iterations': 1000, 'task': task},
    }


def test_finds_previous_run_dir(tmp_path):
    """別 run_dir（前試行）の latest.pt を横断走査で見つける。"""
    base = str(tmp_path)
    run_dir, path = _write_ckpt(base, '20260101_000000', 300, _metadata())
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=_metadata(),
        num_iterations=1000, args_match_keys=('seed', 'num_iterations'),
        log=lambda _msg: None,
    )
    assert found is not None
    assert found['path'] == path
    assert found['run_dir'] == run_dir
    assert found['step'] == 300


def test_picks_max_step_not_latest_timestamp(tmp_path):
    """複数候補ではタイムスタンプ順ではなく step が最大のものを選ぶ。"""
    base = str(tmp_path)
    _, deep = _write_ckpt(base, '20260101_000000', 900, _metadata())
    _write_ckpt(base, '20260102_000000', 100, _metadata())
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=_metadata(),
        num_iterations=1000, args_match_keys=('seed',), log=lambda _msg: None,
    )
    assert found is not None
    assert found['path'] == deep
    assert found['step'] == 900


def test_rejects_completed_run(tmp_path):
    """step >= num_iterations の完了済み run は再開対象にしない。"""
    base = str(tmp_path)
    _write_ckpt(base, '20260101_000000', 1000, _metadata())
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=_metadata(),
        num_iterations=1000, args_match_keys=('seed',), log=lambda _msg: None,
    )
    assert found is None


def test_rejects_metadata_mismatch(tmp_path):
    """seed / task / config が違う ckpt は誤って拾わない。"""
    base = str(tmp_path)
    _write_ckpt(base, '20260101_000000', 300, _metadata(seed=7))
    _write_ckpt(base, '20260101_000001', 400, _metadata(task='reverse'))
    _write_ckpt(base, '20260101_000002', 500, _metadata(d_emb=16))
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=_metadata(),
        num_iterations=1000, args_match_keys=('seed', 'num_iterations', 'task'),
        log=lambda _msg: None,
    )
    assert found is None


def test_ignores_other_prefix_dirs(tmp_path):
    """prefix が前方一致するだけの別実験ディレクトリを拾わない。"""
    base = str(tmp_path)
    other = os.path.join(base, f'{_PREFIX}_extra_20260101_000000')
    os.makedirs(other, exist_ok=True)
    model = _model()
    save_checkpoint(
        os.path.join(other, 'latest.pt'), model,
        torch.optim.SGD(model.parameters(), lr=0.1), None, 800, 0.9, _metadata(),
    )
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=_metadata(),
        num_iterations=1000, args_match_keys=('seed',), log=lambda _msg: None,
    )
    assert found is None


def test_tuple_list_equivalence(tmp_path):
    """config の tuple/list 差では不一致にしない（保存往復で型が揺れるため）。"""
    base = str(tmp_path)
    stored = _metadata()
    stored['config'] = {'m_layer_sizes': (64, 128), 'seq_len': 8}
    expected = _metadata()
    expected['config'] = {'m_layer_sizes': [64, 128], 'seq_len': 8}
    _write_ckpt(base, '20260101_000000', 300, stored)
    found = find_resumable_checkpoint(
        _PREFIX, base_dir=base, expected_metadata=expected,
        num_iterations=1000, args_match_keys=('seed',), log=lambda _msg: None,
    )
    assert found is not None
    assert found['step'] == 300
