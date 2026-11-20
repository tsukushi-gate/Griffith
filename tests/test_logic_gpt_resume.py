"""LogicLadderGPT --resume(途中再開+wandb継続記録)のユニットテスト。

CPU・wandbオフで高速に検証する(test_textcnn.py方式でexperiments/を
sys.pathへ追加しtrain_logic_gpt.pyをトップレベルモジュールとしてimport)。
チェックポイントはtmp_path配下(cwdをtmp_pathへ切替え、--run-dirで
checkpoints/<NAME>/固定パスへ書かせる)に隔離し、実 checkpoints/ を汚さない。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from checkpoint_utils import load_checkpoint  # noqa: E402
from train_logic_gpt import load_recipe, load_shakespeare, main  # noqa: E402

from griffith_logic.models import LogicLadderGPT, LogicLadderGPTConfig  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
_RECIPE = os.path.abspath(
    os.path.join(_REPO_ROOT, 'configs', 'recipes', 'logic_gpt_shakespeare.yaml')
)
_DATA_DIR = os.path.abspath(os.path.join(_REPO_ROOT, 'data-shakespeare'))

_COMMON_ARGS = [
    '--config', _RECIPE, '--data-dir', _DATA_DIR,
    '--batch-size', '8', '--no-wandb', '--device', 'cpu',
    # レシピ既定は全val評価(本走用)のため、テストは部分評価で高速化する
    # (resume契約の検証に評価スコープは無関係)
    '--eval-batches', '5',
]


def _run(monkeypatch, argv):
    """train_logic_gpt.main() を sys.argv 経由で実行する。"""
    monkeypatch.setattr(sys, 'argv', ['train_logic_gpt.py', *argv])
    main()


def test_latest_checkpoint_has_full_state(tmp_path, monkeypatch):
    """eval毎にlatest.ptがcheckpoint_utils形式の全項目付きで保存される。"""
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, [
        *_COMMON_ARGS, '--max-steps', '20', '--eval-freq', '10',
        '--run-dir', 'run_a',
    ])

    # --run-dir指定時は checkpoints/<NAME>/latest.pt の固定パスに保存される
    latest_path = tmp_path / 'checkpoints' / 'run_a' / 'latest.pt'
    assert latest_path.exists()

    ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
    assert ckpt['step'] == 20
    assert ckpt['model_state_dict']
    assert ckpt['optimizer_state_dict'] is not None
    assert len(ckpt['optimizer_state_dict']['state']) > 0
    assert ckpt['best_metric'] < float('inf')
    assert ckpt['metadata']['variant'] == 'L0'
    assert ckpt['metadata']['config_path'] == _RECIPE
    # --no-wandb実行なのでwandb_run_idはNoneのまま記録される
    assert ckpt['metadata']['wandb_run_id'] is None
    assert 'torch_rng' in ckpt['extra_state']


def test_resume_restores_model_and_optimizer_state(tmp_path, monkeypatch):
    """load_checkpointによる復元直後、モデル/optimizer状態が保存時点と一致する。"""
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, [
        *_COMMON_ARGS, '--max-steps', '20', '--eval-freq', '10',
        '--run-dir', 'run_b',
    ])
    latest_path = tmp_path / 'checkpoints' / 'run_b' / 'latest.pt'
    raw = torch.load(latest_path, map_location='cpu', weights_only=False)

    # train_logic_gpt.py内部と同じ手順でモデル/optimizerを再構築
    model_cfg, train_cfg, variant = load_recipe(_RECIPE, None)
    context_len = int(train_cfg.get('context_len', 64))
    _, _, meta = load_shakespeare(
        batch_size=8, context_len=context_len, data_dir=_DATA_DIR,
    )
    config = LogicLadderGPTConfig.from_model_cfg(
        model_cfg, vocab_size=meta['vocab_size'], seq_len=context_len,
    )
    model2 = LogicLadderGPT(config)
    optimizer2 = torch.optim.AdamW(
        model2.parameters(),
        lr=float(train_cfg.get('lr', 3.0e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 0.0)),
    )

    ckpt_info = load_checkpoint(str(latest_path), model2, optimizer2, device='cpu')
    assert ckpt_info['step'] == 20
    assert ckpt_info['best_metric'] == raw['best_metric']

    # 復元直後、パラメータが保存時点の値とbitwiseで一致する
    for (name, restored), (raw_name, saved) in zip(
        model2.state_dict().items(), raw['model_state_dict'].items(),
    ):
        assert name == raw_name
        assert torch.equal(restored, saved)

    # optimizerのmomentum等の内部状態も復元されている
    assert len(optimizer2.state_dict()['state']) == len(raw['optimizer_state_dict']['state'])


def test_resume_continues_from_start_step_and_carries_best(tmp_path, monkeypatch):
    """--resumeでstart_step+1から継続し、best_val_lossが引き継がれる。"""
    monkeypatch.chdir(tmp_path)
    run_dir_name = 'run_c'
    _run(monkeypatch, [
        *_COMMON_ARGS, '--max-steps', '20', '--eval-freq', '10',
        '--run-dir', run_dir_name,
    ])
    latest_path = tmp_path / 'checkpoints' / run_dir_name / 'latest.pt'
    ckpt_at_20 = torch.load(latest_path, map_location='cpu', weights_only=False)
    assert ckpt_at_20['step'] == 20
    best_at_20 = ckpt_at_20['best_metric']

    _run(monkeypatch, [
        *_COMMON_ARGS, '--max-steps', '30', '--eval-freq', '10',
        '--run-dir', run_dir_name, '--resume', str(latest_path),
    ])
    ckpt_at_30 = torch.load(latest_path, map_location='cpu', weights_only=False)
    # start_step=20から継続し、ステップ21..30を実行してstep=30まで到達している
    assert ckpt_at_30['step'] == 30
    # best_val_lossは引き継がれ、再開直後にリセット(infから再スタート)されていない
    assert ckpt_at_30['best_metric'] <= best_at_20
