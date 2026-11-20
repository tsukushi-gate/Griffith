"""共通ランナー(training/runner.py)のテスト。

- logic:xor + logic_mlp小モデルの学習スモーク(hard bit acc=1.0)
- 保存されたconfig.yamlの再読込(同一スキーマ)とチェックポイント形式
- kind/タスク不整合・未対応データセット等のエラー
- VISION_DATASETSレジストリのメタ情報(ローダは呼ばない)
- @slow: MNISTサブセット1エポック(ローカルdata-mnist存在時のみ)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.datasets.vision import VISION_DATASETS
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import (
    evaluate_checkpoint,
    run_training,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')


def _xor_config(**extra_overrides):
    """logic:xor + 小logic_mlp(既知の収束レシピ)の設定を作る"""
    overrides = [
        'data.name=logic:xor',
        'data.batch_size=32',
        'model.kind=logic_mlp',
        'model.k=8',
        'model.layers=2',
        'optim.lr=0.05',
        'run.epochs=40',
        'run.seed=0',
        'run.device=cpu',
    ]
    overrides += [f'{k}={v}' for k, v in extra_overrides.items()]
    return load_config(None, overrides)


# ============================================================
# XORスモーク: 数百stepでhard bit acc=1.0
# ============================================================

class TestXorSmoke:
    def test_xor_reaches_bit_acc_1(self, tmp_path):
        cfg = _xor_config()
        result = run_training(cfg, out_dir=str(tmp_path / 'run'))
        assert result['hard_acc'] == 1.0
        assert result['best_hard_acc'] == 1.0
        assert result['hard_exact_match'] == 1.0  # binary_vectorの追加メトリクス
        # repeat自動水増し(256行/エポック, bs=32)×40エポック=320step
        assert result['steps'] == 320

    def test_saved_config_rerunnable_and_checkpoints(self, tmp_path):
        """保存config.yamlが同一スキーマで再読込できる+ckpt形式互換"""
        out = tmp_path / 'run'
        cfg = _xor_config(**{'run.epochs': 2})
        result = run_training(cfg, out_dir=str(out))

        # 保存されたconfig.yamlをそのまま読める(=同一スキーマ・同一設定)
        config_path = out / 'config.yaml'
        assert config_path.exists()
        assert str(config_path) == result['config_path']
        assert load_config(str(config_path), []) == cfg

        # チェックポイント: experiments/checkpoint_utils.pyと同一dictスキーマ
        latest = out / 'checkpoints' / 'latest.pt'
        best = out / 'checkpoints' / 'best.pt'
        assert latest.exists() and best.exists()
        data = torch.load(str(latest), map_location='cpu', weights_only=False)
        assert {
            'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
            'step', 'best_metric', 'metadata',
        } <= set(data)
        # metadata.configから設定を完全復元できる(md eval用)
        from griffith_logic.training.config import TrainConfig
        assert TrainConfig.model_validate(data['metadata']['config']) == cfg

    def test_evaluate_checkpoint(self, tmp_path):
        """md evalの中身: チェックポイントからモデル復元→hard eval"""
        out = tmp_path / 'run'
        cfg = _xor_config()
        result = run_training(cfg, out_dir=str(out))
        evaluated = evaluate_checkpoint(result['checkpoint_best'], device='cpu')
        assert evaluated['dataset'] == 'logic:xor'
        assert evaluated['hard_acc'] == result['best_hard_acc'] == 1.0

    def test_eval_interval_steps(self, tmp_path):
        """eval_interval>0でステップ単位のhard evalが走る"""
        cfg = _xor_config(**{'run.epochs': 2, 'run.eval_interval': 4})
        result = run_training(cfg, out_dir=str(tmp_path / 'run'))
        assert result['steps'] == 16
        assert 0.0 <= result['hard_acc'] <= 1.0


# ============================================================
# エラー系
# ============================================================

class TestErrors:
    def test_dlgn_with_logic_dataset_raises(self, tmp_path):
        """dlgn(GroupSum終端)×binary_vectorは不整合"""
        cfg = load_config(None, [
            'data.name=logic:xor', 'model.kind=dlgn', 'run.device=cpu',
        ])
        with pytest.raises(ValueError, match='logic_mlp'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))

    def test_logic_mlp_with_vision_dataset_raises(self, tmp_path):
        cfg = load_config(None, [
            'data.name=mnist', 'model.kind=logic_mlp', 'run.device=cpu',
        ])
        with pytest.raises(ValueError, match='dlgn'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))

    def test_unknown_dataset_raises(self, tmp_path):
        cfg = load_config(None, ['data.name=sonzai-shinai', 'run.device=cpu'])
        with pytest.raises(ValueError, match='未対応のデータセット'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))

    def test_unknown_logic_key_raises(self, tmp_path):
        cfg = load_config(None, [
            'data.name=logic:nand99', 'model.kind=logic_mlp', 'run.device=cpu',
        ])
        with pytest.raises(ValueError, match='未対応の論理データセット'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))

    def test_unsupported_method_raises(self, tmp_path):
        cfg = _xor_config(**{'run.method': 'eggroll'})
        with pytest.raises(NotImplementedError, match='bp'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))


# ============================================================
# VISION_DATASETSレジストリ(メタ情報のみ。ローダは呼ばない)
# ============================================================

class TestVisionRegistry:
    def test_registry_metadata(self):
        assert VISION_DATASETS['mnist'].in_width == 784
        assert VISION_DATASETS['mnist'].classes == 10
        assert VISION_DATASETS['mnist20x20'].in_width == 400
        assert VISION_DATASETS['cifar-10-3-thresholds'].in_width == 3 * 32 * 32 * 3
        assert VISION_DATASETS['cifar-100-31-thresholds'].classes == 100
        for spec in VISION_DATASETS.values():
            assert callable(spec.load)


# ============================================================
# @slow: MNISTサブセット1エポック(ローカルdata-mnist存在時のみ)
# ============================================================

def _has_local_mnist() -> bool:
    return os.path.isdir(os.path.join(REPO_ROOT, 'data-mnist'))


@pytest.mark.slow
@pytest.mark.skipif(not _has_local_mnist(), reason='ローカルdata-mnistが無い')
class TestMnistSubsetSlow:
    def test_mnist_2000_samples_one_epoch(self, tmp_path):
        pytest.importorskip('torchvision')
        cfg = load_config(None, [
            'data.name=mnist',
            'data.batch_size=64',
            'data.params.limit=2000',
            'data.params.download=false',
            'model.kind=dlgn',
            'model.k=1000',
            'model.layers=2',
            'model.tau=10',
            'optim.lr=0.05',
            'run.epochs=1',
            'run.seed=0',
            'run.device=cpu',
        ])
        result = run_training(cfg, out_dir=str(tmp_path / 'run'))
        # スモーク基準(実測較正: seed0=0.458 / seed1=0.346 / seed42=0.503)
        assert result['hard_acc'] >= 0.25
