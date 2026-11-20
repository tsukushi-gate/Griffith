"""レシピからwandbのtaxonomyを渡す設定(RunConfig.wandb_*)のテスト。

検証する仕様:
    1. 既定値は従来のランナーの挙動と同一(group='md-train' / job_type='train' /
       追加タグなし / notes・hypothesis なし)。既存レシピの記録内容は変わらない
    2. レシピ(YAML)とドット記法オーバーライドの両方から指定できる。未知の
       job_type は extra='forbid' の他フィールドと同じくバリデーションで弾く
    3. `_wandb_tags` はランナー既定タグへレシピのタグを**追加**する(重複は足さない)
    4. run_training が init_wandb へ group / job_type / tags / notes / hypothesis を
       そのまま渡す(runner.wandb_utils を差し替えて呼び出し引数を捕まえる)
"""

import os
import sys
import types

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.training import runner as runner_mod
from griffith_logic.training.config import RunConfig, load_config, save_config
from griffith_logic.training.runner import (
    ResolvedData,
    _wandb_tags,
    run_training,
)


def _data(kind: str = 'classification') -> ResolvedData:
    return ResolvedData(
        name='synthetic', kind=kind, in_width=16, out_width=2,
        train_loader=None, eval_loader=None,
    )


class TestRunConfigWandbDefaults:
    def test_defaults_match_previous_runner_behaviour(self):
        run = RunConfig()
        assert run.wandb_group == 'md-train'
        assert run.wandb_job_type == 'train'
        assert run.wandb_tags == []
        assert run.wandb_notes is None
        assert run.wandb_hypothesis is None

    def test_existing_recipe_is_unchanged(self, tmp_path):
        """wandb_* を書かないレシピは従来と同じ設定になる"""
        path = tmp_path / 'recipe.yaml'
        path.write_text(
            'data:\n  name: mnist\nrun:\n  epochs: 3\n  wandb: true\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert cfg.run.wandb is True
        assert cfg.run.wandb_group == 'md-train'
        assert cfg.run.wandb_job_type == 'train'
        assert cfg.run.wandb_tags == []


class TestRunConfigWandbFromRecipe:
    def test_from_yaml(self, tmp_path):
        path = tmp_path / 'recipe.yaml'
        path.write_text(
            'run:\n'
            '  wandb: true\n'
            '  wandb_group: dlgn-ffn\n'
            '  wandb_job_type: ablation\n'
            '  wandb_tags: [computed-wiring-overlay, P6]\n'
            '  wandb_notes: rot 配線の較正アンカー再走\n'
            '  wandb_hypothesis: rot は固定ランダム配線と同等以上\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert cfg.run.wandb_group == 'dlgn-ffn'
        assert cfg.run.wandb_job_type == 'ablation'
        assert cfg.run.wandb_tags == ['computed-wiring-overlay', 'P6']
        assert cfg.run.wandb_notes == 'rot 配線の較正アンカー再走'
        assert cfg.run.wandb_hypothesis == 'rot は固定ランダム配線と同等以上'

    def test_from_dot_overrides(self):
        cfg = load_config(None, [
            'run.wandb_group=dlgn-ffn',
            'run.wandb_job_type=smoke',
            'run.wandb_tags=[a,b]',
        ])
        assert cfg.run.wandb_group == 'dlgn-ffn'
        assert cfg.run.wandb_job_type == 'smoke'
        assert cfg.run.wandb_tags == ['a', 'b']

    def test_yaml_roundtrip(self, tmp_path):
        cfg = load_config(None, [
            'run.wandb_group=dlgn-ffn',
            'run.wandb_job_type=ablation',
            'run.wandb_tags=[cwo,P1]',
            'run.wandb_notes=note',
            'run.wandb_hypothesis=hyp',
        ])
        path = str(tmp_path / 'config.yaml')
        save_config(cfg, path)
        assert load_config(path, []) == cfg

    def test_rejects_unknown_job_type(self):
        with pytest.raises(ValidationError):
            load_config(None, ['run.wandb_job_type=benchmark'])

    def test_rejects_typo_key(self):
        with pytest.raises(ValidationError):
            load_config(None, ['run.wandb_grp=dlgn-ffn'])


class TestWandbTags:
    def test_default_tags_are_unchanged(self):
        cfg = load_config(None, [])
        assert _wandb_tags(cfg, _data()) == [
            'md-train', 'dlgn', 'classification',
        ]

    def test_recipe_tags_are_appended(self):
        cfg = load_config(None, ['run.wandb_tags=[computed-wiring-overlay,P6]'])
        assert _wandb_tags(cfg, _data()) == [
            'md-train', 'dlgn', 'classification',
            'computed-wiring-overlay', 'P6',
        ]

    def test_duplicate_tags_are_not_repeated(self):
        cfg = load_config(None, ['run.wandb_tags=[dlgn,md-train,extra]'])
        assert _wandb_tags(cfg, _data()) == [
            'md-train', 'dlgn', 'classification', 'extra',
        ]

    def test_binary_vector_task_tag(self):
        cfg = load_config(None, ['model.kind=logic_mlp', 'run.wandb_tags=[x]'])
        assert _wandb_tags(cfg, _data('binary_vector')) == [
            'md-train', 'logic_mlp', 'binary_vector', 'x',
        ]


class _FakeWandbUtils:
    """init_wandb の呼び出し引数を捕まえるだけの偽 wandb_utils モジュール代用"""

    def __init__(self):
        self.init_kwargs = None
        self.finished = None

    def init_wandb(self, **kwargs):
        self.init_kwargs = kwargs
        return True

    def log_metrics(self, payload, step=None):
        pass

    def log_checkpoint_artifact(self, path, name=None, stage='final',
                                metadata=None, model=None):
        pass

    def finish_wandb(self, status='finished', final_metrics=None):
        self.finished = status


class TestInitWandbCall:
    """run_training が init_wandb へレシピの値をそのまま渡す"""

    @staticmethod
    def _xor_config(**overrides):
        base = [
            'data.name=logic:xor',
            'data.batch_size=32',
            'model.kind=logic_mlp',
            'model.k=8',
            'model.layers=2',
            'run.epochs=1',
            'run.seed=0',
            'run.device=cpu',
            'run.wandb=true',
        ]
        base += [f'{k}={v}' for k, v in overrides.items()]
        return load_config(None, base)

    def _run(self, monkeypatch, tmp_path, cfg):
        fake = _FakeWandbUtils()
        monkeypatch.setattr(
            runner_mod, 'wandb_utils',
            types.SimpleNamespace(
                init_wandb=fake.init_wandb,
                log_metrics=fake.log_metrics,
                log_checkpoint_artifact=fake.log_checkpoint_artifact,
                finish_wandb=fake.finish_wandb,
            ),
        )
        run_training(cfg, out_dir=str(tmp_path / 'run'))
        assert fake.init_kwargs is not None, 'init_wandb が呼ばれていない'
        return fake

    def test_passes_defaults(self, monkeypatch, tmp_path):
        fake = self._run(monkeypatch, tmp_path, self._xor_config())
        assert fake.init_kwargs['group'] == 'md-train'
        assert fake.init_kwargs['job_type'] == 'train'
        assert fake.init_kwargs['tags'] == [
            'md-train', 'logic_mlp', 'binary_vector',
        ]
        assert fake.init_kwargs['notes'] is None
        assert fake.init_kwargs['hypothesis'] is None
        assert fake.finished == 'finished'

    def test_passes_recipe_values(self, monkeypatch, tmp_path):
        cfg = self._xor_config(**{
            'run.wandb_group': 'dlgn-ffn',
            'run.wandb_job_type': 'ablation',
            'run.wandb_tags': '[computed-wiring-overlay,P6]',
            'run.wandb_notes': 'anchor',
            'run.wandb_hypothesis': 'rot >= random',
        })
        fake = self._run(monkeypatch, tmp_path, cfg)
        assert fake.init_kwargs['group'] == 'dlgn-ffn'
        assert fake.init_kwargs['job_type'] == 'ablation'
        assert fake.init_kwargs['tags'] == [
            'md-train', 'logic_mlp', 'binary_vector',
            'computed-wiring-overlay', 'P6',
        ]
        assert fake.init_kwargs['notes'] == 'anchor'
        assert fake.init_kwargs['hypothesis'] == 'rot >= random'
