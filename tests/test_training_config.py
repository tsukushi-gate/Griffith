"""設定スキーマ(training/config.py)のテスト。

- 既定値とYAML読込
- ドット記法オーバーライド(型変換・ネスト・不正キー/書式のエラー)
- YAML往復(save_config→load_configで同一設定)
- JSON Schema出力(model_json_schema)
"""

import json
import os
import sys

import pytest
import yaml
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.training.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    dump_config,
    load_config,
    save_config,
)

# ============================================================
# 既定値とYAML読込
# ============================================================

class TestDefaults:
    def test_default_config(self):
        cfg = load_config(None, [])
        assert cfg == TrainConfig()
        assert cfg.data == DataConfig(name='mnist', batch_size=128, params={})
        assert cfg.model == ModelConfig(kind='dlgn', k=2000, layers=3)
        assert cfg.optim == OptimConfig(name='adam', lr=0.01)
        assert cfg.run == RunConfig(seed=42, epochs=3, eval_interval=0)

    def test_load_yaml_file(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text(
            'data:\n  name: logic:xor\n  batch_size: 16\n'
            'model:\n  kind: logic_mlp\n  k: 8\n'
            'run:\n  epochs: 7\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert cfg.data.name == 'logic:xor'
        assert cfg.data.batch_size == 16
        assert cfg.model.kind == 'logic_mlp'
        assert cfg.model.k == 8
        assert cfg.run.epochs == 7
        # 未指定セクションは既定値
        assert cfg.optim == OptimConfig()

    def test_empty_yaml_is_defaults(self, tmp_path):
        path = tmp_path / 'empty.yaml'
        path.write_text('', encoding='utf-8')
        assert load_config(str(path), []) == TrainConfig()

    def test_non_mapping_root_raises(self, tmp_path):
        path = tmp_path / 'list.yaml'
        path.write_text('- 1\n- 2\n', encoding='utf-8')
        with pytest.raises(ValueError, match='マッピング'):
            load_config(str(path), [])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / 'nai.yaml'), [])


# ============================================================
# ドット記法オーバーライド
# ============================================================

class TestOverrides:
    def test_type_conversion(self):
        """YAMLスカラー解釈でint/float/bool/nullに変換される"""
        cfg = load_config(None, [
            'model.k=64',
            'optim.lr=0.05',
            'run.wandb=true',
            'run.run_name=null',
        ])
        assert cfg.model.k == 64 and isinstance(cfg.model.k, int)
        assert cfg.optim.lr == pytest.approx(0.05)
        assert cfg.run.wandb is True
        assert cfg.run.run_name is None

    def test_nested_params(self):
        cfg = load_config(None, [
            'data.params.binarize_threshold=0.25',
            'data.params.limit=2000',
        ])
        assert cfg.data.params == {'binarize_threshold': 0.25, 'limit': 2000}

    def test_override_wins_over_yaml(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text('run:\n  epochs: 3\n  seed: 1\n', encoding='utf-8')
        cfg = load_config(str(path), ['run.epochs=9'])
        assert cfg.run.epochs == 9
        assert cfg.run.seed == 1  # 上書きしていないキーはYAML値を維持

    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.nonexistent=1'])

    def test_unknown_section_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['bogus.key=1'])

    def test_bad_type_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.k=abc'])

    def test_bad_literal_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.kind=transformer'])

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match='key=value'):
            load_config(None, ['model.k'])

    def test_intermediate_non_mapping_raises(self):
        with pytest.raises(ValueError, match='マッピング'):
            load_config(None, ['data.name=mnist', 'data.name.x=1'])


# ============================================================
# YAML往復(土台)
# ============================================================

class TestRoundTrip:
    def test_save_and_reload_identical(self, tmp_path):
        cfg = load_config(None, [
            'data.name=logic:parity3',
            'data.params.repeat=8',
            'model.kind=logic_mlp',
            'model.k=16',
            'run.epochs=5',
            'run.run_name=roundtrip',
        ])
        path = save_config(cfg, str(tmp_path / 'config.yaml'))
        assert load_config(path, []) == cfg

    def test_dump_config_is_yaml_json_safe(self):
        cfg = TrainConfig()
        dumped = dump_config(cfg)
        assert isinstance(dumped, dict)
        # YAML/JSONの両方でシリアライズ可能
        yaml.safe_dump(dumped)
        json.dumps(dumped)
        # 往復で同一
        assert TrainConfig.model_validate(dumped) == cfg


# ============================================================
# JSON Schema
# ============================================================

class TestJsonSchema:
    def test_model_json_schema(self):
        schema = TrainConfig.model_json_schema()
        # expected_metricsは追加された正式フィールド(レシピの期待値照合)。
        # distillは追加された蒸留設定(optional、省略時None=従来動作)。
        # regularizationは先行実験で追加(optional、省略時None=従来動作)。
        assert set(schema['properties']) == {
            'data', 'model', 'optim', 'run', 'distill', 'regularization',
            'expected_metrics',
        }
        json.dumps(schema)  # JSONシリアライズ可能

    def test_nested_models_in_defs(self):
        """ネストしたpydanticモデルは$defsに出る(先行実験で層ごと配線を追加)"""
        schema = TrainConfig.model_json_schema()
        assert {'ConnectionsConfig', 'LayerConnectionsConfig'} <= set(
            schema['$defs'])
        layer = schema['$defs']['LayerConnectionsConfig']
        # 構造化配線の欄(step / shifts / stride / offsets / grid / bits_per_cell /
        # dedupe)と kind(rot_n / butterfly / local1d / local2d)が増えた
        assert set(layer['properties']) == {
            'kind', 's_a', 's_b', 'step', 'shifts', 'stride', 'offsets', 'grid',
            'bits_per_cell', 'dedupe',
        }
        assert set(layer['properties']['kind']['enum']) == {
            'rot', 'stretched_rot', 'perm', 'random_seeded', 'random_uniform',
            'rot_n', 'butterfly', 'local1d', 'local2d',
        }
        conn = schema['$defs']['ConnectionsConfig']
        assert set(conn['properties']) == {
            'kind', 's_a', 's_b', 'step', 'layers', 'strides', 'order', 'grid',
            'bits_per_cell', 'layer0_offsets', 'assign', 'dedupe',
        }
        assert set(conn['properties']['kind']['enum']) == {
            'rot', 'perm', 'random_seeded', 'random_uniform', 'per_layer',
            'butterfly', 'local1d', 'local2d',
        }
        # conv_dlgn の形は別モデルとして $defs に出る
        assert 'ConvDLGNModelConfig' in schema['$defs']


# ============================================================
# RegConfig(先行実験縮約耐性正則化)
# ============================================================

class TestRegConfig:
    def test_default_is_none(self):
        assert TrainConfig().regularization is None
        assert load_config(None, []).regularization is None

    def test_yaml_lambda_alias(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text(
            'regularization:\n  kind: irreducibility\n'
            '  lambda: 0.1\n  ramp_frac: 0.5\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert cfg.regularization.kind == 'irreducibility'
        assert cfg.regularization.lambda_ == 0.1
        assert cfg.regularization.ramp_frac == 0.5

    def test_defaults_within_section(self):
        cfg = load_config(None, ['regularization.kind=irreducibility'])
        assert cfg.regularization.lambda_ == 0.0
        assert cfg.regularization.ramp_frac == 0.5

    def test_dot_override_lambda(self):
        cfg = load_config(None, [
            'regularization.kind=irreducibility', 'regularization.lambda=0.01',
        ])
        assert cfg.regularization.lambda_ == 0.01

    def test_roundtrip(self, tmp_path):
        cfg = load_config(None, [
            'regularization.kind=irreducibility',
            'regularization.lambda=0.1',
            'regularization.ramp_frac=0.25',
        ])
        path = save_config(cfg, str(tmp_path / 'config.yaml'))
        assert load_config(path, []) == cfg

    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['regularization.bogus=1'])

    def test_unknown_kind_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['regularization.kind=spectral'])

    def test_negative_lambda_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['regularization.lambda=-0.1'])

    def test_ramp_frac_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['regularization.ramp_frac=1.5'])


# ============================================================
# 先行実験のModelConfig / RunConfig拡張
# ============================================================

class TestExp099ModelConfig:
    def test_defaults_are_legacy(self):
        m = load_config(None, []).model
        assert m.selection == 'soft'
        assert m.init_mode is None
        assert m.init_scale == 5.0
        assert m.grad_factor == 1.0
        assert m.gate_mask_hidden is None
        assert m.gate_mask_final is None
        assert m.connections == 'random'

    def test_run_defaults_are_legacy(self):
        r = load_config(None, []).run
        assert r.export_netlist is False
        assert r.save_eval_history is False

    def test_string_connections_still_accepted(self):
        assert load_config(None, ['model.connections=unique']).model.connections \
            == 'unique'

    def test_rot_connections(self, tmp_path):
        path = tmp_path / 'c.yaml'
        path.write_text(
            'model:\n  layers: 6\n  connections:\n    kind: rot\n'
            '    s_a: [3, 0, 0, 0, 0, 0]\n'
            '    s_b: [96, 4, 20, 90, 400, 1800]\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert cfg.model.connections.kind == 'rot'
        assert cfg.model.connections.s_a == [3, 0, 0, 0, 0, 0]
        assert cfg.model.connections.s_b == [96, 4, 20, 90, 400, 1800]

    def test_rot_length_mismatch_raises(self, tmp_path):
        path = tmp_path / 'c.yaml'
        path.write_text(
            'model:\n  layers: 6\n  connections:\n    kind: rot\n'
            '    s_a: [3, 0]\n    s_b: [96, 4]\n',
            encoding='utf-8',
        )
        with pytest.raises(ValidationError):
            load_config(str(path), [])

    def test_rot_without_shifts_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.connections.kind=rot'])

    def test_perm_override_keeps_unused_shifts(self, tmp_path):
        """B0レシピに model.connections.kind=perm を1行上書きしてW腕にできる"""
        path = tmp_path / 'c.yaml'
        path.write_text(
            'model:\n  layers: 6\n  connections:\n    kind: rot\n'
            '    s_a: [3, 0, 0, 0, 0, 0]\n'
            '    s_b: [96, 4, 20, 90, 400, 1800]\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), ['model.connections.kind=perm'])
        assert cfg.model.connections.kind == 'perm'

    def test_unknown_connections_kind_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.connections.kind=affine'])

    def test_gate_mask_range_checked(self):
        cfg = load_config(None, ['model.gate_mask_hidden=[0,15,3,5,10,12]'])
        assert cfg.model.gate_mask_hidden == [0, 15, 3, 5, 10, 12]
        with pytest.raises(ValidationError):
            load_config(None, ['model.gate_mask_final=[16]'])
        with pytest.raises(ValidationError):
            load_config(None, [f'model.gate_mask_hidden={list(range(16))}'])

    def test_init_mode_literal(self):
        for mode in ('randn', 'ri', 'ri_randn', 'rri', 'uniform'):
            assert load_config(None, [f'model.init_mode={mode}']).model.init_mode \
                == mode
        with pytest.raises(ValidationError):
            load_config(None, ['model.init_mode=bogus'])

    def test_selection_literal(self):
        assert load_config(None, ['model.selection=hard_st']).model.selection \
            == 'hard_st'
        with pytest.raises(ValidationError):
            load_config(None, ['model.selection=argmax'])

    def test_grad_factor_positive(self):
        assert load_config(None, ['model.grad_factor=2.0']).model.grad_factor == 2.0
        with pytest.raises(ValidationError):
            load_config(None, ['model.grad_factor=0'])

    def test_init_scale(self):
        """先行実験 Phase 3 の OP 腕は残差初期化の加算量が 1.0"""
        assert load_config(None, ['model.init_scale=1.0']).model.init_scale == 1.0
        with pytest.raises(ValidationError):
            load_config(None, ['model.init_scale=0'])
        with pytest.raises(ValidationError):
            load_config(None, ['model.init_scale=-1'])

    def test_roundtrip_with_all_exp099_keys(self, tmp_path):
        cfg = load_config(None, [
            'model.layers=6',
            'model.selection=hard_st',
            'model.init_mode=ri_randn',
            'model.init_scale=1.0',
            'model.grad_factor=1.5',
            'model.gate_mask_hidden=[0,15,3,5,10,12]',
            'model.gate_mask_final=[3,5,10,12]',
            'model.connections.kind=rot',
            'model.connections.s_a=[3,0,0,0,0,0]',
            'model.connections.s_b=[96,4,20,90,400,1800]',
            'run.export_netlist=true',
            'run.save_eval_history=true',
            'regularization.kind=irreducibility',
            'regularization.lambda=0.1',
        ])
        path = save_config(cfg, str(tmp_path / 'config.yaml'))
        assert load_config(path, []) == cfg


# ============================================================
# 先行実験層ごとの配線指定(per_layer)
# ============================================================

# 先行実験 model.yaml 末尾の腕ごとの上書き(1 引数の丸ごと差し替え)
PR_OVERRIDE = (
    'model.connections={kind: per_layer, layers: ['
    '{kind: perm}, {kind: rot, s_a: 0, s_b: 4}, {kind: rot, s_a: 0, s_b: 20}, '
    '{kind: rot, s_a: 0, s_b: 90}, {kind: rot, s_a: 0, s_b: 400}, '
    '{kind: rot, s_a: 0, s_b: 1800}]}'
)
SR_OVERRIDE = (
    'model.connections={kind: per_layer, layers: ['
    '{kind: stretched_rot, s_a: 3, s_b: 96}, {kind: rot, s_a: 0, s_b: 4}, '
    '{kind: rot, s_a: 0, s_b: 20}, {kind: rot, s_a: 0, s_b: 90}, '
    '{kind: rot, s_a: 0, s_b: 400}, {kind: rot, s_a: 0, s_b: 1800}]}'
)


class TestExp100PerLayerConnections:
    def test_dot_override_replaces_whole_mapping(self):
        """model.yaml 末尾の PR 腕の上書きが既存のドット記法で通る。

        `_set_by_dots` が値を yaml.safe_load するので、フロー形式の
        マッピング/リストをそのまま1引数で渡せる(JSON文字列の口は不要)。
        """
        cfg = load_config(None, ['model.layers=6', PR_OVERRIDE])
        conn = cfg.model.connections
        assert conn.kind == 'per_layer'
        assert [c.kind for c in conn.layers] == ['perm'] + ['rot'] * 5
        assert [c.s_b for c in conn.layers] == [None, 4, 20, 90, 400, 1800]
        assert conn.layers[0].s_a is None

    def test_dot_override_on_top_of_rot_recipe(self, tmp_path):
        """R 腕のレシピ(kind=rot + s_a/s_b リスト)へ上書きして SR 腕にできる"""
        path = tmp_path / 'c.yaml'
        path.write_text(
            'model:\n  layers: 6\n  connections:\n    kind: rot\n'
            '    s_a: [3, 0, 0, 0, 0, 0]\n'
            '    s_b: [96, 4, 20, 90, 400, 1800]\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [SR_OVERRIDE])
        conn = cfg.model.connections
        assert conn.kind == 'per_layer'
        assert conn.s_a is None and conn.s_b is None   # リストは持ち越さない
        assert conn.layers[0].kind == 'stretched_rot'
        assert (conn.layers[0].s_a, conn.layers[0].s_b) == (3, 96)

    def test_yaml_form(self, tmp_path):
        path = tmp_path / 'c.yaml'
        path.write_text(
            'model:\n  layers: 2\n  connections:\n    kind: per_layer\n'
            '    layers:\n'
            '      - kind: stretched_rot\n        s_a: 3\n        s_b: 96\n'
            '      - kind: perm\n',
            encoding='utf-8',
        )
        cfg = load_config(str(path), [])
        assert [c.kind for c in cfg.model.connections.layers] == [
            'stretched_rot', 'perm']

    def test_length_must_match_layers(self):
        with pytest.raises(ValidationError):
            load_config(None, [
                'model.layers=3',
                'model.connections={kind: per_layer, layers: '
                '[{kind: perm}, {kind: perm}]}',
            ])

    def test_per_layer_without_layers_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, ['model.connections.kind=per_layer'])

    def test_rot_layer_requires_shifts(self):
        for kind in ('rot', 'stretched_rot'):
            with pytest.raises(ValidationError):
                load_config(None, [
                    'model.layers=1',
                    'model.connections={kind: per_layer, layers: '
                    f'[{{kind: {kind}}}]}}',
                ])
            with pytest.raises(ValidationError):
                load_config(None, [
                    'model.layers=1',
                    'model.connections={kind: per_layer, layers: '
                    f'[{{kind: {kind}, s_a: 3}}]}}',
                ])

    def test_perm_layer_ignores_missing_shifts(self):
        cfg = load_config(None, [
            'model.layers=1',
            'model.connections={kind: per_layer, layers: [{kind: perm}]}',
        ])
        assert cfg.model.connections.layers[0].s_a is None

    def test_unknown_layer_kind_raises(self):
        with pytest.raises(ValidationError):
            load_config(None, [
                'model.layers=1',
                'model.connections={kind: per_layer, layers: '
                '[{kind: affine}]}',
            ])

    def test_unknown_layer_key_raises(self):
        """層要素も extra='forbid'(タイポ検出)"""
        with pytest.raises(ValidationError):
            load_config(None, [
                'model.layers=1',
                'model.connections={kind: per_layer, layers: '
                '[{kind: perm, sa: 3}]}',
            ])

    def test_layer_shifts_must_be_scalars(self):
        """層ごとの s_a / s_b はリストではなく整数"""
        with pytest.raises(ValidationError):
            load_config(None, [
                'model.layers=1',
                'model.connections={kind: per_layer, layers: '
                '[{kind: rot, s_a: [3], s_b: [96]}]}',
            ])

    def test_random_kinds_at_top_level(self):
        for kind in ('random_seeded', 'random_uniform'):
            cfg = load_config(None, [f'model.connections.kind={kind}'])
            assert cfg.model.connections.kind == kind
            assert cfg.model.connections.layers is None

    def test_roundtrip(self, tmp_path):
        cfg = load_config(None, ['model.layers=6', SR_OVERRIDE])
        path = save_config(cfg, str(tmp_path / 'config.yaml'))
        assert load_config(path, []) == cfg
        dumped = dump_config(cfg)
        json.dumps(dumped)
        # 構造化配線の欄は既定値のまま dump される(rot 系の意味は不変)
        assert dumped['model']['connections']['layers'][0] == {
            'kind': 'stretched_rot', 's_a': 3, 's_b': 96, 'step': 1, 'shifts': None,
            'stride': None, 'offsets': None, 'grid': None, 'bits_per_cell': 1,
            'dedupe': False}
