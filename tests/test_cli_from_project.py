"""griffith train --from-projectのテスト。

エディタプロジェクトJSON(ビジュアルエディタの ProjectSchema 形式、
outputs/editor_projects/*.json)から学習設定を解決し、共通ランナーへ
渡して学習を完走できることを検証する。

- プロジェクトJSONは直接dict/jsonで組み立てる(tools.editor.backend側は
  console script実行時にimportできないため、mdのCLI実装もJSONを直接読む
  設計。テストではその生成物の形をそのまま模す)
- ネットワーク(IR)はexport_ir経由で作った妥当なIRを使う(存在検証のみに
  使われ、実際のモデル構築はtrain_configsのmodelセクションを使う)
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LutLayer, cli
from griffith_logic.export import export_ir
from griffith_logic.ir import ir_to_dict

XOR_MODEL_CFG = {'kind': 'logic_mlp', 'k': 8, 'layers': 2, 'n_inputs': 2}


def _xor_ir_doc(seed: int = 0) -> dict:
    """XORのlogic_mlp(GroupSumなし生ビット出力)相当のIRを作る"""
    torch.manual_seed(seed)
    model = nn.Sequential(
        LutLayer(2, XOR_MODEL_CFG['k'], n_inputs=2, parameterization='softmax'),
        LutLayer(XOR_MODEL_CFG['k'], 1, n_inputs=2, parameterization='softmax'),
    )
    model.eval()
    graph = export_ir(model, input_width=2, name='xor_net')
    return ir_to_dict(graph)


def _write_project(
    path,
    *,
    networks: dict | None = None,
    train_configs: dict | None = None,
    optimizers: dict | None = None,
) -> str:
    now = time.time()
    doc = {
        'id': 'proj-xor',
        'name': 'proj-xor',
        'description': '',
        'networks': networks if networks is not None else {'xor_net': _xor_ir_doc()},
        'train_configs': train_configs if train_configs is not None else {
            'global': {
                'data': {'name': 'logic:xor', 'batch_size': 32},
                'model': dict(XOR_MODEL_CFG),
                'optim': {'name': 'adam', 'lr': 0.05},
                'run': {'epochs': 3, 'seed': 0, 'device': 'cpu'},
            },
        },
        'dataset_refs': [],
        'run_refs': [],
        'optimizers': optimizers if optimizers is not None else {},
        'monitors': {},
        'executors': {},
        'created_at': now,
        'updated_at': now,
        'deleted': False,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    return str(path)


class TestFromProject:
    def test_trains_and_completes(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        out = tmp_path / 'run'
        code = cli.main([
            'train', '--from-project', project_path, '--out-dir', str(out),
        ])
        assert code == 0
        assert (out / 'config.yaml').exists()
        assert (out / 'checkpoints' / 'latest.pt').exists()

    def test_explicit_network_and_config_name(self, tmp_path):
        project_path = _write_project(
            tmp_path / 'proj.json',
            networks={'net_a': _xor_ir_doc(1)},
            train_configs={
                'cfg_a': {
                    'data': {'name': 'logic:xor', 'batch_size': 32},
                    'model': dict(XOR_MODEL_CFG),
                    'optim': {'name': 'adam', 'lr': 0.05},
                    'run': {'epochs': 1, 'seed': 0, 'device': 'cpu'},
                },
            },
        )
        out = tmp_path / 'run'
        code = cli.main([
            'train', '--from-project', project_path,
            '--network', 'net_a', '--config', 'cfg_a',
            '--out-dir', str(out),
        ])
        assert code == 0

    def test_optimizer_override(self, tmp_path):
        project_path = _write_project(
            tmp_path / 'proj.json',
            optimizers={'fast': {'lr': 0.2}},
        )
        out = tmp_path / 'run'
        code = cli.main([
            'train', '--from-project', project_path, '--optimizer', 'fast',
            '--out-dir', str(out),
        ])
        assert code == 0
        from griffith_logic.training.config import load_config
        cfg = load_config(str(out / 'config.yaml'), [])
        assert cfg.optim.lr == 0.2

    def test_dotted_overrides_apply_on_top(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        out = tmp_path / 'run'
        code = cli.main([
            'train', '--from-project', project_path, 'run.epochs=1',
            '--out-dir', str(out),
        ])
        assert code == 0
        from griffith_logic.training.config import load_config
        cfg = load_config(str(out / 'config.yaml'), [])
        assert cfg.run.epochs == 1

    def test_missing_network_name_errors(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        code = cli.main([
            'train', '--from-project', project_path, '--network', 'nai',
        ])
        assert code == 1

    def test_missing_config_name_errors(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        code = cli.main([
            'train', '--from-project', project_path, '--config', 'nai',
        ])
        assert code == 1

    def test_missing_project_file_errors(self, tmp_path):
        code = cli.main([
            'train', '--from-project', str(tmp_path / 'nai.json'),
        ])
        assert code == 1

    def test_invalid_ir_network_errors(self, tmp_path):
        project_path = _write_project(
            tmp_path / 'proj.json',
            networks={'broken': {'nodes': 'not-a-list'}},
        )
        code = cli.main(['train', '--from-project', project_path])
        assert code == 1

    def test_from_project_and_from_run_conflict(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        code = cli.main([
            'train', '--from-project', project_path,
            '--from-run', str(tmp_path),
        ])
        assert code == 2

    def test_from_project_and_config_conflict(self, tmp_path):
        project_path = _write_project(tmp_path / 'proj.json')
        code = cli.main([
            'train', 'config.yaml', '--from-project', project_path,
        ])
        assert code == 2

    def test_network_flag_without_from_project_errors(self):
        code = cli.main(['train', 'run.epochs=1', '--network', 'x'])
        assert code == 2
