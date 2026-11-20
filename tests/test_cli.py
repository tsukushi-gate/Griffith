"""griffith統一CLI(griffith_logic/cli.py)のテスト。

subprocessではなく main(argv) 直呼びで高速に検証する。
- --help(日本語で全サブコマンドを説明)
- griffith train: exit code / config.yaml保存 / --from-run再実行
- griffith eval: hard acc出力
- griffith export-hdl: IR→Verilog/VHDL生成
"""

import os
import sys

import pytest
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LutLayer, cli

XOR_TRAIN_ARGS = [
    'data.name=logic:xor',
    'data.batch_size=32',
    'model.kind=logic_mlp',
    'model.k=8',
    'model.layers=2',
    'optim.lr=0.05',
    'run.epochs=3',
    'run.seed=0',
    'run.device=cpu',
]


# ============================================================
# --help
# ============================================================

class TestHelp:
    def test_help_exits_zero_and_lists_subcommands(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(['--help'])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        # 全サブコマンドが日本語説明つきで載っている
        for name in ('train', 'eval', 'export-hdl', 'editor'):
            assert name in out
        assert '学習' in out
        assert 'エディタ' in out

    def test_subcommand_help(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(['train', '--help'])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert 'ドット記法' in out

    def test_no_command_prints_help(self, capsys):
        assert cli.main([]) == 2
        assert 'train' in capsys.readouterr().out


# ============================================================
# griffith train
# ============================================================

class TestTrain:
    def test_train_with_overrides_only(self, tmp_path):
        """位置引数なし(key=value列のみ)で学習でき、config.yamlが保存される"""
        out = tmp_path / 'run'
        code = cli.main(['train', *XOR_TRAIN_ARGS, '--out-dir', str(out)])
        assert code == 0
        assert (out / 'config.yaml').exists()
        assert (out / 'checkpoints' / 'latest.pt').exists()
        assert (out / 'checkpoints' / 'best.pt').exists()

    def test_train_with_config_file(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text(
            'data:\n  name: logic:xor\n  batch_size: 32\n'
            'model:\n  kind: logic_mlp\n  k: 8\n  layers: 2\n'
            'optim:\n  lr: 0.05\n'
            'run:\n  epochs: 2\n  seed: 0\n  device: cpu\n',
            encoding='utf-8',
        )
        out = tmp_path / 'run'
        code = cli.main(['train', str(path), 'run.epochs=1', '--out-dir', str(out)])
        assert code == 0
        # オーバーライドが保存configにも反映される
        from griffith_logic.training.config import load_config
        assert load_config(str(out / 'config.yaml'), []).run.epochs == 1

    def test_from_run_rerun(self, tmp_path):
        """--from-run で保存済みconfig.yamlから再実行できる"""
        first = tmp_path / 'first'
        assert cli.main(['train', *XOR_TRAIN_ARGS, '--out-dir', str(first)]) == 0
        second = tmp_path / 'second'
        code = cli.main([
            'train', '--from-run', str(first),
            'run.epochs=1', '--out-dir', str(second),
        ])
        assert code == 0
        assert (second / 'config.yaml').exists()

    def test_from_run_and_config_conflict(self, tmp_path):
        code = cli.main([
            'train', 'config.yaml', '--from-run', str(tmp_path),
        ])
        assert code == 2

    def test_invalid_override_returns_1(self, capsys):
        assert cli.main(['train', 'model.bogus=1']) == 1
        err = capsys.readouterr().err
        assert 'エラー' in err and 'model.bogus' in err

    def test_missing_config_file_returns_1(self, tmp_path):
        assert cli.main(['train', str(tmp_path / 'nai.yaml')]) == 1


# ============================================================
# griffith eval
# ============================================================

class TestEval:
    def test_eval_prints_hard_acc(self, tmp_path, capsys):
        out = tmp_path / 'run'
        assert cli.main(['train', *XOR_TRAIN_ARGS, '--out-dir', str(out)]) == 0
        capsys.readouterr()  # trainの出力を捨てる
        code = cli.main([
            'eval', str(out / 'checkpoints' / 'best.pt'), '--device', 'cpu',
        ])
        assert code == 0
        eval_out = capsys.readouterr().out
        assert 'hard_acc=' in eval_out
        assert 'logic:xor' in eval_out

    def test_eval_with_dataset_override(self, tmp_path, capsys):
        """--datasetで評価データセットを差し替えられる(同一入出力幅)"""
        out = tmp_path / 'run'
        assert cli.main(['train', *XOR_TRAIN_ARGS, '--out-dir', str(out)]) == 0
        capsys.readouterr()
        code = cli.main([
            'eval', str(out / 'checkpoints' / 'best.pt'),
            '--dataset', 'logic:half_adder', '--device', 'cpu',
        ])
        # half_adderは2入力2出力でXORモデル(2入力1出力)と不整合→エラー終了
        assert code == 1

    def test_eval_missing_checkpoint_returns_1(self, tmp_path):
        assert cli.main(['eval', str(tmp_path / 'nai.pt')]) == 1


# ============================================================
# griffith export-hdl
# ============================================================

@pytest.fixture
def ir_json(tmp_path):
    """小さなLUTモデルからIR JSONを生成する"""
    from griffith_logic.export import export_ir
    from griffith_logic.ir import save_ir

    model = nn.Sequential(
        LutLayer(2, 4, n_inputs=2),
        LutLayer(4, 1, n_inputs=2),
    )
    model.eval()
    ir = export_ir(model, input_width=2, name='cli_test')
    return save_ir(ir, str(tmp_path / 'cli_test.json'))


class TestExportHdl:
    def test_verilog(self, ir_json, tmp_path):
        out = str(tmp_path / 'out.v')
        assert cli.main(['export-hdl', ir_json, '--lang', 'verilog', '-o', out]) == 0
        with open(out, encoding='utf-8') as f:
            assert 'module' in f.read()

    def test_vhdl(self, ir_json, tmp_path):
        out = str(tmp_path / 'out.vhdl')
        assert cli.main(['export-hdl', ir_json, '--lang', 'vhdl', '-o', out]) == 0
        with open(out, encoding='utf-8') as f:
            assert 'entity' in f.read()

    def test_default_output_path(self, ir_json):
        assert cli.main(['export-hdl', ir_json]) == 0
        assert os.path.exists(ir_json[:-5] + '.v')

    def test_missing_source_returns_1(self, tmp_path):
        assert cli.main(['export-hdl', str(tmp_path / 'nai.json')]) == 1
