"""AI駆動実験ループCLIのテスト。

wandb/Modalへの実通信は行わない(全てモック/縮退経路)。
- griffith launch --dry-run のグリッド展開
- griffith runs list / compare(一時jsonl・縮退動作)
- configs/recipes/logic_xor.yaml のload_config+expected_metrics照合
- griffith report のmarkdownフォールバック生成
- griffith sweep のガイダンス(WANDB_API_KEY無し)とspec検証
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import cli
from griffith_logic.training import launch_cli, runs_cli, sweep
from griffith_logic.training.config import check_expected_metrics, load_config

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
XOR_RECIPE = os.path.join(REPO_ROOT, 'configs', 'recipes', 'logic_xor.yaml')
MNIST_RECIPE = os.path.join(REPO_ROOT, 'configs', 'recipes', 'dlgn_mnist_quick.yaml')
EXAMPLE_SWEEP = os.path.join(REPO_ROOT, 'configs', 'sweeps', 'example_mnist.yaml')


# ============================================================
# --help に新サブコマンドが載る
# ============================================================

class TestHelp:
    def test_help_lists_new_subcommands(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(['--help'])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        for name in ('sweep', 'launch', 'collect', 'runs', 'report'):
            assert name in out


# ============================================================
# griffith launch --grid
# ============================================================

class TestGridExpansion:
    def test_parse_grid_axes_and_fixed(self):
        axes, fixed = launch_cli.parse_grid(
            'optim.lr=[0.001,0.01] model.k=[64,256] run.epochs=3'
        )
        assert axes == [('optim.lr', ['0.001', '0.01']), ('model.k', ['64', '256'])]
        assert fixed == ['run.epochs=3']

    def test_parse_grid_requires_list_axis(self):
        with pytest.raises(ValueError):
            launch_cli.parse_grid('run.epochs=3')

    def test_parse_grid_rejects_garbage(self):
        with pytest.raises(ValueError):
            launch_cli.parse_grid('optim.lr=[0.1] ???')

    def test_expand_grid_cartesian_product_with_names(self):
        axes, fixed = launch_cli.parse_grid('optim.lr=[0.001,0.01] model.k=[64,256]')
        combos = launch_cli.expand_grid(axes, fixed)
        assert len(combos) == 4
        names = [c['name'] for c in combos]
        assert names == ['lr0.001_k64', 'lr0.001_k256', 'lr0.01_k64', 'lr0.01_k256']
        # run名がrun.run_nameオーバーライドとして入る
        assert 'run.run_name=lr0.001_k64' in combos[0]['overrides']
        assert 'optim.lr=0.001' in combos[0]['overrides']
        assert 'model.k=64' in combos[0]['overrides']

    def test_expand_grid_prefixes_base_run_name(self):
        combos = launch_cli.expand_grid(
            [('optim.lr', ['0.1'])], [], base_run_name='base'
        )
        assert combos[0]['name'] == 'base_lr0.1'

    def test_cli_dry_run_prints_table(self, capsys):
        code = cli.main([
            'launch', '--grid', 'optim.lr=[0.001,0.01] model.k=[64,256]',
            '--dry-run',
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert '4件' in out
        assert 'lr0.001_k64' in out
        assert 'lr0.01_k256' in out

    def test_cli_invalid_grid_returns_1(self, capsys):
        code = cli.main(['launch', '--grid', 'run.epochs=3', '--dry-run'])
        assert code == 1
        assert 'エラー' in capsys.readouterr().err


class TestLaunchLocal:
    def test_local_runs_each_combo_with_overrides(self, monkeypatch, capsys):
        """--localが組合せごとにrun_trainingを呼ぶ(実学習はモック)"""
        seen = []

        def fake_run_training(cfg, **kwargs):
            seen.append(cfg)
            return {'hard_acc': 0.5, 'best_hard_acc': 0.5, 'steps': 1}

        import griffith_logic.training.runner as runner_mod
        monkeypatch.setattr(runner_mod, 'run_training', fake_run_training)

        code = cli.main(['launch', '--grid', 'optim.lr=[0.01,0.02]', '--local'])
        assert code == 0
        assert [cfg.optim.lr for cfg in seen] == [0.01, 0.02]
        # run名にグリッド値が含まれる
        assert seen[0].run.run_name == 'lr0.01'
        assert seen[1].run.run_name == 'lr0.02'
        assert 'launch結果' in capsys.readouterr().out


# ============================================================
# griffith runs list / compare
# ============================================================

def _write_jsonl(path, entries):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')


class TestRunsList:
    def test_list_shows_latest_first(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / '.wandb_runs.jsonl', [
            {'run_id': f'run{i}', 'run_name': f'name{i}', 'group': 'g',
             'tags': [], 'started_at': f'2026-07-0{i}T00:00:00'}
            for i in (1, 2, 3)
        ])
        assert cli.main(['runs', 'list', '-n', '2']) == 0
        out = capsys.readouterr().out
        assert 'run3' in out and 'run2' in out
        assert 'run1' not in out  # -n 2で最新2件のみ
        assert out.index('run3') < out.index('run2')  # 新しい順
        assert '| run_id |' in out  # markdown表

    def test_list_without_jsonl_is_not_an_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli.main(['runs', 'list']) == 0
        assert '.wandb_runs.jsonl' in capsys.readouterr().out

    def test_list_skips_broken_lines(self, tmp_path):
        p = tmp_path / '.wandb_runs.jsonl'
        p.write_text('{"run_id": "ok"}\n{broken\n', encoding='utf-8')
        entries = runs_cli.read_runs_jsonl(p)
        assert [e['run_id'] for e in entries] == ['ok']


class TestRunsCompareDegraded:
    def test_compare_without_api_key_uses_jsonl(self, tmp_path, monkeypatch, capsys):
        """WANDB_API_KEY無しはwandb通信せずjsonl情報のみで縮退動作する"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('WANDB_API_KEY', raising=False)
        _write_jsonl(tmp_path / '.wandb_runs.jsonl', [
            {'run_id': 'abc', 'run_name': 'xor_run', 'group': 'md-train',
             'tags': ['md-train'], 'started_at': '2026-07-02T10:00:00'},
        ])
        assert cli.main(['runs', 'compare', 'abc', 'zzz']) == 0
        out = capsys.readouterr().out
        assert '縮退' in out
        assert 'xor_run' in out
        assert '(jsonl未記録)' in out  # 未知IDも行として出る

    def test_compare_table_formats(self):
        rows = [
            {'run_id': 'a', 'name': 'run_a', 'state': 'finished',
             'summary': {'hard_acc': 0.91}, 'config': {'optim.lr': 0.01}},
            {'run_id': 'b', 'name': 'run_b', 'state': 'finished',
             'summary': {'hard_acc': 0.95}, 'config': {'optim.lr': 0.02}},
        ]
        md = runs_cli.build_compare_table(rows, metrics=['hard_acc'])
        assert '| hard_acc | 0.9100 | 0.9500 |' in md
        assert 'config: optim.lr' in md  # 差分configのみ載る
        csv = runs_cli.build_compare_table(rows, metrics=['hard_acc'], fmt='csv')
        assert csv.splitlines()[0] == '項目,a,b'

    def test_compare_table_hides_identical_config(self):
        rows = [
            {'run_id': 'a', 'name': 'x', 'state': 'finished',
             'summary': {}, 'config': {'seed': 42}},
            {'run_id': 'b', 'name': 'y', 'state': 'finished',
             'summary': {}, 'config': {'seed': 42}},
        ]
        assert 'config: seed' not in runs_cli.build_compare_table(rows)


# ============================================================
# expected_metrics
# ============================================================

class TestExpectedMetrics:
    def test_xor_recipe_loads_with_expected_metrics(self):
        cfg = load_config(XOR_RECIPE)
        assert cfg.data.name == 'logic:xor'
        assert cfg.expected_metrics == {
            'hard_acc_min': 1.0, 'hard_exact_match_min': 1.0,
        }

    def test_mnist_recipe_loads(self):
        cfg = load_config(MNIST_RECIPE)
        assert cfg.model.k == 2000 and cfg.run.epochs == 3
        assert cfg.expected_metrics == {'hard_acc_min': 0.90}

    def test_invalid_expected_key_rejected(self):
        with pytest.raises(ValueError):
            load_config(XOR_RECIPE, ['expected_metrics.hard_acc=1.0'])

    def test_check_min_and_max_bounds(self):
        result = {'hard_acc': 0.95, 'final_loss': 0.2}
        verdict = check_expected_metrics(
            {'hard_acc_min': 0.9, 'final_loss_max': 0.5}, result
        )
        assert verdict['ok'] is True
        verdict = check_expected_metrics({'hard_acc_min': 0.99}, result)
        assert verdict['ok'] is False
        assert verdict['checks'][0]['actual'] == 0.95

    def test_check_missing_metric_is_unmet(self):
        verdict = check_expected_metrics({'nai_metric_min': 1.0}, {'hard_acc': 1.0})
        assert verdict['ok'] is False
        assert verdict['checks'][0]['actual'] is None

    def test_strict_exit_code_with_mocked_runner(self, tmp_path, monkeypatch):
        """--strict: 照合未達で非0 / 合格で0(run_trainingはモック)"""
        def make_fake(ok):
            def fake_run_training(cfg, **kwargs):
                return {
                    'hard_acc': 1.0, 'best_hard_acc': 1.0, 'steps': 1,
                    'out_dir': str(tmp_path), 'config_path': 'config.yaml',
                    'expected_metrics': {'ok': ok, 'checks': []},
                }
            return fake_run_training

        import griffith_logic.training.runner as runner_mod
        monkeypatch.setattr(runner_mod, 'run_training', make_fake(False))
        assert cli.main(['train', XOR_RECIPE, '--strict']) == 1
        monkeypatch.setattr(runner_mod, 'run_training', make_fake(True))
        assert cli.main(['train', XOR_RECIPE, '--strict']) == 0
        # --strict無しは未達でも0(照合結果は返り値/表示のみ)
        monkeypatch.setattr(runner_mod, 'run_training', make_fake(False))
        assert cli.main(['train', XOR_RECIPE]) == 0

    def test_xor_recipe_trains_and_verifies(self, tmp_path, capsys):
        """`griffith train configs/recipes/logic_xor.yaml` が照合込みで動く(実学習)"""
        code = cli.main([
            'train', XOR_RECIPE, '--strict', '--out-dir', str(tmp_path / 'run'),
        ])
        out = capsys.readouterr().out
        assert code == 0, out
        assert 'expected_metrics照合' in out
        assert '全件合格' in out


# ============================================================
# griffith report のmarkdownフォールバック
# ============================================================

class TestReportFallback:
    @pytest.fixture(autouse=True)
    def no_wandb_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv('WANDB_API_KEY', raising=False)
        monkeypatch.chdir(tmp_path)

    def test_generates_markdown_with_template_sections(self, tmp_path, capsys):
        _write_jsonl(tmp_path / '.wandb_runs.jsonl', [
            {'run_id': 'abc123', 'run_name': 'xor_run', 'group': 'md-train',
             'tags': [], 'started_at': '2026-07-02T10:00:00'},
        ])
        code = cli.main(['report', 'abc123', '--title', 'XOR収束確認'])
        assert code == 0
        out = capsys.readouterr().out
        assert 'markdownレポート' in out

        auto_dir = tmp_path / 'reports' / 'auto'
        files = list(auto_dir.glob('*_XOR収束確認.md'))
        assert len(files) == 1
        text = files[0].read_text(encoding='utf-8')
        # 手書きレポートの構造(目的/設定/結果/結論)を踏襲
        for section in ('# XOR収束確認', '## 1. 目的', '## 2. 設定',
                        '## 3. 結果', '## 4. 結論'):
            assert section in text
        assert 'abc123' in text and 'xor_run' in text
        assert 'TODO' in text  # 結論はプレースホルダ

    def test_custom_dest(self, tmp_path):
        code = cli.main(['report', 'r1', '--dest', str(tmp_path / 'alt')])
        assert code == 0
        assert list((tmp_path / 'alt').glob('*.md'))


# ============================================================
# griffith sweep: ガイダンスとspec検証(wandb通信なし)
# ============================================================

class TestSweep:
    def test_guidance_without_api_key(self, monkeypatch, capsys):
        monkeypatch.delenv('WANDB_API_KEY', raising=False)
        code = cli.main(['sweep', EXAMPLE_SWEEP])
        assert code == 1
        out = capsys.readouterr().out
        assert 'WANDB_API_KEY' in out  # 日本語ガイダンス(エラーでなく案内)

    def test_load_example_sweep_spec(self):
        spec = sweep.load_sweep_spec(EXAMPLE_SWEEP)
        assert spec['method'] == 'bayes'
        assert spec['metric']['name'] == 'hard_acc'
        assert 'optim.lr' in spec['parameters']

    def test_load_spec_rejects_bad_method(self, tmp_path):
        p = tmp_path / 's.yaml'
        p.write_text(
            'method: hillclimb\nmetric: {name: acc}\n'
            'parameters: {optim.lr: {values: [0.1]}}\n', encoding='utf-8',
        )
        with pytest.raises(ValueError):
            sweep.load_sweep_spec(str(p))

    def test_load_spec_rejects_non_dotted_parameter(self, tmp_path):
        p = tmp_path / 's.yaml'
        p.write_text(
            'method: grid\nmetric: {name: acc}\nparameters: {lr: {values: [0.1]}}\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError):
            sweep.load_sweep_spec(str(p))

    def test_build_sweep_config_embeds_base_config(self):
        spec = sweep.load_sweep_spec(EXAMPLE_SWEEP)
        config = sweep.build_sweep_config(spec)
        assert config['method'] == 'bayes'
        embedded = config['parameters'][sweep.BASE_CONFIG_PARAM]['value']
        base = json.loads(embedded)
        assert base['model']['k'] == 2000  # dlgn_mnist_quick.yaml由来

    def test_build_sweep_config_rejects_typo_dotpath(self, tmp_path):
        p = tmp_path / 's.yaml'
        p.write_text(
            'method: grid\nmetric: {name: acc}\n'
            'parameters: {model.bogus: {values: [1]}}\n', encoding='utf-8',
        )
        spec = sweep.load_sweep_spec(str(p))
        with pytest.raises(ValueError):  # pydantic ValidationErrorを含む
            sweep.build_sweep_config(spec)

    def test_config_from_sweep_params(self):
        base = json.dumps({'data': {'name': 'logic:xor'},
                           'model': {'kind': 'logic_mlp', 'k': 8}})
        cfg = sweep.config_from_sweep_params({
            sweep.BASE_CONFIG_PARAM: base,
            'optim.lr': 0.02,
            'model.k': 16,
            '_wandb_internal': 'ignored',  # ドット無しキーは無視される
        })
        assert cfg.optim.lr == 0.02
        assert cfg.model.k == 16
        assert cfg.data.name == 'logic:xor'


# ============================================================
# griffith collect: 新規ディレクトリ選別(Modal通信なし)
# ============================================================

class TestCollectSelection:
    def test_select_new_dirs_skips_existing(self, tmp_path):
        (tmp_path / 'run_a').mkdir()
        new = launch_cli.select_new_dirs(['run_a', 'run_b'], tmp_path)
        assert new == ['run_b']

    def test_select_new_dirs_with_missing_dest(self, tmp_path):
        new = launch_cli.select_new_dirs(['x'], tmp_path / 'nai')
        assert new == ['x']
