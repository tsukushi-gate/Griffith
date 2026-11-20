"""パッケージ自立化のテスト。

`pip install griffith-logic` しただけの環境(リポジトリの experiments/ scripts/
tools/ configs/ が存在しない環境)でパッケージが自立して動くことを検証する。

- 軽量テスト: wheelの中身検査・パッケージ列挙の網羅・experiments依存の非存在
- @slow: 実際にwheelをビルドしクリーンvenvへインストールして import / CLI /
  最小学習(XOR 1エポック)まで通す

wheelのビルドとクリーンvenv作成には uv を使う。uv が無い環境ではskipする。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# pyproject.toml の [tool.setuptools] packages に列挙すべきパッケージ
_PYPROJECT = REPO_ROOT / 'pyproject.toml'


def _read_pyproject() -> dict:
    import tomllib

    with open(_PYPROJECT, 'rb') as f:
        return tomllib.load(f)


def _discover_packages() -> set[str]:
    """griffith_logic 配下の __init__.py を持つディレクトリを列挙する。"""
    found = set()
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT / 'griffith_logic'):
        dirnames[:] = [
            d for d in dirnames if not d.startswith('.') and d != '__pycache__'
        ]
        if '__init__.py' in filenames:
            rel = Path(dirpath).relative_to(REPO_ROOT)
            found.add(str(rel).replace(os.sep, '.'))
    return found


# ============================================================
# pyproject メタデータ(軽量)
# ============================================================

class TestPyprojectMetadata:
    def test_packages_are_explicit_and_complete(self):
        """packages の明示列挙にサブパッケージの取りこぼしが無いこと。

        新しいサブパッケージを追加したら pyproject.toml の
        [tool.setuptools] packages にも追加すること(この検査で気づける)。
        """
        declared = set(_read_pyproject()['tool']['setuptools']['packages'])
        assert _discover_packages() == declared

    def test_core_dependencies_include_torch(self):
        deps = _read_pyproject()['project']['dependencies']
        assert any(d.split('>')[0].split('=')[0].strip() == 'torch' for d in deps)

    def test_project_metadata_fields(self):
        proj = _read_pyproject()['project']
        assert proj['readme'] == 'README.md'
        assert proj['license']  # SPDX文字列
        assert proj['authors']
        assert proj['keywords']
        assert proj['classifiers']
        for key in ('Homepage', 'Documentation', 'Repository', 'Issues'):
            assert key in proj['urls']
        # SPDX licenseを指定したときはLicense classifierを併記しない(PyPI仕様)
        assert not any(c.startswith('License ::') for c in proj['classifiers'])


# ============================================================
# パッケージがリポジトリ相対パスに依存していないこと(軽量)
# ============================================================

class TestNoRepoRelativeDependency:
    def test_runner_has_no_dynamic_experiments_loader(self):
        """runner.py が cwd の experiments/ を動的ロードしないこと。"""
        text = (REPO_ROOT / 'griffith_logic' / 'training' / 'runner.py').read_text(
            encoding='utf-8'
        )
        assert '_load_experiments_module' not in text
        assert 'Path.cwd()' not in text

    def test_editor_does_not_use_cwd_as_app_dir(self):
        """griffith editor が cwd を無条件に app_dir へ渡さないこと。"""
        text = (REPO_ROOT / 'griffith_logic' / 'cli.py').read_text(encoding='utf-8')
        code = '\n'.join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith('#')
        )
        assert 'os.getcwd()' not in code

    def test_wandb_utils_lives_in_package(self):
        from griffith_logic.training import wandb_utils

        assert Path(wandb_utils.__file__).resolve().parent == (
            REPO_ROOT / 'griffith_logic' / 'training'
        )

    def test_checkpoint_utils_lives_in_package(self):
        from griffith_logic.training import checkpoint_utils

        assert Path(checkpoint_utils.__file__).resolve().parent == (
            REPO_ROOT / 'griffith_logic' / 'training'
        )

    def test_experiments_shims_reexport(self):
        """experiments/ の後方互換シムが実装と同じオブジェクトを返すこと。

        bare名(`import wandb_utils`)はテスト実行順でsys.pathが汚れると別実体を
        掴みうるため、ここではパッケージ修飾(`experiments.*`)で解決する。
        """
        import importlib

        ckpt_shim = importlib.import_module('experiments.checkpoint_utils')
        wandb_shim = importlib.import_module('experiments.wandb_utils')
        from griffith_logic.training import checkpoint_utils, wandb_utils

        assert wandb_shim.init_wandb is wandb_utils.init_wandb
        assert wandb_shim.finish_wandb is wandb_utils.finish_wandb
        assert ckpt_shim.save_checkpoint is checkpoint_utils.save_checkpoint
        assert ckpt_shim.load_checkpoint is checkpoint_utils.load_checkpoint

    def test_ledger_skipped_when_absent(self, tmp_path, monkeypatch):
        """台帳ファイルが無い環境(pip版)では追記をスキップすること。"""
        from griffith_logic.training import wandb_utils

        monkeypatch.chdir(tmp_path)
        assert wandb_utils._ledger_enabled() is False
        assert not (tmp_path / 'docs').exists()

    def test_launch_repo_root_env_override(self, tmp_path, monkeypatch):
        from griffith_logic.training import launch_cli

        (tmp_path / 'scripts').mkdir()
        (tmp_path / 'scripts' / 'modal_app.py').write_text('', encoding='utf-8')
        monkeypatch.setenv(launch_cli.REPO_ROOT_ENV, str(tmp_path))
        monkeypatch.setattr(launch_cli, '_PACKAGE_PARENT', tmp_path)
        assert launch_cli.resolve_repo_root('scripts/modal_app.py') == tmp_path
        assert launch_cli.resolve_repo_root('scripts/absent_xyz.py') is None

    def test_editor_reports_missing_repo(self, tmp_path, monkeypatch, capsys):
        """tools/ が無い環境では griffith editor が案内を出して1で終了する。"""
        from griffith_logic import cli
        from griffith_logic.training import launch_cli

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(launch_cli.REPO_ROOT_ENV, str(tmp_path))
        # site-packages相当(=ソースツリーを引かない)を模すためフォールバックも差し替える
        monkeypatch.setattr(launch_cli, '_PACKAGE_PARENT', tmp_path)
        assert cli.main(['editor']) == 1
        err = capsys.readouterr().err
        assert 'clone' in err and 'uv sync --extra editor' in err

    def test_version_flag(self, capsys):
        from griffith_logic import __version__, cli

        with pytest.raises(SystemExit) as exc:
            cli.main(['--version'])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out


# ============================================================
# wheelの中身・クリーンvenv(@slow)
# ============================================================

def _uv() -> str:
    path = shutil.which('uv')
    if path is None:
        pytest.skip('uv が見つからないためスキップ')
    return path


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise AssertionError(
            f'コマンド失敗: {" ".join(cmd)}\n--- stdout ---\n{proc.stdout}\n'
            f'--- stderr ---\n{proc.stderr}'
        )
    return proc


@pytest.fixture(scope='module')
def built_wheel(tmp_path_factory) -> Path:
    """リポジトリから wheel をビルドしてパスを返す。"""
    uv = _uv()
    dist = tmp_path_factory.mktemp('dist')
    _run([uv, 'build', '--wheel', '-o', str(dist)], cwd=REPO_ROOT)
    wheels = sorted(dist.glob('griffith_logic-*.whl'))
    assert len(wheels) == 1, f'wheelが1つではない: {wheels}'
    return wheels[0]


@pytest.mark.slow
class TestWheelContents:
    def test_wheel_contains_package_and_cuda_sources(self, built_wheel):
        names = zipfile.ZipFile(built_wheel).namelist()
        assert 'griffith_logic/__init__.py' in names
        assert 'griffith_logic/cuda/lut_kernels.cu' in names
        assert 'griffith_logic/cuda/lut_bindings.cpp' in names

    def test_wheel_excludes_legacy_shim(self, built_wheel):
        names = zipfile.ZipFile(built_wheel).namelist()
        assert not [n for n in names if n.startswith('magicdelta_logic')]

    def test_wheel_excludes_repo_only_dirs(self, built_wheel):
        names = zipfile.ZipFile(built_wheel).namelist()
        for prefix in ('experiments/', 'scripts/', 'tools/', 'tests/'):
            assert not [n for n in names if n.startswith(prefix)]


@pytest.fixture(scope='module')
def clean_venv(built_wheel, tmp_path_factory) -> Path:
    """wheelだけを入れたクリーンvenvを作り、python実行ファイルのディレクトリを返す。"""
    uv = _uv()
    venv = tmp_path_factory.mktemp('venv') / 'v'
    _run([uv, 'venv', str(venv)])
    env = dict(os.environ)
    env.pop('VIRTUAL_ENV', None)
    _run(
        [uv, 'pip', 'install', '--python', str(venv / 'bin' / 'python'),
         str(built_wheel)],
        env=env,
    )
    return venv / 'bin'


@pytest.mark.slow
class TestCleanVenv:
    def test_import_and_version(self, clean_venv):
        proc = _run([
            str(clean_venv / 'python'), '-c',
            'import griffith_logic as gl; import griffith_logic.models, '
            'griffith_logic.training; print(gl.__version__)',
        ])
        assert proc.stdout.strip() and proc.stdout.strip() != '0.0.0+unknown'

    def test_cli_help_and_version(self, clean_venv):
        assert 'サブコマンド' in _run([str(clean_venv / 'griffith'), '--help']).stdout
        assert 'griffith-logic' in _run(
            [str(clean_venv / 'griffith'), '--version']
        ).stdout

    def test_cuda_import_does_not_fail_on_cpu(self, clean_venv):
        _run([
            str(clean_venv / 'python'), '-c',
            'from griffith_logic.cuda import cuda_available, get_cuda_module; '
            'assert cuda_available() in (True, False)',
        ])

    def test_editor_without_repo_exits_1(self, clean_venv, tmp_path):
        """tools/ が無いのでエディタは起動せず案内して終了する。"""
        proc = subprocess.run(
            [str(clean_venv / 'griffith'), 'editor'],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert proc.returncode == 1
        assert '別配布' in proc.stderr

    def test_minimal_xor_training(self, clean_venv, tmp_path):
        """configs/ が無い環境でも Python API で XOR を1エポック学習できること。

        `griffith train configs/recipes/logic_xor.yaml` 相当を、リポジトリの
        configs/ に依存しない形(TrainConfigをコードで構築)で実行する。
        """
        script = tmp_path / 'smoke.py'
        script.write_text(
            'import json\n'
            'from griffith_logic.training.config import TrainConfig\n'
            'from griffith_logic.training.runner import run_training\n'
            'cfg = TrainConfig.model_validate({\n'
            "    'data': {'name': 'logic:xor', 'batch_size': 32},\n"
            "    'model': {'kind': 'logic_mlp', 'k': 16, 'layers': 2, "
            "'n_inputs': 2},\n"
            "    'optim': {'name': 'adam', 'lr': 0.1},\n"
            "    'run': {'epochs': 1, 'seed': 0, 'device': 'cpu', "
            "'wandb': False},\n"
            '})\n'
            f"result = run_training(cfg, out_dir={str(tmp_path / 'out')!r})\n"
            "print('RESULT_JSON=' + json.dumps("
            "{'steps': result['steps'], 'hard_acc': result['hard_acc']}))\n",
            encoding='utf-8',
        )
        proc = _run([str(clean_venv / 'python'), str(script)], cwd=tmp_path)
        line = [
            ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT_JSON=')
        ]
        assert line, proc.stdout
        result = json.loads(line[-1].removeprefix('RESULT_JSON='))
        assert result['steps'] > 0
        assert 0.0 <= result['hard_acc'] <= 1.0
