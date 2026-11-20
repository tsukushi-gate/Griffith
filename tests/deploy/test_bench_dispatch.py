"""統一エントリ(scripts/bench_deploy.py)のスモークテスト。

重いベンチ実行(実際のC/Metalビルドや長い計測ループ)はここでは行わない
——動作確認は `scripts/bench_deploy.py --models har_dlgn_k600_gumbel --reps 50`
のような手動/別経路で行う(perf-hunt skill参照)。ここでは追加した
「モデル種別自動判定(detect_kind)」「RDDLGN対象名の解決」「旧スクリプト
(bench_rddlgn.py / bench_rddlgn_decode_gpu.py / bench_rddlgn_gpu.py)から
統一実装への引数委譲」の配線だけを軽く確認する。

scripts/ はパッケージ外なので、tests/deploy/conftest.py::_load_export_webbench
と同じ流儀でファイルパスから直接importする。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def bench_deploy():
    return _load('bench_deploy', 'scripts/bench_deploy.py')


# ---------------------------------------------------------------------------
# detect_kind: モデル種別の自動判定
# ---------------------------------------------------------------------------
class TestDetectKind:
    def test_dense_non_image_preproc(self, bench_deploy):
        doc = {'ir': {'nodes': [{'type': 'lut_layer', 'params': {}}]},
              'preproc': {'kind': 'thermometer'}}
        assert bench_deploy.detect_kind(doc) == 'dense'

    def test_conv_image_preproc(self, bench_deploy):
        doc = {'ir': {'nodes': [{'type': 'conv2d', 'params': {}}]},
              'preproc': {'kind': 'image_thermometer'}}
        assert bench_deploy.detect_kind(doc) == 'conv'

    def test_conv_image_lowlevel_preproc(self, bench_deploy):
        doc = {'ir': {'nodes': []}, 'preproc': {'kind': 'image_lowlevel'}}
        assert bench_deploy.detect_kind(doc) == 'conv'

    def test_rddlgn_top_level_recurrent(self, bench_deploy):
        doc = {'ir': {'nodes': [{'type': 'recurrent',
                                 'params': {'body': {'nodes': []}}}]},
              'preproc': {}}
        assert bench_deploy.detect_kind(doc) == 'rddlgn'

    def test_rddlgn_recurrent_wins_over_image_preproc(self, bench_deploy):
        """recurrentノードがあればpreproc種別に関わらずrddlgn優先。"""
        doc = {'ir': {'nodes': [{'type': 'recurrent',
                                 'params': {'body': {'nodes': []}}}]},
              'preproc': {'kind': 'image_thermometer'}}
        assert bench_deploy.detect_kind(doc) == 'rddlgn'

    def test_rddlgn_nested_recurrent_in_body(self, bench_deploy):
        """recurrentが別ノードのbody(入れ子IRGraph)の中に埋まっていても
        拾う(recurrent bodyは入れ子グラフ、
        griffith_logic/ir/serialize.py参照)。"""
        doc = {'ir': {'nodes': [
            {'type': 'concat', 'params': {
                'body': {'nodes': [{'type': 'recurrent',
                                    'params': {'body': {'nodes': []}}}]}}}]},
              'preproc': {}}
        assert bench_deploy.detect_kind(doc) == 'rddlgn'

    def test_missing_preproc_defaults_dense(self, bench_deploy):
        doc = {'ir': {'nodes': []}}
        assert bench_deploy.detect_kind(doc) == 'dense'

    def test_missing_ir_defaults_dense(self, bench_deploy):
        assert bench_deploy.detect_kind({}) == 'dense'


# ---------------------------------------------------------------------------
# main(): 対象解決(cc探索やビルドに入る前の早期リターン経路だけを確認、
# 実際のビルド・計測はしない)
# ---------------------------------------------------------------------------
class TestMainTargetResolution:
    def test_no_target_errors(self, bench_deploy, capsys):
        rc = bench_deploy.main(['--models', 'no_such_json_stem_xyz'])
        assert rc == 1
        assert '対象モデルが無い' in capsys.readouterr().err

    def test_unknown_rddlgn_model_name_errors(self, bench_deploy, capsys):
        rc = bench_deploy.main(['--rddlgn-only', '--rddlgn-models',
                                'no_such_rddlgn_model_xyz'])
        assert rc == 1
        assert '未知のRDDLGNモデル' in capsys.readouterr().err

    def test_cuda_rddlgn_without_host_errors(self, bench_deploy, capsys):
        rc = bench_deploy.main(['--backend', 'cuda', '--rddlgn-only',
                                '--rddlgn-models', 'tiny'])
        assert rc == 1
        assert '--host' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 旧scripts/bench_rddlgn*.py: 統一実装への引数委譲(deploy.main呼び出しは
# モックし、実ビルド・実計測は走らせない)
# ---------------------------------------------------------------------------
class _FakeDeploy:
    """bench_deploy.main呼び出しを記録するだけのダミー。"""

    def __init__(self):
        self.calls = []

    def main(self, argv):
        self.calls.append(list(argv))
        return 0


class TestBenchRddlgnWrapper:
    def test_forwards_models_and_flags(self, monkeypatch, capsys):
        rddlgn = _load('bench_rddlgn', 'scripts/bench_rddlgn.py')
        fake = _FakeDeploy()
        monkeypatch.setattr(rddlgn, '_load_bench_deploy', lambda: fake)
        monkeypatch.setattr(sys, 'argv', [
            'bench_rddlgn.py', '--models', 'copy_tiny', '--skip-verify',
            '--max-thr', '4', '--out', 'out.json'])

        rc = rddlgn.main()

        assert rc == 0
        assert len(fake.calls) == 1
        argv = fake.calls[0]
        assert '--rddlgn-only' in argv
        assert argv[argv.index('--rddlgn-models') + 1] == 'copy_tiny'
        assert '--rddlgn-skip-verify' in argv
        assert argv[argv.index('--rddlgn-max-thr') + 1] == '4'
        assert argv[argv.index('--rddlgn-out') + 1] == 'out.json'
        assert '非推奨' in capsys.readouterr().err

    def test_no_models_forwards_rddlgn_flag_not_models(self, monkeypatch):
        """--models省略時は旧既定(全4モデル)と同じ意味の--rddlgnを送る
        (--rddlgn-modelsは送らない、bench_deploy.py側のsorted(all_rddlgn)
        分岐に委ねる)。"""
        rddlgn = _load('bench_rddlgn', 'scripts/bench_rddlgn.py')
        fake = _FakeDeploy()
        monkeypatch.setattr(rddlgn, '_load_bench_deploy', lambda: fake)
        monkeypatch.setattr(sys, 'argv', ['bench_rddlgn.py'])

        rddlgn.main()

        argv = fake.calls[0]
        assert '--rddlgn-only' in argv
        assert '--rddlgn-models' not in argv


class TestBenchRddlgnDecodeGpuWrapper:
    def test_forwards_decode_metrics(self, monkeypatch):
        decode_gpu = _load('bench_rddlgn_decode_gpu',
                           'scripts/bench_rddlgn_decode_gpu.py')
        fake = _FakeDeploy()
        monkeypatch.setattr(decode_gpu, '_load_bench_deploy', lambda: fake)
        monkeypatch.setattr(sys, 'argv', [
            'bench_rddlgn_decode_gpu.py', '--backend', 'metal',
            '--models', 'tiny', '--lanes', '1,32'])

        rc = decode_gpu.main()

        assert rc == 0
        argv = fake.calls[0]
        assert argv[argv.index('--backend') + 1] == 'metal'
        assert '--rddlgn-only' in argv
        assert argv[argv.index('--rddlgn-gpu-metrics') + 1] == 'decode'
        assert argv[argv.index('--rddlgn-models') + 1] == 'tiny'
        assert argv[argv.index('--rddlgn-lanes') + 1] == '1,32'

    def test_cuda_without_host_errors_before_delegating(self, monkeypatch,
                                                         capsys):
        decode_gpu = _load('bench_rddlgn_decode_gpu',
                           'scripts/bench_rddlgn_decode_gpu.py')
        fake = _FakeDeploy()
        monkeypatch.setattr(decode_gpu, '_load_bench_deploy', lambda: fake)
        monkeypatch.setattr(sys, 'argv',
                            ['bench_rddlgn_decode_gpu.py', '--backend', 'cuda'])

        rc = decode_gpu.main()

        assert rc == 1
        assert not fake.calls
        assert '--host' in capsys.readouterr().err


class TestBenchRddlgnGpuWrapper:
    def test_forwards_tf_metrics(self, monkeypatch):
        tf_gpu = _load('bench_rddlgn_gpu', 'scripts/bench_rddlgn_gpu.py')
        fake = _FakeDeploy()
        monkeypatch.setattr(tf_gpu, '_load_bench_deploy', lambda: fake)
        monkeypatch.setattr(sys, 'argv', [
            'bench_rddlgn_gpu.py', '--backend', 'metal',
            '--models', 'copy_tiny', '--bench-batches', '1,8'])

        rc = tf_gpu.main()

        assert rc == 0
        argv = fake.calls[0]
        assert '--rddlgn-only' in argv
        assert argv[argv.index('--rddlgn-gpu-metrics') + 1] == 'tf'
        assert argv[argv.index('--rddlgn-models') + 1] == 'copy_tiny'
        assert argv[argv.index('--rddlgn-tf-batches') + 1] == '1,8'
