"""RDDLGNClassifier(tiny) の C ランタイム端到端検証。

bundle → generate_c → cc ビルド → 実行 が ir/simulate と整数厳密に
一致することを、既存の `verify_c_matches_simulate`(deploy/verify.py、
tests/deploy/test_verify_c.py 等で既に使われている経路)でそのまま確かめる。
recurrent専用の新しいCヘルパは要らなかった(既存のbits入力経路が
無改修で通る)。
"""

import numpy as np
import pytest

from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.verify import verify_c_matches_simulate
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import build_rddlgn_tiny_classifier, rddlgn_classifier_doc


class TestVerifyCMatchesSimulate:
    def test_random_samples_match_exactly(self, cc, tmp_path, rddlgn_tiny_bundle):
        result = verify_c_matches_simulate(
            rddlgn_tiny_bundle, n_samples=8, seed=0, cc=cc,
            work_dir=tmp_path / 'verify')
        assert result['max_abs_diff'] == 0
        assert result['path'] == 'bits'
        assert result['n_out'] == 3

    def test_extreme_inputs_match(self, cc, tmp_path, rddlgn_tiny_bundle):
        """全0/全1の埋め込みビット列でも一致する(端数ビットの扱い含む)。"""
        from griffith_logic.deploy.verify import build_c_model, run_c_model

        graph = ir_from_dict(rddlgn_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        exe = build_c_model(rddlgn_tiny_bundle, tmp_path / 'gen', cc=cc)
        for label, x in (('zeros', np.zeros(in_width, dtype=np.uint8)),
                         ('ones', np.ones(in_width, dtype=np.uint8))):
            want = simulate(graph, x[None, :])[0].astype(np.int64)
            got = run_c_model(exe, x.tobytes(), n_out)
            assert np.array_equal(got, want), label


class TestMultipleModelConfigurations:
    """モデルの重みが違っても一致する(1つの固定シードで偶然揃った、を排除)。"""

    @pytest.mark.parametrize('seed', [1, 2, 3])
    def test_different_seeds_match(self, cc, tmp_path, seed):
        model, cfg = build_rddlgn_tiny_classifier(seed=seed)
        doc = rddlgn_classifier_doc(model, cfg.seq_len, labels=['a', 'b', 'c'])
        from griffith_logic.deploy.bundle import build_bundle
        bundle = build_bundle(doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=3, seed=seed, cc=cc,
            work_dir=tmp_path / f'verify{seed}')
        assert result['max_abs_diff'] == 0

    def test_init_mode_one_matches(self, cc, tmp_path):
        """init_mode='one'(全1初期状態)でも一致する。"""
        import torch

        from griffith_logic.models.rddlgn import RDDLGNClassifier, RDDLGNConfig

        torch.manual_seed(0)
        cfg = RDDLGNConfig.tiny()
        cfg.init_mode = 'one'
        model = RDDLGNClassifier(cfg, n_classes=3)
        model.eval()
        doc = rddlgn_classifier_doc(model, cfg.seq_len, labels=['a', 'b', 'c'])
        from griffith_logic.deploy.bundle import build_bundle
        bundle = build_bundle(doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=3, seed=0, cc=cc, work_dir=tmp_path / 'verify_one')
        assert result['max_abs_diff'] == 0


class TestGeneratedSourceProperties:
    def test_no_heap_allocation(self, rddlgn_tiny_bundle):
        files = generate_c(rddlgn_tiny_bundle, name='model', validate=False)
        source = files['model.c']
        for bad in ('malloc', 'calloc', 'realloc', 'free('):
            assert bad not in source, f'生成コードに {bad} がある'

    def test_no_pthread_dependency(self, rddlgn_tiny_bundle):
        files = generate_c(rddlgn_tiny_bundle, name='model', validate=False)
        for text in files.values():
            assert 'pthread' not in text

    def test_header_exposes_sizes(self, rddlgn_tiny_bundle):
        files = generate_c(rddlgn_tiny_bundle, name='model', validate=False)
        header = files['model.h']
        graph = ir_from_dict(rddlgn_tiny_bundle['ir'])
        assert f'#define MDL_INPUT_WIDTH {int(graph.inputs[0].width)}' in header
        assert f'#define MDL_OUTPUT_WIDTH {int(graph.outputs[0].width)}' in header

    def test_step_functions_are_emitted(self, rddlgn_tiny_bundle):
        """recurrentのbodyがmdl_step_<id>という静的関数へ切り出されている。"""
        files = generate_c(rddlgn_tiny_bundle, name='model', validate=False)
        source = files['model.c']
        assert 'static void mdl_step_enc_n_rec(' in source
        assert 'static void mdl_step_enc_k_rec(' in source

    def test_compiles_with_strict_warnings(self, cc, tmp_path, rddlgn_tiny_bundle):
        import subprocess

        from .conftest import RUNTIME_C

        files = generate_c(rddlgn_tiny_bundle, name='model', validate=False)
        work = tmp_path / 'strict'
        work.mkdir()
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              '-c', str(work / 'model.c'), '-o', str(work / 'model.o')]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
