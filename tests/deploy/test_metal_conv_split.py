"""Metalランタイムの「ocごとの命令列分割」新旧一致検証。

`griffith_logic/deploy/metalbackend.py` は既定で conv_engine='split'
(mdl_k_conv_exec_split、並列度=(batch,oc,ワード位置))を使うようになった。
`tests/deploy/test_metal.py` / `tests/deploy/test_verify_metal.py` は
既に(生成時のconv_engineを指定しないので)この新しい既定経路を
`ir/simulate.py` に対して検証している。ここではさらに:

    1. 旧経路(conv_engine='combined')が**まだ動く**ことを固定する
       (「既存 mdl_k_conv_exec は残す」というタスク指示の固定)。
    2. 新旧2つの生成物を**同じMetal実行ファイル無しに直接比較**するのではなく、
       同じ入力に対してそれぞれ実行し、両方が同じ出力(ひいては
       `ir/simulate.py`)に一致することを見る(新旧一致検証)。

macOS + Apple GPU + clang++ が無い環境では自動skip。
"""

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.metalverify import (
    build_metal_model as _build_metal_model,
)
from griffith_logic.deploy.metalverify import (
    has_metal,
    run_metal_model,
)
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

pytestmark = pytest.mark.skipif(
    not has_metal(), reason='macOS + Metal のある環境でのみ実行')


def _run_bits(bundle: dict, work_dir, *, conv_engine: str,
             x: np.ndarray) -> np.ndarray:
    graph = ir_from_dict(bundle['ir'])
    n_out = int(graph.outputs[0].width)
    exe = _build_metal_model(bundle, work_dir, conv_engine=conv_engine)
    return run_metal_model(exe, x.tobytes(), n_out)


@pytest.mark.slow
class TestSplitVsCombinedEngine:
    """conv_engine='split'(既定)と'combined'(旧)が同じ結果を返す。"""

    def test_seg_doc_split_and_combined_both_match_simulate(self, seg_doc,
                                                             tmp_path):
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(7)
        x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        want = simulate(graph, x[None])[0].astype(np.int64)

        got_split = _run_bits(bundle, tmp_path / 'split', conv_engine='split',
                              x=x).astype(np.int64)
        got_combined = _run_bits(bundle, tmp_path / 'combined',
                                 conv_engine='combined', x=x).astype(np.int64)

        assert np.array_equal(got_split, want), (
            'split(既定)経路がsimulateと不一致')
        assert np.array_equal(got_combined, want), (
            'combined(旧)経路がsimulateと不一致'
            '(mdl_k_conv_execを壊していないことの固定)')
        assert np.array_equal(got_split, got_combined), (
            '新旧2経路が互いに不一致(simulateとは別々に一致していても'
            'これは起きてはいけない)')

    def test_seg_doc_multi_sample_agreement(self, seg_doc, tmp_path):
        """複数サンプルで安定して一致する(1サンプルだけの偶然一致を排除)。"""
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(99)

        exe_split = _build_metal_model(bundle, tmp_path / 'split',
                                       conv_engine='split')
        exe_combined = _build_metal_model(bundle, tmp_path / 'combined',
                                          conv_engine='combined')
        n_out = int(graph.outputs[0].width)
        for i in range(5):
            x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
            want = simulate(graph, x[None])[0].astype(np.int64)
            got_split = run_metal_model(exe_split, x.tobytes(),
                                        n_out).astype(np.int64)
            got_combined = run_metal_model(exe_combined, x.tobytes(),
                                           n_out).astype(np.int64)
            assert np.array_equal(got_split, want), f'サンプル{i}: split不一致'
            assert np.array_equal(got_combined, want), (
                f'サンプル{i}: combined不一致')
