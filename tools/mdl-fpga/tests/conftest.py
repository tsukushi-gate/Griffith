"""mdl-fpga ツールのテスト設定。

自己完結ディレクトリのため、リポジトリ本体の tests/ とは独立に
`uv run pytest tools/mdl-fpga/tests -q` で実行する。
iverilog が必要なテストは discover() で見つからなければ skip する。
"""

import os
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR / 'runtime'))   # mdl_runtime(配布ランタイム)

from mdl_fpga.toolchain import discover  # noqa: E402

_tc = discover()

requires_iverilog = pytest.mark.skipif(
    _tc.iverilog is None or _tc.vvp is None,
    reason='iverilog/vvp が見つからない(doctor 参照)',
)

# 実機(ボード接続)が必要なテスト。既定 skip、環境変数で明示オプトイン
requires_hw = pytest.mark.skipif(
    not os.environ.get('MDL_FPGA_HW_TESTS'),
    reason='実機テストは MDL_FPGA_HW_TESTS=1 で明示的に有効化',
)


def build_parity4() -> dict:
    """4bit パリティ(XOR 木)の IR dict。examples/parity4.ir.json と同一構成。"""
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': 'parity4'},
        'inputs': [{'id': 'x', 'width': 4}],
        'outputs': [{'id': 'y', 'width': 1}],
        'nodes': [
            {'id': 'xor_l1', 'type': 'lut_layer', 'params': {
                'in_width': 4, 'out_width': 2, 'n_inputs': 2,
                'indices': [[0, 2], [1, 3]], 'truth_tables': ['0x6', '0x6']}},
            {'id': 'xor_l2', 'type': 'lut_layer', 'params': {
                'in_width': 2, 'out_width': 1, 'n_inputs': 2,
                'indices': [[0], [1]], 'truth_tables': ['0x6']}},
        ],
        'edges': [
            {'src': 'x', 'dst': 'xor_l1'},
            {'src': 'xor_l1', 'dst': 'xor_l2'},
            {'src': 'xor_l2', 'dst': 'y'},
        ],
    }


def build_groupsum_model() -> dict:
    """lut_layer(4→4) + group_sum(k=2) の IR dict(スコアパック検証用)。"""
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': 'gs_demo'},
        'inputs': [{'id': 'x', 'width': 4}],
        'outputs': [{'id': 'y', 'width': 2, 'semantics': 'class_scores'}],
        'nodes': [
            {'id': 'gates', 'type': 'lut_layer', 'params': {
                'in_width': 4, 'out_width': 4, 'n_inputs': 2,
                'indices': [[0, 1, 2, 3], [1, 2, 3, 0]],
                'truth_tables': ['0x8', '0xe', '0x6', '0x9']}},
            {'id': 'gs', 'type': 'group_sum', 'params': {
                'k': 2, 'in_width': 4, 'tau': 1.0}},
        ],
        'edges': [
            {'src': 'x', 'dst': 'gates'},
            {'src': 'gates', 'dst': 'gs'},
            {'src': 'gs', 'dst': 'y'},
        ],
    }


@pytest.fixture
def parity4_path(tmp_path):
    import json
    p = tmp_path / 'parity4.ir.json'
    p.write_text(json.dumps(build_parity4()), encoding='utf-8')
    return p


@pytest.fixture
def groupsum_path(tmp_path):
    import json
    p = tmp_path / 'gs_demo.ir.json'
    p.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    return p


@pytest.fixture
def toolchain():
    return _tc


# ---------------------------------------------------------------- overlay


def build_rot_ffn(w: int, depth: int, in_bits: int, classes: int,
                  s_a, s_b, kinds=None, seed: int = 0) -> dict:
    """計算型配線(rot / 引き伸ばし rot)の FFN の IR dict。

    層 i の配線は kinds[i] に従い閉形式で作る:
        rot            idx[g] = (g + s) mod W_in
        stretched_rot  idx[g] = ((2*g*W_in + W) // (2*W) + s) mod W_in
    真理値表は seed 固定の乱数(オーバーレイの検証はゲート値に依存しない)。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    kinds = list(kinds) if kinds else ['rot'] * depth
    nodes, edges = [], []
    prev = 'x'
    for i in range(depth):
        w_in = in_bits if i == 0 else w
        g = np.arange(w, dtype=np.int64)
        if kinds[i] == 'rot':
            base = g % w_in
        else:
            base = (2 * g * w_in + w) // (2 * w) % w_in
        a = (base + s_a[i]) % w_in
        b = (base + s_b[i]) % w_in
        tts = rng.integers(0, 16, size=w)
        nodes.append({'id': f'l{i}', 'type': 'lut_layer', 'params': {
            'in_width': int(w_in), 'out_width': int(w), 'n_inputs': 2,
            'indices': [a.tolist(), b.tolist()],
            'truth_tables': [f'0x{int(t):x}' for t in tts]}})
        edges.append({'src': prev, 'dst': f'l{i}'})
        prev = f'l{i}'
    nodes.append({'id': 'gs', 'type': 'group_sum',
                  'params': {'k': classes, 'in_width': w, 'tau': 1.0}})
    edges.append({'src': prev, 'dst': 'gs'})
    edges.append({'src': 'gs', 'dst': 'y'})
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': f'rot_w{w}_d{depth}'},
        'inputs': [{'id': 'x', 'width': int(in_bits)}],
        'outputs': [{'id': 'y', 'width': int(classes),
                     'semantics': 'class_scores'}],
        'nodes': nodes, 'edges': edges,
    }


def build_rot_tiny(seed: int = 0) -> dict:
    """全層 rot の tiny ネット(W=32, DEPTH=3, IN=24, CLASSES=2)。"""
    return build_rot_ffn(32, 3, 24, 2, [1, 0, 0], [3, 5, 7], seed=seed)


def build_stretched_tiny(seed: int = 0) -> dict:
    """層 0 が引き伸ばし rot の tiny ネット(IN=22 → W=32)。

    IN=22 / W=32 / V=8 では 整数歩幅 5・余り歩幅 32・除数 64 となり、
    余りアキュムレータの桁上げが実際に起きる(rem_step=0 に縮退しない)。
    """
    return build_rot_ffn(32, 3, 22, 2, [1, 0, 0], [3, 5, 7],
                         kinds=['stretched_rot', 'rot', 'rot'], seed=seed)


@pytest.fixture
def rot_tiny_path(tmp_path):
    import json
    p = tmp_path / 'rot_tiny.ir.json'
    p.write_text(json.dumps(build_rot_tiny()), encoding='utf-8')
    return p


@pytest.fixture
def stretched_tiny_path(tmp_path):
    import json
    p = tmp_path / 'stretched_tiny.ir.json'
    p.write_text(json.dumps(build_stretched_tiny()), encoding='utf-8')
    return p


# ------------------------------------------------- overlay(保存型・自由配線)


def build_free_ffn(w: int, depth: int, in_bits: int, classes: int,
                   kind: str = 'random', seed: int = 0) -> dict:
    """**自由配線**(閉形式でない任意の添字)の FFN の IR dict。

    kind='random' は無制約一様乱択(先行実験の P0 相当)、'perm' は入力幅と
    出力幅が同じ層で置換(全単射)。層 0 は幅が違うので常に乱択にする。
    保存型オーバーレイの入力になる。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    nodes, edges = [], []
    prev = 'x'
    for i in range(depth):
        w_in = in_bits if i == 0 else w
        if kind == 'perm' and w_in == w:
            a = rng.permutation(w_in)
            b = rng.permutation(w_in)
        else:
            a = rng.integers(0, w_in, size=w)
            b = rng.integers(0, w_in, size=w)
        tts = rng.integers(0, 16, size=w)
        nodes.append({'id': f'l{i}', 'type': 'lut_layer', 'params': {
            'in_width': int(w_in), 'out_width': int(w), 'n_inputs': 2,
            'indices': [a.tolist(), b.tolist()],
            'truth_tables': [f'0x{int(t):x}' for t in tts]}})
        edges.append({'src': prev, 'dst': f'l{i}'})
        prev = f'l{i}'
    nodes.append({'id': 'gs', 'type': 'group_sum',
                  'params': {'k': classes, 'in_width': w, 'tau': 1.0}})
    edges.append({'src': prev, 'dst': 'gs'})
    edges.append({'src': 'gs', 'dst': 'y'})
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': f'{kind}_w{w}_d{depth}'},
        'inputs': [{'id': 'x', 'width': int(in_bits)}],
        'outputs': [{'id': 'y', 'width': int(classes),
                     'semantics': 'class_scores'}],
        'nodes': nodes, 'edges': edges,
    }


def build_random_tiny(seed: int = 0) -> dict:
    """全層ランダム配線の tiny ネット(W=32, DEPTH=3, IN=24, CLASSES=2)。

    rot の tiny(build_rot_tiny)と同じ形・同じサイクル数で、配線だけが
    自由(= 保存型オーバーレイでしか載らない)。
    """
    return build_free_ffn(32, 3, 24, 2, kind='random', seed=seed)


@pytest.fixture
def random_tiny_path(tmp_path):
    import json
    p = tmp_path / 'random_tiny.ir.json'
    p.write_text(json.dumps(build_random_tiny()), encoding='utf-8')
    return p
