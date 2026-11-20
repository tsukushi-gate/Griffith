"""E3: 分類デモシステム(ROM+FSM+argmax+BCD+7セグ)の検証。"""

import numpy as np
import pytest
from conftest import build_groupsum_model, requires_iverilog

from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from mdl_fpga.check import run_system_check
from mdl_fpga.hdlgen.system import (
    argmax_tree_self_test,
    eval_tree,
    generate_system,
    to_bcd,
    tree_order,
    write_system_vectors,
)


class TestArgmaxTree:
    def test_matches_numpy_argmax(self):
        # k=10(MNIST)と k=2(フィクスチャ)、同点多発ケース込み
        argmax_tree_self_test(k=10, max_score=400)
        argmax_tree_self_test(k=2, max_score=2)

    def test_tie_break_smallest_index(self):
        assert eval_tree([5, 5, 5]) == 0
        assert eval_tree([0, 7, 7]) == 1
        assert eval_tree([1, 2, 3, 3]) == 2

    def test_tree_topology_k10(self):
        # 社内先行実験 gen_core.py と同型(ペア縮約+奇数持ち上げ、9ノード)
        nodes = tree_order(10)
        assert len(nodes) == 9
        assert nodes[0] == (0, ('s', 0), ('s', 1))
        assert nodes[-1][1:] == (('n', 7), ('n', 4))  # 根 = (n7, n4)


def test_to_bcd():
    assert to_bcd(938, 4) == 0x0938
    assert to_bcd(0, 4) == 0x0000
    assert to_bcd(1000, 4) == 0x1000
    assert to_bcd(10, 2) == 0x10


def _fixture_dataset(n=12, seed=3):
    """gs_demo(4bit入力, k=2)用のミニデータセット。ラベルは半分故意に誤り。"""
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 2, size=(n, 4), dtype=np.uint8)
    graph = ir_from_dict(build_groupsum_model())
    preds = np.argmax(simulate(graph, images.astype(np.int64)), axis=1)
    labels = preds.copy()
    labels[::3] = 1 - labels[::3]   # 1/3 を不正解化(カウント検証を非自明に)
    return graph, images, labels


def test_generate_system_structure():
    graph, images, labels = _fixture_dataset()
    gen = generate_system(graph, n_images=len(images))
    assert gen.k == 2 and gen.score_bits == 2 and gen.latency == 2
    assert gen.rom_word_bits == 4 + 1   # 画像4bit+ラベル1bit
    v = gen.top_verilog
    assert 'mdl_core_pipe u_core' in v
    assert 'sys_rom u_rom' in v
    assert 'seg7 u_h0' in v and 'seg7 u_h5' in v
    assert 'S_THROTTLE' in v and 'restart' in v
    assert '$readmemh("rom.hex", mem);' in gen.rom_verilog
    assert 'ramstyle = "M10K"' in gen.rom_verilog
    # システムメタ(ビューア用)
    meta = gen.system_meta
    assert meta['allow_set_x'] is False
    ids = [n['id'] for n in meta['nodes']]
    assert {'img_idx', 'label', 'pred', 'correct', 'fsm'} <= set(ids)


def test_write_system_vectors_roundtrip(tmp_path):
    graph, images, labels = _fixture_dataset()
    golden = write_system_vectors(graph, images, labels, tmp_path,
                                  label_bits=1)
    assert golden['n_images'] == len(images)
    # rom.hex の word からラベルと画像を復元して一致確認
    lines = (tmp_path / 'rom.hex').read_text().split()
    assert len(lines) == len(images)
    for i, line in enumerate(lines):
        word = int(line, 16)
        assert (word >> 4) & 1 == labels[i]
        bits = [(word >> j) & 1 for j in range(4)]
        assert bits == images[i].tolist()
    # golden の正解数 = simulate と一致(1/3 不正解化を反映)
    preds = np.argmax(simulate(graph, images.astype(np.int64)), axis=1)
    assert golden['correct'] == int((preds == labels).sum())
    assert golden['correct'] < len(images)   # 非自明(全問正解ではない)


@requires_iverilog
def test_small_system_rtl_allpass(tmp_path):
    """小型システムの2パス自己検証(高速→リスタート→低速throttle)。"""
    graph, images, labels = _fixture_dataset(n=12)
    res = run_system_check(graph, images, labels, tmp_path / 'run')
    assert res.passed, res.render() + '\n' + res.stdout


@requires_iverilog
def test_small_system_decade_boundary(tmp_path):
    """正解数が 9→10 を跨ぐケース(BCD 桁上げの実証)。"""
    rng = np.random.default_rng(7)
    graph = ir_from_dict(build_groupsum_model())
    images = rng.integers(0, 2, size=(15, 4), dtype=np.uint8)
    preds = np.argmax(simulate(graph, images.astype(np.int64)), axis=1)
    labels = preds.copy()          # 全問正解 → 15 = 0x15(BCD) で桁上げ発生
    res = run_system_check(graph, images, labels, tmp_path / 'run')
    assert res.passed, res.render() + '\n' + res.stdout


from pathlib import Path  # noqa: E402

MNIST_IR = Path(__file__).parent.parent / 'examples' / 'mnist' / 'mnist24k.ir.json'
MNIST_NPZ = MNIST_IR.parent / 'mnist_test_1000.npz'


@requires_iverilog
@pytest.mark.slow
@pytest.mark.skipif(not (MNIST_IR.exists() and MNIST_NPZ.exists()),
                    reason='MNIST 資産未生成(examples/mnist 参照)')
def test_mnist_system_smoke(tmp_path):
    """MNIST 24K ゲート実モデル×3枚のシステムスモーク(vvp 数十秒)。"""
    from mdl_fpga.ir_io import load_graph
    graph = load_graph(MNIST_IR)
    data = np.load(MNIST_NPZ)
    images, labels = data['images_bits'][:3], data['labels'][:3]
    res = run_system_check(graph, images, labels, tmp_path / 'run')
    assert res.passed, res.render() + '\n' + res.stdout
