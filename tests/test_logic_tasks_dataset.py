"""train.py の論理関数タスク(xor/xor3/maj3/rnd4/parity3n)データセットのテスト。

- DATASET_INFO のエントリ形状が既存(mnist等)と同じキー構造を持つこと
- load_dataset() が返す train/test ローダーが全2^nパターンの正しい真理値表を含むこと
  (xor3=3入力パリティ、maj3=多数決、rnd4=distill_undercap_3input.TARGETSと同一の固定真理値表、
  parity3n=関連3ビットparity+ノイズ3ビットを埋め込んだ配線学習検証用タスク)
- build_model() が論理タスクの入力次元で正しくLUTネットワークを構築できること
"""

import argparse
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from distill_undercap_3input import TARGETS as UNDERCAP_TARGETS
from train import (
    DATASET_INFO,
    LOGIC_TARGETS,
    LOGIC_TASKS,
    PARITY3N_N_NOISE,
    PARITY3N_N_RELEVANT,
    build_model,
    load_dataset,
    logic_task_dataset,
)

from griffith_logic import LutLayer


def _make_args(dataset, **overrides):
    defaults = dict(
        dataset=dataset,
        n_inputs=2,
        num_neurons=16,
        num_layers=2,
        tau=1.0,
        lr=0.01,
        batch_size=4,
        connections='random',
        parameterization='softmax',
        tau_warp=1.0,
        no_gumbel=False,
        grad_factor=1.0,
        device='cpu',
    )
    defaults.update(overrides)
    ns = argparse.Namespace(**defaults)
    # build_model()はargs.n_inputsを読むが、argparseの--n-inputsはdest='n_inputs'
    return ns


# ============================================================
# DATASET_INFO 形状
# ============================================================

class TestDatasetInfoShape:
    @pytest.mark.parametrize('name', ['xor', 'xor3', 'maj3', 'rnd4', 'parity3n'])
    def test_entry_has_same_keys_as_existing_entries(self, name):
        logic_keys = set(DATASET_INFO[name].keys())
        vision_keys = set(DATASET_INFO['mnist'].keys())
        assert vision_keys <= logic_keys, (
            f'{name}のエントリは既存(mnist)のキーを全て含む必要があります: '
            f'不足={vision_keys - logic_keys}'
        )
        assert 'in_dim' in DATASET_INFO[name]
        assert 'num_classes' in DATASET_INFO[name]

    def test_two_class_classification(self):
        # 全論理タスクは2クラス分類として扱う(GroupSum(2)で層幅を割り切れる必要がある)
        for name in ['xor', 'xor3', 'maj3', 'rnd4', 'parity3n']:
            assert DATASET_INFO[name]['num_classes'] == 2

    def test_in_dim_matches_input_bit_count(self):
        assert DATASET_INFO['xor']['in_dim'] == 2
        assert DATASET_INFO['xor3']['in_dim'] == 3
        assert DATASET_INFO['maj3']['in_dim'] == 3
        assert DATASET_INFO['rnd4']['in_dim'] == 4
        assert DATASET_INFO['parity3n']['in_dim'] == PARITY3N_N_RELEVANT + PARITY3N_N_NOISE

    def test_logic_tasks_set_matches_dataset_info(self):
        assert LOGIC_TASKS == {'xor', 'xor3', 'maj3', 'rnd4', 'parity3n'}
        for name in LOGIC_TASKS:
            assert name in DATASET_INFO


# ============================================================
# logic_task_dataset(): 真理値表の正しさ
# ============================================================

class TestLogicTaskDataset:
    def test_xor_truth_table(self):
        ds = logic_task_dataset('xor')
        x, y = ds.tensors
        assert x.shape == (4, 2)
        assert y.shape == (4,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        for xi, yi in zip(x.tolist(), y.tolist()):
            a, b = int(xi[0]), int(xi[1])
            assert yi == (a ^ b)

    def test_xor3_is_3input_parity(self):
        ds = logic_task_dataset('xor3')
        x, y = ds.tensors
        assert x.shape == (8, 3)
        for xi, yi in zip(x.tolist(), y.tolist()):
            bits = [int(v) for v in xi]
            assert yi == (bits[0] ^ bits[1] ^ bits[2])

    def test_maj3_is_majority_vote(self):
        ds = logic_task_dataset('maj3')
        x, y = ds.tensors
        assert x.shape == (8, 3)
        for xi, yi in zip(x.tolist(), y.tolist()):
            bits = [int(v) for v in xi]
            assert yi == int(sum(bits) >= 2)

    def test_rnd4_matches_fixed_truth_table(self):
        ds = logic_task_dataset('rnd4')
        x, y = ds.tensors
        assert x.shape == (16, 4)
        nin, fn = UNDERCAP_TARGETS['rnd4']
        assert nin == 4
        for xi, yi in zip(x.tolist(), y.tolist()):
            bits = tuple(int(v) for v in xi)
            assert yi == fn(bits)

    def test_all_2n_patterns_covered_exactly_once(self):
        for name, n in [('xor', 2), ('xor3', 3), ('maj3', 3), ('rnd4', 4)]:
            ds = logic_task_dataset(name)
            x, _ = ds.tensors
            rows = {tuple(int(v) for v in row) for row in x.tolist()}
            expected = set(itertools.product([0, 1], repeat=n))
            assert rows == expected, f'{name}: 全{2**n}パターンが揃っていません'

    def test_repeat_duplicates_all_patterns(self):
        ds = logic_task_dataset('xor3', repeat=3)
        x, y = ds.tensors
        assert x.shape == (24, 3)
        assert y.shape == (24,)

    def test_targets_consistent_with_distill_undercap_3input(self):
        # xor3/maj3/rnd4はdistill_undercap_3input.TARGETSと同一定義を流用している
        for name in ['xor3', 'maj3', 'rnd4']:
            assert LOGIC_TARGETS[name][0] == UNDERCAP_TARGETS[name][0]
            nin = LOGIC_TARGETS[name][0]
            for bits in itertools.product([0, 1], repeat=nin):
                assert LOGIC_TARGETS[name][1](bits) == UNDERCAP_TARGETS[name][1](bits)


# ============================================================
# parity3n: 配線学習の系統検証タスク
# ============================================================

class TestParity3nDataset:
    """関連3ビットparity(=xor3と同一ラベル関数)をノイズビットに埋め込んだタスク。

    出典: tests/test_learnable_wiring.py::TestLearnableWiringConvergenceSmoke
    (N_RELEVANT=3, N_NOISE=9)。train.pyのparity3nは既存logic_task_datasetの
    「全2^n列挙」方式に合わせるためノイズ数を3に固定(PARITY3N_N_NOISE)している。
    """

    def test_constants(self):
        assert PARITY3N_N_RELEVANT == 3
        assert PARITY3N_N_NOISE >= 1

    def test_in_dataset_info_and_logic_tasks(self):
        assert 'parity3n' in LOGIC_TASKS
        assert 'parity3n' in DATASET_INFO
        n = PARITY3N_N_RELEVANT + PARITY3N_N_NOISE
        assert DATASET_INFO['parity3n']['in_dim'] == n
        assert DATASET_INFO['parity3n']['num_classes'] == 2
        assert DATASET_INFO['parity3n']['logic_n_inputs'] == n

    def test_label_depends_only_on_relevant_bits(self):
        ds = logic_task_dataset('parity3n')
        x, y = ds.tensors
        n = PARITY3N_N_RELEVANT + PARITY3N_N_NOISE
        assert x.shape == (2 ** n, n)
        for xi, yi in zip(x.tolist(), y.tolist()):
            bits = [int(v) for v in xi]
            relevant = bits[:PARITY3N_N_RELEVANT]
            assert yi == (sum(relevant) % 2)

    def test_matches_xor3_on_relevant_bits(self):
        # 関連3ビットのparityはxor3と同一関数(sum%2 == a^b^c)
        _, xor3_fn = LOGIC_TARGETS['xor3']
        _, parity3n_fn = LOGIC_TARGETS['parity3n']
        for bits in itertools.product([0, 1], repeat=PARITY3N_N_RELEVANT):
            noise_padded = bits + (0,) * PARITY3N_N_NOISE
            assert parity3n_fn(noise_padded) == xor3_fn(bits)

    def test_all_2n_patterns_covered_exactly_once(self):
        ds = logic_task_dataset('parity3n')
        x, _ = ds.tensors
        n = PARITY3N_N_RELEVANT + PARITY3N_N_NOISE
        rows = {tuple(int(v) for v in row) for row in x.tolist()}
        expected = set(itertools.product([0, 1], repeat=n))
        assert rows == expected


# ============================================================
# load_dataset(): DataLoader統合
# ============================================================

class TestLoadDatasetLogicTasks:
    @pytest.mark.parametrize('name,n', [
        ('xor', 2), ('xor3', 3), ('maj3', 3), ('rnd4', 4),
        ('parity3n', PARITY3N_N_RELEVANT + PARITY3N_N_NOISE),
    ])
    def test_train_test_loaders_cover_full_truth_table(self, name, n):
        args = _make_args(name, batch_size=4)
        train_loader, test_loader = load_dataset(args)

        for loader in (train_loader, test_loader):
            all_x, all_y = [], []
            for x, y in loader:
                all_x.append(x)
                all_y.append(y)
            x = torch.cat(all_x)
            y = torch.cat(all_y)
            rows = {tuple(int(v) for v in row) for row in x.tolist()}
            expected = set(itertools.product([0, 1], repeat=n))
            assert rows == expected

            n_inputs, fn = LOGIC_TARGETS[name]
            for xi, yi in zip(x.tolist(), y.tolist()):
                bits = tuple(int(v) for v in xi)
                assert yi == fn(bits)

    def test_loaders_nonempty_even_with_large_batch_size(self):
        # batch_size(default 128) > 2^n(xor=4)でもdrop_last=Trueで空にならない
        args = _make_args('xor', batch_size=128)
        train_loader, test_loader = load_dataset(args)
        assert len(train_loader) >= 1
        assert len(test_loader) >= 1


# ============================================================
# build_model(): 論理タスクの入力次元でLUTネットワーク構築
# ============================================================

class TestBuildModelLogicTasks:
    @pytest.mark.parametrize('name,n', [
        ('xor', 2), ('xor3', 3), ('maj3', 3), ('rnd4', 4),
        ('parity3n', PARITY3N_N_RELEVANT + PARITY3N_N_NOISE),
    ])
    def test_builds_and_forward_runs(self, name, n):
        args = _make_args(name, num_neurons=8, num_layers=2, n_inputs=2)
        model = build_model(args)

        lut_layers = [m for m in model.modules() if isinstance(m, LutLayer)]
        assert len(lut_layers) == 2
        assert lut_layers[0].in_dim == DATASET_INFO[name]['in_dim']

        x = torch.rand(5, n).round()
        out = model(x)
        assert out.shape == (5, DATASET_INFO[name]['num_classes'])

    def test_hard_eval_forward_is_binary_stable(self):
        # 推論モード相当(離散化入力)でforwardが例外なく通る
        args = _make_args('xor3', num_neurons=8, num_layers=1, n_inputs=2)
        model = build_model(args)
        model.eval()
        x, _ = logic_task_dataset('xor3').tensors
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 2)
        assert torch.isfinite(out).all()
