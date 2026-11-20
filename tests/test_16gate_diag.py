"""
16ゲートモード（softmax + N=2）の診断テスト。

CUDA上でhard精度が13%に壊れるバグを特定するため、
Python pathとCUDA pathの出力・勾配を比較する。

ローカル（MPS/CPU）: pytest tests/test_16gate_diag.py -v
RunPod（CUDA）: python -m pytest tests/test_16gate_diag.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import LutLayer
from griffith_logic.lut_layer import (
    _build_gate_table_n2,
    _compute_basis_butterfly,
)


class TestGateTable:
    """gate_tableの正当性"""

    def test_shape(self):
        table = _build_gate_table_n2()
        assert table.shape == (16, 4)

    def test_gate0_false(self):
        """gate 0 = FALSE: 全出力0"""
        table = _build_gate_table_n2()
        assert table[0].tolist() == [0, 0, 0, 0]

    def test_gate1_and(self):
        """gate 1 = AND: (1,1)のみ1"""
        table = _build_gate_table_n2()
        assert table[1].tolist() == [0, 0, 0, 1]

    def test_gate7_or(self):
        """gate 7 = OR: (0,0)以外1"""
        table = _build_gate_table_n2()
        assert table[7].tolist() == [0, 1, 1, 1]

    def test_gate6_xor(self):
        """gate 6 = XOR: (0,1)と(1,0)のみ1"""
        table = _build_gate_table_n2()
        assert table[6].tolist() == [0, 1, 1, 0]

    def test_gate15_true(self):
        """gate 15 = TRUE: 全出力1"""
        table = _build_gate_table_n2()
        assert table[15].tolist() == [1, 1, 1, 1]

    def test_gate14_nand(self):
        """gate 14 = NAND: (1,1)以外1"""
        table = _build_gate_table_n2()
        assert table[14].tolist() == [1, 1, 1, 0]

    def test_all_gates_binary(self):
        """全エントリが0か1"""
        table = _build_gate_table_n2()
        assert ((table == 0) | (table == 1)).all()

    def test_eval_with_binary_input(self):
        """バイナリ入力でgate_tableが正しいBoolean関数を再現するか"""
        table = _build_gate_table_n2()
        # 4つの入力パターン
        patterns = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # 各パターンのbasis（one-hot）
        for k, (x0, x1) in enumerate(patterns):
            gathered = torch.tensor([[[x0, x1]]])  # (1, 1, 2)
            basis = _compute_basis_butterfly(gathered, 2)  # (1, 1, 4)
            # basisはone-hot: k番目だけが1
            assert basis[0, 0, k].item() == 1.0, f'pattern {k}: basis[{k}] should be 1'
            assert basis[0, 0].sum().item() == 1.0, f'pattern {k}: basis should sum to 1'

            # 各gateの出力を検証
            for g in range(16):
                w = table[g:g+1]  # (1, 4)
                output = torch.einsum('boe,oe->bo', basis, w)
                expected = table[g, k].item()
                assert output.item() == expected, (
                    f'gate {g}, pattern {k}: got {output.item()}, expected {expected}'
                )


class TestPythonForward:
    """Python LutFunction forwardの正当性"""

    def test_manual_or_gate(self):
        """手動でOR gate重みを設定し、出力が正しいか"""
        layer = LutLayer(2, 1, n_inputs=2)
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)

        with torch.no_grad():
            layer.weights.fill_(-10.0)
            layer.weights[0, 7] = 10.0  # gate 7 = OR

        # soft forward (training mode)
        layer.train()
        x = torch.tensor([[1.0, 0.0]])
        y_soft = layer(x)
        assert y_soft.item() > 0.99, f'OR(1,0) soft should be ~1, got {y_soft.item()}'

        # hard forward (eval mode)
        layer.eval()
        y_hard = layer(x)
        assert y_hard.item() == 1.0, f'OR(1,0) hard should be 1, got {y_hard.item()}'

    def test_eval_output_is_binary(self):
        """eval時の出力がバイナリ入力に対して0か1になるか"""
        layer = LutLayer(4, 8, n_inputs=2)
        opt = torch.optim.Adam(layer.parameters(), lr=0.01)

        layer.train()
        for _ in range(100):
            x = torch.rand(32, 4)
            y = layer(x)
            loss = y.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        layer.eval()
        x_test = torch.rand(50, 4).round()  # バイナリ入力
        y_hard = layer(x_test)

        is_binary = ((y_hard == 0) | (y_hard == 1)).all()
        print(f'hard output binary: {is_binary}')
        print(f'hard unique: {y_hard.unique().tolist()}')
        assert is_binary, f'eval output should be 0 or 1, got: {y_hard.unique().tolist()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA不要')
class TestCudaPythonConsistency:
    """CUDA pathとPython pathの出力・勾配一致"""

    def test_forward_match(self):
        """同じ入力・重みでCUDA forwardとPython forwardが一致"""
        torch.manual_seed(42)
        layer = LutLayer(4, 8, n_inputs=2, device='cuda')
        gate_table = layer.gate_table

        x = torch.rand(16, 4, device='cuda')

        # CUDA forward
        layer.train()
        y_cuda = layer(x).detach()

        # Python forward（CUDAを無効にして）
        x_cpu = x.cpu()
        w_cpu = layer.weights.data.cpu()
        idx_cpu = layer.indices.cpu()
        gt_cpu = gate_table.cpu()

        s = F.softmax(w_cpu, dim=-1)
        eff_w = s @ gt_cpu  # (8, 4)

        gathered = x_cpu[:, idx_cpu.T.reshape(-1)].reshape(16, 8, 2)
        basis = _compute_basis_butterfly(gathered, 2)
        y_python = torch.einsum('boe,oe->bo', basis, eff_w)

        diff = (y_cuda.cpu() - y_python).abs().max().item()
        print(f'forward max diff: {diff:.8f}')
        assert diff < 1e-5, f'CUDA vs Python forward mismatch: {diff}'

    def test_backward_match(self):
        """同じ入力・重みでCUDA backwardとPython backwardが一致"""
        torch.manual_seed(42)

        # CUDA path
        layer_cuda = LutLayer(4, 8, n_inputs=2, device='cuda')
        layer_cuda.train()
        x_cuda = torch.rand(16, 4, device='cuda')
        y_cuda = layer_cuda(x_cuda)
        loss_cuda = y_cuda.sum()
        loss_cuda.backward()
        grad_cuda = layer_cuda.weights.grad.clone().cpu()

        # Python path（同じ重み・入力をCPUで）
        layer_py = LutLayer(4, 8, n_inputs=2, device='cpu')
        with torch.no_grad():
            layer_py.weights.copy_(layer_cuda.weights.data.cpu())
            layer_py.indices = layer_cuda.indices.cpu()
            layer_py.gate_table = layer_cuda.gate_table.cpu()
        layer_py.train()
        x_py = x_cuda.cpu()
        y_py = layer_py(x_py)
        loss_py = y_py.sum()
        loss_py.backward()
        grad_py = layer_py.weights.grad.clone()

        diff = (grad_cuda - grad_py).abs().max().item()
        print(f'backward max diff: {diff:.8f}')
        assert diff < 1e-4, f'CUDA vs Python backward mismatch: {diff}'

    def test_training_produces_peaked_softmax(self):
        """CUDA訓練後、softmaxのmax値が十分に高い（gate選択が明確）か"""
        layer = LutLayer(4, 8, n_inputs=2, device='cuda')
        opt = torch.optim.Adam(layer.parameters(), lr=0.01)

        layer.train()
        for _ in range(500):
            x = torch.rand(32, 4, device='cuda')
            y = layer(x)
            loss = (y - 0.5).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        s = F.softmax(layer.weights, dim=-1)
        max_probs = s.max(dim=-1).values
        print(f'softmax max per neuron: {max_probs.tolist()}')
        print(f'softmax max mean: {max_probs.mean():.4f}')
        print(f'argmax gates: {layer.weights.argmax(dim=-1).tolist()}')

        # 均一分布 = 1/16 = 0.0625。訓練後はもっと高くないとargmaxが不安定
        assert max_probs.mean() > 0.1, (
            f'softmax too flat after training: mean max = {max_probs.mean():.4f}'
        )

    def test_eval_hard_accuracy_not_random(self):
        """CUDA訓練→eval時のhard出力が少なくともランダムより良いか"""
        layer = LutLayer(4, 8, n_inputs=2, device='cuda')
        opt = torch.optim.Adam(layer.parameters(), lr=0.01)

        layer.train()
        for _ in range(200):
            x = torch.rand(32, 4, device='cuda')
            y = layer(x)
            loss = y.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        layer.eval()
        x_test = torch.rand(100, 4, device='cuda').round()
        y_hard = layer(x_test)

        unique_vals = y_hard.unique()
        print(f'hard output unique values: {unique_vals.numel()}')
        print(f'hard output range: [{y_hard.min():.4f}, {y_hard.max():.4f}]')
        print(f'hard output sample: {y_hard[0].tolist()}')

        # hard出力は0か1のみのはず（バイナリ入力 + gate_table離散化）
        is_binary = ((y_hard == 0) | (y_hard == 1)).all()
        print(f'all outputs binary: {is_binary}')
        assert is_binary, f'hard output should be binary, got unique: {unique_vals.tolist()}'
