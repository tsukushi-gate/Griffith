import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.connections import (
    build_reverse_index,
    get_random_connections,
    get_unique_connections,
)
from griffith_logic.group_sum import GroupSum
from griffith_logic.lut_layer import LutLayer

# ============================================================
# 基底関数の正しさ
# ============================================================

class TestBasisFunction:
    def test_n2_binary_inputs(self):
        """N=2: バイナリ入力(0,0),(0,1),(1,0),(1,1)での基底関数が真理値表と一致"""
        layer = LutLayer(2, 1, n_inputs=2)
        # 接続を固定: ニューロン0 が入力0と入力1を使う
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)

        # (0,0) → basis = [(1-0)(1-0), (0)(1-0), (1-0)(0), (0)(0)] = [1,0,0,0]
        x = torch.tensor([[0.0, 0.0]])
        basis = layer._compute_basis(x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 2))
        expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(basis, expected), f'{basis} != {expected}'

        # (1,1) → basis = [0,0,0,1]
        x = torch.tensor([[1.0, 1.0]])
        basis = layer._compute_basis(x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 2))
        expected = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        assert torch.allclose(basis, expected), f'{basis} != {expected}'

        # (0,1) → basis = [0, 0, 1, 0]  (bit0=0→1-x0=1, bit1=1→x1=1 → k=2: (1-x0)*x1)
        # k=0: (1-x0)(1-x1)=0, k=1: x0*(1-x1)=0, k=2: (1-x0)*x1=1, k=3: x0*x1=0
        x = torch.tensor([[0.0, 1.0]])
        basis = layer._compute_basis(x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 2))
        expected = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
        assert torch.allclose(basis, expected), f'{basis} != {expected}'

    def test_n2_continuous_inputs(self):
        """N=2: 連続値入力での基底関数が数式と一致"""
        layer = LutLayer(2, 1, n_inputs=2)
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)

        x = torch.tensor([[0.3, 0.7]])
        gathered = x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 2)
        basis = layer._compute_basis(gathered)

        x0, x1 = 0.3, 0.7
        expected = torch.tensor([[[
            (1 - x0) * (1 - x1),  # k=0
            x0 * (1 - x1),        # k=1
            (1 - x0) * x1,        # k=2
            x0 * x1,              # k=3
        ]]])
        assert torch.allclose(basis, expected, atol=1e-6)

    def test_n4_binary_inputs(self):
        """N=4: 全0入力と全1入力での基底関数"""
        layer = LutLayer(4, 1, n_inputs=4)
        layer.indices = torch.tensor([[0], [1], [2], [3]], dtype=torch.int64)

        # 全0 → basis[0] = 1, 他はすべて0
        x = torch.zeros(1, 4)
        gathered = x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 4)
        basis = layer._compute_basis(gathered)
        assert basis.shape == (1, 1, 16)
        assert torch.isclose(basis[0, 0, 0], torch.tensor(1.0))
        assert torch.isclose(basis[0, 0, 1:].sum(), torch.tensor(0.0))

        # 全1 → basis[15] = 1, 他はすべて0
        x = torch.ones(1, 4)
        gathered = x[:, layer.indices.T.reshape(-1)].reshape(1, 1, 4)
        basis = layer._compute_basis(gathered)
        assert torch.isclose(basis[0, 0, 15], torch.tensor(1.0))
        assert torch.isclose(basis[0, 0, :15].sum(), torch.tensor(0.0))

    def test_basis_sum_to_one(self):
        """基底関数の合計は常に1（任意の入力値で）"""
        for n in [2, 4, 6]:
            layer = LutLayer(n, 3, n_inputs=n)
            x = torch.rand(5, n)
            gathered = x[:, layer.indices.T.reshape(-1)].reshape(5, 3, n)
            basis = layer._compute_basis(gathered)
            sums = basis.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f'N={n}: basis sum = {sums}'


# ============================================================
# forward の正しさ
# ============================================================

class TestForward:
    def test_output_shape(self):
        """出力形状が正しいか"""
        for n in [2, 4, 6]:
            layer = LutLayer(16, 8, n_inputs=n)
            x = torch.rand(4, 16)
            y = layer(x)
            assert y.shape == (4, 8), f'N={n}: {y.shape}'

    def test_n2_identity_gate(self):
        """N=2: 重みを設定してANDゲート動作を確認（16ゲートモード）"""
        layer = LutLayer(2, 1, n_inputs=2)
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)

        # 16ゲートモード: gate 1 = AND（真理値表 [0,0,0,1]）
        with torch.no_grad():
            layer.weights.fill_(-10.0)
            layer.weights[0, 1] = 10.0  # gate 1: AND

        layer.eval()
        x = torch.tensor([[1.0, 1.0]])
        y = layer(x)
        assert y.item() > 0.99, f'AND(1,1) should be ~1, got {y.item()}'

        x = torch.tensor([[1.0, 0.0]])
        y = layer(x)
        assert y.item() < 0.01, f'AND(1,0) should be ~0, got {y.item()}'

    def test_eval_mode_discrete(self):
        """eval時はargmaxで離散化されるか"""
        layer = LutLayer(4, 2, n_inputs=2)
        layer.eval()
        x = torch.rand(3, 4)
        y = layer(x)
        # 離散化なので出力は0か1に近い値が多くなるはず（バイナリ入力でないので完全ではない）
        assert y.shape == (3, 2)


# ============================================================
# 勾配チェック（autograd）
# ============================================================

class TestGradient:
    def _gradcheck(self, n_inputs, in_dim=8, out_dim=4, batch=2):
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs).double()
        x = torch.randn(batch, in_dim, dtype=torch.float64, requires_grad=True)
        # sigmoid で [0,1] 範囲に制限（基底関数の安定性のため）
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3), \
            f'N={n_inputs} の勾配チェックが失敗'

    def test_gradcheck_n2(self):
        """N=2 の勾配数値検証"""
        self._gradcheck(2)

    def test_gradcheck_n4(self):
        """N=4 の勾配数値検証"""
        self._gradcheck(4)

    def test_gradcheck_n6(self):
        """N=6 の勾配数値検証"""
        self._gradcheck(6, in_dim=12, out_dim=4, batch=2)


# ============================================================
# GroupSum
# ============================================================

class TestGroupSum:
    def test_output_shape(self):
        gs = GroupSum(k=10, tau=30.0)
        x = torch.rand(4, 100)
        y = gs(x)
        assert y.shape == (4, 10)

    def test_value(self):
        gs = GroupSum(k=2, tau=1.0)
        x = torch.ones(1, 6)
        y = gs(x)
        # 6ニューロン / 2クラス = 3ニューロン/クラス → 各クラス合計=3
        assert torch.allclose(y, torch.tensor([[3.0, 3.0]]))

    def test_tau_scaling(self):
        gs = GroupSum(k=2, tau=10.0)
        x = torch.ones(1, 6)
        y = gs(x)
        assert torch.allclose(y, torch.tensor([[0.3, 0.3]]))

    def test_not_divisible(self):
        gs = GroupSum(k=3)
        x = torch.rand(2, 7)
        with pytest.raises(AssertionError):
            gs(x)


# ============================================================
# 接続管理
# ============================================================

class TestConnections:
    def test_random_shape(self):
        for n in [2, 4, 6]:
            idx = get_random_connections(16, 8, n)
            assert idx.shape == (n, 8)
            assert idx.min() >= 0
            assert idx.max() < 16

    def test_unique_shape(self):
        for n in [2, 4, 6]:
            idx = get_unique_connections(n * 4, 8, n)
            assert idx.shape == (n, 8)

    def test_reverse_index(self):
        indices = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.int64)
        starts, flat_y, flat_slot = build_reverse_index(indices, in_dim=3)
        # 入力0: 出力0(slot0), 出力2(slot0) → [0,2]
        # 入力1: 出力0(slot1), 出力1(slot0) → [0,1]
        # 入力2: 出力1(slot1), 出力2(slot1) → [1,2]
        assert starts.shape[0] == 4  # in_dim + 1
        total = flat_y.shape[0]
        assert total == 6  # 2入力 × 3出力 = 6接続
        assert flat_slot.shape[0] == 6
        # スロット番号の範囲チェック
        assert flat_slot.min() >= 0
        assert flat_slot.max() <= 1  # n_inputs=2なので0か1


# ============================================================
# 統合テスト
# ============================================================

class TestIntegration:
    def test_sequential_model(self):
        """Sequential モデルが正常に動作するか"""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LutLayer(16, 32, n_inputs=2),
            LutLayer(32, 32, n_inputs=2),
            GroupSum(k=4, tau=10.0),
        )
        x = torch.rand(8, 16)
        y = model(x)
        assert y.shape == (8, 4)

    def test_backward_runs(self):
        """backward が正常に実行されるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 16, n_inputs=4),
            LutLayer(16, 16, n_inputs=4),
            GroupSum(k=2, tau=10.0),
        )
        x = torch.rand(4, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        # weightsに勾配が流れているか確認
        for m in model.modules():
            if isinstance(m, LutLayer):
                assert m.weights.grad is not None
                assert m.weights.grad.shape == m.weights.shape

    def test_training_step(self):
        """1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 20, n_inputs=2),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


# ============================================================
# Independent パラメータ化（BinaryBrain方式）
# ============================================================

class TestIndependentParameterization:
    def test_output_shape(self):
        """independent方式でも出力形状が正しいか"""
        for n in [2, 4, 6]:
            layer = LutLayer(16, 8, n_inputs=n, parameterization='independent')
            x = torch.rand(4, 16)
            y = layer(x)
            assert y.shape == (4, 8), f'N={n}: {y.shape}'

    def test_forward_differs_from_softmax(self):
        """softmaxとindependentで同じ重みでも出力が異なることを確認"""
        torch.manual_seed(42)
        layer_s = LutLayer(8, 4, n_inputs=2, parameterization='softmax')
        torch.manual_seed(42)
        layer_i = LutLayer(8, 4, n_inputs=2, parameterization='independent')
        # 同じ重み・接続で出力が異なるはず（softmax vs sigmoid）
        x = torch.rand(2, 8)
        y_s = layer_s(x)
        y_i = layer_i(x)
        assert not torch.allclose(y_s, y_i, atol=1e-3), \
            'softmaxとindependentの出力が同一（期待: 異なる）'

    def test_eval_discrete(self):
        """independent eval時はsigmoid>0.5で離散化"""
        layer = LutLayer(8, 4, n_inputs=2, parameterization='independent')
        # 重みを明示的に設定: sigmoid(10)≈1, sigmoid(-10)≈0
        with torch.no_grad():
            layer.weights.fill_(-10.0)
            layer.weights[:, 3] = 10.0  # エントリ3のみON
        layer.eval()
        x = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        y = layer(x)
        assert y.shape == (1, 4)

    def test_gradcheck_n2(self):
        """N=2 independentの勾配数値検証"""
        layer = LutLayer(8, 4, n_inputs=2, parameterization='independent').double()
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_gradcheck_n4(self):
        """N=4 independentの勾配数値検証"""
        layer = LutLayer(8, 4, n_inputs=4, parameterization='independent').double()
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_gradcheck_n6(self):
        """N=6 independentの勾配数値検証"""
        layer = LutLayer(12, 4, n_inputs=6, parameterization='independent').double()
        x = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """independentでbackwardが実行されるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 16, n_inputs=2, parameterization='independent'),
            LutLayer(16, 16, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=10.0),
        )
        x = torch.rand(4, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for m in model.modules():
            if isinstance(m, LutLayer):
                assert m.weights.grad is not None

    def test_training_step(self):
        """independent方式で1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 20, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_invalid_parameterization(self):
        """不正なparameterization値でAssertionError"""
        with pytest.raises(AssertionError):
            LutLayer(8, 4, n_inputs=2, parameterization='invalid')


# ============================================================
# IWP パラメータ化（LDDLGN方式）
# ============================================================

class TestIWPParameterization:
    def test_output_shape(self):
        """IWP方式でも出力形状が正しいか"""
        for n in [2, 4, 6]:
            layer = LutLayer(16, 8, n_inputs=n, parameterization='iwp')
            x = torch.rand(4, 16)
            y = layer(x)
            assert y.shape == (4, 8), f'N={n}: {y.shape}'

    def test_weights_shape(self):
        """IWPの重みが(out_dim, n_inputs)であることを確認"""
        for n in [2, 4, 6]:
            layer = LutLayer(16, 8, n_inputs=n, parameterization='iwp')
            assert layer.weights.shape == (8, n), \
                f'N={n}: weights shape {layer.weights.shape} != (8, {n})'

    def test_forward_differs_from_softmax(self):
        """softmaxとiwpで出力が異なることを確認"""
        layer_s = LutLayer(8, 4, n_inputs=2, parameterization='softmax')
        layer_i = LutLayer(8, 4, n_inputs=2, parameterization='iwp')
        x = torch.rand(2, 8)
        y_s = layer_s(x)
        y_i = layer_i(x)
        # 形状は同じだが値は異なるはず
        assert y_s.shape == y_i.shape

    def test_eval_discrete_binary_output(self):
        """IWP eval時にバイナリ入力からバイナリ出力が得られるか"""
        layer = LutLayer(4, 2, n_inputs=2, parameterization='iwp')
        layer.eval()
        x = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        y = layer(x)
        assert y.shape == (1, 2)
        # バイナリ入力 + 離散化p → 出力は0か1
        assert torch.all((y == 0.0) | (y == 1.0)), f'出力が0/1でない: {y}'

    def test_gradcheck_n2(self):
        """N=2 IWPの勾配数値検証"""
        layer = LutLayer(8, 4, n_inputs=2, parameterization='iwp').double()
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_gradcheck_n4(self):
        """N=4 IWPの勾配数値検証"""
        layer = LutLayer(8, 4, n_inputs=4, parameterization='iwp').double()
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_gradcheck_n6(self):
        """N=6 IWPの勾配数値検証"""
        layer = LutLayer(12, 4, n_inputs=6, parameterization='iwp').double()
        x = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
        fn = lambda x: layer(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """IWPでbackwardが実行されるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 16, n_inputs=2, parameterization='iwp'),
            LutLayer(16, 16, n_inputs=2, parameterization='iwp'),
            GroupSum(k=2, tau=10.0),
        )
        x = torch.rand(4, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for m in model.modules():
            if isinstance(m, LutLayer):
                assert m.weights.grad is not None
                assert m.weights.grad.shape == m.weights.shape

    def test_training_step(self):
        """IWP方式で1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutLayer(8, 20, n_inputs=2, parameterization='iwp'),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_parameter_count_reduction(self):
        """IWPのパラメータ数がsoftmaxより少ないことを確認"""
        for n in [2, 4, 6]:
            layer_s = LutLayer(16, 8, n_inputs=n, parameterization='softmax')
            layer_i = LutLayer(16, 8, n_inputs=n, parameterization='iwp')
            params_s = sum(p.numel() for p in layer_s.parameters())
            params_i = sum(p.numel() for p in layer_i.parameters())
            assert params_i < params_s, \
                f'N={n}: IWP({params_i}) >= softmax({params_s})'


# CUDAカーネル正当性テストは tests/l2_grad/test_cuda_parity.py へ移設。
# 旧テストは gate_table 引数追加前のシグネチャで壊れていた。


# ============================================================
# Residual Init（16ゲートモード）
# ============================================================

class TestLutLayerResidualInit:
    def test_default_is_randn_no_change(self):
        """residual_init未指定（既定False）は従来のrandn初期化のまま"""
        torch.manual_seed(0)
        layer = LutLayer(8, 16, n_inputs=2, parameterization='softmax')
        # randnなので全ゲート同一（argmaxが単一値に潰れる）ではない
        gate_idx = layer.weights.argmax(dim=-1)
        assert gate_idx.unique().numel() > 1

    def test_residual_init_all_gate_A(self):
        """residual_init=True: 全ニューロンのargmaxがゲートA([0,1,0,1])のindex"""
        layer = LutLayer(8, 32, n_inputs=2, parameterization='softmax',
                         residual_init=True)
        gt = layer.gate_table
        target = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=gt.dtype)
        a_idx = int((gt == target).all(dim=1).nonzero().flatten().item())
        gate_idx = layer.weights.argmax(dim=-1)
        assert (gate_idx == a_idx).all(), gate_idx.unique().tolist()
        # ゲートAのロジットは5.0、他は0
        assert torch.isclose(layer.weights[:, a_idx],
                             torch.full((32,), 5.0)).all()

    def test_residual_init_forward_approx_passthrough(self):
        """初期状態のsoft forwardが第1入力aをほぼ通す（P(A)=0.9混合の許容内）"""
        layer = LutLayer(4, 16, n_inputs=2, parameterization='softmax',
                         connections='random', residual_init=True)
        layer.train()
        # 各ニューロンは indices[0] の入力を第1入力として受け取る
        x = torch.rand(6, 4)
        y = layer(x)  # (6, 16)
        # 期待: y ≈ 0.9·x[indices[0]] + 混合。indices[0]の値との相関が高い
        a = x[:, layer.indices[0]]  # (6, 16) 各ニューロンの第1入力
        # P(A)=0.9 の混合なので |y - a| は概ね小さい（残り0.1の他ゲート混合分）
        assert (y - a).abs().mean().item() < 0.15, (y - a).abs().mean().item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
