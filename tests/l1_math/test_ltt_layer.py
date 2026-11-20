"""L1: LTT(線形閾値ゲート)層の定義。

    - y = 1[w·x + b > 0]、eval 出力は {0,1}、素朴ループと一致
    - 真理値表の列挙(truth_tables_from_ltt)が eval forward と全パターンで一致
    - n=2 の閾値関数は XOR / XNOR を表せない(乱択 w, b で一度も出ない)
    - clamp: 訓練時の forward で w, b が [−1, 1] に入る(eval では触らない)
    - 代理勾配: z=0 で 1、clip は窓の外で 0、sigmoid は 4σ(1−σ)
    - 配線: random / unique / Tensor / learnable(argmax が実効配線)
"""

import pytest
import torch

from griffith_logic.core.reference import reference_ltt_forward
from griffith_logic.export.discretize import truth_tables_from_ltt
from griffith_logic.layers.ltt import LTTLayer, ltt_ste_surrogate


def _x(batch, in_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, in_dim), generator=g).float()


def _patterns(n):
    return torch.tensor([[(k >> j) & 1 for j in range(n)] for k in range(2 ** n)],
                        dtype=torch.float32)


class TestForward:
    def test_matches_naive_loop(self):
        torch.manual_seed(0)
        layer = LTTLayer(10, 6, n_inputs=3).eval()
        x = _x(16, 10)
        y = layer(x)
        assert y.shape == (16, 6)
        assert set(y.unique().tolist()) <= {0.0, 1.0}
        w, b = layer.w.detach(), layer.b.detach()
        for i in range(16):
            for j in range(6):
                z = float(b[j])
                for k in range(3):
                    z += float(w[j, k]) * float(x[i, int(layer.indices[k, j])])
                assert float(y[i, j]) == (1.0 if z > 0 else 0.0)

    def test_eval_matches_reference(self):
        torch.manual_seed(1)
        layer = LTTLayer(12, 5, n_inputs=4).eval()
        x = _x(8, 12)
        assert torch.equal(layer(x), reference_ltt_forward(x, layer.w, layer.b, layer.indices))

    def test_soft_forward_is_sigmoid_of_scaled_preactivation(self):
        torch.manual_seed(2)
        layer = LTTLayer(8, 3, n_inputs=2, ste_scale=2.0).eval()
        x = _x(5, 8)
        z = layer.pre_activation(x)
        assert torch.allclose(layer.soft_forward(x), torch.sigmoid(z / 2.0))

    def test_train_forward_values_are_hard_and_equal_eval(self):
        torch.manual_seed(3)
        layer = LTTLayer(8, 4, n_inputs=2)
        x = _x(6, 8)
        y_train = layer.train()(x)
        y_eval = layer.eval()(x)
        assert set(y_train.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(y_train, y_eval)

    def test_width_mismatch_rejected(self):
        layer = LTTLayer(8, 4, n_inputs=2)
        with pytest.raises(AssertionError):
            layer(_x(2, 9))


class TestTruthTables:
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 6])
    def test_enumeration_matches_eval(self, n):
        """ゲートが入力 0..n−1 をこの順に読む配線で全 2^n パターンを流すと、
        出力 = 真理値表の bit k(k = パターン番号。bit j = 入力スロット j)"""
        torch.manual_seed(n)
        idx = torch.arange(n).unsqueeze(1).expand(n, 5).contiguous()
        layer = LTTLayer(n, 5, n_inputs=n, connections=idx).eval()
        y = layer(_patterns(n))                      # (2^n, 5)
        tts = truth_tables_from_ltt(layer)
        assert tts.shape == (5,)
        for u in range(5):
            assert 0 <= int(tts[u]) < (1 << (1 << n))
            for k in range(2 ** n):
                assert ((int(tts[u]) >> k) & 1) == int(y[k, u])

    def test_n2_threshold_functions_never_xor(self):
        torch.manual_seed(1)
        idx = torch.tensor([[0] * 2000, [1] * 2000])
        layer = LTTLayer(2, 2000, n_inputs=2, connections=idx)
        tts = {int(t) for t in truth_tables_from_ltt(layer)}
        assert 0b0110 not in tts
        assert 0b1001 not in tts
        assert len(tts) >= 10                         # 閾値関数 14 種の大半が出る


class TestClampAndSte:
    def test_clamp_only_in_training_forward(self):
        layer = LTTLayer(8, 4, n_inputs=2)
        with torch.no_grad():
            layer.w.fill_(5.0)
            layer.b.fill_(-3.0)
        layer.eval()(_x(4, 8))
        assert float(layer.w.detach().max()) == 5.0   # eval では clamp しない
        layer.train()(_x(4, 8))
        assert float(layer.w.detach().max()) == 1.0
        assert float(layer.b.detach().min()) == -1.0

    def test_clamp_can_be_disabled(self):
        layer = LTTLayer(8, 4, n_inputs=2, clamp=False)
        with torch.no_grad():
            layer.w.fill_(5.0)
        layer.train()(_x(4, 8))
        assert float(layer.w.detach().max()) == 5.0

    @pytest.mark.parametrize('mode', LTTLayer.STE_MODES)
    def test_surrogate_is_one_at_zero(self, mode):
        z = torch.zeros(3, dtype=torch.float64)
        assert torch.allclose(ltt_ste_surrogate(z, mode, 1.0), torch.ones(3, dtype=torch.float64))

    def test_clip_window(self):
        z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert ltt_ste_surrogate(z, 'clip', 1.0).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
        assert ltt_ste_surrogate(z, 'clip', 3.0).tolist() == [1.0] * 5
        assert ltt_ste_surrogate(z, 'identity', 1.0).tolist() == [1.0] * 5

    def test_sigmoid_formula(self):
        z = torch.linspace(-3, 3, 7, dtype=torch.float64)
        s = torch.sigmoid(z / 2.0)
        assert torch.allclose(ltt_ste_surrogate(z, 'sigmoid', 2.0), 4 * s * (1 - s))

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            LTTLayer(8, 4, n_inputs=2, ste='tanh')
        with pytest.raises(ValueError):
            ltt_ste_surrogate(torch.zeros(2), 'tanh', 1.0)


class TestWiring:
    def test_random_and_unique(self):
        torch.manual_seed(4)
        rnd = LTTLayer(8, 6, n_inputs=2, connections='random')
        uni = LTTLayer(8, 6, n_inputs=2, connections='unique')
        for layer in (rnd, uni):
            assert layer.indices.shape == (2, 6)
            assert int(layer.indices.min()) >= 0
            assert int(layer.indices.max()) < 8
        assert torch.equal(rnd.effective_indices(), rnd.indices)
        assert uni.mapping_mode == 'unique'

    def test_explicit_tensor_validation(self):
        with pytest.raises(ValueError):
            LTTLayer(8, 4, n_inputs=2, connections=torch.zeros(3, 4, dtype=torch.int64))
        with pytest.raises(ValueError):
            LTTLayer(8, 4, n_inputs=2, connections=torch.full((2, 4), 8, dtype=torch.int64))
        with pytest.raises(ValueError):
            LTTLayer(8, 4, n_inputs=2, connections='perm')

    def test_learnable_mapping(self):
        torch.manual_seed(5)
        layer = LTTLayer(12, 5, n_inputs=3, connections='learnable')
        assert layer.mapping_mode == 'learnable'
        idx = layer.effective_indices()
        assert idx.shape == (3, 5)
        assert int(idx.max()) < 12
        x = _x(4, 12)
        y = layer.eval()(x)
        assert torch.equal(y, reference_ltt_forward(x, layer.w, layer.b, idx))

    def test_extra_repr(self):
        layer = LTTLayer(8, 4, n_inputs=6)
        assert 'n_inputs=6' in layer.extra_repr()
        assert 'ste=clip/s=1.0' in layer.extra_repr()
        assert 'connections=random' in layer.extra_repr()
