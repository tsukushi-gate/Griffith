"""C1直接転写(convert/transcribe.py)の数理性質テスト。

参照意味論(transcribe.pyモジュールdocstring):
    隠れ層: y_i = 1[Σ_j W_ij x_j + b_i > 0]
    最終層: score_c = Σ_j q_cj x_j + β_c(共有スケール整数量子化)、argmax保存

検証はすべてビット/整数の厳密比較(許容誤差なし)。
"""

import numpy as np
import pytest
import torch
from torch import nn

from griffith_logic.convert import (
    TranscribeSpec,
    enumerate_truth_table,
    isf_truth_tables,
    sparsify_fanin,
    threshold_truth_tables,
    transcribe_mlp,
)
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _all_binary_inputs(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


def _reference_forward(linears: list[nn.Linear], x: np.ndarray) -> np.ndarray:
    """参照意味論のnumpy実装(隠れ=発火判定、最終=実数スコア)"""
    h = x.astype(np.float64)
    for lin in linears[:-1]:
        w = lin.weight.detach().numpy().astype(np.float64)
        b = lin.bias.detach().numpy().astype(np.float64)
        h = ((h @ w.T + b) > 0).astype(np.float64)
    lin = linears[-1]
    w = lin.weight.detach().numpy().astype(np.float64)
    b = lin.bias.detach().numpy().astype(np.float64)
    return h @ w.T + b


def _mlp(*layers: nn.Module) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(*layers)


def _set_linear(lin: nn.Linear, w, b) -> nn.Linear:
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(w, dtype=lin.weight.dtype))
        lin.bias.copy_(torch.tensor(b, dtype=lin.bias.dtype))
    return lin


class TestEnumerate:
    def test_xor_by_callable(self):
        tt = enumerate_truth_table(lambda x: x[0] ^ x[1], 2)
        assert tt == 0b0110  # addr1(x0=1),addr2(x1=1)で1

    def test_threshold_matches_callable(self):
        """閾値ゲートの全列挙が汎用列挙と一致する"""
        torch.manual_seed(1)
        w = torch.randn(5, 4, dtype=torch.float64)
        b = torch.randn(5, dtype=torch.float64)
        tts = threshold_truth_tables(w, b)
        for o in range(5):
            expected = enumerate_truth_table(
                lambda x, o=o: int(
                    (w[o] @ torch.tensor(x, dtype=torch.float64) + b[o]) > 0
                ),
                4,
            )
            assert tts[o] == expected


class TestSparsify:
    def test_full_k_is_identity(self):
        """k=in_widthなら全接続が残り、bias_shift=0"""
        torch.manual_seed(2)
        w = torch.randn(3, 6, dtype=torch.float64)
        indices, w_sel, shift = sparsify_fanin(w, 6)
        assert indices.shape == (6, 3)
        assert torch.equal(
            w_sel.sort(dim=1).values, w.sort(dim=1).values
        )
        assert torch.all(shift == 0)

    def test_topk_selection(self):
        w = torch.tensor([[0.1, -5.0, 0.2, 3.0]], dtype=torch.float64)
        indices, w_sel, _ = sparsify_fanin(w, 2)
        assert set(indices[:, 0].tolist()) == {1, 3}

    def test_bias_recalibration(self):
        """落とした接続の寄与 = Σ w_j·mean(x_j) がバイアスへ移る"""
        w = torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float64)
        calib = np.array([[1, 1, 1], [1, 1, 0]])  # mean = [1, 1, 0.5]
        _, _, shift = sparsify_fanin(w, 2, calib_inputs=calib)
        # 上位2 = {0, 1}、落ちるのはj=2(w=0.5, mean=0.5)
        assert shift.item() == pytest.approx(0.25)


class TestExactTranscription:
    """fan-in制約内のMLPは厳密転写できる(サーベイ§3の無損失変換クラス)"""

    def test_xor_mlp_bitexact(self):
        """XORを計算する解析重みMLP → 全入力でsimulate一致"""
        h = _set_linear(nn.Linear(2, 2), [[1.0, 1.0], [-1.0, -1.0]],
                        [-0.5, 1.5])   # OR, NAND
        o = _set_linear(nn.Linear(2, 2), [[1.0, 1.0], [-1.0, -1.0]],
                        [-1.5, 1.5])   # AND(OR,NAND)=XOR, その否定
        model = _mlp(h, nn.ReLU(), o)
        x = _all_binary_inputs(2)
        ir = transcribe_mlp(model, x, TranscribeSpec(n_inputs=2))
        assert validate(ir) == []
        scores = simulate(ir, x)
        ref = _reference_forward([h, o], x)
        assert np.array_equal(scores.argmax(1), ref.argmax(1))
        # XORクラス(class 0)が正しい入力で勝つ
        assert scores.argmax(1).tolist() == [1, 0, 0, 1]

    @pytest.mark.parametrize('n_inputs', [2, 4, 6])
    def test_random_threshold_net_bitexact(self, n_inputs):
        """fan-in≤n_inputsのランダム閾値ネット: 参照意味論とビット厳密一致"""
        torch.manual_seed(42 + n_inputs)
        in_w = n_inputs  # 疎化が恒等になる幅
        h1 = nn.Linear(in_w, 8).double()
        h2 = nn.Linear(8, 8).double()
        out = nn.Linear(8, 3).double()
        model = nn.Sequential(h1, nn.ReLU(), h2, nn.ReLU(), out)

        x = _all_binary_inputs(in_w)
        spec = TranscribeSpec(n_inputs=n_inputs, weight_bits=8)
        ir = transcribe_mlp(model, x, spec)
        assert validate(ir) == []
        scores = simulate(ir, x)

        # 隠れ層はfan-in>n_inputsで疎化されるため厳密でない場合がある。
        # ここではfan-in=in_w=n_inputs(h1)は厳密、h2(fan-in 8>6)は疎化近似。
        # 厳密性を保証できる構成: h2をfan-in n_inputs以下に制限した版で検証
        h2_sparse = nn.Linear(8, 8).double()
        with torch.no_grad():
            mask = torch.zeros(8, 8, dtype=torch.float64)
            for i in range(8):
                cols = torch.randperm(8)[:n_inputs]
                mask[i, cols] = 1.0
            h2_sparse.weight.mul_(mask)
        model_sparse = nn.Sequential(h1, nn.ReLU(), h2_sparse, nn.ReLU(), out)
        ir2 = transcribe_mlp(model_sparse, x, spec)
        scores2 = simulate(ir2, x)
        ref2 = _reference_forward([h1, h2_sparse, out], x)
        assert np.array_equal(scores2.argmax(1), ref2.argmax(1)), \
            'fan-in制約内のネットはargmax厳密保存'

    def test_single_linear_sign_argmax(self):
        """単層Linear+sign量子化: argmax(sign(W)x+β)を厳密保存"""
        torch.manual_seed(7)
        lin = nn.Linear(6, 4).double()
        # sign量子化が恒等になるよう重みを±1にする
        with torch.no_grad():
            lin.weight.copy_(torch.randn(4, 6).sign().double())
            lin.bias.copy_(torch.randint(-2, 3, (4,)).double()
                           * lin.weight.abs().mean())
        model = nn.Sequential(lin)
        x = _all_binary_inputs(6)
        ir = transcribe_mlp(model, x, TranscribeSpec(weight_bits=1))
        assert validate(ir) == []
        scores = simulate(ir, x)
        ref = _reference_forward([lin], x)
        assert np.array_equal(scores.argmax(1), ref.argmax(1))

    def test_group_offset_is_constant(self):
        """count_c = score_c + K(クラス共通オフセット)の構成的性質"""
        torch.manual_seed(9)
        lin = nn.Linear(5, 3).double()
        with torch.no_grad():
            lin.weight.copy_(torch.randn(3, 5).sign().double())
            lin.bias.zero_()
        x = _all_binary_inputs(5)
        ir = transcribe_mlp(nn.Sequential(lin), x, TranscribeSpec())
        scores = simulate(ir, x)
        ref = _reference_forward([lin], x)  # 整数値(±1重み、bias 0)
        diff = scores - ref.astype(np.int64)
        assert np.all(diff == diff[0, 0]), '全クラス・全入力で同一オフセット'


class TestISF:
    def test_observed_match_exact(self):
        """観測アドレス上ではISF=厳密列挙"""
        torch.manual_seed(3)
        w = torch.randn(4, 3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        exact = threshold_truth_tables(w, b)
        addrs = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [1, 3, 5, 7]])  # (B=3, out=4)
        for policy in ('zero', 'one', 'majority'):
            tts = isf_truth_tables(w, b, addrs, policy)
            for o in range(4):
                for a in np.unique(addrs[:, o]):
                    assert (tts[o] >> int(a)) & 1 == (exact[o] >> int(a)) & 1

    def test_dont_care_fill(self):
        """未観測アドレスは方策どおりに埋まる"""
        w = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([-0.5], dtype=torch.float64)  # OR
        addrs = np.array([[3]])  # addr=3のみ観測(出力1)
        tt_zero = isf_truth_tables(w, b, addrs, 'zero')[0]
        tt_one = isf_truth_tables(w, b, addrs, 'one')[0]
        tt_maj = isf_truth_tables(w, b, addrs, 'majority')[0]
        assert tt_zero == 0b1000            # 観測bit3のみ1
        assert tt_one == 0b1111             # 未観測が全部1
        assert tt_maj == 0b1111             # 観測値{1}の多数決=1

    def test_full_observation_equals_exact(self):
        """全アドレス観測ならISF=厳密(方策非依存)"""
        torch.manual_seed(4)
        w = torch.randn(3, 4, dtype=torch.float64)
        b = torch.randn(3, dtype=torch.float64)
        exact = threshold_truth_tables(w, b)
        addrs = np.tile(np.arange(16)[:, None], (1, 3))
        for policy in ('zero', 'one', 'majority'):
            assert isf_truth_tables(w, b, addrs, policy) == exact

    def test_isf_via_transcribe(self):
        """transcribe_mlp(isf=True)が較正データ上で厳密版と同一出力"""
        torch.manual_seed(5)
        h = nn.Linear(4, 6).double()
        o = nn.Linear(6, 2).double()
        model = nn.Sequential(h, nn.ReLU(), o)
        x = _all_binary_inputs(4)  # 全パターン較正=ISFも厳密
        ir_exact = transcribe_mlp(model, x, TranscribeSpec(n_inputs=4))
        ir_isf = transcribe_mlp(
            model, x, TranscribeSpec(n_inputs=4, isf=True, dont_care='zero')
        )
        assert np.array_equal(simulate(ir_exact, x), simulate(ir_isf, x))


class TestSpecValidation:
    def test_bad_n_inputs(self):
        with pytest.raises(ValueError, match='n_inputs'):
            TranscribeSpec(n_inputs=3)

    def test_bad_dont_care(self):
        with pytest.raises(ValueError, match='dont_care'):
            TranscribeSpec(dont_care='random')

    def test_isf_requires_calib(self):
        model = nn.Sequential(nn.Linear(4, 2).double())
        with pytest.raises(ValueError, match='calib'):
            transcribe_mlp(model, None, TranscribeSpec(isf=True))

    def test_no_linear(self):
        with pytest.raises(ValueError, match='Linear'):
            transcribe_mlp(nn.Sequential(nn.ReLU()), None)
