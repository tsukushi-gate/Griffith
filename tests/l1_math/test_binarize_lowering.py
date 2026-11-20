"""C2 学習後二値化lowering(convert/binarize.py)の厳密性テスト。

参照意味論(binarize.pyモジュールdocstring):
    二値化: q = sign(W)、α = mean|W|、b'_i = round(b_i/α)
    隠れ層: y_i = 1[Σ_j q_ij·(2x_j-1) + b'_i > 0] = 1[XNORカウント_i >= θ_i]
            θ_i = floor((n - b'_i)/2) + 1
    最終層: argmax_c(Σ_j q_cj x_j + β_c) を保存(sign量子化=±1)

核心の主張:
    「精度損失はPTB段階(実数→±1)のみ。lowering自体はビット厳密」。
    → bipolar BNN(numpy参照)とIR simulateは全一致でなければならない。
      実数MLPとBNNは一致しなくてよい。

検証はすべてビット/整数の厳密比較(許容誤差なし。conftestの定数も不要)。
"""

import math

import numpy as np
import pytest
import torch
from torch import nn

from griffith_logic.convert.binarize import (
    _hidden_thresholds,
    binarize_weights,
    lower_bnn_mlp,
    lower_int_mlp,
    popcount_threshold_nodes,
    quantize_threshold_layer,
)
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.schema import IREdge, IRGraph, IRPort
from griffith_logic.ir.simulate import simulate, simulate_trace
from griffith_logic.ir.validate import validate


def _all_binary_inputs(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


def _build_popcount_graph(width: int, thresholds: list[int]) -> IRGraph:
    """popcount_threshold_nodesを単体IRとして組み立てる(入力x→...→y)。"""
    n_out = len(thresholds)
    input_cols = np.tile(np.arange(width)[None, :], (n_out, 1))
    nodes, edges, last, _ = popcount_threshold_nodes(
        width, thresholds, 'pc', in_width=width, input_cols=input_cols
    )
    edges = (
        [IREdge(src='x', dst='pc_p0')]
        + edges
        + [IREdge(src=last, dst='y')]
    )
    return IRGraph(
        metadata={},
        inputs=[IRPort(id='x', width=width)],
        outputs=[IRPort(id='y', width=n_out)],
        nodes=nodes,
        edges=edges,
    )


def _bnn_forward(linears: list[nn.Linear], x: np.ndarray):
    """bipolar BNNのnumpy参照実装(sign二値化、XNORカウント閾値)。

    Returns:
        (scores, hidden_bits): scoresは最終層の実数スコア(±1×整数)、
        hidden_bitsは各隠れ層の0/1出力リスト。
    """
    h = x.astype(np.int64)
    hidden_bits = []
    for lin in linears[:-1]:
        w = lin.weight.detach().numpy().astype(np.float64)
        b = lin.bias.detach().numpy().astype(np.float64)
        q = np.where(w >= 0, 1, -1)
        alpha = np.abs(w).mean()
        b_prime = np.round(b / alpha)
        # y_i = 1[Σ_j q_ij(2x_j-1) + b'_i > 0]
        s = (q[None, :, :] * (2 * h[:, None, :] - 1)).sum(2) + b_prime[None, :]
        h = (s > 0).astype(np.int64)
        hidden_bits.append(h.copy())
    lin = linears[-1]
    w = lin.weight.detach().numpy().astype(np.float64)
    b = lin.bias.detach().numpy().astype(np.float64)
    q = np.where(w >= 0, 1.0, -1.0)
    alpha = np.abs(w).mean()
    beta = np.round(b / alpha)
    scores = h @ q.T + beta[None, :]
    return scores, hidden_bits


class TestBinarizeWeights:
    def test_sign_and_scale(self):
        w = torch.tensor([[2.0, -1.0], [0.0, -3.0]], dtype=torch.float64)
        q, alpha = binarize_weights(w)
        # sign(0)=+1(境界は正符号扱い)
        assert q.tolist() == [[1, -1], [1, -1]]
        assert alpha == pytest.approx((2 + 1 + 0 + 3) / 4)

    def test_bad_method(self):
        with pytest.raises(ValueError, match='sign'):
            binarize_weights(torch.zeros(2, 2), method='billm_like')


class TestPopcountThreshold:
    @pytest.mark.parametrize('width', list(range(1, 9)))
    def test_full_enumeration(self, width):
        """width∈{1..8}を全入力網羅でnumpy参照(popcount>=θ)と厳密一致。"""
        thresholds = list(range(0, width + 2))  # 0..width+1(常真・常偽含む)
        g = _build_popcount_graph(width, thresholds)
        assert validate(g) == []
        x = _all_binary_inputs(width)
        out = simulate(g, x)
        pc = x.sum(1)
        ref = np.stack([(pc >= t).astype(np.int64) for t in thresholds], axis=1)
        assert np.array_equal(out, ref)

    def test_width20_random(self):
        """width=20をランダム512パターンで一致(θ複数)。"""
        width = 20
        thresholds = [0, 1, 5, 10, 11, 15, 20, 21]
        g = _build_popcount_graph(width, thresholds)
        assert validate(g) == []
        rng = np.random.default_rng(20)
        x = rng.integers(0, 2, size=(512, width)).astype(np.int64)
        out = simulate(g, x)
        pc = x.sum(1)
        ref = np.stack([(pc >= t).astype(np.int64) for t in thresholds], axis=1)
        assert np.array_equal(out, ref)

    def test_clamped_thresholds(self):
        """θ≤0は常時真、θ>widthは常時偽へクランプされる(意味保存)。"""
        width = 5
        g = _build_popcount_graph(width, [-3, 0, 6, 100])
        assert validate(g) == []
        x = _all_binary_inputs(width)
        out = simulate(g, x)
        assert np.all(out[:, 0] == 1)  # θ=-3 → 常真
        assert np.all(out[:, 1] == 1)  # θ=0  → 常真
        assert np.all(out[:, 2] == 0)  # θ=6>5 → 常偽
        assert np.all(out[:, 3] == 0)  # θ=100 → 常偽


class TestThresholdDerivation:
    def test_hand_example_width3(self):
        """θ_i = floor((n - b'_i)/2) + 1 を手計算例で検証(width=3)。"""
        q = torch.tensor([[1, 1, 1]])  # n=3
        for b_prime in range(-3, 4):
            b = torch.tensor([float(b_prime)])
            theta = _hidden_thresholds(q, b, 1.0)[0]  # alpha=1でb'=b
            assert theta == math.floor((3 - b_prime) / 2) + 1

    def test_derivation_matches_bnn_semantics(self):
        """θ導出が発火判定 1[Σq(2x-1)+b'>0] と全入力で一致(width=6)。"""
        torch.manual_seed(11)
        n = 6
        w = torch.randn(4, n, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        q, alpha = binarize_weights(w)
        thetas = _hidden_thresholds(q, b, alpha)
        x = _all_binary_inputs(n)
        b_prime = np.round(b.numpy() / alpha)
        qn = q.numpy()
        for i in range(4):
            s = (qn[i][None, :] * (2 * x - 1)).sum(1) + b_prime[i]
            fire_ref = (s > 0).astype(np.int64)
            xnor_count = np.where(qn[i][None, :] > 0, x, 1 - x).sum(1)
            fire_theta = (xnor_count >= thetas[i]).astype(np.int64)
            assert np.array_equal(fire_ref, fire_theta)


class TestLowerBnnMlp:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_bitexact_vs_bnn_reference(self, seed):
        """2層MLP(8→12→3): BNN参照とIR simulateが argmax+隠れ層とも全一致。"""
        torch.manual_seed(seed)
        h1 = nn.Linear(8, 12).double()
        h2 = nn.Linear(12, 10).double()
        out = nn.Linear(10, 3).double()
        model = nn.Sequential(h1, nn.ReLU(), h2, nn.ReLU(), out)

        ir = lower_bnn_mlp(model)
        assert validate(ir) == []

        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(256, 8)).astype(np.int64)
        scores_ref, hidden_ref = _bnn_forward([h1, h2, out], x)

        trace = simulate_trace(ir, x)
        scores = simulate(ir, x)
        # 隠れ層出力もビット一致(中間トレース)
        assert np.array_equal(trace['n0_pc_ge'], hidden_ref[0])
        assert np.array_equal(trace['n1_pc_ge'], hidden_ref[1])
        # argmax厳密保存
        assert np.array_equal(scores.argmax(1), scores_ref.argmax(1))

    def test_single_hidden_small_full_enum(self):
        """小MLP(4→5→2)を全16入力で網羅照合(隠れ+argmax)。"""
        torch.manual_seed(7)
        h1 = nn.Linear(4, 5).double()
        out = nn.Linear(5, 2).double()
        model = nn.Sequential(h1, nn.ReLU(), out)
        ir = lower_bnn_mlp(model)
        assert validate(ir) == []
        x = _all_binary_inputs(4)
        scores_ref, hidden_ref = _bnn_forward([h1, out], x)
        trace = simulate_trace(ir, x)
        assert np.array_equal(trace['n0_pc_ge'], hidden_ref[0])
        assert np.array_equal(simulate(ir, x).argmax(1), scores_ref.argmax(1))

    def test_single_linear_no_hidden(self):
        """単層Linear(隠れなし): 最終層のみでargmax保存。"""
        torch.manual_seed(5)
        lin = nn.Linear(6, 4).double()
        ir = lower_bnn_mlp(nn.Sequential(lin))
        assert validate(ir) == []
        x = _all_binary_inputs(6)
        scores_ref, _ = _bnn_forward([lin], x)
        assert np.array_equal(simulate(ir, x).argmax(1), scores_ref.argmax(1))

    def test_metadata_circuit_stats(self):
        """metadataにsourceと回路統計(lut数・段数)が入る。"""
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(6, 8).double(), nn.ReLU(),
                              nn.Linear(8, 3).double())
        ir = lower_bnn_mlp(model)
        assert ir.metadata['source'] == 'convert.lower_bnn_mlp'
        assert ir.metadata['binarization'] == 'sign'
        circ = ir.metadata['circuit']
        assert circ['total_lut_outputs'] > 0
        assert circ['n_lut_layers'] > 0

    def test_no_linear_raises(self):
        with pytest.raises(ValueError, match='Linear'):
            lower_bnn_mlp(nn.Sequential(nn.ReLU()))


class TestBuildTorchCompat:
    """build_torch互換性(回復学習経路の前提)。

    LutLayerは n_inputs∈{2,4,6} をassertするため、C2出力IRの全lut_layerが
    この規約を守り、build_torch→hard evalがIR simulateと厳密一致すること。
    """

    def _small_ir(self, seed=3):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(8, 12).double(), nn.ReLU(),
            nn.Linear(12, 10).double(), nn.ReLU(),
            nn.Linear(10, 3).double(),
        )
        return lower_bnn_mlp(model)

    def test_all_lut_layers_have_allowed_n_inputs(self):
        """全lut_layerノードで n_inputs∈{2,4,6}(LutLayerのassert対応)。"""
        ir = self._small_ir()
        lut_nodes = [n for n in ir.nodes if n.type == 'lut_layer']
        assert lut_nodes, 'lut_layerノードが存在するはず'
        for n in lut_nodes:
            assert n.params['n_inputs'] in (2, 4, 6), (
                f"{n.id}: n_inputs={n.params['n_inputs']} はLutLayer非対応"
            )

    def test_build_torch_hard_eval_matches_simulate(self):
        """統合: build_torchが例外なく構築でき、hard eval ⇔ simulate 厳密一致
        (3段等価性の第1段の流儀)。"""
        ir = self._small_ir()
        assert validate(ir) == []
        model = build_torch(ir)  # LutLayer構築(n_inputs assert)を通過
        model.eval()
        rng = np.random.default_rng(31)
        x = rng.integers(0, 2, size=(256, 8)).astype(np.int64)
        with torch.no_grad():
            y = model(torch.tensor(x, dtype=torch.float32))
        hard = np.round(y.numpy()).astype(np.int64)
        assert np.array_equal(hard, simulate(ir, x))

    def test_build_torch_popcount_standalone(self):
        """popcount単体IRもbuild_torch可能でhard eval一致(width=6全列挙)。"""
        width = 6
        thresholds = [0, 2, 4, 6, 7]
        ir = _build_popcount_graph(width, thresholds)
        assert validate(ir) == []
        model = build_torch(ir)
        model.eval()
        x = _all_binary_inputs(width)
        with torch.no_grad():
            y = model(torch.tensor(x, dtype=torch.float32))
        hard = np.round(y.numpy()).astype(np.int64)
        assert np.array_equal(hard, simulate(ir, x))


class TestPtbLossOnlyClaim:
    """「精度損失はPTB段階のみ」: 実数⇔BNN⇔IR の3者関係を明示する。

    BNN⇔IRは厳密一致(loweringはビット厳密)。実数⇔BNNは一致しなくてよい
    (PTBで符号量子化するため)。この非自明な構成で主張を実証する。
    """

    def test_bnn_equals_ir_but_real_may_differ(self):
        torch.manual_seed(13)
        # 実数重みが±1から離れるようスケールを崩す → PTBで情報が落ちる
        h1 = nn.Linear(8, 10).double()
        out = nn.Linear(10, 4).double()
        with torch.no_grad():
            h1.weight.mul_(3.0)
            h1.bias.mul_(3.0)
        model = nn.Sequential(h1, nn.ReLU(), out)

        rng = np.random.default_rng(13)
        x = rng.integers(0, 2, size=(200, 8)).astype(np.int64)

        # 実数MLPのargmax(参照): 隠れは発火判定(実数重み)、最終は実数スコア
        hr = x.astype(np.float64)
        w1 = h1.weight.detach().numpy()
        b1 = h1.bias.detach().numpy()
        hr = ((hr @ w1.T + b1) > 0).astype(np.float64)
        wo = out.weight.detach().numpy()
        bo = out.bias.detach().numpy()
        real_argmax = (hr @ wo.T + bo).argmax(1)

        # BNN参照(sign二値化)とIR
        ir = lower_bnn_mlp(model)
        assert validate(ir) == []
        bnn_scores, _ = _bnn_forward([h1, out], x)
        bnn_argmax = bnn_scores.argmax(1)
        ir_argmax = simulate(ir, x).argmax(1)

        # 主張1: BNN ⇔ IR は厳密一致(lowering無損失)
        assert np.array_equal(bnn_argmax, ir_argmax)
        # 主張2: 実数 ⇔ BNN は一致しない場合がある(PTBが精度損失の源)
        assert not np.array_equal(real_argmax, bnn_argmax), (
            'この構成ではPTBで予測が変わるはず(主張の非自明性)'
        )


# ============================================================
# 整数量子化閾値ニューロンの加算木厳密lowering
# ============================================================

def _int_mlp_forward(linears: list[nn.Linear], x: np.ndarray,
                     bits_hidden: int, bits_final: int):
    """整数量子化MLPのnumpy参照(lower_int_mlp が厳密化する意味論)。

    隠れ: ニューロン別スケール量子化 q,β → count=Σ q x + C0、
          y=1[count>=θ]、θ=C0-β+1(= Σ q x + β > 0)。
    最終: 共有スケール量子化 → score=Σ q x + β、argmaxを保存。

    Returns:
        (scores, hidden_bits): scores=最終スコア(整数)、hidden_bits=各隠れ層0/1。
    """
    h = x.astype(np.int64)
    hidden_bits = []
    for lin in linears[:-1]:
        w = lin.weight.detach()
        b = lin.bias.detach()
        q, beta = quantize_threshold_layer(w, b, bits_hidden)
        qn = q.numpy()
        bn = beta.numpy()
        s = h @ qn.T + bn[None, :]        # Σ q x + β
        h = (s > 0).astype(np.int64)      # ⟺ count >= θ
        hidden_bits.append(h.copy())
    lin = linears[-1]
    w = lin.weight.detach()
    b = lin.bias.detach()
    # 最終層は transcribe._quantize_final と同一(共有スケール)。
    from griffith_logic.convert.transcribe import _quantize_final
    q, beta = _quantize_final(w.to(torch.float64), b.to(torch.float64),
                              bits_final)
    scores = h @ q.numpy().T + beta.numpy()[None, :]
    return scores, hidden_bits


class TestQuantizeThresholdLayer:
    def test_bits1_is_sign(self):
        """bits=1 は sign 量子化(±1)。"""
        w = torch.tensor([[2.0, -1.0], [0.0, -3.0]], dtype=torch.float64)
        b = torch.zeros(2, dtype=torch.float64)
        q, beta = quantize_threshold_layer(w, b, 1)
        assert q.tolist() == [[1, -1], [1, -1]]

    def test_bits_range_and_per_neuron_scale(self):
        """bits>=2: |q| <= qmax、行ごとに独立スケール(max|w_i|=qmax)。"""
        w = torch.tensor([[4.0, -2.0, 1.0], [0.5, -0.5, 0.25]],
                         dtype=torch.float64)
        b = torch.zeros(2, dtype=torch.float64)
        q, _ = quantize_threshold_layer(w, b, 4)  # qmax=7
        assert q.abs().max().item() <= 7
        # 各行の最大絶対重みは ±qmax に量子化される(ニューロン別スケール)
        assert int(q[0].abs().max()) == 7
        assert int(q[1].abs().max()) == 7

    def test_bad_bits(self):
        with pytest.raises(ValueError, match='bits'):
            quantize_threshold_layer(torch.zeros(2, 2), torch.zeros(2), 0)


class TestLowerIntMlp:
    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_bitexact_vs_int_reference(self, bits):
        """2層MLP: 整数量子化参照とIR simulateが隠れ+argmax全一致(全16入力)。"""
        torch.manual_seed(bits)
        h1 = nn.Linear(4, 6).double()
        h2 = nn.Linear(6, 5).double()
        out = nn.Linear(5, 3).double()
        model = nn.Sequential(h1, nn.ReLU(), h2, nn.ReLU(), out)
        ir = lower_int_mlp(model, weight_bits_hidden=bits, weight_bits_final=6)
        assert validate(ir) == []
        x = _all_binary_inputs(4)
        scores_ref, hidden_ref = _int_mlp_forward([h1, h2, out], x, bits, 6)
        trace = simulate_trace(ir, x)
        assert np.array_equal(trace['n0_pc_ge'], hidden_ref[0])
        assert np.array_equal(trace['n1_pc_ge'], hidden_ref[1])
        assert np.array_equal(simulate(ir, x).argmax(1), scores_ref.argmax(1))

    def test_bits1_is_sign_lowering(self):
        """bits=1: q=sign(W) の unipolar 閾値ニューロン 1[Σqx+β>0] を厳密lowering。

        lower_int_mlp は unipolar 入力 x∈{0,1} の閾値ニューロン
        1[Σ_j q_ij x_j + β_i > 0] を実装する(count=Σqx+C0、θ=C0-β+1)。
        bits=1 では q=sign(W)=±1 となり、これが本 lowering の sign 特殊化。
        lower_bnn_mlp の bipolar 版 1[Σq(2x-1)+b'>0] とは入力エンコーディングが
        異なる(bipolar vs unipolar)ため発火関数は一般に別物 — ここでは
        「bits=1 が sign 量子化の厳密 unipolar lowering になる」ことを確認する。
        """
        torch.manual_seed(21)
        h1 = nn.Linear(6, 8).double()
        out = nn.Linear(8, 4).double()
        model = nn.Sequential(h1, nn.ReLU(), out)
        # q が sign(W)=±1 と一致(bits=1 の定義)
        q_int, _ = quantize_threshold_layer(h1.weight.detach(),
                                            h1.bias.detach(), 1)
        assert np.array_equal(q_int.numpy(),
                              np.where(h1.weight.detach().numpy() >= 0, 1, -1))
        # unipolar 整数参照(sign重み)と IR simulate がビット一致
        ir = lower_int_mlp(model, weight_bits_hidden=1, weight_bits_final=1)
        assert validate(ir) == []
        x = _all_binary_inputs(6)
        _, hidden_ref = _int_mlp_forward([h1, out], x, 1, 1)
        assert np.array_equal(simulate_trace(ir, x)['n0_pc_ge'], hidden_ref[0])

    def test_threshold_clamp_always_true_and_false(self):
        """θクランプ端: 常真ニューロン(θ<=0)と常偽ニューロン(θ>W_i)を厳密再現。

        重みを手で構成: ニューロン0は全正+大バイアス(常に発火=常真)、
        ニューロン1は全負+大負バイアス(常に不発火=常偽)。
        """
        h1 = nn.Linear(4, 2).double()
        out = nn.Linear(2, 2).double()
        with torch.no_grad():
            h1.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                          [-1.0, -1.0, -1.0, -1.0]]))
            h1.bias.copy_(torch.tensor([10.0, -10.0]))  # 常真 / 常偽
            out.weight.copy_(torch.eye(2))
            out.bias.zero_()
        model = nn.Sequential(h1, nn.ReLU(), out)
        ir = lower_int_mlp(model, weight_bits_hidden=3, weight_bits_final=3)
        assert validate(ir) == []
        x = _all_binary_inputs(4)
        hidden = simulate_trace(ir, x)['n0_pc_ge']
        assert np.all(hidden[:, 0] == 1), 'ニューロン0は常真のはず'
        assert np.all(hidden[:, 1] == 0), 'ニューロン1は常偽のはず'
        # 整数参照とも一致
        _, hidden_ref = _int_mlp_forward([h1, out], x, 3, 3)
        assert np.array_equal(hidden, hidden_ref[0])

    def test_single_linear_no_hidden(self):
        """単層Linear: 最終層のみでargmax保存。"""
        torch.manual_seed(5)
        lin = nn.Linear(6, 4).double()
        ir = lower_int_mlp(nn.Sequential(lin), weight_bits_final=8)
        assert validate(ir) == []
        x = _all_binary_inputs(6)
        scores_ref, _ = _int_mlp_forward([lin], x, 4, 8)
        assert np.array_equal(simulate(ir, x).argmax(1), scores_ref.argmax(1))

    def test_metadata_circuit_stats(self):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(6, 8).double(), nn.ReLU(),
                              nn.Linear(8, 3).double())
        ir = lower_int_mlp(model, weight_bits_hidden=4, weight_bits_final=8)
        assert ir.metadata['source'] == 'convert.lower_int_mlp'
        assert ir.metadata['weight_bits_hidden'] == 4
        circ = ir.metadata['circuit']
        assert circ['total_lut_outputs'] > 0
        assert circ['n_lut_layers'] > 0
        assert len(circ['hidden_unary_widths']) == 1

    def test_no_linear_raises(self):
        with pytest.raises(ValueError, match='Linear'):
            lower_int_mlp(nn.Sequential(nn.ReLU()))


class TestLowerIntBuildTorchCompat:
    """build_torch互換(n_inputs∈{2,4,6})+ hard eval ⇔ simulate 一致。"""

    @pytest.mark.parametrize('n_inputs', [2, 4, 6])
    def test_build_torch_hard_eval_matches_simulate(self, n_inputs):
        torch.manual_seed(n_inputs)
        model = nn.Sequential(
            nn.Linear(n_inputs, 8).double(), nn.ReLU(),
            nn.Linear(8, 3).double(),
        )
        ir = lower_int_mlp(model, weight_bits_hidden=3, weight_bits_final=6)
        assert validate(ir) == []
        for n in ir.nodes:
            if n.type == 'lut_layer':
                assert n.params['n_inputs'] in (2, 4, 6)
        m = build_torch(ir)
        m.eval()
        x = _all_binary_inputs(n_inputs)
        with torch.no_grad():
            y = m(torch.tensor(x, dtype=torch.float32))
        hard = np.round(y.numpy()).astype(np.int64)
        assert np.array_equal(hard, simulate(ir, x))
