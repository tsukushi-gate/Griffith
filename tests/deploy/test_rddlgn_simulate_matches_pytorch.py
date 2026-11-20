"""PyTorch hard eval ⇔ ir/simulate のRDDLGNゴールデン検証。

これがこのタスクの要(deploykit-integration スキル②)。
「検証対象自身からゴールデンを作らない」(教訓): 基準は必ず
RDDLGNClassifier.eval()のforward(PyTorch本体)であり、IR/simulate側の
自己整合ではない。

固定するのは2段:
    1. 最終出力(argmax): PyTorch hard eval と simulate の予測ラベルが
       複数の乱数トークン列で一致する
    2. encoderのK-layer内部状態: forward_hookで観測した各tickのk_prevと、
       simulateが(内部トレースフックで記録した)各tickのstate値が
       ビット厳密に一致する(最終出力だけでなく、リカレントの途中経過も
       壊れていないことを保証する)
"""

import importlib

import numpy as np
import pytest
import torch

from griffith_logic.export.rddlgn_ir import export_rddlgn_classifier_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

from .conftest import build_rddlgn_tiny_classifier

_sim_mod = importlib.import_module('griffith_logic.ir.simulate')


def _embed_bits(model, tokens: torch.Tensor) -> np.ndarray:
    """LogicEmbedding(BOUNDARY)を通した[batch, seq_len*d_emb]平坦ビット列。

    export_rddlgn_classifier_irのIR入力ポート契約(埋め込み・Heaviside済み)
    に合わせる。
    """
    with torch.no_grad():
        emb = model.embedding(tokens)
    batch = tokens.shape[0]
    return emb.numpy().astype(np.int64).reshape(batch, -1)


class TestFinalOutputMatchesHardEval:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_argmax_matches_across_random_token_sequences(self, seed):
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        assert validate(graph) == []

        rng = np.random.default_rng(seed)
        batch = 6
        tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len)),
            dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(tokens)
        pt_pred = logits.argmax(dim=-1).numpy()

        x = _embed_bits(model, tokens)
        scores = simulate(graph, x)
        sim_pred = scores.argmax(axis=-1)
        assert np.array_equal(pt_pred, sim_pred), (
            f'seed={seed}: PyTorch hard eval と simulate の予測が不一致\n'
            f'pt={pt_pred}\nsim={sim_pred}'
        )

    def test_scores_are_a_fixed_tau_multiple_of_pytorch_logits(self):
        """simulateは整数popcount、PyTorchはGroupSumで/tauされたfloat。

        tauはargmax不変なので(ir/simulate.py docstring)、simulateは
        意図的にtauを掛けない。scores*(1/tau) == logits で数値も
        きちんと対応していることまで確認する(タイブレークの偶然一致で
        argmaxだけ揃っている、というごまかしを防ぐ)。
        """
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        rng = np.random.default_rng(42)
        tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(4, cfg.seq_len)),
            dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(tokens)
        x = _embed_bits(model, tokens)
        scores = simulate(graph, x).astype(np.float64)
        tau = float(model.group_sum.tau)
        assert np.allclose(scores / tau, logits.numpy(), atol=1e-4)


class TestKLayerInternalStateMatchesHardEval:
    """encoderの各tickのk状態がforward_hookとsimulate内部トレースで一致する。

    K-layerのrecurrentノード(state 1個・output_mode='final_state')は
    このタスクの中核なので、最終出力だけでなく途中の各tickも見る。
    """

    @pytest.mark.parametrize('seed', [10, 11])
    def test_all_ticks_bit_exact(self, seed):
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)

        rng = np.random.default_rng(seed)
        batch = 5
        tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len)),
            dtype=torch.long)

        k_states_pt: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            k_states_pt.append(output.detach().clone())

        handle = model.encoder.k_layers.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(tokens)
        finally:
            handle.remove()
        assert len(k_states_pt) == cfg.seq_len

        x = _embed_bits(model, tokens)
        _sim_mod._recurrent_trace_sink = {}
        try:
            simulate(graph, x)
            trace = _sim_mod._recurrent_trace_sink['enc_k_rec']
        finally:
            _sim_mod._recurrent_trace_sink = None
        assert len(trace) == cfg.seq_len

        for t in range(cfg.seq_len):
            pt_bits = (k_states_pt[t].numpy() > 0.5).astype(np.int64)
            sim_bits = trace[t]['sv_state']
            assert np.array_equal(pt_bits, sim_bits), (
                f'seed={seed}, t={t}: k状態がPyTorchとsimulateで不一致'
            )

    def test_state_starts_all_zero_for_zero_init_mode(self):
        """init_mode='zero'はhard evalで0.5→0(strict >)に倒れる。"""
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        assert model.encoder.init_mode == 'zero'
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)

        rng = np.random.default_rng(0)
        tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(2, cfg.seq_len)),
            dtype=torch.long)
        x = _embed_bits(model, tokens)

        _sim_mod._recurrent_trace_sink = {}
        try:
            simulate(graph, x)
            trace = _sim_mod._recurrent_trace_sink['enc_k_rec']
        finally:
            _sim_mod._recurrent_trace_sink = None
        # t=0でのstate_valsは「1つ目のtickを処理した*後*」の値なので、
        # 初期値そのものを見るにはbody内のscan_varノードのinitを直接見る。
        k_rec = graph.node_by_id('enc_k_rec')
        init_bits = k_rec.params['body'].node_by_id('sv_state').params['init']['bits']
        assert init_bits == [0] * len(init_bits)
        assert len(trace) == cfg.seq_len
