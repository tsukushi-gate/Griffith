"""PyTorch hard eval ⇔ ir/simulate のRDDLGN seq2seq(教師強制)ゴールデン検証。

test_rddlgn_simulate_matches_pytorch.py(エンコーダ+分類ヘッド)の
続き。ここではRDDLGN本体(エンコーダ+デコーダ、教師強制)を対象にする。
基準は必ずRDDLGN.eval()のforward(PyTorch本体)であり、IR/simulate側の
自己整合ではない(「検証対象自身からゴールデンを作らない」教訓)。

固定するのは2段:
    1. 各tickの出力(argmax): PyTorch hard evalとsimulateの予測トークンが
       複数の乱数src/tgt系列で一致する(教師強制の全tickぶん)
    2. デコーダP-layer状態: forward_hook不要のreturn_hidden経路で観測した
       各tickのp_tと、simulateが(内部トレースフックで記録した)各tickの
       state値がビット厳密に一致する(P+M+GroupSumを1つのrecurrent bodyへ
       畳んだ中核なので、最終出力だけでなく途中経過も見る)
"""

import importlib

import numpy as np
import pytest
import torch

from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

from .conftest import build_rddlgn_tiny_seq2seq

_sim_mod = importlib.import_module('griffith_logic.ir.simulate')


def _embed_bits(model, src_tokens: torch.Tensor,
                tgt_tokens: torch.Tensor) -> np.ndarray:
    """src/tgt埋め込み(Heaviside済み)を連結した[batch, (src+tgt)*d_emb]。

    export_rddlgn_irのIR入力ポート契約(x_src ++ x_tgt を平坦連結、
    ともに埋め込み・Heaviside済みビット列)に合わせる。
    """
    with torch.no_grad():
        src_emb = model.src_embedding(src_tokens)
        tgt_emb = model.tgt_embedding(tgt_tokens)
    batch = src_tokens.shape[0]
    src_bits = src_emb.numpy().astype(np.int64).reshape(batch, -1)
    tgt_bits = tgt_emb.numpy().astype(np.int64).reshape(batch, -1)
    return np.concatenate([src_bits, tgt_bits], axis=1)


class TestFinalOutputMatchesHardEval:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_argmax_matches_across_random_token_sequences(self, seed):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        src_len = tgt_len = cfg.seq_len
        graph = export_rddlgn_ir(model, src_len, tgt_len)
        assert validate(graph) == []

        rng = np.random.default_rng(seed)
        batch = 5
        src_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, src_len)),
            dtype=torch.long)
        tgt_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, tgt_len)),
            dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(src_tokens, tgt_tokens)
        pt_pred = logits.argmax(dim=-1).numpy()   # (batch, tgt_len)

        x = _embed_bits(model, src_tokens, tgt_tokens)
        scores = simulate(graph, x)                # (batch, tgt_len*vocab)
        sim_pred = scores.reshape(batch, tgt_len, cfg.vocab_size).argmax(-1)
        assert np.array_equal(pt_pred, sim_pred), (
            f'seed={seed}: PyTorch hard eval と simulate の予測が不一致\n'
            f'pt={pt_pred}\nsim={sim_pred}'
        )

    def test_scores_are_a_fixed_tau_multiple_of_pytorch_logits(self):
        """simulateは整数popcount、PyTorchはGroupSumで/tauされたfloat。

        tauはargmax不変なので(ir/simulate.py docstring)simulateは
        意図的にtauを掛けない。scores*(1/tau) == logits で数値も
        きちんと対応していることまで確認する(タイブレークの偶然一致で
        argmaxだけ揃っている、というごまかしを防ぐ。
        test_rddlgn_simulate_matches_pytorch.py と同じ手口)。
        """
        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        src_len = tgt_len = cfg.seq_len
        graph = export_rddlgn_ir(model, src_len, tgt_len)

        rng = np.random.default_rng(42)
        batch = 4
        src_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, src_len)),
            dtype=torch.long)
        tgt_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, tgt_len)),
            dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(src_tokens, tgt_tokens)
        x = _embed_bits(model, src_tokens, tgt_tokens)
        scores = simulate(graph, x).astype(np.float64).reshape(
            batch, tgt_len, cfg.vocab_size)
        tau = float(model.decoder.group_sum.tau)
        assert np.allclose(scores / tau, logits.numpy(), atol=1e-4)


class TestDecoderStateMatchesHardEval:
    """デコーダのP-layer自己回帰状態(dec_pm_recのsv_p)が各tickで一致する。

    P+M+GroupSumを1つのrecurrent bodyへ畳んだ(中核)ので、
    最終出力(GroupSum後のスコア)だけでなく、bodyの外へ持ち越される
    stateそのものが壊れていないことを別途固定する。
    """

    @pytest.mark.parametrize('seed', [10, 11])
    def test_all_ticks_bit_exact(self, seed):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        src_len = tgt_len = cfg.seq_len
        graph = export_rddlgn_ir(model, src_len, tgt_len)

        rng = np.random.default_rng(seed)
        batch = 4
        src_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, src_len)),
            dtype=torch.long)
        tgt_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, tgt_len)),
            dtype=torch.long)

        with torch.no_grad():
            _, _, hidden = model(src_tokens, tgt_tokens, return_hidden=True)
        # hidden = [context, p_stack]。p_stack: (batch, tgt_len, p_out_dim)
        # ([0,1]論理域、GroupSum前の自己回帰リカレント状態。
        # models/rddlgn.py::RDDLGN.forwardのdocstring参照)
        p_stack = hidden[1]
        assert p_stack.shape[1] == tgt_len

        x = _embed_bits(model, src_tokens, tgt_tokens)
        _sim_mod._recurrent_trace_sink = {}
        try:
            simulate(graph, x)
            trace = _sim_mod._recurrent_trace_sink['dec_pm_rec']
        finally:
            _sim_mod._recurrent_trace_sink = None
        assert len(trace) == tgt_len

        for t in range(tgt_len):
            pt_bits = (p_stack[:, t, :].numpy() > 0.5).astype(np.int64)
            sim_bits = trace[t]['sv_p']
            assert np.array_equal(pt_bits, sim_bits), (
                f'seed={seed}, t={t}: デコーダのP状態がPyTorchとsimulateで不一致'
            )

    def test_state_starts_all_zero_for_zero_init_mode(self):
        """init_mode='zero'はhard evalで0.5→0(strict >)に倒れる。

        decoder.init_modeもencoderと同じ_init_hidden規則を共有する
        (models/rddlgn.py::RDDLGNDecoder.forward)。
        """
        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        assert model.decoder.init_mode == 'zero'
        src_len = tgt_len = cfg.seq_len
        graph = export_rddlgn_ir(model, src_len, tgt_len)

        pm_rec = graph.node_by_id('dec_pm_rec')
        init_bits = pm_rec.params['body'].node_by_id('sv_p').params['init']['bits']
        assert init_bits == [0] * len(init_bits)
