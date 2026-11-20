"""統合: FP GRU 対照モデルのモデル級契約。

`experiments/train_fp_seq_mixer.py` の :class:`FPGRUTranslator` は、論文
RDDLGN の GRU ベースライン(BLEU 5.41)を**我々の評価系**で測り直すための対照。
固定する契約:
    - forward契約: logits (B, T, vocab_size) を単体で返す(FPSeqMixerと同一)
    - **因果性(最重要)**: tgt_tokens の位置t0より後ろを任意に変えても
      logits[:, :t0+1] がビット厳密に不変(全t0掃引)
    - 呼び出し面: encode / decode_logits が greedy_decode の想定どおり働く
      (memory は tgt に一切依存しない)
    - パラメータ予算: 解析式と実測が一致し、既定探索が FP Mixer の
      40,660,648 へ十分近い hidden を返す
"""

import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments'))

from train_fp_seq_mixer import (  # noqa: E402
    FP_MIXER_PAPER_PARAMS,
    FPGRUConfig,
    FPGRUTranslator,
    estimate_gru_parameters,
    solve_gru_hidden,
)

PARAM_BUDGET_TOLERANCE = 0.30


def _tiny(**overrides) -> FPGRUConfig:
    base = FPGRUConfig(vocab_size=50, d_emb=32, seq_len=8, hidden=24, n_layers=2)
    return replace(base, **overrides) if overrides else base


def _tokens(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


def _build(cfg: FPGRUConfig, seed: int = 42) -> FPGRUTranslator:
    torch.manual_seed(seed)
    return FPGRUTranslator(cfg)


class TestForwardContract:
    @pytest.mark.parametrize('batch', [1, 2, 3])
    def test_returns_logits_only(self, batch):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        src = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=1)
        tgt = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=2)
        logits = model(src, tgt)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch, cfg.seq_len, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_variable_src_and_tgt_lengths(self):
        """S と T を独立に変えても形状契約を満たす(teacher forcing の T=S+1 含む)。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        for s_len in (cfg.seq_len - 1, cfg.seq_len):
            for t_len in (1, cfg.seq_len, cfg.seq_len + 1):
                src = _tokens(2, s_len, cfg.vocab_size, seed=3)
                tgt = _tokens(2, t_len, cfg.vocab_size, seed=4)
                logits = model(src, tgt)
                assert logits.shape == (2, t_len, cfg.vocab_size), \
                    f'S={s_len}, T={t_len} で形状契約違反: {tuple(logits.shape)}'

    def test_memory_is_independent_of_tgt(self):
        """encode は src だけの関数(greedy decode が memory を使い回せる根拠)。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=9)
        with torch.no_grad():
            mem = model.encode(src)
            assert mem.shape == (2, cfg.n_layers, cfg.hidden)
            tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=10)
            assert torch.equal(model(src, tgt), model.decode_logits(tgt, mem))


class TestCausality:
    """★最重要: 単方向GRUの因果性(未来トークンが過去のlogitsに漏れない)。"""

    def _assert_causal(self, model: FPGRUTranslator) -> None:
        cfg = model.config
        model.eval()
        batch, t_len = 2, cfg.seq_len
        src = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=5)
        tgt = _tokens(batch, t_len, cfg.vocab_size, seed=6)

        with torch.no_grad():
            base_logits = model(src, tgt)

        for t0 in range(t_len):
            if t_len - t0 - 1 == 0:
                continue
            tgt_mod = tgt.clone()
            tgt_mod[:, t0 + 1:] = (tgt[:, t0 + 1:] + 1) % cfg.vocab_size
            assert not torch.equal(tgt, tgt_mod), '未来トークンが変化していない'
            with torch.no_grad():
                mod_logits = model(src, tgt_mod)
            assert torch.equal(base_logits[:, :t0 + 1], mod_logits[:, :t0 + 1]), \
                (f'因果性違反: t0={t0} より後ろのトークン変更が '
                 f'logits[:, :{t0 + 1}] に影響している')

    @pytest.mark.parametrize('n_layers', [1, 2])
    def test_causal(self, n_layers):
        self._assert_causal(_build(_tiny(n_layers=n_layers)))


class TestParameterBudget:
    @pytest.mark.parametrize('n_layers', [1, 2, 3])
    @pytest.mark.parametrize('shared_vocab', [True, False])
    def test_estimate_matches_actual(self, n_layers, shared_vocab):
        cfg = _tiny(n_layers=n_layers, shared_vocab=shared_vocab)
        model = _build(cfg)
        assert model.count_parameters()['total'] == estimate_gru_parameters(cfg)
        assert model.count_parameters()['total'] == sum(
            p.numel() for p in model.parameters())

    def test_solved_hidden_matches_fp_mixer_budget(self):
        """既定探索の hidden が FP Mixer 40,660,648 の ±30% 以内へ収まる。"""
        base = FPGRUConfig(vocab_size=16000, d_emb=1024, seq_len=16, n_layers=2)
        hidden = solve_gru_hidden(FP_MIXER_PAPER_PARAMS, base)
        total = estimate_gru_parameters(replace(base, hidden=hidden))
        ratio = abs(total - FP_MIXER_PAPER_PARAMS) / FP_MIXER_PAPER_PARAMS
        assert ratio <= PARAM_BUDGET_TOLERANCE, \
            f'hidden={hidden}: 総パラメータ {total:,} が {ratio:.1%} 乖離'

    def test_solver_is_near_optimal(self):
        """探索解の左右1点より誤差が小さい(=最適近傍)。"""
        base = FPGRUConfig(vocab_size=16000, d_emb=1024, seq_len=16, n_layers=2)
        hidden = solve_gru_hidden(FP_MIXER_PAPER_PARAMS, base)

        def err(h):
            return abs(estimate_gru_parameters(replace(base, hidden=h))
                       - FP_MIXER_PAPER_PARAMS)

        assert err(hidden) <= err(hidden - 1)
        assert err(hidden) <= err(hidden + 1)


class TestConfigValidation:
    def test_invalid_layers(self):
        with pytest.raises(ValueError, match='n_layers'):
            FPGRUConfig(n_layers=0)

    def test_invalid_hidden(self):
        with pytest.raises(ValueError, match='hidden'):
            FPGRUConfig(hidden=0)


class TestGradientFlow:
    def test_all_parameters_receive_gradient(self):
        cfg = _tiny()
        model = _build(cfg)
        model.train()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=11)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=12)
        model(src, tgt).sum().backward()
        missing = [n for n, p in model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not missing, f'勾配が届かない/非有限なパラメータ: {missing}'
