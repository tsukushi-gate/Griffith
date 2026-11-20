"""統合: LogicSeqMixerLM(/ decoder-only Mixer 文字LM)のモデル級契約。

固定する契約:
    - forward契約: (B, T) -> (B, T, vocab_size) の logits 単体(tupleではない)
    - **因果性(最重要)**: eval時、位置 t0 より後ろのトークンを任意に変えても
      logits[:, :t0+1] が**ビット厳密に**不変(全 t0 を掃引)。
      positional / residual_mode / gate の全組み合わせで確認する
    - hard離散性: eval時、埋め込み出力と各LutLayer出力が厳密に {0,1}
    - train/eval分岐: 出力が異なる(離散化が実際に効いている)
    - ゲート数整合: count_gates()['total'] == estimate_gates_lm()['total']
                    == Σ num_neurons
    - 生成: generate が context_len を超えても落ちず、長さが伸びる
    - bpc定義: bits_per_character(x) == x / ln(2)
    - 設定検証: n_blocks超過 / xor残差×residual_init / 出力層のvocab_size非整除
"""

import math
import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import assert_all_finite

from griffith_logic.lut_layer import LutLayer
from griffith_logic.models.logic_seq_mixer_lm import (
    LogicSeqMixerLM,
    LogicSeqMixerLMConfig,
    bits_per_character,
    estimate_gates_lm,
)
from griffith_logic.models.rddlgn import LogicEmbedding

# 決定的評価のためGumbelノイズを切ったtiny構成
_VOCAB = 12


def _tiny(**overrides) -> LogicSeqMixerLMConfig:
    base = LogicSeqMixerLMConfig(
        vocab_size=_VOCAB,
        d_emb=16,
        context_len=8,
        d_channels=24,
        channel_hidden=48,
        n_blocks=2,
        channel_depth=1,
        token_kernels=(2, 3, 4),
        token_depth=2,
        m_layer_sizes=[36, _VOCAB * 2],
        n_pos_bits=8,
        selection='soft',
        gumbel=False,
    )
    return replace(base, **overrides) if overrides else base


def _tokens(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


def _build(cfg: LogicSeqMixerLMConfig, seed: int = 42) -> LogicSeqMixerLM:
    torch.manual_seed(seed)
    return LogicSeqMixerLM(cfg)


class TestForwardContract:
    """forwardの戻り値・形状契約。"""

    @pytest.mark.parametrize('batch', [1, 2, 3])
    def test_returns_logits_of_expected_shape(self, batch):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        x = _tokens(batch, cfg.context_len, cfg.vocab_size, seed=1)

        logits = model(x)
        assert isinstance(logits, torch.Tensor), \
            f'forwardはTensorを返す契約: {type(logits)}'
        assert logits.shape == (batch, cfg.context_len, cfg.vocab_size), \
            f'logits形状契約違反: {tuple(logits.shape)}'
        assert_all_finite(logits, 'logits')

    @pytest.mark.parametrize('length', [1, 3, 8])
    def test_accepts_variable_length(self, length):
        """context_len以下の任意長を受け付ける(生成時にprefixが伸びるため)。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        x = _tokens(2, length, cfg.vocab_size, seed=7)
        assert model(x).shape == (2, length, cfg.vocab_size)

    def test_rejects_non_2d_input(self):
        model = _build(_tiny())
        with pytest.raises(ValueError, match='2次元'):
            model(torch.zeros(2, 3, 4, dtype=torch.long))

    def test_aux_loss_is_scalar(self):
        cfg = _tiny()
        model = _build(cfg)
        model(_tokens(2, cfg.context_len, cfg.vocab_size, seed=3))
        aux = model.pop_aux_loss()
        assert isinstance(aux, torch.Tensor) and aux.dim() == 0
        assert_all_finite(aux, 'aux_loss')


class TestCausality:
    """因果性(最重要): 位置 t0 の出力は t0 より後ろのトークンに依存しない。"""

    @staticmethod
    def _assert_causal(model, cfg, seed_a: int, seed_b: int) -> None:
        model.eval()
        batch, length = 2, cfg.context_len
        x = _tokens(batch, length, cfg.vocab_size, seed=seed_a)
        alt = _tokens(batch, length, cfg.vocab_size, seed=seed_b)
        base = model(x)

        for t0 in range(length):
            perturbed = x.clone()
            # t0 より後ろを全部差し替える(最終位置は差し替え対象が無いので
            # 恒等になり、その場合はスキップ)
            if t0 + 1 >= length:
                continue
            perturbed[:, t0 + 1:] = alt[:, t0 + 1:]
            assert not torch.equal(perturbed[:, t0 + 1:], x[:, t0 + 1:]) \
                or torch.equal(alt[:, t0 + 1:], x[:, t0 + 1:]), \
                'テスト前提: 未来トークンが実際に変化していること'
            out = model(perturbed)
            assert torch.equal(out[:, :t0 + 1], base[:, :t0 + 1]), (
                f'因果性違反: t0={t0} で未来トークンを変えたら過去の出力が変化した '
                f'(最大差={float((out[:, :t0 + 1] - base[:, :t0 + 1]).abs().max()):.3e})'
            )

    def test_causal_default(self):
        cfg = _tiny()
        self._assert_causal(_build(cfg), cfg, 11, 12)

    @pytest.mark.parametrize('positional', ['learned_bits', 'none'])
    @pytest.mark.parametrize('residual_mode', ['xor', 'none'])
    def test_causal_all_structural_variants(self, positional, residual_mode):
        cfg = _tiny(positional=positional, residual_mode=residual_mode)
        self._assert_causal(_build(cfg), cfg, 21, 22)

    @pytest.mark.parametrize('gate', ['and', 'none'])
    def test_causal_all_gates(self, gate):
        cfg = _tiny(gate=gate)
        self._assert_causal(_build(cfg), cfg, 31, 32)

    @pytest.mark.parametrize('n_blocks', [1, 3])
    def test_causal_all_depths(self, n_blocks):
        cfg = _tiny(n_blocks=n_blocks)
        self._assert_causal(_build(cfg), cfg, 41, 42)

    def test_causal_in_train_mode_too(self):
        """soft(train)モードでも因果性は構造的に成り立つ。

        gumbel=False・selection='soft' なので train でも決定的に評価できる。
        """
        cfg = _tiny()
        model = _build(cfg)
        model.train()
        batch, length = 2, cfg.context_len
        x = _tokens(batch, length, cfg.vocab_size, seed=51)
        alt = _tokens(batch, length, cfg.vocab_size, seed=52)
        base = model(x)
        for t0 in range(length - 1):
            perturbed = x.clone()
            perturbed[:, t0 + 1:] = alt[:, t0 + 1:]
            out = model(perturbed)
            assert torch.equal(out[:, :t0 + 1], base[:, :t0 + 1]), \
                f'train(soft)モードで因果性違反: t0={t0}'

    def test_prefix_consistency(self):
        """短いprefixを与えた出力が、長い系列の同じprefix部分と一致する。

        因果性の系(自己回帰生成が首尾一貫であることの実体)。
        """
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        x = _tokens(2, cfg.context_len, cfg.vocab_size, seed=61)
        full = model(x)
        for t in range(1, cfg.context_len + 1):
            part = model(x[:, :t])
            assert torch.equal(part, full[:, :t]), \
                f'prefix長{t}の出力が全系列のprefix部分と一致しない'


class TestHardDiscreteness:
    """eval時、埋め込みと全LutLayer出力が厳密に {0,1} になる(collapsed可能性)。"""

    def test_all_activations_binary_in_eval(self):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()

        seen: list[torch.Tensor] = []

        def hook(_module, _inp, out):
            seen.append(out.detach())

        handles = [
            m.register_forward_hook(hook)
            for m in model.modules() if isinstance(m, LutLayer | LogicEmbedding)
        ]
        try:
            model(_tokens(2, cfg.context_len, cfg.vocab_size, seed=71))
        finally:
            for h in handles:
                h.remove()

        assert seen, 'LutLayer/LogicEmbeddingの出力を1つも捕捉できていない'
        for i, out in enumerate(seen):
            is_binary = ((out == 0.0) | (out == 1.0)).all()
            assert is_binary, (
                f'eval時に非二値の出力: モジュール#{i} '
                f'min={float(out.min())} max={float(out.max())} '
                f'(厳密に0/1であること)'
            )

    def test_position_bits_binary_in_eval(self):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        bits = model.pos(cfg.context_len)
        assert ((bits == 0.0) | (bits == 1.0)).all(), \
            'eval時の位置ビットが二値でない'

    def test_train_and_eval_differ(self):
        """離散化が実際に効いている(soft/hardで出力が異なる)。"""
        cfg = _tiny()
        model = _build(cfg)
        x = _tokens(2, cfg.context_len, cfg.vocab_size, seed=81)
        model.train()
        soft = model(x)
        model.eval()
        hard = model(x)
        assert not torch.allclose(soft, hard), \
            'train/evalで出力が同一 — 離散化が効いていない'


class TestGateCount:
    """ゲート数(LUTニューロン数)の整合。"""

    def test_count_matches_estimate_and_module_scan(self):
        cfg = _tiny()
        model = _build(cfg)
        counted = model.count_gates()
        estimated = estimate_gates_lm(cfg)
        scanned = sum(
            int(m.num_neurons) for m in model.modules()
            if hasattr(m, 'num_neurons')
        )
        assert counted == estimated, \
            f'count_gates と estimate_gates_lm が不一致: {counted} != {estimated}'
        assert counted['total'] == scanned, \
            f'total({counted["total"]}) が全モジュール走査({scanned})と不一致'

    def test_default_config_matches_rddlgn_charlm_budget(self):
        """既定構成が RDDLGN char-LM(1,954ゲート)の ±30% に収まる。

        1,954 は experiments/train_rddlgn_charlm.py の既定
        (N=[256,256], K=[512,448], M=[222,260])の実測ゲート数。
        """
        total = estimate_gates_lm(LogicSeqMixerLMConfig())['total']
        baseline = 1954
        ratio = total / baseline
        assert 0.7 <= ratio <= 1.3, \
            f'既定構成のゲート数 {total} がベースライン {baseline} の±30%外 (比{ratio:.3f})'


class TestGenerate:
    """自己回帰生成の契約。"""

    def test_generate_extends_sequence(self):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        idx = torch.zeros(1, 1, dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5)
        assert out.shape == (1, 6)
        assert (out >= 0).all() and (out < cfg.vocab_size).all()

    def test_generate_beyond_context_len(self):
        """context_lenを超えて生成しても落ちない(左から切り詰める)。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        idx = torch.zeros(1, 1, dtype=torch.long)
        n_new = cfg.context_len + 5
        out = model.generate(idx, max_new_tokens=n_new)
        assert out.shape == (1, 1 + n_new)


class TestBpc:
    """bpc(bits per character)の定義。"""

    @pytest.mark.parametrize('nats', [0.0, 1.0, 4.7266, 4.1744])
    def test_bpc_is_nats_over_ln2(self, nats):
        assert bits_per_character(nats) == pytest.approx(nats / math.log(2.0))


class TestConfigValidation:
    """設定の値域検証。"""

    def test_rejects_too_many_blocks(self):
        with pytest.raises(ValueError, match='n_blocks'):
            _tiny(n_blocks=5)

    def test_rejects_zero_blocks(self):
        with pytest.raises(ValueError, match='n_blocks'):
            _tiny(n_blocks=0)

    def test_rejects_xor_with_residual_init(self):
        with pytest.raises(ValueError, match='residual_init'):
            _tiny(residual_mode='xor', residual_init=True)

    def test_rejects_head_not_divisible_by_vocab(self):
        with pytest.raises(ValueError, match='割り切れる'):
            _tiny(m_layer_sizes=[36, _VOCAB * 2 + 1])

    def test_rejects_channels_not_divisible_by_kernels(self):
        with pytest.raises(ValueError, match='割り切れる'):
            _tiny(d_channels=25)
