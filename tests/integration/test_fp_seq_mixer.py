"""統合: FPSeqMixerのモデル級契約。

論理版 `tests/integration/test_logic_seq_mixer.py` と対になるテスト。
固定する契約:
    - forward契約: logits (B, T, vocab_size) を単体で返す
      (論理版の第2要素 emb_reg_loss は float 版に存在しない)
    - **因果性(最重要)**: tgt_tokens の位置t0より後ろを任意に変えても
      logits[:, :t0+1] がビット厳密に不変(全t0掃引)。cross_attention の
      両アーム × context_mode(broadcast/gather/concat_all)で確認
    - パラメータ予算: count_parameters()['total'] == estimate_parameters(config)
      かつ from_paper() が論理版 40,847,232 の ±30% 以内
    - 設定検証: ブロック数超過 / 不正な列挙値 が ValueError
    - アーム: cross_attention='single_head' で cross モジュールが生えて
      パラメータが増え、'none' では存在しない
"""

import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.fp_seq_mixer import (
    FPSeqMixer,
    FPSeqMixerConfig,
    ToeplitzTokenMixFP,
    estimate_parameters,
)

# 論理版 LogicSeqMixerConfig.from_paper() の総パラメータ実測値(比較の基準)
LOGIC_PAPER_PARAMS = 40_847_232
PARAM_BUDGET_TOLERANCE = 0.30


def _tiny(**overrides) -> FPSeqMixerConfig:
    base = FPSeqMixerConfig.tiny()
    return replace(base, **overrides) if overrides else base


def _tokens(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


def _build(cfg: FPSeqMixerConfig, seed: int = 42) -> FPSeqMixer:
    torch.manual_seed(seed)
    return FPSeqMixer(cfg)


class TestForwardContract:
    """forwardの戻り値・形状契約。"""

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

    @pytest.mark.parametrize('context_mode', ['broadcast', 'gather', 'concat_all'])
    def test_forward_contract_all_context_modes(self, context_mode):
        cfg = _tiny(context_mode=context_mode)
        model = _build(cfg)
        model.eval()
        # teacher forcing の実長 = seq_len + 1([BOS, tgt, EOS][:-1])
        t_len = cfg.seq_len + 1
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=15)
        tgt = _tokens(2, t_len, cfg.vocab_size, seed=16)
        logits = model(src, tgt)
        assert logits.shape == (2, t_len, cfg.vocab_size), \
            f'context_mode={context_mode} で形状契約違反: {tuple(logits.shape)}'

    def test_variable_src_and_tgt_lengths(self):
        """S と T を独立に変えても形状契約を満たす。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        lengths = sorted({max(1, cfg.seq_len - 1), cfg.seq_len})
        for s_len in lengths:
            for t_len in lengths:
                src = _tokens(2, s_len, cfg.vocab_size, seed=3)
                tgt = _tokens(2, t_len, cfg.vocab_size, seed=4)
                logits = model(src, tgt)
                assert logits.shape == (2, t_len, cfg.vocab_size), \
                    f'S={s_len}, T={t_len} で形状契約違反: {tuple(logits.shape)}'

    def test_return_hidden(self):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=7)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=8)
        logits, hidden = model(src, tgt, return_hidden=True)
        assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
        assert len(hidden) == 2
        assert hidden[0].shape == (2, cfg.seq_len, cfg.d_model)
        assert hidden[1].shape == (2, cfg.seq_len, cfg.d_model)


class TestCausality:
    """★最重要: デコーダの因果性(未来トークンが過去のlogitsに漏れない)。"""

    def _assert_causal(self, model: FPSeqMixer) -> None:
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
            with torch.no_grad():
                mod_logits = model(src, tgt_mod)

            assert torch.equal(base_logits[:, :t0 + 1], mod_logits[:, :t0 + 1]), \
                (f'因果性違反: t0={t0} より後ろのトークン変更が '
                 f'logits[:, :{t0 + 1}] に影響している')
            assert not torch.equal(tgt, tgt_mod), '未来トークンが変化していない'

    def test_causal_arm_i(self):
        """アーム(i): cross_attention='none' で因果性が成立する。"""
        self._assert_causal(_build(_tiny(cross_attention='none')))

    def test_causal_arm_ii(self):
        """アーム(ii): single_head cross-attention でも因果性が成立する。"""
        self._assert_causal(_build(_tiny(cross_attention='single_head')))

    @pytest.mark.parametrize('context_mode', ['broadcast', 'gather', 'concat_all'])
    def test_causal_all_context_modes(self, context_mode):
        self._assert_causal(_build(_tiny(context_mode=context_mode)))

    def test_encoder_block_is_not_causal(self):
        """encoder(causal=False)は未来を見る。因果テストが自明に通っていない証拠。"""
        mix = ToeplitzTokenMixFP(8, 6, (3, 3), depth=2, causal=False)
        torch.manual_seed(0)
        x = torch.randn(1, 8, 6)
        y_base = mix(x)
        x_mod = x.clone()
        x_mod[:, 5:] += 1.0
        y_mod = mix(x_mod)
        # 位置4は非因果窓(K=3の中心揃え)で位置5を見るため必ず変化する
        assert not torch.equal(y_base[:, 4], y_mod[:, 4])

    def test_causal_window_shape_and_padding(self):
        """causal窓は [t-K+1, t] を切り出す(左詰めゼロパディング)。"""
        mix = ToeplitzTokenMixFP(4, 3, (3,), depth=2, causal=True)
        x = torch.arange(4.0).view(1, 4, 1).expand(1, 4, 3).contiguous()
        win = mix._window(x, 3)
        assert win.shape == (1, 4, 3, 3)
        # t=0 は [0,0,x0]、t=2 は [x0,x1,x2]
        assert torch.equal(win[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.equal(win[0, 2, 0], torch.tensor([0.0, 1.0, 2.0]))


class TestParameterBudget:
    """パラメータ数の解析式と実測の一致・論理版との予算整合。"""

    @pytest.mark.parametrize('cfg_name', ['tiny', 'from_paper'])
    @pytest.mark.parametrize('cross', ['none', 'single_head'])
    @pytest.mark.parametrize('norm', ['layer', 'none'])
    def test_estimate_matches_actual(self, cfg_name, cross, norm):
        base = getattr(FPSeqMixerConfig, cfg_name)()
        cfg = replace(base, cross_attention=cross, norm=norm)
        if cfg_name == 'from_paper':
            # 実体化せず解析式のみ検証すると意味がないので tiny 相当へ縮める
            cfg = replace(cfg, vocab_size=200, d_emb=32, d_model=24,
                          channel_hidden=48, n_pos=8)
        model = _build(cfg)
        assert model.count_parameters()['total'] == estimate_parameters(cfg)

    def test_paper_config_within_logic_budget(self):
        """from_paper() の総パラメータが論理版 40,847,232 の ±30% 以内。"""
        for cross in ('none', 'single_head'):
            cfg = replace(FPSeqMixerConfig.from_paper(), cross_attention=cross)
            total = estimate_parameters(cfg)
            ratio = abs(total - LOGIC_PAPER_PARAMS) / LOGIC_PAPER_PARAMS
            assert ratio <= PARAM_BUDGET_TOLERANCE, \
                (f'cross={cross}: 総パラメータ {total:,} が論理版 '
                 f'{LOGIC_PAPER_PARAMS:,} から {ratio:.1%} 乖離')

    def test_count_parameters_has_total(self):
        model = _build(_tiny())
        counts = model.count_parameters()
        assert 'total' in counts
        assert counts['total'] == sum(p.numel() for p in model.parameters())


class TestArms:
    """cross アームの有無。"""

    def test_arm_i_has_no_cross_module(self):
        model = _build(_tiny(cross_attention='none'))
        assert model.decoder.cross_attn is None

    def test_arm_ii_adds_parameters(self):
        p_i = estimate_parameters(_tiny(cross_attention='none'))
        p_ii = estimate_parameters(_tiny(cross_attention='single_head'))
        assert p_ii > p_i


class TestConfigValidation:
    """不正な設定は ValueError。"""

    def test_too_many_blocks(self):
        with pytest.raises(ValueError, match='4以下'):
            FPSeqMixerConfig(n_enc_blocks=3, n_dec_blocks=3)

    def test_invalid_context_mode(self):
        with pytest.raises(ValueError, match='context_mode'):
            FPSeqMixerConfig(context_mode='bogus')

    def test_invalid_cross_attention(self):
        with pytest.raises(ValueError, match='cross_attention'):
            FPSeqMixerConfig(cross_attention='logic_single_head')

    def test_token_width_not_divisible(self):
        with pytest.raises(ValueError, match='割り切れる'):
            FPSeqMixerConfig(d_model=10, channel_hidden=20, token_kernels=(3, 8, 16))


class TestGradientFlow:
    """全パラメータに勾配が流れる(配線ミスの検出)。"""

    def test_all_parameters_receive_gradient(self):
        cfg = _tiny(cross_attention='single_head')
        model = _build(cfg)
        model.train()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=11)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=12)
        logits = model(src, tgt)
        logits.sum().backward()
        missing = [n for n, p in model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        # 埋め込みは使われたIDの行にのみ勾配が入る(grad自体は存在する)
        assert not missing, f'勾配が届かない/非有限なパラメータ: {missing}'
