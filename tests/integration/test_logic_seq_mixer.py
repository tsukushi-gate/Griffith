"""統合: LogicSeqMixerのモデル級契約。

固定する契約:
    - forward契約: (logits (B,T,vocab_size), emb_reg_loss 0次元Tensor) の2-tuple
    - **因果性(最重要)**: eval時、tgt_tokensの位置t0より後ろを任意に変えても
      logits[:, :t0+1] がビット厳密に不変(全t0掃引)。cross_attentionの
      両アーム × context_mode(broadcast/gather/concat_all)で確認
    - hard離散性: eval時、埋め込み出力と各LutLayer出力が厳密に {0,1}
    - train/eval分岐: 出力が異なる(離散化が実際に効いている)
    - ゲート数整合: count_gates()['total'] == Σ num_neurons
    - 設定検証: ブロック数超過 / xor残差×residual_init / M最終層の
      vocab_size非整除 が ValueError
    - アーム(ii): cross_attention='logic_single_head' でも2-tuple契約と
      因果性が保たれ、ゲート数がアーム(i)より増える。'none'では
      crossモジュールが存在しない
"""

import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import assert_all_finite

from griffith_logic.lut_layer import LutLayer
from griffith_logic.models.logic_seq_mixer import (
    LogicSeqMixer,
    LogicSeqMixerConfig,
)


def _tiny(**overrides) -> LogicSeqMixerConfig:
    """決定的評価のためGumbelノイズを切ったtiny構成(上書き可)。"""
    base = replace(LogicSeqMixerConfig.tiny(), selection='soft', gumbel=False)
    return replace(base, **overrides) if overrides else base


def _tokens(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


def _build(cfg: LogicSeqMixerConfig, seed: int = 42) -> LogicSeqMixer:
    torch.manual_seed(seed)
    return LogicSeqMixer(cfg)


class TestForwardContract:
    """forwardの戻り値・形状契約。"""

    @pytest.mark.parametrize('batch', [1, 2, 3])
    def test_returns_logits_and_scalar_reg(self, batch, seed):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        src = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=1)
        tgt = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=2)

        out = model(src, tgt)
        assert isinstance(out, tuple) and len(out) == 2, \
            f'forwardは2-tupleを返す契約: {type(out)}'
        logits, emb_reg = out
        assert logits.shape == (batch, cfg.seq_len, cfg.vocab_size), \
            f'logits形状契約違反: {tuple(logits.shape)}'
        assert isinstance(emb_reg, torch.Tensor) and emb_reg.dim() == 0, \
            f'emb_reg_lossは0次元Tensor: {emb_reg}'
        assert_all_finite(logits, 'logits')
        assert_all_finite(emb_reg, 'emb_reg_loss')

    @pytest.mark.parametrize('context_mode', ['broadcast', 'gather', 'concat_all'])
    def test_forward_contract_all_context_modes(self, context_mode, seed):
        """context_mode(broadcast/gather/concat_all)いずれでも2-tuple・形状契約。"""
        cfg = _tiny(context_mode=context_mode)
        model = _build(cfg)
        model.eval()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=13)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=14)

        out = model(src, tgt)
        assert isinstance(out, tuple) and len(out) == 2, \
            f'context_mode={context_mode} で2-tuple契約違反: {type(out)}'
        logits, emb_reg = out
        assert logits.shape == (2, cfg.seq_len, cfg.vocab_size), \
            f'context_mode={context_mode} で形状契約違反: {tuple(logits.shape)}'
        assert emb_reg.dim() == 0
        assert_all_finite(logits, f'logits(context_mode={context_mode})')

    @pytest.mark.parametrize('context_mode', ['broadcast', 'gather', 'concat_all'])
    def test_shorter_tgt_than_src_all_context_modes(self, context_mode, seed):
        """S != T(デコード長がソース長と異なる)でも形状契約を満たす。"""
        cfg = _tiny(context_mode=context_mode)
        model = _build(cfg)
        model.eval()
        t_len = max(1, cfg.seq_len - 2)
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=15)
        tgt = _tokens(2, t_len, cfg.vocab_size, seed=16)
        logits, _ = model(src, tgt)
        assert logits.shape == (2, t_len, cfg.vocab_size), \
            f'context_mode={context_mode} で形状契約違反: {tuple(logits.shape)}'

    def test_variable_src_and_tgt_lengths(self, seed):
        """S(ソース長)とT(ターゲット長)を独立に変えても形状契約を満たす。"""
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        lengths = sorted({max(1, cfg.seq_len - 1), cfg.seq_len})
        for s_len in lengths:
            for t_len in lengths:
                src = _tokens(2, s_len, cfg.vocab_size, seed=3)
                tgt = _tokens(2, t_len, cfg.vocab_size, seed=4)
                logits, _ = model(src, tgt)
                assert logits.shape == (2, t_len, cfg.vocab_size), \
                    f'S={s_len}, T={t_len} で形状契約違反: {tuple(logits.shape)}'


class TestCausality:
    """★最重要: デコーダの因果性(未来トークンが過去のlogitsに漏れない)。"""

    def _assert_causal(self, model: LogicSeqMixer) -> None:
        cfg = model.config
        model.eval()
        batch, t_len = 2, cfg.seq_len
        src = _tokens(batch, cfg.seq_len, cfg.vocab_size, seed=5)
        tgt = _tokens(batch, t_len, cfg.vocab_size, seed=6)

        with torch.no_grad():
            base_logits, _ = model(src, tgt)

        for t0 in range(t_len):
            n_future = t_len - t0 - 1
            if n_future == 0:
                continue
            # t0より後ろのトークンを別の値へ総取っ替え(必ず変わるようずらす)
            tgt_mod = tgt.clone()
            tgt_mod[:, t0 + 1:] = (tgt[:, t0 + 1:] + 1) % cfg.vocab_size
            with torch.no_grad():
                mod_logits, _ = model(src, tgt_mod)

            assert torch.equal(base_logits[:, :t0 + 1], mod_logits[:, :t0 + 1]), \
                (f'因果性違反: t0={t0} より後ろのトークン変更が '
                 f'logits[:, :{t0 + 1}] に影響している')
            assert not torch.equal(tgt, tgt_mod), '未来トークンが変化していない'

    def test_causal_arm_i(self, seed):
        """アーム(i): cross_attention='none' で因果性が成立する。"""
        self._assert_causal(_build(_tiny(cross_attention='none')))

    def test_causal_arm_ii(self, seed):
        """アーム(ii): logic_single_head cross-attentionでも因果性が成立する。"""
        self._assert_causal(
            _build(_tiny(cross_attention='logic_single_head')))

    @pytest.mark.parametrize('context_mode', ['broadcast', 'gather', 'concat_all'])
    def test_causal_all_context_modes(self, context_mode, seed):
        """context_modeを変えても因果性が成立する(contextはsrc由来で
        tgtの未来には依存しないため、どのモードでも壊れてはいけない)。"""
        self._assert_causal(_build(_tiny(context_mode=context_mode)))


class TestHardDiscreteness:
    """eval時、論理域の中間表現が厳密に {0,1} であること。"""

    def test_embedding_and_lut_outputs_are_binary(self, seed):
        cfg = _tiny()
        model = _build(cfg)
        model.eval()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=7)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=8)

        # 埋め込み出力
        with torch.no_grad():
            emb = model.src_embedding(src)
        assert torch.all((emb == 0.0) | (emb == 1.0)), \
            'eval時のLogicEmbedding出力が {0,1} でない'

        # LutLayer出力(forward hookで収集)
        lut_outputs: list[torch.Tensor] = []

        def hook(module, inputs, output):
            lut_outputs.append(output.detach())

        handles = [m.register_forward_hook(hook)
                   for m in model.modules() if isinstance(m, LutLayer)]
        assert handles, 'モデルにLutLayerが1つも無い'
        with torch.no_grad():
            model(src, tgt)
        for h in handles:
            h.remove()

        assert lut_outputs, 'LutLayerが1度も実行されていない'
        for i, y in enumerate(lut_outputs):
            assert torch.all((y == 0.0) | (y == 1.0)), \
                f'eval時の中間LutLayer出力[{i}]が {{0,1}} でない'

    def test_train_and_eval_outputs_differ(self, seed):
        """離散化が実際に効いている(train出力とeval出力が一致しない)。"""
        cfg = _tiny()
        model = _build(cfg)
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=9)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=10)

        model.train()
        with torch.no_grad():
            train_logits, _ = model(src, tgt)
        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(src, tgt)

        assert not torch.allclose(train_logits, eval_logits), \
            'train/evalで出力が同一(離散化が効いていない)'


class TestGateCount:
    """ゲート数の集計整合。"""

    def test_count_gates_total_matches_num_neurons(self, seed):
        model = _build(_tiny())
        expected = sum(
            m.num_neurons for m in model.modules() if hasattr(m, 'num_neurons')
        )
        counts = model.count_gates()
        assert 'total' in counts, "count_gates()に'total'キーが無い"
        assert counts['total'] == expected, \
            f"count_gates()['total']={counts['total']} != Σnum_neurons={expected}"

    def test_count_parameters_has_total(self, seed):
        model = _build(_tiny())
        counts = model.count_parameters()
        assert 'total' in counts, "count_parameters()に'total'キーが無い"
        assert counts['total'] == sum(p.numel() for p in model.parameters()), \
            'count_parameters()のtotalが実パラメータ数と一致しない'


class TestConfigValidation:
    """設定の不正組み合わせがValueErrorになること。"""

    def test_too_many_blocks(self):
        with pytest.raises(ValueError):
            replace(LogicSeqMixerConfig.tiny(), n_enc_blocks=3, n_dec_blocks=3)

    def test_xor_residual_with_residual_init(self):
        with pytest.raises(ValueError):
            replace(LogicSeqMixerConfig.tiny(),
                    residual_mode='xor', residual_init=True)

    def test_m_last_layer_not_divisible_by_vocab(self):
        base = LogicSeqMixerConfig.tiny()
        sizes = list(base.m_layer_sizes)
        sizes[-1] = base.vocab_size * 2 + 1
        with pytest.raises(ValueError):
            replace(base, m_layer_sizes=sizes)


class TestCrossAttentionArms:
    """アーム(i) cross_attention='none' / アーム(ii) 'logic_single_head'。"""

    def test_arm_ii_adds_gates(self, seed):
        model_i = _build(_tiny(cross_attention='none'))
        model_ii = _build(_tiny(cross_attention='logic_single_head'))
        total_i = model_i.count_gates()['total']
        total_ii = model_ii.count_gates()['total']
        assert total_ii > total_i, \
            (f"cross_attention='logic_single_head' でゲート数が増えていない: "
             f'{total_ii} <= {total_i}')

    def test_arm_i_has_no_cross_module(self, seed):
        model_i = _build(_tiny(cross_attention='none'))
        cross_modules = [
            name for name, _ in model_i.named_modules() if 'cross' in name
        ]
        assert not cross_modules, \
            f"cross_attention='none' なのにcrossモジュールが存在: {cross_modules}"

    def test_arm_ii_forward_contract(self, seed):
        cfg = _tiny(cross_attention='logic_single_head')
        model = _build(cfg)
        model.eval()
        src = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=11)
        tgt = _tokens(2, cfg.seq_len, cfg.vocab_size, seed=12)

        out = model(src, tgt)
        assert isinstance(out, tuple) and len(out) == 2, \
            f'アーム(ii)でも2-tuple契約: {type(out)}'
        logits, emb_reg = out
        assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
        assert emb_reg.dim() == 0
        assert_all_finite(logits, 'logits(arm ii)')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
