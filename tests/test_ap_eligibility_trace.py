"""適格度トレース付きRDDLGN APトレーナーのテスト。

中核はλ=0回帰: EligibilityRDDLGNAPTrainer(lambda=0, norm='none')が
基底RDDLGNAPTrainerと厳密一致すること(受け入れ基準1)。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_trainer_rddlgn import RDDLGNAPTrainer
from ap_trainer_rddlgn_eligibility import EligibilityRDDLGNAPTrainer

from griffith_logic import RDDLGN, RDDLGNConfig


def _tiny_config():
    return RDDLGNConfig(
        vocab_size=10,
        d_emb=16,
        seq_len=4,
        n_layer_sizes=[32, 32],
        k_layer_sizes=[48, 32],
        l_layer_sizes=[32, 32],
        p_layer_sizes=[48, 32],
        m_layer_sizes=[64, 64, 100],
        tau=1.0,
        n_inputs=2,
        parameterization='independent',
        connections='random',
    )


def _make_batch(batch, src_len, tgt_len, vocab=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(3, vocab, (batch, src_len), generator=g)
    bos = torch.full((batch, 1), 1, dtype=torch.long)
    eos = torch.full((batch, 1), 2, dtype=torch.long)
    tgt_tokens = torch.randint(3, vocab, (batch, tgt_len), generator=g)
    tgt_full = torch.cat([bos, tgt_tokens, eos], dim=1)
    return src, tgt_full


def _lut_weights(model):
    from griffith_logic import LutLayer
    return [
        m.weights.data.clone()
        for m in model.modules()
        if isinstance(m, LutLayer)
    ]


def _all_close_identical(model_a, model_b):
    """全LUT重み+全Embeddingが厳密一致か。"""
    from griffith_logic import LutLayer
    luts_a = [m.weights.data for m in model_a.modules() if isinstance(m, LutLayer)]
    luts_b = [m.weights.data for m in model_b.modules() if isinstance(m, LutLayer)]
    for wa, wb in zip(luts_a, luts_b):
        if not torch.equal(wa, wb):
            return False
    ea = model_a.src_embedding.embedding.weight.data
    eb = model_b.src_embedding.embedding.weight.data
    if not torch.equal(ea, eb):
        return False
    ea2 = model_a.tgt_embedding.embedding.weight.data
    eb2 = model_b.tgt_embedding.embedding.weight.data
    return torch.equal(ea2, eb2)


def _build_pair(update_mode=1, seed=123):
    """同一初期重みのbase/eligibilityトレーナー対を構築。"""
    cfg = _tiny_config()
    torch.manual_seed(seed)
    model_base = RDDLGN(cfg)
    trainer_base = RDDLGNAPTrainer(model_base, lr=0.1, update_mode=update_mode)
    torch.manual_seed(seed)
    model_elig = RDDLGN(cfg)
    trainer_elig = EligibilityRDDLGNAPTrainer(
        model_elig, lr=0.1, update_mode=update_mode,
        eligibility_lambda=0.0, trace_norm='none',
    )
    # 初期重みが厳密一致していることを前提として確認
    assert _all_close_identical(model_base, model_elig), '初期重みが不一致'
    return trainer_base, model_base, trainer_elig, model_elig


class TestLambdaZeroRegression:
    """λ=0で基底クラスと厳密一致(受け入れ基準1)。"""

    @pytest.mark.parametrize('update_mode', [1, 2])
    def test_single_step_identical(self, update_mode):
        tb, mb, te, me = _build_pair(update_mode=update_mode)
        src, tgt = _make_batch(4, 4, 4, seed=7)
        tb.train_step(src.clone(), tgt.clone())
        te.train_step(src.clone(), tgt.clone())
        assert _all_close_identical(mb, me), \
            f'λ=0が基底クラスと不一致 (update_mode={update_mode})'

    def test_multi_step_identical(self):
        tb, mb, te, me = _build_pair(update_mode=1)
        for step in range(5):
            src, tgt = _make_batch(4, 4, 4, seed=100 + step)
            tb.train_step(src.clone(), tgt.clone())
            te.train_step(src.clone(), tgt.clone())
            assert _all_close_identical(mb, me), f'step {step}で不一致'

    def test_one_minus_lambda_norm_zero_lambda_identical(self):
        """trace_norm='one_minus_lambda'もλ=0では(1-0)=1で一致。"""
        cfg = _tiny_config()
        torch.manual_seed(55)
        mb = RDDLGN(cfg)
        tb = RDDLGNAPTrainer(mb, lr=0.1)
        torch.manual_seed(55)
        me = RDDLGN(cfg)
        te = EligibilityRDDLGNAPTrainer(
            me, lr=0.1, eligibility_lambda=0.0,
            trace_norm='one_minus_lambda',
        )
        src, tgt = _make_batch(4, 4, 4, seed=9)
        tb.train_step(src.clone(), tgt.clone())
        te.train_step(src.clone(), tgt.clone())
        assert _all_close_identical(mb, me)


class TestLambdaEffect:
    """λ>0で更新が実際に変化する。"""

    def test_lambda_changes_updates(self):
        cfg = _tiny_config()
        torch.manual_seed(3)
        m0 = RDDLGN(cfg)
        t0 = EligibilityRDDLGNAPTrainer(m0, lr=0.1, eligibility_lambda=0.0)
        torch.manual_seed(3)
        m9 = RDDLGN(cfg)
        t9 = EligibilityRDDLGNAPTrainer(m9, lr=0.1, eligibility_lambda=0.9)
        # 系列長を長めにしてトレースの効果が出るようにする
        src, tgt = _make_batch(4, 8, 4, seed=11)
        t0.train_step(src.clone(), tgt.clone())
        t9.train_step(src.clone(), tgt.clone())
        # K-layers(エンコーダ再帰)の重みが異なるはず
        k0 = [l.weights.data for l in m0.encoder.k_layers.layers]
        k9 = [l.weights.data for l in m9.encoder.k_layers.layers]
        differs = any(not torch.equal(a, b) for a, b in zip(k0, k9))
        assert differs, 'λ=0.9でK-layers更新がλ=0と変わらない'


class TestMemoryBound:
    """トレースメモリがO(basis数)でTに非依存(Phase1-5)。"""

    def test_trace_numel_independent_of_seq_len(self):
        cfg = _tiny_config()
        torch.manual_seed(0)
        model = RDDLGN(cfg)
        trainer = EligibilityRDDLGNAPTrainer(
            model, lr=0.1, eligibility_lambda=0.9,
        )
        # 同一batchで系列長を変える → last_trace_numelは不変であるべき
        numels = []
        for src_len in (4, 8, 16):
            src, tgt = _make_batch(4, src_len, 4, seed=src_len)
            trainer.train_step(src.clone(), tgt.clone())
            numels.append(trainer.last_trace_numel)
        assert len(set(numels)) == 1, \
            f'トレースメモリが系列長に依存: {numels}'
        assert numels[0] > 0


class TestValidation:
    def test_rejects_bad_lambda(self):
        cfg = _tiny_config()
        model = RDDLGN(cfg)
        with pytest.raises(ValueError, match='eligibility_lambda'):
            EligibilityRDDLGNAPTrainer(model, eligibility_lambda=1.5)

    def test_rejects_bad_norm(self):
        cfg = _tiny_config()
        model = RDDLGN(cfg)
        with pytest.raises(ValueError, match='trace_norm'):
            EligibilityRDDLGNAPTrainer(model, trace_norm='bogus')

    @pytest.mark.parametrize('norm', ['none', 'one_minus_lambda', 'seq_len'])
    def test_all_norms_run(self, norm):
        cfg = _tiny_config()
        torch.manual_seed(0)
        model = RDDLGN(cfg)
        trainer = EligibilityRDDLGNAPTrainer(
            model, lr=0.1, eligibility_lambda=0.9, trace_norm=norm,
        )
        src, tgt = _make_batch(4, 6, 4, seed=1)
        metrics = trainer.train_step(src, tgt)
        assert 'loss' in metrics and 'accuracy' in metrics
