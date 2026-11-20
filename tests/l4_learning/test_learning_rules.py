"""L4: 学習ルールプラグイン(HybridTrainer)の検証。

基準テスト: 「全層bp指定のHybridTrainer == 素のtorch訓練ループ」の
bitwise一致。これがルール切替機構自体の正しさの根拠になる。
"""

import copy
import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import (
    GroupSum,
    HybridTrainer,
    LearningSpec,
    LutLayer,
    with_learning,
)
from griffith_logic.eggroll import EGGROLLConfig

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _xor_data():
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    y = torch.tensor([0, 1, 1, 0])
    return x, y


def _xor_model(param='softmax', seed=3):
    torch.manual_seed(seed)
    return nn.Sequential(
        LutLayer(2, 16, n_inputs=2, parameterization=param),
        LutLayer(16, 16, n_inputs=2, parameterization=param),
        GroupSum(k=2, tau=2.0),
    )


class TestBPBitwiseIdentity:
    def test_all_bp_equals_plain_torch_loop(self):
        """全bp(注釈なし)のHybridTrainerが素のtorchループとbitwise一致"""
        x, y = _xor_data()

        model_a = _xor_model(seed=11)
        model_b = copy.deepcopy(model_a)

        opt_a = torch.optim.Adam(model_a.parameters(), lr=0.05)
        trainer = HybridTrainer(model_a, optimizer=opt_a)

        opt_b = torch.optim.Adam(model_b.parameters(), lr=0.05)

        for _ in range(5):
            trainer.train_step(x, y)

            model_b.train()
            opt_b.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model_b(x), y)
            loss.backward()
            opt_b.step()

        for (ka, va), (kb, vb) in zip(
            model_a.state_dict().items(), model_b.state_dict().items(),
        ):
            assert ka == kb
            assert torch.equal(va, vb), f'{ka} がbitwise不一致'

    def test_bp_annotation_is_equivalent_to_no_annotation(self):
        """明示的rule='bp'注釈と注釈なしが同一挙動"""
        x, y = _xor_data()
        model_a = _xor_model(seed=13)
        model_b = copy.deepcopy(model_a)
        for m in model_b:
            if isinstance(m, LutLayer):
                with_learning(m, LearningSpec(rule='bp'))

        ta = HybridTrainer(model_a, torch.optim.Adam(model_a.parameters(), lr=0.05))
        tb = HybridTrainer(model_b, torch.optim.Adam(model_b.parameters(), lr=0.05))
        for _ in range(3):
            ta.train_step(x, y)
            tb.train_step(x, y)
        for va, vb in zip(model_a.state_dict().values(),
                          model_b.state_dict().values()):
            assert torch.equal(va, vb)


class TestAPRule:
    def test_ap_delegation_learns_xor(self, seed):
        """rule='ap'注釈でAPTrainerに委譲され、XORが学習できる"""
        torch.manual_seed(5)
        model = nn.Sequential(
            with_learning(
                LutLayer(2, 32, n_inputs=2, parameterization='independent'),
                LearningSpec(rule='ap', lr=0.5),
            ),
            with_learning(
                LutLayer(32, 16, n_inputs=2, parameterization='independent'),
                LearningSpec(rule='ap', lr=0.5),
            ),
            GroupSum(k=2, tau=2.0),
        )
        trainer = HybridTrainer(model)
        assert trainer._ap is not None  # 委譲確認

        x, y = _xor_data()
        acc = 0.0
        for _ in range(300):
            metrics = trainer.train_step(x, y)
            acc = metrics['accuracy']
            if acc == 1.0:
                break
        assert acc == 1.0, f'AP法でXORが学習できない: acc={acc}'

    def test_ap_mixed_with_dfa_raises(self):
        model = nn.Sequential(
            with_learning(
                LutLayer(2, 8, parameterization='independent'),
                LearningSpec(rule='ap'),
            ),
            with_learning(LutLayer(8, 8), LearningSpec(rule='dfa')),
            GroupSum(k=2),
        )
        with pytest.raises(NotImplementedError):
            HybridTrainer(model)


class TestDFARule:
    def _dfa_model(self, seed=7):
        torch.manual_seed(seed)
        return nn.Sequential(
            with_learning(
                LutLayer(2, 32, n_inputs=2), LearningSpec(rule='dfa'),
            ),
            with_learning(
                LutLayer(32, 16, n_inputs=2), LearningSpec(rule='dfa'),
            ),
            GroupSum(k=2, tau=2.0),
        )

    def test_dfa_learns_xor(self, seed):
        model = self._dfa_model()
        trainer = HybridTrainer(
            model, torch.optim.Adam(model.parameters(), lr=0.05),
        )
        x, y = _xor_data()
        for _ in range(600):
            trainer.train_step(x, y)
        trainer.close()
        model.eval()
        with torch.no_grad():
            acc = (model(x).argmax(-1) == y).float().mean().item()
        assert acc == 1.0, f'DFAでXORが学習できない: hard acc={acc}'

    def test_dfa_blocks_global_backprop_into_layer(self, seed):
        """DFA層のparam.gradが大域BPではなく注入由来であること:
        フックによりlogitsがDFA層のパラメータに依存しない(局所グラフ分離)"""
        model = self._dfa_model()
        trainer = HybridTrainer(model)
        x, y = _xor_data()
        logits = model(x)
        # 全層DFAなのでlogitsは大域グラフから切断されている
        assert not logits.requires_grad
        trainer.close()

    def test_dfa_alignment_with_bp(self, seed):
        """アライメント検定: cos(ΔW_DFA, ΔW_BP) > 0 が学習中平均で成立
        (Lillicrap 2016: feedback alignmentの必要条件)"""
        model = self._dfa_model(seed=17)
        trainer = HybridTrainer(
            model, torch.optim.SGD(model.parameters(), lr=0.05),
        )
        x, y = _xor_data()

        cosines = []
        for _ in range(80):
            # BP勾配: 同一重みの独立クローン(フックなし・新規構築)で計算
            # (deepcopyはDFAフックごと複製してしまうため使わない)
            ref = self._dfa_model(seed=17)
            ref.load_state_dict(model.state_dict())
            ref_loss = F.cross_entropy(ref(x), y)
            ref_loss.backward()
            bp_grads = {
                name: p.grad.clone()
                for name, p in ref.named_parameters() if p.grad is not None
            }

            trainer.train_step(x, y)
            for name, p in model.named_parameters():
                if p.grad is None or name not in bp_grads:
                    continue
                c = F.cosine_similarity(
                    p.grad.flatten(), bp_grads[name].flatten(), dim=0,
                )
                cosines.append(c.item())

        trainer.close()
        mean_cos = sum(cosines) / len(cosines)
        assert mean_cos > 0.0, f'DFA-BPアライメントが負: {mean_cos:.4f}'


class TestEggrollRule:
    def test_eggroll_improves_xor_fitness(self, seed):
        """rule='eggroll'で勾配なしにXORのfitnessが改善する"""
        torch.manual_seed(23)
        cfg = EGGROLLConfig(
            rank=2, sigma=0.1, alpha=0.05, n_workers=64,
            antithetic=True, score_function='centered_rank',
        )
        model = nn.Sequential(
            with_learning(
                LutLayer(2, 16, n_inputs=2),
                LearningSpec(rule='eggroll', eggroll=cfg),
            ),
            GroupSum(k=2, tau=2.0),
        )
        trainer = HybridTrainer(model)
        # eggrollパラメータはautogradから除外されている
        assert all(not p.requires_grad for p in model[0].parameters())

        x, y = _xor_data()
        first_loss = None
        last = None
        for _ in range(150):
            last = trainer.train_step(x, y)
            if first_loss is None:
                first_loss = last['loss']
        assert last['loss'] < first_loss, (
            f'EGGROLLでlossが改善しない: {first_loss:.4f} → {last["loss"]:.4f}'
        )
        assert last['accuracy'] >= 0.75, f'acc={last["accuracy"]}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
