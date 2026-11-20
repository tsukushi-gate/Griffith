"""D3: 論理リザバーコンピューティング(LogicReservoir)のユニットテスト。

高速スイート内で数秒に収める。リザバーは凍結なので状態は一度だけ計算し、
readout のみを学習する(reservoir computing の本質)。
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models import LogicReservoir, LogicReservoirConfig


# ------------------------------------------------------------
# データ生成(1bit ストリームを width 本に複製して入力駆動を確保)
# ------------------------------------------------------------
def _make_delay(n, T, k, width, seed):
    """遅延ビット再生: 目標 = 時刻 T-1-k の入力ビット。"""
    g = torch.Generator().manual_seed(seed)
    bit = (torch.rand(n, T, 1, generator=g) > 0.5).float()
    return bit.repeat(1, 1, width), bit[:, T - 1 - k, 0].long()


def _perturb_early(xs, n_early, seed):
    """先頭 n_early ステップだけを別のランダムビットに差し替えた系列を返す。"""
    g = torch.Generator().manual_seed(seed)
    pert = xs.clone()
    b, _, d = xs.shape
    pert[:, :n_early, :] = (torch.rand(b, n_early, d, generator=g) > 0.5).float()
    return pert


# ============================================================
# 1. 凍結性: 学習後にリザバーの state_dict が bitwise 不変・readout のみ更新
# ============================================================
class TestFrozen:
    def test_reservoir_frozen_readout_updates(self, seed):
        cfg = LogicReservoirConfig(
            input_size=16, reservoir_size=128, num_classes=2,
            depth=2, leak=1.0, readout='group_sum', seed=3,
        )
        torch.manual_seed(0)
        model = LogicReservoir(cfg)

        # リザバーの凍結前スナップショット(bitwise 比較用)
        res_before = {k: v.clone() for k, v in model.reservoir.state_dict().items()}
        readout_before = [p.clone() for p in model.trainable_parameters()]

        xs, y = _make_delay(128, 4, 1, 16, seed=1)
        opt = torch.optim.Adam(model.trainable_parameters(), lr=0.05)
        for _ in range(30):
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xs), y).backward()
            opt.step()

        # リザバーは1ビットも変わらない
        res_after = model.reservoir.state_dict()
        for k, v in res_before.items():
            assert torch.equal(v, res_after[k]), f'リザバー {k} が変化した'

        # readout は更新されている
        readout_after = list(model.trainable_parameters())
        changed = any(
            not torch.equal(a, b)
            for a, b in zip(readout_before, readout_after)
        )
        assert changed, 'readout が更新されていない'

    def test_trainable_parameters_excludes_reservoir(self, seed):
        cfg = LogicReservoirConfig(
            input_size=8, reservoir_size=32, num_classes=2, readout='lut', seed=0,
        )
        model = LogicReservoir(cfg)
        trainable = set(id(p) for p in model.trainable_parameters())
        # リザバーのパラメータは trainable に含まれず、requires_grad=False
        for p in model.reservoir.parameters():
            assert id(p) not in trainable
            assert p.requires_grad is False
        # trainable_parameters は readout パラメータ全てと一致
        assert trainable == set(id(p) for p in model.readout.parameters())


# ============================================================
# 2. 再現性: 同一 seed で同一状態系列、異なる seed で異なる系列
# ============================================================
class TestReproducibility:
    def test_same_seed_same_states(self, seed):
        xs, _ = _make_delay(4, 6, 1, 8, seed=1)
        m1 = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=48, num_classes=2, seed=5))
        m2 = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=48, num_classes=2, seed=5))
        assert torch.equal(m1.states(xs), m2.states(xs))

    def test_different_seed_different_states(self, seed):
        xs, _ = _make_delay(4, 6, 1, 8, seed=1)
        m1 = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=48, num_classes=2, seed=5))
        m2 = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=48, num_classes=2, seed=6))
        assert not torch.equal(m1.states(xs), m2.states(xs))

    def test_seed_independent_of_ambient_rng(self, seed):
        """リザバー回路は config.seed のみに依存し、大域 RNG 状態から独立。"""
        xs, _ = _make_delay(4, 5, 1, 8, seed=1)
        torch.manual_seed(111)
        s_a = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=32, num_classes=2, seed=9)).states(xs)
        torch.manual_seed(222)
        s_b = LogicReservoir(LogicReservoirConfig(
            input_size=8, reservoir_size=32, num_classes=2, seed=9)).states(xs)
        assert torch.equal(s_a, s_b)


# ============================================================
# 3. 状態の離散性: leak=1.0 の states() は {0,1}
# ============================================================
class TestStateDiscreteness:
    def test_leak_one_states_are_binary(self, seed):
        cfg = LogicReservoirConfig(
            input_size=8, reservoir_size=64, num_classes=2, depth=3, leak=1.0, seed=2)
        m = LogicReservoir(cfg)
        xs, _ = _make_delay(6, 7, 1, 8, seed=1)
        st = m.states(xs)
        assert bool(((st == 0.0) | (st == 1.0)).all()), \
            'leak=1.0 の状態が {0,1} でない'

    def test_leaky_states_are_continuous(self, seed):
        """leak<1.0 では soft leaky 統合により状態が連続値になる(離散ではない)。"""
        cfg = LogicReservoirConfig(
            input_size=8, reservoir_size=64, num_classes=2, leak=0.5, seed=2)
        m = LogicReservoir(cfg)
        xs, _ = _make_delay(6, 7, 1, 8, seed=1)
        st = m.states(xs)
        assert not bool(((st == 0.0) | (st == 1.0)).all()), \
            'leak<1.0 でも状態が離散のまま(leaky 統合が効いていない)'


# ============================================================
# 4. fading memory(スモーク): 先頭だけ変えた2系列の状態差が T 増加で減衰
# ============================================================
class TestFadingMemory:
    def test_leaky_state_difference_decays(self, seed):
        # leak<1.0 の低域通過 + 離散ブール回路の縮約力で、先頭のみ異なる摂動が
        # 時間とともに減衰する(echo state property の定性チェック)。
        cfg = LogicReservoirConfig(
            input_size=4, reservoir_size=96, num_classes=2,
            depth=2, leak=0.5, seed=0)
        m = LogicReservoir(cfg)
        xs, _ = _make_delay(48, 24, 1, 4, seed=123)
        pert = _perturb_early(xs, n_early=3, seed=456)

        diff = (m.states(xs) - m.states(pert)).abs().mean(dim=(0, 2))  # (T,)
        early = diff[2:5].mean().item()   # 摂動直後の窓
        late = diff[18:24].mean().item()  # 十分先の窓
        # 実測: late/early ≈ 0.002〜0.17(seed 依存)。安全側に「半減以下」を要求。
        assert late < 0.5 * early, f'状態差が減衰しない: early={early}, late={late}'
        assert late < early, 'fading memory が成立していない'


# ============================================================
# 5. 収束(L4級): 遅延ビット再生を readout 学習のみで hard acc=100%
# ============================================================
class TestConvergence:
    def test_delayed_bit_recall_hard_acc(self, seed):
        # 遅延ビット再生(T=4, k=1)は凍結ランダム論理リザバー+線形 readout の
        # 学習のみで hard eval 100% に到達する(reservoir seed 非依存で実測確認済み)。
        # (時間パリティは記憶×XOR で reservoir 難タスクのため本テストは遅延再生を採用)
        cfg = LogicReservoirConfig(
            input_size=16, reservoir_size=128, num_classes=2,
            depth=2, leak=1.0, readout='group_sum', seed=0)
        torch.manual_seed(0)
        model = LogicReservoir(cfg)

        xtr, ytr = _make_delay(256, 4, 1, 16, seed=1)
        # リザバーは凍結 → 最終状態を一度だけ計算し readout を学習
        feat = model.states(xtr)[:, -1, :]
        opt = torch.optim.Adam(model.trainable_parameters(), lr=0.05)
        for _ in range(200):
            model.train()
            opt.zero_grad()
            F.cross_entropy(model.readout(feat), ytr).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model.readout(feat).argmax(-1) == ytr).float().mean().item()
        assert acc == 1.0, f'遅延ビット再生 hard acc={acc}'


# ============================================================
# 6. 形状・API
# ============================================================
class TestShapes:
    def test_forward_and_states_shapes(self, seed):
        cfg = LogicReservoirConfig(
            input_size=8, reservoir_size=40, num_classes=3, seed=0)
        m = LogicReservoir(cfg)
        xs, _ = _make_delay(5, 6, 1, 8, seed=1)
        assert m.states(xs).shape == (5, 6, 40)
        assert m(xs).shape == (5, 3)
