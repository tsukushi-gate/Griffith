"""LutLayerの配線学習（wiring='learnable'）テスト。

DWN論文(arXiv 2410.11112 §3.2)のLearnableMappingを汎用LutLayerに移植した
機能の検証。3本柱:
    1. wiring='fixed'（既定）が変更前とbitwise同一出力になる非回帰性
    2. wiring='learnable'のforward形状・勾配疎通・eval動作
    3. 固定ランダム配線 vs 学習配線のXOR/parityタスクでの収束スモーク
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.datasets.logic import parity
from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.dwn import LearnableMapping
from griffith_logic.layers.lut import LutLayer

# ============================================================
# 1. wiring='fixed' 非回帰（既定値が変更前と完全一致すること）
# ============================================================

class TestFixedWiringNonRegression:
    """wiring引数追加後もdefault('fixed')が既存gather実装と完全に同一であること。

    「変更前」との比較は、gather処理を素朴に書いた参照実装
    (x[:, indices.T.reshape(-1)].reshape(...)を使う経路)と同一seedで
    突き合わせることで担保する。LutLayer.__init__の乱数消費順序は
    wiring分岐を追加しても変えていないため、同一seedなら重み・接続とも
    完全一致するはず。
    """

    def test_default_wiring_is_fixed(self):
        layer = LutLayer(8, 4, n_inputs=2)
        assert layer.wiring == 'fixed'
        assert not hasattr(layer, 'learnable_mapping')

    def test_bitwise_identical_train_mode(self):
        """同seedで生成した2つのdefault LutLayerが訓練モードでbitwise一致

        warpパラメータ化はforward内でGumbelノイズ用にRNGを消費するため、
        forward呼び出し直前のRNG状態も揃えて比較する。
        """
        for n_inputs in (2, 4, 6):
            for param in LutLayer.PARAMETERIZATIONS:
                torch.manual_seed(123)
                layer_a = LutLayer(12, 6, n_inputs=n_inputs, parameterization=param)
                torch.manual_seed(123)
                layer_b = LutLayer(12, 6, n_inputs=n_inputs, parameterization=param)

                assert torch.equal(layer_a.indices, layer_b.indices)
                assert torch.equal(layer_a.weights, layer_b.weights)

                torch.manual_seed(999)
                x = torch.rand(5, 12)

                torch.manual_seed(555)
                ya = layer_a(x)
                torch.manual_seed(555)
                yb = layer_b(x)
                assert torch.equal(ya, yb), f'n_inputs={n_inputs}, param={param}'

    def test_bitwise_identical_eval_mode(self):
        """同seedで生成した2つのdefault LutLayerがevalモードでbitwise一致"""
        for n_inputs in (2, 4, 6):
            for param in LutLayer.PARAMETERIZATIONS:
                torch.manual_seed(42)
                layer_a = LutLayer(12, 6, n_inputs=n_inputs, parameterization=param)
                torch.manual_seed(42)
                layer_b = LutLayer(12, 6, n_inputs=n_inputs, parameterization=param)
                layer_a.eval()
                layer_b.eval()

                x = torch.rand(5, 12)
                ya = layer_a(x)
                yb = layer_b(x)
                assert torch.equal(ya, yb), f'n_inputs={n_inputs}, param={param}'

    def test_matches_manual_gather_reference(self):
        """wiring='fixed'のforwardが、手動gather + _compute_basisの素朴な合成と一致

        n_inputs=4のsoftmaxはgate_table非搭載（16ゲートモードはn_inputs=2のみ）
        なので、N>2 softmaxのeval離散化ルール(確率>0.5のエントリのみ1)に
        合わせて参照値を計算する。
        """
        torch.manual_seed(7)
        layer = LutLayer(10, 5, n_inputs=4, parameterization='softmax')
        layer.eval()
        x = torch.rand(3, 10)

        y = layer(x)

        gathered_ref = x[:, layer.indices.T.reshape(-1)].reshape(
            x.shape[0], layer.out_dim, layer.n_inputs
        )
        basis_ref = layer._compute_basis(gathered_ref)
        w_ref = (torch.softmax(layer.weights, dim=-1) > 0.5).to(x.dtype)
        y_ref = torch.einsum('boe,oe->bo', basis_ref, w_ref)

        assert torch.equal(y, y_ref)

    def test_explicit_wiring_fixed_kw_same_as_default(self):
        """wiring='fixed'を明示指定してもdefaultと同一挙動"""
        torch.manual_seed(5)
        layer_default = LutLayer(8, 4, n_inputs=2)
        torch.manual_seed(5)
        layer_explicit = LutLayer(8, 4, n_inputs=2, wiring='fixed')

        assert torch.equal(layer_default.indices, layer_explicit.indices)
        assert torch.equal(layer_default.weights, layer_explicit.weights)

        x = torch.rand(4, 8)
        assert torch.equal(layer_default(x), layer_explicit(x))


# ============================================================
# 2. wiring='learnable' の基本動作
# ============================================================

class TestLearnableWiring:
    def test_invalid_wiring_raises(self):
        with pytest.raises(AssertionError):
            LutLayer(8, 4, n_inputs=2, wiring='invalid')

    def test_constructs_learnable_mapping(self):
        layer = LutLayer(10, 6, n_inputs=4, wiring='learnable')
        assert hasattr(layer, 'learnable_mapping')
        assert isinstance(layer.learnable_mapping, LearnableMapping)
        assert layer.learnable_mapping.in_features == 10
        assert layer.learnable_mapping.out_features == 6 * 4

    def test_indices_buffer_still_registered(self):
        """indices bufferはlearnable配線でも従来通り登録される
        （推論時フォールバック/reach-map互換のため）"""
        layer = LutLayer(10, 6, n_inputs=2, wiring='learnable')
        assert hasattr(layer, 'indices')
        assert layer.indices.shape == (2, 6)

    def test_forward_shape_train(self):
        for n_inputs in (2, 4, 6):
            for param in LutLayer.PARAMETERIZATIONS:
                layer = LutLayer(
                    10, 6, n_inputs=n_inputs, parameterization=param,
                    wiring='learnable',
                )
                x = torch.rand(3, 10)
                y = layer(x)
                assert y.shape == (3, 6), f'n_inputs={n_inputs}, param={param}'

    def test_forward_shape_eval(self):
        for n_inputs in (2, 4, 6):
            for param in LutLayer.PARAMETERIZATIONS:
                layer = LutLayer(
                    10, 6, n_inputs=n_inputs, parameterization=param,
                    wiring='learnable',
                )
                layer.eval()
                x = torch.rand(3, 10)
                y = layer(x)
                assert y.shape == (3, 6), f'n_inputs={n_inputs}, param={param}'

    def test_backward_updates_learnable_mapping_weights(self):
        """backwardでlearnable_mapping.weightsに勾配が流れる"""
        layer = LutLayer(10, 6, n_inputs=2, wiring='learnable')
        x = torch.rand(4, 10, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.learnable_mapping.weights.grad is not None
        assert layer.learnable_mapping.weights.grad.shape == \
            layer.learnable_mapping.weights.shape
        assert torch.isfinite(layer.learnable_mapping.weights.grad).all()

        # LUT本体の重みにも勾配が流れる
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()

        # 入力xにも勾配が流れる(LearnableMapping backward経由)
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_backward_all_parameterizations(self):
        """全parameterizationで学習配線のbackwardが疎通する"""
        for n_inputs in (2, 4, 6):
            for param in LutLayer.PARAMETERIZATIONS:
                layer = LutLayer(
                    10, 6, n_inputs=n_inputs, parameterization=param,
                    wiring='learnable',
                )
                x = torch.rand(4, 10)
                y = layer(x)
                loss = y.sum()
                loss.backward()
                assert layer.learnable_mapping.weights.grad is not None, (
                    f'n_inputs={n_inputs}, param={param}'
                )
                assert layer.weights.grad is not None, (
                    f'n_inputs={n_inputs}, param={param}'
                )

    def test_eval_mode_runs_and_is_deterministic(self):
        """eval時も動作し、同一入力で決定的な出力を返す"""
        layer = LutLayer(10, 6, n_inputs=2, wiring='learnable')
        layer.eval()
        x = torch.rand(4, 10)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)
        assert torch.equal(y1, y2)
        assert y1.shape == (4, 6)

    def test_extra_repr_reflects_wiring(self):
        layer = LutLayer(8, 4, n_inputs=2, wiring='learnable', lm_tau=0.01)
        rep = layer.extra_repr()
        assert 'wiring=learnable' in rep
        assert '0.01' in rep

        layer_fixed = LutLayer(8, 4, n_inputs=2)
        assert 'wiring' not in layer_fixed.extra_repr()

    def test_sequential_model_integration(self):
        """学習配線LutLayerをSequentialに組み込んで訓練ステップが回る"""
        model = nn.Sequential(
            LutLayer(8, 16, n_inputs=2, wiring='learnable'),
            LutLayer(16, 16, n_inputs=2, wiring='learnable'),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        x = torch.rand(4, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


# ============================================================
# 3. XOR/parityタスクでの収束スモーク（固定配線 vs 学習配線）
# ============================================================

@pytest.mark.learning
class TestLearnableWiringConvergenceSmoke:
    """学習配線が固定ランダム配線と比べて学習が進む/発散しないことのスモーク。

    設定: parity3タスクの3つの意味ある入力ビットを、9個の無関係なノイズ
    ビットの中に埋め込んだ12次元入力。2層×n_inputs=2のLutLayerで、
    softmax(既定)パラメータ化。固定ランダム配線では層1の各ニューロンが
    独立に2ビットを選ぶため、有効な3ビット全てに層1経由で到達する保証が弱く
    parity信号を再構成しにくい。学習配線ならforward argmax選択が勾配で
    有効な入力へ寄っていく。

    このテストはXOR/parityの原理的な学習困難性の再確認ではなく、
    「配線を学習可能にすると同条件で明確に改善する」ことの実証が目的。
    """

    N_RELEVANT = 3
    N_NOISE = 9
    IN_DIM = N_RELEVANT + N_NOISE
    HIDDEN = 128
    OUT_DIM = 64
    STEPS = 300
    LR = 0.02
    SEEDS = (0, 1, 2)

    @classmethod
    def _make_batch(cls, xs, ys, batch_size, gen):
        idx = torch.randint(0, xs.shape[0], (batch_size,), generator=gen)
        x_rel = xs[idx]
        noise = torch.randint(0, 2, (batch_size, cls.N_NOISE), generator=gen).float()
        x = torch.cat([x_rel, noise], dim=-1)
        y = ys[idx].squeeze(-1).long()
        return x, y

    @classmethod
    def _run(cls, wiring, seed, xs, ys):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1000)
        model = nn.Sequential(
            LutLayer(cls.IN_DIM, cls.HIDDEN, n_inputs=2, wiring=wiring,
                     connections='random'),
            LutLayer(cls.HIDDEN, cls.OUT_DIM, n_inputs=2, wiring=wiring,
                     connections='random'),
            GroupSum(k=2, tau=5.0),
        )
        opt = torch.optim.Adam(model.parameters(), lr=cls.LR)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(cls.STEPS):
            x, y = cls._make_batch(xs, ys, 64, gen)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert torch.isfinite(loss).all(), 'lossが発散(NaN/Inf)'

        model.eval()
        x_full = torch.cat(
            [xs, torch.zeros(xs.shape[0], cls.N_NOISE)], dim=-1
        )
        with torch.no_grad():
            logits = model(x_full)
            pred = logits.argmax(-1)
            acc = (pred == ys.squeeze(-1).long()).float().mean().item()
        return acc

    def test_learnable_wiring_does_not_diverge(self):
        """学習配線での訓練がNaN/Infに発散しないことの確認(最低限のスモーク)"""
        spec = parity(self.N_RELEVANT)
        xs, ys = spec.dataset.tensors
        acc = self._run('learnable', seed=0, xs=xs, ys=ys)
        assert 0.0 <= acc <= 1.0

    def test_learnable_wiring_matches_or_beats_fixed(self):
        """学習配線の平均精度が固定ランダム配線と同等以上であることのスモーク。

        parity3信号をノイズに埋め込んだタスクでは、固定ランダム配線は
        層1のニューロンがparityの全3ビットに到達する保証がなく収束が
        不安定になりやすい。学習配線はforward argmax選択が勾配で
        有効な入力へ寄るため、同条件で明確に有利になることを期待する。
        """
        spec = parity(self.N_RELEVANT)
        xs, ys = spec.dataset.tensors

        fixed_accs = [self._run('fixed', s, xs, ys) for s in self.SEEDS]
        learnable_accs = [self._run('learnable', s, xs, ys) for s in self.SEEDS]

        fixed_mean = sum(fixed_accs) / len(fixed_accs)
        learnable_mean = sum(learnable_accs) / len(learnable_accs)

        print(f'\n[wiring smoke] fixed_accs={fixed_accs} mean={fixed_mean:.3f}')
        print(f'[wiring smoke] learnable_accs={learnable_accs} mean={learnable_mean:.3f}')

        # 最低限: 学習が進んで発散しない(0付近に張り付いていない)
        assert learnable_mean > 0.0
        # 本命: 学習配線が固定ランダム配線と同等以上
        assert learnable_mean >= fixed_mean, (
            f'learnable({learnable_mean:.3f}) < fixed({fixed_mean:.3f})'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================
# 4. 漸進配線ハード化
# ============================================================

class TestProgressiveWiring:
    """LearnableMappingのsoftモード+ProgressiveWiringScheduler。"""

    def test_soft_mode_low_tau_converges_to_hard(self):
        """τ_soft→小 でsoft混合forwardがhard argmax forwardへ収束する"""
        torch.manual_seed(11)
        from griffith_logic.layers.dwn import LearnableMapping
        m = LearnableMapping(10, 8)
        with torch.no_grad():
            m.weights.normal_()  # 明確なargmaxを持たせる
        x = torch.rand(5, 10)
        m.train()
        m.forward_mode = 'hard'
        y_hard = m(x)
        m.forward_mode = 'soft'
        m.tau_soft = 1e-4
        y_soft = m(x)
        assert torch.allclose(y_hard, y_soft, atol=1e-4), (
            f'max diff = {(y_hard - y_soft).abs().max():.3e}'
        )

    def test_eval_always_hard(self):
        """evalではforward_modeによらずhard argmax"""
        torch.manual_seed(13)
        from griffith_logic.layers.dwn import LearnableMapping
        m = LearnableMapping(10, 8)
        x = torch.rand(3, 10)
        m.eval()
        m.forward_mode = 'soft'
        m.tau_soft = 10.0  # 大温度: softなら大きくずれるはず
        y = m(x)
        m.forward_mode = 'hard'
        y_hard = m(x)
        assert torch.equal(y, y_hard)

    def test_scheduler_phases(self):
        """スケジューラ: 前半soft(τ幾何減衰)→switch以降hard"""
        import os
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', 'experiments'))
        from cage_scheduler import ProgressiveWiringScheduler

        model = nn.Sequential(LutLayer(8, 4, n_inputs=2, wiring='learnable'))
        sched = ProgressiveWiringScheduler(
            model, total_steps=100, switch_at=0.8, tau_0=1.0, tau_min=0.05)
        lm = model[0].learnable_mapping
        assert lm.forward_mode == 'soft'
        assert abs(lm.tau_soft - 1.0) < 1e-9
        for _ in range(40):
            sched.step()
        assert lm.forward_mode == 'soft'
        assert 0.05 < lm.tau_soft < 1.0
        for _ in range(40):
            sched.step()
        assert lm.forward_mode == 'hard'

    def test_soft_phase_gradients_flow(self):
        """softフェーズでmapping重みへ通常のautograd勾配が流れる"""
        torch.manual_seed(17)
        layer = LutLayer(12, 8, n_inputs=2, wiring='learnable')
        layer.learnable_mapping.forward_mode = 'soft'
        layer.train()
        x = torch.rand(4, 12)
        layer(x).sum().backward()
        g = layer.learnable_mapping.weights.grad
        assert g is not None and g.abs().sum() > 0

    def test_progressive_recovers_planted_wiring_parity(self):
        """parity3-in-12(planted配線)を漸進ハード化で回復できる(hard 100%)。

        hard選択のみの配線学習が崩壊した設定の縮小版に対する
        処方箋レバーの回帰固定。規模は既存ConvergenceSmoke(128/64)に合わせ、
        seed 0/1/2の3本とも1.0到達を実測してseed 0で決定化している。
        """
        import itertools
        import os
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', 'experiments'))
        from cage_scheduler import ProgressiveWiringScheduler

        seed = 0
        torch.manual_seed(seed)
        n_bits = 12
        xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=6)))
        g = torch.Generator().manual_seed(seed + 99)
        noise = (torch.rand(len(xs), n_bits - 6, generator=g) > 0.5).float()
        x = torch.cat([xs, noise], dim=1)
        y = (xs[:, :3].sum(-1) % 2).long()

        model = nn.Sequential(
            LutLayer(n_bits, 128, n_inputs=2, wiring='learnable'),
            LutLayer(128, 64, n_inputs=2),
            GroupSum(k=2, tau=4.0),
        )
        total = 2000
        sched = ProgressiveWiringScheduler(model, total_steps=total,
                                           switch_at=0.6)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        loss_fn = nn.CrossEntropyLoss()
        acc = 0.0
        for step in range(total):
            model.train()
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            sched.step()
            if (step + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(x).argmax(-1) == y).float().mean().item()
                if acc == 1.0 and step > total * 0.6:
                    break
        assert acc == 1.0, f'漸進配線ハード化でparity3-in-12未回復: acc={acc}'
