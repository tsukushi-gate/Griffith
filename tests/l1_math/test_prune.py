"""C1' 枝刈り部品(convert/prune.py)の数理性質テスト(')。

検証(合成・数秒):
    - fanin_mask: 各行の |W| 上位k本だけが1、非ゼロ数<=k
    - apply_fanin_pruning 後: 隠れ層の実効fan-in(max_fanin)が最終目標以下
    - 枝刈り済みモデルは transcribe_mlp(n_inputs=最終fanin) で厳密転写
      (小ネット全入力列挙で teacher argmax と simulate argmax が全一致)

厳密性の比較はビット/整数(argmax)一致で行う(許容誤差なし)。
"""

import numpy as np
import torch
from torch import nn

from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    fanin_mask,
    max_fanin,
)
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _all_binary_inputs(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


class _SteMLP(nn.Module):
    """隠れ活性 y=1[Wx+b>0](HeavisideSTE)の小型MLP(転写意味論に一致)。"""

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(x.shape[0], -1)
        for lin in self.linears[:-1]:
            h = HeavisideSTE.apply(lin(h))
        return self.linears[-1](h)


def _teacher_argmax(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
    return logits.argmax(dim=-1).numpy()


def _simulate_argmax(ir, x: np.ndarray) -> np.ndarray:
    return simulate(ir, x).argmax(axis=1)


# ============================================================
# fanin_mask
# ============================================================

class TestFaninMask:
    def test_keeps_top_k_by_magnitude(self):
        w = torch.tensor([[0.1, -0.9, 0.5, 0.05],
                          [1.0, 0.2, -0.3, 0.8]])
        mask = fanin_mask(w, 2)
        # 行0: |W|上位2 = idx1(0.9), idx2(0.5)
        assert mask[0].tolist() == [0.0, 1.0, 1.0, 0.0]
        # 行1: |W|上位2 = idx0(1.0), idx3(0.8)
        assert mask[1].tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_row_nnz_exactly_k(self):
        torch.manual_seed(0)
        w = torch.randn(7, 20)
        for k in (1, 3, 6):
            mask = fanin_mask(w, k)
            assert torch.all((mask != 0).sum(dim=1) == k)

    def test_k_clamped_to_in_dim(self):
        w = torch.randn(3, 4)
        mask = fanin_mask(w, 10)
        # k>in なら全接続
        assert torch.all((mask != 0).sum(dim=1) == 4)

    def test_mask_is_binary_same_shape(self):
        w = torch.randn(5, 8)
        mask = fanin_mask(w, 3)
        assert mask.shape == w.shape
        assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


# ============================================================
# max_fanin / 枝刈り後のfan-in保証
# ============================================================

class TestMaxFanin:
    def test_reports_max_nonzero_per_row(self):
        model = _SteMLP((4, 3, 2))
        with torch.no_grad():
            # 隠れ層(linears[0]): 行ごとに非ゼロ 2,3,1 → max=3
            model.linears[0].weight.copy_(torch.tensor([
                [1.0, 0.0, 2.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]))
        assert max_fanin(model) == 3

    def test_excludes_final_layer(self):
        # 最終層は密でもfan-inに数えない
        model = _SteMLP((6, 2, 5))
        with torch.no_grad():
            model.linears[0].weight.zero_()
            model.linears[0].weight[:, :2] = 1.0  # 隠れfan-in=2
        assert max_fanin(model) == 2


def _identity_train_fn(model, epochs):
    """枝刈りマスクのみを検証するダミーtrain(重み更新なし)。"""
    return {'epochs': epochs, 'noop': True}


class TestApplyPruning:
    def test_final_fanin_respected(self):
        model = _SteMLP((16, 8, 5))
        sched = PruneSchedule(fanins=(10, 6, 3), epochs_per_stage=1)
        hist = apply_fanin_pruning(
            model, sched, _identity_train_fn, lambda m: 0.0
        )
        # 段階記録・最終fan-in<=3
        assert [s['fanin'] for s in hist['stages']] == [10, 6, 3]
        assert hist['final_fanin'] == 3
        assert max_fanin(model) <= 3
        # 各段の max_fanin_after が単調に縮む
        mfs = [s['max_fanin_after'] for s in hist['stages']]
        assert mfs[0] <= 10 and mfs[1] <= 6 and mfs[2] <= 3

    def test_grad_mask_freezes_pruned_weights(self):
        """fine-tune中も枝刈り位置は0のまま(勾配hook+乗算)。"""
        model = _SteMLP((12, 6, 4))
        pruned_zero_positions = {}

        def train_fn(m, epochs):
            # 実際に1step更新して復活しないことを確認する
            opt = torch.optim.Adam(m.parameters(), lr=0.5)
            x = torch.randint(0, 2, (8, 12), dtype=torch.float32)
            y = torch.randint(0, 4, (8,))
            for _ in range(epochs):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(x), y)
                loss.backward()
                opt.step()
            return {'epochs': epochs}

        sched = PruneSchedule(fanins=(4,), epochs_per_stage=3)
        apply_fanin_pruning(model, sched, train_fn, lambda m: 0.0)
        # 隠れ層の各行の非ゼロ<=4(復活していない)
        assert max_fanin(model) <= 4


# ============================================================
# 厳密転写(核心の主張): 枝刈り済みモデル ⇔ transcribe simulate 全一致
# ============================================================

class TestExactTranscription:
    def _pruned_model(self, dims, final_fanin, seed=0):
        torch.manual_seed(seed)
        model = _SteMLP(dims).double()
        # doubleにするとtranscribe内部(float64)と数値挙動が完全一致し
        # 境界 preact==0 の判定ゆらぎを排除できる
        sched = PruneSchedule(fanins=(final_fanin,), epochs_per_stage=1)
        apply_fanin_pruning(model, sched, _identity_train_fn, lambda m: 0.0)
        return model.float()

    def test_pruned_mlp_transcribes_exactly(self):
        """小ネット全入力列挙で teacher argmax と simulate argmax が全一致。"""
        dims = (8, 6, 4)
        final_fanin = 6
        model = self._pruned_model(dims, final_fanin, seed=1)
        assert max_fanin(model) <= final_fanin

        x_all = _all_binary_inputs(dims[0])          # (256, 8) 全列挙
        spec = TranscribeSpec(n_inputs=final_fanin, weight_bits=8, isf=False)
        ir = transcribe_mlp(model, x_all, spec)
        assert not validate(ir)

        teacher_pred = _teacher_argmax(model, x_all)
        sim_pred = _simulate_argmax(ir, x_all)
        assert np.array_equal(teacher_pred, sim_pred), (
            f'厳密転写違反: 不一致 {(teacher_pred != sim_pred).sum()}/{len(x_all)}'
        )

    def test_multiple_seeds_exact(self):
        dims = (6, 5, 3)
        final_fanin = 4
        x_all = _all_binary_inputs(dims[0])
        for seed in range(4):
            model = self._pruned_model(dims, final_fanin, seed=seed)
            spec = TranscribeSpec(n_inputs=final_fanin, weight_bits=8)
            ir = transcribe_mlp(model, x_all, spec)
            assert not validate(ir)
            assert np.array_equal(
                _teacher_argmax(model, x_all), _simulate_argmax(ir, x_all)
            ), f'seed={seed} で厳密転写違反'
