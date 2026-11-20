"""バタフライQAT転写の意味論テスト(L1)。

正しさの機械的保証:
    1. butterfly_pairs の配線性質(i XOR 2^level、対合、行あたり非ゼロ≤2)
    2. **転写bit厳密の全数掃引**: width≤8の全2^W入力で
       IR simulate == ButterflyMLP hard forward(等値assert)
    3. 転写後gate_idヒストグラムが16ゲート域内
    4. validate通過・n_inputs=2(native DLGN対応)
"""

import itertools

import numpy as np
import pytest
import torch

from griffith_logic.convert.butterfly_qat import (
    ButterflyMLP,
    butterfly_pairs,
    transcribe_butterfly,
)
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


class TestButterflyPairs:
    @pytest.mark.parametrize('width', [2, 4, 8, 16])
    def test_xor_wiring(self, width):
        for level in range(int(np.log2(width))):
            pairs = butterfly_pairs(width, level)
            stride = 1 << level
            assert (pairs[:, 0] == np.arange(width)).all()
            assert (pairs[:, 1] == np.arange(width) ^ stride).all()
            # 対合: partner(partner(i)) == i
            assert (pairs[pairs[:, 1], 1] == pairs[:, 0]).all()

    def test_row_nonzero_at_most_2(self):
        """各出力は高々2入力(構造でfan-in=2を保証)。"""
        pairs = butterfly_pairs(8, 1)
        assert pairs.shape == (8, 2)
        assert (pairs[:, 0] != pairs[:, 1]).all()

    def test_non_power_of_two_rejected(self):
        with pytest.raises(AssertionError):
            butterfly_pairs(6, 0)


class TestTranscribeBitExact:
    """②転写bit厳密の全数掃引(Phase 3の中核保証)。"""

    @pytest.mark.parametrize('seed', range(4))
    @pytest.mark.parametrize('in_width,n_blocks', [(6, 1), (8, 2), (5, 1)])
    def test_full_sweep(self, seed, in_width, n_blocks):
        torch.manual_seed(seed)
        model = ButterflyMLP(in_width=in_width, num_classes=2,
                             n_blocks=n_blocks).double()
        graph = transcribe_butterfly(model)
        validate(graph)
        X = np.array(list(itertools.product([0, 1], repeat=in_width)),
                     dtype=np.int64)
        scores_ir = simulate(graph, X)
        model.eval()
        with torch.no_grad():
            scores_t = model(torch.as_tensor(X, dtype=torch.float64))
        # GroupSumスコア自体の厳密一致(argmaxより強い保証)
        assert (scores_ir == scores_t.numpy().astype(np.int64)).all(), \
            '転写IRとhard forwardのスコア不一致(ロスレス転写破れ)'

    def test_gate_histogram_and_ninputs(self):
        torch.manual_seed(7)
        model = ButterflyMLP(in_width=8, num_classes=2).double()
        graph = transcribe_butterfly(model)
        hist = graph.metadata['butterfly']['gate_histogram']
        assert sum(hist.values()) == len(model.factors) * model.width
        assert all(0 <= int(k) < 16 for k in hist)
        for node in graph.nodes:
            if node.type == 'lut_layer':
                assert node.params['n_inputs'] == 2

    def test_transcribe_after_training_step(self):
        """勾配更新後(master重み変化後)も転写は厳密。"""
        torch.manual_seed(11)
        model = ButterflyMLP(in_width=6, num_classes=2).double()
        X = (torch.rand(64, 6) > 0.5).double()
        y = (X[:, 0].long() ^ X[:, 1].long())
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(10):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(X), y).backward()
            opt.step()
        graph = transcribe_butterfly(model)
        validate(graph)
        Xa = np.array(list(itertools.product([0, 1], repeat=6)),
                      dtype=np.int64)
        model.eval()
        with torch.no_grad():
            st = model(torch.as_tensor(Xa, dtype=torch.float64))
        assert (simulate(graph, Xa) == st.numpy().astype(np.int64)).all()
