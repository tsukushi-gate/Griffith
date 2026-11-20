"""TaskSpec/損失レジストリのテスト。"""

import math

import pytest
import torch

from griffith_logic.training.tasks import LOSSES, TASK_KINDS, make_task


class TestLosses:
    def test_ce_matches_torch(self):
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.0]])
        y = torch.tensor([0, 1])
        expected = torch.nn.functional.cross_entropy(logits, y)
        assert torch.allclose(LOSSES['ce'](logits, y), expected)

    def test_bce_handles_saturated_outputs(self):
        """LUTのsoft出力が丁度0/1に張り付いてもBCEが有限。"""
        out = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = LOSSES['bce'](out, y)
        assert math.isfinite(loss.item())

    def test_bce_numeric_case(self):
        out = torch.tensor([[0.8]])
        y = torch.tensor([[1.0]])
        expected = -math.log(0.8)
        assert abs(LOSSES['bce'](out, y).item() - expected) < 1e-5

    def test_mse(self):
        out = torch.tensor([[0.5, 1.0]])
        y = torch.tensor([[0.0, 1.0]])
        assert abs(LOSSES['mse'](out, y).item() - 0.125) < 1e-7


class TestTaskSpec:
    def test_classification_metrics(self):
        task = make_task('classification')
        out = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
        y = torch.tensor([0, 1, 1])  # 3問中2問正解
        m = task.metrics(out, y)
        assert abs(m['acc'] - 2 / 3) < 1e-6

    def test_binary_vector_metrics(self):
        task = make_task('binary_vector')
        # 2サンプル×2bit: 1サンプル目は全bit正解、2サンプル目は1bit不正解
        out = torch.tensor([[0.9, 0.1], [0.9, 0.9]])
        y = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        m = task.metrics(out, y)
        assert abs(m['acc'] - 3 / 4) < 1e-6          # bit accuracy
        assert abs(m['exact_match'] - 1 / 2) < 1e-6  # 全bit一致率

    def test_defaults(self):
        assert make_task('classification').loss_name == 'ce'
        assert make_task('binary_vector').loss_name == 'bce'
        assert make_task('binary_vector', 'mse').loss_name == 'mse'

    def test_invalid_combinations(self):
        with pytest.raises(ValueError):
            make_task('classification', 'bce')
        with pytest.raises(ValueError):
            make_task('binary_vector', 'ce')
        with pytest.raises(ValueError):
            make_task('seq2seq')  # 未対応kind

    def test_kinds_registry(self):
        assert set(TASK_KINDS) == {'classification', 'binary_vector'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
