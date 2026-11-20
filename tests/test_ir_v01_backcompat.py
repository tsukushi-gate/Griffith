"""IR v0.1 後方互換ガードレール。

tests/fixtures/ir_v01/ に凍結したv0.1 IR(+golden入出力)が、
今後のIR拡張(v0.2 DAG化等)でも無変更で load / validate / simulate /
build_torch でき、出力がビット単位で不変であることを固定する。

フィクスチャは freeze時点の simulate 出力を golden として持つ。
**このファイルとフィクスチャの更新は「意図的な互換性破壊」を意味する**ため、
更新時はROADMAP/PROGRESSに理由を記録すること。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from griffith_logic.ir import build_torch, ir_from_dict, ir_to_dict, simulate, validate

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'ir_v01'
FIXTURES = sorted(FIXTURE_DIR.glob('*.json'))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize('path', FIXTURES, ids=[p.stem for p in FIXTURES])
class TestIRv01Backcompat:
    def test_load_validate(self, path):
        doc = _load(path)
        graph = ir_from_dict(doc['ir'])
        validate(graph)  # 例外なし

    def test_dict_roundtrip_stable(self, path):
        """ir_from_dict→ir_to_dictがフィクスチャと完全一致(表現の安定性)。"""
        doc = _load(path)
        assert ir_to_dict(ir_from_dict(doc['ir'])) == doc['ir']

    def test_simulate_matches_golden(self, path):
        doc = _load(path)
        graph = ir_from_dict(doc['ir'])
        x = np.array(doc['golden_inputs'], dtype=np.uint8)
        y = np.asarray(simulate(graph, x))
        expected = np.array(doc['golden_outputs'])
        assert np.array_equal(y, expected), f'simulate出力がgoldenと不一致: {path.stem}'

    def test_build_torch_hard_eval_matches_golden(self, path):
        """PyTorch hard eval ⇔ simulate golden の等価性(3段等価性の第1段)。"""
        doc = _load(path)
        graph = ir_from_dict(doc['ir'])
        model = build_torch(graph)
        model.eval()
        x = torch.tensor(doc['golden_inputs'], dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        expected = np.array(doc['golden_outputs'], dtype=np.float64)
        np.testing.assert_allclose(
            y.numpy().astype(np.float64), expected, atol=1e-5,
            err_msg=f'hard eval出力がgoldenと不一致: {path.stem}',
        )


def test_fixtures_exist():
    """フィクスチャが誤って消されていないことの番犬。"""
    assert len(FIXTURES) >= 4, f'ir_v01フィクスチャが不足: {[p.name for p in FIXTURES]}'
