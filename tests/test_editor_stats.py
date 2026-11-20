"""統計API(tools/editor/backend/stats.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する
(STATE注入はstats.configure()。main.pyへの統合は別途)。
"""

import json
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.export.estimate import estimate_resources
from griffith_logic.ir import ir_from_dict, ir_to_dict
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from tools.editor.backend import stats as stats_api

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f'{name}.json')) as f:
        return json.load(f)


class FakeState:
    """main.AppState互換の最小状態(main.py非依存)"""

    def __init__(self):
        self.ir = None


@pytest.fixture
def ctx():
    state = FakeState()
    app = FastAPI()
    app.include_router(stats_api.configure(state))
    return state, TestClient(app, base_url='http://localhost:8765')


def _blackbox_ir_dict() -> dict:
    """blackboxノードを含む最小IR(x(2)→blackbox(2,2)→y(2))"""
    graph = IRGraph(
        metadata={'name': 'fx_blackbox'},
        inputs=[IRPort(id='x', width=2)],
        outputs=[IRPort(id='y', width=2)],
        nodes=[
            IRNode(
                id='n1_blackbox',
                type='blackbox',
                params={
                    'torch_class': 'torch.nn.Identity',
                    'config': {},
                    'in_width': 2,
                    'out_width': 2,
                },
            ),
        ],
        edges=[
            IREdge(src='x', dst='n1_blackbox'),
            IREdge(src='n1_blackbox', dst='y'),
        ],
    )
    return ir_to_dict(graph)


class TestStatsEndpointMatchesEstimate:
    def test_xor_matches_estimate_resources(self, ctx):
        """xorサンプルIRでestimate_resources直呼びと一致する"""
        state, client = ctx
        fx = _load_fixture('xor')
        state.ir = fx['ir']
        expected = estimate_resources(ir_from_dict(fx['ir']))

        res = client.get('/api/stats')
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['total_luts'] == expected['total_luts']
        assert {int(k): v for k, v in body['lut_counts'].items()} == \
            expected['lut_counts']
        assert body['logic_depth'] == expected['logic_depth']
        assert body['wire_count'] == expected['wire_count']
        assert body['group_sums'] == expected['group_sums']
        assert body['node_counts'] == expected['node_counts']
        assert body['lowered'] == expected['lowered']
        assert body['partial'] is False
        # n1_lutlayer: n_inputs=2, out_width=1 → 2^2 * 1 = 4
        assert body['num_params'] == 4

    def test_demo_mnist_conv_lowered_matches_estimate(self, ctx):
        """conv2d/or_pool2d含みIRでもlowering後のestimate_resourcesと一致する"""
        state, client = ctx
        fx = _load_fixture('demo_mnist')
        state.ir = fx['ir']
        expected = estimate_resources(ir_from_dict(fx['ir']))

        res = client.get('/api/stats')
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['total_luts'] == expected['total_luts']
        assert body['logic_depth'] == expected['logic_depth']
        assert body['wire_count'] == expected['wire_count']
        assert body['lowered'] == expected['lowered']
        assert body['partial'] is False
        assert body['num_params'] > 0


class TestStatsEndpointBlackbox:
    def test_blackbox_returns_partial_with_computable_values(self, ctx):
        """blackbox含有IRはpartial=trueで算出可能な部分値を返す"""
        state, client = ctx
        state.ir = _blackbox_ir_dict()

        res = client.get('/api/stats')
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['partial'] is True
        assert body['node_counts'] == {'blackbox': 1}
        assert body['total_luts'] == 0
        assert body['num_params'] == 0
        # wire_count = 入力幅2 + blackbox出力幅2
        assert body['wire_count'] == 4


class TestStatsEndpointErrors:
    def test_no_ir_404(self, ctx):
        _, client = ctx
        res = client.get('/api/stats')
        assert res.status_code == 404
        assert 'IR未設定' in res.json()['detail']

    def test_invalid_ir_422(self, ctx):
        state, client = ctx
        fx = _load_fixture('xor')
        ir = fx['ir']
        # 存在しないノードへのエッジを追加して構造検証を破綻させる
        ir = json.loads(json.dumps(ir))
        ir['edges'].append({'src': 'n1_lutlayer', 'dst': 'nonexistent_node'})
        state.ir = ir

        res = client.get('/api/stats')
        assert res.status_code == 422
        assert 'IRが不正' in res.json()['detail']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
