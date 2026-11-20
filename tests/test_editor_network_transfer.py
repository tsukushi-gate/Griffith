"""重み転写API(tools/editor/backend/network_transfer.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する
(このAPIはSTATEを使わない純関数的APIのためconfigure不要)。
"""

import os
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.ir import ir_from_dict, ir_to_dict, simulate, validate
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from tools.editor.backend import network_transfer as nt_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(nt_api.router)
    return TestClient(app, base_url='http://localhost:8765')


def _lut_ir(name: str, truth_table_hex: str, indices=None) -> dict:
    """x(2)→lut_layer(2,1)→y(1) の最小IR。"""
    if indices is None:
        indices = [[0], [1]]
    graph = IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=2)],
        outputs=[IRPort(id='y', width=1)],
        nodes=[
            IRNode(
                id='n1_lutlayer',
                type='lut_layer',
                params={
                    'in_width': 2,
                    'out_width': 1,
                    'n_inputs': 2,
                    'indices': indices,
                    'truth_tables': [truth_table_hex],
                },
                train_info={'parameterization': 'softmax', 'selection': 'soft'},
            ),
        ],
        edges=[
            IREdge(src='x', dst='n1_lutlayer'),
            IREdge(src='n1_lutlayer', dst='y'),
        ],
    )
    return ir_to_dict(graph)


class TestTransferMatchesSimulate:
    """(a) 同名レイヤ転写→simulate出力がビット一致"""

    def test_transfer_then_simulate_matches_source(self, client):
        # source: XOR(0x6), target: 別の真理値表(AND=0x8)で初期化しておく
        source_ir = _lut_ir('fx_source', '0x6')
        target_ir = _lut_ir('fx_target', '0x8')

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': target_ir},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['transferred'] == ['n1_lutlayer']
        assert body['skipped'] == []

        target_graph = ir_from_dict(body['target_ir'])
        source_graph = ir_from_dict(source_ir)

        xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        out_transferred = simulate(target_graph, xs)
        out_source = simulate(source_graph, xs)
        assert np.array_equal(out_transferred, out_source)


class TestTransferWidthMismatch:
    """(b) 幅不一致で422+日本語エラー"""

    def test_out_width_mismatch_returns_422_japanese(self, client):
        source_ir = _lut_ir('fx_source', '0x6')
        # targetのlut_layerをout_width=2(2出力)に変える(幅不一致)
        target_ir = _lut_ir('fx_target', '0x8')
        target_ir['nodes'][0]['params']['out_width'] = 2
        target_ir['nodes'][0]['params']['truth_tables'] = ['0x8', '0x6']
        target_ir['nodes'][0]['params']['indices'] = [[0], [1]]
        target_ir['outputs'][0]['width'] = 2

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': target_ir},
        )
        assert res.status_code == 422, res.text
        detail = res.json()['detail']
        assert '幅不一致' in detail
        assert 'n1_lutlayer' in detail


class TestTransferSkipsUnmatched:
    """(c) 名前不一致ノードはスキップされ内訳に載る"""

    def test_missing_source_node_is_skipped(self, client):
        source_ir = _lut_ir('fx_source', '0x6')
        # sourceに存在しないidをtargetに使う
        target_ir = _lut_ir('fx_target', '0x8')
        target_ir['nodes'][0]['id'] = 'n2_other_lutlayer'
        target_ir['edges'][0]['dst'] = 'n2_other_lutlayer'
        target_ir['edges'][1]['src'] = 'n2_other_lutlayer'

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': target_ir},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['transferred'] == []
        assert len(body['skipped']) == 1
        assert body['skipped'][0]['id'] == 'n2_other_lutlayer'
        assert 'sourceに同名ノードが存在しない' in body['skipped'][0]['reason']

    def test_type_mismatch_node_is_skipped(self, client):
        source_ir = _lut_ir('fx_source', '0x6')
        # sourceの同名ノードを非対象型(blackbox)に差し替える(型不一致)
        source_ir['nodes'][0]['type'] = 'blackbox'
        source_ir['nodes'][0]['params'] = {
            'torch_class': 'torch.nn.Identity',
            'config': {},
            'in_width': 2,
            'out_width': 1,
        }
        target_ir = _lut_ir('fx_target', '0x8')

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': target_ir},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['transferred'] == []
        assert len(body['skipped']) == 1
        assert body['skipped'][0]['id'] == 'n1_lutlayer'
        assert '型不一致' in body['skipped'][0]['reason']


class TestTransferredIrValidates:
    """(d) 転写後IRがvalidate通過"""

    def test_target_ir_validates_after_transfer(self, client):
        source_ir = _lut_ir('fx_source', '0x6')
        target_ir = _lut_ir('fx_target', '0x8')

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': target_ir},
        )
        assert res.status_code == 200, res.text
        body = res.json()

        target_graph = ir_from_dict(body['target_ir'])
        # 例外が出ないことを確認する
        validate(target_graph)


class TestTransferInvalidIr:
    """不正なIRを渡した場合は422になる(呼び出し側でのIR検証エラー捕捉)"""

    def test_broken_ir_format_returns_422(self, client):
        source_ir = _lut_ir('fx_source', '0x6')
        broken_target = {'format': 'not-magicdelta-ir'}

        res = client.post(
            '/api/networks/transfer',
            json={'source_ir': source_ir, 'target_ir': broken_target},
        )
        assert res.status_code == 422, res.text
        assert 'IRが不正' in res.json()['detail']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
