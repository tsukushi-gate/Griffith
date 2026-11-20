"""HDLエクスポートAPI(tools/editor/backend/hdl.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する。
生成ソースが to_verilog/to_vhdl 直呼びとバイト一致することと、
estimateの数値、HDL不可IR(blackbox等)の400応答を確認する。
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.export.verify import iverilog_available
from griffith_logic.export.verilog import to_verilog
from griffith_logic.export.vhdl import to_vhdl
from griffith_logic.ir import ir_from_dict, ir_to_dict
from griffith_logic.pooling import OrPooling2d
from tools.editor.backend import hdl as hdl_api
from tools.editor.backend import schemas


def _blackbox_ir() -> dict:
    """HDL不可なblackbox入りIR(preflight/verifyのブロッカー検出用)。"""
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': 'hybrid'},
        'inputs': [{'id': 'x', 'width': 4, 'shape': None,
                    'encoding': None, 'semantics': None}],
        'outputs': [{'id': 'y', 'width': 4, 'shape': None,
                     'encoding': None, 'semantics': None}],
        'nodes': [{'id': 'bb', 'type': 'blackbox',
                   'params': {'torch_class': 'Linear', 'config': {},
                              'in_width': 4, 'out_width': 4},
                   'train_info': {}, 'meta': {}}],
        'edges': [{'src': 'x', 'dst': 'bb'}, {'src': 'bb', 'dst': 'y'}],
    }

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_fixture_ir(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f'{name}.json')) as f:
        return json.load(f)['ir']


class FakeState:
    """main.AppState互換の最小状態(main.py非依存)"""

    def __init__(self):
        self.model = None
        self.ir = None
        self.ir_dirty = False
        self.input_width = None


@pytest.fixture
def ctx():
    state = FakeState()
    app = FastAPI()
    app.include_router(hdl_api.configure(state))
    return state, TestClient(app, base_url='http://localhost:8765')


class TestHdlExportSamples:
    def test_xor_verilog_matches_direct_call(self, ctx):
        """XOR: 生成ソースがto_verilog直呼びとバイト一致+estimate数値"""
        state, client = ctx
        doc = _load_fixture_ir('xor')
        state.ir = doc

        res = client.post('/api/export/hdl', json={'lang': 'verilog'})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['source'] == to_verilog(ir_from_dict(doc))
        assert body['filename'] == 'fx_xor.v'
        assert body['lang'] == 'verilog'
        est = body['estimate']
        assert est['total_luts'] == 1
        assert est['logic_depth'] == 1
        assert est['wire_count'] == 3
        assert est['lut_counts'] == {'2': 1}  # JSONキーは文字列化される

    def test_parity3_both_langs_match_direct_call(self, ctx):
        state, client = ctx
        doc = _load_fixture_ir('parity3')
        state.ir = doc
        graph = ir_from_dict(doc)

        res_v = client.post('/api/export/hdl', json={'lang': 'verilog'})
        assert res_v.status_code == 200, res_v.text
        assert res_v.json()['source'] == to_verilog(graph)
        assert res_v.json()['estimate']['total_luts'] == 3

        res_h = client.post('/api/export/hdl', json={'lang': 'vhdl'})
        assert res_h.status_code == 200, res_h.text
        assert res_h.json()['source'] == to_vhdl(graph)
        assert res_h.json()['filename'] == 'fx_parity3.vhdl'

    def test_demo_with_group_sum(self, ctx):
        """群和(group_sum)つきdemo IRでも生成成功"""
        state, client = ctx
        doc = _load_fixture_ir('demo_mnist')
        state.ir = doc

        res = client.post('/api/export/hdl', json={'lang': 'verilog'})
        assert res.status_code == 200, res.text
        body = res.json()
        assert 'module logic_core' in body['source']
        assert body['source'] == to_verilog(ir_from_dict(doc))
        assert body['estimate']['group_sums'] == 1
        assert body['estimate']['total_luts'] == 800


class TestHdlExportConvLowering:
    def _conv_ir(self):
        torch.manual_seed(31)
        model = nn.Sequential(
            LutConv2d(1, 4, kernel_size=3, tree_depth=2),
            OrPooling2d(kernel_size=2),
            nn.Flatten(),
            LutLayer(36, 24, n_inputs=2),
            GroupSum(k=4, tau=1.0),
        )
        model.eval()
        return ir_to_dict(export_ir(model, input_shape=(1, 8, 8),
                                    name='cnn', native_conv=True))

    def test_conv_native_ir_lowered_and_exported(self, ctx):
        """conv2dネイティブIRはlower→to_verilogで生成される"""
        state, client = ctx
        state.ir = self._conv_ir()

        res = client.post('/api/export/hdl', json={'lang': 'verilog'})
        assert res.status_code == 200, res.text
        body = res.json()
        assert 'w_n0_lutconv2d_lv0' in body['source']
        assert body['source'] == to_verilog(
            lower_to_primitives(ir_from_dict(state.ir)))
        assert body['estimate']['lowered'] is True

    def test_conv_without_lower_400(self, ctx):
        """lower=falseでconv2dが残っていれば400+日本語エラー"""
        state, client = ctx
        state.ir = self._conv_ir()
        res = client.post('/api/export/hdl',
                          json={'lang': 'verilog', 'lower': False})
        assert res.status_code == 400
        assert 'lower_to_primitives' in res.json()['detail']


class TestHdlExportErrors:
    def test_no_ir_400(self, ctx):
        _, client = ctx
        res = client.post('/api/export/hdl', json={})
        assert res.status_code == 400
        assert 'IR未設定' in res.json()['detail']

    def test_blackbox_400(self, ctx):
        """blackbox到達は400+日本語エラー"""
        state, client = ctx
        state.ir = {
            'format': 'magicdelta-ir', 'version': '0.2.0',
            'metadata': {'name': 'hybrid'},
            'inputs': [{'id': 'x', 'width': 4, 'shape': None,
                        'encoding': None, 'semantics': None}],
            'outputs': [{'id': 'y', 'width': 4, 'shape': None,
                         'encoding': None, 'semantics': None}],
            'nodes': [{'id': 'bb', 'type': 'blackbox',
                       'params': {'torch_class': 'Linear', 'config': {},
                                  'in_width': 4, 'out_width': 4},
                       'train_info': {}, 'meta': {}}],
            'edges': [{'src': 'x', 'dst': 'bb'}, {'src': 'bb', 'dst': 'y'}],
        }
        res = client.post('/api/export/hdl', json={'lang': 'verilog'})
        assert res.status_code == 400
        detail = res.json()['detail']
        assert 'HDL生成不可' in detail and 'blackbox' in detail

    def test_broken_ir_422(self, ctx):
        state, client = ctx
        state.ir = {'format': 'other-format', 'nodes': []}
        res = client.post('/api/export/hdl', json={})
        assert res.status_code == 422

    def test_invalid_lang_422(self, ctx):
        state, client = ctx
        state.ir = _load_fixture_ir('xor')
        res = client.post('/api/export/hdl', json={'lang': 'systemverilog'})
        assert res.status_code == 422  # pydanticパターン検証


class TestPreflight:
    def test_xor_exportable(self, ctx):
        """XOR: HDL出力可+ブロッカーなし+見積り+スキーマ整合"""
        state, client = ctx
        state.ir = _load_fixture_ir('xor')

        res = client.get('/api/models/current/preflight')
        assert res.status_code == 200, res.text
        body = res.json()
        schemas.PreflightResponse.model_validate(body)  # スキーマ整合
        assert body['hdl_exportable'] is True
        assert body['blockers'] == []
        assert body['estimate']['total_luts'] == 1
        assert body['estimate']['logic_depth'] == 1
        assert body['draft_node_count'] == 0
        assert set(body['verify_available']) == {'iverilog', 'ghdl'}

    def test_blackbox_blocked(self, ctx):
        """blackbox: 出力不可+node_id付きブロッカー(構造は健全なので見積りは出る)"""
        state, client = ctx
        state.ir = _blackbox_ir()

        body = client.get('/api/models/current/preflight').json()
        schemas.PreflightResponse.model_validate(body)
        assert body['hdl_exportable'] is False
        # 構造検証(generic)は通るため見積りは算出される(ブロッカーと併記)
        assert body['estimate'] is not None
        kinds = {(b['kind'], b['node_id']) for b in body['blockers']}
        assert ('blackbox', 'bb') in kinds

    def test_structure_error_no_estimate(self, ctx):
        """構造エラー(幅不整合)はstructureブロッカー+estimateはNone"""
        state, client = ctx
        doc = _load_fixture_ir('xor')
        doc['inputs'][0]['width'] = 3  # lut_layerのin_width(2)と不整合
        state.ir = doc

        body = client.get('/api/models/current/preflight').json()
        schemas.PreflightResponse.model_validate(body)
        assert body['hdl_exportable'] is False
        assert body['estimate'] is None
        assert any(b['kind'] == 'structure' for b in body['blockers'])

    def test_add_sum_blocked(self, ctx):
        """add(mode='sum'): 出力不可+add_sumブロッカー(全ノード列挙)"""
        state, client = ctx
        state.ir = {
            'format': 'magicdelta-ir', 'version': '0.2.0',
            'metadata': {'name': 'sum'},
            'inputs': [{'id': 'x', 'width': 4, 'shape': None,
                        'encoding': None, 'semantics': None}],
            'outputs': [{'id': 'y', 'width': 4, 'shape': None,
                         'encoding': None, 'semantics': None}],
            'nodes': [{'id': 'a', 'type': 'add',
                       'params': {'width': 4, 'mode': 'sum'},
                       'train_info': {}, 'meta': {}}],
            'edges': [{'src': 'x', 'dst': 'a', 'dst_slot': 0},
                      {'src': 'x', 'dst': 'a', 'dst_slot': 1},
                      {'src': 'a', 'dst': 'y'}],
        }
        body = client.get('/api/models/current/preflight').json()
        assert body['hdl_exportable'] is False
        assert any(b['kind'] == 'add_sum' and b['node_id'] == 'a'
                   for b in body['blockers'])

    def test_draft_nodes_counted(self, ctx):
        """train_info.initialized=Falseは未離散化draftとして数える"""
        state, client = ctx
        doc = _load_fixture_ir('xor')
        doc['nodes'][0]['train_info'] = {'initialized': False}
        state.ir = doc

        body = client.get('/api/models/current/preflight').json()
        assert body['draft_node_count'] == 1

    def test_no_ir_400(self, ctx):
        _, client = ctx
        res = client.get('/api/models/current/preflight')
        assert res.status_code == 400
        assert 'IR未設定' in res.json()['detail']


class TestVerify:
    def test_unavailable_optional(self, ctx, monkeypatch):
        """処理系未導入時はavailable=Falseで実行せず返す(オプショナル設計)"""
        state, client = ctx
        state.ir = _load_fixture_ir('xor')
        monkeypatch.setattr(hdl_api, 'iverilog_available', lambda: False)

        body = client.post('/api/export/hdl/verify', json={'lang': 'verilog'}).json()
        schemas.HdlVerifyResponse.model_validate(body)
        assert body['available'] is False
        assert body['passed'] is None
        assert body['n_vectors'] == 0

    def test_blackbox_400(self, ctx):
        state, client = ctx
        state.ir = _blackbox_ir()
        res = client.post('/api/export/hdl/verify', json={'lang': 'verilog'})
        assert res.status_code == 400
        assert 'HDL生成不可' in res.json()['detail']

    @pytest.mark.skipif(not iverilog_available(), reason='iverilog未導入')
    def test_xor_equivalence_passes(self, ctx):
        """iverilogがある環境ではXORの等価性検証が通る(全4ベクトル)"""
        state, client = ctx
        state.ir = _load_fixture_ir('xor')

        body = client.post('/api/export/hdl/verify', json={'lang': 'verilog'}).json()
        schemas.HdlVerifyResponse.model_validate(body)
        assert body['available'] is True
        assert body['passed'] is True
        assert body['n_vectors'] == 4  # 2入力の全列挙


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
