"""エディタ往復同一性テスト(E2)。

IR → generate_code → exec → build_model() → export_ir → IR の往復で、
(a) canonical正規化一致 (b) tt/indicesビット厳密一致 (c) simulate整数スコア
完全一致、を全パラメータ化で検証する。フロントエンド・editor extra
(fastapi)なしで走る(codegen.pyは純関数)。
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import (
    IREdge,
    IRGraph,
    IRNode,
    IRPort,
    build_torch,
    ir_from_dict,
    ir_to_dict,
    simulate,
    validate,
)
from griffith_logic.ir.serialize import tt_from_hex
from tools.editor.backend.codegen import generate_code

IN_WIDTH = 16
HIDDEN = 24
CLASSES = 4
TAU = 2.5
SIM_BATCH = 64
SIM_SEED = 7

PARAMS_NS = [
    ('softmax', 2), ('softmax', 4), ('softmax', 6),
    ('independent', 2), ('independent', 4), ('independent', 6),
    ('iwp', 2), ('iwp', 4),
    ('warp', 2), ('warp', 4), ('warp', 6),
]
IDS = [f'{p}_n{n}' for p, n in PARAMS_NS]


def _model(param, n, seed=31):
    torch.manual_seed(seed)
    model = nn.Sequential(
        LutLayer(IN_WIDTH, HIDDEN, n_inputs=n, parameterization=param),
        LutLayer(HIDDEN, HIDDEN, n_inputs=n, parameterization=param),
        GroupSum(k=CLASSES, tau=TAU),
    )
    model.eval()
    return model


def _binary_input(batch=SIM_BATCH, width=IN_WIDTH, seed=SIM_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)


def _exec_build_model(code: str):
    """生成コードをexecして build_model() の戻り値を返す"""
    ns = {}
    exec(compile(code, '<codegen>', 'exec'), ns)
    model = ns['build_model']()
    model.eval()
    return model


def _canonical(graph: IRGraph) -> dict:
    """比較用の正規形。ノードidはトポ位置で正規化し、meta/metadataは除外。

    truth_tables は tt_from_hex でint化して表記ゆれ(大文字/0埋め)を吸収する。
    """
    order = graph.topological_order()
    pos_of = {node.id: f'#{i}' for i, node in enumerate(order)}

    nodes = []
    for node in order:
        params = dict(node.params)
        if node.type == 'lut_layer':
            params['indices'] = np.asarray(params['indices']).astype(int).tolist()
            params['truth_tables'] = tt_from_hex(params['truth_tables'])
        nodes.append({
            'type': node.type,
            'params': params,
            'train_info': node.train_info,
        })
    edges = sorted(
        (pos_of.get(e.src, e.src), pos_of.get(e.dst, e.dst)) for e in graph.edges
    )
    return {
        'nodes': nodes,
        'edges': edges,
        'inputs': [vars(p) for p in graph.inputs],
        'outputs': [vars(p) for p in graph.outputs],
    }


class TestCodegenRoundtrip:
    @pytest.mark.parametrize('param,n', PARAMS_NS, ids=IDS)
    def test_roundtrip_identity(self, param, n):
        """IR→codegen→exec→build_model→export_ir が元IRと同一"""
        model = _model(param, n)
        ir1 = export_ir(model, input_width=IN_WIDTH, name=f'{param}_n{n}')
        validate(ir1)

        code = generate_code(ir1)
        model2 = _exec_build_model(code)
        ir2 = export_ir(model2, input_width=IN_WIDTH, name=f'{param}_n{n}')

        # (a) canonical正規化一致
        assert _canonical(ir1) == _canonical(ir2)

        # (b) tt/indicesビット厳密一致(明示チェック)
        for n1, n2 in zip(ir1.topological_order(), ir2.topological_order()):
            if n1.type != 'lut_layer':
                continue
            assert tt_from_hex(n1.params['truth_tables']) == \
                tt_from_hex(n2.params['truth_tables'])
            assert (np.asarray(n1.params['indices'])
                    == np.asarray(n2.params['indices'])).all()

        # (c) simulate整数スコア完全一致(固定seed 64×16入力)
        x = _binary_input()
        assert (simulate(ir1, x) == simulate(ir2, x)).all()


class TestHttpPath:
    def test_roundtrip_http_path(self):
        """ir_to_dict→JSON→ir_from_dict の経路でvalidate+simulateが不変"""
        model = _model('softmax', 2)
        ir1 = export_ir(model, input_width=IN_WIDTH, name='http_path')
        doc = json.loads(json.dumps(ir_to_dict(ir1)))
        ir2 = ir_from_dict(doc)
        validate(ir2)

        x = _binary_input()
        assert (simulate(ir1, x) == simulate(ir2, x)).all()

    def test_ir_from_dict_rejects_bad_format(self):
        with pytest.raises(ValueError, match='format'):
            ir_from_dict({'format': 'nosuch', 'nodes': [], 'edges': [],
                          'inputs': [], 'outputs': []})

    def test_ir_from_dict_rejects_data_ref(self):
        model = _model('softmax', 2)
        doc = ir_to_dict(export_ir(model, input_width=IN_WIDTH))
        doc['nodes'][0]['params']['indices'] = {'$data': 'x.npz#k'}
        with pytest.raises(ValueError, match=r'\$data'):
            ir_from_dict(doc)


def _pass_through_tt(n_inputs: int) -> str:
    """draft規約のplaceholder真理値表(pass-through 'A')"""
    return '0x' + 'a' * ((1 << n_inputs) // 4)


def _placeholder_indices(in_width, out_width, n_inputs):
    """draft規約のplaceholder接続: indices[j][o] = (o*n+j) % in_width"""
    return [
        [(o * n_inputs + j) % in_width for o in range(out_width)]
        for j in range(n_inputs)
    ]


def _draft_lut_node(node_id, in_width, out_width, n_inputs=2):
    return IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': _placeholder_indices(in_width, out_width, n_inputs),
            'truth_tables': [_pass_through_tt(n_inputs)] * out_width,
        },
        train_info={
            'parameterization': 'softmax',
            'connections': 'random',
            'selection': 'soft',
            'initialized': False,
        },
    )


def _draft_ir() -> IRGraph:
    """フロントのdraftノード規約に沿った未学習2層MLPのIR"""
    nodes = [
        _draft_lut_node('u1_lut_layer', IN_WIDTH, HIDDEN),
        _draft_lut_node('u2_lut_layer', HIDDEN, HIDDEN),
        IRNode(id='u3_group_sum', type='group_sum',
               params={'in_width': HIDDEN, 'k': CLASSES, 'tau': 1.0}),
    ]
    edges = [
        IREdge(src='x', dst='u1_lut_layer'),
        IREdge(src='u1_lut_layer', dst='u2_lut_layer'),
        IREdge(src='u2_lut_layer', dst='u3_group_sum'),
        IREdge(src='u3_group_sum', dst='y'),
    ]
    return IRGraph(
        metadata={'name': 'draft_mlp'},
        inputs=[IRPort(id='x', width=IN_WIDTH)],
        outputs=[IRPort(id='y', width=CLASSES)],
        nodes=nodes,
        edges=edges,
    )


class TestDraft:
    def test_draft_build(self, seed):
        """draft IR→build→1step訓練→export が通る"""
        ir = _draft_ir()
        assert validate(ir) == []

        model = build_torch(ir)
        model.train()
        x = torch.randint(0, 2, (8, IN_WIDTH)).float()
        y = torch.randint(0, CLASSES, (8,))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)

        model.eval()
        ir2 = export_ir(model, input_width=IN_WIDTH, name='draft_built')
        assert validate(ir2) == []
        # draft確定後は initialized 指定なし(=学習済み扱い)で再現可能
        code = generate_code(ir2)
        model3 = _exec_build_model(code)
        ir3 = export_ir(model3, input_width=IN_WIDTH, name='draft_built')
        assert _canonical(ir2) == _canonical(ir3)

    def test_codegen_draft_no_restore(self):
        """draftノードはrestoreせず「未学習」コメントになる"""
        ir = _draft_ir()
        code = generate_code(ir)
        assert 'restore_lut_layer' not in code
        assert '未学習(ランダム初期化)' in code
        # 生成コードは実行可能で、build後のexportも通る
        model = _exec_build_model(code)
        ir2 = export_ir(model, input_width=IN_WIDTH)
        assert validate(ir2) == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
