"""エディタのモデル部品パレット(Attention/Transformer/PE/Embedding/DWN/UpSample)の検証。

フロント(tools/editor/frontend/src/blocks/templates.ts)がパレットから生成する
ノードのparams生成規約を Python 側で同型に再現し、バックエンドの
validate/build/simulate を通ることを保証する。本体 griffith_logic は
一切変更せず、読み取り(validate/build/simulate/blocks_to_ir)のみで確認する。

要点(実測): blackbox(Attention/Transformer)は validate は通るが build_torch は
NotImplementedError(IRからの再構築不可)。よってエディタのIRビルド学習経路では
学習できず、学習は Python import 経路のモデルで行う(設計判断)。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.blocks.attention import LogicAttention
from griffith_logic.blocks.positional import position_codes
from griffith_logic.blocks.to_ir import blocks_to_ir
from griffith_logic.blocks.transformer import LogicTransformerBlock
from griffith_logic.ir import build_torch, ir_from_dict, simulate, validate


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _pass_through_tt(n_inputs: int) -> str:
    return '0x' + 'a' * ((1 << n_inputs) // 4)


def _placeholder_indices(in_width, out_width, n):
    return [[(o * n + j) % in_width for o in range(out_width)] for j in range(n)]


# ============================================================
# フロント generator の規約を Python で同型再現(templates.ts と一致)
# ============================================================

def _front_attention_node(d_model, seq_len, n_heads, mode):
    in_width = seq_len * d_model
    out_width = in_width  # d_v = d_model
    s = in_width // d_model
    return {
        'id': 'n0_attn',
        'type': 'blackbox',
        'params': {
            'torch_class': 'LogicAttention',
            'config': {'d_model': d_model, 'n_heads': n_heads,
                       'mode': mode, 'similarity': 'xnor'},
            'in_width': in_width,
            'out_width': out_width,
        },
        'train_info': {
            'contract': {
                'input': f'(B, S, {d_model})のNCHW flatten(S={s})',
                'output': f'(B, S, {out_width // s})のflatten',
                'hdl': 'softmax/argmaxを含むためHDL生成不可',
            },
        },
        'meta': {},
    }


def _front_transformer_node(d_model, seq_len, n_heads, norm, merge):
    in_width = seq_len * d_model
    return {
        'id': 'n0_tf',
        'type': 'blackbox',
        'params': {
            'torch_class': 'LogicTransformerBlock',
            'config': {'d_model': d_model, 'n_heads': n_heads,
                       'norm': norm, 'merge': merge},
            'in_width': in_width,
            'out_width': in_width,
        },
        'train_info': {},
        'meta': {},
    }


def _doc(nodes, edges, in_width, out_width):
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'parts'},
        'inputs': [_port('x', in_width)],
        'outputs': [_port('y', out_width)],
        'nodes': nodes,
        'edges': edges,
    }


# ============================================================
# Attention/Transformer blackbox
# ============================================================

class TestAttentionBlackbox:
    def test_front_attention_matches_to_ir_contract(self):
        """フロント生成の blackbox params/contract が to_ir._emit_attention_blackbox と一致。"""
        d_model, seq_len, n_heads, mode = 4, 3, 2, 'soft'
        front = _front_attention_node(d_model, seq_len, n_heads, mode)
        # バックエンドの真実源: 実 LogicAttention を blocks_to_ir でblackbox化
        attn = LogicAttention(d_model, n_heads, mode=mode)
        ir = blocks_to_ir(attn, seq_len * d_model)
        bb = next(n for n in ir.nodes if n.type == 'blackbox')
        assert bb.params['torch_class'] == front['params']['torch_class']
        assert bb.params['config'] == front['params']['config']
        assert bb.params['in_width'] == front['params']['in_width']
        assert bb.params['out_width'] == front['params']['out_width']
        assert bb.train_info['contract'] == front['train_info']['contract']

    def test_front_transformer_matches_to_ir(self):
        d_model, seq_len, n_heads = 4, 3, 2
        front = _front_transformer_node(d_model, seq_len, n_heads, 'layernorm', 'xor')
        tf = LogicTransformerBlock(d_model, n_heads, norm='layernorm', merge='xor')
        ir = blocks_to_ir(tf, seq_len * d_model)
        bb = next(n for n in ir.nodes if n.type == 'blackbox')
        assert bb.params['torch_class'] == 'LogicTransformerBlock'
        assert bb.params['config'] == front['params']['config']
        assert bb.params['in_width'] == front['params']['in_width']
        assert bb.params['out_width'] == front['params']['out_width']

    def test_blackbox_validates_but_build_fails(self):
        """blackbox混在IRは validate 通過するが build_torch は不可(=学習はPython経路)。"""
        node = _front_attention_node(4, 3, 1, 'soft')
        w = node['params']['in_width']
        doc = _doc([node], [{'src': 'x', 'dst': 'n0_attn'},
                            {'src': 'n0_attn', 'dst': 'y'}], w, w)
        graph = ir_from_dict(doc)
        assert validate(graph) == []  # 配置・形状診断は通る
        with pytest.raises(NotImplementedError):
            build_torch(graph)


# ============================================================
# PositionalEncoding / Embedding
# ============================================================

class TestPositionalEncoding:
    @pytest.mark.parametrize('code,seq_len,width', [
        ('binary', 4, 3), ('gray', 4, 3), ('thermometer', 4, 4),
    ])
    def test_pe_constant_concat_matches_position_codes(self, code, seq_len, width):
        """生成マクロ(constant+concat)のsimulateが positional.py position_codes とビット一致。"""
        bits = [int(b) for b in position_codes(seq_len, width, code).reshape(-1).tolist()]
        const_width = seq_len * width
        in_width = 3
        const = {'id': 'c', 'type': 'constant',
                 'params': {'width': const_width, 'bits': bits},
                 'train_info': {'from': 'LogicPositionalEncoding', 'code': code},
                 'meta': {}}
        cat = {'id': 'cat', 'type': 'concat',
               'params': {'in_widths': [in_width, const_width]},
               'train_info': {}, 'meta': {}}
        doc = _doc([const, cat], [
            {'src': 'x', 'dst': 'cat'},
            {'src': 'c', 'dst': 'cat', 'dst_slot': 1},
            {'src': 'cat', 'dst': 'y'},
        ], in_width, in_width + const_width)
        graph = ir_from_dict(doc)
        assert validate(graph) == []
        x = np.array([[1, 1, 0]], dtype=np.int64)
        y = np.asarray(simulate(graph, x)).astype(np.int64).flatten().tolist()
        # 先頭=入力、末尾=位置定数(bits と一致)
        assert y[:in_width] == [1, 1, 0]
        assert y[in_width:] == bits

    def test_embedding_lut_builds(self):
        """簡易Embedding(one-hot vocab → dim の lut_layer)が validate+build 可能。"""
        vocab, dim, n = 8, 4, 2
        node = {'id': 'n0_emb', 'type': 'lut_layer',
                'params': {'in_width': vocab, 'out_width': dim, 'n_inputs': n,
                           'indices': _placeholder_indices(vocab, dim, n),
                           'truth_tables': [_pass_through_tt(n)] * dim},
                'train_info': {'kind': 'embedding', 'parameterization': 'softmax',
                               'initialized': False},
                'meta': {}}
        doc = _doc([node], [{'src': 'x', 'dst': 'n0_emb'},
                            {'src': 'n0_emb', 'dst': 'y'}], vocab, dim)
        graph = ir_from_dict(doc)
        assert validate(graph) == []
        assert build_torch(graph) is not None


# ============================================================
# DWN / UpSample2d
# ============================================================

class TestDwnBinaryBrain:
    def test_dwn_lut_builds_with_kind(self):
        """DWN生成ノードは lut_layer+train_info.kind='dwn' で validate+build 可能(規約)。"""
        in_w, out_w, n = 12, 8, 6
        node = {'id': 'n0_dwn', 'type': 'lut_layer',
                'params': {'in_width': in_w, 'out_width': out_w, 'n_inputs': n,
                           'indices': _placeholder_indices(in_w, out_w, n),
                           'truth_tables': [_pass_through_tt(n)] * out_w},
                'train_info': {'kind': 'dwn', 'mapping': 'random',
                               'parameterization': 'softmax', 'initialized': False},
                'meta': {}}
        doc = _doc([node], [{'src': 'x', 'dst': 'n0_dwn'},
                            {'src': 'n0_dwn', 'dst': 'y'}], in_w, out_w)
        graph = ir_from_dict(doc)
        assert validate(graph) == []
        assert build_torch(graph) is not None
        # kind が抽出器の期待(DWNLayer→lut_layer, train_info.kind='dwn')と整合
        assert graph.nodes[0].train_info['kind'] == 'dwn'

    def test_upsample_nearest_duplicates(self):
        """UpSample2d恒等配線(n_inputs=2 slot0素通し)が nearest 複製で simulate 一致。"""
        in_c, in_h, in_w, s = 1, 2, 2, 2
        h_out, w_out = in_h * s, in_w * s
        width_out = in_c * h_out * w_out
        in_width = in_c * in_h * in_w
        row = [0] * width_out
        for c in range(in_c):
            for y in range(h_out):
                for x in range(w_out):
                    row[(c * h_out + y) * w_out + x] = \
                        (c * in_h + y // s) * in_w + x // s
        node = {'id': 'n0_up', 'type': 'lut_layer',
                'params': {'in_width': in_width, 'out_width': width_out,
                           'n_inputs': 2, 'indices': [row, list(row)],
                           'truth_tables': [_pass_through_tt(2)] * width_out},
                'train_info': {'kind': 'binarybrain', 'from': 'UpSample2d',
                               'initialized': True},
                'meta': {}}
        doc = _doc([node], [{'src': 'x', 'dst': 'n0_up'},
                            {'src': 'n0_up', 'dst': 'y'}], in_width, width_out)
        graph = ir_from_dict(doc)
        assert validate(graph) == []
        # nearest拡大: 各出力画素は対応する入力画素の複製
        x = np.array([[1, 0, 0, 1]], dtype=np.int64)  # 2x2
        y = np.asarray(simulate(graph, x)).astype(np.int64).flatten().tolist()
        expected = [x[0, row[o]] for o in range(width_out)]
        assert y == expected

    def test_upsample_builds(self):
        in_c, in_h, in_w, s = 1, 2, 2, 2
        h_out, w_out = in_h * s, in_w * s
        width_out = in_c * h_out * w_out
        in_width = in_c * in_h * in_w
        row = [((c * in_h + y // s) * in_w + x // s)
               for c in range(in_c) for y in range(h_out) for x in range(w_out)]
        node = {'id': 'n0_up', 'type': 'lut_layer',
                'params': {'in_width': in_width, 'out_width': width_out,
                           'n_inputs': 2, 'indices': [row, list(row)],
                           'truth_tables': [_pass_through_tt(2)] * width_out},
                'train_info': {'kind': 'binarybrain', 'initialized': True},
                'meta': {}}
        doc = _doc([node], [{'src': 'x', 'dst': 'n0_up'},
                            {'src': 'n0_up', 'dst': 'y'}], in_width, width_out)
        graph = ir_from_dict(doc)
        model = build_torch(graph)
        assert model is not None
