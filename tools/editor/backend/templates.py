"""モデルテンプレートプロジェクト集。

「最初の一歩をゼロから書かなくていい」ためのテンプレプロジェクトを同梱し、
GUIの新規プロジェクトダイアログから1クリックで生成できるようにする。

REST:
    GET  /api/templates
        → テンプレ一覧(id/name/description/category/verification/…のメタのみ)。
          IRの重い組み立ては行わない(冪等・軽量)。
    POST /api/templates/{id}/instantiate
        body = {name?: str}
        → テンプレを ProjectSchema として組み立て・保存し、その project payload を返す。
          既存の projects_router と同じ PROJECTS_ROOT 規約に保存する。

設計方針(本体不変原則):
  - IRは本モジュール内の純関数(dict組み立て)で構成する。ノード生成規約は
    フロント(blocks/templates.ts / ir/graphToIr.ts)および conv/recurrent の
    既存テスト(test_editor_conv_palette_n7 / test_editor_recurrent_n7)と同型で、
    griffith_logic 本体は build/validate を「読み取り」でのみ利用する。
  - draft lut_layer(train_info.initialized=false)は build 時にランダム初期化される
    学習可能層(build.py:_build_lut_layer 参照)。テンプレはこの draft を基本とする。

main.py への組み込み手順(probe.py/hdl.py と同じ configure 方式):
    from . import templates
    app.include_router(templates.configure(STATE))
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from griffith_logic.ir.validate import conv2d_out_shape, or_pool2d_out_shape

from .project import ProjectSchema, save_project

# templates.py -> backend -> editor -> tools -> リポジトリルート(projects_routerと同一)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECTS_ROOT = PROJECT_ROOT / 'outputs' / 'editor_projects'

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入。本routerは現状未使用)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


# ---------------------------------------------------------------------------
# IRノード生成ヘルパ(pure dict。フロント blocks/templates.ts と同型規約)
# ---------------------------------------------------------------------------

def _port(pid: str, width: int, *, semantics: str | None = None) -> dict:
    return {'id': pid, 'width': width, 'shape': None,
            'encoding': None, 'semantics': semantics}


def _pass_through_tt(n_inputs: int) -> str:
    """passThroughTT: '0x' + 'a'×(2^n/4)(slot0素通し)。graphToIr.ts と一致。"""
    return '0x' + 'a' * ((1 << n_inputs) // 4)


def _placeholder_indices(in_width: int, out_width: int, n_inputs: int) -> list[list[int]]:
    """draft接続: indices[j][o] = (o*n + j) % in_width。graphToIr.ts と一致。"""
    return [
        [(o * n_inputs + j) % in_width for o in range(out_width)]
        for j in range(n_inputs)
    ]


def _draft_lut(node_id: str, in_width: int, out_width: int, n_inputs: int = 2) -> dict:
    """draft lut_layer(build時ランダム初期化される学習可能層)。"""
    return {
        'id': node_id,
        'type': 'lut_layer',
        'params': {
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': _placeholder_indices(in_width, out_width, n_inputs),
            'truth_tables': [_pass_through_tt(n_inputs)] * out_width,
        },
        'train_info': {
            'parameterization': 'softmax',
            'connections': 'random',
            'selection': 'soft',
            'initialized': False,
        },
        'meta': {},
    }


def _group_sum(node_id: str, in_width: int, k: int, tau: float = 1.0) -> dict:
    return {
        'id': node_id,
        'type': 'group_sum',
        'params': {'in_width': in_width, 'k': k, 'tau': tau},
        'train_info': {},
        'meta': {},
    }


def _conv2d(node_id: str, in_shape: tuple[int, int, int], out_channels: int,
            kernel: int, stride: int, padding: int, tree_depth: int,
            n_inputs: int, seed: int) -> dict:
    """conv2d ノード(test_editor_conv_palette_n7._conv2d_node と同型)。"""
    rng = np.random.default_rng(seed)
    c, _, _ = in_shape
    rf = c * kernel * kernel
    n_leaves = n_inputs ** tree_depth
    n_internal = tree_depth if n_inputs == 1 else (n_leaves - 1) // (n_inputs - 1)
    indices_flat = [
        [int(rng.integers(0, rf)) for _ in range(n_leaves)]
        for _ in range(out_channels)
    ]
    nibbles = (1 << n_inputs) // 4
    tts = [
        '0x' + ''.join(f'{int(rng.integers(0, 16)):x}' for _ in range(nibbles))
        for _ in range(out_channels * n_internal)
    ]
    return {
        'id': node_id,
        'type': 'conv2d',
        'params': {
            'in_shape': list(in_shape),
            'out_channels': out_channels,
            'kernel_size': [kernel, kernel],
            'stride': [stride, stride],
            'padding': [padding, padding],
            'tree_depth': tree_depth,
            'n_inputs': n_inputs,
            'indices_flat': indices_flat,
            'truth_tables': tts,
        },
        'train_info': {'parameterization': 'independent'},
        'meta': {},
    }


def _or_pool2d(node_id: str, in_shape: tuple[int, int, int],
               kernel: int, stride: int, padding: int = 0) -> dict:
    return {
        'id': node_id,
        'type': 'or_pool2d',
        'params': {
            'in_shape': list(in_shape),
            'kernel_size': kernel,
            'stride': stride,
            'padding': padding,
        },
        'train_info': {},
        'meta': {},
    }


def _concat(node_id: str, in_widths: list[int]) -> dict:
    return {'id': node_id, 'type': 'concat',
            'params': {'in_widths': list(in_widths)}, 'train_info': {}, 'meta': {}}


def _or_add(node_id: str, width: int) -> dict:
    return {'id': node_id, 'type': 'add',
            'params': {'width': width, 'mode': 'or'}, 'train_info': {}, 'meta': {}}


def _const0(node_id: str, width: int) -> dict:
    return {'id': node_id, 'type': 'constant',
            'params': {'width': width, 'bits': [0] * width}, 'train_info': {}, 'meta': {}}


def _position_bits_flat(seq_len: int, width: int, code: str = 'binary') -> list[int]:
    """位置→ビット列の位置優先flatten(positional.py の position_codes と同一規約)。

    binary/gray: bit(p, j) = ((p or gray(p)) >> j) & 1、index = p*width + j。
    """
    bits: list[int] = []
    for p in range(seq_len):
        g = p ^ (p >> 1) if code == 'gray' else p
        bits.extend((g >> j) & 1 for j in range(width))
    return bits


def _const_bits(node_id: str, bits: list[int], *, train_info: dict | None = None) -> dict:
    """任意ビット列の constant ノード(LogicPositionalEncoding の定数線)。"""
    return {
        'id': node_id,
        'type': 'constant',
        'params': {'width': len(bits), 'bits': list(bits)},
        'train_info': train_info or {},
        'meta': {},
    }


def _fixed_lut(node_id: str, in_width: int, out_width: int, n_inputs: int,
               indices: list[list[int]], tts: list[str]) -> dict:
    """固定(initialized)ブールlut。状態更新用のAND/NOT/passに使う。"""
    return {
        'id': node_id,
        'type': 'lut_layer',
        'params': {'in_width': in_width, 'out_width': out_width,
                   'n_inputs': n_inputs, 'indices': indices, 'truth_tables': tts},
        'train_info': {'initialized': True},
        'meta': {},
    }


def _slice(node_id: str, total: int, offset: int, out_w: int) -> dict:
    """スライス(slot0素通しの恒等lut)。"""
    idx = [offset + o for o in range(out_w)]
    return _fixed_lut(node_id, total, out_w, 2, [idx, list(idx)],
                      ['0xa'] * out_w)  # PASS_TT


def _and_lut(node_id: str, width: int) -> dict:
    """2W→W の要素ANDlut(out o = busA[o] ∧ busB[o])。"""
    a = list(range(width))
    b = list(range(width, 2 * width))
    return _fixed_lut(node_id, 2 * width, width, 2, [a, b], ['0x8'] * width)  # AND_TT


def _not_lut(node_id: str, width: int) -> dict:
    idx = list(range(width))
    return _fixed_lut(node_id, width, width, 2, [idx, list(idx)], ['0x1'] * width)  # NOT_TT


def _graph(name: str, in_width: int, out_width: int, nodes: list[dict],
           edges: list[dict], *, out_semantics: str | None = None) -> dict:
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': name},
        'inputs': [_port('x', in_width)],
        'outputs': [_port('y', out_width, semantics=out_semantics)],
        'nodes': nodes,
        'edges': edges,
    }


# ---------------------------------------------------------------------------
# 学習設定(TrainConfig dict。schemaのみ検証される)
# ---------------------------------------------------------------------------

def _train_config(dataset: str, *, batch_size: int = 128, lr: float = 0.05,
                  epochs: int = 5, loss: str | None = None) -> dict:
    """推奨 Global Config(TrainConfig 互換 dict)。"""
    cfg: dict[str, Any] = {
        'data': {'name': dataset, 'batch_size': batch_size},
        'optim': {'name': 'adam', 'lr': lr},
        'run': {'epochs': epochs, 'seed': 42},
    }
    if loss is not None:
        cfg['run']['loss'] = loss
    return cfg


# ---------------------------------------------------------------------------
# テンプレIRビルダ(instantiate時のみ呼ばれる。GET一覧では実行しない)
# ---------------------------------------------------------------------------

CLASSIFIER_WIDTH = 60  # 10クラス × 6ニューロン/クラス


def _build_logreg() -> dict:
    """ロジスティック回帰: 784 → lut_layer → GroupSum(10)(1層直結)。"""
    lut = _draft_lut('logits', 784, CLASSIFIER_WIDTH)
    gs = _group_sum('classes', CLASSIFIER_WIDTH, k=10)
    graph = _graph('logreg', 784, 10, [lut, gs], [
        {'src': 'x', 'dst': 'logits'},
        {'src': 'logits', 'dst': 'classes'},
        {'src': 'classes', 'dst': 'y'},
    ], out_semantics='class_scores')
    return {
        'networks': {'Main': graph},
        'train_configs': {'GlobalConfig': _train_config('mnist')},
    }


def _build_mlp() -> dict:
    """MLP: 784 → LutLayer×2 → GroupSum(10)。"""
    l1 = _draft_lut('hidden1', 784, 256)
    l2 = _draft_lut('hidden2', 256, CLASSIFIER_WIDTH)
    gs = _group_sum('classes', CLASSIFIER_WIDTH, k=10)
    graph = _graph('mlp', 784, 10, [l1, l2, gs], [
        {'src': 'x', 'dst': 'hidden1'},
        {'src': 'hidden1', 'dst': 'hidden2'},
        {'src': 'hidden2', 'dst': 'classes'},
        {'src': 'classes', 'dst': 'y'},
    ], out_semantics='class_scores')
    return {
        'networks': {'Main': graph},
        'train_configs': {'GlobalConfig': _train_config('mnist')},
    }


def _build_lenet() -> dict:
    """LeNet風CNN: conv2d → or_pool2d → conv2d → or_pool2d → LutLayer → GroupSum。

    入力 1×28×28(=784, MNIST)。conv/pool の形状は本体の shape 伝播で確定する。
    """
    in_shape = (1, 28, 28)
    conv1 = _conv2d('conv1', in_shape, out_channels=8, kernel=3, stride=1,
                    padding=0, tree_depth=2, n_inputs=2, seed=101)
    s1 = conv2d_out_shape(conv1['params'])            # (8, 26, 26)
    pool1 = _or_pool2d('pool1', s1, kernel=2, stride=2)
    s2 = or_pool2d_out_shape(pool1['params'])         # (8, 13, 13)
    conv2 = _conv2d('conv2', s2, out_channels=16, kernel=3, stride=1,
                    padding=0, tree_depth=2, n_inputs=2, seed=202)
    s3 = conv2d_out_shape(conv2['params'])            # (16, 11, 11)
    pool2 = _or_pool2d('pool2', s3, kernel=2, stride=2)
    s4 = or_pool2d_out_shape(pool2['params'])         # (16, 5, 5)
    flat_width = s4[0] * s4[1] * s4[2]                # 400
    head = _draft_lut('head', flat_width, CLASSIFIER_WIDTH)
    gs = _group_sum('classes', CLASSIFIER_WIDTH, k=10)
    graph = _graph('lenet', 784, 10,
                   [conv1, pool1, conv2, pool2, head, gs], [
                       {'src': 'x', 'dst': 'conv1'},
                       {'src': 'conv1', 'dst': 'pool1'},
                       {'src': 'pool1', 'dst': 'conv2'},
                       {'src': 'conv2', 'dst': 'pool2'},
                       {'src': 'pool2', 'dst': 'head'},
                       {'src': 'head', 'dst': 'classes'},
                       {'src': 'classes', 'dst': 'y'},
                   ], out_semantics='class_scores')
    return {
        'networks': {'Main': graph},
        'train_configs': {'GlobalConfig': _train_config('mnist', batch_size=64, lr=0.02)},
    }


AE_IN_WIDTH = 64
AE_CODE_WIDTH = 16


def _build_autoencoder() -> dict:
    """AutoEncoder: 学習ネット(enc+dec)と推論ネット(enc)の2ネット構成。

    学習後、POST /api/networks/transfer で enc ノードの重みを Runtime へ転写する
    (network_transfer.py。id='enc' が両ネットで一致し幅シグネチャも一致する)。
    """
    enc_train = _draft_lut('enc', AE_IN_WIDTH, AE_CODE_WIDTH)
    dec_train = _draft_lut('dec', AE_CODE_WIDTH, AE_IN_WIDTH)
    train_graph = _graph('autoencoder_train', AE_IN_WIDTH, AE_IN_WIDTH,
                         [enc_train, dec_train], [
                             {'src': 'x', 'dst': 'enc'},
                             {'src': 'enc', 'dst': 'dec'},
                             {'src': 'dec', 'dst': 'y'},
                         ])
    enc_runtime = _draft_lut('enc', AE_IN_WIDTH, AE_CODE_WIDTH)
    runtime_graph = _graph('autoencoder_runtime', AE_IN_WIDTH, AE_CODE_WIDTH,
                           [enc_runtime], [
                               {'src': 'x', 'dst': 'enc'},
                               {'src': 'enc', 'dst': 'y'},
                           ])
    return {
        'networks': {'Train': train_graph, 'Runtime': runtime_graph},
        'train_configs': {
            'GlobalConfig': _train_config('mnist', batch_size=64,
                                          lr=0.05, loss='bce'),
        },
    }


LSTM_UNROLL = 4
LSTM_IN_WIDTH = 4
LSTM_HIDDEN = 4


def _build_lstm() -> dict:
    """LSTM系列: セルをT回時間展開(recurrentUnit.ts と同型の状態代数)。

    ゲート/候補は draft lut_layer(学習可能)、状態更新は固定ブールlut/addで構成する。
    unroll は各step独立層(重み非共有)で、系列モデルの実地サンプルを兼ねる。
    """
    unroll, in_width, hidden = LSTM_UNROLL, LSTM_IN_WIDTH, LSTM_HIDDEN
    total = unroll * in_width
    z_width = in_width + hidden
    nodes: list[dict] = [_slice('rnn_in', total, 0, total),
                         _const0('init_h', hidden), _const0('init_c', hidden)]
    edges: list[dict] = [{'src': 'x', 'dst': 'rnn_in'}]
    h_prev, c_prev = 'init_h', 'init_c'

    for t in range(unroll):
        xt = f'x{t}'
        nodes.append(_slice(xt, total, t * in_width, in_width))
        edges.append({'src': 'rnn_in', 'dst': xt})
        z = f'z{t}'
        nodes.append(_concat(z, [in_width, hidden]))
        edges.append({'src': xt, 'dst': z, 'dst_slot': 0})
        edges.append({'src': h_prev, 'dst': z, 'dst_slot': 1})

        # 4ゲート(f/i/o/候補g)= draft lut(z_width → hidden)、学習可能
        for gate in ('f', 'i', 'o', 'g'):
            gid = f'{gate}{t}'
            nodes.append(_draft_lut(gid, z_width, hidden))
            edges.append({'src': z, 'dst': gid})

        # ic = and(i, g)
        nodes.append(_concat(f'ic_cat{t}', [hidden, hidden]))
        edges.append({'src': f'i{t}', 'dst': f'ic_cat{t}', 'dst_slot': 0})
        edges.append({'src': f'g{t}', 'dst': f'ic_cat{t}', 'dst_slot': 1})
        nodes.append(_and_lut(f'ic{t}', hidden))
        edges.append({'src': f'ic_cat{t}', 'dst': f'ic{t}'})

        # notf = not(f)
        nodes.append(_not_lut(f'nf{t}', hidden))
        edges.append({'src': f'f{t}', 'dst': f'nf{t}'})

        # t1 = and(f, c_prev)
        nodes.append(_concat(f't1cat{t}', [hidden, hidden]))
        edges.append({'src': f'f{t}', 'dst': f't1cat{t}', 'dst_slot': 0})
        edges.append({'src': c_prev, 'dst': f't1cat{t}', 'dst_slot': 1})
        nodes.append(_and_lut(f't1_{t}', hidden))
        edges.append({'src': f't1cat{t}', 'dst': f't1_{t}'})

        # t2 = and(notf, ic)
        nodes.append(_concat(f't2cat{t}', [hidden, hidden]))
        edges.append({'src': f'nf{t}', 'dst': f't2cat{t}', 'dst_slot': 0})
        edges.append({'src': f'ic{t}', 'dst': f't2cat{t}', 'dst_slot': 1})
        nodes.append(_and_lut(f't2_{t}', hidden))
        edges.append({'src': f't2cat{t}', 'dst': f't2_{t}'})

        # c_t = or(t1, t2)
        c_new = f'c{t}'
        nodes.append(_or_add(c_new, hidden))
        edges.append({'src': f't1_{t}', 'dst': c_new, 'dst_slot': 0})
        edges.append({'src': f't2_{t}', 'dst': c_new, 'dst_slot': 1})

        # h_t = and(o, c_t)
        nodes.append(_concat(f'hcat{t}', [hidden, hidden]))
        edges.append({'src': f'o{t}', 'dst': f'hcat{t}', 'dst_slot': 0})
        edges.append({'src': c_new, 'dst': f'hcat{t}', 'dst_slot': 1})
        h_new = f'h{t}'
        nodes.append(_and_lut(h_new, hidden))
        edges.append({'src': f'hcat{t}', 'dst': h_new})

        h_prev, c_prev = h_new, c_new

    edges.append({'src': h_prev, 'dst': 'y'})
    graph = _graph('lstm_seq', total, hidden, nodes, edges)
    return {
        'networks': {'Main': graph},
        'train_configs': {'GlobalConfig': _train_config('mnist', batch_size=64)},
    }


TF_SEQ_LEN = 4
TF_D_MODEL = 8


def _attention_contract(d_model: int, in_width: int, out_width: int) -> dict:
    s = in_width // d_model
    return {
        'input': f'(B, S, {d_model})のNCHW flatten(S={s})',
        'output': f'(B, S, {out_width // max(s, 1)})のflatten',
        'hdl': 'softmax/argmaxを含むためHDL生成不可',
    }


def _build_transformer() -> dict:
    """Transformer(blackbox版): LogicTransformerBlock を blackbox ノードで表現。

    blackbox は build/HDL 不可(build.py:_build_blackbox が raise)。学習は
    Python import 経路(griffith_logic.blocks.LogicTransformerBlock)で行う。
    エディタ上では validate 通過=構造の妥当性のみを保証する。
    """
    seq_len, d_model = TF_SEQ_LEN, TF_D_MODEL
    width = seq_len * d_model
    node = {
        'id': 'transformer',
        'type': 'blackbox',
        'params': {
            'torch_class': 'LogicTransformerBlock',
            'config': {'d_model': d_model, 'n_heads': 2,
                       'norm': 'layernorm', 'merge': 'or'},
            'in_width': width,
            'out_width': width,
        },
        'train_info': {'contract': _attention_contract(d_model, width, width)},
        'meta': {'editor': {'detail': f'LogicTransformerBlock d={d_model} h=2'}},
    }
    graph = _graph('transformer_block', width, width, [node], [
        {'src': 'x', 'dst': 'transformer'},
        {'src': 'transformer', 'dst': 'y'},
    ])
    return {
        'networks': {'Main': graph},
        'train_configs': {},
    }


def _transformer_blackbox(node_id: str, seq_len: int, d_model: int,
                          n_heads: int, *, detail: str) -> dict:
    """LogicTransformerBlock の blackbox ノード(_build_transformer と同型)。"""
    width = seq_len * d_model
    return {
        'id': node_id,
        'type': 'blackbox',
        'params': {
            'torch_class': 'LogicTransformerBlock',
            'config': {'d_model': d_model, 'n_heads': n_heads,
                       'norm': 'layernorm', 'merge': 'or'},
            'in_width': width,
            'out_width': width,
        },
        'train_info': {'contract': _attention_contract(d_model, width, width)},
        'meta': {'editor': {'detail': detail}},
    }


# LLViT複合テンプレのパラメータ。GUIで組める規模に縮約した実地サンプル。
# 実寸(models/llvit.py: dim=192/depth=12/224px)ではなく、validate/表示に足る小型値。
LLVIT_SEQ_LEN = 4          # トークン数(実寸は patch_tokens+1。ここでは小型固定)
LLVIT_D_EMBED = 6          # パッチ埋め込みビット幅/トークン
LLVIT_POS_BITS = 2         # 位置コード幅(binary: seq_len ≦ 2^pos_bits が必要)
LLVIT_D_MODEL = LLVIT_D_EMBED + LLVIT_POS_BITS  # 8(PE連結後のトークン幅)
LLVIT_DEPTH = 2            # Transformerブロック段数
LLVIT_N_HEADS = 2
LLVIT_PATCH_BITS = 12      # 入力の1トークンあたりビット幅(ビット化済みパッチ)


def _build_llvit() -> dict:
    """LL-ViT複合テンプレ: パッチ埋込 → PositionalEncoding → Transformer×N → 分類ヘッド。

    IR表現の判断(パッチ埋め込みの前段):
      models/llvit.py の PatchEmbed は画像→パッチへの nn.Conv2d 射影(連続値 dim ベクトル)で、
      二値論理IRでは表現できない ANN 前段。したがってテンプレ入力は「ビット化済みパッチ列」
      (幅 = seq_len × patch_bits)とし、画像→パッチ分割→ビット化の前処理は学習スクリプト
      (experiments/train_llvit.py)側の責務として説明文・入力仕様で明示する。IR上のパッチ埋込は
      draft lut_layer(patch_bits/トークン → d_embed/トークン、build時ランダム初期化の学習可能層)で表す。

    構成ノード: lut_layer(patch_embed/head)・constant(PE定数線)・concat(PE連結)・
    blackbox(LogicTransformerBlock×N)・group_sum(分類)。blackbox を含むため build/HDL 不可
    (validate のみ)。学習は Python import 経路(experiments/train_llvit.py)で行う。
    """
    seq_len, d_embed = LLVIT_SEQ_LEN, LLVIT_D_EMBED
    pos_bits, d_model = LLVIT_POS_BITS, LLVIT_D_MODEL
    in_width = seq_len * LLVIT_PATCH_BITS          # ビット化済みパッチ列(前段はscript責務)
    embed_width = seq_len * d_embed                # 埋め込み後(位置なし)
    pos_width = seq_len * pos_bits                 # 位置コード定数
    token_width = seq_len * d_model                # PE連結後 = Transformer入出力幅

    # パッチ埋め込み: ビット化パッチ → トークン埋め込み(draft、学習可能)
    patch_embed = _draft_lut('patch_embed', in_width, embed_width)
    # PositionalEncoding: 位置コード定数線 + 連結(positional.py 規約)
    pe_const = _const_bits(
        'pe_const', _position_bits_flat(seq_len, pos_bits, 'binary'),
        train_info={'from': 'LogicPositionalEncoding', 'code': 'binary',
                    'seq_len': seq_len},
    )
    pe_concat = _concat('pe_concat', [embed_width, pos_width])
    pe_concat['train_info'] = {'from': 'LogicPositionalEncoding'}

    nodes: list[dict] = [patch_embed, pe_const, pe_concat]
    edges: list[dict] = [
        {'src': 'x', 'dst': 'patch_embed'},
        {'src': 'patch_embed', 'dst': 'pe_concat', 'dst_slot': 0},
        {'src': 'pe_const', 'dst': 'pe_concat', 'dst_slot': 1},
    ]

    # Transformerブロック×N(blackbox、幅不変 token_width→token_width)
    prev = 'pe_concat'
    for i in range(LLVIT_DEPTH):
        tid = f'transformer{i}'
        nodes.append(_transformer_blackbox(
            tid, seq_len, d_model, LLVIT_N_HEADS,
            detail=f'LogicTransformerBlock d={d_model} h={LLVIT_N_HEADS} #{i}'))
        edges.append({'src': prev, 'dst': tid})
        prev = tid

    # 分類ヘッド: lut_layer(token_width → 10クラス×6) + GroupSum(10)
    head = _draft_lut('head', token_width, CLASSIFIER_WIDTH)
    gs = _group_sum('classes', CLASSIFIER_WIDTH, k=10)
    nodes.extend([head, gs])
    edges.extend([
        {'src': prev, 'dst': 'head'},
        {'src': 'head', 'dst': 'classes'},
        {'src': 'classes', 'dst': 'y'},
    ])

    graph = _graph('llvit', in_width, 10, nodes, edges, out_semantics='class_scores')
    return {
        'networks': {'Main': graph},
        'train_configs': {},
    }


# ---------------------------------------------------------------------------
# テンプレ登録簿(単一情報源)
# ---------------------------------------------------------------------------

class TemplateDef:
    """テンプレ1件の定義。verification は腐敗防止CIの水準に一致する。

    verification:
        'smoke'    — validate + 学習スモークで hard acc 下限を満たす
        'build'    — validate + build(forward有限)まで
        'transfer' — validate + build + 重み転写(enc)が機能
        'validate' — validate のみ(blackbox など build/学習不可)
    """

    def __init__(self, id: str, name: str, description: str, category: str,
                 verification: str, builder: Callable[[], dict],
                 recommended_dataset: str | None = None):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.verification = verification
        self.builder = builder
        self.recommended_dataset = recommended_dataset


TEMPLATES: list[TemplateDef] = [
    TemplateDef(
        'logreg', 'ロジスティック回帰',
        '784→LutLayer→GroupSum(10)の1層直結分類器。MNIST向けの最小モデル。',
        'basic', 'smoke', _build_logreg, 'mnist',
    ),
    TemplateDef(
        'mlp', 'MLP(多層パーセプトロン)',
        '784→LutLayer×2→GroupSum(10)。隠れ層2段の分類器。',
        'basic', 'smoke', _build_mlp, 'mnist',
    ),
    TemplateDef(
        'lenet', 'LeNet風CNN',
        'conv2d→OrPool2d→conv2d→OrPool2d→LutLayer→GroupSum(10)。'
        '論理畳み込みによる画像分類。',
        'vision', 'smoke', _build_lenet, 'mnist',
    ),
    TemplateDef(
        'autoencoder', 'AutoEncoder(重み転写)',
        '学習ネット(enc+dec)と推論ネット(enc)の2ネット構成。学習後に'
        'POST /api/networks/transfer で enc の重みを推論ネットへ転写する。',
        'advanced', 'transfer', _build_autoencoder, None,
    ),
    TemplateDef(
        'lstm', 'LSTM系列',
        f'LSTMセルを{LSTM_UNROLL}回時間展開(unroll)した系列モデル。'
        'ゲートはLutLayer、状態更新は固定ブール論理で構成。',
        'advanced', 'build', _build_lstm, None,
    ),
    TemplateDef(
        'transformer', 'Transformer(blackbox)',
        'LogicTransformerBlockのblackboxノード。エディタ上はvalidateのみ'
        '(学習はPython import経路、HDL生成は不可)。',
        'advanced', 'validate', _build_transformer, None,
    ),
    TemplateDef(
        'llvit', 'LL-ViT(複合)',
        f'パッチ埋込(lut_layer)→PositionalEncoding(constant+concat)→'
        f'LogicTransformerBlock(blackbox)×{LLVIT_DEPTH}→分類ヘッド(lut_layer+GroupSum)。'
        '入力はビット化済みパッチ列(画像→パッチ→ビット化の前段はexperiments/train_llvit.py'
        'の責務)。blackboxを含むためエディタ上はvalidateのみ、学習はPython import経路。',
        'advanced', 'validate', _build_llvit, None,
    ),
]

TEMPLATES_BY_ID = {t.id: t for t in TEMPLATES}


def _template_summary(t: TemplateDef) -> dict:
    """一覧用サマリ(IRの組み立ては行わない軽量メタ)。"""
    return {
        'id': t.id,
        'name': t.name,
        'description': t.description,
        'category': t.category,
        'verification': t.verification,
        'recommended_dataset': t.recommended_dataset,
    }


def build_template_project(template_id: str, *, name: str | None = None) -> ProjectSchema:
    """テンプレを ProjectSchema として組み立てる(保存は呼び出し側)。"""
    t = TEMPLATES_BY_ID.get(template_id)
    if t is None:
        raise KeyError(template_id)
    built = t.builder()
    now = time.time()
    return ProjectSchema(
        id=uuid.uuid4().hex,
        name=name or t.name,
        description=t.description,
        networks=built['networks'],
        train_configs=built.get('train_configs', {}),
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

class TemplateSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    name: str
    description: str
    category: str
    verification: str
    recommended_dataset: str | None = None


class TemplateListResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    templates: list[TemplateSummary]


class InstantiateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str | None = None


@router.get('/api/templates', response_model=TemplateListResponse)
def get_templates() -> TemplateListResponse:
    """テンプレ一覧を返す(GET一覧ではIRを組み立てない)。"""
    return TemplateListResponse(
        templates=[TemplateSummary(**_template_summary(t)) for t in TEMPLATES],
    )


@router.post('/api/templates/{template_id}/instantiate')
def instantiate_template(template_id: str, req: InstantiateRequest):
    """テンプレを ProjectSchema として生成・保存し、project payload を返す。"""
    try:
        project = build_template_project(template_id, name=req.name)
    except KeyError as e:
        raise HTTPException(
            status_code=404, detail=f'テンプレが見つかりません: {template_id}',
        ) from e
    save_project(project, PROJECTS_ROOT)
    return project.model_dump(mode='json')
