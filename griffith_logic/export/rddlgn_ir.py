"""RDDLGN(models/rddlgn.py) → IR抽出(デコーダ追加)。

export/extract.py の直列走査(export_ir)はnn.Sequentialを前提にしており、
RDDLGNのリカレント構造(N/K/L/P-layersのタイムステップループ)を表現できない。
ここでは専用の抽出関数を用意し、IRの`recurrent`ノード(ir/schema.py)へ
落とす。

設計の要点:
    - bodyはunrollしない(重み複製による配信サイズ爆発を避ける。
      「bodyを1回記述し、ランタイムがsteps回実行する」ONNX Scan型)
    - N-layers/L-layers(タイムステップ間で重み共有・状態を持たない
      フィードフォワード)は state 0個の recurrent(output_mode='stacked')
      として表現する(weight共有を保つため。重複エクスポートしない)
    - K-layers(エンコーダのリカレント本体)は state 1個 +
      output_mode='final_state'
    - P-layers/M-layers/GroupSum(デコーダの自己回帰部)は
      1つのrecurrentへ畳む: state=p_prev(P-layer出力)、
      invariant=context(エンコーダ出力。全tickで共通)、
      step=l_t(L-layers出力の該当tick分)、output_mode='stacked'
      (GroupSum後の整数スコア行を全tickぶんstack)。group_sumがbody内に
      現れるのはこれが初(N/K/L-layersのbodyはlut_layer/concatのみ)
    - LogicEmbedding(トークンID→ビット列の境界)はIR化しない(BOUNDARY)。
      export_rddlgn_encoder_ir/export_rddlgn_classifier_irのIR入力ポートは
      「埋め込み・Heaviside済みビット列[seq_len*d_emb]平坦」を受ける契約。
      export_rddlgn_ir(seq2seq全体)はトップレベル入力ポートが
      1本という既存不変条件(deploy/plan.py::build_plan)に合わせ、
      srcとtgt_inputの埋め込みビット列を連結した1本の平坦ベクトルを受け、
      グラフ内で恒等選択lut_layer(_slice_lut)により2つへ分割する
"""

import numpy as np

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex
from ..models.rddlgn import (
    RDDLGN,
    LogicEmbedding,
    LutStack,
    RDDLGNClassifier,
    RDDLGNDecoder,
    RDDLGNEncoder,
)
from .discretize import truth_tables_from_lut_layer
from .extract import _IDENTITY_TT_1IN
from .rddlgn_optimize import reduce_lutstack_nodes

__all__ = ['export_rddlgn_encoder_ir', 'export_rddlgn_classifier_ir',
           'export_rddlgn_ir', 'export_rddlgn_decoder_step_ir',
           'extract_embed_table', 'simulate_rddlgn_greedy',
           'simulate_rddlgn_greedy_bits', 'extract_folded_n_table',
           'extract_folded_l_table', 'extract_folded_table',
           'should_fold_vocab_table']


def _lutstack_nodes(stack: LutStack, id_prefix: str, entry_id: str,
                    entry_width: int) -> tuple[list[IRNode], list[IREdge],
                                                str, int]:
    """LutStackの各LutLayerをlut_layer IRノード化し、entry_idから直列に配線する。

    export/extract.py::_extract_lut_layerと同じ抽出ロジック
    (truth_tables_from_lut_layerで全パターンを流して真理値表を確定する)を、
    LutStack内の複数LutLayerに対して繰り返し適用する共通ヘルパ。

    Returns:
        (nodes, edges, 最終ノードid, 出力幅)
    """
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id = entry_id
    prev_width = entry_width
    for i, layer in enumerate(stack.layers):
        assert layer.in_dim == prev_width, (
            f'{id_prefix}{i}: in_dim({layer.in_dim}) != 前段幅({prev_width})'
        )
        node_id = f'{id_prefix}{i}'
        tables = truth_tables_from_lut_layer(layer)
        nodes.append(IRNode(
            id=node_id,
            type='lut_layer',
            params={
                'in_width': layer.in_dim,
                'out_width': layer.out_dim,
                'n_inputs': layer.n_inputs,
                'indices': layer.indices.cpu().numpy().astype(np.int64),
                'truth_tables': tt_to_hex(tables.tolist()),
            },
            train_info={
                'parameterization': layer.parameterization,
                'selection': layer.selection,
            },
        ))
        edges.append(IREdge(src=prev_id, dst=node_id))
        prev_id = node_id
        prev_width = layer.out_dim
    return nodes, edges, prev_id, prev_width


def _build_lutstack(stack: LutStack, id_prefix: str, entry_id: str,
                    entry_width: int, *, optimize_degenerate: bool
                    ) -> tuple[list[IRNode], list[IREdge], str, int]:
    """`_lutstack_nodes` / `reduce_lutstack_nodes`を出し分ける薄い分岐。

    optimize_degenerate=False(既定)なら従来通り`_lutstack_nodes`を使う
    (既存出力がバイト一致で不変)。Trueなら退化ゲート除去パス
    (export/rddlgn_optimize.py)を通す。戻り値の契約
    (nodes, edges, 最終ノードid, 出力幅)はどちらも同じ。
    """
    if optimize_degenerate:
        return reduce_lutstack_nodes(stack, id_prefix, entry_id, entry_width)
    return _lutstack_nodes(stack, id_prefix, entry_id, entry_width)


def _init_state_bits(init_mode: str, width: int) -> list[int]:
    """K/P-layerのstate初期化bitsを確定する(hard eval規則と同一)。

    'zero'(既定): sigmoid(0)=0.5固定 → hard evalの閾値規則(x>0.5→1、
        タイブレーク0.5→0)でちょうど0.5は0側に倒れるので全0。
    'one': 1.0固定 → 全1。
    'gaussian': hard evalの初期状態が**推論のたびに違う乱数**になり
        (models/rddlgn.py::_init_hiddenがtorch.randnをその場で引く)、
        入力にのみ依存する決定的回路として表現できない。IR化は原理的に不可。
    """
    if init_mode == 'zero':
        return [0] * width
    if init_mode == 'one':
        return [1] * width
    if init_mode == 'gaussian':
        raise NotImplementedError(
            "init_mode='gaussian'はhard eval時の初期状態が入力非依存の乱数"
            '(models/rddlgn.py::_init_hiddenがforwardのたびにtorch.randnを'
            '引く)であり、固定された入力→固定された出力の決定的回路である'
            'IRには表現できない。デプロイするには init_mode を zero/one'
            'へ変更した学習済みモデルを使うこと'
        )
    raise NotImplementedError(f'init_mode={init_mode!r} は未対応')


def export_rddlgn_encoder_ir(
    encoder: RDDLGNEncoder,
    seq_len: int,
    *,
    id_prefix: str = 'enc_',
    optimize_degenerate: bool = False,
) -> tuple[IRGraph, int, int]:
    """RDDLGNEncoderをIR化する(N-layers→K-layersの2つのrecurrentノード)。

    Args:
        encoder: 抽出対象(.eval()済みであること。抽出自体はeval()を強制
            しないが、truth_tables_from_lut_layerがLutLayerをdeepcopyして
            probe評価するため、encoder自体のmode(train/eval)は結果に
            影響しない — 学習可能配線等はindices/重みに既に確定している)
        seq_len: 系列長(RDDLGNEncoderは構造上seq_len非依存だが、IRの
            recurrentノードはsteps回数を固定値として焼き込む必要がある)
        id_prefix: ノードidの接頭辞(classifier全体へ埋め込むとき用)
        optimize_degenerate: Trueなら退化ゲート除去パス
            (export/rddlgn_optimize.py)を通す。既定Falseで既存出力を
            バイト一致で保つ(opt-in)

    Returns:
        (encoder_ir, d_emb, k_out_dim) — encoder_irは
        入力ポート x: [seq_len*d_emb]平坦(埋め込み・Heaviside済み)、
        出力ポート y: [k_out_dim](エンコーダcontext)を持つIRGraph
    """
    n_layers = encoder.n_layers
    k_layers = encoder.k_layers
    d_emb = n_layers.layers[0].in_dim
    n_out = n_layers.out_dim
    k_out = encoder.k_out_dim

    # ---- N-layers: state 0個のrecurrent(output_mode='stacked')。----
    # 全タイムステップに同じ重みを適用してstackするだけ(重み共有を保つ)。
    n_sv = IRNode(id='sv_step', type='scan_var',
                  params={'width': d_emb, 'role': 'step'})
    n_nodes, n_edges, n_last, n_last_w = _build_lutstack(
        n_layers, 'lut', 'sv_step', d_emb,
        optimize_degenerate=optimize_degenerate)
    assert n_last_w == n_out
    n_body = IRGraph(nodes=[n_sv] + n_nodes, edges=n_edges)
    n_rec = IRNode(
        id=f'{id_prefix}n_rec',
        type='recurrent',
        params={
            'steps': int(seq_len),
            'body': n_body,
            'scan_vars': [{'id': 'sv_step', 'role': 'step', 'width': d_emb}],
            'state_update': {},
            'output_mode': 'stacked',
            'output_source_id': n_last,
            'output_width': n_out,
        },
    )

    # ---- K-layers: state 1個(k_prev)+recurrent(output_mode='final_state')。----
    init_bits = _init_state_bits(encoder.init_mode, k_out)
    k_sv_state = IRNode(id='sv_state', type='scan_var', params={
        'width': k_out, 'role': 'state',
        'init': {'kind': 'const', 'bits': init_bits},
    })
    k_sv_step = IRNode(id='sv_step', type='scan_var',
                       params={'width': n_out, 'role': 'step'})
    # k_input = cat([h_t, k_prev]): slot0=step(h_t), slot1=state(k_prev)
    # (models/rddlgn.py::RDDLGNEncoder.forwardのtorch.cat順と一致させる)
    k_concat = IRNode(id='concat0', type='concat',
                      params={'in_widths': [n_out, k_out]})
    k_lut_nodes, k_lut_edges, k_last, k_last_w = _build_lutstack(
        k_layers, 'lut', 'concat0', n_out + k_out,
        optimize_degenerate=optimize_degenerate)
    assert k_last_w == k_out
    k_body = IRGraph(
        nodes=[k_sv_state, k_sv_step, k_concat] + k_lut_nodes,
        edges=[
            IREdge(src='sv_step', dst='concat0', dst_slot=0),
            IREdge(src='sv_state', dst='concat0', dst_slot=1),
            *k_lut_edges,
        ],
    )
    k_rec = IRNode(
        id=f'{id_prefix}k_rec',
        type='recurrent',
        params={
            'steps': int(seq_len),
            'body': k_body,
            'scan_vars': [
                {'id': 'sv_state', 'role': 'state', 'width': k_out},
                {'id': 'sv_step', 'role': 'step', 'width': n_out},
            ],
            'state_update': {'sv_state': k_last},
            'output_mode': 'final_state',
            'output_source_id': k_last,
            'output_width': k_out,
        },
    )

    in_width = int(seq_len) * d_emb
    graph = IRGraph(
        metadata={'name': 'rddlgn_encoder'},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k_out)],
        nodes=[n_rec, k_rec],
        edges=[
            IREdge(src='x', dst=n_rec.id),
            IREdge(src=n_rec.id, dst=k_rec.id),
            IREdge(src=k_rec.id, dst='y'),
        ],
    )
    return graph, d_emb, k_out


def export_rddlgn_classifier_ir(
    model: RDDLGNClassifier,
    seq_len: int,
    *,
    name: str = 'rddlgn_classifier',
    optimize_degenerate: bool = False,
) -> IRGraph:
    """RDDLGNClassifier(エンコーダ+分類ヘッド)をIR化する。

    構成(models/rddlgn.py::RDDLGNClassifier.forward参照):
        embedding(BOUNDARY、IR化しない) → encoder(2つのrecurrent)
        → head(LutStack) → group_sum(k=n_classes)

    Args:
        model: 抽出対象
        seq_len: 系列長(トークン列の長さ。IRのrecurrent.stepsに焼き込む)
        name: metadata.name

    Returns:
        入力ポート x: [seq_len*d_emb]平坦(埋め込み・Heaviside済みビット列)、
        出力ポート y: [n_classes](GroupSumの整数スコア)を持つIRGraph
    """
    encoder_ir, d_emb, k_out = export_rddlgn_encoder_ir(
        model.encoder, seq_len, id_prefix='enc_',
        optimize_degenerate=optimize_degenerate)

    head_nodes, head_edges, head_last, head_last_w = _build_lutstack(
        model.head, 'head_lut', encoder_ir.nodes[-1].id, k_out,
        optimize_degenerate=optimize_degenerate)

    gs = model.group_sum
    gs_node = IRNode(
        id='group_sum',
        type='group_sum',
        params={'in_width': head_last_w, 'k': gs.k, 'tau': float(gs.tau)},
    )

    # encoder_ir自身のエッジは x/y ポートも含む3本
    # (x->n_rec, n_rec->k_rec, k_rec->y)。分類器グラフではencoder出力を
    # headへ直結するので、末尾(k_rec->y)だけ落として引き継ぐ。
    encoder_internal_edges = encoder_ir.edges[:-1]

    in_width = int(seq_len) * d_emb
    graph = IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=gs.k, semantics='class_scores')],
        nodes=[*encoder_ir.nodes, *head_nodes, gs_node],
        edges=[
            *encoder_internal_edges,
            *head_edges,
            IREdge(src=head_last, dst='group_sum'),
            IREdge(src='group_sum', dst='y'),
        ],
    )
    return graph


# ---------------------------------------------------------------------------
# デコーダ・seq2seq全体
# ---------------------------------------------------------------------------

def _slice_lut(node_id: str, in_width: int, start: int, width: int) -> IRNode:
    """入力ビット列の連続範囲 [start, start+width) を恒等選択するlut_layer。

    トップレベル入力ポートは1本のみという既存不変条件(deploy/plan.py::
    build_plan の `len(graph.inputs) != 1` チェック)があるため、
    src埋め込みビット列とtgt埋め込みビット列を連結した1本の平坦入力から
    各部分列を取り出すのに使う。export/lowering.py の
    `_permute_node_to_lut` / `_upsample2d_node_to_lut` と同じ手口
    (n_inputs=1・恒等真理値表 `_IDENTITY_TT_1IN`=0x2)で、新しいノード型は
    要らない。
    """
    indices = np.arange(start, start + width, dtype=np.int64).reshape(1, width)
    return IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': width,
            'n_inputs': 1,
            'indices': indices,
            'truth_tables': tt_to_hex([_IDENTITY_TT_1IN] * width),
        },
        train_info={'lowered_from': 'InputSlice'},
    )


def _decoder_l_recurrent(l_layers: LutStack, tgt_len: int, d_emb: int,
                         *, id_prefix: str = 'dec_',
                         optimize_degenerate: bool = False
                         ) -> tuple[IRNode, int]:
    """L-layers: state 0個のrecurrent(output_mode='stacked')。

    export_rddlgn_encoder_irのN-layers抽出と同型(全タイムステップに
    同じ重みを適用してstackするだけ。重み共有を保つため複製しない)。
    """
    sv_step = IRNode(id='sv_step', type='scan_var',
                     params={'width': d_emb, 'role': 'step'})
    nodes, edges, last, last_w = _build_lutstack(
        l_layers, 'lut', 'sv_step', d_emb,
        optimize_degenerate=optimize_degenerate)
    body = IRGraph(nodes=[sv_step] + nodes, edges=edges)
    rec = IRNode(
        id=f'{id_prefix}l_rec',
        type='recurrent',
        params={
            'steps': int(tgt_len),
            'body': body,
            'scan_vars': [{'id': 'sv_step', 'role': 'step', 'width': d_emb}],
            'state_update': {},
            'output_mode': 'stacked',
            'output_source_id': last,
            'output_width': last_w,
        },
    )
    return rec, last_w


def _decoder_pm_recurrent(
    decoder: RDDLGNDecoder, tgt_len: int, context_dim: int,
    *, id_prefix: str = 'dec_', optimize_degenerate: bool = False,
) -> tuple[IRNode, int, int, int]:
    """P-layers + M-layers + GroupSumを1つのrecurrent bodyへ畳む。

    models/rddlgn.py::RDDLGNDecoder.forwardの自己回帰ループ本体
    (P-layer→M-layer→GroupSumを1tickぶん)と一致させる:

        p_input = cat([p_prev, context, l_t])   # P-layers(state=p_prev)
        p_t = P(p_input)
        m_input = cat([p_t, context, l_t])      # M-layers(concat順は同型)
        m_t = M(m_input)
        logits_t = group_sum(m_t)               # GroupSum(k=vocab, tau)

    group_sumがrecurrentのbody内に現れるのはこれが初(N/K-layers
    bodyはlut_layer/concatのみだった)。simulate側は_simulate_valuesが
    ノード型を汎用的に扱うため無改修で動くが、deploy/plan.py側は
    `_plan_recurrent`がrecurrent自身の出力値種別を無条件にKIND_FLATへ
    決め打ちしていたため、body内でgroup_sum(KIND_SCORES=int32)を経由する
    今回のケースだけ値種別が壊れる。plan.py側で対応する(このタスクの
    「group_sum-in-bodyで必要になった対応」)。

    Returns:
        (recurrentノード, p_out, l_out, vocab_size)
    """
    p_layers = decoder.p_layers
    m_layers = decoder.m_layers
    l_out = decoder.l_layers.out_dim
    p_out = decoder.p_out_dim
    vocab = int(decoder.group_sum.k)

    sv_p = IRNode(id='sv_p', type='scan_var', params={
        'width': p_out, 'role': 'state',
        'init': {'kind': 'const',
                'bits': _init_state_bits(decoder.init_mode, p_out)},
    })
    sv_ctx = IRNode(id='sv_context', type='scan_var',
                    params={'width': context_dim, 'role': 'invariant'})
    sv_l = IRNode(id='sv_l', type='scan_var',
                 params={'width': l_out, 'role': 'step'})

    in_widths = [p_out, context_dim, l_out]
    p_concat = IRNode(id='p_concat', type='concat',
                      params={'in_widths': list(in_widths)})
    p_nodes, p_edges, p_last, p_last_w = _build_lutstack(
        p_layers, 'p_lut', 'p_concat', sum(in_widths),
        optimize_degenerate=optimize_degenerate)
    assert p_last_w == p_out, (
        f'P-layersの出力幅({p_last_w})がp_out_dim({p_out})と不一致')

    m_concat = IRNode(id='m_concat', type='concat',
                      params={'in_widths': list(in_widths)})
    m_nodes, m_edges, m_last, m_last_w = _build_lutstack(
        m_layers, 'm_lut', 'm_concat', sum(in_widths),
        optimize_degenerate=optimize_degenerate)

    gs_node = IRNode(
        id='group_sum', type='group_sum',
        params={'in_width': m_last_w, 'k': vocab,
               'tau': float(decoder.group_sum.tau)},
    )

    body = IRGraph(
        nodes=[sv_p, sv_ctx, sv_l, p_concat, *p_nodes, m_concat, *m_nodes,
              gs_node],
        edges=[
            IREdge(src='sv_p', dst='p_concat', dst_slot=0),
            IREdge(src='sv_context', dst='p_concat', dst_slot=1),
            IREdge(src='sv_l', dst='p_concat', dst_slot=2),
            *p_edges,
            IREdge(src=p_last, dst='m_concat', dst_slot=0),
            IREdge(src='sv_context', dst='m_concat', dst_slot=1),
            IREdge(src='sv_l', dst='m_concat', dst_slot=2),
            *m_edges,
            IREdge(src=m_last, dst='group_sum'),
        ],
    )
    rec = IRNode(
        id=f'{id_prefix}pm_rec',
        type='recurrent',
        params={
            'steps': int(tgt_len),
            'body': body,
            # 親からの入力エッジ(dst_slot昇順)はrole∈{step,invariant}の
            # 並びに1対1対応する(schema.py規約)。ここでは
            # [sv_context(invariant), sv_l(step)]の順(sv_pはstateなので
            # 親からの外部入力には現れない)。
            'scan_vars': [
                {'id': 'sv_p', 'role': 'state', 'width': p_out},
                {'id': 'sv_context', 'role': 'invariant', 'width': context_dim},
                {'id': 'sv_l', 'role': 'step', 'width': l_out},
            ],
            'state_update': {'sv_p': p_last},
            'output_mode': 'stacked',
            'output_source_id': 'group_sum',
            'output_width': vocab,
        },
    )
    return rec, p_out, l_out, vocab


def export_rddlgn_ir(
    model: RDDLGN,
    src_len: int,
    tgt_len: int,
    *,
    name: str = 'rddlgn_seq2seq',
    optimize_degenerate: bool = False,
) -> IRGraph:
    """RDDLGN(教師強制seq2seq全体)をIR化する。

    構成(models/rddlgn.py::RDDLGN.forward参照。embeddingはBOUNDARYなので
    IR化しない):
        入力x = concat([src埋め込みビット列, tgt_input埋め込みビット列])
          → x_src/x_tgt(恒等選択lut_layerで分割)
          → encoder(M1のexport_rddlgn_encoder_irをそのまま再利用。
            N-layers + K-layers → context)
          → decoder: L-layers(state0 recurrent、stacked)
                    → P+M+GroupSumを1つのrecurrent bodyへ畳んだノード
                      (state=p_prev、invariant=context、step=l_t)

    Args:
        model: 抽出対象(.eval()済みであること。理由はexport_rddlgn_encoder_ir
            と同じ — truth_tables_from_lut_layerがLutLayerをdeepcopyして
            probe評価するため、model自体のmode(train/eval)は結果に影響しない)
        src_len: ソース系列長(recurrent.stepsに焼き込む)
        tgt_len: ターゲット系列長(教師強制。同じくrecurrent.stepsへ焼き込む)
        name: metadata.name
        optimize_degenerate: Trueなら退化ゲート除去パスを通す
            (既定Falseで既存出力をバイト一致で保つ、opt-in)

    Returns:
        入力ポート x: [src_len*d_emb + tgt_len*d_emb]平坦
            (src埋め込み・tgt_input埋め込みを連結。ともにHeaviside済みビット列。
            encoding='binary_threshold_0.5')、
        出力ポート y: [tgt_len*vocab_size](教師強制の全tickぶんGroupSum
            整数スコアをstackしたもの)を持つIRGraph
    """
    d_emb = model.src_embedding.embedding.embedding_dim
    src_width = int(src_len) * d_emb
    tgt_width = int(tgt_len) * d_emb
    total_width = src_width + tgt_width

    x_src = _slice_lut('x_src', total_width, 0, src_width)
    x_tgt = _slice_lut('x_tgt', total_width, src_width, tgt_width)

    encoder_ir, _, k_out = export_rddlgn_encoder_ir(
        model.encoder, src_len, id_prefix='enc_',
        optimize_degenerate=optimize_degenerate)
    # encoder_ir自身のエッジは x/y ポートも含む3本
    # (x->n_rec, n_rec->k_rec, k_rec->y)。x_srcへ差し替えて引き継ぐ
    # (末尾k_rec->yは落とす。export_rddlgn_classifier_irと同じ手口)。
    encoder_internal_edges = [
        IREdge(src=('x_src' if e.src == 'x' else e.src), dst=e.dst,
              dst_slot=e.dst_slot)
        for e in encoder_ir.edges[:-1]
    ]
    context_id = encoder_ir.nodes[-1].id   # k_rec

    l_rec, l_out = _decoder_l_recurrent(
        model.decoder.l_layers, tgt_len, d_emb, id_prefix='dec_',
        optimize_degenerate=optimize_degenerate)
    pm_rec, p_out, l_out2, vocab = _decoder_pm_recurrent(
        model.decoder, tgt_len, k_out, id_prefix='dec_',
        optimize_degenerate=optimize_degenerate)
    assert l_out2 == l_out, (
        f'L-layers出力幅がdecoder内部で不一致(l_rec={l_out}, pm={l_out2})')

    graph = IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=total_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=int(tgt_len) * vocab,
                        semantics='class_scores')],
        nodes=[x_src, x_tgt, *encoder_ir.nodes, l_rec, pm_rec],
        edges=[
            IREdge(src='x', dst='x_src'),
            IREdge(src='x', dst='x_tgt'),
            *encoder_internal_edges,
            IREdge(src='x_tgt', dst=l_rec.id),
            IREdge(src=context_id, dst=pm_rec.id, dst_slot=0),
            IREdge(src=l_rec.id, dst=pm_rec.id, dst_slot=1),
            IREdge(src=pm_rec.id, dst='y'),
        ],
    )
    return graph


def export_rddlgn_decoder_step_ir(
    model: RDDLGN,
    *,
    include_l: bool = True,
    name: str = 'rddlgn_decoder_step',
    optimize_degenerate: bool = False,
) -> tuple[IRGraph, dict]:
    """RDDLGN デコーダの 1 ステップ分を組み合わせ IR にする。

    FPGA 側の再帰ステップラッパ(状態レジスタ + 帰還)で自己回帰生成を回すための形で、
    契約は docs/reference/step-ir-contract.md(`export/step_contract.py::check_step_ir`):

        入力 x = [外部入力 | 状態] = [emb_t(include_l)または l_t | context | p_prev]
        出力 y = [次状態 | 読み出し] = [p_t | m_t]

    m_t は M-layers の最終層ビットで、GroupSum は置かない(出力が group_sum だと状態ビットが
    取れない。クラス和の popcount・argmax・次トークンの埋め込み ROM・EOS 停止はラッパ側)。
    1 ステップの本体は models/rddlgn.py::RDDLGNDecoder.forward の 1 tick と同じ:

        l_t = L(emb_t)                          (include_l=True。False なら l_t を外から受ける)
        p_t = P(cat([p_prev, context, l_t]))
        m_t = M(cat([p_t, context, l_t]))

    `export_rddlgn_ir` / `_decoder_pm_recurrent`(recurrent ノード版、simulate 用)は変えない。
    既存の `export_rddlgn_encoder_ir` の出力形も不変(FPGA 側の recurrent_encoder_to_step の前提)。

    Args:
        model: .eval() 済み RDDLGN(init_mode は zero / one。gaussian は不可)
        include_l: True なら L-layers を含め外部入力を埋め込みビット列 emb_t(幅 d_emb)にする。
            False なら外部入力は L-layers の出力 l_t(幅 l_out)
        name: metadata.name
        optimize_degenerate: 退化ゲート除去パスを通すか

    Returns:
        (graph, spec)。spec は FPGA 側へ渡す付帯情報:
            ext_width / state_width / readout_width、state_in / state_out(契約の区間)、
            init_bits(p_prev の初期値)、layout(x の各部分の区間)、
            group_sum(k / group_size / tau。読み出しビットからクラススコアを作る規則)
    """
    decoder = model.decoder
    d_emb = model.tgt_embedding.embedding.embedding_dim
    context_dim = int(model.config.k_layer_sizes[-1])
    l_out = int(decoder.l_layers.out_dim)
    p_out = int(decoder.p_out_dim)
    pfx = 'dec_step_'

    first_w = d_emb if include_l else l_out
    in_width = first_w + context_dim + p_out
    x_first = _slice_lut(f'{pfx}x_emb' if include_l else f'{pfx}x_l', in_width, 0, first_w)
    x_ctx = _slice_lut(f'{pfx}x_ctx', in_width, first_w, context_dim)
    x_p = _slice_lut(f'{pfx}x_p', in_width, first_w + context_dim, p_out)
    nodes: list[IRNode] = [x_first, x_ctx, x_p]
    edges: list[IREdge] = [IREdge(src='x', dst=n.id) for n in nodes]

    if include_l:
        l_nodes, l_edges, l_last, l_w = _build_lutstack(
            decoder.l_layers, f'{pfx}l_lut', x_first.id, d_emb,
            optimize_degenerate=optimize_degenerate)
        nodes += l_nodes
        edges += l_edges
        assert l_w == l_out, (l_w, l_out)
    else:
        l_last, l_w = x_first.id, l_out

    in_widths = [p_out, context_dim, l_w]
    p_concat = IRNode(id=f'{pfx}p_concat', type='concat', params={'in_widths': list(in_widths)})
    p_nodes, p_edges, p_last, p_w = _build_lutstack(
        decoder.p_layers, f'{pfx}p_lut', p_concat.id, sum(in_widths),
        optimize_degenerate=optimize_degenerate)
    assert p_w == p_out, (f'P-layersの出力幅({p_w})がp_out_dim({p_out})と不一致')
    nodes += [p_concat, *p_nodes]
    edges += [
        IREdge(src=x_p.id, dst=p_concat.id, dst_slot=0),
        IREdge(src=x_ctx.id, dst=p_concat.id, dst_slot=1),
        IREdge(src=l_last, dst=p_concat.id, dst_slot=2),
        *p_edges,
    ]

    m_concat = IRNode(id=f'{pfx}m_concat', type='concat', params={'in_widths': list(in_widths)})
    m_nodes, m_edges, m_last, m_w = _build_lutstack(
        decoder.m_layers, f'{pfx}m_lut', m_concat.id, sum(in_widths),
        optimize_degenerate=optimize_degenerate)
    nodes += [m_concat, *m_nodes]
    edges += [
        IREdge(src=p_last, dst=m_concat.id, dst_slot=0),
        IREdge(src=x_ctx.id, dst=m_concat.id, dst_slot=1),
        IREdge(src=l_last, dst=m_concat.id, dst_slot=2),
        *m_edges,
    ]

    out = IRNode(id=f'{pfx}out', type='concat', params={'in_widths': [p_out, m_w]})
    nodes.append(out)
    edges += [
        IREdge(src=p_last, dst=out.id, dst_slot=0),
        IREdge(src=m_last, dst=out.id, dst_slot=1),
        IREdge(src=out.id, dst='y'),
    ]

    graph = IRGraph(
        metadata={'name': name, 'step_contract': 'docs/reference/step-ir-contract.md'},
        inputs=[IRPort(id='x', width=in_width, encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=p_out + m_w)],
        nodes=nodes,
        edges=edges,
    )
    vocab = int(decoder.group_sum.k)
    spec = {
        'ext_width': first_w + context_dim,
        'state_width': p_out,
        'readout_width': m_w,
        'state_in': [first_w + context_dim, in_width],
        'state_out': [0, p_out],
        'init_bits': _init_state_bits(decoder.init_mode, p_out),
        'layout': {
            ('emb' if include_l else 'l'): [0, first_w],
            'context': [first_w, first_w + context_dim],
            'state': [first_w + context_dim, in_width],
        },
        'group_sum': {'k': vocab, 'group_size': m_w // vocab,
                      'tau': float(decoder.group_sum.tau),
                      'assign': getattr(decoder.group_sum, 'assign', 'contig')},
    }
    return graph, spec


# ---------------------------------------------------------------------------
# 自己回帰生成
# ---------------------------------------------------------------------------

def _embed_hard_bits(embedding: LogicEmbedding) -> np.ndarray:
    """LogicEmbedding.forwardのeval経路と同じHeaviside規則で埋め込み表を確定する。

    models/rddlgn.py::LogicEmbedding.forward の eval 経路
    (`(sigmoid(e) > 0.5)`、strict `>`、ちょうど0.5は0側)を
    embedding.weight全体に一括適用する。

    Returns:
        (vocab, d_emb) の0/1(uint8)配列
    """
    import torch

    with torch.no_grad():
        bits = (torch.sigmoid(embedding.embedding.weight) > 0.5)
    return bits.to(torch.uint8).cpu().numpy()


def _pack_bits_lsb(bits: np.ndarray) -> list[int]:
    """(rows, width)の0/1配列を行ごとに words_per_row(width) 語へパックする。

    runtime/c/include/mdl_kernels.hのビットプレーン契約と同じ並び
    (列xはワードx>>5のビットx&31、LSB-first。端数ビットは0)。
    """
    rows, width = bits.shape
    wpr = (int(width) + 31) // 32
    pad = wpr * 32 - width
    padded = np.pad(bits, ((0, 0), (0, pad))) if pad else bits
    grouped = padded.reshape(rows, wpr, 32).astype(np.uint32)
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))[None, None, :]
    words = (grouped * weights).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    return [int(w) for w in words.reshape(-1)]


def extract_embed_table(embedding: LogicEmbedding) -> dict:
    """埋め込み表をHeaviside済みビット列として配布バンドル用に抽出する。

    自己回帰生成(C側)はトークンID→ビット列の変換を実行時に行う必要が
    あるため(教師強制と違い、tgt側の埋め込みビット列を呼び出し側が
    事前に用意できない)、バンドルの新セクション `embed_table` として
    焼き込む。LogicEmbedding自体はIR化しない(BOUNDARY、モジュール
    docstring参照)ので、ここが埋め込み表を確定させる唯一の場所になる。
    params/バンドルにfloatを残さないプロジェクトの規約に従い、ここで
    Heaviside済みのビット列(0/1)へ確定させてから整数(u32)としてパックする。

    Returns:
        {'vocab': V, 'd_emb': D, 'bits': [u32,...]}。
        bitsは行(トークン)ごとに ceil(D/32) 語、LSB-firstでパックした配列
        (deploy/cbackend.py::_emit_embed_tableがmdl_embed_lookupへ焼き込む)。
    """
    bits = _embed_hard_bits(embedding)
    vocab, d_emb = bits.shape
    return {'vocab': int(vocab), 'd_emb': int(d_emb), 'bits': _pack_bits_lsb(bits)}


# ---------------------------------------------------------------------------
# 語彙折り畳み: Embedding+N/L-layersをトークン表に事前計算する
# ---------------------------------------------------------------------------

def _lutstack_weight_bytes(stack: LutStack) -> int:
    """LutStackがdeploy/cbackend.py::emit_lut_denseで出す重みバイト数を見積もる。

    deploy/plan.py::slice_truth_tables と同じ添字/真理値表の算術を使い、
    実際にCソースへ`static const`配列として出るバイト数と一致させる。
    添字は層の**入力**幅(layer.in_dim、1つ前の層の出力幅)が65536以下なら
    uint16_t(2byte/index)、そうでなければint32_t(4byte/index)——
    cbackend.emit_lut_denseが選ぶのと同じ`lut_dense_indices_fit_u16`
    判定をそのまま使う(折り畳みの勝敗判定を、今のcbackendが実際に
    出すバイト数でやるため)。
    """
    from ..deploy.plan import lut_dense_indices_fit_u16, slice_truth_tables

    total = 0
    for layer in stack.layers:
        tables = truth_tables_from_lut_layer(layer)
        idx_elem_bytes = 2 if lut_dense_indices_fit_u16(layer.in_dim) else 4
        idx_bytes = layer.n_inputs * layer.out_dim * idx_elem_bytes
        tt_words = len(slice_truth_tables(tables.tolist(), layer.n_inputs))
        total += idx_bytes + tt_words * 4
    return total


def should_fold_vocab_table(table_bytes: int, weight_bytes: int, *,
                            force: bool | None = None) -> bool:
    """語彙折り畳みの自動判定。

    表サイズ(vocab×ceil(width/32)×4B)が元のLutStack重みサイズを**下回る**
    ときだけ折り畳む(配信サイズが確実に増えない場合だけ有効化する保守的な
    しきい値。ちょうど等しい境界はfoldしない側=既存動作を保つ側に倒す)。

    Args:
        force: Noneなら自動判定。True/Falseを渡すと判定を無視して強制する
            (実験・デバッグ用。paper_scale1のような大語彙で意図的に
            折り畳みを試したい場合などに使う)。
    """
    if force is not None:
        return bool(force)
    return table_bytes < weight_bytes


def extract_folded_table(stack: LutStack, embed_bits: np.ndarray, *,
                         id_prefix: str) -> dict:
    """LutStackを全vocab行に一括適用し、折り畳み表を確定する。

    export_rddlgn_encoder_ir(N-layers)/_decoder_l_recurrent(L-layers)が
    bodyを組み立てるのと**まったく同じ手口**(`_lutstack_nodes`でsv_step
    からチェーンする)でbodyを作り、`_simulate_values`で全vocab行を
    1回のバッチ呼び出しにより評価する。teacher forcing/greedy生成が
    実際にN/L-layersを評価するのと同じコードパスを流用するので、
    折り畳み表とN/L-layers直接評価の一致は構造的に保証される
    (「折り畳み経路と非折り畳み経路が食い違う」という種類のバグが
    そもそも入りにくい設計)。

    Args:
        stack: 折り畳む対象(encoder.n_layers または decoder.l_layers)
        embed_bits: (vocab, in_width) 0/1(uint8)配列。_embed_hard_bitsの
            出力そのもの(N用はsrc_embedding、L用はtgt_embedding)
        id_prefix: bodyノードidの接頭辞(N/Lで衝突しないよう呼び出し側が
            分ける。このbodyは単体で使い捨てなので他IDと混ざる心配はない)

    Returns:
        {'vocab': V, 'width': W, 'bits': [u32,...]}(embed_tableと同じ
        u32 LSB-firstパック形式。deploy/cbackend.py::_emit_folded_tableが
        mdl_folded_n_lookup/mdl_folded_l_lookupへ焼き込む)
    """
    from ..ir.simulate import _simulate_values

    vocab, in_width = embed_bits.shape
    sv = IRNode(id='sv_step', type='scan_var',
               params={'width': in_width, 'role': 'step'})
    nodes, edges, last, last_w = _lutstack_nodes(
        stack, id_prefix, 'sv_step', in_width)
    body = IRGraph(nodes=[sv] + nodes, edges=edges)
    vals = _simulate_values(body, {'sv_step': embed_bits.astype(np.int64)})
    out_bits = vals[last].astype(np.uint8)
    return {'vocab': int(vocab), 'width': int(last_w),
            'bits': _pack_bits_lsb(out_bits)}


def _decide_and_fold(stack: LutStack, embed_bits: np.ndarray, *,
                     id_prefix: str, force: bool | None) -> dict | None:
    vocab, _ = embed_bits.shape
    out_width = stack.out_dim
    wpr = (out_width + 31) // 32
    table_bytes = vocab * wpr * 4
    weight_bytes = _lutstack_weight_bytes(stack)
    if not should_fold_vocab_table(table_bytes, weight_bytes, force=force):
        return None
    return extract_folded_table(stack, embed_bits, id_prefix=id_prefix)


def extract_folded_n_table(model: RDDLGN, *,
                           force: bool | None = None) -> dict | None:
    """N-layers折り畳み表。

    エンコーダのN-layers(state 0のrecurrent、`export_rddlgn_encoder_ir`
    docstring参照)はトークンIDのみの関数(状態非依存)なので、有限語彙
    なら全トークン分を事前計算して「vocab×N出力幅」の表に焼き込める。
    自動判定(表サイズ≥N-layers重みサイズ)で見送られると`None`を返す —
    呼び出し側(バンドルdocビルダ)は`None`ならbundleへ'folded_n_table'
    節を追加しない、という規約(既存動作を完全に保つ)。

    Returns:
        extract_folded_tableの戻り値、または折り畳まない場合None
    """
    embed = _embed_hard_bits(model.src_embedding)
    return _decide_and_fold(model.encoder.n_layers, embed,
                            id_prefix='fold_n', force=force)


def extract_folded_l_table(model: RDDLGN, *,
                           force: bool | None = None) -> dict | None:
    """L-layers折り畳み表。

    デコーダのL-layersも同じくトークンIDのみの関数。tgt_embeddingを使う
    (shared_vocab=False でもsrc/tgtの表を正しく使い分ける。自己回帰生成
    ハーネスの制約(shared_vocab=True限定、export_rddlgn_ir.pyの
    simulate_rddlgn_greedy docstring参照)とは独立 — 折り畳み表自体は
    教師強制のtgt埋め込みにも一般に使えるため、ここではshared_vocabを
    要求しない)。

    Returns:
        extract_folded_tableの戻り値、または折り畳まない場合None
    """
    embed = _embed_hard_bits(model.tgt_embedding)
    return _decide_and_fold(model.decoder.l_layers, embed,
                            id_prefix='fold_l', force=force)


def simulate_rddlgn_greedy(
    model: RDDLGN, src_tokens, *, bos_token: int, eos_token: int,
    max_new_tokens: int, optimize_degenerate: bool = False,
) -> list[int]:
    """ir/simulateベースの自己回帰貪欲デコード(golden)。

    PyTorch側の対応品は models/rddlgn.py::RDDLGN.generate_greedy。
    両者はtruth_tables_from_lut_layerで確定した**同じ真理値表**を使うため、
    一致すればPyTorch hard evalとの3者一致になる(argmaxのタイブレークは
    どちらも「最初の最大値」= runtime/c mdl_argmaxと同じ規約)。

    トークン列を埋め込み表で確定ビット列へ変換してから
    `simulate_rddlgn_greedy_bits`(実装本体)へ委譲する薄いラッパ。
    C側API(`mdl_rddlgn_generate_greedy`)がsrc自体は「埋め込み済み
    ビット列」で受け取る契約に合わせ、ビット列を主インタフェースにして
    ロジックを1箇所に保つ(テストが全0/全1のような実在トークンに
    対応しない極端なビット列でも突き合わせられるように、というのが
    理由。トークン起点だと表現できない)。

    Args:
        model: 抽出対象(.eval()済み、shared_vocab=True限定 — 生成時は
            tgt側もsrc側と同じ埋め込み表を引くため。shared_vocab=Falseの
            モデルは未対応 — models/rddlgn.py::_init_state_bits の
            gaussian init と同じ「原理的に未実装」の扱い)
        src_tokens: (src_len,) 整数トークンID
        bos_token / eos_token: 開始・終了トークンID
        max_new_tokens: 生成する最大トークン数(EOSで早期終了)

    Returns:
        生成トークン列(EOSに達したらそれを含めて打ち切る)
    """
    src_tokens = np.asarray(src_tokens, dtype=np.int64)
    embed = _embed_hard_bits(model.src_embedding).astype(np.int64)
    src_bits = embed[src_tokens].reshape(-1)
    return simulate_rddlgn_greedy_bits(
        model, src_bits, bos_token=bos_token, eos_token=eos_token,
        max_new_tokens=max_new_tokens, optimize_degenerate=optimize_degenerate)


def simulate_rddlgn_greedy_bits(
    model: RDDLGN, src_bits, *, bos_token: int, eos_token: int,
    max_new_tokens: int, optimize_degenerate: bool = False,
) -> list[int]:
    """simulate_rddlgn_greedyの実装本体。srcを埋め込み済みビット列で受ける。

    ir/simulate.py 自体は変更しない。エンコーダはexport_rddlgn_encoder_ir
    をそのまま`simulate()`に通し、デコーダ側(L-layers/P+M+GroupSum)は
    tgt_len=1のrecurrentノードを作ってbodyだけを取り出し、
    `ir.simulate._simulate_values`(内部API、recurrentのbody評価が使うのと
    同じ関数)へscan_varの値を直接注入して1tickぶんだけ評価する
    (recurrentノード自体は使わない — state=p_prevはPython側ループで
    numpy配列として手動管理する。P-layers/M-layers/GroupSumはこの1tick
    評価にとって単なるLutStack+group_sumの直列なので、recurrentの
    steps=1に固定しても意味論は変わらない)。

    Args:
        src_bits: (src_len*d_emb,) の0/1(埋め込み・Heaviside済み、
            mdl_rddlgn_generate_greedyのsrc_bits引数と同じ形)
        optimize_degenerate: Trueなら退化ゲート除去パスを通して
            encoder/L/PM各LutStackをIR化する(既定Falseで従来どおり)
        他の引数は simulate_rddlgn_greedy と同じ

    Returns:
        生成トークン列(EOSに達したらそれを含めて打ち切る)
    """
    if not model.config.shared_vocab:
        raise NotImplementedError(
            'shared_vocab=Falseのモデルはsimulate_rddlgn_greedy未対応'
            '(生成ループがtgt側の埋め込み表としてsrc_embeddingを使うため)')

    from ..ir.simulate import _simulate_values, simulate

    d_emb = model.src_embedding.embedding.embedding_dim
    src_bits = np.asarray(src_bits, dtype=np.int64).reshape(-1)
    src_len = src_bits.shape[0] // d_emb
    embed = _embed_hard_bits(model.src_embedding).astype(np.int64)  # (vocab, d_emb)

    encoder_ir, _, k_out = export_rddlgn_encoder_ir(
        model.encoder, src_len, id_prefix='enc_',
        optimize_degenerate=optimize_degenerate)
    context = simulate(encoder_ir, src_bits[None, :])[0]

    l_rec, l_out = _decoder_l_recurrent(
        model.decoder.l_layers, 1, d_emb,
        optimize_degenerate=optimize_degenerate)
    pm_rec, p_out, l_out2, vocab = _decoder_pm_recurrent(
        model.decoder, 1, k_out, optimize_degenerate=optimize_degenerate)
    assert l_out2 == l_out
    l_body = l_rec.params['body']
    pm_body = pm_rec.params['body']
    sv_p_update_id = pm_rec.params['state_update']['sv_p']

    p_prev = np.asarray(
        _init_state_bits(model.decoder.init_mode, p_out), dtype=np.int64
    )[None, :]
    context_row = context[None, :].astype(np.int64)

    tokens: list[int] = []
    prev_token = int(bos_token)
    for _ in range(int(max_new_tokens)):
        l_in = embed[prev_token][None, :]
        l_vals = _simulate_values(l_body, {'sv_step': l_in})
        l_t = l_vals[l_rec.params['output_source_id']]

        pm_vals = _simulate_values(pm_body, {
            'sv_p': p_prev, 'sv_context': context_row, 'sv_l': l_t,
        })
        logits = pm_vals[pm_rec.params['output_source_id']][0]
        p_prev = pm_vals[sv_p_update_id]

        token = int(np.argmax(logits))   # 最初の最大値(strict >、numpy既定)
        tokens.append(token)
        if token == int(eos_token):
            break
        prev_token = token
    return tokens
