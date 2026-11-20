"""RDDLGN: LutStackを跨いだ退化ゲート除去パス。

静的解析(デプロイキット設計メモ節)で、学習済みRDDLGNの
38.5〜46.5%のゲートが退化真理値表(定数0/1・恒等A/B・NOT A/B)である
ことが分かった。CSE・dead gate・埋め込み定数伝播はいずれも~0%だった
ため、ここでは「退化tt除去 → 下流indices張り替え」のみを実装する。

export/optimize.py(conv2d重み共有木の縮約)との関係:
    tt代数(tt_eval・TT_NOT_A/B・CONST_FOLD_A/B・TT_DEGEN・SAME_FOLD)は
    そのまま輸入する。ただし build_conv_program 自体は「1個の重み共有
    バランス木、oc(出力チャネル)ごとに独立」という前提で書かれており、
    RDDLGNのLutStack(複数LutLayerが直列に連なり、層をまたいで参照が
    伸びる)には使えない(木専用)。ここでは同じ代数を、LutStackの
    「層0→層1→…→層L-1(境界)」という直列DAGに適用する新しいドライバを
    書く。

設計(詳細はデプロイキット設計メモ節):
    - LutStackの**最終層(境界層)は出力幅を変えない**。境界層の出力は
      state_update・recurrentのoutput_source_id(stacked/final_state)・
      GroupSum入力のいずれかとして「幅ぴったりのベクトル」を要求される
      ため、境界層自身のゲート数は変わらない(が、そのゲートの参照先・
      tt自体は簡約されうる)。
    - **境界より前の層(中間層)は物理的にゲート数を減らせる**。退化した
      位置は「実ゲートを置かない」(代わりに定数/上流参照+反転フラグの
      記述子として扱う)。下流(次の層、または境界層)はこの記述子を
      解決してから自分のtt/参照を組み立てる(定数畳み込み・NOT吸収・
      tt退化・同一参照折り畳みを、build_conv_programのemit_gateと同じ
      アルゴリズムで、ただし木ではなく層をまたぐ直列DAGとして適用する)。
    - 中間層で生き残った実ゲートは、既存のLutStackの入力空間
      (entry。N/Lならsv_step、K/P/Mならconcat0/p_concat/m_concat)と
      並べて**アキュムレータ空間**を作る。次の層はこのアキュムレータを
      入力とする新しいlut_layerノードとして生成する(層の入力が「直前の
      1ノードの出力」でなければならないというIRのlut_layer制約
      (in_width == 直前ノードのout_width)を守るため、アキュムレータは
      concatノード1個で「entry ++ 中間層の生存ゲート」を束ねる)。
    - NOT(A/B)の除去は**反転フラグの遅延**で実現する(build_conv_program
      のように「まずNOTゲートを作ってから次のemit_gateで吸収する」方式
      だと、吸収しきれず参照されずに残った標準NOTゲートがdead codeに
      なりうる — このモジュールはdead gate除去を実装しない方針なので、
      そもそもNOTゲートを作らない設計にした。反転フラグは次段のtt上で
      TT_NOT_A/TT_NOT_B により合成され、最終的に実ゲートが必要になった
      時点(境界層、または両オペランドが真に2変数の実ゲート)で初めて
      tt側へ焼き込まれる)。
    - **状態ループ(K/Pの自己ループ、t-1の出力がt同層の入力)は跨がない**。
      entry(状態scan_var・invariant・step)は常にopaqueなタップとして
      扱い、境界層(state_updateの対象)は常に全幅の実ゲートを維持する
      ので、このモジュール自体の設計として自己ループを跨いだ書き換えは
      発生しない。

呼び出し方: `_lutstack_nodes`(export/rddlgn_ir.py)と同じ引数
(stack, id_prefix, entry_id, entry_width)を取り、同じ4値
(nodes, edges, last_id, last_width)を返す — drop-in差し替え可能。
"""

import numpy as np

from ..ir.schema import IREdge, IRNode
from ..ir.serialize import tt_to_hex
from .discretize import truth_tables_from_lut_layer
from .optimize import (
    CONST_FOLD_A,
    CONST_FOLD_B,
    RED_BINARY,
    RED_CONST0,
    RED_CONST1,
    RED_ID,
    SAME_FOLD,
    TT_DEGEN,
    TT_DEGEN_ON_B,
    TT_NOT_A,
    TT_NOT_B,
    tt_eval,
)

__all__ = ['reduce_lutstack_nodes']

# 境界層でconst(両オペランド定数畳み込み)を焼き込むときのダミー添字。
# tt=0x0/0xfは両オペランドの値を無視するので、ACC空間内の有効な添字なら
# 何でもよい(entryは常に幅>=1なので0は常に有効)。
_DUMMY_ACC_INDEX = 0


class _V:
    """解決済みの値の記述子。

    is_const=True: 定数(val∈{0,1})。
    is_const=False: refはACC空間(entry ++ これまでの生存中間ゲート)の
        添字、invは「まだtt側へ焼き込んでいない反転」の遅延フラグ
        (0/1)。同じrefへ複数回reduceを重ねてもinvはXORで畳み込まれる
        (このモジュール内ではcombine呼び出し時に毎回tt側へ折り込むので
        invが2以上に育つことはない)。
    """

    __slots__ = ('is_const', 'val', 'ref', 'inv')

    def __init__(self, is_const, val=0, ref=0, inv=0):
        self.is_const = is_const
        self.val = val
        self.ref = ref
        self.inv = inv


def _const(v: int) -> _V:
    return _V(True, val=int(v))


def _ref(r: int, inv: int = 0) -> _V:
    return _V(False, ref=r, inv=inv)


def _finalize_const(val: int, force: bool, draft: list) -> _V:
    """定数値の確定。非境界なら伝播(実ゲート無し)、境界なら強制実体化。"""
    if not force:
        return _const(val)
    idx = len(draft)
    draft.append((0x0 if val == 0 else 0xf, _DUMMY_ACC_INDEX, _DUMMY_ACC_INDEX))
    return _ref(-(idx + 1))


def _finalize_unary(code: int, v: _V, force: bool, draft: list) -> _V:
    """単項簡約コード(CONST0/1・ID・NOT)の確定。vは非定数(呼び出し元で保証)。"""
    if code == RED_CONST0:
        return _finalize_const(0, force, draft)
    if code == RED_CONST1:
        return _finalize_const(1, force, draft)
    if code == RED_ID:
        if not force:
            return _ref(v.ref)
        idx = len(draft)
        draft.append((0xa, v.ref, v.ref))
        return _ref(-(idx + 1))
    # RED_NOT
    if not force:
        return _ref(v.ref, 1)
    idx = len(draft)
    draft.append((0x5, v.ref, v.ref))
    return _ref(-(idx + 1))


def _combine(tt: int, va: _V, vb: _V, force: bool, draft: list) -> _V:
    """1ゲート(tt, va, vb)を簡約する。draftへ実体化した場合は負添字(=-(local_idx+1))を返す。

    export/optimize.pyのemit_gateと同じ手順(NOT吸収→定数畳み込み→
    tt自体の退化→同一参照折り畳み→実ゲート)を踏む。ただしNOT吸収は
    「まずNOTゲートを作ってから読み取り時に剥がす」のではなく、va/vbの
    invフラグをtt側へ先に折り込む(=NOTゲート自体を作らない)。
    """
    tt &= 0xf
    if not va.is_const and va.inv:
        tt = TT_NOT_A[tt]
    if not vb.is_const and vb.inv:
        tt = TT_NOT_B[tt]

    if va.is_const and vb.is_const:
        val = tt_eval(tt, va.val, vb.val)
        return _finalize_const(val, force, draft)
    if vb.is_const:
        code = CONST_FOLD_B[tt * 2 + vb.val]
        return _finalize_unary(code, va, force, draft)
    if va.is_const:
        code = CONST_FOLD_A[tt * 2 + va.val]
        return _finalize_unary(code, vb, force, draft)

    # ここから先はva/vbともに非定数(ACC空間内の実参照、inv=0に正規化済み)
    degen = TT_DEGEN[tt]
    if degen != RED_BINARY:
        target = vb if TT_DEGEN_ON_B[tt] else va
        return _finalize_unary(degen, target, force, draft)

    if va.ref == vb.ref:
        code = SAME_FOLD[tt]
        return _finalize_unary(code, va, force, draft)

    # 真に2変数へ依存する実ゲート。CSEはやらない(静的解析でcse_dup_rate
    # ~0%と確認済み、デプロイキット設計メモ節)。
    idx = len(draft)
    draft.append((tt, va.ref, vb.ref))
    return _ref(-(idx + 1))


def _layer_arrays(layer) -> tuple[np.ndarray, np.ndarray, int]:
    if int(layer.n_inputs) != 2:
        raise ValueError(
            f'reduce_lutstack_nodes: n_inputs={layer.n_inputs} は非対応'
            '(縮約はn_inputs=2固定、export/optimize.pyと同じ前提)'
        )
    indices = layer.indices.detach().cpu().numpy().astype(np.int64)
    tts_obj = truth_tables_from_lut_layer(layer)
    tt = np.array([int(v) for v in tts_obj], dtype=np.int64)
    return indices, tt, int(layer.out_dim)


def _emit_lutstack_ir(
    id_prefix: str, entry_id: str, intermediate: list[dict],
    boundary_i: int, boundary_draft: list, boundary_out_dim: int,
    final_width: int, stats_out: list | None,
) -> tuple[list, list, str, int]:
    """ 退化ゲート除去の縮約結果をIRノード列へ組み立てる(遅延実体化)。

    `reduce_lutstack_nodes`の第1段(縮約の走査)がこの関数を呼ぶ時点で、
    グループ全体の最終幅(`final_width` = 境界層のin_width)が確定済み。
    中間層のconcatノードへ`accumulate_group`/`accumulate_final_width`を
    一律で付けることで、`deploy/plan.py::_plan_accumulate_concat`が
    「グループの最初のconcatだけ実バッファを1本確保して1回だけentryを
    コピーし、以降のconcatは新層の出力を親バッファ内オフセットへの
    aliasとして返すだけ(物理コピー無し)」という最適化をできるようにする
    (旧設計ではconcatノードが層ごとに独立で、`deploy/plan.py`側が
    毎回`flat_concat`で実コピーしていた — これがO(層数)回・幅が単調増加
    するコピーの原因だった)。

    Args:
        intermediate: `reduce_lutstack_nodes`が走査中に記録した中間層の
            記録(層順)。各要素は
            {'i', 'acc_width_before', 'draft'}(draftは(tt,a,b)のリスト、
            a/bはACC-local添字)
        boundary_draft: 境界層の(tt,a,b)リスト(出力位置と1:1対応)
        final_width: 境界層のin_width(=entry_width + 中間層の生存ゲート数の総和)
    """
    nodes: list = []
    edges: list = []
    acc_id = entry_id
    # 中間層が1つも実体化しない(全層n_real==0)LutStackはグループを作らない
    # (concatノード自体が無いので、alias機構の出番も無い)。
    group_id = id_prefix if intermediate else None

    for rec in intermediate:
        i = rec['i']
        acc_width_before = rec['acc_width_before']
        draft = rec['draft']
        n_real = len(draft)

        new_layer_id = f'{id_prefix}{i}'
        new_node = IRNode(
            id=new_layer_id,
            type='lut_layer',
            params={
                'in_width': acc_width_before,
                'out_width': n_real,
                'n_inputs': 2,
                'indices': np.array(
                    [[d[1] for d in draft], [d[2] for d in draft]],
                    dtype=np.int64,
                ),
                'truth_tables': tt_to_hex([d[0] for d in draft]),
            },
            train_info={'rddlgn_optimize': 'reduced'},
        )
        nodes.append(new_node)
        edges.append(IREdge(src=acc_id, dst=new_layer_id))

        concat_id = f'{id_prefix}{i}_acc'
        concat_node = IRNode(
            id=concat_id, type='concat',
            params={
                'in_widths': [acc_width_before, n_real],
                'accumulate_group': group_id,
                'accumulate_final_width': int(final_width),
            },
        )
        nodes.append(concat_node)
        edges.append(IREdge(src=acc_id, dst=concat_id, dst_slot=0))
        edges.append(IREdge(src=new_layer_id, dst=concat_id, dst_slot=1))

        acc_id = concat_id

    node = IRNode(
        id=f'{id_prefix}{boundary_i}',
        type='lut_layer',
        params={
            'in_width': final_width,
            'out_width': boundary_out_dim,
            'n_inputs': 2,
            'indices': np.array(
                [[t[1] for t in boundary_draft], [t[2] for t in boundary_draft]],
                dtype=np.int64,
            ),
            'truth_tables': tt_to_hex([t[0] for t in boundary_draft]),
        },
        train_info={'rddlgn_optimize': 'boundary'},
    )
    nodes.append(node)
    edges.append(IREdge(src=acc_id, dst=node.id))
    if stats_out is not None:
        stats_out.append(dict(layer=boundary_i, is_boundary=True,
                              out_dim=boundary_out_dim, n_real=boundary_out_dim))
    return nodes, edges, node.id, boundary_out_dim


def reduce_lutstack_nodes(
    stack, id_prefix: str, entry_id: str, entry_width: int,
    *, stats_out: list | None = None,
) -> tuple[list, list, str, int]:
    """LutStackを退化ゲート除去つきでIRノード化する(`_lutstack_nodes`のdrop-in置換)。

    Args:
        stack: models/rddlgn.py::LutStack(複数LutLayerの直列)
        id_prefix: ノードidの接頭辞(既存の`_lutstack_nodes`と同じ規約)
        entry_id / entry_width: このLutStackへの入力ノードid・幅
        stats_out: Noneでなければ、層別の削減統計
            {'layer': i, 'is_boundary': bool, 'out_dim': int, 'n_real': int}
            を追記する(呼び出し側の計測・報告用。戻り値の互換性には
            影響しない)

    Returns:
        (nodes, edges, last_id, last_width) — `_lutstack_nodes`と同じ契約。
        last_widthは元のLutStackの最終層out_dimと**必ず一致**する
        (境界層は幅を変えない)。

    中間層のIRノード(lut_layer/concat)の**実体化**は、全層を
    走査して最終アキュムレータ幅(`final_width`)が確定するまで
    `_emit_lutstack_ir`へ遅延する(理由は同関数のdocstring参照)。
    縮約そのもの(tt代数によるゲートの生存判定)は元の実装から変更していない
    — `_combine`呼び出しの引数・順序・returnは全て同一で、
    IRノード構築だけを後段へ切り出した。
    """
    layers = list(stack.layers)
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError('reduce_lutstack_nodes: LutStackが空')
    if entry_width <= 0:
        raise ValueError('reduce_lutstack_nodes: entry_widthは正整数が必要')

    acc_width = entry_width
    prev_pos_map: list | None = None  # layer i-1 の元の位置 -> _V(ACC-local)
    intermediate: list[dict] = []

    for i, layer in enumerate(layers):
        indices, tts, out_dim = _layer_arrays(layer)
        if i == 0:
            assert layer.in_dim == entry_width, (
                f'{id_prefix}0: in_dim({layer.in_dim}) != entry_width({entry_width})'
            )
        else:
            prev_out_dim = int(layers[i - 1].out_dim)
            assert layer.in_dim == prev_out_dim, (
                f'{id_prefix}{i}: in_dim({layer.in_dim}) != 前段幅({prev_out_dim})'
            )

        force = (i == n_layers - 1)
        draft: list = []  # このレイヤで新規に実体化したゲート(a/bはacc_width基準)

        if force:
            boundary: list = []
            for j in range(out_dim):
                a_orig = int(indices[0, j])
                b_orig = int(indices[1, j])
                tt = int(tts[j])
                va = _V(False, ref=a_orig) if i == 0 else prev_pos_map[a_orig]
                vb = _V(False, ref=b_orig) if i == 0 else prev_pos_map[b_orig]
                result = _combine(tt, va, vb, True, draft)
                assert not result.is_const and result.inv == 0 and result.ref < 0, (
                    'force=Trueは常にdraftへ実体化した負添字refを返すはず'
                )
                boundary.append(draft[-(result.ref + 1)])

            # acc_widthはここまでの中間層の更新を反映した最終アキュムレータ幅
            # (=境界層のin_width。境界層自身はこの幅を変えない)。
            return _emit_lutstack_ir(
                id_prefix, entry_id, intermediate, i, boundary, out_dim,
                acc_width, stats_out)

        # --- 中間層: 退化した位置は実体化しない ---
        this_pos_map: list = [None] * out_dim
        for j in range(out_dim):
            a_orig = int(indices[0, j])
            b_orig = int(indices[1, j])
            tt = int(tts[j])
            va = _V(False, ref=a_orig) if i == 0 else prev_pos_map[a_orig]
            vb = _V(False, ref=b_orig) if i == 0 else prev_pos_map[b_orig]
            this_pos_map[j] = _combine(tt, va, vb, False, draft)

        n_real = len(draft)
        if stats_out is not None:
            stats_out.append(dict(layer=i, is_boundary=False,
                                  out_dim=out_dim, n_real=n_real))

        acc_width_before = acc_width
        if n_real > 0:
            intermediate.append(dict(i=i, acc_width_before=acc_width_before,
                                     draft=draft))
            acc_width = acc_width_before + n_real

            # this_pos_mapの負添字(このレイヤ内で新規実体化したゲート)を
            # ACC-local添字へ変換する。非負(既存ACCへのパススルー)は
            # そのままprefix位置として有効。
            for j in range(out_dim):
                v = this_pos_map[j]
                if not v.is_const and v.ref < 0:
                    v.ref = acc_width_before + (-(v.ref) - 1)
        # n_real == 0 ならこのレイヤは何も新設しない(this_pos_mapは全て
        # 定数かパススルーのはずで、負添字は存在しないので変換不要)。

        prev_pos_map = this_pos_map

    raise AssertionError('到達しないはず(最終層は必ずforce=Trueで早期returnする)')
