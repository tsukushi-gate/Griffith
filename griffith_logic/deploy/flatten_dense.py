"""dense DLGN の lut_layer 列を縮約済みの `lut_flat` ノードへ置換する。

dense DLGN(lut_layer 列 + group_sum)の C 生成は、学習時の**全ゲート**を
LUT 引きで実行していた。実測では HAR k6000 の 36,000 ゲートのうち実効ゲートは
15,124(58% 削減)で、残りは定数・パススルー/NOT(1 入力関数)・CSE 重複・
未参照ゲートだった。`export/netlist_reduce.py` はこれらを**極性ビット付き参照**
(パススルー = 配線、NOT = 参照の極性反転)へ畳み込む lossless 縮約器だが、
測定装置としてしか使われていなかった。本モジュールはそれをデプロイ経路へ
接続する変換パスである。

    IR(lut_layer×L → group_sum)
        ──flat_netlist_from_ir→ 平坦ネットリスト ──reduce_netlist→ ReduceResult
        ──本モジュール→ IR(lut_flat → group_sum)

設計判断(`lut_flat` の出力形)
----------------------------
`lut_flat` は **置換した lut_layer 列の最終層と同じ out_width 本の出力ビット列**
を返す。group_sum 側は一切変更しない。理由(simulate 一致を最短で保証できる):

- 元 IR の group_sum(整数スコア = グループ内 popcount)の意味論をそのまま
  使えるので、置換前後の一致は「最終層ビット列の一致」だけに帰着する。
  group_sum の `k` / `tau` を `lut_flat` へ複製して整数スコア規約を二重化しない
- 縮約で定数/パススルー/反転に解決した最終層ゲートは、`outputs` の参照
  (信号 0 = 定数 0、または 入力/ゲート信号 + 極性ビット)として
  `lut_flat` 内で最終出力ビット列へ**再構成**される。クラス別バイアスや
  係数 −1 のような「スコア側の補正」を作らないので group_sum がそのまま読める
- `ReduceResult.class_scores`(バイアス + Σ 符号 × 参照値)とは
  group_sum(lut_flat 出力)が整数で一致することをテストで固定する

信号番号・ゲート表記は `ir/schema.py` の `lut_flat` 節が真実源。

多重防御
--------
変換後 IR を返す前に、必ず `ir/simulate.py` で元 IR と変換後 IR の出力を
ランダム入力 N 本(+ 呼び出し側が渡した実データ)で整数比較する。1 本でも
不一致なら ValueError を送出し、壊れた IR を返さない(`resynth.py` と同じ規律)。
ゴールデンは**元 IR の simulate**であり、変換後の IR 自身ではない。

対象外
------
- n_inputs != 2 の lut_layer(縮約器が 2 入力限定。連鎖はそこで切れる)
- recurrent ノードの body 内の dense(トップレベルのグラフだけを見る)
"""

from copy import copy

import numpy as np

from ..export.netlist_reduce import flat_netlist_from_ir, reduce_netlist
from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.simulate import simulate

__all__ = ['demo_bits_from_doc', 'flatten_dense_ir', 'flatten_dense_doc',
           'find_dense_chains', 'DENSE_FLAT_MODES', 'resolve_dense_flat',
           'describe_dense_flat']


def _is_flattenable(node) -> bool:
    return node.type == 'lut_layer' and int(node.params.get('n_inputs', 0)) == 2


def find_dense_chains(graph: IRGraph) -> list[list[IRNode]]:
    """置換対象の「2 入力 lut_layer の直列鎖」を列挙する。

    鎖の条件: 隣り合う lut_layer 同士が、前段の出力を**後段だけ**が読み
    (fanout 1)、後段の入力エッジが 1 本であること。途中の層を別のノードが
    読んでいる(= 途中の出力ビットが外から見える)場合はそこで鎖を切る。
    鎖の最後の層の出力は、group_sum・出力ポートなど何が読んでもよい
    (`lut_flat` が同じ幅の出力ビット列を返すので配線は変わらない)。

    Returns:
        鎖(IRNode のリスト、トポロジカル順)のリスト
    """
    in_edges: dict[str, list] = {}
    out_edges: dict[str, list] = {}
    for e in graph.edges:
        in_edges.setdefault(e.dst, []).append(e)
        out_edges.setdefault(e.src, []).append(e)
    by_id = {n.id: n for n in graph.nodes}

    def continues(prev, nxt) -> bool:
        """nxt が prev の直後に続く鎖要素か。"""
        outs = out_edges.get(prev.id, [])
        ins = in_edges.get(nxt.id, [])
        return (len(outs) == 1 and outs[0].dst == nxt.id
                and len(ins) == 1 and _is_flattenable(nxt))

    in_chain: set[str] = set()
    chains: list[list[IRNode]] = []
    for node in graph.topological_order():
        if node.id in in_chain or not _is_flattenable(node):
            continue
        chain = [node]
        cur = node
        while True:
            outs = out_edges.get(cur.id, [])
            if len(outs) != 1:
                break
            nxt = by_id.get(outs[0].dst)
            if nxt is None or not continues(cur, nxt):
                break
            chain.append(nxt)
            cur = nxt
        in_chain.update(n.id for n in chain)
        chains.append(chain)
    return chains


def _chain_netlist(graph: IRGraph, chain: list[IRNode]) -> dict:
    """鎖を単独の IR(入力ポート + 鎖 + group_sum k=1)へ切り出して平坦化する。

    `flat_netlist_from_ir` の変換規則(op ⇔ tt の対合・in_width 検査)を
    そのまま再利用するため、鎖の末尾へ k=1 の group_sum を仮置きする
    (classes=1, group_size=最終幅 になるだけで、最終層ゲートの参照は
    `ReduceResult.final_refs` から 1 本ずつ取れる)。
    """
    head = chain[0]
    in_width = int(head.params['in_width'])
    last_w = int(chain[-1].params['out_width'])
    sub = IRGraph(
        inputs=[IRPort(id='__in', width=in_width)],
        outputs=[IRPort(id='__out', width=1)],
    )
    prev = '__in'
    for n in chain:
        sub.nodes.append(n)
        sub.edges.append(IREdge(src=prev, dst=n.id))
        prev = n.id
    gs = IRNode(id='__gs', type='group_sum',
                params={'in_width': last_w, 'k': 1, 'tau': 1.0})
    sub.nodes.append(gs)
    sub.edges.append(IREdge(src=prev, dst='__gs'))
    sub.edges.append(IREdge(src='__gs', dst='__out'))
    return flat_netlist_from_ir(sub)


def _lut_flat_params(netlist: dict, result) -> tuple[dict, dict]:
    """ReduceResult → lut_flat の params と、その鎖の内訳統計。

    縮約器は「作ったが出力から到達しなくなったゲート」(`reach` が偽)も
    `gates` に残すので、ここで到達可能なものだけに詰め直して添字を振り直す。
    """
    in_w = int(netlist['input_bits'])
    reach = result.reach
    new_index: dict[int, int] = {}
    for n in range(len(result.gates)):
        if reach[n]:
            new_index[n] = len(new_index)

    def signal(ref) -> int:
        kind, idx = ref[0], ref[1]
        if kind == 'i':
            return 1 + idx
        if kind == 'g':
            return 1 + in_w + new_index[idx]
        raise ValueError(f'ゲート入力に定数参照が残っている: {ref}')

    gates = []
    for n, (tt, ra, rb, _li) in enumerate(result.gates):
        if not reach[n]:
            continue
        if ra[2] or rb[2]:
            raise ValueError('縮約後ゲートの入力に極性が残っている(縮約器の不変条件違反)')
        gates.append([int(tt), signal(ra), signal(rb)])

    outputs = []
    for ref in result.final_refs:
        if ref[0] == 'c':
            outputs.append(int(ref[1]))            # 信号0(定数0)の極性 = 定数値
        else:
            outputs.append(2 * signal(ref) + int(ref[2]))
    params = {
        'in_width': in_w,
        'out_width': len(outputs),
        'gates': gates,
        'outputs': outputs,
    }
    st = result.stats
    info = {
        'gates_before': int(st['total_gates']),
        'gates_after': len(gates),
        'constants': int(st['const_folded']),
        'passthrough': int(st['passthrough_folded']),
        'cse': int(st['cse_merged']),
        'unreachable_after_fold': int(st['unreachable_after_fold']),
        'structurally_unreferenced': int(st['structurally_unreferenced']),
        'widths': [int(w) for w in st['widths']],
    }
    return params, info


def demo_bits_from_doc(doc: dict) -> np.ndarray | None:
    """docの demo_samples(実画像)+ preproc から検証用の実データビット列を作る。

    画像前処理を持たない(demo_samples が無い、または preproc.kind が記述子化
    できない)docでは None を返す(ランダム入力のみで検証する)。
    """
    from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits
    from .resynth import _decode_demo_images

    demo_samples = doc.get('demo_samples') or []
    preproc = doc.get('preproc') or {}
    if not demo_samples or preproc.get('kind') not in SUPPORTED_KINDS:
        return None
    images = _decode_demo_images(demo_samples, preproc)
    plan = build_preproc_descriptors(preproc)
    return np.asarray(encode_bits(plan, images), dtype=np.uint8)


def flatten_dense_ir(graph: IRGraph, *, n_verify: int = 16, seed: int = 0,
                     demo_bits: np.ndarray | None = None
                     ) -> tuple[IRGraph, dict]:
    """IR の 2 入力 lut_layer 直列鎖を縮約して `lut_flat` ノードへ置換する。

    元の graph は変更しない。置換した `lut_flat` ノードは**鎖の最後の
    lut_layer の id を引き継ぐ**ので、後段(group_sum・出力ポート)のエッジは
    無改造のまま繋がる。鎖が 1 つも無ければ元の graph と空の report を返す。

    Args:
        graph: 検証済み IR(トップレベルの lut_layer 列のみ対象)
        n_verify: 多重防御に使う一様ランダム入力の本数(0 なら省略。
            既定 16。本数を減らす/省くのは故障注入テストなどに限ること)
        seed: 検証用乱数の seed
        demo_bits: 実データのビット列 (M, in_width)。あれば同様に突き合わせる

    Returns:
        (新しい IRGraph, report)。report は
        ``{'chains': [{'node': 末尾id, 'replaced': [置換した層id…],
        'gates_before', 'gates_after', 'constants', 'passthrough', 'cse',
        'unreachable_after_fold', 'structurally_unreferenced', 'widths'}…],
        'gates_before', 'gates_after', 'constants', 'passthrough', 'cse',
        'verify_random', 'verify_demo'}``(合計は鎖ごとの総和)

    Raises:
        ValueError: 変換後 IR の simulate が元 IR と一致しなかった場合
            (縮約器・直列化のバグを塞ぐ多重防御。壊れた IR を返さない)
    """
    chains = find_dense_chains(graph)
    empty = {'chains': [], 'gates_before': 0, 'gates_after': 0,
             'constants': 0, 'passthrough': 0, 'cse': 0,
             'verify_random': 0, 'verify_demo': 0}
    if not chains:
        return graph, empty

    replace: dict[str, IRNode] = {}      # 末尾ノード id → lut_flat ノード
    drop: set[str] = set()               # 置換で消える層 id
    head_in: dict[str, str] = {}         # 先頭層 id → 末尾 id(入力エッジの付け替え)
    chain_reports = []
    for chain in chains:
        netlist = _chain_netlist(graph, chain)
        result = reduce_netlist(netlist)
        params, info = _lut_flat_params(netlist, result)
        last = chain[-1]
        node = IRNode(
            id=last.id, type='lut_flat', params=params,
            train_info={'flattened_from': [n.id for n in chain]},
            meta=dict(last.meta))
        replace[last.id] = node
        drop.update(n.id for n in chain[:-1])
        head_in[chain[0].id] = last.id
        info = {'node': last.id, 'replaced': [n.id for n in chain], **info}
        chain_reports.append(info)

    new_nodes = []
    for n in graph.nodes:
        if n.id in drop:
            continue
        new_nodes.append(replace.get(n.id, n))
    new_edges = []
    for e in graph.edges:
        if e.src in drop:
            continue                        # 鎖の途中の出力エッジ(鎖は fanout 1)
        if e.dst in head_in:
            # 先頭層への入力エッジ → 末尾 id の lut_flat へ付け替える
            # (長さ1の鎖では先頭 = 末尾なので同じ id に戻る)
            e = IREdge(src=e.src, dst=head_in[e.dst], dst_slot=e.dst_slot)
        elif e.dst in drop:
            continue                        # 鎖の途中の入力エッジ
        new_edges.append(e)

    new_graph = IRGraph(
        metadata=copy(graph.metadata), inputs=graph.inputs,
        outputs=graph.outputs, nodes=new_nodes, edges=new_edges,
        format=graph.format, version=graph.version)

    # ---- 多重防御: 元 IR の simulate をゴールデンとして整数一致を確認 ----
    from ..ir.validate import validate
    validate(new_graph)

    in_width = int(graph.inputs[0].width)
    verify_random = 0
    if n_verify > 0:
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=(n_verify, in_width), dtype=np.uint8)
        if not np.array_equal(simulate(graph, bits), simulate(new_graph, bits)):
            raise ValueError(
                'flatten_dense_ir: ランダム入力で縮約前後の simulate が一致しない'
                '(lut_flat 変換が壊れている可能性)')
        verify_random = n_verify
    verify_demo = 0
    if (demo_bits is not None and np.asarray(demo_bits).ndim == 2
            and np.asarray(demo_bits).shape[0] > 0
            and np.asarray(demo_bits).shape[1] == in_width):
        xd = np.asarray(demo_bits, dtype=np.uint8)
        if not np.array_equal(simulate(graph, xd), simulate(new_graph, xd)):
            raise ValueError(
                'flatten_dense_ir: 実データ(demo_samples)で縮約前後の simulate '
                'が一致しない(lut_flat 変換が壊れている可能性)')
        verify_demo = int(xd.shape[0])

    report = {
        'chains': chain_reports,
        'gates_before': sum(c['gates_before'] for c in chain_reports),
        'gates_after': sum(c['gates_after'] for c in chain_reports),
        'constants': sum(c['constants'] for c in chain_reports),
        'passthrough': sum(c['passthrough'] for c in chain_reports),
        'cse': sum(c['cse'] for c in chain_reports),
        'verify_random': verify_random,
        'verify_demo': verify_demo,
    }
    return new_graph, report


def flatten_dense_doc(doc: dict, *, n_verify: int = 16, seed: int = 0
                      ) -> tuple[dict, dict]:
    """docの IR に `flatten_dense_ir` を適用した新 doc と report を返す。

    doc 自体は変更しない(浅いコピー + ir のみ差し替え)。demo_samples が
    あれば実データでも検証する。report は新 doc の
    ``ir['metadata']['dense_flat']`` にも書き込む(`griffith deploy info` が読む)。
    実効ゲートが 1 つも減らない場合(鎖なし)は doc をそのまま返す。
    """
    from ..ir.serialize import ir_from_dict, ir_to_dict

    graph = ir_from_dict(doc['ir'])
    new_graph, report = flatten_dense_ir(
        graph, n_verify=n_verify, seed=seed, demo_bits=demo_bits_from_doc(doc))
    if not report['chains']:
        return doc, report
    new_graph.metadata = dict(new_graph.metadata)
    new_graph.metadata['dense_flat'] = report
    new_doc = dict(doc)
    new_doc['ir'] = ir_to_dict(new_graph)
    return new_doc, report


#: `--dense-flat` の指定値(codegen-c / verify-c)。
DENSE_FLAT_MODES = ('auto', 'on', 'off')


def _decision(mode: str, min_reduction: float, **kw) -> dict:
    out = {'mode': mode, 'applied': False, 'reason': '', 'reduction': None,
           'gates_before': 0, 'gates_after': 0,
           'min_reduction': float(min_reduction)}
    out.update(kw)
    return out


def resolve_dense_flat(bundle: dict, *, mode: str = 'auto',
                       min_reduction: float | None = None,
                       dense_batch: int = 0, n_verify: int = 16,
                       seed: int = 0, target: str = 'c') -> tuple[dict, dict]:
    """生成の直前に、dense-flat(`lut_flat`)を使うかを決めて適用する。

    バンドル(IR が真実源・全ターゲット共通)は変えない。`lut_flat` を実行
    できるのは C と Webなので、その生成の段で**メモリ上の複製バンドル**
    だけを平坦化する。返すバンドルは ``bundle['ir']`` を差し替えた浅いコピー
    (他の節は共有)。

    ``target``: ``'c'``(既定)/ ``'web'`` / ``'gpu'``。``min_reduction`` を
    省略したときの閾値だけが変わる(C は
    ``plan.DENSE_FLAT_AUTO_MIN_REDUCTION``=0.25、Web は
    ``plan.DENSE_FLAT_AUTO_MIN_REDUCTION_WEB``=0 つまり鎖があれば常に採用、
    GPU は ``plan.DENSE_FLAT_AUTO_MIN_REDUCTION_GPU``=0.25)。

    判定(``mode``):
      - ``'off'``: 何もしない(従来経路)
      - ``'on'``: 2 入力 lut_layer 列があれば閾値に関わらず平坦化する
      - ``'auto'``: 平坦化を試行し、実効ゲート削減率
        ``1 - gates_after/gates_before``(置換対象 lut_layer 列の合計)が
        ``min_reduction`` 以上なら採用、未満なら従来経路
        (既定 ``plan.DENSE_FLAT_AUTO_MIN_REDUCTION``=0.25、実測の損益分岐)
      - ``dense_batch`` は判定に使わない(でバッチ軸パック経路も
        ``lut_flat`` に対応した。引数は互換のために残す)
      - IR に既に ``lut_flat`` が焼いてある(``bundle --dense-flat``)ときは
        再平坦化しない(採用済みとして報告する)

    ``auto`` で変換や検証が失敗したときは**従来経路へ倒す**(最適化の失敗で
    出荷物を壊さない。理由に失敗内容を入れる)。``on`` では例外をそのまま送出する。
    変換は `flatten_dense_ir` の多重防御(元 IR の simulate との整数一致)込み。

    Returns:
        (C 生成に使うバンドル, decision)。decision は
        ``{'mode', 'applied', 'reason', 'reduction', 'gates_before',
        'gates_after', 'min_reduction'}``(reduction は判定したときだけ
        float、それ以外 None)

    Raises:
        ValueError: ``mode`` が不正、または ``mode='on'`` で変換が失敗した場合
    """
    from ..ir.serialize import ir_from_dict
    from .plan import (
        DENSE_FLAT_AUTO_MIN_REDUCTION,
        DENSE_FLAT_AUTO_MIN_REDUCTION_GPU,
        DENSE_FLAT_AUTO_MIN_REDUCTION_WEB,
    )

    defaults = {'c': DENSE_FLAT_AUTO_MIN_REDUCTION,
                'web': DENSE_FLAT_AUTO_MIN_REDUCTION_WEB,
                'gpu': DENSE_FLAT_AUTO_MIN_REDUCTION_GPU}
    if target not in defaults:
        raise ValueError(f'target={target!r} は不正(c/web/gpu のみ)')
    if min_reduction is None:
        min_reduction = defaults[target]
    if mode not in DENSE_FLAT_MODES:
        raise ValueError(
            f'dense_flat={mode!r} は不正({"/".join(DENSE_FLAT_MODES)} のみ)')
    if mode == 'off':
        return bundle, _decision(mode, min_reduction,
                                 reason='off 指定(従来経路)')
    del dense_batch   # パック経路も lut_flat に対応したので判定に使わない

    graph = ir_from_dict(bundle['ir'])
    has_flat = any(n.type == 'lut_flat' for n in graph.nodes)
    if not find_dense_chains(graph):
        if has_flat:
            rep = (graph.metadata or {}).get('dense_flat') or {}
            before = int(rep.get('gates_before', 0))
            after = int(rep.get('gates_after', 0))
            return bundle, _decision(
                mode, min_reduction, applied=True,
                reason='バンドルに焼き込み済み(bundle --dense-flat)。再平坦化しない',
                reduction=(1.0 - after / before) if before else None,
                gates_before=before, gates_after=after)
        return bundle, _decision(
            mode, min_reduction,
            reason='対象の2入力lut_layer列が無い(従来経路)')

    try:
        new_doc, report = flatten_dense_doc(bundle, n_verify=n_verify,
                                            seed=seed)
    except Exception as exc:                       # noqa: BLE001
        if mode == 'on':
            raise
        return bundle, _decision(
            mode, min_reduction,
            reason=f'平坦化に失敗したため従来経路へ({type(exc).__name__}: {exc})')
    before = int(report['gates_before'])
    after = int(report['gates_after'])
    reduction = (1.0 - after / before) if before else 0.0
    common = dict(reduction=reduction, gates_before=before, gates_after=after)
    if mode == 'on' or reduction >= min_reduction:
        why = ('on 指定' if mode == 'on' else
               f'削減率が閾値以上({reduction:.1%} >= {min_reduction:.0%})')
        if mode != 'on' and target == 'web':
            why += '。Web は lut_flat が1ゲートあたりも速いので鎖があれば常に採用'
        return new_doc, _decision(mode, min_reduction, applied=True,
                                  reason=why, **common)
    return bundle, _decision(
        mode, min_reduction,
        reason=f'削減率が閾値未満({reduction:.1%} < {min_reduction:.0%})'
               '。従来のbit-sliceカーネルのほうが速い見込み',
        **common)


def describe_dense_flat(decision: dict) -> str:
    """`resolve_dense_flat` の decision を1行にまとめる(標準出力・model.h 用)。"""
    gates = ''
    if decision.get('gates_before'):
        red = decision.get('reduction')
        cut = f'{red:.1%}' if red is not None else '-'
        gates = (f"実効ゲート {decision['gates_before']:,} → "
                 f"{decision['gates_after']:,}(削減率 {cut})。")
    verdict = '採用' if decision['applied'] else '不採用'
    return (f"dense-flat({decision['mode']}): {verdict} — {gates}"
            f"{decision['reason']}")
