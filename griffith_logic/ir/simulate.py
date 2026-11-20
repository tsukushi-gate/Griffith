"""IRのビット厳密リファレンス実行器(golden model、v0.2でDAG対応)。

numpyのみでIRを実行する。HDLシミュレーションとPyTorch推論の
両方に対する照合基準(3段等価性チェーンの中心)。

    PyTorch hard eval ⇔ IR simulate ⇔ Verilogシミュレーション

規約:
    - 信号は {0,1} の uint8
    - lut_layer: addr = Σ_j x[indices[j,o]] << j、出力 = (tt[o] >> addr) & 1
    - group_sum: グループ内popcount(整数スコア。tauはargmax不変なので無視)
    - 多入力ノードはdst_slot昇順に入力を集める(slot 0が先頭)
    - concat: slot 0が出力ビット列の先頭(=整数パック時の下位ビット)
    - add: or/xor=ビットOR/XOR、sum=整数和(出力が{0,1}を超えるため
      論理領域の終端専用)
    - constant: 入力なし(arity 0)。全バッチ同一のbits定数行
    - conv2d/or_pool2d/upsample2d/permute: lower_to_primitives展開後の
      グラフで実行する内部フォールバック(traceキーは元ノードidへ集約。
      upsample2d/permuteは恒等配線ノード。indexはJSONに持たず
      lower_to_primitives呼び出し時(=simulate呼び出し時)に初めて計算する
      ためJSONは小さいまま)
    - lut_flat: 縮約済みdense平坦DAG。信号表 sig = [定数0, 入力ビット…,
      ゲート出力…] を先頭から1ゲートずつ素朴に埋め、outputs[i] = 2*信号+極性
      の参照で出力ビット列を作る(縮約前のlut_layer列の最終層ビットと一致)
    - scan_var/recurrent: scan_varはbody内でのみ意味を持つ
      入力の「穴」で、_simulate_values(内部API)が事前に注入した値を
      そのまま返す(計算しない)。recurrentはbodyをsteps回、
      _simulate_valuesで再帰評価する(unrollしない。ONNX Scan型)。
      公開シグネチャ(simulate/simulate_trace)は入力を(batch,width)の
      配列のまま受けるので変わらない
"""

import numpy as np

from .schema import IRGraph
from .serialize import tt_from_hex

# lowering展開でsimulateするノード型(export/lowering.LOWERABLE_TYPESと同値。
# ir→exportのモジュールレベル循環importを避けるためここに複製する)
_LOWERABLE_TYPES = ('conv2d', 'or_pool2d', 'upsample2d', 'permute')

#: テスト/デバッグ専用の内部フック。Noneなら何もしない
#: (通常経路への影響ゼロ)。dictを渡すと {recurrent node id: [tick毎の
#: state_valsのコピー]} が溜まる。recurrentの全tick・全state値を
#: PyTorch側のforward_hookと突き合わせるテスト用で、公開契約ではない。
_recurrent_trace_sink: dict | None = None


def _simulate_trace_lowered(graph: IRGraph, x: np.ndarray) -> dict:
    """conv2d/or_pool2dを含むIRを展開して実行し、元ノードidへ集約する。

    conv2dノードの値は展開後の最終レベルlut_layerの出力
    (=convノードの出力)として記録する。展開中間ノードはtraceに含めない。
    """
    from ..export.lowering import lower_to_primitives, lowered_output_id

    low_trace = simulate_trace(lower_to_primitives(graph), x)
    trace = {p.id: low_trace[p.id] for p in graph.inputs}
    for node in graph.nodes:
        key = lowered_output_id(node) if node.type in _LOWERABLE_TYPES \
            else node.id
        trace[node.id] = low_trace[key]
    return trace


def _simulate_values(graph: IRGraph, values: dict) -> dict:
    """`values`を初期状態としてIRを実行する(内部API)。

    公開API(simulate/simulate_trace)は入力ポートの値だけを事前に
    詰めて呼ぶが、recurrentノードのbody評価はscan_varノードの値を
    事前に詰めて同じロジックを再帰的に回す。
    `values`に既にキーがあるノード(入力ポート・scan_var)は再計算しない。

    Args:
        graph: 検証済みIR(bodyグラフの場合もある。inputs=[]でよい)
        values: {node_id または input port id: (batch, width)のint64配列}。
            この辞書を書き換えて返す

    Returns:
        values(全ノードぶんの出力を追加したもの)
    """
    # dst_slot昇順の入力元マップ
    in_srcs: dict[str, list[str]] = {}
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    for dst, edges in grouped.items():
        in_srcs[dst] = [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]

    # バッチサイズは既に注入済みの値(入力ポートまたはscan_var)から取る。
    # constant/recurrentの内部評価がこれを使う('x'という外側の変数には
    # 依存しない — bodyグラフの評価にはそもそも入力ポートが無いため)
    batch = next(iter(values.values())).shape[0]

    for node in graph.topological_order():
        if node.id in values:
            continue      # 入力ポート・scan_varとして既に注入済み
        # constant等のarity 0ノードは入力エッジを持たない
        ins = [values[s] for s in in_srcs.get(node.id, [])]
        h = ins[0] if ins else None
        p = node.params

        if node.type == 'lut_layer':
            indices = np.asarray(p['indices'], dtype=np.int64)  # (n, out)
            n_inputs = p['n_inputs']
            gathered = h[:, indices]              # (batch, n, out)
            weights = (1 << np.arange(n_inputs, dtype=np.int64))[None, :, None]
            addr = (gathered * weights).sum(axis=1)            # (batch, out)
            tt_ints = tt_from_hex(p['truth_tables'])
            if n_inputs <= 6:
                # 真理値表 2^n ≤ 64 bit は uint64 のシフトで引く(従来経路、bitwise 不変)
                tts = np.asarray(tt_ints, dtype=np.uint64)
                out = ((tts[None, :] >> addr.astype(np.uint64)) & np.uint64(1)).astype(np.int64)
            else:
                # n > 6(例: ThermometerEncoder の 8 入力 LUT)は 2^n bit が uint64 に
                # 収まらないので、表を (out, 2^n) のビット行列に展開して添字で引く
                n_entries = 1 << n_inputs
                table = np.zeros((len(tt_ints), n_entries), dtype=np.int64)
                for o, t in enumerate(tt_ints):
                    t = int(t)
                    if t:
                        bits = np.frombuffer(
                            t.to_bytes((n_entries + 7) // 8, 'little'), dtype=np.uint8)
                        table[o] = np.unpackbits(bits, bitorder='little')[:n_entries]
                out = table[np.arange(table.shape[0])[None, :], addr]
        elif node.type == 'group_sum':
            k = p['k']
            out = h.reshape(h.shape[0], k, -1).sum(axis=-1)  # 整数スコア
            if p.get('bias') is not None:
                # 判定バイアス: クラスごとの整数定数を足す(argmax に効く)
                out = out + np.asarray(p['bias'], dtype=np.int64)[None, :]
        elif node.type == 'lut_flat':
            out = _simulate_lut_flat(p, h)
        elif node.type == 'binarize':
            out = (h > p['threshold']).astype(np.int64)
        elif node.type == 'flatten':
            out = h.reshape(h.shape[0], -1)
        elif node.type == 'concat':
            # ビット順規約: slot 0が出力ビット列の先頭(下位ビット側)
            out = np.concatenate(ins, axis=1)
        elif node.type == 'add':
            mode = p['mode']
            out = ins[0].copy()
            for t in ins[1:]:
                if mode == 'or':
                    out |= t
                elif mode == 'xor':
                    out ^= t
                else:  # sum: 整数和(論理領域の終端専用)
                    out = out + t
        elif node.type == 'constant':
            # 定数行: 全バッチ同一のビット列
            bits = np.asarray(p['bits'], dtype=np.int64)
            out = np.tile(bits, (batch, 1))
        elif node.type == 'scan_var':
            # bodyの外(recurrent自身の直接の子でない位置)に出てきたら
            # 注入されていないということなので、それ自体がエラー
            raise NotImplementedError(
                f'{node.id}: scan_varはrecurrentのbody評価時にのみ注入される値'
                '(親グラフの直接評価では現れないはず)'
            )
        elif node.type == 'recurrent':
            out = _simulate_recurrent(node, ins, batch)
        elif node.type == 'blackbox':
            raise NotImplementedError(
                f'{node.id}: blackboxノードはsimulate不可'
            )
        else:
            raise NotImplementedError(f'未対応ノード型: {node.type}')

        values[node.id] = out

    return values


def _simulate_lut_flat(p: dict, h: np.ndarray) -> np.ndarray:
    """lut_flatノードを素朴に評価する(信号番号規約は schema.py 参照)。

    ゴールデンなので速度は不要: ゲートを先頭から1つずつ評価する。

    Args:
        p: ノードparams(in_width/gates/outputs)
        h: 入力ビット列 (batch, in_width)

    Returns:
        (batch, out_width) のint64 0/1配列
    """
    in_width = int(p['in_width'])
    gates = np.asarray(p['gates'], dtype=np.int64).reshape(-1, 3)
    outputs = np.asarray(p['outputs'], dtype=np.int64)
    batch = h.shape[0]
    sig = np.zeros((1 + in_width + gates.shape[0], batch), dtype=np.int64)
    sig[1:1 + in_width] = h.T
    base = 1 + in_width
    for g in range(gates.shape[0]):
        tt, a, b = (int(v) for v in gates[g])
        sig[base + g] = (tt >> ((sig[b] << 1) | sig[a])) & 1
    return (sig[outputs >> 1] ^ (outputs & 1)[:, None]).T


def _simulate_recurrent(node, ins: list, batch: int) -> np.ndarray:
    """recurrentノードを評価する。

    bodyはunrollせず、steps回 _simulate_values を再帰的に呼ぶ
    (numpy素朴実装。速度は不要 — ここはゴールデン)。
    """
    p = node.params
    body: IRGraph = p['body']
    steps = int(p['steps'])
    scan_vars = p['scan_vars']
    ext_svs = [sv for sv in scan_vars if sv['role'] in ('step', 'invariant')]
    state_svs = [sv for sv in scan_vars if sv['role'] == 'state']
    assert len(ext_svs) == len(ins), (
        f"{node.id}: scan_vars(step/invariant)={len(ext_svs)}本 だが "
        f'入力エッジ={len(ins)}本'
    )

    # initはrecurrentレベルのscan_vars要約には無く、body内のscan_var
    # ノード自身が持つ(schema.py参照。単一情報源をbody側に保つ)。
    state_vals: dict[str, np.ndarray] = {}
    for sv in state_svs:
        init = body.node_by_id(sv['id']).params['init']
        bits = np.asarray(init['bits'], dtype=np.int64)
        state_vals[sv['id']] = np.tile(bits, (batch, 1))

    stacked_rows: list[np.ndarray] = []
    out_t = None
    for t in range(steps):
        seed = dict(state_vals)
        for sv, ext_in in zip(ext_svs, ins, strict=True):
            w = int(sv['width'])
            if sv['role'] == 'step':
                seed[sv['id']] = ext_in[:, t * w:(t + 1) * w]
            else:  # invariant: 全ステップで共通
                seed[sv['id']] = ext_in
        body_values = _simulate_values(body, seed)
        for sv_id, body_node_id in p['state_update'].items():
            state_vals[sv_id] = body_values[body_node_id]
        if _recurrent_trace_sink is not None:
            _recurrent_trace_sink.setdefault(node.id, []).append(
                {k: v.copy() for k, v in state_vals.items()})
        out_t = body_values[p['output_source_id']]
        if p['output_mode'] == 'stacked':
            stacked_rows.append(out_t)

    if p['output_mode'] == 'stacked':
        return np.concatenate(stacked_rows, axis=1)
    return out_t


def simulate_trace(graph: IRGraph, x: np.ndarray) -> dict:
    """IRを実行し、全ノードの出力を返す(推論プローブA5の土台)。

    Args:
        graph: 検証済みIR
        x: (batch, in_width) の0/1配列

    Returns:
        {node_id: (batch, width)のint64配列}。入力ポートidの値も含む
    """
    if any(n.type in _LOWERABLE_TYPES for n in graph.nodes):
        return _simulate_trace_lowered(graph, x)

    x = np.asarray(x)
    assert set(np.unique(x)).issubset({0, 1}), '入力は0/1のみ'

    values: dict[str, np.ndarray] = {
        p.id: x.astype(np.int64) for p in graph.inputs
    }
    return _simulate_values(graph, values)


def simulate(graph: IRGraph, x: np.ndarray) -> np.ndarray:
    """IRを実行し最終出力を返す(simulate_traceのwrapper)。

    Args:
        graph: 検証済みIR
        x: (batch, in_width) の0/1配列

    Returns:
        出力ポートに接続されたノードの出力(未接続なら最終トポノード)。
        group_sumで終わる場合は(batch, k)の整数スコア
    """
    trace = simulate_trace(graph, x)
    out_src = graph.output_source()
    if out_src is not None:
        return trace[out_src]
    return trace[graph.topological_order()[-1].id]
