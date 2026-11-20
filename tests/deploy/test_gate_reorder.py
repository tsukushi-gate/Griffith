"""配線の近さによるゲート並べ替え。

固定したい性質:

    - **値が1ビットも変わらない**。並べ替えは出力番号の付け替えなので、
      読み手の添字を書き換え損ねると静かに壊れる
    - **順序に意味がある層は動かさない**。group_sum は入力の並びが
      クラス分けそのものなので、その手前の層を並べ替えてはいけない
    - **しきい値以下では何もしない**。効くのは層の出力がL1を溢れてから
      (実測: 6KB で 1.00倍、391KB で 1.27倍)なので、小さいモデルでは
      生成物が1バイトも変わらないことを固定する

しきい値超えの検査は、巨大なモデルを組む代わりに**しきい値のほうを
下げて**行う(並べ替えの論理は幅に依存しないため)。
"""

import numpy as np

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.plan import build_plan, reorder_dense_gates
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .test_c_codegen_e2e import run_generated


def _luts(plan):
    return [o for o in plan.ops if o.kind == 'lut_dense']


def _snapshot(plan):
    return [(list(o.attrs['indices']), list(o.attrs['truth_tables']))
            for o in _luts(plan)]


def test_small_model_is_left_alone(dense_doc):
    """しきい値以下のモデルは1バイトも変わらない。"""
    plan = build_plan(build_bundle(dense_doc))
    before = _snapshot(plan)
    assert reorder_dense_gates(plan) == []
    assert _snapshot(plan) == before


def test_reorders_only_the_layers_it_may(monkeypatch, dense_doc):
    """並べ替わるのは「読み手が lut_dense ただ1つ」の層だけ。"""
    monkeypatch.setattr('griffith_logic.deploy.plan.REORDER_MIN_BYTES', 8)
    plan = build_plan(build_bundle(dense_doc))
    luts = _luts(plan)
    assert len(luts) >= 2, 'この検査には dense が2層以上要る'

    head = luts[0]
    w = head.attrs['out_width']
    assert head.attrs['indices'][:w] == sorted(head.attrs['indices'][:w]), \
        '前段が a 添字の昇順になっていない'

    tail = luts[-1]                       # 読み手は group_sum
    w = tail.attrs['out_width']
    idx = list(tail.attrs['indices'][:w])
    assert idx != sorted(idx), 'group_sum 直前の層まで並べ替えてしまっている'


def test_reordered_plan_gives_the_same_answer(monkeypatch, dense_doc):
    """並べ替えの有無で、素朴に回した結果が完全に一致する。"""
    plain = build_plan(build_bundle(dense_doc))
    monkeypatch.setattr('griffith_logic.deploy.plan.REORDER_MIN_BYTES', 8)
    shuffled = build_plan(build_bundle(dense_doc))
    assert _snapshot(plain) != _snapshot(shuffled), '並べ替えが起きていない'

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(6, plain.input_width)).astype(np.uint8)
    assert np.array_equal(_run_plan(plain, bits), _run_plan(shuffled, bits))


def test_generated_c_still_matches_simulate(monkeypatch, cc, tmp_path,
                                            dense_doc):
    """並べ替えたモデルの生成Cが `ir/simulate.py` と整数厳密一致する。"""
    monkeypatch.setattr('griffith_logic.deploy.plan.REORDER_MIN_BYTES', 8)
    bundle = build_bundle(dense_doc)
    graph = ir_from_dict(bundle['ir'])
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, size=(1, int(graph.inputs[0].width)),
                     dtype=np.uint8)
    want = simulate(graph, x)[0].astype(np.int64)
    assert np.array_equal(run_generated(cc, tmp_path, bundle, x[0]), want)


def _run_plan(plan, bits):
    """プランを素朴に実行する(参照。速度は問わない)。"""
    out = []
    for row in bits:
        vals = {plan.input_value: [int(v) for v in row]}
        for op in plan.ops:
            a = op.attrs
            if op.kind == 'lut_dense':
                x = vals[op.inputs[0]]
                w, n = a['out_width'], a['n_inputs']
                idx, tt = a['indices'], a['truth_tables']
                vals[op.output] = [
                    (tt[g] >> sum(x[idx[j * w + g]] << j
                                  for j in range(n))) & 1
                    for g in range(w)]
            elif op.kind == 'reduce_scores':
                x = vals[op.inputs[0]]
                gs = a['in_width'] // a['k']
                vals[op.output] = [sum(x[i * gs:(i + 1) * gs])
                                   for i in range(a['k'])]
            elif op.kind == 'to_scores':
                vals[op.output] = list(vals[op.inputs[0]])
            else:
                raise AssertionError(f'想定外の op: {op.kind}')
        out.append(vals[plan.output_value])
    return np.asarray(out, dtype=np.int64)
