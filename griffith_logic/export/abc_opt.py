"""ABC(UC Berkeley論理合成ツール)によるIR最適化(compile-pipeline)。

blif.pyのIR⇔BLIF変換を使い、group_sum手前のlut_layer core部分を
ABCの等価保存フローで最適化する。最適化後は必ずsimulateで元IRとの
ビット厳密一致を検証する(等価性検証はskip不可)。

    abc_available          abc/berkeley-abc/yosys-abcの探索(キャッシュ)
    with_input_projection  温度計グループの射影LUT層を入力直後に前置
    run_abc_optimize       core→BLIF→ABC→BLIF→IR+等価性検証
    run_abc_cec            ABC cecによる2つのcoreの厳密等価判定

射影トリック(EXDC直接方式の代替、プローブ検証済み):
    ABCの.exdcセクションはmfs/mfs2/mfseのどれも実際には使わない。
    代わりに「有効符号語への射影LUT層」を回路の前に合成してから
    通常の等価保存フローにかけると、無効パターンが射影後段の
    充足不能ドントケアになり自動活用される(mfs2/rewrite系が
    射影との合成関数として簡約する)。

ABCスクリプトの注意:
    - `resyn2` はabc.rcのエイリアスであり、-cのバッチ実行では
      未定義(cmd error)。既定スクリプトでは標準展開をインライン化する
    - `sweep` はロジックネットワーク用コマンドなのでmfs2後に置く
    - ABCはcwdにabc.historyを吐くため必ずworkdir内で実行する
      (subprocess cwd=workdir)
"""

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from ..ir.schema import IREdge, IRGraph, IRNode
from ..ir.serialize import tt_to_hex
from ..ir.simulate import simulate
from ..ir.validate import validate
from .blif import (
    blif_to_ir,
    pad_tt,
    read_blif,
    reattach_group_sum,
    split_lut_core,
    write_blif,
)
from .estimate import estimate_resources

# resyn2エイリアスの標準展開(abc.rc非依存)
_RESYN2 = ('balance; rewrite; refactor; balance; rewrite; rewrite -z; '
           'balance; refactor -z; rewrite -z; balance')

# mfs2 は既定で don't care を使わない(Sfm_ParSetDefault の fUseDcs=0)。
# `-l` でDC活用をONにする。射影トリック(with_input_projection)で
# 作った充足不能DCはこの `-l` で初めて効き、SOPキューブ数が減る
# (ランダム16ケース実測: 合計 5370→5084 = -5.3%、LUT数は同数、失敗0件)。
#
# `-a`(純area最小化)は**付けてはいけない**: 実測でキューブ数・LUT数とも
# 悪化するうえ、射影付きの中規模ケースで16件中5件が
# `Assertion failed: (nVars == 0), Sfm_TruthToCnf, sfmCnf.c:79` でabort する。
DEFAULT_SCRIPT = f'strash; dc2; {_RESYN2}; strash; if -K 6; mfs2 -l; sweep'

_ABC_CANDIDATES = ('abc', 'berkeley-abc', 'yosys-abc')
_abc_cmd_cache: str | None = None
_abc_searched = False


def abc_available() -> bool:
    """ABCコマンドが利用可能か(見つかったコマンド名はキャッシュ)。"""
    global _abc_cmd_cache, _abc_searched
    if not _abc_searched:
        for cand in _ABC_CANDIDATES:
            if shutil.which(cand) is not None:
                _abc_cmd_cache = cand
                break
        _abc_searched = True
    return _abc_cmd_cache is not None


def _abc_cmd() -> str:
    assert abc_available(), (
        f'ABCが見つからない(探索: {"/".join(_ABC_CANDIDATES)})'
    )
    return _abc_cmd_cache


def _run_abc(command: str, workdir: str) -> str:
    """abc -c "{command}" をworkdir内で実行しstdoutを返す。

    非0終了・cmd errorはRuntimeError。abc.historyがcwdに落ちるため
    必ずcwd=workdirで実行する。
    """
    proc = subprocess.run(
        [_abc_cmd(), '-c', command],
        cwd=workdir, capture_output=True, text=True,
    )
    if proc.returncode != 0 or 'cmd error' in proc.stdout:
        raise RuntimeError(
            f'abc実行失敗(exit={proc.returncode}, cmd={command!r}):\n'
            f'--- stdout ---\n{proc.stdout[-2000:]}\n'
            f'--- stderr ---\n{proc.stderr[-2000:]}'
        )
    return proc.stdout


def _thermo_tt(b: int, level: int) -> int:
    """bビットグループの出力ビットlevel: 1[popcount(t) >= level+1] の真理値表。

    アドレスkのbit j = グループ内スロットj(しきい値昇順)の入力値。
    """
    tt = 0
    for k in range(1 << b):
        if bin(k).count('1') >= level + 1:
            tt |= 1 << k
    return tt


def with_input_projection(graph: IRGraph, groups: list[list[int]]) -> IRGraph:
    """温度計グループごとの射影LUT層を入力直後に前置したIRを返す(純関数)。

    Args:
        graph: 単一入力ポートのIR(元グラフは変更しない)
        groups: 各グループの入力bit index列(温度計しきい値昇順)。
            グループ長b≦6。出力ビットi(=位置groups[g][i])は
            1[popcount(グループ入力) >= i+1]。単調有効符号語上では恒等、
            無効パターンは同popcountの有効符号語へ射影される

    グループに属さない入力bitは恒等通過(fan-in 1の恒等ttをpad_ttで
    fan-in統一)。射影層の出力ビット順=入力ビット順なので後続ノードの
    indicesは変更不要(入力ポート発エッジの付け替えのみ)。
    """
    validate(graph)
    assert len(graph.inputs) == 1, 'with_input_projectionは単一入力ポートのみ対応'
    in_port = graph.inputs[0]
    width = int(in_port.width)

    flat = [int(i) for g in groups for i in g]
    assert len(set(flat)) == len(flat), 'グループ間で入力bitが重複している'
    assert all(0 <= i < width for i in flat), (
        f'グループのbit indexが入力幅({width})を超えている'
    )
    group_of: dict[int, tuple[list[int], int]] = {}
    for g in groups:
        assert 1 <= len(g) <= 6, f'グループ長{len(g)}は1..6のみ対応'
        for lvl, bit in enumerate(g):
            group_of[int(bit)] = ([int(i) for i in g], lvl)

    n_inputs = max([len(g) for g in groups] + [1])
    indices = np.zeros((n_inputs, width), dtype=np.int64)
    tts: list[int] = []
    for o in range(width):
        if o in group_of:
            g, lvl = group_of[o]
            srcs = g
            tt = _thermo_tt(len(g), lvl)
        else:
            srcs = [o]
            tt = 0x2  # fan-in 1の恒等
        padded = srcs + [srcs[0]] * (n_inputs - len(srcs))
        for j in range(n_inputs):
            indices[j, o] = padded[j]
        tts.append(pad_tt(tt, len(srcs), n_inputs))

    existing = {n.id for n in graph.nodes}
    proj_id = 'proj_in'
    while proj_id in existing:
        proj_id += '_'
    proj_node = IRNode(
        id=proj_id,
        type='lut_layer',
        params={
            'in_width': width,
            'out_width': width,
            'n_inputs': n_inputs,
            'indices': indices,
            'truth_tables': tt_to_hex(tts),
        },
        train_info={
            'source': 'with_input_projection',
            'groups': [[int(i) for i in g] for g in groups],
        },
    )

    in_id = in_port.id
    new_edges = [IREdge(src=in_id, dst=proj_id)]
    for e in graph.edges:
        src = proj_id if e.src == in_id else e.src
        new_edges.append(IREdge(src=src, dst=e.dst, dst_slot=e.dst_slot))

    new_graph = IRGraph(
        metadata={
            **graph.metadata,
            'input_projection': [[int(i) for i in g] for g in groups],
        },
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        nodes=[proj_node, *graph.nodes],
        edges=new_edges,
    )
    validate(new_graph)
    return new_graph


def valid_codewords(
    rng: np.random.Generator, n: int, width: int, groups: list[list[int]],
) -> np.ndarray:
    """有効符号語のみのランダム入力を生成する。

    各グループでレベル0..bを一様に引いて単調列化
    (bit i = 1[level >= i+1])。グループ外bitは一様ランダム。
    """
    x = rng.integers(0, 2, size=(n, width), dtype=np.int64)
    for g in groups:
        levels = rng.integers(0, len(g) + 1, size=n)
        for i, bit in enumerate(g):
            x[:, int(bit)] = (levels >= i + 1).astype(np.int64)
    return x


def _count_cubes(blif_path: str) -> int:
    """BLIFファイルの総cube行数(.namesのSOP行数)。"""
    net = read_blif(blif_path)
    return sum(len(cubes) for _fanins, _out, cubes in net.names)


def run_abc_optimize(
    graph: IRGraph,
    *,
    script: str = DEFAULT_SCRIPT,
    projection_groups: list[list[int]] | None = None,
    workdir: str | None = None,
    verify_vectors: int = 1024,
    calib_x: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[IRGraph, dict]:
    """coreをABCで最適化し、等価性検証済みのIRと統計を返す。

    フロー: (射影前置) → split_lut_core → write_blif → abc(script)
    → read_blif → blif_to_ir → reattach_group_sum → simulate照合。

    等価性検証(必須、skip不可):
        - projection_groups=None: ランダムverify_vectors本(+calib_x)で
          元graphとsimulate完全一致をassert
        - projection_groups指定: 有効符号語のみ(+calib_x)で
          射影適用済みgraph(=最適化入力)との一致をassert。
          calib_xは有効符号語であることが呼び出し側の責任

    Returns:
        (optimized_graph, stats)。statsのluts/depth_beforeは
        射影適用後(=最適化入力)基準。射影追加LUT数はn_projection_lutsに別掲
    """
    base = graph
    if projection_groups:
        base = with_input_projection(graph, projection_groups)

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='abc_opt_')
    os.makedirs(workdir, exist_ok=True)

    core, gs_node = split_lut_core(base)
    core_path = os.path.join(workdir, 'core.blif')
    out_path = os.path.join(workdir, 'out.blif')
    write_blif(core, core_path)

    t0 = time.perf_counter()
    stdout = _run_abc(f'read_blif core.blif; {script}; write_blif out.blif',
                      workdir)
    wall_sec = time.perf_counter() - t0
    if not os.path.exists(out_path):
        raise RuntimeError(
            f'abcがout.blifを生成しなかった:\n{stdout[-2000:]}'
        )

    in_width = sum(int(p.width) for p in core.inputs)
    opt_core = blif_to_ir(read_blif(out_path), in_width=in_width)
    optimized = reattach_group_sum(opt_core, gs_node)

    # 等価性検証(必須)
    rng = np.random.default_rng(seed)
    if projection_groups:
        x = valid_codewords(rng, verify_vectors, in_width, projection_groups)
    else:
        x = rng.integers(0, 2, size=(verify_vectors, in_width),
                         dtype=np.int64)
    if calib_x is not None:
        x = np.vstack([x, np.asarray(calib_x, dtype=np.int64)])
    expected = simulate(base, x)
    actual = simulate(optimized, x)
    assert np.array_equal(expected, actual), (
        f'ABC最適化後IRが元と不一致'
        f'({int((expected != actual).any(axis=1).sum())}/{len(x)}本)'
    )

    before = estimate_resources(base)
    after = estimate_resources(optimized)
    n_projection_luts = int(graph.inputs[0].width) if projection_groups else 0
    stats = {
        'luts_before': before['total_luts'],
        'luts_after': after['total_luts'],
        'depth_before': before['logic_depth'],
        'depth_after': after['logic_depth'],
        'n_cubes_before': _count_cubes(core_path),
        'n_cubes_after': _count_cubes(out_path),
        'wall_sec': wall_sec,
        'abc_stdout': stdout[-2000:],
        'script': script,
        'n_projection_luts': n_projection_luts,
        'workdir': workdir,
    }
    return optimized, stats


def run_abc_cec(g1: IRGraph, g2: IRGraph, workdir: str | None = None) -> dict:
    """ABC cecで2つのIRのcore部分の厳密等価性を判定する。

    両IRをsplit_lut_core→BLIF化し `cec a.blif b.blif` を実行。
    PI/PO名(x_i/y_i)はwrite_blifの規約で揃うため名前対応で照合される。

    Returns:
        {'equivalent': bool, 'stdout': str}
    """
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='abc_cec_')
    os.makedirs(workdir, exist_ok=True)
    core1, _ = split_lut_core(g1)
    core2, _ = split_lut_core(g2)
    write_blif(core1, os.path.join(workdir, 'a.blif'), model_name='a')
    write_blif(core2, os.path.join(workdir, 'b.blif'), model_name='b')
    stdout = _run_abc('cec a.blif b.blif', workdir)
    # 等価: "Networks are equivalent." / 非等価: "Networks are NOT EQUIVALENT."
    return {
        'equivalent': 'Networks are equivalent' in stdout,
        'stdout': stdout,
    }
