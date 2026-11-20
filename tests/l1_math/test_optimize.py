"""L1: conv2d重み共有木の縮約(griffith_logic/export/optimize.py)。

webbench(ブラウザ比較ツール)の convProgram.ts(TS版、挙動の真実源)の
Pythonミラー。ここでは:
    1. 既知パターン: 全パススルー木→0命令、定数木→定数
    2. ランダム小木のproperty test: 縮約前後の全数一致(verify_conv_program) +
       内訳の検算(const+alias+not+cse+emitted == before)
    3. クロス検証: webbench の gate_stats.mjs --json の出力と
       5モデルでgates_after(+全stats内訳)が一致することを確認する
       (モデルJSON・node実行環境が無い場合はskip。両方ともgitignore対象の
       ローカル成果物なのでCIでは基本skipになる)
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from griffith_logic.export.optimize import (
    REF_KIND_CONST,
    REF_KIND_TAP,
    ConvTreeSpec,
    build_conv_program,
    reduce_conv_tree,
    ref_idx,
    ref_kind,
    verify_conv_program,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBBENCH_DIR = REPO_ROOT / 'tools' / 'webbench'
MODELS_DIR = WEBBENCH_DIR / 'public' / 'models'

# gate_stats.mjs の既定5モデル(MODELS定数)と同一
MODELS = {
    'det': 'det_multimnist_lut_k256.json',
    'detmono': 'det_multimnist_mono_lut_k256.json',
    'detface': 'det_face_lut_k256.json',
    'detreal': 'det_fewclass_lut_k64.json',
    'seg': 'seg_convdlgn_k32.json',
}

# ConvProgramStats(snake_case) <-> convProgram.ts ConvProgramStats(camelCase)
FIELD_MAP = {
    'gatesBefore': 'gates_before',
    'gatesAfter': 'gates_after',
    'constFolded': 'const_folded',
    'aliasFolded': 'alias_folded',
    'notFolded': 'not_folded',
    'cseHits': 'cse_hits',
    'emitted': 'emitted',
    'instrsCreated': 'instrs_created',
    'deadRemoved': 'dead_removed',
    'notInstrs': 'not_instrs',
    'tapsBefore': 'taps_before',
    'nTaps': 'n_taps',
    'nSlots': 'n_slots',
    'ocConst': 'oc_const',
    'ocTap': 'oc_tap',
    'ocUnique': 'oc_unique',
}


class TestKnownPatterns:
    """既知パターン(全パススルー木・定数木)での縮約結果。"""

    def test_all_passthrough_tree_reduces_to_bare_tap(self):
        """全ゲートがtt=0xa(f(a,b)=a、恒等)の木は命令を1つも発行せず、
        oc出力は木の最左葉タップへの直接参照になる。"""
        spec = ConvTreeSpec(
            out_channels=2, tree_depth=2,
            indices_flat=[[10, 11, 12, 13], [20, 21, 22, 23]],
            truth_tables=[0xa, 0xa, 0xa, 0xa, 0xa, 0xa],
        )
        prog = build_conv_program(spec)
        verify_conv_program(spec, prog)

        assert prog.stats.gates_after == 0
        assert prog.stats.const_folded == 0
        assert prog.stats.alias_folded == prog.stats.gates_before
        for r in prog.oc_out:
            assert ref_kind(r) == REF_KIND_TAP
        assert [spec.indices_flat[0][0], spec.indices_flat[1][0]] == prog.taps

    @pytest.mark.parametrize('tt_const,expect_bit', [(0x0, 0), (0xf, 1)])
    def test_all_const_tree_reduces_to_constant(self, tt_const, expect_bit):
        """全ゲートが定数tt(0x0/0xf)の木は入力に関係なく定数へ畳まれる。"""
        spec = ConvTreeSpec(
            out_channels=3, tree_depth=2,
            indices_flat=[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]],
            truth_tables=[tt_const] * 9,
        )
        prog = build_conv_program(spec)
        verify_conv_program(spec, prog)

        assert prog.stats.gates_after == 0
        assert prog.stats.const_folded == prog.stats.gates_before
        assert prog.stats.n_taps == 0  # 定数木はタップを一切読まない
        for r in prog.oc_out:
            assert ref_kind(r) == REF_KIND_CONST
            assert ref_idx(r) == expect_bit


class TestRandomTreeProperties:
    """ランダム小木のproperty test(縮約前後の全数一致+内訳の検算)。"""

    @given(
        depth=st.integers(min_value=1, max_value=3),
        oc=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_reduction_is_lossless_and_breakdown_sums_to_before(self, depth, oc, data):
        n_leaves = 1 << depth
        n_internal = n_leaves - 1
        # タップIDのプールを狭くして再利用(same-ref/CSE/dead除去の経路)を誘発する
        n_tap_pool = data.draw(
            st.integers(min_value=1, max_value=max(1, n_leaves * oc // 2)),
            label='n_tap_pool',
        )
        indices_flat = data.draw(
            st.lists(
                st.lists(
                    st.integers(min_value=0, max_value=n_tap_pool - 1),
                    min_size=n_leaves, max_size=n_leaves,
                ),
                min_size=oc, max_size=oc,
            ),
            label='indices_flat',
        )
        truth_tables = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=oc * n_internal, max_size=oc * n_internal,
            ),
            label='truth_tables',
        )
        spec = ConvTreeSpec(
            out_channels=oc, tree_depth=depth,
            indices_flat=indices_flat, truth_tables=truth_tables,
        )
        prog = build_conv_program(spec)
        verify_conv_program(spec, prog)  # 縮約前後の関数等価性(全数照合)

        s = prog.stats
        assert s.gates_before == oc * n_internal
        # 内訳の検算: 元ゲートは必ずこの5分類のどれか1つに落ちる
        # (const/alias/not畳み込み、CSE一致、新規発行)。
        assert (s.const_folded + s.alias_folded + s.not_folded
                + s.cse_hits + s.emitted) == s.gates_before
        # emitted(2入力CSEでの新規発行)は draft へ追加されるが、単項NOTの
        # 新規発行(not_foldedの一部、CSE一致時は draft は増えない)も
        # draft へ追加され得るため instrs_created >= emitted。
        assert s.emitted <= s.instrs_created
        assert s.dead_removed == s.instrs_created - s.gates_after
        assert 0 <= s.gates_after <= s.instrs_created
        assert 0 <= s.n_taps <= s.taps_before
        assert s.oc_const + s.oc_tap <= oc
        assert 1 <= s.oc_unique <= oc


def _run_node_gate_stats() -> dict | None:
    """webbench の gate_stats.mjs --json を実行し {task: result} を返す。

    nodeが無い/実行に失敗した場合はNoneを返す(呼び出し側でskip)。
    """
    if shutil.which('node') is None:
        return None
    try:
        proc = subprocess.run(
            ['node', 'scripts/gate_stats.mjs', '--json'],
            cwd=WEBBENCH_DIR, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return {r['task']: r for r in results}


def _models_available() -> bool:
    return MODELS_DIR.is_dir() and all((MODELS_DIR / f).is_file() for f in MODELS.values())


@pytest.mark.skipif(
    not _models_available(),
    reason='webbench のモデルJSON(gitignore対象のローカル成果物)が無い',
)
class TestCrossValidateWithTsGateStats:
    """gate_stats.mjs(TS版convProgram.ts)とreduce_conv_tree(Pyミラー)の一致検証。"""

    @pytest.fixture(scope='class')
    def ts_results(self):
        results = _run_node_gate_stats()
        if results is None:
            pytest.skip('node実行不可(未インストール、または gate_stats.mjs 実行失敗)')
        return results

    def test_gates_after_and_stats_match_ts_for_all_models(self, ts_results):
        mismatches = []
        checked_nodes = 0
        for task, fname in MODELS.items():
            doc = json.loads((MODELS_DIR / fname).read_text())
            ts_nodes = {n['id']: n for n in ts_results[task]['nodes']}
            for node in doc['ir']['nodes']:
                if node['type'] != 'conv2d':
                    continue
                assert int(node['params'].get('n_inputs', 2)) == 2
                stats = reduce_conv_tree(node['params'])
                ts_node = ts_nodes[node['id']]
                checked_nodes += 1
                for ts_key, py_key in FIELD_MAP.items():
                    py_val = getattr(stats, py_key)
                    if ts_node[ts_key] != py_val:
                        mismatches.append(
                            f'{task}/{node["id"]}: {ts_key}(ts)={ts_node[ts_key]} != '
                            f'{py_key}(py)={py_val}'
                        )
        assert checked_nodes > 0, 'conv2dノードが1つも見つからなかった'
        assert not mismatches, '\n'.join(mismatches)
