"""tools/nas/search.py のテスト。

方針:
    - 純ロジック(genome変異・パレート判定)は高速に単体検証する。
    - 探索の完走・中断再開・決定性ゴールデンは小規模XOR(max_trials<=6、
      epoch数最小)で数秒に収める。
    - 決定性ゴールデンは構造指標(genome/luts/depth/gates/partial/
      parent/pareto)を厳密照合し、accはプラットフォーム間BLAS差を吸収する
      許容帯で照合する。更新は MD_UPDATE_GOLDEN=1 のみ(規約)。
"""

import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.ir import ir_from_dict, validate
from tools.nas.search import (
    SearchConfig,
    SearchSpace,
    Trial,
    load_trial_ir,
    mutate_genome,
    pareto_front,
    random_genome,
    run_search,
)

GOLDEN_PATH = Path(__file__).parent / 'golden' / 'nas_xor_search.json'
# accの許容絶対誤差(構造指標は厳密、accのみBLAS差を吸収)
ACC_ATOL = 1e-3

_SMALL_SPACE = SearchSpace(
    layers_choices=(1, 2), width_choices=(4, 8), n_inputs_choices=(2,)
)


def _cfg(out_root, **kw) -> SearchConfig:
    base = dict(
        search_id='t', dataset='logic:xor', model_kind='logic_mlp',
        max_trials=6, epochs=3, seed=0, out_root=str(out_root), space=_SMALL_SPACE,
    )
    base.update(kw)
    return SearchConfig(**base)


# ------------------------------------------------------------------
# 純ロジック
# ------------------------------------------------------------------


class TestGenome:
    def test_random_genome_in_space(self):
        rng = random.Random(0)
        for _ in range(50):
            g = random_genome(rng, _SMALL_SPACE)
            assert g['layers'] in _SMALL_SPACE.layers_choices
            assert g['width'] in _SMALL_SPACE.width_choices
            assert g['n_inputs'] in _SMALL_SPACE.n_inputs_choices

    def test_mutate_genome_stays_in_space_and_changes_one_axis(self):
        rng = random.Random(1)
        parent = {'layers': 2, 'width': 8, 'n_inputs': 2}
        for _ in range(50):
            child = mutate_genome(rng, parent, _SMALL_SPACE)
            assert child['layers'] in _SMALL_SPACE.layers_choices
            assert child['width'] in _SMALL_SPACE.width_choices
            # 高々1軸だけ変わる
            diff = sum(child[k] != parent[k] for k in parent)
            assert diff <= 1

    def test_mutate_is_nondestructive(self):
        rng = random.Random(2)
        parent = {'layers': 2, 'width': 4, 'n_inputs': 2}
        before = dict(parent)
        mutate_genome(rng, parent, _SMALL_SPACE)
        assert parent == before


class TestPareto:
    def _t(self, tid, acc, luts, partial=False, state='done'):
        return Trial(trial_id=tid, genome={}, state=state, acc=acc, best_acc=acc,
                     total_luts=luts, logic_depth=1, wire_count=1, group_sums=0,
                     gates=1, num_params=1, partial=partial, parent_id=None,
                     steps=0, seed=0)

    def test_nondominated_only(self):
        trials = [
            self._t(0, 0.5, 10),   # 支配される
            self._t(1, 0.9, 5),    # 非劣
            self._t(2, 1.0, 8),    # 非劣(accで勝る)
            self._t(3, 0.9, 8),    # trial1に支配される
        ]
        assert pareto_front(trials) == [1, 2]

    def test_partial_excluded(self):
        trials = [
            self._t(0, 1.0, 1, partial=True),   # 中空=除外
            self._t(1, 0.6, 3),
        ]
        assert pareto_front(trials) == [1]

    def test_failed_excluded(self):
        trials = [self._t(0, 0.0, 1, state='failed'), self._t(1, 0.7, 2)]
        assert pareto_front(trials) == [1]


# ------------------------------------------------------------------
# 探索の完走・IR健全性・中断再開
# ------------------------------------------------------------------


@pytest.mark.learning
class TestSearchRun:
    def test_completes_and_records_metrics(self, tmp_path):
        ledger = run_search(_cfg(tmp_path))
        assert len(ledger) == 6
        for t in ledger:
            assert t.state == 'done'
            assert 0.0 <= t.acc <= 1.0
            assert t.total_luts >= 1
            assert t.logic_depth >= 1

    def test_trial_ir_is_valid(self, tmp_path):
        run_search(_cfg(tmp_path, max_trials=3))
        cfg = _cfg(tmp_path, max_trials=3)
        for tid in range(3):
            ir_json = load_trial_ir(cfg, tid)
            assert ir_json is not None
            validate(ir_from_dict(ir_json))  # 例外なし=健全

    def test_resume_is_identical_to_uninterrupted(self, tmp_path):
        # 無中断
        full = run_search(_cfg(tmp_path / 'full'))
        # 中断(3)→再開(6)
        run_search(_cfg(tmp_path / 'res', max_trials=3))
        resumed = run_search(_cfg(tmp_path / 'res', max_trials=6))
        assert len(resumed) == 6
        for a, b in zip(full, resumed, strict=True):
            assert a.trial_id == b.trial_id
            assert a.genome == b.genome
            assert a.parent_id == b.parent_id
            assert a.total_luts == b.total_luts
            assert a.acc == pytest.approx(b.acc, abs=ACC_ATOL)

    def test_stop_check_halts_early(self, tmp_path):
        calls = {'n': 0}

        def stop():
            calls['n'] += 1
            return calls['n'] > 2  # 2trial後に停止

        ledger = run_search(_cfg(tmp_path), stop_check=stop)
        assert len(ledger) == 2


# ------------------------------------------------------------------
# 決定性ゴールデン
# ------------------------------------------------------------------


@pytest.mark.learning
class TestGolden:
    def test_xor_search_matches_golden(self, tmp_path):
        ledger = run_search(_cfg(tmp_path))
        pareto = pareto_front(ledger)
        actual = {
            'seed': 0, 'max_trials': 6, 'epochs': 3,
            'space': {
                'layers_choices': list(_SMALL_SPACE.layers_choices),
                'width_choices': list(_SMALL_SPACE.width_choices),
                'n_inputs_choices': list(_SMALL_SPACE.n_inputs_choices),
            },
            'trials': [
                {
                    'trial_id': t.trial_id, 'genome': t.genome,
                    'acc': round(t.acc, 4), 'total_luts': t.total_luts,
                    'logic_depth': t.logic_depth, 'gates': t.gates,
                    'partial': t.partial, 'parent_id': t.parent_id,
                    'state': t.state,
                }
                for t in ledger
            ],
            'pareto': pareto,
        }

        if os.environ.get('MD_UPDATE_GOLDEN') == '1':
            GOLDEN_PATH.write_text(
                json.dumps(actual, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            pytest.skip('ゴールデンを更新した(MD_UPDATE_GOLDEN=1)')

        golden = json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))
        assert golden['pareto'] == pareto
        assert len(golden['trials']) == len(actual['trials'])
        for g, a in zip(golden['trials'], actual['trials'], strict=True):
            # 構造指標は厳密照合
            assert g['trial_id'] == a['trial_id']
            assert g['genome'] == a['genome']
            assert g['total_luts'] == a['total_luts']
            assert g['logic_depth'] == a['logic_depth']
            assert g['gates'] == a['gates']
            assert g['partial'] == a['partial']
            assert g['parent_id'] == a['parent_id']
            assert g['state'] == a['state']
            # accのみ許容帯(BLAS差吸収)
            assert a['acc'] == pytest.approx(g['acc'], abs=ACC_ATOL)
