"""L3: 計算型配線(rot)の較正アンカー C3 / C4(@paper)。

固定配線どうしの比較で成り立つと主張している 2 点を、**本番構成のまま**
(configs/recipes/cwo_*.yaml をそのまま、8 エポック × 2 seed)再走して assert する。

    C3(MNIST 48K): rot(P1)は固定ランダム配線(P0)を 2 seed 平均で
                   +0.4pt 以上上回る(原値 +0.95pt。先行実験実測表掃引A)
    C4(CIFAR-10 48K): 受容野を満たした rot(SR2、クラス受容野 100%)は
                   置換配線(P)を 2 seed 平均で下回らない(原値 +1.10 / +1.31pt)

判定は「差」で行う。腕ごとの絶対値は環境(GPU・torch・データの取得元)で
数 pt 動きうるが、同じ環境で同じ epoch 数だけ回した固定配線どうしの差は主張の本体で、
これが崩れたら主張が崩れる。閾値は非公開の主張台帳(C3 / C4 の許容幅)からの転記で、
実装の実測に合わせて緩めない。

所要: GPU で C3 が数分 × 4 run、C4 が十数分 × 4 run。@paper なので既定の
addopts から外れる(夜間 / Modal 実行向け)。

実行:
    uv run pytest -m paper tests/l3_paper/test_computed_wiring_anchors.py
    CWO_ANCHOR_SEEDS=0 uv run pytest -m paper \
        tests/l3_paper/test_computed_wiring_anchors.py   # 1 seed だけで様子見
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.training.config import load_config
from griffith_logic.training.runner import run_training

pytestmark = pytest.mark.paper

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RECIPE_DIR = os.path.join(REPO_ROOT, 'configs', 'recipes')
ANCHORS_PATH = os.path.join(
    REPO_ROOT, 'experiments', 'computed-wiring-overlay', 'anchors.json'
)

# 環境変数で seed を絞れる(既定は claims.json と同じ 2 seed)
SEEDS = tuple(
    int(s) for s in os.environ.get('CWO_ANCHOR_SEEDS', '0,1').split(',')
)


def _anchors() -> dict:
    with open(ANCHORS_PATH, encoding='utf-8') as f:
        return json.load(f)['anchors']


def _rule(anchor_id: str, name: str) -> dict:
    for rule in _anchors()[anchor_id]['difference_rules']:
        if rule['name'] == name:
            return rule
    raise AssertionError(f'{anchor_id} に差の規則 {name!r} が無い')


def _passes(delta_pt: float, rule: dict) -> bool:
    """差の規則の判定(anchors.json の comparison。'gte' = 以上 / 'gt' = 超過)。"""
    minimum = float(rule['min_pt'])
    comparison = rule.get('comparison', 'gte')
    if comparison == 'gt':
        return delta_pt > minimum
    if comparison == 'gte':
        return delta_pt >= minimum
    raise AssertionError(f'未知の comparison: {comparison!r}')


def _mean_metric(recipe: str, metric: str, tmp_path) -> float:
    """レシピを seed ごとに本番構成で回し、指標の seed 平均を返す。"""
    values = []
    for seed in SEEDS:
        cfg = load_config(
            os.path.join(RECIPE_DIR, f'{recipe}.yaml'),
            [f'run.seed={seed}', 'run.wandb=false',
             f'run.run_name={recipe}_s{seed}'],
        )
        result = run_training(
            cfg, out_dir=str(tmp_path / f'{recipe}_s{seed}')
        )
        assert result[metric] is not None
        values.append(float(result[metric]))
    return sum(values) / len(values)


class TestC3RotBeatsRandomWiring:
    """C3: MNIST 48K で rot(P1)が固定ランダム配線(P0)を上回る"""

    def test_p1_minus_p0_at_least_min_pt(self, tmp_path):
        rule = _rule('C3', 'P1 - P0 (48K)')
        metric = _anchors()['C3']['metric']
        p1 = _mean_metric(rule['positive'], metric, tmp_path)
        p0 = _mean_metric(rule['negative'], metric, tmp_path)
        delta_pt = (p1 - p0) * 100
        assert _passes(delta_pt, rule), (
            f'C3 が崩れた: P1 {p1:.4f} - P0 {p0:.4f} = {delta_pt:+.2f}pt '
            f'は下限 {rule["min_pt"]:+.2f}pt を満たさない(原値 {rule["original_pt"]:+.2f}pt、'
            f'seeds={SEEDS})'
        )


class TestC4ReceptiveFieldCondition:
    """C4: 受容野を満たした rot(SR2)が置換配線(P)を下回らない"""

    def test_sr2_minus_p_at_least_min_pt(self, tmp_path):
        rule = _rule('C4', 'SR2 - P')
        metric = _anchors()['C4']['metric']
        sr2 = _mean_metric(rule['positive'], metric, tmp_path)
        perm = _mean_metric(rule['negative'], metric, tmp_path)
        delta_pt = (sr2 - perm) * 100
        assert _passes(delta_pt, rule), (
            f'C4 が崩れた: SR2 {sr2:.4f} - P {perm:.4f} = {delta_pt:+.2f}pt '
            f'は下限 {rule["min_pt"]:+.2f}pt を満たさない(原値 {rule["original_pt"]:+.2f}pt、'
            f'seeds={SEEDS})'
        )
