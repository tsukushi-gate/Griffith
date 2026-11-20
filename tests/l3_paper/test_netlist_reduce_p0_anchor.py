"""L3(較正アンカー 2): 縮約器が社内先行実験 P0 の実測値を再現するか。

先行実験 phase 1 の主指標 U(実効ゲート ÷ N)の測定装置として
`export/netlist_reduce.py` を移植した。その忠実性は「P0 が縮約済みの
既存ネットに掛けて内訳が完全一致するか」でしか示せないので、
凍結アーカイブ社内先行実験の実ネットリストを golden に使う。

golden の出所(読み取り専用):
    <社内先行実験>/pilot/experiments/irreducible-dlgn-p0/
        p0_results.json                          … 46 ネットの内訳(正)
        runs/exp056/<arm>/netlist.json           … 48,000 ゲートの実ネット
        mnist_test_bin.npz                       … MNIST テストの二値化ビット

閾値は無い(整数の完全一致のみ)。参照値は P0 の実測なので、
合わなければ移植の誤りであり、期待値を動かして通してはならない。

パスが解決できない環境(アーカイブ未配置)では skip する。
`GRIFFITH_P0_ANCHOR_DIR` 環境変数で明示指定もできる。

実行: uv run pytest tests/l3_paper/test_netlist_reduce_p0_anchor.py -q -m paper
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.export.netlist_reduce import analyze

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CANDIDATES = (
    os.environ.get('GRIFFITH_P0_ANCHOR_DIR', ''),
    os.path.join(_REPO, '..', '社内先行実験', 'pilot', 'experiments',
                 'irreducible-dlgn-p0'),
    os.path.join(_REPO, '..', '..', '社内先行実験', 'pilot', 'experiments',
                 'irreducible-dlgn-p0'),
)

# 較正アンカー 2 の対象(先行実験群 D、w8000×d6 = 48,000 ゲート)
ARMS = ('P6_w8000_s0', 'P7_w8000_s0', 'P0_w8000_s0')

# 内訳の一致から外す欄: 実行時間(実測)と等価性(別途 assert する)
_SKIP_FIELDS = ('reduce_time_s', 'equivalence')

# P0 の実測アンカー(`p0_results.json` からの転記。取り違え検出用の固定値)
_ANCHOR_EFFECTIVE = {
    'P6_w8000_s0': (14718, 48000, 0.306625),
    'P7_w8000_s0': (17770, 48000, 0.3702083333333333),
    'P0_w8000_s0': (14219, 48000, 0.29622916666666665),
}


def _p0_dir():
    for cand in _CANDIDATES:
        if not cand:
            continue
        path = os.path.abspath(cand)
        if os.path.isfile(os.path.join(path, 'p0_results.json')):
            return path
    return None


P0_DIR = _p0_dir()
pytestmark = [pytest.mark.paper,
              pytest.mark.skipif(
                  P0_DIR is None,
                  reason='社内先行実験の P0 パイロット結果が見つからない'
                         '(GRIFFITH_P0_ANCHOR_DIR で指定可)')]


@pytest.fixture(scope='module')
def p0_reference():
    """p0_results.json の nets を name -> エントリの dict にして返す。"""
    with open(os.path.join(P0_DIR, 'p0_results.json'), encoding='utf-8') as f:
        return {e['name']: e for e in json.load(f)['nets']}


@pytest.fixture(scope='module')
def mnist_bits():
    """MNIST テストの二値化ビット(無ければ None → ランダムのみで検証)。"""
    path = os.path.join(P0_DIR, 'mnist_test_bin.npz')
    if not os.path.isfile(path):
        return None
    return np.load(path)['X'][:2000].astype(np.uint8)


@pytest.fixture(scope='module')
def results(p0_reference, mnist_bits):
    """3 腕を 1 回ずつ縮約して結果をキャッシュする(1 腕 ≈ 1.4 秒)。"""
    out = {}
    for arm in ARMS:
        path = os.path.join(P0_DIR, 'runs', 'exp056', arm, 'netlist.json')
        assert os.path.isfile(path), f'ネットリストが無い: {path}'
        with open(path, encoding='utf-8') as f:
            netlist = json.load(f)
        out[arm] = analyze(netlist, X_data=mnist_bits, n_random=2000, seed=0)
    return out


@pytest.mark.parametrize('arm', ARMS)
class TestP0Anchor:
    def test_all_p0_fields_match(self, arm, results, p0_reference):
        """P0 の内訳・構造統計・層別内訳が 1 欄も違わずに一致する。"""
        got = results[arm]
        ref = p0_reference['exp056_' + arm]
        compared = 0
        mismatch = []
        for key, value in ref.items():
            if key in _SKIP_FIELDS or key not in got:
                continue
            compared += 1
            if got[key] != value:
                mismatch.append((key, value, got[key]))
        assert not mismatch, f'{arm}: P0 と不一致 {mismatch}'
        # 交差が空のまま「一致」になる事故を防ぐ
        assert compared >= 30, f'比較できた欄が {compared} しかない'

    def test_effective_gates_and_ratio(self, arm, results):
        """実効ゲート数・総ゲート数・実効率が P0 の実測値と一致する。"""
        eff, total, ratio = _ANCHOR_EFFECTIVE[arm]
        got = results[arm]
        assert got['effective_gates'] == eff
        assert got['total_gates'] == total
        assert got['effective_ratio'] == ratio
        assert got['reduction_ratio'] == 1.0 - ratio

    def test_equivalence_passes(self, arm, results, mnist_bits):
        """縮約前後のクラス別 group-sum が整数完全一致(不一致行 0)。"""
        eq = results[arm]['equivalence']
        assert eq['overall'] == 'pass', eq
        assert eq['random'] == 'pass'
        assert eq['random_n'] == 2000
        assert eq['random_mismatch_rows'] == 0
        if mnist_bits is None:
            assert eq['data'] == 'skip'
        else:
            assert eq['data'] == 'pass'
            assert eq['data_n'] == 2000
            assert eq['data_mismatch_rows'] == 0


def test_structural_unreferenced_separates_wiring(results):
    """P0 の主要な発見: rot 系(P6/P7)は構造未参照 0、rand 由来の P0 腕は > 0。

    2 指標を分けていないと測れない量なので、移植でも分離が保たれることを
    アンカーで確認する(値は p0_results.json の実測)。
    """
    assert results['P6_w8000_s0']['structurally_unreferenced'] == 0
    assert results['P7_w8000_s0']['structurally_unreferenced'] == 0
    assert results['P0_w8000_s0']['structurally_unreferenced'] == 7339
    assert results['P0_w8000_s0']['unreachable_after_fold'] == 6469
