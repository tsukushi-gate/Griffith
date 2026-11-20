"""SAT容量判定器(analysis/sat_capacity)のテスト。

正しさの機械的保証:
    1. 極小トポロジーで全ゲート割当(16^M)ブルートフォースとSAT/UNSATが完全一致
    2. 既知境界の再現(xor2=1LUT SAT、parity3=depth1 UNSAT/depth3 SAT等)
    3. witness 3-way一致(torch hard eval == IR simulate == target)
    4. don't-care単調性(制約を減らしてSAT→UNSATにはならない)
    5. fixed-UNSATかつfree-SATの実例(配線税の存在証明)
"""

import itertools

import numpy as np
import pytest
import torch

pytest.importorskip('pysat')

from griffith_logic.analysis.sat_capacity import (  # noqa: E402
    LutTopology,
    TargetSpec,
    can_realize,
    min_width_search,
    wiring_tax,
    witness_to_model,
)

XOR3 = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2])
MAJ3 = TargetSpec(fn=lambda b: int(sum(b) >= 2))


def _brute_force_bit(n_in: int, widths: list[int],
                     wiring: list[np.ndarray], labels: np.ndarray) -> bool:
    """bit読み出し・固定配線の全ゲート割当列挙(整数TTで直接シミュレート)。"""
    patterns = list(itertools.product([0, 1], repeat=n_in))
    n_neurons = sum(widths)
    for assign in itertools.product(range(16), repeat=n_neurons):
        pos = 0
        ok = True
        for p, row in enumerate(patterns):
            vals = list(row)
            pos = 0
            for li, w in enumerate(widths):
                nxt = []
                for o in range(w):
                    tt = assign[pos + o]
                    a = vals[wiring[li][0][o]]
                    b = vals[wiring[li][1][o]]
                    nxt.append((tt >> (a + 2 * b)) & 1)
                vals = nxt
                pos += w
            if vals[0] != labels[p]:
                ok = False
                break
        if ok:
            return True
    return False


class TestBruteForceCrossCheck:
    """①極小トポロジーでSAT/UNSAT判定がブルートフォースと完全一致。"""

    @pytest.mark.parametrize('target_name,fn', [
        ('xor2', lambda b: b[0] ^ b[1]),
        ('and2', lambda b: b[0] & b[1]),
        ('xor3', lambda b: b[0] ^ b[1] ^ b[2]),
        ('maj3', lambda b: int(sum(b) >= 2)),
    ])
    @pytest.mark.parametrize('widths', [[1], [2, 1]])
    def test_matches_brute_force(self, target_name, fn, widths):
        n_in = 3 if target_name in ('xor3', 'maj3') else 2
        target = TargetSpec(fn=fn)
        patterns, labels = target.patterns_labels(n_in)
        topo = LutTopology(n_in=n_in, layer_widths=widths, readout='bit')
        for seed in range(4):
            r = can_realize(topo, target, wiring='fixed_random', seed=seed)
            # 同一配線をブルートフォースへ
            from griffith_logic.analysis.sat_capacity import _fixed_wiring
            wiring = _fixed_wiring(topo, seed)
            bf = _brute_force_bit(n_in, widths, wiring, labels)
            assert (r.status == 'sat') == bf, (
                f'{target_name} W={widths} seed={seed}: SAT={r.status} BF={bf}'
            )


class TestKnownBoundaries:
    """②既知の表現境界の再現。"""

    def test_xor2_single_lut_sat(self):
        topo = LutTopology(n_in=2, layer_widths=[1], readout='bit')
        r = can_realize(topo, TargetSpec(fn=lambda b: b[0] ^ b[1]),
                        wiring='free')
        assert r.status == 'sat'
        assert r.truth_tables[0][0] == 0b0110  # XOR真理値表

    def test_parity3_depth1_unsat_any_wiring(self):
        """parity3は単層(bit読み出し)では自由配線でも不可能。

        1個の2入力LUTは3変数に依存できない(N=2の構造的な壁)。
        """
        topo = LutTopology(n_in=3, layer_widths=[1], readout='bit')
        r = can_realize(topo, XOR3, wiring='free')
        assert r.status == 'unsat'

    def test_parity3_depth1_groupsum_unsat(self):
        """単層+GroupSum(2)でもparity3は不可能(事前の実測: chance張り付き)。"""
        for w in (4, 8):
            topo = LutTopology(n_in=3, layer_widths=[w], readout='groupsum')
            r = can_realize(topo, XOR3, wiring='free')
            assert r.status == 'unsat', f'W={w}'

    def test_parity3_depth2_free_sat(self):
        """xor(a, xor(b, c))は2段で組めるので自由配線depth2はSAT。"""
        topo = LutTopology(n_in=3, layer_widths=[2, 1], readout='bit')
        r = can_realize(topo, XOR3, wiring='free')
        assert r.status == 'sat'

    def test_maj3_depth1_groupsum_wide_sat(self):
        """majorityは幅広単層+GroupSumで表現可能(事前の実測: 100%)。"""
        topo = LutTopology(n_in=3, layer_widths=[16], readout='groupsum')
        r = can_realize(topo, MAJ3, wiring='free')
        assert r.status == 'sat'


class TestWitnessRoundTrip:
    """③witness 3-way一致(hard eval == IR simulate == target)。

    can_realize(verify=True)内でIR simulate == targetは検証済みなので、
    ここではtorch hard evalとの一致を追加で確認する。
    """

    @pytest.mark.parametrize('target,name', [(XOR3, 'xor3'), (MAJ3, 'maj3')])
    def test_torch_hard_eval_matches(self, target, name):
        topo = LutTopology(n_in=3, layer_widths=[8, 8], readout='groupsum')
        r = can_realize(topo, target, wiring='free')
        assert r.status == 'sat'
        assert r.stats['verified']

        model = witness_to_model(r, topo)
        model.eval()
        patterns, labels = target.patterns_labels(3)
        x = torch.tensor(patterns, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).argmax(-1).numpy()
        assert (pred == labels).all(), f'{name}: torch hard evalが標的と不一致'


class TestDontCareMonotonicity:
    """④制約の緩和(don't care追加)はSAT→UNSATに変わらない。"""

    def test_partial_never_harder(self):
        topo = LutTopology(n_in=3, layer_widths=[4], readout='groupsum')
        full = can_realize(topo, MAJ3, wiring='fixed_random', seed=0)
        patterns, labels = MAJ3.patterns_labels(3)
        # 半分のパターンだけ制約
        partial = TargetSpec(dataset=(patterns[:4], labels[:4]))
        r = can_realize(topo, partial, wiring='fixed_random', seed=0)
        if full.status == 'sat':
            assert r.status == 'sat'

    def test_dc_flips_unsat_to_sat(self):
        """全制約ではUNSATだがdon't careで緩めるとSATになる実例。

        parity3は単層で表現不能だが、4点だけの部分関数なら1 LUTでも
        (関連2変数のみで)フィット可能なことがある。
        """
        topo = LutTopology(n_in=3, layer_widths=[1], readout='bit')
        assert can_realize(topo, XOR3, wiring='free').status == 'unsat'
        patterns, labels = XOR3.patterns_labels(3)
        # b[2]=0 の4点のみ → xor2に退化 → 1 LUTで可
        keep = patterns[:, 2] == 0
        partial = TargetSpec(dataset=(patterns[keep], labels[keep]))
        r = can_realize(topo, partial, wiring='free')
        assert r.status == 'sat'


class TestTotalizerAuxCollision:
    """回帰: totalizer補助変数の衝突(IDPool.occupyがtopを更新しない)。

    クロスマップ実験で「配線UNSATなのに学習がhard 100%到達」という
    論理的に不可能な結果として顕在化したバグ。ブルートフォースで
    解(0,1,12,12)が存在すると確認済みの固定配線インスタンス。
    """

    def test_maj3_degenerate_wiring_is_sat(self):
        wiring = [np.array([[1, 0, 1, 0], [2, 0, 2, 1]])]
        topo = LutTopology(n_in=3, layer_widths=[4], readout='groupsum')
        r = can_realize(topo, MAJ3, wiring=wiring)
        assert r.status == 'sat', (
            'totalizer補助変数衝突の偽UNSATが再発している'
        )

    def test_groupsum_bruteforce_cross_check(self):
        """groupsum読み出しでもブルートフォースとSAT/UNSAT一致。"""
        for seed in range(6):
            topo = LutTopology(n_in=3, layer_widths=[4], readout='groupsum')
            from griffith_logic.analysis.sat_capacity import _fixed_wiring
            wiring = _fixed_wiring(topo, seed)
            r = can_realize(topo, MAJ3, wiring=wiring)
            patterns, labels = MAJ3.patterns_labels(3)
            found = False
            for assign in itertools.product(range(16), repeat=4):
                ok = True
                for row, lab in zip(patterns.tolist(), labels.tolist()):
                    outs = [(assign[o] >> (row[wiring[0][0][o]]
                                           + 2 * row[wiring[0][1][o]])) & 1
                            for o in range(4)]
                    pred = 0 if sum(outs[:2]) >= sum(outs[2:]) else 1
                    if pred != lab:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            assert (r.status == 'sat') == found, (seed, r.status, found)


class TestKClassReadout:
    """k>2 GroupSum読み出しのブルートフォース照合+witness検証。"""

    @staticmethod
    def _brute_force_kclass(wiring, width, k, patterns, labels):
        gsz = width // k
        for assign in itertools.product(range(16), repeat=width):
            ok = True
            for row, lab in zip(patterns.tolist(), labels.tolist()):
                outs = [(assign[o] >> (row[wiring[0][0][o]]
                                       + 2 * row[wiring[0][1][o]])) & 1
                        for o in range(width)]
                sums = [sum(outs[i * gsz:(i + 1) * gsz]) for i in range(k)]
                pred = max(range(k), key=lambda c: (sums[c], -c))
                if pred != lab:
                    ok = False
                    break
            if ok:
                return True
        return False

    @pytest.mark.parametrize('seed', list(range(6)))
    def test_3class_bruteforce_cross_check_fast(self, seed):
        """3クラス(W=3, グループ幅1)でSAT/UNSATがブルートフォース一致。"""
        fn = lambda b: 0 if sum(b) <= 1 else (1 if sum(b) == 2 else 2)  # noqa: E731,E501
        target = TargetSpec(fn=fn)
        topo = LutTopology(n_in=3, layer_widths=[3], readout='groupsum',
                           groupsum_k=3)
        from griffith_logic.analysis.sat_capacity import _fixed_wiring
        wiring = _fixed_wiring(topo, seed)
        r = can_realize(topo, target, wiring=wiring)
        patterns, labels = target.patterns_labels(3)
        bf = self._brute_force_kclass(wiring, 3, 3, patterns, labels)
        assert (r.status == 'sat') == bf, (seed, r.status, bf)

    @pytest.mark.slow
    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_3class_bruteforce_cross_check(self, seed):
        """3クラス(W=6)でSAT/UNSATがブルートフォース一致(重い網羅版)。"""
        fn = lambda b: 0 if sum(b) <= 1 else (1 if sum(b) == 2 else 2)  # noqa: E731,E501
        target = TargetSpec(fn=fn)
        topo = LutTopology(n_in=3, layer_widths=[6], readout='groupsum',
                           groupsum_k=3)
        from griffith_logic.analysis.sat_capacity import _fixed_wiring
        wiring = _fixed_wiring(topo, seed)
        r = can_realize(topo, target, wiring=wiring)
        patterns, labels = target.patterns_labels(3)
        bf = self._brute_force_kclass(wiring, 6, 3, patterns, labels)
        assert (r.status == 'sat') == bf, (seed, r.status, bf)

    def test_4class_witness_torch_match(self):
        """4クラス: witnessのtorch hard evalが標的一致(argmax同点規約込み)。"""
        fn = lambda b: (b[0] + 2 * b[1])  # noqa: E731 2bitデコーダ
        target = TargetSpec(fn=fn)
        topo = LutTopology(n_in=4, layer_widths=[8, 8], readout='groupsum',
                           groupsum_k=4)
        r = can_realize(topo, target, wiring='free')
        assert r.status == 'sat'
        model = witness_to_model(r, topo)
        model.eval()
        patterns, labels = target.patterns_labels(4)
        x = torch.tensor(patterns, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).argmax(-1).numpy()
        assert (pred == labels).all()


class TestFindSatWiring:
    def test_guard_finds_sat_wiring_on_taxed_cell(self):
        """配線税セル(parity3-in-6)でSAT配線を発見し、UNSAT配線を棄却する。"""
        from griffith_logic.analysis.sat_capacity import find_sat_wiring
        fn = lambda b: b[0] ^ b[1] ^ b[2]  # noqa: E731
        topo = LutTopology(n_in=6, layer_widths=[8, 8], readout='groupsum')
        res = find_sat_wiring(topo, TargetSpec(fn=fn), max_tries=32)
        assert res['wiring'] is not None, res['statuses']
        # 見つかった配線が本当にSATであることを再確認
        r = can_realize(topo, TargetSpec(fn=fn), wiring=res['wiring'])
        assert r.status == 'sat'


class TestWiringTax:
    """⑤fixed-UNSATかつfree-SATの実例=配線税の存在。"""

    def test_parity_in_noise_wiring_tax(self):
        """parity3-in-6bits: 自由配線ならSATだが、ランダム配線は
        関連3変数が揃わずUNSATになるseedが存在する。
        """
        fn = lambda b: b[0] ^ b[1] ^ b[2]  # noqa: E731 (残り3bitはノイズ)
        target = TargetSpec(fn=fn)
        tax = wiring_tax(6, 2, 4, target, n_wiring_seeds=8)
        assert tax['free_sat'] is True
        assert tax['fixed_sat_frac'] < 1.0, (
            f'配線税が観測されるはず: {tax}'
        )

    def test_min_width_search(self):
        r = min_width_search(3, 2, XOR3, widths=(2, 4, 8), wiring='free',
                             readout='groupsum')
        assert r['min_width'] in (2, 4), r


def _brute_force_bit_arity(n_in: int, widths: list[int], arity: int,
                           wiring: list[np.ndarray],
                           labels: np.ndarray) -> bool:
    """任意arity・bit読み出し・固定配線の全ゲート割当列挙。

    2^(2^arity)^ニューロン数 の列挙なので単一ニューロン級の
    極小トポロジー専用(arity=4で65536通り/ニューロン)。
    """
    patterns = list(itertools.product([0, 1], repeat=n_in))
    n_neurons = sum(widths)
    n_tts = 1 << (1 << arity)
    for assign in itertools.product(range(n_tts), repeat=n_neurons):
        ok = True
        for p, row in enumerate(patterns):
            vals = list(row)
            pos = 0
            for li, w in enumerate(widths):
                nxt = []
                for o in range(w):
                    tt = assign[pos + o]
                    addr = sum(vals[wiring[li][j][o]] << j
                               for j in range(arity))
                    nxt.append((tt >> addr) & 1)
                vals = nxt
                pos += w
            if vals[0] != labels[p]:
                ok = False
                break
        if ok:
            return True
    return False


class TestArityGeneralization:
    """arity∈{4,6}一般化の正しさ(fan-in掃引の前提)。"""

    @pytest.mark.parametrize('target_name,fn', [
        ('xor4', lambda b: b[0] ^ b[1] ^ b[2] ^ b[3]),
        ('maj4', lambda b: int(sum(b) >= 3)),
        ('and4', lambda b: b[0] & b[1] & b[2] & b[3]),
    ])
    def test_arity4_matches_brute_force(self, target_name, fn):
        """①固定配線・単一LUT4でSAT/UNSATがブルートフォースと完全一致。

        偽UNSAT(対称性破り・エンコードバグ)の機械的検出。ランダム配線は
        ピン重複で標的の関連変数を落とすことがあり、SAT/UNSAT両方を踏む。
        """
        from griffith_logic.analysis.sat_capacity import _fixed_wiring
        n_in = 4
        target = TargetSpec(fn=fn)
        patterns, labels = target.patterns_labels(n_in)
        topo = LutTopology(n_in=n_in, layer_widths=[1], readout='bit',
                           lut_arity=4)
        for seed in range(4):
            r = can_realize(topo, target, wiring='fixed_random', seed=seed)
            wiring = _fixed_wiring(topo, seed)
            bf = _brute_force_bit_arity(n_in, [1], 4, wiring, labels)
            assert (r.status == 'sat') == bf, (
                f'{target_name} seed={seed}: SAT={r.status} BF={bf}'
            )

    def test_parity_junta_boundaries(self):
        """②既知境界: k-juntaはarity≥kの単一LUTでのみ実現可能。

        parity5は4-junta不可(単一LUT4 UNSAT)、単一LUT6ではSAT。
        parity3は単一LUT2不可(既存テスト)だが単一LUT4でSAT。
        """
        p5 = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4])
        topo4 = LutTopology(n_in=5, layer_widths=[1], readout='bit',
                            lut_arity=4)
        assert can_realize(topo4, p5, wiring='free').status == 'unsat'
        topo6 = LutTopology(n_in=5, layer_widths=[1], readout='bit',
                            lut_arity=6)
        assert can_realize(topo6, p5, wiring='free').status == 'sat'

        p3 = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2])
        topo4b = LutTopology(n_in=3, layer_widths=[1], readout='bit',
                             lut_arity=4)
        assert can_realize(topo4b, p3, wiring='free').status == 'sat'

    def test_arity6_witness_roundtrip(self):
        """③witness復元(independent)のtorch hard evalが標的一致。"""
        fn = lambda b: b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5]  # noqa: E731
        target = TargetSpec(fn=fn)
        topo = LutTopology(n_in=6, layer_widths=[1], readout='bit',
                           lut_arity=6)
        r = can_realize(topo, target, wiring='free')
        assert r.status == 'sat'
        assert r.stats['verified']

        model = witness_to_model(r, topo)
        model.eval()
        patterns, labels = target.patterns_labels(6)
        x = torch.tensor(patterns, dtype=torch.float32)
        with torch.no_grad():
            pred = (model(x)[:, 0] > 0.5).long().numpy()
        assert (pred == labels).all()

    def test_arity4_groupsum_witness_roundtrip(self):
        """③' groupsum読み出し+深さ2でもwitness往復が一致(arity=4)。"""
        target = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2] ^ b[3])
        topo = LutTopology(n_in=4, layer_widths=[4, 4], readout='groupsum',
                           lut_arity=4)
        r = can_realize(topo, target, wiring='free')
        assert r.status == 'sat'
        model = witness_to_model(r, topo)
        model.eval()
        patterns, labels = target.patterns_labels(4)
        x = torch.tensor(patterns, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).argmax(-1).numpy()
        assert (pred == labels).all()

    def test_required_width_shrinks_with_arity(self):
        """④frontierの単調性: xor4はarity=2 depth1でUNSAT、arity=4でSAT。

        fan-in増が必要幅/深さを縮める(掃引の理論骨子)の最小実証。
        """
        target = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2] ^ b[3])
        r2 = min_width_search(4, 1, target, widths=(2, 4, 8), wiring='free',
                              readout='groupsum', lut_arity=2)
        assert r2['min_width'] is None, r2
        r4 = min_width_search(4, 1, target, widths=(2, 4, 8), wiring='free',
                              readout='groupsum', lut_arity=4)
        assert r4['min_width'] is not None, r4

    def test_arity4_depth2_free_neuron_symmetry_sound(self):
        """⑤多層自由配線(ニューロン対称性破り)がSATを潰さない。

        parity5-in-5は[2,1]のLUT4×3で構成可能(xor4 + パススルー → 結合)。
        """
        p5 = TargetSpec(fn=lambda b: b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4])
        topo = LutTopology(n_in=5, layer_widths=[2, 1], readout='bit',
                           lut_arity=4)
        r = can_realize(topo, p5, wiring='free')
        assert r.status == 'sat'
        assert r.stats['verified']
