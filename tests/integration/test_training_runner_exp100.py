"""統合: 先行実験で追加した層ごとの配線指定(per_layer)の配管。

- per_layer レシピから組んだモデルの各層の connections が生成器の出力と bitwise 一致
- kind='rot' / 'perm'(先行実験の腕)が先行実験の分岐追加後も bitwise 不変
- 先行実験 model.yaml 末尾の腕ごとの上書き(PR / RP / SR / RAND)が load_config で通り、
  層0 の被覆率が期待値になる(R 87.81% / P 100% / SR 100% / RAND 100% / RAND_U 82.4%)
(モデル級・統合テストは CONTRIBUTING の規約どおり tests/integration/ に置く)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.core.connections import (
    connection_stats,
    get_permutation_connections,
    get_random_connections,
    get_rot_connections,
    get_stretched_rot_connections,
    get_uniform_random_connections,
)
from griffith_logic.layers.lut import LutLayer
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import (
    _WIRING_SEED_OFFSET,
    ResolvedData,
    _build_model,
    _resolve_connections,
)

# 先行実験層0 の実寸(CIFAR-10 3値サーモメータ 9216 → 幅 8000)
IN0, OUT0 = 9216, 8000

# 先行実験 model.yaml 末尾の腕ごとの上書き(そのまま貼れる形)
R_OVERRIDES = [
    'model.connections.kind=rot',
    'model.connections.s_a=[3,0,0,0,0,0]',
    'model.connections.s_b=[96,4,20,90,400,1800]',
]
P_OVERRIDES = [*R_OVERRIDES, 'model.connections.kind=perm']
PR_OVERRIDES = [
    'model.connections={kind: per_layer, layers: ['
    '{kind: perm}, {kind: rot, s_a: 0, s_b: 4}, {kind: rot, s_a: 0, s_b: 20}, '
    '{kind: rot, s_a: 0, s_b: 90}, {kind: rot, s_a: 0, s_b: 400}, '
    '{kind: rot, s_a: 0, s_b: 1800}]}'
]
RP_OVERRIDES = [
    'model.connections={kind: per_layer, layers: ['
    '{kind: rot, s_a: 3, s_b: 96}, {kind: perm}, {kind: perm}, {kind: perm}, '
    '{kind: perm}, {kind: perm}]}'
]
SR_OVERRIDES = [
    'model.connections={kind: per_layer, layers: ['
    '{kind: stretched_rot, s_a: 3, s_b: 96}, {kind: rot, s_a: 0, s_b: 4}, '
    '{kind: rot, s_a: 0, s_b: 20}, {kind: rot, s_a: 0, s_b: 90}, '
    '{kind: rot, s_a: 0, s_b: 400}, {kind: rot, s_a: 0, s_b: 1800}]}'
]
RAND_OVERRIDES = ['model.connections.kind=random_seeded']
RAND_UNIFORM_OVERRIDES = ['model.connections.kind=random_uniform']


def _wiring_gen(seed=0):
    """runner._build_model と同じ配線用 generator(先行実験の別ストリーム)"""
    return torch.Generator().manual_seed(seed + _WIRING_SEED_OFFSET)


class TestPerLayerModelBuilding:
    """per_layer レシピで組んだモデルの各層の配線が生成器の出力と一致する"""

    @staticmethod
    def _data(in_width=16, out_width=2):
        return ResolvedData(
            name='synthetic', kind='classification', in_width=in_width,
            out_width=out_width, train_loader=None, eval_loader=None,
        )

    @staticmethod
    def _luts(model):
        return [m for m in model.modules() if isinstance(m, LutLayer)]

    def _build(self, *overrides, in_width=16, k=8, layers=3, seed=0):
        cfg = load_config(None, [
            'model.kind=dlgn', f'model.k={k}', f'model.layers={layers}',
            *overrides,
        ])
        return _build_model(cfg.model, self._data(in_width), 'cpu', seed=seed)

    def test_layer0_perm_hidden_rot(self):
        """PR 腕の縮小版: 層0 だけ perm、隠れ層は rot"""
        model = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: perm}, {kind: rot, s_a: 0, s_b: 3}, '
            '{kind: rot, s_a: 0, s_b: 5}]}'
        )
        gen = _wiring_gen()
        expected = [
            get_permutation_connections(16, 8, gen),
            get_rot_connections(8, 8, 0, 3),
            get_rot_connections(8, 8, 0, 5),
        ]
        for layer, exp in zip(self._luts(model), expected, strict=True):
            assert torch.equal(layer.indices, exp)

    def test_layer0_rot_hidden_perm(self):
        """RP 腕の縮小版: 層0 は rot(乱数を引かない)、隠れ層が generator を順に消費"""
        model = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: rot, s_a: 3, s_b: 5}, {kind: perm}, {kind: perm}]}'
        )
        gen = _wiring_gen()
        expected = [
            get_rot_connections(16, 8, 3, 5),
            get_permutation_connections(8, 8, gen),
            get_permutation_connections(8, 8, gen),
        ]
        for layer, exp in zip(self._luts(model), expected, strict=True):
            assert torch.equal(layer.indices, exp)

    def test_stretched_rot_layer0(self):
        """SR 腕の縮小版: 層0 が引き伸ばし rot(16 → 8)。

        被覆率は本番条件(9216 → 8000)では 100% だが、この縮小版は
        in/out が整数比(2)なので基準位置が偶数に限られる。
        s_a=3 / s_b=6(差が奇数)なら 16 本すべてを読む。
        """
        model = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: stretched_rot, s_a: 3, s_b: 6}, '
            '{kind: rot, s_a: 0, s_b: 3}, {kind: rot, s_a: 0, s_b: 5}]}'
        )
        luts = self._luts(model)
        assert torch.equal(
            luts[0].indices, get_stretched_rot_connections(16, 8, 3, 6))
        assert connection_stats(luts[0].indices, 16)['coverage'] == 1.0
        # 同じずらし量の rot は入力の先頭側 11 本しか読まない
        assert connection_stats(
            get_rot_connections(16, 8, 3, 6), 16)['coverage'] == 11 / 16

    def test_random_kinds_consume_generator_in_layer_order(self):
        model = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: random_seeded}, {kind: random_uniform}, {kind: perm}]}'
        )
        gen = _wiring_gen()
        expected = [
            get_random_connections(16, 8, 2, 'cpu', gen),
            get_uniform_random_connections(8, 8, 2, 'cpu', gen),
            get_permutation_connections(8, 8, gen),
        ]
        for layer, exp in zip(self._luts(model), expected, strict=True):
            assert torch.equal(layer.indices, exp)

    def test_all_rot_per_layer_matches_list_form(self):
        """per_layer で全層 rot にすると kind='rot'(リスト形式)と bitwise 一致"""
        a = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: rot, s_a: 3, s_b: 5}, {kind: rot, s_a: 0, s_b: 3}, '
            '{kind: rot, s_a: 0, s_b: 5}]}'
        )
        b = self._build(
            'model.connections.kind=rot',
            'model.connections.s_a=[3,0,0]',
            'model.connections.s_b=[5,3,5]',
        )
        for la, lb in zip(self._luts(a), self._luts(b), strict=True):
            assert torch.equal(la.indices, lb.indices)

    def test_all_perm_per_layer_matches_short_form(self):
        """per_layer で全層 perm にすると kind='perm' と bitwise 一致(消費順も同じ)"""
        a = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: perm}, {kind: perm}, {kind: perm}]}'
        )
        b = self._build('model.connections.kind=perm')
        for la, lb in zip(self._luts(a), self._luts(b), strict=True):
            assert torch.equal(la.indices, lb.indices)

    def test_layer0_wiring_shared_between_arms_with_same_layer0_kind(self):
        """層0 が同じ種なら腕をまたいで層0 の配線が一致する(generator の消費順)。

        P 腕(全層 perm)と PR 腕(層0 だけ perm)は層0 が bitwise 一致する。
        逆に、層0 が乱数を引かない RP 腕では隠れ層の perm が P 腕とずれる
        (層0 の消費分だけ generator の位置が違うため)。
        """
        p = self._build('model.connections.kind=perm')
        pr = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: perm}, {kind: rot, s_a: 0, s_b: 3}, '
            '{kind: rot, s_a: 0, s_b: 5}]}'
        )
        rp = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: rot, s_a: 3, s_b: 5}, {kind: perm}, {kind: perm}]}'
        )
        assert torch.equal(self._luts(p)[0].indices, self._luts(pr)[0].indices)
        assert not torch.equal(
            self._luts(p)[1].indices, self._luts(rp)[1].indices)

    def test_wiring_is_independent_of_init_mode(self):
        """先行実験と同じく配線用 generator は初期化と別ストリーム"""
        base = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: perm}, {kind: perm}, {kind: perm}]}',
            'model.init_mode=ri_randn')
        other = self._build(
            'model.connections={kind: per_layer, layers: ['
            '{kind: perm}, {kind: perm}, {kind: perm}]}',
            'model.init_mode=rri',
            'model.gate_mask_hidden=[0,15,3,5,10,12]')
        for a, b in zip(self._luts(base), self._luts(other), strict=True):
            assert torch.equal(a.indices, b.indices)

    def test_wiring_depends_on_seed(self):
        args = ('model.connections={kind: per_layer, layers: ['
                '{kind: perm}, {kind: rot, s_a: 0, s_b: 3}, '
                '{kind: rot, s_a: 0, s_b: 5}]}',)
        a = self._build(*args, seed=0)
        b = self._build(*args, seed=1)
        assert not torch.equal(
            self._luts(a)[0].indices, self._luts(b)[0].indices)
        # rot 層は seed に依らない
        assert torch.equal(
            self._luts(a)[1].indices, self._luts(b)[1].indices)

    def test_exp099_recipes_are_unchanged(self):
        """先行実験の腕(kind='rot' / 'perm')が先行実験の分岐追加後も bitwise 不変"""
        rot = self._build(
            'model.connections.kind=rot',
            'model.connections.s_a=[3,0,0]',
            'model.connections.s_b=[5,3,5]',
        )
        expected_rot = [
            get_rot_connections(16, 8, 3, 5),
            get_rot_connections(8, 8, 0, 3),
            get_rot_connections(8, 8, 0, 5),
        ]
        for layer, exp in zip(self._luts(rot), expected_rot, strict=True):
            assert torch.equal(layer.indices, exp)

        perm = self._build('model.connections.kind=perm')
        gen = _wiring_gen()
        expected_perm = [
            get_permutation_connections(16, 8, gen),
            get_permutation_connections(8, 8, gen),
            get_permutation_connections(8, 8, gen),
        ]
        for layer, exp in zip(self._luts(perm), expected_perm, strict=True):
            assert torch.equal(layer.indices, exp)

    def test_string_connections_still_bypass_generators(self):
        """'random' / 'unique' は従来どおり LutLayer 側の生成に素通しする"""
        torch.manual_seed(0)
        a = self._build()
        torch.manual_seed(0)
        b = self._build('model.connections=random')
        for la, lb in zip(self._luts(a), self._luts(b), strict=True):
            assert torch.equal(la.indices, lb.indices)


class TestArmOverridesAtRealDimensions:
    """先行実験 model.yaml の腕ごとの上書きと、層0(9216 → 8000)の被覆率"""

    @staticmethod
    def _layer0(overrides, seed=0):
        cfg = load_config(None, [
            'model.kind=dlgn', 'model.k=8000', 'model.layers=6',
            'data.name=cifar-10-3-thresholds', *overrides,
        ])
        idx = _resolve_connections(
            cfg.model, 0, IN0, OUT0, _wiring_gen(seed), 'cpu')
        return connection_stats(idx, IN0)

    def test_r_arm_layer0_coverage(self):
        """R(rot): 8093/9216 = 87.81%(先行実験で測った既知値)"""
        stats = self._layer0(R_OVERRIDES)
        assert stats['coverage'] == 8093 / IN0
        assert round(stats['coverage'] * 100, 2) == 87.81
        assert stats['usage_hist']['0'] == 1123
        assert stats['n_gates'] == OUT0

    def test_p_arm_layer0_coverage(self):
        """P(perm): 100%"""
        stats = self._layer0(P_OVERRIDES)
        assert stats['coverage'] == 1.0
        assert stats['usage_hist'] == {
            '0': 0, '1': 2432, '2': 6784, '3+': 0}

    def test_pr_arm_layer0_is_perm(self):
        assert self._layer0(PR_OVERRIDES)['coverage'] == 1.0

    def test_rp_arm_layer0_is_rot(self):
        assert self._layer0(RP_OVERRIDES)['coverage'] == 8093 / IN0

    def test_sr_arm_layer0_coverage_is_full(self):
        """SR(引き伸ばし rot): 100%(実測。plan の見積もりは「約 98%」だった)"""
        stats = self._layer0(SR_OVERRIDES)
        assert stats['coverage'] == 1.0
        assert stats['usage_hist'] == {
            '0': 0, '1': 2432, '2': 6784, '3+': 0}
        assert stats['same_input'] == 0

    def test_rand_arm_layer0_coverage_is_full_not_82(self):
        """RAND(random_seeded): **100%**(実測)。

        既存の 'random' 生成器は使用回数を均衡させるので、
        層0 では perm と同じ使用回数分布になる。plan.md が RAND 腕に
        期待した「一様乱択・被覆 約 82%・未参照 約 15%」は
        random_uniform でなければ得られない(下のテスト)。
        """
        stats = self._layer0(RAND_OVERRIDES)
        assert stats['coverage'] == 1.0
        assert stats['usage_hist'] == {
            '0': 0, '1': 2432, '2': 6784, '3+': 0}

    def test_random_uniform_layer0_coverage_is_82_percent(self):
        """random_uniform: 1 - (1-1/9216)^16000 = 82.44%(±1pt)"""
        stats = self._layer0(RAND_UNIFORM_OVERRIDES)
        expected = 1.0 - (1.0 - 1.0 / IN0) ** (2 * OUT0)
        assert abs(stats['coverage'] - expected) < 0.01
        assert stats['usage_hist']['0'] > 1000
        assert stats['usage_hist']['3+'] > 1000

    def test_all_arms_load_and_cover_six_layers(self):
        """6 層すべてが解決でき、隠れ層(8000→8000)も仕様どおり"""
        for overrides in (R_OVERRIDES, P_OVERRIDES, PR_OVERRIDES,
                          RP_OVERRIDES, SR_OVERRIDES, RAND_OVERRIDES):
            cfg = load_config(None, [
                'model.kind=dlgn', 'model.k=8000', 'model.layers=6',
                *overrides,
            ])
            gen = _wiring_gen()
            for i in range(6):
                in_dim = IN0 if i == 0 else OUT0
                idx = _resolve_connections(
                    cfg.model, i, in_dim, OUT0, gen, 'cpu')
                assert idx.shape == (2, OUT0)
                assert int(idx.min()) >= 0 and int(idx.max()) < in_dim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
