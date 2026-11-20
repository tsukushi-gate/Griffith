"""前処理記述子(Pythonミラー)の検証。

固定する契約:
    - 記述子実行 ⇔ 参照エンコーダ(griffith_logic/encoding/image_features.py を
      **float64で**評価したもの)が全ビット一致する
    - 整数化した比較(D >= cut / D <= cut)が、実数比較 d > t / d < -t と
      u8の全定義域で一致する
    - thermometer の256ビットLUT と THERMO_STEP 畳み込みが等価
    - チャネル順・チャネル数が image_features の契約と一致する

参照を float64 で取るのは、Web配信経路(convpreproc.ts)が f64 で計算しており、
記述子はその厳密整数化だから。学習時のtorch float32 とは境界でごく稀に
1ビットずれる(下の TestFloat32Divergence がその差を明示的に測る)。
"""

import numpy as np
import pytest
import torch

from griffith_logic.deploy.preproc_desc import (
    DIR_GE,
    DIR_LE,
    PREPROC_DESC_STRIDE,
    PREPROC_OP_THERMO,
    PREPROC_OP_THERMO_STEP,
    SCALE,
    build_preproc_descriptors,
    cut_greater,
    cut_less,
    cut_of,
    encode_bits,
    quant_table,
    run_preproc_descriptors,
)
from griffith_logic.encoding.image_features import (
    CHROMA_PAIRS,
    CURV_OFFSETS,
    EDGE_OFFSETS,
    encode_image_features,
    n_output_channels,
)

EDGE_THR = (0.1, 0.3)
CURV_THR = (0.2,)
CHROMA_THR = (0.15, 0.4)
N_THR = 3


def lowlevel_preproc(c: int = 3, h: int = 9, w: int = 11) -> dict:
    return {
        'kind': 'image_lowlevel', 'mean': [0.0] * c, 'std': [1.0] * c,
        'height': h, 'width': w,
        'thresholds': [[(i + 1) / (N_THR + 1) for i in range(N_THR)]
                       for _ in range(c)],
        'edge_thresholds': list(EDGE_THR),
        'curv_thresholds': list(CURV_THR),
        'chroma_thresholds': list(CHROMA_THR),
        'edge_offsets': [list(o) for o in EDGE_OFFSETS],
        'curv_offsets': [list(o) for o in CURV_OFFSETS],
        'chroma_pairs': [list(p) for p in CHROMA_PAIRS],
    }


def thermo_preproc(c: int = 3, h: int = 7, w: int = 5, t: int = 3) -> dict:
    rng = np.random.default_rng(2)
    return {
        'kind': 'image_thermometer', 'mean': list(rng.normal(0.5, 0.1, c)),
        'std': list(rng.uniform(0.2, 0.5, c)),
        'height': h, 'width': w,
        'thresholds': [sorted(rng.normal(0, 1, t)) for _ in range(c)],
    }


def reference_bits_f64(images_u8: np.ndarray, preproc: dict) -> np.ndarray:
    """image_features を float64 で評価した参照(配信経路と同じ数値規約)。"""
    x = (images_u8.astype(np.float32) / np.float32(255)).astype(np.float64)
    if preproc['kind'] == 'image_thermometer':
        mean = np.asarray(preproc['mean'])[None, :, None, None]
        std = np.asarray(preproc['std'])[None, :, None, None]
        thr = np.asarray(preproc['thresholds'])
        z = (x - mean) * (1.0 / std)
        return (z[:, :, None] > thr[None, :, :, None, None]).astype(np.uint8) \
            .reshape(x.shape[0], -1, x.shape[2], x.shape[3])
    return encode_image_features(
        torch.from_numpy(x), n_thresholds=N_THR,
        edge_thresholds=EDGE_THR, curv_thresholds=CURV_THR,
        chroma_thresholds=CHROMA_THR,
    ).numpy().astype(np.uint8)


def random_images(shape, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


class TestQuantTable:
    def test_values_are_exact_integers(self):
        q = quant_table()
        assert len(q) == 256
        assert q[0] == 0
        assert q[255] == SCALE
        for k in range(256):
            assert q[k] == float(np.float32(k / 255)) * SCALE
            assert 0 <= q[k] <= 0xffffffff

    def test_monotonic(self):
        q = quant_table()
        assert all(q[k] < q[k + 1] for k in range(255))


class TestIntegerComparison:
    """d > t ⟺ D >= cut、d < -t ⟺ D <= cut を u8 全定義域で確認する。"""

    @pytest.mark.parametrize('t', [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.75])
    def test_pairwise_difference_all_256_squared(self, t):
        q = np.asarray(quant_table(), dtype=np.int64)
        xf = np.asarray([float(np.float32(k / 255)) for k in range(256)],
                        dtype=np.float64)
        a, b = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        d_real = xf[a] - xf[b]
        d_int = q[a] - q[b]
        assert np.array_equal(d_int >= cut_greater(t), d_real > t)
        assert np.array_equal(d_int <= cut_less(t), d_real < -t)

    @pytest.mark.parametrize('t', [0.2, 0.4])
    def test_curvature_triples_sampled(self, t):
        """curv は D の値域が34bit。ランダム三つ組+端値で厳密一致を見る。"""
        q = np.asarray(quant_table(), dtype=np.int64)
        xf = np.asarray([float(np.float32(k / 255)) for k in range(256)],
                        dtype=np.float64)
        rng = np.random.default_rng(7)
        ks = rng.integers(0, 256, size=(3, 200_000))
        edge = np.array([[0, 0, 0, 255, 255, 255, 0, 255],
                         [0, 255, 0, 0, 255, 0, 255, 255],
                         [0, 0, 255, 255, 0, 255, 255, 0]])
        ks = np.concatenate([ks, edge], axis=1)
        a, m, c = ks
        d_real = xf[a] - 2.0 * xf[m] + xf[c]
        d_int = q[a] - 2 * q[m] + q[c]
        assert np.array_equal(d_int >= cut_greater(t), d_real > t)
        assert np.array_equal(d_int <= cut_less(t), d_real < -t)

    @pytest.mark.parametrize('t', [0.0, 0.2, 0.4])
    def test_curvature_triples_exhaustive(self, t):
        """全 256^3 三つ組。メモリのためaで分割して回す(実測0.03s/閾値)。"""
        q = np.asarray(quant_table(), dtype=np.int64)
        xf = np.asarray([float(np.float32(k / 255)) for k in range(256)],
                        dtype=np.float64)
        m, c = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        cg, cl = cut_greater(t), cut_less(t)
        base_int = -2 * q[m] + q[c]
        base_real = -2.0 * xf[m] + xf[c]
        for a in range(256):
            d_int = q[a] + base_int
            d_real = xf[a] + base_real
            assert np.array_equal(d_int >= cg, d_real > t), f'a={a}'
            assert np.array_equal(d_int <= cl, d_real < -t), f'a={a}'

    def test_cut_words_roundtrip(self):
        """cutの64bit二の補数2語が元の整数へ厳密に戻る(C側int64と同じ扱い)。"""
        plan = build_preproc_descriptors(lowlevel_preproc())
        desc = plan['desc']
        for ch in range(plan['channels']):
            base = ch * PREPROC_DESC_STRIDE
            if desc[base] in (PREPROC_OP_THERMO, PREPROC_OP_THERMO_STEP):
                continue
            cut = cut_of(desc, base)
            assert abs(cut) <= 2 ** 33, f'cut={cut} が想定の値域外'


class TestThermometer:
    def test_lut_matches_formula_for_all_256(self):
        p = thermo_preproc()
        plan = build_preproc_descriptors(p, fold_thermo_step=False)
        desc = plan['desc']
        ch = 0
        for c in range(len(p['mean'])):
            for k_idx in range(len(p['thresholds'][c])):
                base = ch * PREPROC_DESC_STRIDE
                assert desc[base] == PREPROC_OP_THERMO
                th = p['thresholds'][c][k_idx]
                inv = 1.0 / p['std'][c]
                for k in range(256):
                    bit = (desc[base + 8 + (k >> 5)] >> (k & 31)) & 1
                    want = (float(np.float32(k / 255)) - p['mean'][c]) * inv > th
                    assert bit == int(want), f'c={c} th={th} k={k}'
                ch += 1

    def test_step_folding_is_equivalent(self):
        """THERMO_STEP に畳んだ結果とLUT参照のままの結果が全ビット一致する。"""
        p = thermo_preproc()
        img = random_images((4, 3, p['height'], p['width']), seed=1)
        folded = run_preproc_descriptors(build_preproc_descriptors(p), img)
        raw = run_preproc_descriptors(
            build_preproc_descriptors(p, fold_thermo_step=False), img)
        assert np.array_equal(folded, raw)

    def test_monotonic_thermometer_always_folds(self):
        """std>0 の thermometer は単調なので必ず1閾値へ畳める。"""
        plan = build_preproc_descriptors(thermo_preproc())
        assert plan['counts']['thermo_lut'] == 0
        assert plan['counts']['thermo_step'] == plan['channels']

    def test_saturating_thresholds_fold_to_constants(self):
        """全画素が閾値を超える/超えない場合も階段関数として畳める(k0=0/256)。"""
        p = thermo_preproc(c=1, t=2)
        p['mean'] = [0.0]
        p['std'] = [1.0]
        p['thresholds'] = [[-5.0, 5.0]]   # 常に真 / 常に偽
        plan = build_preproc_descriptors(p)
        assert plan['counts']['thermo_step'] == 2
        img = random_images((2, 1, p['height'], p['width']), seed=3)
        bits = run_preproc_descriptors(plan, img)
        assert bits[:, 0].all()
        assert not bits[:, 1].any()


class TestAgainstReferenceEncoder:
    def test_lowlevel_matches_float64_reference(self):
        p = lowlevel_preproc()
        img = random_images((4, 3, p['height'], p['width']), seed=5)
        plan = build_preproc_descriptors(p)
        assert np.array_equal(run_preproc_descriptors(plan, img),
                              reference_bits_f64(img, p))

    def test_thermometer_matches_float64_reference(self):
        p = thermo_preproc()
        img = random_images((4, 3, p['height'], p['width']), seed=6)
        plan = build_preproc_descriptors(p)
        assert np.array_equal(run_preproc_descriptors(plan, img),
                              reference_bits_f64(img, p))

    def test_extreme_pixels(self):
        """端値(0/255)飽和入力でも一致する。"""
        p = lowlevel_preproc()
        h, w = p['height'], p['width']
        img = np.zeros((3, 3, h, w), dtype=np.uint8)
        img[1] = 255
        img[2, :, :, ::2] = 255
        plan = build_preproc_descriptors(p)
        assert np.array_equal(run_preproc_descriptors(plan, img),
                              reference_bits_f64(img, p))

    def test_channel_count_and_order(self):
        p = lowlevel_preproc()
        plan = build_preproc_descriptors(p)
        assert plan['channels'] == n_output_channels(
            3, n_thresholds=N_THR, edge_thresholds=EDGE_THR,
            curv_thresholds=CURV_THR, chroma_thresholds=CHROMA_THR)

    def test_encode_bits_flattens_in_ir_input_order(self):
        p = lowlevel_preproc()
        img = random_images((2, 3, p['height'], p['width']), seed=8)
        plan = build_preproc_descriptors(p)
        flat = encode_bits(plan, img)
        bits = run_preproc_descriptors(plan, img)
        assert flat.shape == (2, plan['channels'] * p['height'] * p['width'])
        assert np.array_equal(flat, bits.reshape(2, -1))


class TestFloat32Divergence:
    """学習時エンコーダ(torch float32)との差を測って明示的に記録する。

    配信経路(convpreproc.ts)も記述子も **f64相当** で計算するため、
    torch float32 で学習時に作ったビットとは境界でごく稀に食い違う。
    これは今回の実装で生じた差ではなく、Web経路にも元からある差である。
    """

    def test_divergence_is_rare_and_only_from_f32_rounding(self):
        p = lowlevel_preproc()
        img = random_images((8, 3, p['height'], p['width']), seed=9)
        got = run_preproc_descriptors(build_preproc_descriptors(p), img)
        want64 = reference_bits_f64(img, p)
        assert np.array_equal(got, want64), '記述子はf64参照と厳密一致すること'

        x32 = img.astype(np.float32) / np.float32(255)
        want32 = encode_image_features(
            torch.from_numpy(x32), n_thresholds=N_THR,
            edge_thresholds=EDGE_THR, curv_thresholds=CURV_THR,
            chroma_thresholds=CHROMA_THR,
        ).numpy().astype(np.uint8)
        diff = int((want32 != want64).sum())
        # 実測: 26433ビット中3件(0.01%)。桁が変わったら数値経路が変わった合図
        assert diff / want64.size < 1e-3, f'f32との差が想定より大きい({diff}件)'


class TestErrors:
    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match='未対応'):
            build_preproc_descriptors({'kind': 'dense_thermometer'})

    def test_rejects_nonpositive_std(self):
        p = thermo_preproc()
        p['std'][0] = 0.0
        with pytest.raises(ValueError, match='std'):
            build_preproc_descriptors(p)

    def test_rejects_shape_mismatch(self):
        p = thermo_preproc()
        plan = build_preproc_descriptors(p)
        with pytest.raises(ValueError, match='画像形状'):
            run_preproc_descriptors(plan, random_images((1, 3, 4, 4)))

    def test_rejects_non_u8_input(self):
        p = thermo_preproc()
        plan = build_preproc_descriptors(p)
        bad = np.zeros((1, 3, p['height'], p['width']), dtype=np.float32)
        with pytest.raises(ValueError, match='uint8'):
            run_preproc_descriptors(plan, bad)


class TestDirections:
    def test_dir_constants_appear_in_pairs(self):
        """符号付き検出器は (>t, <-t) の対で並ぶ(image_features の規約)。"""
        plan = build_preproc_descriptors(lowlevel_preproc())
        desc = plan['desc']
        dirs = [desc[ch * PREPROC_DESC_STRIDE + 7]
                for ch in range(plan['channels'])
                if desc[ch * PREPROC_DESC_STRIDE] not in
                (PREPROC_OP_THERMO, PREPROC_OP_THERMO_STEP)]
        assert set(dirs) <= {DIR_GE, DIR_LE}
        assert dirs.count(DIR_GE) > dirs.count(DIR_LE)  # 彩度はGEのみ
