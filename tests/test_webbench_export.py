"""scripts/export_webbench.py のコア関数テスト(W4、ネットワーク不要)。

TS側(tools/webbench)との契約:
    - int16量子化: float = int16 * scale
    - bitパック: LSB-first(bit i → byte[i>>3] のビット (i&7))
    - 画像thermometer: bit = ((c*T+thr)*H + y)*W + x、判定 z > thr
    - dense thermometer: bit = ((ch*L)+t)*T+thr(HAR時代の契約と同一)
    - 合成IR(seg/concat/dense)は ir/simulate.py で実行可能
"""

import base64
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ はパッケージ外なのでファイルパスから直接import
_SPEC = importlib.util.spec_from_file_location(
    'export_webbench',
    Path(__file__).resolve().parents[1] / 'scripts' / 'export_webbench.py',
)
export_webbench = importlib.util.module_from_spec(_SPEC)
sys.modules['export_webbench'] = export_webbench
_SPEC.loader.exec_module(export_webbench)

quantize_i16 = export_webbench.quantize_i16
pack_bits_b64 = export_webbench.pack_bits_b64
image_thermometer_params = export_webbench.image_thermometer_params
encode_image_thermometer = export_webbench.encode_image_thermometer
dense_thermometer_params = export_webbench.dense_thermometer_params
encode_dense_thermometer = export_webbench.encode_dense_thermometer


class TestQuantizeI16:
    def test_roundtrip_error_bound(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 9, 128)).astype(np.float32) * 3
        xi, scale = quantize_i16(x)
        assert xi.dtype == np.dtype('<i2')
        restored = xi.astype(np.float32) * scale
        # 量子化誤差はscale/2以下(float32丸めの境界マージン2%)
        assert np.abs(restored - x).max() <= scale * 0.51

    def test_extremes_within_range(self):
        x = np.array([[-5.0, 5.0]], dtype=np.float32)
        xi, scale = quantize_i16(x)
        assert xi.max() == 32767
        assert xi.min() == -32767

    def test_zero_input(self):
        xi, scale = quantize_i16(np.zeros((2, 3), dtype=np.float32))
        assert scale == 1.0
        assert (xi == 0).all()


class TestPackBits:
    def test_lsb_first_contract(self):
        # bit i → byte[i>>3] の bit (i&7)
        bits = np.zeros(16, dtype=np.uint8)
        bits[0] = 1   # byte0 bit0 → 0x01
        bits[3] = 1   # byte0 bit3 → 0x08
        bits[8] = 1   # byte1 bit0 → 0x01
        raw = base64.b64decode(pack_bits_b64(bits))
        assert raw == bytes([0x09, 0x01])

    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        bits = (rng.random(100) > 0.5).astype(np.uint8)
        raw = base64.b64decode(pack_bits_b64(bits))
        unpacked = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8), bitorder='little'
        )[:100]
        np.testing.assert_array_equal(unpacked, bits)


class TestImageThermometer:
    def test_bit_order_contract(self):
        """bit index = ((c*T+thr)*H + y)*W + x を素朴3重ループと照合"""
        rng = np.random.default_rng(2)
        x_train = rng.normal(size=(16, 2, 4, 5)).astype(np.float32)
        x = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)
        t = 3
        params = image_thermometer_params(x_train, t)
        bits = encode_image_thermometer(x, params)
        n, c, h, w = x.shape
        assert bits.shape == (n, c * t * h * w)
        mean, std = params['mean'], params['std']
        thr = params['thresholds']
        for i in range(n):
            for ch in range(c):
                for k in range(t):
                    for y in range(h):
                        for xx in range(w):
                            z = (x[i, ch, y, xx] - mean[ch]) / std[ch]
                            expected = int(z > thr[ch, k])
                            bi = ((ch * t + k) * h + y) * w + xx
                            assert bits[i, bi] == expected

    def test_strict_greater(self):
        """境界値ちょうどは0(z > thr、>=ではない)"""
        params = {
            'mean': np.array([0.0]), 'std': np.array([1.0]),
            'thresholds': np.array([[0.5]]),
        }
        x = np.array([[[[0.5]]]], dtype=np.float64)
        assert encode_image_thermometer(x, params)[0, 0] == 0
        x2 = np.array([[[[0.5000001]]]], dtype=np.float64)
        assert encode_image_thermometer(x2, params)[0, 0] == 1

    def test_thresholds_monotone(self):
        rng = np.random.default_rng(3)
        x_train = rng.normal(size=(32, 3, 6, 6)).astype(np.float32)
        params = image_thermometer_params(x_train, 4)
        d = np.diff(params['thresholds'], axis=1)
        assert (d >= 0).all()


class TestDenseThermometer:
    def test_bit_order_contract(self):
        """bit index = ((ch*L)+t)*T+thr(HAR契約)を素朴ループと照合"""
        rng = np.random.default_rng(4)
        x_train = rng.normal(size=(16, 3, 5)).astype(np.float32)
        x = rng.normal(size=(2, 3, 5)).astype(np.float32)
        t = 2
        params = dense_thermometer_params(x_train, t)
        bits = encode_dense_thermometer(x, params)
        n, c, l = x.shape
        assert bits.shape == (n, c * l * t)
        for i in range(n):
            for ch in range(c):
                for ti in range(l):
                    for k in range(t):
                        z = ((x[i, ch, ti] - params['mean'][ch])
                             / params['std'][ch])
                        expected = int(z > params['thresholds'][ch, k])
                        bi = (ch * l + ti) * t + k
                        assert bits[i, bi] == expected


class TestSyntheticIr:
    """合成IR(fixtures/demoの源)がsimulate可能で決定的であること"""

    def test_concat_ir_simulates(self):
        from griffith_logic.ir.simulate import simulate
        g = export_webbench.build_concat_ir(seed=1, in_width=24, a=32, b=16,
                                            k=3)
        rng = np.random.default_rng(0)
        x = (rng.random((4, 24)) > 0.5).astype(np.uint8)
        out = simulate(g, x)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out, simulate(g, x))  # 決定的

    def test_dense_ir_simulates(self):
        from griffith_logic.ir.simulate import simulate
        g = export_webbench.build_dense_ir(seed=2, in_width=20,
                                           widths=[32, 16], k=4)
        rng = np.random.default_rng(0)
        x = (rng.random((5, 20)) > 0.5).astype(np.uint8)
        out = simulate(g, x)
        assert out.shape == (5, 4)
        assert out.max() <= 4  # group_size=4のpopcount上限

    @pytest.mark.slow
    def test_seg_ir_simulates(self):
        """conv/pool/upsample/SpatialGroupSum混在IRのsimulate往復(やや重い)"""
        from griffith_logic.ir.simulate import simulate
        g = export_webbench.build_seg_ir(
            seed=3, in_shape=(2, 6, 6), oc1=3, oc2=4, k=2, depth=2,
        )
        types = {n.type for n in g.nodes}
        assert 'conv2d' in types and 'or_pool2d' in types
        rng = np.random.default_rng(0)
        x = (rng.random((2, g.inputs[0].width)) > 0.5).astype(np.uint8)
        out = simulate(g, x)
        assert out.shape == (2, g.outputs[0].width)

    def test_readout_meta_spatial(self):
        g = export_webbench.build_seg_ir(
            seed=3, in_shape=(2, 6, 6), oc1=3, oc2=4, k=2, depth=2,
        )
        meta = export_webbench._readout_meta(g)
        assert meta['kind'] == 'spatial_scores'
        assert meta['k'] == g.outputs[0].width


class TestFaceDataset:
    """WIDER FACE顔検出のckpt非依存export改修(ckptが無くても書ける部分)。"""

    def test_face_in_rgb_datasets(self):
        assert 'face' in export_webbench._RGB_DATASETS

    def test_det_labels_face(self):
        assert export_webbench._det_labels('face', {}, 1) == ['face']

    def test_det_labels_face_ignores_n_classes(self):
        # face は常に1クラスなので n_classes 引数の値に関わらず ['face'] 固定
        assert export_webbench._det_labels('face', {}, 5) == ['face']

    def test_det_labels_fewclass(self):
        assert export_webbench._det_labels(
            'fewclass', {'fewclass_classes': 'person,car'}, 2,
        ) == ['person', 'car']

    def test_det_labels_fewclass_fallback(self):
        # fewclass_classes が空文字ならクラス数からの数値連番へフォールバック
        assert export_webbench._det_labels(
            'fewclass', {'fewclass_classes': ''}, 3,
        ) == ['0', '1', '2']

    def test_det_labels_other(self):
        assert export_webbench._det_labels('multimnist', {}, 3) == \
            ['0', '1', '2']

    def test_build_centernetlut_from_ckpt_face_n_classes_and_in_channels(
            self, tmp_path):
        """合成ckptからの再構築テスト(罠の回帰テスト)。

        a.get('n_classes', 10) は --n-classes という argparse引数が存在しない
        ため常に10へフォールバックする罠がある。face専用分岐が無いと
        n_classes=10・in_channels=3(is_color=Falseなら1*3)のいずれかで
        load_state_dict が形状不一致になる、または黙って誤ったn_classesで
        モデルが再構築される。ここでは実際にCenterNetLUT(n_classes=1,
        in_channels=9)を構築しその state_dict を積んだ合成ckptから
        build_centernetlut_from_ckpt が正しく n_classes=1・in_channels=9 を
        復元することを厳密に検証する。
        """
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
        from detection_model import CenterNetLUT

        args = {
            'dataset': 'face', 'k': 4, 'n_thresholds': 3, 'tree_depth': 2,
            'head_depth': 2, 'n_inputs': 2, 'hm_group': 2, 'size_group': 2,
            'offset_group': 2, 'input_channels_per_tree': 2,
            'parameterization': 'softmax16g', 'n_layers': 3,
            'canvas_size': 32, 'wiring': 'learnable', 'head_type': 'lut',
        }
        # n_rgb(=3, face)*n_thresholds(=3) = 9ch が期待値
        expected_in_channels = 9
        real_model = CenterNetLUT(
            n_classes=1, k=args['k'], in_channels=expected_in_channels,
            tree_depth=args['tree_depth'], head_depth=args['head_depth'],
            n_inputs=args['n_inputs'], hm_group=args['hm_group'],
            size_group=args['size_group'], offset_group=args['offset_group'],
            input_channels_per_tree=args['input_channels_per_tree'],
            parameterization=args['parameterization'],
            n_layers=args['n_layers'], wiring=args['wiring'],
        )
        ckpt_path = tmp_path / 'face_tiny.pt'
        torch.save({
            'model_state_dict': real_model.state_dict(),
            'metadata': {'args': args},
            'best_metric': 0.0,
        }, ckpt_path)

        model, a, best = export_webbench.build_centernetlut_from_ckpt(
            str(ckpt_path))
        assert a['dataset'] == 'face'
        assert model.n_classes == 1
        # in_channels は CenterNetLUT自体には保持されないため、バックボーン
        # 先頭のLutConv2d(stage1[0])の in_channels から間接的に確認する
        # (load_state_dict自体も形状不一致ならRuntimeErrorで既に落ちている)
        assert model.stage1[0].in_channels == expected_in_channels


class TestAboutBlock:
    """説明文を機械可読にする `about` ブロック。

    契約(tools/webbench/src/modelMeta.ts と対応):
        - 値の無いキーは書かない(TS側は欠損を「—」で描画し、嘘の既定値を出さない)
        - `about` は json_bytes/gzip_bytes の計測**前**に doc へ入る
        - 既存モデルJSONは再生成しないため、`about` が無い doc も正当
          (TS側 applyAbout が静的散文へフォールバックする)
    """

    def test_labels_and_recipe_from_args(self):
        a = {
            'dataset': 'face', 'k': 256, 'wiring': 'learnable',
            'n_thresholds': 31, 'input_channels_per_tree': 32,
            'edge_thresholds': '0.15,0.35', 'teacher_ckpt': 'runs/t/best.pt',
        }
        about = export_webbench.build_about(
            task='detection', args_dict=a, source_ckpt='runs/s/best.pt')
        assert about['task'] == 'detection'
        assert about['task_label']
        assert about['dataset'] == 'face'
        assert about['dataset_label'] == 'WIDER FACE'
        assert about['recipe']['k'] == 256
        assert about['recipe']['wiring'] == 'learnable'
        assert about['recipe']['n_thresholds'] == 31
        assert about['teacher']['teacher_ckpt'] == 'runs/t/best.pt'
        assert about['source_ckpt'] == 'runs/s/best.pt'
        assert about['exported_at']

    def test_missing_values_are_omitted_not_defaulted(self):
        about = export_webbench.build_about(task='segmentation', args_dict={})
        # データセット/レシピ/teacher が分からないときは**キーごと出さない**
        assert 'dataset' not in about
        assert 'recipe' not in about
        assert 'teacher' not in about
        assert 'wandb_run_id' not in about
        assert about['task'] == 'segmentation'

    def test_unknown_dataset_keeps_raw_name_without_label(self):
        about = export_webbench.build_about(
            task='detection', args_dict={'dataset': 'brand_new_ds'})
        assert about['dataset'] == 'brand_new_ds'
        assert 'dataset_label' not in about  # 存在しない表示名を捏造しない

    def test_doc_includes_about_and_sizes_account_for_it(self):
        import json as _json

        g = export_webbench.build_seg_ir(
            seed=3, in_shape=(2, 6, 6), oc1=3, oc2=4, k=2, depth=2)
        about = export_webbench.build_about(
            task='segmentation', args_dict={'dataset': 'pet', 'k': 32})
        doc = export_webbench.build_model_doc(
            g, preproc={'kind': 'image_thermometer'}, labels=['background', 'pet'],
            eval_info={'best_hard_miou': 0.5}, about=about)
        assert doc['about']['dataset_label'] == 'Oxford-IIIT Pet'
        # 配信サイズ表示が実ファイルとずれないこと(aboutを含めて計測している)
        rendered = _json.dumps(doc, separators=(',', ':')).encode()
        # json_bytes 自体を後から resources に足しているぶんだけ増えるので、
        # 「about を含めない場合より確実に大きい」ことで検証する
        doc_wo = export_webbench.build_model_doc(
            g, preproc={'kind': 'image_thermometer'}, labels=['background', 'pet'],
            eval_info={'best_hard_miou': 0.5})
        assert 'about' not in doc_wo  # フォールバック契約: about が無い doc も正当
        assert doc['resources']['json_bytes'] > doc_wo['resources']['json_bytes']
        assert doc['resources']['json_bytes'] <= len(rendered)
