"""Metal等価性ガードの検証。

`tests/deploy/test_verify_c.py` と同型: ガード自体が仕事をしているか
(=壊れたモデルを通さないか)を故障注入で確かめる。基準は常に
`ir/simulate.py`(教訓)。macOS + Apple GPU のある環境でのみ動く。
"""

import json

import pytest

from griffith_logic.deploy import build_bundle, save_bundle
from griffith_logic.deploy.metalverify import (
    has_metal,
    verify_metal_matches_simulate,
)

pytestmark = pytest.mark.skipif(
    not has_metal(), reason='macOS + Metal のある環境でのみ実行')


@pytest.mark.slow
class TestVerifyPasses:
    def test_seg_model_passes_image_path(self, seg_doc, tmp_path):
        """画像前処理を持つモデルは mdl_metal_infer_image(GPU前処理込み)で
        検証される(mdl_metal_infer だけを叩く既存の test_metal.py には
        無かった経路)。"""
        result = verify_metal_matches_simulate(
            build_bundle(seg_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['path'] == 'image', '画像経路で検証されていない'
        assert result['n_samples'] == 2
        assert result['max_abs_diff'] == 0

    def test_dense_model_uses_bit_path(self, dense_doc, tmp_path):
        result = verify_metal_matches_simulate(
            build_bundle(dense_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['path'] == 'bits'


@pytest.mark.slow
class TestFaultInjection:
    """故障注入: 壊したら必ず落ちる(ガードが機能している証拠)。

    test_verify_c.py の該当ケースと同じ改竄を Metal 側にも適用する
    (検証対象自身からゴールデンを作らない、という同じ枠組みを踏む)。
    """

    def test_detects_stale_programs(self, seg_doc, tmp_path):
        """IRだけ更新して programs を作り直し忘れた状態(1バイト相当の
        破壊)を検出する。"""
        from griffith_logic.ir.serialize import ir_from_dict

        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            for tt in node['params']['truth_tables']
        ]
        # programs は古いまま(=生成Metalは古い論理、simulateは新しい論理)
        ir_from_dict(bundle['ir'])   # 変更後IRがそれ自体は妥当なことの確認
        with pytest.raises(ValueError, match='一致しない'):
            verify_metal_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)

    def test_detects_corrupted_preproc_descriptor(self, seg_doc, tmp_path):
        """バンドル内の前処理記述子を1要素だけ壊すと画像経路で検出される。"""
        bundle = build_bundle(seg_doc)
        desc = bundle['preproc_desc']['desc']
        desc[2] = (desc[2] + 37) % 256
        with pytest.raises(ValueError, match='一致しない'):
            verify_metal_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)

    def test_detects_rewired_output(self, seg_doc, tmp_path):
        """出力チャネルの読み出し元(oc_out)を張り替えると検出される。"""
        bundle = build_bundle(seg_doc)
        key = sorted(bundle['programs'])[0]
        oc_out = bundle['programs'][key]['oc_out']
        if len(set(oc_out)) < 2:
            pytest.skip('全出力が同じ値を読んでいる')
        oc_out[0] = next(v for v in oc_out if v != oc_out[0])
        with pytest.raises(ValueError, match='一致しない'):
            verify_metal_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)


@pytest.mark.slow
class TestCli:
    def test_verify_metal_command(self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(seg_doc), path)
        assert main(['deploy', 'verify-metal', str(path), '-n', '1']) == 0
        assert '一致' in capsys.readouterr().out

    def test_verify_metal_fails_on_broken_bundle(self, seg_doc, tmp_path):
        from griffith_logic.cli import main

        bundle = build_bundle(seg_doc)
        key = sorted(bundle['programs'])[0]
        oc_out = bundle['programs'][key]['oc_out']
        oc_out[0] = oc_out[-1]          # 出力の張り替え(意味を変える)
        path = tmp_path / 'broken.bundle.json'
        path.write_text(json.dumps(bundle), encoding='utf-8')
        assert main(['deploy', 'verify-metal', str(path), '-n', '1']) == 1
