"""C等価性ガードの検証(K5)。

ガード自体が仕事をしているか(=壊れたモデルを通さないか)を、故障注入で
確かめる。教訓: 検証対象自身からゴールデンを作ると、壊れたもの同士が
一致して全部緑になる。ここでは基準を常に `ir/simulate.py` に置く。
"""

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle, save_bundle
from griffith_logic.deploy.verify import (
    find_cc,
    verify_c_matches_simulate,
)

pytestmark = pytest.mark.skipif(
    find_cc() is None, reason='Cコンパイラが無い環境')


class TestVerifyPasses:
    def test_seg_model_passes(self, seg_doc, tmp_path):
        result = verify_c_matches_simulate(
            build_bundle(seg_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['path'] == 'image', '画像経路で検証されていない'
        assert result['n_samples'] == 2
        assert result['max_abs_diff'] == 0

    def test_dense_model_uses_bit_path(self, dense_doc, tmp_path):
        result = verify_c_matches_simulate(
            build_bundle(dense_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['path'] == 'bits'


@pytest.mark.slow
class TestFaultInjection:
    """故障注入: 壊したら必ず落ちる(ガードが機能している証拠)。"""

    def test_consistent_edit_is_a_different_but_valid_model(
            self, seg_doc, tmp_path):
        """IRとprogramsを揃えて書き換えたものは「別のモデル」として通る。

        検証しているのは「バンドルの各層が互いに整合しているか」であって
        「モデルが意図どおりか」ではない。後者はエクスポート時の
        assert_ir_matches_torch(PyTorch ⇔ IR)が受け持つ。境界をはっきり
        させるためにこの通過ケースを明示的に置いておく。
        """
        from griffith_logic.deploy.programs import build_programs
        from griffith_logic.ir.serialize import ir_from_dict

        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            for tt in node['params']['truth_tables']
        ]
        bundle['programs'] = build_programs(ir_from_dict(bundle['ir']))
        result = verify_c_matches_simulate(
            bundle, n_samples=1, work_dir=tmp_path, verbose=False)
        assert result['max_abs_diff'] == 0

    def test_detects_stale_programs(self, seg_doc, tmp_path):
        """IRだけ更新して programs を作り直し忘れた状態を検出する。"""
        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            for tt in node['params']['truth_tables']
        ]
        # programs は古いまま(=C は古い論理、simulate は新しい論理)
        with pytest.raises(ValueError, match='一致しない'):
            verify_c_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)

    def test_detects_corrupted_preproc_descriptor(self, seg_doc, tmp_path):
        """バンドル内の前処理記述子だけを壊すと画像経路で検出される。

        検証の基準は **preproc宣言から作り直した記述子** なので、
        焼かれた preproc_desc が壊れていれば食い違う。両側で同じ
        preproc_desc を使っていた頃はここが素通りしていた。
        """
        bundle = build_bundle(seg_doc)
        desc = bundle['preproc_desc']['desc']
        desc[2] = (desc[2] + 37) % 256   # 先頭チャネルのTHERMO_STEP閾値(w2)
        with pytest.raises(ValueError, match='一致しない'):
            verify_c_matches_simulate(
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
            verify_c_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)

    def test_write_order_within_a_bucket_is_irrelevant(self, seg_doc, tmp_path):
        """バケット内での write_oc の順序は意味を持たない(仕様の確認)。

        書き出しは (oc → oc_src_off[oc]) の対応で行うので、同じバケットの中で
        順番を入れ替えても結果は変わらない。「順序を壊しても落ちない」のは
        ガードの穴ではなく、そういう仕様であることをここで固定しておく
        (バケットの**境界**を壊せば当然落ちる = 上の張り替えテスト)。
        """
        bundle = build_bundle(seg_doc)
        key = sorted(bundle['programs'])[0]
        prog = bundle['programs'][key]
        start = prog['write_start']
        bucket = next((b for b in range(len(start) - 1)
                       if start[b + 1] - start[b] >= 2), None)
        if bucket is None:
            pytest.skip('2要素以上のバケットが無い')
        i = start[bucket]
        woc = prog['write_oc']
        woc[i], woc[i + 1] = woc[i + 1], woc[i]
        result = verify_c_matches_simulate(
            bundle, n_samples=1, work_dir=tmp_path, verbose=False)
        assert result['max_abs_diff'] == 0


class TestCli:
    def test_verify_c_command(self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(seg_doc), path)
        assert main(['deploy', 'verify-c', str(path), '-n', '1']) == 0
        assert '一致' in capsys.readouterr().out

    def test_verify_c_fails_on_broken_bundle(self, seg_doc, tmp_path):
        import json

        from griffith_logic.cli import main

        bundle = build_bundle(seg_doc)
        key = sorted(bundle['programs'])[0]
        oc_out = bundle['programs'][key]['oc_out']
        oc_out[0] = oc_out[-1]          # 出力の張り替え(意味を変える)
        path = tmp_path / 'broken.bundle.json'
        path.write_text(json.dumps(bundle), encoding='utf-8')
        assert main(['deploy', 'verify-c', str(path), '-n', '1']) == 1


def test_scores_are_plain_integers(seg_doc, tmp_path):
    """スコアが整数のまま出ている(浮動小数点を経由していない)。"""
    from griffith_logic.deploy.preproc_desc import encode_bits
    from griffith_logic.deploy.verify import build_c_model, run_c_model
    from griffith_logic.ir.serialize import ir_from_dict

    bundle = build_bundle(seg_doc)
    graph = ir_from_dict(bundle['ir'])
    exe = build_c_model(bundle, tmp_path, image_entry=True)
    plan = bundle['preproc_desc']
    rng = np.random.default_rng(2)
    image = rng.integers(
        0, 256, size=(plan['image_channels'], plan['height'], plan['width']),
        dtype=np.uint8)
    scores = run_c_model(exe, image.tobytes(), int(graph.outputs[0].width))
    assert scores.dtype == np.int64
    assert scores.min() >= 0, 'group_sumの出力は非負のはず'
    encode_bits(plan, image[None])   # 記述子経路が例外なく通ることも確認
