"""`griffith deploy bundle --sweep`: dense IR への lossless sweep の組み込み。

構成(``test_resynth.py`` と同じ流儀):
    TestSweepDocLossless  doc→doc 変換の性質(等価・減る・入力不変・provenance)
    TestFaultInjection    多重防御が働くこと(壊れた sweep を差し込む)
    TestBundleSweepCli    CLI(--sweep / --sweep-verify-samples / info 表示 / 既定の回帰)
    TestChainToC          sweep → バンドル → C実行 ⇔ simulate(cc が無ければ skip)
    TestRealModels        配信モデルがあれば実モデルで通す(無ければ skip)
"""

import base64
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle, validate_bundle
from griffith_logic.deploy.bundle import attach_bundle_sizes, save_bundle
from griffith_logic.deploy.sweepdoc import sweep_doc_lossless
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate


def _redundant_dense_doc(webbench, seed: int = 4) -> dict:
    """定数ゲート・重複ゲートを意図的に混ぜた dense モデルdoc。

    1層目(24→48): ゲート0..9を定数(0x0/0xf)に、ゲート10..19をゲート20..29の
    完全複製(同じ配線・同じtt)にする。以降の層は乱数配線なので、定数吸収・CSE・
    未参照除去のどれもが働く。
    """
    graph = webbench.build_dense_ir(
        seed=seed, in_width=24, widths=[48, 40, 32], k=4, name='dense_sweep')
    n1 = graph.nodes[0]
    idx = np.asarray(n1.params['indices']).copy()
    tts = list(n1.params['truth_tables'])
    for o in range(10):
        tts[o] = '0x0' if o % 2 == 0 else '0xf'
    for o in range(10, 20):
        idx[:, o] = idx[:, o + 10]
        tts[o] = tts[o + 10]
    n1.params['indices'] = idx
    n1.params['truth_tables'] = tts
    return webbench.build_model_doc(
        graph,
        preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=['a', 'b', 'c', 'd'],
    )


@pytest.fixture
def red_doc(webbench):
    return _redundant_dense_doc(webbench)


def _gates(doc: dict) -> int:
    return sum(int(n['params']['out_width']) for n in doc['ir']['nodes']
               if n['type'] == 'lut_layer')


def _random_bits(doc: dict, n: int = 64, seed: int = 12345) -> np.ndarray:
    width = int(doc['ir']['inputs'][0]['width'])
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, width), dtype=np.uint8)


class TestSweepDocLossless:
    def test_gates_decrease(self, red_doc):
        new_doc, summary = sweep_doc_lossless(red_doc)
        assert summary['gates_before'] == _gates(red_doc)
        assert summary['gates_after'] == _gates(new_doc)
        assert summary['gates_after'] < summary['gates_before']
        assert summary['constants_absorbed'] > 0
        assert summary['cse_merged'] > 0
        assert summary['swept_gates'] <= summary['gates_after']

    def test_outputs_identical_with_and_without_sweep(self, red_doc):
        """sweep あり/なしのバンドルが simulate で整数一致する(実装内部の検証を
        信用せず、別の乱数で独立に再検証)。"""
        plain = build_bundle(red_doc)
        new_doc, _ = sweep_doc_lossless(red_doc)
        swept = build_bundle(new_doc)
        bits = _random_bits(red_doc)
        want = simulate(ir_from_dict(plain['ir']), bits)
        got = simulate(ir_from_dict(swept['ir']), bits)
        assert np.array_equal(want, got)
        assert _gates(swept) < _gates(plain)

    def test_swept_bundle_validates_strict(self, red_doc):
        new_doc, _ = sweep_doc_lossless(red_doc)
        assert validate_bundle(build_bundle(new_doc), strict=True) == []

    def test_original_doc_is_not_mutated(self, red_doc):
        before = copy.deepcopy(red_doc)
        sweep_doc_lossless(red_doc)
        assert red_doc == before, '入力docを書き換えてはいけない'

    def test_provenance_in_about(self, red_doc):
        new_doc, summary = sweep_doc_lossless(red_doc, verify_samples=5)
        prov = build_bundle(new_doc)['about']['sweep']
        assert prov['mode'] == 'lossless'
        for key in ('gates_before', 'gates_after', 'swept_gates',
                    'passthrough_remaining', 'const_remaining',
                    'constants_absorbed', 'cse_merged', 'gates_removed'):
            assert prov[key] == summary[key]
        assert prov['verify_samples'] == 5
        assert prov['verify_demo_samples'] == 0
        json.dumps(prov)  # JSON直列化できる(numpy型が混ざっていない)

    def test_existing_about_is_preserved(self, red_doc):
        red_doc['about'] = {'ckpt': 'x.pt'}
        new_doc, _ = sweep_doc_lossless(red_doc)
        assert new_doc['about']['ckpt'] == 'x.pt'
        assert 'sweep' in new_doc['about']
        assert red_doc['about'] == {'ckpt': 'x.pt'}

    def test_resources_refreshed(self, red_doc):
        new_doc, _ = sweep_doc_lossless(red_doc)
        assert new_doc['resources']['total_luts'] < red_doc['resources']['total_luts']
        assert new_doc['resources']['json_bytes'] > 0

    def test_demo_samples_bits_are_verified(self, red_doc):
        """bits_b64 形式の demo_samples は実データ検証に使われる。"""
        width = int(red_doc['ir']['inputs'][0]['width'])
        bits = _random_bits(red_doc, n=3, seed=7)
        red_doc['demo_samples'] = [
            {'bits_b64': base64.b64encode(
                np.packbits(b, bitorder='little').tobytes()).decode('ascii')}
            for b in bits
        ]
        assert width == bits.shape[1]
        _, summary = sweep_doc_lossless(red_doc)
        assert summary['verify_demo_samples'] == 3

    def test_conv_ir_is_rejected(self, seg_doc):
        """conv2d を含むIRは lut_layer 列へ展開されてしまうので拒否する。"""
        with pytest.raises(ValueError, match='dense'):
            sweep_doc_lossless(seg_doc)

    def test_verify_samples_must_be_positive(self, red_doc):
        with pytest.raises(ValueError, match='verify_samples'):
            sweep_doc_lossless(red_doc, verify_samples=0)

    def test_no_change_when_nothing_to_sweep(self, webbench):
        """冗長の無い層(全ゲートが2入力を実際に使い、重複も無い)では不変。"""
        graph = webbench.build_dense_ir(
            seed=1, in_width=8, widths=[4], k=2, name='tiny')
        n = graph.nodes[0]
        n.params['indices'] = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        n.params['truth_tables'] = ['0x6'] * 4   # XOR、配線は全て別
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b'])
        new_doc, summary = sweep_doc_lossless(doc)
        assert summary['gates_after'] == summary['gates_before'] == 4
        assert summary['constants_absorbed'] == summary['cse_merged'] == 0


class TestFaultInjection:
    """内部検証(多重防御)が実際に機能していることを故障注入で確かめる。"""

    def _corrupting_sweep(self, monkeypatch):
        import griffith_logic.deploy.sweepdoc as mod
        from griffith_logic.ir.sweep import sweep as real_sweep

        def fake_sweep(graph):
            g, report = real_sweep(graph)
            # 最終 lut_layer(group_sum が位置で全ビットを読む)の先頭ゲートの
            # 真理値表を全ビット反転: 全入力でそのゲートの出力が反転し、
            # group_sum のスコアが必ず変わる。
            last = [n for n in g.nodes if n.type == 'lut_layer'][-1]
            tts = list(last.params['truth_tables'])
            tts[0] = hex(int(tts[0], 16) ^ 0xf)
            last.params['truth_tables'] = tts
            return g, report

        monkeypatch.setattr(mod, 'sweep', fake_sweep)

    def test_corrupted_sweep_is_detected(self, red_doc, monkeypatch):
        self._corrupting_sweep(monkeypatch)
        with pytest.raises(ValueError, match='一致しない'):
            sweep_doc_lossless(red_doc)

    def test_corrupted_sweep_aborts_cli(self, red_doc, tmp_path, monkeypatch,
                                        capsys):
        """CLIは壊れた sweep ではバンドルを書かず非0終了する。"""
        from griffith_logic.cli import main

        self._corrupting_sweep(monkeypatch)
        src = tmp_path / 'm.json'
        src.write_text(json.dumps(red_doc), encoding='utf-8')
        out = tmp_path / 'm.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out), '--sweep'])
        assert rc != 0
        assert not out.exists()
        assert '一致しない' in capsys.readouterr().err

    def test_corrupted_sweep_detected_with_demo_samples(self, red_doc,
                                                        monkeypatch):
        """demo_samples(実データ)があっても壊れた sweep は ValueError になる。"""
        width = int(red_doc['ir']['inputs'][0]['width'])
        bits = _random_bits(red_doc, n=2, seed=3)
        assert bits.shape[1] == width
        red_doc['demo_samples'] = [
            {'bits_b64': base64.b64encode(
                np.packbits(b, bitorder='little').tobytes()).decode('ascii')}
            for b in bits
        ]
        self._corrupting_sweep(monkeypatch)
        with pytest.raises(ValueError, match='一致しない'):
            sweep_doc_lossless(red_doc)


class TestBundleSweepCli:
    def _write(self, tmp_path, doc):
        src = tmp_path / 'model.json'
        src.write_text(json.dumps(doc), encoding='utf-8')
        return src

    def test_sweep_flag_produces_strict_clean_bundle(self, red_doc, tmp_path,
                                                     capsys):
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        out = tmp_path / 'model.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out), '--sweep',
                   '--sweep-verify-samples', '4'])
        assert rc == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert validate_bundle(bundle, strict=True) == []
        prov = bundle['about']['sweep']
        assert prov['gates_after'] < prov['gates_before']
        assert prov['verify_samples'] == 4
        text = capsys.readouterr().out
        assert 'sweep' in text and '→' in text

    def test_info_shows_sweep_gate_counts(self, red_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        out = tmp_path / 'model.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--sweep']) == 0
        capsys.readouterr()
        assert main(['deploy', 'info', str(out)]) == 0
        text = capsys.readouterr().out
        prov = json.loads(out.read_text(encoding='utf-8'))['about']['sweep']
        assert f"{prov['gates_before']:,} → {prov['gates_after']:,}" in text

    def test_info_without_sweep_has_no_sweep_line(self, red_doc, tmp_path,
                                                  capsys):
        """既定(--compress fast)は sweep を自動適用するので、sweep 行が
        出ないのは --compress none のとき。"""
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        out = tmp_path / 'model.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none']) == 0
        capsys.readouterr()
        assert main(['deploy', 'info', str(out)]) == 0
        assert 'sweep' not in capsys.readouterr().out

    def test_conv_model_with_sweep_fails_cleanly(self, seg_doc, tmp_path,
                                                 capsys):
        from griffith_logic.cli import main

        src = self._write(tmp_path, seg_doc)
        out = tmp_path / 'model.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out), '--sweep'])
        assert rc == 2
        assert not out.exists()
        assert '--resynth' in capsys.readouterr().err

    def test_compress_none_output_is_byte_identical_to_legacy_pipeline(
            self, red_doc, tmp_path):
        """--compress none の出力は、旧版の経路(build_bundle →
        attach_bundle_sizes → save_bundle)と1バイトも変わらない
        (で既定が fast になったので none を明示する)。"""
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        out = tmp_path / 'cli.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none']) == 0

        legacy_doc = json.loads(src.read_text(encoding='utf-8'))
        legacy = build_bundle(legacy_doc, verify=True)
        attach_bundle_sizes(legacy)
        legacy_out = tmp_path / 'legacy.bundle.json'
        save_bundle(legacy, legacy_out)

        assert out.read_bytes() == legacy_out.read_bytes()
        assert 'sweep' not in json.loads(out.read_text(encoding='utf-8')).get(
            'about', {})

    def test_verify_samples_flag_alone_changes_nothing(self, red_doc, tmp_path):
        """--sweep-verify-samples は sweep を行わないとき(--compress none)は
        無視される(出力不変)。: 既定の fast では sweep が走るので効く。"""
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        a = tmp_path / 'a.bundle.json'
        b = tmp_path / 'b.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(a),
                     '--compress', 'none']) == 0
        assert main(['deploy', 'bundle', str(src), '-o', str(b),
                     '--compress', 'none', '--sweep-verify-samples', '3']) == 0
        assert a.read_bytes() == b.read_bytes()

    def test_verify_samples_flag_applies_to_default_auto_sweep(
            self, red_doc, tmp_path):
        """既定(fast)の自動 sweep にも --sweep-verify-samples が効く。"""
        from griffith_logic.cli import main

        src = self._write(tmp_path, red_doc)
        out = tmp_path / 'c.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--sweep-verify-samples', '3']) == 0
        prov = json.loads(out.read_text(encoding='utf-8'))['about']['sweep']
        assert prov['verify_samples'] == 3


class TestChainToC:
    """sweep → バンドル化 → Cランタイム検証の鎖が壊れていないこと。

    sweep はパススルー/NOTゲートを層から除去しない(層IRは層をまたぐエイリアス
    を表現できない)ので、build_plan / codegen-c は従来の dense と同じ形を受け取る。
    """

    def test_swept_bundle_verify_c(self, red_doc, tmp_path, cc):
        from griffith_logic.deploy.verify import verify_c_matches_simulate

        new_doc, summary = sweep_doc_lossless(red_doc)
        bundle = build_bundle(new_doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=3, work_dir=tmp_path, verbose=False)
        assert result['max_abs_diff'] == 0
        assert summary['gates_after'] < summary['gates_before']

    def test_swept_bundle_matches_unswept_c_outputs(self, red_doc, tmp_path, cc):
        """C実行の出力が sweep 有無で一致する(simulate 経由でなくバンドル同士)。"""
        from griffith_logic.deploy.verify import verify_c_matches_simulate

        plain = build_bundle(red_doc)
        new_doc, _ = sweep_doc_lossless(red_doc)
        swept = build_bundle(new_doc)
        for name, bundle in (('plain', plain), ('swept', swept)):
            work = tmp_path / name
            work.mkdir()
            res = verify_c_matches_simulate(
                bundle, n_samples=4, seed=9, work_dir=work, verbose=False)
            assert res['max_abs_diff'] == 0


# ---------------------------------------------------------------------------
# 実モデル(配信モデルがある環境のみ)
# ---------------------------------------------------------------------------
MODELS_DIR = (Path(__file__).resolve().parents[2] / 'tools' / 'webbench'
              / 'public' / 'models')


def _dense_model_paths() -> list[Path]:
    """配信モデルのうち conv2d を含まない(sweep 対象の)ものだけ。"""
    if not MODELS_DIR.is_dir():
        return []
    out = []
    for path in sorted(MODELS_DIR.glob('*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not (isinstance(doc, dict) and 'ir' in doc):
            continue
        types = {n['type'] for n in doc['ir']['nodes']}
        if types & {'conv2d', 'or_pool2d', 'upsample2d', 'permute', 'recurrent'}:
            continue
        out.append(path)
    return out


DENSE_PATHS = _dense_model_paths()


@pytest.mark.slow
@pytest.mark.skipif(not DENSE_PATHS,
                    reason='dense の配信モデル(tools/webbench/public/models)が無い環境')
@pytest.mark.parametrize('path', DENSE_PATHS, ids=lambda p: p.stem)
def test_real_dense_model_sweep_chain(path, tmp_path, cc):
    """実モデル: sweep でゲートが減らない事は無く、バンドル→verify-c まで通る。"""
    from griffith_logic.deploy.verify import verify_c_matches_simulate

    doc = json.loads(path.read_text(encoding='utf-8'))
    new_doc, summary = sweep_doc_lossless(doc)
    assert summary['gates_after'] <= summary['gates_before']
    bundle = build_bundle(new_doc)
    assert validate_bundle(bundle, strict=True) == []
    result = verify_c_matches_simulate(
        bundle, n_samples=3, work_dir=tmp_path, verbose=False)
    assert result['max_abs_diff'] == 0
