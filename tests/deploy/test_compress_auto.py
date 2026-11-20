"""回路圧縮の自動適用: `bundle --compress` と `codegen-c`/`verify-c` の `--dense-flat auto`。

構成:
    TestBundleCompress    (a)(b)(c) bundle の既定=fast(dense に sweep、conv はスキップ記録)、
                          max(conv に resynth)、none(従来同一)、個別フラグの優先、fail-safe
    TestDenseFlatAuto     (d)(f) codegen-c の auto 判定(閾値で両経路を踏む)、dense-batch は off
    TestVerifyCAuto       (e) verify-c が auto 経路で通る(ゴールデンは元バンドルIRの simulate)
    TestInfo              info の表示
    TestXorExample        examples/06_deploy_c の XOR を既定コマンドのまま通す

ゴールデンは常に**圧縮前の元 IR の simulate**(圧縮後 IR 自身ではない)。
"""

import copy
import json

import numpy as np
import pytest

from griffith_logic.cli import main
from griffith_logic.deploy import build_bundle, load_bundle, validate_bundle
from griffith_logic.deploy.flatten_dense import (
    describe_dense_flat,
    resolve_dense_flat,
)
from griffith_logic.deploy.plan import DENSE_FLAT_AUTO_MIN_REDUCTION
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate

from .conftest import CC
from .test_dense_flat import _dense_ir, _random_layers


def _write(tmp_path, doc, name='model.json'):
    src = tmp_path / name
    src.write_text(json.dumps(doc), encoding='utf-8')
    return src


def _bits(doc, n=64, seed=777):
    width = int(doc['ir']['inputs'][0]['width'])
    return np.random.default_rng(seed).integers(
        0, 2, size=(n, width), dtype=np.uint8)


def _same_outputs(doc_a, doc_b) -> bool:
    x = _bits(doc_a)
    return np.array_equal(simulate(ir_from_dict(doc_a['ir']), x),
                          simulate(ir_from_dict(doc_b['ir']), x))


def _flat_doc(p_trivial: float, *, seed=3, nontrivial_only=False) -> dict:
    """lut_layer 列 + group_sum のモデル doc。p_trivial で平坦化の削減率を振る。"""
    rng = np.random.default_rng(seed)
    layers = _random_layers(rng, 40, [64, 64, 48], p_trivial=p_trivial)
    if nontrivial_only:
        # 自明でない 2 入力関数(AND系/XOR/XNOR)だけにして削減率を最小にする
        nontriv = np.array([0x1, 0x2, 0x4, 0x6, 0x7, 0x8, 0x9, 0xb, 0xd, 0xe])
        layers = [(idx, rng.choice(nontriv, size=len(tts)))
                  for idx, tts in layers]
    g = _dense_ir(layers, in_width=40, k=6)
    return {'ir': ir_to_dict(g), 'preproc': {}, 'labels': []}


@pytest.fixture
def high_doc():
    """自明ゲートが多く、平坦化で実効ゲートが大きく減る(削減率 >= 閾値)doc。"""
    doc = _flat_doc(0.6)
    _, dec = resolve_dense_flat(doc, mode='on')
    assert dec['reduction'] >= DENSE_FLAT_AUTO_MIN_REDUCTION + 0.1
    return doc


@pytest.fixture
def low_doc():
    """自明ゲートが無く、平坦化してもほぼ減らない(削減率 < 閾値)doc。"""
    doc = _flat_doc(0.0, nontrivial_only=True)
    _, dec = resolve_dense_flat(doc, mode='on')
    assert dec['reduction'] < DENSE_FLAT_AUTO_MIN_REDUCTION
    return doc


# ---------------------------------------------------------------------------
# bundle --compress
# ---------------------------------------------------------------------------
class TestBundleCompress:
    def test_default_applies_sweep_on_dense_and_is_lossless(self, dense_doc,
                                                            tmp_path, capsys):
        """(a) 既定の bundle は dense に sweep を自動適用し、simulate が一致する。"""
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        comp = bundle['about']['compress']
        assert {k: comp[k] for k in ('level', 'applied', 'skipped')} == {
            'level': 'fast', 'applied': ['sweep'], 'skipped': []}
        before, after = comp['gates']['sweep']
        assert after <= before
        assert 'always' not in comp                 # conv が無いので命令列化なし
        assert bundle['about']['sweep']['mode'] == 'lossless'
        assert validate_bundle(bundle, strict=True) == []
        # ゴールデンは圧縮前の元doc(テスト自身が独立に検証)
        assert _same_outputs(dense_doc, bundle)
        text = capsys.readouterr().out
        assert '圧縮(compress=fast)' in text and '適用 sweep' in text

    def test_default_reduces_gates_on_redundant_dense(self, webbench, tmp_path):
        """冗長な dense(定数・重複ゲート入り)では既定でゲートが実際に減る。"""
        from .test_bundle_sweep import _gates, _redundant_dense_doc

        doc = _redundant_dense_doc(webbench)
        src = _write(tmp_path, doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert _gates(bundle) < _gates(doc)
        assert _same_outputs(doc, bundle)

    def test_conv_ir_default_applies_resynth_and_records_program(
            self, seg_doc, tmp_path, capsys):
        """(b): 既定(fast)は conv2d 木にも lossless 再合成をかけ、命令列化の
        削減も同じ形式で記録する。sweep は対象外(lut_layer が無い)なので記録しない。"""
        src = _write(tmp_path, seg_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        comp = bundle['about']['compress']
        assert comp['level'] == 'fast' and comp['applied'] == ['resynth']
        assert comp['skipped'] == []
        b, a = comp['gates']['resynth']
        assert a <= b
        assert comp['always'][0]['pass'] == 'conv-program'
        pb, pa = comp['always'][0]['gates']
        assert pa <= pb
        assert bundle['about']['resynth']['verify'] == 'fast'
        assert 'sweep' not in bundle['about']
        assert validate_bundle(bundle, strict=True) == []
        assert _same_outputs(seg_doc, bundle)       # 元 doc と出力が一致
        text = capsys.readouterr().out
        assert 'conv再合成' in text and 'conv命令列化' in text and '常時' in text

    def test_max_applies_resynth_on_conv(self, seg_doc, tmp_path):
        """(c) --compress max は conv2d を含む IR に resynth lossless を適用する。"""
        src = _write(tmp_path, seg_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'max', '--resynth-max-cost', '3']) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        comp = bundle['about']['compress']
        assert comp['level'] == 'max'
        assert comp['applied'] == ['resynth']
        assert comp['skipped'] == []                # sweep は対象外(lut_layer なし)
        prov = bundle['about']['resynth']
        assert prov['mode'] == 'lossless' and prov['max_cost'] == 3
        assert prov['verify'] == 'simulate'          # max は simulate 照合つき
        assert prov['gates_after'] <= prov['gates_before']
        assert validate_bundle(bundle, strict=True) == []
        assert _same_outputs(seg_doc, bundle)

    def test_max_on_dense_applies_sweep_and_skips_resynth(self, dense_doc,
                                                          tmp_path):
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'max']) == 0
        comp = json.loads(out.read_text(encoding='utf-8'))['about']['compress']
        assert comp['applied'] == ['sweep']
        assert comp['skipped'] == []                # resynth は対象外(conv なし)

    def test_none_is_byte_identical_to_legacy_pipeline(self, dense_doc,
                                                       tmp_path):
        """--compress none は圧縮導入前の経路と1バイトも変わらない。"""
        from griffith_logic.deploy.bundle import attach_bundle_sizes, save_bundle

        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'cli.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none']) == 0
        legacy = build_bundle(json.loads(src.read_text(encoding='utf-8')))
        attach_bundle_sizes(legacy)
        ref = save_bundle(legacy, tmp_path / 'legacy.bundle.json')
        assert out.read_bytes() == ref.read_bytes()
        assert 'compress' not in json.loads(
            out.read_text(encoding='utf-8')).get('about', {})

    def test_explicit_sweep_beats_compress_none(self, dense_doc, tmp_path):
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none', '--sweep']) == 0
        about = json.loads(out.read_text(encoding='utf-8'))['about']
        assert {k: about['compress'][k] for k in ('level', 'applied', 'skipped')} \
            == {'level': 'none', 'applied': ['sweep'], 'skipped': []}
        assert 'sweep' in about

    def test_no_sweep_beats_compress_fast(self, dense_doc, tmp_path):
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--no-sweep']) == 0
        about = json.loads(out.read_text(encoding='utf-8'))['about']
        assert about['compress']['applied'] == []
        assert about['compress']['skipped'] == [
            {'pass': 'sweep', 'reason': '--no-sweep 指定'}]
        assert 'sweep' not in about

    def test_sweep_and_no_sweep_are_mutually_exclusive(self, dense_doc,
                                                       tmp_path):
        src = _write(tmp_path, dense_doc)
        with pytest.raises(SystemExit):
            main(['deploy', 'bundle', str(src), '--sweep', '--no-sweep'])

    def test_explicit_resynth_beats_compress_fast(self, seg_doc, tmp_path):
        """--resynth lossless は fast の自動再合成より優先し、simulate 照合で走る。"""
        src = _write(tmp_path, seg_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--resynth', 'lossless', '--resynth-max-cost', '3']) == 0
        about = json.loads(out.read_text(encoding='utf-8'))['about']
        comp = about['compress']
        assert comp['level'] == 'fast' and comp['applied'] == ['resynth']
        assert about['resynth']['verify'] == 'simulate'
        assert about['resynth']['max_cost'] == 3

    def test_explicit_sweep_on_conv_still_fails_cleanly(self, seg_doc, tmp_path,
                                                        capsys):
        src = _write(tmp_path, seg_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--sweep']) == 2
        assert not out.exists()
        assert '--resynth' in capsys.readouterr().err

    def test_auto_sweep_failure_is_failsafe(self, dense_doc, tmp_path,
                                            monkeypatch, capsys):
        """自動 sweep が壊れていても(多重防御が検出)、圧縮なしでバンドル化を続ける。
        圧縮なしの IR は元から正しいので出荷は止めず、失敗は記録+警告する。"""
        from griffith_logic.deploy import sweepdoc

        def broken(doc, **kw):
            raise ValueError('sweep_doc_lossless: 一致しない(注入)')

        monkeypatch.setattr(sweepdoc, 'sweep_doc_lossless', broken)
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        comp = bundle['about']['compress']
        assert comp['applied'] == []
        assert comp['skipped'][0]['pass'] == 'sweep'
        assert '失敗' in comp['skipped'][0]['reason']
        assert _same_outputs(dense_doc, bundle)
        assert '警告' in capsys.readouterr().err
        # 明示 --sweep では従来どおりエラー
        rc = main(['deploy', 'bundle', str(src), '-o', str(tmp_path / 'x.json'),
                   '--sweep'])
        assert rc == 2

    def test_parser_defaults(self):
        from griffith_logic.cli import _build_parser

        args = _build_parser().parse_args(['deploy', 'bundle', 'm.json'])
        assert args.compress == 'fast'
        assert args.sweep is False and args.no_sweep is False
        args = _build_parser().parse_args(['deploy', 'codegen-c', 'b.json',
                                           '-o', 'out'])
        assert args.dense_flat == 'auto' and args.dense_flat_min_reduction is None
        args = _build_parser().parse_args(['deploy', 'verify-c', 'b.json'])
        assert args.dense_flat == 'auto'

    def test_dense_flat_flag_is_recorded_in_compress(self, high_doc, tmp_path):
        """bundle --dense-flat(IRに焼く経路)は互換で残り、compress に載る。"""
        src = _write(tmp_path, high_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--dense-flat']) == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert bundle['about']['compress']['applied'] == ['sweep', 'dense-flat']
        assert 'lut_flat' in {n['type'] for n in bundle['ir']['nodes']}

    def test_mdlb_roundtrips_about_compress(self, dense_doc, tmp_path):
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.mdlb'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        assert load_bundle(out)['about']['compress']['applied'] == ['sweep']


# ---------------------------------------------------------------------------
# codegen-c の --dense-flat auto
# ---------------------------------------------------------------------------
def _bundle_file(tmp_path, doc, name='m.bundle.json') -> str:
    """doc を --compress none でバンドル化したパス(IR をそのまま使う)。"""
    src = _write(tmp_path, doc, 'm.json')
    out = tmp_path / name
    assert main(['deploy', 'bundle', str(src), '-o', str(out),
                 '--compress', 'none']) == 0
    return str(out)


def _codegen(tmp_path, bundle_path, *extra) -> tuple[str, str]:
    out = tmp_path / 'c'
    assert main(['deploy', 'codegen-c', bundle_path, '-o', str(out),
                 '--no-runtime', *extra]) == 0
    return ((out / 'model.h').read_text(encoding='utf-8'),
            (out / 'model.c').read_text(encoding='utf-8'))


class TestDenseFlatAuto:
    def test_default_uses_flat_when_reduction_above_threshold(self, high_doc,
                                                              tmp_path, capsys):
        """(d) 既定(auto)で削減率 >= 閾値なら lut_flat 経路。"""
        path = _bundle_file(tmp_path, high_doc)
        h, c = _codegen(tmp_path, path)
        assert '#define MDL_DENSE_FLAT 1' in h
        assert 'dense-flat(auto): 採用' in h.split('*/')[0]
        assert 'mdl_lut_flat' in c
        assert 'dense-flat(auto): 採用' in capsys.readouterr().out

    def test_default_keeps_legacy_when_reduction_below_threshold(
            self, low_doc, tmp_path, capsys):
        """(d) 削減率 < 閾値なら従来経路(bit-slice)。理由と削減率は model.h/標準出力に出る。"""
        path = _bundle_file(tmp_path, low_doc)
        h, c = _codegen(tmp_path, path)
        assert '#define MDL_DENSE_FLAT 0' in h
        assert 'dense-flat(auto): 不採用' in h.split('*/')[0]
        assert 'lut_flat' not in c
        assert 'dense-flat(auto): 不採用' in capsys.readouterr().out

    def test_threshold_flag_flips_the_decision(self, high_doc, low_doc,
                                               tmp_path):
        """(d) 閾値をいじって両方の経路を踏む(同じモデルで採否が反転する)。"""
        hp = _bundle_file(tmp_path, high_doc, 'h.bundle.json')
        h, c = _codegen(tmp_path, hp, '--dense-flat-min-reduction', '0.999')
        assert '#define MDL_DENSE_FLAT 0' in h and 'lut_flat' not in c
        lp = _bundle_file(tmp_path, low_doc, 'l.bundle.json')
        h, c = _codegen(tmp_path, lp, '--dense-flat-min-reduction', '0.0')
        assert '#define MDL_DENSE_FLAT 1' in h and 'mdl_lut_flat' in c

    def test_on_and_off_modes(self, high_doc, low_doc, tmp_path):
        lp = _bundle_file(tmp_path, low_doc, 'l.bundle.json')
        h, c = _codegen(tmp_path, lp, '--dense-flat', 'on')
        assert '#define MDL_DENSE_FLAT 1' in h and 'mdl_lut_flat' in c
        hp = _bundle_file(tmp_path, high_doc, 'h.bundle.json')
        h, c = _codegen(tmp_path, hp, '--dense-flat', 'off')
        assert '#define MDL_DENSE_FLAT 0' in h and 'lut_flat' not in c
        assert 'off 指定' in h

    def test_bundle_file_is_not_modified(self, high_doc, tmp_path):
        """バンドル(IR が真実源)は C 生成で書き換わらない。"""
        path = _bundle_file(tmp_path, high_doc)
        before = open(path, 'rb').read()
        _codegen(tmp_path, path)
        assert open(path, 'rb').read() == before

    def test_default_emits_packed_entry_when_flat(self, high_doc, tmp_path):
        """平坦化を採用した dense には既定で 32 枚/語のパック入口が生える
        (lut_flat の表を共有するので追加の表は無い)。"""
        from griffith_logic.deploy.plan import DENSE_BATCH_AUTO

        path = _bundle_file(tmp_path, high_doc)
        h, c = _codegen(tmp_path, path)
        assert '#define MDL_DENSE_FLAT 1' in h
        assert f'#define MDL_DENSE_BATCH {DENSE_BATCH_AUTO}' in h
        assert 'mdl_lut_flat_packed_' in c
        # 0 を明示すれば生やさない
        h, _ = _codegen(tmp_path, path, '--dense-batch', '0')
        assert 'mdl_infer_batch_packed' not in h

    def test_default_has_no_packed_entry_without_flat(self, low_doc, tmp_path):
        """平坦化しない dense には既定では生やさない(パック経路はゲートごとの
        生の表 4B/ゲートを足すので、flash の小さい機器で黙って増やさない)。"""
        path = _bundle_file(tmp_path, low_doc)
        h, _ = _codegen(tmp_path, path)
        assert '#define MDL_DENSE_FLAT 0' in h
        assert 'mdl_infer_batch_packed' not in h

    def test_dense_batch_keeps_dense_flat(self, high_doc, tmp_path, capsys):
        """(f): --dense-batch でも dense-flat を使い、パック経路は lut_flat を
        32 枚/語で評価する(以前は lut_flat 非対応で off にしていた)。"""
        path = _bundle_file(tmp_path, high_doc)
        h, c = _codegen(tmp_path, path, '--dense-batch', '32')
        assert '#define MDL_DENSE_FLAT 1' in h
        assert 'mdl_lut_flat_packed_' in c and 'mdl_infer_batch_packed' in h
        # off を明示すれば従来どおり lut_layer のパック経路
        h, c = _codegen(tmp_path, path, '--dense-batch', '32',
                        '--dense-flat', 'off')
        assert '#define MDL_DENSE_FLAT 0' in h and 'lut_flat' not in c
        assert 'mdl_infer_batch_packed' in h

    def test_prebaked_lut_flat_is_not_flattened_again(self, high_doc, tmp_path):
        """bundle --dense-flat(焼き込み)済みなら codegen-c は再平坦化しない。"""
        src = _write(tmp_path, high_doc)
        out = tmp_path / 'baked.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none', '--dense-flat']) == 0
        h, c = _codegen(tmp_path, str(out))
        assert '#define MDL_DENSE_FLAT 1' in h and '焼き込み済み' in h
        assert 'mdl_lut_flat' in c

    def test_no_dense_chain_means_legacy_path(self, seg_doc, tmp_path):
        """conv 主体(2入力 lut_layer 列なし)は何も起きず、理由が出る。"""
        path = _bundle_file(tmp_path, seg_doc)
        h, c = _codegen(tmp_path, path)
        assert '#define MDL_DENSE_FLAT 0' in h and 'lut_flat' not in c

    def test_auto_failure_falls_back_to_legacy(self, high_doc, monkeypatch):
        """auto で平坦化の多重防御が不一致を検出したら、従来経路へ倒す
        (on なら例外のまま)。"""
        from griffith_logic.deploy import flatten_dense as fd

        real = fd.reduce_netlist

        def broken(netlist):
            res = real(netlist)
            kind, idx, pol = res.final_refs[0]
            res.final_refs[0] = (kind, idx, 1 - pol) if kind != 'c' else (
                'c', 1 - idx, 0)
            return res

        monkeypatch.setattr(fd, 'reduce_netlist', broken)
        bundle = build_bundle(copy.deepcopy(high_doc))
        out, dec = resolve_dense_flat(bundle, mode='auto')
        assert out is bundle and dec['applied'] is False
        assert '失敗' in dec['reason']
        with pytest.raises(ValueError, match='一致しない'):
            resolve_dense_flat(bundle, mode='on')

    def test_invalid_mode_raises(self, high_doc):
        with pytest.raises(ValueError, match='不正'):
            resolve_dense_flat(build_bundle(high_doc), mode='maybe')

    def test_describe_line_has_verdict_and_reduction(self, high_doc):
        _, dec = resolve_dense_flat(build_bundle(high_doc), mode='auto')
        line = describe_dense_flat(dec)
        assert '採用' in line and '削減率' in line and '→' in line

    def test_generated_c_without_decision_is_unchanged(self, dense_doc):
        """ライブラリの generate_c は dense_flat 未指定なら従来と同一
        (MDL_DENSE_FLAT マクロも出さない)。"""
        from griffith_logic.deploy.cbackend import generate_c

        files = generate_c(build_bundle(dense_doc), name='m', validate=False)
        assert 'MDL_DENSE_FLAT' not in files['m.h']

    def test_generate_c_dense_flat_param(self, high_doc):
        from griffith_logic.deploy.cbackend import generate_c

        files = generate_c(build_bundle(high_doc), name='m', validate=False,
                           dense_flat='auto')
        assert '#define MDL_DENSE_FLAT 1' in files['m.h']
        with pytest.raises(ValueError, match='同時'):
            generate_c(build_bundle(high_doc), name='m', validate=False,
                       dense_flat='auto', dense_flat_decision={'applied': True})


# ---------------------------------------------------------------------------
# verify-c(auto 経路)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(CC is None, reason='Cコンパイラが無い環境')
class TestVerifyCAuto:
    def test_verify_c_passes_on_flat_path(self, high_doc, tmp_path, capsys):
        """(e) auto で lut_flat 経路になる bundle でも verify-c が通る。
        ゴールデンは元バンドルIRの simulate なので lossless の実証になる。"""
        path = _bundle_file(tmp_path, high_doc)
        assert main(['deploy', 'verify-c', path, '-n', '6']) == 0
        out = capsys.readouterr().out
        assert 'dense-flat(auto): 採用' in out
        assert 'dense-flat(lut_flat)' in out

    def test_verify_c_passes_on_legacy_path(self, low_doc, tmp_path, capsys):
        path = _bundle_file(tmp_path, low_doc)
        assert main(['deploy', 'verify-c', path, '-n', '6']) == 0
        out = capsys.readouterr().out
        assert 'dense-flat(auto): 不採用' in out
        assert 'dense-flat(lut_flat)' not in out

    def test_verify_c_off_matches_too(self, high_doc, tmp_path, capsys):
        path = _bundle_file(tmp_path, high_doc)
        assert main(['deploy', 'verify-c', path, '--dense-flat', 'off']) == 0
        assert 'off 指定' in capsys.readouterr().out

    def test_verify_c_on_compressed_default_bundle(self, dense_doc, tmp_path):
        """既定の bundle(sweep 済み)→ 既定の verify-c の通し。"""
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        assert main(['deploy', 'verify-c', str(out), '-n', '5']) == 0

    def test_verify_golden_is_the_original_bundle_ir(self, high_doc,
                                                     monkeypatch):
        """verify はゴールデンを平坦化後の IR から作らない: C 生成用バンドルを
        壊しても(lut_flat の出力を反転)不一致を検出する。"""
        from griffith_logic.deploy import verify as vf

        real = vf.build_c_model

        def corrupting(bundle, work_dir, **kw):
            b = dict(bundle)
            ir = copy.deepcopy(bundle['ir'])
            for node in ir['nodes']:
                if node['type'] == 'lut_flat':
                    node['params']['outputs'][0] ^= 1      # 極性反転
            b['ir'] = ir
            return real(b, work_dir, **kw)

        monkeypatch.setattr(vf, 'build_c_model', corrupting)
        with pytest.raises(ValueError, match='一致しない'):
            vf.verify_c_matches_simulate(
                build_bundle(high_doc), n_samples=8, dense_flat='on',
                verbose=False)

    def test_verify_c_also_checks_batch_packed_entry(self, high_doc, tmp_path,
                                                     capsys):
        """平坦化を採用すると codegen-c と同じく mdl_infer_batch_packed が生え、
        verify-c はそれも(語境界を跨ぐ 33 枚で)simulate と照合する。"""
        from griffith_logic.deploy.verify import (
            PACKED_VERIFY_MIN_SAMPLES,
            verify_c_matches_simulate,
        )

        res = verify_c_matches_simulate(build_bundle(high_doc), n_samples=4,
                                        dense_flat='auto', verbose=False)
        assert res['dense_flat']['applied'] is True
        assert res['batch_packed_samples'] == PACKED_VERIFY_MIN_SAMPLES
        path = _bundle_file(tmp_path, high_doc)
        assert main(['deploy', 'verify-c', path, '-n', '3']) == 0
        assert 'mdl_infer_batch_packed 33枚' in capsys.readouterr().out

    def test_verify_c_batch_packed_follows_dense_batch(self, high_doc, low_doc):
        """--dense-batch 0 で入口を作らなければ照合もしない。平坦化しないときも
        明示すれば(従来カーネルの)パック入口を照合する。"""
        from griffith_logic.deploy.verify import verify_c_matches_simulate

        res = verify_c_matches_simulate(build_bundle(high_doc), n_samples=2,
                                        dense_flat='auto', dense_batch=0,
                                        verbose=False)
        assert res['batch_packed_samples'] == 0
        res = verify_c_matches_simulate(build_bundle(low_doc), n_samples=2,
                                        dense_flat='auto', verbose=False)
        assert res['dense_flat']['applied'] is False
        assert res['batch_packed_samples'] == 0          # codegen-c の既定と同じ
        res = verify_c_matches_simulate(build_bundle(low_doc), n_samples=40,
                                        dense_flat='off', dense_batch=16,
                                        verbose=False)
        assert res['batch_packed_samples'] == 40         # 16 枚ごとの区切りも跨ぐ

    def test_verify_c_detects_broken_batch_packed_entry(self, high_doc,
                                                        monkeypatch):
        """パック入口だけが壊れていても検出する(1 枚ずつの mdl_infer は正しいまま)。"""
        from griffith_logic.deploy import verify as vf

        broken = vf._PACKED_DRIVER.replace(
            '    uint32_t v = (uint32_t)sc[i];',
            '    uint32_t v = (uint32_t)(sc[i] + (i == 5 * MDL_OUTPUT_WIDTH));')
        assert broken != vf._PACKED_DRIVER
        monkeypatch.setattr(vf, '_PACKED_DRIVER', broken)
        with pytest.raises(ValueError, match='mdl_infer_batch_packed'):
            vf.verify_c_matches_simulate(build_bundle(high_doc), n_samples=4,
                                         dense_flat='auto', verbose=False)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
class TestInfo:
    def test_info_prints_compress_and_dense_flat_outlook(self, high_doc,
                                                         tmp_path, capsys):
        src = _write(tmp_path, high_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        capsys.readouterr()
        assert main(['deploy', 'info', str(out)]) == 0
        text = capsys.readouterr().out
        assert '圧縮(compress=fast): 適用 sweep' in text
        assert 'codegen-c 既定の見込み: dense-flat(auto): ' in text

    def test_info_for_conv_shows_same_format(self, seg_doc, tmp_path, capsys):
        """conv も dense と同じ形式(パス名 前→後(削減率))で表示し、
        対象外の sweep を「スキップ」と出さない。"""
        src = _write(tmp_path, seg_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0
        capsys.readouterr()
        assert main(['deploy', 'info', str(out)]) == 0
        text = capsys.readouterr().out
        assert '圧縮(compress=fast): 適用 conv命令列化 ' in text
        assert '常時)、うちconv再合成 ' in text and 'スキップ' not in text

    def test_info_without_compress_has_no_compress_line(self, dense_doc,
                                                        tmp_path, capsys):
        src = _write(tmp_path, dense_doc)
        out = tmp_path / 'm.bundle.json'
        assert main(['deploy', 'bundle', str(src), '-o', str(out),
                     '--compress', 'none']) == 0
        capsys.readouterr()
        assert main(['deploy', 'info', str(out)]) == 0
        assert '圧縮(compress' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# XOR(examples/06_deploy_c)を既定コマンドのまま通す
# ---------------------------------------------------------------------------
@pytest.mark.skipif(CC is None, reason='Cコンパイラが無い環境')
class TestXorExample:
    def test_default_commands_end_to_end(self, tmp_path, capsys):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from griffith_logic import GroupSum, LutLayer
        from griffith_logic.export import export_ir

        torch.manual_seed(0)
        xs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        ys = torch.tensor([0, 1, 1, 0])
        model = nn.Sequential(LutLayer(2, 16, n_inputs=2),
                              LutLayer(16, 16, n_inputs=2),
                              GroupSum(k=2, tau=2.0))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            F.cross_entropy(model(xs), ys).backward()
            opt.step()
        model.eval()
        ir = export_ir(model, input_width=2, name='xor')
        doc = {'ir': ir_to_dict(ir), 'preproc': {}, 'labels': []}
        src = _write(tmp_path, doc, 'xor_model.json')
        bundle_path = tmp_path / 'xor.bundle.json'
        # 既定のコマンド(フラグなし)だけで bundle → codegen-c → verify-c
        assert main(['deploy', 'bundle', str(src), '-o', str(bundle_path)]) == 0
        assert main(['deploy', 'codegen-c', str(bundle_path),
                     '-o', str(tmp_path / 'xor_c')]) == 0
        assert main(['deploy', 'verify-c', str(bundle_path)]) == 0
        bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
        comp = bundle['about']['compress']
        assert comp['applied'] == ['sweep']
        before = sum(int(n['params']['out_width']) for n in doc['ir']['nodes']
                     if n['type'] == 'lut_layer')
        after = sum(int(n['params']['out_width']) for n in bundle['ir']['nodes']
                    if n['type'] == 'lut_layer')
        assert after < before                            # 32 → 22
        h = (tmp_path / 'xor_c' / 'model.h').read_text(encoding='utf-8')
        assert '#define MDL_DENSE_FLAT 1' in h
        # 元 IR の simulate と bundle IR の simulate が全入力で一致
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        assert np.array_equal(simulate(ir, x),
                              simulate(ir_from_dict(bundle['ir']), x))
        capsys.readouterr()
