"""Webバックエンドの登録とパッケージング。

Webは他の3バックエンド(C/CUDA/Metal)と違い、Pythonがコードを生成する
わけではない — 実行体は `runtime/web`(TypeScript)側にあり、ここが
やるのは (1) 「どのop種別を実行できるか」の申告と (2) バンドルを配布用
JSONとしてディレクトリへ並べるパッケージングだけ。したがってこの
ウェーブでは node/npm を一切要求しない(runtime/web/dist は
RUNTIME_WEB をmonkeypatchしたダミーで代替する)。
"""

import gzip
import json
import shutil
from pathlib import Path

import pytest

from griffith_logic.deploy import build_bundle, save_bundle
from griffith_logic.deploy.backends import BACKENDS, can_run, capabilities
from griffith_logic.deploy.bundle import validate_bundle
from griffith_logic.deploy.webbackend import generate_web
from griffith_logic.deploy.webverify import RUNTIME_WEB, verify_web_matches_simulate

NODE = shutil.which('node')
WEB_DIST_OK = (RUNTIME_WEB / 'dist' / 'verify.mjs').is_file()
_SKIP_REASON = (
    'runtime/web/dist/verify.mjs が無い環境(先に `make web-runtime`'
    '= npm --prefix tools/webbench run build:runtime でビルドすること)'
)
skip_no_web_runtime = pytest.mark.skipif(
    NODE is None or not WEB_DIST_OK, reason=_SKIP_REASON)


class TestRegistration:
    def test_web_is_registered(self):
        assert 'web' in BACKENDS

    def test_capabilities_match_declared_ops(self):
        cap = capabilities('web')
        assert cap['ops'] == frozenset({
            'pack', 'unpack', 'gate_program', 'plane_op', 'reduce_scores',
            'lut_dense', 'copy_scores', 'to_scores', 'lut_flat',
        })
        assert cap['sub_ops'] == frozenset({
            'or_pool2x2', 'upsample2d_x2', 'permute_channels',
            'concat_planes', 'spatial_group_sum', 'group_sum',
        })

    def test_can_run_seg_model(self, seg_doc):
        ok, reason = can_run(build_bundle(seg_doc), 'web')
        assert (ok, reason) == (True, '')

    def test_can_run_dense_model(self, dense_doc):
        ok, reason = can_run(build_bundle(dense_doc), 'web')
        assert (ok, reason) == (True, '')


class TestGenerateWeb:
    def test_default_output_is_a_valid_bundle(self, seg_doc):
        bundle = build_bundle(seg_doc)
        files = generate_web(bundle, name='m')
        assert set(files) == {'m.bundle.json'}
        parsed = json.loads(files['m.bundle.json'])
        # load_bundle相当の検証をそのまま通す(=正規のバンドルであること)
        warnings = validate_bundle(parsed)
        assert warnings == [] or all(isinstance(w, str) for w in warnings)
        assert parsed['bundle_format'] == bundle['bundle_format']
        # 通常のjsonラウンドトリップと同じ結果であること(int寄りのキーが
        # 文字列化される等の差異は正規化してから比べる)。
        normalized = json.loads(json.dumps(bundle, ensure_ascii=False))
        assert parsed == normalized

    def test_compact_json_matches_save_bundle_convention(self, dense_doc):
        bundle = build_bundle(dense_doc)
        files = generate_web(bundle, name='m')
        payload = files['m.bundle.json']
        assert '\n' not in payload
        assert ', ' not in payload and ': ' not in payload

    def test_slim_output_drops_bundle_only_keys(self, seg_doc):
        bundle = build_bundle(seg_doc)
        files = generate_web(bundle, name='m', slim=True)
        assert set(files) == {'m.json'}
        parsed = json.loads(files['m.json'])
        for key in ('bundle_format', 'bundle_version', 'programs',
                    'preproc_desc'):
            assert key not in parsed, key
        assert 'ir' in parsed
        assert 'preproc' in parsed
        # slim出力はもはや正規のバンドルではない(必須キーが欠ける)
        with pytest.raises(ValueError):
            validate_bundle(parsed)


class TestCodegenWebCli:
    def _fake_dist(self, tmp_path):
        dist = tmp_path / 'fakeruntime' / 'dist'
        (dist / 'assets').mkdir(parents=True)
        (dist / 'index.html').write_text('<html></html>', encoding='utf-8')
        (dist / 'assets' / 'engine.js').write_text('/* wasm glue */',
                                                    encoding='utf-8')
        return dist.parent

    def test_packages_bundle_and_copies_runtime(self, monkeypatch, tmp_path,
                                                 seg_doc):
        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle, webbackend

        runtime_root = self._fake_dist(tmp_path)
        monkeypatch.setattr(webbackend, 'RUNTIME_WEB', runtime_root)

        bundle_path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(seg_doc), bundle_path)
        out_dir = tmp_path / 'out'
        rc = main(['deploy', 'codegen-web', str(bundle_path),
                   '-o', str(out_dir), '--name', 'model'])
        assert rc == 0
        assert (out_dir / 'model.bundle.json').exists()
        assert (out_dir / 'index.html').read_text(encoding='utf-8') == \
            '<html></html>'
        assert (out_dir / 'assets' / 'engine.js').exists()
        parsed = json.loads((out_dir / 'model.bundle.json').read_text(
            encoding='utf-8'))
        assert parsed['bundle_format'] == 'magicdelta-bundle'

    def test_no_runtime_skips_copy(self, monkeypatch, tmp_path, dense_doc):
        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle, webbackend

        runtime_root = self._fake_dist(tmp_path)
        monkeypatch.setattr(webbackend, 'RUNTIME_WEB', runtime_root)

        bundle_path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(dense_doc), bundle_path)
        out_dir = tmp_path / 'out'
        rc = main(['deploy', 'codegen-web', str(bundle_path),
                   '-o', str(out_dir), '--no-runtime'])
        assert rc == 0
        assert (out_dir / 'model.bundle.json').exists()
        assert not (out_dir / 'index.html').exists()

    def test_missing_dist_raises_actionable_error(self, monkeypatch, tmp_path,
                                                   dense_doc):
        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle, webbackend

        monkeypatch.setattr(webbackend, 'RUNTIME_WEB',
                            tmp_path / 'nonexistent_runtime_web')

        bundle_path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(dense_doc), bundle_path)
        out_dir = tmp_path / 'out'
        with pytest.raises(SystemExit, match='make web-runtime'):
            main(['deploy', 'codegen-web', str(bundle_path),
                 '-o', str(out_dir)])

    def test_gzip_writes_gz_alongside(self, monkeypatch, tmp_path, dense_doc):
        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle, webbackend

        runtime_root = self._fake_dist(tmp_path)
        monkeypatch.setattr(webbackend, 'RUNTIME_WEB', runtime_root)

        bundle_path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(dense_doc), bundle_path)
        out_dir = tmp_path / 'out'
        rc = main(['deploy', 'codegen-web', str(bundle_path),
                   '-o', str(out_dir), '--gzip'])
        assert rc == 0
        gz = out_dir / 'model.bundle.json.gz'
        assert gz.exists()
        raw = gzip.decompress(gz.read_bytes())
        payload = (out_dir / 'model.bundle.json').read_text(encoding='utf-8')
        assert raw.decode('utf-8') == payload


@skip_no_web_runtime
class TestVerifyWebPasses:
    """Webランタイム(Node上のverify.mjs) ⇔ ir/simulate の等価性ガード(K36-Py)。"""

    def test_seg_model_passes_bits_and_image(self, seg_doc, tmp_path):
        result = verify_web_matches_simulate(
            build_bundle(seg_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['mode'] == 'bits+image', '画像preprocを持つのに画像経路が回っていない'
        assert result['n_samples'] == 2
        assert result['max_abs_diff'] == 0

    def test_dense_model_uses_bits_only(self, dense_doc, tmp_path):
        result = verify_web_matches_simulate(
            build_bundle(dense_doc), n_samples=2, work_dir=tmp_path,
            verbose=False)
        assert result['mode'] == 'bits'
        assert result['max_abs_diff'] == 0


@skip_no_web_runtime
class TestVerifyWebFaultInjection:
    """ガード自体が仕事をしているか(=壊れたモデルを通さないか)を故障注入で確かめる。

    tests/deploy/test_verify_c.py::TestFaultInjection::test_detects_stale_programs
    と同じ狙い: IRだけ更新して programs を作り直し忘れた状態を検出できるか。
    これは同時に「webランタイムはIRを組み直さず programs(命令列)を実行して
    いる」ことも pin する(そうでなければ古いprogramsが無視され通ってしまう)。
    """

    def test_detects_stale_programs(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            for tt in node['params']['truth_tables']
        ]
        # programs は古いまま(=web は古い論理、simulate は新しい論理)
        with pytest.raises(ValueError, match='一致しない'):
            verify_web_matches_simulate(
                bundle, n_samples=1, work_dir=tmp_path, verbose=False)


@skip_no_web_runtime
class TestVerifyWebCli:
    def test_verify_web_command(self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(seg_doc), path)
        assert main(['deploy', 'verify-web', str(path), '-n', '1']) == 0
        assert '一致' in capsys.readouterr().out


def test_verify_web_help_renders():
    """--help はCLIパーサだけの経路なので node/dist の有無に関わらず動く。"""
    from griffith_logic.cli import main

    with pytest.raises(SystemExit) as exc:
        main(['deploy', 'verify-web', '--help'])
    assert exc.value.code == 0


class TestBundleResynthCli:
    """`griffith deploy bundle --resynth lossless` のCLI組み込み。

    seg_doc は tree_depth=2(木あたり最大3ゲート)の小さな合成IRなので、
    lossless再合成の分枝限定探索は一瞬で終わる(実測1秒未満)。実モデルは
    分オーダーになりうる(検証込みの探索コスト)。
    """

    def test_resynth_lossless_produces_strict_clean_bundle(
            self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'model.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out),
                   '--resynth', 'lossless', '--resynth-max-cost', '5',
                   '--resynth-verify-samples', '8'])
        assert rc == 0

        bundle = json.loads(out.read_text(encoding='utf-8'))
        warnings = validate_bundle(bundle, strict=True)
        assert warnings == []

        prov = bundle['about']['resynth']
        assert prov['mode'] == 'lossless'
        assert prov['max_cost'] == 5
        assert prov['gates_after'] <= prov['gates_before']

        out_text = capsys.readouterr().out
        assert 'resynth' in out_text
        assert '→' in out_text

    def test_resynth_default_flags(self, seg_doc, tmp_path):
        """--resynth-max-cost / --resynth-verify-samples 省略時の既定値。"""
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'model.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out),
                   '--resynth', 'lossless'])
        assert rc == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert bundle['about']['resynth']['max_cost'] == 5

    def test_no_resynth_flag_leaves_about_untouched(self, seg_doc, tmp_path):
        """--compress none なら resynth provenance が付かない(現在は既定の
        fast は conv にも再合成をかけるので、無変換は none で明示する)。"""
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'model.bundle.json'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out),
                   '--compress', 'none'])
        assert rc == 0
        bundle = json.loads(out.read_text(encoding='utf-8'))
        assert 'resynth' not in bundle.get('about', {})


def _real_model_paths() -> list[Path]:
    models_dir = (Path(__file__).resolve().parents[2] / 'tools' / 'webbench'
                  / 'public' / 'models')
    if not models_dir.is_dir():
        return []
    out = []
    for path in sorted(models_dir.glob('*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and 'ir' in doc:
            out.append(path)
    return out


_REAL_MODEL_PATHS = _real_model_paths()
_SMALLEST_REAL_MODEL = (
    min(_REAL_MODEL_PATHS, key=lambda p: p.stat().st_size)
    if _REAL_MODEL_PATHS else None
)


@pytest.mark.slow
@skip_no_web_runtime
@pytest.mark.skipif(
    _SMALLEST_REAL_MODEL is None,
    reason='配信モデル(tools/webbench/public/models)が無い環境',
)
class TestVerifyWebRealModel:
    """配信中の実モデル(最小サイズのもの)で通しの鎖が壊れていないことの確認。"""

    def test_smallest_real_model_passes(self, tmp_path):
        doc = json.loads(_SMALLEST_REAL_MODEL.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)
        result = verify_web_matches_simulate(
            bundle, n_samples=2, work_dir=tmp_path, verbose=False)
        assert result['max_abs_diff'] == 0
