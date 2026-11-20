"""examples/30_deploy_esp32 のホスト検証(ESP-IDF不要)。

XORバンドルを一時生成 → gen_model.py(codegen-c + samples.h) → `cc` でホストビルド
→ 実行して全サンプルの分類・スコアが一致することを確かめる。実機(ESP32)側は
同じ main.c を `idf.py build` したもので、実機⇔ホストの整数一致は
`results/deploy/devices/` のログ(score_hash)で確認する。
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.bundle import save_bundle
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / 'examples' / '30_deploy_esp32'


def _load_gen_model():
    spec = importlib.util.spec_from_file_location(
        'esp32_gen_model', PROJECT / 'gen_model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['esp32_gen_model'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def xor_bundle_path(tmp_path_factory):
    """examples/06_deploy_c の手順でXORを学習し、バンドルを一時ディレクトリへ書く。"""
    work = tmp_path_factory.mktemp('esp32_xor')
    subprocess.run(
        [sys.executable, str(REPO / 'examples' / '06_deploy_c' / 'export_model.py')],
        cwd=work, check=True, capture_output=True)
    doc = json.loads((work / 'out_examples' / 'xor_model.json').read_text())
    path = work / 'xor.bundle.json'
    save_bundle(build_bundle(doc), path)
    return path


def _build_and_run(cc, model_dir: Path, tmp_path: Path) -> dict:
    exe = tmp_path / 'host_demo'
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-DHOST_BUILD',
           '-DMDL_NO_DEFAULT_CTX', f'-I{PROJECT / "main"}', f'-I{model_dir}',
           str(PROJECT / 'main' / 'main.c'), str(model_dir / 'model.c'),
           str(model_dir / 'mdl_kernels.c'), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.rstrip().endswith('DONE')
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
    assert len(lines) == 1
    return json.loads(lines[0][len('RESULT '):])


def test_builtin_xor_samples(cc, tmp_path, xor_bundle_path):
    """--samples 省略(内蔵XOR 4パターン): 全件正解。

    既定は dense-flat auto なので、XOR(32 ゲート → 実効 3 ゲート)は lut_flat で
    実行され、MODEL_GATES・ctx_bytes は平坦化後の値になる。従来経路(32/32)は
    `--dense-flat off`(下のテスト)。
    """
    gen = _load_gen_model()
    model_dir = tmp_path / 'model'
    assert gen.main(['--bundle', str(xor_bundle_path), '--out', str(model_dir)]) == 0
    assert (model_dir / 'model_info.json').exists()
    assert (model_dir / 'samples.h').exists()
    info = json.loads((model_dir / 'model_info.json').read_text())
    assert info['dense_flat']['applied'] is True
    assert '#define MDL_DENSE_FLAT 1' in (model_dir / 'model.h').read_text()

    r = _build_and_run(cc, model_dir, tmp_path)
    assert r['model'] == 'xor' and r['platform'] == 'host'
    assert r['n'] == 4 and r['n_labeled'] == 4 and r['correct'] == 4
    assert r['acc'] == 1.0
    assert r['input_width'] == 2 and r['output_width'] == 2
    assert r['gates'] == info['gates'] < 32
    assert r['ctx_bytes'] < 32


def test_builtin_xor_samples_dense_flat_off(cc, tmp_path, xor_bundle_path):
    """--dense-flat off: 従来経路(bit-slice)。MODEL_GATES=32・ctx_bytes=32。"""
    gen = _load_gen_model()
    model_dir = tmp_path / 'model'
    assert gen.main(['--bundle', str(xor_bundle_path), '--out', str(model_dir),
                     '--dense-flat', 'off']) == 0
    assert '#define MDL_DENSE_FLAT 0' in (model_dir / 'model.h').read_text()
    assert (model_dir / 'model_info.json').exists()
    assert (model_dir / 'samples.h').exists()

    r = _build_and_run(cc, model_dir, tmp_path)
    assert r['model'] == 'xor' and r['platform'] == 'host'
    assert r['n'] == 4 and r['n_labeled'] == 4 and r['correct'] == 4
    assert r['acc'] == 1.0
    assert r['input_width'] == 2 and r['output_width'] == 2
    assert r['ctx_bytes'] == 32 and r['gates'] == 32


def test_samples_json_with_expected_scores(cc, tmp_path, xor_bundle_path):
    """samples JSON(IR simulate 由来の expected_scores 付き): スコアが整数一致する。"""
    gen = _load_gen_model()
    bundle = json.loads(xor_bundle_path.read_text())
    graph = ir_from_dict(bundle['ir'])
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1]], dtype=np.int64)
    scores = simulate(graph, x)
    samples = [{
        'bits': np.packbits(row.astype(np.uint8), bitorder='little').tobytes().hex(),
        'label': int(np.argmax(sc)),
        'expected_scores': [int(v) for v in sc],
    } for row, sc in zip(x, scores)]
    samples_path = tmp_path / 'samples.json'
    samples_path.write_text(json.dumps({'input_width': 2, 'samples': samples}))

    model_dir = tmp_path / 'model'
    assert gen.main(['--bundle', str(xor_bundle_path), '--samples', str(samples_path),
                     '--name', 'xor_scores', '--out', str(model_dir)]) == 0
    r = _build_and_run(cc, model_dir, tmp_path)
    assert r['model'] == 'xor_scores'
    assert r['n'] == 6 and r['correct'] == 6
    assert r['score_checked'] == 6 and r['score_mismatch'] == 0
