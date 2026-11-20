"""配信中の実モデルでバンドル化が通ることの確認(K1)。

`tools/webbench/public/models/*.json` は gitignore 下の学習成果物なので、
無い環境では自動skipする(tests/l1_math/test_optimize.py と同じ方針)。

ここで見るのは「実モデルの形でも壊れないこと」だけで、数値そのものの
end-to-end 一致は K5(`griffith deploy verify-c`)が担当する。
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle, validate_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.memplan import check_no_overlap, plan_memory
from griffith_logic.deploy.plan import build_plan
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_codegen_e2e import DRIVER, run_generated

MODELS_DIR = Path(__file__).resolve().parents[2] / 'tools' / 'webbench' / 'public' / 'models'


def _model_paths() -> list[Path]:
    if not MODELS_DIR.is_dir():
        return []
    out = []
    for path in sorted(MODELS_DIR.glob('*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and 'ir' in doc:
            out.append(path)
    return out


MODEL_PATHS = _model_paths()

pytestmark = pytest.mark.skipif(
    not MODEL_PATHS,
    reason='配信モデル(tools/webbench/public/models)が無い環境',
)


@pytest.mark.slow
@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_real_model_bundles_and_validates(path):
    doc = json.loads(path.read_text(encoding='utf-8'))
    bundle = build_bundle(doc)          # 命令列の全数等価性検証込み
    assert validate_bundle(bundle, strict=True) == []

    programs = bundle['programs']
    if programs:
        before = sum(p['stats']['gates_before'] for p in programs.values())
        after = sum(p['stats']['gates_after'] for p in programs.values())
        assert 0 < after <= before or after == 0

    plan = bundle.get('preproc_desc')
    if plan is not None:
        # 前処理の出力ビット数とIR入力幅の一致は build_bundle が保証するが、
        # ここでも読める形で書いておく(壊れた時の原因特定を早くするため)
        ir_width = int(doc['ir']['inputs'][0]['width'])
        assert plan['channels'] * plan['height'] * plan['width'] == ir_width


# 数値一致まで見るモデル(simulateがconv全展開で重いので小さいものを選ぶ)。
_SMALL = {'demo_seg', 'har_dlgn_k600_gumbel'}
SMALL_PATHS = [p for p in MODEL_PATHS if p.stem in _SMALL]


@pytest.mark.slow
@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_real_model_generates_buildable_c(cc, tmp_path, path):
    """実モデルからCを生成し、依存ゼロでビルドが通る。"""
    doc = json.loads(path.read_text(encoding='utf-8'))
    bundle = build_bundle(doc)
    files = generate_c(bundle, name='model', validate=False)
    work = tmp_path / path.stem
    work.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (work / filename).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(DRIVER, encoding='utf-8')
    build = subprocess.run(
        [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
         f'-I{work}', f'-I{RUNTIME_C / "include"}',
         str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
         str(work / 'driver.c'), '-o', str(work / 'app')],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr


@pytest.mark.slow
@pytest.mark.skipif(not SMALL_PATHS, reason='小さい実モデルが無い環境')
@pytest.mark.parametrize('path', SMALL_PATHS, ids=lambda p: p.stem)
def test_real_model_c_matches_simulate(cc, tmp_path, path):
    """実モデルのC実行が ir/simulate.py と整数厳密一致する。"""
    doc = json.loads(path.read_text(encoding='utf-8'))
    bundle = build_bundle(doc)
    graph = ir_from_dict(bundle['ir'])
    in_width = int(graph.inputs[0].width)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
    want = simulate(graph, x)[0].astype(np.int64)
    got = run_generated(cc, tmp_path, bundle, x[0])
    assert np.array_equal(got, want)


@pytest.mark.slow
@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_memory_plan_reuses_buffers(path):
    """実モデルで生存区間解析が実際に作業RAMを削っている。

    実測(`model.o` の `__bss`):

        seg_convdlgn_k32   488,208 B → 242,848 B(49.7%)
        det_face_lut_k256  当時の値は下の記録を参照

    閾値は「再利用しない場合の8割以下」。実モデルは形がまちまちなので
    緩めの線にしてあるが、**再利用が壊れたら必ず落ちる**水準にはしてある。
    実装に合わせてここを緩めないこと。
    """
    doc = json.loads(path.read_text(encoding='utf-8'))
    bundle = build_bundle(doc)
    plan = build_plan(bundle)
    mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
    check_no_overlap(plan, mem)
    if len(plan.ops) < 3:
        pytest.skip('opが少なすぎて再利用の余地が無いモデル')
    assert mem.ratio() <= 0.80, (
        f'{path.stem}: 再利用が効いていない '
        f'{mem.naive_words * 4:,}B → {mem.arena_words * 4:,}B '
        f'({mem.ratio():.0%})')
