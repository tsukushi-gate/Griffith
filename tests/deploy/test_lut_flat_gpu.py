"""lut_flat の GPU 実行: 段分け(flatlevels)と CUDA/Metal 生成・実行。

等価性の鎖(ゴールデンは常に**元IR(lut_layer 列)の simulate**):

    元IR simulate ⇔ flatten_dense 後IR simulate(test_dense_flat.py で固定)
                  ⇔ levelize_lut_flat → eval_levelized(numpy、段ごと一括評価)
                  ⇔ Metal(融合版 / 段ごと版、batch=1 と batch>1)

CUDA は実機が無い環境では生成ソースの形だけを見る(実行検証は
`scripts/verify_cuda_remote.py`)。
"""

import subprocess

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.backends import can_run
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.flatlevels import eval_levelized, levelize_lut_flat
from griffith_logic.deploy.flatten_dense import flatten_dense_ir
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.ir.serialize import ir_to_dict
from griffith_logic.ir.simulate import simulate, simulate_trace

from .test_batch import METAL_BATCH_DRIVER, N_BATCH
from .test_dense_flat import _dense_ir, _mixed_ir, _random_layers
from .test_metal import METAL_RUNTIME, _has_metal


def _flat_graph(seed=11, widths=(64, 64, 48, 48), in_w=40, k=6, p_trivial=0.4):
    rng = np.random.default_rng(seed)
    g = _dense_ir(_random_layers(rng, in_w, list(widths), p_trivial=p_trivial),
                  in_width=in_w, k=k)
    new, _ = flatten_dense_ir(g, n_verify=4)
    return g, new


def _flat_node(graph):
    return next(n for n in graph.nodes if n.type == 'lut_flat')


def _bits(width, n, seed=0):
    return np.random.default_rng(seed).integers(
        0, 2, size=(n, width), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 段分け
# ---------------------------------------------------------------------------
class TestLevelize:
    def test_matches_lut_flat_simulate(self):
        _, flat = _flat_graph()
        node = _flat_node(flat)
        lf = levelize_lut_flat(node.params)
        x = _bits(lf.in_width, 64, seed=1)
        want = np.asarray(simulate_trace(flat, x)[node.id])
        assert np.array_equal(eval_levelized(lf, x), want)

    def test_levels_are_independent_and_sorted(self):
        """同じ段のゲートは同じ段の出力を読まない(並列評価できる)。"""
        _, flat = _flat_graph(seed=3)
        lf = levelize_lut_flat(_flat_node(flat).params)
        base = 1 + lf.n_in
        for lv in range(lf.n_levels):
            g0, g1 = lf.level_off[lv], lf.level_off[lv + 1]
            assert g1 > g0, '空の段がある'
            for g in range(g0, g1):
                assert lf.a[g] < base + g0 and lf.b[g] < base + g0

    def test_depth_does_not_exceed_original_layers(self):
        """段数 <= 置換前の lut_layer の層数(4)。"""
        _, flat = _flat_graph(seed=5)
        assert levelize_lut_flat(_flat_node(flat).params).n_levels <= 4

    def test_only_referenced_inputs_are_kept(self):
        params = {'in_width': 10, 'out_width': 2,
                  'gates': [[6, 3, 8]], 'outputs': [2 * 11, 2 * 5 + 1]}
        lf = levelize_lut_flat(params)
        assert lf.in_idx == [2, 4, 7]           # 入力 3,5,8 番(信号 3,5,8)
        x = _bits(10, 32, seed=2)
        ref = np.stack([x[:, 2] ^ x[:, 7], 1 - x[:, 4]], axis=1)
        assert np.array_equal(eval_levelized(lf, x), ref)

    def test_rejects_forward_reference(self):
        with pytest.raises(ValueError, match='自分以降'):
            levelize_lut_flat({'in_width': 2, 'out_width': 1,
                               'gates': [[6, 1, 3]], 'outputs': [6]})

    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(0, 2 ** 16), depth=st.integers(1, 5))
    def test_property_random_dense(self, seed, depth):
        rng = np.random.default_rng(seed)
        widths = [int(rng.integers(4, 24)) for _ in range(depth)]
        widths[-1] = 2 * max(1, widths[-1] // 2)
        g = _dense_ir(_random_layers(rng, 9, widths, p_trivial=0.5),
                      in_width=9, k=2)
        flat, rep = flatten_dense_ir(g, n_verify=0)
        if not rep['chains']:
            return
        node = _flat_node(flat)
        lf = levelize_lut_flat(node.params)
        x = ((np.arange(512)[:, None] >> np.arange(9)[None, :]) & 1).astype(np.uint8)
        assert np.array_equal(eval_levelized(lf, x),
                              np.asarray(simulate_trace(flat, x)[node.id]))


# ---------------------------------------------------------------------------
# バックエンド申告と生成
# ---------------------------------------------------------------------------
def _flat_bundle(graph):
    return build_bundle({'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []})


class TestGpuGeneration:
    def test_all_backends_can_run_lut_flat(self):
        _, flat = _flat_graph()
        bundle = _flat_bundle(flat)
        for backend in ('c', 'web', 'cuda', 'metal'):
            assert can_run(bundle, backend) == (True, ''), backend

    @pytest.mark.parametrize('gen,prefix', [(generate_metal, 'mdl_metal_'),
                                            (generate_cuda, 'mdl_cuda_')])
    def test_auto_picks_fused_when_signal_table_fits(self, gen, prefix):
        _, flat = _flat_graph()
        src = ''.join(gen(_flat_bundle(flat), batch=4).values())
        assert f'{prefix}lut_flat_fused(' in src
        assert f'{prefix}lut_flat_level(' not in src

    @pytest.mark.parametrize('gen,prefix', [(generate_metal, 'mdl_metal_'),
                                            (generate_cuda, 'mdl_cuda_')])
    def test_levels_mode_emits_one_launch_per_level(self, gen, prefix):
        _, flat = _flat_graph()
        lf = levelize_lut_flat(_flat_node(flat).params)
        src = ''.join(gen(_flat_bundle(flat), batch=4,
                          lut_flat_mode='levels').values())
        assert src.count(f'{prefix}lut_flat_level(') == lf.n_levels
        assert f'{prefix}lut_flat_init(' in src and f'{prefix}lut_flat_out(' in src

    def test_packed_mode_needs_batch_32(self):
        from griffith_logic.deploy.plan import UnsupportedNode

        _, flat = _flat_graph()
        with pytest.raises(UnsupportedNode, match='batch>=32'):
            generate_metal(_flat_bundle(flat), batch=8, lut_flat_mode='packed')
        src = ''.join(generate_cuda(_flat_bundle(flat), batch=64,
                                    lut_flat_mode='packed').values())
        assert 'mdl_cuda_lut_flat_pk_level(' in src

    def test_python_api_does_not_flatten(self):
        """Python API の generate_metal はバンドルをそのまま生成する
        (平坦化は CLI / verify の段で resolve_dense_flat(target='gpu') が行う。C と同じ流儀)。"""
        orig, _ = _flat_graph()
        src = ''.join(generate_metal(_flat_bundle(orig), batch=128).values())
        assert 'mdl_metal_lut_flat_' not in src.split('MDL_SHADER_SRC')[-1]

    @pytest.mark.parametrize('cmd,fn', [('codegen-metal', 'mdl_metal_lut_flat_'),
                                        ('codegen-cuda', 'mdl_cuda_lut_flat_')])
    @pytest.mark.parametrize('extra,expect', [
        ([], False),                                  # GPU の既定は off(opt-in)
        (['--dense-flat', 'off'], False),
        (['--dense-flat', 'auto'], True),             # 削減率が閾値以上なら採用
        (['--dense-flat', 'on'], True),
        (['--dense-flat', 'on', '--dense-packed', '--batch', '128'], False),  # 明示が勝つ
    ])
    def test_cli_flattens_only_when_opted_in(self, tmp_path, capsys, cmd, fn, extra,
                                             expect):
        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle

        orig, _ = _flat_graph()
        src = tmp_path / 'm.bundle.json'
        save_bundle(_flat_bundle(orig), src)
        out = tmp_path / 'out'
        assert main(['deploy', cmd, str(src), '-o', str(out), '--no-runtime',
                     *extra]) == 0
        assert 'dense-flat(' in capsys.readouterr().out
        text = ''.join(p.read_text(encoding='utf-8') for p in out.iterdir())
        body = text.split('MDL_SHADER_SRC')[-1]
        assert (f'{fn}fused(' in body or f'{fn}level(' in body) is expect

    def test_gpu_threshold(self):
        from griffith_logic.deploy.flatten_dense import resolve_dense_flat
        from griffith_logic.deploy.plan import DENSE_FLAT_AUTO_MIN_REDUCTION_GPU

        orig, _ = _flat_graph()
        _, dec = resolve_dense_flat(_flat_bundle(orig), mode='auto',
                                    n_verify=0, target='gpu')
        assert dec['min_reduction'] == DENSE_FLAT_AUTO_MIN_REDUCTION_GPU
        assert dec['applied'] is (dec['reduction'] >= DENSE_FLAT_AUTO_MIN_REDUCTION_GPU)

    def test_invalid_mode_raises(self):
        _, flat = _flat_graph()
        with pytest.raises(ValueError, match='lut_flat_mode'):
            generate_metal(_flat_bundle(flat), lut_flat_mode='magic')

    def test_fused_refuses_oversized_signal_table(self, monkeypatch):
        from griffith_logic.deploy.metalbackend import MetalBackend
        from griffith_logic.deploy.plan import UnsupportedNode

        _, flat = _flat_graph()
        monkeypatch.setattr(MetalBackend, 'LUT_FLAT_FUSED_MAX_BYTES', 16)
        with pytest.raises(UnsupportedNode, match='共有メモリ上限'):
            generate_metal(_flat_bundle(flat), lut_flat_mode='fused')
        # auto は段ごと版へ倒れる
        src = ''.join(generate_metal(_flat_bundle(flat)).values())
        assert 'mdl_metal_lut_flat_level(' in src


# ---------------------------------------------------------------------------
# Metal 実行(macOS のみ)
# ---------------------------------------------------------------------------
def _run_metal_batch(bundle, x, tmp_path, mode):
    n = x.shape[0]
    files = generate_metal(bundle, name='model_metal', batch=n,
                           lut_flat_mode=mode)
    work = tmp_path / mode
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.mm').write_text(
        METAL_BATCH_DRIVER.replace(f'#define N {N_BATCH}', f'#define N {n}'),
        encoding='utf-8')
    build = subprocess.run(
        ['clang++', '-std=c++17', '-O2', '-fobjc-arc', f'-I{METAL_RUNTIME}',
         '-framework', 'Metal', '-framework', 'Foundation',
         str(work / 'model_metal.mm'), str(METAL_RUNTIME / 'mdl_metal.mm'),
         str(work / 'driver.mm'), '-o', str(work / 'app')],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[:2000]
    proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                          capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode(errors='replace')[:500]
    return np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)


@pytest.mark.slow
@pytest.mark.skipif(not _has_metal(), reason='macOS + Metal のある環境でのみ実行')
class TestMetalLutFlat:
    @pytest.mark.parametrize('mode,n', [('fused', N_BATCH), ('levels', N_BATCH),
                                        ('packed', 70)])
    def test_batch_matches_original_simulate(self, tmp_path, mode, n):
        """packed は語境界(32/64)を跨ぐ 70 枚で見る。"""
        orig, flat = _flat_graph(seed=21)
        x = _bits(40, n, seed=7)
        want = simulate(orig, x).astype(np.int64).ravel()
        got = _run_metal_batch(_flat_bundle(flat), x, tmp_path, mode)
        assert np.array_equal(got, want)

    @pytest.mark.parametrize('mode,n', [('fused', N_BATCH), ('levels', N_BATCH),
                                        ('packed', 33)])
    def test_mixed_ir_with_constants_and_polarity(self, tmp_path, mode, n):
        orig = _mixed_ir()
        flat, _ = flatten_dense_ir(orig, n_verify=4)
        x = _bits(6, n, seed=3)
        want = simulate(orig, x).astype(np.int64).ravel()
        got = _run_metal_batch(_flat_bundle(flat), x, tmp_path, mode)
        assert np.array_equal(got, want)

    def test_verify_metal_default_flat_golden_is_original(self, tmp_path):
        """verify-metal で平坦化(dense_flat='on')して検証しても、ゴールデンは元IRの simulate。"""
        from griffith_logic.deploy.metalverify import verify_metal_matches_simulate

        orig, _ = _flat_graph(seed=8)
        res = verify_metal_matches_simulate(_flat_bundle(orig), n_samples=4,
                                            verbose=False, dense_flat='on')
        assert res['dense_flat']['applied'] is True and res['max_abs_diff'] == 0

    def test_verify_metal_single(self, tmp_path):
        from griffith_logic.deploy.metalverify import verify_metal_matches_simulate

        _, flat = _flat_graph(seed=4)
        res = verify_metal_matches_simulate(_flat_bundle(flat), n_samples=4,
                                            verbose=False)
        assert res['max_abs_diff'] == 0
