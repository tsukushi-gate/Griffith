"""C codegen の端到端検証(K3)。

バンドル → Cソース生成 → cc → 実行 の結果が `ir/simulate.py` と
**整数厳密**に一致することを確かめる。全ノード型(conv2d / or_pool2d /
upsample2d / permute / concat / group_sum / lut_layer)を踏むグラフで回す。

未対応の形は黙って別経路へ落ちず例外になること(=検証していないコードを
出荷しないこと)も併せて固定する。
"""

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cbackend import UnsupportedNode, generate_c
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32

DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t input[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  int32_t i;
  if (fread(input, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) {
    fprintf(stderr, "input short read\\n");
    return 1;
  }
  if (mdl_infer(input, scores) != 0) return 2;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)scores[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
  return 0;
}
'''


def run_generated(cc, tmp_path, bundle, x_bits: np.ndarray, *,
                  fuse: bool = False) -> np.ndarray:
    """バンドルからCを生成・ビルドし、与えたビット列で推論する。"""
    import subprocess

    files = generate_c(bundle, name='model', validate=False, fuse=fuse)
    work = tmp_path / 'gen'
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(DRIVER, encoding='utf-8')

    exe = work / 'app'
    cmd = [
        cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
        f'-I{work}', f'-I{RUNTIME_C / "include"}',
        str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
        str(work / 'driver.c'), '-o', str(exe),
    ]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        raise AssertionError(f'生成Cのビルドに失敗:\n{build.stderr}')
    proc = subprocess.run([str(exe)], input=x_bits.astype(np.uint8).tobytes(),
                          capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(
            f'実行失敗 (exit={proc.returncode}): '
            f'{proc.stderr.decode(errors="replace")}')
    return read_u32(proc.stdout).astype(np.int64)


def doc_of(graph: IRGraph, webbench, **kw) -> dict:
    return webbench.build_model_doc(
        graph, preproc=kw.pop('preproc', {}),
        labels=kw.pop('labels', []), **kw)


class TestSegLikeGraph:
    """conv2d / or_pool2d / upsample2d / concat / SpatialGroupSum を踏む。"""

    def test_matches_simulate(self, cc, tmp_path, seg_doc):
        bundle = build_bundle(seg_doc)
        graph = IRGraph  # noqa: F841  (型参照を避けるためのダミー)
        from griffith_logic.ir.serialize import ir_from_dict
        g = ir_from_dict(bundle['ir'])
        in_width = int(g.inputs[0].width)

        rng = np.random.default_rng(0)
        for trial in range(3):
            x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
            want = simulate(g, x)[0].astype(np.int64)
            got = run_generated(cc, tmp_path / f't{trial}', bundle, x[0])
            assert np.array_equal(got, want), f'trial={trial}'

    def test_extreme_inputs(self, cc, tmp_path, seg_doc):
        """全0 / 全1 の入力でも一致する(端数ビットの扱いが効く)。"""
        bundle = build_bundle(seg_doc)
        from griffith_logic.ir.serialize import ir_from_dict
        g = ir_from_dict(bundle['ir'])
        in_width = int(g.inputs[0].width)
        for label, x in (('zeros', np.zeros(in_width, dtype=np.uint8)),
                         ('ones', np.ones(in_width, dtype=np.uint8))):
            want = simulate(g, x[None, :])[0].astype(np.int64)
            got = run_generated(cc, tmp_path / label, bundle, x)
            assert np.array_equal(got, want), label


class TestDenseGraph:
    """lut_layer + group_sum だけのモデル(HAR型)。パック経路を通らない。"""

    def test_matches_simulate(self, cc, tmp_path, dense_doc):
        bundle = build_bundle(dense_doc)
        from griffith_logic.ir.serialize import ir_from_dict
        g = ir_from_dict(bundle['ir'])
        in_width = int(g.inputs[0].width)
        rng = np.random.default_rng(4)
        x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want = simulate(g, x)[0].astype(np.int64)
        got = run_generated(cc, tmp_path, bundle, x[0])
        assert np.array_equal(got, want)


class TestGeneratedInterface:
    def test_header_exposes_sizes_and_labels(self, seg_doc):
        bundle = build_bundle(seg_doc)
        files = generate_c(bundle, name='model', validate=False)
        header = files['model.h']
        from griffith_logic.ir.serialize import ir_from_dict
        g = ir_from_dict(bundle['ir'])
        assert f'#define MDL_INPUT_WIDTH {int(g.inputs[0].width)}' in header
        assert f'#define MDL_OUTPUT_WIDTH {int(g.outputs[0].width)}' in header
        assert f"#define MDL_NUM_LABELS {len(bundle['labels'])}" in header
        assert 'mdl_labels' in header

    def test_no_heap_allocation_in_generated_source(self, seg_doc):
        """実行時のヒープ確保ゼロが契約(バッファは全て static)。"""
        files = generate_c(build_bundle(seg_doc), name='model', validate=False)
        source = files['model.c']
        for bad in ('malloc', 'calloc', 'realloc', 'free('):
            assert bad not in source, f'生成コードに {bad} がある'

    def test_preproc_geometry_in_header(self, seg_doc):
        bundle = build_bundle(seg_doc)
        header = generate_c(bundle, name='model', validate=False)['model.h']
        plan = bundle['preproc_desc']
        assert f"#define MDL_INPUT_CHANNELS {plan['channels']}" in header
        assert f"#define MDL_INPUT_HEIGHT {plan['height']}" in header


class TestUnsupportedForms:
    """未対応の形は黙って通さず、生成時に落とす。"""

    def _wrap(self, node: IRNode, in_width: int, out_width: int) -> dict:
        graph = IRGraph(
            metadata={'name': 'x'},
            inputs=[IRPort(id='x', width=in_width)],
            outputs=[IRPort(id='y', width=out_width)],
            nodes=[node],
            edges=[IREdge(src='x', dst=node.id), IREdge(src=node.id, dst='y')],
        )
        from griffith_logic.ir.serialize import ir_to_dict
        return {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}

    def test_rejects_non_2x2_pool(self):
        node = IRNode(id='p', type='or_pool2d', params={
            'in_shape': [1, 8, 8], 'kernel_size': 3, 'stride': 3, 'padding': 0,
        })
        doc = self._wrap(node, 64, 4)
        with pytest.raises(UnsupportedNode, match='or_pool2d'):
            generate_c(build_bundle(doc), validate=False)

    def test_rejects_scale3_upsample(self):
        node = IRNode(id='u', type='upsample2d', params={
            'in_shape': [1, 4, 4], 'scale': 3,
        })
        doc = self._wrap(node, 16, 144)
        with pytest.raises(UnsupportedNode, match='upsample2d'):
            generate_c(build_bundle(doc), validate=False)

    def test_rejects_spatial_crossing_permute(self):
        node = IRNode(id='pm', type='permute', params={
            'in_shape': [2, 3, 4, 5], 'perm': [0, 1, 3, 2],
        })
        doc = self._wrap(node, 120, 120)
        with pytest.raises(UnsupportedNode, match='permute'):
            generate_c(build_bundle(doc), validate=False)

    def test_rejects_unknown_node_type(self):
        node = IRNode(id='bb', type='blackbox', params={
            'torch_class': 'Foo', 'config': {}, 'in_width': 8, 'out_width': 4,
        })
        doc = self._wrap(node, 8, 4)
        with pytest.raises(UnsupportedNode, match='blackbox'):
            generate_c(build_bundle(doc), validate=False)

    def test_rejects_high_arity_lut(self):
        node = IRNode(id='l', type='lut_layer', params={
            'in_width': 16, 'out_width': 4, 'n_inputs': 6,
            'indices': [[0] * 4 for _ in range(6)],
            'truth_tables': ['0x1'] * 4,
        })
        doc = self._wrap(node, 16, 4)
        with pytest.raises(UnsupportedNode, match='n_inputs'):
            generate_c(build_bundle(doc), validate=False)


class TestCli:
    def test_codegen_c_writes_buildable_tree(self, cc, tmp_path, seg_doc):
        import json
        import subprocess

        from griffith_logic.cli import main
        from griffith_logic.deploy import save_bundle

        bundle_path = tmp_path / 'm.bundle.json'
        save_bundle(build_bundle(seg_doc), bundle_path)
        out_dir = tmp_path / 'out'
        assert main(['deploy', 'codegen-c', str(bundle_path),
                     '-o', str(out_dir), '--name', 'model']) == 0
        for name in ('model.h', 'model.c', 'mdl_kernels.h', 'mdl_simd.h',
                     'mdl_kernels.c', 'mdl_stdio.h'):
            assert (out_dir / name).exists(), name

        # コピーされたランタイムだけでビルドが通る(外部依存が無いことの確認)
        (out_dir / 'driver.c').write_text(DRIVER, encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{out_dir}', str(out_dir / 'model.c'),
             str(out_dir / 'mdl_kernels.c'), str(out_dir / 'driver.c'),
             '-o', str(out_dir / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        assert json.loads(bundle_path.read_text())['bundle_version']


@pytest.mark.slow
class TestFusedGateProgram:
    """命令列を直線展開した経路が、インタプリタ経路と**同じ値**を出す。

    融合は真理値表を生成時に確定させ、中間スロットをレジスタへ載せる
    最適化で、**数値の意味は1ビットも変えてはいけない**。ここが崩れると
    「速いが間違っている」コードを出荷することになる。
    """

    def test_fused_matches_interpreted_and_simulate(self, cc, tmp_path,
                                                    seg_doc):
        from griffith_logic.deploy import build_bundle as _bb

        bundle = _bb(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(515)
        x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)

        interp = run_generated(cc, tmp_path / 'i', bundle, x[0])
        fused = run_generated(cc, tmp_path / 'f', bundle, x[0], fuse=True)
        assert np.array_equal(interp, want), 'インタプリタ経路が simulate と不一致'
        assert np.array_equal(fused, want), '融合経路が simulate と不一致'
        assert np.array_equal(fused, interp), '融合とインタプリタで値が違う'

    def test_fused_source_has_no_truth_table_dispatch(self, seg_doc):
        """融合経路では真理値表の実行時分岐が消えている。"""
        from griffith_logic.deploy import build_bundle as _bb

        bundle = _bb(seg_doc)
        src = generate_c(bundle, name='m', validate=False, fuse=True)['m.c']
        assert '_fused(' in src, '融合関数が生成されていない'
        assert 'mdl_conv_exec' not in src, (
            '融合したのにインタプリタ呼び出しが残っている')
        assert 'mdl_conv_build_planes' in src, 'タップ平面の構築が抜けている'
