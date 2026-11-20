"""配線・プーリング・読み出し系Cカーネルの検証。

いずれも `ir/simulate.py`(numpyのgolden model)を基準に整数厳密比較する。
対象は配信中の実モデルが実際に使う形だけ:

    or_pool2d  : kernel=2, stride=2, padding=0
    upsample2d : scale=2
    permute    : チャネル軸のみ / SpatialGroupSum型([0,2,3,1] + group_sum)
    concat     : チャネル軸
    lut_layer  : n_inputs <= 5
    group_sum  : 素の総和

実モデルに現れない形(汎用pool/upsample、add、binarize、constant)は
Cランタイムに載せていない。使われていない経路を無検証で持つより、
エクスポータ側で明示的に拒否する方針(K3で実装)。
"""

import numpy as np
import pytest

from griffith_logic.deploy.layout import tail_mask, words_per_row
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.simulate import simulate

from .conftest import c_i32_array, c_u32_array
from .test_c_kernels import (
    PRELUDE,
    pack_planes_ref,
    read_u32,
    unpack_planes_ref,
)


def read_i32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype='<i4')


def single_node_graph(node: IRNode, in_width: int, out_width: int) -> IRGraph:
    return IRGraph(
        metadata={'name': node.type},
        inputs=[IRPort(id='x', width=in_width)],
        outputs=[IRPort(id='y', width=out_width)],
        nodes=[node],
        edges=[IREdge(src='x', dst=node.id), IREdge(src=node.id, dst='y')],
    )


class TestOrPool2x2:
    @pytest.mark.parametrize('c,h,w', [
        (1, 10, 33),
        pytest.param(1, 4, 4, marks=pytest.mark.slow),
        pytest.param(3, 8, 8, marks=pytest.mark.slow),
        pytest.param(2, 6, 40, marks=pytest.mark.slow),
        pytest.param(2, 8, 66, marks=pytest.mark.slow),
    ])
    def test_matches_simulate(self, build_and_run, c, h, w):
        h_out, w_out = (h - 2) // 2 + 1, (w - 2) // 2 + 1
        node = IRNode(id='p', type='or_pool2d', params={
            'in_shape': [c, h, w], 'kernel_size': 2, 'stride': 2, 'padding': 0,
        })
        graph = single_node_graph(node, c * h * w, c * h_out * w_out)
        rng = np.random.default_rng(c * 1000 + h * 10 + w)
        x = rng.integers(0, 2, size=(1, c * h * w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.uint8)

        packed = pack_planes_ref(x.reshape(c, h, w))
        n_out_words = c * h_out * words_per_row(w_out)
        src = PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
static uint32_t out[{n_out_words}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_or_pool2x2(IN, out, {c}, {h}, {w});
  emit(out, {n_out_words});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        back = unpack_planes_ref(got, c, h_out, w_out).ravel()
        assert np.array_equal(back, want)

    def test_tail_bits_are_masked(self, build_and_run):
        """全1入力でも出力の端数ビットは0(不変条件の維持)。"""
        c, h, w = 1, 4, 10          # w_out=5 → 27ビットが端数
        w_out = (w - 2) // 2 + 1
        packed = pack_planes_ref(np.ones((c, h, w), dtype=np.uint8))
        n_out_words = c * ((h - 2) // 2 + 1) * words_per_row(w_out)
        src = PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
static uint32_t out[{n_out_words}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_or_pool2x2(IN, out, {c}, {h}, {w});
  emit(out, {n_out_words});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        assert all(int(v) == tail_mask(w_out) for v in got)


class TestUpsample2dX2:
    @pytest.mark.parametrize('c,h,w', [
        (3, 5, 17),
        pytest.param(1, 2, 2, marks=pytest.mark.slow),
        pytest.param(2, 4, 8, marks=pytest.mark.slow),
        pytest.param(2, 3, 33, marks=pytest.mark.slow),
    ])
    def test_matches_simulate(self, build_and_run, c, h, w):
        node = IRNode(id='u', type='upsample2d', params={
            'in_shape': [c, h, w], 'scale': 2,
        })
        graph = single_node_graph(node, c * h * w, c * 2 * h * 2 * w)
        rng = np.random.default_rng(c * 77 + h * 7 + w)
        x = rng.integers(0, 2, size=(1, c * h * w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.uint8)

        packed = pack_planes_ref(x.reshape(c, h, w))
        n_out_words = c * (2 * h) * words_per_row(2 * w)
        src = PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
static uint32_t out[{n_out_words}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_upsample2d_x2(IN, out, {c}, {h}, {w});
  emit(out, {n_out_words});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        back = unpack_planes_ref(got, c, 2 * h, 2 * w).ravel()
        assert np.array_equal(back, want)


class TestPermuteChannels:
    def test_matches_simulate_for_channel_swap(self, build_and_run):
        """実モデルの perm=[1,0,2,3](thermometer の rgb↔thr 並べ替え)。"""
        c0, c1, h, w = 3, 4, 6, 10   # in_shape=[c0,c1,H,W] を [c1,c0,H,W] へ
        c = c0 * c1
        node = IRNode(id='pm', type='permute', params={
            'in_shape': [c0, c1, h, w], 'perm': [1, 0, 2, 3],
        })
        graph = single_node_graph(node, c * h * w, c * h * w)
        rng = np.random.default_rng(5)
        x = rng.integers(0, 2, size=(1, c * h * w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.uint8)

        # チャネル順: 出力チャネル o=(i1*c0+i0) の中身は入力チャネル i0*c1+i1
        order = [i0 * c1 + i1 for i1 in range(c1) for i0 in range(c0)]
        packed = pack_planes_ref(x.reshape(c, h, w))
        wpr = words_per_row(w)
        src = PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
{c_i32_array('ORDER', order)}
static uint32_t out[{c * h * wpr}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_permute_channels(IN, out, ORDER, {c}, {h}, {wpr});
  emit(out, {c * h * wpr});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        back = unpack_planes_ref(got, c, h, w).ravel()
        assert np.array_equal(back, want)


class TestConcatPlanes:
    def test_matches_simulate(self, build_and_run):
        """slot 0 が出力の先頭に来る規約を simulate と突き合わせる。

        simulate は入力ポート1本のグラフしか受けないので、恒等 lut_layer で
        入力を3枚に切り出してから concat する形にする(実モデルの
        skip接続と同じ「複数ノード → concat」の構造になる)。
        """
        h, w = 5, 12
        widths = [2, 3, 1]
        total = sum(widths)
        plane = h * w
        in_width = total * plane

        nodes, edges = [], []
        offset = 0
        for i, n in enumerate(widths):
            out_w = n * plane
            # n_inputs=1 の恒等 LUT(tt bit0=f(0), bit1=f(1) → 0b10 = 0x2)
            nodes.append(IRNode(id=f's{i}', type='lut_layer', params={
                'in_width': in_width, 'out_width': out_w, 'n_inputs': 1,
                'indices': [[offset + j for j in range(out_w)]],
                'truth_tables': ['0x2'] * out_w,
            }))
            edges.append(IREdge(src='x', dst=f's{i}'))
            edges.append(IREdge(src=f's{i}', dst='cat', dst_slot=i))
            offset += out_w
        nodes.append(IRNode(id='cat', type='concat', params={
            'in_widths': [n * plane for n in widths],
        }))
        edges.append(IREdge(src='cat', dst='y'))
        graph = IRGraph(
            metadata={'name': 'concat'},
            inputs=[IRPort(id='x', width=in_width)],
            outputs=[IRPort(id='y', width=in_width)],
            nodes=nodes, edges=edges,
        )

        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.uint8)

        wpr = words_per_row(w)
        parts, offset = [], 0
        for n in widths:
            chunk = x[0, offset:offset + n * plane].reshape(n, h, w)
            parts.append(pack_planes_ref(chunk))
            offset += n * plane
        arrays = '\n'.join(
            c_u32_array(f'IN{i}', p.ravel()) for i, p in enumerate(parts))
        src = PRELUDE + f'''
{arrays}
{c_i32_array('CIN', widths)}
static const uint32_t *const SRCS[] = {{{",".join(f"IN{i}" for i in range(len(widths)))}}};
static uint32_t out[{total * h * wpr}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_concat_planes(SRCS, CIN, {len(widths)}, out, {h}, {wpr});
  emit(out, {total * h * wpr});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        back = unpack_planes_ref(got, total, h, w).ravel()
        assert np.array_equal(back, want)


class TestSpatialGroupSum:
    @pytest.mark.parametrize('n_out,gs,h,w', [
        (10, 16, 6, 24),   # 実モデル(det)の形
        pytest.param(2, 4, 4, 4, marks=pytest.mark.slow),
        pytest.param(3, 16, 5, 12, marks=pytest.mark.slow),
        pytest.param(2, 32, 3, 40, marks=pytest.mark.slow),
        pytest.param(1, 7, 2, 33, marks=pytest.mark.slow),
    ])
    def test_matches_simulate(self, build_and_run, n_out, gs, h, w):
        """permute[0,2,3,1] + group_sum の融合形が素朴実行と一致する。"""
        c = n_out * gs
        width = c * h * w
        perm = IRNode(id='pm', type='permute', params={
            'in_shape': [n_out, gs, h, w], 'perm': [0, 2, 3, 1],
        })
        gsum = IRNode(id='gs', type='group_sum', params={
            'in_width': width, 'k': n_out * h * w, 'tau': 1.0,
        })
        graph = IRGraph(
            metadata={'name': 'sgs'},
            inputs=[IRPort(id='x', width=width)],
            outputs=[IRPort(id='y', width=n_out * h * w)],
            nodes=[perm, gsum],
            edges=[IREdge(src='x', dst='pm'), IREdge(src='pm', dst='gs'),
                   IREdge(src='gs', dst='y')],
        )
        rng = np.random.default_rng(n_out * 31 + gs)
        x = rng.integers(0, 2, size=(1, width), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)

        packed = pack_planes_ref(x.reshape(c, h, w))
        wpr = words_per_row(w)
        n_planes = max(1, int(gs).bit_length())
        src = PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
static uint32_t planes[{n_planes * wpr}];
static int32_t scores[{n_out * h * w}];
static uint32_t emitted[{n_out * h * w}];
int main(void) {{
  MDL_STDIO_BINARY();
  int32_t i;
  mdl_spatial_group_sum(IN, scores, {n_out}, {gs}, {h}, {w}, planes);
  for (i = 0; i < {n_out * h * w}; i++) emitted[i] = (uint32_t)scores[i];
  emit(emitted, {n_out * h * w});
  return 0;
}}
'''
        got = read_i32(build_and_run(src)).astype(np.int64)
        assert np.array_equal(got, want)

    def test_plane_count_formula(self, build_and_run):
        """平面数 = ceil(log2(gs+1))。総和 gs が桁あふれしないことの根拠。"""
        cases = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64]
        src = PRELUDE + f'''
{c_i32_array('GS', cases)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[{len(cases)}];
  int32_t i;
  for (i = 0; i < {len(cases)}; i++)
    out[i] = (uint32_t)mdl_spatial_group_sum_planes(GS[i]);
  emit(out, {len(cases)});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        for i, gs in enumerate(cases):
            assert int(got[i]) == gs.bit_length(), f'gs={gs}'
            assert gs < (1 << int(got[i])), f'gs={gs} が平面数で表せない'


class TestUnpackedLayers:
    def test_lut_layer_matches_simulate(self, build_and_run):
        rng = np.random.default_rng(21)
        in_w, out_w, n_in = 24, 40, 2
        indices = rng.integers(0, in_w, size=(n_in, out_w))
        tts = [f'0x{int(v):x}' for v in rng.integers(0, 1 << (1 << n_in),
                                                     size=out_w)]
        node = IRNode(id='l', type='lut_layer', params={
            'in_width': in_w, 'out_width': out_w, 'n_inputs': n_in,
            'indices': indices.tolist(), 'truth_tables': tts,
        })
        graph = single_node_graph(node, in_w, out_w)
        x = rng.integers(0, 2, size=(1, in_w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)

        src = PRELUDE + f'''
static const uint8_t X[] = {{{','.join(str(int(v)) for v in x[0])}}};
{c_i32_array('IDX', indices.ravel())}
{c_u32_array('TT', [int(t, 16) for t in tts])}
static uint8_t out[{out_w}];
static uint32_t o[{out_w}];
int main(void) {{
  MDL_STDIO_BINARY();
  int32_t i;
  mdl_lut_layer(X, out, {n_in}, {out_w}, IDX, TT);
  for (i = 0; i < {out_w}; i++) o[i] = (uint32_t)out[i];
  emit(o, {out_w});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).astype(np.int64)
        assert np.array_equal(got, want)

    @pytest.mark.parametrize('n_in', [
        5,
        pytest.param(1, marks=pytest.mark.slow),
        pytest.param(3, marks=pytest.mark.slow),
        pytest.param(4, marks=pytest.mark.slow),
    ])
    def test_lut_layer_all_supported_arities(self, build_and_run, n_in):
        rng = np.random.default_rng(100 + n_in)
        in_w, out_w = 16, 12
        indices = rng.integers(0, in_w, size=(n_in, out_w))
        tts = [f'0x{int(v):x}'
               for v in rng.integers(0, 1 << (1 << n_in), size=out_w)]
        node = IRNode(id='l', type='lut_layer', params={
            'in_width': in_w, 'out_width': out_w, 'n_inputs': n_in,
            'indices': indices.tolist(), 'truth_tables': tts,
        })
        graph = single_node_graph(node, in_w, out_w)
        x = rng.integers(0, 2, size=(1, in_w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)
        src = PRELUDE + f'''
static const uint8_t X[] = {{{','.join(str(int(v)) for v in x[0])}}};
{c_i32_array('IDX', indices.ravel())}
{c_u32_array('TT', [int(t, 16) for t in tts])}
static uint8_t out[{out_w}];
static uint32_t o[{out_w}];
int main(void) {{
  MDL_STDIO_BINARY();
  int32_t i;
  mdl_lut_layer(X, out, {n_in}, {out_w}, IDX, TT);
  for (i = 0; i < {out_w}; i++) o[i] = (uint32_t)out[i];
  emit(o, {out_w});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).astype(np.int64)
        assert np.array_equal(got, want)

    def test_group_sum_matches_simulate(self, build_and_run):
        rng = np.random.default_rng(31)
        in_w, k = 60, 6
        node = IRNode(id='g', type='group_sum', params={
            'in_width': in_w, 'k': k, 'tau': 1.0,
        })
        graph = single_node_graph(node, in_w, k)
        x = rng.integers(0, 2, size=(1, in_w), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)
        src = PRELUDE + f'''
static const uint8_t X[] = {{{','.join(str(int(v)) for v in x[0])}}};
static int32_t out[{k}];
static uint32_t o[{k}];
int main(void) {{
  MDL_STDIO_BINARY();
  int32_t i;
  mdl_group_sum(X, out, {in_w}, {k});
  for (i = 0; i < {k}; i++) o[i] = (uint32_t)out[i];
  emit(o, {k});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).astype(np.int64)
        assert np.array_equal(got, want)
