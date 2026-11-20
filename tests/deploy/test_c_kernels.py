"""Cカーネルの検証。

錨の置き方:
    - ワードレイアウト契約(LSB-first / 行頭ワード境界 / 端数ビット常0)は
      **ワード配列そのものをバイト比較**して固定する。ここがずれると
      上位のどの検証も意味を失うので最初に置く
    - conv2d の実行結果は `ir/simulate.py`(numpyのgolden model)と
      整数厳密比較する。素朴実装を基準にするので、命令列化・共有シフト平面・
      tail_mask といった最適化が壊れれば必ず落ちる

Cコンパイラが無い環境では conftest の `cc` フィクスチャが自動skipする。
"""

import numpy as np
import pytest

from griffith_logic.deploy.layout import (
    build_conv_layout,
    tail_mask,
    words_per_row,
)
from griffith_logic.deploy.programs import build_programs
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.simulate import simulate

from .conftest import c_i32_array, c_u32_array

# ---------------------------------------------------------------------------
# Python側のビットプレーン参照(素朴実装。速度は問わない)
# ---------------------------------------------------------------------------


def pack_planes_ref(bits: np.ndarray) -> np.ndarray:
    """[C,H,W] の0/1 → [C,H,wpr] のワード列(LSB-first、端数ビット0)。"""
    c, h, w = bits.shape
    wpr = words_per_row(w)
    out = np.zeros((c, h, wpr), dtype=np.uint32)
    for ch in range(c):
        for y in range(h):
            for x in range(w):
                if bits[ch, y, x]:
                    out[ch, y, x >> 5] |= np.uint32(1) << np.uint32(x & 31)
    return out


def unpack_planes_ref(words: np.ndarray, c: int, h: int, w: int) -> np.ndarray:
    flat = words.reshape(c, h, -1)
    out = np.zeros((c, h, w), dtype=np.uint8)
    for ch in range(c):
        for y in range(h):
            for x in range(w):
                out[ch, y, x] = (int(flat[ch, y, x >> 5]) >> (x & 31)) & 1
    return out


def shared_shift_plane_ref(plane_bits: np.ndarray, p_h: int, s: int) -> np.ndarray:
    """共有シフト平面の参照。行 r の内容 = 入力行 y=r-p_h を s ビットシフト。

    走査するのは W ではなく **行のワード幅ぶん(32*wpr)** の全ビット位置。
    左シフト(s<0)は元の bit W-1 を位置 W へ運ぶので、この平面では端数ビットが
    立ちうる(TS側 shiftRowInto も同じで、マスクしていない)。端数ビットが0に
    戻るのは conv の出力書き出しで tail_mask を掛ける時点であり、その不変条件は
    TestConvExec::test_output_tail_bits_are_masked が別途固定する。
    """
    h, w = plane_bits.shape
    wpr = words_per_row(w)
    bits_per_row = 32 * wpr
    rows = h + 2 * p_h
    out = np.zeros((rows, wpr), dtype=np.uint32)
    for r in range(rows):
        y = r - p_h
        if not (0 <= y < h):
            continue
        for x in range(bits_per_row):
            src_x = x + s
            if 0 <= src_x < bits_per_row and src_x < w and plane_bits[y, src_x]:
                out[r, x >> 5] |= np.uint32(1) << np.uint32(x & 31)
    return out


def read_u32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype='<u4')


PRELUDE = '''#include "mdl_kernels.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
static void emit(const uint32_t *p, int32_t n) {
  int32_t i;
  for (i = 0; i < n; i++) {
    unsigned char b[4];
    b[0] = (unsigned char)(p[i] & 0xffu);
    b[1] = (unsigned char)((p[i] >> 8) & 0xffu);
    b[2] = (unsigned char)((p[i] >> 16) & 0xffu);
    b[3] = (unsigned char)((p[i] >> 24) & 0xffu);
    fwrite(b, 1, 4, stdout);
  }
}
'''


class TestGate2:
    def test_all_16_truth_tables(self, build_and_run):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2 ** 32, size=8, dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 2 ** 32, size=8, dtype=np.uint64).astype(np.uint32)
        src = PRELUDE + f'''
{c_u32_array('A', a)}
{c_u32_array('B', b)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[16 * 8];
  uint32_t tt; int32_t i;
  for (tt = 0; tt < 16u; tt++)
    for (i = 0; i < 8; i++) out[tt * 8 + i] = mdl_gate2(tt, A[i], B[i]);
  emit(out, 16 * 8);
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).reshape(16, 8)
        for tt in range(16):
            for i in range(8):
                want = 0
                for bit in range(32):
                    av = (int(a[i]) >> bit) & 1
                    bv = (int(b[i]) >> bit) & 1
                    if (tt >> ((bv << 1) | av)) & 1:
                        want |= 1 << bit
                assert int(got[tt, i]) == want, f'tt={tt:#x} i={i}'

    def test_range_matches_scalar(self, build_and_run):
        rng = np.random.default_rng(1)
        n = 5
        buf = rng.integers(0, 2 ** 32, size=3 * n, dtype=np.uint64).astype(np.uint32)
        src = PRELUDE + f'''
{c_u32_array('INIT', buf)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t buf[3 * {n}];
  uint32_t out[16 * {n}];
  uint32_t tt; int32_t i;
  for (tt = 0; tt < 16u; tt++) {{
    for (i = 0; i < 3 * {n}; i++) buf[i] = INIT[i];
    mdl_gate2_range(buf, 2 * {n}, 0, {n}, tt, {n});
    for (i = 0; i < {n}; i++) out[tt * {n} + i] = buf[2 * {n} + i];
  }}
  emit(out, 16 * {n});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).reshape(16, n)
        a, b = buf[:n], buf[n:2 * n]
        for tt in range(16):
            for i in range(n):
                want = 0
                for bit in range(32):
                    av = (int(a[i]) >> bit) & 1
                    bv = (int(b[i]) >> bit) & 1
                    if (tt >> ((bv << 1) | av)) & 1:
                        want |= 1 << bit
                assert int(got[tt, i]) == want, f'tt={tt:#x} i={i}'


class TestWordLayout:
    """レイアウト契約の錨。ここが崩れたら上位の検証は全て無意味になる。"""

    def test_words_per_row_and_tail_mask(self, build_and_run):
        widths = [1, 7, 31, 32, 33, 64, 65, 100, 128, 173]
        src = PRELUDE + f'''
{c_i32_array('WIDTHS', widths)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[2 * {len(widths)}];
  int32_t i;
  for (i = 0; i < {len(widths)}; i++) {{
    out[2 * i] = (uint32_t)mdl_words_per_row(WIDTHS[i]);
    out[2 * i + 1] = mdl_tail_mask(WIDTHS[i]);
  }}
  emit(out, 2 * {len(widths)});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).reshape(-1, 2)
        for i, w in enumerate(widths):
            assert int(got[i, 0]) == words_per_row(w), f'W={w}'
            assert int(got[i, 1]) == tail_mask(w), f'W={w}'

    @pytest.mark.parametrize('shape', [
        (2, 3, 7),
        pytest.param((1, 1, 1), marks=pytest.mark.slow),
        pytest.param((3, 4, 32), marks=pytest.mark.slow),
        (2, 2, 33),
        pytest.param((1, 5, 70), marks=pytest.mark.slow),
    ])
    def test_pack_matches_reference_word_for_word(self, build_and_run, shape):
        c, h, w = shape
        rng = np.random.default_rng(hash(shape) % 1000)
        bits = rng.integers(0, 2, size=shape, dtype=np.uint8)
        want = pack_planes_ref(bits)
        src = PRELUDE + f'''
static const uint8_t SRC[] = {{{",".join(str(int(v)) for v in bits.ravel())}}};
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[{c * h * words_per_row(w)}];
  mdl_pack_planes(SRC, out, {c}, {h}, {w});
  emit(out, {c * h * words_per_row(w)});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        assert got.tobytes() == want.ravel().tobytes()

    def test_pack_leaves_tail_bits_zero(self, build_and_run):
        """x>=W のビットが常に0(全ビット1の入力でも)。"""
        c, h, w = 1, 2, 5
        bits = np.ones((c, h, w), dtype=np.uint8)
        src = PRELUDE + f'''
static const uint8_t SRC[] = {{{",".join(['1'] * (c * h * w))}}};
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[{c * h}];
  mdl_pack_planes(SRC, out, {c}, {h}, {w});
  emit(out, {c * h});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        assert all(int(v) == (1 << w) - 1 for v in got)
        assert got.tobytes() == pack_planes_ref(bits).ravel().tobytes()

    def test_pack_treats_nonzero_as_one(self, build_and_run):
        """0/1 以外のバイト値も非ゼロ=1として扱う(JS側と同じ意味論)。"""
        vals = [0, 1, 2, 255, 128, 7, 0, 3]
        src = PRELUDE + f'''
static const uint8_t SRC[] = {{{",".join(str(v) for v in vals)}}};
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[1];
  mdl_pack_planes(SRC, out, 1, 1, {len(vals)});
  emit(out, 1);
  return 0;
}}
'''
        got = int(read_u32(build_and_run(src))[0])
        want = sum(1 << i for i, v in enumerate(vals) if v != 0)
        assert got == want

    @pytest.mark.parametrize('shape', [
        (1, 4, 33),
        pytest.param((2, 3, 7), marks=pytest.mark.slow),
        pytest.param((3, 2, 64), marks=pytest.mark.slow),
    ])
    def test_unpack_roundtrip(self, build_and_run, shape):
        c, h, w = shape
        rng = np.random.default_rng(hash(shape) % 997)
        bits = rng.integers(0, 2, size=shape, dtype=np.uint8)
        src = PRELUDE + f'''
static const uint8_t SRC[] = {{{",".join(str(int(v)) for v in bits.ravel())}}};
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t packed[{c * h * words_per_row(w)}];
  uint8_t back[{c * h * w}];
  uint32_t out[{c * h * w}];
  int32_t i;
  mdl_pack_planes(SRC, packed, {c}, {h}, {w});
  mdl_unpack_planes(packed, back, {c}, {h}, {w});
  for (i = 0; i < {c * h * w}; i++) out[i] = back[i];
  emit(out, {c * h * w});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).reshape(shape)
        assert np.array_equal(got.astype(np.uint8), bits)


class TestSharedShiftPlane:
    SHIFTS = (-2, -1, 0, 1, 2)

    @pytest.mark.parametrize('w', [
        33,
        pytest.param(7, marks=pytest.mark.slow),
        pytest.param(32, marks=pytest.mark.slow),
        pytest.param(70, marks=pytest.mark.slow),
    ])
    def test_matches_reference(self, build_and_run, w):
        """1つのwについて全シフト量をまとめて実行する(コンパイル回数を抑える)。"""
        h, p_h = 5, 1
        wpr = words_per_row(w)
        rows = h + 2 * p_h
        plane_words = rows * wpr
        rng = np.random.default_rng(w * 100)
        bits = rng.integers(0, 2, size=(1, h, w), dtype=np.uint8)
        packed = pack_planes_ref(bits)
        shifts = self.SHIFTS
        src = PRELUDE + f'''
{c_u32_array('SRC', packed.ravel())}
{c_i32_array('SHIFTS', shifts)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t dst[{len(shifts) * plane_words}];
  int32_t k, i;
  for (i = 0; i < {len(shifts) * plane_words}; i++) dst[i] = 0xdeadbeefu;
  for (k = 0; k < {len(shifts)}; k++) {{
    mdl_shared_shift_plane(SRC, 0, {h}, {wpr}, {p_h}, SHIFTS[k],
                           dst, k * {plane_words});
  }}
  emit(dst, {len(shifts) * plane_words});
  return 0;
}}
'''
        got = read_u32(build_and_run(src)).reshape(len(shifts), plane_words)
        for k, s in enumerate(shifts):
            want = shared_shift_plane_ref(bits[0], p_h, s)
            assert got[k].tobytes() == want.ravel().tobytes(), f'W={w} s={s}'

    def test_large_shift_falls_back_correctly(self, build_and_run):
        """|s|>=32 の想定外ジオメトリでも行ごとの汎用路で正しい。"""
        h, w, p_h, s = 3, 80, 0, 35
        wpr = words_per_row(w)
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=(1, h, w), dtype=np.uint8)
        packed = pack_planes_ref(bits)
        want = shared_shift_plane_ref(bits[0], p_h, s)
        src = PRELUDE + f'''
{c_u32_array('SRC', packed.ravel())}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t dst[{h * wpr}];
  mdl_shared_shift_plane(SRC, 0, {h}, {wpr}, {p_h}, {s}, dst, 0);
  emit(dst, {h * wpr});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        assert got.tobytes() == want.ravel().tobytes()


# ---------------------------------------------------------------------------
# conv2d: ir/simulate.py を基準にした整数厳密比較
# ---------------------------------------------------------------------------


def make_conv_graph(rng, *, c, h, w, oc, kh=3, kw=3, depth=2, stride=1, pad=1):
    """conv2dノード1個だけのIR(simulateで素朴に評価できる)。"""
    khw = kh * kw
    n_leaves = 1 << depth
    n_internal = n_leaves - 1
    indices_flat = rng.integers(0, c * khw, size=(oc, n_leaves)).tolist()
    tts = [f'0x{int(v):x}' for v in rng.integers(0, 16, size=oc * n_internal)]
    h_out = (h + 2 * pad - kh) // stride + 1
    w_out = (w + 2 * pad - kw) // stride + 1
    node = IRNode(id='conv', type='conv2d', params={
        'in_shape': [c, h, w], 'out_channels': oc,
        'kernel_size': [kh, kw], 'stride': [stride, stride],
        'padding': [pad, pad], 'tree_depth': depth, 'n_inputs': 2,
        'indices_flat': indices_flat, 'truth_tables': tts,
    })
    graph = IRGraph(
        metadata={'name': 'conv_only'},
        inputs=[IRPort(id='x', width=c * h * w)],
        outputs=[IRPort(id='y', width=oc * h_out * w_out)],
        nodes=[node],
        edges=[IREdge(src='x', dst='conv'), IREdge(src='conv', dst='y')],
    )
    return graph, node, h_out, w_out


CONV_CASES = [
    # (c, h, w, oc, kh, kw, depth, stride, pad) — 高速路と汎用路の両方を踏む
    # 常時: 高速路(端数ビットあり)・汎用路(stride>1)を1本ずつ踏む
    pytest.param(4, 5, 33, 5, 3, 3, 3, 1, 1, id='fast_w33_tailbit'),
    pytest.param(2, 8, 8, 3, 3, 3, 2, 2, 1, id='strided_s2'),
    # 以下は形状スイープ(回帰用)
    pytest.param(3, 8, 8, 4, 3, 3, 2, 1, 1, id='fast_w8',
                 marks=pytest.mark.slow),
    pytest.param(2, 6, 40, 3, 3, 3, 2, 1, 1, id='fast_w40_multiword',
                 marks=pytest.mark.slow),
    pytest.param(3, 9, 9, 4, 3, 3, 2, 1, 0, id='nopad',
                 marks=pytest.mark.slow),
    pytest.param(2, 7, 7, 3, 2, 2, 2, 1, 0, id='k2',
                 marks=pytest.mark.slow),
    pytest.param(1, 4, 4, 2, 3, 3, 1, 1, 1, id='depth1',
                 marks=pytest.mark.slow),
]


class TestConvExec:
    @pytest.mark.parametrize('c,h,w,oc,kh,kw,depth,stride,pad', CONV_CASES)
    def test_matches_simulate(self, build_and_run, c, h, w, oc, kh, kw,
                              depth, stride, pad):
        rng = np.random.default_rng(abs(hash((c, h, w, oc, stride, pad))) % 10_000)
        graph, node, h_out, w_out = make_conv_graph(
            rng, c=c, h=h, w=w, oc=oc, kh=kh, kw=kw, depth=depth,
            stride=stride, pad=pad)
        programs = build_programs(graph)
        layout = build_conv_layout(node.params, programs['conv'])

        x_bits = rng.integers(0, 2, size=(1, c * h * w), dtype=np.uint8)
        want = simulate(graph, x_bits)[0].astype(np.uint8)

        packed = pack_planes_ref(x_bits.reshape(c, h, w))
        got_words = read_u32(build_and_run(
            _conv_main(layout, packed)))
        got = unpack_planes_ref(
            got_words, layout.out_channels, h_out, w_out).ravel()
        assert np.array_equal(got, want)

    def test_output_tail_bits_are_masked(self, build_and_run):
        """NOT系ゲートが端数ビットに立てた1が出力に残らない(不変条件)。"""
        rng = np.random.default_rng(7)
        c, h, w, oc = 2, 4, 5, 3   # w=5 → 27ビットが端数
        graph, node, h_out, w_out = make_conv_graph(
            rng, c=c, h=h, w=w, oc=oc, depth=2, stride=1, pad=1)
        # 全ゲートをNOT系(0x5 = ~a)にして端数ビットへ1を立てさせる
        node.params['truth_tables'] = ['0x5'] * len(node.params['truth_tables'])
        programs = build_programs(graph)
        layout = build_conv_layout(node.params, programs['conv'])

        x_bits = rng.integers(0, 2, size=(1, c * h * w), dtype=np.uint8)
        packed = pack_planes_ref(x_bits.reshape(c, h, w))
        got = read_u32(build_and_run(_conv_main(layout, packed)))
        mask = tail_mask(w_out)
        for oy in range(h_out * layout.out_channels):
            word = int(got[oy * layout.wpr_out + layout.wpr_out - 1])
            assert word & ~mask == 0, '端数ビットにマスクされ残りがある'

        want = simulate(graph, x_bits)[0].astype(np.uint8)
        back = unpack_planes_ref(got, layout.out_channels, h_out, w_out).ravel()
        assert np.array_equal(back, want)

    def test_all_gates_folded_program(self, build_and_run):
        """命令列が空(全ゲートが畳み込まれた)conv でも正しく動く。"""
        rng = np.random.default_rng(3)
        graph, node, h_out, w_out = make_conv_graph(
            rng, c=2, h=6, w=6, oc=4, depth=2, stride=1, pad=1)
        node.params['truth_tables'] = ['0xa'] * len(node.params['truth_tables'])
        programs = build_programs(graph)
        assert programs['conv']['n_instrs'] == 0, '前提: 全て畳み込まれる'
        layout = build_conv_layout(node.params, programs['conv'])

        x_bits = rng.integers(0, 2, size=(1, 2 * 6 * 6), dtype=np.uint8)
        packed = pack_planes_ref(x_bits.reshape(2, 6, 6))
        got = read_u32(build_and_run(_conv_main(layout, packed)))
        back = unpack_planes_ref(got, layout.out_channels, h_out, w_out).ravel()
        assert np.array_equal(back, simulate(graph, x_bits)[0].astype(np.uint8))


def _conv_main(layout, packed: np.ndarray) -> str:
    """レイアウトと入力を焼いたconv実行main。"""
    return PRELUDE + f'''
{c_u32_array('IN', packed.ravel())}
{c_i32_array('PLANE_C', layout.plane_c)}
{c_i32_array('PLANE_DY', layout.plane_dy)}
{c_i32_array('PLANE_DX', layout.plane_dx)}
{c_i32_array('INSTR', layout.instr_flat)}
{c_i32_array('OC_SRC', layout.oc_src_off)}
{c_i32_array('WSTART', layout.write_start)}
{c_i32_array('WOC', layout.write_oc)}
static uint32_t scratch[{layout.scratch_words}];
static uint32_t out[{layout.out_plane_words}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_conv_layout L;
  L.h = {layout.h}; L.w = {layout.w};
  L.h_out = {layout.h_out}; L.w_out = {layout.w_out};
  L.s_h = {layout.s_h}; L.s_w = {layout.s_w};
  L.p_h = {layout.p_h}; L.p_w = {layout.p_w};
  L.out_channels = {layout.out_channels};
  L.wpr_in = {layout.wpr_in}; L.wpr_out = {layout.wpr_out};
  L.tap_wpr = {layout.tap_wpr}; L.tap_rows_words = {layout.tap_rows_words};
  L.mask_out = {layout.mask_out}u;
  L.use_fast_path = {1 if layout.use_fast_path else 0};
  L.plane_c = PLANE_C; L.plane_dy = PLANE_DY; L.plane_dx = PLANE_DX;
  L.n_planes = {layout.n_planes}; L.plane_stride = {layout.plane_stride};
  L.plane_rows = {layout.plane_rows};
  L.in_plane_words = {layout.in_plane_words};
  L.out_plane_words = {layout.out_plane_words};
  L.instr_flat = INSTR; L.n_instrs = {layout.n_instrs};
  L.oc_src_off = OC_SRC; L.write_start = WSTART; L.write_oc = WOC;
  L.pool_base = {layout.pool_base}; L.reg_base = {layout.reg_base};
  L.scratch_words = {layout.scratch_words};
  mdl_conv_init_scratch(&L, scratch);
  mdl_conv_exec(&L, IN, scratch, out);
  emit(out, {layout.out_plane_words});
  return 0;
}}
'''
