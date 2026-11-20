"""ゲート軸のビット並列 — 真理値表のスライス化と集計の縦カウンタ化。

どちらも「1ワードに32個の**別々のもの**を詰めて一度に進める」変形で、
固定したい性質は共通してひとつ:**速くしても答えを1ビットも変えない**。

    - `mdl_lut_layer_slice` ⇔ `mdl_lut_layer`(表の並びが違っても同値)
    - `mdl_lut_layer_packed` ⇔ 単枚のスライス経路(表の並びが違っても同値)
    - `mdl_group_sum_packed` ⇔ 1画像ずつ数えた素朴版

端数(32の倍数でない幅・レーン)と n_inputs / bwords の全域を通す。
2入力だけ専用の展開経路に落ちるので、2入力とそれ以外の両方を必ず踏むこと。
"""

import random

import pytest

from griffith_logic.deploy.plan import slice_truth_tables

from .conftest import c_i32_array, c_u16_array, c_u32_array

# 32の倍数・その前後・小さすぎる幅を混ぜる(端数処理の取りこぼし対策)
WIDTHS = [1, 31, 32, 33, 64, 100, 129]


def _case(n_inputs: int, out_width: int, seed: int):
    rng = random.Random(seed)
    n_in = max(8, out_width // 2)
    x = [rng.randrange(2) for _ in range(n_in)]
    idx = [rng.randrange(n_in)
           for _ in range(n_inputs * out_width)]
    tts = [rng.randrange(1 << (1 << n_inputs)) for _ in range(out_width)]
    return x, idx, tts, n_in


def _expected(x, idx, tts, n_inputs, out_width):
    out = []
    for o in range(out_width):
        addr = 0
        for j in range(n_inputs):
            addr |= x[idx[j * out_width + o]] << j
        out.append((tts[o] >> addr) & 1)
    return out


def test_slice_truth_tables_is_a_transpose():
    """スライス表現はビットの置き換えでしかない(情報を落とさない)。"""
    rng = random.Random(0)
    tts = [rng.randrange(16) for _ in range(70)]
    sl = slice_truth_tables(tts, 2)
    wpr = (len(tts) + 31) // 32
    assert len(sl) == 4 * wpr
    for g, tt in enumerate(tts):
        got = 0
        for m in range(4):
            got |= ((sl[m * wpr + (g >> 5)] >> (g & 31)) & 1) << m
        assert got == tt, f'ゲート{g}'


@pytest.mark.parametrize('n_inputs', [1, 2, 3, 4])
@pytest.mark.parametrize('out_width', WIDTHS)
def test_slice_matches_scalar_path(build_and_run, n_inputs, out_width):
    """スライス経路 ⇔ 既存の `mdl_lut_layer`(整数厳密)。"""
    x, idx, tts, n_in = _case(n_inputs, out_width, seed=n_inputs * 991 + out_width)
    src = f'''
#include <stdio.h>
#include "mdl_stdio.h"
#include "mdl_kernels.h"
{c_u32_array('xs', x)}
{c_i32_array('idx', idx)}
{c_u32_array('tt', tts)}
{c_u32_array('ts', slice_truth_tables(tts, n_inputs))}
int main(void) {{
  MDL_STDIO_BINARY();
  uint8_t x[{n_in}], a[{out_width}], b[{out_width}];
  int i;
  for (i = 0; i < {n_in}; i++) x[i] = (uint8_t)xs[i];
  mdl_lut_layer(x, a, {n_inputs}, {out_width}, idx, tt);
  mdl_lut_layer_slice(x, b, {n_inputs}, {out_width}, idx, ts);
  for (i = 0; i < {out_width}; i++) putchar(a[i] == b[i] ? '0' + a[i] : 'X');
  return 0;
}}
'''
    got = build_and_run(src).decode()
    assert 'X' not in got, 'スライス経路がスカラー経路と食い違った'
    want = ''.join(str(v)
                   for v in _expected(x, idx, tts, n_inputs, out_width))
    assert got == want


@pytest.mark.parametrize('n_inputs', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('out_width', WIDTHS)
def test_slice_u16_matches_i32_path(build_and_run, n_inputs, out_width):
    """添字u16版 ⇔ 添字i32版の `mdl_lut_layer_slice`(整数厳密)。

    `mdl_lut_layer_slice_u16` は同じマクロ本体(MDL_LUT_LAYER_SLICE_BODY)を
    `indices` の型だけ変えて展開したものなので、値は1ビットも変わらない
    はず——n_inputs=1..5(2入力特化経路とそれ以外の汎用経路の両方)、
    32の倍数でない幅(端数レーン)を混ぜて固定する。
    """
    x, idx, tts, n_in = _case(n_inputs, out_width, seed=n_inputs * 773 + out_width)
    assert n_in <= 0xffff, 'u16の値域内であることがテストの前提'
    src = f'''
#include <stdio.h>
#include "mdl_stdio.h"
#include "mdl_kernels.h"
{c_u32_array('xs', x)}
{c_i32_array('idx32', idx)}
{c_u16_array('idx16', idx)}
{c_u32_array('ts', slice_truth_tables(tts, n_inputs))}
int main(void) {{
  MDL_STDIO_BINARY();
  uint8_t x[{n_in}], a[{out_width}], b[{out_width}];
  int i;
  for (i = 0; i < {n_in}; i++) x[i] = (uint8_t)xs[i];
  mdl_lut_layer_slice(x, a, {n_inputs}, {out_width}, idx32, ts);
  mdl_lut_layer_slice_u16(x, b, {n_inputs}, {out_width}, idx16, ts);
  for (i = 0; i < {out_width}; i++) putchar(a[i] == b[i] ? '0' + a[i] : 'X');
  return 0;
}}
'''
    got = build_and_run(src).decode()
    assert 'X' not in got, 'u16添字経路がi32添字経路と食い違った'
    want = ''.join(str(v)
                   for v in _expected(x, idx, tts, n_inputs, out_width))
    assert got == want


# bwords が 1 / 2 / 4 / 5 になる幅と、端数レーンのある幅を混ぜる
# (縦カウンタ版 group_sum は語ごとにレーン数を切るので、ここが崩れやすい)
@pytest.mark.parametrize('batch', [1, 32, 40, 128, 129])
def test_packed_path_agrees_with_slice_path(build_and_run, batch):
    """バッチ軸パック経路と単枚のスライス経路が同じ答えを出す。

    表の並びは経路ごとに違う(パック側はゲートあたり1語のまま。組み直すと
    7%遅いと実測した)ので、**並びが違っても値が同じ**ことを固定する。
    """
    n_inputs, out_width = 2, 70
    rng = random.Random(batch)
    n_in = 48
    imgs = [[rng.randrange(2) for _ in range(n_in)] for _ in range(batch)]
    idx = [rng.randrange(n_in) for _ in range(n_inputs * out_width)]
    tts = [rng.randrange(16) for _ in range(out_width)]
    bw = (batch + 31) // 32
    flat = [v for img in imgs for v in img]
    src = f'''
#include <stdio.h>
#include "mdl_stdio.h"
#include "mdl_kernels.h"
{c_u32_array('xs', flat)}
{c_i32_array('idx', idx)}
{c_u32_array('tt', tts)}
static uint8_t ins[{batch * n_in}];
static uint32_t px[{n_in * bw}], po[{out_width * bw}];
int main(void) {{
  MDL_STDIO_BINARY();
  int i, o;
  for (i = 0; i < {batch * n_in}; i++) ins[i] = (uint8_t)xs[i];
  mdl_pack_batch(ins, px, {n_in}, {batch});
  mdl_lut_layer_packed(px, po, {n_inputs}, {out_width}, idx, tt, {bw});
  for (i = 0; i < {batch}; i++)
    for (o = 0; o < {out_width}; o++)
      putchar('0' + (int)((po[(size_t)o * {bw} + (i >> 5)] >> (i & 31)) & 1u));
  return 0;
}}
'''
    got = build_and_run(src).decode()
    want = ''.join(
        ''.join(str(v) for v in _expected(img, idx, tts, n_inputs, out_width))
        for img in imgs)
    assert got == want


@pytest.mark.parametrize('n_inputs', [2, 3])
@pytest.mark.parametrize('batch', [1, 32, 40])
def test_packed_u16_matches_i32_path(build_and_run, batch, n_inputs):
    """添字u16版の `mdl_lut_layer_packed_u16` ⇔ 添字i32版(整数厳密)。

    n_inputs=2(特化経路)と n_inputs=3(汎用フォールバック)の両方で、
    u16添字がi32添字と1ビットも変わらないことを固定する
    (packed teacher forcing、RDDLGNのM-layersの主要経路)。
    """
    out_width = 70
    rng = random.Random(batch * 97 + n_inputs)
    n_in = 48
    assert n_in <= 0xffff
    imgs = [[rng.randrange(2) for _ in range(n_in)] for _ in range(batch)]
    idx = [rng.randrange(n_in) for _ in range(n_inputs * out_width)]
    tts = [rng.randrange(1 << (1 << n_inputs)) for _ in range(out_width)]
    bw = (batch + 31) // 32
    flat = [v for img in imgs for v in img]
    src = f'''
#include <stdio.h>
#include "mdl_stdio.h"
#include "mdl_kernels.h"
{c_u32_array('xs', flat)}
{c_i32_array('idx32', idx)}
{c_u16_array('idx16', idx)}
{c_u32_array('tt', tts)}
static uint8_t ins[{batch * n_in}];
static uint32_t px[{n_in * bw}], poa[{out_width * bw}], pob[{out_width * bw}];
int main(void) {{
  MDL_STDIO_BINARY();
  int i, o;
  for (i = 0; i < {batch * n_in}; i++) ins[i] = (uint8_t)xs[i];
  mdl_pack_batch(ins, px, {n_in}, {batch});
  mdl_lut_layer_packed(px, poa, {n_inputs}, {out_width}, idx32, tt, {bw});
  mdl_lut_layer_packed_u16(px, pob, {n_inputs}, {out_width}, idx16, tt, {bw});
  for (i = 0; i < {batch}; i++)
    for (o = 0; o < {out_width}; o++) {{
      const uint32_t a = (poa[(size_t)o * {bw} + (i >> 5)] >> (i & 31)) & 1u;
      const uint32_t b = (pob[(size_t)o * {bw} + (i >> 5)] >> (i & 31)) & 1u;
      putchar(a == b ? '0' + (int)a : 'X');
    }}
  return 0;
}}
'''
    got = build_and_run(src).decode()
    assert 'X' not in got, 'u16添字経路がi32添字経路と食い違った'
    want = ''.join(
        ''.join(str(v) for v in _expected(img, idx, tts, n_inputs, out_width))
        for img in imgs)
    assert got == want


@pytest.mark.parametrize('batch,k', [(1, 3), (33, 4), (128, 6), (200, 5)])
def test_group_sum_packed_matches_scalar(build_and_run, batch, k):
    """縦カウンタ版 group_sum ⇔ 1画像ずつ数えた素朴版(整数厳密)。

    「画像ごとに1ビット抜き出す」形から「32画像ぶんのカウンタを
    ビットスライスで同時に進める」形へ変えた。**語ごとの端数レーン**と
    **平面数(= ceil(log2(gsz+1)))** が主な壊れどころなので、bwords と
    群の大きさをまたいで固定する。
    """
    rng = random.Random(batch * 31 + k)
    gsz = 70                      # 平面7枚。32の倍数でない群幅を選ぶ
    width = gsz * k
    bw = (batch + 31) // 32
    bits = [[rng.randrange(2) for _ in range(width)] for _ in range(batch)]
    # [width][bwords] のバッチ軸パック
    words = [0] * (width * bw)
    for b, row in enumerate(bits):
        for i, v in enumerate(row):
            if v:
                words[i * bw + (b >> 5)] |= 1 << (b & 31)
    src = f'''
#include <stdio.h>
#include "mdl_stdio.h"
#include "mdl_kernels.h"
{c_u32_array('xs', words)}
static int32_t sc[{batch * k}];
int main(void) {{
  MDL_STDIO_BINARY();
  int b, g;
  mdl_group_sum_packed(xs, sc, {width}, {k}, {batch}, {bw});
  for (b = 0; b < {batch}; b++)
    for (g = 0; g < {k}; g++) printf("%d ", sc[b * {k} + g]);
  return 0;
}}
'''
    got = build_and_run(src).decode().split()
    want = [str(sum(row[g * gsz:(g + 1) * gsz])) for row in bits for g in range(k)]
    assert got == want


@pytest.mark.parametrize('n_in', [1, 31, 32, 33, 100])
@pytest.mark.parametrize('batch', [1, 5, 32, 33, 64, 100])
def test_pack_batch_matches_reference(build_and_run, n_in, batch):
    """`mdl_pack_batch`(で 32x32 ビット転置へ高速化)⇔ 素朴な定義。

    [batch][n_in] の 0/非0 バイト → [n_in][ceil(batch/32)] 語、枚 b は語
    b>>5 のビット b&31。**枚数の端数(b >= batch)のビットは常に 0**。
    非0 は 1 とみなす(0/1 以外の値も混ぜて確かめる)。
    """
    rng = random.Random(n_in * 1000 + batch)
    vals = [rng.choice([0, 0, 1, 1, 2, 255]) for _ in range(batch * n_in)]
    bw = (batch + 31) // 32
    src = f'''
#include <stdio.h>
#include "mdl_kernels.h"
{c_u32_array('xs', vals)}
static uint8_t ins[{batch * n_in}];
static uint32_t px[{n_in * bw}];
int main(void) {{
  int i;
  for (i = 0; i < {batch * n_in}; i++) ins[i] = (uint8_t)xs[i];
  for (i = 0; i < {n_in * bw}; i++) px[i] = 0xdeadbeefu;  /* 書き残しを検出 */
  mdl_pack_batch(ins, px, {n_in}, {batch});
  for (i = 0; i < {n_in * bw}; i++) printf("%u\\n", px[i]);
  return 0;
}}
'''
    got = [int(v) for v in build_and_run(src).decode().split()]
    want = []
    for i in range(n_in):
        for w in range(bw):
            word = 0
            for lane in range(32):
                b = w * 32 + lane
                if b < batch and vals[b * n_in + i]:
                    word |= 1 << lane
            want.append(word)
    assert got == want
