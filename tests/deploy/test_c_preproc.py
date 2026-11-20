"""C側の前処理・読み出しの検証(K4)。

    - mdl_preproc_exec(画像u8 → ビットプレーン)が Python の記述子実行と
      全ビット一致する。Python側は既に float64 参照エンコーダと一致済み
      (test_preproc_desc.py)なので、これで「生画像 → ビット」の鎖が繋がる
    - 生成コードの mdl_infer_image(前処理+推論)が
      mdl_infer(符号化済みビット)と同じスコアを返す
    - argmax / 画素ごとargmax が numpy と一致する
"""

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.layout import words_per_row
from griffith_logic.deploy.preproc_desc import (
    build_preproc_descriptors,
    encode_bits,
    run_preproc_descriptors,
)

from .conftest import RUNTIME_C, c_i32_array, c_u32_array
from .test_c_kernels import PRELUDE, pack_planes_ref, read_u32, unpack_planes_ref
from .test_c_preproc_data import LOWLEVEL_PREPROC, THERMO_PREPROC


def _preproc_main(plan: dict, image: np.ndarray) -> str:
    c_img, h, w = image.shape
    n_words = plan['channels'] * h * words_per_row(w)
    flat = ','.join(str(int(v)) for v in image.ravel())
    return PRELUDE + f'''
static const uint8_t IMG[] = {{{flat}}};
{c_i32_array('DESC', plan['desc'])}
{c_u32_array('QVAL', plan['qval'])}
static uint32_t out[{n_words}];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_preproc_exec(IMG, {c_img}, {h}, {w}, DESC, {plan['channels']}, QVAL, out);
  emit(out, {n_words});
  return 0;
}}
'''


class TestPreprocExec:
    @pytest.mark.parametrize('preproc', [THERMO_PREPROC, LOWLEVEL_PREPROC],
                             ids=['thermometer', 'lowlevel'])
    def test_matches_python_descriptors(self, build_and_run, preproc):
        plan = build_preproc_descriptors(preproc)
        h, w = preproc['height'], preproc['width']
        c_img = len(preproc['mean'])
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(c_img, h, w), dtype=np.uint8)
        want = run_preproc_descriptors(plan, image[None])[0]

        got_words = read_u32(build_and_run(_preproc_main(plan, image)))
        got = unpack_planes_ref(got_words, plan['channels'], h, w)
        assert np.array_equal(got, want)

    @pytest.mark.parametrize('preproc', [THERMO_PREPROC, LOWLEVEL_PREPROC],
                             ids=['thermometer', 'lowlevel'])
    def test_extreme_pixels(self, build_and_run, preproc):
        """端値(0/255)飽和入力。境界のreplicateと閾値の端が効く。"""
        plan = build_preproc_descriptors(preproc)
        h, w = preproc['height'], preproc['width']
        c_img = len(preproc['mean'])
        image = np.zeros((c_img, h, w), dtype=np.uint8)
        image[:, ::2] = 255
        image[0, :, 0] = 255
        want = run_preproc_descriptors(plan, image[None])[0]
        got_words = read_u32(build_and_run(_preproc_main(plan, image)))
        got = unpack_planes_ref(got_words, plan['channels'], h, w)
        assert np.array_equal(got, want)

    def test_produces_zero_tail_bits(self, build_and_run):
        """出力の端数ビットは常に0(ビットプレーン不変条件)。"""
        preproc = dict(THERMO_PREPROC, width=37)   # 37 → 2語目に27ビットの端数
        plan = build_preproc_descriptors(preproc)
        h, w = preproc['height'], preproc['width']
        image = np.full((len(preproc['mean']), h, w), 255, dtype=np.uint8)
        got = read_u32(build_and_run(_preproc_main(plan, image)))
        wpr = words_per_row(w)
        mask = ~((1 << (w & 31)) - 1) & 0xffffffff
        for i in range(wpr - 1, len(got), wpr):
            assert int(got[i]) & mask == 0, f'語{i}の端数ビットが立っている'


class TestReadout:
    def test_argmax_matches_numpy(self, build_and_run):
        rng = np.random.default_rng(9)
        cases = [rng.integers(-50, 50, size=7) for _ in range(6)]
        cases.append(np.array([3, 3, 1, 3, 0, 0, 0]))   # 同点は先勝ち
        flat = [int(v) for row in cases for v in row]
        src = PRELUDE + f'''
{c_i32_array('S', flat)}
int main(void) {{
  MDL_STDIO_BINARY();
  uint32_t out[{len(cases)}];
  int32_t i;
  for (i = 0; i < {len(cases)}; i++)
    out[i] = (uint32_t)mdl_argmax(S + i * 7, 7);
  emit(out, {len(cases)});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        for i, row in enumerate(cases):
            assert int(got[i]) == int(np.argmax(row)), f'case={i}'

    def test_argmax_spatial_matches_numpy(self, build_and_run):
        rng = np.random.default_rng(10)
        n_out, n_pix = 4, 25
        scores = rng.integers(0, 8, size=(n_out, n_pix))
        src = PRELUDE + f'''
{c_i32_array('S', scores.ravel())}
int main(void) {{
  MDL_STDIO_BINARY();
  uint8_t labels[{n_pix}];
  uint32_t out[{n_pix}];
  int32_t i;
  mdl_argmax_spatial(S, {n_out}, {n_pix}, labels);
  for (i = 0; i < {n_pix}; i++) out[i] = labels[i];
  emit(out, {n_pix});
  return 0;
}}
'''
        got = read_u32(build_and_run(src))
        assert np.array_equal(got, scores.argmax(axis=0))


class TestGeneratedImageEntry:
    """生成コードの mdl_infer_image が mdl_infer と同じスコアを返す。"""

    IMAGE_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t image[MDL_IMAGE_CHANNELS * MDL_INPUT_HEIGHT
                       * MDL_INPUT_WIDTH_PX];
  static int32_t a[MDL_OUTPUT_WIDTH];
  static uint8_t bits[MDL_INPUT_WIDTH];
  static int32_t b[MDL_OUTPUT_WIDTH];
  int32_t i;
  size_t n_img = sizeof(image);
  if (fread(image, 1, n_img, stdin) != n_img) return 1;
  if (fread(bits, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH)
    return 1;
  if (mdl_infer_image(image, a) != 0) return 2;
  if (mdl_infer(bits, b) != 0) return 3;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    if (a[i] != b[i]) { fprintf(stderr, "mismatch at %d\\n", (int)i); return 4; }
  }
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char o[4];
    uint32_t v = (uint32_t)a[i];
    o[0]=(unsigned char)(v&0xffu); o[1]=(unsigned char)((v>>8)&0xffu);
    o[2]=(unsigned char)((v>>16)&0xffu); o[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(o, 1, 4, stdout);
  }
  return 0;
}
'''

    def test_image_entry_matches_bit_entry(self, cc, tmp_path, seg_doc):
        import subprocess

        bundle = build_bundle(seg_doc)
        plan = bundle['preproc_desc']
        files = generate_c(bundle, name='model', validate=False)
        assert 'mdl_infer_image' in files['model.h'], (
            '前処理付きエントリが生成されていない')

        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(self.IMAGE_DRIVER, encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
             str(work / 'driver.c'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        rng = np.random.default_rng(12)
        image = rng.integers(
            0, 256,
            size=(plan['image_channels'], plan['height'], plan['width']),
            dtype=np.uint8)
        bits = encode_bits(plan, image[None])[0]
        proc = subprocess.run(
            [str(work / 'app')],
            input=image.tobytes() + bits.tobytes(), capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        assert len(read_u32(proc.stdout)) > 0

    def test_no_pack_when_preproc_shares_buffer(self, seg_doc):
        """前処理の出力先と入力バッファを共有し、中間の0/1列を作らない。"""
        files = generate_c(build_bundle(seg_doc), name='model', validate=False)
        # mdl_run の本体だけを切り出す(ctx化し、次に来る関数が
        # mdl_infer から mdl_infer_ctx に変わった)。
        run_body = files['model.c'].split('static int mdl_run')[1] \
            .split('int mdl_infer_ctx(')[0]
        assert 'mdl_pack_planes' not in run_body, (
            'グラフ本体に入力パックが残っている(バッファ共有が効いていない)')


def test_pack_reference_helpers_agree():
    """テスト用ヘルパ自身の健全性(pack ↔ unpack が恒等)。"""
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(3, 5, 37), dtype=np.uint8)
    packed = pack_planes_ref(bits)
    assert np.array_equal(unpack_planes_ref(packed.ravel(), 3, 5, 37), bits)
