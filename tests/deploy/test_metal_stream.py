"""Metalランタイムのストリーム・パイプライン化の検証。

`mdl_metal_infer_batch_submit`/`_collect`(commit_asyncでコミットして
待たず、次のスロットへ積み進める)が、既存の同期API
(`mdl_metal_infer_batch`、1枚ずつ即wait)と**全枚整数厳密一致**することを
確かめる。教訓(バッファ共有の誤りは1回きりの実行では見えない)を
踏まえ、「N回連続で送る」ことをテストの核にする(1回だけの呼び出しでは
スロット間のバッファ独立性が問われない)。

macOS + Apple GPU のある環境でのみ動く(それ以外は自動skip)。
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.deploy.metalverify import has_metal

from .conftest import RUNTIME_C

RUNTIME_METAL = RUNTIME_C.parent / 'metal'

pytestmark = pytest.mark.skipif(
    not has_metal(), reason='macOS + Metal のある環境でのみ実行')


def _clangxx() -> str:
    path = shutil.which('clang++')
    assert path is not None, 'clang++ が見つからない'
    return path


def _build(bundle: dict, work_dir: Path, *, batch: int, n_streams: int,
          extra_driver_source: str) -> Path:
    """generate_metal(n_streams=N) + 手書きドライバをビルドする。"""
    for filename, text in generate_metal(
            bundle, name='model_metal', batch=batch,
            n_streams=n_streams).items():
        (work_dir / filename).write_text(text, encoding='utf-8')
    (work_dir / 'driver.mm').write_text(extra_driver_source, encoding='utf-8')
    exe = work_dir / 'app'
    build = subprocess.run(
        [_clangxx(), '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{RUNTIME_METAL}', '-framework', 'Metal', '-framework',
         'Foundation', str(work_dir / 'model_metal.mm'),
         str(RUNTIME_METAL / 'mdl_metal.mm'), str(work_dir / 'driver.mm'),
         '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f'ビルド失敗:\n{build.stderr}')
    return exe


# 「K回、毎回ランダムなバッチ枚数nで送る」ドライバ。逐次経路(mdl_metal_infer_
# batch)と、ストリーム経路(submit N深さぶん先行してからcollect)の両方を
# 同じ入力列で実行し、全出力をmemcmpする。バッチ枚数を回ごとに変えるのは、
# 「スロットのバッファサイズがMDL_MAX_BATCH固定でも、実行時nが変わっても
# 壊れない」ことも一緒に確かめるため。
_BITS_DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {{
int mdl_metal_infer_batch(const unsigned char*, int*, int);
int mdl_metal_infer_batch_submit(const unsigned char*, int, int);
int mdl_metal_infer_batch_collect(int*, int, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_metal_max_batch(void);
int mdl_model_init(void);
}}

#define K {k}
#define NSLOT {nslot}

static const int NS[K] = {{{ns_csv}}};

int main(void) {{
  const int in_w = mdl_metal_input_width();
  const int out_w = mdl_metal_output_width();
  const int maxb = mdl_metal_max_batch();
  unsigned char *inputs[K];
  int *seq[K];
  int *strm[K];
  int k;
  size_t i;

  if (mdl_model_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  for (k = 0; k < K; k++) {{
    const int n = NS[k];
    inputs[k] = (unsigned char*)malloc((size_t)n * in_w);
    seq[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    strm[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    for (i = 0; i < (size_t)n * in_w; i++) {{
      inputs[k][i] = (unsigned char)(((i + 1) * 2654435761u * (k + 1)) & 1);
    }}
  }}
  if (maxb < 1) {{ fprintf(stderr, "bad maxb\\n"); return 4; }}

  for (k = 0; k < K; k++) {{
    if (mdl_metal_infer_batch(inputs[k], seq[k], NS[k]) != 0) {{
      fprintf(stderr, "seq infer failed k=%d\\n", k); return 2;
    }}
  }}

  {{
    int pending = 0, next_submit = 0, next_collect = 0;
    while (next_collect < K) {{
      while (next_submit < K && pending < NSLOT) {{
        const int slot = next_submit % NSLOT;
        if (mdl_metal_infer_batch_submit(inputs[next_submit], NS[next_submit],
                                         slot) != 0) {{
          fprintf(stderr, "submit failed k=%d\\n", next_submit); return 2;
        }}
        next_submit++; pending++;
      }}
      {{
        const int slot = next_collect % NSLOT;
        if (mdl_metal_infer_batch_collect(strm[next_collect], NS[next_collect],
                                          slot) != 0) {{
          fprintf(stderr, "collect failed k=%d\\n", next_collect); return 2;
        }}
        next_collect++; pending--;
      }}
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (memcmp(seq[k], strm[k], (size_t)NS[k] * out_w * sizeof(int)) != 0) {{
      fprintf(stderr, "MISMATCH at k=%d (n=%d)\\n", k, NS[k]);
      return 1;
    }}
  }}
  printf("OK\\n");
  return 0;
}}
'''

_IMAGE_DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {{
int mdl_metal_infer_image_batch(const unsigned char*, int*, int);
int mdl_metal_infer_image_batch_submit(const unsigned char*, int, int);
int mdl_metal_infer_image_batch_collect(int*, int, int);
int mdl_metal_output_width(void);
int mdl_metal_max_batch(void);
int mdl_model_init(void);
}}

#define K {k}
#define NSLOT {nslot}
static const int n_img = {n_img};
static const int NS[K] = {{{ns_csv}}};

int main(void) {{
  const int out_w = mdl_metal_output_width();
  unsigned char *images[K];
  int *seq[K];
  int *strm[K];
  int k;
  size_t i;

  if (mdl_model_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

  for (k = 0; k < K; k++) {{
    const int n = NS[k];
    images[k] = (unsigned char*)malloc((size_t)n * n_img);
    seq[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    strm[k] = (int*)malloc((size_t)n * out_w * sizeof(int));
    for (i = 0; i < (size_t)n * n_img; i++) {{
      images[k][i] = (unsigned char)(((i + 7) * 2654435761u * (k + 3)) >> 24);
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (mdl_metal_infer_image_batch(images[k], seq[k], NS[k]) != 0) {{
      fprintf(stderr, "seq infer failed k=%d\\n", k); return 2;
    }}
  }}

  {{
    int pending = 0, next_submit = 0, next_collect = 0;
    while (next_collect < K) {{
      while (next_submit < K && pending < NSLOT) {{
        const int slot = next_submit % NSLOT;
        if (mdl_metal_infer_image_batch_submit(images[next_submit],
                                               NS[next_submit], slot) != 0) {{
          fprintf(stderr, "submit failed k=%d\\n", next_submit); return 2;
        }}
        next_submit++; pending++;
      }}
      {{
        const int slot = next_collect % NSLOT;
        if (mdl_metal_infer_image_batch_collect(strm[next_collect],
                                                NS[next_collect], slot) != 0) {{
          fprintf(stderr, "collect failed k=%d\\n", next_collect); return 2;
        }}
        next_collect++; pending--;
      }}
    }}
  }}

  for (k = 0; k < K; k++) {{
    if (memcmp(seq[k], strm[k], (size_t)NS[k] * out_w * sizeof(int)) != 0) {{
      fprintf(stderr, "MISMATCH at k=%d (n=%d)\\n", k, NS[k]);
      return 1;
    }}
  }}
  printf("OK\\n");
  return 0;
}}
'''


@pytest.mark.slow
class TestStreamMatchesSequential:
    """N回連続ストリーム ⇔ 1枚(1回)ずつN回、が全て一致する。"""

    def test_dense_bits_path(self, dense_doc, tmp_path):
        bundle = build_bundle(dense_doc)
        rng = np.random.default_rng(0)
        batch = 6
        k, nslot = 24, 3
        ns = rng.integers(1, batch + 1, size=k).tolist()
        src = _BITS_DRIVER_TMPL.format(
            k=k, nslot=nslot, ns_csv=','.join(str(n) for n in ns))
        exe = _build(bundle, tmp_path, batch=batch, n_streams=nslot,
                    extra_driver_source=src)
        proc = subprocess.run([str(exe)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'OK' in proc.stdout

    def test_image_path(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        p = bundle['preproc']
        n_img = p['rgb_channels'] * p['height'] * p['width']
        batch = 5
        k, nslot = 18, 2
        rng = np.random.default_rng(1)
        ns = rng.integers(1, batch + 1, size=k).tolist()
        src = _IMAGE_DRIVER_TMPL.format(
            k=k, nslot=nslot, n_img=n_img,
            ns_csv=','.join(str(n) for n in ns))
        exe = _build(bundle, tmp_path, batch=batch, n_streams=nslot,
                    extra_driver_source=src)
        proc = subprocess.run([str(exe)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'OK' in proc.stdout

    def test_bits_path_with_preproc_model(self, seg_doc, tmp_path):
        """前処理(image)付きモデルのビット入力submit経路(対称の回帰確認)。

        seg_doc は画像前処理(preproc)を持つモデルなので、`generate_metal`は
        `mdl_metal_infer_batch`(ビット入力)と`mdl_metal_infer_image_batch`
        (画像入力)の両方の入口を生成する。非streamingのビット入力entry
        (`mdl_metal_infer_batch`)は呼び出しのたびに`mdl_metal_pack`で
        bitplaneバッファを更新してから本体を実行するが、導入時の
        streaming版(`mdl_metal_infer_batch_submit`)はこのpack呼び出しが
        欠落しており、bitplaneバッファがそのスロットの前回内容のまま
        化けて古い入力に対するスコアを返しうる(CUDA側に見つかった
        のと同型のバグ)。

        この経路を「スロット数(nslot)<送信回数(k)のラウンドロビン」+
        「毎回異なる入力」でsubmit→collectし、逐次経路
        (`mdl_metal_infer_batch`)と全送信ぶん整数厳密一致することを
        確かめる(1回きりの呼び出しでは各スロットが初回しか使われず、
        pack欠落があっても偶然一致してしまいうる ─ 教訓どおり
        N回連続がテストの核)。
        """
        bundle = build_bundle(seg_doc)
        rng = np.random.default_rng(2)
        batch = 6
        k, nslot = 24, 3
        ns = rng.integers(1, batch + 1, size=k).tolist()
        src = _BITS_DRIVER_TMPL.format(
            k=k, nslot=nslot, ns_csv=','.join(str(n) for n in ns))
        exe = _build(bundle, tmp_path, batch=batch, n_streams=nslot,
                    extra_driver_source=src)
        proc = subprocess.run([str(exe)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'OK' in proc.stdout


_GUARD_DRIVER = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
int mdl_metal_infer_batch_submit(const unsigned char*, int, int);
int mdl_metal_infer_batch_collect(int*, int, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}

int main(void) {
  const int in_w = mdl_metal_input_width();
  const int out_w = mdl_metal_output_width();
  unsigned char *bits = (unsigned char*)calloc((size_t)in_w, 1);
  int *scores = (int*)malloc((size_t)out_w * sizeof(int));
  int rc1, rc2;

  if (mdl_model_init() != 0) return 3;

  rc1 = mdl_metal_infer_batch_submit(bits, 1, 0);
  /* collectせずに同じスロットへもう一度submitする(警告する
   * 「バッファ共有の誤り」そのものの操作)。ガードが働いて-1を返す
   * べき(黙って前回ぶんの入力バッファを上書きしてはいけない)。 */
  rc2 = mdl_metal_infer_batch_submit(bits, 1, 0);
  printf("rc1=%d rc2=%d\\n", rc1, rc2);
  if (rc1 != 0 || rc2 == 0) {
    fprintf(stderr, "ガードが機能していない(2回目もOKを返した)\\n");
    return 1;
  }
  /* 後始末: pendingのままだとshutdownが待つだけなので、ここでcollectして
   * おく(このプロセスの終了確認には不要だが、行儀のため)。 */
  if (mdl_metal_infer_batch_collect(scores, 1, 0) != 0) return 2;
  printf("OK\\n");
  return 0;
}
'''


@pytest.mark.slow
def test_commit_async_rejects_unreleased_slot(dense_doc, tmp_path):
    """collectを挟まずに同じスロットへ二重commit_asyncすると拒否される。

    これは「スクラッチ/入力バッファをスロット間で共有すると2枚目が1枚目を
    壊す」(教訓)という事故そのものを、実行時ガードが未然に防いで
    いることの確認(タスク指示の故障注入に相当)。ガードが無ければ
    ここは黙って進み、1枚目の入力バッファへ2枚目の入力を上書きしてから
    1枚目のGPUカーネルを走らせる可能性がある壊れ方になる。
    """
    bundle = build_bundle(dense_doc)
    exe = _build(bundle, tmp_path, batch=1, n_streams=2,
                extra_driver_source=_GUARD_DRIVER)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout


def test_n_streams_out_of_range_rejected(dense_doc):
    """MDL_METAL_MAX_STREAMS を超える n_streams は生成時点で拒否される。"""
    from griffith_logic.deploy.metalbackend import MDL_METAL_MAX_STREAMS

    bundle = build_bundle(dense_doc)
    with pytest.raises(ValueError, match='MDL_METAL_MAX_STREAMS'):
        generate_metal(bundle, n_streams=MDL_METAL_MAX_STREAMS + 1)
