"""CUDAランタイムのストリーム・パイプライン化(Metal ミラー)の検証。

このMacにはCUDA実機/nvccが無いため、常時実行できるのは**生成構造の検証**
(`generate_cuda`が吐くソース文字列の中身を見る、実行はしない)だけに絞る。
`tests/deploy/test_metal_stream.py`と同じ「N回連続で送る」ドライバ雛形も
用意してあるが、実際にビルド・実行するテストは `nvcc` が無い環境では自動
skipする(nvcc のある環境 ─ リモートGPU機やCUDA入りCIランナー ─ でだけ
実行される。タスクの実機検証はリモートSSH機で別途 `scripts/
verify_cuda_stream_remote.py` を使って行った)。

対称性の確認observationは`test_metal_stream.py`のMetal版と同じ発想:
教訓(バッファ共有の誤りは1回きりの実行では見えない)を踏まえ、
「N回連続で送る」ドライバをビルド用に用意しておく(実行できる環境向け)。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cudabackend import (
    MDL_CUDA_MAX_STREAMS,
    generate_cuda,
)

from .conftest import RUNTIME_C

RUNTIME_CUDA = RUNTIME_C.parent / 'cuda'
RUNTIME_GPU = RUNTIME_C.parent / 'gpu'


def _nvcc() -> str | None:
    return shutil.which('nvcc')


def has_cuda() -> bool:
    return _nvcc() is not None


# ---------------------------------------------------------------------------
# 生成構造の検証(実行なし、常時実行)
# ---------------------------------------------------------------------------

def test_n_streams_out_of_range_rejected(dense_doc):
    """MDL_CUDA_MAX_STREAMS を超える n_streams は生成時点で拒否される
    (Metal版 test_n_streams_out_of_range_rejected と同じ形)。"""
    bundle = build_bundle(dense_doc)
    with pytest.raises(ValueError, match='MDL_CUDA_MAX_STREAMS'):
        generate_cuda(bundle, n_streams=MDL_CUDA_MAX_STREAMS + 1)


def test_n_streams_zero_rejected(dense_doc):
    bundle = build_bundle(dense_doc)
    with pytest.raises(ValueError, match='1以上'):
        generate_cuda(bundle, n_streams=0)


def test_dense_packed_with_n_streams_rejected(dense_doc):
    """dense_packed=True と n_streams>1 の併用はv1未対応(Metal版と同じ
    制約)。dense_docはbatch>=32でなければdense_packed自体が拒否される
    ので、n_streams>1の方の判定が先に発火することを確かめる(__init__の
    バリデーション順)。"""
    bundle = build_bundle(dense_doc)
    with pytest.raises(ValueError, match='dense_packed.*n_streams'):
        generate_cuda(bundle, n_streams=2, dense_packed=True)


def test_n_streams_1_output_unchanged_structurally(dense_doc):
    """n_streams=1(既定)の生成物は、ストリーミング関連のシンボルを一切
    含まない(既存の非ストリーミング経路が変わっていないことの構造的な
    確認)。"""
    bundle = build_bundle(dense_doc)
    src = generate_cuda(bundle, batch=4)['model_cuda.cu']
    assert 'mdl_cuda_infer_batch_submit' not in src
    assert 'mdl_cuda_infer_batch_collect' not in src
    assert 'mdl_model_encode' not in src
    assert 'mdl_cuda_streams_init' not in src
    assert 'static int mdl_cuda_run(int *scores, int n) {' in src


def test_n_streams_generates_submit_collect(dense_doc):
    """n_streams>1で mdl_cuda_infer_batch_submit/_collect(ビット列経路)が
    生成され、バッファがn_streams本の配列宣言になっていることを確認する。"""
    bundle = build_bundle(dense_doc)
    src = generate_cuda(bundle, batch=4, n_streams=3)['model_cuda.cu']
    assert 'int mdl_cuda_infer_batch_submit(' in src
    assert 'int mdl_cuda_infer_batch_collect(' in src
    assert 'static int mdl_model_encode(int slot, int n) {' in src
    assert 'static int mdl_cuda_run(int *scores, int n, int slot) {' in src
    assert 'mdl_cuda_streams_init(3)' in src
    assert 'mdl_cuda_streams_shutdown();' in src
    assert 'mdl_cuda_select_stream(slot)' in src
    assert 'mdl_cuda_select_default_stream();' in src
    assert 'mdl_cuda_commit_async(slot)' in src
    assert 'mdl_cuda_wait_slot(slot)' in src
    # 既存の非ストリーミング入口は「そのまま同じ形」で残る(常にslot0)。
    assert 'int mdl_cuda_infer_batch(const unsigned char *input_bits' in src
    assert 'int mdl_cuda_infer(const unsigned char *input_bits' in src
    assert 'return mdl_cuda_run(scores, n, 0);' in src


def test_n_streams_image_path_generates_submit_collect(seg_doc):
    """画像入力(preproc)ありのモデルでは image 版 submit/collect も
    追加生成される(Metal版と同じ「両方(bits+image)を持つ」構造)。
    ビット列経路の方は前処理付きモデルでも mdl_cuda_pack を明示的に呼ぶ
    (前処理をバイパスして既パック済みbitsを渡す代替入口のため)。"""
    bundle = build_bundle(seg_doc)
    src = generate_cuda(bundle, batch=4, n_streams=2)['model_cuda.cu']
    assert 'int mdl_cuda_infer_image_batch_submit(' in src
    assert 'int mdl_cuda_infer_image_batch_collect(' in src
    assert 'int mdl_cuda_infer_batch_submit(' in src
    # ビット列streaming submitでも前処理ありモデルはpackを呼ぶこと
    # (呼ばないとbitplaneバッファが前回の内容のまま=壊れるバグになる)。
    assert 'mdl_cuda_pack(d_mdl_input[slot], d_mdl_preproc_out[slot]' in src


def test_dense_fuse_48kb_cuda_only(dense_doc):
    """CUDAのdense-fuse共有メモリ上限はMetalの32KBではなく48KB
    (`CUDA_DENSE_FUSE_MAX_SHARED_BYTES`)を使う。dense_docは小さいので
    どちらの上限でも融合されるが、少なくとも --dense-fuse 指定時に生成が
    壊れないことと、融合カーネルが実際に生成されることを確認する
    (32KB/48KBの境界にかかるケースは下の
    test_dense_fuse_32kb_48kb_gap_is_backend_specific で確認)。"""
    bundle = build_bundle(dense_doc)
    src = generate_cuda(bundle, batch=32, dense_fuse=True)['model_cuda.cu']
    assert 'mdl_k_densefuse_' in src


def test_dense_fuse_32kb_48kb_gap_is_backend_specific(webbench):
    """本命の確認: 2*act_bytesが32KBと48KBの間に落ちる鎖は、
    Metal既定(32KB)では融合されずCUDA(48KB、`generate_cuda`が明示的に
    `CUDA_DENSE_FUSE_MAX_SHARED_BYTES`を渡す)では融合される。

    **注記(実測訂正)**: 当初「mnist_adder(層幅45.9KB)は32KB縛り
    だけで融合対象外になっている」という前提で48KB化を計画したが、
    実際に`results/compile-pipeline/mnist_adder_s0_20260712_092520_
    ir_final.json`をbuild_plan(dense_fuse=True)へ通すと、対象の鎖
    (n0_pb_pc_ge→n1_final→group_sum)は2*act_bytes=2*45900=91,800B
    (=約89.6KB)を要求し、**32KBはもちろん48KBにも収まらない**
    (48KBで解禁されるという前提は誤りだった)。そのためこのテストでは、
    32KB〜48KBの間に正確に収まる合成モデル(2*act_bytes=40,000B)で
    「48KB化の配線が機能している」ことを確認する。
    """
    graph = webbench.build_dense_ir(seed=3, in_width=64,
                                    widths=[20000, 20000], k=4,
                                    name='k55_gap')
    doc = webbench.build_model_doc(
        graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=['a', 'b', 'c', 'd'])
    bundle = build_bundle(doc)
    # Metal既定(32KB相当)は generate_metal 自体を呼ばずに
    # build_plan(dense_fuse_max_shared_bytes未指定=32KB既定)で代用する
    # (このファイルはCUDA専用テストなのでmetalbackendはimportしない)。
    from griffith_logic.deploy.plan import build_plan

    plan_32kb = build_plan(bundle, batch=1, dense_fuse=True)
    assert 'dense_fuse' not in [op.kind for op in plan_32kb.ops]

    src = generate_cuda(bundle, batch=1, dense_fuse=True)['model_cuda.cu']
    assert 'mdl_k_densefuse_' in src
    # 融合されたことの直接証拠: 個別lut_layer呼び出し(mdl_cuda_lut_layer)
    # がこの鎖ぶん残っていない(1つの融合カーネル呼び出しへ差し替わって
    # いる)。
    assert src.count('mdl_cuda_lut_layer(') == 0


# ---------------------------------------------------------------------------
# ビルド+実行(nvccがある環境でのみ。実機検証はリモートSSH機で
# scripts/verify_cuda_stream_remote.py を使って別途実施した)
# ---------------------------------------------------------------------------

pytestmark_build = pytest.mark.skipif(
    not has_cuda(), reason='nvcc のある環境でのみ実行(実機検証は'
                          'リモートSSH機で別途実施)')


def _nvcc_build(bundle: dict, work_dir: Path, *, batch: int, n_streams: int,
                extra_driver_source: str, driver_name: str) -> Path:
    for filename, text in generate_cuda(
            bundle, name='model_cuda', batch=batch,
            n_streams=n_streams).items():
        (work_dir / filename).write_text(text, encoding='utf-8')
    (work_dir / driver_name).write_text(extra_driver_source,
                                        encoding='utf-8')
    for rel in ('mdl_cuda.h', 'mdl_cuda.cu'):
        (work_dir / rel).write_text(
            (RUNTIME_CUDA / rel).read_text(encoding='utf-8'),
            encoding='utf-8')
    (work_dir / 'mdl_gpu_kernels.h').write_text(
        (RUNTIME_GPU / 'mdl_gpu_kernels.h').read_text(encoding='utf-8'),
        encoding='utf-8')
    exe = work_dir / 'app'
    build = subprocess.run(
        [_nvcc(), '-O2', '-arch=sm_89', '-I', str(work_dir),
         str(work_dir / 'model_cuda.cu'), str(work_dir / 'mdl_cuda.cu'),
         str(work_dir / driver_name), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f'ビルド失敗:\n{build.stderr}')
    return exe


_BITS_DRIVER_TMPL = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {{
int mdl_cuda_infer_batch(const unsigned char*, int*, int);
int mdl_cuda_infer_batch_submit(const unsigned char*, int, int);
int mdl_cuda_infer_batch_collect(int*, int, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_max_batch(void);
int mdl_cuda_init(void);
}}

#define K {k}
#define NSLOT {nslot}

static const int NS[K] = {{{ns_csv}}};

int main(void) {{
  const int in_w = mdl_cuda_input_width();
  const int out_w = mdl_cuda_output_width();
  const int maxb = mdl_cuda_max_batch();
  unsigned char *inputs[K];
  int *seq[K];
  int *strm[K];
  int k;
  size_t i;

  if (mdl_cuda_init() != 0) {{ fprintf(stderr, "init failed\\n"); return 3; }}

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
    if (mdl_cuda_infer_batch(inputs[k], seq[k], NS[k]) != 0) {{
      fprintf(stderr, "seq infer failed k=%d\\n", k); return 2;
    }}
  }}

  {{
    int pending = 0, next_submit = 0, next_collect = 0;
    while (next_collect < K) {{
      while (next_submit < K && pending < NSLOT) {{
        const int slot = next_submit % NSLOT;
        if (mdl_cuda_infer_batch_submit(inputs[next_submit], NS[next_submit],
                                        slot) != 0) {{
          fprintf(stderr, "submit failed k=%d\\n", next_submit); return 2;
        }}
        next_submit++; pending++;
      }}
      {{
        const int slot = next_collect % NSLOT;
        if (mdl_cuda_infer_batch_collect(strm[next_collect], NS[next_collect],
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

_GUARD_DRIVER = '''#include "mdl_cuda.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
int mdl_cuda_infer_batch_submit(const unsigned char*, int, int);
int mdl_cuda_infer_batch_collect(int*, int, int);
int mdl_cuda_input_width(void);
int mdl_cuda_output_width(void);
int mdl_cuda_init(void);
}

int main(void) {
  const int in_w = mdl_cuda_input_width();
  const int out_w = mdl_cuda_output_width();
  unsigned char *bits = (unsigned char*)calloc((size_t)in_w, 1);
  int *scores = (int*)malloc((size_t)out_w * sizeof(int));
  int rc1, rc2;

  if (mdl_cuda_init() != 0) return 3;

  rc1 = mdl_cuda_infer_batch_submit(bits, 1, 0);
  /* collectせずに同じスロットへもう一度submitする(警告する
   * 「バッファ共有の誤り」そのものの操作)。ガードが働いて-1を返す
   * べき(黙って前回ぶんの入力バッファを上書きしてはいけない)。 */
  rc2 = mdl_cuda_infer_batch_submit(bits, 1, 0);
  printf("rc1=%d rc2=%d\\n", rc1, rc2);
  if (rc1 != 0 || rc2 == 0) {
    fprintf(stderr, "ガードが機能していない(2回目もOKを返した)\\n");
    return 1;
  }
  if (mdl_cuda_infer_batch_collect(scores, 1, 0) != 0) return 2;
  printf("OK\\n");
  return 0;
}
'''


@pytest.mark.slow
@pytestmark_build
class TestCudaStreamMatchesSequential:
    """N回連続ストリーム ⇔ 1枚(1回)ずつN回、が全て一致する
    (nvccのある環境でのみ実行)。"""

    def test_dense_bits_path(self, dense_doc, tmp_path):
        import numpy as np

        bundle = build_bundle(dense_doc)
        rng = np.random.default_rng(0)
        batch = 6
        k, nslot = 24, 3
        ns = rng.integers(1, batch + 1, size=k).tolist()
        src = _BITS_DRIVER_TMPL.format(
            k=k, nslot=nslot, ns_csv=','.join(str(n) for n in ns))
        exe = _nvcc_build(bundle, tmp_path, batch=batch, n_streams=nslot,
                          extra_driver_source=src, driver_name='driver.cu')
        proc = subprocess.run([str(exe)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'OK' in proc.stdout


@pytest.mark.slow
@pytestmark_build
def test_commit_async_rejects_unreleased_slot(dense_doc, tmp_path):
    """collectを挟まずに同じスロットへ二重commit_asyncすると拒否される
    (nvccのある環境でのみ実行、Metal版の同名テストと対称)。"""
    bundle = build_bundle(dense_doc)
    exe = _nvcc_build(bundle, tmp_path, batch=1, n_streams=2,
                      extra_driver_source=_GUARD_DRIVER,
                      driver_name='guard_driver.cu')
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout
