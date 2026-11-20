"""RDDLGN自己回帰生成(decode)のGPUドライバ検証。

`deploy/rddlgn_decode_gpu.py` が生成する `mdl_metal_rddlgn_decode`
(Metal)を、CPU golden(`export/rddlgn_ir.py::simulate_rddlgn_greedy_bits`。
`tests/deploy/test_c_rddlgn_greedy.py` が使うのと同じ関数)と突き合わせる。

GPUのdecodeは**EOS判定をデバイス側で行わない**(design方針、
rddlgn_decode_gpu.pyのモジュールdocstring参照)。全レーンが常に
`steps`(=IRのtgt_len)ぶん走り切るため、CPU golden(EOSで早期終了)との
比較は「goldenの長さぶんのプレフィックスが一致するか」で行う
(EOS以降はCPU側が生成を打ち切っているのでもう突き合わせようがない)。

CUDAは`generate_rddlgn_decode_cuda`のソース生成のみこのファイルで検証する
(実行はこのMacにCUDA実機が無いため。`scripts/bench_rddlgn_decode_gpu.py`が
リモートでの実行検証を担う)。
"""

import platform
import shutil
import subprocess

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.rddlgn_decode_gpu import (
    generate_rddlgn_decode_cuda,
    generate_rddlgn_decode_metal,
)
from griffith_logic.export.rddlgn_ir import (
    extract_embed_table,
    simulate_rddlgn_greedy_bits,
)

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

RUNTIME = RUNTIME_C.parent
METAL_RUNTIME = RUNTIME / 'metal'

DRIVER = r'''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_rddlgn_decode(const unsigned char*, int, int*, int);
int mdl_metal_rddlgn_decode_steps(void);
}
/* 引数: [枚数n] [src_width] [bos] [batch容量]。stdinからn*src_width
   バイトのsrc_bitsを読み、[steps][batch容量]のint32を吐く
   (Pythonテスト側がn列ぶんだけ切り出す)。 */
int main(int argc, char** argv) {
  int steps = mdl_metal_rddlgn_decode_steps();
  int n = atoi(argv[1]);
  int src_width = atoi(argv[2]);
  int bos = atoi(argv[3]);
  int batch_cap = atoi(argv[4]);
  unsigned char* src = (unsigned char*)malloc((size_t)n * src_width);
  int* out = (int*)malloc((size_t)steps * batch_cap * sizeof(int));
  if (fread(src, 1, (size_t)n * src_width, stdin) != (size_t)n * src_width) {
    return 1;
  }
  if (mdl_metal_rddlgn_decode(src, bos, out, n) != 0) return 2;
  fwrite(out, sizeof(int), (size_t)steps * batch_cap, stdout);
  return 0;
}
'''


def _has_metal() -> bool:
    if platform.system() != 'Darwin':
        return False
    return shutil.which('clang++') is not None and METAL_RUNTIME.is_dir()


pytestmark = pytest.mark.skipif(
    not _has_metal(), reason='macOS + Metal のある環境でのみ実行')


def _embed_src_bits(model, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()
    return embed[tokens]


@pytest.fixture(scope='module')
def decode_model():
    return build_rddlgn_tiny_seq2seq(seed=0)


@pytest.fixture(scope='module')
def decode_bundle(decode_model):
    model, cfg = decode_model
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


def _build_decode_app(tmp_path, bundle, batch):
    files = generate_rddlgn_decode_metal(bundle, name='model_metal', batch=batch)
    work = tmp_path / f'decode_gen_b{batch}'
    work.mkdir(parents=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.mm').write_text(DRIVER, encoding='utf-8')
    build = subprocess.run(
        ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{METAL_RUNTIME}', '-framework', 'Metal', '-framework', 'Foundation',
         str(work / 'model_metal.mm'), str(METAL_RUNTIME / 'mdl_metal.mm'),
         str(work / 'driver.mm'), '-o', str(work / 'app')],
        capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f'Metalのビルド不可: {build.stderr[:500]}')
    return work / 'app'


def _run_decode(app, src_bits: np.ndarray, bos_token: int, steps: int,
                batch_cap: int) -> np.ndarray:
    """src_bits: (n, src_width) uint8。戻り値: (n, steps) int32(GPU出力)。"""
    n, src_width = src_bits.shape
    proc = subprocess.run(
        [str(app), str(n), str(src_width), str(bos_token), str(batch_cap)],
        input=src_bits.tobytes(), capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f'Metal実行不可(GPU無し等): '
                   f'{proc.stderr.decode(errors="replace")[:300]}')
    out = np.frombuffer(proc.stdout, dtype='<i4').reshape(steps, batch_cap)
    return out[:, :n].T  # -> (n, steps)


class TestMetalDecodeMatchesGolden:
    """generate_rddlgn_decode_metalの出力トークン列が、CPU golden
    (simulate_rddlgn_greedy_bits)のプレフィックスと全数一致する
    (核心の検証)。"""

    @pytest.mark.slow
    @pytest.mark.parametrize('batch', [1, 8, 32])
    def test_random_samples_match_prefix(self, tmp_path, decode_model,
                                         decode_bundle, batch):
        model, cfg = decode_model
        app = _build_decode_app(tmp_path, decode_bundle, batch)
        rng = np.random.default_rng(11 + batch)

        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)

        goldens = []
        for b in range(batch):
            g = simulate_rddlgn_greedy_bits(
                model, src_bits[b], bos_token=bos_token, eos_token=999999,
                max_new_tokens=cfg.seq_len)
            goldens.append(g)

        got = _run_decode(app, src_bits.astype(np.uint8), bos_token,
                          cfg.seq_len, batch)
        for b in range(batch):
            got_prefix = got[b, :len(goldens[b])].tolist()
            assert got_prefix == goldens[b], (
                f'lane={b}: got={got_prefix} want={goldens[b]}')

    @pytest.mark.slow
    def test_extreme_src_bits_match(self, tmp_path, decode_model, decode_bundle):
        """全0/全1の埋め込みビット列でも一致する(端数ビットの扱い含む、
        test_c_rddlgn_greedy.pyの同名テストとの対称)。"""
        model, cfg = decode_model
        app = _build_decode_app(tmp_path, decode_bundle, batch=2)
        d_emb = model.src_embedding.embedding.embedding_dim
        src_width = cfg.seq_len * d_emb
        src_bits = np.stack([
            np.zeros(src_width, dtype=np.uint8),
            np.ones(src_width, dtype=np.uint8),
        ])
        bos_token = 0
        goldens = [
            simulate_rddlgn_greedy_bits(
                model, src_bits[b], bos_token=bos_token, eos_token=999999,
                max_new_tokens=cfg.seq_len)
            for b in range(2)
        ]
        got = _run_decode(app, src_bits, bos_token, cfg.seq_len, batch_cap=2)
        for b in range(2):
            assert got[b, :len(goldens[b])].tolist() == goldens[b]

    @pytest.mark.slow
    def test_multiple_lanes_independent_srcs(self, tmp_path, decode_model,
                                              decode_bundle):
        """同一呼び出しの複数レーンが互いに干渉しない
        (B系列レーン並列の基本性質)。"""
        model, cfg = decode_model
        batch = 16
        app = _build_decode_app(tmp_path, decode_bundle, batch)
        rng = np.random.default_rng(999)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)
        goldens = [
            simulate_rddlgn_greedy_bits(
                model, src_bits[b], bos_token=bos_token, eos_token=999999,
                max_new_tokens=cfg.seq_len)
            for b in range(batch)
        ]
        got = _run_decode(app, src_bits.astype(np.uint8), bos_token,
                          cfg.seq_len, batch)
        mismatches = [b for b in range(batch)
                     if got[b, :len(goldens[b])].tolist() != goldens[b]]
        assert not mismatches, f'一致しないレーン: {mismatches}'


class TestCudaDecodeSourceGeneration:
    """CUDA側はソース生成のみ検証する(このMacに実機が無いため。
    実行検証はscripts/bench_rddlgn_decode_gpu.py --backend cuda がリモートで
    行う)。"""

    def test_generates_decode_entry_symmetric_to_metal(self, decode_bundle):
        """のtickループ経路(`megakernel=False`)固有のシンボル構造を見る
        テストなので明示的に`megakernel=False`を渡す(megakernelの
        既定値がFalse→True(fits時)へ変わったため、既定のままだとこの
        モデルはmegakernel経路になり`mdl_cuda_embed_lookup`/
        `mdl_cuda_argmax`という個別呼び出しが出なくなる)。"""
        files = generate_rddlgn_decode_cuda(decode_bundle, batch=4,
                                            megakernel=False)
        assert 'model_cuda.cu' in files
        text = files['model_cuda.cu']
        assert 'mdl_cuda_rddlgn_decode' in text
        assert 'mdl_k_embed_lookup' not in text  # 共有カーネル側(mdl_cuda.cu本体)にある
        assert 'mdl_cuda_embed_lookup' in text
        assert 'mdl_cuda_argmax' in text

    def test_requires_embed_table(self):
        from griffith_logic.deploy.bundle import build_bundle as _bb
        from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
        from griffith_logic.ir.serialize import ir_to_dict
        from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

        torch.manual_seed(0)
        cfg = RDDLGNConfig.tiny()
        model = RDDLGN(cfg)
        model.eval()
        graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len)
        doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
        bundle_no_embed = _bb(doc)
        with pytest.raises(ValueError, match='embed_table'):
            generate_rddlgn_decode_cuda(bundle_no_embed, batch=1)
        with pytest.raises(ValueError, match='embed_table'):
            generate_rddlgn_decode_metal(bundle_no_embed, batch=1)
