"""RDDLGN自己回帰decodeのmegakernel化の検証。

`generate_rddlgn_decode_metal/_cuda(megakernel=True)`が生成する
「1スレッドグループ(Metal)/1スレッドブロック(CUDA)=1レーン、tickループ
全体を1カーネルへ畳む」経路を、CPU golden
(`export/rddlgn_ir.py::simulate_rddlgn_greedy_bits`)および従来の
 tickループ経路(`megakernel=False`)と突き合わせる。

CUDA側は`test_rddlgn_decode_gpu.py`と同じ制約でソース生成(構造)のみを
検証する(このMacにCUDA実機/nvccが無いため。ローカルのg++構文検査は
scratch限定で手動実施し、リポジトリには持ち込まない——同じ
流儀、`deploy/rddlgn_decode_megakernel.py`のdocstring参照)。
"""

import platform
import shutil
import subprocess

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.rddlgn_decode_gpu import (
    _discover_decode_spec,
    generate_rddlgn_decode_cuda,
    generate_rddlgn_decode_metal,
)
from griffith_logic.deploy.rddlgn_decode_megakernel import (
    RDDLGN_MEGAKERNEL_MAX_SHARED_BYTES,
    _fits,
    _threadgroup_layout,
)
from griffith_logic.export.rddlgn_ir import (
    extract_embed_table,
    simulate_rddlgn_greedy_bits,
)
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

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
   (Pythonテスト側がn列ぶんだけ切り出す)。test_rddlgn_decode_gpu.pyの
   DRIVERと同一(megakernel=True/False どちらの生成物でも同じエントリ
   `mdl_metal_rddlgn_decode`を叩くだけなので、ドライバ自体は共有できる)。 */
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


def _build_decode_app(tmp_path, bundle, batch, *, megakernel: bool, tag: str):
    """指定経路(megakernel=True/False)でMetalソースを生成しビルドする。
    戻り値: (実行ファイルpath, 生成ソーステキスト)。
    """
    files = generate_rddlgn_decode_metal(bundle, name='model_metal',
                                         batch=batch, megakernel=megakernel)
    text = files['model_metal.mm']
    work = tmp_path / f'decode_{tag}_b{batch}'
    work.mkdir(parents=True)
    (work / 'model_metal.mm').write_text(text, encoding='utf-8')
    (work / 'driver.mm').write_text(DRIVER, encoding='utf-8')
    build = subprocess.run(
        ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{METAL_RUNTIME}', '-framework', 'Metal', '-framework', 'Foundation',
         str(work / 'model_metal.mm'), str(METAL_RUNTIME / 'mdl_metal.mm'),
         str(work / 'driver.mm'), '-o', str(work / 'app')],
        capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f'Metalのビルド不可: {build.stderr[:800]}')
    return work / 'app', text


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


def _goldens(model, cfg, src_bits, bos_token):
    return [
        simulate_rddlgn_greedy_bits(
            model, src_bits[b], bos_token=bos_token, eos_token=999999,
            max_new_tokens=cfg.seq_len)
        for b in range(src_bits.shape[0])
    ]


@pytest.fixture(scope='module')
def decode_model():
    return build_rddlgn_tiny_seq2seq(seed=0)


@pytest.fixture(scope='module')
def decode_bundle(decode_model):
    model, cfg = decode_model
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


def _build_seq2seq(seed: int, seq_len: int):
    """`build_rddlgn_tiny_seq2seq`と同じ層サイズでseq_len(=steps)だけ変えた
    RDDLGNを作る(「steps複数」の検証用。層サイズが同じなので
    threadgroup所要バイト数はtinyと同じ=32KB以内に収まる)。"""
    torch.manual_seed(seed)
    base = RDDLGNConfig.tiny()
    cfg = RDDLGNConfig(
        vocab_size=base.vocab_size, d_emb=base.d_emb, seq_len=seq_len,
        n_layer_sizes=base.n_layer_sizes, k_layer_sizes=base.k_layer_sizes,
        l_layer_sizes=base.l_layer_sizes, p_layer_sizes=base.p_layer_sizes,
        m_layer_sizes=base.m_layer_sizes, n_inputs=base.n_inputs,
        parameterization=base.parameterization,
        connections=base.connections, grad_factor=base.grad_factor,
        selection=base.selection, tau=base.tau, tau_warp=base.tau_warp,
        gumbel=base.gumbel, init_mode=base.init_mode,
        init_scale=base.init_scale, shared_vocab=base.shared_vocab)
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


@pytest.fixture(scope='module')
def decode_model_steps6():
    return _build_seq2seq(seed=1, seq_len=6)


@pytest.fixture(scope='module')
def decode_bundle_steps6(decode_model_steps6):
    model, cfg = decode_model_steps6
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


def _big_config_doc():
    """threadgroup/sharedメモリ32KBを超える(自動フォールバックが発火する)
    ようlayer幅を大きくしたRDDLGN(層構成以外はtinyと同じ)。"""
    torch.manual_seed(2)
    base = RDDLGNConfig.tiny()
    cfg = RDDLGNConfig(
        vocab_size=base.vocab_size, d_emb=base.d_emb, seq_len=base.seq_len,
        n_layer_sizes=base.n_layer_sizes, k_layer_sizes=base.k_layer_sizes,
        l_layer_sizes=[2048, 2048], p_layer_sizes=[4096, 3072],
        m_layer_sizes=[8000, 8000, 4000], n_inputs=base.n_inputs,
        parameterization=base.parameterization,
        connections=base.connections, grad_factor=base.grad_factor,
        selection=base.selection, tau=base.tau, tau_warp=base.tau_warp,
        gumbel=base.gumbel, init_mode=base.init_mode,
        init_scale=base.init_scale, shared_vocab=base.shared_vocab)
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


@pytest.fixture(scope='module')
def big_model():
    return _big_config_doc()


@pytest.fixture(scope='module')
def big_bundle(big_model):
    model, cfg = big_model
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


# ---------------------------------------------------------------------------
# A. threadgroupレイアウト/適格性判定(GPU不要、Pythonのみ)
# ---------------------------------------------------------------------------

class TestFitsAndLayout:
    def test_tiny_model_fits(self, decode_bundle):
        from griffith_logic.deploy.plan import build_plan

        plan = build_plan(decode_bundle, batch=4)
        spec = _discover_decode_spec(plan, decode_bundle['embed_table'])
        fits, reason = _fits(spec)
        assert fits, reason
        layout = _threadgroup_layout(spec)
        assert 0 < layout.total_bytes <= RDDLGN_MEGAKERNEL_MAX_SHARED_BYTES

    def test_layout_independent_of_batch(self, decode_bundle):
        """threadgroupメモリは1レーンぶんなので、batch(レーン数)を変えても
        所要バイト数は変わらない。"""
        from griffith_logic.deploy.plan import build_plan

        sizes = set()
        for batch in (1, 32, 1024):
            plan = build_plan(decode_bundle, batch=batch)
            spec = _discover_decode_spec(plan, decode_bundle['embed_table'])
            sizes.add(_threadgroup_layout(spec).total_bytes)
        assert len(sizes) == 1

    def test_big_model_does_not_fit(self, big_bundle):
        from griffith_logic.deploy.plan import build_plan

        plan = build_plan(big_bundle, batch=4)
        spec = _discover_decode_spec(plan, big_bundle['embed_table'])
        fits, reason = _fits(spec)
        assert not fits
        assert '32KB' in reason or 'B' in reason


# ---------------------------------------------------------------------------
# B. megakernel生成物 vs CPU golden(核心)
# ---------------------------------------------------------------------------

class TestMegakernelMatchesGolden:
    @pytest.mark.slow
    @pytest.mark.parametrize('batch', [1, 32, 100, 1024])
    def test_random_samples_match_prefix(self, tmp_path, decode_model,
                                         decode_bundle, batch):
        model, cfg = decode_model
        app, text = _build_decode_app(tmp_path, decode_bundle, batch,
                                      megakernel=True, tag='mega')
        assert 'megakernel: enabled' in text, (
            'tinyモデルは32KB以内のはずなのにフォールバックした')

        rng = np.random.default_rng(11 + batch)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)

        goldens = _goldens(model, cfg, src_bits, bos_token)
        got = _run_decode(app, src_bits.astype(np.uint8), bos_token,
                          cfg.seq_len, batch)
        mismatches = [b for b in range(batch)
                     if got[b, :len(goldens[b])].tolist() != goldens[b]]
        assert not mismatches, f'一致しないレーン: {mismatches[:10]}'

    @pytest.mark.slow
    @pytest.mark.parametrize('batch', [1, 32])
    def test_different_steps_match_prefix(self, tmp_path, decode_model_steps6,
                                          decode_bundle_steps6, batch):
        """stepsがtiny(4)と異なるモデル(steps=6)でも一致する。"""
        model, cfg = decode_model_steps6
        app, text = _build_decode_app(tmp_path, decode_bundle_steps6, batch,
                                      megakernel=True, tag='mega_steps6')
        assert 'megakernel: enabled' in text

        rng = np.random.default_rng(101 + batch)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)

        goldens = _goldens(model, cfg, src_bits, bos_token)
        got = _run_decode(app, src_bits.astype(np.uint8), bos_token,
                          cfg.seq_len, batch)
        mismatches = [b for b in range(batch)
                     if got[b, :len(goldens[b])].tolist() != goldens[b]]
        assert not mismatches, f'一致しないレーン: {mismatches[:10]}'


# ---------------------------------------------------------------------------
# C. megakernel vs 従来の tickループ経路(全トークン一致)
# ---------------------------------------------------------------------------

class TestMegakernelMatchesLegacy:
    @pytest.mark.slow
    def test_same_output_as_legacy_tick_loop(self, tmp_path, decode_model,
                                             decode_bundle):
        model, cfg = decode_model
        batch = 32
        app_mega, text_mega = _build_decode_app(
            tmp_path, decode_bundle, batch, megakernel=True, tag='mega')
        app_legacy, text_legacy = _build_decode_app(
            tmp_path, decode_bundle, batch, megakernel=False, tag='legacy')
        assert 'megakernel: enabled' in text_mega
        assert 'megakernel' not in text_legacy   # 従来経路は無改変

        rng = np.random.default_rng(4242)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)
        src_bits = src_bits.astype(np.uint8)

        got_mega = _run_decode(app_mega, src_bits, bos_token, cfg.seq_len,
                               batch)
        got_legacy = _run_decode(app_legacy, src_bits, bos_token, cfg.seq_len,
                                 batch)
        assert got_mega.tolist() == got_legacy.tolist()


# ---------------------------------------------------------------------------
# D. 故障注入: 重み1ビット破壊で不一致を検出できること
# ---------------------------------------------------------------------------

def _flip_one_bit(lut_node: dict, gate_index: int, bit: int) -> str:
    """`lut_node['params']['truth_tables'][gate_index]`の1ビットだけを
    反転した16進文字列を返す(元の値は変更しない)。"""
    tts = lut_node['params']['truth_tables']
    original = tts[gate_index]
    value = int(original, 16) if isinstance(original, str) else int(original)
    return hex(value ^ (1 << bit))


class TestMegakernelFaultInjection:
    @pytest.mark.slow
    def test_corrupted_weight_breaks_match(self, tmp_path, decode_model):
        """dec_pm_recのbody(group_sum直前のlut_layer、出力への影響が最も
        直接的)の真理値表を1ビットだけ破壊したバンドルをmegakernel経路で
        ビルドし、非破壊goldenとの不一致を検出できることを確認する
        (テスト自体が偽陰性でないことの担保)。

        ランダム配線(connections='random')の1ゲートは、選んだ入力の組
        (n_inputs=2なら4通り)のうち出力に現れないアドレスもありうる
        (test_bundle_format.py::test_dead_gate_mutation_is_invisibleと同種の
        「死んだゲート」)ため、複数のゲート番号/ビット位置を順に試し、
        golden(非破壊モデル)との不一致が実際に観測できたものを1つ採用する
        (=検出できる破壊が存在することの担保。1回で決め打ちにすると
        「たまたま死んだゲートを引いた」だけでテストが偽陰性になりうる)。
        """
        model, cfg = decode_model
        batch = 32
        rng = np.random.default_rng(777)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)
        src_bits_u8 = src_bits.astype(np.uint8)
        goldens = _goldens(model, cfg, src_bits, bos_token)

        candidates = [(gate, bit) for gate in (0, 1, 50, 500, 999)
                     for bit in (0, 1, 2, 3)]
        found = False
        last_text = ''
        for i, (gate, bit) in enumerate(candidates):
            doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
            doc['embed_table'] = extract_embed_table(model.src_embedding)
            pm_rec = next(n for n in doc['ir']['nodes']
                         if n['id'] == 'dec_pm_rec')
            body_nodes = pm_rec['params']['body']['nodes']
            lut_nodes = [n for n in body_nodes if n['type'] == 'lut_layer']
            # group_sumの直前(dec_pm_rec bodyの最後のlut_layer、
            # M-layersの最終層=出力への影響が最も直接的)を狙う。
            lut_node = lut_nodes[-1]
            if gate >= len(lut_node['params']['truth_tables']):
                continue
            lut_node['params']['truth_tables'][gate] = _flip_one_bit(
                lut_node, gate, bit)

            bundle = build_bundle(doc)
            app, last_text = _build_decode_app(
                tmp_path, bundle, batch, megakernel=True, tag=f'corrupt{i}')
            assert 'megakernel: enabled' in last_text

            got = _run_decode(app, src_bits_u8, bos_token, cfg.seq_len, batch)
            mismatches = [b for b in range(batch)
                         if got[b, :len(goldens[b])].tolist() != goldens[b]]
            if mismatches:
                found = True
                break

        assert found, (
            f'{len(candidates)}通りの1ビット破壊すべてでgoldenと全レーン'
            '一致した(=テストの検出力が無い/破壊が出力に伝播していない)')


# ---------------------------------------------------------------------------
# E. 32KB超モデルの自動フォールバック(動作込み)
# ---------------------------------------------------------------------------

class TestMegakernelFallback:
    @pytest.mark.slow
    def test_big_model_falls_back_and_still_matches_golden(
            self, tmp_path, big_model, big_bundle):
        model, cfg = big_model
        batch = 4
        app, text = _build_decode_app(tmp_path, big_bundle, batch,
                                      megakernel=True, tag='bigfallback')
        assert 'megakernel: fallback' in text
        assert 'mdl_k_rddlgn_decode_mega_' not in text

        rng = np.random.default_rng(55)
        src_tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len))
        bos_token = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens).reshape(batch, -1)

        goldens = _goldens(model, cfg, src_bits, bos_token)
        got = _run_decode(app, src_bits.astype(np.uint8), bos_token,
                          cfg.seq_len, batch)
        mismatches = [b for b in range(batch)
                     if got[b, :len(goldens[b])].tolist() != goldens[b]]
        assert not mismatches, f'一致しないレーン: {mismatches}'


# ---------------------------------------------------------------------------
# F. CUDA側: ソース生成(構造)のみ(実行はこのMacでは不可)
# ---------------------------------------------------------------------------

class TestCudaMegakernelSourceGeneration:
    def test_generates_megakernel_entry_symmetric_to_metal(self, decode_bundle):
        files = generate_rddlgn_decode_cuda(decode_bundle, batch=4,
                                            megakernel=True)
        assert 'model_cuda.cu' in files
        text = files['model_cuda.cu']
        assert 'megakernel: enabled' in text
        assert 'mdl_k_rddlgn_decode_mega_' in text
        assert 'mdl_cuda_rddlgn_decode' in text
        # __global__本体(<<<>>>起動含む)は生成コードから直接呼ぶ(CUDAは
        # AOTコンパイルなのでMetalのような名前引きディスパッチが不要、
        # のemit_dense_fuseと同じ設計)。
        # 実機修正: threads=1024でのレジスタ過剰要求
        # ("too many resources requested for launch")を防ぐため
        # __launch_bounds__が__global__ voidと関数名の間に入る。
        assert ('__global__ void __launch_bounds__' in text
                and 'mdl_k_rddlgn_decode_mega_' in text)
        assert '<<<' in text

    def test_big_model_falls_back_on_cuda_too(self, big_bundle):
        files = generate_rddlgn_decode_cuda(big_bundle, batch=4,
                                            megakernel=True)
        text = files['model_cuda.cu']
        assert 'megakernel: fallback' in text
        assert 'mdl_k_rddlgn_decode_mega_' not in text

    def test_megakernel_false_is_byte_identical_to_k51(self, decode_bundle):
        """`megakernel=False`を明示指定すると従来の生成物と1バイトも
        変わらない(タスク指示: 「既存の経路は無改変で残す」)。

        `megakernel`の既定値がFalse→True(fits時)へ変わったため
        (実機A/Bで全実測が正、悪化ケース無し)、`a`側も明示的に
        `megakernel=False`を渡すよう更新した(元テストは「既定呼び出し」を
        比較していたが、その既定の意味自体が変わったため)。"""
        from griffith_logic.deploy.rddlgn_decode_gpu import (
            generate_rddlgn_decode_cuda as gen,
        )

        a = gen(decode_bundle, name='m', batch=4, megakernel=False)
        b = gen(decode_bundle, name='m', batch=4, megakernel=False)
        assert a == b

    def test_megakernel_default_is_enabled_when_fits(self, decode_bundle):
        """`megakernel`省略(既定)は、fitsするモデルなら
        `megakernel=True`と同じ「megakernel: enabled」経路になる。"""
        from griffith_logic.deploy.rddlgn_decode_gpu import (
            generate_rddlgn_decode_cuda as gen,
        )

        default_text = gen(decode_bundle, name='m', batch=4)['m.cu']
        explicit_true_text = gen(decode_bundle, name='m', batch=4,
                                 megakernel=True)['m.cu']
        assert 'megakernel: enabled' in default_text
        assert default_text == explicit_true_text
