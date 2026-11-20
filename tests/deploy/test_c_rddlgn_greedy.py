"""RDDLGN 自己回帰生成(greedy decode)の3者一致検証(続き)。

tests/deploy/test_c_rddlgn_seq2seq.py(教師強制)の続き。教師強制と違い、
自己回帰生成は「前ステップの出力トークンをargmaxで選び、埋め込み直して
次ステップの入力にする」ループを持つ。IRには乗らない(blackboxと同種の
線引き、デプロイキット設計メモ節)ので、手書きハーネス
runtime/c/src/mdl_rddlgn.c + Python側の2つの参照実装
(models/rddlgn.py::RDDLGN.generate_greedy、export/rddlgn_ir.py::
simulate_rddlgn_greedy)の3者一致を確かめる:

    PyTorch greedy(model.eval()、1トークンずつ)
        ⇔ ir/simulateベースのgreedy(golden)
        ⇔ C(mdl_rddlgn_generate_greedy、deploy/cbackend.py::generate_rddlgn_c)
"""

import struct
import subprocess

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cbackend import generate_rddlgn_c
from griffith_logic.export.rddlgn_ir import (
    extract_embed_table,
    simulate_rddlgn_greedy,
    simulate_rddlgn_greedy_bits,
)

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

# 生成ハーネスとの単純なバイナリプロトコルで動くドライバ。1回の起動で
# 複数クエリ(src_bits + bos + eos + max_new_tokens、いずれもu32LE)を
# 順に処理し、行ごとに「n [token...]」を吐く(n<0のときはトークン無し)。
# **同一ctxを使い回す**(コンパイル1回で複数シナリオ=複数行の
# stdinレコードを流せば、state独立性の検証にもそのまま使える)。
_DRIVER = '''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"

static uint32_t work[MDL_CTX_BYTES / 4];

int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
  uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
  uint32_t bos, eos, max_new;
  int n, i;

  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {
    size_t got = fread(src_bits, 1, sizeof(src_bits), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits)) return 11;
    if (fread(&bos, sizeof(bos), 1, stdin) != 1) return 12;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 13;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 14;
    n = mdl_rddlgn_generate_greedy(&ctx, src_bits, bos, eos, max_new,
                                   out_tokens);
    printf("%d", n);
    if (n > 0) {
      for (i = 0; i < n; i++) printf(" %u", out_tokens[i]);
    }
    printf("\\n");
  }
  return 0;
}
'''


def _embed_src_bits(model, tokens: np.ndarray) -> np.ndarray:
    """srcトークン列 → 埋め込み・Heaviside済みビット列(1バイト=1ビット)。

    models/rddlgn.py::LogicEmbedding.forwardのeval経路と同じ規則
    (`sigmoid(e) > 0.5`)。extract_embed_tableが作るパック表と同じ規則
    だが、ここではCの`src_bits`引数に渡す**非パック**形が要る。
    """
    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()
    return embed[tokens].reshape(-1).astype(np.uint8)


def _encode_query(src_bits: np.ndarray, bos: int, eos: int,
                  max_new: int) -> bytes:
    return src_bits.tobytes() + struct.pack('<III', int(bos), int(eos),
                                            int(max_new))


def _run_queries(exe, queries: list[bytes]) -> list[list[int] | int]:
    proc = subprocess.run([str(exe)], input=b''.join(queries),
                          capture_output=True)
    assert proc.returncode == 0, (
        f'C実行が失敗 (exit={proc.returncode}): '
        f'{proc.stderr.decode(errors="replace")}')
    lines = proc.stdout.decode().strip('\n').split('\n')
    assert len(lines) == len(queries), (
        f'クエリ数{len(queries)}に対し出力行数{len(lines)}')
    results: list[list[int] | int] = []
    for line in lines:
        parts = [int(v) for v in line.split()]
        n = parts[0]
        results.append(n if n < 0 else parts[1:1 + n])
    return results


# ---------------------------------------------------------------------------
# 固定モデル・バンドル・実行ファイル(モジュール内で使い回す。ビルドは1回)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def greedy_model():
    return build_rddlgn_tiny_seq2seq(seed=0)


@pytest.fixture(scope='module')
def greedy_bundle(greedy_model):
    model, cfg = greedy_model
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


@pytest.fixture(scope='module')
def greedy_exe(cc, tmp_path_factory, greedy_bundle):
    work = tmp_path_factory.mktemp('rddlgn_greedy_gen')
    files = generate_rddlgn_c(greedy_bundle, name='model', validate=False)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(_DRIVER, encoding='utf-8')
    exe = work / 'app'
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
          f'-I{work}', f'-I{RUNTIME_C / "include"}',
          str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
          str(RUNTIME_C / 'src' / 'mdl_rddlgn.c'), str(work / 'driver.c'),
          '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe


def _random_query(rng, model, cfg, *, eos=999999):
    src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,))
    bos = int(rng.integers(0, cfg.vocab_size))
    src_bits = _embed_src_bits(model, src_tokens)
    return src_tokens, bos, src_bits


class TestPyTorchMatchesSimulate:
    """PyTorch greedy ⇔ simulate greedy(Cを経由しない、純Python)。"""

    def test_random_samples_match(self, greedy_model):
        model, cfg = greedy_model
        rng = np.random.default_rng(7)
        for _ in range(8):
            src_tokens, bos, _ = _random_query(rng, model, cfg)
            eos = int(rng.integers(0, cfg.vocab_size))
            py = model.generate_greedy(
                torch.from_numpy(src_tokens), bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len)
            sim = simulate_rddlgn_greedy(
                model, src_tokens, bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len)
            assert py == sim, (src_tokens, bos, eos, py, sim)


class TestCMatchesSimulate:
    """simulate greedy ⇔ C greedy(複数乱数src、トークン列全一致)。"""

    def test_random_samples_match_exactly(self, greedy_exe, greedy_model):
        model, cfg = greedy_model
        rng = np.random.default_rng(11)
        queries: list[bytes] = []
        expected: list[list[int]] = []
        for _ in range(10):
            src_tokens, bos, src_bits = _random_query(rng, model, cfg)
            eos = int(rng.integers(0, cfg.vocab_size))
            sim = simulate_rddlgn_greedy(
                model, src_tokens, bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len)
            queries.append(_encode_query(src_bits, bos, eos, cfg.seq_len))
            expected.append(sim)

        got = _run_queries(greedy_exe, queries)
        assert got == expected

    def test_extreme_src_bits_match(self, greedy_exe, greedy_model):
        """全0/全1の埋め込みビット列でも一致する(端数ビットの扱い含む)。

        simulate_rddlgn_greedy_bits(simulate_rddlgn_greedyから
        切り出した、srcを生ビット列で受ける版)を使えば、実在のトークンに
        対応しない全0/全1ビット列でもgoldenが作れる。
        """
        model, cfg = greedy_model
        src_width = cfg.seq_len * model.src_embedding.embedding.embedding_dim
        queries = []
        expected = []
        for bits in (np.zeros(src_width, dtype=np.uint8),
                    np.ones(src_width, dtype=np.uint8)):
            sim = simulate_rddlgn_greedy_bits(
                model, bits, bos_token=0, eos_token=999999,
                max_new_tokens=cfg.seq_len)
            queries.append(_encode_query(bits, 0, 999999, cfg.seq_len))
            expected.append(sim)
        got = _run_queries(greedy_exe, queries)
        assert got == expected


class TestEosAndBounds:
    def test_eos_stops_early(self, greedy_exe, greedy_model):
        """simulate側で確定した将来のトークンをeosに指定し、その手前で
        打ち切られることをC側と突き合わせて確かめる。"""
        model, cfg = greedy_model
        # seed=0はfull=[47,25,14,14]のように非自明な(全ステップ同一
        # トークンへ縮退しない)系列になる実測済みの種。未学習の小型モデルは
        # 容易に単一トークンへ縮退する(下のfirst_hitがどこであっても
        # テスト自体は正しく動くようindex()で機械的に求めているが、
        # position 0で毎回止まる自明ケースだけ踏むのを避けるために選んだ)。
        rng = np.random.default_rng(0)
        src_tokens, bos, src_bits = _random_query(rng, model, cfg)
        full = simulate_rddlgn_greedy(
            model, src_tokens, bos_token=bos, eos_token=999999,
            max_new_tokens=cfg.seq_len)
        assert len(full) == cfg.seq_len, (
            'eos_token=999999(語彙外)ならmax_new_tokensぶん生成される前提')
        cut = cfg.seq_len // 2
        eos_token = full[cut]
        # eos_tokenがcutより前に既に現れる可能性がある(小規模未学習モデルは
        # 同じトークンへ収束しがち)ので、期待打ち切り位置は「最初に
        # eos_tokenが現れた位置」から機械的に求める(固定でcutを仮定しない)。
        first_hit = full.index(eos_token)

        sim_early = simulate_rddlgn_greedy(
            model, src_tokens, bos_token=bos, eos_token=eos_token,
            max_new_tokens=cfg.seq_len)
        assert sim_early == full[:first_hit + 1]

        got = _run_queries(greedy_exe, [
            _encode_query(src_bits, bos, eos_token, cfg.seq_len)])
        assert got == [sim_early]

    def test_max_new_tokens_off_by_one(self, greedy_exe, greedy_model):
        """max_new_tokens<=STEPSは成功、STEPS+1はC側が-1で拒否する。"""
        model, cfg = greedy_model
        rng = np.random.default_rng(29)
        _src_tokens, bos, src_bits = _random_query(rng, model, cfg)
        steps = cfg.seq_len   # dec_pm_recのコンパイル時steps=tgt_len

        ok_at_bound, over_bound = _run_queries(greedy_exe, [
            _encode_query(src_bits, bos, 999999, steps),
            _encode_query(src_bits, bos, 999999, steps + 1),
        ])
        assert isinstance(ok_at_bound, list) and len(ok_at_bound) == steps
        assert over_bound == -1


class TestSameContextTwoGenerations:
    """同一ctxで2系列連続生成しても2回目が単独実行と一致する
    (state リセットの検証)。"""

    def test_second_generation_matches_standalone(self, greedy_exe,
                                                   greedy_model):
        model, cfg = greedy_model
        rng = np.random.default_rng(31)
        _t1, bos1, bits1 = _random_query(rng, model, cfg)
        _t2, bos2, bits2 = _random_query(rng, model, cfg)
        q1 = _encode_query(bits1, bos1, 999999, cfg.seq_len)
        q2 = _encode_query(bits2, bos2, 999999, cfg.seq_len)

        # 同一プロセス(=同一ctx)で q1 → q2 の順に2回生成。
        both = _run_queries(greedy_exe, [q1, q2])
        # 別プロセス(=新しいctx)で q2 だけを単独実行。
        standalone = _run_queries(greedy_exe, [q2])

        assert both[1] == standalone[0], (
            '2回目の生成が単独実行と食い違う(1回目の生成状態が'
            '持ち越されている疑い)')
