"""RDDLGN語彙折り畳みのC側検証。

tests/deploy/test_c_rddlgn_greedy.py(非折り畳み経路の3者一致)の
延長。ここでは新しく追加した `mdl_rddlgn_generate_greedy_folded`
(runtime/c/src/mdl_rddlgn.c)が、既存の `mdl_rddlgn_generate_greedy`
(src_bits版、非折り畳み)とビット厳密に一致することを、以下の3通りの
バンドルで確認する:

    - 両方(N/L)折り畳んだバンドル(RDDLGNConfig.tiny()は自動判定で
      両方foldする)
    - 折り畳みを強制的に見送ったバンドル(force=False。新関数の
      フォールバック経路 == 旧関数、を確認する)
    - N/Lどちらか一方だけ折り畳んだバンドル(独立に自動判定される
      ことの確認)

さらに、Python golden(`simulate_rddlgn_greedy`)とも突き合わせ、
折り畳み経路が3者(PyTorch含む既存チェーンの延長として simulate)一致
であることも固定する。

`TestDropFoldedWeights` は配信サイズの実削減側
(`generate_rddlgn_c(..., drop_folded_weights=True)`、N/L-layersの重み・
step関数・mdl_infer/mdl_runからの呼び出しを生成物から丸ごと落とす
ビルド)の正しさとサイズ削減、および安全側フォールバック(片方しか
foldしていなければ黙って通常ビルドに戻る)を検証する。
"""

import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cbackend import generate_rddlgn_c
from griffith_logic.export.rddlgn_ir import (
    extract_embed_table,
    extract_folded_l_table,
    extract_folded_n_table,
    simulate_rddlgn_greedy,
)

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

# 1回の起動で複数クエリを処理する。各クエリ:
#   src_bits (MDL_BUF_WIDTH_X_SRC バイト) + src_tokens (u32 x MDL_STEPS_ENC_N_REC)
#   + bos(u32) + eos(u32) + max_new(u32)
# 出力は行ごとに:
#   "OK n [token...]"          両関数が一致(nは生成数)
#   "MISMATCH_N n_bits n_tok"  トークン数自体が食い違う
#   "MISMATCH_TOK"             トークン列が食い違う
_DRIVER = '''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>

static uint32_t work_bits[MDL_CTX_BYTES / 4];
static uint32_t work_tok[MDL_CTX_BYTES / 4];

int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx_bits, ctx_tok;
  uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
  uint32_t src_tokens[MDL_STEPS_ENC_N_REC];
  uint32_t out_bits[MDL_STEPS_DEC_PM_REC];
  uint32_t out_tok[MDL_STEPS_DEC_PM_REC];
  uint32_t bos, eos, max_new;
  int n_bits, n_tok, i;

  if (mdl_ctx_init(&ctx_bits, work_bits, sizeof(work_bits)) != 0) return 10;
  if (mdl_ctx_init(&ctx_tok, work_tok, sizeof(work_tok)) != 0) return 11;
  for (;;) {
    size_t got = fread(src_bits, 1, sizeof(src_bits), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits)) return 12;
    if (fread(src_tokens, sizeof(uint32_t), MDL_STEPS_ENC_N_REC, stdin)
        != (size_t)MDL_STEPS_ENC_N_REC) return 13;
    if (fread(&bos, sizeof(bos), 1, stdin) != 1) return 14;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 15;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 16;

    n_bits = mdl_rddlgn_generate_greedy(&ctx_bits, src_bits, bos, eos,
                                        max_new, out_bits);
    n_tok = mdl_rddlgn_generate_greedy_folded(&ctx_tok, src_tokens, bos, eos,
                                              max_new, out_tok);
    if (n_bits != n_tok) {
      printf("MISMATCH_N %d %d\\n", n_bits, n_tok);
      continue;
    }
    if (n_bits > 0
        && memcmp(out_bits, out_tok, sizeof(uint32_t) * (size_t)n_bits) != 0) {
      printf("MISMATCH_TOK\\n");
      continue;
    }
    printf("OK %d", n_bits);
    if (n_bits > 0) {
      for (i = 0; i < n_bits; i++) printf(" %u", out_bits[i]);
    }
    printf("\\n");
  }
  return 0;
}
'''


def _embed_src_bits(model, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()
    return embed[tokens].reshape(-1).astype(np.uint8)


def _encode_query(src_bits: np.ndarray, src_tokens: np.ndarray, bos: int,
                  eos: int, max_new: int) -> bytes:
    return (src_bits.tobytes()
           + src_tokens.astype('<u4').tobytes()
           + struct.pack('<III', int(bos), int(eos), int(max_new)))


def _run_queries(exe, queries: list[bytes]) -> list[tuple[str, list[int] | None]]:
    proc = subprocess.run([str(exe)], input=b''.join(queries),
                          capture_output=True)
    assert proc.returncode == 0, (
        f'C実行が失敗 (exit={proc.returncode}): '
        f'{proc.stderr.decode(errors="replace")}')
    lines = proc.stdout.decode().strip('\n').split('\n')
    assert len(lines) == len(queries), (
        f'クエリ数{len(queries)}に対し出力行数{len(lines)}: {lines}')
    results = []
    for line in lines:
        parts = line.split()
        tag = parts[0]
        if tag == 'OK':
            n = int(parts[1])
            results.append((tag, [int(v) for v in parts[2:2 + n]]))
        else:
            results.append((tag, None))
    return results


@pytest.fixture(scope='module')
def folded_model():
    return build_rddlgn_tiny_seq2seq(seed=0)


def _build_bundle(model, cfg, *, force_n, force_l):
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    doc['folded_n_table'] = extract_folded_n_table(model, force=force_n)
    doc['folded_l_table'] = extract_folded_l_table(model, force=force_l)
    return build_bundle(doc)


def _build_exe(cc, tmp_path_factory, bundle, tag: str):
    work = tmp_path_factory.mktemp(f'rddlgn_folded_{tag}')
    files = generate_rddlgn_c(bundle, name='model', validate=False)
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


def _random_queries(model, cfg, n: int, seed: int):
    rng = np.random.default_rng(seed)
    queries = []
    src_token_list = []
    for _ in range(n):
        src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,),
                                  dtype=np.int64)
        bos = int(rng.integers(0, cfg.vocab_size))
        eos = int(rng.integers(0, cfg.vocab_size))
        src_bits = _embed_src_bits(model, src_tokens)
        queries.append(_encode_query(
            src_bits, src_tokens.astype(np.uint32), bos, eos, cfg.seq_len))
        src_token_list.append((src_tokens, bos, eos))
    return queries, src_token_list


class TestFoldedMatchesUnfolded:
    """折り畳み経路(mdl_rddlgn_generate_greedy_folded) ==
    非折り畳み経路(mdl_rddlgn_generate_greedy)。ドライバ自身がC内部で
    突き合わせて MISMATCH_* を吐くので、テスト側は全行が OK かどうかだけ
    見ればよい。"""

    @pytest.mark.parametrize('force_n,force_l,tag', [
        (None, None, 'auto'),   # RDDLGNConfig.tiny()は両方自動foldする
        (False, False, 'none'),   # 折り畳み無し(新関数のフォールバック確認)
        (True, False, 'n_only'),
        (False, True, 'l_only'),
    ])
    def test_random_samples_match(self, folded_model, cc, tmp_path_factory,
                                  force_n, force_l, tag):
        model, cfg = folded_model
        bundle = _build_bundle(model, cfg, force_n=force_n, force_l=force_l)
        exe = _build_exe(cc, tmp_path_factory, bundle, tag)

        queries, _ = _random_queries(model, cfg, n=12, seed=5)
        results = _run_queries(exe, queries)
        mismatches = [r for r in results if r[0] != 'OK']
        assert not mismatches, f'{tag}: {mismatches}'

    def test_auto_config_actually_folds_both(self, folded_model):
        """このテストファイルの前提(tiny()は自動で両方foldする)が
        崩れていないことの自己チェック。崩れたら'auto'ケースは
        'none'ケースの重複になり、折り畳み経路自体を検証していない
        ことになるため。"""
        model, _cfg = folded_model
        assert extract_folded_n_table(model) is not None
        assert extract_folded_l_table(model) is not None


class TestFoldedMatchesPythonGolden:
    """折り畳みC経路 ⇔ simulate_rddlgn_greedy(Python golden、非折り畳み)。

    C内部の突き合わせ(TestFoldedMatchesUnfolded)は「新旧C関数が一致する」
    ことしか示さないので、ここで独立に検証されたgoldenとも突き合わせ、
    折り畳み経路の絶対的な正しさを確認する。
    """

    def test_random_samples_match_golden(self, folded_model, cc,
                                         tmp_path_factory):
        model, cfg = folded_model
        bundle = _build_bundle(model, cfg, force_n=None, force_l=None)
        exe = _build_exe(cc, tmp_path_factory, bundle, 'golden')

        queries, meta = _random_queries(model, cfg, n=10, seed=17)
        results = _run_queries(exe, queries)
        for (tag, got), (src_tokens, bos, eos) in zip(results, meta,
                                                       strict=True):
            assert tag == 'OK'
            want = simulate_rddlgn_greedy(
                model, src_tokens, bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len)
            assert got == want, (src_tokens, bos, eos)


# ---------------------------------------------------------------------------
# drop_folded_weights: N/L-layersの重みを生成物から丸ごと削る
# 「折り畳み専用・配信サイズ削減」ビルド。
# ---------------------------------------------------------------------------
_DROPPED_DRIVER = '''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"

static uint32_t work[MDL_CTX_BYTES / 4];

int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  uint32_t src_tokens[MDL_STEPS_ENC_N_REC];
  uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
  uint32_t bos, eos, max_new;
  int n, i;

  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {
    size_t got = fread(src_tokens, sizeof(uint32_t), MDL_STEPS_ENC_N_REC,
                       stdin);
    if (got == 0) break;
    if (got != (size_t)MDL_STEPS_ENC_N_REC) return 11;
    if (fread(&bos, sizeof(bos), 1, stdin) != 1) return 12;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 13;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 14;
    n = mdl_rddlgn_generate_greedy_folded(&ctx, src_tokens, bos, eos,
                                          max_new, out_tokens);
    printf("%d", n);
    if (n > 0) for (i = 0; i < n; i++) printf(" %u", out_tokens[i]);
    printf("\\n");
  }
  return 0;
}
'''

#: mdl_rddlgn_generate_greedy(旧・bits版)を直接呼ぶだけの最小プログラム。
#: drop_folded_weights=Trueのビルドではこのシンボル自体が存在しないので、
#: リンクが失敗することを確認するのに使う。
_OLD_FN_PROBE = '''#include "model.h"
#include "mdl_rddlgn.h"
int main(void) {
  static uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
  static uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
  return mdl_rddlgn_generate_greedy(0, src_bits, 0, 0, 0, out_tokens);
}
'''


def _const_bytes(obj_path) -> int:
    """オブジェクトの読み取り専用データ節のバイト数。

    Mach-O(macOS)は `size -m` の `__TEXT, __const`、ELF(Linux)と COFF(Windows
    MinGW)は GNU `size -A` の `.rodata*` / `.rdata*` を合計する。
    """
    import re

    size_bin = shutil.which('size')
    if size_bin is None:
        pytest.skip('binutils の size が無い環境')
    if sys.platform == 'darwin':
        r = subprocess.run([size_bin, '-m', str(obj_path)], capture_output=True,
                           text=True)
        m = re.search(r'__TEXT, __const\): (\d+)', r.stdout)
        assert m, f'size -m の出力を解釈できない: {r.stdout!r}'
        return int(m.group(1))
    r = subprocess.run([size_bin, '-A', str(obj_path)], capture_output=True,
                       text=True)
    total = 0
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.match(r'^\.(rodata|rdata)(\W|$)', parts[0]):
            total += int(parts[1])
    assert total > 0, f'size -A の出力に .rodata/.rdata が無い: {r.stdout!r}'
    return total


class TestDropFoldedWeights:
    """generate_rddlgn_c(..., drop_folded_weights=True) の配信サイズ削減と
    正しさ・安全側フォールバックを確認する。"""

    def _compile_o(self, cc, work, bundle, tag: str, *, drop: bool):
        work = work / tag
        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  drop_folded_weights=drop)
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        obj = work / 'model.o'
        r = subprocess.run(
            [cc, '-std=c99', '-O2', f'-I{work}', f'-I{RUNTIME_C / "include"}',
             '-c', str(work / 'model.c'), '-o', str(obj)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return work, obj

    def test_dropped_const_section_is_smaller_than_baseline(
            self, folded_model, cc, tmp_path):
        """折り畳み+重み削除は、折り畳み無しの元バンドルより.rodataが
        小さくなる(配信サイズ純減、勝ち基準そのもの)。"""
        model, cfg = folded_model
        baseline_bundle = _build_bundle(model, cfg, force_n=False,
                                        force_l=False)
        dropped_bundle = _build_bundle(model, cfg, force_n=None,
                                       force_l=None)
        assert dropped_bundle['folded_n_table'] is not None
        assert dropped_bundle['folded_l_table'] is not None

        _, baseline_obj = self._compile_o(
            cc, tmp_path, baseline_bundle, 'baseline', drop=False)
        _, dropped_obj = self._compile_o(
            cc, tmp_path, dropped_bundle, 'dropped', drop=True)

        baseline_bytes = _const_bytes(baseline_obj)
        dropped_bytes = _const_bytes(dropped_obj)
        assert dropped_bytes < baseline_bytes, (
            f'baseline={baseline_bytes} dropped={dropped_bytes}: '
            '折り畳み+重み削除が配信サイズを減らしていない')

    def test_kept_weights_is_larger_than_baseline(self, folded_model, cc,
                                                   tmp_path):
        """(回帰確認・負の結果の記録) 表を足すだけ(重みは残す)だと
        配信サイズは逆に**増える** — drop_folded_weightsが必要な理由その
        もの。タスクの事前試算表に載せる根拠。"""
        model, cfg = folded_model
        baseline_bundle = _build_bundle(model, cfg, force_n=False,
                                        force_l=False)
        kept_bundle = _build_bundle(model, cfg, force_n=None, force_l=None)

        _, baseline_obj = self._compile_o(
            cc, tmp_path, baseline_bundle, 'baseline2', drop=False)
        _, kept_obj = self._compile_o(
            cc, tmp_path, kept_bundle, 'kept', drop=False)

        assert _const_bytes(kept_obj) > _const_bytes(baseline_obj)

    def test_dropped_build_matches_golden(self, folded_model, cc, tmp_path):
        model, cfg = folded_model
        bundle = _build_bundle(model, cfg, force_n=None, force_l=None)
        work, _ = self._compile_o(cc, tmp_path, bundle, 'run', drop=True)
        (work / 'driver.c').write_text(_DROPPED_DRIVER, encoding='utf-8')
        exe = work / 'app'
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              str(work / 'model.c'),
              str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
              str(RUNTIME_C / 'src' / 'mdl_rddlgn.c'), str(work / 'driver.c'),
              '-o', str(exe)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        rng = np.random.default_rng(23)
        payload = b''
        metas = []
        for _ in range(10):
            src_tokens = rng.integers(0, cfg.vocab_size,
                                      size=(cfg.seq_len,), dtype=np.uint32)
            bos = int(rng.integers(0, cfg.vocab_size))
            eos = int(rng.integers(0, cfg.vocab_size))
            payload += src_tokens.astype('<u4').tobytes() + struct.pack(
                '<III', bos, eos, cfg.seq_len)
            metas.append((src_tokens.astype(np.int64), bos, eos))

        proc = subprocess.run([str(exe)], input=payload, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.decode().strip('\n').split('\n')
        assert len(lines) == len(metas)
        for line, (src_tokens, bos, eos) in zip(lines, metas, strict=True):
            parts = [int(v) for v in line.split()]
            n = parts[0]
            got = parts[1:1 + n] if n > 0 else []
            want = simulate_rddlgn_greedy(
                model, src_tokens, bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len)
            assert got == want, (src_tokens, bos, eos)

    def test_old_function_is_unlinkable_when_dropped(self, folded_model, cc,
                                                       tmp_path):
        """MDL_RDDLGN_WEIGHTS_DROPPEDが立つと、旧関数の宣言/定義ごと消え、
        参照するとリンクエラーになる(誤用が即座に分かることの確認)。"""
        model, cfg = folded_model
        bundle = _build_bundle(model, cfg, force_n=None, force_l=None)
        work, _ = self._compile_o(cc, tmp_path, bundle, 'probe', drop=True)
        (work / 'probe.c').write_text(_OLD_FN_PROBE, encoding='utf-8')
        cmd = [cc, '-std=c99', '-O2', f'-I{work}', f'-I{RUNTIME_C / "include"}',
              str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
              str(RUNTIME_C / 'src' / 'mdl_rddlgn.c'), str(work / 'probe.c'),
              '-o', str(work / 'probe')]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0, (
            '旧mdl_rddlgn_generate_greedyが重み削除ビルドでも呼べてしまった')
        assert 'mdl_rddlgn_generate_greedy' in r.stderr, (
            f'期待した理由(未宣言/未定義シンボル)でのエラーではない: {r.stderr}')

    def test_drop_requested_but_not_both_folded_falls_back_safely(
            self, folded_model, cc, tmp_path):
        """drop_folded_weights=Trueでも、N/L片方しか折り畳まれていなければ
        黙って通常ビルド(重みを落とさない)にフォールバックする。教師強制や
        bits版greedyを不用意に壊さない安全側の既定。"""
        model, cfg = folded_model
        bundle = _build_bundle(model, cfg, force_n=True, force_l=False)
        assert bundle['folded_n_table'] is not None
        assert bundle['folded_l_table'] is None

        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  drop_folded_weights=True)
        assert 'MDL_RDDLGN_WEIGHTS_DROPPED' not in files['model.h']
