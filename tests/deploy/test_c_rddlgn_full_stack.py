"""RDDLGN全部載せ(packed teacher forcing +語彙折り畳み +
退化ゲート除去 +添字u16化)の統合検証。

各機能は個別のテストファイルで検証済み:
    - test_c_rddlgn_packed_tf.py::TestPackedWithOptimizeDegenerate
    -単独: test_c_rddlgn_folded_greedy.py
    -単独: test_rddlgn_optimize_degenerate.py
    -単独: test_lut_slice.py

だが、3ブランチ(k27-multicore/k29-u16-indices/k28-onnx-baseline)を
feature/rddlgn-deploykitへ統合するマージ作業で、
4機能を**同時に**有効化した「全部載せ」のC生成物が整数厳密一致することを
固定するテストが無いことが分かった(マージ後の検証タスク)。

char-LM相当の合成モデル(RDDLGNConfig.tiny()、vocab_size=50)で、
(N/L-layers語彙折り畳み、force=True)・(optimize_degenerate=True)を
同時に有効化したIRから、(packed teacher forcing)を有効化した**1個の**
生成C(model.c/h)を作り、2つのエントリポイント
(mdl_infer_packed_tf_ctx / mdl_rddlgn_generate_greedy_folded)がそれぞれ
Python golden(ir/simulate・simulate_rddlgn_greedy)と整数厳密一致すること
を確認する。(添字u16化)はin_width<=65536で自動的に効くため、生成C内に
`_u16`カーネル呼び出しが実際に現れることも併せて固定する。

folded_n_table/folded_l_tableは`_lutstack_nodes`(非最適化パス)で
構築され、optimize_degenerateはtt代数で値を変えずに物理ゲートだけを
減らす変換なので、両者は構造的に独立(節参照)——このテストはその
独立性が生成Cレベルでも保たれていることの裏取りでもある。
"""

import struct
import subprocess

import numpy as np

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cbackend import generate_rddlgn_c
from griffith_logic.export.rddlgn_ir import (
    export_rddlgn_ir,
    extract_embed_table,
    extract_folded_l_table,
    extract_folded_n_table,
    simulate_rddlgn_greedy,
)
from griffith_logic.ir.serialize import ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq

#:: packed teacher forcing。生の入力ビット列を1行読み、
#: mdl_infer_ctx(既存経路)とmdl_infer_packed_tf_ctxを突き合わせて
#: 一致していればpackedのint32スコアを返す(不一致ならexit 5)。
_PACKED_TF_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>

static uint32_t work[MDL_CTX_BYTES / 4];
static uint32_t pwork[MDL_DEC_PM_REC_PACKED_WORK_BYTES / 4 + 4];
static uint8_t input_bits[MDL_INPUT_WIDTH];
static int32_t scores_normal[MDL_OUTPUT_WIDTH];
static int32_t scores_packed[MDL_OUTPUT_WIDTH];

int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int32_t i;

  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {
    size_t got = fread(input_bits, 1, sizeof(input_bits), stdin);
    if (got == 0) break;
    if (got != sizeof(input_bits)) return 11;

    if (mdl_infer_ctx(&ctx, input_bits, scores_normal) != 0) return 2;
    if (mdl_infer_packed_tf_ctx(&ctx, input_bits, pwork, sizeof(pwork),
                                scores_packed) != 0) return 3;
    if (memcmp(scores_normal, scores_packed, sizeof(scores_normal)) != 0) {
      return 5;
    }
    for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
      unsigned char b[4];
      uint32_t v = (uint32_t)scores_packed[i];
      b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
      b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
      fwrite(b, 1, 4, stdout);
    }
  }
  return 0;
}
'''

#:: 語彙折り畳み経由の自己回帰貪欲生成。トークン列クエリを1件読み、
#: mdl_rddlgn_generate_greedy_folded(N/L-layersをfoldedテーブル参照で
#: 置き換えた経路)の生成トークン列を1行に書き出す。
_FOLDED_GREEDY_DRIVER = '''#include "model.h"
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


def _build_full_stack_bundle(seed: int = 11) -> tuple[dict, object, object]:
    """(force fold)+(optimize_degenerate)を両方有効にしたbundleを作る。

    Returns:
        (bundle, model, cfg)
    """
    model, cfg = build_rddlgn_tiny_seq2seq(seed=seed)
    graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                             optimize_degenerate=True)
    assert validate(graph) == []
    doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    doc['folded_n_table'] = extract_folded_n_table(model, force=True)
    doc['folded_l_table'] = extract_folded_l_table(model, force=True)
    assert doc['folded_n_table'] is not None
    assert doc['folded_l_table'] is not None
    bundle = build_bundle(doc)
    return bundle, model, cfg


class TestFullStackIntegerMatch:
    """同時に有効化した1個の生成Cが、2エントリポイント
    ともPython goldenと整数厳密一致することを固定する。"""

    def test_generated_source_uses_k24_k25_and_k29(self, tmp_path, cc):
        """生成物に4機能ぶんの痕跡(エントリポイント、u16
        カーネル呼び出し)が実際に現れていることの自己チェック。これが
        無ければ、以下の一致テストは「そもそも機能が有効化されていない」
        だけで偶然一致する空振りになりうる。"""
        bundle, _model, _cfg = _build_full_stack_bundle()
        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
        header = files['model.h']
        source = files['model.c']
        assert 'mdl_infer_packed_tf_ctx' in header, 'エントリポイントが無い'
        # mdl_rddlgn_generate_greedy_folded自体はruntime/c/include/
        # mdl_rddlgn.h(固定のランタイムヘッダ)で宣言される手書き関数
        # なので、生成物(model.h)には現れない。折り畳みテーブルが
        # 実際に組み込まれた痕跡はこのMDL_HAS_FOLDED_*マクロ
        # (mdl_rddlgn.hがこれを見てfolded lookup経路を選ぶ)。
        assert 'MDL_HAS_FOLDED_N_TABLE' in header, 'のN折り畳み表が無い'
        assert 'MDL_HAS_FOLDED_L_TABLE' in header, 'のL折り畳み表が無い'
        assert 'mdl_folded_n_lookup' in header
        assert 'mdl_folded_l_lookup' in header
        assert ('mdl_lut_layer_slice_u16(' in source
               or 'mdl_lut_layer_packed_u16(' in source), (
            'のu16カーネル呼び出しが生成物に現れていない'
            '(in_widthが65536を超えるレイヤしか無い?)')

    def test_packed_tf_matches_simulate(self, tmp_path, cc):
        """(packed teacher forcing)が共存する生成物で、
        ir/simulate(教師強制のgolden)と整数厳密一致する。"""
        bundle, _model, _cfg = _build_full_stack_bundle(seed=11)
        from griffith_logic.ir.serialize import ir_from_dict

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)

        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
        work = tmp_path / 'packed_tf'
        work.mkdir()
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(_PACKED_TF_DRIVER, encoding='utf-8')
        exe = work / 'app'
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
              str(work / 'driver.c'), '-o', str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=(8, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, (
            f'C実行が失敗(exit={proc.returncode}、5は正常/packed不一致): '
            f'{proc.stderr.decode(errors="replace")}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        got = got.reshape(8, n_out)
        assert np.array_equal(got, want)

    def test_folded_greedy_matches_python_golden(self, tmp_path, cc):
        """(語彙折り畳み)経由のgreedy生成が、(退化ゲート除去)
        込みのsimulate_rddlgn_greedy(golden)と一致する。同じbundleは
       (packed_teacher_forcing)も同時に有効化されている
        (test_packed_tf_matches_simulateと同一生成元)。"""
        bundle, model, cfg = _build_full_stack_bundle(seed=13)

        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
        work = tmp_path / 'folded_greedy'
        work.mkdir()
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(_FOLDED_GREEDY_DRIVER,
                                       encoding='utf-8')
        exe = work / 'app'
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
              str(RUNTIME_C / 'src' / 'mdl_rddlgn.c'),
              str(work / 'driver.c'), '-o', str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        rng = np.random.default_rng(13)
        queries = []
        meta = []
        for _ in range(8):
            src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,),
                                      dtype=np.int64)
            bos = int(rng.integers(0, cfg.vocab_size))
            eos = int(rng.integers(0, cfg.vocab_size))
            queries.append(src_tokens.astype('<u4').tobytes()
                          + struct.pack('<III', bos, eos, cfg.seq_len))
            meta.append((src_tokens, bos, eos))

        proc = subprocess.run([str(exe)], input=b''.join(queries),
                              capture_output=True)
        assert proc.returncode == 0, (
            f'C実行が失敗(exit={proc.returncode}): '
            f'{proc.stderr.decode(errors="replace")}')
        lines = proc.stdout.decode().strip('\n').split('\n')
        assert len(lines) == len(queries)

        for line, (src_tokens, bos, eos) in zip(lines, meta, strict=True):
            parts = line.split()
            n = int(parts[0])
            got = [int(v) for v in parts[1:1 + n]]
            want = simulate_rddlgn_greedy(
                model, src_tokens, bos_token=bos, eos_token=eos,
                max_new_tokens=cfg.seq_len, optimize_degenerate=True)
            assert got == want, (src_tokens, bos, eos)
