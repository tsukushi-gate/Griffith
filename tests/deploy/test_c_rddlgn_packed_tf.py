"""RDDLGN 教師強制の時間バッチ化の整数厳密一致検証。

デプロイキット設計メモ節ベースライン計測で確定した通り、decode時間の88〜99%はP+M+GroupSum
(`mdl_step_dec_pm_rec`)が占める。P-layersは真のリカレント(state依存)
だがM-layers+GroupSumはtickごとに独立(`state_t`だけの純関数)なので、
`deploy/cbackend.py::CBackend._emit_packed_teacher_forcing`が
recurrent bodyをhead(P-layers、順にsteps回)/tail(M+GroupSum、steps本
まとめて`mdl_lut_layer_packed`で1回評価)へ分割し、追加entrypoint
`mdl_infer_packed_tf_ctx`を生やす(既存の`mdl_infer_ctx`/`mdl_run`は
一切変更しない、既存経路を残して選択制)。

このテストは新entrypointの値が既存経路(`mdl_infer_ctx`、ひいては
`ir/simulate`)と完全一致することを固定する。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

_DRIVER = '''#include "model.h"
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


def _build(bundle, work_dir, cc):
    files = generate_c(bundle, name='model', validate=False,
                       expose_recurrent=frozenset({'dec_pm_rec'}),
                       packed_teacher_forcing='dec_pm_rec')
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work_dir / name).write_text(text, encoding='utf-8')
    (work_dir / 'driver.c').write_text(_DRIVER, encoding='utf-8')
    exe = work_dir / 'app'
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
          f'-I{work_dir}', f'-I{RUNTIME_C / "include"}',
          str(work_dir / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
          str(work_dir / 'driver.c'), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe, files


class TestPackedMatchesNormalAndSimulate:
    """mdl_infer_packed_tf_ctx == mdl_infer_ctx == ir/simulate(整数厳密)。"""

    def test_random_samples_match_exactly(self, cc, tmp_path,
                                          rddlgn_seq2seq_tiny_bundle):
        graph = ir_from_dict(rddlgn_seq2seq_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        exe, _ = _build(rddlgn_seq2seq_tiny_bundle, tmp_path / 'gen', cc)

        rng = np.random.default_rng(0)
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

    def test_extreme_inputs_match(self, cc, tmp_path,
                                  rddlgn_seq2seq_tiny_bundle):
        """全0/全1の埋め込みビット列でも一致する(端数ビットの扱い含む)。"""
        graph = ir_from_dict(rddlgn_seq2seq_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        exe, _ = _build(rddlgn_seq2seq_tiny_bundle, tmp_path / 'gen', cc)
        for label, x in (('zeros', np.zeros((1, in_width), dtype=np.uint8)),
                         ('ones', np.ones((1, in_width), dtype=np.uint8))):
            want = simulate(graph, x).astype(np.int64)
            proc = subprocess.run([str(exe)], input=x.tobytes(),
                                  capture_output=True)
            assert proc.returncode == 0, label
            got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
            got = got.reshape(1, n_out)
            assert np.array_equal(got, want), label

    @pytest.mark.parametrize('seed,init_mode', [(1, 'zero'), (2, 'one')])
    def test_different_weights_and_init_mode_match(self, cc, tmp_path, seed,
                                                    init_mode):
        """重みや初期状態モードが違っても一致する(1つの固定シードで
        偶然揃った、を排除)。"""
        model, cfg = build_rddlgn_tiny_seq2seq(seed=seed)
        cfg.init_mode = init_mode
        import torch

        from griffith_logic.models.rddlgn import RDDLGN

        torch.manual_seed(seed)
        model = RDDLGN(cfg)
        model.eval()
        doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
        bundle = build_bundle(doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        exe, _ = _build(bundle, tmp_path / f'gen{seed}', cc)

        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(5, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)
        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        got = got.reshape(5, n_out)
        assert np.array_equal(got, want)


class TestGeneratedSourceProperties:
    def test_mdl_run_body_is_byte_identical_without_flag(
            self, rddlgn_seq2seq_tiny_bundle):
        """packed_teacher_forcing=Noneのときは既存生成物と1バイトも
        変わらない(既存経路を残して選択制、設計方針)。"""
        with_flag = generate_c(
            rddlgn_seq2seq_tiny_bundle, name='model', validate=False,
            expose_recurrent=frozenset({'dec_pm_rec'}))['model.c']
        without = generate_c(
            rddlgn_seq2seq_tiny_bundle, name='model', validate=False,
            expose_recurrent=frozenset({'dec_pm_rec'}),
            packed_teacher_forcing=None)['model.c']
        assert with_flag == without

    def test_no_heap_or_pthread(self, rddlgn_seq2seq_tiny_bundle):
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False,
                           expose_recurrent=frozenset({'dec_pm_rec'}),
                           packed_teacher_forcing='dec_pm_rec')
        source = files['model.c']
        for bad in ('malloc', 'calloc', 'realloc', 'free(', 'pthread'):
            assert bad not in source, f'生成コードに {bad} がある'

    def test_compiles_with_strict_warnings(self, cc, tmp_path,
                                           rddlgn_seq2seq_tiny_bundle):
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False,
                           expose_recurrent=frozenset({'dec_pm_rec'}),
                           packed_teacher_forcing='dec_pm_rec')
        work = tmp_path / 'strict'
        work.mkdir()
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              '-c', str(work / 'model.c'), '-o', str(work / 'model.o')]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr


class TestPackedWithOptimizeDegenerate:
    """(packed teacher forcing)と(退化ゲート除去)の併用。

    中間層の生存ゲートをアキュムレータ(`concat`ノード、
    `deploy/plan.py::_plan_accumulate_concat`がalias化)へ追記する設計に
    変わったため(デプロイキット設計メモ節「との
    併用」参照)、`_emit_packed_teacher_forcing`のtailにこのconcat由来の
    1入力`flat_concat`(シード)が現れる。側がこれを認識してpm_x0を
    accumulate-groupの最終幅へ拡張し、各層をpm_x0内の絶対オフセットへ
    直接読み書きすることで、有効時でも`packed_teacher_forcing`が
    有効化できる(拒否されない)ことと、値がoptimize_degenerate=Falseの
    既存経路(mdl_infer_ctx)・ir/simulateと厳密一致することを固定する。
    """

    def test_packed_tf_matches_normal_and_simulate_with_k26(self, cc, tmp_path):
        from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
        from griffith_logic.ir.serialize import ir_to_dict
        from griffith_logic.ir.validate import validate

        model, cfg = build_rddlgn_tiny_seq2seq(seed=3)
        graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                 optimize_degenerate=True)
        assert validate(graph) == []
        doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
        bundle = build_bundle(doc)

        exe, _ = _build(bundle, tmp_path / 'gen_k26', cc)

        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=(8, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, (
            f'C実行が失敗(exit={proc.returncode}、5はpacked不一致): '
            f'{proc.stderr.decode(errors="replace")}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        got = got.reshape(8, n_out)
        assert np.array_equal(got, want)

    def test_no_flat_concat_rejection_with_k26(self, tmp_path):
        """有効時、tailにflat_concat(シード)があっても`generate_c`が
        ValueErrorで拒否しない(節「との併用」で報告した
        従来の非対応が解消されたことの回帰止め)。"""
        from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
        from griffith_logic.ir.serialize import ir_to_dict
        from griffith_logic.ir.validate import validate

        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                 optimize_degenerate=True)
        assert validate(graph) == []
        doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
        bundle = build_bundle(doc)

        files = generate_c(bundle, name='model', validate=False,
                           expose_recurrent=frozenset({'dec_pm_rec'}),
                           packed_teacher_forcing='dec_pm_rec')
        assert 'mdl_infer_packed_tf_ctx' in files['model.h']


class TestEligibilityErrors:
    """パターンに合わないnode_idを指定すると、黙って無視せず明示的に
    落ちる(opt-inした機能が効いていないことに気づけるように)。"""

    def test_wrong_node_id_raises(self, rddlgn_seq2seq_tiny_bundle):
        with pytest.raises(ValueError, match='dec_l_rec'):
            generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                      validate=False,
                      expose_recurrent=frozenset({'dec_l_rec'}),
                      packed_teacher_forcing='dec_l_rec')

    def test_not_exposed_raises(self, rddlgn_seq2seq_tiny_bundle):
        with pytest.raises(ValueError, match='expose_recurrent'):
            generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                      validate=False,
                      packed_teacher_forcing='dec_pm_rec')

    def test_combining_with_drop_folded_recurrents_raises(
            self, rddlgn_seq2seq_tiny_bundle):
        """`packed_teacher_forcing`と`drop_folded_recurrents`の
        同時指定は未対応(マージ時に固定)。`mdl_infer_packed_tf_ctx`
       は対象recurrent直前までの`self.body`(他recurrentの
        host_loop_block込み)を再利用してprefixを組み立てるが、
        `drop_folded_recurrents`はそのnode_idのhost_loop_block
        自体を生成しないため、packed_teacher_forcing側のprefixが
        欠落した状態で黙って壊れた値を返す事故になる。どちらも
        opt-inの機能なので黙ってどちらかを無視せずValueErrorで落ちる。"""
        with pytest.raises(ValueError, match='packed_teacher_forcing'):
            generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                      validate=False,
                      expose_recurrent=frozenset({'dec_pm_rec'}),
                      packed_teacher_forcing='dec_pm_rec',
                      drop_folded_recurrents=frozenset({'enc_n_rec'}))
