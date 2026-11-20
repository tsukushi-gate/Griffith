"""RDDLGN語彙折り畳みのPython側検証。

対象: export/rddlgn_ir.py の
    - _lutstack_weight_bytes / should_fold_vocab_table(自動判定・強制フラグ)
    - extract_folded_table / extract_folded_n_table / extract_folded_l_table
    - deploy/bundle.py::_check_folded_table(bundle検証)

C側の折り畳み経路(mdl_rddlgn_generate_greedy_folded)の検証は
tests/deploy/test_c_rddlgn_folded_greedy.py。ここはPython側の表の
確定と自動判定だけを対象にする。
"""

import dataclasses

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import _check_folded_table, build_bundle
from griffith_logic.export.rddlgn_ir import (
    _embed_hard_bits,
    _lutstack_nodes,
    _lutstack_weight_bytes,
    extract_folded_l_table,
    extract_folded_n_table,
    should_fold_vocab_table,
)
from griffith_logic.ir.schema import IRGraph, IRNode
from griffith_logic.ir.simulate import _simulate_values
from griffith_logic.models.rddlgn import RDDLGN, LutStack, RDDLGNConfig

from .conftest import build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc


# ---------------------------------------------------------------------------
# should_fold_vocab_table: 自動判定・強制フラグ
# ---------------------------------------------------------------------------
class TestShouldFoldVocabTable:
    def test_smaller_table_folds(self):
        assert should_fold_vocab_table(table_bytes=100, weight_bytes=200) is True

    def test_larger_table_does_not_fold(self):
        assert should_fold_vocab_table(table_bytes=200, weight_bytes=100) is False

    def test_equal_boundary_does_not_fold(self):
        """境界(表==重み)はfoldしない側に倒す(配信サイズを確実に増やさないため)。"""
        assert should_fold_vocab_table(table_bytes=100, weight_bytes=100) is False

    def test_force_true_overrides_autodecision(self):
        assert should_fold_vocab_table(
            table_bytes=10_000_000, weight_bytes=1, force=True) is True

    def test_force_false_overrides_autodecision(self):
        assert should_fold_vocab_table(
            table_bytes=1, weight_bytes=10_000_000, force=False) is False


# ---------------------------------------------------------------------------
# _lutstack_weight_bytes: cbackendが実際に出すバイト数と一致すること
# ---------------------------------------------------------------------------
class TestLutstackWeightBytes:
    def test_single_layer_matches_hand_computed_bytes(self):
        """1層(n_inputs=2、out_dim=4、in_dim=8)の手計算値と一致することを確認する。

        deploy/cbackend.py::emit_lut_dense の算術(添字はin_dim<=65536
        ならuint16_t、そうでなければint32_t):
            indices: n_inputs * out_dim 個 × (in_dim=8<=65536なので2byte)
            truth_tables: slice_truth_tables(...) のu32語数 * 4byte
        n_inputs=2, out_dim=4 なら indices=2*4*2=16byte、
        truth_tablesは wpr=ceil(4/32)=1、(1<<2)*wpr=4語*4byte=16byte、
        合計32byte。
        """
        torch.manual_seed(0)
        stack = LutStack(in_dim=8, layer_sizes=[4], n_inputs=2)
        assert _lutstack_weight_bytes(stack) == 32

    def test_multi_layer_sums_across_layers(self):
        torch.manual_seed(0)
        stack = LutStack(in_dim=8, layer_sizes=[4], n_inputs=2)
        one_layer_bytes = _lutstack_weight_bytes(stack)
        stack2 = LutStack(in_dim=8, layer_sizes=[4, 4], n_inputs=2)
        assert _lutstack_weight_bytes(stack2) == 2 * one_layer_bytes

    def test_large_in_dim_falls_back_to_i32_indices(self):
        """in_dim>65536(u16の値域外)ではemit_lut_denseと同じくint32_tへ
        フォールバックする。paper_scale1のM-layers級で実際に踏む経路
        (デプロイキット設計メモ節)。

        n_inputs=2, out_dim=4, in_dim=70000(>65536)なら
        indices=2*4*4=32byte(4byte/index)、truth_tablesは前テストと同じ
        16byte、合計48byte(u16なら32byteになるところ、ならないことを固定)。
        """
        torch.manual_seed(0)
        stack = LutStack(in_dim=70000, layer_sizes=[4], n_inputs=2)
        assert _lutstack_weight_bytes(stack) == 48


# ---------------------------------------------------------------------------
# extract_folded_table: 直接simulateした値と一致すること
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tiny_seq2seq():
    return build_rddlgn_tiny_seq2seq(seed=0)


class TestExtractFoldedTable:
    def test_n_table_folds_for_tiny_config(self, tiny_seq2seq):
        """RDDLGNConfig.tiny()(vocab=50, n_layer_sizes=[512,512])は表が
        重みより小さいので自動的に折り畳まれる(事前試算どおり)。"""
        model, _cfg = tiny_seq2seq
        table = extract_folded_n_table(model)
        assert table is not None
        assert table['vocab'] == 50
        assert table['width'] == model.encoder.n_layers.out_dim

    def test_l_table_folds_for_tiny_config(self, tiny_seq2seq):
        model, _cfg = tiny_seq2seq
        table = extract_folded_l_table(model)
        assert table is not None
        assert table['vocab'] == 50
        assert table['width'] == model.decoder.l_layers.out_dim

    def test_force_false_returns_none(self, tiny_seq2seq):
        model, _cfg = tiny_seq2seq
        assert extract_folded_n_table(model, force=False) is None
        assert extract_folded_l_table(model, force=False) is None

    def test_n_table_rows_match_direct_simulate(self, tiny_seq2seq):
        """折り畳み表の各行 == N-layersのbodyを独立に組み立てて直接simulate
        した結果(このテストは extract_folded_table とは別に body を
        組み立てるので、パッキング/インデックスのバグを検出できる)。
        """
        model, _cfg = tiny_seq2seq
        table = extract_folded_n_table(model, force=True)
        embed = _embed_hard_bits(model.src_embedding)
        d_emb = embed.shape[1]

        sv = IRNode(id='sv_step', type='scan_var',
                   params={'width': d_emb, 'role': 'step'})
        nodes, edges, last, last_w = _lutstack_nodes(
            model.encoder.n_layers, 'chk', 'sv_step', d_emb)
        body = IRGraph(nodes=[sv] + nodes, edges=edges)

        rng = np.random.default_rng(0)
        width = table['width']
        wpr = (width + 31) // 32
        bits = table['bits']
        for tok in rng.integers(0, table['vocab'], size=8):
            tok = int(tok)
            vals = _simulate_values(
                body, {'sv_step': embed[tok:tok + 1].astype(np.int64)})
            want = vals[last][0]
            row = bits[tok * wpr:(tok + 1) * wpr]
            got = np.array(
                [(row[j >> 5] >> (j & 31)) & 1 for j in range(width)],
                dtype=np.int64)
            assert np.array_equal(got, want), f'token={tok}'

    def test_l_table_uses_tgt_embedding_when_not_shared(self):
        """shared_vocab=Falseのとき、L折り畳み表はtgt_embeddingの行を
        使う(src_embeddingではない)ことを確認する。"""
        torch.manual_seed(1)
        cfg = dataclasses.replace(RDDLGNConfig.tiny(), shared_vocab=False)
        model = RDDLGN(cfg)
        model.eval()
        assert model.tgt_embedding is not model.src_embedding

        l_table = extract_folded_l_table(model, force=True)
        tgt_embed = _embed_hard_bits(model.tgt_embedding)
        d_emb = tgt_embed.shape[1]
        sv = IRNode(id='sv_step', type='scan_var',
                   params={'width': d_emb, 'role': 'step'})
        nodes, edges, last, last_w = _lutstack_nodes(
            model.decoder.l_layers, 'chk', 'sv_step', d_emb)
        body = IRGraph(nodes=[sv] + nodes, edges=edges)
        vals = _simulate_values(
            body, {'sv_step': tgt_embed[3:4].astype(np.int64)})
        want = vals[last][0]

        width = l_table['width']
        wpr = (width + 31) // 32
        row = l_table['bits'][3 * wpr:4 * wpr]
        got = np.array(
            [(row[j >> 5] >> (j & 31)) & 1 for j in range(width)],
            dtype=np.int64)
        assert np.array_equal(got, want)

    def test_large_vocab_small_layer_auto_disables_folding(self):
        """語彙が大きくN-layersが小さいと、表(vocab*width)が重みより
        大きくなり自動判定でfoldしない(paper_scale1級の想定を縮小再現)。
        """
        torch.manual_seed(2)
        cfg = dataclasses.replace(
            RDDLGNConfig.tiny(), vocab_size=200_000, d_emb=8,
            n_layer_sizes=[4], seq_len=2,
            m_layer_sizes=[8, 8, 200_000])
        model = RDDLGN(cfg)
        model.eval()
        assert extract_folded_n_table(model) is None


# ---------------------------------------------------------------------------
# bundle検証: _check_folded_table
# ---------------------------------------------------------------------------
class TestCheckFoldedTable:
    def test_none_is_noop(self):
        _check_folded_table('folded_n_table', None)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match='必須キー'):
            _check_folded_table('folded_n_table', {'vocab': 2, 'width': 4})

    def test_nonpositive_dims_raise(self):
        with pytest.raises(ValueError, match='正でなければならない'):
            _check_folded_table(
                'folded_n_table', {'vocab': 0, 'width': 4, 'bits': []})

    def test_wrong_bits_length_raises(self):
        with pytest.raises(ValueError, match='bits長'):
            _check_folded_table(
                'folded_n_table', {'vocab': 2, 'width': 4, 'bits': [0]})

    def test_out_of_range_word_raises(self):
        with pytest.raises(ValueError, match='u32の範囲外'):
            _check_folded_table(
                'folded_n_table',
                {'vocab': 1, 'width': 4, 'bits': [1 << 33]})

    def test_valid_table_passes(self):
        _check_folded_table(
            'folded_n_table', {'vocab': 2, 'width': 4, 'bits': [0, 0]})

    def test_build_bundle_roundtrips_folded_sections(self, tiny_seq2seq):
        """build_bundleがdocのfolded_n_table/folded_l_tableをそのまま
        引き継ぎ、validate_bundleを通ることを確認する(embed_tableと
        同じ扱いであることの回帰確認)。"""
        from griffith_logic.deploy.bundle import validate_bundle

        model, cfg = tiny_seq2seq
        doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
        doc['folded_n_table'] = extract_folded_n_table(model, force=True)
        doc['folded_l_table'] = extract_folded_l_table(model, force=True)
        bundle = build_bundle(doc)
        assert bundle['folded_n_table'] is doc['folded_n_table']
        assert bundle['folded_l_table'] is doc['folded_l_table']
        assert validate_bundle(bundle) == []
