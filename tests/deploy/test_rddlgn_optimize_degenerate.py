"""RDDLGNの退化ゲート除去パス(export/rddlgn_optimize.py)の検証。

静的解析(デプロイキット設計メモ節)で学習済みRDDLGNの
38.5〜46.5%のゲートが退化真理値表(定数0/1・恒等A/B・NOT A/B)である
ことが分かったので、LutStackを跨いだ「退化ゲート除去→下流indices
張り替え」パスを実装した(opt-in、既定off)。

固定すること:
    1. optimize_degenerate=False(既定)は既存の`_lutstack_nodes`経路と
       完全に同じ(このファイルでは境界条件だけ軽く触れ、主眼は
       tests/deploy/test_rddlgn_*.py の既存127件がoptimize_degenerate
       未指定=Falseのまま全緑であることに委ねる)
    2. optimize_degenerate=True: simulateの最終出力(教師強制)・
       自己回帰生成トークン列(greedy)が、False側と全サンプルで
       完全一致する(タイニー合成モデルで高速に、実チェックポイント
       3本でslowマークして厳密に)
    3. 物理ゲート数(count_lut_gates相当)が実際に減っていること
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.memplan import check_no_overlap, plan_memory
from griffith_logic.deploy.plan import build_plan
from griffith_logic.export.rddlgn_ir import (
    export_rddlgn_classifier_ir,
    export_rddlgn_encoder_ir,
    export_rddlgn_ir,
    simulate_rddlgn_greedy_bits,
)
from griffith_logic.ir.serialize import ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

from .conftest import build_rddlgn_tiny_classifier, build_rddlgn_tiny_seq2seq

REPO = Path(__file__).resolve().parents[2]

CKPTS = {
    'copy_tiny': (REPO / 'checkpoints/rddlgn_copy_tiny_20260815_074318/best.pt',
                 1, 2),
    'reverse_custom': (
        REPO / 'checkpoints/rddlgn_reverse_custom_20260815_081056/best.pt',
        1, 2),
    'charlm_scale005': (
        REPO / 'checkpoints/rddlgn_charlm_scale005/'
              '2wsppkmu_alpha1.0_pureCE_best.pt',
        66, 66_000),
}


def _load_ckpt(path: Path) -> tuple[RDDLGN, RDDLGNConfig]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = RDDLGNConfig(**ck['metadata']['config'])
    model = RDDLGN(cfg)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    return model, cfg


def _count_lut_gates(graph) -> int:
    total = 0

    def walk(g):
        nonlocal total
        for node in g.nodes:
            if node.type == 'lut_layer':
                total += int(node.params['out_width'])
            elif node.type == 'recurrent':
                walk(node.params['body'])

    walk(graph)
    return total


def _embed_bits(model, src_tokens: torch.Tensor,
                tgt_tokens: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        src_emb = model.src_embedding(src_tokens)
        tgt_emb = model.tgt_embedding(tgt_tokens)
    batch = src_tokens.shape[0]
    src_bits = src_emb.numpy().astype(np.int64).reshape(batch, -1)
    tgt_bits = tgt_emb.numpy().astype(np.int64).reshape(batch, -1)
    return np.concatenate([src_bits, tgt_bits], axis=1)


# ---------------------------------------------------------------------------
# 高速: タイニー合成モデル(常時グリーンの高速スイート)
# ---------------------------------------------------------------------------
class TestTinySynthetic:
    def test_encoder_matches_and_validates(self):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        seq_len = cfg.seq_len
        g_off, _, _ = export_rddlgn_encoder_ir(model.encoder, seq_len)
        g_on, _, _ = export_rddlgn_encoder_ir(
            model.encoder, seq_len, optimize_degenerate=True)
        assert validate(g_off) == []
        assert validate(g_on) == []

        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(8, g_off.inputs[0].width))
        want = simulate(g_off, x)
        got = simulate(g_on, x)
        assert np.array_equal(want, got)

    def test_classifier_matches(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=1, n_classes=3)
        seq_len = cfg.seq_len
        g_off = export_rddlgn_classifier_ir(model, seq_len)
        g_on = export_rddlgn_classifier_ir(
            model, seq_len, optimize_degenerate=True)
        assert validate(g_off) == []
        assert validate(g_on) == []

        rng = np.random.default_rng(1)
        x = rng.integers(0, 2, size=(8, g_off.inputs[0].width))
        assert np.array_equal(simulate(g_off, x), simulate(g_on, x))

    def test_seq2seq_teacher_forcing_matches(self):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=2)
        src_len = tgt_len = cfg.seq_len
        g_off = export_rddlgn_ir(model, src_len, tgt_len)
        g_on = export_rddlgn_ir(model, src_len, tgt_len,
                                optimize_degenerate=True)
        assert validate(g_off) == []
        assert validate(g_on) == []

        rng = np.random.default_rng(2)
        batch = 6
        src_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, src_len)),
            dtype=torch.long)
        tgt_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, tgt_len)),
            dtype=torch.long)
        x = _embed_bits(model, src_tokens, tgt_tokens)
        want = simulate(g_off, x)
        got = simulate(g_on, x)
        assert np.array_equal(want, got)

    def test_greedy_matches(self):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=3)
        rng = np.random.default_rng(3)
        for _ in range(4):
            src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,))
            with torch.no_grad():
                embed = (torch.sigmoid(
                    model.src_embedding.embedding.weight) > 0.5)
            embed = embed.to(torch.int64).cpu().numpy()
            src_bits = embed[src_tokens].reshape(-1)
            want = simulate_rddlgn_greedy_bits(
                model, src_bits, bos_token=1, eos_token=2,
                max_new_tokens=cfg.seq_len)
            got = simulate_rddlgn_greedy_bits(
                model, src_bits, bos_token=1, eos_token=2,
                max_new_tokens=cfg.seq_len, optimize_degenerate=True)
            assert want == got

    def test_gate_count_reduced_or_equal(self):
        """タイニーモデルは学習不十分/乱数初期化に近く退化率が低いこともある
        ので、削減率の下限は課さず「増えない」ことだけ固定する。実チェック
        ポイント(TestRealCheckpoints)側で実際の削減率を厳密に見る。"""
        model, cfg = build_rddlgn_tiny_seq2seq(seed=4)
        g_off = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len)
        g_on = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                optimize_degenerate=True)
        before = _count_lut_gates(g_off)
        after = _count_lut_gates(g_on)
        assert after <= before


# ---------------------------------------------------------------------------
# 実チェックポイント3本(重い。slowマーク)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestRealCheckpoints:
    @pytest.mark.parametrize('name', list(CKPTS))
    def test_teacher_forcing_and_greedy_match_and_gates_reduced(self, name):
        path, bos, eos = CKPTS[name]
        if not path.exists():
            pytest.skip(f'checkpoint not found: {path}')
        model, cfg = _load_ckpt(path)
        src_len = tgt_len = cfg.seq_len

        g_off = export_rddlgn_ir(model, src_len, tgt_len)
        g_on = export_rddlgn_ir(model, src_len, tgt_len,
                                optimize_degenerate=True)
        assert validate(g_off) == []
        assert validate(g_on) == []

        before = _count_lut_gates(g_off)
        after = _count_lut_gates(g_on)
        assert after < before, f'{name}: ゲート数が削減されていない'
        print(f'\n  [{name}] 物理ゲート数 {before:,} -> {after:,} '
              f'({100.0 * (1 - after / before):.1f}%削減)')

        # 教師強制: simulateの最終出力(スコア)が全サンプル一致
        rng = np.random.default_rng(0)
        batch = 8
        src_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, src_len)),
            dtype=torch.long)
        tgt_tokens = torch.tensor(
            rng.integers(0, cfg.vocab_size, size=(batch, tgt_len)),
            dtype=torch.long)
        x = _embed_bits(model, src_tokens, tgt_tokens)
        want_scores = simulate(g_off, x)
        got_scores = simulate(g_on, x)
        assert np.array_equal(want_scores, got_scores), (
            f'{name}: 教師強制のスコアが最適化あり/なしで不一致')

        # greedy: 生成トークン列が全サンプル一致(shared_vocab限定)
        if cfg.shared_vocab:
            with torch.no_grad():
                embed = (torch.sigmoid(
                    model.src_embedding.embedding.weight) > 0.5)
            embed = embed.to(torch.int64).cpu().numpy()
            rng2 = np.random.default_rng(1)
            for _ in range(6):
                src = rng2.integers(0, cfg.vocab_size, size=(src_len,))
                src_bits = embed[src].reshape(-1)
                want = simulate_rddlgn_greedy_bits(
                    model, src_bits, bos_token=bos, eos_token=eos,
                    max_new_tokens=src_len)
                got = simulate_rddlgn_greedy_bits(
                    model, src_bits, bos_token=bos, eos_token=eos,
                    max_new_tokens=src_len, optimize_degenerate=True)
                assert want == got, (
                    f'{name}: greedy生成トークン列が最適化あり/なしで不一致\n'
                    f'want={want}\ngot={got}')


# ---------------------------------------------------------------------------
# plan.py/memplan.pyのアキュムレータalias機構
# (退化ゲート除去のレイテンシ回帰修正、デプロイキット設計メモ節)
# ---------------------------------------------------------------------------
class TestK26bAccumulateAlias:
    """`deploy/plan.py::_plan_accumulate_concat`が導入するalias機構の
    直接検証。旧設計では中間層のconcatが毎回`flat_concat`で実コピーして
    おり(O(層数)回・幅が単調増加)、これがcharlmのencode+96%回帰の原因
    だった。グループの最初のconcatでだけ実バッファを1本確保して
    1回だけentryを物理コピーし、以降はlut_denseの出力を「親バッファ+
    オフセット」のaliasへ差し替えるだけにした。ここではop列レベルで
    flat_concatの数が減っていること、plan_memory/check_no_overlapが
    壊れていないこと(タスク要求「memplanのcheck_no_overlapとliveness
    を壊さないこと」)を固定する。
    """

    def _bundle(self, seed=5, optimize_degenerate=True):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=seed)
        graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                 optimize_degenerate=optimize_degenerate)
        assert validate(graph) == []
        doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
        return build_bundle(doc)

    def test_flat_concat_op_count_is_one_per_group_not_per_layer(self):
        """のaccumulate_group付きconcatノード数に対し、実際に
        flat_concat opとして実体化される「シード」(1入力のflat_concat、
        `_plan_accumulate_concat`が最初のconcatでだけ発行する)は
        グループごとに高々1つ。旧設計は各concatノードが毎回
        (2入力の)flat_concatを1つ発行していたので、この数はconcatノード数と
        一致していた——ここではconcatノード数 > シードop数を確認する
        (由来の通常concat(state/step束ね、2入力以上)はシードの
        1入力という形と区別できるので、この数に混ざらない)。
        """
        from griffith_logic.ir.serialize import ir_from_dict

        bundle = self._bundle()
        plan = build_plan(bundle)
        n_seed_ops = sum(1 for op in plan.ops
                         if op.kind == 'flat_concat' and len(op.inputs) == 1)
        n_concat_nodes = 0

        def walk(g):
            nonlocal n_concat_nodes
            for node in g.nodes:
                if node.type == 'concat' and node.params.get('accumulate_group'):
                    n_concat_nodes += 1
                elif node.type == 'recurrent':
                    walk(node.params['body'])

        walk(ir_from_dict(bundle['ir']))
        assert n_concat_nodes > 0, (
            'このseedではaccumulate-groupのconcatノードが1つも無い'
            '(テスト前提を満たさない)')
        assert 0 < n_seed_ops < n_concat_nodes, (
            f'シードop数({n_seed_ops})はconcatノード数({n_concat_nodes})'
            '未満(グループごとに高々1つ)のはず')

    def test_memplan_excludes_aliases_and_no_overlap_holds(self):
        bundle = self._bundle()
        plan = build_plan(bundle)
        aliased = [v for v in plan.values if v.alias_of is not None]
        assert aliased, 'このseedではalias値が1つも無い(テスト前提を満たさない)'

        mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
        for v in aliased:
            assert v.id not in mem.offsets, (
                f'alias値v{v.id}はプールから除外されているはず'
                '(独自領域を持たないので独立割り当てされてはいけない)')
        check_no_overlap(plan, mem)  # 例外が飛ばなければ成功

    def test_check_no_overlap_catches_out_of_bounds_alias(self):
        """故意にalias_offsetを親バッファの範囲外へ壊し、check_no_overlap
        が検出することを固定する(要求「memplanのcheck_no_overlapを
        壊さないこと」の裏取り — 壊れたら黙って通らずここで落ちる)。
        """
        bundle = self._bundle()
        plan = build_plan(bundle)
        aliased = [v for v in plan.values if v.alias_of is not None]
        assert aliased
        v = aliased[0]
        parent = plan.value(v.alias_of)
        v.alias_offset = parent.nbytes() + 1000   # 明らかに範囲外
        mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
        with pytest.raises(AssertionError, match='はみ出している'):
            check_no_overlap(plan, mem)

    def test_check_no_overlap_catches_overlapping_siblings(self):
        """同じ親を指す2つのalias値のオフセット範囲を故意に重ねて、
        検出されることを固定する。"""
        bundle = self._bundle()
        plan = build_plan(bundle)
        aliased = [v for v in plan.values if v.alias_of is not None]
        by_parent: dict = {}
        for v in aliased:
            by_parent.setdefault(v.alias_of, []).append(v)
        pair = next((vs for vs in by_parent.values() if len(vs) >= 2), None)
        if pair is None:
            pytest.skip('同一親を持つalias値が2つ以上あるグループが無い')
        a, b = pair[0], pair[1]
        b.alias_offset = a.alias_offset  # 意図的に衝突させる
        mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
        with pytest.raises(AssertionError, match='重なっている'):
            check_no_overlap(plan, mem)
