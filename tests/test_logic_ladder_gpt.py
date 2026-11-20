"""LogicLadderGPT(D1: 自己回帰言語モデルの論理ゲート化)のユニットテスト。

極小次元で高速スイート(2〜3秒以内)に収める。ladder等価テストは
test_textcnn.py 方式で experiments/attention_ladder/src を sys.path 追加して
`LadderLanguageModel` を import し、同一 state_dict の forward 一致を確認する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models import LogicLadderGPT, LogicLadderGPTConfig

# ------------------------------------------------------------
# テスト用の極小構成(L0=fp/fp, logic=射影ゲート化+選択器+論理FFN)
# ------------------------------------------------------------
_VOCAB = 17
_SEQ_LEN = 10

_L0_CFG = {
    'd_model': 8, 'n_heads': 2, 'n_layers': 2, 'mlp_ratio': 2,
    'attention': 'fp', 'ffn': 'fp', 'dropout': 0.0,
}
# l3c_shakespeare.yaml 相当(選択器MUX構成)。数値次元のみ極小化
_L3C_CFG = {
    'd_model': 8, 'n_heads': 2, 'n_layers': 2, 'mlp_ratio': 2,
    'attention': 'logic_proj', 'ffn': 'logic', 'score': 'xnor_popcount',
    'norm': 'mux_gumbel', 'qk_bits': 8, 'v_bits': 8, 'n_inputs': 2,
    'parameterization': 'warp', 'proj_depth': 1, 'tau': 1.0, 'dropout': 0.0,
}


def _build(model_cfg: dict, seed: int = 0) -> LogicLadderGPT:
    """model_cfg から LogicLadderGPT を決定的に構築する。"""
    torch.manual_seed(seed)
    cfg = LogicLadderGPTConfig.from_model_cfg(
        model_cfg, vocab_size=_VOCAB, seq_len=_SEQ_LEN,
    )
    return LogicLadderGPT(cfg)


# ============================================================
# 1. causality: 未来トークン改変が位置tまでのlogitsに影響しない
# ============================================================

class TestCausality:
    """位置tより未来の入力を改変しても位置tまでのlogitsがbitwise不変。"""

    @pytest.mark.parametrize('model_cfg', [_L0_CFG, _L3C_CFG])
    def test_future_tokens_do_not_leak(self, model_cfg):
        model = _build(model_cfg)
        model.eval()  # 選択器(mux_gumbel)を決定的なhard argmaxにする
        torch.manual_seed(123)
        x = torch.randint(0, _VOCAB, (3, _SEQ_LEN))

        with torch.no_grad():
            base = model(x)

        cut = _SEQ_LEN // 2  # 位置 cut までは未来改変の影響を受けない
        x_mod = x.clone()
        # cut より後(未来)の全トークンを別の値へ改変
        x_mod[:, cut + 1:] = (x[:, cut + 1:] + 1) % _VOCAB
        with torch.no_grad():
            mod = model(x_mod)

        # 位置 0..cut の logits は bitwise 不変
        assert torch.equal(base[:, : cut + 1], mod[:, : cut + 1])
        # 未来を変えたので cut より後は変化する(causalマスクの健全性確認)
        assert not torch.equal(base[:, cut + 1:], mod[:, cut + 1:])


# ============================================================
# 2. ladder等価: 同一state_dictでLadderLanguageModelとforward一致(atol=0)
# ============================================================

class TestLadderEquivalence:
    """experiments/attention_ladder の LadderLanguageModel と数値等価。"""

    @staticmethod
    def _import_ladder():
        ladder_src = os.path.join(
            os.path.dirname(__file__), '..',
            'experiments', 'attention_ladder', 'src',
        )
        sys.path.insert(0, ladder_src)
        from models import LadderLanguageModel  # noqa: E402
        return LadderLanguageModel

    @pytest.mark.parametrize('model_cfg', [_L0_CFG, _L3C_CFG])
    def test_forward_matches_ladder(self, model_cfg):
        ladder_cls = self._import_ladder()
        meta = {'task': 'lm', 'seq_len': _SEQ_LEN, 'vocab_size': _VOCAB}
        ladder = ladder_cls(model_cfg, meta)

        cfg = LogicLadderGPTConfig.from_model_cfg(
            model_cfg, vocab_size=_VOCAB, seq_len=_SEQ_LEN,
        )
        gpt = LogicLadderGPT(cfg)

        # サブモジュール名が一致するため state_dict を直接転写できる
        ladder.load_state_dict(gpt.state_dict(), strict=True)

        ladder.eval()
        gpt.eval()
        torch.manual_seed(7)
        x = torch.randint(0, _VOCAB, (3, _SEQ_LEN))
        with torch.no_grad():
            out_ladder = ladder(x)
            out_gpt = gpt(x)
        # atol=0(bitwise一致)
        assert torch.equal(out_ladder, out_gpt)


# ============================================================
# 3. generate: shape / greedy決定性 / 再現性 / top_k=1==greedy
# ============================================================

class TestGenerate:
    """自己回帰生成の契約(shape・決定性・再現性)。"""

    def test_output_shape(self):
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 3))
        out = model.generate(idx, max_new_tokens=5, greedy=True)
        assert out.shape == (2, 3 + 5)

    def test_context_crop(self):
        """入力が seq_len を超えても(末尾クロップで)生成できる。"""
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, _SEQ_LEN + 4))
        out = model.generate(idx, max_new_tokens=3, greedy=True)
        assert out.shape == (2, _SEQ_LEN + 4 + 3)

    def test_greedy_deterministic(self):
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        a = model.generate(idx, max_new_tokens=6, greedy=True)
        b = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(a, b)

    def test_temperature_reproducible_with_generator(self):
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))

        g1 = torch.Generator().manual_seed(2024)
        a = model.generate(idx, max_new_tokens=6, temperature=1.0, generator=g1)
        g2 = torch.Generator().manual_seed(2024)
        b = model.generate(idx, max_new_tokens=6, temperature=1.0, generator=g2)
        assert torch.equal(a, b)

    def test_top_k_1_equals_greedy(self):
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        g = torch.Generator().manual_seed(1)
        sampled = model.generate(
            idx, max_new_tokens=6, temperature=1.0, top_k=1, generator=g,
        )
        greedy = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(sampled, greedy)

    def test_temperature_zero_is_greedy(self):
        model = _build(_L0_CFG)
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        zero = model.generate(idx, max_new_tokens=6, temperature=0.0)
        greedy = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(zero, greedy)


# ============================================================
# 4. weight tying: head と token_embed の共有・両経路の勾配
# ============================================================

class TestWeightTying:
    """weight_tying=True で head と embed が重み共有し、勾配が流れる。"""

    def test_shared_weight_identity(self):
        cfg = LogicLadderGPTConfig.from_model_cfg(_L0_CFG, vocab_size=_VOCAB, seq_len=_SEQ_LEN)
        cfg.weight_tying = True
        model = LogicLadderGPT(cfg)
        assert model.head.weight is model.token_embed.weight

    def test_untied_by_default(self):
        model = _build(_L0_CFG)
        assert model.head.weight is not model.token_embed.weight

    def test_gradient_flows_both_paths(self):
        cfg = LogicLadderGPTConfig.from_model_cfg(_L0_CFG, vocab_size=_VOCAB, seq_len=_SEQ_LEN)
        cfg.weight_tying = True
        model = LogicLadderGPT(cfg)
        model.train()

        idx = torch.randint(0, _VOCAB, (2, _SEQ_LEN))
        logits = model(idx)
        # 埋め込み経路とヘッド経路の双方から共有重みへ勾配が流れる
        loss = logits.sum()
        loss.backward()

        shared = model.head.weight
        assert shared.grad is not None
        assert not torch.all(shared.grad == 0)

    def test_count_parameters_no_double_count(self):
        cfg = LogicLadderGPTConfig.from_model_cfg(_L0_CFG, vocab_size=_VOCAB, seq_len=_SEQ_LEN)
        cfg.weight_tying = True
        model = LogicLadderGPT(cfg)
        counts = model.count_parameters()
        # 共有重みは total で二重計上しない(実体ベース)
        unique = sum(
            p.numel() for p in {id(p): p for p in model.parameters()}.values()
        )
        assert counts['total'] == unique


# ============================================================
# 5. config変換: l3c相当の model_cfg 辞書から from_model_cfg で構築
# ============================================================

class TestConfigConversion:
    """yamlの model_cfg 辞書 → Config への変換ヘルパ。"""

    def test_from_model_cfg_l0(self):
        cfg = LogicLadderGPTConfig.from_model_cfg(_L0_CFG, vocab_size=_VOCAB, seq_len=_SEQ_LEN)
        assert cfg.attention == 'fp'
        assert cfg.ffn == 'fp'
        assert cfg.attention_kwargs == {}  # 上位キーのみ、透過キーなし

    def test_from_model_cfg_l3c(self):
        cfg = LogicLadderGPTConfig.from_model_cfg(_L3C_CFG, vocab_size=_VOCAB, seq_len=_SEQ_LEN)
        assert cfg.attention == 'logic_proj'
        assert cfg.ffn == 'logic'
        # 選択器種別・温度等が attention_kwargs に透過されている
        assert cfg.attention_kwargs['score'] == 'xnor_popcount'
        assert cfg.attention_kwargs['norm'] == 'mux_gumbel'
        assert cfg.attention_kwargs['n_inputs'] == 2
        # 実際に構築でき forward が通る
        model = LogicLadderGPT(cfg)
        x = torch.randint(0, _VOCAB, (2, _SEQ_LEN))
        out = model(x)
        assert out.shape == (2, _SEQ_LEN, _VOCAB)
