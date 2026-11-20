"""LogicWaveNet(dilated causal conv の論理ゲート化)のユニットテスト。

極小次元で高速スイートに収める。因果性(未来非漏洩+受容野)を最重要テストとし、
shape契約・勾配疎通・hard離散性・generate契約・極小収束を検証する。

models/__init__.py は他エージェント並列作業中のため、モジュールから直接importする。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.logic_wavenet import LogicWaveNet, LogicWaveNetConfig

_VOCAB = 17


def _build(seed: int = 0, **overrides) -> LogicWaveNet:
    """極小構成の LogicWaveNet を決定的に構築する。"""
    torch.manual_seed(seed)
    cfg_kwargs = dict(
        vocab_size=_VOCAB, d_channels=16, n_layers=3, kernel_size=2,
    )
    cfg_kwargs.update(overrides)
    return LogicWaveNet(LogicWaveNetConfig(**cfg_kwargs))


# ============================================================
# 1. causality(最重要): 未来非漏洩 + 受容野の検証
# ============================================================

class TestCausality:
    """dilated causal conv の因果性: 未来トークンは過去のlogitsに漏れない。"""

    def test_future_tokens_do_not_leak(self):
        """位置tより未来の入力を改変しても位置tまでのlogitsがbitwise不変。"""
        model = _build()
        model.eval()
        torch.manual_seed(123)
        t = 30
        x = torch.randint(0, _VOCAB, (3, t))
        with torch.no_grad():
            base = model(x)

        # 末尾トークンを改変(最も未来)
        x_mod = x.clone()
        x_mod[:, -1] = (x[:, -1] + 1) % _VOCAB
        with torch.no_grad():
            mod = model(x_mod)

        # 末尾を除く全位置の logits は bitwise 不変
        assert torch.equal(base[:, :-1], mod[:, :-1])
        # 末尾は当然変化する(自己位置は自分を見る)
        assert not torch.equal(base[:, -1], mod[:, -1])

    @pytest.mark.parametrize('n_layers', [2, 3, 4])
    def test_receptive_field(self, n_layers):
        """位置pの改変は logits[t<p](因果)と logits[t>=p+RF](受容野外)に不影響、
        受容野内 [p, p+RF) の一部は変化する。RF = 2^n_layers(kernel=2)。"""
        model = _build(n_layers=n_layers)
        model.eval()
        rf = model.receptive_field
        assert rf == 2 ** n_layers  # kernel_size=2 の受容野

        torch.manual_seed(7)
        t = rf + 10
        x = torch.randint(0, _VOCAB, (3, t))
        with torch.no_grad():
            base = model(x)

        p = 2
        x_mod = x.clone()
        x_mod[:, p] = (x[:, p] + 1) % _VOCAB
        with torch.no_grad():
            mod = model(x_mod)

        # 位置p未満は因果性で不変
        assert torch.equal(base[:, :p], mod[:, :p])
        # 受容野の外(t >= p+RF)は不変
        assert torch.equal(base[:, p + rf:], mod[:, p + rf:])
        # 受容野内 [p, p+RF) は変化する
        assert not torch.equal(base[:, p: p + rf], mod[:, p: p + rf])


# ============================================================
# 2. shape契約
# ============================================================

class TestShape:
    """forward の入出力 shape 契約。"""

    def test_forward_shape(self):
        model = _build()
        x = torch.randint(0, _VOCAB, (4, 12))
        out = model(x)
        assert out.shape == (4, 12, _VOCAB)

    def test_receptive_field_formula(self):
        """RF = (K-1)*(2^n_layers-1)+1。kernel_size=3 でも成立。"""
        model = _build(kernel_size=3, n_layers=2)
        assert model.receptive_field == (3 - 1) * (2 ** 2 - 1) + 1  # = 7

    def test_variable_length(self):
        """T が異なっても(位置埋め込みが無いため)通る。"""
        model = _build()
        for t in (1, 5, 33):
            out = model(torch.randint(0, _VOCAB, (2, t)))
            assert out.shape == (2, t, _VOCAB)


# ============================================================
# 3. 勾配疎通(最深層まで)
# ============================================================

class TestGradient:
    """埋め込み・全conv層・ヘッドまで勾配が流れる。"""

    def test_grad_flows_to_all_layers(self):
        model = _build(n_layers=4)
        model.train()
        x = torch.randint(0, _VOCAB, (2, 20))
        loss = model(x).sum()
        loss.backward()

        # 埋め込み
        assert model.embed.embedding.weight.grad is not None
        assert not torch.all(model.embed.embedding.weight.grad == 0)
        # ヘッド
        assert model.head.weight.grad is not None
        assert not torch.all(model.head.weight.grad == 0)
        # 全 filter/gate 層(最深層含む)
        for stack in list(model.filters) + list(model.gates):
            for layer in stack.layers:
                g = layer.weights.grad
                assert g is not None
                assert not torch.all(g == 0)


# ============================================================
# 4. hard離散性(eval時、conv層出力={0,1})
# ============================================================

class TestHardDiscreteness:
    """eval時、埋め込みとゲート付き活性 z=f*g が厳密に {0,1} になる。"""

    def test_embedding_is_binary(self):
        model = _build()
        model.eval()
        x = torch.randint(0, _VOCAB, (3, 15))
        with torch.no_grad():
            e = model.embed(x)
        uniq = torch.unique(e)
        assert torch.all((uniq == 0.0) | (uniq == 1.0))

    def test_gated_activation_is_binary(self):
        """z = filter ⊙ gate が eval時に厳密ビットAND({0,1})になる。"""
        model = _build()
        model.eval()
        x = torch.randint(0, _VOCAB, (3, 15))
        with torch.no_grad():
            e = model.embed(x)
            batch, seq_len, _ = e.shape
            for dilation, filt, gate in zip(
                model.dilations, model.filters, model.gates, strict=True,
            ):
                concat = model._dilated_concat(e, dilation)
                flat = concat.reshape(batch * seq_len, -1)
                z = filt(flat) * gate(flat)
                uniq = torch.unique(z)
                assert torch.all((uniq == 0.0) | (uniq == 1.0)), (
                    f'dilation={dilation} で z が非ブール: {uniq.tolist()}'
                )


# ============================================================
# 5. generate契約(LogicGPT と同一)
# ============================================================

class TestGenerate:
    """自己回帰生成の契約(shape・決定性・再現性)。"""

    def test_output_shape(self):
        model = _build()
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 3))
        out = model.generate(idx, max_new_tokens=5, greedy=True)
        assert out.shape == (2, 3 + 5)

    def test_greedy_deterministic(self):
        model = _build()
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        a = model.generate(idx, max_new_tokens=6, greedy=True)
        b = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(a, b)

    def test_temperature_zero_is_greedy(self):
        model = _build()
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        zero = model.generate(idx, max_new_tokens=6, temperature=0.0)
        greedy = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(zero, greedy)

    def test_temperature_reproducible_with_generator(self):
        model = _build()
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        g1 = torch.Generator().manual_seed(2024)
        a = model.generate(idx, max_new_tokens=6, temperature=1.0, generator=g1)
        g2 = torch.Generator().manual_seed(2024)
        b = model.generate(idx, max_new_tokens=6, temperature=1.0, generator=g2)
        assert torch.equal(a, b)

    def test_top_k_1_equals_greedy(self):
        model = _build()
        model.eval()
        idx = torch.randint(0, _VOCAB, (2, 4))
        g = torch.Generator().manual_seed(1)
        sampled = model.generate(
            idx, max_new_tokens=6, temperature=1.0, top_k=1, generator=g,
        )
        greedy = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(sampled, greedy)


# ============================================================
# 6. 極小収束: 周期パターンの次トークン予測 hard acc=100%
# ============================================================

class TestConvergence:
    """周期パターン 'aab'(次トークンが prev と curr の両方に依存)を学習し、
    hard eval で次トークン予測が 100% 一致する。dilated tap(x_{t-1})を
    正しく使わないと解けないため、因果convの学習能力を検証する。"""

    def test_periodic_pattern_hard_acc_100(self):
        torch.manual_seed(0)
        unit = [0, 0, 1]  # 'aab' 相当。next は prev,curr の両方に依存
        data = torch.tensor(unit * 40, dtype=torch.long)
        t = 30
        x = data[:t].unsqueeze(0)
        y = data[1: t + 1].unsqueeze(0)

        cfg = LogicWaveNetConfig(
            vocab_size=2, d_channels=32, n_layers=2, kernel_size=2,
        )
        model = LogicWaveNet(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)

        model.train()
        for _ in range(500):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        assert acc == 1.0, f'hard acc={acc} (期待1.0)'
