"""L4: DWNのトイ問題学習収束スモーク。

DWNのforwardは常にハード(二値LUT+argmax接続)なので、soft/hardの
離散化ギャップが構造的に存在しない。訓練accがそのまま回路の精度になる。
合格基準: XOR / parity3 で acc 100%。
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.dwn import DWNClassifier

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _bits_dataset(n_bits, label_fn):
    """全2^n_bits入力とラベルの完全データセット"""
    xs = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=n_bits))
    )
    ys = torch.tensor([label_fn(row) for row in xs.long().tolist()])
    return xs, ys


def _train_until(model, x, y, *, lr=0.01, max_steps=2000, check_every=25):
    """訓練し、eval accが100%になったら早期終了。最終accを返す"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def _model(in_bits, n_classes, widths, *, n_inputs, mapping='random', efd='paper'):
    torch.manual_seed(0)
    # 入力は既に{0,1}なのでbits=1のuniform符号化(しきい値0.5)が恒等変換になる
    return DWNClassifier(
        in_bits, n_classes, layer_widths=widths, bits=1, encoder_mode='uniform',
        n_inputs=n_inputs, mapping=mapping, efd=efd,
    )


class TestXOR:
    @pytest.mark.parametrize('efd', ['paper', 'official'])
    def test_xor(self, efd, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _model(2, 2, (16, 16), n_inputs=2, efd=efd)
        acc = _train_until(model, x, y, max_steps=1000)
        assert acc == 1.0, f'XOR({efd}): acc={acc}'

    def test_xor_learnable_mapping(self, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _model(2, 2, (16, 16), n_inputs=2, mapping='learnable')
        acc = _train_until(model, x, y, max_steps=1000)
        assert acc == 1.0, f'XOR(learnable): acc={acc}'


class TestParity:
    @pytest.mark.parametrize('n_inputs', [2, 4])
    def test_parity3(self, n_inputs, seed):
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _model(3, 2, (32, 32), n_inputs=n_inputs)
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'parity3 n={n_inputs}: acc={acc}'


class TestSpectralPenalty:
    def test_penalty_backward(self, seed):
        """spectral_penaltyが損失に加算でき勾配が流れる"""
        x, y = _bits_dataset(2, lambda b: b[0] & b[1])
        model = _model(2, 2, (8,), n_inputs=2)
        loss = F.cross_entropy(model(x), y) + 0.01 * model.spectral_penalty()
        loss.backward()
        for layer in model.layers:
            assert layer.luts.grad is not None
            assert torch.isfinite(layer.luts.grad).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
