"""L3: DLGN(Petersen 2022)のMNIST再現(@paper)。

既定(quick): 短縮構成(784入力二値化・4層・tau≈28.28・20k×30ep)で
hard eval(離散回路)の回帰基準を固定する。
DLGN_FULL=1: difflogic公式レシピ(mnist20x20・6層・tau=10・bs=100・
訓練入力は連続値[0,1]・unique接続・200k iter)で論文水準を再現する。

実測較正(Modal T4/A10G):
    quick: 6k×20ep=0.8725 / 20k×30ep=0.9140 → 閾値0.90
    旧FULL(60k×60ep, 4層/tau28/二値化/先頭2000評価): 0.9285 → 旧閾値0.92
    レシピ監査: 論文97.69%との残差−4.4ptは複合レシピ差
    (τ=28.28で−1.0pt、訓練入力二値化−0.1pt、iter不足、先頭2000サンプル
    部分評価の測定差−2.6pt等)と特定。公式レシピ一致で
    50k iter=0.9739 / 200k iter=0.9784(全10k評価, A10G)を実測し、
    FULL構成を公式レシピに変更、閾値0.97に較正。
    詳細: DLGN レシピ監査レポート(内部)

実行: uv run pytest -m paper tests/l3_paper/test_dlgn_mnist.py
      (Modal: uv run modal run scripts/modal_app.py::pytest_cuda --marker paper)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer

pytestmark = pytest.mark.paper


FULL_MODE = os.environ.get('DLGN_FULL', '0') == '1'
N_TRAIN = 20000
N_EPOCHS = 30
THRESHOLD = 0.97 if FULL_MODE else 0.90
FULL_ITERATIONS = 200_000

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data-mnist')

_BLACK_MAX = 14   # 片側の黒行/列カウント上限(difflogic準拠)
_BORDER = 8       # 除去する黒行/列の合計(28→20)


def _trim_counts(black):
    """黒い行(列)のブール列(28,)から両端の除去量を決める(difflogic準拠)。"""
    lead = 0
    while black[lead] and lead < _BLACK_MAX:
        lead += 1
    trail = 0
    while black[-trail - 1] and trail < _BLACK_MAX:
        trail += 1
    while lead + trail >= _BORDER + 2:
        if lead > 0:
            lead -= 1
        if trail > 0:
            trail -= 1
    if lead + trail == _BORDER + 1:
        if lead > 0:
            lead -= 1
        else:
            trail -= 1
    assert lead + trail == _BORDER, (lead, trail)
    return lead, trail


def _remove_border(img):
    """(28,28)画像から黒境界を適応除去して(20,20)にする(difflogic準拠)。"""
    top, bottom = _trim_counts((img == 0.0).all(dim=1))
    img = img[top:28 - bottom]
    left, right = _trim_counts((img == 0.0).all(dim=0))
    return img[:, left:28 - right]


def _load_mnist20x20(device):
    """MNIST全件を境界除去済み(N,400)テンソルとして返す(公式レシピ準拠)。"""
    import torchvision

    out = []
    for train_flag in (True, False):
        ds = torchvision.datasets.MNIST(
            DATA_ROOT, train=train_flag, download=train_flag,
        )
        imgs = ds.data.to(torch.float32) / 255.0
        imgs = torch.stack([_remove_border(im) for im in imgs])
        out += [imgs.reshape(imgs.shape[0], -1).to(device),
                ds.targets.to(device)]
    return out


class TestDLGNMnist:
    def _run_full(self):
        """公式レシピ(difflogic README small: 97.69%)の再現。

        python experiments/main.py -bs 100 -t 10 --dataset mnist20x20
            -ni 200_000 -k 8_000 -l 6 相当。
        訓練入力は連続値のまま、評価時のみround()で二値化する。
        """
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
        train_x, train_y, test_x, test_y = _load_mnist20x20(device)

        torch.manual_seed(0)
        k = 8000
        llkw = dict(n_inputs=2, device=device, connections='unique')
        model = nn.Sequential(
            LutLayer(400, k, **llkw),
            LutLayer(k, k, **llkw),
            LutLayer(k, k, **llkw),
            LutLayer(k, k, **llkw),
            LutLayer(k, k, **llkw),
            LutLayer(k, k, **llkw),
            GroupSum(k=10, tau=10.0),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        bs = 100
        n = train_x.shape[0]
        steps_per_epoch = n // bs  # drop_last相当
        model.train()
        perm = None
        for it in range(FULL_ITERATIONS):
            step = it % steps_per_epoch
            if step == 0:
                perm = torch.randperm(n, device=device)
            idx = perm[step * bs:(step + 1) * bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, test_x.shape[0], 2000):
                x = test_x[s:s + 2000].round()
                correct += (model(x).argmax(-1) == test_y[s:s + 2000]).sum().item()
        acc = correct / test_x.shape[0]
        assert acc >= THRESHOLD, (
            f'DLGN MNIST(公式レシピ) hard acc = {acc:.4f} < {THRESHOLD}'
            f' (mnist20x20, 6層×8000, tau=10, {FULL_ITERATIONS} iter)'
        )

    def _run_quick(self):
        """短縮構成(従来較正済み): 784入力二値化・4層・tau≈28.28。"""
        import torchvision
        from torch.utils.data import DataLoader, Subset

        tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: (t > 0.5).float().flatten()),
        ])
        train = Subset(
            torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                       transform=tfm),
            range(N_TRAIN),
        )
        test = Subset(
            torchvision.datasets.MNIST(DATA_ROOT, train=False, transform=tfm),
            range(2000),
        )
        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        test_loader = DataLoader(test, batch_size=512)

        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')

        torch.manual_seed(0)
        k = 8000
        model = nn.Sequential(
            LutLayer(784, k, n_inputs=2, device=device),
            LutLayer(k, k, n_inputs=2, device=device),
            LutLayer(k, k, n_inputs=2, device=device),
            LutLayer(k, k, n_inputs=2, device=device),
            GroupSum(k=10, tau=(k // 10) ** 0.5),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(N_EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        acc = correct / total
        assert acc >= THRESHOLD, (
            f'DLGN MNIST hard acc = {acc:.4f} < {THRESHOLD}'
            f' (train={N_TRAIN}, epochs={N_EPOCHS})'
        )

    def test_mnist_subset(self, seed):
        if FULL_MODE:
            self._run_full()
        else:
            self._run_quick()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'paper'])
