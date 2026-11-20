"""L3: DWN論文再現(Bacellar et al., ICML 2024)。

論文Table 1: DWN(n=6; lg) はMNISTフルセットで98.31%。
ここでは高速化のため6,000枚サブセット+縮小構成(LUT 1000→500)で
テストacc ≥ 92% を再現の下限として固定する。

構成は公式実装 examples/mnist.py に準拠:
    DistributiveThermometer(3) → LUTLayer(2352, 1000, n=6, learnable)
    → LUTLayer(1000, 500, n=6) → GroupSum(k=10, tau=1/0.3)、Adam lr=1e-2
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.dwn import DWNClassifier

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data-mnist')

N_TRAIN = 6000
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-2
LR_STEP = 10       # 10エポック目以降はlr×0.1(公式のStepLR方式)
TARGET_ACC = 0.92


def _load_mnist():
    import torchvision

    train_set = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True)
    test_set = torchvision.datasets.MNIST(DATA_ROOT, train=False)
    x_train = train_set.data[:N_TRAIN].reshape(N_TRAIN, -1).float() / 255.0
    y_train = train_set.targets[:N_TRAIN]
    x_test = test_set.data.reshape(len(test_set), -1).float() / 255.0
    y_test = test_set.targets
    return x_train, y_train, x_test, y_test


@torch.no_grad()
def _accuracy(model, x, y, batch_size=512):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        pred = model(x[i:i + batch_size]).argmax(-1)
        correct += (pred == y[i:i + batch_size]).sum().item()
    return correct / len(x)


@pytest.mark.paper
def test_mnist_6k_subset_accuracy():
    torch.manual_seed(42)
    x_train, y_train, x_test, y_test = _load_mnist()

    model = DWNClassifier(
        in_features=784, n_classes=10, layer_widths=(1000, 500),
        bits=3, encoder_mode='distributive', n_inputs=6,
        mapping='learnable', tau=1.0 / 0.3,
    ).fit_encoder(x_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=0.1)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    acc = _accuracy(model, x_test, y_test)
    assert acc >= TARGET_ACC, f'MNIST 6kサブセット: test acc={acc:.4f} < {TARGET_ACC}'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'paper'])
