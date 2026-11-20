"""エディタ学習用のデータセット解決。

dataset文字列を (タスク種別, 入出力幅, DataLoader factory) へ解決する。
  - 'mnist'            : torchvision MNIST(784入力/10クラス、classification)
  - 'logic:<key>'      : griffith_logic.datasets.LOGIC_DATASETS(binary_vector)
                         例: logic:xor / logic:parity3 / logic:half_adder
  - 'truth_table'      : リクエストで渡された真理値表ターゲット(binary_vector)。
                         params = {'table': [[出力bit,...] パターンk順], 'n_inputs', 'n_outputs'}
                         エディタの検証テーブルで編集した期待値をそのまま学習できる
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MNIST_ROOT = str(PROJECT_ROOT / 'data-mnist')
MNIST_IN_WIDTH = 784
MNIST_CLASSES = 10
LLVIT_IMAGE_SHAPE = [3, 224, 224]
LLVIT_IN_WIDTH = 3 * 224 * 224
LLVIT_DATASETS = {
    'cifar10': ('CIFAR-10', 10),
    'cifar100': ('CIFAR-100', 100),
    'flowers102': ('Flowers-102', 102),
    'tinyimagenet': ('Tiny-ImageNet', 200),
}

# 小さい論理データセットは行数がこの目安になるようrepeatで水増しする
# (1エポックが数stepで終わってタイムライン/チャートが寂しくなるのを防ぐ)
LOGIC_TARGET_ROWS = 1024


@dataclass
class ResolvedDataset:
    """解決済みデータセット(TrainingSessionへ渡す)。"""

    name: str
    kind: str            # 'classification' | 'binary_vector'
    in_width: int        # モデル入力幅の要件
    out_width: int       # モデル出力幅の要件(classificationはクラス数)
    loader_factory: Callable[[], torch.utils.data.DataLoader]


# データセットセレクタの表示ラベル
DATASET_LABELS = {
    'mnist': 'MNIST',
    'llvit_synthetic_debug': 'LL-ViT Synthetic Debug',
    'llvit:cifar10': 'LL-ViT CIFAR-10',
    'llvit:cifar100': 'LL-ViT CIFAR-100',
    'llvit:flowers102': 'LL-ViT Flowers-102',
    'llvit:tinyimagenet': 'LL-ViT Tiny-ImageNet',
    'xor': 'XOR',
    'parity3': 'パリティ3',
    'parity4': 'パリティ4',
    'majority3': '多数決3',
    'half_adder': '半加算器',
    'full_adder': '全加算器',
    'mux2': 'MUX 2:1',
    'kv_attention2': 'KV Lookup Toy',
    'seq101': '系列101検出',
    # n bit 加算器 / 乗算器(LSB-first。adder{n} は 2n → n+1、mul{n} は 2n → 2n)
    'adder2': '2bit 加算器',
    'adder3': '3bit 加算器',
    'adder4': '4bit 加算器',
    'mul2': '2bit 乗算器',
    'mul3': '3bit 乗算器',
    'mul4': '4bit 乗算器',
}


def list_datasets() -> list[dict]:
    """学習UIのデータセットカタログ。LOGIC_DATASETSから自動生成する。

    truth_tableはリクエストパラメータ必須のためカタログには載せない
    (エディタの検証テーブル側から直接dataset_paramsで送る)。
    """
    from griffith_logic.datasets import LOGIC_DATASETS

    items = [{
        'name': 'mnist',
        'label': DATASET_LABELS['mnist'],
        'in_width': MNIST_IN_WIDTH,
        'out_width': MNIST_CLASSES,
        'kind': 'classification',
        'input_spec': {'shape': [1, 28, 28], 'width': MNIST_IN_WIDTH},
        'output_spec': {'width': MNIST_CLASSES, 'semantics': 'class_scores'},
    }, {
        'name': 'llvit_synthetic_debug',
        'label': DATASET_LABELS['llvit_synthetic_debug'],
        'in_width': 32,
        'out_width': 4,
        'kind': 'classification',
        'input_spec': {'shape': [4, 8], 'width': 32},
        'output_spec': {'width': 4, 'semantics': 'class_scores'},
    }]
    for key, (_, classes) in LLVIT_DATASETS.items():
        name = f'llvit:{key}'
        items.append({
            'name': name,
            'label': DATASET_LABELS[name],
            'in_width': LLVIT_IN_WIDTH,
            'out_width': classes,
            'kind': 'classification',
            'input_spec': {
                'shape': LLVIT_IMAGE_SHAPE,
                'width': LLVIT_IN_WIDTH,
                'encoding': 'rgb_224_deit',
            },
            'output_spec': {'width': classes, 'semantics': 'class_scores'},
        })
    for key in sorted(LOGIC_DATASETS):
        spec = LOGIC_DATASETS[key]()
        items.append({
            'name': f'logic:{key}',
            'label': DATASET_LABELS.get(key, key),
            'in_width': spec.n_inputs,
            'out_width': spec.n_outputs,
            'kind': 'binary_vector',
            'input_spec': {'width': spec.n_inputs, 'encoding': 'binary'},
            'output_spec': {'width': spec.n_outputs, 'semantics': 'binary_vector'},
        })
    return items


def _mnist_loader_factory(batch_size: int) -> Callable[[], torch.utils.data.DataLoader]:
    def factory():
        import torchvision

        transforms = torchvision.transforms.ToTensor()
        train_set = torchvision.datasets.MNIST(
            MNIST_ROOT, train=True, download=True, transform=transforms
        )
        return torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )

    return factory


def _vision_transform():
    import torchvision

    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])


def _fake_llvit_loader_factory(
    batch_size: int,
    *,
    classes: int,
    rows: int = 64,
    seed: int = 31,
) -> Callable[[], torch.utils.data.DataLoader]:
    def factory():
        generator = torch.Generator().manual_seed(seed)
        x = torch.rand(rows, *LLVIT_IMAGE_SHAPE, generator=generator)
        y = torch.randint(0, classes, (rows,), generator=generator)
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    return factory


def _resolve_workspace_dataset_root(path: str | None) -> Path:
    base = Path(path) if path else PROJECT_ROOT / 'data-tiny-imagenet-200'
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    resolved = base.resolve()
    root = PROJECT_ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f'workspace外のdataset pathは使用できません: {resolved}')
    return resolved


def _llvit_loader_factory(
    key: str,
    batch_size: int,
    params: dict | None,
) -> Callable[[], torch.utils.data.DataLoader]:
    if key not in LLVIT_DATASETS:
        raise ValueError(f'未対応のLL-ViTデータセット: {key}(対応: {sorted(LLVIT_DATASETS)})')
    _, classes = LLVIT_DATASETS[key]
    p = params or {}
    if bool(p.get('fake')):
        return _fake_llvit_loader_factory(
            batch_size,
            classes=classes,
            rows=int(p.get('fake_size', 64)),
            seed=int(p.get('seed', 31)),
        )

    def factory():
        if key == 'tinyimagenet':
            tiny_root = _resolve_workspace_dataset_root(p.get('root'))
            train_dir = tiny_root / 'train'
            if not train_dir.exists():
                raise ValueError(
                    'Tiny-ImageNetが見つかりません。dataset_params.rootに'
                    'workspace内のtiny-imagenet-200ディレクトリを指定してください'
                    f'(期待: {train_dir})'
                )
        import torchvision

        transform = _vision_transform()
        root = str(PROJECT_ROOT / 'data-vision')
        if key == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root, train=True, download=True, transform=transform,
            )
        elif key == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(
                root, train=True, download=True, transform=transform,
            )
        elif key == 'flowers102':
            dataset = torchvision.datasets.Flowers102(
                root, split='train', download=True, transform=transform,
            )
        elif key == 'tinyimagenet':
            dataset = torchvision.datasets.ImageFolder(str(train_dir), transform=transform)
        else:  # pragma: no cover - key検証済み
            raise ValueError(key)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    return factory


def resolve_dataset(
    name: str, *, batch_size: int, params: dict | None = None,
) -> ResolvedDataset:
    """dataset文字列を解決する。未対応・不正はValueError(RESTは400に変換)。"""
    if name == 'mnist':
        return ResolvedDataset(
            name='mnist', kind='classification',
            in_width=MNIST_IN_WIDTH, out_width=MNIST_CLASSES,
            loader_factory=_mnist_loader_factory(batch_size),
        )

    if name in ('llvit_synthetic_debug', 'llvit_tiny'):
        def factory():
            generator = torch.Generator().manual_seed(17)
            x = torch.randn(LOGIC_TARGET_ROWS, 4, 8, generator=generator)
            mean_score = x.mean(dim=(1, 2))
            first_token_score = x[:, 0, :].mean(dim=1)
            y = ((mean_score > 0).long() * 2 + (first_token_score > 0).long()).clamp(max=3)
            dataset = torch.utils.data.TensorDataset(x, y)
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

        return ResolvedDataset(
            name='llvit_synthetic_debug', kind='classification',
            in_width=32, out_width=4,
            loader_factory=factory,
        )

    if name.startswith('llvit:'):
        key = name.split(':', 1)[1]
        if key not in LLVIT_DATASETS:
            raise ValueError(f'未対応のLL-ViTデータセット: {key}(対応: {sorted(LLVIT_DATASETS)})')
        return ResolvedDataset(
            name=name, kind='classification',
            in_width=LLVIT_IN_WIDTH, out_width=LLVIT_DATASETS[key][1],
            loader_factory=_llvit_loader_factory(key, batch_size, params),
        )

    if name.startswith('logic:'):
        from griffith_logic.datasets import LOGIC_DATASETS

        key = name.split(':', 1)[1]
        if key not in LOGIC_DATASETS:
            raise ValueError(
                f'未対応の論理データセット: {key}(対応: {sorted(LOGIC_DATASETS)})'
            )
        # 行数がLOGIC_TARGET_ROWS程度になるようrepeatを自動決定
        probe = LOGIC_DATASETS[key]()
        n_rows = len(probe.dataset)
        repeat = max(1, LOGIC_TARGET_ROWS // n_rows)
        spec = LOGIC_DATASETS[key](repeat=repeat)

        def factory(spec=spec):
            return spec.loader(min(batch_size, len(spec.dataset)), shuffle=True)

        return ResolvedDataset(
            name=name, kind='binary_vector',
            in_width=spec.n_inputs, out_width=spec.n_outputs,
            loader_factory=factory,
        )

    if name == 'truth_table':
        from griffith_logic.datasets import truth_table_dataset

        p = params or {}
        try:
            n_inputs = int(p['n_inputs'])
            n_outputs = int(p['n_outputs'])
            table = p['table']
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f'truth_tableデータセットにはparams(table/n_inputs/n_outputs)が必要: {e}'
            )
        repeat = max(1, int(p.get('repeat', LOGIC_TARGET_ROWS // (2 ** n_inputs) or 1)))
        dataset = truth_table_dataset(table, n_inputs, n_outputs, repeat=repeat)

        def factory(dataset=dataset):
            return torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
            )

        return ResolvedDataset(
            name='truth_table', kind='binary_vector',
            in_width=n_inputs, out_width=n_outputs,
            loader_factory=factory,
        )

    raise ValueError(
        f'未対応のデータセット: {name}(対応: mnist / llvit:<key> / logic:<key> / truth_table)'
    )
