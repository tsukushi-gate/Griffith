"""データセット生成のサブパッケージ(A4系列)。

logic系はtorch標準のみに依存する。vision系はtorchvisionを
関数内で遅延importする(コア依存には追加しない)。
"""

from .har import (
    UCI_HAR_CHANNELS,
    UCI_HAR_LABELS,
    UCI_HAR_URL,
    fit_thermometer_thresholds,
    load_uci_har,
    thermometer_encode,
)
from .logic import (
    LOGIC_DATASETS,
    LogicDatasetSpec,
    full_adder,
    half_adder,
    kv_attention2,
    majority3,
    mux,
    nbit_adder,
    nbit_multiplier,
    parity,
    random_logic,
    seq101,
    truth_table_dataset,
    xor2,
)
from .segmentation import (
    OxfordPetBinarySeg,
    load_oxford_pet_seg,
    trimap_to_binary_mask,
)
from .vision import (
    VISION_DATASETS,
    VisionDatasetSpec,
    load_cifar10,
    load_cifar100,
    load_fashion_mnist,
    load_mnist,
    load_mnist20x20,
    resolve_data_root,
)

__all__ = [
    'LOGIC_DATASETS',
    'LogicDatasetSpec',
    'OxfordPetBinarySeg',
    'UCI_HAR_CHANNELS',
    'UCI_HAR_LABELS',
    'UCI_HAR_URL',
    'VISION_DATASETS',
    'VisionDatasetSpec',
    'fit_thermometer_thresholds',
    'full_adder',
    'half_adder',
    'kv_attention2',
    'load_cifar10',
    'load_cifar100',
    'load_fashion_mnist',
    'load_mnist',
    'load_mnist20x20',
    'load_oxford_pet_seg',
    'load_uci_har',
    'majority3',
    'mux',
    'nbit_adder',
    'nbit_multiplier',
    'parity',
    'random_logic',
    'resolve_data_root',
    'seq101',
    'thermometer_encode',
    'trimap_to_binary_mask',
    'truth_table_dataset',
    'xor2',
]
