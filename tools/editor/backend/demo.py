"""デモモデル生成とIR→JSON変換。

チェックポイント読み込みのセキュリティ(/ 監査B4):
`torch.load(weights_only=False)` はファイル内のpickleを実行するため、
ワークスペース内の任意 `.pt` を読ませるAPI(`POST /api/models/import`)経路では
任意コード実行の踏み台になる。そこで
  1. 既定は `checkpoint_utils.torch_load`(`weights_only=True`)の安全モード、
  2. state_dict形式(`{'state_dict': ..., 'config': ...}`)を**優先**して読み、
     構造は `config` からコードで組み立てる(pickleを一切使わない)、
  3. モデルオブジェクト全体をpickleした旧形式は後方互換のオプトイン
     (`GRIFFITH_ALLOW_UNSAFE_CKPT=1`)時のみ読む、
の3段構えにした。
"""

from pathlib import Path

import torch

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import ir_to_dict
from griffith_logic.training.checkpoint_utils import UNSAFE_CKPT_ENV, torch_load

__all__ = ['MNIST_IN_DIM', 'MNIST_CLASSES', 'create_demo_model',
           'demo_model_config', 'build_model_from_config', 'save_demo_checkpoint',
           'export_model_ir', 'ir_to_dict', 'load_model_from_checkpoint']

# モデル全体をpickleした旧形式ckptを既定で拒否するときの案内。
UNSAFE_CKPT_MESSAGE = (
    'このチェックポイントはモデル全体を pickle しています。'
    f'信頼できる場合のみ `{UNSAFE_CKPT_ENV}=1` で読み込めます'
    "(推奨は state_dict 形式 {'state_dict': ..., 'config': ...} への作り直しです)。"
)

# build_model_from_config が解釈するアーキテクチャ識別子。
SEQUENTIAL_LUT_ARCH = 'sequential_lut'

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MNIST_IN_DIM = 784
MNIST_CLASSES = 10


def create_demo_model(k: int = 2000, layers: int = 3) -> torch.nn.Module:
    """MNIST用デモモデルを生成する。

    構成: Flatten → LutLayer(784, k, N=2) × layers → GroupSum(10)
    """
    assert layers >= 1, 'layersは1以上'
    assert k % MNIST_CLASSES == 0, f'kはクラス数({MNIST_CLASSES})の倍数にすること: {k}'
    modules: list[torch.nn.Module] = [torch.nn.Flatten()]
    in_dim = MNIST_IN_DIM
    for _ in range(layers):
        modules.append(LutLayer(in_dim, k, n_inputs=2, connections='unique'))
        in_dim = k
    modules.append(GroupSum(k=MNIST_CLASSES, tau=1.0))
    model = torch.nn.Sequential(*modules)
    model.eval()
    return model


# ir_to_dict は griffith_logic.ir.serialize へ移設(E2)。ここでは再エクスポートのみ。


def export_model_ir(model: torch.nn.Module, input_width: int, name: str) -> dict:
    """モデルをIRへ変換しJSON dictで返す。

    ネイティブ抽出(conv2d / or_pool2d / permute をノードのまま出す)。エディタは
    build のたびに再抽出するので、描いた構造(conv2d ノード等)が lut_layer 列へ
    展開されず保たれる。HDL 生成時は lowering が展開する。
    """
    graph = export_ir(model, input_width=input_width, name=name, native_conv=True)
    return ir_to_dict(graph)


def demo_model_config(k: int = 2000, layers: int = 3) -> dict:
    """`create_demo_model` と同じ構成を表す config dict を返す。

    state_dict形式ckptの `config` に入れる。pickleを介さずに構造を再現するための
    最小の記述(JSON化可能なプリミティブのみ)。
    """
    return {
        'arch': SEQUENTIAL_LUT_ARCH,
        'in_dim': MNIST_IN_DIM,
        'hidden': [k] * layers,
        'n_inputs': 2,
        'connections': 'unique',
        'classes': MNIST_CLASSES,
        'tau': 1.0,
        'flatten': True,
    }


def build_model_from_config(config: dict) -> torch.nn.Module:
    """config dict から `torch.nn.Sequential` を組み立てる(pickle不使用)。

    `arch='sequential_lut'` のみ対応: [Flatten?] → LutLayer × len(hidden) → GroupSum。
    """
    if not isinstance(config, dict):
        raise ValueError(f'configがdictではありません: {type(config)}')
    arch = config.get('arch', SEQUENTIAL_LUT_ARCH)
    if arch != SEQUENTIAL_LUT_ARCH:
        raise ValueError(
            f'未対応のarchです: {arch!r}(対応: {SEQUENTIAL_LUT_ARCH!r})'
        )
    hidden = config.get('hidden') or []
    if not hidden:
        raise ValueError("configの'hidden'(各LutLayerの出力次元リスト)が空です")

    in_dim = int(config.get('in_dim', MNIST_IN_DIM))
    n_inputs = int(config.get('n_inputs', 2))
    connections = str(config.get('connections', 'unique'))
    classes = int(config.get('classes', MNIST_CLASSES))
    tau = float(config.get('tau', 1.0))

    modules: list[torch.nn.Module] = []
    if config.get('flatten', True):
        modules.append(torch.nn.Flatten())
    for out_dim in hidden:
        modules.append(
            LutLayer(in_dim, int(out_dim), n_inputs=n_inputs, connections=connections)
        )
        in_dim = int(out_dim)
    modules.append(GroupSum(k=classes, tau=tau))
    model = torch.nn.Sequential(*modules)
    model.eval()
    return model


def save_demo_checkpoint(path: str | Path, k: int = 2000, layers: int = 3) -> Path:
    """デモモデルを **state_dict形式** で保存する(B4: pickle非依存のckpt正本)。

    形式: ``{'state_dict': model.state_dict(), 'config': demo_model_config(...)}``
    これは `weights_only=True` のまま読めるため、読み込み側でpickle実行が起きない。
    """
    model = create_demo_model(k=k, layers=layers)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'state_dict': model.state_dict(), 'config': demo_model_config(k=k, layers=layers)},
        out,
    )
    return out


def _model_from_state_dict_ckpt(ckpt: dict, device: str) -> torch.nn.Module:
    """state_dict形式ckpt(`state_dict`/`model_state_dict` + `config`)を復元する。"""
    state_dict = ckpt.get('state_dict')
    if state_dict is None:
        state_dict = ckpt.get('model_state_dict')
    config = ckpt.get('config')
    if config is None:
        raise ValueError(
            "state_dict形式のチェックポイントには構造情報 'config' が必要です"
            "(例: {'state_dict': ..., 'config': {'arch': 'sequential_lut', ...}})。"
        )
    model = build_model_from_config(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = 'cpu', *, allow_unsafe: bool = False,
) -> torch.nn.Module:
    """チェックポイントからモデルをロードする(安全既定、B4)。

    優先順:
      1. state_dict形式(`state_dict`/`model_state_dict` + `config`)
         — `weights_only=True` で読めるのでpickle実行なし。
      2. モデル全体をpickleした旧形式 — `allow_unsafe=True` か
         ``GRIFFITH_ALLOW_UNSAFE_CKPT=1`` のオプトイン時のみ。既定は拒否。

    Raises:
        ValueError: 旧形式を既定(安全モード)で読もうとした場合、または構造を
            復元できない場合。
    """
    try:
        ckpt = torch_load(checkpoint_path, map_location=device, allow_unsafe=allow_unsafe)
    except RuntimeError as e:
        # checkpoint_utils.torch_load が weights_only=True で失敗したとき。
        # = モデル全体pickle等、安全モードで復元できない内容。
        raise ValueError(UNSAFE_CKPT_MESSAGE) from e

    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model_state_dict' in ckpt):
        return _model_from_state_dict_ckpt(ckpt, device)

    # ここに来るのはオプトイン時のみ(安全モードならnn.Moduleは復元されない)
    model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if not isinstance(model, torch.nn.Module):
        raise ValueError(
            f'モデルを取り出せない: {type(model)}。{UNSAFE_CKPT_MESSAGE}'
        )
    model.eval()
    return model
