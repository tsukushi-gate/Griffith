"""事前学習モデル/蒸留teacherの統一抽象(ROADMAP-v5 Track D)。

C1/C2(transcribe/binarize)の変換元モデル読込と、C4(distill)の教師ロジット
供給を同じ`Teacher`入口に統一する。変換パイプラインは個々のモデル形式
(torch nn.Module / onnxruntime / チェックポイントファイル)を意識せず、
`predict_logits(batch) -> (B, C)`とメタ情報(num_classes/input_size/
fingerprint)だけを見ればよい。

fingerprintはロジットキャッシュ(convert/cache.py予定)のキーに使う
安定識別子。同一モデル(同一重み)なら常に同じ値、重みが変われば必ず
値が変わることを要求する(内容ベースのキャッシュ無効化。
`griffith_logic/cuda/__init__.py`のソースハッシュと同じ考え方)。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import warnings
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    'TEACHER_TRANSFORMS',
    'OnnxTeacher',
    'Teacher',
    'TorchTeacher',
    'load_checkpoint_safely',
    'resolve_teacher',
]

# 非テンソルのオプトイン読込を明示するための環境変数。
UNSAFE_CKPT_ENV = 'GRIFFITH_ALLOW_UNSAFE_CKPT'


def _register_safe_globals() -> None:
    """既知のteacherモデルクラスだけを weights_only=True の許可リストへ登録する。

    `ckpt:` チェックポイントの metadata['model_class'] はクラス/callableであり
    テンソルではないため、weights_only=True では既定で拒否される。本プロジェクトが
    自前で定義するteacherクラスに限り明示的に許可する(任意コードのunpickleは
    引き続き禁止)。
    """
    add = getattr(torch.serialization, 'add_safe_globals', None)
    if add is None:  # 古いtorch(<2.1)ではAPIが無い
        return
    from . import teacher_models

    allowed = [
        teacher_models.SmallTeacherCNN,
        teacher_models.TeacherMLP,
        teacher_models.TeacherMLPSte,
        teacher_models.HeavisideSTE,
    ]
    try:
        add(allowed)
    except Exception:  # noqa: BLE001 — 重複登録等は無視(冪等にしたい)
        pass


def load_checkpoint_safely(
    path: str,
    *,
    map_location: str = 'cpu',
    allow_unsafe: bool = False,
    what: str = 'チェックポイント',
):
    """`torch.load` を既定 weights_only=True で行う。

    weights_only=True で読めない(=pickleで任意オブジェクトの復元が必要な)
    チェックポイントは、既定では ValueError で拒否する。信頼できるファイルに
    限り `allow_unsafe=True` もしくは環境変数 ``GRIFFITH_ALLOW_UNSAFE_CKPT=1``
    で明示的にオプトインでき、その場合は警告を出したうえで
    weights_only=False にフォールバックする。

    Args:
        path: チェックポイントのパス
        map_location: torch.load の map_location
        allow_unsafe: 呼び出し側(CLIフラグ等)からの明示オプトイン
        what: エラーメッセージに出す対象の呼び名

    Raises:
        ValueError: weights_only=True で読めず、オプトインも無い場合
    """
    _register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, AttributeError, RuntimeError) as e:
        opted_in = allow_unsafe or os.environ.get(UNSAFE_CKPT_ENV) == '1'
        if not opted_in:
            raise ValueError(
                f'{what}を安全な weights_only=True で読み込めませんでした: {path}\n'
                f'  理由: {e}\n'
                '  テンソル以外(モデルクラス等)の復元が必要なチェックポイントは'
                '任意コード実行のリスクがあります。信頼できるファイルの場合のみ'
                f' 環境変数 {UNSAFE_CKPT_ENV}=1 (またはCLIの'
                ' --allow-unsafe-checkpoint)で明示的に許可してください。'
                '  自作クラスは torch.serialization.add_safe_globals([...]) で'
                '個別に許可することもできます。'
            ) from e
        warnings.warn(
            f'{UNSAFE_CKPT_ENV}=1 のため weights_only=False で{what}を読み込みます'
            f'(任意コード実行のリスクがあります): {path}',
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


@runtime_checkable
class Teacher(Protocol):
    """変換元/蒸留teacherの統一プロトコル。

    後続タスク(C1/C2/C4)はこの4属性・メソッドのみに依存するため、
    属性名・シグネチャは固定(TorchTeacher/OnnxTeacher/カスタム実装のいずれもこれに従う)。
    """

    num_classes: int
    input_size: tuple[int, ...]  # 例 (1, 28, 28)。チャネル+空間次元(バッチ次元は含まない)
    fingerprint: str  # キャッシュキー用の安定識別子(重みが変われば変わる)

    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, *input_size)の入力から(B, num_classes)のロジットを返す。"""
        ...


def _auto_fingerprint(model: nn.Module) -> str:
    """クラス名+パラメータ形状+全パラメータ内容のsha256短縮からfingerprintを自動生成する。

    どの層の重みが変わっても必ず値が変わる(部分fine-tune済みモデルの
    キャッシュ衝突防止)。全パラメータをストリーミングでハッシュするため
    追加メモリはかからず、軽量モデル(本プロジェクトの対象)では十分速い。
    パラメータが1つもないモデル(定数関数など)ではクラス名+形状のみになるが、
    そのようなモデルをteacherにする実用ケースは想定しない。
    """
    h = hashlib.sha256()
    h.update(type(model).__name__.encode('utf-8'))
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode('utf-8'))
        h.update(str(tuple(p.shape)).encode('utf-8'))
        if p.numel():
            h.update(p.detach().to('cpu').contiguous().numpy().tobytes())
    return f'{type(model).__name__}-{h.hexdigest()[:16]}'


class TorchTeacher:
    """任意のnn.Moduleをteacherとしてラップする。

    eval()モードに固定し、`predict_logits`はno_grad配下+指定transform
    (正規化等)を適用してから推論する。deviceはコンストラクタで指定した
    ものへモデル・入力の両方を移す。

    Args:
        model: ラップ対象のnn.Module(推論専用。学習は行わない)
        num_classes: 出力クラス数
        input_size: 入力形状(バッチ次元を除く。例 (1, 28, 28))
        transform: 入力に適用する前処理(例: 正規化)。Noneなら恒等
        device: 推論デバイス。Noneならmodelの現在のdeviceを踏襲
        fingerprint: キャッシュキー。省略時はクラス名+パラメータ形状+
            先頭パラメータのsha256短縮から自動生成する
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        num_classes: int,
        input_size: tuple[int, ...],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
        fingerprint: str | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.input_size = input_size
        self.transform = transform if transform is not None else _identity_transform
        self.fingerprint = fingerprint if fingerprint is not None else _auto_fingerprint(model)

        self.device = torch.device(device) if device is not None else next(
            model.parameters(), torch.zeros(0)
        ).device
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, *input_size)からtransform適用後(B, num_classes)のロジットを返す。"""
        x = self.transform(batch.to(self.device))
        return self.model(x)


class OnnxTeacher:
    """onnxruntime InferenceSessionのラップ(resnet18_cifar10形式等を想定)。

    onnxruntime未導入環境では、インスタンス化時に導入方法を含む
    ImportErrorを送出する(遅延送出: モジュールimport時点では発生しない)。
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        num_classes: int,
        input_size: tuple[int, ...],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fingerprint: str | None = None,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                'OnnxTeacherにはonnxruntimeが必要です。'
                "`uv sync --extra teachers` で導入してください。"
            ) from e

        self.onnx_path = onnx_path
        self.num_classes = num_classes
        self.input_size = input_size
        self.transform = transform if transform is not None else _identity_transform
        self.fingerprint = fingerprint if fingerprint is not None else f'onnx-{onnx_path}'

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ['CPUExecutionProvider']
        )
        self._input_name = self.session.get_inputs()[0].name

    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, *input_size)からtransform適用後(B, num_classes)のロジットを返す。"""
        x = self.transform(batch)
        inputs = {self._input_name: x.detach().cpu().numpy()}
        (out,) = self.session.run(None, inputs)
        return torch.as_tensor(out)


def _identity_transform(x: torch.Tensor) -> torch.Tensor:
    """変換なし(恒等)。"""
    return x


def _binarize05_transform(x: torch.Tensor) -> torch.Tensor:
    """閾値0.5で二値化する(ROADMAP-v5)。

    `load_raw` は未二値化の生[0,1]画像を返すが、二値化MNIST(threshold=0.5)で
    学習したteacher(例 experiments/convert_transcribe_mnist.py の teacher-float/
    teacher-ste)にそのまま食わせると入力分布が学習時と異なりロジットが崩れる。
    `datasets/vision.BinarizeTransform` と同じ閾値・同じ意味論(x>0.5→1)を
    teacher入力側に適用し、student側の二値化MNISTローダと整合させる。
    """
    return (x > 0.5).to(x.dtype)


class _Normalize:
    """チャネルごとの平均/標準偏差正規化(torchvision.transforms.Normalize相当の最小実装)。

    ネットワークアクセス禁止のテスト環境でもtorchvisionに依存せず使えるように、
    このモジュール内で完結させる。
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        std = self.std.to(dtype=x.dtype, device=x.device)
        return (x - mean) / std


TEACHER_TRANSFORMS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'none': _identity_transform,
    'mnist': _Normalize((0.1307,), (0.3081,)),
    'cifar10': _Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
    # edadaltocg/resnet18_cifar10 系(HF)の学習時std。'cifar10'のstd(0.247系)だと
    # -0.5pt劣化することをGPU機D実測で確認(0.9445 vs 0.9498)
    'cifar10-resnet18-hf': _Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
    'binarize05': _binarize05_transform,
}


def resolve_teacher(source: str, *, device: str | torch.device | None = None) -> Teacher:
    """文字列指定からTeacherを解決する統一入口。

    対応スキーム:
        'torchvision:<name>' — torchvision.models.get_model(name, weights='DEFAULT')。
            num_classes=1000, input_size=(3, 224, 224)既定(torchvisionはlazy import)
        'ckpt:<path>'        — torch.load(map_location='cpu')。
            {'model_state_dict', 'metadata': {'model_class', 'num_classes', 'input_size', ...}}
            形式のチェックポイント(experiments/checkpoint_utils.pyのsave_checkpoint出力)を想定。
            metadataにmodel_classが無ければ復元できないため分かりやすいValueErrorを送出する
        'onnx:<path>'        — OnnxTeacher(num_classes/input_sizeはmetadata埋め込みが
            無いためONNX側では自明でなく、現状は既定値(1000, (3, 224, 224))を用いる。
            詳細な指定が要る場合はOnnxTeacherを直接構築すること)
        'hf:<repo_id>' — HuggingFace Hub の CIFAR-10 resnet18。
            **edadaltocg/resnet18_cifar10 形式に限定実装**(汎用AutoModelはスコープ外)。
            huggingface-hub を遅延importして config.json / pytorch_model.bin を取得し、
            torchvision.models.resnet18 の stem を CIFAR 化(3x3 stride1 conv + maxpool除去、
            fc=10クラス)して重みをロードする。返すTorchTeacherは transform=None(素)。
            **KDに使う正規化(TEACHER_TRANSFORMS['cifar10'])は呼び出し側で teacher.transform
            を上書きすること**(runner._apply_distill が teacher_transform 指定で行う)。

    Args:
        source: 'scheme:value' 形式の文字列
        device: 推論デバイス(torch系teacherにのみ適用)

    Returns:
        Teacherプロトコルを満たすインスタンス

    Raises:
        ValueError: sourceの形式が不正、またはスキームが未知、または'hf:'で未対応repo
        ImportError: 'onnx:'でonnxruntime未導入、または'hf:'でhuggingface-hub未導入
    """
    if ':' not in source:
        raise ValueError(
            f"sourceは'scheme:value'形式が必要です(例 'torchvision:resnet18'): {source!r}"
        )
    scheme, _, value = source.partition(':')

    if scheme == 'torchvision':
        return _resolve_torchvision_teacher(value, device=device)
    if scheme == 'ckpt':
        return _resolve_ckpt_teacher(value, device=device)
    if scheme == 'onnx':
        return OnnxTeacher(value, num_classes=1000, input_size=(3, 224, 224))
    if scheme == 'hf':
        return _resolve_hf_teacher(value, device=device)
    raise ValueError(
        f"未知のteacherスキーム: {scheme!r}"
        "(対応: 'torchvision:' / 'ckpt:' / 'onnx:' / 'hf:')"
    )


def _resolve_torchvision_teacher(
    name: str, *, device: str | torch.device | None
) -> TorchTeacher:
    """'torchvision:<name>' を解決する(重みDLが伴うためlazy import)。"""
    import torchvision

    model = torchvision.models.get_model(name, weights='DEFAULT')
    return TorchTeacher(
        model,
        num_classes=1000,
        input_size=(3, 224, 224),
        device=device,
    )


def _resolve_ckpt_teacher(path: str, *, device: str | torch.device | None) -> TorchTeacher:
    """'ckpt:<path>' を解決する。

    experiments/checkpoint_utils.pyのsave_checkpoint/load_checkpointが扱う
    チェックポイント形式(dict、キー'model_state_dict'+'metadata')に合わせる。
    metadataには変換元モデルを再構築するための以下のキーを要求する:
        model_class: nn.Moduleサブクラス(callable)、またはそれを返す0引数callable
        num_classes:  出力クラス数
        input_size:   入力形状(バッチ次元を除くtuple)
        model_kwargs: (任意) model_classへ渡すkwargs
        fingerprint:  (任意) 明示指定。無ければ復元後のstate_dictから自動生成
        transform:    (任意) TEACHER_TRANSFORMSのキー文字列
    """
    import os

    if not os.path.exists(path):
        raise ValueError(f'チェックポイントファイルが存在しません: {path}')

    data = load_checkpoint_safely(path, what=f'teacherチェックポイント ckpt:{path}')
    if not isinstance(data, dict) or 'model_state_dict' not in data:
        raise ValueError(
            f"ckpt:{path} は非対応形式です。"
            "{'model_state_dict': ..., 'metadata': {...}} 形式"
            "(experiments/checkpoint_utils.save_checkpoint出力)を期待します。"
        )

    metadata = data.get('metadata') or {}
    model_class = metadata.get('model_class')
    if model_class is None:
        raise ValueError(
            f"ckpt:{path} のmetadataに'model_class'がありません。"
            "resolve_teacher('ckpt:...')にはmodel_class/num_classes/input_sizeを"
            "含むmetadataが必要です(experiments/checkpoint_utils.pyのsave_checkpoint参照)。"
        )
    if 'num_classes' not in metadata:
        raise ValueError(f"ckpt:{path} のmetadataに'num_classes'がありません。")
    if 'input_size' not in metadata:
        raise ValueError(f"ckpt:{path} のmetadataに'input_size'がありません。")

    model_kwargs = metadata.get('model_kwargs') or {}
    model = model_class(**model_kwargs)
    model.load_state_dict(data['model_state_dict'])

    transform_key = metadata.get('transform')
    transform = TEACHER_TRANSFORMS[transform_key] if transform_key else None

    return TorchTeacher(
        model,
        num_classes=metadata['num_classes'],
        input_size=tuple(metadata['input_size']),
        transform=transform,
        device=device,
        fingerprint=metadata.get('fingerprint'),
    )


# 'hf:' で限定サポートする repo_id。汎用AutoModelはスコープ外。
_HF_CIFAR10_RESNET18_REPOS = frozenset({'edadaltocg/resnet18_cifar10'})


def _build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """torchvision resnet18 を CIFAR-10 用 stem に改変した nn.Module を返す。

    edadaltocg/resnet18_cifar10(および chenyaofo系のCIFAR resnet18)は、
    ImageNet版resnet18の stem を次のように差し替えたもの:
        - conv1: 7x7 stride2 → **3x3 stride1 pad1**(32x32入力で空間解像度を保つ)
        - maxpool: 3x3 stride2 → **Identity(除去)**
        - fc: 1000クラス → num_classes
    それ以外(layer1〜4 の block構成・downsample・BN・avgpool)は torchvision の
    resnet18 と完全に同一で、state_dict のキー名も torchvision 命名そのまま。
    したがって state_dict キー変換は不要で、この改変モデルに直接 load_state_dict できる。
    """
    import torchvision

    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _resolve_hf_teacher(repo_id: str, *, device: str | torch.device | None) -> TorchTeacher:
    """'hf:<repo_id>' を解決する(CIFAR-10 resnet18 に限定)。

    huggingface-hub を遅延importして config.json / pytorch_model.bin を取得し、
    CIFAR化した torchvision resnet18 に重みをロードする。汎用のAutoModel解決は
    スコープ外で、対応repoは `_HF_CIFAR10_RESNET18_REPOS` のみ。

    返り値の TorchTeacher は transform=None(素)。正規化は呼び出し側で
    teacher.transform を上書きする規約(runner._apply_distill が
    TEACHER_TRANSFORMS[teacher_transform] を代入)。fingerprint は全パラメータの
    自動sha256(_auto_fingerprint)。
    """
    if repo_id not in _HF_CIFAR10_RESNET18_REPOS:
        raise ValueError(
            f'hf:スキームは現状 {sorted(_HF_CIFAR10_RESNET18_REPOS)} のみ対応です'
            f'(汎用AutoModelはスコープ外): {repo_id!r}'
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "hf:スキームには huggingface-hub が必要です。"
            "`uv sync --extra teachers` で導入してください。"
        ) from e

    # R11: 供給元の差し替えを防ぐには revision にコミットSHAを固定するのが望ましい
    # (例: hf_hub_download(repo_id, 'pytorch_model.bin', revision='<commit sha>'))。
    # 現状は既定リビジョンを取得している。
    bin_path = hf_hub_download(repo_id, 'pytorch_model.bin')
    state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)

    model = _build_cifar_resnet18(num_classes=10)
    model.load_state_dict(state_dict)

    return TorchTeacher(
        model,
        num_classes=10,
        input_size=(3, 32, 32),
        transform=None,  # 正規化は呼び出し側で teacher.transform を上書きする(要指定)
        device=device,
    )
