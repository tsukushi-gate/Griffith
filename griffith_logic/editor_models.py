"""Pythonモデルをエディタへ公開するための軽量登録API。

このモジュールはエディタ専用のメタデータ層で、学習や推論の意味論は
通常のPyTorch ``nn.Module`` を正とする。IRへ完全にloweringできない
Attention/LayerNorm等を含むモデルでも、表示・学習・コード生成の入口を
共有できるようにする。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BuildFn = Callable[[], Any]


@dataclass
class EditorModelSpec:
    """エディタへ公開するPythonモデルの仕様。

    ``build`` は ``@editor_model`` が付けた関数/クラスから自動設定される。
    ``input_spec`` / ``output_spec`` は ``width`` または ``shape`` を含む
    dictを想定する。``shape`` だけの場合は要素積をwidthとして扱う。
    """

    id: str
    name: str
    task: str = 'classification'
    input_spec: Mapping[str, Any] = field(default_factory=dict)
    output_spec: Mapping[str, Any] = field(default_factory=dict)
    datasets: Sequence[str] = field(default_factory=tuple)
    training_presets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    stages: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    hierarchy: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    lowering: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    paper_ref: Mapping[str, Any] | None = None
    dataset_presets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    description: str | None = None
    build: BuildFn | None = field(default=None, repr=False, compare=False)

    def build_model(self):
        """登録済みbuilderからPyTorchモデルを生成する。"""
        if self.build is None:
            raise ValueError(f'{self.id}: build関数が設定されていません')
        return self.build()

    def to_metadata(self) -> dict[str, Any]:
        """IR metadataへ安全に埋め込めるJSON互換dictへ変換する。"""
        return {
            'id': self.id,
            'name': self.name,
            'task': self.task,
            'input_spec': dict(self.input_spec),
            'output_spec': dict(self.output_spec),
            'datasets': list(self.datasets),
            'training_presets': {
                str(k): dict(v) for k, v in self.training_presets.items()
            },
            'capabilities': dict(self.capabilities),
            'stages': [dict(s) for s in self.stages],
            'hierarchy': [dict(h) for h in self.hierarchy],
            'lowering': {
                str(k): dict(v) for k, v in self.lowering.items()
            },
            'paper_ref': dict(self.paper_ref) if self.paper_ref is not None else None,
            'dataset_presets': {
                str(k): dict(v) for k, v in self.dataset_presets.items()
            },
            'description': self.description,
        }


def editor_model(
    *,
    id: str,
    name: str,
    task: str = 'classification',
    input_spec: Mapping[str, Any] | None = None,
    output_spec: Mapping[str, Any] | None = None,
    datasets: Sequence[str] = (),
    training_presets: Mapping[str, Mapping[str, Any]] | None = None,
    capabilities: Mapping[str, bool] | None = None,
    stages: Sequence[Mapping[str, Any]] = (),
    hierarchy: Sequence[Mapping[str, Any]] = (),
    lowering: Mapping[str, Mapping[str, Any]] | None = None,
    paper_ref: Mapping[str, Any] | None = None,
    dataset_presets: Mapping[str, Mapping[str, Any]] | None = None,
    description: str | None = None,
):
    """関数またはクラスをエディタ用モデルbuilderとして登録する。

    例:
        @editor_model(id='toy', name='Toy', input_spec={'width': 4}, ...)
        def build_model():
            return nn.Sequential(...)
    """

    def deco(builder):
        spec = EditorModelSpec(
            id=id,
            name=name,
            task=task,
            input_spec=input_spec or {},
            output_spec=output_spec or {},
            datasets=tuple(datasets),
            training_presets=training_presets or {},
            capabilities=capabilities or {},
            stages=tuple(stages),
            hierarchy=tuple(hierarchy),
            lowering=lowering or {},
            paper_ref=paper_ref,
            dataset_presets=dataset_presets or {},
            description=description,
            build=builder,
        )
        builder.__magicdelta_editor_model__ = spec
        return builder

    return deco


def get_editor_model_spec(obj: Any) -> EditorModelSpec | None:
    """オブジェクトからEditorModelSpecを取り出す。"""
    if isinstance(obj, EditorModelSpec):
        return obj
    spec = getattr(obj, '__magicdelta_editor_model__', None)
    return spec if isinstance(spec, EditorModelSpec) else None


def infer_width(spec: Mapping[str, Any], *, field_name: str) -> int:
    """spec dictの ``width`` または ``shape`` から平坦幅を推定する。"""
    width = spec.get('width')
    if width is not None:
        return int(width)
    shape = spec.get('shape')
    if not isinstance(shape, Sequence) or isinstance(shape, str | bytes):
        raise ValueError(f'{field_name}にはwidthまたはshapeが必要です')
    out = 1
    for v in shape:
        out *= int(v)
    return out


__all__ = ['EditorModelSpec', 'editor_model', 'get_editor_model_spec', 'infer_width']
