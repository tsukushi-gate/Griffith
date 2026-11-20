"""学習済みチェックポイント(.pt) → モデルdoc(webbench形式)。

`griffith deploy bundle model.pt` の入口。チェックポイントからモデルを再構築し、
`export_ir` で離散化(各ゲートの演算を1つに確定)したIRに、readout・
labels・由来を添えて build_bundle が受け取れるdocにする。

前処理は「入力はすでにビット列」を宣言する(`griffith train` のモデルは
データセット側で二値化済みの入力を受けるため)。
"""

from datetime import datetime
from pathlib import Path

from ..export.estimate import estimate_resources
from ..export.extract import export_ir
from ..ir.serialize import ir_to_dict

CHECKPOINT_SUFFIXES = ('.pt', '.pth')


def is_checkpoint_path(path) -> bool:
    return Path(path).suffix.lower() in CHECKPOINT_SUFFIXES


def doc_from_checkpoint(checkpoint, *, config: str | None = None) -> dict:
    """チェックポイントをbuild_bundle用のモデルdocへ変換する。

    Args:
        checkpoint: `griffith train` / run_training が保存した .pt
        config: 学習設定YAML(省略時はチェックポイント内の設定を使う)
    """
    from ..training.runner import load_model_from_checkpoint

    model, cfg, shape, data = load_model_from_checkpoint(
        str(checkpoint), config=config)
    graph = export_ir(
        model, input_width=shape.in_width,
        name=Path(checkpoint).stem,
        metadata={'dataset': cfg.data.name, 'model_kind': cfg.model.kind},
    )

    if shape.kind == 'classification':
        labels = [str(i) for i in range(shape.out_width)]
        readout = {'kind': 'class_scores', 'k': shape.out_width, 'spatial': None}
    else:
        labels = [f'out{i}' for i in range(shape.out_width)]
        readout = {'kind': 'bits', 'k': None, 'spatial': None}

    about = {
        'source_ckpt': str(checkpoint),
        'dataset': cfg.data.name,
        'exported_at': datetime.now().astimezone().isoformat(),
    }
    metadata = data.get('metadata') or {}
    if metadata.get('run_name'):
        about['run_name'] = str(metadata['run_name'])
    if data.get('step') is not None:
        about['step'] = int(data['step'])

    return {
        'ir': ir_to_dict(graph),
        'preproc': {'kind': 'bits', 'input_kind': 'bits'},
        'labels': labels,
        'readout': readout,
        'resources': estimate_resources(graph),
        'eval': {},
        'about': about,
    }
