"""多環境デプロイキット(Track K)。

学習済みモデルを **1つの配布バンドル**にまとめ、Web / C(組み込み) /
CUDA / モバイルのどのターゲットでも同じファイルから動かすためのレイヤ。

責務分離(このパッケージの設計原則):
    - **最適化はエクスポータ側(=ここ)で終わらせる**。conv2d木の命令列化
      (export/optimize.py)や前処理の整数化をバンドル生成時に実行し、
      結果を配布物に焼き込む。ターゲット側ランタイムは「焼かれた命令列を
      実行するだけ」で、縮約ロジックを一切持たない
      (ONNX Runtimeのモバイル向けORTフォーマットと同じ考え方)
    - 前後処理も**データ**(記述子)としてバンドルに同梱し、アプリ側の
      コードを最小化する(MediaPipeの.taskバンドルと同じ考え方)

真実源は依然としてIR。バンドルの programs / preproc_desc は IR から
決定的に再構築できる冗長データであり、書き出し前に必ずセルフチェック
(verify_conv_program 等)を通す。
"""

from .bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    build_bundle,
    load_bundle,
    save_bundle,
    validate_bundle,
)
from .preproc_desc import (
    build_preproc_descriptors,
    encode_bits,
    run_preproc_descriptors,
)
from .programs import build_programs, program_from_dict, program_to_dict

__all__ = [
    'BUNDLE_FORMAT',
    'BUNDLE_VERSION',
    'build_bundle',
    'load_bundle',
    'save_bundle',
    'validate_bundle',
    'build_programs',
    'program_to_dict',
    'program_from_dict',
    'build_preproc_descriptors',
    'run_preproc_descriptors',
    'encode_bits',
]
