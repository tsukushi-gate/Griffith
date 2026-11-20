"""ハイブリッドNN(実数層+論理ゲート層)のサブパッケージ。

replace系は遅延インポート: layers.logic_linear が本パッケージの
policy を参照するため、eagerに replace を読むと循環importになる。
"""

from .policy import LogicSpec

_LAZY = {
    'convert_linear_to_logic': '.replace',
    'replace_mlp_with_lut': '.replace',
    'replace_modules': '.replace',
    'split_param_groups': '.replace',
}

__all__ = ['LogicSpec', *sorted(_LAZY)]


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(
            f"module 'griffith_logic.hybrid' has no attribute {name!r}"
        ) from None
    import importlib
    module = importlib.import_module(_LAZY[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
