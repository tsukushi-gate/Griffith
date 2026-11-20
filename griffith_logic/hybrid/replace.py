"""既存PyTorchモデルの部分的な論理ゲート化API。

torch.ao.quantizationと同じ「named_modules走査 + 親のsetattr」定石で
モジュールを置換する。LL-ViTのように「Attentionは実数のまま、
MLPだけLUT化」といったハイブリッド構成を1関数で実現する。
"""

import copy
import fnmatch
from collections.abc import Callable

import torch.nn as nn

from ..layers.dwn import DWNLayer
from ..layers.logic_linear import LogicLinear, LogicMLP
from ..layers.lut import LutLayer
from .policy import LogicSpec

MatchSpec = type | tuple | Callable[[str, nn.Module], bool] | str


def _to_predicate(match: MatchSpec) -> Callable[[str, nn.Module], bool]:
    if isinstance(match, str):
        return lambda name, m: fnmatch.fnmatch(name, match)
    if isinstance(match, type) or isinstance(match, tuple):
        return lambda name, m: isinstance(m, match)
    return match


def replace_modules(
    model: nn.Module,
    *,
    match: MatchSpec,
    factory: Callable[[str, nn.Module], nn.Module],
    inplace: bool = False,
) -> tuple[nn.Module, list[str]]:
    """matchに一致するサブモジュールをfactoryの生成物で置換する。

    Args:
        match: 型 / 型タプル / 述語(name, module)->bool / 名前glob文字列
        factory: (name, 旧モジュール) -> 新モジュール
        inplace: Falseならdeepcopyしてから置換(元モデル保全)

    Returns:
        (置換後モデル, 置換したモジュール名リスト)

    Note:
        一致したモジュールの内部はそれ以上走査しない
        (親子が両方一致する場合は親を優先)。
    """
    pred = _to_predicate(match)
    if not inplace:
        model = copy.deepcopy(model)

    replaced: list[str] = []
    matched: list[str] = []
    for name, module in model.named_modules():
        if name == '':
            continue
        # 既に一致した親の内部はスキップ
        if any(name.startswith(p + '.') for p in matched):
            continue
        if pred(name, module):
            matched.append(name)

    for name in matched:
        old = model.get_submodule(name)
        new = factory(name, old)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new)
        replaced.append(name)

    return model, replaced


def _name_filter(include, exclude):
    def ok(name: str) -> bool:
        if include is not None and not fnmatch.fnmatch(name, include):
            return False
        if exclude is not None and fnmatch.fnmatch(name, exclude):
            return False
        return True
    return ok


def convert_linear_to_logic(
    model: nn.Module,
    *,
    spec: LogicSpec | None = None,
    include: str | None = None,
    exclude: str | None = None,
    inplace: bool = False,
) -> tuple[nn.Module, list[str]]:
    """nn.Linear を LogicLinear に置換する。

    Args:
        spec: 論理ゲート化レシピ
        include: この名前globに一致するLinearだけ置換(例: '*.mlp.*')
        exclude: この名前globに一致するLinearは除外(例: '*head*')
    """
    name_ok = _name_filter(include, exclude)

    def match(name: str, m: nn.Module) -> bool:
        return isinstance(m, nn.Linear) and name_ok(name)

    def factory(name: str, old: nn.Linear) -> nn.Module:
        return LogicLinear(old.in_features, old.out_features, spec=spec)

    return replace_modules(model, match=match, factory=factory, inplace=inplace)


def _find_linears(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def replace_mlp_with_lut(
    model: nn.Module,
    *,
    spec: LogicSpec | None = None,
    mlp_class: type | None = None,
    mlp_name_pattern: str = '*mlp*',
    inplace: bool = False,
) -> tuple[nn.Module, list[str]]:
    """FFN/MLPブロック全体を LogicMLP に置換する(LL-ViT用途)。

    Attention・LayerNorm・埋め込みはPyTorchのまま残し、MLP部分だけを
    LUTベースの論理ゲート型に置き換える。

    検出条件(いずれか):
        - mlp_class 指定時: そのクラスのインスタンス
        - 名前の末尾要素が mlp_name_pattern に一致し、内部にnn.Linearを
          2つ以上含むモジュール

    in/out次元は内部の最初/最後のnn.Linearから自動推定する。
    """
    def match(name: str, m: nn.Module) -> bool:
        if mlp_class is not None:
            return isinstance(m, mlp_class)
        leaf = name.rpartition('.')[2]
        if not fnmatch.fnmatch(leaf.lower(), mlp_name_pattern):
            return False
        return len(_find_linears(m)) >= 2

    def factory(name: str, old: nn.Module) -> nn.Module:
        linears = _find_linears(old)
        assert linears, f'{name}: 内部にnn.Linearが見つからない'
        return LogicMLP(
            linears[0].in_features, linears[-1].out_features, spec=spec,
        )

    return replace_modules(model, match=match, factory=factory, inplace=inplace)


def split_param_groups(
    model: nn.Module,
    *,
    logic_lr: float | None = None,
    ann_lr: float | None = None,
) -> list[dict]:
    """LUT系パラメータとANN系パラメータをoptimizer param groupに分割する。

    論理ゲート層(lr=0.01前後)と実数層(lr=1e-3前後)で適切な学習率が
    1桁異なるため、分離して個別指定できるようにする。

    Returns:
        torch.optim.*(param_groups, ...) にそのまま渡せるリスト
    """
    logic_params, ann_params = [], []
    logic_ids = set()
    for m in model.modules():
        if isinstance(m, LutLayer | DWNLayer):
            for p in m.parameters(recurse=True):
                logic_ids.add(id(p))
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (logic_params if id(p) in logic_ids else ann_params).append(p)

    groups = []
    if logic_params:
        g = {'params': logic_params, 'name': 'logic'}
        if logic_lr is not None:
            g['lr'] = logic_lr
        groups.append(g)
    if ann_params:
        g = {'params': ann_params, 'name': 'ann'}
        if ann_lr is not None:
            g['lr'] = ann_lr
        groups.append(g)
    return groups
