"""オプティマイザのパラメータグループ構築(IWP系のweight decayガード)。"""

import warnings

import torch.nn as nn

# weight decay を掛けてはいけないパラメータ化。
# いずれも重みが「エントリ確率ω∈[0,1]の(直接あるいは推定器経由の)
# パラメータ化」であり、Ω→0 は ω≈0.5 = 最も離散化誤差の大きい状態を意味する。
_NO_DECAY_PARAMETERIZATIONS = ('independent', 'iwp', 'raw', 'warp')


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """weight decay を掛けてはいけない論理ゲート重みを wd=0 グループへ分離する。

    論文 arXiv:2510.03250 §C.1: IWP系(ω∈[0,1]の直接パラメータ化)では Ω→0 が
    ω≈0.5(=最も離散化誤差の大きい状態)を意味するため、weight decay は
    離散化誤差を積極的に悪化させる。ここでの「IWP系」は上記の性質を共有する
    パラメータ化群の総称であり、論文IWPそのものは 'independent' のみ
    (既存の 'iwp' はリテラル積で論文IWPとは別物だが、同じ性質を持つため同列に扱う)。

    softmax/softmax16g は「エントリ間の競合ロジット」であり、Ω→0 は一様分布
    (=特定ゲートへの偏りがない)を意味するだけで離散化を悪化させないため、
    通常のweight decayグループに残す。

    Args:
        model: 対象モデル
        weight_decay: 既定のweight decay係数

    Returns:
        torch.optim へ渡すパラメータグループのリスト。
        weight_decay==0.0 なら分離不要なので単一グループを返す。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if weight_decay == 0.0:
        return [{'params': params}]

    no_decay_ids: set[int] = set()
    for module in model.modules():
        param = getattr(module, 'weights', None)
        if not isinstance(param, nn.Parameter) or not param.requires_grad:
            continue
        parameterization = getattr(module, 'parameterization', None)
        if parameterization in _NO_DECAY_PARAMETERIZATIONS:
            no_decay_ids.add(id(param))

    decay = [p for p in params if id(p) not in no_decay_ids]
    no_decay = [p for p in params if id(p) in no_decay_ids]

    if not no_decay:
        return [{'params': params}]

    warnings.warn(
        f'IWP系パラメータ {len(no_decay)} 個を weight decay 対象から除外しました'
        '(ω∈[0,1]の直接パラメータ化ではΩ→0がω≈0.5=最悪の離散化を意味するため。'
        '出典: arXiv:2510.03250 §C.1)',
        stacklevel=2,
    )
    return [
        {'params': decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
