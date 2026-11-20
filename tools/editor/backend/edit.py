"""ビルド済みLutLayerの単一ニューロンin-place編集。

restore_lut_layer(全ニューロン一括)の部分適用版。真理値表編集は
weights_from_truth_tables で該当行のみ差し替え、接続編集は該当列の
差し替え+逆引きバッファ(rev_starts/rev_flat_y/rev_flat_slot)の再計算を行う。
fastapiに依存しないためテストから直接importできる。
"""

import torch

from griffith_logic.core.connections import build_reverse_index
from griffith_logic.ir import weights_from_truth_tables


@torch.no_grad()
def set_neuron_truth_table(layer, neuron: int, tt: int) -> None:
    """LutLayerの1ニューロンのweightsを真理値表からin-place更新する。

    Args:
        layer: 対象LutLayer
        neuron: ニューロン番号(0 <= neuron < out_dim)
        tt: 真理値表整数(bit k=butterfly基底順)

    Raises:
        ValueError: 範囲外のneuron、またはパラメータ化で表現不能な真理値表
    """
    if not 0 <= neuron < layer.out_dim:
        raise ValueError(f'neuronが範囲外: {neuron} (out_dim={layer.out_dim})')
    param = 'softmax16g' if getattr(layer, '_use_16gate', False) \
        else layer.parameterization
    w = weights_from_truth_tables(param, layer.n_inputs, [tt])
    layer.weights[neuron].copy_(w[0].to(layer.weights.device, layer.weights.dtype))


@torch.no_grad()
def set_neuron_indices(layer, neuron: int, slots: list[int]) -> None:
    """LutLayerの1ニューロンの接続スロットを差し替える。

    indices更新後は build_reverse_index で逆引きバッファを必ず再計算する
    (漏れるとCUDA backwardが壊れる — restore_lut_layer と同じ規約)。

    Args:
        layer: 対象LutLayer
        neuron: ニューロン番号(0 <= neuron < out_dim)
        slots: 長さn_inputsの入力インデックス列(各 0 <= v < in_dim)

    Raises:
        ValueError: 範囲外のneuron/slots、スロット数不一致
    """
    if not 0 <= neuron < layer.out_dim:
        raise ValueError(f'neuronが範囲外: {neuron} (out_dim={layer.out_dim})')
    if len(slots) != layer.n_inputs:
        raise ValueError(f'slots数({len(slots)}) != n_inputs({layer.n_inputs})')
    if any(not 0 <= int(v) < layer.in_dim for v in slots):
        raise ValueError(f'slotsが入力幅(in_dim={layer.in_dim})を超えている: {slots}')

    layer.indices[:, neuron] = torch.tensor(
        [int(v) for v in slots], dtype=torch.int64, device=layer.indices.device
    )
    rev_starts, rev_flat_y, rev_flat_slot = build_reverse_index(
        layer.indices, layer.in_dim
    )
    layer.rev_starts = rev_starts
    layer.rev_flat_y = rev_flat_y
    layer.rev_flat_slot = rev_flat_slot


def generate_wiring(
    kind: str, in_width: int, out_width: int, n_inputs: int,
    spec: dict | None = None, *, dedupe: bool = False,
) -> dict:
    """配線生成 API の本体。core.connections.make_connections で indices を作り、
    被覆率などの統計(connection_stats)を添えて返す。

    STATE に触れない純関数。フロントは返った indices を IR の lut_layer.params.indices へ
    書き込む(構造は不変なので truth_tables は保たれ、build で LutLayer に復元される)。

    Args:
        kind: 配線種(core.connections.WIRING_KINDS)
        in_width / out_width / n_inputs: 対象 lut_layer の形
        spec: 種ごとの引数(core.connections.WIRING_SPEC_KEYS)
        dedupe: 真なら重複ピンを潰す

    Returns:
        {'kind', 'indices': (n_inputs, out_width) のリスト, 'stats': 被覆率・使用回数の要約}

    Raises:
        ValueError: 未知の種、引数の欠落・過剰、入力数の不一致(呼び出し側が 422 にする)
    """
    from griffith_logic.core.connections import connection_stats, make_connections

    idx = make_connections(
        kind, int(in_width), int(out_width), int(n_inputs),
        dedupe=dedupe, **dict(spec or {}))
    stats = connection_stats(idx, int(in_width))
    return {
        'kind': kind,
        'indices': idx.tolist(),
        'stats': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in stats.items()},
    }
