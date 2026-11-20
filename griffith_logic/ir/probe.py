"""推論プローブ: 全ノードの中間値を取得するAPI。

hard(ビット厳密)とsoft(torch実数)の2系統:
    probe      — simulate_traceの公開wrapper。numpyのみでビット厳密に実行
    probe_soft — torchモデルのforwardから全ノード出力を捕捉。
                 eval modeで二値入力を与えれば、ビット化(>0.5)した値が
                 probeと全ノードで一致する(テストで担保)

IRノードidとの対応付け:
    - IRGraphModule: forward_trace(node_id→Tensor)をそのまま使う
    - nn.Sequential: 各トップレベルモジュールにforward hookを張り、
      ビジュアルエディタの lut_layers_with_ids と同じ
      採番規約 n{i}_クラス名(i=Sequential内の位置)で対応付ける。
      規約がgraphのノードidと合わない場合(build_torch直組み等)は、
      非Flattenモジュールとトポロジカル順ノードの位置対応へ
      フォールバックする。nn.Flattenは恒等(平坦入力前提)なので記録しない
"""

import numpy as np
import torch
import torch.nn as nn

from .build import IRGraphModule
from .schema import IRGraph
from .simulate import simulate_trace


def probe(graph: IRGraph, x_bits: np.ndarray) -> dict:
    """IRをビット厳密に実行し、全ノードの出力を返す。

    Args:
        graph: 検証済みIR
        x_bits: (batch, in_width) の0/1配列

    Returns:
        {node_id: (batch, width)のint64配列}。入力ポートidの値も含む
    """
    return simulate_trace(graph, np.asarray(x_bits))


def _sequential_trace(model: nn.Sequential, graph: IRGraph,
                      x: torch.Tensor) -> dict:
    """Sequentialの各トップレベルモジュール出力をforward hookで捕捉する"""
    modules = list(model)
    captured: list[tuple[int, torch.Tensor]] = []
    handles = []

    def _make_hook(i: int):
        def hook(_mod, _inputs, output):
            captured.append((i, output))
        return hook

    for i, m in enumerate(modules):
        if isinstance(m, nn.Flatten):
            continue  # 恒等(平坦入力前提)。IRノードにも現れない
        handles.append(m.register_forward_hook(_make_hook(i)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    node_ids = {n.id for n in graph.nodes}
    keys = [f'n{i}_{type(modules[i]).__name__.lower()}' for i, _ in captured]
    if not set(keys) <= node_ids:
        # フォールバック: 非Flattenモジュール ↔ トポロジカル順ノードの位置対応
        topo = [n for n in graph.topological_order() if n.type != 'flatten']
        if len(topo) != len(captured):
            raise ValueError(
                f'モデルとIRの対応付けに失敗: モジュール{len(captured)}個に対し'
                f'IRノード{len(topo)}個(n{{i}}_クラス名規約も不一致)'
            )
        keys = [n.id for n in topo]

    trace: dict[str, torch.Tensor] = {}
    if graph.inputs:
        trace[graph.inputs[0].id] = x
    for key, (_, out) in zip(keys, captured, strict=False):
        trace[key] = out
    return trace


def probe_soft(model: nn.Module, graph: IRGraph, x: torch.Tensor) -> dict:
    """torchモデルのforwardから全ノードの出力を捕捉する。

    ビット化照合(== probe)にはmodelをeval modeにして二値入力を渡すこと。
    train modeなら連続緩和の中間値がそのまま得られる(可視化用)。

    Args:
        model: IRGraphModule または nn.Sequential(build_torch/エディタの構築物)
        graph: modelに対応するIR(ノードidの対応付けに使用)
        x: (batch, in_width) のfloatテンソル(幅規約=C*H*W flatten)

    Returns:
        {node_id: (batch, width)のTensor}。入力ポートidの値も含む。
        flattenノードは恒等のため含まれない(Sequential経路)
    """
    if isinstance(model, IRGraphModule):
        with torch.no_grad():
            return model.forward_trace(x)
    if isinstance(model, nn.Sequential):
        return _sequential_trace(model, graph, x)
    raise TypeError(
        f'probe_soft対象はIRGraphModuleかnn.Sequentialのみ: '
        f'{type(model).__name__}'
    )
