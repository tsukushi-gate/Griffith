"""コンパイル時のモデル等価配線ソート(gatherのメモリ局所性改善)。

DLGNの配線(LutLayer.indices)は学習中固定(register_buffer)なので、出力
ニューロンの番号をモデルの意味を変えずに付け替えられる。各LutLayerの出力
ニューロンを「第1入力インデックス(indices[0, :])昇順」に安定ソートして
renumberすると、次層のgatherが番地順に近いアクセスになる。

効果の実測(RTX 4090): gather単体ではランダム順の約2.1倍の実効帯域になる一方、
融合CUDAカーネルを使う学習ステップ全体では0.5〜0.7%の改善に留まる。融合
カーネルはサンプル方向に連続した番地をまとめて読む設計で、入力の並び順に
依存する部分が小さいため。opt-inのユーティリティとして提供する。

数学的根拠:
    層lの出力ニューロンをpermで並べ替える(new_row[i] = old_row[perm[i]])
    ⇔ 新ニューロンiの計算内容 = 旧ニューロンperm[i]の計算内容
    ⇔ 新出力ベクトル y' は旧出力ベクトル y の並べ替え: y'[i] = y[perm[i]]
    次層が旧ニューロンjの出力を参照していた場合、新ベクトルではy[j]は
    位置 inv_perm[j] (permの逆置換) に移っているため、次層のindices中の
    値jはinv_perm[j]へremapする必要がある。
    最終層の出力を直接消費するGroupSumは連続kブロックの和を取るだけなので、
    ブロック内の順序を変えてもブロック単位の和は不変
    → 最終層は「ブロック内限定」の置換にすればremap不要のまま安全。

対象: nn.Sequential内の、連続するLutLayer列(すべてwiring='fixed') +
（あれば）末尾のGroupSum。それ以外の構造(非Sequential、LutLayerが
連続していない、wiring='learnable'を含む)は明示的にUnsupportedWiringErrorを
送出する(無変更で失敗を握りつぶすと「効いていないのに気づかない」
リスクがあるため、既定は例外)。
"""

from __future__ import annotations

import copy
from typing import cast

import torch
import torch.nn as nn

from ..group_sum import GroupSum
from ..layers.lut import LutLayer
from .connections import build_reverse_index


class UnsupportedWiringError(ValueError):
    """sort_model_wiringが対応できないモデル構造・配線方式の場合に送出。"""


def _indices(layer: LutLayer) -> torch.Tensor:
    """LutLayer.indices(register_buffer)をTensorとして返す。

    nn.Module.__getattr__ の戻り型は ``Tensor | Module`` なので、mypy向けに
    Tensorへ明示的にcastする(実行時は常にTensor)。
    """
    return cast(torch.Tensor, layer.indices)


def _invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """permの逆置換を返す: inv[perm[i]] == i"""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.shape[0], device=perm.device, dtype=perm.dtype)
    return inv


def _sort_key(layer: LutLayer) -> torch.Tensor:
    """層の出力ニューロンごとの「第1入力インデックス」(indices[0, :])。

    indices の形状は (n_inputs, out_dim) — indices[0]が各出力ニューロンの
    第1入力スロットの参照先(前層の出力番号 or データ次元)。
    """
    return _indices(layer)[0].to(torch.int64)


def _permute_layer_outputs(
    layer: LutLayer,
    perm: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """LutLayerの出力ニューロンをpermに従って並べ替える(in-place)。

    weights・indices の「出力次元」側の行/列をpermで並べ替え、
    逆引きバッファ(rev_starts/rev_flat_y/rev_flat_slot)を再構築する。
    weights の勾配(.grad)と、optimizer が渡されればその per-parameter 状態
    (Adam の exp_avg / exp_avg_sq 等、weights と同じ形のテンソル)も同じ置換で
    並べ替える(並べ替えないとモーメントが別のニューロンに適用される)。
    パラメータ化方式によらず weights の dim 0 が常に out_dim なので
    (softmax/independent/raw/iwp/16gate/warp いずれも (out_dim, ...))
    この処理は分岐不要で共通に適用できる。
    """
    indices = _indices(layer)
    perm = perm.to(device=indices.device, dtype=torch.int64)
    with torch.no_grad():
        if layer.weights.grad is not None:
            grad = layer.weights.grad
            grad.copy_(grad.index_select(0, perm.to(grad.device)))
        if optimizer is not None:
            for value in optimizer.state.get(layer.weights, {}).values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.shape == layer.weights.shape
                ):
                    value.copy_(value.index_select(0, perm.to(value.device)))
        layer.weights.data = layer.weights.data.index_select(0, perm).contiguous()
        # indices: (n_inputs, out_dim) の出力次元(dim=1)を並べ替え
        layer.indices = indices.index_select(1, perm).contiguous()

    rev_starts, rev_flat_y, rev_flat_slot = build_reverse_index(
        _indices(layer), layer.in_dim
    )
    layer.rev_starts = rev_starts
    layer.rev_flat_y = rev_flat_y
    layer.rev_flat_slot = rev_flat_slot


def _remap_consumer_indices(next_layer: LutLayer, inv_perm: torch.Tensor) -> None:
    """次層のindicesが参照する「前層の出力番号」を旧→新へremapする。"""
    indices = _indices(next_layer)
    inv_perm = inv_perm.to(device=indices.device, dtype=torch.int64)
    with torch.no_grad():
        next_layer.indices = inv_perm[indices].contiguous()

    rev_starts, rev_flat_y, rev_flat_slot = build_reverse_index(
        _indices(next_layer), next_layer.in_dim
    )
    next_layer.rev_starts = rev_starts
    next_layer.rev_flat_y = rev_flat_y
    next_layer.rev_flat_slot = rev_flat_slot


def _find_lut_run_and_groupsum(model: nn.Sequential):
    """Sequential内の連続LutLayer列と、それに続くGroupSum(あれば)を探す。

    Returns:
        (lut_layers, groupsum_or_None)

    Raises:
        UnsupportedWiringError: LutLayerが複数箇所に分かれて存在する
            (連続run以外に混在する)、またはLutLayerが1つも無い場合。
    """
    positions = [i for i, m in enumerate(model) if isinstance(m, LutLayer)]
    if not positions:
        return [], None

    expected = list(range(positions[0], positions[0] + len(positions)))
    if positions != expected:
        raise UnsupportedWiringError(
            '非対応構造: LutLayerがnn.Sequential内で連続していません'
            f'(検出位置: {positions})。sort_model_wiringは'
            '「連続するLutLayer列(+末尾GroupSum)」構造のみ対応です。'
        )

    lut_layers = [model[i] for i in positions]

    groupsum = None
    tail_pos = positions[-1] + 1
    if tail_pos < len(model) and isinstance(model[tail_pos], GroupSum):
        groupsum = model[tail_pos]

    return lut_layers, groupsum


def sort_model_wiring(
    model: nn.Sequential,
    *,
    inplace: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
) -> nn.Sequential:
    """各LutLayerの出力ニューロンを第1入力インデックス昇順にrenumberする。

    モデル等価な置換(forward出力・backward勾配は対応付けの上で不変)。
    ゲートを読み出し先の番地順に並べ、gatherのメモリ局所性を改善するのが目的。
    融合CUDAカーネルを使う学習では速度への効果は小さい(実測0.5〜0.7%)。

    optimizer構築前に呼ぶのが基本。構築後(学習途中を含む)に呼ぶ場合は
    `optimizer=` を渡すこと。Adam等のモーメント(weightsと同じ形の状態)を同じ
    置換で並べ替える。渡さないとモーメントが別のニューロンに適用されたまま
    学習が続く(本関数からはoptimizerの存在を検出できないため、呼び出し側の
    責務となる)。

    Args:
        model: nn.Sequential(必要ならFlatten等の非LutLayer要素も可)。
            連続するLutLayer列(すべてwiring='fixed') + 末尾GroupSum(あれば)
            という構造のみ対応。
        inplace: Trueなら引数のmodelを直接書き換えて返す(既定)。
            Falseならdeepcopyしたモデルを書き換えて返し、元のmodelは
            変更しない(before/afterの比較テストや、フォールバック用に
            元modelを残したい呼び出し元向け)。
        optimizer: modelのパラメータを持つoptimizer(構築後に呼ぶ場合)。
            per-parameter状態をweightsと同じ置換で並べ替える。
            inplace=Falseとは併用できない(コピー側のパラメータはoptimizerが
            持つパラメータと別物のため)。

    Returns:
        置換適用後のモデル(inplace=Trueなら引数と同一オブジェクト)。

    Raises:
        UnsupportedWiringError:
            - modelがnn.Sequentialでない
            - LutLayerがSequential内で連続していない
            - いずれかのLutLayerがwiring='learnable'
            - 最終LutLayerの出力次元がGroupSumのクラス数で割り切れない
              (LutLayer/GroupSum自体の不整合。通常は起こらない)
        ValueError: optimizer と inplace=False を同時に指定した
    """
    if not isinstance(model, nn.Sequential):
        raise UnsupportedWiringError(
            f'非対応構造: nn.Sequentialのみ対応です(受け取った型: {type(model)})'
        )

    if optimizer is not None and not inplace:
        raise ValueError(
            'optimizer を渡す場合は inplace=True にしてください'
            '(inplace=False はモデルをコピーするため、optimizer が持つ'
            'パラメータとは別物になります)'
        )
    if not inplace:
        model = copy.deepcopy(model)

    lut_layers, groupsum = _find_lut_run_and_groupsum(model)
    if not lut_layers:
        # 対象(LutLayer)が無いモデル。無変更でそのまま返す。
        return model

    for layer in lut_layers:
        if layer.wiring != 'fixed':
            raise UnsupportedWiringError(
                "sort_model_wiringはwiring='fixed'のLutLayerのみ対応です"
                f"(検出: wiring={layer.wiring!r})。learnable wiringは"
                'LearnableMapping側のパラメータ整合も必要になり非対応のため、'
                'そのようなLutLayerを含むモデルには適用できません。'
            )

    n_layers = len(lut_layers)
    for li, layer in enumerate(lut_layers):
        key = _sort_key(layer)
        out_dim = key.shape[0]
        is_final = li == n_layers - 1

        if is_final and groupsum is not None:
            # 最終層: GroupSumのグループ構造(out_dim // k の連続ブロック)を
            # 壊さないよう、置換をグループ内に限定する。
            k = groupsum.k
            if out_dim % k != 0:
                raise UnsupportedWiringError(
                    f'最終LutLayerのout_dim({out_dim})がGroupSumのクラス数'
                    f'({k})で割り切れません'
                )
            group_size = out_dim // k
            perm = torch.arange(out_dim, device=key.device)
            for g in range(k):
                s, e = g * group_size, (g + 1) * group_size
                local_order = torch.argsort(key[s:e], stable=True)
                perm[s:e] = local_order + s
        else:
            perm = torch.argsort(key, stable=True)

        _permute_layer_outputs(layer, perm, optimizer)

        if li + 1 < n_layers:
            inv_perm = _invert_permutation(perm)
            _remap_consumer_indices(lut_layers[li + 1], inv_perm)

    return model
