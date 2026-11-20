"""Don't-care修復パス: hard化後のLUT網を訓練データ上で離散修復する。

NullaNet/ReducedLUT(FPGA 2025)の考え方の適用:
訓練データ多様体上で**観測されない**LUT入力パターンはdon't careであり、
そのビットは自由に選べる。逆に観測パターンのビットは、soft参照
(連続緩和モデルの同ニューロン活性)の多数決、または訓練誤差最小化で
選び直せる。緩和(勾配)を一切経ない離散修復。

提供する修復:
    repair_refit   — 各ニューロンのTTを観測パターン上でsoft参照に再適合
                     (未観測ビットは現状維持=don't care)
    repair_flip    — 貪欲ゲートflip(coordinate descent):
                     各ニューロンの16択を訓練hard精度最大へ貪欲に再選択
    repair_output_layer_sat — 最終LUT層をSAT(部分関数+don't care)で
                     厳密再合成(analysis/sat_capacityを利用)

いずれも「訓練hard精度が非減少の場合のみ採択」する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..ir.weights import weights_from_truth_tables
from ..layers import LutLayer


@dataclass
class RepairReport:
    """修復の記録。"""

    acc_before: float
    acc_after: float
    layer_changes: list[dict] = field(default_factory=list)


def _lut_layers(model: nn.Module) -> list[LutLayer]:
    return [m for m in model.modules() if isinstance(m, LutLayer)]


def _assert_repairable(layer: LutLayer) -> None:
    assert layer.n_inputs == 2 and layer._use_16gate, (
        'don\'t-care修復はN=2 softmax(16ゲートモード)のみ対応'
    )


@torch.no_grad()
def _hard_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    was = model.training
    model.eval()
    acc = (model(x.round()).argmax(-1) == y).float().mean().item()
    model.train(was)
    return acc


@torch.no_grad()
def _layer_inputs_hard(model: nn.Sequential, x: torch.Tensor) -> list[torch.Tensor]:
    """各LutLayerへのhard入力(eval経路)を層順に返す。"""
    model.eval()
    inputs = []
    h = x.round()
    for m in model:
        if isinstance(m, LutLayer):
            inputs.append(h.clone())
        h = m(h)
    return inputs


@torch.no_grad()
def _layer_soft_outputs(model: nn.Sequential, x: torch.Tensor) -> list[torch.Tensor]:
    """各LutLayerのsoft出力(train経路、勾配なし)を層順に返す。"""
    was = model.training
    model.train()
    outputs = []
    h = x
    for m in model:
        h = m(h)
        if isinstance(m, LutLayer):
            outputs.append(h.clone())
    model.train(was)
    return outputs


@torch.no_grad()
def _current_tts(layer: LutLayer) -> list[int]:
    """現在のargmaxゲートのTT(butterfly順4bit整数)を返す。"""
    gate_idx = layer.weights.argmax(-1)
    tt_bits = layer.gate_table[gate_idx]  # (out_dim, 4)
    tts = []
    for o in range(layer.out_dim):
        tt = 0
        for k in range(4):
            if tt_bits[o, k] > 0.5:
                tt |= (1 << k)
        tts.append(tt)
    return tts


@torch.no_grad()
def _write_tts(layer: LutLayer, tts: list[int]) -> None:
    w = weights_from_truth_tables('softmax16g', 2, tts)
    layer.weights.copy_(w.to(layer.weights.device, layer.weights.dtype))


@torch.no_grad()
def repair_refit(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor,
                 *, passes: int = 2) -> RepairReport:
    """観測パターン再適合: TTビット=soft参照活性(>0.5)の観測多数決。

    未観測パターンのビットはdon't care(現状維持)。層ごとに書き換え、
    訓練hard精度が下がる場合はその層をロールバックする。
    層を書き換えると下流の観測パターンが変わるため、パスを複数回まわす。
    """
    layers = _lut_layers(model)
    for layer in layers:
        _assert_repairable(layer)

    report = RepairReport(acc_before=_hard_acc(model, x, y),
                          acc_after=0.0)
    acc = report.acc_before
    for p in range(passes):
        soft_out = _layer_soft_outputs(model, x)
        hard_in = _layer_inputs_hard(model, x)
        for li, layer in enumerate(layers):
            gathered = hard_in[li][:, layer.indices.T.reshape(-1)].reshape(
                x.shape[0], layer.out_dim, 2,
            )
            addr = (gathered[..., 0] + 2 * gathered[..., 1]).long()  # (B, out)
            ref = (soft_out[li] > 0.5)  # (B, out)

            old_tts = _current_tts(layer)
            new_tts = []
            for o in range(layer.out_dim):
                tt = old_tts[o]
                for k in range(4):
                    m = addr[:, o] == k
                    n_obs = int(m.sum())
                    if n_obs == 0:
                        continue  # don't care: 現状維持
                    bit = int(ref[m, o].float().mean().item() > 0.5)
                    tt = (tt & ~(1 << k)) | (bit << k)
                new_tts.append(tt)

            if new_tts == old_tts:
                continue
            _write_tts(layer, new_tts)
            new_acc = _hard_acc(model, x, y)
            if new_acc >= acc:
                report.layer_changes.append({
                    'pass': p, 'layer': li, 'kind': 'refit',
                    'n_changed': sum(a != b for a, b in zip(new_tts, old_tts, strict=True)),
                    'acc': new_acc,
                })
                acc = new_acc
                # 下流の観測パターンが変わったので再計算
                hard_in = _layer_inputs_hard(model, x)
            else:
                _write_tts(layer, old_tts)
    report.acc_after = acc
    return report


@torch.no_grad()
def repair_flip(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor,
                *, passes: int = 2) -> RepairReport:
    """貪欲ゲートflip: 各ニューロンの16択を訓練hard精度最大へ座標降下。

    改善が無くなるか最大passesで停止。O(ニューロン数×16×forward)。
    """
    layers = _lut_layers(model)
    for layer in layers:
        _assert_repairable(layer)

    report = RepairReport(acc_before=_hard_acc(model, x, y), acc_after=0.0)
    acc = report.acc_before
    for p in range(passes):
        improved = False
        for li, layer in enumerate(layers):
            tts = _current_tts(layer)
            for o in range(layer.out_dim):
                best_tt, best_acc = tts[o], acc
                orig = tts[o]
                for cand in range(16):
                    if cand == orig:
                        continue
                    tts[o] = cand
                    _write_tts(layer, tts)
                    a = _hard_acc(model, x, y)
                    if a > best_acc:
                        best_tt, best_acc = cand, a
                tts[o] = best_tt
                _write_tts(layer, tts)
                if best_acc > acc:
                    report.layer_changes.append({
                        'pass': p, 'layer': li, 'neuron': o, 'kind': 'flip',
                        'tt': best_tt, 'acc': best_acc,
                    })
                    acc = best_acc
                    improved = True
        if not improved:
            break
    report.acc_after = acc
    return report


@torch.no_grad()
def repair_output_layer_sat(model: nn.Sequential, x: torch.Tensor,
                            y: torch.Tensor) -> RepairReport:
    """最終LUT層をSAT(部分関数+don't care)で厳密再合成する。

    上流層のhard出力を入力パターンとし、観測パターンのみ制約して
    最終層TT+GroupSum読み出しをSATで解く。UNSATなら現状維持。
    """
    from .sat_capacity import LutTopology, TargetSpec, can_realize

    layers = _lut_layers(model)
    for layer in layers:
        _assert_repairable(layer)
    last = layers[-1]

    # GroupSum(k=2)がモデル末尾にある前提
    from ..group_sum import GroupSum
    gs = [m for m in model.modules() if isinstance(m, GroupSum)]
    assert gs and gs[-1].k == 2, 'repair_output_layer_satはGroupSum(k=2)前提'

    report = RepairReport(acc_before=_hard_acc(model, x, y), acc_after=0.0)

    hard_in = _layer_inputs_hard(model, x)[-1].numpy().astype(np.int64)
    topo = LutTopology(n_in=last.in_dim, layer_widths=[last.out_dim],
                       readout='groupsum')
    target = TargetSpec(dataset=(hard_in, y.numpy().astype(np.int64)))
    wiring = [last.indices.numpy().copy()]
    r = can_realize(topo, target, wiring=wiring)

    if r.status == 'sat':
        old_tts = _current_tts(last)
        _write_tts(last, r.truth_tables[0])
        new_acc = _hard_acc(model, x, y)
        if new_acc >= report.acc_before:
            report.layer_changes.append({
                'kind': 'sat_output',
                'n_changed': sum(a != b for a, b in
                                 zip(r.truth_tables[0], old_tts, strict=True)),
                'acc': new_acc,
            })
            report.acc_after = new_acc
            return report
        _write_tts(last, old_tts)
    report.acc_after = report.acc_before
    return report
