"""監視データのエンコード/フレーム組立。

u8テンソルエンベロープはWS/RESTで共通形式:
    {"encoding": "u8", "shape": [...], "scale": <float>, "data": "<base64>"}
値 = round(v / scale * 255)。量子化誤差は scale/255 以下。
確率分布は scale=1.0 固定、その他はフレーム内max。
※ u8は符号なしのため、値は非負前提(勾配はニューロン別ノルムに集約してから使う)。
"""

from __future__ import annotations

import base64

import torch


def encode_u8(t: torch.Tensor, *, scale: float | None = None) -> dict:
    """非負テンソルをu8量子化エンベロープへ(scale省略時はabs-max)。"""
    t = t.detach().to(torch.float32).cpu()
    if scale is None:
        m = float(t.abs().max().item())
        scale = m if m > 0 else 1.0
    q = (t / scale * 255.0).round().clamp_(0, 255).to(torch.uint8)
    return {
        'encoding': 'u8',
        'shape': list(t.shape),
        'scale': scale,
        'data': base64.b64encode(q.numpy().tobytes()).decode('ascii'),
    }


def decode_u8(env: dict) -> torch.Tensor:
    """u8エンベロープを復元する(テスト用の逆変換)。"""
    import numpy as np

    raw = base64.b64decode(env['data'])
    q = torch.from_numpy(
        np.frombuffer(raw, dtype=np.uint8).copy()
    ).reshape(env['shape'])
    return q.to(torch.float32) / 255.0 * float(env['scale'])


# --- WSフレーム組立 ---

# これを超える要素数(out_dim × n_entries)の層は集約モードへ格下げする
DETAIL_MAX_ELEMS = 262144


def build_capture_config(session) -> dict:
    """接続直後(+capture変更時)に送るcapture設定フレーム。"""
    from .capture import snapshot_bytes_estimate

    return {
        'type': 'capture_config',
        'level': session.capture_level,
        'stride': session.history.stride,
        'method': session.method,
        'snapshot_bytes': snapshot_bytes_estimate(
            session.lut_layers, session.capture_level
        ),
        'detail_max_elems': DETAIL_MAX_ELEMS,
    }


def build_gate_distribution(
    name: str, weights: torch.Tensor, parameterization: str,
    n_inputs: int, step: int,
) -> dict:
    """ゲート確率分布フレーム(現行layer_summaryのconfidence max潰しの非集約版)。

    大規模層は自動で mode:'aggregate'(argmaxヒスト+エントロピー)へ格下げする。
    """
    from .extract import gate_entropy, gate_probs_from_weights

    probs = gate_probs_from_weights(weights, parameterization, n_inputs)
    base = {
        'type': 'gate_distribution',
        'layer': name,
        'step': step,
        'n_inputs': n_inputs,
        'parameterization': parameterization,
    }
    if probs is None:
        return {**base, 'mode': 'aggregate', 'hist': [], 'entropy_mean': None}

    entropy = float(gate_entropy(probs).mean().item())
    if probs.numel() > DETAIL_MAX_ELEMS:
        n_bins = probs.shape[-1]
        hist = torch.bincount(probs.argmax(dim=-1), minlength=n_bins)
        return {
            **base,
            'mode': 'aggregate',
            'hist': [int(v) for v in hist],
            'entropy_mean': round(entropy, 4),
        }
    return {
        **base,
        'mode': 'dense',
        'probs': encode_u8(probs, scale=1.0),
        'entropy_mean': round(entropy, 4),
    }


def build_activations(name: str, act_mean: torch.Tensor, step: int) -> dict:
    """中間活性フレーム(forward hookのバッチ平均、[0,1])。"""
    act = act_mean.detach().to(torch.float32).cpu().clamp(0.0, 1.0)
    saturation = float(((act - 0.5).abs() > 0.45).float().mean().item())
    return {
        'type': 'activations',
        'layer': name,
        'step': step,
        'mean': encode_u8(act, scale=1.0),
        'saturation': round(saturation, 4),
    }


def build_ap_delta(name: str, delta: torch.Tensor, step: int) -> dict:
    """AP法更新量フレーム(型は確定、実配線は)。"""
    d = delta.detach().to(torch.float32).cpu()
    return {
        'type': 'ap_delta',
        'layer': name,
        'step': step,
        'delta_norm': round(float(d.norm().item()), 6),
        'delta': encode_u8(d.abs()),
    }
