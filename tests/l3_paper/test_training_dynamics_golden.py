"""学習動態のゴールデン回帰。

「最終精度は同じだが学習動態が変わった」変更(勾配スケール・更新則の退行)を、
固定seedの小規模学習の中間統計をゴールデンJSONと帯域照合して検出する。
AP法scatterバグ(発覚)級の退行はこの種のテストで即検出できる。

ゴールデン更新の手順(勝手な緩和禁止):
    MD_UPDATE_GOLDEN=1 uv run pytest tests/l3_paper/test_training_dynamics_golden.py
を実行し、更新理由を記録した上でレビューを受けること。

帯域について: 完全一致はBLAS実装差(プラットフォーム間)で壊れるため、
構造チェック(形状・ゲート数)は厳密、統計値は相対帯域で照合する。
"""

import json
import os
from pathlib import Path

import pytest
import torch

from griffith_logic import GroupSum, LutLayer

GOLDEN_PATH = Path(__file__).parent / 'golden' / 'tiny_dlgn_dynamics.json'
SEED = 20260702
STEPS = 60
# 統計値の許容相対帯域(プラットフォーム間のBLAS差を吸収しつつ動態変化を検出)
REL_BAND = 0.25
# ゲートヒストグラムのL1距離許容(総ニューロン40に対して)
HIST_L1_MAX = 10


def _run_tiny_training() -> dict:
    """固定seedの小規模学習を実行し、中間統計を収集する。"""
    torch.manual_seed(SEED)
    layer1 = LutLayer(32, 40, n_inputs=2, connections='random')
    layer2 = LutLayer(40, 20, n_inputs=2, connections='random')
    model = torch.nn.Sequential(layer1, layer2, GroupSum(k=10, tau=1.0))

    gen = torch.Generator().manual_seed(SEED)
    x = torch.rand(256, 32, generator=gen)
    y = torch.randint(0, 10, (256,), generator=gen)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = torch.nn.CrossEntropyLoss()

    grad_norms_l1: list[float] = []
    grad_norms_l2: list[float] = []
    losses: list[float] = []
    for i in range(STEPS):
        lo = (i * 32) % 256
        xb, yb = x[lo:lo + 32], y[lo:lo + 32]
        out = model(xb)
        loss = loss_fn(out, yb)
        optimizer.zero_grad()
        loss.backward()
        grad_norms_l1.append(float(layer1.weights.grad.norm()))
        grad_norms_l2.append(float(layer2.weights.grad.norm()))
        optimizer.step()
        losses.append(float(loss))

    with torch.no_grad():
        gate_hist = torch.bincount(
            layer1.weights.argmax(dim=-1), minlength=16
        ).tolist()

    def stats(xs: list[float]) -> dict:
        t = torch.tensor(xs)
        return {'mean': float(t.mean()), 'std': float(t.std()),
                'first': xs[0], 'last': xs[-1]}

    return {
        'seed': SEED,
        'steps': STEPS,
        'loss': stats(losses),
        'grad_norm_layer1': stats(grad_norms_l1),
        'grad_norm_layer2': stats(grad_norms_l2),
        'gate_hist_layer1': gate_hist,
        'grad_ratio_l2_over_l1': (
            float(torch.tensor(grad_norms_l2).mean())
            / max(float(torch.tensor(grad_norms_l1).mean()), 1e-12)
        ),
    }


def _assert_band(actual: float, golden: float, name: str):
    scale = max(abs(golden), 1e-9)
    assert abs(actual - golden) <= REL_BAND * scale, (
        f'{name} が帯域外: actual={actual:.6g}, golden={golden:.6g} '
        f'(許容±{REL_BAND:.0%})'
    )


class TestTrainingDynamicsGolden:
    def test_dynamics_match_golden(self):
        record = _run_tiny_training()

        if os.environ.get('MD_UPDATE_GOLDEN') == '1':
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(record, indent=1))
            pytest.skip(f'ゴールデンを更新した: {GOLDEN_PATH}')

        assert GOLDEN_PATH.exists(), (
            'ゴールデン未生成。MD_UPDATE_GOLDEN=1 で生成し、コミットすること'
        )
        golden = json.loads(GOLDEN_PATH.read_text())
        assert record['seed'] == golden['seed']
        assert record['steps'] == golden['steps']

        for key in ('loss', 'grad_norm_layer1', 'grad_norm_layer2'):
            for stat in ('mean', 'std', 'first', 'last'):
                _assert_band(record[key][stat], golden[key][stat], f'{key}.{stat}')

        # 層間の勾配比(勾配消失/爆発の退行検出)
        _assert_band(
            record['grad_ratio_l2_over_l1'], golden['grad_ratio_l2_over_l1'],
            'grad_ratio_l2_over_l1',
        )

        # ゲート種分布(学習の到達点の質)はL1距離で
        l1 = sum(
            abs(a - b)
            for a, b in zip(record['gate_hist_layer1'], golden['gate_hist_layer1'])
        )
        assert l1 <= HIST_L1_MAX, (
            f'ゲート分布のL1距離が大きすぎる: {l1} > {HIST_L1_MAX}\n'
            f'actual={record["gate_hist_layer1"]}\n'
            f'golden={golden["gate_hist_layer1"]}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
