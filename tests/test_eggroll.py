"""EGGROLL Optimizer の単体テスト。

論文 "Evolution Strategies at the Hyperscale" (Sarkar 2026) の主要要素が
実装通り動作することを検証する:
- counter-based RNG の決定論性
- low-rank 摂動の shape と分散
- Hardware-Efficient form の数値等価性
- 凸関数での収束（Algorithm 1 の動作確認）
- score function の挙動
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD  # noqa: E402

from griffith_logic.eggroll import (
    EGGROLLConfig,
    EGGROLLOptimizer,
    _make_seed,
)
from griffith_logic.eggroll_lut import (
    batched_lut_forward,
    batched_model_forward,
    hw_efficient_linear_batched,
    naive_linear_batched,
)
from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.lut import LutLayer

# ----------------------------------------------------------------------
# Counter-based RNG
# ----------------------------------------------------------------------

def test_counter_based_rng_determinism():
    """同じ (t, worker, layer, kind) で同じ A,B が再構築できる"""
    p = torch.zeros(50, 8)
    cfg = EGGROLLConfig(rank=2, n_workers=4, base_seed=123)
    opt1 = EGGROLLOptimizer([p.clone()], cfg)
    opt2 = EGGROLLOptimizer([p.clone()], cfg)
    A1, B1 = opt1._sample_AB(t=3, worker=2, layer_idx=0)
    A2, B2 = opt2._sample_AB(t=3, worker=2, layer_idx=0)
    assert torch.equal(A1, A2)
    assert torch.equal(B1, B2)


def test_distinct_seeds_distinct_perturbations():
    """異なる worker/timestep では異なる摂動が得られる"""
    p = torch.zeros(50, 8)
    cfg = EGGROLLConfig(rank=2, n_workers=4)
    opt = EGGROLLOptimizer([p], cfg)
    E1 = opt.perturbation(t=0, worker=0, layer_idx=0)
    E2 = opt.perturbation(t=0, worker=1, layer_idx=0)
    E3 = opt.perturbation(t=1, worker=0, layer_idx=0)
    assert not torch.allclose(E1, E2)
    assert not torch.allclose(E1, E3)


def test_seed_function_collision_avoidance():
    """seed が容易に衝突しないことを確認"""
    seeds = set()
    for t in range(10):
        for w in range(10):
            for l in range(3):
                for k in (0, 1):
                    seeds.add(_make_seed(42, t, w, l, k))
    assert len(seeds) == 10 * 10 * 3 * 2


# ----------------------------------------------------------------------
# Low-rank 摂動
# ----------------------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 4, 16])
def test_perturbation_shape(rank):
    """E = (1/√r) A B^T の shape が (m, n) になる"""
    m, n = 100, 30
    p = torch.zeros(m, n)
    cfg = EGGROLLConfig(rank=rank)
    opt = EGGROLLOptimizer([p], cfg)
    E = opt.perturbation(t=0, worker=0, layer_idx=0)
    assert E.shape == (m, n)


def test_perturbation_variance_bounded():
    """異なる rank で要素分散が約 1 に bounded（論文 (1/√r) スケールの効果）"""
    torch.manual_seed(0)
    m, n = 200, 100
    p = torch.zeros(m, n)
    for r in [1, 4, 16]:
        cfg = EGGROLLConfig(rank=r)
        opt = EGGROLLOptimizer([p], cfg)
        Es = torch.stack([
            opt.perturbation(t=0, worker=i, layer_idx=0)
            for i in range(50)
        ])
        # Var[E_ij] = (1/r) × Σ_k Var[A_ik B_jk] = (1/r) × r × 1 = 1
        var = Es.var().item()
        assert 0.7 < var < 1.3, f'rank={r}: variance={var}'


# ----------------------------------------------------------------------
# Hardware-Efficient form（論文 Section 4.2）
# ----------------------------------------------------------------------

def test_hw_efficient_form_equivalence():
    """論文の hardware-efficient form がナイーブ展開と数値一致"""
    torch.manual_seed(0)
    N, batch, m, n, r = 8, 32, 64, 16, 2
    sigma = 0.05
    inputs = torch.randn(batch, n)
    M = torch.randn(m, n)
    A = torch.randn(N, m, r)
    B = torch.randn(N, n, r)
    out_eff = hw_efficient_linear_batched(inputs, M, A, B, sigma, r)
    out_naive = naive_linear_batched(inputs, M, A, B, sigma, r)
    assert out_eff.shape == (N, batch, m)
    assert torch.allclose(out_eff, out_naive, atol=1e-5)


@pytest.mark.parametrize('rank', [1, 4, 8])
def test_hw_efficient_form_various_ranks(rank):
    torch.manual_seed(rank)
    N, batch, m, n = 4, 16, 32, 12
    sigma = 0.1
    inputs = torch.randn(batch, n)
    M = torch.randn(m, n)
    A = torch.randn(N, m, rank)
    B = torch.randn(N, n, rank)
    out_eff = hw_efficient_linear_batched(inputs, M, A, B, sigma, rank)
    out_naive = naive_linear_batched(inputs, M, A, B, sigma, rank)
    assert torch.allclose(out_eff, out_naive, atol=1e-5)


# ----------------------------------------------------------------------
# Optimizer の動作
# ----------------------------------------------------------------------

def test_zero_step_preserves_M():
    """σ=α=0 なら step 後も M が変わらない（中間状態が漏れない）"""
    torch.manual_seed(0)
    M = torch.randn(20, 10)
    M_before = M.clone()
    cfg = EGGROLLConfig(rank=1, sigma=0.0, alpha=0.0, n_workers=16)
    opt = EGGROLLOptimizer([M], cfg)
    opt.step(lambda: 0.0, t=0)
    assert torch.allclose(M, M_before, atol=1e-7)


def test_step_returns_expected_stats_keys():
    """step が想定どおりの統計キーを返す"""
    M = torch.randn(10, 5)
    cfg = EGGROLLConfig(rank=1, sigma=0.01, alpha=0.01, n_workers=8)
    opt = EGGROLLOptimizer([M], cfg)
    stats = opt.step(lambda: 0.5, t=0)
    for key in ('fitness/mean', 'fitness/max', 'fitness/min', 'fitness/std',
                'schedule/sigma', 'schedule/alpha', 'params/M_norm_0'):
        assert key in stats


def test_optimizer_quadratic_convergence():
    """凸関数 f(M) = -||M - M*||_F^2 で M が M* に向けて収束する"""
    torch.manual_seed(42)
    m, n = 20, 10
    M_target = torch.randn(m, n) * 0.5
    M = torch.zeros(m, n)
    cfg = EGGROLLConfig(
        rank=2, sigma=0.1, alpha=0.1, n_workers=64,
        base_seed=42, score_function='gaussian',
    )
    opt = EGGROLLOptimizer([M], cfg)

    def fitness():
        return -float(((M - M_target) ** 2).sum())

    initial_dist = float((M - M_target).norm())
    for t in range(200):
        opt.step(fitness, t)
    final_dist = float((M - M_target).norm())
    assert final_dist < initial_dist * 0.3, (
        f'凸関数で十分収束していない: {initial_dist:.4f} → {final_dist:.4f}'
    )


def test_optimizer_rejects_non_2d_params():
    """1D や 3D テンソルは弾く（DLGN の 2D matrix 専用）"""
    p_1d = torch.zeros(10)
    p_3d = torch.zeros(4, 5, 6)
    with pytest.raises(ValueError):
        EGGROLLOptimizer([p_1d], EGGROLLConfig())
    with pytest.raises(ValueError):
        EGGROLLOptimizer([p_3d], EGGROLLConfig())


# ----------------------------------------------------------------------
# Score function
# ----------------------------------------------------------------------

def test_centered_rank_score_function():
    """centered_rank が [-0.5, 0.5] を取り、極値の順序を保つ"""
    fitnesses = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
    p = torch.zeros(10, 5)
    cfg = EGGROLLConfig(score_function='centered_rank')
    opt = EGGROLLOptimizer([p], cfg)
    norm = opt._normalize_fitness(fitnesses)
    assert norm.min() >= -0.5 - 1e-6
    assert norm.max() <= 0.5 + 1e-6
    assert norm.argmax() == fitnesses.argmax()
    assert norm.argmin() == fitnesses.argmin()


def test_unknown_score_function_raises():
    p = torch.zeros(5, 3)
    cfg = EGGROLLConfig(score_function='unknown_xyz')
    opt = EGGROLLOptimizer([p], cfg)
    with pytest.raises(ValueError):
        opt._normalize_fitness(torch.tensor([1.0, 2.0]))


# ----------------------------------------------------------------------
# Multi-layer
# ----------------------------------------------------------------------

def test_antithetic_quadratic_convergence():
    """Antithetic sampling でも凸関数で収束する。
    1 step あたり 2N forward だが信号対雑音比が大幅改善する。
    """
    torch.manual_seed(42)
    m, n = 20, 10
    M_target = torch.randn(m, n) * 0.5
    M = torch.zeros(m, n)
    cfg = EGGROLLConfig(
        rank=2, sigma=0.1, alpha=0.1, n_workers=32,
        base_seed=42, antithetic=True,
    )
    opt = EGGROLLOptimizer([M], cfg)

    def fitness():
        return -float(((M - M_target) ** 2).sum())

    initial_dist = float((M - M_target).norm())
    for t in range(100):
        opt.step(fitness, t)
    final_dist = float((M - M_target).norm())
    # antithetic は通常版より少ないステップで収束すべき
    assert final_dist < initial_dist * 0.3, (
        f'antithetic で収束不十分: {initial_dist:.4f} → {final_dist:.4f}'
    )


def test_multi_layer_optimization():
    """複数の matrix を同時最適化できる（DLGN は複数 LutLayer を持つ）"""
    torch.manual_seed(7)
    M1 = torch.zeros(15, 8)
    M2 = torch.zeros(8, 5)
    T1 = torch.randn(15, 8) * 0.3
    T2 = torch.randn(8, 5) * 0.3
    cfg = EGGROLLConfig(rank=2, sigma=0.1, alpha=0.1, n_workers=64)
    opt = EGGROLLOptimizer([M1, M2], cfg)

    def fitness():
        d1 = ((M1 - T1) ** 2).sum()
        d2 = ((M2 - T2) ** 2).sum()
        return -float(d1 + d2)

    init_loss = -fitness()
    for t in range(150):
        opt.step(fitness, t)
    final_loss = -fitness()
    assert final_loss < init_loss * 0.4, (
        f'multi-layer で十分収束していない: {init_loss:.4f} → {final_loss:.4f}'
    )


# ----------------------------------------------------------------------
# batched=True（ワーカー次元のバッチ並列化）
# ----------------------------------------------------------------------
#
# batched=False（既定）が既存コードパスそのまま(bitwise不変)であることは
# 上の全テストがそのまま無変更でパスすることで担保する。以下は
# batched=True が (1) 汎用fitness_fnに対して直列版と等価な更新を生むこと、
# (2) LutLayerの実forward数理と一致するバッチ化forwardを提供すること、を
# それぞれ単体で検証したうえで、(3) 実際のLutLayer MLP全体を使った
# end-to-endの直列 vs バッチ 更新一致で締める。


def test_batched_optimizer_matches_serial_generic_fitness_single_layer():
    """batched=Trueの汎用fitness_fnがbatched=Falseと同一の1step更新を生む
    （単層、float64で浮動小数点再結合誤差を排除）。"""
    torch.manual_seed(0)
    m, n = 12, 7
    target = torch.randn(m, n, dtype=torch.float64) * 0.3
    M_serial = torch.zeros(m, n, dtype=torch.float64)
    M_batched = M_serial.clone()

    cfg = EGGROLLConfig(rank=2, sigma=0.1, alpha=0.1, n_workers=32, base_seed=123)
    opt_serial = EGGROLLOptimizer([M_serial], cfg, batched=False)
    opt_batched = EGGROLLOptimizer([M_batched], cfg, batched=True)

    def fitness_serial():
        return -float(((M_serial - target) ** 2).sum())

    def fitness_batched(perturbed):
        (Mw,) = perturbed  # (N, m, n)
        return -((Mw - target.unsqueeze(0)) ** 2).sum(dim=(1, 2))

    stats_s = opt_serial.step(fitness_serial, t=5)
    stats_b = opt_batched.step(fitness_batched, t=5)

    assert torch.allclose(M_serial, M_batched, atol=TOL_F64_EQ)
    for key in ('fitness/mean', 'fitness/max', 'fitness/min', 'fitness/std'):
        assert abs(stats_s[key] - stats_b[key]) < TOL_F64_EQ, key


def test_batched_optimizer_matches_serial_multi_layer_and_centered_rank():
    """複数layer + centered_rankスコアでも直列/バッチが一致する。"""
    torch.manual_seed(1)
    shapes = [(10, 6), (6, 4)]
    targets = [torch.randn(*s, dtype=torch.float64) * 0.2 for s in shapes]
    Ms_serial = [torch.zeros(*s, dtype=torch.float64) for s in shapes]
    Ms_batched = [M.clone() for M in Ms_serial]

    cfg = EGGROLLConfig(
        rank=3, sigma=0.05, alpha=0.2, n_workers=24, base_seed=7,
        score_function='centered_rank',
    )
    opt_serial = EGGROLLOptimizer(Ms_serial, cfg, batched=False)
    opt_batched = EGGROLLOptimizer(Ms_batched, cfg, batched=True)

    def fitness_serial():
        return -float(sum(
            ((M - T) ** 2).sum() for M, T in zip(Ms_serial, targets)
        ))

    def fitness_batched(perturbed):
        total = None
        for Mw, T in zip(perturbed, targets):
            d = ((Mw - T.unsqueeze(0)) ** 2).sum(dim=tuple(range(1, Mw.dim())))
            total = d if total is None else total + d
        return -total

    for t in range(3):
        stats_s = opt_serial.step(fitness_serial, t)
        stats_b = opt_batched.step(fitness_batched, t)
        for M_s, M_b in zip(Ms_serial, Ms_batched):
            assert torch.allclose(M_s, M_b, atol=TOL_F64_GRAD)
        assert abs(stats_s['fitness/mean'] - stats_b['fitness/mean']) < TOL_F64_GRAD


def test_batched_optimizer_matches_serial_antithetic():
    """antithetic=Trueでも直列/バッチが一致する(mirror samplingの2回呼び出し)。"""
    torch.manual_seed(2)
    m, n = 8, 5
    target = torch.randn(m, n, dtype=torch.float64) * 0.4
    M_serial = torch.zeros(m, n, dtype=torch.float64)
    M_batched = M_serial.clone()

    cfg = EGGROLLConfig(
        rank=1, sigma=0.1, alpha=0.1, n_workers=16, base_seed=99, antithetic=True,
    )
    opt_serial = EGGROLLOptimizer([M_serial], cfg, batched=False)
    opt_batched = EGGROLLOptimizer([M_batched], cfg, batched=True)

    def fitness_serial():
        return -float(((M_serial - target) ** 2).sum())

    def fitness_batched(perturbed):
        (Mw,) = perturbed
        return -((Mw - target.unsqueeze(0)) ** 2).sum(dim=(1, 2))

    opt_serial.step(fitness_serial, t=0)
    opt_batched.step(fitness_batched, t=0)
    assert torch.allclose(M_serial, M_batched, atol=TOL_F64_EQ)


def test_batched_step_rejects_wrong_fitness_shape():
    """バッチfitness_fnが(N,)以外を返したら明確に弾く。"""
    M = torch.zeros(5, 3)
    cfg = EGGROLLConfig(rank=1, n_workers=8)
    opt = EGGROLLOptimizer([M], cfg, batched=True)
    with pytest.raises(ValueError):
        opt.step(lambda perturbed: torch.zeros(3), t=0)


# ----------------------------------------------------------------------
# batched_lut_forward / batched_model_forward の数理一致
# ----------------------------------------------------------------------

def test_batched_lut_forward_matches_16gate_softmax_loop():
    """16ゲートsoftmax(n_inputs=2)で、batched_lut_forwardが
    per-worker LutLayer.forward呼び出しループと一致する。"""
    torch.manual_seed(3)
    in_dim, out_dim, N, batch = 6, 5, 4, 8
    layer = LutLayer(in_dim=in_dim, out_dim=out_dim, n_inputs=2,
                      parameterization='softmax')
    x = torch.bernoulli(torch.full((batch, in_dim), 0.5))
    weights_stack = torch.randn(N, out_dim, 16)

    with torch.no_grad():
        batched_out = batched_lut_forward(
            x, weights_stack, layer.indices, layer.n_inputs, layer.gate_table,
        )
        assert batched_out.shape == (N, batch, out_dim)

        serial_outs = []
        original = layer.weights.detach().clone()
        layer.train()
        for i in range(N):
            layer.weights.copy_(weights_stack[i])
            serial_outs.append(layer(x).clone())
        layer.weights.copy_(original)
    serial_out = torch.stack(serial_outs, dim=0)
    assert torch.allclose(batched_out, serial_out, **GRADCHECK_KW)


def _build_small_softmax_mlp():
    in_dim, k, num_classes = 6, 8, 2
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(in_dim=in_dim, out_dim=k, n_inputs=2, parameterization='softmax'),
        LutLayer(in_dim=k, out_dim=k, n_inputs=2, parameterization='softmax'),
        GroupSum(num_classes),
    )
    return model, in_dim, num_classes


def test_batched_model_forward_matches_serial_snapshot_loop():
    """2層LutLayer(16ゲートsoftmax) + GroupSum のMLPで、
    batched_model_forwardが per-worker forwardループと一致する
    （worker毎に入力が異なる第2層以降の畳み込みも含めて検証）。"""
    torch.manual_seed(4)
    model, in_dim, _num_classes = _build_small_softmax_mlp()
    lut_layers = [m for m in model.modules() if isinstance(m, LutLayer)]
    batch, N = 5, 6
    x = torch.bernoulli(torch.full((batch, in_dim), 0.5))

    stacks = [
        layer.weights.detach().unsqueeze(0)
        + 0.3 * torch.randn(N, *layer.weights.shape)
        for layer in lut_layers
    ]
    weights_stacks = dict(zip(lut_layers, stacks))

    model.train()
    with torch.no_grad():
        batched_out = batched_model_forward(model, x, weights_stacks)

        originals = [layer.weights.detach().clone() for layer in lut_layers]
        serial_outs = []
        for i in range(N):
            for layer, stack in zip(lut_layers, stacks):
                layer.weights.copy_(stack[i])
            serial_outs.append(model(x).clone())
        for layer, orig in zip(lut_layers, originals):
            layer.weights.copy_(orig)
    serial_out = torch.stack(serial_outs, dim=0)
    assert torch.allclose(batched_out, serial_out, **GRADCHECK_KW)


def _serial_worker_loop(model, x, lut_layers, stacks):
    """worker毎に重みを書き換えて通常のforwardを回す(直列版の参照)。"""
    originals = [layer.weights.detach().clone() for layer in lut_layers]
    outs = []
    with torch.no_grad():
        for i in range(stacks[0].shape[0]):
            for layer, stack in zip(lut_layers, stacks):
                layer.weights.copy_(stack[i])
            outs.append(model(x).clone())
        for layer, orig in zip(lut_layers, originals):
            layer.weights.copy_(orig)
    return torch.stack(outs, dim=0)


def _build_masked_mlp(**lut_kwargs):
    in_dim, k = 6, 8
    model = torch.nn.Sequential(
        LutLayer(in_dim=in_dim, out_dim=k, n_inputs=2, **lut_kwargs),
        LutLayer(in_dim=k, out_dim=k, n_inputs=2, **lut_kwargs),
        GroupSum(2),
    )
    return model, in_dim


@pytest.mark.parametrize('training', [True, False])
def test_batched_model_forward_respects_gate_mask(training):
    """gate_mask(除外ゲート)をLutLayerと同じく適用する(train=soft / eval=hard)。"""
    torch.manual_seed(6)
    model, in_dim = _build_masked_mlp(gate_mask=[0, 3, 5, 10, 15])
    lut_layers = [m for m in model if isinstance(m, LutLayer)]
    x = torch.bernoulli(torch.full((5, in_dim), 0.5))
    N = 4
    # 除外ゲートのロジットを大きくして、マスクを無視すると結果が変わるようにする
    stacks = []
    for layer in lut_layers:
        st = layer.weights.detach().unsqueeze(0) + 0.3 * torch.randn(N, *layer.weights.shape)
        st[..., [0, 3, 5, 10, 15]] += 5.0
        stacks.append(st)
    model.train(training)
    with torch.no_grad():
        batched_out = batched_model_forward(model, x, dict(zip(lut_layers, stacks)))
    serial_out = _serial_worker_loop(model, x, lut_layers, stacks)
    assert torch.allclose(batched_out, serial_out, **GRADCHECK_KW)


@pytest.mark.parametrize('n_inputs', [2, 4])
def test_batched_model_forward_eval_uses_hard_evaluation(n_inputs):
    """評価モードではLutLayerの推論時と同じhard評価になる(soft評価ではない)。"""
    torch.manual_seed(7)
    in_dim, k = 8, 8
    model = torch.nn.Sequential(
        LutLayer(in_dim=in_dim, out_dim=k, n_inputs=n_inputs, parameterization='softmax'),
        LutLayer(in_dim=k, out_dim=k, n_inputs=n_inputs, parameterization='softmax'),
        GroupSum(2),
    )
    lut_layers = [m for m in model if isinstance(m, LutLayer)]
    x = torch.bernoulli(torch.full((5, in_dim), 0.5))
    stacks = [
        layer.weights.detach().unsqueeze(0) + 2.0 * torch.randn(3, *layer.weights.shape)
        for layer in lut_layers
    ]
    model.eval()
    with torch.no_grad():
        batched_out = batched_model_forward(model, x, dict(zip(lut_layers, stacks)))
    serial_out = _serial_worker_loop(model, x, lut_layers, stacks)
    assert torch.equal(batched_out, serial_out)
    # 同じ重みでも訓練モード(soft)とは値が異なる(=hard評価が効いている)
    model.train()
    with torch.no_grad():
        soft_out = batched_model_forward(model, x, dict(zip(lut_layers, stacks)))
    assert not torch.allclose(soft_out, batched_out)


def test_batched_model_forward_applies_gumbel_noise():
    """gumbel_noise=True の訓練モードでは LutLayer と同じ式でノイズを注入する。

    単層なら worker 毎のノイズが直列ループと同じ順に乱数を消費するので、
    同じseedで直列forwardのループと一致する。評価モードではノイズは入らない。
    """
    torch.manual_seed(8)
    in_dim, k, N = 6, 8, 5
    layer = LutLayer(in_dim=in_dim, out_dim=k, n_inputs=2, gumbel_noise=True,
                     gumbel_tau=0.7)
    model = torch.nn.Sequential(layer, GroupSum(2))
    x = torch.bernoulli(torch.full((4, in_dim), 0.5))
    stacks = [layer.weights.detach().unsqueeze(0) + 0.3 * torch.randn(N, *layer.weights.shape)]

    model.train()
    torch.manual_seed(100)
    with torch.no_grad():
        batched_out = batched_model_forward(model, x, {layer: stacks[0]})
    torch.manual_seed(100)
    serial_out = _serial_worker_loop(model, x, [layer], stacks)
    assert torch.allclose(batched_out, serial_out, **GRADCHECK_KW)

    # ノイズが実際に入っている(seedを変えると結果が変わる)
    torch.manual_seed(101)
    with torch.no_grad():
        other = batched_model_forward(model, x, {layer: stacks[0]})
    assert not torch.allclose(other, batched_out)

    # 評価モードではノイズなし(seedに依らない)
    model.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        e1 = batched_model_forward(model, x, {layer: stacks[0]})
    torch.manual_seed(2)
    with torch.no_grad():
        e2 = batched_model_forward(model, x, {layer: stacks[0]})
    assert torch.equal(e1, e2)


def test_batched_model_forward_rejects_hard_selection_in_training():
    """訓練モードの selection='hard_st' は未対応として明確に弾く(評価モードは可)。"""
    layer = LutLayer(in_dim=4, out_dim=3, n_inputs=2, selection='hard_st')
    model = torch.nn.Sequential(layer)
    stack = layer.weights.detach().unsqueeze(0).repeat(2, 1, 1)
    x = torch.bernoulli(torch.full((3, 4), 0.5))
    model.train()
    with pytest.raises(NotImplementedError):
        batched_model_forward(model, x, {layer: stack})
    model.eval()
    with torch.no_grad():
        assert batched_model_forward(model, x, {layer: stack}).shape == (2, 3, 3)


def test_batched_model_forward_rejects_unsupported_parameterization():
    """softmax/soft/fixed以外のLutLayerは明確にNotImplementedErrorで弾く。"""
    layer = LutLayer(in_dim=4, out_dim=3, n_inputs=2, parameterization='independent')
    model = torch.nn.Sequential(layer)
    stack = layer.weights.detach().unsqueeze(0).repeat(2, 1, 1)
    x = torch.bernoulli(torch.full((3, 4), 0.5))
    with pytest.raises(NotImplementedError):
        batched_model_forward(model, x, {layer: stack})


# ----------------------------------------------------------------------
# end-to-end: EGGROLLOptimizer(batched=True) + 実LutLayer MLP
# ----------------------------------------------------------------------

def test_eggroll_batched_end_to_end_matches_serial_on_lut_mlp():
    """LutLayer MLP + cross entropy のEGGROLL学習で、
    batched=Trueが直列版と数値的に等価な1step更新を生む。

    同一seedで初期化した2つのモデルに対し、直列版はEGGROLLOptimizer(既定)
    +model.forward()をそのまま使い、バッチ版はEGGROLLOptimizer(batched=True)
    +batched_model_forwardを使う。両者は独立な計算経路(直列forwardの
    ループ vs バッチ化einsum)だが、同一の摂動集合・同一のスコア正規化
    からは同一の更新が導かれるべき、という本タスクの数値等価性契約を
    検証する。
    """
    def build():
        torch.manual_seed(42)
        return _build_small_softmax_mlp()

    model_serial, in_dim, num_classes = build()
    model_batched, _, _ = build()
    torch.manual_seed(123)
    batch = 16
    x = torch.bernoulli(torch.full((batch, in_dim), 0.5))
    y = torch.randint(0, num_classes, (batch,))

    lut_serial = [m for m in model_serial.modules() if isinstance(m, LutLayer)]
    lut_batched = [m for m in model_batched.modules() if isinstance(m, LutLayer)]
    for a, b in zip(lut_serial, lut_batched):
        assert torch.equal(a.weights, b.weights), '同一seed初期化のはずが不一致'
        a.weights.requires_grad_(False)
        b.weights.requires_grad_(False)

    cfg = EGGROLLConfig(rank=2, sigma=0.05, alpha=0.05, n_workers=12, base_seed=17)
    opt_serial = EGGROLLOptimizer([m.weights for m in lut_serial], cfg, batched=False)
    opt_batched = EGGROLLOptimizer([m.weights for m in lut_batched], cfg, batched=True)

    model_serial.train()
    model_batched.train()

    def fitness_serial():
        with torch.no_grad():
            out = model_serial(x)
            return -float(F.cross_entropy(out, y))

    def fitness_batched(perturbed):
        weights_stacks = dict(zip(lut_batched, perturbed))
        with torch.no_grad():
            out = batched_model_forward(model_batched, x, weights_stacks)
            N, B, C = out.shape
            logp = F.log_softmax(out, dim=-1)
            nll = -logp.gather(
                -1, y.view(1, B, 1).expand(N, B, 1)
            ).squeeze(-1)
            return -nll.mean(dim=1)

    stats_s = opt_serial.step(fitness_serial, t=0)
    stats_b = opt_batched.step(fitness_batched, t=0)

    for layer_s, layer_b in zip(lut_serial, lut_batched):
        assert torch.allclose(layer_s.weights, layer_b.weights, **GRADCHECK_KW), (
            'batched=Trueとbatched=Falseで更新後weightsが不一致'
        )
    assert abs(stats_s['fitness/mean'] - stats_b['fitness/mean']) < (
        GRADCHECK_KW['atol'] + GRADCHECK_KW['rtol'] * abs(stats_s['fitness/mean'])
    )
