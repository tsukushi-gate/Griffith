"""EGGROLL: Evolution Strategies at the Hyperscale (Sarkar et al., 2026) の Optimizer 実装。

論文の Algorithm 1 を忠実に再現する。
- low-rank 摂動 E = (1/√r) A B^T  （A∈R^{m×r}, B∈R^{n×r}, i.i.d. 標準正規）
- counter-based RNG で摂動を seed から決定的に再構築（共有は scalar fitness のみ）
- Gaussian score function ŝ(E) = -E（論文 Eq.5）
- 集約更新 M ← M + (α/N) Σ_i E_i f_i（論文 Eq.6, maximize）

backprop を使わず、forward のみで 2D matrix パラメータを最適化する。
DLGN（LutLayer の logits）の非微分的な学習を主用途とする。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


@dataclass
class EGGROLLConfig:
    """EGGROLL のハイパーパラメータ"""
    rank: int = 1
    sigma: float = 0.01
    alpha: float = 0.01
    n_workers: int = 256
    base_seed: int = 42
    # "gaussian": ŝ(E)=-E（Eq.5、論文の主要設定）
    # "centered_rank": Salimans 2017 の utility transform（fine-tuning に有効）
    score_function: str = "gaussian"
    # mirror sampling（Salimans 2017）: 各 worker で +E と -E の両方を評価し
    # (f(+E)-f(-E))/2 を信号にする。1 step あたりの forward 評価が 2 倍になる代わりに
    # 有限サンプル分散が大幅に低減し、softmax 非線形を経由する DLGN logits 等
    # 信号対雑音比が小さい設定で必須。論文 Algorithm 1 には明示されていないが
    # 実用 ES の事実上の標準。
    antithetic: bool = False


# counter-based seed 用の素数（衝突回避）
_PRIME_T = 2654435761
_PRIME_W = 1597334677
_PRIME_L = 805459861
_PRIME_K = 100003


def _make_seed(base_seed: int, t: int, worker: int, layer: int, kind: int) -> int:
    """論文の "known random seeds ς + counter" を 1 つの seed に hash する。

    kind: 0=A行列, 1=B行列。
    出力は torch.Generator が許容する範囲（int32 max）に収める。
    """
    h = base_seed & 0xFFFFFFFF
    h ^= (t * _PRIME_T) & 0xFFFFFFFF
    h ^= (worker * _PRIME_W) & 0xFFFFFFFF
    h ^= (layer * _PRIME_L) & 0xFFFFFFFF
    h ^= (kind * _PRIME_K) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


class EGGROLLOptimizer:
    """論文 Algorithm 1 を実装。各 2D matrix パラメータ M_t を in-place で更新する。

    Args:
        params: 訓練対象の 2D テンソル（LutLayer.weights など）
        cfg:    ハイパーパラメータ

    Note:
        params 自体が M_t（mean）として保持される。step() 内で各 worker の評価のために
        一時的に M を上書きするが、step 終了時には更新後の M に戻る。
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        cfg: EGGROLLConfig,
        batched: bool = False,
    ):
        """
        Args:
            params: 訓練対象の2D matrix群
            cfg: ハイパーパラメータ
            batched: True の場合、N_workers 回の直列forward評価の代わりに
                1回のバッチ化forwardでN worker分の摂動評価をまとめて行う
                。既定Falseは従来の直列step()と完全に同一の
                コードパスを通り、bitwise不変（後方互換の最重要契約）。
                batched=True 時は step() に渡す fitness_fn のシグネチャが
                変わる: `fitness_fn(perturbed_params) -> Tensor[N]` を
                要求する（perturbed_params は self.params と同じ順序の
                リストで、各要素は (N, *param.shape) の worker毎の
                perturbed weights）。同一seed・同一score正規化であれば、
                直列版と数値的に等価な更新になる
                （tests/test_eggroll.py::test_batched_optimizer_matches_serial* 参照）。
                perturbed weights は worker 次元ごと実体化するため、
                評価中は各paramについて N 倍のメモリを使う(直列版は1倍)。
        """
        self.params: list[torch.Tensor] = list(params)
        if not self.params:
            raise ValueError("EGGROLL: params が空")
        for p in self.params:
            if p.dim() != 2:
                raise ValueError(
                    f"EGGROLL は 2D matrix のみ対応: shape={tuple(p.shape)}"
                )
        self.cfg = cfg
        self.batched = batched
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype

    # ------------------------------------------------------------------
    # 摂動生成（counter-based RNG）
    # ------------------------------------------------------------------

    def _sample_AB(
        self, t: int, worker: int, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """worker i, layer_idx に対する A, B を seed 決定的にサンプル。

        A ∈ R^{m×r}, B ∈ R^{n×r}, 各要素 i.i.d. N(0, 1)（Assumption 1）。
        """
        m, n = self.params[layer_idx].shape
        r = self.cfg.rank
        seed_a = _make_seed(self.cfg.base_seed, t, worker, layer_idx, 0)
        seed_b = _make_seed(self.cfg.base_seed, t, worker, layer_idx, 1)
        g_a = torch.Generator(device=self.device).manual_seed(seed_a)
        g_b = torch.Generator(device=self.device).manual_seed(seed_b)
        A = torch.randn(m, r, generator=g_a, device=self.device, dtype=self.dtype)
        B = torch.randn(n, r, generator=g_b, device=self.device, dtype=self.dtype)
        return A, B

    def perturbation(self, t: int, worker: int, layer_idx: int) -> torch.Tensor:
        """E = (1/√r) A B^T （rank-r low-rank 摂動）"""
        A, B = self._sample_AB(t, worker, layer_idx)
        return (A @ B.T) / (self.cfg.rank ** 0.5)

    def _sample_AB_stack(
        self, t: int, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """全 worker 分の (A_i, B_i) を stack して再構築（batched評価用）。

        counter-based RNGはseedからの決定的再構築であり、worker間で
        共有ストリームを消費しないため、直列版のPhase 2ループ
        （`for i in range(N): A_stack[i], B_stack[i] = self._sample_AB(...)`）
        と全く同じ値を返す（呼び出し回数・順序に依存しない）。
        """
        m, n = self.params[layer_idx].shape
        r = self.cfg.rank
        N = self.cfg.n_workers
        A_stack = torch.empty(N, m, r, device=self.device, dtype=self.dtype)
        B_stack = torch.empty(N, n, r, device=self.device, dtype=self.dtype)
        for i in range(N):
            A_stack[i], B_stack[i] = self._sample_AB(t, i, layer_idx)
        return A_stack, B_stack

    # ------------------------------------------------------------------
    # スコア関数
    # ------------------------------------------------------------------

    def _normalize_fitness(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """論文の score function に応じた変換。

        - gaussian: ŝ(E)=-E（論文 Eq.5）。fitness の mean を引く（理論的に等価
          で有限サンプル分散を抑制。論文 Section 6.3 の RWKV 実験では z-score
          正規化を採用している）
        - centered_rank: Salimans 2017 の utility transform（[-0.5, 0.5]）
        """
        sf = self.cfg.score_function
        if sf == "gaussian":
            return fitnesses - fitnesses.mean()
        if sf == "centered_rank":
            order = fitnesses.argsort()
            ranks = torch.zeros_like(fitnesses)
            ranks[order] = torch.arange(
                len(fitnesses), device=fitnesses.device, dtype=fitnesses.dtype,
            )
            denom = max(len(fitnesses) - 1, 1)
            return ranks / denom - 0.5
        raise ValueError(f"unknown score_function: {sf}")

    # ------------------------------------------------------------------
    # 1 step 実行
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, fitness_fn, t: int) -> dict:
        """論文 Algorithm 1 の 1 timestep を実行。

        `self.batched` に応じて直列版(`_step_serial`)/バッチ版
        (`_step_batched`)へディスパッチする。`batched=False`(既定)の
        パスは従来コードそのままでbitwise不変。

        Args:
            fitness_fn: batched=False の場合は
                `Callable[[], float]`（self.params の現在値でfitnessを
                返す。loss最小化なら-lossを返す）。batched=True の場合は
                `Callable[[list[torch.Tensor]], torch.Tensor]`
                （perturbed_params → shape (N,) のfitnessテンソル。
                __init__のdocstring参照）。
            t: timestep（counter-based RNG用）

        Returns:
            学習統計の辞書
        """
        if self.batched:
            return self._step_batched(fitness_fn, t)
        return self._step_serial(fitness_fn, t)

    def _step_serial(self, fitness_fn: Callable[[], float], t: int) -> dict:
        """従来の直列実装(N_workers回forwardを1つずつ評価)。batched=False の既定経路。"""
        N = self.cfg.n_workers
        sigma = self.cfg.sigma

        # Phase 1: M を保存し、各 worker で fitness を評価
        original_M = [M.clone() for M in self.params]
        fitnesses = torch.zeros(N, device=self.device, dtype=self.dtype)

        if self.cfg.antithetic:
            # Mirror sampling: 各 worker で +E_i と -E_i の両方を評価し
            # f_eff = (f(M+σE) - f(M-σE)) / 2 を信号にする（Salimans 2017）
            for i in range(N):
                # +E_i 評価
                for layer_idx, M in enumerate(self.params):
                    E = self.perturbation(t, i, layer_idx)
                    M.copy_(original_M[layer_idx]).add_(E, alpha=sigma)
                f_plus = float(fitness_fn())
                # -E_i 評価
                for layer_idx, M in enumerate(self.params):
                    E = self.perturbation(t, i, layer_idx)
                    M.copy_(original_M[layer_idx]).add_(E, alpha=-sigma)
                f_minus = float(fitness_fn())
                fitnesses[i] = (f_plus - f_minus) / 2.0
        else:
            for i in range(N):
                for layer_idx, M in enumerate(self.params):
                    E = self.perturbation(t, i, layer_idx)
                    M.copy_(original_M[layer_idx]).add_(E, alpha=sigma)
                fitnesses[i] = float(fitness_fn())

        # M を保存値に戻す（更新前の M_t）
        for layer_idx, M in enumerate(self.params):
            M.copy_(original_M[layer_idx])

        # Phase 2: 集約更新 ΔM = (α/(N√r)) Σ_i f_i A_i B_i^T
        f_norm = self._normalize_fitness(fitnesses)
        scale = self.cfg.alpha / (N * (self.cfg.rank ** 0.5))

        for layer_idx, M in enumerate(self.params):
            m, n = M.shape
            r = self.cfg.rank
            # 全 worker の (A_i, B_i) を seed から再構築（論文 Algorithm 1 Phase 2）
            A_stack = torch.empty(N, m, r, device=self.device, dtype=self.dtype)
            B_stack = torch.empty(N, n, r, device=self.device, dtype=self.dtype)
            for i in range(N):
                A_stack[i], B_stack[i] = self._sample_AB(t, i, layer_idx)
            # ΔM[m, n] = scale × Σ_i f_i × Σ_k A_stack[i,m,k] × B_stack[i,n,k]
            # 個々の E_i を materialize しない最適化（論文 Section 4.2 末尾）
            delta = torch.einsum('imr,inr,i->mn', A_stack, B_stack, f_norm)
            M.add_(delta, alpha=scale)

        stats = {
            "fitness/mean": float(fitnesses.mean()),
            "fitness/max": float(fitnesses.max()),
            "fitness/min": float(fitnesses.min()),
            "fitness/std": float(fitnesses.std()),
            "schedule/sigma": float(sigma),
            "schedule/alpha": float(self.cfg.alpha),
        }
        for i, M in enumerate(self.params):
            stats[f"params/M_norm_{i}"] = float(M.norm())
        return stats

    def _step_batched(self, fitness_fn, t: int) -> dict:
        """N_workers 分の forward を1回のバッチ呼び出しにまとめる。

        直列版との違いは Phase 1 の評価方法のみ:
        - 直列: M を1 workerずつ上書きし、fitness_fn()をN回呼ぶ
        - バッチ: 全workerの perturbed weights (N,*shape) を stack し、
          fitness_fn(perturbed_params)を1回（antithetic時は2回）呼ぶ

        Phase 2（集約更新）は直列版と同一の数式・同一のseed再構築を使う
        （`_sample_AB_stack`は`_sample_AB`のループ版と同じ値を返す）ため、
        fitness_fn が同じ入力に対し同じ値を返す限り数値的に等価な更新になる。

        メモリ: worker毎の perturbed weights (N, m, n) は fitness_fn に渡すために
        実体化する(低ランクのままでは渡せない)。摂動 E_i = A_i B_i^T/√r は
        各符号の評価時にその場で作り、評価をまたいで保持しない。
        """
        N = self.cfg.n_workers
        sigma = self.cfg.sigma
        r = self.cfg.rank

        A_stacks: list[torch.Tensor] = []
        B_stacks: list[torch.Tensor] = []
        for layer_idx in range(len(self.params)):
            A_stack, B_stack = self._sample_AB_stack(t, layer_idx)
            A_stacks.append(A_stack)
            B_stacks.append(B_stack)

        def perturbed(sign: float) -> list[torch.Tensor]:
            out = []
            for M, A, B in zip(self.params, A_stacks, B_stacks, strict=True):
                E = torch.einsum('imr,inr->imn', A, B) / (r ** 0.5)
                out.append(M.unsqueeze(0) + (sign * sigma) * E)
            return out

        if self.cfg.antithetic:
            f_plus = fitness_fn(perturbed(1.0))
            f_minus = fitness_fn(perturbed(-1.0))
            fitnesses = (
                f_plus.to(device=self.device, dtype=self.dtype)
                - f_minus.to(device=self.device, dtype=self.dtype)
            ) / 2.0
        else:
            fitnesses = fitness_fn(perturbed(1.0)).to(
                device=self.device, dtype=self.dtype
            )

        if fitnesses.shape != (N,):
            raise ValueError(
                f'batched fitness_fnはshape ({N},)を返す必要がある: '
                f'{tuple(fitnesses.shape)}'
            )

        # Phase 2: 集約更新（直列版と同一の数式）
        f_norm = self._normalize_fitness(fitnesses)
        scale = self.cfg.alpha / (N * (r ** 0.5))
        for layer_idx, M in enumerate(self.params):
            delta = torch.einsum(
                'imr,inr,i->mn', A_stacks[layer_idx], B_stacks[layer_idx], f_norm,
            )
            M.add_(delta, alpha=scale)

        stats = {
            "fitness/mean": float(fitnesses.mean()),
            "fitness/max": float(fitnesses.max()),
            "fitness/min": float(fitnesses.min()),
            "fitness/std": float(fitnesses.std()),
            "schedule/sigma": float(sigma),
            "schedule/alpha": float(self.cfg.alpha),
        }
        for i, M in enumerate(self.params):
            stats[f"params/M_norm_{i}"] = float(M.norm())
        return stats
