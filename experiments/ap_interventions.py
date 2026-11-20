"""AP法(誤差ブロードキャスト型学習則)の介入実験用トレーナー変種。

仮説:
    「AP法の失敗原因はネットワークの単調性の破れ」。
    GEVB(Clark et al. 2021)は「重み非負(=単調増加)」ネットワークで
    誤差ブロードキャスト更新の符号が勾配と一致することを証明した。
    LUTは反転関数を学習でき単調性が破れうるため、単調性を強制すれば
    AP法が改善するはず、という因果仮説を検証する。

本モジュールは griffith_logic/ のコア実装を一切変更せず、
既存の APTrainer / CNNAPTrainer を継承したアダプタとして実装する。

提供する介入:
    - MonotoneAPTrainer     : 各更新後に重みを単調増加錐へ射影(単調制約AP)
    - DFAliteAPTrainer      : 層固定ランダム±1符号でブロードキャスト誤差を歪める対照
    - MonotoneCNNAPTrainer  : CDD-LGN(Conv+Dense)版の単調制約AP
    - isotonic_project_lut_ : 単調増加錐への近似射影(in-place)

ビット規約は既存と共通: e = Σ_j x_j·2^j(入力0=LSB)。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_trainer import APTrainer, LayerCache
from ap_trainer_cnn import CNNAPTrainer
from monotonicity import collect_lut_conv_layers, collect_lut_layers


def isotonic_project_lut_(w: torch.Tensor, n_inputs: int, iters: int = 3) -> None:
    """LUT重みを単調増加錐へ近似射影する(in-place)。

    各入力jについて、bit_j=0のエントリ集合e0とe1=e0|(1<<j)のペアを取り、
    単調増加違反(w[:,e0] > w[:,e1])の箇所のみ両者を平均に置換する。
    これを iters 回スイープする(pool-adjacent-violators を各入力軸で反復)。

    Args:
        w: (N, 2^n) の重みテンソル(N=ニューロン数 or ゲート数)。in-placeで更新。
        n_inputs: LUTの入力数(2^n = w.shape[1])。
        iters: スイープ回数(既定3)。[0,1]クランプは呼び出し側で行う。
    """
    n_entries = w.shape[1]
    e_all = torch.arange(n_entries, device=w.device)
    for _ in range(iters):
        for j in range(n_inputs):
            bit = 1 << j
            e0 = e_all[(e_all & bit) == 0]  # bit_j==0 のエントリ (2^(n-1),)
            e1 = e0 | bit                    # 対応する bit_j==1 のエントリ
            w0 = w[:, e0]
            w1 = w[:, e1]
            mean = (w0 + w1) / 2.0
            # 単調増加違反(小さいべき側 e0 の方が大きい)の箇所のみ平均化
            violated = w0 > w1
            w[:, e0] = torch.where(violated, mean, w0)
            w[:, e1] = torch.where(violated, mean, w1)


class MonotoneAPTrainer(APTrainer):
    """単調制約AP: 各更新後に全LUT重みを単調増加錐へ射影する。

    GEVB条件(重み非負=単調増加)を各ステップで強制し、単調性が
    AP法の成功条件かを因果検証する。
    """

    def __init__(self, *args, proj_iters: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_iters = proj_iters

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        m = super().train_step(x, target)
        with torch.no_grad():
            for layer in self.lut_layers:
                w = layer.weights.data
                isotonic_project_lut_(w, layer.n_inputs, self.proj_iters)
                w.clamp_(0, 1)
        return m


class DFAliteAPTrainer(APTrainer):
    """対照介入: 層固定ランダム±1符号でブロードキャスト誤差を歪める。

    各層のニューロンに dfa_seed で固定した±1符号を割り当て、
    ブロードキャスト誤差 error_l に乗じてから重み更新する(DFA的な符号ランダム化)。
    単調制約が効くなら符号ランダム化は悪化するはず、という対照。
    """

    def __init__(self, *args, dfa_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dfa_seed = dfa_seed

        # 各層のニューロンに固定ランダム±1符号を生成(CPU生成→使用時にto(device))
        generator = torch.Generator()
        generator.manual_seed(dfa_seed)
        self._dfa_signs: list[torch.Tensor] = []
        for layer in self.lut_layers:
            bits = torch.randint(0, 2, (layer.out_dim,), generator=generator)
            self._dfa_signs.append(bits.to(torch.float32) * 2.0 - 1.0)

    def _ap_backward(
        self,
        caches: list[LayerCache],
        neuron_error: torch.Tensor,
    ) -> None:
        """コアの _ap_backward を写経し、層固定符号を error_l に乗じてから更新する。

        コアと同一の到達マップ(src, dst)による scatter で error_l を構築する点は
        不変。相違は _update_weights へ渡す直前に self._dfa_signs[l] を乗じる点のみ。
        """
        batch = neuron_error.shape[0]
        device = neuron_error.device

        for l, cache in enumerate(caches):
            src, dst = self._reach_per_layer[l]
            src = src.to(device)
            dst = dst.to(device)

            # 出力ニューロンの誤差を到達先に直接scatter(コアと同一)
            gathered = neuron_error[:, src]
            error_l = torch.zeros(batch, cache.layer.out_dim, device=device)
            error_l.scatter_add_(
                1, dst.unsqueeze(0).expand(batch, -1), gathered
            )

            # 層固定±1符号を乗じてから更新(broadcast: (batch,O)×(O,))
            signs = self._dfa_signs[l].to(device)
            self._update_weights(cache, error_l * signs)


class MonotoneCNNAPTrainer(CNNAPTrainer):
    """CDD-LGN版の単調制約AP: 各更新後にConv/Dense全LUTを単調増加錐へ射影する。

    Conv重みは (out_channels, n_internal, 2^n) を (oc*n_internal, 2^n) に畳んで
    ゲート単位で射影し、Dense LutLayer は (out_dim, 2^n) をニューロン単位で射影する。
    """

    def __init__(self, *args, proj_iters: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_iters = proj_iters

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        m = super().train_step(x, target)
        with torch.no_grad():
            for conv in collect_lut_conv_layers(self.model):
                w = conv.weights.data  # (oc, n_internal, 2^n)
                oc, n_internal, n_entries = w.shape
                flat = w.reshape(oc * n_internal, n_entries)
                isotonic_project_lut_(flat, conv.n_inputs, self.proj_iters)
                flat.clamp_(0, 1)
                w.copy_(flat.reshape(oc, n_internal, n_entries))
            for layer in collect_lut_layers(self.model):
                w = layer.weights.data  # (out_dim, 2^n)
                isotonic_project_lut_(w, layer.n_inputs, self.proj_iters)
                w.clamp_(0, 1)
        return m
