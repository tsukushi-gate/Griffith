"""
AP法（Akiho Propagation）による多層LUTネットワーク訓練器。

autogradを使わず、各LutLayerの重みを error × basis × lr ルールで直接更新する。
LutLayerのindependentパラメータ化と対応するが、sigmoid変換を介さず
重みを[0,1]の確率値として直接扱う。

使用例:
    model = build_model(args)  # parameterization='independent'
    trainer = APTrainer(model, lr=0.1, update_mode=1)
    for x, y in dataloader:
        metrics = trainer.train_step(x, y)
"""

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import GroupSum, LutLayer


@dataclass
class LayerCache:
    """LutLayerのforward時にキャッシュする中間データ"""
    layer: LutLayer
    gathered: torch.Tensor       # (batch, out_dim, n_inputs)
    basis: torch.Tensor          # (batch, out_dim, 2^n_inputs)
    intermediates: list          # butterfly中間状態（n_inputs個）
    output: torch.Tensor         # (batch, out_dim)


class APTrainer:
    """
    AP法（Akiho Propagation）による多層LUTネットワーク訓練器。

    LutLayerのforwardを呼ばず、weights.dataを直接読み取って
    独自のforward/backwardを実行する。全処理はtorch.no_grad()内で行う。
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.1,
        update_mode: int = 1,
    ):
        """
        Args:
            model: nn.Sequential(Flatten, LutLayer, ..., LutLayer, GroupSum)
            lr: 学習率
            update_mode: 1=全重み更新, 2=最大basis入力のみ更新
        """
        self.model = model
        self.lr = lr
        self.update_mode = update_mode

        # モデルからLutLayerとGroupSumを抽出
        self.lut_layers: list[LutLayer] = []
        self.group_sum: GroupSum | None = None
        for module in model.modules():
            if isinstance(module, LutLayer):
                if module.parameterization != 'independent':
                    raise ValueError(
                        f'AP法はindependentパラメータ化を要求: {module.parameterization}'
                    )
                # AP法は_build_reach_maps()でlayer.indicesが訓練中
                # 不変であることを前提に出力→各層の到達可能ニューロンの
                # マッピングを事前計算する(__init__時点で1回のみ構築)。
                # wiring='learnable'(DWN方式の学習可能配線、lut.py参照)は
                # 訓練中にindicesが変化しうるため、この前提が崩れて
                # reach-mapが暗黙に不正化する(サイレントに誤った勾配を
                # 与え続ける)。よってAP法とlearnable wiringの組み合わせは
                # 明確に拒否する。
                if getattr(module, 'wiring', 'fixed') != 'fixed':
                    raise ValueError(
                        f'AP法はwiring="fixed"のLutLayerのみ対応: '
                        f'wiring={module.wiring!r}'
                        '(reach-mapが固定indices前提のため、学習可能配線'
                        '(wiring="learnable")とは非対応)'
                    )
                self.lut_layers.append(module)
            elif isinstance(module, GroupSum):
                self.group_sum = module

        if not self.lut_layers:
            raise ValueError('モデルにLutLayerが含まれていません')
        if self.group_sum is None:
            raise ValueError('モデルにGroupSumが含まれていません')

        # 重みをAP法用に[0,1]で再初期化
        self._init_weights()

        # 出力ニューロン→各層の到達可能ニューロンのマッピングを事前構築
        self._build_reach_maps()

    def _init_weights(self):
        """全LutLayerの重みを[0,1]のランダム値で初期化"""
        with torch.no_grad():
            for layer in self.lut_layers:
                layer.weights.data.uniform_(0, 1)

    def _build_reach_maps(self):
        """
        出力ニューロンから各層の到達可能ニューロンへのマッピングを事前構築。

        AP法では出力ニューロンjの誤差e_jが、接続グラフを辿って到達可能な
        全LUTに同一の値として渡される。この到達関係を(src, dst)インデックス
        ペアとして事前計算する。
        """
        num_layers = len(self.lut_layers)
        last_out_dim = self.lut_layers[-1].out_dim

        # 最終層: 各出力ニューロンは自分自身に到達
        # current_reach: ニューロン→到達元の出力ニューロン集合
        current_reach: dict[int, set[int]] = {
            j: {j} for j in range(last_out_dim)
        }

        self._reach_per_layer: list[tuple[torch.Tensor, torch.Tensor]] = []

        # 最終層の(src, dst)ペアを構築
        arange = torch.arange(last_out_dim, dtype=torch.long)
        self._reach_per_layer.append((arange, arange))

        # 最終層から逆順に到達マップを構築
        for l in range(num_layers - 1, 0, -1):
            indices_np = self.lut_layers[l].indices.cpu().numpy()
            n_inputs, out_dim = indices_np.shape

            # 層lの各ニューロンの到達元を、前層の入力ニューロンに伝播
            prev_reach: dict[int, set[int]] = {}
            for j in range(out_dim):
                if j not in current_reach:
                    continue
                outputs_reaching_j = current_reach[j]
                for slot in range(n_inputs):
                    prev_idx = int(indices_np[slot, j])
                    if prev_idx not in prev_reach:
                        prev_reach[prev_idx] = set()
                    prev_reach[prev_idx].update(outputs_reaching_j)

            # (src, dst)ペアに変換
            src_list, dst_list = [], []
            for prev_idx, output_set in prev_reach.items():
                for out_j in output_set:
                    src_list.append(out_j)
                    dst_list.append(prev_idx)

            self._reach_per_layer.insert(0, (
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long),
            ))
            current_reach = prev_reach

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        """
        1バッチの訓練ステップ。

        Args:
            x: (batch, ...) 入力テンソル
            target: (batch,) 正解ラベル（整数）

        Returns:
            {'loss': float, 'accuracy': float}
        """
        with torch.no_grad():
            # 1. Forward（各層のbasis等をキャッシュ）
            logits, caches = self._forward_with_cache(x)

            # 2. メトリクス（更新前に記録）
            pred = logits.argmax(dim=-1)
            acc = (pred == target).float().mean().item()
            loss = F.cross_entropy(logits, target).item()

            # 3. 出力誤差をニューロン単位に変換
            neuron_error = self._compute_output_error(logits, target)

            # 4. AP法: 接続経路を辿って全層に同一誤差を伝播し重み更新
            self._ap_backward(caches, neuron_error)

        return {'loss': loss, 'accuracy': acc}

    def _forward_with_cache(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """各LutLayerのforward結果をキャッシュしながらモデル全体を実行"""
        caches = []
        h = x

        for module in self.model:
            if isinstance(module, nn.Flatten):
                h = h.flatten(1)
            elif isinstance(module, LutLayer):
                cache = self._lut_forward(module, h)
                caches.append(cache)
                h = cache.output
            elif isinstance(module, GroupSum):
                logits = module(h)
            else:
                h = module(h)

        return logits, caches

    def _lut_forward(self, layer: LutLayer, x: torch.Tensor) -> LayerCache:
        """
        LutLayerのforwardをsigmoidなしで実行。
        weights.dataを[0,1]の確率値として直接使う。
        """
        batch = x.shape[0]
        out_dim = layer.out_dim
        n_inputs = layer.n_inputs

        # 入力収集
        gathered = x[:, layer.indices.T.reshape(-1)].reshape(
            batch, out_dim, n_inputs
        )

        # Butterfly基底（中間状態を保存）
        intermediates = []
        basis = torch.ones(
            batch, out_dim, 1, device=x.device, dtype=x.dtype
        )
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j + 1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # AP法: weightsを直接使う（sigmoidなし）
        w = layer.weights.data.clamp(0, 1)
        output = torch.einsum('boe,oe->bo', basis, w)
        output = output.clamp(0, 1)

        return LayerCache(layer, gathered, basis, intermediates, output)

    def _compute_output_error(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        GroupSum出力からニューロン単位の誤差を計算。

        one_hot(target) - softmax(logits) → GroupSumの逆でbroadcast
        """
        k = self.group_sum.k
        tau = self.group_sum.tau

        probs = F.softmax(logits, dim=-1)  # (batch, k)
        target_oh = F.one_hot(target, num_classes=k).float()
        class_error = target_oh - probs  # (batch, k)

        # GroupSumの逆: 各クラスの誤差をそのクラスに属する全ニューロンにbroadcast
        # GroupSum forward: reshape(k, n_per).sum(-1) / tau
        # 逆: repeat_interleave / tau
        last_out_dim = self.lut_layers[-1].out_dim
        n_per = last_out_dim // k
        neuron_error = class_error.repeat_interleave(n_per, dim=-1) / tau

        return neuron_error  # (batch, last_out_dim)

    def _update_weights(self, cache: LayerCache, error: torch.Tensor):
        """
        AP法の重み更新ルール。

        mode 1: w_k += mean(error × basis_k) × lr （全重み更新）
        mode 2: 最大basis入力に対応する重みのみ更新
        """
        layer = cache.layer
        basis = cache.basis  # (batch, out_dim, 2^n)

        if self.update_mode == 1:
            # バッチ平均で全重みを更新
            delta = torch.einsum('bo,boe->oe', error, basis) / error.shape[0]
            layer.weights.data.add_(delta * self.lr)
        elif self.update_mode == 2:
            # 最大basis入力に対応する重みのみ更新
            max_idx = basis.argmax(dim=-1)  # (batch, out_dim)
            max_basis = basis.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)
            delta_scalar = error * max_basis  # (batch, out_dim)

            # scatter_addで該当重みのみに集積(各ニューロン行のmax_idx列へ)
            delta = torch.zeros_like(layer.weights.data)
            for b in range(error.shape[0]):
                delta.scatter_add_(
                    1, max_idx[b].unsqueeze(-1), delta_scalar[b].unsqueeze(-1)
                )
            delta /= error.shape[0]
            layer.weights.data.add_(delta * self.lr)

        # [0, 1]にクリップ
        layer.weights.data.clamp_(0, 1)

    def _ap_backward(
        self,
        caches: list[LayerCache],
        neuron_error: torch.Tensor,
    ):
        """
        AP法の誤差伝播: 出力ニューロンの誤差を、事前計算した到達マップで
        各層に直接scatter。カスケードなし（誤差が層ごとに増幅されない）。

        各層の誤差は出力層のneuron_errorから直接計算される:
        error_l[i] = Σ neuron_error[j] for all output j that reaches neuron i in layer l
        """
        batch = neuron_error.shape[0]
        device = neuron_error.device

        for l, cache in enumerate(caches):
            src, dst = self._reach_per_layer[l]
            src = src.to(device)
            dst = dst.to(device)

            # 出力ニューロンの誤差を到達先に直接scatter
            gathered = neuron_error[:, src]  # (batch, num_pairs)
            error_l = torch.zeros(
                batch, cache.layer.out_dim, device=device
            )
            error_l.scatter_add_(
                1, dst.unsqueeze(0).expand(batch, -1), gathered
            )

            self._update_weights(cache, error_l)

    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        device: str,
        binarize: bool = False,
    ) -> float:
        """
        AP法用の評価。LutLayerのforwardを呼ばず独自にforward。

        Args:
            binarize: Trueなら一時的にbinarizeして評価
        """
        saved_weights = {}
        if binarize:
            for i, layer in enumerate(self.lut_layers):
                saved_weights[i] = layer.weights.data.clone()
                layer.weights.data = (
                    layer.weights.data > 0.5
                ).float()

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                if binarize:
                    x = x.round()
                logits, _ = self._forward_with_cache(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.shape[0]

        # binarize時は元の重みを復元
        if binarize:
            for i, layer in enumerate(self.lut_layers):
                layer.weights.data = saved_weights[i]

        return correct / total if total > 0 else 0.0
