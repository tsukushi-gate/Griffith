"""
CDD-LGN用AP法（Akiho Propagation）トレーナー。

LutConv2d（ツリー構造畳み込み）+ OrPooling2d + LutLayer（Dense）+ GroupSum で
構成されるCDD-LGNモデルをAP法で学習する。

ツリー内の全ゲートに同一の誤差を渡し、各ゲートは自身のbasis値で重み付けして更新。
OrPoolingはmax_pool2dの最大値位置を記録し、max_unpool2dで誤差を逆伝播。

使用例:
    model = build_mnist_model(args)
    trainer = CNNAPTrainer(model, lr=0.1)
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
from ap_trainer import (
    APTrainer,
)

# ap_trainer.pyからLutLayer用のreach-map/更新ロジックを再利用
from ap_trainer import (
    LayerCache as LutLayerCache,
)

from griffith_logic import GroupSum, LutConv2d, LutLayer, OrPooling2d
from griffith_logic.conv import _compute_basis

# ============================================================
# LutConv2d AP法用のキャッシュとユーティリティ
# ============================================================

@dataclass
class ConvCache:
    """LutConv2dのforward時にキャッシュする中間データ"""
    conv: LutConv2d
    level_caches: list  # 各レベルの(basis, gate_offset, n_groups)
    patches: torch.Tensor  # unfold後のパッチ (B, C*kH*kW, L)
    output_shape: tuple  # (B, out_ch, H_out, W_out)
    input_shape: tuple   # (B, C, H, W)


@dataclass
class PoolCache:
    """OrPooling2dのforward時にキャッシュするデータ"""
    indices: torch.Tensor  # max_pool2dの最大値位置
    input_shape: tuple     # プーリング前の形状
    kernel_size: int


def _conv_forward_ap(
    conv: LutConv2d, x: torch.Tensor
) -> tuple[torch.Tensor, ConvCache]:
    """LutConv2dのforwardをAP法用に実行（sigmoidなし）。"""
    B, C, H, W = x.shape
    kH, kW = conv.kernel_size
    sH, sW = conv.stride
    pH, pW = conv.padding

    # Unfold → パッチ抽出
    patches = F.unfold(x, (kH, kW), padding=(pH, pW), stride=(sH, sW))
    L = patches.shape[2]
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    # リーフ入力の収集
    leaves = patches[:, conv.indices_flat]
    leaves = leaves.permute(0, 3, 1, 2)  # (B, L, out_ch, n_leaves)
    leaves = leaves.reshape(B * L, conv.out_channels, conv.n_leaves)

    # ツリー演算（各レベルのbasisをキャッシュ）
    current = leaves
    level_caches = []
    gate_offset = 0

    for level in range(conv.tree_depth):
        n_groups = current.shape[-1] // conv.n_inputs
        grouped = current[..., :n_groups * conv.n_inputs].reshape(
            *current.shape[:-1], n_groups, conv.n_inputs
        )

        basis = _compute_basis(grouped, conv.n_inputs)
        w = conv.weights.data[:, gate_offset:gate_offset + n_groups, :].clamp(0, 1)
        current = (basis * w.unsqueeze(0)).sum(dim=-1)

        level_caches.append((basis, gate_offset, n_groups))
        gate_offset += n_groups

    # 出力整形
    output = current.squeeze(-1)  # (B*L, out_ch)
    output = output.reshape(B, L, conv.out_channels)
    output = output.permute(0, 2, 1).reshape(B, conv.out_channels, H_out, W_out)

    cache = ConvCache(
        conv=conv,
        level_caches=level_caches,
        patches=patches,
        output_shape=(B, conv.out_channels, H_out, W_out),
        input_shape=(B, C, H, W),
    )
    return output, cache


def _conv_update_ap(cache: ConvCache, error_4d: torch.Tensor, lr: float):
    """
    LutConv2dのAP法重み更新。
    ツリー内の全ゲートに同一の誤差を渡す。

    Args:
        error_4d: (B, out_ch, H_out, W_out)
    """
    conv = cache.conv
    B, out_ch, H_out, W_out = error_4d.shape
    L = H_out * W_out

    # (B, out_ch, H_out, W_out) → (B*L, out_ch)
    error = error_4d.permute(0, 2, 3, 1).reshape(B * L, out_ch)

    for basis, gate_offset, n_groups in cache.level_caches:
        # basis: (B*L, out_ch, n_groups, 2^n_inputs)
        # error: (B*L, out_ch) → 全ゲートに同じ値
        error_expanded = error.unsqueeze(-1).unsqueeze(-1)
        delta = (error_expanded * basis).mean(dim=0)  # (out_ch, n_groups, 2^n)
        conv.weights.data[:, gate_offset:gate_offset + n_groups, :] += delta * lr

    conv.weights.data.clamp_(0, 1)


def _conv_backward_to_input(
    cache: ConvCache, error_4d: torch.Tensor
) -> torch.Tensor:
    """
    LutConv2dの誤差をリーフ→パッチ空間→画像空間に伝播。

    Args:
        error_4d: (B, out_ch, H_out, W_out)

    Returns:
        input_error: (B, C, H, W)
    """
    conv = cache.conv
    B, out_ch, H_out, W_out = error_4d.shape
    L = H_out * W_out
    kH, kW = conv.kernel_size
    sH, sW = conv.stride
    pH, pW = conv.padding
    _, C, H, W = cache.input_shape

    # (B, out_ch, H_out, W_out) → (B*L, out_ch)
    error = error_4d.permute(0, 2, 3, 1).reshape(B * L, out_ch)

    # 各リーフにerrorを伝播（ツリー内は同一誤差）
    # 各ツリーのn_leaves個のリーフに同じ誤差を渡す
    n_leaves = conv.n_leaves
    # (B*L, out_ch) → (B*L, out_ch, n_leaves)
    leaf_error = error.unsqueeze(-1).expand(-1, -1, n_leaves)

    # リーフ誤差をパッチ空間にscatter
    # indices_flat: (out_ch, n_leaves) — パッチ空間でのインデックス
    patch_dim = C * kH * kW
    patch_error = torch.zeros(
        B * L, patch_dim, device=error.device, dtype=error.dtype
    )

    # leaf_error: (B*L, out_ch, n_leaves) → scatter先: (B*L, patch_dim)
    indices = conv.indices_flat  # (out_ch, n_leaves)
    for ch in range(out_ch):
        idx = indices[ch].unsqueeze(0).expand(B * L, -1)
        patch_error.scatter_add_(1, idx, leaf_error[:, ch, :])

    # (B*L, patch_dim) → (B, patch_dim, L)
    patch_error = patch_error.reshape(B, L, patch_dim).permute(0, 2, 1)

    # fold: パッチ→画像空間
    input_error = F.fold(
        patch_error, (H, W), (kH, kW),
        padding=(pH, pW), stride=(sH, sW),
    )

    return input_error


def _orpool_forward_ap(
    pool: OrPooling2d, x: torch.Tensor
) -> tuple[torch.Tensor, PoolCache]:
    """OrPooling2dのforward。最大値位置を記録。"""
    output, indices = F.max_pool2d(
        x, pool.kernel_size, pool.stride, pool.padding,
        return_indices=True,
    )
    cache = PoolCache(
        indices=indices,
        input_shape=x.shape,
        kernel_size=pool.kernel_size,
    )
    return output, cache


def _orpool_backward_ap(
    error: torch.Tensor, cache: PoolCache
) -> torch.Tensor:
    """OrPooling2dの誤差逆伝播。最大値位置に誤差を配置。"""
    return F.max_unpool2d(
        error, cache.indices, cache.kernel_size,
        stride=cache.kernel_size, padding=0,
        output_size=cache.input_shape,
    )


# ============================================================
# CNNAPTrainer
# ============================================================

class CNNAPTrainer:
    """CDD-LGN用AP法トレーナー。"""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.1,
        update_mode: int = 1,
    ):
        self.model = model
        self.lr = lr
        self.update_mode = update_mode

        # モデルからモジュールを抽出・分類
        self.conv_layers: list[LutConv2d] = []
        self.pool_layers: list[OrPooling2d] = []
        self.lut_layers: list[LutLayer] = []
        self.group_sum: GroupSum | None = None

        # nn.Sequentialのモジュール順序を記録
        self._module_order: list[tuple[str, nn.Module]] = []
        for module in model:
            if isinstance(module, LutConv2d):
                self._module_order.append(('conv', module))
                self.conv_layers.append(module)
            elif isinstance(module, OrPooling2d):
                self._module_order.append(('pool', module))
                self.pool_layers.append(module)
            elif isinstance(module, nn.Flatten):
                self._module_order.append(('flatten', module))
            elif isinstance(module, LutLayer):
                if module.parameterization != 'independent':
                    raise ValueError(
                        f'AP法はindependentパラメータ化を要求: {module.parameterization}'
                    )
                self._module_order.append(('lut', module))
                self.lut_layers.append(module)
            elif isinstance(module, GroupSum):
                self._module_order.append(('groupsum', module))
                self.group_sum = module

        if self.group_sum is None:
            raise ValueError('モデルにGroupSumが含まれていません')

        # LutLayer用のreach-mapを構築
        self._build_lut_reach_maps()

        # 重みを[0,1]で再初期化
        self._init_weights()

    def _build_lut_reach_maps(self):
        """Dense LutLayer部分のreach-mapを構築"""
        if not self.lut_layers:
            self._lut_reach_maps = []
            return

        num_layers = len(self.lut_layers)
        last_out_dim = self.lut_layers[-1].out_dim

        current_reach: dict[int, set[int]] = {
            j: {j} for j in range(last_out_dim)
        }

        self._lut_reach_maps: list[tuple[torch.Tensor, torch.Tensor]] = []

        arange = torch.arange(last_out_dim, dtype=torch.long)
        self._lut_reach_maps.append((arange, arange))

        for l in range(num_layers - 1, 0, -1):
            indices_np = self.lut_layers[l].indices.cpu().numpy()
            n_inputs, out_dim = indices_np.shape

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

            src_list, dst_list = [], []
            for prev_idx, output_set in prev_reach.items():
                for out_j in output_set:
                    src_list.append(out_j)
                    dst_list.append(prev_idx)

            self._lut_reach_maps.insert(0, (
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long),
            ))
            current_reach = prev_reach

    def _init_weights(self):
        """全LUT重みを[0,1]で再初期化"""
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, LutLayer | LutConv2d):
                    if isinstance(module, LutConv2d):
                        module.weights.data.uniform_(0, 1)
                    else:
                        module.weights.data.uniform_(0, 1)

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        """1バッチの訓練ステップ。"""
        with torch.no_grad():
            # Forward（キャッシュ収集）
            logits, forward_caches = self._forward_with_cache(x)

            # メトリクス
            pred = logits.argmax(dim=-1)
            acc = (pred == target).float().mean().item()
            loss = F.cross_entropy(logits, target).item()

            # 出力誤差
            k = self.group_sum.k
            tau = self.group_sum.tau
            probs = F.softmax(logits, dim=-1)
            target_oh = F.one_hot(target, num_classes=k).float().to(logits.device)
            class_error = target_oh - probs

            # GroupSum逆
            last_lut = self.lut_layers[-1] if self.lut_layers else None
            if last_lut:
                last_out_dim = last_lut.out_dim
            else:
                # Conv直結の場合（Dense層がない場合は未対応）
                raise NotImplementedError('Dense層なしのCDD-LGNは未対応')

            n_per = last_out_dim // k
            neuron_error = class_error.repeat_interleave(n_per, dim=-1) / tau

            # AP逆伝播
            self._ap_backward(forward_caches, neuron_error)

        return {'loss': loss, 'accuracy': acc}

    def _forward_with_cache(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """モデル全体をforward。各層のキャッシュをリストで返す。"""
        caches = []
        h = x

        for kind, module in self._module_order:
            if kind == 'conv':
                h, cache = _conv_forward_ap(module, h)
                caches.append(('conv', cache))
            elif kind == 'pool':
                h, cache = _orpool_forward_ap(module, h)
                caches.append(('pool', cache))
            elif kind == 'flatten':
                caches.append(('flatten', h.shape))
                h = h.flatten(1)
            elif kind == 'lut':
                cache = self._lut_forward(module, h)
                caches.append(('lut', cache))
                h = cache.output
            elif kind == 'groupsum':
                logits = module(h)

        return logits, caches

    def _lut_forward(self, layer: LutLayer, x: torch.Tensor) -> LutLayerCache:
        """LutLayerのforward（sigmoidなし）"""
        batch = x.shape[0]
        out_dim = layer.out_dim
        n_inputs = layer.n_inputs

        gathered = x[:, layer.indices.T.reshape(-1)].reshape(
            batch, out_dim, n_inputs
        )
        basis = torch.ones(batch, out_dim, 1, device=x.device, dtype=x.dtype)
        for j in range(n_inputs):
            xj = gathered[:, :, j:j + 1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        w = layer.weights.data.clamp(0, 1)
        output = torch.einsum('boe,oe->bo', basis, w).clamp(0, 1)

        return LutLayerCache(layer, gathered, basis, [], output)

    def _ap_backward(self, forward_caches: list, neuron_error: torch.Tensor):
        """全層のAP逆伝播。"""
        batch = neuron_error.shape[0]
        device = neuron_error.device
        error = neuron_error

        # LutLayer部分のreach-mapによる更新
        lut_idx = len(self.lut_layers) - 1
        conv_pool_caches = []

        for kind, cache in reversed(forward_caches):
            if kind == 'lut':
                # reach-mapでこの層の重みを更新
                src, dst = self._lut_reach_maps[lut_idx]
                src = src.to(device)
                dst = dst.to(device)
                gathered = neuron_error[:, src]
                error_l = torch.zeros(batch, cache.layer.out_dim, device=device)
                error_l.scatter_add_(
                    1, dst.unsqueeze(0).expand(batch, -1), gathered
                )
                self._lut_update(cache, error_l)

                # 最初のLutLayerからの入力誤差を計算
                if lut_idx == 0:
                    first = cache.layer
                    indices = first.indices
                    error_flat = torch.zeros(
                        batch, first.in_dim, device=device
                    )
                    for slot in range(first.n_inputs):
                        idx = indices[slot].unsqueeze(0).expand(batch, -1)
                        error_flat.scatter_add_(1, idx, error_l)
                    error = error_flat

                lut_idx -= 1

            elif kind == 'flatten':
                # error: (batch, flat_dim) → (batch, C, H, W)
                pre_flatten_shape = cache  # tuple
                error = error.reshape(pre_flatten_shape)

            elif kind == 'pool':
                error = _orpool_backward_ap(error, cache)

            elif kind == 'conv':
                _conv_update_ap(cache, error, self.lr)
                error = _conv_backward_to_input(cache, error)

    def _lut_update(self, cache: LutLayerCache, error: torch.Tensor):
        """LutLayerのAP重み更新。"""
        layer = cache.layer
        basis = cache.basis
        delta = torch.einsum('bo,boe->oe', error, basis) / error.shape[0]
        layer.weights.data.add_(delta * self.lr)
        layer.weights.data.clamp_(0, 1)

    def evaluate(
        self,
        loader,
        device: str,
        binarize: bool = False,
    ) -> float:
        """評価。"""
        saved = {}
        if binarize:
            for i, m in enumerate(self.model.modules()):
                if isinstance(m, LutLayer | LutConv2d):
                    saved[i] = m.weights.data.clone()
                    m.weights.data = (m.weights.data > 0.5).float()

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

        if binarize:
            for i, m in enumerate(self.model.modules()):
                if isinstance(m, LutLayer | LutConv2d) and i in saved:
                    m.weights.data = saved[i]

        return correct / total if total > 0 else 0.0
