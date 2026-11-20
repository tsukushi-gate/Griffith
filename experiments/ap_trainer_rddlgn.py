"""
RDDLGN用AP法（Akiho Propagation）トレーナー。

分類モデル用APTrainerのreach-mapとLUT更新ロジックを再利用しつつ、
RDDLGNのseq2seq構造（Encoder/Decoder、リカレント、Embedding）に対応。

リカレント部分（K-layers, P-layers）は各タイムステップ独立に扱い、
時間方向に誤差を辿らない。

使用例:
    model = RDDLGN(config)
    trainer = RDDLGNAPTrainer(model, lr=0.1)
    for src, tgt in dataloader:
        metrics = trainer.train_step(src, tgt)
"""

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import RDDLGN, GroupSum, LutLayer
from griffith_logic.rddlgn import LogicEmbedding, LutStack, _init_hidden


@dataclass
class LayerCache:
    """LutLayerのforward時にキャッシュする中間データ"""
    layer: LutLayer
    gathered: torch.Tensor       # (batch, out_dim, n_inputs)
    basis: torch.Tensor          # (batch, out_dim, 2^n_inputs)
    output: torch.Tensor         # (batch, out_dim)


def _build_stack_reach_maps(
    stack: LutStack,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    LutStack内のreach-mapを構築。
    出力ニューロン→各層の到達可能ニューロンの(src, dst)ペア。
    """
    lut_layers = list(stack.layers)
    num_layers = len(lut_layers)
    last_out_dim = lut_layers[-1].out_dim

    current_reach: dict[int, set[int]] = {
        j: {j} for j in range(last_out_dim)
    }

    reach_per_layer: list[tuple[torch.Tensor, torch.Tensor]] = []

    arange = torch.arange(last_out_dim, dtype=torch.long)
    reach_per_layer.append((arange, arange))

    for l in range(num_layers - 1, 0, -1):
        indices_np = lut_layers[l].indices.cpu().numpy()
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

        reach_per_layer.insert(0, (
            torch.tensor(src_list, dtype=torch.long),
            torch.tensor(dst_list, dtype=torch.long),
        ))
        current_reach = prev_reach

    return reach_per_layer


def _lut_forward(layer: LutLayer, x: torch.Tensor) -> LayerCache:
    """LutLayerのforwardをsigmoidなしで実行。"""
    batch = x.shape[0]
    out_dim = layer.out_dim
    n_inputs = layer.n_inputs

    gathered = x[:, layer.indices.T.reshape(-1)].reshape(
        batch, out_dim, n_inputs
    )

    basis = torch.ones(
        batch, out_dim, 1, device=x.device, dtype=x.dtype
    )
    for j in range(n_inputs):
        xj = gathered[:, :, j:j + 1]
        basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

    w = layer.weights.data.clamp(0, 1)
    output = torch.einsum('boe,oe->bo', basis, w)
    output = output.clamp(0, 1)

    return LayerCache(layer, gathered, basis, output)


def _stack_forward(
    stack: LutStack, x: torch.Tensor
) -> tuple[torch.Tensor, list[LayerCache]]:
    """LutStackのforwardをキャッシュ付きで実行。"""
    caches = []
    h = x
    for layer in stack.layers:
        cache = _lut_forward(layer, h)
        caches.append(cache)
        h = cache.output
    return h, caches


def _ap_backward_stack(
    caches: list[LayerCache],
    reach_maps: list[tuple[torch.Tensor, torch.Tensor]],
    neuron_error: torch.Tensor,
    lr: float,
    update_mode: int = 1,
) -> torch.Tensor:
    """
    LutStack内のAP法逆伝播。重みを更新し、入力側の誤差を返す。

    Returns:
        input_error: (batch, in_dim) Stack入力に対する誤差
    """
    batch = neuron_error.shape[0]
    device = neuron_error.device

    # 各層の重みを更新
    for l, cache in enumerate(caches):
        src, dst = reach_maps[l]
        src = src.to(device)
        dst = dst.to(device)

        gathered = neuron_error[:, src]
        error_l = torch.zeros(batch, cache.layer.out_dim, device=device)
        error_l.scatter_add_(1, dst.unsqueeze(0).expand(batch, -1), gathered)

        _update_weights(cache, error_l, lr, update_mode)

    # 入力側の誤差を計算（最初の層の入力に到達する誤差）
    first_layer = caches[0].layer
    indices = first_layer.indices  # (n_inputs, out_dim)
    n_inputs = first_layer.n_inputs

    # 最初の層のニューロン誤差を取得
    src_0, dst_0 = reach_maps[0]
    src_0 = src_0.to(device)
    dst_0 = dst_0.to(device)
    gathered_0 = neuron_error[:, src_0]
    first_layer_error = torch.zeros(batch, first_layer.out_dim, device=device)
    first_layer_error.scatter_add_(
        1, dst_0.unsqueeze(0).expand(batch, -1), gathered_0
    )

    # 最初の層の接続を通じて入力空間に伝播
    input_error = torch.zeros(batch, first_layer.in_dim, device=device)
    for slot in range(n_inputs):
        idx = indices[slot].unsqueeze(0).expand(batch, -1)
        input_error.scatter_add_(1, idx, first_layer_error)

    return input_error


def _update_weights(
    cache: LayerCache, error: torch.Tensor, lr: float, update_mode: int = 1
):
    """AP法の重み更新。"""
    layer = cache.layer
    basis = cache.basis

    if update_mode == 1:
        delta = torch.einsum('bo,boe->oe', error, basis) / error.shape[0]
        layer.weights.data.add_(delta * lr)
    elif update_mode == 2:
        max_idx = basis.argmax(dim=-1)
        max_basis = basis.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)
        delta_scalar = error * max_basis
        delta = torch.zeros_like(layer.weights.data)
        # indexは(out_dim, 1)形状にする(発覚したscatter方向バグの修正。
        # 旧: max_idx[b:b+1]=(1,out_dim)だと全ニューロンの更新が行0に集積していた)
        for b in range(error.shape[0]):
            delta.scatter_add_(
                1, max_idx[b].unsqueeze(-1), delta_scalar[b].unsqueeze(-1)
            )
        delta /= error.shape[0]
        layer.weights.data.add_(delta * lr)

    layer.weights.data.clamp_(0, 1)


class RDDLGNAPTrainer:
    """RDDLGN用AP法トレーナー。"""

    def __init__(
        self,
        model: RDDLGN,
        lr: float = 0.1,
        update_mode: int = 1,
        pad_token: int = 0,
    ):
        self.model = model
        self.lr = lr
        self.update_mode = update_mode
        self.pad_token = pad_token

        config = model.config

        # 全LutLayerがindependentパラメータ化であることを確認
        for name, module in model.named_modules():
            if isinstance(module, LutLayer):
                if module.parameterization != 'independent':
                    raise ValueError(
                        f'AP法はindependentパラメータ化を要求: '
                        f'{name} is {module.parameterization}'
                    )

        # 各LutStackのreach-mapを事前構築
        self._n_reach = _build_stack_reach_maps(model.encoder.n_layers)
        self._k_reach = _build_stack_reach_maps(model.encoder.k_layers)
        self._l_reach = _build_stack_reach_maps(model.decoder.l_layers)
        self._p_reach = _build_stack_reach_maps(model.decoder.p_layers)
        self._m_reach = _build_stack_reach_maps(model.decoder.m_layers)

        # GroupSum
        self._group_sum = model.decoder.group_sum

        # 次元情報（連結の分割に使用）
        self._p_out_dim = model.decoder.p_out_dim
        self._context_dim = config.k_layer_sizes[-1]
        self._l_out_dim = config.l_layer_sizes[-1]

        # LutLayerの重みを[0,1]で再初期化
        self._init_lut_weights()

    def _init_lut_weights(self):
        """全LutLayerの重みを[0,1]のランダム値で初期化"""
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, LutLayer):
                    module.weights.data.uniform_(0, 1)

    def train_step(
        self,
        src: torch.Tensor,
        tgt_full: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
        kd_alpha: float = 1.0,
    ) -> dict:
        """
        1バッチの訓練ステップ。

        Args:
            src: (batch, src_len) ソーストークンID
            tgt_full: (batch, tgt_len+1) ターゲットトークンID（BOS付き）
            teacher_probs: (batch, tgt_len, vocab_size) teacherのsoftmax確率分布。
                Noneの場合は既存のone-hotターゲットのみを使用（非KD、従来挙動）。
            kd_alpha: one-hotターゲットとteacher分布の混合率。
                target_dist = kd_alpha * one_hot + (1 - kd_alpha) * teacher_probs。
                kd_alpha=1.0（既定）はteacher_probsを渡してもone-hotのみと完全一致する。

        Returns:
            {'loss': float, 'accuracy': float}
        """
        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        with torch.no_grad():
            batch = src.shape[0]
            device = src.device
            tgt_len = tgt_input.shape[1]
            vocab_size = self.model.config.vocab_size

            # === Encoder Forward ===
            src_emb = self._embedding_forward(self.model.src_embedding, src)
            n_out, n_caches_per_t = self._encode_n_layers(src_emb)
            context, k_caches_final = self._encode_k_layers(
                n_out, batch, device
            )

            # === Decoder Forward + 各timestepでAP更新 ===
            tgt_emb = self._embedding_forward(self.model.tgt_embedding, tgt_input)
            l_out, l_caches_per_t = self._decode_l_layers(tgt_emb)

            p_prev = _init_hidden(
                batch, self._p_out_dim, device,
                self.model.config.init_mode, self.model.config.init_scale,
            )

            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            context_error_accum = torch.zeros(
                batch, self._context_dim, device=device
            )

            for t in range(tgt_len):
                l_t = l_out[:, t, :]

                # P-layers forward
                p_input = torch.cat([p_prev, context, l_t], dim=-1)
                p_t, p_caches = _stack_forward(
                    self.model.decoder.p_layers, p_input
                )

                # M-layers forward
                m_input = torch.cat([p_t, context, l_t], dim=-1)
                m_t, m_caches = _stack_forward(
                    self.model.decoder.m_layers, m_input
                )

                # GroupSum → logits
                logits_t = self._group_sum(m_t)

                # メトリクス
                label_t = tgt_label[:, t]
                mask = label_t != self.pad_token
                if mask.sum() > 0:
                    loss_t = F.cross_entropy(
                        logits_t, label_t, ignore_index=self.pad_token
                    )
                    total_loss += loss_t.item()
                    pred_t = logits_t.argmax(dim=-1)
                    total_correct += ((pred_t == label_t) & mask).sum().item()
                    total_tokens += mask.sum().item()

                # === AP backward for this timestep ===

                # 1. 出力誤差
                probs = F.softmax(logits_t, dim=-1)
                target_oh = F.one_hot(label_t, num_classes=vocab_size).float()
                if teacher_probs is None:
                    target_dist = target_oh
                else:
                    target_dist = (
                        kd_alpha * target_oh
                        + (1.0 - kd_alpha) * teacher_probs[:, t]
                    )
                class_error = target_dist - probs  # (batch, vocab_size)

                # PADマスク適用
                class_error = class_error * mask.float().unsqueeze(-1)

                # GroupSum逆
                tau = self._group_sum.tau
                m_out_dim = self.model.decoder.m_layers.out_dim
                n_per = m_out_dim // vocab_size
                neuron_error = class_error.repeat_interleave(n_per, dim=-1) / tau

                # 2. M-layers AP backward → M入力誤差
                m_input_error = _ap_backward_stack(
                    m_caches, self._m_reach, neuron_error,
                    self.lr, self.update_mode,
                )

                # 3. M入力 = [p_t | context | l_t] → 分割
                p_dim = self._p_out_dim
                c_dim = self._context_dim
                l_dim = self._l_out_dim
                p_t_error = m_input_error[:, :p_dim]
                context_error_accum += m_input_error[:, p_dim:p_dim + c_dim]
                l_t_error_from_m = m_input_error[:, p_dim + c_dim:]

                # 4. P-layers AP backward → P入力誤差
                p_input_error = _ap_backward_stack(
                    p_caches, self._p_reach, p_t_error,
                    self.lr, self.update_mode,
                )

                # P入力 = [p_prev | context | l_t] → 分割
                # p_prev誤差は破棄（時間方向に辿らない）
                context_error_accum += p_input_error[:, p_dim:p_dim + c_dim]
                l_t_error_from_p = p_input_error[:, p_dim + c_dim:]

                # 5. L-layers AP backward → Embedding更新
                l_t_error = l_t_error_from_m + l_t_error_from_p
                l_input_error = _ap_backward_stack(
                    l_caches_per_t[t], self._l_reach, l_t_error,
                    self.lr, self.update_mode,
                )
                self._update_embedding(
                    self.model.tgt_embedding, tgt_input[:, t], l_input_error
                )

                p_prev = p_t

            # === Encoder AP backward ===
            # 蓄積context誤差 → K-layers（最終stepのみ）
            k_input_error = _ap_backward_stack(
                k_caches_final, self._k_reach, context_error_accum,
                self.lr, self.update_mode,
            )

            # K入力 = [n_final | k_prev] → n_final誤差のみ使用
            n_out_dim = self.model.config.n_layer_sizes[-1]
            n_final_error = k_input_error[:, :n_out_dim]

            # N-layers AP backward（最終timestepのみ）
            src_len = src.shape[1]
            n_input_error = _ap_backward_stack(
                n_caches_per_t[src_len - 1], self._n_reach, n_final_error,
                self.lr, self.update_mode,
            )
            self._update_embedding(
                self.model.src_embedding, src[:, -1], n_input_error
            )

        avg_loss = total_loss / tgt_len if tgt_len > 0 else 0.0
        acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return {'loss': avg_loss, 'accuracy': acc}

    def _embedding_forward(
        self, embedding: LogicEmbedding, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Embedding forward。sigmoidを適用。"""
        e = embedding.embedding(tokens)
        return torch.sigmoid(e)

    def _encode_n_layers(
        self, src_emb: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[LayerCache]]]:
        """N-layers: 各timestepを独立にforward。キャッシュをtimestep別に保存。"""
        batch, seq_len, d_emb = src_emb.shape
        n_caches_per_t = []
        outputs = []

        for t in range(seq_len):
            h_t = src_emb[:, t, :]
            out_t, caches_t = _stack_forward(
                self.model.encoder.n_layers, h_t
            )
            n_caches_per_t.append(caches_t)
            outputs.append(out_t)

        n_out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_out_dim)
        return n_out, n_caches_per_t

    def _encode_k_layers(
        self, n_out: torch.Tensor, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """K-layers: リカレント。最終stepのキャッシュのみ返す。"""
        seq_len = n_out.shape[1]
        k_out_dim = self.model.encoder.k_out_dim
        k_prev = _init_hidden(
            batch, k_out_dim, device,
            self.model.config.init_mode, self.model.config.init_scale,
        )

        k_caches_final = None
        for t in range(seq_len):
            k_input = torch.cat([n_out[:, t, :], k_prev], dim=-1)
            k_prev, k_caches = _stack_forward(
                self.model.encoder.k_layers, k_input
            )
            k_caches_final = k_caches

        return k_prev, k_caches_final

    def _decode_l_layers(
        self, tgt_emb: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[LayerCache]]]:
        """L-layers: 各timestepを独立にforward。"""
        batch, tgt_len, d_emb = tgt_emb.shape
        l_caches_per_t = []
        outputs = []

        for t in range(tgt_len):
            h_t = tgt_emb[:, t, :]
            out_t, caches_t = _stack_forward(
                self.model.decoder.l_layers, h_t
            )
            l_caches_per_t.append(caches_t)
            outputs.append(out_t)

        l_out = torch.stack(outputs, dim=1)
        return l_out, l_caches_per_t

    def _update_embedding(
        self,
        embedding: LogicEmbedding,
        token_ids: torch.Tensor,
        error: torch.Tensor,
    ):
        """
        Embedding重みをAP風に更新。

        sigmoid前のlogit空間で: W[token_id] += mean(error) * lr
        """
        batch = token_ids.shape[0]
        with torch.no_grad():
            for b in range(batch):
                tid = token_ids[b].item()
                embedding.embedding.weight.data[tid] += error[b] * self.lr

    def evaluate(
        self,
        loader,
        device: str,
        pad_token: int = 0,
        bos_token: int = 1,
        eos_token: int = 2,
    ) -> float:
        """トークンレベル精度を計算。"""
        correct = 0
        total = 0

        with torch.no_grad():
            for src, tgt in loader:
                src = src.to(device)
                tgt = tgt.to(device)

                batch = src.shape[0]
                bos = torch.full(
                    (batch, 1), bos_token, dtype=torch.long, device=device
                )
                eos = torch.full(
                    (batch, 1), eos_token, dtype=torch.long, device=device
                )
                tgt_full = torch.cat([bos, tgt, eos], dim=1)

                tgt_input = tgt_full[:, :-1]
                tgt_label = tgt_full[:, 1:]

                # forward（AP用のforward）
                src_emb = self._embedding_forward(
                    self.model.src_embedding, src
                )
                n_out, _ = self._encode_n_layers(src_emb)
                context, _ = self._encode_k_layers(
                    n_out, batch, src.device
                )

                tgt_emb = self._embedding_forward(
                    self.model.tgt_embedding, tgt_input
                )
                l_out, _ = self._decode_l_layers(tgt_emb)

                p_prev = _init_hidden(
                    batch, self._p_out_dim, src.device,
                    self.model.config.init_mode,
                    self.model.config.init_scale,
                )

                tgt_len = tgt_input.shape[1]
                for t in range(tgt_len):
                    l_t = l_out[:, t, :]
                    p_input = torch.cat([p_prev, context, l_t], dim=-1)
                    p_t, _ = _stack_forward(
                        self.model.decoder.p_layers, p_input
                    )
                    m_input = torch.cat([p_t, context, l_t], dim=-1)
                    m_t, _ = _stack_forward(
                        self.model.decoder.m_layers, m_input
                    )
                    logits_t = self._group_sum(m_t)

                    label_t = tgt_label[:, t]
                    mask = label_t != pad_token
                    pred_t = logits_t.argmax(dim=-1)
                    correct += ((pred_t == label_t) & mask).sum().item()
                    total += mask.sum().item()

                    p_prev = p_t

        return correct / total if total > 0 else 0.0
