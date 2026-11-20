"""
適格度トレース(eligibility trace)付きRDDLGN AP法トレーナー。

現行の `RDDLGNAPTrainer` はリカレント共有層の時間方向信用割当を欠く:
  - K-layers(エンコーダ再帰): 最終エンコーダstepのbasisのみで1回更新
  - P-layers(デコーダ再帰): 各stepで更新するが p_prev 経由の誤差は破棄

本トレーナーはスパイキングNNのe-prop(Bellec et al., 2020, Nat. Commun.)の
LUT版として、basis値の指数移動平均トレースを導入する:

    trace_l(t) = lambda * trace_l(t-1) + basis_l(t)

更新は `w_l += error * trace_l * lr`(errorは従来どおり、basisをトレースに置換)。
lambda=0 では trace_l = basis_l(現step) となり、基底クラスと厳密一致する
(回帰テスト tests/test_ap_eligibility_trace.py で保証)。

メモリ増分はトレース(K/P各層のbasisと同形状)のみでO(basis数)、系列長Tに非依存。

禁止事項の遵守:
  - ライブラリコア(griffith_logic/)は変更せず、トレーナー側の拡張として実装
  - 基底クラス `RDDLGNAPTrainer` は一切変更せず、train_stepを複製して2箇所のみ改変
  - decoderの各step独立更新(M/L層、Pのstep単位更新)の仕様は維持
    (共有リカレント層のbasis→trace置換のみが変更点)
"""

import os
import sys
from dataclasses import replace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from ap_trainer_rddlgn import (  # noqa: E402
    LayerCache,
    RDDLGNAPTrainer,
    _ap_backward_stack,
    _init_hidden,
    _stack_forward,
)

# トレース正規化モード
TRACE_NORM_NONE = 'none'                 # 素のトレース(lambda=0で基底クラス一致)
TRACE_NORM_ONE_MINUS_LAMBDA = 'one_minus_lambda'  # (1-lambda)スケール
TRACE_NORM_SEQ_LEN = 'seq_len'           # 系列長Tで除算
VALID_TRACE_NORMS = (
    TRACE_NORM_NONE, TRACE_NORM_ONE_MINUS_LAMBDA, TRACE_NORM_SEQ_LEN,
)


def _norm_factor(trace_norm: str, lam: float, seq_len: int) -> float:
    """トレースに掛ける正規化係数を返す。"""
    if trace_norm == TRACE_NORM_NONE:
        return 1.0
    if trace_norm == TRACE_NORM_ONE_MINUS_LAMBDA:
        return 1.0 - lam
    if trace_norm == TRACE_NORM_SEQ_LEN:
        return 1.0 / max(1, seq_len)
    raise ValueError(f'未知のtrace_norm: {trace_norm}')


def _accumulate_trace(
    trace: list[torch.Tensor] | None,
    caches: list[LayerCache],
    lam: float,
) -> list[torch.Tensor]:
    """basisトレースをin-placeで更新して返す。

    trace_l <- lambda * trace_l + basis_l  (各層l)
    trace が None なら basis のクローンで初期化(=最初のstep)。
    """
    if trace is None:
        return [c.basis.clone() for c in caches]
    for l, c in enumerate(caches):
        trace[l].mul_(lam).add_(c.basis)
    return trace


def _trace_caches(
    caches: list[LayerCache],
    trace: list[torch.Tensor],
    factor: float,
) -> list[LayerCache]:
    """basisをトレース(×正規化係数)に差し替えたLayerCache列を作る。

    _ap_backward_stack / _update_weights はbasis以外(layer/gathered/output)を
    重み更新に使わないため、basisのみ差し替えれば更新にトレースが反映される。
    入力側誤差の逆伝播はbasis非依存なので影響しない。
    """
    scaled = trace if factor == 1.0 else [t * factor for t in trace]
    return [replace(c, basis=scaled[l]) for l, c in enumerate(caches)]


class EligibilityRDDLGNAPTrainer(RDDLGNAPTrainer):
    """適格度トレース付きRDDLGN APトレーナー。"""

    def __init__(
        self,
        model,
        lr: float = 0.1,
        update_mode: int = 1,
        pad_token: int = 0,
        eligibility_lambda: float = 0.0,
        trace_norm: str = TRACE_NORM_NONE,
    ):
        super().__init__(
            model, lr=lr, update_mode=update_mode, pad_token=pad_token,
        )
        if not 0.0 <= eligibility_lambda <= 1.0:
            raise ValueError(
                f'eligibility_lambdaは[0,1]: {eligibility_lambda}'
            )
        if trace_norm not in VALID_TRACE_NORMS:
            raise ValueError(
                f'trace_normは{VALID_TRACE_NORMS}: {trace_norm}'
            )
        self.eligibility_lambda = eligibility_lambda
        self.trace_norm = trace_norm
        # メモリ検証用: 直近train_stepで確保したトレース要素数(Tに非依存のはず)
        self.last_trace_numel = 0

    def _encode_k_layers_traced(
        self, n_out: torch.Tensor, batch: int, device: torch.device,
    ) -> tuple[torch.Tensor, list[LayerCache], list[torch.Tensor]]:
        """K-layers再帰forward。最終stepキャッシュとbasisトレースを返す。"""
        seq_len = n_out.shape[1]
        k_out_dim = self.model.encoder.k_out_dim
        k_prev = _init_hidden(
            batch, k_out_dim, device,
            self.model.config.init_mode, self.model.config.init_scale,
        )
        lam = self.eligibility_lambda
        k_caches_final = None
        k_trace = None
        for t in range(seq_len):
            k_input = torch.cat([n_out[:, t, :], k_prev], dim=-1)
            k_prev, k_caches = _stack_forward(
                self.model.encoder.k_layers, k_input
            )
            k_trace = _accumulate_trace(k_trace, k_caches, lam)
            k_caches_final = k_caches
        return k_prev, k_caches_final, k_trace

    def train_step(
        self,
        src: torch.Tensor,
        tgt_full: torch.Tensor,
    ) -> dict:
        """適格度トレース付き1バッチ訓練ステップ。

        基底クラスtrain_stepを複製し、以下2箇所のみ改変:
          (A) K-layers更新: 最終stepのbasis -> トレース trace_K(T_enc)
          (B) P-layers更新: 各stepのbasis -> トレース trace_P(t)
        """
        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        lam = self.eligibility_lambda

        with torch.no_grad():
            batch = src.shape[0]
            device = src.device
            tgt_len = tgt_input.shape[1]
            src_len = src.shape[1]
            vocab_size = self.model.config.vocab_size

            # === Encoder Forward ===
            src_emb = self._embedding_forward(self.model.src_embedding, src)
            n_out, n_caches_per_t = self._encode_n_layers(src_emb)
            # (A) K-layers: basisトレースを蓄積
            context, k_caches_final, k_trace = self._encode_k_layers_traced(
                n_out, batch, device
            )

            # === Decoder Forward + 各timestepでAP更新 ===
            tgt_emb = self._embedding_forward(
                self.model.tgt_embedding, tgt_input
            )
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
            p_trace = None  # (B) P-layers basisトレース

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
                probs = F.softmax(logits_t, dim=-1)
                target_oh = F.one_hot(
                    label_t, num_classes=vocab_size
                ).float()
                class_error = target_oh - probs
                class_error = class_error * mask.float().unsqueeze(-1)

                tau = self._group_sum.tau
                m_out_dim = self.model.decoder.m_layers.out_dim
                n_per = m_out_dim // vocab_size
                neuron_error = (
                    class_error.repeat_interleave(n_per, dim=-1) / tau
                )

                # M-layers AP backward(現行仕様: step独立、basisそのまま)
                m_input_error = _ap_backward_stack(
                    m_caches, self._m_reach, neuron_error,
                    self.lr, self.update_mode,
                )

                p_dim = self._p_out_dim
                c_dim = self._context_dim
                p_t_error = m_input_error[:, :p_dim]
                context_error_accum += m_input_error[:, p_dim:p_dim + c_dim]
                l_t_error_from_m = m_input_error[:, p_dim + c_dim:]

                # (B) P-layers: basisトレースを更新し、トレースで更新
                p_trace = _accumulate_trace(p_trace, p_caches, lam)
                p_factor = _norm_factor(self.trace_norm, lam, tgt_len)
                p_caches_used = _trace_caches(p_caches, p_trace, p_factor)
                p_input_error = _ap_backward_stack(
                    p_caches_used, self._p_reach, p_t_error,
                    self.lr, self.update_mode,
                )

                # P入力 = [p_prev | context | l_t]。p_prev誤差は破棄(現行仕様)
                context_error_accum += p_input_error[:, p_dim:p_dim + c_dim]
                l_t_error_from_p = p_input_error[:, p_dim + c_dim:]

                # L-layers AP backward → Embedding更新(現行仕様: step独立)
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
            # (A) 蓄積context誤差 → K-layers: 最終stepのbasisをトレースに置換
            k_factor = _norm_factor(self.trace_norm, lam, src_len)
            k_caches_used = _trace_caches(k_caches_final, k_trace, k_factor)
            k_input_error = _ap_backward_stack(
                k_caches_used, self._k_reach, context_error_accum,
                self.lr, self.update_mode,
            )

            n_out_dim = self.model.config.n_layer_sizes[-1]
            n_final_error = k_input_error[:, :n_out_dim]

            n_input_error = _ap_backward_stack(
                n_caches_per_t[src_len - 1], self._n_reach, n_final_error,
                self.lr, self.update_mode,
            )
            self._update_embedding(
                self.model.src_embedding, src[:, -1], n_input_error
            )

            # メモリ検証: トレースはK/P各層のbasis同形状のみ(Tに非依存)
            self.last_trace_numel = (
                sum(t.numel() for t in k_trace)
                + (sum(t.numel() for t in p_trace) if p_trace else 0)
            )

        avg_loss = total_loss / tgt_len if tgt_len > 0 else 0.0
        acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return {'loss': avg_loss, 'accuracy': acc}
