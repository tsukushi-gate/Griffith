"""
RDDLGNClassifier用AP法（Akiho Propagation）トレーナー。

RDDLGNClassifier（LogicEmbedding → RDDLGNEncoder(N/K-layers) → head(LutStack)
→ GroupSum、テキスト分類向け）をAP法(BP-free)で学習する。ap_trainer_rddlgn.py
（seq2seqのRDDLGN本体、encoder/decoder付き）が持つreach-map/forward/backward
ユーティリティ（_build_stack_reach_maps, _stack_forward, _ap_backward_stack）
をそのまま再利用し、RDDLGNClassifierのシンプルな構造
（decoder無し、自己回帰ループ無し、1回のforwardで分類ロジットが出る）
に合わせて訓練ループを新規に組む。

train_kd_sst2.pyのstudent='rddlgn-enc'（RDDLGNClassifier）と同一のモデル構築を
使う。ap_trainer.py/ap_trainer_cnn.pyの出力誤差計算（one_hot(target)-softmax
→ GroupSum逆）をそのまま踏襲し、KDはtrain_ap_rddlgn.py/RDDLGNAPTrainerと同じ
`target_dist = kd_alpha*one_hot + (1-kd_alpha)*teacher_probs` 混合方式を使う。

使用例:
    model = RDDLGNClassifier(config, n_classes=2)
    trainer = RDDLGNClassifierAPTrainer(model, lr=0.1)
    for tokens, labels in dataloader:
        metrics = trainer.train_step(tokens, labels)
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ap_trainer_rddlgn import (
    _ap_backward_stack,
    _build_stack_reach_maps,
    _stack_forward,
)

from griffith_logic import RDDLGNClassifier
from griffith_logic.lut_layer import LutLayer


class RDDLGNClassifierAPTrainer:
    """RDDLGNClassifier（テキスト分類）用AP法トレーナー。

    LogicEmbedding → RDDLGNEncoder(N-layers/K-layers) → head(LutStack)
    → GroupSum の一直線構造。decoderの自己回帰ループが無いため、
    RDDLGNAPTrainerよりも単純な1パスforward/backwardで済む。
    """

    def __init__(
        self,
        model: RDDLGNClassifier,
        lr: float = 0.1,
        update_mode: int = 1,
    ):
        self.model = model
        self.lr = lr
        self.update_mode = update_mode

        # 全LutLayerがindependentパラメータ化・wiring='fixed'であることを確認
        # (AP法の前提。ap_trainer.py/ap_trainer_rddlgn.pyと同じ検証)
        for name, module in model.named_modules():
            if isinstance(module, LutLayer):
                if module.parameterization != 'independent':
                    raise ValueError(
                        f'AP法はindependentパラメータ化を要求: '
                        f'{name} is {module.parameterization}'
                    )
                if getattr(module, 'wiring', 'fixed') != 'fixed':
                    raise ValueError(
                        f'AP法はwiring="fixed"のLutLayerのみ対応: '
                        f'{name} wiring={module.wiring!r}'
                    )

        # N-layers / K-layers（エンコーダ）と head（分類ヘッド）のreach-mapを
        # 事前構築（ap_trainer_rddlgn.pyのRDDLGNAPTrainerと同じ手法）
        self._n_reach = _build_stack_reach_maps(model.encoder.n_layers)
        self._k_reach = _build_stack_reach_maps(model.encoder.k_layers)
        self._head_reach = _build_stack_reach_maps(model.head)

        self._group_sum = model.group_sum
        self._n_out_dim = model.config.n_layer_sizes[-1]
        self._k_out_dim = model.config.k_layer_sizes[-1]

        # LutLayerの重みを[0,1]で再初期化（AP法規約、ap_trainer_rddlgn.py踏襲）
        self._init_lut_weights()

    def _init_lut_weights(self):
        """全LutLayerの重みを[0,1]のランダム値で初期化。"""
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, LutLayer):
                    module.weights.data.uniform_(0, 1)

    def _embedding_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """LogicEmbedding forward。sigmoidを適用（train_ap_rddlgn系と同じ規約）。"""
        e = self.model.embedding.embedding(tokens)
        return torch.sigmoid(e)

    def _encode_n_layers(self, src_emb: torch.Tensor):
        """N-layers: 各timestepを独立にforward。キャッシュをtimestep別に保存。"""
        batch, seq_len, d_emb = src_emb.shape
        n_caches_per_t = []
        outputs = []

        for t in range(seq_len):
            h_t = src_emb[:, t, :]
            out_t, caches_t = _stack_forward(self.model.encoder.n_layers, h_t)
            n_caches_per_t.append(caches_t)
            outputs.append(out_t)

        n_out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_out_dim)
        return n_out, n_caches_per_t

    def _encode_k_layers(self, n_out: torch.Tensor, batch: int, device: torch.device):
        """K-layers: リカレント。最終stepのキャッシュのみ返す（末尾コンテキストを使う規約）。"""
        from griffith_logic.rddlgn import _init_hidden

        seq_len = n_out.shape[1]
        k_prev = _init_hidden(
            batch, self._k_out_dim, device,
            self.model.config.init_mode, self.model.config.init_scale,
        )

        k_caches_final = None
        for t in range(seq_len):
            k_input = torch.cat([n_out[:, t, :], k_prev], dim=-1)
            k_prev, k_caches = _stack_forward(self.model.encoder.k_layers, k_input)
            k_caches_final = k_caches

        return k_prev, k_caches_final

    def _update_embedding(self, token_ids: torch.Tensor, error: torch.Tensor):
        """Embedding重みをAP風に更新（RDDLGNAPTrainer._update_embeddingと同一）。

        sigmoid前のlogit空間で: W[token_id] += mean(error) * lr。
        """
        batch = token_ids.shape[0]
        with torch.no_grad():
            for b in range(batch):
                tid = token_ids[b].item()
                self.model.embedding.embedding.weight.data[tid] += error[b] * self.lr

    def train_step(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
        kd_alpha: float = 1.0,
    ) -> dict:
        """1バッチの訓練ステップ。

        Args:
            tokens: (batch, seq_len) トークンID
            labels: (batch,) 正解ラベル（整数、pseudo-labels時はteacher argmax）
            teacher_probs: (batch, n_classes) teacherのsoftmax確率分布。
                Noneの場合は既存のone-hotターゲットのみを使用（非KD）。
            kd_alpha: one-hotターゲットとteacher分布の混合率。
                target_dist = kd_alpha * one_hot + (1 - kd_alpha) * teacher_probs。
                kd_alpha=1.0（既定）はteacher_probsを渡してもone-hotのみと完全一致する
                （train_ap_rddlgn.py/RDDLGNAPTrainerと同一の混合規約）。

        Returns:
            {'loss': float, 'accuracy': float}
        """
        with torch.no_grad():
            batch = tokens.shape[0]
            device = tokens.device
            n_classes = self.model.n_classes

            # === Forward（キャッシュ収集） ===
            emb = self._embedding_forward(tokens)
            n_out, n_caches_per_t = self._encode_n_layers(emb)
            context, k_caches_final = self._encode_k_layers(n_out, batch, device)

            head_out, head_caches = _stack_forward(self.model.head, context)
            logits = self._group_sum(head_out)

            # === メトリクス ===
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
            loss = F.cross_entropy(logits, labels).item()

            # === 出力誤差 ===
            target_oh = F.one_hot(labels, num_classes=n_classes).float().to(device)
            if teacher_probs is None:
                target_dist = target_oh
            else:
                target_dist = kd_alpha * target_oh + (1.0 - kd_alpha) * teacher_probs
            class_error = target_dist - probs  # (batch, n_classes)

            # GroupSum逆
            tau = self._group_sum.tau
            head_out_dim = self.model.head.out_dim
            n_per = head_out_dim // n_classes
            neuron_error = class_error.repeat_interleave(n_per, dim=-1) / tau

            # === head(LutStack) AP backward → head入力(=context)誤差 ===
            context_error = _ap_backward_stack(
                head_caches, self._head_reach, neuron_error,
                self.lr, self.update_mode,
            )

            # === K-layers AP backward（最終timestepのみ、リカレント規約） ===
            k_input_error = _ap_backward_stack(
                k_caches_final, self._k_reach, context_error,
                self.lr, self.update_mode,
            )
            # K入力 = [n_final | k_prev] → n_final誤差のみ使用（時間方向には辿らない）
            n_final_error = k_input_error[:, :self._n_out_dim]

            # === N-layers AP backward（最終timestepのみ） ===
            seq_len = tokens.shape[1]
            n_input_error = _ap_backward_stack(
                n_caches_per_t[seq_len - 1], self._n_reach, n_final_error,
                self.lr, self.update_mode,
            )
            self._update_embedding(tokens[:, -1], n_input_error)

        return {'loss': loss, 'accuracy': acc}

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        device: str,
        use_soft: bool = False,
    ) -> float:
        """正解ラベルに対する分類精度を計算する。

        Args:
            use_soft: True=sigmoid連続値のまま評価(soft)、
                False=Heaviside離散化評価(hard、AP法のweights.data>0.5に相当)。
                AP法はLutLayer.forwardを呼ばず独自forwardのため、
                weights.dataを一時的にbinarizeしてhard評価する。
        """
        saved_weights = {}
        if not use_soft:
            for i, module in enumerate(self.model.modules()):
                if isinstance(module, LutLayer):
                    saved_weights[i] = module.weights.data.clone()
                    module.weights.data = (module.weights.data > 0.5).float()

        correct = 0
        total = 0
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            batch = tokens.shape[0]

            emb = self._embedding_forward(tokens)
            if not use_soft:
                emb = (emb > 0.5).float()
            n_out, _ = self._encode_n_layers(emb)
            context, _ = self._encode_k_layers(n_out, batch, device)
            head_out, _ = _stack_forward(self.model.head, context)
            logits = self._group_sum(head_out)

            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.shape[0]

        if not use_soft:
            for i, module in enumerate(self.model.modules()):
                if isinstance(module, LutLayer) and i in saved_weights:
                    module.weights.data = saved_weights[i]

        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def evaluate_teacher_agreement(
        self,
        loader,
        teacher_val_logits: torch.Tensor,
        device: str,
        use_soft: bool = False,
    ) -> float:
        """teacher-student のargmax一致率を検証データ全体で計算する（インデックス整列前提）。

        teacher_val_logits: [N_val, C]（validation全体、get_or_compute_logitsの
        インデックス順）。loader は shuffle=False の validation loader である必要がある。
        """
        saved_weights = {}
        if not use_soft:
            for i, module in enumerate(self.model.modules()):
                if isinstance(module, LutLayer):
                    saved_weights[i] = module.weights.data.clone()
                    module.weights.data = (module.weights.data > 0.5).float()

        agree = 0
        total = 0
        offset = 0
        for tokens, _labels in loader:
            bsz = tokens.shape[0]
            tokens = tokens.to(device)

            emb = self._embedding_forward(tokens)
            if not use_soft:
                emb = (emb > 0.5).float()
            n_out, _ = self._encode_n_layers(emb)
            context, _ = self._encode_k_layers(n_out, bsz, device)
            head_out, _ = _stack_forward(self.model.head, context)
            logits = self._group_sum(head_out)

            student_preds = logits.argmax(dim=-1).cpu()
            teacher_preds = teacher_val_logits[offset:offset + bsz].argmax(dim=-1)
            agree += (student_preds == teacher_preds).sum().item()
            total += bsz
            offset += bsz

        if not use_soft:
            for i, module in enumerate(self.model.modules()):
                if isinstance(module, LutLayer) and i in saved_weights:
                    module.weights.data = saved_weights[i]

        return agree / total if total > 0 else 0.0
