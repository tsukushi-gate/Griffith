"""RNN→RDDLGN構造転写(D-RNN1): 再帰teacherの構造転移はhard精度で効くか。

(算術MLP→DLGN)は「教師の関数がロジック表現可能(fan-in枝刈り後のMLP)なら
構造転写(transcribe→recover soft磨き)がIO-KD/scratchを圧倒する」ことを示した。
本スクリプトは同じ
パラダイムを**再帰**teacher(vanilla nn.RNNベースのシーケンスモデル)へ拡張し、
「連続隠れ状態+tanh非線形性+再帰」という新しい構造がMLPと同様に転写できるか
を検証する、事前の下調べでは**否定的結果になり得る**境界ケース実験である
(閾値ゲート転写は隠れ状態を毎ステップ1bitへ丸めるため、tanhの連続的な記憶が
再帰的に失われる懸念がある)。

## タスク・teacher

copyタスク(vocab≈10, seq_len≈4): 各ステップでトークンx_tを読み、同じ位置の
トークンをそのまま出力する(CopyDataset)。teacher = 二値化埋め込み
(LogicEmbeddingと同じsigmoid+閾値、STE)→ vanilla RNNセル(tanh、weight_ih/
weight_hh/bias_ih/bias_hhを直接保持)→ 各ステップの隠れ状態から
nn.Linear(d_hidden, vocab)で読み出し。

## 構造転写のターゲット: native RDDLGN再帰LutStack

RDDLGNEncoder.k_layers(griffith_logic/models/rddlgn.py)は
「(N出力+K出力) → k_layer_sizes」のLutStack(LutLayer列)をタイムステップ
ループで適用する再帰そのものである。本実験はteacherのRNNセル
(1層、d_hidden出力)を **1層のLutStack**(= LutLayer 1枚)として構築し、
`transcribe_rnn_cell`が返す真理値表・配線を`ir.weights.restore_lut_layer`で
直接書き込む(training-freeのin-place初期化)。これは
「RDDLGNの再帰LutStackへネイティブに初期化する」経路であり、
「固定長unrolled serial DLGNへのフォールバック」ではない
(タスク側が要求するフォールバック条件「配線が侵襲的すぎる場合」に該当しない:
LutLayerは元々indices/weightsをin-placeで書き換え可能に設計されている)。

studentモデル(RnnStructDlgnStudent)は以下の3層で構成する(RDDLGNのK/M層の
縮小版、encoder/decoderの分離やN/L/P層は本タスクの最小検証には不要と判断し
省略。理由は本ファイルdocstring末尾の「省略した理由」参照):
    embed:  LogicEmbedding(vocab, d_emb)                — token→0/1ベクトル
    k_layer: LutLayer(d_emb+d_hidden -> d_hidden, fanin) — 再帰(teacher RNNセル転写先)
    head:   LutStack(d_hidden -> head_sizes) + GroupSum(k=vocab) — 各ステップ読み出し

## 4アーム比較(arm/table規約を踏襲)

    A(scratch)   ランダム初期化のRnnStructDlgnStudentをラベルCEで直接学習。
    B(IO-KDのみ) 同じランダム初期化を系列KD(convert.distill.kd_kl_loss_seq)で学習。
                 構造転写なし。
    C(本命: 構造転写+soft磨き) teacher RNNセルをtranscribe_rnn_cellで転写し
                 k_layerをrestore_lut_layerでin-place初期化(parameterization=
                 'warp'、真理値表を厳密表現できる方式)。headはランダム初期化の
                 まま(readoutはteacherのnn.Linear読み出しに対応する転写機構が
                 ないため、教師強制ラベルCEで軽くsoft磨きする)。
    teacher      teacher自身のhard/soft copy精度(比較の天井)。

exact-match精度(系列全体が一致して初めて正解、CopyDataset規約)で報告する。

## 省略した理由(RDDLGN本体の全構造でなく縮小版を使う理由)

RDDLGNEncoder/Decoderの全機構(N/L/P層、エンコーダ・デコーダ分離、teacher
forcingのBOS/EOS)はWMT14スケールの翻訳を想定した設計であり、copyタスクの
「各ステップでその場のトークンを出力する」という単純な写像には過剰である。
本タスクが検証したい対象は**K-layer再帰そのもの**(RDDLGNの再帰核)への構造
転写が効くかどうかであり、全体をラップするEncoder/Decoderクラスをそのまま
使うと(a)エンコーダ全体を読んでからデコードする設計のため「同時刻の入力を
即座に出力する」copyの意味論と噛み合わない、(b)N/L/P層という転写対象で
ない層が増えるだけで検証対象がぼやける。よってRDDLGNと**同じLutStack/
LutLayer部品**を使い、re-implementationでなく最小構成で組んだ
(griffith_logic.models.rddlgn.LutStackをそのままimportして使用)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

JST = timezone(timedelta(hours=9))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.synthetic import SPECIAL_TOKENS, CopyDataset
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.distill import kd_kl_loss_seq
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import sparsify_fanin, threshold_truth_tables
from griffith_logic.convert.transcribe_rnn import transcribe_rnn_cell
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.weights import restore_lut_layer
from griffith_logic.layers.lut import LutLayer
from griffith_logic.models.rddlgn import LogicEmbedding, LutStack

load_dotenv()

EPS = 1e-8


# ============================================================
# データ: copyタスク(CopyDataset、trainとevalで独立サンプル)
# ============================================================

def make_copy_loaders(vocab_size: int, seq_len: int, n_train: int, n_eval: int,
                      batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train_set = CopyDataset(vocab_size, seq_len, n_train)
    eval_set = CopyDataset(vocab_size, seq_len, n_eval)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g, drop_last=True,
    )
    eval_loader = DataLoader(eval_set, batch_size=max(n_eval, 1), shuffle=False)
    return train_loader, eval_loader


# ============================================================
# teacher: 二値化埋め込み + vanilla RNNセル(tanh) + 線形読み出し
# ============================================================

class RnnCopyTeacher(nn.Module):
    """copyタスク用のvanilla-RNN teacher。

    embed: LogicEmbeddingと同じsigmoid+閾値(0.5)二値化(STE勾配)。
    cell:  weight_ih/weight_hh/bias_ih/bias_hhを直接保持するRNNセル
           (nn.RNNCellと同一パラメータ形状規約: weight_ih (d_hidden,d_in)、
           weight_hh (d_hidden,d_hidden))。tanh活性(vanilla RNN標準)。
           transcribe_rnn_cellがこの4テンソルをそのまま受け取れる。
    head:  nn.Linear(d_hidden, vocab)の各ステップ読み出し(copyラベルを予測)。
    """

    def __init__(self, vocab_size: int, d_emb: int, d_hidden: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_hidden = d_hidden

        self.embedding = nn.Embedding(vocab_size, d_emb)
        self.weight_ih = nn.Parameter(torch.empty(d_hidden, d_emb))
        self.weight_hh = nn.Parameter(torch.empty(d_hidden, d_hidden))
        self.bias_ih = nn.Parameter(torch.zeros(d_hidden))
        self.bias_hh = nn.Parameter(torch.zeros(d_hidden))
        nn.init.uniform_(self.weight_ih, -1.0 / d_hidden ** 0.5, 1.0 / d_hidden ** 0.5)
        nn.init.uniform_(self.weight_hh, -1.0 / d_hidden ** 0.5, 1.0 / d_hidden ** 0.5)
        self.head = nn.Linear(d_hidden, vocab_size)

    def _embed_binary(self, tokens: torch.Tensor) -> torch.Tensor:
        """LogicEmbeddingと同じsigmoid+閾値二値化(STE)。(B,S)->(B,S,d_emb)。"""
        e = torch.sigmoid(self.embedding(tokens))
        if self.training:
            # STE: forward hard threshold, backward straight-through
            hard = (e > 0.5).to(e.dtype)
            return e + (hard - e).detach()
        return (e > 0.5).to(e.dtype)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, S) 整数トークンID

        Returns:
            logits: (B, S, vocab) 各ステップの読み出しロジット
            h_all:  (B, S, d_hidden) 各ステップの隠れ状態(tanh後、[-1,1])
        """
        batch, seq_len = tokens.shape
        x = self._embed_binary(tokens)  # (B, S, d_emb) in {0,1}
        h = torch.zeros(batch, self.d_hidden, device=tokens.device, dtype=x.dtype)
        h_list = []
        for t in range(seq_len):
            z = x[:, t, :] @ self.weight_ih.T + h @ self.weight_hh.T \
                + self.bias_ih + self.bias_hh
            h = torch.tanh(z)
            h_list.append(h)
        h_all = torch.stack(h_list, dim=1)  # (B, S, d_hidden)
        logits = self.head(h_all)
        return logits, h_all

    @torch.no_grad()
    def hard_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """閾値ゲート意味論(tanh(z)>0 <=> z>0)での完全二値hard forward。

        transcribe_rnn_cellが転写する関数と厳密に同じ計算(埋め込みも隠れ状態も
        {0,1}で保持し、readoutのみ実数nn.Linearをargmaxする)。厳密性assertに使う。
        """
        self.eval()
        batch, seq_len = tokens.shape
        x = self._embed_binary(tokens)  # (B,S,d_emb) 0/1 (evalなので閾値のみ)
        h = torch.zeros(batch, self.d_hidden, device=tokens.device, dtype=x.dtype)
        preds = []
        for t in range(seq_len):
            z = x[:, t, :] @ self.weight_ih.T + h @ self.weight_hh.T \
                + self.bias_ih + self.bias_hh
            h = (z > 0).to(x.dtype)  # 閾値ゲート意味論(transcribe_rnn_cellと同一)
            logit = self.head(h)
            preds.append(logit.argmax(-1))
        return torch.stack(preds, dim=1)  # (B, S)


def train_teacher(teacher: RnnCopyTeacher, train_loader: DataLoader,
                  eval_loader: DataLoader, *, epochs: int, lr: float,
                  device: str) -> float:
    """teacherをcopyタスクで学習し、最終exact-match精度(soft forward)を返す。"""
    teacher.to(device).train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    for _ in range(epochs):
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits, _ = teacher(src)
            loss = F.cross_entropy(logits.reshape(-1, teacher.vocab_size), tgt.reshape(-1))
            loss.backward()
            opt.step()
    return exact_match_acc_soft(teacher, eval_loader, device)


@torch.no_grad()
def exact_match_acc_soft(teacher: RnnCopyTeacher, loader: DataLoader, device: str) -> float:
    teacher.eval()
    correct, total = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits, _ = teacher(src)
        pred = logits.argmax(-1)
        correct += int((pred == tgt).all(dim=1).sum())
        total += int(src.shape[0])
    teacher.train()
    return correct / total if total else 0.0


@torch.no_grad()
def exact_match_acc_hard_teacher(teacher: RnnCopyTeacher, loader: DataLoader,
                                 device: str) -> float:
    """teacherの閾値ゲート意味論(hard_forward)でのexact-match精度。"""
    correct, total = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        pred = teacher.hard_forward(src)
        correct += int((pred == tgt).all(dim=1).sum())
        total += int(src.shape[0])
    return correct / total if total else 0.0


# ============================================================
# teacher RNNセルのfan-in枝刈り(combined [x_t;h_prev]の結合fan-in)
# ============================================================

def combined_fanin_prune_(teacher: RnnCopyTeacher, target_fanin: int,
                          train_loader: DataLoader, *,
                          epochs_per_stage: int, lr: float, device: str) -> dict:
    """weight_ih/weight_hhを[x_t;h_prev]結合fan-inがtarget_fanin以下になるまで
    段階的に枝刈りする(transcribe.pyのgentle_fanin_scheduleと同じ倍々刻み方針)。

    convert/prune.pyのapply_fanin_pruningは単一nn.Linearの行内fan-inしか
    扱えず、weight_ih/weight_hhを跨いだ結合fan-in制約には非対応のため、本関数を
    専用に実装する(transcribe_rnn_cellのsparsify_fanin適用と同じ「結合ベクトル
    の|W|上位k本」意味論に合わせる目的の、実験スクリプト固有のグルーコード)。

    Returns:
        stage_history dict({'stages': [...], 'skipped': bool})
    """
    d_hidden, d_in = teacher.weight_ih.shape
    combined_width = d_in + d_hidden
    if combined_width <= target_fanin:
        return {'stages': [], 'skipped': True}

    # 倍々刻みスケジュール(gentle_fanin_scheduleと同じ思想)
    fanins = [combined_width]
    cur = combined_width
    while cur > target_fanin:
        nxt = max(target_fanin, cur // 2)
        fanins.append(nxt)
        cur = nxt

    history = {'stages': [], 'skipped': False}
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)

    for k in fanins[1:]:
        with torch.no_grad():
            w_combined = torch.cat([teacher.weight_ih.data, teacher.weight_hh.data], dim=1)
            order = w_combined.abs().argsort(dim=1, descending=True)
            keep = order[:, :k]  # (d_hidden, k)
            mask = torch.zeros_like(w_combined)
            mask.scatter_(1, keep, 1.0)
            mask_ih = mask[:, :d_in]
            mask_hh = mask[:, d_in:]
            teacher.weight_ih.data.mul_(mask_ih)
            teacher.weight_hh.data.mul_(mask_hh)

        h_ih = teacher.weight_ih.register_hook(lambda g, m=mask_ih: g * m)
        h_hh = teacher.weight_hh.register_hook(lambda g, m=mask_hh: g * m)

        teacher.train()
        for _ in range(epochs_per_stage):
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                opt.zero_grad()
                logits, _ = teacher(src)
                loss = F.cross_entropy(
                    logits.reshape(-1, teacher.vocab_size), tgt.reshape(-1),
                )
                loss.backward()
                opt.step()

        h_ih.remove()
        h_hh.remove()
        with torch.no_grad():
            teacher.weight_ih.data.mul_(mask_ih)
            teacher.weight_hh.data.mul_(mask_hh)

        acc = exact_match_acc_soft(teacher, train_loader, device)
        history['stages'].append({'fanin': k, 'train_acc': acc})

    return history


# ============================================================
# student: RnnStructDlgnStudent(RDDLGN K-layer相当の再帰 + readout head)
# ============================================================

class RnnStructDlgnStudent(nn.Module):
    """copyタスク用の最小再帰DLGN(RDDLGN K-layer + M-layer相当、モジュールdocstring参照)。

    embed:   LogicEmbedding(vocab, d_emb)
    k_layer: LutStack(d_emb+d_hidden -> [d_hidden], fanin)      — 再帰(1層)
    head:    LutStack(d_hidden -> head_sizes) + GroupSum(k=vocab) — 読み出し
    """

    def __init__(self, vocab_size: int, d_emb: int, d_hidden: int, *,
                fanin: int, head_sizes: list[int], parameterization: str = 'warp',
                device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_hidden = d_hidden

        self.embedding = LogicEmbedding(vocab_size, d_emb)
        self.k_layer = LutLayer(
            d_emb + d_hidden, d_hidden, n_inputs=fanin, device=device,
            parameterization=parameterization, connections='random',
        )

        head_sizes = list(head_sizes)
        group_factor = max(1, head_sizes[-1] // vocab_size)
        head_sizes[-1] = vocab_size * group_factor
        self.head = LutStack(
            d_hidden, head_sizes, n_inputs=fanin, device=device,
            parameterization=parameterization, connections='random',
        )
        tau = max(1.0, (head_sizes[-1] / vocab_size) ** 0.5)
        self.group_sum = GroupSum(k=vocab_size, tau=tau)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        device = tokens.device
        x = self.embedding(tokens)  # (B, S, d_emb) in [0,1] (soft) or {0,1} (eval)
        h = torch.full((batch, self.d_hidden), 0.5, device=device, dtype=x.dtype)
        logits_list = []
        for t in range(seq_len):
            combined = torch.cat([x[:, t, :], h], dim=-1)
            h = self.k_layer(combined)
            m = self.head(h)
            logits_list.append(self.group_sum(m))
        return torch.stack(logits_list, dim=1)  # (B, S, vocab)


def build_scratch_student(vocab_size: int, d_emb: int, d_hidden: int, *,
                          fanin: int, head_sizes: list[int], device: str,
                          seed: int) -> RnnStructDlgnStudent:
    torch.manual_seed(seed)
    return RnnStructDlgnStudent(
        vocab_size, d_emb, d_hidden, fanin=fanin, head_sizes=head_sizes,
        parameterization='warp', device=device,
    ).to(device)


@torch.no_grad()
def exact_match_acc_student(model: RnnStructDlgnStudent, loader: DataLoader,
                            device: str, *, hard: bool) -> float:
    """studentのexact-match精度。hard=Trueならmodel.eval()(自動離散化)、
    hard=Falseならmodel.train()相当のsoft forwardで測る。"""
    was_training = model.training
    model.eval() if hard else model.train()
    correct, total = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        pred = logits.argmax(-1)
        correct += int((pred == tgt).all(dim=1).sum())
        total += int(src.shape[0])
    model.train(was_training)
    return correct / total if total else 0.0


# ============================================================
# arm A: scratch
# ============================================================

def run_arm_a(train_loader, eval_loader, *, vocab_size, d_emb, d_hidden, args,
             device, seed) -> dict:
    model = build_scratch_student(
        vocab_size, d_emb, d_hidden, fanin=args.fanin, head_sizes=args.head_sizes,
        device=device, seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.epochs):
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(src)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            opt.step()
    soft_acc = exact_match_acc_student(model, eval_loader, device, hard=False)
    hard_acc = exact_match_acc_student(model, eval_loader, device, hard=True)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


# ============================================================
# arm B: IO-KDのみ(構造転写なし)
# ============================================================

def run_arm_b(train_loader, eval_loader, teacher, *, vocab_size, d_emb, d_hidden,
             args, device, seed) -> dict:
    model = build_scratch_student(
        vocab_size, d_emb, d_hidden, fanin=args.fanin, head_sizes=args.head_sizes,
        device=device, seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    teacher.eval()
    for _ in range(args.epochs):
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            with torch.no_grad():
                teacher_logits, _ = teacher(src)
            opt.zero_grad()
            logits = model(src)
            ce = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            kd = kd_kl_loss_seq(logits, teacher_logits, temperature=args.kd_temperature)
            loss = args.kd_alpha * ce + (1.0 - args.kd_alpha) * kd
            loss.backward()
            opt.step()
    soft_acc = exact_match_acc_student(model, eval_loader, device, hard=False)
    hard_acc = exact_match_acc_student(model, eval_loader, device, hard=True)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


# ============================================================
# arm C: 構造転写(native RDDLGN再帰LutStack初期化) + soft磨き
# ============================================================

def build_structural_student(teacher: RnnCopyTeacher, *, vocab_size, d_emb,
                             d_hidden, fanin, head_sizes, device, seed,
                             calib_tokens: torch.Tensor) -> tuple[RnnStructDlgnStudent, dict]:
    """teacher RNNセルをtranscribe_rnn_cellで転写し、k_layerをin-place初期化する。

    埋め込みは転写しない(LogicEmbeddingとteacherの二値化埋め込みは独立に
    初期化された別々のEmbeddingテーブルのため、ビット単位の対応が無い。
    teacherの埋め込みテーブルをstudentのLogicEmbedding.embeddingへ直接コピー
    することで、意味的に対応する二値表現をstudentへ引き継ぐ)。

    Returns:
        (student, transcribe_info): transcribe_infoはfan-in・厳密性assert結果。
    """
    torch.manual_seed(seed)
    student = RnnStructDlgnStudent(
        vocab_size, d_emb, d_hidden, fanin=fanin, head_sizes=head_sizes,
        parameterization='warp', device=device,
    ).to(device)

    # 埋め込みテーブルを引き継ぐ(teacherのsigmoid+閾値二値化意味論と
    # LogicEmbeddingのsigmoid+閾値二値化意味論が完全一致するため、
    # 同じ実数埋め込みテーブルをコピーすればstudent側でも同じbit列になる)。
    with torch.no_grad():
        student.embedding.embedding.weight.copy_(
            teacher.embedding.weight.detach().to(device)
        )

    # teacher RNNセルを較正データ上で転写
    calib_x = teacher._embed_binary(calib_tokens.to(device))  # (B,S,d_emb)
    with torch.no_grad():
        h = torch.zeros(calib_tokens.shape[0], d_hidden, device=device)
        combined_rows = []
        for t in range(calib_tokens.shape[1]):
            combined_rows.append(torch.cat([calib_x[:, t, :], h], dim=-1))
            z = calib_x[:, t, :] @ teacher.weight_ih.T + h @ teacher.weight_hh.T \
                + teacher.bias_ih + teacher.bias_hh
            h = (z > 0).to(calib_x.dtype)
        calib_combined = torch.cat(combined_rows, dim=0).cpu().numpy().astype(np.int64)

    result = transcribe_rnn_cell(
        teacher.weight_ih.detach().cpu(), teacher.weight_hh.detach().cpu(),
        teacher.bias_ih.detach().cpu(), teacher.bias_hh.detach().cpu(),
        n_inputs=fanin, calib=calib_combined,
    )
    restore_lut_layer(student.k_layer, result['indices'], result['truth_tables'])

    # 厳密性assert: 転写直後のk_layer(eval、離散化forward)が、
    # 較正データ上でteacher.hard_forwardの隠れ状態更新則と一致することを
    # 確認する(fan-in枝刈りが恒等でない場合は近似のため、一致率を報告するのみ
    # でSTOPはしない。恒等疎化の場合は100%一致が期待値)。
    student.eval()
    with torch.no_grad():
        h_student = torch.full(
            (calib_tokens.shape[0], d_hidden), 0.5, device=device,
        )
        h_teacher = torch.zeros(calib_tokens.shape[0], d_hidden, device=device)
        agree_bits = 0
        total_bits = 0
        for t in range(calib_tokens.shape[1]):
            x_t = calib_x[:, t, :]
            combined = torch.cat([x_t, h_student], dim=-1)
            h_student = student.k_layer(combined)
            z = x_t @ teacher.weight_ih.T + h_teacher @ teacher.weight_hh.T \
                + teacher.bias_ih + teacher.bias_hh
            h_teacher = (z > 0).to(x_t.dtype)
            agree_bits += int((h_student == h_teacher).sum())
            total_bits += h_student.numel()
    student.train()

    combined_width = d_emb + d_hidden
    transcribe_info = {
        'fanin': fanin,
        'combined_width': combined_width,
        'exact_expected': combined_width <= fanin,
        'k_layer_agreement': agree_bits / max(total_bits, 1),
    }
    return student, transcribe_info


def polish_structural_student(student: RnnStructDlgnStudent, train_loader,
                              *, vocab_size, args, device) -> None:
    """arm Cのsoft磨き: 転写初期化済みstudent全体を低LRのCEでfine-tuneする。

    `recover()`は直列feed-forward IR専用(group_sum終端のIRGraphを要求)
    で再帰モデルには非適用のため、ここではrecover()と同じ精神(良い離散化
    初期値からの軽いsoft micro-tune、hard_stは使わない)を、再帰student全体への
    直接end-to-end soft backpropとして実装する(知見: 良い初期値に対する
    hard_stは破壊的なので、既定selectionは'soft'のみを使うLutLayerのまま
    据え置く。K-layerはparameterization='warp'かつselection既定'soft'なので
    追加変更不要)。
    """
    opt = torch.optim.Adam(student.parameters(), lr=args.polish_lr)
    student.train()
    for _ in range(args.polish_epochs):
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = student(src)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            opt.step()


def run_arm_c(teacher, train_loader, eval_loader, calib_tokens, *, vocab_size,
             d_emb, d_hidden, args, device, seed) -> dict:
    student, transcribe_info = build_structural_student(
        teacher, vocab_size=vocab_size, d_emb=d_emb, d_hidden=d_hidden,
        fanin=args.fanin, head_sizes=args.head_sizes, device=device, seed=seed,
        calib_tokens=calib_tokens,
    )
    soft_acc_before = exact_match_acc_student(student, eval_loader, device, hard=False)
    hard_acc_before = exact_match_acc_student(student, eval_loader, device, hard=True)

    polish_structural_student(
        student, train_loader, vocab_size=vocab_size, args=args, device=device,
    )

    soft_acc_after = exact_match_acc_student(student, eval_loader, device, hard=False)
    hard_acc_after = exact_match_acc_student(student, eval_loader, device, hard=True)

    return {
        'soft_acc_before': soft_acc_before, 'hard_acc_before': hard_acc_before,
        'soft_acc_after': soft_acc_after, 'hard_acc_after': hard_acc_after,
        'transcribe_info': transcribe_info,
    }


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab_size = args.vocab_size
    seq_len = args.seq_len
    d_emb = args.d_emb
    d_hidden = args.d_hidden

    train_loader, eval_loader = make_copy_loaders(
        vocab_size, seq_len, args.num_train, args.num_eval, args.batch_size,
        seed=seed,
    )
    calib_tokens = next(iter(DataLoader(
        CopyDataset(vocab_size, seq_len, args.calib_samples), batch_size=args.calib_samples,
    )))[0]

    result: dict = {'seed': seed}

    # ---- teacher学習 ----
    torch.manual_seed(seed)
    teacher = RnnCopyTeacher(vocab_size, d_emb, d_hidden).to(device)
    teacher_soft_acc = train_teacher(
        teacher, train_loader, eval_loader, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device,
    )
    print(f'  [seed={seed}] teacher soft exact-match acc={teacher_soft_acc:.4f}')

    # ---- teacherのRNNセルfan-in枝刈り(combined width > fanin の場合のみ) ----
    prune_history = combined_fanin_prune_(
        teacher, args.fanin, train_loader,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr, device=device,
    )
    teacher_soft_acc = exact_match_acc_soft(teacher, eval_loader, device)
    teacher_hard_acc = exact_match_acc_hard_teacher(teacher, eval_loader, device)
    print(f'  [seed={seed}] teacher fan-in枝刈り後: soft={teacher_soft_acc:.4f} '
          f'hard(閾値ゲート意味論)={teacher_hard_acc:.4f} (fanin<={args.fanin})')
    result['teacher_soft_acc'] = teacher_soft_acc
    result['teacher_hard_acc'] = teacher_hard_acc
    result['prune_stages'] = prune_history.get('stages', [])

    # ---- arm A: scratch ----
    t0 = time.time()
    res_a = run_arm_a(
        train_loader, eval_loader, vocab_size=vocab_size, d_emb=d_emb,
        d_hidden=d_hidden, args=args, device=device, seed=seed + 1000,
    )
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm A(scratch): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} gap={res_a['gap']:+.4f} ({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ ----
    t0 = time.time()
    res_b = run_arm_b(
        train_loader, eval_loader, teacher, vocab_size=vocab_size, d_emb=d_emb,
        d_hidden=d_hidden, args=args, device=device, seed=seed + 2000,
    )
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm B(IO-KDのみ): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} gap={res_b['gap']:+.4f} ({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm C: 構造転写(native RDDLGN再帰LutStack) + soft磨き ----
    t0 = time.time()
    res_c = run_arm_c(
        teacher, train_loader, eval_loader, calib_tokens, vocab_size=vocab_size,
        d_emb=d_emb, d_hidden=d_hidden, args=args, device=device, seed=seed + 3000,
    )
    res_c['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm C(構造転写+soft磨き): "
          f"soft {res_c['soft_acc_before']:.4f}->{res_c['soft_acc_after']:.4f} "
          f"hard {res_c['hard_acc_before']:.4f}->{res_c['hard_acc_after']:.4f} "
          f"k_layer_agreement={res_c['transcribe_info']['k_layer_agreement']:.4f} "
          f"({res_c['seconds']}s)")
    result['C'] = res_c

    result['lift_vs_scratch'] = res_c['hard_acc_after'] - res_a['hard_acc']
    result['lift_vs_iokd'] = res_c['hard_acc_after'] - res_b['hard_acc']

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 96)
    print('RNN→RDDLGN構造転写(D-RNN1): 4アーム exact-match hard精度 比較')
    print('=' * 96)

    header = '| arm | soft_acc(mean±std) | hard_acc(mean±std) | gap(mean±std) |'
    print(header)
    print('|' + '---|' * 4)
    summary: dict = {'arms': {}}
    for arm in ('A', 'B', 'C'):
        if arm in ('A', 'B'):
            softs = [r[arm]['soft_acc'] for r in results]
            hards = [r[arm]['hard_acc'] for r in results]
        else:
            softs = [r[arm]['soft_acc_after'] for r in results]
            hards = [r[arm]['hard_acc_after'] for r in results]
        gaps = [s - h for s, h in zip(softs, hards, strict=True)]
        sm, ss = _mean_std(softs)
        hm, hs = _mean_std(hards)
        gm, gs = _mean_std(gaps)
        label = {
            'A': 'A: scratch', 'B': 'B: IO-KDのみ',
            'C': 'C: 構造転写(native RDDLGN k_layer)+soft磨き(本命)',
        }[arm]
        print(f'| {label} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} | {gm:+.4f}±{gs:.4f} |')
        summary['arms'][arm] = {
            'soft_acc_mean': sm, 'soft_acc_std': ss,
            'hard_acc_mean': hm, 'hard_acc_std': hs,
            'gap_mean': gm, 'gap_std': gs,
        }

    teacher_soft = [r['teacher_soft_acc'] for r in results]
    teacher_hard = [r['teacher_hard_acc'] for r in results]
    tsm, tss = _mean_std(teacher_soft)
    thm, ths = _mean_std(teacher_hard)
    print(f'\nteacher参照: soft exact-match={tsm:.4f}±{tss:.4f}  '
          f'hard(閾値ゲート意味論)exact-match={thm:.4f}±{ths:.4f}')
    summary['teacher_soft_acc_mean'] = tsm
    summary['teacher_hard_acc_mean'] = thm

    k_agree = [r['C']['transcribe_info']['k_layer_agreement'] for r in results]
    kam, kas = _mean_std(k_agree)
    print(f'k_layer転写後agreement(較正データ上、teacher隠れ状態更新則との一致率)'
          f'={kam:.4f}±{kas:.4f}')
    summary['k_layer_agreement_mean'] = kam

    lift_scratch = [r['lift_vs_scratch'] for r in results]
    lift_iokd = [r['lift_vs_iokd'] for r in results]
    lsm, lss = _mean_std(lift_scratch)
    lim, lis = _mean_std(lift_iokd)
    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift_vs_scratch (C - A) : {lsm:+.4f}±{lss:.4f}')
    print(f'lift_vs_iokd    (C - B) : {lim:+.4f}±{lis:.4f}')
    summary['lift_vs_scratch_mean'] = lsm
    summary['lift_vs_scratch_std'] = lss
    summary['lift_vs_iokd_mean'] = lim
    summary['lift_vs_iokd_std'] = lis

    print('\n--- 仮説判定(本手法 = arm C) ---')
    print(f"H(構造転写>IO-KDのみ, hard exact-match): lift_vs_iokd={lim:+.4f} "
          f"→ {'支持' if lim > 0 else '不支持'}")
    print(f"H(構造転写>scratch, hard exact-match): lift_vs_scratch={lsm:+.4f} "
          f"→ {'支持' if lsm > 0 else '不支持'}")

    return summary


# ============================================================
# メイン
# ============================================================

def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='RNN→RDDLGN構造転写(D-RNN1): 再帰teacherの構造転移がhard精度に'
                     '効くかを検証する境界ケース実験(negative結果も価値ある成果物)',
    )
    parser.add_argument('--vocab-size', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--d-emb', type=int, default=8)
    parser.add_argument('--d-hidden', type=int, default=16)
    parser.add_argument('--fanin', type=int, default=6, choices=(2, 4, 6),
                        help='k_layer/headのLUT fan-in(LutLayer制約で2/4/6のみ)')
    parser.add_argument('--head-sizes', type=int, nargs='+', default=[64, 32])

    parser.add_argument('--num-train', type=int, default=2000)
    parser.add_argument('--num-eval', type=int, default=500)
    parser.add_argument('--calib-samples', type=int, default=256)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)

    parser.add_argument('--teacher-epochs', type=int, default=30)
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-prune-epochs', type=int, default=20,
                        help='RNNセルfan-in枝刈り各段のre-fitエポック数')

    parser.add_argument('--epochs', type=int, default=100, help='arm A/Bの学習エポック数')
    parser.add_argument('--lr', type=float, default=5e-2, help='arm A/BのAdam学習率')
    parser.add_argument('--kd-alpha', type=float, default=0.25)
    parser.add_argument('--kd-temperature', type=float, default=2.0)

    parser.add_argument('--polish-epochs', type=int, default=30,
                        help='arm C soft磨きのfine-tuneエポック数')
    parser.add_argument('--polish-lr', type=float, default=0.02)

    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・少epoch、<60s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str, default='outputs/rnn_to_rddlgn')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.quick:
        args.seeds = 1
        args.num_train = 500
        args.num_eval = 200
        args.calib_samples = 128
        args.teacher_epochs = 10
        args.teacher_prune_epochs = 5
        args.epochs = 20
        args.polish_epochs = 10
        args.head_sizes = [32, 16]

    print(f'デバイス: {device}')
    print(f'copyタスク: vocab={args.vocab_size} seq_len={args.seq_len} '
          f'seeds={args.seeds}')
    print(f'teacher: d_emb={args.d_emb} d_hidden={args.d_hidden} '
          f'epochs={args.teacher_epochs}')
    print(f'student: fanin={args.fanin} head_sizes={args.head_sizes} '
          f'epochs={args.epochs} polish_epochs={args.polish_epochs}')

    use_wandb = init_wandb(
        run_name=f'rnn_to_rddlgn_v{args.vocab_size}_s{args.seq_len}_h{args.d_hidden}',
        config_dict=vars(args),
        device=device,
        tags=['structural-transfer', 'rnn', 'rddlgn', 'd-rnn1'],
        group='convert-rnn-to-rddlgn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_soft_acc': r['teacher_soft_acc'],
                f'seed{seed}/teacher_hard_acc': r['teacher_hard_acc'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/A_soft_acc': r['A']['soft_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/B_soft_acc': r['B']['soft_acc'],
                f'seed{seed}/C_hard_acc': r['C']['hard_acc_after'],
                f'seed{seed}/C_soft_acc': r['C']['soft_acc_after'],
                f'seed{seed}/C_k_layer_agreement':
                    r['C']['transcribe_info']['k_layer_agreement'],
                f'seed{seed}/lift_vs_scratch': r['lift_vs_scratch'],
                f'seed{seed}/lift_vs_iokd': r['lift_vs_iokd'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_{ts}.json')
    dump = {
        'args': vars(args),
        'summary': summary,
        'per_seed': results,
    }
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
