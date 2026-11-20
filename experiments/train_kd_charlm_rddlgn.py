"""
data-free文字レベル言語モデルKD訓練スクリプト（ROADMAP-v5）

nanoGPT文字レベルteacher(`hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights`)
から生成したシーケンス(gen_teacher_corpus.py)のみを使い、実テキストに一切触れずに
RDDLGN/DLGNAttentionTranslator/単純next-charベースラインへ蒸留する。

data-freeの根拠: 訓練ループが読むデータは TeacherCorpusDataset(teacherの自己回帰
サンプル)のみ。tiny_shakespeare実データは evaluate_bpc_real() 内でのみ読み込み、
勾配は一切流さない(BPC評価専用、teacher/studentどちらも同じ実データウィンドウで
比較する「アンカー」として使う)。

studentトークン空間の設計（重要、PROGRESS/設計ノート参照）:
    teacher文字vocab(65) + PAD(65) + BOS(66) → student vocab_size = 67。
    teacherロジットは[.,.,65]なので、KD比較のためにPAD/BOS位置を大きな負定数
    (-1e9)でパディングして[.,.,67]に揃える(採用案。理由: student側の
    GroupSum出力をスライスする代替案より「teacher/studentのlogits形状が
    最初から一致する」ほうがmake_seq_kd_loss呼び出し側のコードが単純になる。
    -1e9はsoftmax後の確率質量が実質0になる値で、実際の学習信号を歪めない)。

シーケンス整列（設計ノートより、重要）:
    teacherはlength=64の文字列を生成。
        src        = chars[0:32]                (エンコーダ文脈)
        tgt_input  = [BOS] + chars[32:63]        (デコーダ入力, teacher forcing)
        tgt_target = chars[32:64]                (デコーダ正解)
    teacher.next_token_logits(chars[0:64]) の位置iは「char[i+1]を予測するロジット」
    (nanoGPTの標準的な次トークン予測)。よってchar[32+t]を予測するteacherロジットは
    位置31+tにある。student側のデコードステップtはtgt_input[t]を消費して
    tgt_target[t]=chars[32+t]を予測するので、teacherロジット[:, 31:63, :]
    (63-31=32個、t=0..31に対応)がstudentの32ステップ分と正確に整列する。
    この整列は tests/test_charlm_kd_alignment.py で決定的teacherを使って検証済み。

使用例:
    # スモーク(rddlgn、CPU、tiny構成)
    WANDB_MODE=offline uv run python experiments/train_kd_charlm_rddlgn.py \\
        --student rddlgn --config tiny \\
        --corpus data-charlm/corpus_XXXXXXXX_500x64_T1.0_s42.pt \\
        --eval-corpus data-charlm/corpus_XXXXXXXX_500x64_T1.0_s42_eval.pt \\
        -ni 100 -ef 50 -bs 16

    # nextcharベースライン
    WANDB_MODE=offline uv run python experiments/train_kd_charlm_rddlgn.py \\
        --student nextchar --corpus data-charlm/corpus_*.pt \\
        --eval-corpus data-charlm/*_eval.pt -ni 50 -ef 25 -bs 16

    # attention translator arm
    WANDB_MODE=offline uv run python experiments/train_kd_charlm_rddlgn.py \\
        --student attention --config tiny --corpus data-charlm/corpus_*.pt \\
        --eval-corpus data-charlm/*_eval.pt -ni 50 -ef 25 -bs 16

    # フルスケール(GPU、Modal経由)
    uv run modal run scripts/modal_app.py --script train_kd_charlm_rddlgn.py \\
        --args "--student rddlgn --config paper --scale 0.05 --corpus data-charlm/corpus_....pt --eval-corpus data-charlm/....pt -ni 50000 -ef 2000"
"""

import argparse
import glob
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_metrics import ConvergenceTracker, WallClock
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from data.char_corpus import (
    BOS_ID,
    PAD_ID,
    STUDENT_VOCAB_SIZE,
    RealCharDataset,
    TeacherCorpusDataset,
    load_tiny_shakespeare_train,
    load_tiny_shakespeare_val,
)
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import (
    RDDLGN,
    DLGNAttentionTranslator,
    LogicGPT,
    LogicGPTConfig,
    RDDLGNConfig,
    TranslatorConfig,
)
from griffith_logic.convert.distill import (
    LogitExitHead,
    make_exit_kd_loss,
    make_seq_kd_loss,
    make_seq_kd_loss_v2,
)
from griffith_logic.convert.text_teacher import decode_shakespeare_char, resolve_text_teacher
from griffith_logic.group_sum import GroupSum
from griffith_logic.models.rddlgn import LogicEmbedding, LutStack

load_dotenv()

WINDOW = 64      # teacherが生成する1シーケンスの長さ
SRC_LEN = 32     # エンコーダ文脈長 / デコーダ系列長
TEACHER_VOCAB = 65  # teacher側の文字vocabサイズ(PAD/BOSを含まない)


# ============================================================
# teacherロジットのオンライン計算 + 整列スライス
# ============================================================

def compute_teacher_logits_for_batch(
    teacher, raw_tokens: torch.Tensor, device: str,
) -> torch.Tensor:
    """(B, 64)の生トークン列からKD対象32ステップ分のteacherロジットを返す。

    Args:
        teacher: CharLMTeacher(next_token_logitsを持つ)。
        raw_tokens: (B, 64) long。teacher vocab(0..64)のみを含む。
        device: teacherの推論デバイス(CPU固定運用でも明示的に渡す)。

    Returns:
        (B, 32, 65) — 位置t(0..31)はtgt_target[t]=chars[32+t]の予測ロジット。
        整列根拠はモジュールdocstring参照:
            teacher.next_token_logits(chars[0:64])[:, i, :] は chars[i+1] の予測。
            chars[32+t] の予測は i=31+t にある → スライス[:, 31:63, :]。
    """
    with torch.no_grad():
        full_logits = teacher.next_token_logits(raw_tokens.to(device))  # (B, 64, 65) CPU
    # next_token_logitsはCPU floatを返す仕様(text_teacher.py参照)。
    aligned = full_logits[:, 31:63, :]  # (B, 32, 65)
    assert aligned.shape[1] == SRC_LEN, aligned.shape
    return aligned


def pad_teacher_logits_to_student_vocab(
    teacher_logits: torch.Tensor, student_vocab_size: int = STUDENT_VOCAB_SIZE,
    neg_fill: float = -1e9,
) -> torch.Tensor:
    """teacherロジット[B,S,65]をstudent語彙幅[B,S,67]へ大負定数パディングする。

    PAD(65)/BOS(66)の列を-1e9で埋めることで、softmax後の確率質量がそれらの
    トークンにほぼ0になる(実数値だが指数的に無視できるレベル)。studentが
    これらのIDをtargetとして持つことは無いため(tgt_targetは常にteacher vocab
    範囲0..64)、この設計はKD項の数値にもCE項にも悪影響を与えない。
    """
    b, s, v = teacher_logits.shape
    assert v == TEACHER_VOCAB, v
    pad_width = student_vocab_size - v
    if pad_width <= 0:
        return teacher_logits
    filler = torch.full(
        (b, s, pad_width), neg_fill, dtype=teacher_logits.dtype, device=teacher_logits.device,
    )
    return torch.cat([teacher_logits, filler], dim=-1)


# ============================================================
# nextcharベースラインモデル（1ステップKDサニティ）
# ============================================================

class NextCharBaseline(nn.Module):
    """LogicEmbedding + LutLayerスタック + GroupSumで「src直後の1文字」だけを予測する
    最小構成(サニティベースライン)。

    RDDLGN/AttentionTranslatorのような系列デコーダを持たず、srcの最終位置の
    埋め込みを直接分類ヘッドに通す。1ポジションのみのKDで学習が機能するかを
    確認する土台として使う。train_kd_charlm_rddlgn.py の系列KDインターフェース
    (forward(src, tgt_input) -> (logits[B,T,V], emb_reg_loss))には合わせず、
    専用の学習ループ分岐で扱う（下記train_step_nextchar参照）。
    """

    def __init__(self, vocab_size: int, d_emb: int, hidden_sizes: list[int],
                 n_inputs: int = 2, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = LogicEmbedding(vocab_size, d_emb)

        # 最終隠れ層はstudent vocab_size(67)の倍数に丸める(GroupSum契約)。
        hidden_sizes = list(hidden_sizes)
        group_factor = max(1, hidden_sizes[-1] // vocab_size)
        hidden_sizes[-1] = vocab_size * group_factor

        self.stack = LutStack(
            in_dim=d_emb * SRC_LEN, layer_sizes=hidden_sizes,
            n_inputs=n_inputs, device=device,
        )
        self.group_sum = GroupSum(k=vocab_size, tau=1.0)

    def forward(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: (B, 32) long

        Returns:
            logits: (B, vocab_size) — src直後の1文字(chars[32])の予測ロジット
            emb_reg_loss: スカラー
        """
        emb = self.embedding(src_tokens)  # (B, 32, d_emb)
        b = emb.shape[0]
        flat = emb.reshape(b, -1)
        h = self.stack(flat)
        logits = self.group_sum(h)
        return logits, self.embedding.binary_reg_loss()


# ============================================================
# ランプ・損失(train_rddlgn.pyのget_emb_reg_weightをコピー)
# ============================================================

def get_emb_reg_weight(step: int, ramp_start: int = 1000, ramp_end: int = 100000,
                       max_weight: float = 0.1) -> float:
    """二値化正則化の重みを線形ランプで計算(train_rddlgn.pyと同一)。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


# Spectral正則化係数。mainでargs.spectral_regから設定される。
# 6種のtrain_step_*全てが_backward_and_step経由なためここで一元適用する。
_SPECTRAL_REG = 0.0


def _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip: float = 0.0):
    """全train_step_*共通のbackward+optimizer.step(+任意でgrad clip/scheduler)。

   (学習安定化): grad_clip<=0 かつ scheduler=None のとき、既存の
    「zero_grad→backward→step」と完全に同じ(bitwise)挙動になる
    (デフォルトOFF、CLIで明示した時のみ有効化する設計方針)。

    Args:
        model: パラメータのnormを計算するモデル(grad_clip>0の時のみ使用)。
        optimizer: optimizer.step()を呼ぶ対象。
        scheduler: Noneまたはstep()を持つLRスケジューラ。Noneなら何もしない。
        total_loss: backward対象のスカラー損失。
        grad_clip: >0のとき、backward後・step前にclip_grad_norm_を適用する。
    """
    if _SPECTRAL_REG > 0:
        from griffith_logic.regularization import spectral_penalty
        total_loss = total_loss + _SPECTRAL_REG * spectral_penalty(model)
    optimizer.zero_grad()
    total_loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def make_pure_ce_seq_loss(pad_id: int = PAD_ID):
    """--kd off(scratch)専用の純CE系列損失。

    make_seq_kd_loss(alpha=1.0, ...)のCE項と数値的に同一(同じ
    ignore_index=pad_idマスク規約)だが、teacher_logits引数を受け取らず
    KD項の計算を一切行わない。train_step_*_scratch系がteacherロジット
    計算を呼ばずに済む(GPU無駄+scratch純度)ためのシグネチャ分離。

    Returns:
        loss(student_logits[B,S,V], target_ids[B,S]) -> スカラー。
    """
    def loss_fn(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        vocab = student_logits.shape[-1]
        return F.cross_entropy(
            student_logits.reshape(-1, vocab), target_ids.reshape(-1).long(),
            ignore_index=pad_id,
        )
    return loss_fn


# ============================================================
# データ準備
# ============================================================

def resolve_glob(pattern: str) -> str:
    """globパターン(シェル展開されなかった場合)を解決して1ファイルを返す。"""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'該当ファイルなし: {pattern}')
    if len(matches) > 1:
        print(f'警告: 複数一致、先頭を採用: {matches[0]} (全候補: {matches})')
    return matches[0]


def collate_with_raw(batch):
    """(src, tgt_input, tgt_target)のバッチ化。Dataset.__getitem__の3-tuple用。"""
    src = torch.stack([b[0] for b in batch])
    tgt_input = torch.stack([b[1] for b in batch])
    tgt_target = torch.stack([b[2] for b in batch])
    return src, tgt_input, tgt_target


def collate_raw_window(batch):
    """(64,)の生トークン列のバッチ化。TeacherCorpusDataset(mode='decoder_only')専用。"""
    return torch.stack(list(batch))


# ============================================================
# 訓練ステップ(seq2seq: rddlgn / attention)
# ============================================================

def train_step_seq2seq(
    model, teacher, src, tgt_input, tgt_target, raw_tokens,
    optimizer, step, loss_fn, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0, exit_ctx=None,
):
    """RDDLGN/DLGNAttentionTranslator共通の1ステップ訓練(同一forward契約)。

    init_step: --init-ckptでfine-tuneする際、emb_reg_ramp/loss_fnの
    step依存スケジュールが巻き戻らないよう`step + init_step`を評価stepとして使う
    (既定0なので通常時は無改造と同じ挙動)。

    exit_ctx(Noneなら無効=既存挙動を完全維持):
        dict(exit_kd_loss=Callable, exit_weight=float, exit_layers=list[int])。
        指定時、model(src, tgt_input, return_hidden=True)を呼び、exit_layers
        で選んだ中間活性をexit_kd_lossに渡してteacher_logitsへ蒸留する
        (追加のteacher forwardは行わず、既に計算済みのteacher_logitsを再利用する)。
        RDDLGNのhidden_states[0]=context([B,context_dim]、時間軸なし)は
        tgt_targetの系列長ぶんexpandしてから渡す(kd_kl_loss_seqの[B,S,V]契約
        に合わせるため。値は複製のみで意味は変えない)。
    """
    model.train()
    effective_step = step + init_step

    teacher_logits = compute_teacher_logits_for_batch(teacher, raw_tokens, device)
    teacher_logits = pad_teacher_logits_to_student_vocab(teacher_logits).to(device)

    if exit_ctx is None:
        logits, emb_loss = model(src, tgt_input)  # (B, 32, 67)
    else:
        logits, emb_loss, hidden_states = model(src, tgt_input, return_hidden=True)

    main_loss = loss_fn(logits, tgt_target, teacher_logits, step=effective_step)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    exit_loss_value = 0.0
    if exit_ctx is not None:
        seq_len = tgt_target.shape[1]
        selected = []
        for idx in exit_ctx['exit_layers']:
            h = hidden_states[idx]
            if h.dim() == 2:
                # context([B, context_dim])は時間軸を持たないため、
                # kd_kl_loss_seqの[B,S,V]契約に合わせて系列長ぶんexpandする。
                h = h.unsqueeze(1).expand(-1, seq_len, -1)
            selected.append(h)
        mask = tgt_target != PAD_ID
        exit_loss = exit_ctx['exit_kd_loss'](selected, teacher_logits, mask=mask)
        total_loss = total_loss + exit_ctx['exit_weight'] * exit_loss
        exit_loss_value = exit_loss.item()

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item(), exit_loss_value


def train_step_seq2seq_scratch(
    model, src, tgt_input, tgt_target, optimizer, step, loss_fn_ce, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0,
):
    """RDDLGN/DLGNAttentionTranslator共通のscratch(--kd off)1ステップ訓練。

    train_step_seq2seqとの唯一の違いはteacherを一切呼ばないこと(引数にteacher/
    raw_tokensを取らない)。損失は純CE(make_pure_ce_seq_loss)。
    init_stepの扱いはtrain_step_seq2seqと同様。
    """
    model.train()
    effective_step = step + init_step

    logits, emb_loss = model(src, tgt_input)  # (B, 32, 67)

    main_loss = loss_fn_ce(logits, tgt_target)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item()


def train_step_nextchar_scratch(
    model, src, tgt_target_first, optimizer, step, loss_fn_ce_seq, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0,
):
    """nextcharベースラインのscratch(--kd off)1ステップ訓練(teacher呼び出し無し)。"""
    model.train()
    effective_step = step + init_step

    logits, emb_loss = model(src)  # (B, vocab_size)
    logits_seq = logits.unsqueeze(1)  # (B, 1, vocab_size)
    target_seq = tgt_target_first.unsqueeze(1)  # (B, 1)

    main_loss = loss_fn_ce_seq(logits_seq, target_seq)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item()


def train_step_decoder_only_scratch(
    model, window, optimizer, step, loss_fn_ce, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0,
):
    """LogicGPT(decoder-only)のscratch(--kd off)1ステップ訓練(teacher呼び出し無し)。

    input/targetのオフセット整列はtrain_step_decoder_onlyと同一
    (input=window[:, :-1], target=window[:, 1:])。
    """
    model.train()
    effective_step = step + init_step

    inp = window[:, :-1].to(device)   # (B, 63)
    target = window[:, 1:].to(device)  # (B, 63)

    logits, emb_loss = model(inp)  # (B, 63, vocab_size)

    main_loss = loss_fn_ce(logits, target)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item()


def train_step_nextchar(
    model, teacher, src, tgt_target_first, raw_tokens,
    optimizer, step, loss_fn_seq, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0,
):
    """nextcharベースライン専用の1ステップ訓練(1ポジションのみのKD)。

    make_seq_kd_loss(loss_fn_seq)はseq長Sを仮定するので、S=1の系列として
    扱うためlogits/targetsに長さ1の次元を追加してmake_seq_kd_lossへ渡す
    (関数のshape契約[B,S,V]/[B,S]をそのまま満たせるため無改造で使い回せる)。
    """
    model.train()
    effective_step = step + init_step

    teacher_logits_full = compute_teacher_logits_for_batch(teacher, raw_tokens, device)
    # nextcharは「src直後の1文字」= tgt_target[0] のみを予測するため、
    # teacherロジットも対応する先頭1ステップ分だけを使う。
    teacher_logits_1 = teacher_logits_full[:, :1, :]  # (B, 1, 65)
    teacher_logits_1 = pad_teacher_logits_to_student_vocab(teacher_logits_1).to(device)

    logits, emb_loss = model(src)  # (B, vocab_size)
    logits_seq = logits.unsqueeze(1)  # (B, 1, vocab_size)
    target_seq = tgt_target_first.unsqueeze(1)  # (B, 1)

    main_loss = loss_fn_seq(logits_seq, target_seq, teacher_logits_1, step=effective_step)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item()


def train_step_decoder_only(
    model, teacher, window, optimizer, step, loss_fn, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0, init_step=0, exit_ctx=None,
):
    """LogicGPT(decoder-only)専用の1ステップ訓練。

    window: (B, 64) の生トークン列(teacher vocab範囲、PAD/BOS無し)。
    student入力/正解のオフセット整列(load-bearing、重要):
        input  = window[:, :-1]   (63トークン, 位置0..62)
        target = window[:, 1:]    (63トークン, 位置1..63)
        student位置i(0-indexed, i=0..62)はinput[:, :i+1]から
        target[i]=window[i+1]を予測する。
    teacherロジット整列(seq2seqアームの[:, 31:63, :]オフセット-31トリックとは異なり、
    ここではオフセット無しでそのまま対応する):
        teacher.next_token_logits(window)[:, i, :] は window[i+1] の予測。
        student位置iもwindow[i+1]を予測するので、teacher位置i ⇔ student位置iが
        既に1対1対応している(i=0..62の63ステップ全体、スライス[:, 0:63, :])。

    exit_ctx(Noneなら無効=既存挙動を完全維持):
        dict(exit_kd_loss=Callable, exit_weight=float, exit_layers=list[int])。
        指定時、model(inp, return_hidden=True)を呼び、exit_layersで選んだ
        LogicTransformerBlock出力(各[B,63,d_model]、時間軸ありでexpand不要)を
        exit_kd_lossに渡してteacher_logitsへ蒸留する。
    """
    model.train()
    effective_step = step + init_step

    with torch.no_grad():
        full_logits = teacher.next_token_logits(window.to(device))  # (B, 64, 65) CPU
    teacher_logits = full_logits[:, :63, :]  # (B, 63, 65) — オフセット無しでinput/targetと整列
    teacher_logits = pad_teacher_logits_to_student_vocab(teacher_logits).to(device)

    inp = window[:, :-1].to(device)   # (B, 63)
    target = window[:, 1:].to(device)  # (B, 63)

    if exit_ctx is None:
        logits, emb_loss = model(inp)  # (B, 63, vocab_size)
    else:
        logits, emb_loss, hidden_states = model(inp, return_hidden=True)

    main_loss = loss_fn(logits, target, teacher_logits, step=effective_step)

    emb_weight = get_emb_reg_weight(effective_step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    exit_loss_value = 0.0
    if exit_ctx is not None:
        selected = [hidden_states[idx] for idx in exit_ctx['exit_layers']]
        mask = target != PAD_ID
        exit_loss = exit_ctx['exit_kd_loss'](selected, teacher_logits, mask=mask)
        total_loss = total_loss + exit_ctx['exit_weight'] * exit_loss
        exit_loss_value = exit_loss.item()

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item(), exit_loss_value


# ============================================================
# 評価: (1) teacher-forced top-1一致率、(2) 実テキストBPC、(3) soft/hard gap、
#       (4) 生成サンプル
# ============================================================

@torch.no_grad()
def evaluate_teacher_agreement(model, student_kind, loader, teacher, device, use_soft=True):
    """held-out teacherコーパスでの「student argmax == teacher argmax」一致率。

    loaderはstudent_kindによって異なるバッチ形式を返す前提:
        - 'logicgpt': (B, 64) 生トークン列(collate_raw_window)
        - それ以外(rddlgn/attention/nextchar): (src, tgt_input, tgt_target) 3-tuple
    """
    was_training = model.training
    model.train(mode=use_soft)

    correct = 0
    total = 0

    if student_kind == 'logicgpt':
        for window in loader:
            # compute_teacher_logits_for_batchはseq2seqアームの[31:63]オフセットを
            # 前提とするため、logicgptではそれを使わずtrain_step_decoder_onlyと
            # 同じ[0:63]整列を直接計算する。
            with torch.no_grad():
                full_logits = teacher.next_token_logits(window.to(device))  # (B,64,65)
            teacher_logits = full_logits[:, :63, :]
            teacher_pred = teacher_logits.argmax(dim=-1).to(device)  # (B, 63)

            inp = window[:, :-1].to(device)
            student_logits, _ = model(inp)  # (B, 63, vocab_size)
            student_pred = student_logits.argmax(dim=-1)  # (B, 63)
            correct += (student_pred == teacher_pred).sum().item()
            total += teacher_pred.numel()
        model.train(mode=was_training)
        return correct / total if total > 0 else 0.0

    for src, tgt_input, tgt_target in loader:
        src = src.to(device)
        # raw_tokens(64) = src(32) + tgt_target(32) がそのまま元の64トークン列
        # に一致する(split_src_tgtの定義: src=chars[0:32], tgt_target=chars[32:64])。
        raw_tokens = torch.cat([src.cpu(), tgt_target.cpu()], dim=1)

        teacher_logits = compute_teacher_logits_for_batch(teacher, raw_tokens, device)
        teacher_pred = teacher_logits.argmax(dim=-1).to(device)  # (B, 32) teacher vocab範囲

        if student_kind == 'nextchar':
            tgt_input = tgt_input.to(device)
            student_logits, _ = model(src)  # (B, vocab)
            student_pred = student_logits.argmax(dim=-1)  # (B,)
            correct += (student_pred == teacher_pred[:, 0]).sum().item()
            total += student_pred.shape[0]
        else:
            tgt_input = tgt_input.to(device)
            student_logits, _ = model(src, tgt_input)  # (B, 32, 67)
            student_pred = student_logits.argmax(dim=-1)  # (B, 32)
            correct += (student_pred == teacher_pred).sum().item()
            total += teacher_pred.numel()

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_bpc(model, student_kind, windows: torch.Tensor, device, batch_size=64, use_soft=True):
    """実テキストwindows[N,64]に対するteacher-forced NLL/ln(2)per-char(BPC)を計算。

    student==Noneの場合はteacher自身のBPCを計算する(呼び出し側でmodel=teacherを
    渡す想定。teacherのnext_token_logitsは既にCPU float、no_gradなのでそのまま使う)。

    use_soft: Trueならtrain mode(sigmoid連続=soft)、Falseならeval mode
        (Heaviside離散化後=hard、論理ゲートの実機性能相当)でBPCを計算する。
        evaluate_teacher_agreementと同じモード切替パターンを踏襲する。
    """
    was_training = model.training
    model.train(mode=use_soft)

    total_nll = 0.0
    total_chars = 0
    n = windows.shape[0]

    for start in range(0, n, batch_size):
        batch = windows[start:start + batch_size].to(device)
        src = batch[:, :SRC_LEN]
        tgt_target = batch[:, SRC_LEN:WINDOW]
        bos = torch.full((batch.shape[0], 1), BOS_ID, dtype=torch.long, device=device)
        tgt_input = torch.cat([bos, batch[:, SRC_LEN:WINDOW - 1]], dim=1)

        if student_kind == 'nextchar':
            logits, _ = model(src)  # (B, vocab) — chars[32]のみ予測可能
            log_probs = torch.log_softmax(logits[:, :TEACHER_VOCAB], dim=-1)
            target = tgt_target[:, 0]
            nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
            total_nll += nll.sum().item()
            total_chars += target.shape[0]
        elif student_kind == 'logicgpt':
            inp = batch[:, :-1]     # (B, 63)
            target = batch[:, 1:]   # (B, 63)
            logits, _ = model(inp)  # (B, 63, vocab_size)
            log_probs = torch.log_softmax(logits[:, :, :TEACHER_VOCAB], dim=-1)
            nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, 63)
            total_nll += nll.sum().item()
            total_chars += target.numel()
        else:
            logits, _ = model(src, tgt_input)  # (B, 32, 67)
            log_probs = torch.log_softmax(logits[:, :, :TEACHER_VOCAB], dim=-1)
            nll = -log_probs.gather(2, tgt_target.unsqueeze(-1)).squeeze(-1)  # (B, 32)
            total_nll += nll.sum().item()
            total_chars += tgt_target.numel()

    model.train(mode=was_training)

    mean_nll_nats = total_nll / total_chars
    bpc = mean_nll_nats / math.log(2)
    return bpc


@torch.no_grad()
def evaluate_teacher_bpc(teacher, windows: torch.Tensor, device, batch_size=64):
    """teacher自身の実テキストBPC(比較アンカー)。teacher-forced(自身のnext_token_logits)。"""
    total_nll = 0.0
    total_chars = 0
    n = windows.shape[0]
    for start in range(0, n, batch_size):
        batch = windows[start:start + batch_size].to(device)
        logits = teacher.next_token_logits(batch)  # (B, 64, 65) CPU float
        # 位置iはchar[i+1]を予測。char[1..63]をラベルとして使う(char[0]は文脈のみ)。
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (B, 63, 65)
        targets = batch[:, 1:].cpu()  # (B, 63)
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        total_chars += targets.numel()
    return (total_nll / total_chars) / math.log(2)


@torch.no_grad()
def generate_sample(model, student_kind, teacher, device, length=300, temperature=0.8, greedy=True):
    """先頭BOS(nextchar以外)/種文字から貪欲またはsamplingで文字列を生成する。

    RDDLGN/AttentionTranslatorはencoder-decoder構造なので、生成のたびに
    「直近32文字」をsrcとして再エンコードし1文字ずつ自己回帰させる
    (sliding window。teacher.generateのblock_sizeクロップと同じ考え方)。
    """
    model.eval()
    # teacherの生成物を種にする(data-freeの範�囲内: teacher自身のサンプル)。
    seed_tokens = teacher.generate(1, SRC_LEN, temperature=1.0, seed=12345, device=device)[0]
    generated = seed_tokens.tolist()  # teacher vocab範囲の文字ID列

    if student_kind == 'nextchar':
        for _ in range(length):
            ctx = generated[-SRC_LEN:]
            if len(ctx) < SRC_LEN:
                ctx = [0] * (SRC_LEN - len(ctx)) + ctx
            src = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = model(src)
            logits = logits[:, :TEACHER_VOCAB]
            if greedy:
                next_id = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
    elif student_kind == 'logicgpt':
        block_size = model.config.block_size
        for _ in range(length):
            ctx = generated[-block_size:]
            if len(ctx) < block_size:
                ctx = [0] * (block_size - len(ctx)) + ctx  # nextcharと同じ左パディング流儀
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = model(inp)  # (1, len(ctx), vocab_size)
            step_logits = logits[:, -1, :TEACHER_VOCAB]
            if greedy:
                next_id = step_logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
    else:
        for _ in range(length):
            ctx = generated[-SRC_LEN:]
            if len(ctx) < SRC_LEN:
                ctx = [0] * (SRC_LEN - len(ctx)) + ctx
            src = torch.tensor([ctx], dtype=torch.long, device=device)
            gen_ids = [BOS_ID]
            for _t in range(SRC_LEN):
                tgt_input = torch.tensor([gen_ids], dtype=torch.long, device=device)
                logits, _ = model(src, tgt_input)
                step_logits = logits[:, -1, :TEACHER_VOCAB]
                if greedy:
                    next_id = step_logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(step_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                gen_ids.append(next_id)
                # 1文字生成できたところで次のsliding windowへ移る(全32文字を
                # デコードしなくても、通常のLM生成と同様に1文字ずつ進める)。
                if _t == 0:
                    generated.append(next_id)
                    break

    text = decode_shakespeare_char(generated)
    return text


# ============================================================
# モデル構築
# ============================================================

def build_config_rddlgn(args) -> RDDLGNConfig:
    if args.config == 'paper':
        config = RDDLGNConfig.from_paper(scale=args.scale)
    else:
        config = RDDLGNConfig.tiny()
    # fan-in/パラメータ化(従来はnextchar以外n=2固定だった)。
    # getattr既定はテスト等がNamespaceを部分構築する場合の互換用
    config.n_inputs = getattr(args, 'n_inputs', 2)
    config.parameterization = getattr(args, 'parameterization', 'softmax')
    config.tau_warp = getattr(args, 'tau_warp', 1.0)
    config.gumbel = not getattr(args, 'no_gumbel', False)
    config.vocab_size = STUDENT_VOCAB_SIZE
    config.seq_len = SRC_LEN
    m_last = config.m_layer_sizes[-1]
    if m_last % config.vocab_size != 0:
        group_factor = max(1, m_last // config.vocab_size)
        config.m_layer_sizes[-1] = config.vocab_size * max(group_factor, 1)
    return config


def build_config_attention(args) -> TranslatorConfig:
    if args.config == 'paper':
        config = TranslatorConfig.from_paper(scale=args.scale)
    else:
        config = TranslatorConfig.tiny()
    # fan-in/パラメータ化(従来はnextchar以外n=2固定だった)。
    # getattr既定はテスト等がNamespaceを部分構築する場合の互換用
    config.n_inputs = getattr(args, 'n_inputs', 2)
    config.parameterization = getattr(args, 'parameterization', 'softmax')
    config.tau_warp = getattr(args, 'tau_warp', 1.0)
    config.gumbel = not getattr(args, 'no_gumbel', False)
    config.vocab_size = STUDENT_VOCAB_SIZE
    config.src_len = SRC_LEN
    config.tgt_len = SRC_LEN
    m_last = config.m_layer_sizes[-1]
    if m_last % config.vocab_size != 0:
        group_factor = max(1, m_last // config.vocab_size)
        config.m_layer_sizes[-1] = config.vocab_size * max(group_factor, 1)
    return config


def build_config_logicgpt(args) -> LogicGPTConfig:
    """LogicGPT(decoder-only)のconfigを構築する。

    block_size=WINDOW-1=63固定(studentは64トークン窓のうち63トークンを
    input/targetのオフセットペアとして消費する。train_step_decoder_only参照)。
    """
    if args.config == 'paper':
        config = LogicGPTConfig.from_paper(scale=args.scale)
    else:
        config = LogicGPTConfig.tiny()
    # fan-in/パラメータ化/selection(LogicGPTにも配管)
    config.n_inputs = getattr(args, 'n_inputs', 2)
    config.parameterization = getattr(args, 'parameterization', 'softmax')
    config.selection = getattr(args, 'selection', 'soft')
    config.vocab_size = STUDENT_VOCAB_SIZE
    config.block_size = WINDOW - 1
    # CLI引数は指定時のみ上書き(None=プリセット値を維持)。これにより
    # --config paperのfrom_paper(~1.24M)構成が既定値で潰れないようにする。
    if args.gpt_layers is not None:
        config.n_layers = args.gpt_layers
    if args.gpt_heads is not None:
        config.n_heads = args.gpt_heads
    if args.gpt_dmodel is not None:
        config.d_model = args.gpt_dmodel
    if args.gpt_mlp_ratio is not None:
        config.mlp_ratio = args.gpt_mlp_ratio
    # pos_widthはblock_size<=2^pos_widthを満たす必要がある(63 <= 2^6=64)。
    config.pos_width = max(config.pos_width, math.ceil(math.log2(config.block_size + 1)))
    if config.d_model < config.pos_width:
        config.pos_width = config.d_model
    return config


# ============================================================
# ログ
# ============================================================

def setup_logger(student, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'kd_charlm_{student}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='data-free文字レベルLM KD訓練(RDDLGN/Attention/NextChar)')

    parser.add_argument('--student', type=str, default='rddlgn',
                        choices=['rddlgn', 'attention', 'nextchar', 'logicgpt'])
    parser.add_argument('--corpus', type=str, default=None,
                        help='gen_teacher_corpus.pyの訓練コーパス.pt(--train-data teacher時は必須)')
    parser.add_argument('--eval-corpus', type=str, required=True, help='held-out評価コーパス.pt(全arm必須)')
    parser.add_argument('--teacher', type=str,
                        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')

    # データ効率実験: 訓練データソースの切り替え
    parser.add_argument('--train-data', type=str, default='teacher', choices=['teacher', 'real'],
                        help='訓練データソース。teacher(既定)=既存data-free挙動を完全維持'
                             '(TeacherCorpusDataset)。real=tiny_shakespeare実データの'
                             'prefixサブセット(RealCharDataset、--data-fraction指定)。')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='--train-data real時のtrain pool取得割合(0<fraction<=1)。'
                             '1%%/5%%/25%%/100%%はnested(小さいfractionは大きいfractionのprefix)。')
    parser.add_argument('--kd', type=str, default='on', choices=['on', 'off'],
                        help='on(既定)=KD損失(make_seq_kd_loss)。off=scratch純CE'
                             '(teacher logits計算をtrain step内で一切呼ばない)。'
                             'eval側のteacher使用(BPCアンカー/agreement)は--kdに関わらず維持。'
                             '注意: KD-onをreal訓練データで使う場合、alpha=1.0はscratchと'
                             '数値的に同一になるためalpha<1(既定0.25)で走らせる想定'
                             '(コード上の強制はしない)。')

    # モデル構成(rddlgn/attention共通プリセット)
    parser.add_argument('--config', type=str, default='tiny', choices=['tiny', 'paper'])
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式(全student対応。logicgptにも配管)')
    parser.add_argument('--tau-anneal', action='store_true',
                        help='TauAnnealScheduler(τ_b幾何減衰+Gumbelノイズ減衰)を有効化')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'],
                        help='パラメータ化方式(rddlgn/attention/nextchar。'
                             'default: softmax)')
    parser.add_argument('--tau-warp', type=float, default=1.0,
                        help='WARP温度パラメータ(warp時のみ)')
    parser.add_argument('--no-gumbel', action='store_true',
                        help='WARPのGumbel smoothingを無効化')
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')
    parser.add_argument('--scale', type=float, default=0.05, help='config=paper時のスケール')
    parser.add_argument('--d-emb', type=int, default=64, help='nextchar専用: 埋め込み次元')
    parser.add_argument('--nextchar-hidden', type=str, default='512,512',
                        help='nextchar専用: 隠れ層サイズ(カンマ区切り)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--gpt-layers', type=int, default=None, help='logicgpt専用: LogicTransformerBlock段数(None=preset)')
    parser.add_argument('--gpt-heads', type=int, default=None, help='logicgpt専用: attentionヘッド数(None=preset)')
    parser.add_argument('--gpt-dmodel', type=int, default=None, help='logicgpt専用: d_model(None=preset)')
    parser.add_argument('--gpt-mlp-ratio', type=int, default=None, help='logicgpt専用: MLP中間幅倍率(None=preset)')

    # KD損失
    parser.add_argument('--distill-alpha', type=float, default=0.25)
    parser.add_argument('--distill-temperature', type=float, default=4.0)
    parser.add_argument('--distill-standardize', dest='distill_standardize', action='store_true', default=True)
    parser.add_argument('--no-distill-standardize', dest='distill_standardize', action='store_false')

    # KD損失v2: 全てデフォルトのときはmake_seq_kd_loss(v1)をそのまま使う
    # (bitwise非回帰)。いずれかが非デフォルトのときのみmake_seq_kd_loss_v2に切替。
    parser.add_argument('--kd-kind', type=str, default='kl', choices=['kl', 'dist'],
                        help="kl(既定、make_seq_kd_loss互換)。dist=DIST風Pearson相関損失"
                             "(arXiv:2205.10536)。'dist'指定時のみv2損失に切り替わる。")
    parser.add_argument('--alpha-schedule', type=str, default=None,
                        help="alphaのスケジュール仕様('const:V'/'linear:S:E:D'/'cosine:S:E:D')。"
                             "None(既定)=--distill-alphaをそのまま定数として使う"
                             "(v1のalpha=--distill-alphaと完全に同じ)。指定時のみv2損失に切替。")
    parser.add_argument('--temp-schedule', type=str, default=None,
                        help="temperatureのスケジュール仕様(形式は--alpha-schedule同様)。"
                             "None(既定)=--distill-temperatureをそのまま定数として使う。"
                             "指定時のみv2損失に切替。")
    parser.add_argument('--alpha-mode', type=str, default='fixed', choices=['fixed', 'confidence'],
                        help="fixed(既定)=alphaをそのまま使う。confidence=teacherの位置別確信度に"
                             "応じてalphaを動的調整(distill.make_seq_kd_loss_v2参照)。"
                             "'confidence'指定時のみv2損失に切替。")
    parser.add_argument('--hard-agree-weight', type=float, default=0.0,
                        help="teacher argmaxとの一致を促す補助CE損失の重み(既定0.0=無効)。"
                             ">0指定時のみv2損失に切替。")

    # exit-head蒸留(OFA-KD式ロジット空間投影。既定None=無効で既存挙動を維持)
    parser.add_argument('--exit-layers', type=str, default=None,
                        help="使う中間活性のindex(カンマ区切り、例 '0,1')。"
                             "None(既定)=exit-head蒸留を使わない(既存挙動を完全維持)。"
                             "student=rddlgn/logicgptのみ対応(return_hidden実装済み)。"
                             "rddlgnはhidden_states=[encoder context, decoder P-layer出力]"
                             "の2点、logicgptは各LogicTransformerBlock出力"
                             "(config.n_layers点)からindexで選ぶ。--kd on"
                             "(teacherロジットが必要)前提であり、--kd offと併用不可。")
    parser.add_argument('--exit-weight', type=float, default=0.3,
                        help='exit KD損失の重み(total_lossへの加算係数)。')

    # 訓練
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=20000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--eval-max-chars', type=int, default=100_000,
                        help='tiny_shakespeare検証データの末尾文字数')
    parser.add_argument('--emb-reg-ramp-start', type=int, default=1000)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=100000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)

    # 学習安定化: 全てデフォルトOFF(既存挙動をbitwise維持)。
    # CLIで明示した時のみ有効化する。
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='>0のときclip_grad_norm_を適用(既定0.0=off)。')
    parser.add_argument('--lr-schedule', type=str, default='const',
                        choices=['const', 'warmup-cosine'],
                        help='const(既定)=固定lr。warmup-cosine=線形warmup→cosine減衰。')
    parser.add_argument('--warmup-iters', type=int, default=500,
                        help='lr-schedule=warmup-cosine時の線形warmupステップ数。')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='lr-schedule=warmup-cosine時のcosine減衰の下限(lr*ratio)。')
    parser.add_argument('--gs-tau', type=float, default=None,
                        help='指定時、モデル内の全GroupSum.tauをこの値で上書きする。')

    # チェックポイント
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--init-ckpt', type=str, default=None,
                        help="訓練開始前にこのckptから**重みのみ**をロードする"
                             "(optimizer/schedulerは新規)。ckpt metadata['args']の"
                             "student/config/scaleが現行argsと不一致ならraiseする。")
    parser.add_argument('--init-step', type=int, default=0,
                        help="--init-ckptからfine-tuneする際、emb_reg/スケジュール系の"
                             "step評価に加算するオフセット(既定0)。fine-tune時にstep=0から"
                             "再開すると二値化正則化ランプ(get_emb_reg_weight)やalpha/temp"
                             "スケジュールが巻き戻ってしまう罠を避けるために使う。")

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    # Spectral正則化の一元設定(_backward_and_stepが参照)
    global _SPECTRAL_REG
    _SPECTRAL_REG = args.spectral_reg

    if args.train_data == 'teacher' and args.corpus is None:
        parser.error(
            '--train-data teacher(既定)では --corpus が必須です'
            '(gen_teacher_corpus.pyで生成した訓練コーパス.ptを指定してください)。'
            ' 実データを使う場合は --train-data real --data-fraction <割合> を指定してください。'
        )

    # exit-head蒸留の起動時バリデーション(exit-headはteacherロジットが
    # 必須なので--kd on前提。対応studentもreturn_hidden実装済みのrddlgn/logicgptのみ)。
    if args.exit_layers is not None:
        if args.kd == 'off':
            parser.error(
                '--exit-layers は --kd on 前提です(exit-head蒸留はteacherロジットへの'
                'KLが必須で、--kd off(scratch純CE)とは組み合わせられません)。'
            )
        if args.student not in ('rddlgn', 'logicgpt'):
            parser.error(
                f'--exit-layers は student=rddlgn/logicgpt のみ対応です'
                f'(return_hidden未実装: student={args.student!r})。'
            )

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_corpus_path = resolve_glob(args.eval_corpus)
    print(f'評価コーパス: {eval_corpus_path}')

    # logicgptはフル64トークン窓が必要(src/tgt_input/tgt_targetの3分割をしない)ため、
    # 専用modeのDatasetと専用collate_fnで別のloaderを構築する。
    dataset_mode = 'decoder_only' if args.student == 'logicgpt' else 'seq2seq'
    collate_fn = collate_raw_window if args.student == 'logicgpt' else collate_with_raw

    if args.train_data == 'real':
        # 実データprefixサブセット。val_reserve_charsはload_tiny_shakespeare_val
        # のmax_charsと揃えてdisjoint性を保つ(下のval_windows読み込みで同じ
        # args.eval_max_charsを使う)。
        train_ids = load_tiny_shakespeare_train(
            args.data_fraction, val_reserve_chars=args.eval_max_chars,
        )
        train_set = RealCharDataset(train_ids, mode=dataset_mode)
        corpus_path = f'real(fraction={args.data_fraction})'
        n_train_chars = len(train_ids)
    else:
        corpus_path = resolve_glob(args.corpus)
        print(f'訓練コーパス: {corpus_path}')
        train_set = TeacherCorpusDataset(corpus_path, mode=dataset_mode)
        n_train_chars = None

    eval_set = TeacherCorpusDataset(eval_corpus_path, mode=dataset_mode)
    n_train_windows = len(train_set)
    print(f'訓練サンプル: {len(train_set):,}  評価サンプル: {len(eval_set):,}')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
        collate_fn=collate_fn,
    )

    # teacherロード(オンラインロジット計算用。CPU固定運用でも明示デバイス)
    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=args.device)
    print(f'  teacher fingerprint={teacher.fingerprint}')

    # モデル構築
    if args.student == 'rddlgn':
        config = build_config_rddlgn(args)
        config.selection = args.selection
        model = RDDLGN(config, device=args.device).to(args.device)
        config_dict = asdict(config)
    elif args.student == 'attention':
        config = build_config_attention(args)
        config.selection = args.selection
        model = DLGNAttentionTranslator(config, device=args.device).to(args.device)
        config_dict = asdict(config)
    elif args.student == 'logicgpt':
        config = build_config_logicgpt(args)
        model = LogicGPT(config).to(args.device)
        config_dict = asdict(config)
    else:  # nextchar
        hidden_sizes = [int(x) for x in args.nextchar_hidden.split(',')]
        model = NextCharBaseline(
            vocab_size=STUDENT_VOCAB_SIZE, d_emb=args.d_emb, hidden_sizes=hidden_sizes,
            n_inputs=args.n_inputs, device=args.device,
        ).to(args.device)
        config_dict = {'d_emb': args.d_emb, 'hidden_sizes': hidden_sizes}
        if args.selection != 'soft':
            from griffith_logic.layers import LutLayer as _LutLayer
            for _m in model.modules():
                if isinstance(_m, _LutLayer):
                    _m.selection = args.selection

    # 重みロード: --init-ckpt指定時、モデル構築直後に**重みのみ**をロードする
    # (optimizer/schedulerは新規作成。fine-tune時にoptimizerモーメントを引き継がない設計)。
    # 事前にckpt metadataのstudent/config/scaleを検証してから重みをmodelへ流し込む
    # (load_checkpointはmodel.load_state_dictを即座に実行するため、一致検証は
    # torch.loadで先にmetadataだけ覗いてから行う)。
    init_ckpt_step = 0
    if args.init_ckpt is not None:
        ckpt_probe = torch.load(args.init_ckpt, map_location=args.device, weights_only=False)
        ckpt_args = ckpt_probe.get('metadata', {}).get('args', {})
        mismatches = []
        for key in ('student', 'config', 'scale'):
            ckpt_val = ckpt_args.get(key)
            cur_val = getattr(args, key)
            if ckpt_val != cur_val:
                mismatches.append(f'{key}: ckpt={ckpt_val!r} vs current={cur_val!r}')
        if mismatches:
            raise ValueError(
                f'--init-ckpt {args.init_ckpt} は現行argsと構成が一致しません: '
                f'{"; ".join(mismatches)}。'
                ' student/config/scaleが一致するckptを指定してください。'
            )
        del ckpt_probe  # model.load_state_dictはload_checkpoint内で行うため不要
        info = load_checkpoint(args.init_ckpt, model, optimizer=None, scheduler=None, device=args.device)
        init_ckpt_step = info['step']
        print(
            f'--init-ckpt {args.init_ckpt} から重みをロードしました'
            f' (step={info["step"]}, best_metric={info["best_metric"]})'
        )

    # τ_gs上書き: 指定時のみモデル内の全GroupSum.tauを差し替える。
    if args.gs_tau is not None:
        n_overridden = 0
        for m in model.modules():
            if isinstance(m, GroupSum):
                m.tau = args.gs_tau
                n_overridden += 1
        print(f'GroupSum.tau を {args.gs_tau} に上書き: {n_overridden}個')

    counts = model.count_parameters() if hasattr(model, 'count_parameters') else {
        'total': sum(p.numel() for p in model.parameters())
    }
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')

    # exit-head蒸留: --exit-layers指定時のみexit headsを構築する。
    # 訓練時のみのfloat足場(LogitExitHead)であり、推論時・変換時には破棄される
    # (griffith_logic/convert/distill.py の LogitExitHead docstring参照)。
    exit_ctx = None
    exit_head_modules = None
    exit_param_count = 0
    if args.exit_layers is not None:
        exit_layer_indices = [int(x) for x in args.exit_layers.split(',')]

        if args.student == 'rddlgn':
            hidden_dims = {
                0: config.k_layer_sizes[-1],   # encoder context
                1: config.p_layer_sizes[-1],   # decoder P-layer出力
            }
        else:  # logicgpt
            hidden_dims = dict.fromkeys(range(config.n_layers), config.d_model)

        n_points = 2 if args.student == 'rddlgn' else config.n_layers
        for idx in exit_layer_indices:
            if idx not in hidden_dims:
                parser.error(
                    f'--exit-layers の index {idx} は無効です'
                    f'(student={args.student} の中間活性は0..{n_points - 1})。'
                )

        exit_head_modules = nn.ModuleList([
            LogitExitHead(hidden_dim=hidden_dims[idx], vocab_size=STUDENT_VOCAB_SIZE)
            for idx in exit_layer_indices
        ]).to(args.device)
        exit_param_count = sum(p.numel() for p in exit_head_modules.parameters())
        print(
            f'exit-head蒸留: layers={exit_layer_indices} weight={args.exit_weight} '
            f'params={exit_param_count:,}'
        )

        exit_kd_loss = make_exit_kd_loss(
            exit_heads=list(exit_head_modules),
            weights=[1.0] * len(exit_layer_indices),
            temperature=args.distill_temperature,
        )
        exit_ctx = {
            'exit_kd_loss': exit_kd_loss,
            'exit_weight': args.exit_weight,
            'exit_layers': exit_layer_indices,
        }

    # 損失関数: --kd on(既定)はKD損失、--kd offはteacher非依存の純CE
    # kd-kind/alpha-schedule/temp-schedule/alpha-mode/hard-agree-weightが
    # 全てデフォルトのときはmake_seq_kd_loss(v1)をそのまま使う(bitwise非回帰)。
    # いずれかが非デフォルトのときのみmake_seq_kd_loss_v2に切替える。
    use_kd_v2 = (
        args.kd_kind != 'kl'
        or args.alpha_schedule is not None
        or args.temp_schedule is not None
        or args.alpha_mode != 'fixed'
        or args.hard_agree_weight != 0.0
    )
    if args.kd == 'off':
        loss_fn_ce = make_pure_ce_seq_loss(pad_id=PAD_ID)
        loss_fn = None
    elif use_kd_v2:
        alpha_spec = args.alpha_schedule if args.alpha_schedule is not None else args.distill_alpha
        temp_spec = args.temp_schedule if args.temp_schedule is not None else args.distill_temperature
        loss_fn_v2 = make_seq_kd_loss_v2(
            alpha=alpha_spec,
            temperature=temp_spec,
            standardize=args.distill_standardize,
            pad_id=PAD_ID,
            kind=args.kd_kind,
            alpha_mode=args.alpha_mode,
            hard_agree_weight=args.hard_agree_weight,
        )
        # v1クロージャ(loss_fn(student, target, teacher))と同じ3引数呼び出しに
        # stepを紛れ込ませずに済むよう、train_step_*からは常に`step=`キーワードで
        # 明示的に渡す設計にする。stepの値そのものへのinit_step/init_ckpt_step
        # オフセット適用は呼び出し元のtrain_step_*(init_step引数)側で行う
        # (ここで二重に加算しない)。
        def loss_fn(student_logits, target_ids, teacher_logits, step=0):
            return loss_fn_v2(student_logits, target_ids, teacher_logits, step=step)
        loss_fn_ce = None
    else:
        # v1(既存make_seq_kd_loss)は無改造。ただしtrain_step_*からは(v2と揃えて)
        # 常にstep=を渡せるよう、stepを吸収するだけの薄いラッパで包む
        # (v1クロージャ自体は3引数のまま、非回帰は変わらない)。
        loss_fn_v1 = make_seq_kd_loss(
            alpha=args.distill_alpha,
            temperature=args.distill_temperature,
            standardize=args.distill_standardize,
            pad_id=PAD_ID,
        )

        def loss_fn(student_logits, target_ids, teacher_logits, step=0):
            return loss_fn_v1(student_logits, target_ids, teacher_logits)
        loss_fn_ce = None
    emb_reg_ramp = (args.emb_reg_ramp_start, args.emb_reg_ramp_end, args.emb_reg_max_weight)

    # 実データ検証セット(BPC専用、data-free主張はこの評価専用利用で成立)
    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    # exit headsは同じoptimizerに追加する(独立optimizerを作らない設計。
    # モデル本体と同じlr/weight_decay/betasでまとめて最適化する)。
    optimizer_params = list(model.parameters())
    if exit_head_modules is not None:
        optimizer_params += list(exit_head_modules.parameters())

    optimizer = torch.optim.AdamW(
        optimizer_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # lrスケジュール: const(既定)はscheduler=None(既存挙動維持)。
    # warmup-cosineは線形warmup(0→1)後、cosineでmin_lr_ratioまで減衰する。
    if args.lr_schedule == 'warmup-cosine':
        warmup_iters = args.warmup_iters
        total_iters = args.num_iterations
        min_lr_ratio = args.min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_iters:
                return (current_step + 1) / max(1, warmup_iters)
            progress = (current_step - warmup_iters) / max(1, total_iters - warmup_iters)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    log, log_file = setup_logger(args.student, args.log_dir)
    log(f'START student={args.student} config={args.config} corpus={corpus_path} '
        f'train_data={args.train_data} data_fraction={args.data_fraction} kd={args.kd} '
        f'bs={args.batch_size} ni={args.num_iterations} device={args.device} '
        f'params={counts["total"]:,}')

    ckpt_prefix = f'kd_charlm_{args.student}_{args.config}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(ckpt_prefix, base_dir=args.checkpoint_dir)
    # hard BPC(eval mode=離散化後、論理ゲート実機性能相当)基準のbestは
    # 既存のsoft best.ptとは別ファイルに保存する(setup_checkpoint_dirは変更しない、
    # 最小侵襲のためbest_pathと同ディレクトリに派生パスを作るのみ)。
    best_hard_path = os.path.join(os.path.dirname(best_path), 'best_hard.pt')
    ckpt_metadata = {
        'args': vars(args),
        'config': config_dict,
        'teacher_fingerprint': teacher.fingerprint,
    }
    print(f'チェックポイント保存先: {run_dir}/')

    # teacher自身のBPC(起動時1回、比較アンカー)
    print('teacher自身のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    # run_name/tags拡張: train_dataの値(real{fraction}/teacher)とkd(scratch/kd_a{alpha})
    # を既存命名規則(kd_charlm_{student}_{config}_...)に追記する。
    if args.train_data == 'real':
        data_tag = f'real{args.data_fraction}'
    else:
        data_tag = 'teacher'
    kd_tag = 'scratch' if args.kd == 'off' else f'kd_a{args.distill_alpha}'
    run_name = (
        f'kd_charlm_{args.student}_{args.config}_{data_tag}_{kd_tag}'
        f'_T{args.distill_temperature}_ni{args.num_iterations}'
    )
    # 学習安定化フラグがデフォルト値以外のときのみrun_nameに追記する
    # (既存run名の互換性を維持するため、デフォルト時は何も付かない)。
    if args.grad_clip > 0:
        run_name += f'_clip{args.grad_clip}'
    if args.lr_schedule == 'warmup-cosine':
        run_name += '_wc'
    if args.gs_tau is not None:
        run_name += f'_gstau{args.gs_tau}'
    # HG(hard-gap): ゲート選択方式(gumbel_st/hard_st)とtau-annealを非デフォルト時のみ
    # 追記する(Phase 1のarch×selection格子を識別。既存soft runの名前は不変)。
    if args.selection != 'soft':
        run_name += f'_sel{args.selection}'
        if args.tau_anneal:
            run_name += 'anneal'
    # KD損失v2関連フラグが非デフォルトのときのみrun_nameに追記する。
    if args.kd_kind == 'dist':
        run_name += '_dist'
    if args.hard_agree_weight != 0.0:
        run_name += f'_ha{args.hard_agree_weight}'
    if args.alpha_schedule is not None:
        run_name += '_asch'
    if args.temp_schedule is not None:
        run_name += '_tsch'
    if args.alpha_mode == 'confidence':
        run_name += '_conf'
    # --init-ckpt指定時のみrun_nameに追記する。
    if args.init_ckpt is not None:
        run_name += f'_initstep{init_ckpt_step}'
    # --exit-layers指定時のみrun_nameに追記する
    # (既存run名の互換性を維持するため、未指定時は何も付かない)。
    if args.exit_layers is not None:
        exit_layers_tag = args.exit_layers.replace(',', '-')
        run_name += f'_exit{exit_layers_tag}w{args.exit_weight}'
    wandb_config = {
        **vars(args), **config_dict, 'teacher_fingerprint': teacher.fingerprint,
        'teacher_bpc': teacher_bpc,
        'data_fraction': args.data_fraction,
        'train_data': args.train_data,
        'kd': args.kd,
        'n_train_chars': n_train_chars,
        'n_train_windows': n_train_windows,
        'init_ckpt_step': init_ckpt_step,
        'exit_param_count': exit_param_count,
    }
    use_wandb = False
    if not args.no_wandb:
        tags = ['kd', 'charlm', 'datafree', args.student, 'dataeff', args.train_data]
        if args.kd == 'off':
            tags.append('scratch')
        # HG(hard-gap): selection/tau-annealでフィルタできるようにタグ付け。
        tags.append(f'sel_{args.selection}')
        if args.tau_anneal:
            tags.append('tau_anneal')
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=tags,
            group='kd-charlm',
        )

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション')
    print('=' * 60)

    tau_sched = None
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        tau_sched = TauAnnealScheduler(
            model, tau_0=args.tau_b0, tau_min=args.tau_b_min,
            total_steps=args.num_iterations,
            anneal_noise=(args.selection == 'gumbel_st'),
        )

    train_start = time.time()
    best_bpc = float('inf')
    best_bpc_hard = float('inf')

    # BP-free学習の収束・停滞・時間計測(hard BPC基準)。
    bpc_hard_tracker = ConvergenceTracker(mode='min')
    wall_clock = WallClock()
    BPC_HARD_THRESHOLD = 3.0

    # --init-ckptからのfine-tune時、emb_reg_ramp/損失スケジュールの評価stepに
    # 加算するオフセット(--init-step明示値 + ckptに記録されたstep)。既定は両方0。
    total_init_step = args.init_step + init_ckpt_step

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='train', total=args.num_iterations,
    )

    for step, batch in pbar:
        exit_loss_value = 0.0
        if args.student == 'logicgpt':
            # batch = (B, 64) 生トークン列(collate_raw_window)
            window = batch
            if args.kd == 'off':
                # scratch: teacher.next_token_logitsを一切呼ばない(GPU無駄+純度)。
                total_loss, main_loss, emb_loss = train_step_decoder_only_scratch(
                    model, window,
                    optimizer, step, loss_fn_ce, args.device, emb_reg_ramp,
                    scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                )
            else:
                total_loss, main_loss, emb_loss, exit_loss_value = train_step_decoder_only(
                    model, teacher, window,
                    optimizer, step, loss_fn, args.device, emb_reg_ramp,
                    scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                    exit_ctx=exit_ctx,
                )
        else:
            # batch = (src, tgt_input, tgt_target) 3-tuple(collate_with_raw)
            src, tgt_input, tgt_target = batch
            src = src.to(args.device)
            tgt_input = tgt_input.to(args.device)
            tgt_target_dev = tgt_target.to(args.device)

            if args.kd == 'off':
                # scratch: compute_teacher_logits_for_batch(=teacher呼び出し)を
                # 一切行わない。raw_tokens連結もteacher専用の前処理なので省略する。
                if args.student == 'nextchar':
                    total_loss, main_loss, emb_loss = train_step_nextchar_scratch(
                        model, src, tgt_target_dev[:, 0],
                        optimizer, step, loss_fn_ce, args.device, emb_reg_ramp,
                        scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                    )
                else:
                    total_loss, main_loss, emb_loss = train_step_seq2seq_scratch(
                        model, src, tgt_input, tgt_target_dev,
                        optimizer, step, loss_fn_ce, args.device, emb_reg_ramp,
                        scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                    )
            else:
                # raw_tokens(64) = src(32) + tgt_target(32)。split_src_tgtの定義通り
                # 連結すれば元の64トークン列に一致する(teacherロジット計算用)。
                raw_tokens = torch.cat([src.cpu(), tgt_target.cpu()], dim=1)

                if args.student == 'nextchar':
                    total_loss, main_loss, emb_loss = train_step_nextchar(
                        model, teacher, src, tgt_target_dev[:, 0], raw_tokens,
                        optimizer, step, loss_fn, args.device, emb_reg_ramp,
                        scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                    )
                else:
                    total_loss, main_loss, emb_loss, exit_loss_value = train_step_seq2seq(
                        model, teacher, src, tgt_input, tgt_target_dev, raw_tokens,
                        optimizer, step, loss_fn, args.device, emb_reg_ramp,
                        scheduler=scheduler, grad_clip=args.grad_clip, init_step=total_init_step,
                        exit_ctx=exit_ctx,
                    )

        if tau_sched is not None:
            tau_sched.step()

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{total_loss:.4f}', main=f'{main_loss:.4f}', eta=eta_str)

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            agree_soft = evaluate_teacher_agreement(model, args.student, eval_loader, teacher, args.device, use_soft=True)
            agree_hard = evaluate_teacher_agreement(model, args.student, eval_loader, teacher, args.device, use_soft=False)
            bpc_soft = evaluate_bpc(model, args.student, val_windows, args.device, use_soft=True)
            bpc_hard = evaluate_bpc(model, args.student, val_windows, args.device, use_soft=False)

            is_best = bpc_soft < best_bpc
            if is_best:
                best_bpc = bpc_soft

            is_best_hard = bpc_hard < best_bpc_hard
            if is_best_hard:
                best_bpc_hard = bpc_hard

            # BP-free学習の収束・停滞・時間計測(hard BPC基準)。
            saved_step_for_metrics = step + 1 + total_init_step
            bpc_hard_tracker.record(saved_step_for_metrics, bpc_hard)
            steps_to_bpc3 = bpc_hard_tracker.steps_to_threshold(BPC_HARD_THRESHOLD, mode='min')
            plateau_step = bpc_hard_tracker.plateau_step()
            wall_clock_sec = wall_clock.elapsed_sec()

            # 保存するstepは「このrunの反復数」ではなく「累積step」
            # (total_init_step込み)にする。--init-ckptで多段fine-tuneする場合に
            # 次段のinit_ckpt_stepが正しく引き継がれるようにするため。
            saved_step = step + 1 + total_init_step
            save_checkpoint(latest_path, model, optimizer, scheduler, saved_step, best_bpc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler, saved_step, best_bpc, ckpt_metadata)
            if is_best_hard:
                save_checkpoint(
                    best_hard_path, model, optimizer, scheduler, saved_step, best_bpc_hard, ckpt_metadata,
                )

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1)
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            emb_w = get_emb_reg_weight(step + total_init_step, *emb_reg_ramp)

            sample_text = generate_sample(model, args.student, teacher, args.device, length=300)

            current_lr = optimizer.param_groups[0]['lr']
            exit_str = f' exit={exit_loss_value:.4f}' if exit_ctx is not None else ''
            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f}(w={emb_w:.4f}){exit_str}  '
                f'agree(soft)={agree_soft:.4f} agree(hard)={agree_hard:.4f}  '
                f'BPC(soft)={bpc_soft:.4f}  BPC(hard)={bpc_hard:.4f}  teacher_BPC={teacher_bpc:.4f}  lr={current_lr:.6f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'{"  ★ BEST(hard)" if is_best_hard else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )
            print(f'  生成サンプル(300字): {sample_text[:300]!r}')

            log(f'ITER {step+1}/{args.num_iterations} loss={total_loss:.4f} main={main_loss:.4f} '
                f'emb={emb_loss:.4f} exit={exit_loss_value:.4f} agree_soft={agree_soft:.4f} agree_hard={agree_hard:.4f} '
                f'bpc_soft={bpc_soft:.4f} bpc_hard={bpc_hard:.4f} teacher_bpc={teacher_bpc:.4f} '
                f'best={str(is_best).lower()} best_hard={str(is_best_hard).lower()}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'loss/emb_reg': emb_loss,
                    'loss/emb_reg_weight': emb_w,
                    'loss/exit_kd': exit_loss_value,
                    'agreement/soft': agree_soft,
                    'agreement/hard': agree_hard,
                    'bpc/student_soft': bpc_soft,
                    'bpc/student_hard': bpc_hard,
                    'bpc/best_hard': best_bpc_hard,
                    'metrics/steps_to_bpc3.0': steps_to_bpc3 if steps_to_bpc3 is not None else -1,
                    'metrics/plateau_step': plateau_step if plateau_step is not None else -1,
                    'metrics/wall_clock_sec': wall_clock_sec,
                    'bpc/teacher': teacher_bpc,
                    'bpc/best': best_bpc,
                    'sample_text': sample_text,
                    'lr': optimizer.param_groups[0]['lr'],
                }, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終BPC(best soft): {best_bpc:.4f}  最終BPC(best hard): {best_bpc_hard:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_bpc={best_bpc:.4f} best_bpc_hard={best_bpc_hard:.4f} total_time={total_time}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_bpc': best_bpc, 'teacher_bpc': teacher_bpc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
