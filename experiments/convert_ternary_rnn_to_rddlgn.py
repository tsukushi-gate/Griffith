"""量子化起点の転写(RNN版): 量子化(1bit/1.58bit)RNN → RDDLGN 転写 → polish(copyタスク)。

`convert_rnn_to_rddlgn.py`(D-RNN1、float RNN → RDDLGN構造転写)を土台に、
source RNNセルを **量子化**(1bit sign / 1.58bit ternary)してから
`transcribe_rnn_cell` で RDDLGN の再帰 k_layer へ転写する版。量子化MLPの転写実験
(`convert_ternary_to_dwn_mnist.py`、量子化MLP→DWN)の「量子化起点が転写崖を
回避するか」という問いを、**再帰(RNN)**へ拡張する。

## 中心対照(4アーム、同一パイプライン・同一fan-in F を共有)

    float    枝刈りfloat重みのRNNセルを転写(既存D-RNN1相当の基準)。
    bin      1bit sign 量子化(0マスクなし=密、binarize_weights)。
    ternary  1.58bit absmean 量子化(0マスク=疎、quantize_ternary+
             transcribe_rnn_cell(masks=...) でマスク内のみ転写)。
    scratch  同容量RDDLGN studentをランダム初期化からCE polish(下限対照)。

## 予測される新課題(設計に織り込む)

1. **連続活性の二値化ギャップ + 再帰の誤差蓄積**: tanh RNNを閾値ゲート化する
   と隠れ状態が毎ステップ1bitへ丸められ、再帰でステップを重ねると誤差が
   蓄積し得る。→ 量子化teacherを `hard_forward`(完全二値化、閾値ゲート意味論
   =transcribe_rnn_cellと同一)で再評価して**参照精度(source acc)**を測り、
   seq_len別に転写後hard精度の劣化を測る。
2. **fan-in疎化が量子化マスクを無視**: 従来 transcribe_rnn_cell は
   sparsify_fanin のみ。→ Step1で追加した `masks` 引数(sparsify_from_mask
   経由)で ternary の0マスク(=枝刈り済み構造)を尊重して転写する。

## QAT(量子化aware fine-tune)

`convert_ternary_to_dwn_mnist.py` の `_qat_finetune` に倣い、RNNの再帰forwardで
STE量子化重み(forward=量子化、backward=shadow float)を使って source を回復
する。枝刈り0は step ごとに再ゼロして維持する(順序=枝刈り→量子化→QAT→bake)。

## 指標

    source(hard_forward) exact-match acc / transcribe直後 hard exact-match acc /
    polish後 soft・hard acc / oracle_loss(=source − transcribe) /
    k_layer_agreement(転写直後 vs teacher hard隠れ状態) / seq_len別劣化。
    arm比較: ternary−scratch(一次指標)、ternary−bin(0マスクの純効果)。

## 使用例

    # 煙実行(数十秒、wandb無効、1 seed・少epoch)
    uv run python experiments/convert_ternary_rnn_to_rddlgn.py --quick --no-wandb

    # 本実行(SSH GPU/ローカル、wandb online)
    uv run python experiments/convert_ternary_rnn_to_rddlgn.py --fanin 6 --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# D-RNN1(float版)から機構を最大限再利用する。
from convert_rnn_to_rddlgn import (
    RnnCopyTeacher,
    RnnStructDlgnStudent,
    _auto_device,
    build_scratch_student,
    combined_fanin_prune_,
    exact_match_acc_hard_teacher,
    exact_match_acc_soft,
    exact_match_acc_student,
    make_copy_loaders,
    train_teacher,
)
from data.synthetic import CopyDataset
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.transcribe_rnn import transcribe_rnn_cell
from griffith_logic.ir.weights import restore_lut_layer

load_dotenv()

ARMS = ('float', 'bin', 'ternary')


# ============================================================
# source RNNセルの量子化(枝刈り0を保持)+ STE QAT
# ============================================================

def _quantize_rnn_cell_(teacher: RnnCopyTeacher, arm: str) -> torch.Tensor:
    """teacherのweight_ih/weight_hhを結合fan-inのarm別量子化で in-place bake。

    combined=[w_ih|w_hh] を量子化し、bin/ternaryとも既存の枝刈り0
    (weight==0の位置)を保持する。ternaryはさらにabsmean三値化の0(=枝刈り)を
    加える。戻り値は combined マスク (d_hidden, d_in+d_hidden) bool。

    - float:   量子化せず、既存の枝刈り0だけをマスクとして返す。
    - bin:     sign×alpha(枝刈り内平均|W|)。0マスクは枝刈り0のみ(密)。
    - ternary: absmean per-channel 三値化。0マスク=枝刈り0 ∪ 三値化0。
    """
    d_in = teacher.weight_ih.shape[1]
    with torch.no_grad():
        w_ih = teacher.weight_ih.data
        w_hh = teacher.weight_hh.data
        w_comb = torch.cat([w_ih, w_hh], dim=1)  # (d_hidden, d_in+d_hidden)
        prune_mask = (w_comb != 0)               # 既存の枝刈り0を保持

        if arm == 'float':
            mask = prune_mask
            q_comb = w_comb
        elif arm == 'bin':
            pm = prune_mask.to(w_comb.dtype)
            alpha = (w_comb.abs() * pm).sum() / pm.sum().clamp_min(1.0)
            q_comb = torch.sign(w_comb) * alpha * pm
            mask = prune_mask  # 0マスクなし=密(枝刈り0のみ)
        elif arm == 'ternary':
            # binarize.quantize_ternary(per='channel')の枝刈り版:
            # scaleは枝刈り内平均|W|(枝刈り0が平均を薄めないため)。
            pm = prune_mask.to(w_comb.dtype)
            denom = pm.sum(dim=1, keepdim=True).clamp_min(1.0)
            scale = ((w_comb.abs() * pm).sum(dim=1, keepdim=True) / denom
                     ).clamp_min(1e-5)
            q = torch.clamp(torch.round(w_comb / scale), -1.0, 1.0) * pm
            q_comb = q * scale
            mask = (q != 0)  # 枝刈り0 ∪ 三値化0
        else:
            raise ValueError(f'未対応 arm: {arm}')

        teacher.weight_ih.data.copy_(q_comb[:, :d_in])
        teacher.weight_hh.data.copy_(q_comb[:, d_in:])
    return mask.cpu()


def _quantize_rnn_cell_ste(teacher: RnnCopyTeacher, arm: str,
                           prune_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """STE量子化した (w_ih, w_hh)(forward=量子化、backward=shadow)を返す。

    QAT forward用。combined重みをarm別に量子化し、STEで勾配をshadow(float)
    重みへ素通しする。scale/alphaは prune_mask 内平均で計算する。
    """
    d_in = teacher.weight_ih.shape[1]
    w_ih, w_hh = teacher.weight_ih, teacher.weight_hh
    w_comb = torch.cat([w_ih, w_hh], dim=1)
    pm = prune_mask.to(w_comb.dtype).to(w_comb.device)
    if arm == 'bin':
        alpha = (w_comb.detach().abs() * pm).sum() / pm.sum().clamp_min(1.0)
        q = torch.sign(w_comb) * alpha
    elif arm == 'ternary':
        denom = pm.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = ((w_comb.detach().abs() * pm).sum(dim=1, keepdim=True) / denom
                 ).clamp_min(1e-5)
        q = torch.clamp(torch.round(w_comb / scale), -1.0, 1.0) * scale
    else:  # float
        q = w_comb
    q = q * pm
    q_ste = w_comb + (q - w_comb).detach()  # STE
    return q_ste[:, :d_in], q_ste[:, d_in:]


def qat_finetune_rnn(teacher: RnnCopyTeacher, arm: str, prune_mask: torch.Tensor,
                     train_loader: DataLoader, *, epochs: int, lr: float,
                     device: str) -> None:
    """量子化aware fine-tune(STE forward)でRNNセルの量子化 source を回復する。

    毎forwardで weight_ih/weight_hh をSTE量子化し、再帰RNN forward(閾値ゲート
    でなくtanh soft、勾配を通すため)+線形readoutでCE。backwardはshadow float
    重みへ。枝刈り0はstepごとに再ゼロして維持する(shadow復活を防ぐ)。
    """
    if arm == 'float' or epochs <= 0:
        return
    d_in = teacher.weight_ih.shape[1]
    pm_ih = prune_mask[:, :d_in].to(device)
    pm_hh = prune_mask[:, d_in:].to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    teacher.to(device).train()
    for _ in range(epochs):
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            x = teacher._embed_binary(src)  # (B,S,d_emb) STE二値
            batch, seq_len = src.shape
            wq_ih, wq_hh = _quantize_rnn_cell_ste(teacher, arm, prune_mask)
            h = torch.zeros(batch, teacher.d_hidden, device=device, dtype=x.dtype)
            h_list = []
            for t in range(seq_len):
                z = x[:, t, :] @ wq_ih.T + h @ wq_hh.T \
                    + teacher.bias_ih + teacher.bias_hh
                h = torch.tanh(z)
                h_list.append(h)
            logits = teacher.head(torch.stack(h_list, dim=1))
            opt.zero_grad()
            loss = F.cross_entropy(
                logits.reshape(-1, teacher.vocab_size), tgt.reshape(-1),
            )
            loss.backward()
            opt.step()
            with torch.no_grad():  # 枝刈り0を維持
                teacher.weight_ih.mul_(pm_ih)
                teacher.weight_hh.mul_(pm_hh)


# ============================================================
# seq_len別のhard劣化測定(再帰誤差蓄積の可視化)
# ============================================================

@torch.no_grad()
def _per_step_agreement(student: RnnStructDlgnStudent, teacher: RnnCopyTeacher,
                        calib_tokens: torch.Tensor, d_hidden: int,
                        device: str) -> list[float]:
    """各タイムステップでの k_layer隠れ状態 vs teacher hard隠れ状態の一致率。

    再帰の誤差蓄積を可視化する(t=0で一致率高くても後段で崩れるか)。
    """
    student.eval()
    calib_x = teacher._embed_binary(calib_tokens.to(device))
    seq_len = calib_tokens.shape[1]
    h_student = torch.full((calib_tokens.shape[0], d_hidden), 0.5, device=device)
    h_teacher = torch.zeros(calib_tokens.shape[0], d_hidden, device=device)
    per_step = []
    for t in range(seq_len):
        x_t = calib_x[:, t, :]
        combined = torch.cat([x_t, h_student], dim=-1)
        h_student = student.k_layer(combined)
        z = x_t @ teacher.weight_ih.T + h_teacher @ teacher.weight_hh.T \
            + teacher.bias_ih + teacher.bias_hh
        h_teacher = (z > 0).to(x_t.dtype)
        per_step.append(float((h_student == h_teacher).float().mean()))
    student.train()
    return per_step


# ============================================================
# 転写 student 構築(量子化マスク対応)
# ============================================================

def build_quantized_structural_student(
    teacher: RnnCopyTeacher, arm: str, comb_mask: torch.Tensor, *,
    vocab_size, d_emb, d_hidden, fanin, head_sizes, device, seed,
    calib_tokens: torch.Tensor,
) -> tuple[RnnStructDlgnStudent, dict]:
    """量子化済みteacher RNNセルを転写し、student k_layer をin-place初期化する。

    ternary armは comb_mask(結合0マスク)を transcribe_rnn_cell(masks=...) へ渡し、
    sparsify_from_mask 経由でマスク内のみ転写する。float/binはmasks=Noneで
    従来の|W|上位k疎化(binは0マスクなし=密なため)。
    """
    torch.manual_seed(seed)
    student = RnnStructDlgnStudent(
        vocab_size, d_emb, d_hidden, fanin=fanin, head_sizes=head_sizes,
        parameterization='warp', device=device,
    ).to(device)

    # 埋め込みテーブルを引き継ぐ(D-RNN1と同じ、意味論一致)
    with torch.no_grad():
        student.embedding.embedding.weight.copy_(
            teacher.embedding.weight.detach().to(device)
        )

    # teacher(量子化済み hard 意味論)を較正データ上で走らせ combined を収集
    calib_x = teacher._embed_binary(calib_tokens.to(device))
    with torch.no_grad():
        h = torch.zeros(calib_tokens.shape[0], d_hidden, device=device)
        combined_rows = []
        for t in range(calib_tokens.shape[1]):
            combined_rows.append(torch.cat([calib_x[:, t, :], h], dim=-1))
            z = calib_x[:, t, :] @ teacher.weight_ih.T + h @ teacher.weight_hh.T \
                + teacher.bias_ih + teacher.bias_hh
            h = (z > 0).to(calib_x.dtype)
        calib_combined = torch.cat(combined_rows, dim=0).cpu().numpy().astype(np.int64)

    # ternaryのみ量子化0マスクを尊重して転写(bin/floatは密→|W|上位k)
    masks = comb_mask if arm == 'ternary' else None
    result = transcribe_rnn_cell(
        teacher.weight_ih.detach().cpu(), teacher.weight_hh.detach().cpu(),
        teacher.bias_ih.detach().cpu(), teacher.bias_hh.detach().cpu(),
        n_inputs=fanin, calib=calib_combined, masks=masks,
    )
    restore_lut_layer(student.k_layer, result['indices'], result['truth_tables'])

    per_step = _per_step_agreement(student, teacher, calib_tokens, d_hidden, device)
    combined_width = d_emb + d_hidden
    info = {
        'fanin': fanin,
        'combined_width': combined_width,
        'exact_expected': combined_width <= fanin,
        'k_layer_agreement': float(np.mean(per_step)) if per_step else 0.0,
        'per_step_agreement': per_step,
        'comb_mask_mean_fanin': float(comb_mask.float().sum(dim=1).mean()),
        'comb_mask_min_fanin': int(comb_mask.float().sum(dim=1).min()),
    }
    return student, info


def polish_student(student: RnnStructDlgnStudent, train_loader, eval_loader, *,
                   vocab_size, args, device, curriculum: bool) -> dict:
    """転写起点studentのsoft磨き(良init破壊を避け控えめLR・best採用)。

    epochごとにhard accを測り最良を採る(convert_ternary_to_dwn_mnist._polish
    と同方針)。curriculum=Trueなら後半epochでk_layer/headのselectionを
    'hard_st'(STE)へ切り替える soft→hard_st curriculum(翻訳知見)。
    """
    from griffith_logic.lut_layer import LutLayer

    def _set_selection(sel: str):
        for m in student.modules():
            if isinstance(m, LutLayer):
                m.selection = sel

    hard_before = exact_match_acc_student(student, eval_loader, device, hard=True)
    best_hard = hard_before
    opt = torch.optim.Adam(student.parameters(), lr=args.polish_lr)
    switch_at = args.polish_epochs // 2 if curriculum else args.polish_epochs + 1
    for ep in range(args.polish_epochs):
        if curriculum and ep == switch_at:
            _set_selection('hard_st')
        student.train()
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = student(src)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            opt.step()
        best_hard = max(best_hard, exact_match_acc_student(
            student, eval_loader, device, hard=True))
    if curriculum:
        _set_selection('soft')  # 評価一貫性のため既定へ戻す
    soft_after = exact_match_acc_student(student, eval_loader, device, hard=False)
    hard_after = exact_match_acc_student(student, eval_loader, device, hard=True)
    return {
        'hard_before': hard_before, 'soft_after': soft_after,
        'hard_after': hard_after, 'hard_best': best_hard,
    }


# ============================================================
# 1 arm のパイプライン(量子化→QAT→転写→polish)
# ============================================================

def run_arm(arm, base_teacher, train_loader, eval_loader, calib_tokens, *,
            vocab_size, d_emb, d_hidden, args, device, seed) -> tuple[dict, RnnStructDlgnStudent]:
    print(f'\n=== arm {arm} ===')
    teacher = copy.deepcopy(base_teacher).to(device)

    # ---- 量子化(枝刈り0保持) + QAT回復 + 最終bake ----
    prune_mask = (torch.cat([teacher.weight_ih.data, teacher.weight_hh.data],
                            dim=1) != 0).cpu()
    _quantize_rnn_cell_(teacher, arm)  # 初期bake(QAT初期点)
    qat_finetune_rnn(teacher, arm, prune_mask, train_loader,
                     epochs=args.qat_epochs, lr=args.qat_lr, device=device)
    comb_mask = _quantize_rnn_cell_(teacher, arm)  # QAT後の最終bake+最終マスク

    # ---- source(閾値ゲート hard 意味論)参照精度 ----
    src_hard = exact_match_acc_hard_teacher(teacher, eval_loader, device)
    print(f'  [{arm}] 量子化source hard exact-match acc={src_hard:.4f}  '
          f'comb_mask mean-fanin={float(comb_mask.float().sum(dim=1).mean()):.2f}')

    # ---- 転写(量子化マスク対応) ----
    student, info = build_quantized_structural_student(
        teacher, arm, comb_mask, vocab_size=vocab_size, d_emb=d_emb,
        d_hidden=d_hidden, fanin=args.fanin, head_sizes=args.head_sizes,
        device=device, seed=seed + 3000, calib_tokens=calib_tokens,
    )
    transcribe_hard = exact_match_acc_student(student, eval_loader, device, hard=True)
    ps = ', '.join(f'{a:.2f}' for a in info['per_step_agreement'])
    print(f'  [{arm}] transcribe直後 hard exact-match acc={transcribe_hard:.4f}  '
          f'k_layer_agreement={info["k_layer_agreement"]:.4f}  '
          f'per-step=[{ps}]')

    # ---- soft polish(best採用、controlで curriculum) ----
    t0 = time.time()
    pol = polish_student(student, train_loader, eval_loader, vocab_size=vocab_size,
                         args=args, device=device, curriculum=args.curriculum)
    dt = round(time.time() - t0, 2)
    print(f"  [{arm}] polish hard {pol['hard_before']:.4f}->{pol['hard_after']:.4f} "
          f"(best {pol['hard_best']:.4f}) soft={pol['soft_after']:.4f} ({dt}s)")

    return {
        'arm': arm,
        'source_hard_acc': src_hard,
        'transcribe_hard_acc': transcribe_hard,
        'polish_hard_before': pol['hard_before'],
        'polish_hard_after': pol['hard_after'],
        'polish_hard_best': pol['hard_best'],
        'polish_soft_after': pol['soft_after'],
        'oracle_loss': src_hard - transcribe_hard,
        'polish_gain': pol['hard_best'] - pol['hard_before'],
        'k_layer_agreement': info['k_layer_agreement'],
        'per_step_agreement': info['per_step_agreement'],
        'comb_mask_mean_fanin': info['comb_mask_mean_fanin'],
        'comb_mask_min_fanin': info['comb_mask_min_fanin'],
        'polish_seconds': dt,
    }, student


# ============================================================
# 1 seed
# ============================================================

def run_one_seed(seed: int, args, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab_size, seq_len = args.vocab_size, args.seq_len
    d_emb, d_hidden = args.d_emb, args.d_hidden

    train_loader, eval_loader = make_copy_loaders(
        vocab_size, seq_len, args.num_train, args.num_eval, args.batch_size,
        seed=seed,
    )
    calib_tokens = next(iter(DataLoader(
        CopyDataset(vocab_size, seq_len, args.calib_samples),
        batch_size=args.calib_samples,
    )))[0]

    # ---- teacher(float)学習 → 結合fan-in枝刈り(transcribe前提のF制約) ----
    torch.manual_seed(seed)
    teacher = RnnCopyTeacher(vocab_size, d_emb, d_hidden).to(device)
    teacher_soft = train_teacher(
        teacher, train_loader, eval_loader, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device,
    )
    combined_fanin_prune_(
        teacher, args.fanin, train_loader,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    teacher_soft = exact_match_acc_soft(teacher, eval_loader, device)
    teacher_hard = exact_match_acc_hard_teacher(teacher, eval_loader, device)
    print(f'[seed={seed}] teacher(枝刈り後) soft={teacher_soft:.4f} '
          f'hard(閾値ゲート)={teacher_hard:.4f} (fanin<={args.fanin})')

    result: dict = {
        'seed': seed, 'teacher_soft_acc': teacher_soft,
        'teacher_hard_acc': teacher_hard,
    }

    # ---- 量子化アーム(float/bin/ternary) ----
    ternary_student = None
    for arm in ARMS:
        res, student = run_arm(
            arm, teacher, train_loader, eval_loader, calib_tokens,
            vocab_size=vocab_size, d_emb=d_emb, d_hidden=d_hidden,
            args=args, device=device, seed=seed,
        )
        result[arm] = res
        if arm == 'ternary':
            ternary_student = student

    # ---- scratch対照(同容量RDDLGN studentをランダム初期化→同epochs polish) ----
    scratch = build_scratch_student(
        vocab_size, d_emb, d_hidden, fanin=args.fanin, head_sizes=args.head_sizes,
        device=device, seed=seed + 7000,
    )
    s_before = exact_match_acc_student(scratch, eval_loader, device, hard=True)
    opt = torch.optim.Adam(scratch.parameters(), lr=args.polish_lr)
    best_s = s_before
    for _ in range(args.polish_epochs):
        scratch.train()
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = scratch(src)
            F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1)).backward()
            opt.step()
        best_s = max(best_s, exact_match_acc_student(scratch, eval_loader, device, hard=True))
    result['scratch'] = {'hard_before': s_before, 'hard_best': best_s}
    print(f'[seed={seed}] scratch polish hard {s_before:.4f}->{best_s:.4f}')

    # ---- 派生指標(best hard で比較。良init破壊を避けbest採用方針) ----
    result['ternary_vs_scratch'] = (
        result['ternary']['polish_hard_best'] - best_s)
    result['ternary_vs_bin'] = (
        result['ternary']['polish_hard_best'] - result['bin']['polish_hard_best'])
    result['ternary_vs_float'] = (
        result['ternary']['polish_hard_best'] - result['float']['polish_hard_best'])
    del ternary_student
    return result


# ============================================================
# 集計
# ============================================================

def _mean_std(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results, args) -> dict:
    print('\n' + '=' * 96)
    print('量子化起点の転写(RNN): 量子化RNN → RDDLGN 転写→polish(copyタスク) — 結果')
    print('=' * 96)
    summary = {'arms': {}}
    print('| arm | source hard | transcribe hard | polish後 hard(best) | '
          'oracle_loss | k_agree | mean-fanin |')
    print('|' + '---|' * 7)
    for arm in ARMS:
        src = _mean_std([r[arm]['source_hard_acc'] for r in results])
        tr = _mean_std([r[arm]['transcribe_hard_acc'] for r in results])
        aft = _mean_std([r[arm]['polish_hard_best'] for r in results])
        orc = _mean_std([r[arm]['oracle_loss'] for r in results])
        ka = _mean_std([r[arm]['k_layer_agreement'] for r in results])
        mf = _mean_std([r[arm]['comb_mask_mean_fanin'] for r in results])
        print(f'| {arm} | {src[0]:.4f} | {tr[0]:.4f} | {aft[0]:.4f}±{aft[1]:.4f} | '
              f'{orc[0]:+.4f} | {ka[0]:.4f} | {mf[0]:.2f} |')
        summary['arms'][arm] = {
            'source_hard_mean': src[0], 'transcribe_hard_mean': tr[0],
            'polish_best_mean': aft[0], 'polish_best_std': aft[1],
            'oracle_loss_mean': orc[0], 'k_layer_agreement_mean': ka[0],
            'comb_mask_mean_fanin_mean': mf[0],
        }
    sc = _mean_std([r['scratch']['hard_best'] for r in results])
    summary['scratch_hard_best_mean'] = sc[0]
    print(f'| scratch(下限) | - | - | {sc[0]:.4f}±{sc[1]:.4f} | - | - | - |')

    tt = _mean_std([r['teacher_hard_acc'] for r in results])
    print(f'\nfloat teacher参照: hard(閾値ゲート)exact-match={tt[0]:.4f}±{tt[1]:.4f}')
    summary['teacher_hard_acc_mean'] = tt[0]

    # seq_len別劣化(ternary armのper-step agreement平均)
    n_steps = args.seq_len
    per_step_mean = []
    for t in range(n_steps):
        vals = [r['ternary']['per_step_agreement'][t] for r in results
                if len(r['ternary']['per_step_agreement']) > t]
        per_step_mean.append(round(float(np.mean(vals)), 4) if vals else None)
    print(f'ternary per-step k_layer_agreement(t=0..{n_steps - 1}, 再帰誤差蓄積): '
          f'{per_step_mean}')
    summary['ternary_per_step_agreement'] = per_step_mean

    print('\n--- 仮説判定(mean over seeds、best hard) ---')
    tvs = _mean_std([r['ternary_vs_scratch'] for r in results])
    tvb = _mean_std([r['ternary_vs_bin'] for r in results])
    tvf = _mean_std([r['ternary_vs_float'] for r in results])
    print(f'ternary − scratch : {tvs[0]:+.4f}±{tvs[1]:.4f}  '
          f"→ {'支持(転写起点>スクラッチ)' if tvs[0] > 0 else '不支持'}")
    print(f'ternary − bin     : {tvb[0]:+.4f}±{tvb[1]:.4f}  (0マスクの純効果)')
    print(f'ternary − float   : {tvf[0]:+.4f}±{tvf[1]:.4f}')
    orc_t = summary['arms']['ternary']['oracle_loss_mean']
    print(f'ternary oracle_loss(source − transcribe)={orc_t:+.4f}')
    summary['ternary_vs_scratch_mean'] = tvs[0]
    summary['ternary_vs_bin_mean'] = tvb[0]
    summary['ternary_vs_float_mean'] = tvf[0]
    return summary


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='量子化起点の転写(RNN): 量子化(1bit/1.58bit)RNN→RDDLGN 転写→polish',
    )
    parser.add_argument('--vocab-size', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--d-emb', type=int, default=8)
    parser.add_argument('--d-hidden', type=int, default=16)
    parser.add_argument('--fanin', type=int, default=6, choices=(2, 4, 6))
    parser.add_argument('--head-sizes', type=int, nargs='+', default=[64, 32])

    parser.add_argument('--num-train', type=int, default=2000)
    parser.add_argument('--num-eval', type=int, default=500)
    parser.add_argument('--calib-samples', type=int, default=256)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)

    parser.add_argument('--teacher-epochs', type=int, default=30)
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-prune-epochs', type=int, default=20)
    parser.add_argument('--qat-epochs', type=int, default=15,
                        help='量子化aware fine-tune(STE)のepoch数(0でPTQ)')
    parser.add_argument('--qat-lr', type=float, default=5e-3)

    parser.add_argument('--polish-epochs', type=int, default=30)
    parser.add_argument('--polish-lr', type=float, default=0.02,
                        help='良init破壊を避けるため控えめ(安定化知見)')
    parser.add_argument('--curriculum', action='store_true',
                        help='polish後半でselection=hard_stへ切替(soft→hard_st)')

    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・少epoch、<60s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str, default='outputs/ternary_rnn_to_rddlgn')
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
        args.qat_epochs = 5
        args.polish_epochs = 8
        args.head_sizes = [32, 16]

    print(f'デバイス: {device}')
    print(f'copyタスク: vocab={args.vocab_size} seq_len={args.seq_len} '
          f'seeds={args.seeds}')
    print(f'teacher: d_emb={args.d_emb} d_hidden={args.d_hidden} '
          f'qat_epochs={args.qat_epochs}')
    print(f'student: fanin={args.fanin} head_sizes={args.head_sizes} '
          f'polish_epochs={args.polish_epochs} curriculum={args.curriculum}')

    use_wandb = init_wandb(
        run_name=f'ternary_rnn_v{args.vocab_size}_s{args.seq_len}_h{args.d_hidden}_F{args.fanin}',
        config_dict=vars(args),
        device=device,
        tags=['D18', 'track-a', 'ternary', 'rnn', 'rddlgn', 'transfer',
              f'F{args.fanin}'],
        group='ternary-dwn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        elapsed = time.time() - started
        done = seed
        if done > 0:
            per = elapsed / done
            remaining = per * (args.seeds - done)
            eta = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
            el = str(timedelta(seconds=int(elapsed)))
            rm = str(timedelta(seconds=int(remaining)))
            print(f'\n########## seed {seed + 1}/{args.seeds} '
                  f'[{el}<{rm}, 終了予定 {eta}] ##########')
        else:
            print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_hard_acc': r['teacher_hard_acc'],
                f'seed{seed}/float_polish_best': r['float']['polish_hard_best'],
                f'seed{seed}/bin_polish_best': r['bin']['polish_hard_best'],
                f'seed{seed}/ternary_polish_best': r['ternary']['polish_hard_best'],
                f'seed{seed}/scratch_polish_best': r['scratch']['hard_best'],
                f'seed{seed}/ternary_oracle_loss': r['ternary']['oracle_loss'],
                f'seed{seed}/ternary_vs_scratch': r['ternary_vs_scratch'],
                f'seed{seed}/ternary_vs_bin': r['ternary_vs_bin'],
                f'seed{seed}/ternary_k_agreement': r['ternary']['k_layer_agreement'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_F{args.fanin}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary, 'per_seed': results},
                  f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')
    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
