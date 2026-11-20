"""DFA(Direct Feedback Alignment)でLogicGPT/DLGNAttentionTranslatorを学習。

文献的背景(Launay et al. 2020, arXiv:2006.12878):
    DFAでGPT様Transformerを学習できるが、完全BP-freeなattentionは未達
    (Q/K/V射影の更新にはattention機構を通じたBPが依然必要)。DFA単独は
    BPに大差で劣る。block単位(macro)が層単位(micro)より良好。
    → 本スクリプトの現実的レシピ: LUT層(LogicMLP/head)をDFA、
      attention(q/k/v射影+softmax混合)はBPのハイブリッド。
      完全BP-free(--dfa-targets mlp,head,attn)も探索枠として提供する
      (成否どちらも記録する)。

実装方式:
    griffith_logic.training.rules.dfa.DFARule を直接使う(HybridTrainer
    は model(x)->単一logitsテンソルの契約を前提とするが、LogicGPT/
    DLGNAttentionTranslatorは(logits, emb_reg_loss)の2-tupleを返すため、
    train_kd_charlm_rddlgn.py既存のPADマスク付きKD損失や二値化正則化と
    組み合わせるには専用ループが必要)。
    DFA注釈は griffith_logic.models.logic_gpt.LogicGPT /
    griffith_logic.models.attention.attention_translator.DLGNAttentionTranslator
    の dfa_targets/dfa_mode 引数で行う。

    1step: model.train() → forward → loss計算 → DFA誤差e取得
        → optimizer.zero_grad() → loss.backward()(BP層へ)
        → dfa_rule.inject(e)(DFA層へ) → optimizer.step()

    --dfa-mode off は同じスクリプト内でDFA注釈なし(=完全BP)の対照実験
    として走らせるためのモード(dfa_targets=Noneで構築するのみ)。

データ/評価/teacher整列は train_kd_charlm_rddlgn.py の実装を再利用する
(import専用、無改造)。char-LMのdata-free主張・整列根拠は同モジュールの
docstring参照。

使用例:
    # LogicGPT, micro DFA(mlp+head), BP対照と比較する場合は--dfa-mode offも走らせる
    WANDB_MODE=offline uv run python experiments/train_dfa_charlm.py \\
        --student logicgpt --dfa-mode micro --dfa-targets mlp,head \\
        --corpus data-charlm/corpus_*.pt --eval-corpus data-charlm/*_eval.pt \\
        -ni 2000 -ef 200 -bs 32

    # macro(block単位)
    WANDB_MODE=offline uv run python experiments/train_dfa_charlm.py \\
        --student logicgpt --dfa-mode macro --dfa-targets mlp,head \\
        --corpus data-charlm/corpus_*.pt --eval-corpus data-charlm/*_eval.pt \\
        -ni 2000 -ef 200 -bs 32

    # BP対照(--dfa-mode off)
    WANDB_MODE=offline uv run python experiments/train_dfa_charlm.py \\
        --student logicgpt --dfa-mode off \\
        --corpus data-charlm/corpus_*.pt --eval-corpus data-charlm/*_eval.pt \\
        -ni 2000 -ef 200 -bs 32

    # attention translator(encoder + l_layersのみDFA)
    WANDB_MODE=offline uv run python experiments/train_dfa_charlm.py \\
        --student attention --dfa-mode micro --dfa-targets encoder,l_layers \\
        --corpus data-charlm/corpus_*.pt --eval-corpus data-charlm/*_eval.pt \\
        -ni 2000 -ef 200 -bs 32
"""

import argparse
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_metrics import ConvergenceTracker, WallClock
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from data.char_corpus import (
    PAD_ID,
    STUDENT_VOCAB_SIZE,
    TeacherCorpusDataset,
    load_tiny_shakespeare_val,
)
from env_utils import load_dotenv

# train_kd_charlm_rddlgn.py はimport専用(編集禁止)。データ/評価/teacher整列
# の既存実装をそのまま再利用する(新規に再実装しない)。
from train_kd_charlm_rddlgn import (
    SRC_LEN,
    WINDOW,
    build_config_attention,
    build_config_logicgpt,
    collate_raw_window,
    collate_with_raw,
    compute_teacher_logits_for_batch,
    evaluate_bpc,
    evaluate_teacher_agreement,
    evaluate_teacher_bpc,
    generate_sample,
    get_emb_reg_weight,
    load_n,
    make_pure_ce_seq_loss,
    pad_teacher_logits_to_student_vocab,
    resolve_glob,
)
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_checkpoint_artifact, log_metrics

from griffith_logic import DLGNAttentionTranslator, LogicGPT
from griffith_logic.convert.distill import make_seq_kd_loss
from griffith_logic.convert.text_teacher import resolve_text_teacher
from griffith_logic.training.rules.dfa import DFARule

load_dotenv()

DFA_MODE_CHOICES = ('micro', 'macro', 'off')


# ============================================================
# DFA train_step(モデル種別非依存)
# ============================================================

def make_output_error_fn(loss_fn_for_grad):
    """DFA用の出力誤差 e=∂L/∂logits を返すクロージャ(trainer.pyの
    HybridTrainer._output_errorと同じ考え方。系列(B,S,V)対応)。

    Args:
        loss_fn_for_grad: (logits) -> main_lossスカラー。headまでDFA化されて
            logitsが大域グラフから完全に切断された場合のフォールバックで、
            検出時にlogitsの独立コピーへrequires_grad_(True)して
            再計算するために使う(HybridTrainerのsoftmax-onehot解析式は
            素のcross_entropyしか厳密に成立しないため、KD損失など任意の
            main_loss関数でも厳密に動く汎用フォールバックとしてこちらを採用)。
    """

    def output_error(logits: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        if logits.requires_grad:
            (e,) = torch.autograd.grad(loss, logits, retain_graph=True)
            return e.detach()
        # 全層DFA(headまでDFA化)でlogitsが大域グラフから完全に切断されている
        # 場合: 検出済みlogits値の独立コピーにrequires_grad_(True)し、
        # main_loss関数を再適用してそこだけのローカルグラフでautograd.gradを
        # 取る(厳密。teacher/target等はクロージャ内で固定済みなので
        # 再計算コストはsoftmax+loss相当のみで軽い)。
        logits_leaf = logits.detach().clone().requires_grad_(True)
        loss_local = loss_fn_for_grad(logits_leaf)
        (e,) = torch.autograd.grad(loss_local, logits_leaf)
        return e.detach()

    return output_error


def dfa_train_step(
    model, dfa_rule, optimizer, forward_fn, loss_fn, make_output_error,
    grad_clip: float = 0.0,
):
    """DFA(+部分BP)の1ステップ訓練。

    Args:
        model: nn.Module
        dfa_rule: DFARule または None(--dfa-mode off時)
        forward_fn: () -> (logits, emb_reg_loss)。呼び出し側でtarget等をクロージャに含める
        loss_fn: (logits, emb_reg_loss) -> (total_loss, main_loss)
        make_output_error: main_loss_only_fn(logits)->スカラー を受け取り、
            output_error(logits, main_loss)->error tensor を返すファクトリ
            (make_output_error_fn参照。dfa_rule=None時は未使用)。
    """
    model.train()
    if dfa_rule is not None:
        dfa_rule.begin_step()

    logits, emb_reg_loss = forward_fn()
    total_loss, main_loss, main_loss_only_fn = loss_fn(logits, emb_reg_loss)

    e = None
    if dfa_rule is not None:
        output_error = make_output_error(main_loss_only_fn)
        e = output_error(logits, main_loss)

    optimizer.zero_grad(set_to_none=True)
    if total_loss.requires_grad:
        total_loss.backward()
    if dfa_rule is not None:
        dfa_rule.inject(e)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return total_loss.item(), main_loss.item()


# ============================================================
# モデル構築
# ============================================================

def parse_dfa_targets(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    return [s.strip() for s in spec.split(',') if s.strip()]


def build_student(args):
    """--studentに応じてLogicGPT/DLGNAttentionTranslatorを構築する。

    Returns:
        (model, config_dict, dataset_mode, collate_fn)
    """
    dfa_targets = None if args.dfa_mode == 'off' else parse_dfa_targets(args.dfa_targets)

    if args.student == 'logicgpt':
        config = build_config_logicgpt(args)
        model = LogicGPT(
            config, dfa_targets=dfa_targets,
            dfa_mode='micro' if args.dfa_mode == 'off' else args.dfa_mode,
            dfa_lr=args.dfa_lr,
        ).to(args.device)
        return model, asdict(config), 'decoder_only', collate_raw_window

    config = build_config_attention(args)
    model = DLGNAttentionTranslator(
        config, device=args.device, dfa_targets=dfa_targets, dfa_lr=args.dfa_lr,
    ).to(args.device)
    return model, asdict(config), 'seq2seq', collate_with_raw


def collect_dfa_entries(model) -> list[tuple[str, torch.nn.Module]]:
    """with_learning(rule='dfa')で注釈済みのモジュールを(name, module)で収集する。

    HybridTrainer._scan相当のミニマル版(親の注釈があれば子はスキップする、
    親の走査順=named_modules()の順=登録順を利用)。AP/EGGROLLはスコープ外
    のためチェックしない(dfa_targets指定時は常にrule='dfa'のみ発生する)。
    """
    entries = []
    annotated_prefixes: list[str] = []
    for name, module in model.named_modules():
        spec = getattr(module, '_learning_spec', None)
        if spec is None:
            continue
        if any(name.startswith(p + '.') for p in annotated_prefixes):
            continue
        annotated_prefixes.append(name)
        if spec.rule == 'dfa':
            entries.append((name, module))
    return entries


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DFA(Direct Feedback Alignment)でLogicGPT/attentionを学習',
    )

    parser.add_argument('--student', type=str, default='logicgpt',
                        choices=['logicgpt', 'attention'])
    parser.add_argument('--corpus', type=str, required=True,
                        help='gen_teacher_corpus.pyの訓練コーパス.pt')
    parser.add_argument('--eval-corpus', type=str, required=True, help='held-out評価コーパス.pt')
    parser.add_argument('--teacher', type=str,
                        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')

    # DFA設定
    parser.add_argument('--dfa-mode', type=str, default='micro', choices=list(DFA_MODE_CHOICES),
                        help="micro(既定)=LogicMLP/head単体を注釈。"
                             "macro=LogicTransformerBlock単位で注釈(logicgptのみ)。"
                             "off=DFA注釈なし(=完全BP対照実験)。")
    parser.add_argument('--dfa-targets', type=str, default='mlp,head',
                        help="カンマ区切りのDFA注釈対象。student=logicgptは"
                             "'mlp'/'head'/'attn'(実験的、完全BP-free探索用)。"
                             "student=attentionは'encoder'/'l_layers'。"
                             "--dfa-mode off時は無視される。")
    parser.add_argument('--dfa-lr', type=float, default=None,
                        help='DFA注釈モジュール向けのLearningSpec.lr(Noneならoptimizer共通lrに従う)')

    # KD設定(主に純タスク=data-free next-char予測でDFAの学習成立を見る)
    parser.add_argument('--kd', type=str, default='off', choices=['on', 'off'],
                        help="off(既定)=純CE(teacherコーパス上での自己回帰予測タスク、"
                             "teacher logitsへのKLは使わない)。on=make_seq_kd_loss"
                             "(--distill-alpha等)を使う(student=attentionのみ対応)。")
    parser.add_argument('--distill-alpha', type=float, default=0.25)
    parser.add_argument('--distill-temperature', type=float, default=4.0)
    parser.add_argument('--distill-standardize', dest='distill_standardize',
                        action='store_true', default=True)
    parser.add_argument('--no-distill-standardize', dest='distill_standardize',
                        action='store_false')

    # モデル構成
    parser.add_argument('--config', type=str, default='tiny', choices=['tiny', 'paper'])
    parser.add_argument('--scale', type=float, default=0.05, help='config=paper時のスケール')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--gpt-layers', type=int, default=None)
    parser.add_argument('--gpt-heads', type=int, default=None)
    parser.add_argument('--gpt-dmodel', type=int, default=None)
    parser.add_argument('--gpt-mlp-ratio', type=int, default=None)

    # 訓練
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=32)
    parser.add_argument('-ni', '--num-iterations', type=int, default=2000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=200)
    parser.add_argument('--eval-max-chars', type=int, default=20_000)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--emb-reg-ramp-start', type=int, default=1000)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=100000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)

    # チェックポイント/環境
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    if args.kd == 'on' and args.student != 'attention':
        parser.error('--kd on は現状 --student attention のみ対応です'
                     '(logicgptはtrain_kd_charlm_rddlgn.pyのdecoder-only KD経路を使用してください)')
    if args.dfa_mode == 'macro' and args.student != 'logicgpt':
        parser.error('--dfa-mode macro は --student logicgpt のみ対応です'
                     '(block単位=LogicTransformerBlockはlogicgpt専用の概念のため)')

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

    corpus_path = resolve_glob(args.corpus)
    eval_corpus_path = resolve_glob(args.eval_corpus)
    print(f'訓練コーパス: {corpus_path}')
    print(f'評価コーパス: {eval_corpus_path}')

    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=args.device)
    print(f'  teacher fingerprint={teacher.fingerprint}')

    model, config_dict, dataset_mode, collate_fn = build_student(args)

    counts = model.count_parameters() if hasattr(model, 'count_parameters') else {
        'total': sum(p.numel() for p in model.parameters())
    }
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')

    dfa_entries = [] if args.dfa_mode == 'off' else collect_dfa_entries(model)
    dfa_rule = DFARule(dfa_entries) if dfa_entries else None
    n_dfa_params = sum(p.numel() for _, m in dfa_entries for p in m.parameters())
    n_total_params = counts['total']
    print(f'DFA注釈モジュール: {[n for n, _ in dfa_entries] if dfa_entries else "(なし=完全BP)"}')
    print(f'  DFAパラメータ数: {n_dfa_params:,} / 全体 {n_total_params:,} '
          f'({100 * n_dfa_params / max(1, n_total_params):.1f}%)')
    if args.dfa_mode != 'off' and not dfa_entries:
        print('警告: --dfa-mode指定に反しDFA注釈モジュールが0件です(--dfa-targetsを確認)')

    train_set = TeacherCorpusDataset(corpus_path, mode=dataset_mode)
    eval_set = TeacherCorpusDataset(eval_corpus_path, mode=dataset_mode)
    print(f'訓練サンプル: {len(train_set):,}  評価サンプル: {len(eval_set):,}')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
        collate_fn=collate_fn,
    )

    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    emb_reg_ramp = (args.emb_reg_ramp_start, args.emb_reg_ramp_end, args.emb_reg_max_weight)

    if args.kd == 'on':
        kd_loss_seq = make_seq_kd_loss(
            alpha=args.distill_alpha, temperature=args.distill_temperature,
            standardize=args.distill_standardize, pad_id=PAD_ID,
        )
    else:
        ce_loss_seq = make_pure_ce_seq_loss(pad_id=PAD_ID)

    student_kind = args.student

    def make_forward_and_loss(batch, step):
        """batchから(forward_fn, loss_fn)を組み立てる(student種別ごとの整列規約に従う)。"""
        emb_w = get_emb_reg_weight(step, *emb_reg_ramp)

        if student_kind == 'logicgpt':
            window = batch
            inp = window[:, :-1].to(args.device)
            target = window[:, 1:].to(args.device)

            def forward_fn():
                return model(inp)

            if args.kd == 'on':
                with torch.no_grad():
                    full_logits = teacher.next_token_logits(window.to(args.device))
                teacher_logits = full_logits[:, :63, :]
                teacher_logits = pad_teacher_logits_to_student_vocab(teacher_logits).to(args.device)

                def main_loss_only_fn(logits):
                    return kd_loss_seq(logits, target, teacher_logits)
            else:
                def main_loss_only_fn(logits):
                    return ce_loss_seq(logits, target)

            def loss_fn(logits, emb_reg_loss):
                main = main_loss_only_fn(logits)
                return main + emb_w * emb_reg_loss, main, main_loss_only_fn

            return forward_fn, loss_fn

        # attention
        src, tgt_input, tgt_target = batch
        src = src.to(args.device)
        tgt_input = tgt_input.to(args.device)
        tgt_target = tgt_target.to(args.device)

        def forward_fn():
            return model(src, tgt_input)

        if args.kd == 'on':
            raw_tokens = torch.cat([src.cpu(), tgt_target.cpu()], dim=1)
            teacher_logits = compute_teacher_logits_for_batch(teacher, raw_tokens, args.device)
            teacher_logits = pad_teacher_logits_to_student_vocab(teacher_logits).to(args.device)

            def main_loss_only_fn(logits):
                return kd_loss_seq(logits, tgt_target, teacher_logits)
        else:
            def main_loss_only_fn(logits):
                return ce_loss_seq(logits, tgt_target)

        def loss_fn(logits, emb_reg_loss):
            main = main_loss_only_fn(logits)
            return main + emb_w * emb_reg_loss, main, main_loss_only_fn

        return forward_fn, loss_fn

    dfa_tag = 'bp' if args.dfa_mode == 'off' else f'dfa-{args.dfa_mode}'
    targets_tag = 'none' if args.dfa_mode == 'off' else parse_dfa_targets(args.dfa_targets)
    run_name = (
        f'dfa_charlm_{args.student}_{args.config}_{dfa_tag}'
        f'_t{"-".join(targets_tag) if isinstance(targets_tag, list) else targets_tag}'
        f'_kd{args.kd}_ni{args.num_iterations}_seed{args.seed}'
    )

    ckpt_prefix = f'dfa_charlm_{args.student}_{args.config}_{dfa_tag}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(ckpt_prefix, base_dir=args.checkpoint_dir)
    ckpt_metadata = {'args': vars(args), 'config': config_dict, 'teacher_fingerprint': teacher.fingerprint}
    print(f'チェックポイント保存先: {run_dir}/')

    print('teacher自身のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    wandb_config = {
        **vars(args), **config_dict, 'teacher_fingerprint': teacher.fingerprint,
        'teacher_bpc': teacher_bpc, 'n_dfa_params': n_dfa_params, 'n_total_params': n_total_params,
    }
    use_wandb = False
    if not args.no_wandb:
        tags = ['dfa', 'charlm', args.student, dfa_tag]
        use_wandb = init_wandb(
            run_name=run_name, config_dict=wandb_config, model=model,
            device=args.device, tags=tags, group='kd-charlm',
        )

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション (dfa_mode={args.dfa_mode})')
    print('=' * 60)

    train_start = time.time()
    best_bpc_hard = float('inf')
    bpc_hard_tracker = ConvergenceTracker(mode='min')
    wall_clock = WallClock()
    BPC_HARD_THRESHOLD = 3.0

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='train', total=args.num_iterations,
    )

    for step, batch in pbar:
        forward_fn, loss_fn = make_forward_and_loss(batch, step)
        total_loss, main_loss = dfa_train_step(
            model, dfa_rule, optimizer, forward_fn, loss_fn, make_output_error_fn,
            grad_clip=args.grad_clip,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{total_loss:.4f}', main=f'{main_loss:.4f}', eta=eta_str)

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            agree_soft = evaluate_teacher_agreement(model, student_kind, eval_loader, teacher, args.device, use_soft=True)
            agree_hard = evaluate_teacher_agreement(model, student_kind, eval_loader, teacher, args.device, use_soft=False)
            bpc_soft = evaluate_bpc(model, student_kind, val_windows, args.device, use_soft=True)
            bpc_hard = evaluate_bpc(model, student_kind, val_windows, args.device, use_soft=False)

            is_best_hard = bpc_hard < best_bpc_hard
            if is_best_hard:
                best_bpc_hard = bpc_hard

            saved_step = step + 1
            bpc_hard_tracker.record(saved_step, bpc_hard)
            steps_to_bpc3 = bpc_hard_tracker.steps_to_threshold(BPC_HARD_THRESHOLD, mode='min')
            plateau_step = bpc_hard_tracker.plateau_step()
            wall_clock_sec = wall_clock.elapsed_sec()

            save_checkpoint(latest_path, model, optimizer, None, saved_step, best_bpc_hard, ckpt_metadata)
            if is_best_hard:
                save_checkpoint(best_path, model, optimizer, None, saved_step, best_bpc_hard, ckpt_metadata)

            sample_text = generate_sample(model, student_kind, teacher, args.device, length=200)

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={total_loss:.4f} main={main_loss:.4f} '
                f'agree(soft)={agree_soft:.4f} agree(hard)={agree_hard:.4f}  '
                f'BPC(soft)={bpc_soft:.4f} BPC(hard)={bpc_hard:.4f} teacher_BPC={teacher_bpc:.4f}'
                f'{"  * BEST(hard)" if is_best_hard else ""}'
                f'  plateau_step={plateau_step}'
            )
            print(f'  生成サンプル(200字): {sample_text[:200]!r}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'agreement/soft': agree_soft,
                    'agreement/hard': agree_hard,
                    'bpc/student_soft': bpc_soft,
                    'bpc/student_hard': bpc_hard,
                    'bpc/best_hard': best_bpc_hard,
                    'bpc/teacher': teacher_bpc,
                    'metrics/steps_to_bpc3.0': steps_to_bpc3 if steps_to_bpc3 is not None else -1,
                    'metrics/plateau_step': plateau_step if plateau_step is not None else -1,
                    'metrics/wall_clock_sec': wall_clock_sec,
                    'sample_text': sample_text,
                }, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終BPC(best hard): {best_bpc_hard:.4f}  (総訓練時間: {total_time})')

    if dfa_rule is not None:
        dfa_rule.remove()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_bpc_hard': best_bpc_hard, 'teacher_bpc': teacher_bpc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
