"""AP法(Akiho Propagation)によるdata-free文字レベルLM KD訓練。

train_kd_charlm_rddlgn.py（BP版）と同一のデータ/teacher/eval基盤を再利用し、
訓練ループだけを RDDLGNAPTrainer（experiments/ap_trainer_rddlgn.py）に置換する。
studentは RDDLGN 固定（AP法は LutLayer.parameterization == 'independent' を
要求するため、softmax/warp系のattention/nextchar/logicgptパスは対象外）。

データ整列・vocab設計（train_kd_charlm_rddlgn.pyのdocstringと同一、要約）:
    teacher文字vocab(65) + PAD(65) + BOS(66) → student vocab_size = 67。
    src        = chars[0:32]                (エンコーダ文脈)
    tgt_input  = [BOS] + chars[32:63]        (デコーダ入力, teacher forcing)
    tgt_target = chars[32:64]                (デコーダ正解)
    teacherロジット[:, 31:63, :] がstudentの32デコードステップと整列する
    (compute_teacher_logits_for_batch、train_kd_charlm_rddlgn.py流用)。

RDDLGNAPTrainer.train_step(src, tgt_full, teacher_probs, kd_alpha) の契約:
    tgt_full[:, :-1] == tgt_input, tgt_full[:, 1:] == tgt_target を満たす
    長さ33の系列が必要。tgt_input[:, 0] == BOS かつ tgt_input[:, 1:] ==
    tgt_target[:, :-1] なので、tgt_full = cat([tgt_input[:, :1], tgt_target])
    = [BOS, chars[32], ..., chars[63]] で両条件を満たす（コピータスク版
    train_ap_rddlgn.pyのprepare_batchが作る[BOS, tgt, EOS]と同型、EOS無し版）。
    teacher_probs[:, t] は tgt_full上のデコードステップt（t=0..31、
    tgt_target[t]=chars[32+t]の予測）に対応する必要があるため、
    pad_teacher_logits_to_student_vocab後のteacher_logits[:, 31:63, :]を
    そのままsoftmax(/T)して渡せばよい（BP版のteacher_logits引数と同一の
    テンソルを使い回せる）。

使用例:
    # スモーク(rddlgn tiny、CPU、offline)
    HF_HUB_OFFLINE=1 WANDB_MODE=offline uv run python experiments/train_ap_charlm_rddlgn.py \\
        --student rddlgn --config tiny --device cpu \\
        --corpus data-charlm/corpus_*_500x64_T1.0_s42.pt \\
        --eval-corpus data-charlm/corpus_*_500x64_T1.0_s42_eval.pt \\
        -ni 40 -ef 20 -bs 16

    # フルスケール(GPU)
    uv run modal run scripts/modal_app.py --script train_ap_charlm_rddlgn.py \\
        --args "--config paper --scale 0.05 --corpus data-charlm/corpus_....pt --eval-corpus data-charlm/....pt -ni 50000 -ef 2000"
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_metrics import ConvergenceTracker, WallClock
from ap_trainer_rddlgn import RDDLGNAPTrainer
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
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
from train_kd_charlm_rddlgn import (
    SRC_LEN,
    WINDOW,
    build_config_rddlgn,
    collate_with_raw,
    compute_teacher_logits_for_batch,
    evaluate_bpc,
    evaluate_teacher_agreement,
    evaluate_teacher_bpc,
    generate_sample,
    load_n,
    pad_teacher_logits_to_student_vocab,
    resolve_glob,
    setup_logger,
)
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import RDDLGN
from griffith_logic.convert.text_teacher import resolve_text_teacher

load_dotenv()


# ============================================================
# tgt_full構築（RDDLGNAPTrainer.train_stepの契約に合わせる）
# ============================================================

def build_tgt_full(tgt_input: torch.Tensor, tgt_target: torch.Tensor) -> torch.Tensor:
    """(tgt_input, tgt_target)からRDDLGNAPTrainer用のtgt_fullを構築する。

    tgt_full = [BOS] + tgt_target = cat([tgt_input[:, :1], tgt_target], dim=1)。
    tgt_input[:, 0] == BOS_ID かつ tgt_input[:, 1:] == tgt_target[:, :-1]
    （split_src_tgtの定義）であることから、この構成は
        tgt_full[:, :-1] == tgt_input
        tgt_full[:, 1:]  == tgt_target
    を厳密に満たす（モジュールdocstring参照、tests/test_ap_charlm.pyで検証）。

    Args:
        tgt_input: (B, 32) long、先頭がBOS_ID。
        tgt_target: (B, 32) long。

    Returns:
        (B, 33) long。
    """
    return torch.cat([tgt_input[:, :1], tgt_target], dim=1)


def compute_teacher_probs_for_batch(
    teacher, raw_tokens: torch.Tensor, device: str, temperature: float,
) -> torch.Tensor:
    """AP用teacher確率分布(student vocab幅、位置整列済み)を計算する。

    compute_teacher_logits_for_batch(BP版と共有)でteacherロジット[B,32,65]を
    取得し、pad_teacher_logits_to_student_vocabで[B,32,67]に拡張後、
    温度Tでsoftmaxを取る。RDDLGNAPTrainer.train_stepはteacher_probs[:, t]を
    デコードステップt(t=0..31)で参照するため、この[B,32,67]がそのまま
    tgt_full上のステップ0..31（tgt_len=32）に対応する。

    Returns:
        (B, 32, 67) — 各行が確率分布（PAD/BOS列はほぼ0）。
    """
    teacher_logits = compute_teacher_logits_for_batch(teacher, raw_tokens, device)
    teacher_logits = pad_teacher_logits_to_student_vocab(teacher_logits).to(device)
    return F.softmax(teacher_logits / temperature, dim=-1)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AP法によるdata-free文字レベルLM KD訓練(RDDLGN固定)',
    )

    parser.add_argument('--student', type=str, default='rddlgn', choices=['rddlgn'],
                        help='AP法はLutLayer.parameterization==independent必須のため'
                             'rddlgnのみ対応(将来拡張余地を残すため引数自体は残す)。')
    parser.add_argument('--corpus', type=str, default=None,
                        help='gen_teacher_corpus.pyの訓練コーパス.pt(--train-data teacher時は必須)')
    parser.add_argument('--eval-corpus', type=str, required=True, help='held-out評価コーパス.pt')
    parser.add_argument('--teacher', type=str,
                        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')

    parser.add_argument('--train-data', type=str, default='teacher', choices=['teacher', 'real'],
                        help='teacher(既定)=TeacherCorpusDataset。real=tiny_shakespeare実データ'
                             'prefixサブセット(RealCharDataset、--data-fraction指定)。')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='--train-data real時のtrain pool取得割合(0<fraction<=1)。')

    # モデル構成
    parser.add_argument('--config', type=str, default='tiny', choices=['tiny', 'paper'])
    parser.add_argument('--scale', type=float, default=0.05, help='config=paper時のスケール')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])

    # KD設定(RDDLGNAPTrainer.train_step契約に合わせる)
    parser.add_argument('--kd-alpha', type=float, default=0.25,
                        help='one-hotターゲットとteacher分布の混合率。'
                             'target_dist = kd_alpha*one_hot + (1-kd_alpha)*teacher_probs。'
                             '1.0はteacher無し(scratch)と数値的に同一。')
    parser.add_argument('--kd-temperature', type=float, default=4.0,
                        help='teacher softmaxの温度T。teacher_probs = softmax(teacher_logits / T)。')

    # AP法設定
    parser.add_argument('--lr', type=float, default=0.1, help='AP法の重み更新率。')
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2])

    # 訓練
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=20000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--eval-max-chars', type=int, default=100_000,
                        help='tiny_shakespeare検証データの末尾文字数')

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    if args.train_data == 'teacher' and args.corpus is None:
        parser.error(
            '--train-data teacher(既定)では --corpus が必須です'
            '(gen_teacher_corpus.pyで生成した訓練コーパス.ptを指定してください)。'
            ' 実データを使う場合は --train-data real --data-fraction <割合> を指定してください。'
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

    dataset_mode = 'seq2seq'
    collate_fn = collate_with_raw

    if args.train_data == 'real':
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

    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=args.device)
    print(f'  teacher fingerprint={teacher.fingerprint}')

    # モデル構築(AP法はindependentパラメータ化必須)
    config = build_config_rddlgn(args)
    config.parameterization = 'independent'
    model = RDDLGN(config, device=args.device).to(args.device)
    config_dict = asdict(config)

    counts = model.count_parameters() if hasattr(model, 'count_parameters') else {
        'total': sum(p.numel() for p in model.parameters())
    }
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')

    # APTrainer(全LutLayerがindependentであることをコンストラクタ内で検証する)
    trainer = RDDLGNAPTrainer(
        model, lr=args.lr, update_mode=args.update_mode, pad_token=PAD_ID,
    )
    print(f'AP法: update_mode={args.update_mode}, lr={args.lr}, '
          f'kd_alpha={args.kd_alpha}, kd_temperature={args.kd_temperature}')

    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    log, log_file = setup_logger(f'ap_{args.student}', args.log_dir)
    log(f'START student={args.student} config={args.config} corpus={corpus_path} '
        f'train_data={args.train_data} data_fraction={args.data_fraction} '
        f'kd_alpha={args.kd_alpha} kd_temperature={args.kd_temperature} '
        f'lr={args.lr} update_mode={args.update_mode} '
        f'bs={args.batch_size} ni={args.num_iterations} device={args.device} '
        f'params={counts["total"]:,}')

    ckpt_prefix = f'ap_kd_charlm_{args.student}_{args.config}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(ckpt_prefix, base_dir=args.checkpoint_dir)
    best_hard_path = os.path.join(os.path.dirname(best_path), 'best_hard.pt')
    ckpt_metadata = {
        'args': vars(args),
        'config': config_dict,
        'teacher_fingerprint': teacher.fingerprint,
        'trainer': 'ap_charlm_rddlgn',
    }
    print(f'チェックポイント保存先: {run_dir}/')

    print('teacher自身のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    if args.train_data == 'real':
        data_tag = f'real{args.data_fraction}'
    else:
        data_tag = 'teacher'
    run_name = (
        f'ap_kd_charlm_{args.student}_{args.config}_{data_tag}_a{args.kd_alpha}'
        f'_T{args.kd_temperature}_lr{args.lr}_ni{args.num_iterations}'
    )
    wandb_config = {
        **vars(args), **config_dict, 'teacher_fingerprint': teacher.fingerprint,
        'teacher_bpc': teacher_bpc,
        'data_fraction': args.data_fraction,
        'train_data': args.train_data,
        'n_train_chars': n_train_chars,
        'n_train_windows': n_train_windows,
    }
    use_wandb = False
    if not args.no_wandb:
        tags = ['ap', 'charlm', 'kd', 'datafree', args.student, args.train_data]
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

    train_start = time.time()
    best_bpc = float('inf')
    best_bpc_hard = float('inf')

    # AP法(BP-free)学習の収束・停滞・時間計測(hard BPC基準)。
    bpc_hard_tracker = ConvergenceTracker(mode='min')
    wall_clock = WallClock()
    BPC_HARD_THRESHOLD = 3.0

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='train', total=args.num_iterations,
    )

    for step, batch in pbar:
        src, tgt_input, tgt_target = batch
        src = src.to(args.device)
        tgt_input = tgt_input.to(args.device)
        tgt_target = tgt_target.to(args.device)

        # raw_tokens(64) = src(32) + tgt_target(32)（split_src_tgtの定義通り）
        raw_tokens = torch.cat([src.cpu(), tgt_target.cpu()], dim=1)

        teacher_probs = compute_teacher_probs_for_batch(
            teacher, raw_tokens, args.device, args.kd_temperature,
        )
        tgt_full = build_tgt_full(tgt_input, tgt_target)

        metrics = trainer.train_step(
            src, tgt_full, teacher_probs=teacher_probs, kd_alpha=args.kd_alpha,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{metrics["loss"]:.4f}', acc=f'{metrics["accuracy"]:.4f}', eta=eta_str)

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

            saved_step = step + 1
            bpc_hard_tracker.record(saved_step, bpc_hard)
            steps_to_bpc3 = bpc_hard_tracker.steps_to_threshold(BPC_HARD_THRESHOLD, mode='min')
            plateau_step = bpc_hard_tracker.plateau_step()
            wall_clock_sec = wall_clock.elapsed_sec()

            # AP法はoptimizer/schedulerを持たないため、save_checkpointにはNoneを渡す
            # (checkpoint_utils.save_checkpointはoptimizer.state_dict()の呼び出しを
            # optimizer is not Noneガード無しで行うため、代わりにoptimizer=None固有の
            # 軽量dictを自前で組み立てる)。
            _save_ap_checkpoint(latest_path, model, saved_step, best_bpc, ckpt_metadata)
            if is_best:
                _save_ap_checkpoint(best_path, model, saved_step, best_bpc, ckpt_metadata)
            if is_best_hard:
                _save_ap_checkpoint(best_hard_path, model, saved_step, best_bpc_hard, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1)
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')

            sample_text = generate_sample(model, args.student, teacher, args.device, length=300)

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={metrics["loss"]:.4f} train_acc={metrics["accuracy"]:.4f}  '
                f'agree(soft)={agree_soft:.4f} agree(hard)={agree_hard:.4f}  '
                f'BPC(soft)={bpc_soft:.4f}  BPC(hard)={bpc_hard:.4f}  teacher_BPC={teacher_bpc:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'{"  ★ BEST(hard)" if is_best_hard else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )
            print(f'  生成サンプル(300字): {sample_text[:300]!r}')

            log(f'ITER {step+1}/{args.num_iterations} loss={metrics["loss"]:.4f} '
                f'train_acc={metrics["accuracy"]:.4f} agree_soft={agree_soft:.4f} agree_hard={agree_hard:.4f} '
                f'bpc_soft={bpc_soft:.4f} bpc_hard={bpc_hard:.4f} teacher_bpc={teacher_bpc:.4f} '
                f'best={str(is_best).lower()} best_hard={str(is_best_hard).lower()}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss/total': metrics['loss'],
                    'train_acc': metrics['accuracy'],
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


def _save_ap_checkpoint(path, model, step, best_metric, metadata):
    """AP法用チェックポイント保存(optimizer/schedulerを持たないためNone固定)。

    checkpoint_utils.save_checkpointはoptimizer引数を必須（optimizer.state_dict()
    を無条件に呼ぶ）としているため、AP法（optimizerを使わない）専用にここで薄い
    保存関数を用意する。train_ap_rddlgn.pyのsave_ap_checkpointと同一のdata構造
    （'optimizer_state_dict': None, 'scheduler_state_dict': None）を使うことで、
    load_checkpoint(checkpoint_utils.py)からの読み込み互換性を保つ。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }, path)


if __name__ == '__main__':
    main()
