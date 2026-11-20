"""
RDDLGNClassifier SST-2 AP法（Akiho Propagation, BP-free）訓練スクリプト。

train_kd_sst2.py（BP版KD）と同一のデータ/tokenize/eval基盤を再利用し、
訓練ループだけを RDDLGNClassifierAPTrainer（experiments/ap_trainer_rddlgn_classifier.py）
に置換する。studentは RDDLGNClassifier 固定（AP法は LutLayer.parameterization
== 'independent' かつ wiring == 'fixed' を要求するため、TextCNN-DLGN
（LutConv2dの並列ブランチ+concat構造。ap_trainer_cnn.pyのCNNAPTrainerは単一
sequential構造のみ対応で非互換）は対象外）。

KD設定はtrain_ap_rddlgn.py/RDDLGNAPTrainerと同じ
`target_dist = kd_alpha*one_hot + (1-kd_alpha)*teacher_probs` 混合方式
（BP版のα·CE+(1-α)·KL·T²とは異なる、AP法特有の規約）。
teacherはtrain_kd_sst2.pyと同じ`hf-seqcls:`スキーム経由で公開DistilBERTを解決する。

使用例:
    # scratch（KD無し、tinyスモーク）
    uv run python experiments/train_ap_sst2.py \\
        --train-limit 2000 -ni 60 -ef 20 -bs 16 --device cpu

    # AP + KD（DistilBERT teacher、ラベルあり）
    uv run python experiments/train_ap_sst2.py \\
        --distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' \\
        --train-limit 2000 -ni 200 -ef 50 -bs 32 --device cpu

    # AP + KD + pseudo-labels（完全ラベルフリー）
    uv run python experiments/train_ap_sst2.py \\
        --distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' \\
        --distill-pseudo-labels --kd-alpha 0.25 --kd-temperature 2.0 \\
        -ni 20000 -ef 1000

    # 本番相当（GPU、Modal経由）
    uv run modal run scripts/modal_app.py --script train_ap_sst2.py \\
        --args "--distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' -ni 20000 -ef 1000"
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ap_metrics import ConvergenceTracker, WallClock
from ap_trainer_rddlgn_classifier import RDDLGNClassifierAPTrainer
from checkpoint_utils import setup_checkpoint_dir
from env_utils import load_dotenv
from train_kd_sst2 import parse_int_list
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import RDDLGNClassifier, RDDLGNConfig

load_dotenv()

N_CLASSES = 2  # SST-2: 0=negative, 1=positive


# ============================================================
# データ
# ============================================================

def load_dataset_sst2(args):
    """SST-2データセットをロードする（train_kd_sst2.pyと同一）。"""
    from data.sst2 import download_and_tokenize
    train_set, val_set, tokenizer = download_and_tokenize(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        train_limit=args.train_limit,
    )
    return train_set, val_set, tokenizer


# ============================================================
# ユーティリティ
# ============================================================

def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ（train_ap_rddlgn.py等と同一）。"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


def setup_logger(log_dir='logs'):
    """ログファイルを作成する。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'ap_sst2_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


def save_ap_checkpoint(path, model, step, best_metric, metadata):
    """AP法用チェックポイント保存（optimizerなし。train_ap_rddlgn.pyと同一形式）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }, path)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='RDDLGNClassifier SST-2 AP法(BP-free)訓練スクリプト'
    )

    # データ設定
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='語彙サイズ (default: 8000)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='最大系列長 (default: 64)')
    parser.add_argument('--data-dir', type=str, default='./data-sst2',
                        help='データ保存先 (default: ./data-sst2)')
    parser.add_argument('--train-limit', type=int, default=None,
                        help='訓練データを先頭N件に決定的サブサンプル'
                             '(デバッグ/スモーク用。teacherキャッシュキーにも反映)')

    # RDDLGNClassifier構成（train_kd_sst2.py --student rddlgn-encと同一の引数体系）
    parser.add_argument('--rddlgn-d-emb', type=int, default=128,
                        help='埋め込み次元 (default: 128)')
    parser.add_argument('--rddlgn-n-sizes', type=str, default='256,256',
                        help='N層(feedforward)サイズ (default: 256,256)')
    parser.add_argument('--rddlgn-k-sizes', type=str, default='384,256',
                        help='K層(リカレント)サイズ (default: 384,256)')
    parser.add_argument('--rddlgn-head-sizes', type=str, default='256,128',
                        help='分類ヘッドサイズ (default: 256,128)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='Dense層の接続方式 (default: random)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数(AP法では不使用だがConfig互換のため保持)')

    # AP法設定
    parser.add_argument('--lr', type=float, default=0.1, help='AP法の重み更新率 (default: 0.1)')
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2],
                        help='1=全重み更新, 2=最大basis入力のみ更新 (default: 1)')

    # 訓練設定
    parser.add_argument('-bs', '--batch-size', type=int, default=32,
                        help='バッチサイズ (default: 32)')
    parser.add_argument('-ni', '--num-iterations', type=int, default=20000,
                        help='反復回数 (default: 20000)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000,
                        help='評価頻度 (default: 1000)')

    # 知識蒸留(KD): RDDLGNAPTrainer/train_ap_rddlgn.pyと同じkd_alpha混合方式
    parser.add_argument('--distill-teacher', type=str, default='',
                        help="蒸留teacher指定(例 'hf-seqcls:distilbert/"
                             "distilbert-base-uncased-finetuned-sst-2-english')。"
                             "空文字でKD無効 (default: '')")
    parser.add_argument('--kd-alpha', type=float, default=0.25,
                        help='one-hotターゲットとteacher分布の混合率。'
                             'target_dist = kd_alpha*one_hot + (1-kd_alpha)*teacher_probs。'
                             '1.0はteacher無しと数値的に同一 (default: 0.25)')
    parser.add_argument('--kd-temperature', type=float, default=2.0,
                        help='teacher softmaxの温度T。teacher_probs = '
                             'softmax(teacher_logits / T)。'
                             'SST-2(二値)はT=2がT=4を圧倒する実測あり'
                             '(model-conversion-kdスキル§8.3-1) (default: 2.0)')
    parser.add_argument('--distill-pseudo-labels', action='store_true',
                        help='hard項の正解ラベルをteacher argmax(疑似ラベル)にする'
                             '(完全ラベルフリー) (default: off)')
    parser.add_argument('--distill-cache', action='store_true', default=True,
                        help='teacherロジットを data-teacher-cache/ にキャッシュする '
                             '(default: True)')
    parser.add_argument('--no-distill-cache', dest='distill_cache',
                        action='store_false',
                        help='teacherロジットのキャッシュを無効化する(常時再計算)')

    # 環境
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード (default: 42)')
    parser.add_argument('--device', type=str, default=None, help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs', help='ログ出力先ディレクトリ')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化する')

    args = parser.parse_args()

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

    # データ読み込み
    train_set, val_set, tokenizer = load_dataset_sst2(args)
    actual_vocab_size = tokenizer.vocab_size

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )

    print('\nデータセット: SST-2')
    print(f'訓練: {len(train_set):,} サンプル, 検証: {len(val_set):,} サンプル')
    print(f'語彙サイズ: {actual_vocab_size:,}')

    # ============================================================
    # 知識蒸留(KD)セットアップ(--distill-teacher指定時のみ)
    # ============================================================
    distill_on = bool(args.distill_teacher)
    teacher_train_logits = None
    teacher_val_logits = None
    teacher_fingerprint = None

    if distill_on:
        from data.sst2 import SST2RawTextDataset

        from griffith_logic.convert.cache import get_or_compute_logits, teacher_cache_key
        from griffith_logic.convert.text_teacher import resolve_text_teacher

        teacher = resolve_text_teacher(args.distill_teacher, device=args.device)
        teacher_fingerprint = teacher.fingerprint

        # ラベル順序チェック: teacher.id2labelがSST-2金ラベル(0=negative,1=positive)
        # と一致することをassertする(train_kd_sst2.pyと同一の安全策)。
        if hasattr(teacher, 'id2label'):
            id2label = {int(k): str(v).lower() for k, v in teacher.id2label.items()}
            expected = {0: 'negative', 1: 'positive'}
            for idx, name in expected.items():
                label = id2label.get(idx, '')
                if name not in label and label not in name:
                    raise AssertionError(
                        f'teacher.id2labelがSST-2の金ラベル順序と不一致: '
                        f'{id2label} (期待: {expected})。'
                        'このteacherをそのまま使うとラベル順序が反転した蒸留になる。'
                    )

        raw_train = SST2RawTextDataset('train', data_dir=args.data_dir, train_limit=args.train_limit)
        raw_val = SST2RawTextDataset('validation', data_dir=args.data_dir)
        assert len(raw_train) == len(train_set), (
            f'raw_train({len(raw_train)}) と train_set({len(train_set)}) の件数が不一致'
            '(インデックス整列が崩れている)'
        )

        key_params = {'seq_len_irrelevant': True, 'train_limit': args.train_limit}
        key = teacher_cache_key(
            teacher.fingerprint, 'sst2', key_params, split='train', n=len(raw_train),
        )
        val_key = teacher_cache_key(
            teacher.fingerprint, 'sst2', key_params, split='validation', n=len(raw_val),
        )
        print(f'\n[KD] teacher={args.distill_teacher} '
              f'kd_alpha={args.kd_alpha} T={args.kd_temperature} '
              f'pseudo_labels={args.distill_pseudo_labels}')
        print(f'[KD] teacherロジット計算/ロード中(cache={args.distill_cache}, key={key})...')
        if args.distill_cache:
            teacher_train_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device,
            )
            teacher_val_logits = get_or_compute_logits(
                teacher, raw_val, key=val_key, device=args.device,
            )
        else:
            import tempfile
            tmp_root = tempfile.mkdtemp(prefix='md-ap-distill-nocache-')
            teacher_train_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device, root=tmp_root,
            )
            teacher_val_logits = get_or_compute_logits(
                teacher, raw_val, key=val_key, device=args.device, root=tmp_root,
            )
        print(f'[KD] teacherロジット shape={tuple(teacher_train_logits.shape)}')

        if args.distill_pseudo_labels:
            print('[KD] pseudo_labels有効: 訓練hardラベルをteacher argmaxに置換(ラベルフリー)')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0,
    )
    # shuffle=Falseにする理由: teacher_train_logitsをtrain_setと同一インデックス順で
    # 引くため(PackedDistillDatasetを介さず素朴にoffsetで整列させる簡易実装のため)。

    # モデル構築(AP法はindependentパラメータ化必須)
    config = RDDLGNConfig(
        vocab_size=actual_vocab_size,
        d_emb=args.rddlgn_d_emb,
        seq_len=args.seq_len,
        n_layer_sizes=parse_int_list(args.rddlgn_n_sizes),
        k_layer_sizes=parse_int_list(args.rddlgn_k_sizes),
        n_inputs=args.n_inputs,
        parameterization='independent',
        connections=args.connections,
        grad_factor=args.grad_factor,
    )
    head_sizes = parse_int_list(args.rddlgn_head_sizes)
    model = RDDLGNClassifier(
        config, n_classes=N_CLASSES, head_sizes=head_sizes, device=args.device,
    ).to(args.device)

    counts = model.count_parameters() if hasattr(model, 'count_parameters') else {
        'total': sum(p.numel() for p in model.parameters())
    }
    print('\nモデル: rddlgn-enc (RDDLGNClassifier)')
    print('パラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')

    trainer = RDDLGNClassifierAPTrainer(model, lr=args.lr, update_mode=args.update_mode)
    print(f'AP法: update_mode={args.update_mode}, lr={args.lr}, '
          f'kd_alpha={args.kd_alpha}, kd_temperature={args.kd_temperature}')

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(f'START student=rddlgn-enc vocab={actual_vocab_size} seq_len={args.seq_len} '
        f'distill={args.distill_teacher or "off"} lr={args.lr} update_mode={args.update_mode} '
        f'bs={args.batch_size} ni={args.num_iterations} seed={args.seed} '
        f'device={args.device} params={counts["total"]:,}')

    # チェックポイント
    ckpt_prefix = 'ap_sst2_rddlgn-enc'
    run_dir, latest_path, best_path = setup_checkpoint_dir(ckpt_prefix, base_dir=args.checkpoint_dir)
    ckpt_metadata = {'args': vars(args), 'config': vars(config), 'trainer': 'ap_sst2'}
    if teacher_fingerprint is not None:
        ckpt_metadata['teacher_fingerprint'] = teacher_fingerprint
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = f'ap_sst2_rddlgn-enc_v{actual_vocab_size}_s{args.seq_len}_ni{args.num_iterations}'
    wandb_tags = ['ap', 'sst2', 'rddlgn-enc']
    if distill_on:
        wandb_tags.append('kd')
        if args.distill_pseudo_labels:
            wandb_tags.append('pseudo-labels')
        run_name = f'ap_kd_{run_name[3:]}'
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=wandb_tags,
            group='ap-sst2',
        )

    print(f'\n訓練開始: {args.num_iterations:,}イテレーション (評価={args.eval_freq:,}イテレーションごと)')
    print('=' * 70)

    train_start = time.time()
    best_hard_acc = 0.0

    # BP-free学習の収束・停滞・時間計測(hard acc基準)
    acc_hard_tracker = ConvergenceTracker(mode='max')
    wall_clock = WallClock()
    ACC_HARD_THRESHOLD = 0.75

    batches_per_epoch = len(train_set) // args.batch_size

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='train', total=args.num_iterations,
    )

    for step, (tokens, labels) in pbar:
        tokens = tokens.to(args.device)
        labels_orig = labels.to(args.device)

        teacher_probs = None
        labels_for_step = labels_orig
        if distill_on:
            # train_loaderはshuffle=False+drop_last=Trueのため、エポック内の
            # バッチ順序はteacher_train_logits計算時のraw_train順序と一致する
            # (load_nはエポック境界で単純にloaderを再開するため、
            # step % batches_per_epoch でエポック内バッチ番号を復元できる)。
            batch_idx = step % batches_per_epoch
            start = batch_idx * args.batch_size
            end = start + tokens.shape[0]
            batch_teacher_logits = teacher_train_logits[start:end].to(args.device)
            teacher_probs = F.softmax(batch_teacher_logits / args.kd_temperature, dim=-1)
            if args.distill_pseudo_labels:
                labels_for_step = batch_teacher_logits.argmax(dim=-1)

        metrics = trainer.train_step(
            tokens, labels_for_step, teacher_probs=teacher_probs, kd_alpha=args.kd_alpha,
        )

        elapsed_t = time.time() - train_start
        sec_per_iter = elapsed_t / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{metrics["loss"]:.4f}', acc=f'{metrics["accuracy"]:.4f}', eta=eta_str)

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            acc_hard = trainer.evaluate(val_loader, args.device, use_soft=False)
            acc_soft = trainer.evaluate(val_loader, args.device, use_soft=True)
            gap = acc_soft - acc_hard
            agreement = None
            if teacher_val_logits is not None:
                agreement = trainer.evaluate_teacher_agreement(
                    val_loader, teacher_val_logits, args.device, use_soft=False,
                )

            is_best = acc_hard > best_hard_acc
            if is_best:
                best_hard_acc = acc_hard

            saved_step = step + 1
            acc_hard_tracker.record(saved_step, acc_hard)
            steps_to_thresh = acc_hard_tracker.steps_to_threshold(ACC_HARD_THRESHOLD, mode='max')
            plateau_step = acc_hard_tracker.plateau_step()
            wall_clock_sec = wall_clock.elapsed_sec()

            save_ap_checkpoint(latest_path, model, saved_step, best_hard_acc, ckpt_metadata)
            if is_best:
                save_ap_checkpoint(best_path, model, saved_step, best_hard_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            agree_str = f'  agree={agreement:.4f}' if agreement is not None else ''

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={metrics["loss"]:.4f} train_acc={metrics["accuracy"]:.4f}  '
                f'acc(hard)={acc_hard:.4f}  acc(soft)={acc_soft:.4f}  gap={gap:.4f}'
                f'{agree_str}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )

            log(f'ITER {step+1}/{args.num_iterations} loss={metrics["loss"]:.4f} '
                f'train_acc={metrics["accuracy"]:.4f} acc_hard={acc_hard:.4f} '
                f'acc_soft={acc_soft:.4f} gap={gap:.4f} agreement={agreement} '
                f'best={str(is_best).lower()} elapsed={elapsed_str} eta={remain_str}')

            if use_wandb:
                wandb_metrics = {
                    'iteration': step + 1,
                    'loss': metrics['loss'],
                    'train_acc': metrics['accuracy'],
                    'acc_hard': acc_hard,
                    'acc_soft': acc_soft,
                    'soft_hard_gap': gap,
                    'best_acc_hard': best_hard_acc,
                    'metrics/steps_to_acc0.75': steps_to_thresh if steps_to_thresh is not None else -1,
                    'metrics/plateau_step': plateau_step if plateau_step is not None else -1,
                    'metrics/wall_clock_sec': wall_clock_sec,
                }
                if agreement is not None:
                    wandb_metrics['teacher_student_agreement'] = agreement
                log_metrics(wandb_metrics, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終検証精度(hard): {best_hard_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_acc_hard={best_hard_acc:.4f} total_time={total_time}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_acc_hard': best_hard_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
