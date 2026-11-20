"""
CDDLGN/RDDLGN SST-2 知識蒸留(KD)訓練スクリプト（英語感情分析、ROADMAP-v5）

SST-2(Stanford Sentiment Treebank v2、二値感情分類)を題材に、HuggingFace
`AutoModelForSequenceClassification` teacher(既定: DistilBERT SST-2ファインチューン済み)
からのKD(知識蒸留)を行う。student はTextCNN(train_textcnn.pyのTextCNNModel)または
RDDLGNEncoder + 分類ヘッド(RDDLGNClassifier)から選べる。

KD配線はtrain_cnn.pyの--distill-*ブロックを踏襲する(resolve_text_teacher→
teacher_cache_key→get_or_compute_logits→PackedDistillDataset→make_kd_loss)。
teacherのラベル順序(id2label)がSST-2の金ラベル(0=negative, 1=positive)と
一致するかを起動時にassertする。

使用例:
    # TextCNN student、KDなし（デバッグ用の小構成）
    uv run python experiments/train_kd_sst2.py --student textcnn \\
        --d-emb 32 --n-filters 8 --dense-dims 128,64 -ni 200 -ef 100

    # TextCNN student + DistilBERT teacher蒸留（train-limitでスモーク）
    uv run python experiments/train_kd_sst2.py --student textcnn \\
        --distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' \\
        --train-limit 2000 -ni 60 -ef 30 -bs 32

    # RDDLGNEncoder student（tiny構成相当）+ KD
    uv run python experiments/train_kd_sst2.py --student rddlgn-enc \\
        --distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' \\
        --train-limit 2000 -ni 60 -ef 30 -bs 32

    # 本番相当（全件・エポックモード）
    uv run python experiments/train_kd_sst2.py --student textcnn \\
        --distill-teacher 'hf-seqcls:distilbert/distilbert-base-uncased-finetuned-sst-2-english' \\
        -e 20
"""

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from train_textcnn import TextCNNModel
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import RDDLGNClassifier, RDDLGNConfig

# .envから環境変数を読み込み
load_dotenv()

N_CLASSES = 2  # SST-2: 0=negative, 1=positive


# ============================================================
# モデルラッパー
# ============================================================

class _ModelWrapper(nn.Module):
    """TextCNNModel / RDDLGNClassifier の forward 差異を吸収する薄いラッパー。

    どちらも forward(tokens[B,S]) -> (logits[B,C], emb_reg_loss) の契約を
    既に満たしている(TextCNNModel.forward / RDDLGNClassifier.forward参照)ため、
    実質パススルーだが、将来 forward契約が変わった際の緩衝材として保持する。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(tokens)

    def count_parameters(self) -> dict:
        if hasattr(self.model, 'count_parameters'):
            return self.model.count_parameters()
        total = sum(p.numel() for p in self.model.parameters())
        return {'total': total}


def build_model(args, vocab_size: int) -> nn.Module:
    """--student に応じてモデルを構築する。"""
    if args.student == 'textcnn':
        kernel_sizes = parse_int_list(args.kernel_sizes)
        dense_dims = parse_int_list(args.dense_dims)
        if dense_dims[-1] % N_CLASSES != 0:
            dense_dims[-1] = (dense_dims[-1] // N_CLASSES) * N_CLASSES
            print(f'Dense最終層を {dense_dims[-1]} に調整（{N_CLASSES}の倍数）')
        model = TextCNNModel(
            vocab_size=vocab_size,
            d_emb=args.d_emb,
            n_filters=args.n_filters,
            kernel_sizes=kernel_sizes,
            dense_dims=dense_dims,
            n_classes=N_CLASSES,
            tree_depth=args.tree_depth,
            n_inputs=args.n_inputs,
            device=args.device,
            grad_factor=args.grad_factor,
            parameterization=args.parameterization,
            connections=args.connections,
            tau=args.tau,
            selection=args.selection,
        )
    elif args.student == 'rddlgn-enc':
        # RDDLGNConfig: student用に小さく明示構築する（specialトークン込みの
        # vocab_size、d_emb/seq_lenは共有引数、n/k層は--rddlgn-*で指定）。
        config = RDDLGNConfig(
            vocab_size=vocab_size,
            d_emb=args.rddlgn_d_emb,
            seq_len=args.seq_len,
            n_layer_sizes=parse_int_list(args.rddlgn_n_sizes),
            k_layer_sizes=parse_int_list(args.rddlgn_k_sizes),
            n_inputs=args.n_inputs,
            parameterization=args.parameterization,
            connections=args.connections,
            grad_factor=args.grad_factor,
            selection=args.selection,
        )
        head_sizes = parse_int_list(args.rddlgn_head_sizes)
        model = RDDLGNClassifier(
            config, n_classes=N_CLASSES, head_sizes=head_sizes, device=args.device,
        )
    else:
        raise ValueError(f'未知の --student: {args.student}')

    if args.selection != 'soft':
        from griffith_logic.layers import LutLayer as _LutLayer
        for _m in model.modules():
            if isinstance(_m, _LutLayer):
                _m.selection = args.selection

    return _ModelWrapper(model).to(args.device)


# ============================================================
# データセット
# ============================================================

def load_dataset_sst2(args):
    """SST-2データセットをロードする（trainはtrain-limitで決定的サブサンプル可）。"""
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

def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。"""
    return [int(x.strip()) for x in s.split(',')]


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ。"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


def get_emb_reg_weight(step: int, ramp_start: int = 500, ramp_end: int = 20000,
                       max_weight: float = 0.1) -> float:
    """二値化正則化の重みを線形ランプで計算する（train_rddlgn.py/train_textcnn.py踏襲）。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


# ============================================================
# ログ
# ============================================================

def setup_logger(log_dir='logs'):
    """ログファイルを作成する。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'kd_sst2_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, target, loss_fn, optimizer, step, tau_sched=None,
               spectral_reg=0.0):
    """1バッチの訓練ステップ。emb_reg_lossはランプ重みで加算する。"""
    model.train()
    logits, emb_loss = model(x)
    main_loss = loss_fn(logits, target)

    emb_weight = get_emb_reg_weight(step)
    total_loss = main_loss + emb_weight * emb_loss

    # Spectral正則化(DWN §3.4)
    if spectral_reg > 0:
        from griffith_logic.regularization import spectral_penalty
        total_loss = total_loss + spectral_reg * spectral_penalty(model)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if tau_sched is not None:
        tau_sched.step()

    return total_loss.item(), main_loss.item(), emb_loss.item()


@torch.no_grad()
def evaluate_acc(model, loader, device, use_soft=True, desc=None):
    """正解ラベルに対する分類精度を計算する。

    Args:
        use_soft: True=訓練モード（softmax重み）, False=推論モード（離散化）
            リポジトリ規約: model.eval() = hard/Heaviside離散化、
            model.train() = soft(train_textcnn.py.evaluateと同じ規約)。
    """
    was_training = model.training
    model.train(mode=use_soft)

    correct = 0
    total = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader

    for token_ids, labels in it:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        logits, _ = model(token_ids)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        if desc and isinstance(it, tqdm):
            it.set_postfix(acc=f'{correct / total:.4f}')

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_teacher_agreement(model, loader, teacher_val_logits, device,
                               use_soft=False, desc=None):
    """teacher-student のargmax一致率を検証データ全体で計算する（インデックス整列前提）。

    teacher_val_logits: [N_val, C]（validation全体、get_or_compute_logitsのインデックス
    順）。loader は shuffle=False の validation loader である必要がある。
    """
    was_training = model.training
    model.train(mode=use_soft)

    agree = 0
    total = 0
    offset = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader

    for token_ids, _labels in it:
        bsz = token_ids.shape[0]
        token_ids = token_ids.to(device)
        logits, _ = model(token_ids)
        student_preds = logits.argmax(dim=-1).cpu()
        teacher_preds = teacher_val_logits[offset:offset + bsz].argmax(dim=-1)
        agree += (student_preds == teacher_preds).sum().item()
        total += bsz
        offset += bsz

    model.train(mode=was_training)
    return agree / total if total > 0 else 0.0


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CDDLGN/RDDLGN SST-2 KD訓練スクリプト（英語感情分析）'
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

    # student選択
    parser.add_argument('--student', type=str, default='textcnn',
                        choices=['textcnn', 'rddlgn-enc'],
                        help='studentモデル種別 (default: textcnn)')

    # TextCNN構成
    parser.add_argument('--d-emb', type=int, default=256,
                        help='[textcnn] 埋め込み次元 (default: 256)')
    parser.add_argument('--n-filters', type=int, default=64,
                        help='[textcnn] ブランチあたりのフィルタ数 (default: 64)')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='[textcnn] カーネルサイズ (default: 3,4,5)')
    parser.add_argument('--dense-dims', type=str, default='1024,512',
                        help='[textcnn] Dense層の次元 (default: 1024,512)')

    # RDDLGNEncoder構成（明示的に小構成を組む。from_paperは大きすぎるため）
    parser.add_argument('--rddlgn-d-emb', type=int, default=128,
                        help='[rddlgn-enc] 埋め込み次元 (default: 128)')
    parser.add_argument('--rddlgn-n-sizes', type=str, default='256,256',
                        help='[rddlgn-enc] N層(feedforward)サイズ (default: 256,256)')
    parser.add_argument('--rddlgn-k-sizes', type=str, default='384,256',
                        help='[rddlgn-enc] K層(リカレント)サイズ (default: 384,256)')
    parser.add_argument('--rddlgn-head-sizes', type=str, default='256,128',
                        help='[rddlgn-enc] 分類ヘッドサイズ (default: 256,128)')

    # 共有モデル設定
    parser.add_argument('--tree-depth', type=int, default=3,
                        help='[textcnn] ツリーの深さ (default: 3)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--tau', type=float, default=None,
                        help='[textcnn] GroupSum温度（Noneで自動計算）')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式(LutLayer部のみ。convツリーはsoft)')
    parser.add_argument('--tau-anneal', action='store_true')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'softmax16g',
                                 'warp'],
                        help='パラメータ化方式 (default: softmax。warpは'
                             'LutLayer系のみ、conv木はsoftmaxへフォールバック)')
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='Dense層の接続方式 (default: random)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')

    # 訓練設定
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay (default: 0.001)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='バッチサイズ (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数（--num-iterationsと排他）')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数（--epochsと排他）')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度')

    # 知識蒸留(KD): student + HF sequence classification teacher(train_cnn.py踏襲)
    parser.add_argument('--distill-teacher', type=str, default='',
                        help="蒸留teacher指定(例 'hf-seqcls:distilbert/"
                             "distilbert-base-uncased-finetuned-sst-2-english')。"
                             "空文字でKD無効 (default: '')")
    parser.add_argument('--distill-alpha', type=float, default=0.25,
                        help='hard(CE)項の重み。α=1で純CE、α=0で純KD (default: 0.25)')
    parser.add_argument('--distill-temperature', type=float, default=4.0,
                        help='KD温度 T(二重温度 τ_gs×T の T 成分) (default: 4.0)')
    parser.add_argument('--distill-standardize', action='store_true', default=True,
                        help='KD項の両ロジットをZ-score標準化する(既定ON) (default: True)')
    parser.add_argument('--no-distill-standardize', dest='distill_standardize',
                        action='store_false',
                        help='ロジット標準化を無効化する(非推奨)')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_set, val_set, tokenizer = load_dataset_sst2(args)
    actual_vocab_size = tokenizer.vocab_size

    use_pin_memory = args.device == 'cuda'
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=False, num_workers=0,
    )

    print('\nデータセット: SST-2')
    print(f'訓練: {len(train_set):,} サンプル, 検証: {len(val_set):,} サンプル')
    print(f'語彙サイズ: {actual_vocab_size:,}')

    # ============================================================
    # 知識蒸留(KD)セットアップ(--distill-teacher指定時のみ)
    # ============================================================
    distill_on = bool(args.distill_teacher)
    teacher_val_logits = None
    loss_fn = nn.CrossEntropyLoss()
    train_loader = None

    if distill_on:
        from data.sst2 import SST2RawTextDataset

        from griffith_logic.convert.cache import (
            PackedDistillDataset,
            get_or_compute_logits,
            teacher_cache_key,
        )
        from griffith_logic.convert.distill import make_kd_loss
        from griffith_logic.convert.text_teacher import resolve_text_teacher

        teacher = resolve_text_teacher(args.distill_teacher, device=args.device)

        # ラベル順序チェック: teacher.id2labelがSST-2金ラベル(0=negative,1=positive)
        # と一致することをassertする(順序が違う蒸留は静かに誤学習するため必須)。
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
        n_classes = N_CLASSES

        # teacher推論用の生テキストtrain dataset(student と同一順序・同一件数)
        raw_train = SST2RawTextDataset('train', data_dir=args.data_dir, train_limit=args.train_limit)
        raw_val = SST2RawTextDataset('validation', data_dir=args.data_dir)
        assert len(raw_train) == len(train_set), (
            f'raw_train({len(raw_train)}) と train_set({len(train_set)}) の件数が不一致'
            '(インデックス整列が崩れている)'
        )

        key_params = {'seq_len_irrelevant': True, 'train_limit': args.train_limit}
        key = teacher_cache_key(
            teacher.fingerprint, 'sst2', key_params,
            split='train', n=len(raw_train),
        )
        val_key = teacher_cache_key(
            teacher.fingerprint, 'sst2', key_params,
            split='validation', n=len(raw_val),
        )
        print(f'\n[KD] teacher={args.distill_teacher} '
              f'α={args.distill_alpha} T={args.distill_temperature} '
              f'standardize={args.distill_standardize} '
              f'pseudo_labels={args.distill_pseudo_labels}')
        print(f'[KD] teacherロジット計算/ロード中(cache={args.distill_cache}, '
              f'key={key})...')
        if args.distill_cache:
            teacher_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device,
            )
            teacher_val_logits = get_or_compute_logits(
                teacher, raw_val, key=val_key, device=args.device,
            )
        else:
            import tempfile
            tmp_root = tempfile.mkdtemp(prefix='md-distill-nocache-')
            teacher_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device, root=tmp_root,
            )
            teacher_val_logits = get_or_compute_logits(
                teacher, raw_val, key=val_key, device=args.device, root=tmp_root,
            )
        print(f'[KD] teacherロジット shape={tuple(teacher_logits.shape)}')

        packed_train = PackedDistillDataset(
            train_set, teacher_logits,
            pseudo_labels=args.distill_pseudo_labels,
        )
        train_loader = DataLoader(
            packed_train, batch_size=args.batch_size, shuffle=True,
            pin_memory=use_pin_memory, drop_last=True, num_workers=0,
        )

        _kd = make_kd_loss(
            alpha=args.distill_alpha,
            temperature=args.distill_temperature,
            standardize=args.distill_standardize,
            kind='kl',
        )

        def loss_fn(output, target):
            labels = target[..., 0]
            teacher_l = target[..., 1:1 + n_classes]
            return _kd(output, labels.long(), teacher_l)
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            pin_memory=use_pin_memory, drop_last=True, num_workers=0,
        )

    # モデル構築
    model = build_model(args, actual_vocab_size)

    # パラメータ統計
    counts = model.count_parameters()
    print(f'\nモデル: {args.student}')
    print('パラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(model)

    # オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    tau_sched = None
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        _total = (args.num_iterations if args.num_iterations is not None
                  else (args.epochs or 5) * max(1, len(train_loader)))
        tau_sched = TauAnnealScheduler(
            model, tau_0=args.tau_b0, tau_min=args.tau_b_min,
            total_steps=_total,
            anneal_noise=(args.selection == 'gumbel_st'),
        )

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 20
        use_epochs = True

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(f'START student={args.student} vocab={actual_vocab_size} '
        f'seq_len={args.seq_len} distill={args.distill_teacher or "off"} '
        f'lr={args.lr} wd={args.weight_decay} bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iters=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device} params={counts["total"]:,}')

    # チェックポイント
    ckpt_prefix = f'kd_sst2_{args.student}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = f'{args.student}_v{actual_vocab_size}_s{args.seq_len}'
    if use_epochs:
        run_name += f'_e{args.epochs}'
    else:
        run_name += f'_ni{args.num_iterations}'
    wandb_tags = ['kd', 'sst2', args.student]
    if distill_on:
        wandb_tags.append('distill')
        run_name = f'kd_{run_name}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=wandb_tags,
        group='kd-text',
    )

    # 訓練ループ
    best_hard_acc = 0.0
    batches_per_epoch = len(train_loader)
    global_step = 0

    if use_epochs:
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'\n訓練開始: {args.epochs}エポック '
              f'({batches_per_epoch}バッチ/エポック, '
              f'評価={eval_freq}エポックごと)')
    else:
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'\n訓練開始: {args.num_iterations:,}イテレーション '
              f'(評価={eval_freq:,}イテレーションごと)')

    print('=' * 70)
    train_start = time.time()

    def run_eval(step_label, total_label):
        """soft/hard精度 + teacher一致率 + soft-hardギャップをまとめて計算する。"""
        acc_hard = evaluate_acc(
            model, val_loader, args.device, use_soft=False,
            desc=f'eval(hard) {step_label}/{total_label}',
        )
        acc_soft = evaluate_acc(
            model, val_loader, args.device, use_soft=True,
            desc=f'eval(soft) {step_label}/{total_label}',
        )
        gap = acc_soft - acc_hard
        agreement = None
        if teacher_val_logits is not None:
            agreement = evaluate_teacher_agreement(
                model, val_loader, teacher_val_logits, args.device, use_soft=False,
            )
        return acc_hard, acc_soft, gap, agreement

    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f'epoch {epoch}/{args.epochs}',
                leave=False,
            )
            for x, target in pbar:
                x = x.to(args.device)
                target = target.to(args.device)
                total_loss, main_loss, emb_loss = train_step(
                    model, x, target, loss_fn, optimizer, global_step,
                    tau_sched, spectral_reg=args.spectral_reg,
                )
                epoch_loss += total_loss
                global_step += 1

                elapsed_t = time.time() - train_start
                progress = (epoch - 1 + (pbar.n + 1) / len(train_loader)) / args.epochs
                remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
                pbar.set_postfix(loss=f'{total_loss:.4f}', eta=eta_str)

            avg_loss = epoch_loss / batches_per_epoch

            elapsed = time.time() - train_start
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (args.epochs - epoch)
            eta = datetime.now(JST) + timedelta(seconds=remaining)

            if epoch % eval_freq == 0 or epoch == args.epochs:
                acc_hard, acc_soft, gap, agreement = run_eval(epoch, args.epochs)

                is_best = acc_hard > best_hard_acc
                if is_best:
                    best_hard_acc = acc_hard

                step_now = epoch * batches_per_epoch
                save_checkpoint(latest_path, model, optimizer, None,
                                step_now, best_hard_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step_now, best_hard_acc, ckpt_metadata)

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')
                emb_w = get_emb_reg_weight(global_step)
                agree_str = f'  agree={agreement:.4f}' if agreement is not None else ''

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_loss:.4f} (emb_w={emb_w:.4f})  '
                    f'acc(hard)={acc_hard:.4f}  acc(soft)={acc_soft:.4f}  '
                    f'gap={gap:.4f}{agree_str}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'emb_w={emb_w:.4f} acc_hard={acc_hard:.4f} acc_soft={acc_soft:.4f} '
                    f'gap={gap:.4f} agreement={agreement} '
                    f'best={str(is_best).lower()} elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    metrics = {
                        'epoch': epoch,
                        'step': global_step,
                        'loss': avg_loss,
                        'emb_reg_weight': emb_w,
                        'acc_hard': acc_hard,
                        'acc_soft': acc_soft,
                        'soft_hard_gap': gap,
                        'best_acc_hard': best_hard_acc,
                    }
                    if agreement is not None:
                        metrics['teacher_student_agreement'] = agreement
                    log_metrics(metrics)
    else:
        pbar = tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='train',
            total=args.num_iterations,
        )
        for i, (x, target) in pbar:
            x = x.to(args.device)
            target = target.to(args.device)

            total_loss, main_loss, emb_loss = train_step(
                model, x, target, loss_fn, optimizer, i, tau_sched,
                spectral_reg=args.spectral_reg,
            )
            global_step = i + 1

            epoch_now = global_step / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / global_step
            remaining_sec = sec_per_iter * (args.num_iterations - global_step)
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{total_loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if global_step % eval_freq == 0 or global_step == args.num_iterations:
                acc_hard, acc_soft, gap, agreement = run_eval(global_step, args.num_iterations)

                is_best = acc_hard > best_hard_acc
                if is_best:
                    best_hard_acc = acc_hard

                save_checkpoint(latest_path, model, optimizer, None,
                                global_step, best_hard_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    global_step, best_hard_acc, ckpt_metadata)

                elapsed = time.time() - train_start
                remaining = sec_per_iter * (args.num_iterations - global_step)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                eta_str = eta.strftime('%H:%M:%S')
                emb_w = get_emb_reg_weight(global_step)
                agree_str = f'  agree={agreement:.4f}' if agreement is not None else ''

                print(
                    f'\n[iter {global_step:,} / epoch {epoch_now:.1f}] '
                    f'loss={total_loss:.4f} (emb_w={emb_w:.4f})  '
                    f'acc(hard)={acc_hard:.4f}  acc(soft)={acc_soft:.4f}  '
                    f'gap={gap:.4f}{agree_str}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'ITER {global_step}/{args.num_iterations} '
                    f'epoch={epoch_now:.1f} loss={total_loss:.4f} '
                    f'emb_w={emb_w:.4f} acc_hard={acc_hard:.4f} acc_soft={acc_soft:.4f} '
                    f'gap={gap:.4f} agreement={agreement} '
                    f'best={str(is_best).lower()} elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    metrics = {
                        'iteration': global_step,
                        'epoch': epoch_now,
                        'loss': total_loss,
                        'emb_reg_weight': emb_w,
                        'acc_hard': acc_hard,
                        'acc_soft': acc_soft,
                        'soft_hard_gap': gap,
                        'best_acc_hard': best_hard_acc,
                    }
                    if agreement is not None:
                        metrics['teacher_student_agreement'] = agreement
                    log_metrics(metrics, step=global_step)

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
