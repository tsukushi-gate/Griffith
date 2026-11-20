"""
CDDLGN TextCNN 訓練スクリプト（WRIME 8感情2値分類）

arXiv 2505.00013 の実験条件に準拠し、WRIME ver1 の8感情（Joy, Sadness,
Anticipation, Surprise, Anger, Fear, Disgust, Trust）それぞれの有無を
LUT-based TextCNN で2値分類する。

実験条件（arXiv 2505.00013準拠）:
- データ: WRIME ver1 (43,200件), Avg. Readers アノテーション
- 二値化: 強度 ≥1 で陽性
- 分割: 80/20 ランダム分割
- 評価: 8感情のマクロ平均accuracy・マクロ平均F1

使用例:
    # デフォルト設定（50エポック）
    uv run python experiments/train_textcnn_emotion.py -e 50

    # 小さいモデルでデバッグ
    uv run python experiments/train_textcnn_emotion.py --d-emb 64 --n-filters 16 -e 3

    # クラス重みバランシング
    uv run python experiments/train_textcnn_emotion.py -e 50 --class-weight balanced
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.wrime_emotion import EMOTIONS
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LogicEmbedding, LutConv2d, LutLayer

load_dotenv()

N_EMOTIONS = len(EMOTIONS)
N_CLASSES = N_EMOTIONS * 2  # 8感情 × 2クラス = 16


# ============================================================
# モデル
# ============================================================

class TextCNNModel(nn.Module):
    """TextCNN (Kim 2014) を CDDLGN で実装（8感情2値分類版）。

    出力は (B, 16) のロジット。reshape (B, 8, 2) で各感情の2クラス分類となる。
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        n_filters: int,
        kernel_sizes: list[int],
        dense_dims: list[int],
        tree_depth: int = 3,
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        parameterization: str = 'softmax',
        connections: str = 'random',
        tau: float | None = None,
    ):
        super().__init__()
        self.d_emb = d_emb

        self.embedding = LogicEmbedding(vocab_size, d_emb)

        conv_kw = dict(
            n_inputs=n_inputs,
            device=device,
            grad_factor=grad_factor,
            input_channels_per_tree=1,
            parameterization=parameterization,
        )
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(
                LutConv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(k, d_emb),
                    tree_depth=tree_depth,
                    padding=(k // 2, 0),
                    **conv_kw,
                )
            )

        concat_dim = n_filters * len(kernel_sizes)
        lut_kw = dict(
            n_inputs=n_inputs,
            device=device,
            grad_factor=grad_factor,
            connections=connections,
            parameterization=parameterization,
        )
        self.dense_layers = nn.ModuleList()
        prev = concat_dim
        for dim in dense_dims:
            self.dense_layers.append(LutLayer(
                in_dim=prev, out_dim=dim, **lut_kw,
            ))
            prev = dim

        n_outputs_per_class = prev // N_CLASSES
        if tau is None:
            tau = max(1.0, math.sqrt(n_outputs_per_class))
        self.group_sum = GroupSum(k=N_CLASSES, tau=tau)
        self.tau = tau

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 16) → reshape (B, 8, 2) で各感情の2クラスロジット
            emb_reg_loss: 二値化正則化損失
        """
        emb = self.embedding(token_ids)
        x = emb.unsqueeze(1)

        branch_outputs = []
        for branch in self.branches:
            h = branch(x)
            h = F.adaptive_max_pool2d(h, (1, 1))
            h = h.flatten(1)
            branch_outputs.append(h)

        x = torch.cat(branch_outputs, dim=-1)

        for layer in self.dense_layers:
            x = layer(x)

        logits = self.group_sum(x)
        emb_reg_loss = self.embedding.binary_reg_loss()
        return logits, emb_reg_loss

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        counts['embedding'] = sum(p.numel() for p in self.embedding.parameters())
        counts['conv_branches'] = sum(p.numel() for p in self.branches.parameters())
        counts['dense'] = sum(p.numel() for p in self.dense_layers.parameters())
        counts['total'] = sum(counts.values())
        return counts


# ============================================================
# データセット
# ============================================================

def load_dataset_emotion(args):
    """WRIME 8感情データセットをロードする。"""
    from data.wrime_emotion import download_and_tokenize_emotion
    train_set, test_set, tokenizer = download_and_tokenize_emotion(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        annotation=args.annotation,
        threshold=args.threshold,
        seed=args.seed,
    )
    return train_set, test_set, tokenizer


# ============================================================
# ユーティリティ
# ============================================================

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
    """二値化正則化の重みを線形ランプで計算する。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


def compute_emotion_class_weights(dataset, device: str) -> list[torch.Tensor]:
    """8感情それぞれの逆頻度重みを計算する。

    Returns:
        8個の重みテンソル（各shape (2,)）のリスト
    """
    labels = dataset.labels  # (N, 8)
    weights_list = []
    for i in range(N_EMOTIONS):
        col = labels[:, i]
        n_neg = (col == 0).sum().float()
        n_pos = (col == 1).sum().float()
        total = n_neg + n_pos
        w = total / (2 * torch.stack([n_neg, n_pos]))
        weights_list.append(w.to(device))
    return weights_list


def compute_f1(tp: int, fp: int, fn: int) -> float:
    """Precision, Recall, F1を計算する。"""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# ============================================================
# ログ
# ============================================================

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'textcnn_emotion_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, y, loss_fns, optimizer, step, use_balanced_weights):
    """1バッチの訓練ステップ。

    Args:
        y: (B, 8) 各感情の二値ラベル
        loss_fns: 単一のCrossEntropyLoss、またはclass-weight使用時は8個のリスト
    """
    model.train()
    logits, emb_loss = model(x)  # (B, 16)
    B = logits.shape[0]
    logits_reshaped = logits.view(B, N_EMOTIONS, 2)  # (B, 8, 2)

    if use_balanced_weights:
        main_loss = torch.tensor(0.0, device=logits.device)
        for i in range(N_EMOTIONS):
            main_loss = main_loss + loss_fns[i](logits_reshaped[:, i, :], y[:, i])
        main_loss = main_loss / N_EMOTIONS
    else:
        logits_flat = logits_reshaped.reshape(B * N_EMOTIONS, 2)
        labels_flat = y.reshape(B * N_EMOTIONS)
        main_loss = loss_fns(logits_flat, labels_flat)

    emb_weight = get_emb_reg_weight(step)
    total_loss = main_loss + emb_weight * emb_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), main_loss.item(), emb_loss.item()


@torch.no_grad()
def evaluate(model, loader, device, use_soft=True, desc=None):
    """8感情それぞれのaccuracy・F1を計算する。

    Returns:
        dict: {
            'per_emotion': [{name, accuracy, f1, tp, fp, fn, tn}, ...],
            'mean_accuracy': float,
            'mean_f1': float,
        }
    """
    was_training = model.training
    model.train(mode=use_soft)

    # 感情ごとの集計
    tp = [0] * N_EMOTIONS
    fp = [0] * N_EMOTIONS
    fn = [0] * N_EMOTIONS
    tn = [0] * N_EMOTIONS

    it = tqdm(loader, desc=desc, leave=False) if desc else loader

    for token_ids, labels in it:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        logits, _ = model(token_ids)  # (B, 16)
        B = logits.shape[0]
        preds = logits.view(B, N_EMOTIONS, 2).argmax(dim=-1)  # (B, 8)

        for i in range(N_EMOTIONS):
            p = preds[:, i]
            l = labels[:, i]
            tp[i] += ((p == 1) & (l == 1)).sum().item()
            fp[i] += ((p == 1) & (l == 0)).sum().item()
            fn[i] += ((p == 0) & (l == 1)).sum().item()
            tn[i] += ((p == 0) & (l == 0)).sum().item()

    model.train(mode=was_training)

    per_emotion = []
    total_acc = 0.0
    total_f1 = 0.0
    for i in range(N_EMOTIONS):
        total = tp[i] + fp[i] + fn[i] + tn[i]
        acc = (tp[i] + tn[i]) / total if total > 0 else 0.0
        f1 = compute_f1(tp[i], fp[i], fn[i])
        per_emotion.append({
            'name': EMOTIONS[i],
            'accuracy': acc,
            'f1': f1,
            'tp': tp[i], 'fp': fp[i], 'fn': fn[i], 'tn': tn[i],
        })
        total_acc += acc
        total_f1 += f1

    return {
        'per_emotion': per_emotion,
        'mean_accuracy': total_acc / N_EMOTIONS,
        'mean_f1': total_f1 / N_EMOTIONS,
    }


def format_eval_result(result: dict, mode: str) -> str:
    """評価結果を見やすい文字列にフォーマットする。"""
    lines = [f'  [{mode}] 平均acc={result["mean_accuracy"]:.4f}  平均F1={result["mean_f1"]:.4f}']
    for e in result['per_emotion']:
        lines.append(
            f'    {e["name"]:>13s}: acc={e["accuracy"]:.4f}  F1={e["f1"]:.4f}'
            f'  (TP={e["tp"]} FP={e["fp"]} FN={e["fn"]} TN={e["tn"]})'
        )
    return '\n'.join(lines)


# ============================================================
# メイン
# ============================================================

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='CDDLGN TextCNN 訓練（WRIME 8感情2値分類）'
    )

    # データ設定
    parser.add_argument('--vocab-size', type=int, default=8000)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='./data-wrime-emotion')
    parser.add_argument('--annotation', type=str, default='reader',
                        choices=['reader', 'writer'],
                        help='アノテーション (default: reader, arXiv 2505.00013準拠)')
    parser.add_argument('--threshold', type=int, default=1,
                        help='二値化閾値 (default: 1, ≥1で陽性)')

    # モデル設定
    parser.add_argument('--d-emb', type=int, default=256)
    parser.add_argument('--n-filters', type=int, default=64)
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5')
    parser.add_argument('--dense-dims', type=str, default='1024,512')
    parser.add_argument('--tree-depth', type=int, default=3)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'])
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'])
    parser.add_argument('--grad-factor', type=float, default=1.0)

    # 訓練設定
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=None)
    parser.add_argument('-ni', '--num-iterations', type=int, default=None)
    parser.add_argument('-ef', '--eval-freq', type=int, default=None)
    parser.add_argument('--class-weight', type=str, default=None,
                        choices=['balanced'])

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')

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

    kernel_sizes = parse_int_list(args.kernel_sizes)
    dense_dims = parse_int_list(args.dense_dims)

    # データ読み込み
    train_set, test_set, tokenizer = load_dataset_emotion(args)
    actual_vocab_size = tokenizer.vocab_size

    use_pin_memory = args.device == 'cuda'
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )

    print('\nデータセット: WRIME 8感情2値分類')
    print(f'訓練: {len(train_set):,} サンプル, テスト: {len(test_set):,} サンプル')
    print(f'語彙サイズ: {actual_vocab_size:,}')
    print(f'アノテーション: {args.annotation}, 閾値: ≥{args.threshold}')

    # Dense最終層がN_CLASSESで割り切れるよう調整
    if dense_dims[-1] % N_CLASSES != 0:
        dense_dims[-1] = (dense_dims[-1] // N_CLASSES) * N_CLASSES
        print(f'Dense最終層を {dense_dims[-1]} に調整（{N_CLASSES}の倍数）')

    # モデル構築
    model = TextCNNModel(
        vocab_size=actual_vocab_size,
        d_emb=args.d_emb,
        n_filters=args.n_filters,
        kernel_sizes=kernel_sizes,
        dense_dims=dense_dims,
        tree_depth=args.tree_depth,
        n_inputs=args.n_inputs,
        device=args.device,
        grad_factor=args.grad_factor,
        parameterization=args.parameterization,
        connections=args.connections,
        tau=args.tau,
    ).to(args.device)

    counts = model.count_parameters()
    total_gates = sum(
        m.num_gates for m in model.modules() if isinstance(m, LutConv2d)
    )
    total_neurons = sum(
        m.num_neurons for m in model.modules() if isinstance(m, LutLayer)
    )

    param_labels = {
        'softmax': 'softmax (DLGN方式)',
        'independent': 'independent (BinaryBrain方式)',
        'iwp': 'iwp (LDDLGN方式)',
    }
    print('\nモデル構成:')
    print(f'  埋め込み次元: {args.d_emb}')
    print(f'  カーネルサイズ: {kernel_sizes}')
    print(f'  フィルタ数/ブランチ: {args.n_filters}')
    print(f'  Dense層: {dense_dims}')
    print(f'  ツリー深さ: {args.tree_depth}, ゲート入力数: {args.n_inputs}')
    print(f'  パラメータ化: {param_labels[args.parameterization]}')
    print(f'  出力: GroupSum(k={N_CLASSES}) → (B, {N_CLASSES}) → (B, {N_EMOTIONS}, 2)')
    print(f'  τ = {model.tau:.1f}')
    print('パラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'Conv総ゲート数: {total_gates:,}')
    print(f'Dense総ゲート数: {total_neurons:,}')
    print(f'総ロジックゲート数: {total_gates + total_neurons:,} (LUT{args.n_inputs})')

    # 損失関数
    use_balanced_weights = args.class_weight == 'balanced'
    if use_balanced_weights:
        emotion_weights = compute_emotion_class_weights(train_set, args.device)
        loss_fns = [nn.CrossEntropyLoss(weight=w) for w in emotion_weights]
        print('クラス重み (balanced):')
        for i, emo in enumerate(EMOTIONS):
            print(f'  {emo:>13s}: {emotion_weights[i].tolist()}')
    else:
        loss_fns = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 50
        use_epochs = True

    log, log_file = setup_logger(args.log_dir)
    log(f'START task=8emotion vocab={actual_vocab_size} d_emb={args.d_emb} '
        f'n_filters={args.n_filters} kernels={kernel_sizes} '
        f'dense={dense_dims} tree_depth={args.tree_depth} '
        f'n_inputs={args.n_inputs} param={args.parameterization} '
        f'tau={model.tau:.1f} lr={args.lr} wd={args.weight_decay} '
        f'bs={args.batch_size} annotation={args.annotation} threshold={args.threshold} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iters=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device} params={counts["total"]:,}')

    run_name = (f'textcnn_emo_d{args.d_emb}_f{args.n_filters}'
                f'_k{"".join(str(k) for k in kernel_sizes)}'
                f'_e{args.epochs or args.num_iterations}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['textcnn', 'wrime-emotion', '8emotion', f'n{args.n_inputs}'],
        group='textcnn-emotion',
    )

    # 訓練ループ
    best_mean_acc = 0.0
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

    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f'epoch {epoch}/{args.epochs}',
                leave=False,
            )
            for token_ids, labels in pbar:
                token_ids = token_ids.to(args.device)
                labels = labels.to(args.device)
                total_loss, main_loss, emb_loss = train_step(
                    model, token_ids, labels, loss_fns, optimizer,
                    global_step, use_balanced_weights,
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
                result_hard = evaluate(
                    model, test_loader, args.device, use_soft=False,
                    desc=f'eval(hard) {epoch}/{args.epochs}',
                )
                result_soft = evaluate(
                    model, test_loader, args.device, use_soft=True,
                    desc=f'eval(soft) {epoch}/{args.epochs}',
                )

                is_best = result_hard['mean_accuracy'] > best_mean_acc
                if is_best:
                    best_mean_acc = result_hard['mean_accuracy']

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')
                emb_w = get_emb_reg_weight(global_step)

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_loss:.4f} (emb_w={emb_w:.4f})'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )
                print(format_eval_result(result_hard, 'hard'))
                print(format_eval_result(result_soft, 'soft'))

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'emb_w={emb_w:.4f} '
                    f'mean_acc_hard={result_hard["mean_accuracy"]:.4f} '
                    f'mean_f1_hard={result_hard["mean_f1"]:.4f} '
                    f'mean_acc_soft={result_soft["mean_accuracy"]:.4f} '
                    f'mean_f1_soft={result_soft["mean_f1"]:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')

                if use_wandb:
                    metrics = {
                        'epoch': epoch,
                        'step': global_step,
                        'loss': avg_loss,
                        'emb_reg_weight': emb_w,
                        'mean_acc_hard': result_hard['mean_accuracy'],
                        'mean_f1_hard': result_hard['mean_f1'],
                        'mean_acc_soft': result_soft['mean_accuracy'],
                        'mean_f1_soft': result_soft['mean_f1'],
                        'best_mean_acc': best_mean_acc,
                    }
                    for e in result_hard['per_emotion']:
                        metrics[f'acc_hard_{e["name"]}'] = e['accuracy']
                        metrics[f'f1_hard_{e["name"]}'] = e['f1']
                    log_metrics(metrics)
    else:
        pbar = tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='train',
            total=args.num_iterations,
        )
        for i, (token_ids, labels) in pbar:
            token_ids = token_ids.to(args.device)
            labels = labels.to(args.device)

            total_loss, main_loss, emb_loss = train_step(
                model, token_ids, labels, loss_fns, optimizer,
                i, use_balanced_weights,
            )
            global_step = i + 1

            epoch_now = global_step / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / global_step
            remaining_sec = sec_per_iter * (args.num_iterations - global_step)
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{total_loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if global_step % eval_freq == 0:
                result_hard = evaluate(
                    model, test_loader, args.device, use_soft=False,
                    desc=f'eval(hard) {global_step}/{args.num_iterations}',
                )
                result_soft = evaluate(
                    model, test_loader, args.device, use_soft=True,
                    desc=f'eval(soft) {global_step}/{args.num_iterations}',
                )

                is_best = result_hard['mean_accuracy'] > best_mean_acc
                if is_best:
                    best_mean_acc = result_hard['mean_accuracy']

                elapsed = time.time() - train_start
                remaining = sec_per_iter * (args.num_iterations - global_step)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                eta_str = eta.strftime('%H:%M:%S')
                emb_w = get_emb_reg_weight(global_step)

                print(
                    f'\n[iter {global_step:,} / epoch {epoch_now:.1f}] '
                    f'loss={total_loss:.4f} (emb_w={emb_w:.4f})'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )
                print(format_eval_result(result_hard, 'hard'))
                print(format_eval_result(result_soft, 'soft'))

                log(f'ITER {global_step}/{args.num_iterations} '
                    f'epoch={epoch_now:.1f} loss={total_loss:.4f} '
                    f'emb_w={emb_w:.4f} '
                    f'mean_acc_hard={result_hard["mean_accuracy"]:.4f} '
                    f'mean_f1_hard={result_hard["mean_f1"]:.4f} '
                    f'mean_acc_soft={result_soft["mean_accuracy"]:.4f} '
                    f'mean_f1_soft={result_soft["mean_f1"]:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')

                if use_wandb:
                    metrics = {
                        'iteration': global_step,
                        'epoch': epoch_now,
                        'loss': total_loss,
                        'emb_reg_weight': emb_w,
                        'mean_acc_hard': result_hard['mean_accuracy'],
                        'mean_f1_hard': result_hard['mean_f1'],
                        'mean_acc_soft': result_soft['mean_accuracy'],
                        'mean_f1_soft': result_soft['mean_f1'],
                        'best_mean_acc': best_mean_acc,
                    }
                    for e in result_hard['per_emotion']:
                        metrics[f'acc_hard_{e["name"]}'] = e['accuracy']
                        metrics[f'f1_hard_{e["name"]}'] = e['f1']
                    log_metrics(metrics)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('\n最終結果 (hard mode):')
    print(f'  最良平均accuracy: {best_mean_acc:.4f}')
    print(f'  総訓練時間: {total_time}')
    print('\n比較対象 (arXiv 2505.00013):')
    print('  DeBERTa-v3-base: 87.0%  (F1=0.630)')
    print('  BERT:            85.6%  (F1=0.572)')
    print('  ChatGPT-4o:      84.4%  (F1=0.527)')

    log(f'DONE best_mean_acc={best_mean_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
