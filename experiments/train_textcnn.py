"""
CDDLGN TextCNN 訓練スクリプト（日本語感情分析）

TextCNN (Kim 2014) を微分可能論理ゲートツリー（CDDLGN）で実装。
LutConv2d を kernel_size=(k, d_emb) で使用し、1D畳み込みとして機能させる。

使用例:
    # デフォルト設定（WRIME, 50エポック）
    uv run python experiments/train_textcnn.py -e 50

    # 小さいモデルでデバッグ
    uv run python experiments/train_textcnn.py --d-emb 64 --n-filters 16 -e 5

    # イテレーションモード
    uv run python experiments/train_textcnn.py -ni 20000 -ef 1000

    # IWPパラメータ化
    uv run python experiments/train_textcnn.py --parameterization iwp -e 50
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
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LogicEmbedding, LutConv2d, LutLayer

# .envから環境変数を読み込み
load_dotenv()


# ============================================================
# モデル
# ============================================================

class TextCNNModel(nn.Module):
    """
    TextCNN (Kim 2014) を CDDLGN で実装したモデル。

    アーキテクチャ:
        LogicEmbedding(vocab_size, d_emb)
        → unsqueeze(1): (B, 1, seq_len, d_emb)
        → 複数の LutConv2d ブランチ（kernel_sizes=[3,4,5]）
            各ブランチ: kernel_size=(k, d_emb) → 実質的な1D畳み込み
        → Global Max Pool (adaptive_max_pool2d)
        → Concatenate: (B, n_filters * n_branches)
        → LutLayer stack → GroupSum → n_classes logits

    LutConv2d を kernel_size=(k, d_emb) で使う理由:
        Kim 2014 のフィルタは埋め込み次元全体をカバーする設計。
        kernel_w = d_emb とすることで W_out=1 となり、実質1D出力になる。
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        n_filters: int,
        kernel_sizes: list[int],
        dense_dims: list[int],
        n_classes: int,
        tree_depth: int = 3,
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        parameterization: str = 'softmax',
        connections: str = 'random',
        tau: float | None = None,
        selection: str = 'soft',
    ):
        super().__init__()
        self.d_emb = d_emb
        self.n_classes = n_classes

        # 埋め込み層
        self.embedding = LogicEmbedding(vocab_size, d_emb)

        # Conv2dブランチ（各n-gramサイズに1つ）
        # LutConv2dはwarp未対応のためconv木のみsoftmaxへフォールバック
        # (warpはdense層にのみ適用される)
        conv_param = (parameterization
                      if parameterization in LutConv2d.PARAMETERIZATIONS
                      else 'softmax')
        conv_kw = dict(
            n_inputs=n_inputs,
            device=device,
            grad_factor=grad_factor,
            input_channels_per_tree=1,
            parameterization=conv_param,
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

        # Dense層
        concat_dim = n_filters * len(kernel_sizes)
        lut_kw = dict(
            n_inputs=n_inputs,
            device=device,
            grad_factor=grad_factor,
            connections=connections,
            parameterization=parameterization,
            # selectionはdense層のみ(conv木のhard/gumbel-STは非ゴール)
            selection=selection,
        )
        self.dense_layers = nn.ModuleList()
        prev = concat_dim
        for dim in dense_dims:
            self.dense_layers.append(LutLayer(
                in_dim=prev, out_dim=dim, **lut_kw,
            ))
            prev = dim

        # GroupSum: τの自動計算
        n_outputs_per_class = prev // n_classes
        if tau is None:
            tau = max(1.0, math.sqrt(n_outputs_per_class))
        self.group_sum = GroupSum(k=n_classes, tau=tau)
        self.tau = tau

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (B, seq_len) 整数トークンID

        Returns:
            logits: (B, n_classes)
            emb_reg_loss: 二値化正則化損失（スカラー）
        """
        # (B, seq_len) → (B, seq_len, d_emb)
        emb = self.embedding(token_ids)
        # (B, seq_len, d_emb) → (B, 1, seq_len, d_emb)
        x = emb.unsqueeze(1)

        # 各ブランチで畳み込み→Global Max Pool
        branch_outputs = []
        for branch in self.branches:
            h = branch(x)  # (B, n_filters, H_out, 1)
            h = F.adaptive_max_pool2d(h, (1, 1))  # (B, n_filters, 1, 1)
            h = h.flatten(1)  # (B, n_filters)
            branch_outputs.append(h)

        # 連結
        x = torch.cat(branch_outputs, dim=-1)  # (B, n_filters * n_branches)

        # Dense層
        for layer in self.dense_layers:
            x = layer(x)

        logits = self.group_sum(x)
        emb_reg_loss = self.embedding.binary_reg_loss()
        return logits, emb_reg_loss

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す。"""
        counts = {}
        counts['embedding'] = sum(
            p.numel() for p in self.embedding.parameters()
        )
        counts['conv_branches'] = sum(
            p.numel() for p in self.branches.parameters()
        )
        counts['dense'] = sum(
            p.numel() for p in self.dense_layers.parameters()
        )
        counts['total'] = sum(counts.values())
        return counts


# ============================================================
# データセット
# ============================================================

def load_dataset_wrime(args):
    """WRIMEデータセットをロードする。"""
    from data.wrime import download_and_tokenize
    train_set, test_set, tokenizer = download_and_tokenize(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
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


def compute_class_weights(dataset, n_classes: int, device: str) -> torch.Tensor:
    """クラスの逆頻度重みを計算する。"""
    counts = torch.zeros(n_classes)
    for i in range(len(dataset)):
        _, label = dataset[i]
        counts[label] += 1
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device)


# ============================================================
# ログ
# ============================================================

def setup_logger(log_dir='logs'):
    """ログファイルを作成する。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'textcnn_wrime_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, y, loss_fn, optimizer, step,
               tau_sched=None, spectral_reg=0.0):
    """1バッチの訓練ステップ。"""
    model.train()
    logits, emb_loss = model(x)
    main_loss = loss_fn(logits, y)

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
def evaluate(model, loader, device, use_soft=True, desc=None):
    """精度を計算する。

    Args:
        use_soft: True=訓練モード（softmax重み）, False=推論モード（離散化）
        desc: プログレスバーの説明文
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


# ============================================================
# メイン
# ============================================================

def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。"""
    return [int(x.strip()) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='CDDLGN TextCNN 訓練スクリプト（日本語感情分析）'
    )

    # データ設定
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='語彙サイズ (default: 8000)')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='最大系列長 (default: 128)')
    parser.add_argument('--data-dir', type=str, default='./data-wrime',
                        help='データ保存先 (default: ./data-wrime)')

    # モデル設定
    parser.add_argument('--d-emb', type=int, default=256,
                        help='埋め込み次元 (default: 256)')
    parser.add_argument('--n-filters', type=int, default=64,
                        help='ブランチあたりのフィルタ数 (default: 64)')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='カーネルサイズ (default: 3,4,5)')
    parser.add_argument('--dense-dims', type=str, default='1024,512',
                        help='Dense層の次元 (default: 1024,512)')
    parser.add_argument('--tree-depth', type=int, default=3,
                        help='ツリーの深さ (default: 3)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--tau', type=float, default=None,
                        help='GroupSum温度（Noneで自動計算）')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'],
                        help='パラメータ化方式 (default: softmax。'
                             'warpはdense層のみ、conv木はsoftmaxへフォールバック)')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式(dense層のみ。default: soft)')
    parser.add_argument('--tau-anneal', action='store_true',
                        help='TauAnnealScheduler(τ_b幾何減衰+ノイズ減衰)を有効化')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
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
    parser.add_argument('--class-weight', type=str, default=None,
                        choices=['balanced'],
                        help='クラス重み（balanced=逆頻度重み）')

    # 環境
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')

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

    # カーネルサイズ・Dense次元のパース
    kernel_sizes = parse_int_list(args.kernel_sizes)
    dense_dims = parse_int_list(args.dense_dims)

    # データ読み込み
    train_set, test_set, tokenizer = load_dataset_wrime(args)
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

    print('\nデータセット: WRIME')
    print(f'訓練: {len(train_set):,} サンプル, テスト: {len(test_set):,} サンプル')
    print(f'語彙サイズ: {actual_vocab_size:,}')

    # Dense最終層がn_classesで割り切れるよう調整
    n_classes = 2
    if dense_dims[-1] % n_classes != 0:
        dense_dims[-1] = (dense_dims[-1] // n_classes) * n_classes
        print(f'Dense最終層を {dense_dims[-1]} に調整（{n_classes}の倍数）')

    # モデル構築
    model = TextCNNModel(
        vocab_size=actual_vocab_size,
        d_emb=args.d_emb,
        n_filters=args.n_filters,
        kernel_sizes=kernel_sizes,
        dense_dims=dense_dims,
        n_classes=n_classes,
        tree_depth=args.tree_depth,
        n_inputs=args.n_inputs,
        device=args.device,
        grad_factor=args.grad_factor,
        parameterization=args.parameterization,
        connections=args.connections,
        tau=args.tau,
        selection=args.selection,
    ).to(args.device)

    # パラメータ統計
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
    print(f'  τ = {model.tau:.1f}')
    print('パラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'Conv総ゲート数: {total_gates:,}')
    print(f'Dense総ゲート数: {total_neurons:,}')
    print(f'総ロジックゲート数: {total_gates + total_neurons:,} (LUT{args.n_inputs})')
    print(model)

    # 損失関数
    if args.class_weight == 'balanced':
        class_weights = compute_class_weights(train_set, n_classes, args.device)
        print(f'クラス重み: {class_weights.tolist()}')
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    tau_sched = None
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        _total = (args.num_iterations if args.num_iterations is not None
                  else (args.epochs or 5) * 1000)
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
        args.epochs = 50
        use_epochs = True

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(f'START vocab={actual_vocab_size} d_emb={args.d_emb} '
        f'n_filters={args.n_filters} kernels={kernel_sizes} '
        f'dense={dense_dims} tree_depth={args.tree_depth} '
        f'n_inputs={args.n_inputs} param={args.parameterization} '
        f'tau={model.tau:.1f} lr={args.lr} wd={args.weight_decay} '
        f'bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iters=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device} params={counts["total"]:,}')

    # wandb
    run_name = (f'textcnn_d{args.d_emb}_f{args.n_filters}'
                f'_k{"".join(str(k) for k in kernel_sizes)}'
                f'_e{args.epochs or args.num_iterations}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['textcnn', 'wrime', f'n{args.n_inputs}'],
        group='textcnn',
    )

    # 訓練ループ
    best_test_acc = 0.0
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
                    model, token_ids, labels, loss_fn, optimizer, global_step,
                    tau_sched=tau_sched, spectral_reg=args.spectral_reg,
                )
                epoch_loss += total_loss
                global_step += 1

                elapsed_t = time.time() - train_start
                progress = (epoch - 1 + (pbar.n + 1) / len(train_loader)) / args.epochs
                remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
                pbar.set_postfix(loss=f'{total_loss:.4f}', eta=eta_str)

            avg_loss = epoch_loss / batches_per_epoch

            # 時間計算
            elapsed = time.time() - train_start
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (args.epochs - epoch)
            eta = datetime.now(JST) + timedelta(seconds=remaining)

            # 評価
            if epoch % eval_freq == 0 or epoch == args.epochs:
                test_acc_hard = evaluate(
                    model, test_loader, args.device, use_soft=False,
                    desc=f'eval(hard) {epoch}/{args.epochs}',
                )
                test_acc_soft = evaluate(
                    model, test_loader, args.device, use_soft=True,
                    desc=f'eval(soft) {epoch}/{args.epochs}',
                )

                is_best = test_acc_hard > best_test_acc
                if is_best:
                    best_test_acc = test_acc_hard

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                emb_w = get_emb_reg_weight(global_step)

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_loss:.4f} (emb_w={emb_w:.4f})  '
                    f'test_acc(hard)={test_acc_hard:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'emb_w={emb_w:.4f} '
                    f'test_acc_hard={test_acc_hard:.4f} '
                    f'test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'epoch': epoch,
                        'step': global_step,
                        'loss': avg_loss,
                        'emb_reg_weight': emb_w,
                        'test_acc_hard': test_acc_hard,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })
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
                model, token_ids, labels, loss_fn, optimizer, i,
                tau_sched=tau_sched, spectral_reg=args.spectral_reg,
            )
            global_step = i + 1

            epoch_now = global_step / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / global_step
            remaining_sec = sec_per_iter * (args.num_iterations - global_step)
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{total_loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if global_step % eval_freq == 0:
                test_acc_hard = evaluate(
                    model, test_loader, args.device, use_soft=False,
                    desc=f'eval(hard) {global_step}/{args.num_iterations}',
                )
                test_acc_soft = evaluate(
                    model, test_loader, args.device, use_soft=True,
                    desc=f'eval(soft) {global_step}/{args.num_iterations}',
                )

                is_best = test_acc_hard > best_test_acc
                if is_best:
                    best_test_acc = test_acc_hard

                elapsed = time.time() - train_start
                remaining = sec_per_iter * (args.num_iterations - global_step)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                eta_str = eta.strftime('%H:%M:%S')
                emb_w = get_emb_reg_weight(global_step)

                print(
                    f'\n[iter {global_step:,} / epoch {epoch_now:.1f}] '
                    f'loss={total_loss:.4f} (emb_w={emb_w:.4f})  '
                    f'test_acc(hard)={test_acc_hard:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'ITER {global_step}/{args.num_iterations} '
                    f'epoch={epoch_now:.1f} loss={total_loss:.4f} '
                    f'emb_w={emb_w:.4f} '
                    f'test_acc_hard={test_acc_hard:.4f} '
                    f'test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'iteration': global_step,
                        'epoch': epoch_now,
                        'loss': total_loss,
                        'emb_reg_weight': emb_w,
                        'test_acc_hard': test_acc_hard,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度(hard): {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_test_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
