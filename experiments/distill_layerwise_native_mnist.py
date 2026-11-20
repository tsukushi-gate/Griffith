"""N3 層別二値蒸留 → native DLGN(ROADMAP-v5、本キャンペーンの新規性中核)。

**問い**: fan-in制約STE teacher の隠れ二値活性(300bit/100bit)を per-layer BCE の
教師信号として、native DLGN(LutLayer N=2 softmax + GroupSum)を「ブロック」単位で
層別監督すると、scratch(quick30、6kゲート、steps_to(0.90)中央値3744)より少ない
step で hard acc 0.90 に到達できるか(=学習効率が上がるか)。

**論文的裏付け**:
- OFA-KD(2023 NeurIPS): 構造的に非同一なアーキ間KDでは生の中間特徴模倣は破綻しうる
  → 「teacher層1つに native ブロック2-3層」という多対一マッチング設計はこの裏付けを持つ。
  崩壊時のフォールバックとしてロジット空間投影(GroupSum前集約)への切替を検討。
- OFA-KD (arXiv:2310.19444) の知見「クロスアーキKDはロジット空間監督が安全」。
- difflogic-CA: N=2 DLGN は単純な論理なら深い回路でも安定 → 深さ2-3のブロックは妥当。

**設計(確定、docs参照)**:
- teacher = fan-in制約枝刈りSTE teacher(fanins=(32,12,6)、各2ep、float ckpt 初期化)。
  fan-in6版(hard acc~0.92-0.93見込み)を convert_prune_transcribe_mnist.py の機構で自前生成。
- 教師信号 = teacher の隠れ二値活性(300bit/100bit)を train全60kサンプル分、forward
  hook で1回キャプチャし outputs/ にキャッシュ(数十MB)。
- student = native DLGN。teacher層ごとにブロックを割り当て:
    ブロック構成グリッド(上限4構成、1seedスクリーニング):
        深さ∈{2,3}(ブロック内DLGN層数) × 幅対応 m∈{1,4}
    m=1: ブロック出力幅=teacher層幅(300/100)、soft出力[0,1]を teacher ビットと BCE
    m=4: ブロック出力幅=4倍(1200/400)、m個の出力のグループ平均が teacher ビット1つを BCE監督
    入力: 二値化784 MNIST(teacherと同じ)
- Stage 1: 全ブロックのBCE和をjoint最適化(進まなければ層凍結カリキュラムにフォールバック)。
- Stage 2: GroupSum頭を接続し全体fine-tune。損失{CE, KD(make_kd_loss(0.5,4,standardize=True)、
  teacherはfloat MLP ckptのロジット)}の2種。
- gate判定: stage-2 の学習曲線(step=stage1+stage2の合計)が基線 steps_to(0.90)=3744 を
  上回る(=より少ないstepで0.90到達)こと。少データ版(6kサンプル、基線0.8522)も1本。

使用例:
    # 煙実行(数十秒、wandb無効、少データ・1構成・少step)
    uv run python experiments/distill_layerwise_native_mnist.py --quick --no-wandb

    # スクリーニング(1seed、4構成すべて、CE+KD stage2)
    uv run python experiments/distill_layerwise_native_mnist.py --screen

    # ベスト構成のみ3seed(スクリーニング後に --config で固定)
    uv run python experiments/distill_layerwise_native_mnist.py \
        --config d3_m1 --seeds 42 43 44

    # 少データ(6kサンプル)
    uv run python experiments/distill_layerwise_native_mnist.py --config d3_m1 --train-n 6000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutLayer
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    max_fanin,
)
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.teacher_models import (
    HeavisideSTE,
    TeacherMLP,
    TeacherMLPSte,
)
from griffith_logic.datasets.vision import load_mnist

load_dotenv()

# ---- 基線(の実測値)----
BASE_STEPS_TO_90 = 3744    # scratch quick30 steps_to(0.90) 中央値
BASE_FINAL_ACC = 0.9088    # scratch quick30 最終 hard acc
BASE_LOWDATA_ACC = 0.8522  # scratch 少データ6k 最終 hard acc
BASE_GATES = 6000          # scratch quick30 総ゲート数(k=2000×3層)

TEACHER_HIDDEN = (300, 100)
NUM_CLASSES = 10
IN_WIDTH = 28 * 28

# ブロック構成グリッド(上限4構成): 深さ∈{2,3} × m∈{1,4}
CONFIG_GRID = {
    'd2_m1': {'depth': 2, 'm': 1},
    'd2_m4': {'depth': 2, 'm': 4},
    'd3_m1': {'depth': 3, 'm': 1},
    'd3_m4': {'depth': 3, 'm': 4},
}


# ============================================================
# データ
# ============================================================

def _flatten_binary(dataset, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """二値化済み MNIST を (N,784) float32 0/1 と (N,) int64 ラベルへ展開する。"""
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, IN_WIDTH), dtype=np.float32)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.float32)
        ys[i] = int(label)
    uniq = set(np.unique(xs).tolist())
    assert uniq.issubset({0.0, 1.0}), f'二値化MNISTのはずが0/1以外を検出: {uniq}'
    return xs, ys


def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ============================================================
# teacher(fan-in制約枝刈りSTE)の生成
# ============================================================

def _find_latest_float_ckpt(ckpt_dir: str) -> str | None:
    pattern = os.path.join(ckpt_dir, 'teacher_mlp_float_mnist_*', 'best.pt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _build_ste_from_float(ckpt_path: str, device: str) -> TeacherMLPSte:
    """teacher-float ckpt の Linear 重みで TeacherMLPSte を初期化(ReLU→STE切替)。

    convert_prune_transcribe_mnist.py と同一機構。事前学習の重み初期値+接続の
    相対強度(=どの接続を枝刈りで残すか)を継承する。
    """
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device='cpu')
    src = teacher.model  # TeacherMLP
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    ste = TeacherMLPSte(in_width=IN_WIDTH, hidden=TEACHER_HIDDEN,
                        num_classes=NUM_CLASSES)
    dst_linears = list(ste.linears)
    if len(src_linears) != len(dst_linears):
        raise ValueError(
            f'Linear層数不一致: float={len(src_linears)} ste={len(dst_linears)}'
        )
    with torch.no_grad():
        for s, d in zip(src_linears, dst_linears, strict=True):
            if s.weight.shape != d.weight.shape:
                raise ValueError(
                    f'Linear形状不一致: {s.weight.shape} vs {d.weight.shape}')
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
    return ste.to(device)


@torch.no_grad()
def _eval_ste_acc(model, x: np.ndarray, y: np.ndarray, device: str,
                  batch: int = 4000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def _make_ste_train_fn(x_tr, y_tr, device, lr, batch_size):
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    n = len(x_tr)
    loss_fn = nn.CrossEntropyLoss()

    def train_fn(model, epochs):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        logs = []
        for _ in range(epochs):
            perm = torch.randperm(n)
            ep_loss, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            logs.append(ep_loss / max(nb, 1))
        return {'epoch_losses': logs}

    return train_fn


def build_teacher(args, x_tr, y_tr, x_te, y_te, device) -> TeacherMLPSte:
    """fan-in6 STE teacher を convert_prune_transcribe_mnist.py の機構で自前生成。"""
    print(f'\n=== teacher: fan-in制約枝刈りSTE (fanins={args.teacher_fanins}, '
          f'epochs/stage={args.teacher_epochs}) ===')
    model = _build_ste_from_float(args.float_ckpt, device)
    init_acc = _eval_ste_acc(model, x_te, y_te, device)
    print(f'  STE切替直後 test_acc={init_acc:.4f}')
    sched = PruneSchedule(
        fanins=tuple(args.teacher_fanins),
        epochs_per_stage=args.teacher_epochs,
        lr=args.teacher_lr,
    )
    train_fn = _make_ste_train_fn(
        x_tr, y_tr, device, sched.lr, args.batch_size)
    hist = apply_fanin_pruning(model, sched, train_fn,
                               lambda m: _eval_ste_acc(m, x_te, y_te, device))
    for s in hist['stages']:
        print(f'  stage fanin={s["fanin"]:2d} '
              f'max_fanin_after={s["max_fanin_after"]:2d} acc={s["acc"]:.4f}')
    print(f'  teacher(fan-in={max_fanin(model)}) 最終 test_acc={hist["final_acc"]:.4f}')
    return model, hist['final_acc']


# ============================================================
# 隠れ二値活性のキャプチャ(forward hook、outputs/ キャッシュ)
# ============================================================

def _activation_cache_path(teacher_acc: float, train_n: int,
                           fanins: tuple[int, ...]) -> str:
    tag = 'x'.join(str(f) for f in fanins)
    return os.path.join(
        'outputs',
        f'd5_n3_teacher_acts_mnist_fanin{tag}_n{train_n}_acc{teacher_acc:.4f}.npz',
    )


@torch.no_grad()
def capture_hidden_bits(teacher: TeacherMLPSte, x_tr: np.ndarray, device: str,
                        batch: int = 4000) -> list[np.ndarray]:
    """teacher の各隠れ層の二値活性 y=1[Wx+b>0] を train全サンプル分キャプチャ。

    forward hook で TeacherMLPSte.linears[:-1] の出力に HeavisideSTE を適用した
    値(=隠れ層のブロック出力そのもの)を層ごとに収集する。返り値は層数分の
    (N, width) uint8 配列(300bit, 100bit)。
    """
    teacher.eval()
    captured: dict[int, list[np.ndarray]] = {}
    handles = []

    hidden_lins = list(teacher.linears[:-1])

    def make_hook(layer_i):
        def hook(_module, _inp, out):
            # out は Linear の生 preact。teacher forward と同じ二値化を適用。
            bits = (out > 0).to(torch.uint8).cpu().numpy()
            captured.setdefault(layer_i, []).append(bits)
        return hook

    for li, lin in enumerate(hidden_lins):
        handles.append(lin.register_forward_hook(make_hook(li)))

    try:
        for i in range(0, len(x_tr), batch):
            xb = torch.as_tensor(x_tr[i:i + batch], dtype=torch.float32,
                                 device=device)
            teacher(xb)
    finally:
        for h in handles:
            h.remove()

    layers = []
    for li in range(len(hidden_lins)):
        layers.append(np.concatenate(captured[li], axis=0))
    return layers


def get_or_capture_activations(teacher, x_tr, device, teacher_acc, args):
    """隠れ二値活性を outputs/ からロード、無ければキャプチャして保存。"""
    os.makedirs('outputs', exist_ok=True)
    path = _activation_cache_path(teacher_acc, len(x_tr),
                                  tuple(args.teacher_fanins))
    if os.path.exists(path) and not args.no_cache:
        data = np.load(path)
        layers = [data[k] for k in sorted(data.files)]
        print(f'\n=== 隠れ二値活性キャッシュ読込: {path} '
              f'({" ".join(str(a.shape) for a in layers)}) ===')
        return layers, path

    print('\n=== 隠れ二値活性キャプチャ(forward hook、train全サンプル) ===')
    t0 = time.time()
    layers = capture_hidden_bits(teacher, x_tr, device)
    # 教師ビットの発火率(定数化してないかの sanity)を表示
    for li, a in enumerate(layers):
        rate = float(a.mean())
        print(f'  layer{li}: {a.shape} 発火率={rate:.3f} ({a.nbytes/1e6:.1f}MB)')
    np.savez_compressed(path, **{f'layer{li}': a for li, a in enumerate(layers)})
    print(f'  保存: {path} ({time.time()-t0:.1f}s)')
    return layers, path


# ============================================================
# student: native DLGN ブロック
# ============================================================

class NativeBlock(nn.Module):
    """1つの teacher 層に対応する native DLGN ブロック(LutLayer N=2 softmax 積層)。

    depth 層の LutLayer を積む。最終層の out_dim = teacher層幅 × m。
    m>1 のときは stage1 で m個の出力のグループ平均を teacher ビット1つと BCE する
    (forward_bce)。stage2 では最終層の生 soft 出力(幅 teacher×m)を次段に渡す。

    入力幅は前段(前ブロック)の出力幅。DLGN 慣例に従い grad_factor は層数で調整。
    """

    def __init__(self, in_dim: int, teacher_width: int, depth: int, m: int,
                 device: str, grad_factor: float = 1.0):
        super().__init__()
        self.teacher_width = teacher_width
        self.m = m
        self.out_width = teacher_width * m
        self.depth = depth
        layers = []
        prev = in_dim
        for li in range(depth):
            out = self.out_width  # 全層を出力幅に揃える(素直な積層)
            layers.append(LutLayer(
                in_dim=prev, out_dim=out, n_inputs=2,
                device=device, parameterization='softmax',
                connections='random', grad_factor=grad_factor,
            ))
            prev = out
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lay in self.layers:
            x = lay(x)
        return x  # (B, teacher_width*m) soft [0,1]

    def bce_target_pred(self, block_out: torch.Tensor) -> torch.Tensor:
        """block出力[B, teacher_width*m] → BCE用の予測[B, teacher_width]。

        m=1: そのまま。m>1: m個の出力のグループ平均(GroupSum流の集約)。
        """
        if self.m == 1:
            return block_out
        b = block_out.shape[0]
        return block_out.reshape(b, self.teacher_width, self.m).mean(dim=-1)

    def n_gates(self) -> int:
        return sum(int(lay.out_dim) for lay in self.layers)


class LayerwiseStudent(nn.Module):
    """teacher層ごとの NativeBlock を直列接続し、末尾に GroupSum 頭を持つ student。

    ブロック間はブロック soft 出力をそのまま次段入力にする(値域[0,1]で DLGN 互換)。
    stage2 では最終ブロック出力に GroupSum(k=NUM_CLASSES) を掛けてロジット化する。
    GroupSum が割り切れるよう、最終ブロックの出力幅は NUM_CLASSES の倍数である必要が
    ある(300/100 も 1200/400 も 10 で割り切れる)。
    """

    # 基線 quick30 の GroupSum は幅2000/tau10 = 200(1クラスあたりの実効ロジット
    # スケール)。tau=None のとき、最終ブロック幅から同じ比になるよう tau を自動設定
    # (=幅/200)し、logitスケールを基線と揃える(公平な比較)。
    BASELINE_WIDTH_PER_TAU = 200.0

    def __init__(self, teacher_widths: tuple[int, ...], depth: int, m: int,
                 device: str, tau: float | None, in_dim: int = IN_WIDTH):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.m = m
        prev = in_dim
        total_depth = depth * len(teacher_widths) + 1  # +1: 頭
        gf = (1.0 / total_depth) ** 0.5 if total_depth > 0 else 1.0
        for tw in teacher_widths:
            blk = NativeBlock(prev, tw, depth, m, device, grad_factor=max(gf, 0.1))
            self.blocks.append(blk)
            prev = blk.out_width
        # GroupSum 頭: 最終ブロック出力幅 prev を NUM_CLASSES クラスへ集約
        if prev % NUM_CLASSES != 0:
            raise ValueError(
                f'最終ブロック出力幅 {prev} が {NUM_CLASSES} で割り切れない')
        if tau is None:
            tau = max(prev / self.BASELINE_WIDTH_PER_TAU, 1e-3)
        self.tau = tau
        self.head = GroupSum(k=NUM_CLASSES, tau=tau)

    def block_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """全ブロックの soft 出力を順に返す(stage1 の per-layer BCE 用)。"""
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)  # (B, NUM_CLASSES) ロジット

    def n_gates(self) -> int:
        """GroupSum はゲートを持たない(集約のみ)。LutLayer 出力数の総和。"""
        return sum(blk.n_gates() for blk in self.blocks)


# ============================================================
# 学習(Stage 1: 層別BCE / Stage 2: CE or KD)
# ============================================================

@torch.no_grad()
def eval_hard_acc(student: LayerwiseStudent, x: np.ndarray, y: np.ndarray,
                  device: str, batch: int = 4000) -> float:
    """student の hard(eval モード=離散化 forward)accuracy。"""
    student.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = student(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def stage1_bce(student, x_tr, teacher_bits, device, args, step_state,
               arm_prefix, use_wandb):
    """Stage 1: 全ブロックの per-layer BCE 和を joint 最適化。

    進捗が停滞(初期比の相対低下が閾値未満)したら層凍結カリキュラムにフォールバック。
    返り値: {'mode': 'joint'|'freeze', 'final_bce': [...], 'used_freeze': bool}。
    """
    n = len(x_tr)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    # teacher_bits: list of (N, width) uint8 → float target
    tgt_t = [torch.as_tensor(b, dtype=torch.float32) for b in teacher_bits]
    bce = nn.BCELoss()
    n_blocks = len(student.blocks)

    def block_bce_losses(xb, idx):
        student.train()
        outs = student.block_outputs(xb)
        losses = []
        for bi, out in enumerate(outs):
            pred = student.blocks[bi].bce_target_pred(out).clamp(1e-6, 1 - 1e-6)
            losses.append(bce(pred, tgt_t[bi][idx].to(device)))
        return losses

    def run_epochs(epochs, active_blocks, opt):
        """active_blocks の BCE のみを最適化(joint=全部, freeze=段階)。"""
        last = [0.0] * n_blocks
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                xb = x_t[idx].to(device)
                losses = block_bce_losses(xb, idx)
                loss = sum(losses[bi] for bi in active_blocks)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step_state['step'] += 1
                for bi in range(n_blocks):
                    last[bi] = float(losses[bi].detach())
                if use_wandb and step_state['step'] % 100 == 0:
                    log_metrics({
                        f'{arm_prefix}/stage1_bce_total': float(loss.detach()),
                        **{f'{arm_prefix}/stage1_bce_b{bi}': last[bi]
                           for bi in range(n_blocks)},
                    }, step=step_state['step'])
        return last

    # --- joint 最適化 ---
    opt = torch.optim.Adam(student.parameters(), lr=args.stage1_lr)
    # 初期BCEを記録
    with torch.no_grad():
        student.train()
        xb0 = x_t[:args.batch_size].to(device)
        init_losses = block_bce_losses(xb0, torch.arange(min(args.batch_size, n)))
        init_bce = [float(v.detach()) for v in init_losses]
    last = run_epochs(args.stage1_epochs, list(range(n_blocks)), opt)
    print(f'  [stage1 joint] init_bce={[f"{v:.3f}" for v in init_bce]} '
          f'-> final_bce={[f"{v:.3f}" for v in last]}')

    # 停滞判定: どのブロックも初期比 rel_drop 未満なら凍結カリキュラムへ
    rel_drops = [(init_bce[bi] - last[bi]) / max(init_bce[bi], 1e-6)
                 for bi in range(n_blocks)]
    stalled = all(rd < args.stage1_stall_rel for rd in rel_drops)
    used_freeze = False
    if stalled and not args.no_freeze_fallback:
        used_freeze = True
        print(f'  [stage1] joint 停滞(rel_drops={[f"{r:.2f}" for r in rel_drops]}) '
              f'→ 層凍結カリキュラムにフォールバック')
        # ブロックを1つずつ順に学習(前段は凍結)
        opt = torch.optim.Adam(student.parameters(), lr=args.stage1_lr)
        for bi in range(n_blocks):
            last = run_epochs(args.stage1_epochs, [bi], opt)
            print(f'    [freeze] block{bi} 学習後 bce={last[bi]:.3f}')

    return {'mode': 'freeze' if used_freeze else 'joint',
            'init_bce': init_bce, 'final_bce': last, 'used_freeze': used_freeze,
            'rel_drops': rel_drops}


def stage2_finetune(student, x_tr, y_tr, x_te, y_te, teacher_logits, device,
                    args, step_state, arm_prefix, loss_kind, use_wandb):
    """Stage 2: GroupSum頭を接続し全体fine-tune。損失は CE または KD。

    step は stage1+stage2 の通し(step_state を継続)。500step毎(+末尾)に
    hard acc を評価し wandb へ。steps_to(0.90) を返す。
    """
    n = len(x_tr)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    kd_loss_fn = None
    if loss_kind == 'kd':
        if teacher_logits is None:
            raise ValueError('KD には teacher_logits が必要')
        tl_t = torch.as_tensor(teacher_logits, dtype=torch.float32)
        # 指定: make_kd_loss(0.5, 4, standardize=True)
        kd_loss_fn = make_kd_loss(0.5, 4.0, standardize=True)
    ce = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(student.parameters(), lr=args.stage2_lr)
    steps_to_90 = None
    eval_interval = args.eval_interval

    def do_eval():
        nonlocal steps_to_90
        acc = eval_hard_acc(student, x_te, y_te, device)
        if use_wandb:
            log_metrics({f'{arm_prefix}/hard_acc': acc}, step=step_state['step'])
        if steps_to_90 is None and acc >= 0.90:
            steps_to_90 = step_state['step']
        return acc

    start_step = step_state['step']
    best_acc = do_eval()  # stage2 開始時点
    for ep in range(args.stage2_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            # 各iter先頭でtrainモードへ(mid-epochのdo_evalがevalモードにするため)。
            student.train()
            idx = perm[i:i + args.batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            logits = student(xb)
            if loss_kind == 'kd':
                loss = kd_loss_fn(logits, yb, tl_t[idx].to(device))
            else:
                loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_state['step'] += 1
            if use_wandb and step_state['step'] % 100 == 0:
                log_metrics({f'{arm_prefix}/stage2_loss': float(loss.detach())},
                            step=step_state['step'])
            # 500step毎 hard acc(runner の do_eval 相当)
            if (step_state['step'] - start_step) % eval_interval == 0:
                acc = do_eval()
                best_acc = max(best_acc, acc)
        acc = do_eval()
        best_acc = max(best_acc, acc)
        print(f'    [stage2/{loss_kind}] ep{ep+1}/{args.stage2_epochs} '
              f'step={step_state["step"]} hard_acc={acc:.4f} '
              f'best={best_acc:.4f} steps_to90={steps_to_90}')
    return {'steps_to_90': steps_to_90, 'final_acc': best_acc,
            'total_steps': step_state['step']}


# ============================================================
# 1構成×1seed の実行
# ============================================================

def run_config(cfg_name, cfg, seed, x_tr, y_tr, x_te, y_te, teacher_bits,
               teacher_logits, teacher_acc, args, device):
    """1つの (構成, seed) を stage1→stage2(CE と KD の2損失)で回す。"""
    depth, m = cfg['depth'], cfg['m']
    print(f'\n{"="*60}\n構成 {cfg_name} (depth={depth}, m={m}) seed={seed}\n{"="*60}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {}
    for loss_kind in args.stage2_losses:
        torch.manual_seed(seed)  # 各損失で同一初期化から
        np.random.seed(seed)
        student = LayerwiseStudent(
            teacher_widths=TEACHER_HIDDEN, depth=depth, m=m,
            device=device, tau=args.tau,
        ).to(device)
        n_gates = student.n_gates()
        arm_prefix = f'{cfg_name}_s{seed}_{loss_kind}'

        use_wandb = init_wandb(
            run_name=f'd5_n3_{cfg_name}_{loss_kind}_s{seed}',
            config_dict={
                'config': cfg_name, 'depth': depth, 'm': m, 'seed': seed,
                'stage2_loss': loss_kind, 'n_gates': n_gates,
                'teacher_widths': list(TEACHER_HIDDEN),
                'teacher_acc': teacher_acc, 'tau': student.tau,
                'stage1_epochs': args.stage1_epochs,
                'stage2_epochs': args.stage2_epochs,
                'stage1_lr': args.stage1_lr, 'stage2_lr': args.stage2_lr,
                'train_n': len(x_tr), 'dataset': 'mnist-binarized',
                'base_steps_to_90': BASE_STEPS_TO_90,
            },
            model=student, device=device,
            tags=['D5-4', 'N3', cfg_name, loss_kind,
                  'lowdata' if len(x_tr) < 60000 else 'fulldata'],
            group='convert-mnist-n3',
        )

        step_state = {'step': 0}
        t0 = time.time()
        s1 = stage1_bce(student, x_tr, teacher_bits, device, args,
                        step_state, arm_prefix, use_wandb)
        s2 = stage2_finetune(student, x_tr, y_tr, x_te, y_te, teacher_logits,
                             device, args, step_state, arm_prefix, loss_kind,
                             use_wandb)
        dt = time.time() - t0

        eff = (BASE_STEPS_TO_90 / s2['steps_to_90']
               if s2['steps_to_90'] else None)
        run_id = os.environ.get('WANDB_RUN_ID', '')
        if use_wandb:
            log_metrics({
                f'{arm_prefix}/steps_to_90': s2['steps_to_90'] or -1,
                f'{arm_prefix}/final_hard_acc': s2['final_acc'],
                f'{arm_prefix}/efficiency_ratio': eff or 0.0,
                f'{arm_prefix}/n_gates': n_gates,
                f'{arm_prefix}/stage1_used_freeze': int(s1['used_freeze']),
            }, step=step_state['step'])
            try:
                import wandb
                run_id = wandb.run.id if wandb.run else run_id
            except Exception:
                pass
            wandb_finish()

        print(f'  → {loss_kind}: steps_to90={s2["steps_to_90"]} '
              f'効率比={eff if eff else "N/A"} '
              f'最終acc={s2["final_acc"]:.4f} ゲート={n_gates} '
              f'stage1={s1["mode"]} ({dt:.0f}s) run_id={run_id}')

        results[loss_kind] = {
            'stage1': s1, 'stage2': s2, 'n_gates': n_gates,
            'efficiency_ratio': eff, 'wandb_run_id': run_id,
            'seconds': round(dt, 1),
        }
    return results


# ============================================================
# 集計
# ============================================================

def summarize(all_results: dict, teacher_acc: float, train_n: int) -> None:
    print('\n' + '=' * 88)
    print(f'層別二値蒸留結果表 (teacher fan-in6 acc={teacher_acc:.4f}, '
          f'train_n={train_n})')
    print('=' * 88)
    base = BASE_LOWDATA_ACC if train_n < 60000 else BASE_FINAL_ACC
    print(f'基線: steps_to(0.90)={BASE_STEPS_TO_90} 最終acc={base:.4f} '
          f'({"少データ6k" if train_n < 60000 else "quick30"}) '
          f'総ゲート={BASE_GATES}')
    hdr = ('| 構成 | 深さ | m | stage2損失 | stage1収束 | steps_to(0.90) | '
           '効率比 | 最終acc | ゲート数 | WANDB_RUN_ID |')
    print(hdr)
    print('|' + '---|' * 10)
    for cfg_name, per_seed in all_results.items():
        for seed, per_loss in per_seed.items():
            for lk, r in per_loss.items():
                cfg = CONFIG_GRID[cfg_name]
                s1 = r['stage1']
                s1str = (f'{s1["mode"]} '
                         f'{[f"{v:.2f}" for v in s1["final_bce"]]}')
                st90 = r['stage2']['steps_to_90']
                eff = r['efficiency_ratio']
                gate_flag = ''
                if st90 is not None:
                    gate_flag = ' PASS' if st90 < BASE_STEPS_TO_90 else ' FAIL'
                print(f'| {cfg_name} | {cfg["depth"]} | {cfg["m"]} | {lk} | '
                      f'{s1str} | {st90}{gate_flag} | '
                      f'{f"{eff:.2f}x" if eff else "N/A"} | '
                      f'{r["stage2"]["final_acc"]:.4f} | {r["n_gates"]} | '
                      f'{r["wandb_run_id"]} |')

    print(f'\n--- gate判定(stage2 steps_to(0.90) < 基線{BASE_STEPS_TO_90}) ---')
    best = None
    for cfg_name, per_seed in all_results.items():
        for seed, per_loss in per_seed.items():
            for lk, r in per_loss.items():
                st90 = r['stage2']['steps_to_90']
                if st90 is not None and (best is None or st90 < best[1]):
                    best = (f'{cfg_name}/{lk}/s{seed}', st90,
                            r['efficiency_ratio'], r['stage2']['final_acc'])
    if best:
        verdict = 'PASS' if best[1] < BASE_STEPS_TO_90 else 'FAIL'
        print(f'最良: {best[0]} steps_to(0.90)={best[1]} '
              f'効率比={best[2]:.2f}x 最終acc={best[3]:.4f} → {verdict}')
    else:
        print('どの構成も 0.90 未到達(gate FAIL)')


# ============================================================
# メイン
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description='N3 層別二値蒸留 → native DLGN')
    p.add_argument('--float-ckpt', type=str, default=None,
                   help='teacher初期化元の teacher-float ckpt(省略で最新自動発見)')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--teacher-fanins', type=int, nargs='+', default=[32, 12, 6])
    p.add_argument('--teacher-epochs', type=int, default=2)
    p.add_argument('--teacher-lr', type=float, default=1e-3)
    # student
    p.add_argument('--config', type=str, default=None,
                   help=f'単一構成名({"/".join(CONFIG_GRID)})。省略時は --screen が必要')
    p.add_argument('--screen', action='store_true',
                   help='全4構成を1seedでスクリーニング')
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='実行seed(スクリーニングは先頭のみ、--config指定で全seed)')
    p.add_argument('--stage2-losses', type=str, nargs='+', default=['ce', 'kd'],
                   choices=['ce', 'kd'])
    p.add_argument('--tau', type=float, default=None,
                   help='GroupSum温度。省略で最終ブロック幅から自動設定'
                        '(=幅/200、基線quick30のロジットスケールに整合)')
    p.add_argument('--stage1-epochs', type=int, default=3)
    p.add_argument('--stage2-epochs', type=int, default=30)
    p.add_argument('--stage1-lr', type=float, default=0.01)
    p.add_argument('--stage2-lr', type=float, default=0.01)
    p.add_argument('--stage1-stall-rel', type=float, default=0.05,
                   help='joint停滞判定: 全ブロックの初期比BCE低下がこの値未満で凍結へ')
    p.add_argument('--no-freeze-fallback', action='store_true')
    p.add_argument('-bs', '--batch-size', type=int, default=128)
    p.add_argument('--eval-interval', type=int, default=500)
    p.add_argument('--train-n', type=int, default=None,
                   help='train サンプル数(少データ版は6000)。省略で全60k')
    p.add_argument('--eval-n', type=int, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--no-cache', action='store_true', help='活性キャッシュを使わず再計算')
    p.add_argument('--quick', action='store_true',
                   help='煙実行(少データ・1構成・少step・CEのみ)')
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.float_ckpt is None:
        args.float_ckpt = _find_latest_float_ckpt(args.checkpoint_dir)
        if args.float_ckpt is None:
            print('!! teacher-float ckpt が見つからない(--float-ckpt を指定)')
            sys.exit(1)

    if args.quick:
        args.train_n = args.train_n or 4000
        args.eval_n = args.eval_n or 2000
        args.teacher_fanins = [6]
        args.teacher_epochs = 1
        args.stage1_epochs = 1
        args.stage2_epochs = 2
        args.stage2_losses = ['ce']
        args.config = args.config or 'd2_m1'

    # 構成の決定
    if args.screen:
        configs = list(CONFIG_GRID.keys())
        seeds = [args.seeds[0]]
    elif args.config:
        if args.config not in CONFIG_GRID:
            print(f'!! 未知の構成 {args.config}(有効: {list(CONFIG_GRID)})')
            sys.exit(1)
        configs = [args.config]
        seeds = args.seeds
    else:
        print('!! --screen か --config のいずれかが必要')
        sys.exit(1)

    print(f'デバイス: {device}  teacher初期化元: {args.float_ckpt}')
    print(f'構成: {configs}  seeds: {seeds}  stage2損失: {args.stage2_losses}')

    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])

    print('MNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    x_tr, y_tr = _flatten_binary(train_set, n=args.train_n)
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n)
    print(f'train={len(x_tr)} test={len(x_te)}')

    started = time.time()

    # 1) teacher(fan-in6 STE)生成
    teacher, teacher_acc = build_teacher(args, x_tr, y_tr, x_te, y_te, device)

    # 2) 隠れ二値活性キャプチャ(outputs/ キャッシュ)
    teacher_bits, act_path = get_or_capture_activations(
        teacher, x_tr, device, teacher_acc, args)

    # 3) teacher float ロジット(KD用): float ckpt の生 MLP ロジットを train 分キャッシュ
    teacher_logits = None
    if 'kd' in args.stage2_losses:
        print('\n=== teacher float ロジット(KD用)計算 ===')
        float_teacher = resolve_teacher(f'ckpt:{args.float_ckpt}', device=device)
        fm = float_teacher.model.to(device).eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(x_tr), 4000):
                xb = torch.as_tensor(x_tr[i:i + 4000], dtype=torch.float32,
                                     device=device)
                logits_list.append(fm(xb).cpu().numpy())
        teacher_logits = np.concatenate(logits_list, axis=0).astype(np.float32)
        print(f'  teacher_logits {teacher_logits.shape}')

    # 4) 各構成×seed を実行
    all_results: dict = {}
    for cfg_name in configs:
        all_results[cfg_name] = {}
        for seed in seeds:
            all_results[cfg_name][seed] = run_config(
                cfg_name, CONFIG_GRID[cfg_name], seed,
                x_tr, y_tr, x_te, y_te, teacher_bits, teacher_logits,
                teacher_acc, args, device,
            )

    summarize(all_results, teacher_acc, len(x_tr))

    print(f'\ncaptured活性: {act_path}')
    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
