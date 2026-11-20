"""構造転写(wiring+ゲート初期化)+soft磨き(本命) vs IO-KDのみ/scratch/RKD補助損失
(パイロット実験)。

ROADMAP-v5 Track D の変換パラダイム比較を、整数和分類という小さくクリーンな
testbedで検証する。`transcribe_mlp`/`recover` は分類専用(group_sum終端+
simulate(...).argmax でhard acc測定)なので、加算タスクを**整数和クラス分類**
として組む(多ビットBCEではない)。tests/l4_learning/test_toy_problems.py の
test_full_adder_as_4class と同じ規約(ラベル=a+bの整数値)を n bit へ一般化する。

入力: 2n bit。LSB-firstで a をビット0..n-1、b をビットn..2n-1に連結する
(datasets/logic.py の nbit_adder と同じ規約)。全 2^(2n) パターンを列挙する
(記憶で構わない。ここで測るのは汎化でなくhard回路忠実度)。
ラベル: 整数 a+b ∈ [0, 2^(n+1)-2]、クラス数 K = 2^(n+1)-1。GroupSum(k=K)で集計。

比較する5アーム(arm構成を刷新。旧arm C/C'の主従を入れ替えた):
    A(scratch)   ランダム初期化DLGN(LutLayer列+GroupSum)をラベルのみCEでBP学習。
                 下限の対照。
    B(IO-KDのみ) 同じランダム初期化DLGNを、teacherロジットKD(+CE)混合で学習。
                 構造転写なし(配線もゲート初期値も無関係)。
    C(本命: 構造転写+soft磨き) 小さなSTE-MLP teacherをタスクで学習(暗記させる)
                 → fan-in枝刈り(改善(1): 倍々刻みの多段スケジュールで
                 teacher-after-prune精度を保つ) → transcribe_mlp でLUT網IRへ
                 training-free転写 → recover(parameterization='warp',
                 selection=--polish-selection既定'soft')で磨く。実測
                 (0.855→0.225 for hard_st、0.855→0.888 for soft)により、
                 「良い離散化初期値からの磨きはsoftが正しい」がheadline手法。
    C_hard_st(アブレーション) Cと同じ転写元IRだが selection='hard_st' で磨く。
                 「良い初期値に対してhard_stは破壊的」を示す対照
                 (旧は逆にこちらが本命(C)と誤って位置づけられていた)。
    D(本命+RKD補助損失) Cと同じsoft磨きに、teacherロジットとの関係蒸留
                 (RKD、convert/relational.py)を補助項として追加する
                 (--aux rkd)。distillation-beyond-KDの主賭け。

oracle参照: recoverが返す hard_acc_before(=磨き前、転写直後の関数保存回路の
hard acc)とteacher自身のhard accを天井として報告する(別途手組み回路アームは
用意しない、という指示通り)。oracle_reach = C.hard_acc_after / teacher_acc。

厳密性assert: transcribe_mlp直後のIRについて、simulate(...).argmax がteacher
のtorch eval forward argmaxと全パターンで一致することを確認する(不一致は
transcribe/recoverパイプラインのバグなので緩めずSTOPする)。--faninに複数値を
渡すと各fanin構成で独立にこの厳密性assertを行う(1つでも破れたら即STOP)。

使用例:
    # 煙実行(数十秒、wandb無効、1 seed・小width・少epoch)
    uv run python experiments/convert_arith_structural_transfer.py --quick --no-wandb

    # 本実行(wandb online、既定nbits=4、3 seeds)
    uv run python experiments/convert_arith_structural_transfer.py

    # fan-inスイープ(oracle_reach対fan-inのトレードオフを見る。
    # LutLayer制約によりfan-inは2/4/6のみ対応、6が上限=2^6エントリLUT)
    uv run python experiments/convert_arith_structural_transfer.py --fanin 2 4 6 --no-wandb

    # スケール確認 — より大きな加算(n=6)
    uv run python experiments/convert_arith_structural_transfer.py \
        --task adder --nbits 6 --fanin 6 --seeds 3 --no-wandb

    # スケール確認 — 構造的に難しいタスク(乗算、n=3/4)
    uv run python experiments/convert_arith_structural_transfer.py \
        --task multiplier --nbits 4 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

JST = timezone(timedelta(hours=9))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutLayer
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.prune import PruneSchedule, apply_fanin_pruning
from griffith_logic.convert.recover import RecoverConfig, recover
from griffith_logic.convert.relational import rkd_loss
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import (
    TranscribeSpec,
    _quantize_final,
    transcribe_mlp,
)
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

load_dotenv()

EPS = 1e-8


# ============================================================
# データ: 整数和分類(nbit_adderの分類版)
# ============================================================

TASK_CHOICES = ('adder', 'multiplier')


def _combine(a: int, b: int, task: str) -> int:
    """アーム共通のオペランド結合則(タスク切替の単一情報源)。"""
    if task == 'adder':
        return a + b
    if task == 'multiplier':
        return a * b
    raise ValueError(f'未対応のtask: {task}(対応: {TASK_CHOICES})')


def arith_classify_dataset(n: int, task: str = 'adder') -> tuple[np.ndarray, np.ndarray]:
    """n-bit算術(加算/乗算)の全パターンを整数クラス分類データとして列挙する。

    入力2n bit、LSB-firstで a=ビット0..n-1、b=ビットn..2n-1
    (datasets/logic.py の nbit_adder/nbit_multiplier と同じ規約)。
    ラベルは整数 a+b(task='adder')または a*b(task='multiplier')。

    Args:
        n: オペランドのビット幅(n>=1)
        task: 'adder'(ラベル=a+b、既定)または'multiplier'(ラベル=a*b)

    Returns:
        (X, y): X (2^(2n), 2n) int64 0/1、y (2^(2n),) int64 クラスラベル
    """
    if n < 1:
        raise ValueError(f'nは1以上: {n}')
    if task not in TASK_CHOICES:
        raise ValueError(f'未対応のtask: {task}(対応: {TASK_CHOICES})')
    width = 2 * n
    n_patterns = 1 << width
    x = np.empty((n_patterns, width), dtype=np.int64)
    y = np.empty((n_patterns,), dtype=np.int64)
    for k in range(n_patterns):
        bits = [(k >> j) & 1 for j in range(width)]
        a = sum(bits[j] << j for j in range(n))
        b = sum(bits[n + j] << j for j in range(n))
        x[k] = bits
        y[k] = _combine(a, b, task)
    # 素朴照合: Pythonの整数演算と一致すること(参照実装同様、素朴さを優先)
    for k in (0, 1, n_patterns - 1, n_patterns // 2):
        bits = x[k].tolist()
        a = sum(bits[j] << j for j in range(n))
        b = sum(bits[n + j] << j for j in range(n))
        assert y[k] == _combine(a, b, task), f'arith_classify_dataset: k={k} で不一致'
    return x, y


def num_classes_for(n: int, task: str = 'adder') -> int:
    """n-bit算術の整数クラス数。

    adder: K = 2^(n+1) - 1(a+bの最大値は2(2^n-1))。
    multiplier: K = (2^n-1)^2 + 1(a*bの最大値は(2^n-1)^2、0含む)。
    """
    if task == 'adder':
        return (1 << (n + 1)) - 1
    if task == 'multiplier':
        return ((1 << n) - 1) ** 2 + 1
    raise ValueError(f'未対応のtask: {task}(対応: {TASK_CHOICES})')


# ============================================================
# モデル: DLGN(LutLayer列 + GroupSum) — arm A/B
# ============================================================

def build_dlgn(
    in_width: int, k_classes: int, *, width: int, depth: int,
    n_inputs: int, group_width: int, device: str,
    parameterization: str = 'softmax', seed: int | None = None,
) -> nn.Sequential:
    """ランダム初期化DLGN(LutLayer列+GroupSum)を構築する(arm A/B用)。"""
    if seed is not None:
        torch.manual_seed(seed)
    out_final = k_classes * group_width
    layers: list[nn.Module] = []
    prev = in_width
    for i in range(depth):
        nxt = out_final if i == depth - 1 else width
        layers.append(LutLayer(
            prev, nxt, n_inputs=n_inputs, device=device,
            parameterization=parameterization, connections='random',
        ))
        prev = nxt
    layers.append(GroupSum(k=k_classes, tau=1.0))
    return nn.Sequential(*layers).to(device)


# ============================================================
# teacher: 小さなSTE-MLP(構造転写元、arm C/C_hard_st/D)
# ============================================================

class ArithTeacherMlp(nn.Module):
    """小さなSTE隠れ活性MLP(transcribe_mlp対応: Linear列+HeavisideSTE)。

    転写器(transcribe_mlp)はnn.Linear列をchildren順に走査するため、Linearは
    self.linears(ModuleList)に1度だけ登録する(teacher_models.TeacherMLPSteと
    同じ規約)。隠れ活性はy=1[Wx+b>0](transcribe_mlpの隠れ層意味論と厳密一致)。
    最終層は二値化しない(argmax保存のスコア)。
    """

    def __init__(self, in_width: int, hidden: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        prev = in_width
        for h in hidden:
            self.linears.append(nn.Linear(prev, h))
            prev = h
        self.linears.append(nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        for lin in self.linears[:-1]:
            h = HeavisideSTE.apply(lin(h))
        return self.linears[-1](h)


def train_teacher(
    teacher: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
    epochs: int, lr: float, device: str, batch_size: int = 256,
) -> float:
    """teacherを暗記させる(全パターン列挙なので記憶=正解)。最終train hard accを返す。"""
    teacher.to(device).train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = teacher(xb.float())
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
    teacher.eval()
    with torch.no_grad():
        pred = teacher(x.to(device).float()).argmax(-1).cpu()
    return float((pred == y).float().mean())


# ============================================================
# 評価ヘルパ
# ============================================================

@torch.no_grad()
def soft_hard_acc_dlgn(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """DLGN(nn.Sequential)のsoft-forward argmax精度(hard評価はexport_ir経由が必要なので
    ここではsoft accのみ。arm A/Bのhard accはexport_ir+simulateで別途測る)。"""
    model.eval()
    out = model(x.to(device).float())
    pred = out.argmax(-1).cpu()
    return float((pred == y).float().mean())


def hard_acc_via_export(model: nn.Module, x_np: np.ndarray, y_np: np.ndarray,
                        in_width: int) -> float:
    """DLGN(eval)をexport_irでIR化し、simulateのargmaxでhard accを測る(arm A/B用)。"""
    from griffith_logic.export import export_ir
    model.eval()
    ir = export_ir(model, input_width=in_width, name='arith.dlgn')
    scores = simulate(ir, x_np)
    pred = scores.argmax(axis=-1)
    return float((pred == y_np).mean())


def make_teacher_logits(teacher: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    teacher.eval()
    with torch.no_grad():
        return teacher(x.to(device).float()).detach().cpu()


# ============================================================
# arm A: scratch(ラベルのみCE)
# ============================================================

def run_arm_a(x_tr, y_tr, x_ev_np, y_ev_np, *, in_width, k_classes, args, device, seed):
    model = build_dlgn(
        in_width, k_classes, width=args.width, depth=args.depth,
        n_inputs=args.n_inputs, group_width=args.group_width,
        device=device, parameterization='softmax', seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    soft_acc = soft_hard_acc_dlgn(model, x_tr, y_tr, device)
    hard_acc = hard_acc_via_export(model, x_ev_np, y_ev_np, in_width)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


# ============================================================
# arm B: IO-KDのみ(構造転写なし)
# ============================================================

def run_arm_b(x_tr, y_tr, teacher_logits, x_ev_np, y_ev_np, *, in_width,
              k_classes, args, device, seed):
    model = build_dlgn(
        in_width, k_classes, width=args.width, depth=args.depth,
        n_inputs=args.n_inputs, group_width=args.group_width,
        device=device, parameterization='softmax', seed=seed,
    )
    loss_fn = make_kd_loss(
        alpha=args.alpha, temperature=args.temperature,
        standardize=args.standardize, kind='kl',
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = TensorDataset(x_tr, y_tr, teacher_logits)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device).float(), yb.to(device), tb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb, tb)
            loss.backward()
            opt.step()
    soft_acc = soft_hard_acc_dlgn(model, x_tr, y_tr, device)
    hard_acc = hard_acc_via_export(model, x_ev_np, y_ev_np, in_width)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


# ============================================================
# arm C / C_hard_st / D: 構造転写(transcribe) + recover(soft / hard_st / soft+RKD)
# ============================================================

def gentle_fanin_schedule(max_in: int, n_inputs: int) -> tuple[int, ...]:
    """max_inからn_inputsまでの**倍々刻み**fan-in段階列を作る(改善(1))。

    初版は{max_in, (max_in+n_inputs)//2, n_inputs}の3点(実質1段の
    急激な枝刈り)で teacher hard_acc を 1.0→0.85 へ落としていた
    (初版の限界)。
    convert_prune_transcribe_mnist.py の (32, 12, 6) 相当の「緩やかな
    複数段」に倣い、各段でfan-inをおよそ半分にしてfine-tuneで再適応する
    時間を与える(反復magnitude枝刈りの標準作法: 一度に大きく削ると
    再学習で拾いきれない)。

    Args:
        max_in: teacher隠れ層の現在の最大fan-in(枝刈り前)。
        n_inputs: 最終目標fan-in(transcribe_mlp対応の2/4/6)。

    Returns:
        降順のfan-in段階タプル(先頭=max_in、末尾=n_inputs)。
        max_in<=n_inputsなら(n_inputs,)の1要素(呼び出し側でスキップ判定に使う)。
    """
    if max_in <= n_inputs:
        return (n_inputs,)
    stages = [max_in]
    cur = max_in
    while cur > n_inputs:
        nxt = max(n_inputs, cur // 2)
        if nxt == cur:  # 無限ループ防止(理論上max_in>n_inputsなら発生しない)
            nxt = n_inputs
        stages.append(nxt)
        cur = nxt
    return tuple(stages)


def ensure_teacher_fanin(
    teacher: nn.Module, x_all: np.ndarray, y_all: np.ndarray, *,
    n_inputs: int, epochs_per_stage: int, lr: float, device: str,
) -> tuple[nn.Module, dict]:
    """teacherの隠れ層in_featuresがn_inputsを超える場合のみfan-in枝刈りする。

    transcribe_mlp の sparsify_fanin は各ニューロンの|W|上位n_inputs本だけを
    残し、落とした接続の寄与をバイアス再較正で近似する(実証済みの
    近似誤差源)。teacher の隠れ層 in_features が n_inputs 以下ならこの近似は
    一切発生しない(全接続がそのまま残る)ため、その場合は何もしない。
    in_features > n_inputs のときのみ PruneSchedule で段階的に厳密n_inputs本へ
    枝刈り+re-fit し、sparsify_fanin の近似誤差をゼロにする
    (convert/prune.pyのdocstring: 枝刈り済み重みは非ゼロが正確にn_inputs本な
    ので、転写のtop-k選択は残存接続と一致し、fan-in疎化誤差ゼロ=厳密になる)。

    改善(1)(限界節への対応): 段階を{max_in, mid, n_inputs}の3点から
    gentle_fanin_schedule の倍々刻みへ変更し、各段のfine-tuneエポック数
    (epochs_per_stage、既定を100→300に引き上げ)も増やして、段階間で
    精度が再適応する時間を確保する(oracle_reach = 構造転写hard_acc /
    teacher-after-prune hard_acc を引き上げるのが目的)。

    Returns:
        (teacher, stage_history): stage_historyは
        apply_fanin_pruning の history dict(空枝刈りの場合は
        {'stages': [], 'skipped': True})。
    """
    hidden_linears = [
        m for m in teacher.modules() if isinstance(m, nn.Linear)
    ][:-1]
    max_in = max((lin.in_features for lin in hidden_linears), default=0)
    if max_in <= n_inputs:
        return teacher, {'stages': [], 'skipped': True}

    x_t = torch.as_tensor(x_all, dtype=torch.float32)
    y_t = torch.as_tensor(y_all, dtype=torch.int64)
    loss_fn = nn.CrossEntropyLoss()

    def train_fn(model, n_epochs):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_epochs):
            xb, yb = x_t.to(device), y_t.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        return {}

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            pred = model(x_t.to(device)).argmax(-1).cpu()
        return float((pred == y_t).float().mean())

    fanins = gentle_fanin_schedule(max_in, n_inputs)
    sched = PruneSchedule(fanins=fanins, epochs_per_stage=epochs_per_stage, lr=lr)
    history = apply_fanin_pruning(teacher, sched, train_fn, eval_fn)
    history['skipped'] = False
    return teacher, history


def build_transcribed_ir(teacher: nn.Module, calib_x_np: np.ndarray, *,
                         n_inputs: int, weight_bits: int, device: str):
    """teacherをtranscribe_mlpでIR化し、厳密性assertを行う。

    呼び出し前に ensure_teacher_fanin で teacher の隠れ層fan-inが n_inputs
    以下になっていること(そうでないと sparsify_fanin の近似で厳密転写に
    ならない)。
    """
    teacher_cpu = teacher.to('cpu').eval()
    spec = TranscribeSpec(n_inputs=n_inputs, weight_bits=weight_bits, isf=False)
    ir = transcribe_mlp(teacher_cpu, calib_x_np, spec)
    errs = validate(ir)
    if errs:
        raise RuntimeError(f'転写IR検証失敗: {errs[:3]}')
    return ir, teacher_cpu


def _quantized_forward_pred(teacher_cpu: nn.Module, x_np: np.ndarray,
                            weight_bits: int) -> np.ndarray:
    """transcribe_mlpが実装する意味論(STE隠れ + 最終層weight_bits量子化)のargmax。

    convert_prune_transcribe_mnist.py の_quantized_forward_predと同じ理由:
    素のteacher.forward()は最終層を量子化しないため、weight_bits<∞では
    量子化によりargmaxが素のfloat forwardとずれ得る(バグではない)。厳密性
    assertは「transcribeが表現している関数」と比較する必要がある。
    """
    linears = list(teacher_cpu.linears)
    h = torch.as_tensor(x_np, dtype=torch.float64)
    for lin in linears[:-1]:
        w = lin.weight.detach().to(torch.float64)
        b = lin.bias.detach().to(torch.float64)
        h = ((h @ w.T + b) > 0).to(torch.float64)
    fin = linears[-1]
    q, beta = _quantize_final(
        fin.weight.detach().to(torch.float64),
        fin.bias.detach().to(torch.float64), weight_bits,
    )
    scores = h.numpy() @ q.numpy().T.astype(np.float64) + beta.numpy()[None, :]
    return scores.argmax(axis=1)


def assert_transcribe_exact(ir, teacher_cpu: nn.Module, x_np: np.ndarray,
                            weight_bits: int) -> None:
    """transcribe直後のIR argmaxが、transcribeが表現する量子化済み関数のargmaxと
    全パターン一致することを確認する(lowering無損失の厳密性assert)。

    不一致はtranscribe_mlpパイプラインのバグを意味するため、閾値を緩めず例外で停止する。
    (素のfloat forwardとの差は最終層量子化由来でバグではないため、比較対象は
    quantized forwardにする。convert_prune_transcribe_mnist.pyと同じ規約。)
    """
    ref_pred = _quantized_forward_pred(teacher_cpu, x_np, weight_bits)
    sim_pred = simulate(ir, x_np).argmax(axis=-1)
    n_mismatch = int((ref_pred != sim_pred).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密性assert違反: transcribe直後のIR argmaxが量子化済みteacher関数の'
            f'argmaxと{n_mismatch}/{len(x_np)}件不一致。バグなので停止'
            f'(閾値は緩めない)。'
        )


def _make_polish_loss_fn(args, teacher_logits_tr: torch.Tensor | None):
    """arm C/D共通の磨きloss_fnを組み立てる(CE, +KD, +RKD補助項の合成)。

    改善(3): --aux rkd のとき、既存の(CE または CE+KD)損失に加えて
    最終ロジット同士のRKD(convert/relational.py)を補助項として足す。

    総損失(--aux rkd かつ --polish-kd 時) = alpha*CE + (1-alpha)*KD_KL + rkd_beta*RKD
    (--polish-kd無しなら CE + rkd_beta*RKD)。

    「主要(must-land)」表現選択: 最終ロジット(GroupSum出力) vs teacher最終ロジット
    (--polish-kdのpacked-target機構をそのまま再利用、taskで指示された通り)。
    中間層RKD(stretch)はrecover()がloss_fn(output, target)の2引数契約しか
    公開しておらず、中間活性へのアクセスにはrecover()自体の改造が要る
    (TODO、下記docstring参照)。

    TODO(中間層RKD、stretch、未実装): recover()のtrain loopは
    `loss = loss_fn(out, yb)` で最終出力のみをloss_fnへ渡す(model/hidden
    stateは渡さない)。teacher hidden(ArithTeacherMlp.linears[:-1]の出力)と
    student中間LutLayer出力(nn.Sequentialへforward_hook登録で取得可能、
    次元非依存なRKDなので投影層は不要)間のRKDを行うには、recover()の
    train loopをmodel引用を持つ形へ拡張する必要がある(loss_fn(output,
    target)の2引数契約を壊す非自明な変更)。本タスクでは範囲外とし、
    最終ロジットRKD(上記)を動く成果物として確定させる。
    """
    if teacher_logits_tr is None and args.aux == 'rkd':
        raise ValueError('--aux rkd には teacher_logits_tr が必要')

    use_packed = args.polish_kd or args.aux == 'rkd'
    if not use_packed:
        return None, False

    task_loss = None
    if args.polish_kd:
        task_loss = make_kd_loss(
            alpha=args.alpha, temperature=args.temperature,
            standardize=args.standardize, kind='kl',
        )

    def loss_fn(output, packed_target):
        labels = packed_target[:, 0].long()
        t_logits = packed_target[:, 1:]
        if task_loss is not None:
            loss = task_loss(output, labels, t_logits)
        else:
            loss = F.cross_entropy(output, labels)
        if args.aux == 'rkd':
            loss = loss + args.rkd_beta * rkd_loss(output, t_logits.detach())
        return loss

    return loss_fn, True


def run_arm_c(
    ir, x_tr_np, y_tr_np, x_ev_np, y_ev_np, *, selection: str, args, device,
    teacher_logits_tr: torch.Tensor | None = None,
) -> dict:
    """構造転写IRをrecoverで磨く(selectionでheadline'soft'/ablation'hard_st'を切替、
    aux='rkd'ならDのRKD補助損失版になる)。

    KD磨き(--polish-kd)または RKD補助損失(--aux rkd)を使う場合、
    PackedDistillDatasetでstudent入力+ラベル+teacherロジットを
    index整列させ、recoverのloss_fn(output, target)契約を保ったまま
    2引数へスライスするクロージャを注入する
    (convert_c3_polish_mnist.pyのpolish_oneと同じパターン)。
    """
    x_tr_t = torch.as_tensor(x_tr_np, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr_np, dtype=torch.int64)
    x_ev_t = torch.as_tensor(x_ev_np, dtype=torch.float32)
    y_ev_t = torch.as_tensor(y_ev_np, dtype=torch.int64)

    train_ds = TensorDataset(x_tr_t, y_tr_t)
    loss_fn, use_packed = _make_polish_loss_fn(args, teacher_logits_tr)
    if use_packed:
        from griffith_logic.convert.cache import PackedDistillDataset
        packed_ds = PackedDistillDataset(train_ds, teacher_logits_tr)
        train_loader = DataLoader(packed_ds, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        loss_fn = None  # recover既定 = F.cross_entropy

    eval_loader = DataLoader(TensorDataset(x_ev_t, y_ev_t), batch_size=max(len(x_ev_t), 1))

    cfg = RecoverConfig(
        parameterization='warp', epochs=args.recover_epochs, lr=args.recover_lr,
        device=device, tau=args.recover_tau, selection=selection,
    )
    recovered, metrics = recover(ir, train_loader, eval_loader, cfg, loss_fn=loss_fn)
    return {'recovered_ir': recovered, **metrics}


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = args.nbits
    task = args.task
    in_width = 2 * n
    k_classes = num_classes_for(n, task)
    x_all, y_all = arith_classify_dataset(n, task)
    x_all_t = torch.as_tensor(x_all, dtype=torch.float32)
    y_all_t = torch.as_tensor(y_all, dtype=torch.int64)

    print(f'  [seed={seed}] task={task} n={n} in_width={in_width} '
          f'k_classes={k_classes} n_patterns={len(x_all)}')

    result: dict = {'seed': seed, 'n': n, 'k_classes': k_classes}

    # ---- teacher学習(暗記) ----
    torch.manual_seed(seed)
    teacher = ArithTeacherMlp(in_width, args.teacher_hidden, k_classes).to(device)
    teacher_acc = train_teacher(
        teacher, x_all_t, y_all_t, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device, batch_size=args.batch_size,
    )
    print(f'  [seed={seed}] teacher train hard_acc={teacher_acc:.4f}')

    # ---- teacherのfan-inをtranscribe fan-in以下へ枝刈り(必要な場合のみ) ----
    # (sparsify_faninの近似誤差をゼロにし、厳密転写を保証する。'と同じ機構。
    #  改善(1): 倍々刻みの緩やかな多段スケジュール(gentle_fanin_schedule)で
    #  枝刈り、teacher-after-prune の精度低下を抑える)
    teacher, prune_history = ensure_teacher_fanin(
        teacher, x_all, y_all, n_inputs=args.fanin,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    teacher_acc = float(
        (teacher.to(device).eval()(x_all_t.to(device)).argmax(-1).cpu()
         == y_all_t).float().mean()
    )
    print(f'  [seed={seed}] teacher fan-in枝刈り後 hard_acc={teacher_acc:.4f} '
          f'(fanin<={args.fanin})')
    if not prune_history.get('skipped', True) and prune_history.get('stages'):
        stage_str = ' -> '.join(
            f"{s['fanin']}:{s['acc']:.4f}" for s in prune_history['stages']
        )
        print(f'  [seed={seed}] 枝刈り段階acc(fanin:acc): {stage_str}')
    result['teacher_acc'] = teacher_acc
    result['prune_stages'] = prune_history.get('stages', [])

    # ---- transcribe(training-free) + 厳密性assert ----
    ir, teacher_cpu = build_transcribed_ir(
        teacher, x_all, n_inputs=args.fanin, weight_bits=args.weight_bits,
        device=device,
    )
    assert_transcribe_exact(ir, teacher_cpu, x_all, args.weight_bits)
    result['exactness_ok'] = True
    print(f'  [seed={seed}] transcribe厳密性assert: OK')

    # 転写直後(磨き前)のhard acc(oracle参照)
    from griffith_logic.convert.recover import _hard_accuracy
    transcribe_hard_acc = _hard_accuracy(ir, x_all, y_all)
    result['transcribe_hard_acc'] = transcribe_hard_acc
    print(f'  [seed={seed}] transcribe直後 hard_acc(oracle参照)={transcribe_hard_acc:.4f}')

    # ---- teacherロジット(arm B / arm C(kd)用) ----
    # build_transcribed_ir が teacher を in-place で CPU へ移すため(teacher_cpu
    # と同一オブジェクト)、ロジット計算もCPUで行う(studentは後段でdeviceへ送る)
    teacher_logits_all = make_teacher_logits(teacher_cpu, x_all_t.cpu(), 'cpu')

    # ---- arm A: scratch ----
    t0 = time.time()
    res_a = run_arm_a(
        x_all_t, y_all_t, x_all, y_all, in_width=in_width, k_classes=k_classes,
        args=args, device=device, seed=seed + 1000,
    )
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm A(scratch): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} gap={res_a['gap']:+.4f} "
          f"({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ ----
    t0 = time.time()
    res_b = run_arm_b(
        x_all_t, y_all_t, teacher_logits_all, x_all, y_all,
        in_width=in_width, k_classes=k_classes, args=args, device=device,
        seed=seed + 2000,
    )
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm B(IO-KDのみ): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} gap={res_b['gap']:+.4f} "
          f"({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm C(本命): 構造転写 + --polish-selection磨き(既定'soft') ----
    t0 = time.time()
    res_c = run_arm_c(
        ir, x_all, y_all, x_all, y_all, selection=args.polish_selection,
        args=args, device=device, teacher_logits_tr=teacher_logits_all,
    )
    res_c['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm C(構造転写+{args.polish_selection}磨き, 本命): "
          f"soft {res_c['soft_acc_before']:.4f}->{res_c['soft_acc_after']:.4f} "
          f"hard {res_c['hard_acc_before']:.4f}->{res_c['hard_acc_after']:.4f} "
          f"gap={res_c['soft_acc_after'] - res_c['hard_acc_after']:+.4f} "
          f"({res_c['seconds']}s)")
    result['C'] = res_c

    # ---- arm C_hard_st(アブレーション): 常にselection='hard_st'で磨く ----
    # 「良い離散化初期値に対してhard_stは破壊的」という知見の対照として
    # --polish-selectionの値によらず常に実行する(taskの明示指示: keep it)。
    # ただし--polish-selectionが既にhard_stならCと重複するので再利用する。
    t0 = time.time()
    if args.polish_selection == 'hard_st':
        res_c_hard_st = res_c
    else:
        res_c_hard_st = run_arm_c(
            ir, x_all, y_all, x_all, y_all, selection='hard_st', args=args,
            device=device, teacher_logits_tr=teacher_logits_all,
        )
    res_c_hard_st['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm C_hard_st(構造転写+hard_st磨き, アブレーション): "
          f"soft {res_c_hard_st['soft_acc_before']:.4f}->"
          f"{res_c_hard_st['soft_acc_after']:.4f} "
          f"hard {res_c_hard_st['hard_acc_before']:.4f}->"
          f"{res_c_hard_st['hard_acc_after']:.4f} "
          f"gap={res_c_hard_st['soft_acc_after'] - res_c_hard_st['hard_acc_after']:+.4f} "
          f"({res_c_hard_st['seconds']}s)")
    result['C_hard_st'] = res_c_hard_st

    # ---- arm D(本命+RKD補助損失): Cと同じselectionにRKD補助項を追加 ----
    t0 = time.time()
    d_args = argparse.Namespace(**{**vars(args), 'aux': 'rkd'})
    res_d = run_arm_c(
        ir, x_all, y_all, x_all, y_all, selection=args.polish_selection,
        args=d_args, device=device, teacher_logits_tr=teacher_logits_all,
    )
    res_d['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm D(構造転写+{args.polish_selection}磨き+RKD): "
          f"soft {res_d['soft_acc_before']:.4f}->{res_d['soft_acc_after']:.4f} "
          f"hard {res_d['hard_acc_before']:.4f}->{res_d['hard_acc_after']:.4f} "
          f"gap={res_d['soft_acc_after'] - res_d['hard_acc_after']:+.4f} "
          f"({res_d['seconds']}s)")
    result['D'] = res_d

    # ---- 派生指標 ----
    hard_a = res_a['hard_acc']
    hard_b = res_b['hard_acc']
    hard_c = res_c['hard_acc_after']
    hard_c_hard_st = res_c_hard_st['hard_acc_after']
    hard_d = res_d['hard_acc_after']
    result['lift_vs_scratch'] = hard_c - hard_a
    result['lift_vs_iokd'] = hard_c - hard_b
    result['hard_st_contribution'] = hard_c_hard_st - hard_c
    result['oracle_reach'] = hard_c / max(teacher_acc, EPS)
    result['rkd_lift'] = hard_d - hard_c

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 96)
    print('構造転写(structural transfer)+soft磨き(本命) vs IO-KDのみ/scratch/'
          'hard_stアブレーション/RKD補助損失 — 結果')
    print('=' * 96)

    arms = ['A', 'B', 'C', 'C_hard_st', 'D']
    labels = {
        'A': 'A: scratch',
        'B': 'B: IO-KDのみ',
        'C': f'C: 構造転写+{args.polish_selection}磨き(本命)',
        'C_hard_st': 'C_hard_st: 構造転写+hard_st磨き(アブレーション、破壊的初期値対照)',
        'D': f'D: 構造転写+{args.polish_selection}磨き+RKD補助損失',
    }

    header = '| arm | soft_acc(mean±std) | hard_acc(mean±std) | gap(mean±std) |'
    print(header)
    print('|' + '---|' * 4)
    summary: dict = {'arms': {}}
    for arm in arms:
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
        print(f'| {labels[arm]} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} | '
              f'{gm:+.4f}±{gs:.4f} |')
        summary['arms'][arm] = {
            'soft_acc_mean': sm, 'soft_acc_std': ss,
            'hard_acc_mean': hm, 'hard_acc_std': hs,
            'gap_mean': gm, 'gap_std': gs,
        }

    teacher_accs = [r['teacher_acc'] for r in results]
    oracle_accs = [r['transcribe_hard_acc'] for r in results]
    tm, ts = _mean_std(teacher_accs)
    om, os_ = _mean_std(oracle_accs)
    print(f'\noracle参照: teacher hard_acc={tm:.4f}±{ts:.4f}  '
          f'転写直後(磨き前)hard_acc={om:.4f}±{os_:.4f}')
    summary['teacher_acc_mean'] = tm
    summary['teacher_acc_std'] = ts
    summary['transcribe_hard_acc_mean'] = om
    summary['transcribe_hard_acc_std'] = os_

    lift_scratch = [r['lift_vs_scratch'] for r in results]
    lift_iokd = [r['lift_vs_iokd'] for r in results]
    hard_st_contrib = [r['hard_st_contribution'] for r in results]
    oracle_reach = [r['oracle_reach'] for r in results]
    rkd_lift = [r['rkd_lift'] for r in results]
    lsm, lss = _mean_std(lift_scratch)
    lim, lis = _mean_std(lift_iokd)
    hcm, hcs = _mean_std(hard_st_contrib)
    orm, ors = _mean_std(oracle_reach)
    rkm, rks = _mean_std(rkd_lift)

    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift_vs_scratch (C - A)                  : {lsm:+.4f}±{lss:.4f}')
    print(f'lift_vs_iokd    (C - B)                  : {lim:+.4f}±{lis:.4f}')
    print(f'hard_st_contribution (C_hard_st - C)     : {hcm:+.4f}±{hcs:.4f}')
    print(f'oracle_reach    (C / teacher_acc)        : {orm:.4f}±{ors:.4f}')
    print(f'rkd_lift        (D - C)                  : {rkm:+.4f}±{rks:.4f}')

    summary['lift_vs_scratch_mean'] = lsm
    summary['lift_vs_scratch_std'] = lss
    summary['lift_vs_iokd_mean'] = lim
    summary['lift_vs_iokd_std'] = lis
    summary['hard_st_contribution_mean'] = hcm
    summary['hard_st_contribution_std'] = hcs
    summary['oracle_reach_mean'] = orm
    summary['oracle_reach_std'] = ors
    summary['rkd_lift_mean'] = rkm
    summary['rkd_lift_std'] = rks

    # 仮説判定の「本手法」はarm C(既定--polish-selection='soft')であり、
    # hard_stではない(改善(4):初版のバグ修正。旧版はhard_stを本手法と
    # 決め打ちして過小評価していた)。
    print('\n--- 仮説判定(本手法 = arm C, --polish-selection='
          f"{args.polish_selection}) ---")
    print(f"H(構造転写>IO-KDのみ, hard acc): lift_vs_iokd={lim:+.4f} "
          f"→ {'支持' if lim > 0 else '不支持'}")
    print(f"H(構造転写>scratch, hard acc): lift_vs_scratch={lsm:+.4f} "
          f"→ {'支持' if lsm > 0 else '不支持'}")
    print(f"H(hard_stは良い離散化初期値に対して破壊的): "
          f"hard_st_contribution(C_hard_st - C)={hcm:+.4f} "
          f"→ {'支持(hard_stが悪化=破壊的)' if hcm < 0 else '不支持(hard_stが改善)'}")
    print(f"H(RKD補助損失がsoft磨き単体を上回る): rkd_lift(D - C)={rkm:+.4f} "
          f"→ {'支持' if rkm > 0 else '不支持'}")

    exactness_all = all(r['exactness_ok'] for r in results)
    print(f'\n厳密性assert(全seed): {"OK" if exactness_all else "NG"}')
    summary['exactness_ok'] = exactness_all

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


FANIN_CHOICES = (2, 4, 6)  # LutLayer/TranscribeSpec対応の最大値は6(2^6=64エントリLUT)


def _strip_ir(r: dict) -> dict:
    r2 = dict(r)
    for arm in ('C', 'C_hard_st', 'D'):
        if arm in r2 and isinstance(r2[arm], dict):
            r2[arm] = {k: v for k, v in r2[arm].items() if k != 'recovered_ir'}
    return r2


def run_fanin_config(fanin: int, args, device: str) -> dict:
    """1つのfanin構成でseeds分の全アームを実行し、summaryとper_seedを返す。

    改善(1)の--faninスイープ用に main() の本体をこの関数へ切り出した。
    fanin=argsで既定の6のみ渡された場合は従来のsingle-run実行と等価。
    """
    fanin_args = argparse.Namespace(**{**vars(args), 'fanin': fanin})
    print(f'\n{"#" * 30} task={fanin_args.task} fanin={fanin} {"#" * 30}')
    print(f'nbits={fanin_args.nbits} (入力={2 * fanin_args.nbits}bit) '
          f'transcribe fanin={fanin} weight_bits={fanin_args.weight_bits} '
          f'polish_selection={fanin_args.polish_selection} aux={fanin_args.aux}')

    use_wandb = init_wandb(
        run_name=f'arith_structural_transfer_{fanin_args.task}_'
                 f'n{fanin_args.nbits}_fanin{fanin}',
        config_dict=vars(fanin_args),
        device=device,
        tags=['pilot', 'structural-transfer', 'soft-polish', 'rkd', fanin_args.task],
        group='convert-arith-structural-transfer',
    )

    results = []
    for seed in range(fanin_args.seeds):
        print(f'\n########## fanin={fanin} seed {seed + 1}/{fanin_args.seeds} ##########')
        r = run_one_seed(seed, fanin_args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_acc': r['teacher_acc'],
                f'seed{seed}/transcribe_hard_acc': r['transcribe_hard_acc'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/A_soft_acc': r['A']['soft_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/B_soft_acc': r['B']['soft_acc'],
                f'seed{seed}/C_hard_acc': r['C']['hard_acc_after'],
                f'seed{seed}/C_soft_acc': r['C']['soft_acc_after'],
                f'seed{seed}/C_hard_st_hard_acc': r['C_hard_st']['hard_acc_after'],
                f'seed{seed}/C_hard_st_soft_acc': r['C_hard_st']['soft_acc_after'],
                f'seed{seed}/D_hard_acc': r['D']['hard_acc_after'],
                f'seed{seed}/D_soft_acc': r['D']['soft_acc_after'],
                f'seed{seed}/lift_vs_scratch': r['lift_vs_scratch'],
                f'seed{seed}/lift_vs_iokd': r['lift_vs_iokd'],
                f'seed{seed}/hard_st_contribution': r['hard_st_contribution'],
                f'seed{seed}/oracle_reach': r['oracle_reach'],
                f'seed{seed}/rkd_lift': r['rkd_lift'],
            })

    summary = summarize(results, fanin_args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    return {'fanin': fanin, 'args': vars(fanin_args), 'summary': summary,
            'per_seed': [_strip_ir(r) for r in results]}


def main():
    parser = argparse.ArgumentParser(
        description='構造転写(wiring+ゲート初期化)+soft磨き(本命) vs IO-KDのみ/'
                     'scratch/hard_stアブレーション/RKD補助損失'
                     '(パイロット実験、整数和分類testbed)')
    parser.add_argument('--task', type=str, default='adder', choices=TASK_CHOICES,
                        help="タスク種別(既定'adder'、現行動作を保存)。'adder': "
                             "ラベル=a+b, K=2^(n+1)-1クラス。'multiplier': "
                             "ラベル=a*b, K=(2^n-1)^2+1クラス(構造的に難しいタスクで"
                             '知見のスケール確認、item 2)')
    parser.add_argument('--nbits', type=int, default=4,
                        help='オペランドのビット幅n(既定4。入力2n bit、'
                             'クラス数は--taskに依存)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200,
                        help='arm A/B(scratch DLGN)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=300)
    parser.add_argument('--teacher-prune-epochs', type=int, default=300,
                        help='teacherのfan-in枝刈り段あたりre-fitエポック数'
                             '(in_width>faninのときのみ使用。改善(1)で100→300に'
                             '引き上げ、gentle_fanin_scheduleの緩やかな多段化と'
                             '合わせてteacher-after-prune精度を保つ)')
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-hidden', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--lr', type=float, default=5e-2, help='arm A/BのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128, help='DLGN隠れ層幅')
    parser.add_argument('--depth', type=int, default=3, help='DLGN隠れ層数')
    parser.add_argument('--n-inputs', type=int, default=4, choices=(2, 4, 6),
                        help='arm A/B DLGNのLUT fan-in')
    parser.add_argument('--group-width', type=int, default=4,
                        help='arm A/B DLGN最終層のクラスあたりLUT出力数')
    parser.add_argument('--fanin', type=int, nargs='+', default=None,
                        help='transcribe_mlpのLUT fan-in(arm C/C_hard_st/D、既定[6])。'
                             '複数値を渡すとスイープ実行し、fanin毎に'
                             'oracle_reach/hard_accを個別報告する。'
                             f'LutLayer制約により{FANIN_CHOICES}のみ対応'
                             '(2^n_inputsエントリのLUTでn_inputs<=6が上限)')
    parser.add_argument('--weight-bits', type=int, default=8,
                        help='transcribe_mlp最終層の量子化ビット幅')
    parser.add_argument('--recover-epochs', type=int, default=30)
    parser.add_argument('--recover-lr', type=float, default=0.05)
    parser.add_argument('--recover-tau', type=float, default=4.0)
    parser.add_argument('--polish-selection', type=str, default='soft',
                        choices=('soft', 'hard_st'),
                        help="arm C(本命)の磨きselection(既定'soft'、実測: "
                             "良い離散化初期値からの磨きはhard_stだと0.855→0.225へ"
                             "崩壊、softだと0.855→0.888へ改善)。'hard_st'アブレーション"
                             "アームは--polish-selectionの値によらず常に別途実行される")
    parser.add_argument('--aux', type=str, default='none', choices=('none', 'rkd'),
                        help='磨きへの補助損失(改善(3))。既定none=arm Cは補助損失なし。'
                             "arm Dは常にaux='rkd'で実行される(--auxはarm Cのみに影響)")
    parser.add_argument('--rkd-beta', type=float, default=1.0,
                        help='RKD補助損失の重み(arm D、--aux rkd時)')
    parser.add_argument('--polish-kd', action='store_true',
                        help='arm C/C_hard_st/Dの磨きにKD損失(CE+KL混合)を注入する')
    parser.add_argument('--alpha', type=float, default=0.5, help='KD損失のCE項重み')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD温度T')
    parser.add_argument('--standardize', action='store_true',
                        help='KD項の両ロジットをZ-score標準化する')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小width・少epoch、<60s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str, default='outputs/arith_structural_transfer')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    fanin_explicit = args.fanin is not None
    if args.fanin is None:
        args.fanin = [6]

    if args.quick:
        args.nbits = min(args.nbits, 3)
        args.seeds = 1
        args.epochs = 60
        args.teacher_epochs = 150
        args.teacher_prune_epochs = 30
        args.teacher_hidden = [16]
        args.teacher_lr = 3e-2
        args.width = 32
        args.depth = 2
        args.n_inputs = 4
        args.group_width = 2
        if not fanin_explicit:
            # fanin=6=in_width(2*nbits=6)なのでteacher fan-in枝刈り不要(最速・厳密)。
            # ユーザーが明示的に--faninを指定した場合(スイープ動作確認等)は尊重する。
            args.fanin = [6]
        args.recover_epochs = 5
        args.batch_size = 32

    bad_fanins = [f for f in args.fanin if f not in FANIN_CHOICES]
    if bad_fanins:
        print(f'!! --fanin={bad_fanins} はtranscribe_mlp非対応'
              f'({FANIN_CHOICES}のみ。LutLayer.n_inputsの上限=6のため'
              'fan-inは6を超えられない)')
        sys.exit(1)

    k_classes = num_classes_for(args.nbits, args.task)
    print(f'デバイス: {device}')
    print(f'task={args.task} nbits={args.nbits} (入力={2 * args.nbits}bit, '
          f'K={k_classes}クラス, n_patterns={1 << (2 * args.nbits)}) '
          f'seeds={args.seeds}')
    print(f'DLGN(arm A/B): width={args.width} depth={args.depth} '
          f'n_inputs={args.n_inputs} group_width={args.group_width} '
          f'epochs={args.epochs} lr={args.lr}')
    print(f'teacher(arm C/D): hidden={args.teacher_hidden} '
          f'epochs={args.teacher_epochs} lr={args.teacher_lr} '
          f'prune_epochs_per_stage={args.teacher_prune_epochs}')
    print(f'transcribe: fanin(sweep)={args.fanin} weight_bits={args.weight_bits}')
    print(f'recover: epochs={args.recover_epochs} lr={args.recover_lr} '
          f'tau={args.recover_tau} polish_selection={args.polish_selection} '
          f'polish_kd={args.polish_kd} aux={args.aux} rkd_beta={args.rkd_beta}')

    started = time.time()
    fanin_results = [run_fanin_config(f, args, device) for f in args.fanin]

    # fanin間の比較(oracle_reach vs fanin のトレードオフ、改善(1)の主張点)
    if len(fanin_results) > 1:
        print('\n' + '=' * 96)
        print('fan-inスイープ比較(oracle_reach対fan-inのトレードオフ)')
        print('=' * 96)
        print('| fanin | teacher_acc | transcribe_hard_acc(oracle) | '
              'C hard_acc(本命) | oracle_reach |')
        print('|' + '---|' * 5)
        for fr in fanin_results:
            s = fr['summary']
            print(f"| {fr['fanin']} | {s['teacher_acc_mean']:.4f} | "
                  f"{s['transcribe_hard_acc_mean']:.4f} | "
                  f"{s['arms']['C']['hard_acc_mean']:.4f} | "
                  f"{s['oracle_reach_mean']:.4f} |")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(
        args.out_dir, f'result_{args.task}_n{args.nbits}_{ts}.json',
    )

    dump = {
        'args': vars(args),
        'fanin_sweep': fanin_results,
    }
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
