"""コンパクトnative DLGN(difflogic方式)獲得 — 算術分類での配線+ゲート初期化/KD/層別蒸留
(ROADMAP-v5 Track D、プランpivot)。

(`convert_arith_structural_transfer.py`)は「teacher形状のLUT回路」(warp
パラメータ化、テーブル数=ノード数固定・大きい)への構造転写だった。本タスクは
それとは異なる目標を検証する: **コンパクトなnative DLGN**
(difflogic方式、`LutLayer(n_inputs=2, parameterization='softmax')` の16ゲート
softmaxセル、少ゲート数、ゲート選択を勾配で自由学習)を、算術分類タスクで
「wiring+gate-init(構造転写)」「KD」「層別二値蒸留」経由でscratchより高いhard
精度に到達できるか。

D5(native DLGN 変換の先行実験)はMNISTで
「native DLGNへの事前学習からの変換はscratchを上回らない」と結論した
(teacherが弱い・アーキが最大限に異質)。本タスクはteacherが暗記可能で
scratch native DLGNの性能が低い(~0.20)算術分類という**有利な条件**で
D5の結論を再検証する。

## アーム(すべてnative DLGN、`hard_acc_via_export`/`simulate`でhard acc測定)

- **A(scratch)**: `LutLayer(n_inputs=2, softmax)`列+GroupSumをラベルCEで学習
  (`convert_arith_structural_transfer.build_dlgn`/`run_arm_a`を`n_inputs=2`で
  再利用)。下限対照。
- **B(IO-KDのみ)**: 同じcompact native archをteacher soft-logit KD+CEで学習
  (`run_arm_b`を`n_inputs=2`で再利用)。「KD単体」を測る。
- **E(本命: wiring+gate-init→soft磨き)**: STE teacherを`gentle_fanin_schedule`で
  fan-in=2まで枝刈り→`transcribe_mlp(n_inputs=2)`で厳密転写→
  `_rebuild_as_native16g`でnative softmax16g DLGNへ復元
  (`convert_native_init_mnist.rebuild_softmax16g`と同じパターン: 各lut_layer
  ノードのtrain_info.parameterization='softmax'+initialized=Trueにして
  build_torchへ渡す。n_inputs=2のLutLayerはsoftmax+N=2で自動的に16ゲート
  モードになる)。**復元直後がtranscribe直後のIRとbit-exactであることを
  assert**(2入力真理値表→16ゲートは無損失変換のはず。破れたらSTOPし報告)。
  その後soft磨き(低LR CE fine-tune)。
- **E+KD(init+distill+polish)**: Eの初期化から、磨きをKD(CE+KL混合)で行う。
- **F(層別二値蒸留、N3再訪)**: STE teacherの隠れ二値活性(HeavisideSTE出力)を
  forward hookで層ごとに捕捉し、native DLGN(層ごとにブロック割当、m=1簡易版)を
  per-layer BCEで学習→GroupSum接続→CE/KD fine-tune。(MNIST)は
  「scratch以下」だったので、ここでも honestly re-test して結果をそのまま報告する
  (最小限の単一ステージ版として実装、と明記)。
- **(参照) C(warp、非native)**: `run_arm_c`(structural_transfer, selection='soft')
  をそのまま呼び、同様の巨大teacher形状warp回路の hard acc を非nativeの
  上限参照として併記する(compact native Eとのgapを見る)。

## 派生指標

- lift(E - A), lift(E - B), lift(E+KD - E), gap(E - C)
- fan-in≤2 oracle損失: teacher-after-fanin2-prune acc / teacher-before acc
- E/Cのゲート数比較(nativeがcompactであることの定量化)

## 使用例

    # 煙実行(数十秒、wandb無効、1 seed・小width・少epoch)
    uv run python experiments/convert_arith_to_native_dlgn.py --quick --no-wandb

    # 本実行(既定nbits=4、3 seeds)
    uv run python experiments/convert_arith_to_native_dlgn.py --nbits 4 --seeds 3 --no-wandb
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# の機構を無改造でimport(taskの明示指示: convert_arith_structural_transfer.py
# を改変せず、importして再利用する)
from convert_arith_structural_transfer import (
    ArithTeacherMlp,
    arith_classify_dataset,
    assert_transcribe_exact,
    build_transcribed_ir,
    ensure_teacher_fanin,
    hard_acc_via_export,
    make_teacher_logits,
    num_classes_for,
    run_arm_a,
    run_arm_b,
    run_arm_c,
    soft_hard_acc_dlgn,
    train_teacher,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutLayer
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.recover import _hard_accuracy
from griffith_logic.export.estimate import estimate_resources
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.simulate import simulate

load_dotenv()

EPS = 1e-8
NATIVE_N_INPUTS = 2  # compact native DLGN(difflogic方式)はN=2専用


# ============================================================
# native-16g rebuild(N2パターンの再利用、experiments/convert_native_init_mnist.py
# ::rebuild_softmax16g と同一ロジック。taskの指示によりconvert_native_init_mnist.py
# は改変せずこのファイルへ複製する: 対象ファイルが違うテスト対象であるため)
# ============================================================

def rebuild_as_native16g(graph: IRGraph) -> nn.Module:
    """n_inputs=2の転写IRをsoftmax16g(16ゲート自由選択)の学習可能DLGNへ復元する。

    `convert/recover.py::rebuild_trainable` は softmax16g を許可しない
    (RECOVER_PARAMETERIZATIONS=independent/raw/warp)。LutLayer は
    parameterization='softmax' かつ n_inputs=2 のとき内部で16ゲートモード
    (_use_16gate)になり、`restore_lut_layer`(`ir/weights.py`)が
    `weights_from_truth_tables(is_16gate)`経由で各ノードの真理値表を
    ビット厳密にゲートロジットへ変換する(2入力真理値表→16ゲートIDへの
    写像 `gate_id_from_tt_n2` は全単射なので無損失)。

    Args:
        graph: n_inputs=2で転写されたIR(直列: 入力→lut_layer列→group_sum)。

    Returns:
        native softmax16g DLGN(nn.Sequential、build_torch経由)。

    Raises:
        ValueError: いずれかのlut_layerノードのn_inputsが2でない場合
            (16ゲートモードの前提が満たせない)。
    """
    g = copy.deepcopy(graph)
    for node in g.nodes:
        if node.type == 'lut_layer':
            n_in = int(node.params['n_inputs'])
            if n_in != NATIVE_N_INPUTS:
                raise ValueError(
                    f'native16g復元はn_inputs={NATIVE_N_INPUTS}専用: '
                    f'ノード{node.id}のn_inputs={n_in}'
                )
            node.train_info = {
                **node.train_info,
                'parameterization': 'softmax',  # N=2 → 16ゲートモード
                'initialized': True,
            }
    return build_torch(g)


def assert_native16g_bit_exact(graph: IRGraph, model: nn.Module,
                               x_np: np.ndarray) -> None:
    """rebuild_as_native16g直後のmodelがsimulate(graph)とargmax一致することを確認する。

    2入力真理値表→16ゲートIDは全単射(無損失)なので、この一致は理論上
    常に成立するはず。破れたらtranscribe/rebuildパイプラインのバグなので
    閾値を緩めず例外で停止する。
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x_np, dtype=torch.float32)
        y_model = model(x_t).numpy()
    y_sim = simulate(graph, x_np)
    n_mismatch = int((y_model.argmax(axis=-1) != y_sim.argmax(axis=-1)).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密性assert違反: native16g復元直後のargmaxがtranscribe直後の'
            f'IR argmaxと{n_mismatch}/{len(x_np)}件不一致。理論上2入力'
            'TT→16ゲートは無損失変換のはずなのでバグ(閾値は緩めない)。'
        )


def native_gate_count(graph: IRGraph) -> int:
    """native DLGN(IR)の総ゲート数(lut_layer出力ノード数の総和)。"""
    est = estimate_resources(graph)
    return int(est['total_luts'])


# ============================================================
# arm E / E+KD: wiring+gate-init(native16g) → soft磨き(→ KD磨き)
# ============================================================

def polish_native(
    model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_ev_np: np.ndarray, y_ev_np: np.ndarray, *, in_width: int, device: str,
    epochs: int, lr: float, batch_size: int,
    teacher_logits_tr: torch.Tensor | None = None,
    kd_alpha: float = 0.5, kd_temperature: float = 4.0,
    kd_standardize: bool = True,
) -> dict:
    """native16g DLGNを低LR CE(またはKD)でfine-tuneし、磨き前後のacc/gapを返す。

    teacher_logits_tr が与えられればKD損失(make_kd_loss)、Noneなら素のCE。
    """
    model = model.to(device)
    soft_acc_before = soft_hard_acc_dlgn(model, x_tr, y_tr, device)
    hard_acc_before = hard_acc_via_export(model, x_ev_np, y_ev_np, in_width)

    if teacher_logits_tr is not None:
        loss_fn = make_kd_loss(
            alpha=kd_alpha, temperature=kd_temperature,
            standardize=kd_standardize, kind='kl',
        )
        ds = TensorDataset(x_tr, y_tr, teacher_logits_tr)
    else:
        loss_fn = None
        ds = TensorDataset(x_tr, y_tr)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            if teacher_logits_tr is not None:
                xb, yb, tb = batch
                tb = tb.to(device)
            else:
                xb, yb = batch
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb, tb) if teacher_logits_tr is not None \
                else F.cross_entropy(out, yb)
            loss.backward()
            opt.step()

    soft_acc_after = soft_hard_acc_dlgn(model, x_tr, y_tr, device)
    hard_acc_after = hard_acc_via_export(model, x_ev_np, y_ev_np, in_width)
    return {
        'soft_acc_before': soft_acc_before, 'hard_acc_before': hard_acc_before,
        'soft_acc_after': soft_acc_after, 'hard_acc_after': hard_acc_after,
        'gap_after': soft_acc_after - hard_acc_after,
    }


def run_arm_e(
    native_ir, x_all, y_all, x_all_t, y_all_t, *, in_width, args, device,
    teacher_logits_all: torch.Tensor | None = None,
) -> dict:
    """arm E(wiring+gate-init→soft磨き)またはE+KD(teacher_logits_all指定時)。

    native_ir(n_inputs=2の転写IR)から毎回新規にmodelを復元する
    (E本体とE+KDが同じ初期化から独立に出発できるよう呼び出し側で
    native_irを共有し、本関数内でrebuildする)。
    """
    model = rebuild_as_native16g(native_ir)
    assert_native16g_bit_exact(native_ir, model, x_all)
    res = polish_native(
        model, x_all_t, y_all_t, x_all, y_all, in_width=in_width,
        device=device, epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size,
        teacher_logits_tr=teacher_logits_all,
        kd_alpha=args.alpha, kd_temperature=args.temperature,
        kd_standardize=args.standardize,
    )
    res['exactness_ok'] = True
    return res


# ============================================================
# arm F: 層別二値蒸留(N3再訪、最小単一ステージ版)
# ============================================================

class NativeBlockF(nn.Module):
    """1 teacher隠れ層に対応するnative DLGNブロック(LutLayer N=2 softmax、depth層積層)。

    m=1簡易版(出力幅=teacher層幅、そのままBCE対象)のみ実装する
    (`distill_layerwise_native_mnist.py::NativeBlock` のm=1相当。taskの
    指示通り「最小単一ステージ版」として明示的にスコープを絞る)。
    """

    def __init__(self, in_dim: int, out_width: int, depth: int, device: str):
        super().__init__()
        self.out_width = out_width
        layers = []
        prev = in_dim
        for _ in range(depth):
            layers.append(LutLayer(
                prev, out_width, n_inputs=NATIVE_N_INPUTS, device=device,
                parameterization='softmax', connections='random',
            ))
            prev = out_width
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lay in self.layers:
            x = lay(x)
        return x

    def n_gates(self) -> int:
        return sum(int(lay.out_dim) for lay in self.layers)


class LayerwiseStudentF(nn.Module):
    """teacher隠れ層ごとのNativeBlockFを直列接続し、末尾にGroupSum頭を持つstudent。"""

    def __init__(self, in_width: int, teacher_widths: tuple[int, ...],
                depth: int, k_classes: int, group_width: int, device: str,
                tau: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_width
        for tw in teacher_widths:
            blk = NativeBlockF(prev, tw, depth, device)
            self.blocks.append(blk)
            prev = tw
        # 最終ブロック出力を分類ヘッド用の幅(k_classes*group_width)へ変換する
        # 追加LutLayer1層(層別BCE監督の対象外、CE/KD fine-tune専用)
        out_final = k_classes * group_width
        self.head_lut = LutLayer(
            prev, out_final, n_inputs=NATIVE_N_INPUTS, device=device,
            parameterization='softmax', connections='random',
        )
        self.head_sum = GroupSum(k=k_classes, tau=tau)

    def block_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.head_lut(x)
        return self.head_sum(x)

    def n_gates(self) -> int:
        return sum(b.n_gates() for b in self.blocks) + int(self.head_lut.out_dim)


@torch.no_grad()
def hard_acc_via_eval_forward(model: nn.Module, x_np: np.ndarray, y_np: np.ndarray,
                              device: str) -> float:
    """モデルのeval()forward(離散化hard経路)によるargmax精度。

    `hard_acc_via_export`(structural_transfer)はnn.Sequentialを前提に
    export_ir経由でIR化してからsimulateする。arm Fのstudent(LayerwiseStudentF)
    は素のnn.Sequentialではなく制御フローを持つ自作nn.Moduleのため
    export_irの抽出器が未登録(NotImplementedError)。LutLayer/GroupSumは
    eval()時に厳密な離散化フォワード(hard forward)を行う設計
    (`layers/lut.py`のforward推論時分岐: softmax16gはargmaxでゲートを
    1つ選択、GroupSumはτに依らずargmax不変)なので、IR化を経由せず
    eval()forwardのargmaxを直接取ればexport_ir経由のsimulateと等価な
    hard精度が得られる(N3の`distill_layerwise_native_mnist.py::eval_hard_acc`
    と同じ設計)。
    """
    model.eval()
    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    pred = model(x_t).argmax(dim=-1).cpu().numpy()
    return float((pred == y_np).mean())


@torch.no_grad()
def capture_hidden_bits_arith(teacher: ArithTeacherMlp,
                              x_all: torch.Tensor) -> list[np.ndarray]:
    """teacher(ArithTeacherMlp)の各隠れ層二値活性 y=1[Wx+b>0] を全パターン分捕捉する。

    forward hookでlinears[:-1]の生preactを取得し、HeavisideSTEと同じ二値化
    (>0)をteacher.forwardと厳密に同じ意味論で適用する
    (distill_layerwise_native_mnist.py::capture_hidden_bits と同じ設計)。
    teacherが置かれているデバイス(呼び出し側の責務)で入力を評価する
    (build_transcribed_ir がteacherをCPUへin-placeで移すため、Fで渡される
    teacher_cpuは常にCPU)。
    """
    teacher.eval()
    dev = next(teacher.parameters()).device
    captured: dict[int, np.ndarray] = {}
    handles = []
    hidden_lins = list(teacher.linears[:-1])

    def make_hook(li):
        def hook(_module, _inp, out):
            captured[li] = (out > 0).to(torch.uint8).cpu().numpy()
        return hook

    for li, lin in enumerate(hidden_lins):
        handles.append(lin.register_forward_hook(make_hook(li)))
    try:
        teacher(x_all.to(dev).float())
    finally:
        for h in handles:
            h.remove()
    return [captured[li] for li in range(len(hidden_lins))]


def run_arm_f(
    teacher: ArithTeacherMlp, x_all, y_all, x_all_t, y_all_t, *, in_width,
    k_classes, args, device, teacher_logits_all: torch.Tensor | None = None,
) -> dict:
    """F: 層別二値蒸留(最小単一ステージ版)。

    Stage1: teacher隠れ層の二値活性をper-block BCEで模倣(joint最適化)。
    Stage2: GroupSum頭を接続しCE(またはKD)でfine-tune。
    """
    teacher_widths = tuple(lin.out_features for lin in teacher.linears[:-1])
    teacher_bits = capture_hidden_bits_arith(teacher, x_all_t)

    student = LayerwiseStudentF(
        in_width, teacher_widths, depth=args.f_depth, k_classes=k_classes,
        group_width=args.group_width, device=device, tau=1.0,
    ).to(device)
    n_gates = student.n_gates()

    # ---- Stage1: per-layer BCE(joint) ----
    tgt_t = [torch.as_tensor(b, dtype=torch.float32) for b in teacher_bits]
    bce = nn.BCELoss()
    opt1 = torch.optim.Adam(student.parameters(), lr=args.f_stage1_lr)
    ds1 = TensorDataset(x_all_t, torch.arange(len(x_all_t)))
    loader1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=True)
    student.train()
    stage1_final_bce = [float('nan')] * len(student.blocks)
    for _ in range(args.f_stage1_epochs):
        for xb, idx in loader1:
            xb = xb.to(device).float()
            opt1.zero_grad()
            outs = student.block_outputs(xb)
            losses = []
            for bi, out in enumerate(outs):
                pred = out.clamp(1e-6, 1 - 1e-6)
                losses.append(bce(pred, tgt_t[bi][idx].to(device)))
            loss = sum(losses)
            loss.backward()
            opt1.step()
            stage1_final_bce = [float(v.detach()) for v in losses]

    # ---- Stage2: GroupSum頭でCE(またはKD)fine-tune ----
    if teacher_logits_all is not None:
        loss_fn2 = make_kd_loss(
            alpha=args.alpha, temperature=args.temperature,
            standardize=args.standardize, kind='kl',
        )
        ds2 = TensorDataset(x_all_t, y_all_t, teacher_logits_all)
    else:
        loss_fn2 = None
        ds2 = TensorDataset(x_all_t, y_all_t)
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=True)
    opt2 = torch.optim.Adam(student.parameters(), lr=args.f_stage2_lr)
    student.train()
    for _ in range(args.f_stage2_epochs):
        for batch in loader2:
            if teacher_logits_all is not None:
                xb, yb, tb = batch
                tb = tb.to(device)
            else:
                xb, yb = batch
            xb, yb = xb.to(device).float(), yb.to(device)
            opt2.zero_grad()
            out = student(xb)
            loss = loss_fn2(out, yb, tb) if teacher_logits_all is not None \
                else F.cross_entropy(out, yb)
            loss.backward()
            opt2.step()

    soft_acc = soft_hard_acc_dlgn(student, x_all_t, y_all_t, device)
    hard_acc = hard_acc_via_eval_forward(student, x_all, y_all, device)
    return {
        'soft_acc': soft_acc, 'hard_acc': hard_acc,
        'gap': soft_acc - hard_acc, 'n_gates': n_gates,
        'stage1_final_bce': stage1_final_bce,
        'teacher_widths': list(teacher_widths),
    }


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
    teacher_acc_before_prune = train_teacher(
        teacher, x_all_t, y_all_t, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device, batch_size=args.batch_size,
    )
    print(f'  [seed={seed}] teacher train hard_acc(枝刈り前)='
          f'{teacher_acc_before_prune:.4f}')
    result['teacher_acc_before_prune'] = teacher_acc_before_prune

    # ---- fan-in=2まで枝刈り(compact native DLGNはN=2専用) ----
    teacher, prune_history = ensure_teacher_fanin(
        teacher, x_all, y_all, n_inputs=NATIVE_N_INPUTS,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    teacher_acc_after_prune = float(
        (teacher.to(device).eval()(x_all_t.to(device)).argmax(-1).cpu()
         == y_all_t).float().mean()
    )
    print(f'  [seed={seed}] teacher fan-in=2枝刈り後 hard_acc='
          f'{teacher_acc_after_prune:.4f}')
    result['teacher_acc_after_prune'] = teacher_acc_after_prune
    result['fanin2_oracle_loss'] = (
        teacher_acc_before_prune - teacher_acc_after_prune
    )
    result['prune_stages'] = prune_history.get('stages', [])

    # ---- transcribe(n_inputs=2、training-free)+ 厳密性assert ----
    ir2, teacher_cpu = build_transcribed_ir(
        teacher, x_all, n_inputs=NATIVE_N_INPUTS, weight_bits=args.weight_bits,
        device=device,
    )
    assert_transcribe_exact(ir2, teacher_cpu, x_all, args.weight_bits)
    print(f'  [seed={seed}] fan-in=2転写 厳密性assert: OK')
    transcribe_hard_acc = _hard_accuracy(ir2, x_all, y_all)
    result['transcribe_hard_acc'] = transcribe_hard_acc
    gates_e = native_gate_count(ir2)
    result['gates_E'] = gates_e
    print(f'  [seed={seed}] 転写直後(磨き前, oracle参照) hard_acc='
          f'{transcribe_hard_acc:.4f}  gates(native E)={gates_e}')

    # teacherロジット(arm B/E+KD/F+KD用。fan-in枝刈り後teacherのロジット)
    teacher_logits_all = make_teacher_logits(teacher_cpu, x_all_t.cpu(), 'cpu')

    # ---- arm A: scratch(compact native, n_inputs=2) ----
    t0 = time.time()
    res_a = run_arm_a(
        x_all_t, y_all_t, x_all, y_all, in_width=in_width, k_classes=k_classes,
        args=args, device=device, seed=seed + 1000,
    )
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] A(scratch native): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} ({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ(compact native, n_inputs=2) ----
    t0 = time.time()
    res_b = run_arm_b(
        x_all_t, y_all_t, teacher_logits_all, x_all, y_all,
        in_width=in_width, k_classes=k_classes, args=args, device=device,
        seed=seed + 2000,
    )
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] B(IO-KD native): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} ({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm E: wiring+gate-init(native16g) → soft磨き ----
    t0 = time.time()
    res_e = run_arm_e(
        ir2, x_all, y_all, x_all_t, y_all_t, in_width=in_width, args=args,
        device=device, teacher_logits_all=None,
    )
    res_e['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E(init native+soft磨き, 本命): "
          f"soft {res_e['soft_acc_before']:.4f}->{res_e['soft_acc_after']:.4f} "
          f"hard {res_e['hard_acc_before']:.4f}->{res_e['hard_acc_after']:.4f} "
          f"({res_e['seconds']}s)")
    result['E'] = res_e

    # ---- arm E+KD: 同じ初期化からKD磨き ----
    t0 = time.time()
    res_ekd = run_arm_e(
        ir2, x_all, y_all, x_all_t, y_all_t, in_width=in_width, args=args,
        device=device, teacher_logits_all=teacher_logits_all,
    )
    res_ekd['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E+KD(init native+KD磨き): "
          f"soft {res_ekd['soft_acc_before']:.4f}->{res_ekd['soft_acc_after']:.4f} "
          f"hard {res_ekd['hard_acc_before']:.4f}->{res_ekd['hard_acc_after']:.4f} "
          f"({res_ekd['seconds']}s)")
    result['E_KD'] = res_ekd

    # ---- arm F: 層別二値蒸留(最小単一ステージ版) ----
    t0 = time.time()
    res_f = run_arm_f(
        teacher_cpu, x_all, y_all, x_all_t, y_all_t, in_width=in_width,
        k_classes=k_classes, args=args, device=device,
        teacher_logits_all=(teacher_logits_all if args.f_kd else None),
    )
    res_f['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] F(層別二値蒸留, 最小単一ステージ版): "
          f"soft={res_f['soft_acc']:.4f} hard={res_f['hard_acc']:.4f} "
          f"gates={res_f['n_gates']} ({res_f['seconds']}s)")
    result['F'] = res_f

    # ---- (参照) arm C: warp構造転写+soft磨き(非native、本命の再利用) ----
    # 参照Cはfan-in=args.fanin_c(既定6、n_inputs=2ではなく)で独立に teacher を
    # 学習し直して転写する(本来のwarp回路と同条件で比較するため。E用に
    # fan-in=2枝刈り済みのteacherを使い回すと、参照Cの回路規模が実測と
    # ずれてしまう)。
    t0 = time.time()
    torch.manual_seed(seed)
    teacher_for_c = ArithTeacherMlp(in_width, args.teacher_hidden, k_classes).to(device)
    train_teacher(
        teacher_for_c, x_all_t, y_all_t, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device, batch_size=args.batch_size,
    )
    teacher_for_c, _ = ensure_teacher_fanin(
        teacher_for_c, x_all, y_all, n_inputs=args.fanin_c,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    ir_c, teacher_c_cpu = build_transcribed_ir(
        teacher_for_c, x_all, n_inputs=args.fanin_c, weight_bits=args.weight_bits,
        device=device,
    )
    assert_transcribe_exact(ir_c, teacher_c_cpu, x_all, args.weight_bits)
    teacher_logits_c = make_teacher_logits(teacher_c_cpu, x_all_t.cpu(), 'cpu')
    gates_c = native_gate_count(ir_c)
    res_c = run_arm_c(
        ir_c, x_all, y_all, x_all, y_all, selection='soft', args=args,
        device=device, teacher_logits_tr=teacher_logits_c,
    )
    res_c['seconds'] = round(time.time() - t0, 2)
    res_c['gates_C'] = gates_c
    print(f"  [seed={seed}] (参照)C(warp構造転写+soft磨き, 非native, "
          f"fanin={args.fanin_c}): "
          f"hard {res_c['hard_acc_before']:.4f}->{res_c['hard_acc_after']:.4f} "
          f"gates={gates_c} ({res_c['seconds']}s)")
    result['C_ref'] = {k: v for k, v in res_c.items() if k != 'recovered_ir'}

    # ---- 派生指標 ----
    hard_a = res_a['hard_acc']
    hard_b = res_b['hard_acc']
    hard_e = res_e['hard_acc_after']
    hard_ekd = res_ekd['hard_acc_after']
    hard_f = res_f['hard_acc']
    hard_c = res_c['hard_acc_after']
    result['lift_E_vs_A'] = hard_e - hard_a
    result['lift_E_vs_B'] = hard_e - hard_b
    result['lift_EKD_vs_E'] = hard_ekd - hard_e
    result['lift_F_vs_A'] = hard_f - hard_a
    result['gap_E_vs_C'] = hard_e - hard_c
    result['gate_ratio_E_over_C'] = gates_e / max(gates_c, 1)

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 100)
    print('コンパクトnative DLGN獲得 — wiring+gate-init(E)/IO-KD(B)/'
          '層別蒸留(F)/init+KD(E+KD) vs scratch(A)、warp非native参照(C) — 結果')
    print('=' * 100)

    def get_hard(arm, r):
        if arm == 'A':
            return r['A']['hard_acc']
        if arm == 'B':
            return r['B']['hard_acc']
        if arm == 'E':
            return r['E']['hard_acc_after']
        if arm == 'E_KD':
            return r['E_KD']['hard_acc_after']
        if arm == 'F':
            return r['F']['hard_acc']
        if arm == 'C_ref':
            return r['C_ref']['hard_acc_after']
        raise ValueError(arm)

    def get_soft(arm, r):
        if arm == 'A':
            return r['A']['soft_acc']
        if arm == 'B':
            return r['B']['soft_acc']
        if arm == 'E':
            return r['E']['soft_acc_after']
        if arm == 'E_KD':
            return r['E_KD']['soft_acc_after']
        if arm == 'F':
            return r['F']['soft_acc']
        if arm == 'C_ref':
            return r['C_ref']['soft_acc_after']
        raise ValueError(arm)

    arms = ['A', 'B', 'E', 'E_KD', 'F', 'C_ref']
    labels = {
        'A': 'A: scratch native',
        'B': 'B: IO-KDのみ native',
        'E': 'E: wiring+gate-init native + soft磨き(本命)',
        'E_KD': 'E+KD: wiring+gate-init native + KD磨き',
        'F': 'F: 層別二値蒸留 native(最小単一ステージ)',
        'C_ref': '(参照)C: warp構造転写+soft磨き(非native、本命)',
    }

    print('| arm | soft_acc(mean±std) | hard_acc(mean±std) |')
    print('|---|---|---|')
    summary: dict = {'arms': {}}
    for arm in arms:
        softs = [get_soft(arm, r) for r in results]
        hards = [get_hard(arm, r) for r in results]
        sm, ss = _mean_std(softs)
        hm, hs = _mean_std(hards)
        print(f'| {labels[arm]} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} |')
        summary['arms'][arm] = {
            'soft_acc_mean': sm, 'soft_acc_std': ss,
            'hard_acc_mean': hm, 'hard_acc_std': hs,
        }

    gates_e = [r['gates_E'] for r in results]
    gates_c = [r['C_ref']['gates_C'] for r in results]
    gem, ges = _mean_std([float(g) for g in gates_e])
    gcm, gcs = _mean_std([float(g) for g in gates_c])
    print(f'\nゲート数: native E={gem:.0f}±{ges:.0f}  '
          f'非native C(warp)={gcm:.0f}±{gcs:.0f}  '
          f'比E/C={gem / max(gcm, 1):.4f}')
    summary['gates_E_mean'] = gem
    summary['gates_C_mean'] = gcm
    summary['gate_ratio_E_over_C_mean'] = gem / max(gcm, 1)

    fanin2_loss = [r['fanin2_oracle_loss'] for r in results]
    flm, fls = _mean_std(fanin2_loss)
    print(f'\nfan-in≤2 oracle損失(teacher-before - teacher-after-prune): '
          f'{flm:+.4f}±{fls:.4f}')
    summary['fanin2_oracle_loss_mean'] = flm
    summary['fanin2_oracle_loss_std'] = fls

    lift_e_a = [r['lift_E_vs_A'] for r in results]
    lift_e_b = [r['lift_E_vs_B'] for r in results]
    lift_ekd_e = [r['lift_EKD_vs_E'] for r in results]
    lift_f_a = [r['lift_F_vs_A'] for r in results]
    gap_e_c = [r['gap_E_vs_C'] for r in results]

    leam, leas = _mean_std(lift_e_a)
    lebm, lebs = _mean_std(lift_e_b)
    lkem, lkes = _mean_std(lift_ekd_e)
    lfam, lfas = _mean_std(lift_f_a)
    gecm, gecs = _mean_std(gap_e_c)

    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift(E - A)      : {leam:+.4f}±{leas:.4f}')
    print(f'lift(E - B)      : {lebm:+.4f}±{lebs:.4f}')
    print(f'lift(E+KD - E)   : {lkem:+.4f}±{lkes:.4f}')
    print(f'lift(F - A)      : {lfam:+.4f}±{lfas:.4f}')
    print(f'gap(E - C_ref)   : {gecm:+.4f}±{gecs:.4f}')

    summary['lift_E_vs_A_mean'] = leam
    summary['lift_E_vs_B_mean'] = lebm
    summary['lift_EKD_vs_E_mean'] = lkem
    summary['lift_F_vs_A_mean'] = lfam
    summary['gap_E_vs_C_mean'] = gecm

    print('\n--- 仮説判定 ---')
    print(f'H(wiring+gate-init(E) > scratch(A)): lift(E-A)={leam:+.4f} '
          f"→ {'支持' if leam > 0 else '不支持'}")
    print(f'H(wiring+gate-init(E) > IO-KD(B)): lift(E-B)={lebm:+.4f} '
          f"→ {'支持' if lebm > 0 else '不支持'}")
    print(f'H(init+KD(E+KD) > init単独(E)): lift(E+KD-E)={lkem:+.4f} '
          f"→ {'支持' if lkem > 0 else '不支持'}")
    print(f'H(層別二値蒸留(F) > scratch(A)): lift(F-A)={lfam:+.4f} '
          f"→ {'支持' if lfam > 0 else '不支持'}")
    print(f'gap(compact native E vs 非native warp C)={gecm:+.4f} '
          f'(負={"native Eはwarp Cに届いていない" if gecm < 0 else "nativeがwarpに並ぶ/超える"})')

    exactness_all = all(r['E']['exactness_ok'] for r in results)
    print(f'\nnative16g復元bit-exact厳密性assert(全seed): '
          f'{"OK" if exactness_all else "NG"}')
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


def _strip_ir(r: dict) -> dict:
    r2 = dict(r)
    if 'C_ref' in r2 and isinstance(r2['C_ref'], dict):
        r2['C_ref'] = {k: v for k, v in r2['C_ref'].items() if k != 'recovered_ir'}
    return r2


def main():
    parser = argparse.ArgumentParser(
        description='コンパクトnative DLGN獲得(wiring+gate-init/KD/層別蒸留) '
                    'vs scratch、算術分類testbed')
    parser.add_argument('--task', type=str, default='adder',
                        choices=('adder', 'multiplier'))
    parser.add_argument('--nbits', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200,
                        help='arm A/B(scratch native DLGN)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=300)
    parser.add_argument('--teacher-prune-epochs', type=int, default=300)
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-hidden', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--lr', type=float, default=5e-2, help='arm A/BのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128,
                        help='arm A/B compact native DLGN隠れ層幅')
    parser.add_argument('--depth', type=int, default=3,
                        help='arm A/B compact native DLGN隠れ層数')
    parser.add_argument('--n-inputs', type=int, default=NATIVE_N_INPUTS,
                        choices=(NATIVE_N_INPUTS,),
                        help='compact native DLGNのLUT fan-in(常に2固定)')
    parser.add_argument('--group-width', type=int, default=4)
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--fanin-c', type=int, default=6, choices=(2, 4, 6),
                        help='参照arm C(warp)のtranscribe fan-in(既定6、'
                             '本来のwarp回路条件を再現)')
    # arm E/E+KD 磨き
    # 既定lrは低すぎると事実上のno-opになる: native16g復元直後は
    # DEFAULT_LOGIT_M=10.0(ir/weights.py)で16ゲートsoftmaxが極端に飽和して
    # おり、lr=0.05では実測grad_norm~1e-7で30epoch経ってもaccが不動
    # (convert_transcribe_mnist_conv.pyの Conv-DLGNで報告された「深い
    # 展開回路の勾配消失」と同根の飽和初期化問題)。lr>=0.5でようやく学習が
    # 進むことを実測で確認し、既定を0.05→1.0・epochsを30→60へ引き上げた
    # (recover-lr 0.02→0.3引き上げと同じ対処)。
    parser.add_argument('--polish-epochs', type=int, default=60)
    parser.add_argument('--polish-lr', type=float, default=1.0)
    # arm C(参照)のrecoverハイパラ(run_arm_c/RecoverConfigが読む)
    parser.add_argument('--recover-epochs', type=int, default=30)
    parser.add_argument('--recover-lr', type=float, default=0.05)
    parser.add_argument('--recover-tau', type=float, default=4.0)
    parser.add_argument('--polish-selection', type=str, default='soft',
                        choices=('soft', 'hard_st'))
    parser.add_argument('--aux', type=str, default='none', choices=('none', 'rkd'))
    parser.add_argument('--rkd-beta', type=float, default=1.0)
    parser.add_argument('--polish-kd', action='store_true',
                        help='参照arm CをKD磨きにする(既定false=CE磨き)')
    # KD共通(arm B/E+KD/F+KD)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--standardize', action='store_true', default=True)
    # arm F固有
    parser.add_argument('--f-depth', type=int, default=2,
                        help='arm Fの各ブロック内DLGN層数')
    parser.add_argument('--f-stage1-epochs', type=int, default=20)
    parser.add_argument('--f-stage1-lr', type=float, default=0.01)
    parser.add_argument('--f-stage2-epochs', type=int, default=30)
    parser.add_argument('--f-stage2-lr', type=float, default=0.01)
    parser.add_argument('--f-kd', action='store_true',
                        help='arm FのStage2をKDにする(既定false=CE)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小width・少epoch、<90s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='outputs/arith_to_native_dlgn')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

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
        args.group_width = 2
        args.polish_epochs = 10
        args.polish_lr = 1.0
        args.recover_epochs = 5
        args.f_stage1_epochs = 3
        args.f_stage2_epochs = 5
        args.batch_size = 32
        args.fanin_c = 6

    k_classes = num_classes_for(args.nbits, args.task)
    print(f'デバイス: {device}')
    print(f'task={args.task} nbits={args.nbits} (入力={2 * args.nbits}bit, '
          f'K={k_classes}クラス, n_patterns={1 << (2 * args.nbits)}) '
          f'seeds={args.seeds}')
    print(f'compact native DLGN(arm A/B/E/F): n_inputs={NATIVE_N_INPUTS} '
          f'width={args.width} depth={args.depth} group_width={args.group_width}')
    print(f'teacher: hidden={args.teacher_hidden} epochs={args.teacher_epochs} '
          f'lr={args.teacher_lr}')
    print(f'参照arm C(warp): fanin={args.fanin_c} weight_bits={args.weight_bits}')

    use_wandb = init_wandb(
        run_name=f'arith_to_native_dlgn_{args.task}_n{args.nbits}',
        config_dict=vars(args), device=device,
        tags=['D12', 'native-dlgn', 'structural-transfer', 'kd',
              'layerwise-distill', args.task],
        group='convert-arith-to-native-dlgn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_acc_before_prune': r['teacher_acc_before_prune'],
                f'seed{seed}/teacher_acc_after_prune': r['teacher_acc_after_prune'],
                f'seed{seed}/fanin2_oracle_loss': r['fanin2_oracle_loss'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/E_hard_acc': r['E']['hard_acc_after'],
                f'seed{seed}/EKD_hard_acc': r['E_KD']['hard_acc_after'],
                f'seed{seed}/F_hard_acc': r['F']['hard_acc'],
                f'seed{seed}/C_ref_hard_acc': r['C_ref']['hard_acc_after'],
                f'seed{seed}/gates_E': r['gates_E'],
                f'seed{seed}/gates_C': r['C_ref']['gates_C'],
                f'seed{seed}/lift_E_vs_A': r['lift_E_vs_A'],
                f'seed{seed}/lift_E_vs_B': r['lift_E_vs_B'],
                f'seed{seed}/lift_EKD_vs_E': r['lift_EKD_vs_E'],
                f'seed{seed}/lift_F_vs_A': r['lift_F_vs_A'],
                f'seed{seed}/gap_E_vs_C': r['gap_E_vs_C'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(
        args.out_dir, f'result_{args.task}_n{args.nbits}_{ts}.json',
    )
    dump = {
        'args': vars(args), 'summary': summary,
        'per_seed': [_strip_ir(r) for r in results],
    }
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
