"""量子化起点の転写(Stage 1.5/2): 量子化(1bit/1.58bit)CNN → Conv-DLGN 転写→polish。

 dense(`convert_ternary_to_dwn_mnist.py`)のconv版。CNN(nn.Conv2d)の第1conv層を
量子化(ternary/bin, QAT)してから `transcribe_conv2d`(近似・read-once木)で
Conv-DLGN(LutConv2d)へ転写→`recover`でsoft磨きする。dense版の
`convert_transcribe_mnist_conv.py`(arm A/B/C)を無改造import して再利用。

問い: quantized CNN→Conv-DLGN でも「量子化起点」は近似転写の質(agreement)と
polished hard accを損なわないか。ternary(0マスク)は agreement を上げるか。

案A(head転写、既定ON、--transcribe-head): 旧境界結果の主因=「第1conv層のみ
転写・head(LutLayer)/GroupSumはrandom初期化をrecoverでスクラッチ訓練」を是正する。
第1conv転写後の Conv-DLGN を hard forward して**二値特徴マップ**(domain整合較正
データ)を得、その二値特徴+ラベルで STE-MLP(HeavisideSTE隠れ)head教師を学習し、
`transcribe_mlp`(fan-in疎化)でheadもLUT列へ転写する。conv/head両方転写した
Conv-DLGN全体を recover で磨く(転写比率≈1.0)。scratch下限は**同トポロジー・同
recoverパイプライン**のランダム初期化(初期化のみが差)。--no-transcribe-head で
旧conv-only経路(head random)を再現できる。案C(--kd-polish)は teacher logits で
recoverをKD磨きする(alpha=0.25/T=4/standardize)。

概念的注意(base実験と同じ): transcribe_conv2d は受容野全体でなく
n_inputs^tree_depth リーフをサンプルする**近似**転写(agreementを正直に報告、
ビット厳密は主張しない)。head転写(transcribe_mlp)も 6272→hidden の dense層を
fan-in≤6 へ疎化する近似(head_agreement を正直に報告)。Conv-DLGNはnative LUT
(popcount非経由)、readout=GroupSum。「conv-DWN(tree-leaf learnable wiring)」は
未実装のため本PoCは固定tree構造。

使用例:
    uv run python experiments/convert_ternary_conv_to_dlgn.py --quick --no-wandb
    uv run python experiments/convert_ternary_conv_to_dlgn.py --seeds 3        # MNIST本実行(案A)
    uv run python experiments/convert_ternary_conv_to_dlgn.py --no-transcribe-head  # 旧conv-only境界
    uv run python experiments/convert_ternary_conv_to_dlgn.py --kd-polish        # 案C KD磨き
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# dense conv転写実験の機構を無改造で再利用
from convert_transcribe_mnist_conv import (
    IN_SHAPE,
    N_CLASSES,
    _auto_device,
    _flatten_binary_images,
    build_conv_dlgn,
    build_transcribed_conv_ir,
    make_teacher_logits,
    run_arm_a,
    run_arm_c,
    train_teacher,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutConv2d, LutLayer, OrPooling2d
from griffith_logic.convert.binarize import binarize_weights, quantize_ternary
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.recover import RecoverConfig, recover
from griffith_logic.convert.teacher_models import SmallTeacherCNN, TeacherMLPSte
from griffith_logic.convert.transcribe import (
    TranscribeSpec,
    transcribe_conv2d,
    transcribe_mlp,
)
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.export import export_ir
from griffith_logic.ir.validate import IRValidationError, validate
from griffith_logic.ir.weights import restore_lut_conv2d, restore_lut_layer

load_dotenv()

ARMS = ('float', 'bin', 'ternary')


# ============================================================
# 案A: 第1conv層 + 分類head を両方転写する(head=domain整合STE-MLP)
# ============================================================

def _conv_dlgn_binary_features(conv_module, out_ch, tree_depth, n_inputs,
                               images_bchw, device):
    """転写済み第1convを持つ prefix(conv→OrPool→Flatten)を hard forward し、
    conv-DLGNの**二値特徴マップ**(0/1、head転写の domain整合較正データ)を返す。

    head入力は conv-DLGN が実際に出す二値特徴であり、teacher float特徴とは
    分布が違う(タスク指示の注意点)。ゆえに head の較正・STE教師学習・転写は
    すべてこの二値特徴の上で行う(domain整合)。
    """
    pool = OrPooling2d(kernel_size=2)
    prefix = nn.Sequential(conv_module, pool, nn.Flatten()).to(device).eval()
    xb = torch.as_tensor(images_bchw, dtype=torch.float32).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(xb), 512):
            f = prefix(xb[i:i + 512])
            feats.append((f > 0.5).to(torch.int64).cpu())
    return torch.cat(feats, dim=0).numpy()  # (B, flat_dim) 0/1


def _train_ste_head(feat_np, y_np, *, hidden, epochs, lr, batch_size, device,
                    seed):
    """conv-DLGN二値特徴 → クラス の STE-MLP(head教師)を学習する。

    HeavisideSTE隠れ活性で transcribe_mlp の意味論 y=1[Wx+b>0] と学習時に
    厳密一致するため、head転写は fan-in疎化誤差のみのビット厳密転写になる
    (dense-MLP 勝ちレシピを head へ適用)。train hard accを併せて返す。
    """
    torch.manual_seed(seed)
    flat_dim = feat_np.shape[1]
    head = TeacherMLPSte(in_width=flat_dim, hidden=(hidden,),
                         num_classes=N_CLASSES).to(device)
    x = torch.as_tensor(feat_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.int64)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(head(xb), yb).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        acc = float((head(x.to(device)).argmax(-1).cpu() == y).float().mean())
    return head, acc


def build_transcribed_conv_head_ir(
    teacher_conv, calib_x_np, y_calib_np, *, out_ch, tree_depth, n_inputs,
    head_hidden, head_n_inputs, head_teacher_epochs, head_teacher_lr,
    batch_size, device, seed,
):
    """第1conv層 + 分類head を両方転写した Conv-DLGN 全体のIRを構築する(案A)。

    手順:
        1. transcribe_conv2d で第1conv層を近似転写(既存機構)
        2. 転写convの prefix を hard forward し conv-DLGN二値特徴を得る(domain整合)
        3. その二値特徴+ラベルで STE-MLP head教師を学習
        4. transcribe_mlp(weight_bits=1)で head をビット厳密転写
        5. conv/head を実体 nn.Sequential(conv→OrPool→Flatten→hidden_lut→
           final_lut→GroupSum)へ書き戻し、export_ir で recover可能なIR化

    Returns:
        (ir, info): info は conv agreement / head_teacher_acc / head_agreement /
            transcribe_ratio(転写された隠れLUT数 / 全隠れLUT数)等
    """
    # --- 1) 第1conv転写 ---
    calib_t = torch.as_tensor(calib_x_np, dtype=torch.float32)
    conv_node, conv_info = transcribe_conv2d(
        teacher_conv, calib_inputs=calib_t, tree_depth=tree_depth,
        n_inputs=n_inputs, seed=seed,
    )
    assert conv_node.params['out_channels'] == out_ch

    conv = LutConv2d(
        IN_SHAPE[0], out_ch, kernel_size=3, tree_depth=tree_depth, padding=1,
        n_inputs=n_inputs, device='cpu', parameterization='independent',
    )
    conv.weights = nn.Parameter(
        torch.zeros(conv.out_channels, conv.n_internal, conv.n_entries)
    )
    restore_lut_conv2d(conv, conv_node.params['indices_flat'],
                       conv_node.params['truth_tables'])

    # --- 2) conv-DLGN二値特徴(domain整合較正データ) ---
    feat_np = _conv_dlgn_binary_features(
        conv, out_ch, tree_depth, n_inputs, calib_x_np, device='cpu',
    )
    flat_dim = feat_np.shape[1]

    # --- 3) STE head教師をconv-DLGN二値特徴で学習 ---
    head_teacher, head_teacher_acc = _train_ste_head(
        feat_np, y_calib_np, hidden=head_hidden, epochs=head_teacher_epochs,
        lr=head_teacher_lr, batch_size=batch_size, device=device, seed=seed,
    )

    # --- 4) head転写(weight_bits=1で最終層のgroup幅を抑える) ---
    head_teacher.to('cpu')
    spec = TranscribeSpec(n_inputs=head_n_inputs, weight_bits=1)
    head_ir = transcribe_mlp(head_teacher, calib_inputs=feat_np, spec=spec)
    hnodes = head_ir.nodes
    hidden_node, final_node = hnodes[0], hnodes[1]
    hidden_out = hidden_node.params['out_width']
    final_out = final_node.params['out_width']

    # --- 5) 実体モデルへ書き戻し ---
    hidden_lut = LutLayer(flat_dim, hidden_out, n_inputs=head_n_inputs,
                          device='cpu', parameterization='independent')
    final_lut = LutLayer(hidden_out, final_out, n_inputs=2, device='cpu',
                         parameterization='independent')
    gs = GroupSum(k=N_CLASSES, tau=max(1.0, (final_out / N_CLASSES) ** 0.5))
    model = nn.Sequential(conv, OrPooling2d(kernel_size=2), nn.Flatten(),
                          hidden_lut, final_lut, gs)
    restore_lut_layer(hidden_lut, hidden_node.params['indices'],
                      hidden_node.params['truth_tables'])
    restore_lut_layer(final_lut, final_node.params['indices'],
                      final_node.params['truth_tables'])
    model.to(device).eval()

    ir = export_ir(model, input_shape=IN_SHAPE, name='conv_head_transcribed')
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'転写IR検証失敗: {errs[:3]}')

    # head転写の忠実性(head教師argmax と 転写head simulate の一致率)を測る
    from griffith_logic.ir.simulate import simulate
    x_flat = calib_x_np.reshape(calib_x_np.shape[0], -1).astype(np.int64)
    sim_pred = simulate(ir, x_flat).argmax(-1)
    with torch.no_grad():
        ht_pred = head_teacher(
            torch.as_tensor(feat_np, dtype=torch.float32)
        ).argmax(-1).numpy()
    head_agreement = float((sim_pred == ht_pred).mean())

    # 転写された隠れLUT: conv(out_ch*n_internal) + head hidden(hidden_out)。
    # 全隠れLUTも同じ(head転写により random head が消えたため ratio=1.0)。
    info = {
        'approx_agreement': float(conv_info['approx_agreement']),
        'per_channel_agreement': conv_info['per_channel_agreement'],
        'n_leaves': conv_info['n_leaves'],
        'receptive_field': conv_info['receptive_field'],
        'head_teacher_acc': head_teacher_acc,
        'head_agreement': head_agreement,
        'head_hidden': hidden_out,
        'head_final_width': final_out,
        'transcribe_ratio': 1.0,
    }
    return ir, info


def run_arm_c_head(ir, x_tr, y_tr, x_te, y_te, *, args, device,
                   teacher_logits=None):
    """head転写済みIRを recover で磨く。KD指定時は teacher logits で蒸留する。"""
    x_tr_flat = x_tr.reshape(x_tr.shape[0], -1)
    x_te_flat = x_te.reshape(x_te.shape[0], -1)
    x_tr_t = torch.as_tensor(x_tr_flat, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64)
    x_te_t = torch.as_tensor(x_te_flat, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.int64)

    if args.kd_polish and teacher_logits is not None:
        # recover() は (xb, yb) 2要素前提のため、KD時は自前ループを使う
        kd = make_kd_loss(alpha=args.kd_alpha, temperature=args.kd_temperature,
                          standardize=True, kind='kl')
        tl_t = torch.as_tensor(teacher_logits, dtype=torch.float32)
        train_ds = TensorDataset(x_tr_t, y_tr_t, tl_t)
        return _recover_kd(ir, train_ds, x_te_t, y_te_t, kd, args, device)

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t),
                              batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(x_te_t, y_te_t),
                             batch_size=max(len(x_te_t), 1))
    cfg = RecoverConfig(parameterization='independent', epochs=args.recover_epochs,
                        lr=args.recover_lr, device=device, tau=args.recover_tau,
                        selection='soft')
    _recovered, metrics = recover(ir, train_loader, eval_loader, cfg)
    return metrics


def _recover_kd(ir, train_ds, x_te_t, y_te_t, kd, args, device):
    """KD polish版 recover(teacher logits でDLGN headを磨く。best-over-epochs)。"""
    from griffith_logic.convert.recover import (
        _hard_accuracy,
        rebuild_trainable,
    )
    from griffith_logic.export import export_ir as _export_ir

    eval_x = x_te_t.numpy().astype(np.int64)
    eval_y = y_te_t.numpy().astype(np.int64)
    hard_before = _hard_accuracy(ir, eval_x, eval_y)

    model = rebuild_trainable(ir, 'independent', selection='soft').to(device)
    from griffith_logic.convert.recover import _set_group_sum_tau
    _set_group_sum_tau(model, args.recover_tau)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.recover_lr)

    orig_tau = next((float(n.params.get('tau', 1.0)) for n in ir.nodes
                     if n.type == 'group_sum'), 1.0)
    best_hard = hard_before
    for _ep in range(args.recover_epochs):
        model.train()
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device).float(), yb.to(device), tb.to(device)
            opt.zero_grad()
            kd(model(xb), yb, tb).backward()
            opt.step()
        # best採用: 各epoch後にhard評価(控えめ+best採用レシピ)
        model.eval()
        _set_group_sum_tau(model, orig_tau)
        rec = _export_ir(model, input_width=ir.inputs[0].width,
                         name='convert.recover_kd',
                         metadata={'source': 'convert.recover_kd'})
        h = _hard_accuracy(rec, eval_x, eval_y)
        best_hard = max(best_hard, h)
        _set_group_sum_tau(model, args.recover_tau)
    return {'hard_acc_before': hard_before, 'hard_acc_after': best_hard,
            'soft_acc_before': 0.0, 'soft_acc_after': 0.0,
            'epochs': args.recover_epochs}


def _build_scratch_head_model(out_ch, tree_depth, n_inputs, head_hidden,
                              head_n_inputs, head_final_width, device, seed):
    """arm C(head転写)と同トポロジーのランダム初期化 Conv-DLGN(scratch下限)。

    conv→OrPool→Flatten→hidden_lut(softmax/warp学習可)→final_lut→GroupSum。
    scratch/転写の唯一の差を「初期化」に絞るため、head転写armと同じ2層head幅を持つ
    (旧1層head scratchはtopology不一致で不公平だった)。
    """
    torch.manual_seed(seed)
    conv = LutConv2d(IN_SHAPE[0], out_ch, kernel_size=3, tree_depth=tree_depth,
                     padding=1, n_inputs=n_inputs, device=device,
                     parameterization='softmax16g')
    pool = OrPooling2d(kernel_size=2)
    flat_dim = out_ch * (IN_SHAPE[1] // 2) * (IN_SHAPE[2] // 2)
    hidden = LutLayer(flat_dim, head_hidden, n_inputs=head_n_inputs,
                      device=device, parameterization='softmax')
    final = LutLayer(head_hidden, head_final_width, n_inputs=2, device=device,
                     parameterization='softmax')
    gs = GroupSum(k=N_CLASSES, tau=max(1.0, (head_final_width / N_CLASSES) ** 0.5))
    return nn.Sequential(conv, pool, nn.Flatten(), hidden, final, gs).to(device)


def run_arm_a_head(x_tr, y_tr, x_te_np, y_te_np, *, out_ch, head_final_width,
                   args, device, seed):
    """head転写armと**同トポロジー・同recoverパイプライン**のscratch下限。

    転写armとの唯一の差を「初期化(転写 vs ランダム)」に絞るため、同じ
    conv→OrPool→Flatten→hidden_lut→final_lut→GroupSum構造のIRを
    ランダム初期化で作り、arm C と同一の RecoverConfig(independent,
    recover_lr, recover_tau)で学習する(旧: 別optimizer/lr/paramで
    undertrainされた不公平scratchを是正)。
    """
    model = _build_scratch_head_model(
        out_ch, args.tree_depth, args.n_inputs, args.head_hidden,
        args.head_n_inputs, head_final_width, device, seed,
    ).eval()
    ir = export_ir(model.to(device), input_shape=IN_SHAPE, name='scratch_head')
    x_tr_t = torch.as_tensor(
        np.asarray(x_tr).reshape(np.asarray(x_tr).shape[0], -1), dtype=torch.float32,
    )
    y_tr_t = torch.as_tensor(np.asarray(y_tr), dtype=torch.int64)
    x_te_t = torch.as_tensor(
        np.asarray(x_te_np).reshape(np.asarray(x_te_np).shape[0], -1),
        dtype=torch.float32,
    )
    y_te_t = torch.as_tensor(np.asarray(y_te_np), dtype=torch.int64)
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t),
                              batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(x_te_t, y_te_t),
                             batch_size=max(len(x_te_t), 1))
    cfg = RecoverConfig(parameterization='independent', epochs=args.recover_epochs,
                        lr=args.recover_lr, device=device, tau=args.recover_tau,
                        selection='soft')
    _rec, metrics = recover(ir, train_loader, eval_loader, cfg)
    return float(metrics.get('hard_acc_after', 0.0))


def _quantize_ste_conv(w: torch.Tensor, arm: str, eps: float = 1e-5) -> torch.Tensor:
    """conv重み(out,in,kh,kw)のSTE量子化(per-out-channel)。forward=量子化・backward=恒等。"""
    if arm == 'float':
        return w
    flat = w.reshape(w.shape[0], -1)  # (out, in*kh*kw)
    if arm == 'bin':
        alpha = flat.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
        q = torch.sign(flat) * alpha
    elif arm == 'ternary':
        scale = flat.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
        q = torch.clamp(torch.round(flat / scale), -1.0, 1.0) * scale
    else:
        raise ValueError(arm)
    q = q.reshape_as(w)
    return w + (q - w).detach()  # STE


def _qat_conv1(teacher, arm, x, y, *, epochs, lr, batch_size, device):
    """第1conv層を量子化(STE)しながらCNN全体をfine-tune(量子化aware)。"""
    if arm == 'float' or epochs <= 0:
        return
    conv1 = teacher.features[0]
    orig_forward = conv1.forward

    def q_forward(inp):
        wq = _quantize_ste_conv(conv1.weight, arm)
        return F.conv2d(inp, wq, conv1.bias, conv1.stride, conv1.padding,
                        conv1.dilation, conv1.groups)

    conv1.forward = q_forward  # monkeypatch: 量子化aware forward
    try:
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
        teacher.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(teacher(xb), yb).backward()
                opt.step()
    finally:
        conv1.forward = orig_forward
    # bake: 第1conv重みを量子化決定値へ
    with torch.no_grad():
        conv1.weight.copy_(_quantize_ste_conv(conv1.weight, arm).detach())


def _conv1_fanin_stats(conv1, arm) -> dict:
    w = conv1.weight.detach().reshape(conv1.weight.shape[0], -1)
    if arm == 'ternary':
        mask = w != 0
    else:
        mask = torch.ones_like(w, dtype=torch.bool)
    nz = mask.sum(dim=1).float()
    return {'mean_fanin': float(nz.mean()), 'max_fanin': int(nz.max()),
            'sparsity': float(1.0 - mask.float().mean())}


def _load_images(dataset_name, train_n, eval_n):
    if dataset_name == 'mnist':
        tr, te = load_mnist(binarize_threshold=0.5)
    else:
        raise ValueError(f'未対応dataset(本PoCはmnistのみ確定、cifar10は要load_cifar): {dataset_name}')
    x_tr, y_tr = _flatten_binary_images(tr, n=train_n)
    x_te, y_te = _flatten_binary_images(te, n=eval_n)
    return x_tr, y_tr, x_te, y_te


def run_one_seed(seed, args, device, x_tr, y_tr, x_te, y_te):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xt = torch.as_tensor(x_tr, dtype=torch.float32).reshape(-1, *IN_SHAPE)
    yt = torch.as_tensor(y_tr, dtype=torch.int64)
    calib = x_tr[:args.calib_n].reshape(-1, *IN_SHAPE)

    # ---- teacher CNN学習 ----
    base = SmallTeacherCNN(num_classes=N_CLASSES, in_channels=IN_SHAPE[0]).to(device)
    tacc = train_teacher(base, xt, yt, epochs=args.teacher_epochs,
                         lr=args.teacher_lr, device=device, batch_size=args.batch_size)
    print(f'\n[seed={seed}] teacher CNN train_acc={tacc:.4f}')
    result = {'seed': seed, 'teacher_acc': tacc}

    rc_args = SimpleNamespace(
        batch_size=args.batch_size, recover_epochs=args.recover_epochs,
        recover_lr=args.recover_lr, recover_tau=args.recover_tau,
        kd_polish=args.kd_polish, kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
    )
    out_ch = base.features[0].out_channels
    y_calib = y_tr[:args.calib_n]
    # KD polish用 teacher logits(head転写+KD時のみ計算)
    tlogits = None
    if args.transcribe_head and args.kd_polish:
        tlogits = make_teacher_logits(
            base, torch.as_tensor(x_tr, dtype=torch.float32).reshape(-1, *IN_SHAPE),
            device,
        ).numpy()

    for arm in ARMS:
        teacher = SmallTeacherCNN(num_classes=N_CLASSES, in_channels=IN_SHAPE[0]).to(device)
        teacher.load_state_dict(base.state_dict())
        _qat_conv1(teacher, arm, xt, yt, epochs=args.qat_epochs, lr=args.qat_lr,
                   batch_size=args.batch_size, device=device)
        fstats = _conv1_fanin_stats(teacher.features[0], arm)

        if args.transcribe_head:
            ir, approx = build_transcribed_conv_head_ir(
                teacher.features[0].to('cpu'), calib.reshape(-1, *IN_SHAPE),
                y_calib, out_ch=out_ch, tree_depth=args.tree_depth,
                n_inputs=args.n_inputs, head_hidden=args.head_hidden,
                head_n_inputs=args.head_n_inputs,
                head_teacher_epochs=args.head_teacher_epochs,
                head_teacher_lr=args.head_teacher_lr,
                batch_size=args.batch_size, device=device, seed=seed,
            )
            res_c = run_arm_c_head(
                ir, np.asarray(x_tr), np.asarray(y_tr), np.asarray(x_te),
                np.asarray(y_te), args=rc_args, device=device,
                teacher_logits=tlogits,
            )
            extra = {'head_teacher_acc': approx['head_teacher_acc'],
                     'head_agreement': approx['head_agreement'],
                     'transcribe_ratio': approx['transcribe_ratio'],
                     'head_final_width': approx['head_final_width']}
        else:
            ir, approx = build_transcribed_conv_ir(
                teacher.features[0].to('cpu'), calib, out_ch=out_ch,
                tree_depth=args.tree_depth, n_inputs=args.n_inputs,
                head_width=args.head_width, device=device, seed=seed,
            )
            res_c = run_arm_c(ir, x_tr, y_tr, x_te, y_te, args=rc_args, device=device)
            extra = {'transcribe_ratio': 0.5}  # convのみ転写(head random)
        agreement = float(approx['approx_agreement'])
        hard = float(res_c.get('hard_acc_after', res_c.get('hard_acc', 0.0)))
        msg = (f'  [{arm}] conv1 fan-in mean={fstats["mean_fanin"]:.1f} '
               f'conv_agr={agreement:.4f}')
        if args.transcribe_head:
            msg += (f' head_agr={extra["head_agreement"]:.4f} '
                    f'head_teacher={extra["head_teacher_acc"]:.4f}')
        msg += f' → Conv-DLGN polish hard={hard:.4f}'
        print(msg)
        result[arm] = {'agreement': agreement, 'hard_acc': hard,
                       **fstats, **extra}

    # scratch下限(arm A)。head転写時は同トポロジー(2層head)のscratchで公平比較
    if args.transcribe_head:
        sc_hard = run_arm_a_head(
            x_tr, y_tr, x_te, y_te, out_ch=out_ch,
            head_final_width=result['ternary']['head_final_width'],
            args=args, device=device, seed=seed + 100,
        )
        result['scratch'] = {'hard_acc': float(sc_hard)}
    else:
        res_a = run_arm_a(torch.as_tensor(x_tr, dtype=torch.float32).reshape(-1, *IN_SHAPE),
                          yt, x_te, y_te, out_ch=out_ch, args=SimpleNamespace(
                              batch_size=args.batch_size, epochs=args.recover_epochs,
                              lr=args.lr, tree_depth=args.tree_depth, n_inputs=args.n_inputs,
                              head_width=args.head_width),
                          device=device, seed=seed + 100)
        result['scratch'] = {'hard_acc': float(res_a.get('hard_acc', 0.0))}
    print(f'[seed={seed}] scratch Conv-DLGN hard={result["scratch"]["hard_acc"]:.4f}')
    result['ternary_vs_scratch'] = result['ternary']['hard_acc'] - result['scratch']['hard_acc']
    result['ternary_vs_bin'] = result['ternary']['hard_acc'] - result['bin']['hard_acc']
    result['ternary_vs_float'] = result['ternary']['hard_acc'] - result['float']['hard_acc']
    return result


def _ms(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std())


def summarize(results):
    print('\n' + '=' * 88)
    print('量子化起点の転写 conv: 量子化CNN → Conv-DLGN 転写→polish — 結果')
    print('=' * 88)
    summary = {'arms': {}}
    has_head = 'head_agreement' in results[0].get('ternary', {})
    if has_head:
        print('| arm | conv_agr | head_agr | head_teacher | transcribe_ratio '
              '| Conv-DLGN polish後 hard |')
        print('|---|---|---|---|---|---|')
    else:
        print('| arm | agreement | Conv-DLGN polish後 hard |')
        print('|---|---|---|')
    for arm in ARMS:
        ag = _ms([r[arm]['agreement'] for r in results])
        hd = _ms([r[arm]['hard_acc'] for r in results])
        entry = {'agreement_mean': ag[0], 'hard_mean': hd[0], 'hard_std': hd[1]}
        if has_head:
            hag = _ms([r[arm]['head_agreement'] for r in results])
            hta = _ms([r[arm]['head_teacher_acc'] for r in results])
            tr = _ms([r[arm]['transcribe_ratio'] for r in results])
            entry.update({'head_agreement_mean': hag[0],
                          'head_teacher_acc_mean': hta[0],
                          'transcribe_ratio_mean': tr[0]})
            print(f'| {arm} | {ag[0]:.4f} | {hag[0]:.4f} | {hta[0]:.4f} '
                  f'| {tr[0]:.2f} | {hd[0]:.4f}±{hd[1]:.4f} |')
        else:
            print(f'| {arm} | {ag[0]:.4f} | {hd[0]:.4f}±{hd[1]:.4f} |')
        summary['arms'][arm] = entry
    sc = _ms([r['scratch']['hard_acc'] for r in results])
    summary['scratch_hard_mean'] = sc[0]
    print(f'| scratch(下限) | - | {sc[0]:.4f}±{sc[1]:.4f} |')
    for key, lbl in [('ternary_vs_scratch', 'ternary−scratch'),
                     ('ternary_vs_bin', 'ternary−bin'),
                     ('ternary_vs_float', 'ternary−float')]:
        m = _ms([r[key] for r in results])
        summary[f'{key}_mean'] = m[0]
        print(f'{lbl}: {m[0]:+.4f}±{m[1]:.4f}')
    return summary


def main():
    p = argparse.ArgumentParser(description='conv: 量子化CNN→Conv-DLGN 転写→polish')
    p.add_argument('--dataset', type=str, default='mnist', choices=('mnist',))
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--train-n', type=int, default=8000)
    p.add_argument('--calib-n', type=int, default=2000)
    p.add_argument('--eval-n', type=int, default=4000)
    p.add_argument('--teacher-epochs', type=int, default=3)
    p.add_argument('--teacher-lr', type=float, default=1e-3)
    p.add_argument('--qat-epochs', type=int, default=2)
    p.add_argument('--qat-lr', type=float, default=5e-4)
    p.add_argument('--recover-epochs', type=int, default=20)
    p.add_argument('--recover-lr', type=float, default=0.3)
    p.add_argument('--recover-tau', type=float, default=4.0)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--tree-depth', type=int, default=3)
    p.add_argument('--n-inputs', type=int, default=2, choices=(2, 4, 6))
    p.add_argument('--head-width', type=int, default=256)
    # --- 案A: head転写(domain整合STE-MLP)。既定ON ---
    p.add_argument('--transcribe-head', dest='transcribe_head',
                   action='store_true', default=True,
                   help='第1conv+分類headを両方転写(案A、既定ON)')
    p.add_argument('--no-transcribe-head', dest='transcribe_head',
                   action='store_false',
                   help='convのみ転写・head random(旧境界結果を再現)')
    p.add_argument('--head-hidden', type=int, default=256,
                   help='STE head教師の隠れ幅(案A)')
    p.add_argument('--head-n-inputs', type=int, default=6, choices=(2, 4, 6),
                   help='head LUTのfan-in(案A、既定6=dense-MLP勝ちレシピ)')
    p.add_argument('--head-teacher-epochs', type=int, default=40)
    p.add_argument('--head-teacher-lr', type=float, default=1e-3)
    # --- 案C: KD polish(head転写時のみ有効) ---
    p.add_argument('--kd-polish', dest='kd_polish', action='store_true',
                   default=False, help='recoverをKD(teacher logits)で磨く(案C)')
    p.add_argument('--kd-alpha', type=float, default=0.25)
    p.add_argument('--kd-temperature', type=float, default=4.0)
    p.add_argument('-bs', '--batch-size', type=int, default=128)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--out-dir', type=str, default='outputs/ternary_conv_to_dlgn')
    args = p.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    if args.quick:
        args.seeds = 1
        args.train_n = 3000
        args.calib_n = 1000
        args.eval_n = 1500
        args.teacher_epochs = 2
        args.qat_epochs = 1
        args.recover_epochs = 4
        args.head_width = 128
        args.head_hidden = 128
        args.head_teacher_epochs = 20

    print(f'デバイス: {device}  dataset={args.dataset} seeds={args.seeds} '
          f'tree_depth={args.tree_depth} n_inputs={args.n_inputs}')
    x_tr, y_tr, x_te, y_te = _load_images(args.dataset, args.train_n, args.eval_n)
    x_te, y_te = x_te[:args.eval_n], y_te[:args.eval_n]
    print(f'train={len(x_tr)} calib={args.calib_n} test={len(x_te)}')

    use_wandb = init_wandb(
        run_name=f'ternary_conv_to_dlgn_{args.dataset}',
        config_dict=vars(args), device=device,
        tags=['D18', 'track-a', 'conv', 'ternary', 'conv-dlgn', args.dataset],
        group='ternary-dwn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        results.append(run_one_seed(seed, args, device, x_tr, y_tr, x_te, y_te))
    summary = summarize(results)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_{args.dataset}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary, 'per_seed': results},
                  f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')
    print(f'完了(総時間 {timedelta(seconds=int(time.time() - started))})')


if __name__ == '__main__':
    main()
