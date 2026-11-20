"""CNN→Conv-DLGN構造転写PoC。convert_arith_structural_transfer.pyの
arm規約(A scratch / B IO-KDのみ / C 構造転写+soft磨き)をConvへ拡張する。

背景:(convert_arith_structural_transfer.py)はMLP→DLGNの構造転写
(transcribe_mlp)+soft磨きがscratch/IO-KDよりHARD accで勝ることを示した。
本PoCはこのパラダイムをLinearからConvへ拡張し、同じ問いを立てる:
    CNN(nn.Conv2d)→Conv-DLGN(LutConv2d)の構造転写初期化は、
    scratch/IO-KDに対してHARD accで優位か?

概念的な誠実さ(タスク指示、README必読の注意点):
    Conv-DLGNのtree convは受容野のうちn_inputs^tree_depth本のリーフしか
    read-once N分木でサンプルしない。一方nn.Conv2dのカーネルは受容野
    **全体**を見る閾値関数であり、一般にread-once木では厳密表現できない
    (fan-in制約前のMLP転写と同じ構図)。ゆえに`transcribe_conv2d`
    (convert/transcribe.py)は**近似**構造転写であり、本PoCはその近似初期値
    からのsoft磨きがscratch/IO-KDに勝るかを見る(ビット厳密の転写を主張しない)。

比較する3アーム(A/B/Cに対応、C_hard_st/DはMLP版のみでconv版は範囲外):
    A(scratch)   ランダム初期化Conv-DLGN(LutConv2d→OrPool→LutLayer→GroupSum)
                 をラベルのみCEで学習。
    B(IO-KDのみ) 同じランダム初期化Conv-DLGNを、teacherロジットKDで学習。
                 構造転写なし。
    C(本命)      SmallTeacherCNNの**第1conv層のみ**をtranscribe_conv2dで
                 近似構造転写し、そのLutConv2dへ書き戻した上で、残りの層
                 (OrPool→LutLayer→GroupSum)はランダム初期化のまま、
                 全体をrecover()でsoft磨きする。

厳密性ではなく内部整合性: transcribe_conv2d自体の内部整合性
(構築した真理値表をsimulateが厳密に実現すること)はtests/l1_math/
test_transcribe_conv.pyで担保する。本スクリプトでは近似誤差
(較正データ上のagreement rate)を測定・報告するのみで、0を主張しない。

使用例:
    # 煙実行(数十秒、wandb無効、小データ・少epoch)
    uv run python experiments/convert_transcribe_mnist_conv.py --quick --no-wandb

    # 本実行(wandb online)
    uv run python experiments/convert_transcribe_mnist_conv.py
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
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutConv2d, LutLayer, OrPooling2d
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.recover import RecoverConfig, recover
from griffith_logic.convert.teacher_models import SmallTeacherCNN
from griffith_logic.convert.transcribe import transcribe_conv2d
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.export import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate
from griffith_logic.ir.weights import restore_lut_conv2d

load_dotenv()

IN_SHAPE = (1, 28, 28)
N_CLASSES = 10


# ============================================================
# データ
# ============================================================

def _flatten_binary_images(dataset, n: int | None = None):
    """二値化MNISTを(N,1,28,28)float32の0/1と(N,)ラベルへ展開する。"""
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, 1, 28, 28), dtype=np.float32)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.numpy()
        ys[i] = int(label)
    assert set(np.unique(xs)).issubset({0.0, 1.0}), '二値化MNISTのはずが0/1以外を検出'
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
# teacher: SmallTeacherCNNを軽く学習(構造転写元、arm C)
# ============================================================

def train_teacher(
    teacher: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
    epochs: int, lr: float, device: str, batch_size: int = 128,
) -> float:
    """SmallTeacherCNNをMNISTで学習する。最終train hard accを返す。"""
    teacher.to(device).train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(teacher(xb), yb)
            loss.backward()
            opt.step()
    teacher.eval()
    with torch.no_grad():
        pred = teacher(x.to(device)).argmax(-1).cpu()
    return float((pred == y).float().mean())


def make_teacher_logits(teacher: nn.Module, x: torch.Tensor, device: str,
                        batch: int = 1000) -> torch.Tensor:
    teacher.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(teacher(x[i:i + batch].to(device)).detach().cpu())
    return torch.cat(outs, dim=0)


# ============================================================
# モデル: 小さなConv-DLGN(arm A/B/C共通トポロジー)
# ============================================================

def build_conv_dlgn(
    *, out_ch: int, tree_depth: int, n_inputs: int, head_width: int,
    device: str, seed: int | None = None,
    conv_parameterization: str = 'softmax16g',
) -> nn.Sequential:
    """LutConv2d(1層)→OrPool→Flatten→LutLayer(1層)→GroupSumの小Conv-DLGN。

    conv_parameterizationは既定'softmax16g'(n_inputs=2でscratch/IO-KD学習に
    適した全16ゲート表現)。arm Cはtranscribe_conv2dが'independent'を要求する
    ため、呼び出し側でLutConv2dを別途構築しrestore_lut_conv2dで上書きする。
    """
    if seed is not None:
        torch.manual_seed(seed)
    c, h, w = IN_SHAPE
    conv = LutConv2d(
        c, out_ch, kernel_size=3, tree_depth=tree_depth, padding=1,
        n_inputs=n_inputs, device=device,
        parameterization=conv_parameterization,
    )
    pool = OrPooling2d(kernel_size=2)  # 28x28 -> 14x14
    h_out, w_out = h // 2, w // 2
    flat_dim = out_ch * h_out * w_out
    out_final = N_CLASSES * 4
    # LutLayerはn_inputs=2でparameterization='softmax'を渡すと自動的に
    # 16ゲートモード(_use_16gate)になる(LutConv2dの'softmax16g'相当)。
    # LutLayer.PARAMETERIZATIONSに'softmax16g'という名前は存在しない。
    head = LutLayer(flat_dim, out_final, n_inputs=n_inputs, device=device,
                    parameterization='softmax')
    gs = GroupSum(k=N_CLASSES, tau=max(1.0, (out_final / N_CLASSES) ** 0.5))
    model = nn.Sequential(conv, pool, nn.Flatten(), head, gs).to(device)
    return model


@torch.no_grad()
def hard_acc_via_export(model: nn.Module, x_np: np.ndarray, y_np: np.ndarray,
                        batch: int = 2000) -> float:
    """モデル(eval)をexport_irでIR化し、simulateのargmaxでhard accを測る。"""
    model.eval()
    ir = export_ir(model, input_shape=IN_SHAPE, name='conv_dlgn')
    correct = 0
    x_flat = x_np.reshape(x_np.shape[0], -1).astype(np.int64)
    for i in range(0, len(x_flat), batch):
        scores = simulate(ir, x_flat[i:i + batch])
        pred = scores.argmax(axis=-1)
        correct += int((pred == y_np[i:i + batch]).sum())
    return correct / len(x_flat)


@torch.no_grad()
def soft_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str,
            batch: int = 2000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch].to(device))
        pred = out.argmax(-1).cpu()
        correct += int((pred == y[i:i + batch]).sum())
    model.train()
    return correct / len(x)


# ============================================================
# arm A: scratch
# ============================================================

def run_arm_a(x_tr, y_tr, x_ev_np, y_ev_np, *, out_ch, args, device, seed):
    model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=args.tree_depth, n_inputs=args.n_inputs,
        head_width=args.head_width, device=device, seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size,
                        shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    s_acc = soft_acc(model, x_tr, y_tr, device)
    h_acc = hard_acc_via_export(model, x_ev_np, y_ev_np)
    return {'soft_acc': s_acc, 'hard_acc': h_acc, 'gap': s_acc - h_acc}


# ============================================================
# arm B: IO-KDのみ
# ============================================================

def run_arm_b(x_tr, y_tr, teacher_logits, x_ev_np, y_ev_np, *, out_ch, args,
              device, seed):
    model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=args.tree_depth, n_inputs=args.n_inputs,
        head_width=args.head_width, device=device, seed=seed,
    )
    loss_fn = make_kd_loss(alpha=args.alpha, temperature=args.temperature,
                           standardize=args.standardize, kind='kl')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr, teacher_logits),
                        batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb, tb)
            loss.backward()
            opt.step()
    s_acc = soft_acc(model, x_tr, y_tr, device)
    h_acc = hard_acc_via_export(model, x_ev_np, y_ev_np)
    return {'soft_acc': s_acc, 'hard_acc': h_acc, 'gap': s_acc - h_acc}


# ============================================================
# arm C: 構造転写(第1conv) + soft磨き(recover)
# ============================================================

def build_transcribed_conv_ir(
    teacher_conv: nn.Conv2d, calib_x_np: np.ndarray, *,
    out_ch: int, tree_depth: int, n_inputs: int, head_width: int,
    device: str, seed: int,
):
    """teacherの第1conv層をtranscribe_conv2dで転写し、残りをランダム初期化した
    Conv-DLGN全体をexport_irでIR化する(recover()に渡す前段)。

    Returns:
        (ir, approx_info): approx_infoはtranscribe_conv2dのinfo dict
        (agreement等、正直に報告する近似誤差)
    """
    calib_t = torch.as_tensor(calib_x_np, dtype=torch.float32)
    node, approx_info = transcribe_conv2d(
        teacher_conv, calib_inputs=calib_t, tree_depth=tree_depth,
        n_inputs=n_inputs, seed=seed,
    )
    # 呼び出し側(run_one_seed)がteacher_conv1.out_channelsをそのままout_chとして
    # 渡す契約なので、ここでは内部一貫性の確認のみ行う(不一致はバグ)
    assert node.params['out_channels'] == out_ch, (
        f'transcribe_conv2dが返したout_channels({node.params["out_channels"]}) '
        f'!= 呼び出し側out_ch({out_ch})。内部バグの可能性'
    )

    torch.manual_seed(seed)
    model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=tree_depth, n_inputs=n_inputs,
        head_width=head_width, device='cpu', seed=seed,
    )
    conv_module = model[0]
    assert isinstance(conv_module, LutConv2d)
    # transcribe_conv2dは'independent'パラメータ化を要求する真理値表を
    # 構築するため、LutConv2dも'independent'へ差し替えてから書き戻す。
    conv_module.parameterization = 'independent'
    conv_module.weights = nn.Parameter(
        torch.zeros(conv_module.out_channels, conv_module.n_internal,
                    conv_module.n_entries)
    )
    restore_lut_conv2d(conv_module, node.params['indices_flat'],
                       node.params['truth_tables'])

    model.eval()
    ir = export_ir(model.to(device), input_shape=IN_SHAPE, name='conv_dlgn_transcribed')
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'転写IR検証失敗: {errs[:3]}')
    return ir, approx_info


def run_arm_c(ir, x_tr_np, y_tr_np, x_ev_np, y_ev_np, *, args, device):
    """構造転写Conv-DLGN IRをrecoverでsoft磨きする(headline手法)。"""
    x_tr_flat = x_tr_np.reshape(x_tr_np.shape[0], -1)
    x_ev_flat = x_ev_np.reshape(x_ev_np.shape[0], -1)
    x_tr_t = torch.as_tensor(x_tr_flat, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr_np, dtype=torch.int64)
    x_ev_t = torch.as_tensor(x_ev_flat, dtype=torch.float32)
    y_ev_t = torch.as_tensor(y_ev_np, dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t),
                              batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(x_ev_t, y_ev_t),
                             batch_size=max(len(x_ev_t), 1))

    cfg = RecoverConfig(
        parameterization='independent', epochs=args.recover_epochs,
        lr=args.recover_lr, device=device, tau=args.recover_tau,
        selection='soft',
    )
    recovered, metrics = recover(ir, train_loader, eval_loader, cfg)
    return {'recovered_ir': recovered, **metrics}


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str,
                 x_tr_all, y_tr_all, x_te, y_te) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr = args.train_n
    x_tr = x_tr_all[:n_tr]
    y_tr = y_tr_all[:n_tr]
    calib_x = x_tr_all[:args.calib_n]

    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64)

    result: dict = {'seed': seed}

    # ---- teacher(SmallTeacherCNN)学習 ----
    torch.manual_seed(seed)
    teacher = SmallTeacherCNN(num_classes=N_CLASSES, in_channels=1).to(device)
    teacher_train_acc = train_teacher(
        teacher, x_tr_t, y_tr_t, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device, batch_size=args.batch_size,
    )
    with torch.no_grad():
        teacher.eval()
        pred = teacher(torch.as_tensor(x_te, dtype=torch.float32).to(device))
        teacher_test_acc = float((pred.argmax(-1).cpu().numpy() == y_te).mean())
    print(f'  [seed={seed}] teacher train_acc={teacher_train_acc:.4f} '
          f'test_acc={teacher_test_acc:.4f}')
    result['teacher_train_acc'] = teacher_train_acc
    result['teacher_test_acc'] = teacher_test_acc

    teacher_conv1 = teacher.features[0]
    assert isinstance(teacher_conv1, nn.Conv2d)
    # Conv-DLGNのout_chはteacherの第1conv out_channelsに揃える(SmallTeacherCNN
    # は常に32。--out-chは将来teacher差し替え時のための予約引数で、ここでは
    # 実際のteacher構造を信頼する)
    out_ch = teacher_conv1.out_channels
    # deepcopyしてCPUへ退避(teacher_conv1.to('cpu')は同一Moduleをin-placeで
    # デバイス移動するため、teacher本体(まだ後段でteacher_logits計算に使う)を
    # 破壊してしまう)
    teacher_conv1_cpu = copy.deepcopy(teacher_conv1).to('cpu')

    # ---- arm C: 第1conv層のtranscribe_conv2d(近似)+ 残り層ランダム初期化 ----
    t0 = time.time()
    ir_c, approx_info = build_transcribed_conv_ir(
        teacher_conv1_cpu, calib_x, out_ch=out_ch,
        tree_depth=args.tree_depth, n_inputs=args.n_inputs,
        head_width=args.head_width, device=device, seed=seed,
    )
    print(f'  [seed={seed}] transcribe_conv2d 近似agreement='
          f'{approx_info["approx_agreement"]:.4f} '
          f'(n_leaves={approx_info["n_leaves"]}, '
          f'receptive_field={approx_info["receptive_field"]}) '
          f'({time.time() - t0:.1f}s)')
    result['conv_approx_agreement'] = approx_info['approx_agreement']
    result['conv_approx_per_channel'] = approx_info['per_channel_agreement'].tolist()
    result['conv_n_leaves'] = approx_info['n_leaves']
    result['conv_receptive_field'] = approx_info['receptive_field']

    # ---- teacherロジット(arm B用) ----
    teacher_logits_tr = make_teacher_logits(teacher, x_tr_t, device).cpu()

    # ---- arm A: scratch ----
    t0 = time.time()
    res_a = run_arm_a(x_tr_t, y_tr_t, x_tr, y_tr, out_ch=out_ch, args=args,
                      device=device, seed=seed + 1000)
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm A(scratch): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} gap={res_a['gap']:+.4f} "
          f"({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ ----
    t0 = time.time()
    res_b = run_arm_b(x_tr_t, y_tr_t, teacher_logits_tr, x_tr, y_tr,
                      out_ch=out_ch, args=args, device=device, seed=seed + 2000)
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm B(IO-KDのみ): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} gap={res_b['gap']:+.4f} "
          f"({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm C: 構造転写(近似)+ soft磨き(本命) ----
    # hard acc評価は学習に使ったtrainサブセットで測る(scratch/IO-KDと同一条件で
    # 比較するため、A/Bも同じx_tr/y_trでhard_accを測っている)
    t0 = time.time()
    res_c = run_arm_c(ir_c, x_tr, y_tr, x_tr, y_tr, args=args, device=device)
    res_c['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] arm C(構造転写+soft磨き, 本命): "
          f"soft {res_c['soft_acc_before']:.4f}->{res_c['soft_acc_after']:.4f} "
          f"hard {res_c['hard_acc_before']:.4f}->{res_c['hard_acc_after']:.4f} "
          f"({res_c['seconds']}s)")
    result['C'] = res_c

    result['lift_vs_scratch'] = res_c['hard_acc_after'] - res_a['hard_acc']
    result['lift_vs_iokd'] = res_c['hard_acc_after'] - res_b['hard_acc']
    return result


# ============================================================
# 集計
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _strip_ir(r: dict) -> dict:
    r2 = dict(r)
    if 'C' in r2 and isinstance(r2['C'], dict):
        r2['C'] = {k: v for k, v in r2['C'].items() if k != 'recovered_ir'}
    return r2


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 90)
    print('CNN→Conv-DLGN構造転写PoC — 結果(arm A scratch / B IO-KD / C 構造転写+soft磨き)')
    print('=' * 90)

    header = '| arm | soft_acc(mean±std) | hard_acc(mean±std) |'
    print(header)
    print('|' + '---|' * 3)
    summary: dict = {'arms': {}}
    for arm in ('A', 'B', 'C'):
        if arm in ('A', 'B'):
            softs = [r[arm]['soft_acc'] for r in results]
            hards = [r[arm]['hard_acc'] for r in results]
        else:
            softs = [r[arm]['soft_acc_after'] for r in results]
            hards = [r[arm]['hard_acc_after'] for r in results]
        sm, ss = _mean_std(softs)
        hm, hs = _mean_std(hards)
        print(f'| {arm} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} |')
        summary['arms'][arm] = {'soft_acc_mean': sm, 'soft_acc_std': ss,
                                'hard_acc_mean': hm, 'hard_acc_std': hs}

    agreements = [r['conv_approx_agreement'] for r in results]
    am, as_ = _mean_std(agreements)
    print(f'\n転写conv近似agreement(較正データ, mean±std): {am:.4f}±{as_:.4f}')
    summary['conv_approx_agreement_mean'] = am
    summary['conv_approx_agreement_std'] = as_

    lift_scratch = [r['lift_vs_scratch'] for r in results]
    lift_iokd = [r['lift_vs_iokd'] for r in results]
    lsm, lss = _mean_std(lift_scratch)
    lim, lis = _mean_std(lift_iokd)
    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift_vs_scratch (C - A): {lsm:+.4f}±{lss:.4f}')
    print(f'lift_vs_iokd    (C - B): {lim:+.4f}±{lis:.4f}')
    summary['lift_vs_scratch_mean'] = lsm
    summary['lift_vs_scratch_std'] = lss
    summary['lift_vs_iokd_mean'] = lim
    summary['lift_vs_iokd_std'] = lis

    print('\n--- 仮説判定(本手法 = arm C) ---')
    print(f"H(構造転写>IO-KDのみ, hard acc): lift_vs_iokd={lim:+.4f} "
          f"→ {'支持' if lim > 0 else '不支持'}")
    print(f"H(構造転写>scratch, hard acc): lift_vs_scratch={lsm:+.4f} "
          f"→ {'支持' if lsm > 0 else '不支持'}")
    print(f'\n注意: transcribe_conv2dは近似転写(read-once木がリーフ全体でなく'
          f'受容野内サンプルのみを見るため)。agreement={am:.4f}は「厳密性0.0の'
          f'主張」ではなく実測値。')

    return summary


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CNN→Conv-DLGN構造転写PoC(MNIST)')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--train-n', type=int, default=4000,
                        help='arm A/B/C学習件数(共通、既定4000)')
    parser.add_argument('--calib-n', type=int, default=2000,
                        help='transcribe_conv2d較正件数')
    parser.add_argument('--eval-n', type=int, default=2000,
                        help='test評価件数')
    parser.add_argument('--epochs', type=int, default=5,
                        help='arm A/B(scratch/IO-KD)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=3)
    parser.add_argument('--teacher-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.01, help='arm A/BのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--out-ch', type=int, default=32,
                        help='conv出力チャンネル数(未使用、SmallTeacherCNN第1conv'
                             'のout_channels=32に自動固定。将来teacher差し替え時の'
                             '拡張用に引数のみ残す)')
    parser.add_argument('--tree-depth', type=int, default=3,
                        help='LutConv2d/transcribe_conv2dの木の深さ(既定3、n_leaves=2^3=8)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=(2, 4, 6))
    parser.add_argument('--head-width', type=int, default=256,
                        help='未使用(将来の多層head拡張用に予約)')
    parser.add_argument('--recover-epochs', type=int, default=20)
    parser.add_argument('--recover-lr', type=float, default=0.3,
                        help='recover()のAdam学習率(既定0.3、MLP版既定0.02'
                             'より大幅に大きい)。tree convは1層でtree_depth段の '
                             'lut_layerへ展開されるため(既定tree_depth=3で '
                             'conv3段+OrPool1段+head1段=5段)、MLP版のarith '
                             '実験(2-3段)より深い。independent parameterization '
                             '(sigmoid)は深い直列合成で勾配が急速に減衰するため '
                             '(実測: lr=0.02では10epoch経ってもsoft accが動かない '
                             'ほど勾配が小さい、lr>=0.3で学習が進む)、既定を引き '
                             '上げてある。詳細は開発記録参照')
    parser.add_argument('--recover-tau', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5, help='KD損失のCE項重み')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD温度T')
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='KD項の両ロジットをZ-score標準化する(既定ON)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小データ・少epoch、<60s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='outputs/convert_transcribe_mnist_conv')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.quick:
        args.seeds = 1
        args.train_n = 500
        args.calib_n = 300
        args.eval_n = 300
        args.epochs = 1
        args.teacher_epochs = 1
        args.tree_depth = 2
        args.recover_epochs = 2

    print(f'デバイス: {device}')
    print(f'seeds={args.seeds} train_n={args.train_n} calib_n={args.calib_n} '
          f'eval_n={args.eval_n}')
    print(f'conv: out_ch=(teacherの第1conv out_channelsに自動固定) '
          f'tree_depth={args.tree_depth} n_inputs={args.n_inputs} '
          f'(n_leaves={args.n_inputs ** args.tree_depth})')
    print(f'teacher: epochs={args.teacher_epochs} lr={args.teacher_lr}')
    print(f'arm A/B: epochs={args.epochs} lr={args.lr}')
    print(f'arm C(recover): epochs={args.recover_epochs} lr={args.recover_lr} '
          f'tau={args.recover_tau}')

    print('\nMNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    x_tr_all, y_tr_all = _flatten_binary_images(
        train_set, n=max(args.train_n, args.calib_n) + 1000)
    x_te, y_te = _flatten_binary_images(test_set, n=args.eval_n)
    print(f'train_pool={len(x_tr_all)} test={len(x_te)}')

    use_wandb = init_wandb(
        run_name=f'convert_transcribe_mnist_conv_d{args.tree_depth}',
        config_dict=vars(args),
        device=device,
        tags=['D12', 'structural-transfer', 'conv', 'soft-polish'],
        group='convert-transcribe-mnist-conv',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device, x_tr_all, y_tr_all, x_te, y_te)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_test_acc': r['teacher_test_acc'],
                f'seed{seed}/conv_approx_agreement': r['conv_approx_agreement'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/C_hard_acc': r['C']['hard_acc_after'],
                f'seed{seed}/lift_vs_scratch': r['lift_vs_scratch'],
                f'seed{seed}/lift_vs_iokd': r['lift_vs_iokd'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_{ts}.json')
    dump = {'args': vars(args), 'summary': summary,
            'per_seed': [_strip_ir(r) for r in results]}
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
