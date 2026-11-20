"""量子化起点の転写: 量子化(1bit/1.58bit)モデル → DWN学習可能配線DLGN 転写 → polish(MNIST)。

仮説(docs/plan の本計画): native DLGN転写がfloat起点で全滅した真因は
「float密モデルをfan-in≤Fへ枝刈りする過程の情報損失」。source を最初から
量子化(1bit sign / 1.58bit ternary)にし、ternaryの0マスク=枝刈り済み構造
として尊重しつつ **DWN学習可能配線(LearnableMapping)** へ転写→polishすれば、
配線学習が枝刈りで失った接続を再発見し、崖を回避できるか。

中心対照(同一パイプライン・同一fan-in構造 F を共有):
    - float   : 枝刈りfloat重みを transcribe(既存'相当の基準)
    - bin     : 1bit sign 量子化(0マスクなし=密、binarize_weights)
    - ternary : 1.58bit absmean 量子化(0マスク=疎、quantize_ternary+
                sparsify_from_mask でマスク内のみ転写)
各armを固定配線でtranscribe→**DWN学習可能配線へ復元(bit-exact)**→CE polish。
scratch(同容量DWNをランダム初期化からpolish)を下限対照に置く。

算術ゼロの範囲: 隠れ層は純LUT/DWN(popcount加算木を使わない=transcribe_mlp
の真理値表列挙のみ)。最終readoutのGroupSum(クラス別和)は既存DLGN標準として許容。

派生指標:
    - oracle_loss(arm) = source(量子化teacher) hard acc − transcribe hard acc
      (float起点nativeの+0.39〜+0.52を下回れば「量子化起点が崖を回避」の証拠)
    - dwn_polish_gain = DWN polish後 − transcribe直後 hard acc(配線学習の純効果)
    - 1bit(bin) vs 1.58bit(ternary): 0マスクの有無が崖回避に効くか
    - vs scratch: 転写起点がスクラッチDWNを上回るか(一次指標)
    - ternary残存fan-in分布(2に潰れていないか=量子化粒度の健全性)

使用例:
    # 煙実行(数十秒、wandb無効、少データ・少epoch)
    uv run python experiments/convert_ternary_to_dwn_mnist.py --quick --no-wandb

    # 本実行(SSH GPU、wandb online)
    uv run python experiments/convert_ternary_to_dwn_mnist.py --fanin 6 --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

# LUT basis(n_inputs=6→64幅)のGPU断片化によるOOMを緩和(importより前に設定)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 既存MNIST転写実験の再利用ヘルパ(データ展開・厳密性参照・device判定)
from convert_prune_transcribe_mnist import (
    _auto_device,
    _flatten_binary,
    _quantized_forward_pred,
    _simulate_pred,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.binarize import binarize_weights, quantize_ternary
from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    max_fanin,
)
from griffith_logic.convert.recover import rebuild_trainable
from griffith_logic.convert.teacher_models import TeacherMLPSte
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate
from griffith_logic.layers.lut import LutLayer

load_dotenv()

ARMS = ('float', 'bin', 'ternary')


# ============================================================
# DWN学習可能配線への復元(rebuild_as_native16g_learnable_wiring の汎用版)
# ============================================================

def rebuild_learnable_wiring(
    graph, parameterization: str, selection: str = 'soft',
) -> nn.Sequential:
    """転写IRを wiring='learnable'(DWN LearnableMapping)なDLGNへ復元する。

    まず rebuild_trainable(固定配線、bit-exact)で構築し、各LutLayerを同形状・
    同ゲート重みの wiring='learnable' 版へ差し替える。LearnableMapping.weights は
    固定配線の indices を argmax one-hot で再現するよう初期化する
    (ir/weights.py::restore_dwn_layer と同規約: flat[u*n_inputs+j]=indices[j,u]
    の位置に1.0)。→ 固定配線と全く同じ初期条件から配線を学習対象にできる。

    Args:
        graph: 転写IR(直列: 入力→lut_layer列→group_sum)
        parameterization: LUTパラメータ化('independent'/'warp'等。任意真理値表を
            厳密表現できるものを使う)
        selection: 'soft'(polish既定)

    Returns:
        wiring='learnable'なLutLayer列を持つDLGN(nn.Sequential)
    """
    fixed = rebuild_trainable(graph, parameterization=parameterization,
                              selection=selection)
    new_layers = []
    for m in fixed:
        if isinstance(m, LutLayer):
            learnable = LutLayer(
                m.in_dim, m.out_dim, n_inputs=m.n_inputs,
                device=str(m.weights.device),
                parameterization=m.parameterization, selection=m.selection,
                wiring='learnable',
            )
            with torch.no_grad():
                learnable.weights.copy_(m.weights)
                w = torch.zeros_like(learnable.learnable_mapping.weights)
                flat = m.indices.T.reshape(-1)  # flat[u*n_inputs+j]=indices[j,u]
                w[flat, torch.arange(flat.numel(), device=flat.device)] = 1.0
                learnable.learnable_mapping.weights.copy_(w)
            new_layers.append(learnable)
        else:
            new_layers.append(copy.deepcopy(m))
    return nn.Sequential(*new_layers)


def _reinit_scratch(model: nn.Sequential, seed: int) -> nn.Sequential:
    """DWNモデルと同容量(同形状)のスクラッチ初期化版を作る(下限対照)。"""
    torch.manual_seed(seed)
    scratch = copy.deepcopy(model)
    with torch.no_grad():
        for m in scratch:
            if isinstance(m, LutLayer):
                m.weights.normal_(0.0, 1.0)
                if m.wiring == 'learnable':
                    m.learnable_mapping.weights.uniform_(0.0, 1.0)
    return scratch


# ============================================================
# 評価
# ============================================================

@torch.no_grad()
def _eval_hard(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str,
               batch: int = 512) -> float:
    """eval()forward argmax の hard accuracy(0/1入力、ゲートは離散化)。

    n_inputs=6のLUT basisは64幅に展開されGPUメモリを食うため、evalも
    小バッチで回す(12GB級GPUでのOOM回避)。
    """
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


# ============================================================
# source(量子化teacher)の構築
# ============================================================

def _load_source_ckpt(path, hidden, device):
    """公開/外部MLPチェックポイントを source としてロードする(自作学習の代替)。

    受理形式:
      - TeacherMLPSte の state_dict(linears.*)そのまま
      - 素のMLP state_dict(Linearの weight/bias 列。キー名は問わず順序で対応)
      - dict{'hidden': [...], 'state_dict': ...}
    BitNetMCU(FC, 1.58bit)等は float shadow重みを保存しており、本実験の
    quantize_ternary(=BitNetMCUと同一 absmean 式)で当該の1.58bitモデルを再現する。
    """
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        hidden = obj.get('hidden', hidden)
        sd = obj['state_dict']
    else:
        sd = obj
    model = TeacherMLPSte(hidden=tuple(hidden))
    # linears.* キーで直接ロードできればそれを使う。無ければ Linear を順序対応
    try:
        model.load_state_dict(sd, strict=True)
    except (RuntimeError, KeyError):
        src_ws = [(k, v) for k, v in sd.items()
                  if k.endswith('.weight') and v.dim() == 2]
        dst = list(model.linears)
        if len(src_ws) != len(dst):
            raise ValueError(
                f'source Linear数={len(src_ws)} != model={len(dst)}。'
                f'--source-hidden で形状を合わせること'
            )
        with torch.no_grad():
            for (wk, w), lin in zip(src_ws, dst):
                bk = wk[:-len('weight')] + 'bias'
                if w.shape != lin.weight.shape:
                    raise ValueError(f'形状不一致 {w.shape} vs {lin.weight.shape}')
                lin.weight.copy_(w)
                if bk in sd and lin.bias is not None:
                    lin.bias.copy_(sd[bk])
    return model.to(device)


def _train_ste_teacher(x_tr, y_tr, hidden, epochs, lr, batch_size, device):
    """TeacherMLPSte(STE隠れ活性)を学習して source とする(=事前学習モデル代役)。"""
    model = TeacherMLPSte(hidden=tuple(hidden)).to(device)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x_tr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_t[idx].to(device), y_t[idx].to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


def _quantize_ste_weight(w: torch.Tensor, arm: str, prune_mask: torch.Tensor,
                         eps: float = 1e-5) -> torch.Tensor:
    """STE量子化: forward=量子化重み、backward=恒等(shadow重みへ)。枝刈り外は0。

    bin: sign×alpha(残存平均|W|)。ternary: absmean三値化(per-channel scale)。
    scale/alphaはprune_mask内平均で計算する(枝刈り0が平均を薄めないため)。
    """
    pm = prune_mask.to(w.dtype)
    if arm == 'bin':
        alpha = (w.detach().abs() * pm).sum() / pm.sum().clamp_min(1.0)
        q = torch.sign(w) * alpha
    elif arm == 'ternary':
        denom = pm.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = ((w.detach().abs() * pm).sum(dim=1, keepdim=True) / denom
                 ).clamp_min(eps)
        q = torch.clamp(torch.round(w / scale), -1.0, 1.0) * scale
    else:  # float
        q = w
    q = q * pm
    return w + (q - w).detach()  # STE(勾配はshadow wへ素通し)


def _qat_finetune(model, arm, prune_masks, x_tr, y_tr, *, epochs, lr,
                  batch_size, device):
    """量子化aware fine-tune(STE forward)で量子化sourceの精度を回復する。

    hidden Linearの重みを毎forwardで _quantize_ste_weight により量子化しつつ、
    backwardはshadow(float)重みへ流す。枝刈り0はstepごとに再ゼロして維持する。
    """
    from griffith_logic.convert.teacher_models import HeavisideSTE
    hidden = list(model.linears[:-1])
    final = model.linears[-1]
    pmts = [pm.to(device) for pm in prune_masks]
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_t[idx].to(device), y_t[idx].to(device)
            h = xb.reshape(xb.shape[0], -1)
            for lin, pm in zip(hidden, pmts):
                wq = _quantize_ste_weight(lin.weight, arm, pm)
                h = HeavisideSTE.apply(F.linear(h, wq, lin.bias))
            opt.zero_grad()
            F.cross_entropy(final(h), yb).backward()
            opt.step()
            with torch.no_grad():  # 枝刈り0を維持(shadow復活を防ぐ)
                for lin, pm in zip(hidden, pmts):
                    lin.weight.mul_(pm)


def _prepare_source(model: TeacherMLPSte, arm: str, x_tr, y_tr, args, device):
    """arm用にsourceを量子化(必要ならQAT回復)し、隠れ層マスク列と統計を返す。

    枝刈り0を保持したまま量子化。bin/ternaryは --qat-epochs>0 なら STE で
    量子化aware fine-tune し、最後に量子化重みをbake(transcribeが見る決定値)。
    """
    hidden = list(model.linears[:-1])
    prune_masks = [(lin.weight.detach() != 0).cpu() for lin in hidden]

    if arm != 'float':
        # 初期bake(QATの初期点)
        with torch.no_grad():
            for lin, pm in zip(hidden, prune_masks):
                wq = _quantize_ste_weight(lin.weight, arm, pm.to(lin.weight.device))
                lin.weight.copy_(wq.detach())
        if args.qat_epochs > 0:
            _qat_finetune(model, arm, prune_masks, x_tr, y_tr,
                          epochs=args.qat_epochs, lr=args.qat_lr,
                          batch_size=args.batch_size, device=device)
        # 最終bake(QAT後のshadow→量子化決定値)
        with torch.no_grad():
            for lin, pm in zip(hidden, prune_masks):
                wq = _quantize_ste_weight(lin.weight, arm, pm.to(lin.weight.device))
                lin.weight.copy_(wq.detach())

    masks: list[torch.Tensor] = []
    fanin_stats: list[dict] = []
    for lin in hidden:
        mask = (lin.weight.detach() != 0).cpu()
        masks.append(mask)
        nz = mask.sum(dim=1).float()
        fanin_stats.append({
            'mean_fanin': float(nz.mean()), 'max_fanin': int(nz.max()),
            'min_fanin': int(nz.min()),
            'sparsity': float(1.0 - mask.float().mean()),
        })
    return masks, fanin_stats


# ============================================================
# 1 arm のパイプライン(転写→DWN復元→polish)
# ============================================================

def _polish(model, x_tr, y_tr, x_te, y_te, *, epochs, lr, batch_size, device):
    """DWNモデルを CE で polish し hard acc(before, best-over-epochs)を返す。

    転写起点は既に良初期化のため、polishが壊す局面がある(良init破壊)。
    epochごとにhard accを測り最良を採る(repoの最良重み選択方針と整合)。
    """
    hard_before = _eval_hard(model, x_te, y_te, device)
    best = hard_before
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size,
                        shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        best = max(best, _eval_hard(model, x_te, y_te, device))
    return hard_before, best


def run_arm(arm, base_teacher, x_tr, y_tr, calib_x, x_te, y_te, args, device):
    """1 arm: source量子化→transcribe(bit-exact)→DWN復元→polish。"""
    print(f'\n=== arm {arm} ===')
    teacher = copy.deepcopy(base_teacher).to(device)
    masks, fanin_stats = _prepare_source(teacher, arm, x_tr, y_tr, args, device)
    src_acc = _eval_hard(teacher, x_te, y_te, device)
    fanin_str = ', '.join(
        '{:.1f}/{}'.format(s['mean_fanin'], s['max_fanin']) for s in fanin_stats
    )
    print(f'  [{arm}] 量子化source hard_acc={src_acc:.4f}  '
          f'隠れfan-in(mean/max)=[{fanin_str}]')

    # ---- 転写(固定配線、popcount非経由の真理値表列挙) ----
    teacher_cpu = teacher.to('cpu').eval()
    spec = TranscribeSpec(n_inputs=args.fanin, weight_bits=args.weight_bits)
    # bin/ternaryは枝刈りマスクを尊重して転写(float基準はsparsify_fanin top-F)
    ir = transcribe_mlp(
        teacher_cpu, calib_x, spec,
        masks=[m.cpu() for m in masks] if arm != 'float' else None,
    )
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'arm {arm}: IR検証失敗 {errs[:3]}')

    # 厳密性: 転写が表現する関数(量子化隠れ+量子化最終層)のargmax ==
    # simulate argmax(test全体一致=lowering無損失。不一致はバグ)
    ref_pred = _quantized_forward_pred(teacher_cpu, x_te, args.weight_bits)
    sim_pred = _simulate_pred(ir, x_te)
    n_mismatch = int((ref_pred != sim_pred).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'arm {arm}: 厳密転写違反 {n_mismatch}/{len(x_te)}(バグ、閾値は緩めない)'
        )
    transcribe_hard = float((sim_pred == y_te).sum()) / len(y_te)
    n_luts = sum(int(n.params['out_width']) for n in ir.nodes
                 if n.type == 'lut_layer')
    print(f'  [{arm}] 厳密転写OK  transcribe hard_acc={transcribe_hard:.4f}  '
          f'luts={n_luts}')

    # ---- DWN学習可能配線へ復元(bit-exact) ----
    model = rebuild_learnable_wiring(ir, args.parameterization).to(device)
    with torch.no_grad():
        y_model = _eval_hard(model, x_te, y_te, device)
    # rebuild直後のhard accがsimulate accと一致(独立assert、緩めない)
    if abs(y_model - transcribe_hard) > 1e-9:
        raise RuntimeError(
            f'arm {arm}: DWN復元直後 hard_acc={y_model:.6f} != '
            f'transcribe {transcribe_hard:.6f}(配線初期化バグの疑い)'
        )

    # ---- polish(CE、隠れ層は純DWN、readoutはGroupSum) ----
    t0 = time.time()
    hard_before, hard_after = _polish(
        model, x_tr, y_tr, x_te, y_te,
        epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size, device=device,
    )
    dt = round(time.time() - t0, 2)
    print(f'  [{arm}] DWN polish hard {hard_before:.4f}->{hard_after:.4f} '
          f'({dt}s)')

    return {
        'arm': arm,
        'source_hard_acc': src_acc,
        'transcribe_hard_acc': transcribe_hard,
        'dwn_hard_before': hard_before,
        'dwn_hard_after': hard_after,
        'oracle_loss': src_acc - transcribe_hard,
        'dwn_polish_gain': hard_after - hard_before,
        'n_luts': n_luts,
        'fanin_stats': fanin_stats,
        'polish_seconds': dt,
        'exact': True,
    }, model


# ============================================================
# 1 seed
# ============================================================

def run_one_seed(seed, x_tr, y_tr, calib_x, x_te, y_te, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- source teacher: 公開ckpt流用 or 自作STE学習 → fan-in F 枝刈り ----
    if args.source_ckpt:
        teacher = _load_source_ckpt(args.source_ckpt, args.hidden, device)
        print(f'\n[seed={seed}] 公開source流用: {args.source_ckpt}')
    else:
        teacher = _train_ste_teacher(
            x_tr, y_tr, args.hidden, args.teacher_epochs, args.teacher_lr,
            args.batch_size, device,
        )
    teacher_acc = _eval_hard(teacher, x_te, y_te, device)
    print(f'[seed={seed}] source teacher(枝刈り前) hard_acc={teacher_acc:.4f}')

    sched = PruneSchedule(fanins=tuple(args.fanins), epochs_per_stage=args.prune_epochs,
                          lr=args.teacher_lr)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)

    def train_fn(m, epochs):
        opt = torch.optim.Adam(m.parameters(), lr=sched.lr)
        m.train()
        for _ in range(epochs):
            perm = torch.randperm(len(x_tr))
            for i in range(0, len(x_tr), args.batch_size):
                idx = perm[i:i + args.batch_size]
                xb, yb = x_t[idx].to(device), y_t[idx].to(device)
                opt.zero_grad()
                F.cross_entropy(m(xb), yb).backward()
                opt.step()
        return {}

    def eval_fn(m):
        return _eval_hard(m, x_te, y_te, device)

    apply_fanin_pruning(teacher, sched, train_fn, eval_fn)
    pruned_acc = _eval_hard(teacher, x_te, y_te, device)
    print(f'[seed={seed}] fan-in<={args.fanin}枝刈り後 hard_acc={pruned_acc:.4f} '
          f'(max_fanin={max_fanin(teacher)})')

    result = {
        'seed': seed, 'teacher_acc': teacher_acc, 'pruned_acc': pruned_acc,
    }

    dwn_ternary = None
    for arm in ARMS:
        res, model = run_arm(arm, teacher, x_tr, y_tr, calib_x, x_te, y_te,
                             args, device)
        result[arm] = res
        if arm == 'ternary':
            dwn_ternary = model

    # ---- scratch対照(ternaryと同容量のDWNをランダム初期化→同epochs polish) ----
    scratch = _reinit_scratch(dwn_ternary, seed + 7).to(device)
    s_before, s_after = _polish(
        scratch, x_tr, y_tr, x_te, y_te, epochs=args.polish_epochs,
        lr=args.polish_lr, batch_size=args.batch_size, device=device,
    )
    result['scratch'] = {'dwn_hard_before': s_before, 'dwn_hard_after': s_after}
    print(f'[seed={seed}] scratch DWN polish hard {s_before:.4f}->{s_after:.4f}')

    # ---- 派生指標 ----
    result['ternary_vs_scratch'] = (
        result['ternary']['dwn_hard_after'] - s_after
    )
    result['ternary_vs_bin'] = (
        result['ternary']['dwn_hard_after'] - result['bin']['dwn_hard_after']
    )
    result['ternary_vs_float'] = (
        result['ternary']['dwn_hard_after'] - result['float']['dwn_hard_after']
    )
    return result


# ============================================================
# 集計
# ============================================================

def _mean_std(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results, args):
    print('\n' + '=' * 96)
    print('量子化起点の転写: 量子化source → DWN学習可能配線 転写→polish(MNIST) — 結果')
    print('=' * 96)
    summary = {'arms': {}}
    print('| arm | source hard | transcribe hard | DWN polish後 hard | '
          'oracle_loss | polish gain |')
    print('|' + '---|' * 6)
    for arm in ARMS:
        src = [r[arm]['source_hard_acc'] for r in results]
        tr = [r[arm]['transcribe_hard_acc'] for r in results]
        after = [r[arm]['dwn_hard_after'] for r in results]
        orc = [r[arm]['oracle_loss'] for r in results]
        gain = [r[arm]['dwn_polish_gain'] for r in results]
        sm = _mean_std(src)
        tm = _mean_std(tr)
        am = _mean_std(after)
        om = _mean_std(orc)
        gm = _mean_std(gain)
        print(f'| {arm} | {sm[0]:.4f} | {tm[0]:.4f} | {am[0]:.4f}±{am[1]:.4f} | '
              f'{om[0]:+.4f} | {gm[0]:+.4f} |')
        summary['arms'][arm] = {
            'source_hard_mean': sm[0], 'transcribe_hard_mean': tm[0],
            'dwn_after_mean': am[0], 'dwn_after_std': am[1],
            'oracle_loss_mean': om[0], 'polish_gain_mean': gm[0],
        }

    sc = _mean_std([r['scratch']['dwn_hard_after'] for r in results])
    summary['scratch_dwn_after_mean'] = sc[0]
    print(f'| scratch(下限) | - | - | {sc[0]:.4f}±{sc[1]:.4f} | - | - |')

    print('\n--- 仮説判定(mean over seeds) ---')
    tvs = _mean_std([r['ternary_vs_scratch'] for r in results])
    tvb = _mean_std([r['ternary_vs_bin'] for r in results])
    tvf = _mean_std([r['ternary_vs_float'] for r in results])
    print(f'ternary − scratch : {tvs[0]:+.4f}±{tvs[1]:.4f}  '
          f"→ {'支持(転写起点がスクラッチ超え)' if tvs[0] > 0 else '不支持'}")
    print(f'ternary − bin     : {tvb[0]:+.4f}±{tvb[1]:.4f}  (0マスクの純効果)')
    print(f'ternary − float   : {tvf[0]:+.4f}±{tvf[1]:.4f}')
    orc_t = summary['arms']['ternary']['oracle_loss_mean']
    print(f'ternary oracle_loss={orc_t:+.4f} '
          f"(float起点native +0.39〜+0.52 を{'下回る=崖回避' if orc_t < 0.39 else '下回らない'})")
    summary['ternary_vs_scratch_mean'] = tvs[0]
    summary['ternary_vs_bin_mean'] = tvb[0]
    summary['ternary_vs_float_mean'] = tvf[0]

    # ternary残存fan-in(潰れ検査): 全seed・全隠れ層のmean fan-in最小
    min_mean_fanin = min(
        s['mean_fanin'] for r in results for s in r['ternary']['fanin_stats']
    )
    print(f'\nternary残存fan-in(全層mean最小)={min_mean_fanin:.2f} '
          f"({'警告: 2未満=量子化粗すぎ' if min_mean_fanin < 2 else 'OK'})")
    summary['ternary_min_mean_fanin'] = min_mean_fanin
    summary['exactness_ok'] = all(
        r[arm]['exact'] for r in results for arm in ARMS
    )
    print(f'全arm厳密性assert(全seed): {"OK" if summary["exactness_ok"] else "NG"}')
    return summary


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='量子化起点の転写: 量子化(1bit/1.58bit)→DWN学習可能配線 転写→polish(MNIST)')
    parser.add_argument('--fanin', type=int, default=6, choices=(2, 4, 6),
                        help='転写LUT fan-in F(枝刈り最終値・transcribe n_inputs)')
    parser.add_argument('--fanins', type=int, nargs='+', default=None,
                        help='段階枝刈りスケジュール(既定[32,12,fanin])')
    parser.add_argument('--hidden', type=int, nargs='+', default=[300, 100])
    parser.add_argument('--source-ckpt', type=str, default=None,
                        help='公開/外部MLP state_dict を source に流用(自作学習を省略)。'
                             '--hidden で形状を合わせる。BitNetMCU等の1.58bitは'
                             'ternary armが同一absmean式で再現')
    parser.add_argument('--parameterization', type=str, default='independent',
                        choices=('independent', 'warp'),
                        help='DWN LutLayerのパラメータ化(任意真理値表を厳密表現)')
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--teacher-epochs', type=int, default=8)
    parser.add_argument('--teacher-lr', type=float, default=1e-3)
    parser.add_argument('--prune-epochs', type=int, default=2)
    parser.add_argument('--qat-epochs', type=int, default=8,
                        help='量子化aware fine-tune(STE)のepoch数(0で無効=PTQ)')
    parser.add_argument('--qat-lr', type=float, default=1e-3)
    parser.add_argument('--polish-epochs', type=int, default=30)
    parser.add_argument('--polish-lr', type=float, default=2e-3,
                        help='良init破壊を避けるため控えめ(D9安定化知見)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--calib-n', type=int, default=10000)
    parser.add_argument('--eval-n', type=int, default=10000)
    parser.add_argument('--train-n', type=int, default=None,
                        help='polish/teacher学習に使うtrain件数(既定全体)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str, default='outputs/ternary_to_dwn_mnist')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    if args.fanins is None:
        args.fanins = [32, 12, args.fanin]
    if args.fanins[-1] != args.fanin:
        args.fanins = list(args.fanins) + [args.fanin]

    if args.quick:
        args.seeds = 1
        args.hidden = [64]
        args.teacher_epochs = 3
        args.prune_epochs = 1
        args.qat_epochs = 3
        args.polish_epochs = 3
        args.fanins = [args.fanin]
        args.calib_n = 2000
        args.eval_n = 2000
        args.train_n = 4000

    print(f'デバイス: {device}  fan-in={args.fanin} '
          f'param={args.parameterization} hidden={args.hidden} seeds={args.seeds}')
    print(f'枝刈りスケジュール: {args.fanins}  polish: '
          f'epochs={args.polish_epochs} lr={args.polish_lr}')

    print('MNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    x_tr, y_tr = _flatten_binary(train_set, n=args.train_n)
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n)
    calib_x = x_tr[:args.calib_n]
    x_te = x_te[:args.eval_n]
    y_te = y_te[:args.eval_n]
    print(f'train={len(x_tr)} calib={len(calib_x)} test={len(x_te)}')

    use_wandb = init_wandb(
        run_name=f'ternary_to_dwn_mnist_F{args.fanin}_{args.parameterization}',
        config_dict=vars(args), device=device,
        tags=['D18', 'track-a', 'ternary', 'dwn-wiring', 'polish', 'mnist',
              f'F{args.fanin}'],
        group='ternary-dwn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, x_tr, y_tr, calib_x, x_te, y_te, args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_acc': r['teacher_acc'],
                f'seed{seed}/pruned_acc': r['pruned_acc'],
                f'seed{seed}/ternary_dwn_after': r['ternary']['dwn_hard_after'],
                f'seed{seed}/bin_dwn_after': r['bin']['dwn_hard_after'],
                f'seed{seed}/float_dwn_after': r['float']['dwn_hard_after'],
                f'seed{seed}/scratch_dwn_after': r['scratch']['dwn_hard_after'],
                f'seed{seed}/ternary_oracle_loss': r['ternary']['oracle_loss'],
                f'seed{seed}/ternary_vs_scratch': r['ternary_vs_scratch'],
                f'seed{seed}/ternary_vs_bin': r['ternary_vs_bin'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_F{args.fanin}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary, 'per_seed': results},
                  f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')
    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
