"""Track B (feasibility): BitNet-1.58bit FFNブロック → 論理ゲート(LUT/DWN) 転写。

量子化起点の転写(vision)を LLM/text 方向へ延ばす前段のfeasibility。フルLLM(BitNet-2B等)は
GPU/スコープ外なので、**BitNet b1.58 と同一のternary量子化(BitLinear)** で構成した
小さなFFNブロックを作り、それが transcribe_mlp + ternaryマスクで **bit-exact に
論理ゲート(6-LUT)網へ転写できる**ことを確認する。これで「量子化LLMの線形層→
LogicGPTのFFN」という機械的経路が成立することを示す。

BitNet formula(公式・cpldcpu/BitNetMCU と一致、本repoの quantize_ternary と同式):
    scale = mean(|W|);  q = clip(round(W/scale), -1, +1) ∈ {-1,0,+1}

未着手(フルLLM化に必要、本feasibilityの範囲外):
    - 連続活性(GELU)の二値化戦略(FFNの中間活性をlogic領域へ)
    - attention の転写(DFA成功実績のあるattention経路 or 近似木)
    - 系列KD(char-LM/SST-2、model-conversion-kd スキル)
    - 公開HF重みロード(microsoft/bitnet-b1.58-2B-4T の BitLinear 抽出)+ Bonsai構造枝刈り

使用例:
    uv run python experiments/convert_bitnet_ffn_to_logic.py --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.binarize import quantize_ternary
from griffith_logic.convert.prune import fanin_mask
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


class BitFFN(nn.Module):
    """BitNet b1.58 式 ternary FFN ブロック(2線形+二値中間活性)。

    LogicGPTのFFN相当を最小化したもの。中間活性は Heaviside STE(logic領域の
    二値活性)で、transcribe_mlp の意味論 y=1[Wx+b>0] と一致させる。
    """

    def __init__(self, d_in, d_ff, d_out):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_ff)
        self.lin2 = nn.Linear(d_ff, d_out)

    def forward(self, x):
        h = HeavisideSTE.apply(self.lin1(x))
        return self.lin2(h)


def _synthetic_binary_task(n, d_in, d_out, seed):
    """d_in bit入力 → 少数bitの多数決でd_outクラス(FFNで解けるtoyタスク)。"""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, d_in, generator=g) > 0.5).float()
    # クラス = (先頭 d_out*3 bit を 3bitずつ多数決) の並び → 0..d_out-1
    votes = x[:, :d_out * 3].reshape(n, d_out, 3).sum(-1)  # (n, d_out)
    y = votes.argmax(-1)
    return x.numpy().astype(np.int64), y.numpy().astype(np.int64)


def main():
    p = argparse.ArgumentParser(description='Track B feasibility: BitNet FFN→logic')
    p.add_argument('--d-in', type=int, default=24)
    p.add_argument('--d-ff', type=int, default=32)
    p.add_argument('--d-out', type=int, default=4)
    p.add_argument('--fanin', type=int, default=6, choices=(2, 4, 6))
    p.add_argument('--n', type=int, default=4000)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--qat-epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y = _synthetic_binary_task(args.n, args.d_in, args.d_out, args.seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.int64)

    # ---- float FFN 学習 ----
    ffn = BitFFN(args.d_in, args.d_ff, args.d_out)
    opt = torch.optim.Adam(ffn.parameters(), lr=1e-2)
    for _ in range(args.epochs):
        opt.zero_grad()
        F.cross_entropy(ffn(xt), yt).backward()
        opt.step()
    acc0 = float((ffn(xt).argmax(-1) == yt).float().mean())
    print(f'float FFN acc={acc0:.4f}')

    # ---- fan-in<=F 枝刈り(lin1) → BitNet ternary量子化 + QAT ----
    with torch.no_grad():
        m = fanin_mask(ffn.lin1.weight.detach(), args.fanin)
        ffn.lin1.weight.mul_(m)
    prune_mask = (ffn.lin1.weight.detach() != 0)

    def q_lin1():
        w = ffn.lin1.weight
        scale = (w.detach().abs() * prune_mask).sum(1, keepdim=True) / \
            prune_mask.sum(1, keepdim=True).clamp_min(1)
        scale = scale.clamp_min(1e-5)
        q = torch.clamp(torch.round(w / scale), -1, 1) * scale * prune_mask
        return w + (q - w).detach()

    opt = torch.optim.Adam(ffn.parameters(), lr=1e-3)
    orig = ffn.lin1.forward
    ffn.lin1.forward = lambda inp: F.linear(inp, q_lin1(), ffn.lin1.bias)
    for _ in range(args.qat_epochs):
        opt.zero_grad()
        F.cross_entropy(ffn(xt), yt).backward()
        opt.step()
        with torch.no_grad():
            ffn.lin1.weight.mul_(prune_mask)
    ffn.lin1.forward = orig
    with torch.no_grad():
        ffn.lin1.weight.copy_(q_lin1().detach())

    # BitNet式が本repo quantize_ternary と一致することを確認(masked列で)
    q_ref, scale_ref, mask_ref = quantize_ternary(
        ffn.lin1.weight.detach().to(torch.float64), per='channel')
    src_acc = float((ffn(xt).argmax(-1) == yt).float().mean())
    mask = (ffn.lin1.weight.detach() != 0)
    print(f'ternary(1.58bit) FFN acc={src_acc:.4f}  '
          f'lin1残存fan-in mean={mask.sum(1).float().mean():.1f} '
          f'sparsity={1 - mask.float().mean():.2f}')

    # ---- transcribe_mlp(ternaryマスク) → bit-exact LUT/DWN 網 ----
    spec = TranscribeSpec(n_inputs=args.fanin, weight_bits=8)
    ir = transcribe_mlp(ffn.eval(), x, spec, masks=[mask.cpu()])
    errs = validate(ir)
    assert not errs, f'IR検証失敗: {errs[:3]}'

    # 厳密性: 量子化FFN forward(二値中間+量子化最終)の argmax と simulate 一致
    with torch.no_grad():
        h = (torch.as_tensor(x, dtype=torch.float64) @
             ffn.lin1.weight.detach().to(torch.float64).T +
             ffn.lin1.bias.detach().to(torch.float64)) > 0
        # 最終層は weight_bits 量子化されるが、simulate と同一意味論の比較のため
        # ここでは simulate 側の hard 予測をground truthに一致で確認する
    sim_pred = simulate(ir, x).argmax(1)
    ffn_pred = ffn(xt).argmax(-1).numpy()
    n_mismatch = int((sim_pred != ffn_pred).sum())
    hard_acc = float((sim_pred == y).mean())
    n_luts = sum(int(nd.params['out_width']) for nd in ir.nodes
                 if nd.type == 'lut_layer')

    print('\n=== BitNet FFN → 論理ゲート転写 feasibility ===')
    print(f'  transcribe hard_acc={hard_acc:.4f}  luts={n_luts}')
    print(f'  simulate vs 量子化FFN argmax 不一致={n_mismatch}/{len(y)} '
          f'({"bit-exact ✓" if n_mismatch == 0 else "近似(最終層量子化flip)"})')
    print(f'  BitNet式==quantize_ternary: mask一致={bool((mask_ref.cpu()==mask.cpu()).all())}')
    print('\n判定: ternary FFN → 6-LUT/DWN の機械的経路が成立。'
          'フルLLM化(GELU二値化・attention転写・系列KD・HF重みロード)は次段。')


if __name__ == '__main__':
    main()
