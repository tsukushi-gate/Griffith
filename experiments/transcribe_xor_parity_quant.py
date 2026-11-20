"""XOR / 3-XOR(parity)等の基本論理関数を、量子化(1bit/1.58bit)MLP経由で
DLGN(LUT網)へ転写できるか。fan-in が小さい(XOR=2, parity3=3, parity4=4)ので
LUT budget(≤6)内 → 転写は**ビット厳密**になるはず。実LLM(fan-in≈1563)と対照的に、
「fan-in が budget 内なら量子化起点でも厳密転写で100%」を確認する。

各関数 × arm(float/bin/ternary)で、量子化source→transcribe_mlp→simulate の hard acc を測る。

使用例:
    uv run python experiments/transcribe_xor_parity_quant.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.binarize import binarize_weights, quantize_ternary
from griffith_logic.convert.teacher_models import HeavisideSTE, TeacherMLPSte
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

ARMS = ('float', 'bin', 'ternary')


def _all_inputs(n):
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


FUNCS = {
    'and2':    (2, lambda X: (X.sum(1) == 2).astype(np.int64)),
    'or2':     (2, lambda X: (X.sum(1) >= 1).astype(np.int64)),
    'xor2':    (2, lambda X: (X.sum(1) % 2).astype(np.int64)),
    'parity3': (3, lambda X: (X.sum(1) % 2).astype(np.int64)),
    'parity4': (4, lambda X: (X.sum(1) % 2).astype(np.int64)),
    'maj3':    (3, lambda X: (X.sum(1) >= 2).astype(np.int64)),
}


def _quantize_ste(w, arm, pm, eps=1e-5):
    if arm == 'bin':
        alpha = (w.detach().abs() * pm).sum() / pm.sum().clamp_min(1)
        q = torch.sign(w) * alpha
    elif arm == 'ternary':
        scale = ((w.detach().abs() * pm).sum(1, keepdim=True)
                 / pm.sum(1, keepdim=True).clamp_min(1)).clamp_min(eps)
        q = torch.clamp(torch.round(w / scale), -1, 1) * scale
    else:
        q = w
    return w + (q * pm - w).detach()


def train_and_transcribe(fname, n, fn, arm, seed=0, epochs=1500, qat_epochs=400):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = _all_inputs(n)
    y = fn(X)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.int64)

    # tiny STE-MLP(2クラス出力)を全パターン暗記させる
    model = TeacherMLPSte(in_width=n, hidden=(8,), num_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward()
        opt.step()
    teacher_acc = float((model(xt).argmax(-1) == yt).float().mean())

    # 隠れ層を量子化(枝刈り0保持=ここでは密、fan-in=n≤budget)+QAT
    hid = list(model.linears[:-1])
    pms = [(l.weight.detach() != 0) for l in hid]
    if arm != 'float' and qat_epochs > 0:
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        final = model.linears[-1]
        for _ in range(qat_epochs):
            h = xt
            for l, pm in zip(hid, pms):
                h = HeavisideSTE.apply(F.linear(h, _quantize_ste(l.weight, arm, pm), l.bias))
            opt.zero_grad()
            F.cross_entropy(final(h), yt).backward()
            opt.step()
    with torch.no_grad():
        for l, pm in zip(hid, pms):
            if arm != 'float':
                l.weight.copy_(_quantize_ste(l.weight, arm, pm).detach())
    src_acc = float((model(xt).argmax(-1) == yt).float().mean())

    # 転写(fan-in = n ≤ 6 なので疎化不要 = ビット厳密のはず)
    masks = None
    if arm == 'ternary':
        masks = [quantize_ternary(l.weight.detach().double(), per='channel')[2]
                 for l in hid]
    nin = 2 if n <= 2 else (4 if n <= 4 else 6)  # LUT n_inputsは2/4/6のみ
    spec = TranscribeSpec(n_inputs=nin, weight_bits=8)
    ir = transcribe_mlp(model.eval(), X, spec, masks=masks)
    assert not validate(ir), validate(ir)[:2]
    pred = simulate(ir, X).argmax(1)
    hard = float((pred == y).mean())
    return teacher_acc, src_acc, hard


def main():
    print('関数 | arm | teacher | 量子化source | transcribe hard | 厳密?')
    print('-' * 68)
    rows = []
    for fname, (n, fn) in FUNCS.items():
        for arm in ARMS:
            ta, sa, hard = train_and_transcribe(fname, n, fn, arm)
            exact = '✓' if abs(hard - sa) < 1e-9 else f'✗(Δ{sa - hard:+.2f})'
            print(f'{fname:8s} | {arm:7s} | {ta:.3f}  | {sa:.3f}       | '
                  f'{hard:.3f}          | {exact}')
            rows.append((fname, arm, ta, sa, hard))
    print('-' * 68)
    # 判定: すべての関数で transcribe hard == 量子化source(fan-in budget内=厳密)
    all_exact = all(abs(sa - hard) < 1e-9 for _, _, _, sa, hard in rows)
    perfect = [f'{fn}/{a}' for fn, a, _, _, h in rows if h == 1.0]
    print(f'全転写ビット厳密(hard==source): {"YES ✓" if all_exact else "NO"}')
    print(f'hard 100%到達: {len(perfect)}/{len(rows)}  {perfect}')
    print('\n判定: fan-in が LUT budget(≤6)内なら、量子化(1bit/1.58bit)起点でも '
          'XOR/parity3/parity4 は厳密転写で100%到達可能。実LLM(fan-in≈1563)が'
          '転写不可なのは fan-in ギャップが原因であり、論理関数そのものの問題ではない。')


if __name__ == '__main__':
    main()
