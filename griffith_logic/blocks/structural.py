"""構造部品: Concat / Add / Residual。

IR v0.2のconcat/addノードにそのまま対応するtorch側の明示部品。
soft演算はIRのbuild(ir/build.py の _soft_or/_soft_xor)と同一式を
importして使う(単一情報源。{0,1}入力ではhard evalとビット厳密に一致)。

torch→IR抽出はexport_irのSequential直列走査では捕まらないため、
blocks/to_ir.py の blocks_to_ir で行う(Residual→fanout+concat/add
の部分グラフ)。
"""

import torch
import torch.nn as nn

from ..ir.build import _soft_or, _soft_xor

ADD_MODES = ('or', 'xor', 'sum')
MERGE_MODES = ('concat', 'or', 'xor')


class Concat(nn.Module):
    """dim=-1連結の明示部品(IR concatノード対応)。

    forward(xs): テンソル列を受け取り最後の次元で連結する。
    ビット順はIRのconcat規約と同一(先頭要素が出力ビット列の先頭)。
    """

    def forward(self, xs) -> torch.Tensor:
        return torch.cat(list(xs), dim=-1)


class Add(nn.Module):
    """多入力soft論理マージ(IR addノード対応)。

    mode:
        'or'  — 1-(1-a)(1-b)({0,1}入力でビットORと厳密一致)
        'xor' — a+b-2ab   ({0,1}入力でビットXORと厳密一致)
        'sum' — a+b       (整数和。論理領域の終端専用、HDL生成不可)
    """

    def __init__(self, mode: str):
        super().__init__()
        assert mode in ADD_MODES, f'mode は {ADD_MODES} のいずれか: {mode}'
        self.mode = mode

    def forward(self, xs) -> torch.Tensor:
        xs = list(xs)
        if self.mode == 'or':
            return _soft_or(xs)
        if self.mode == 'xor':
            return _soft_xor(xs)
        out = xs[0]
        for t in xs[1:]:
            out = out + t
        return out

    def extra_repr(self):
        return f'mode={self.mode}'


class Residual(nn.Module):
    """残差接続: forward(x) = merge(x, inner(x))。

    merge:
        'concat'    — cat([x, inner(x)], -1)。幅が拡大する
                      (IRのconcatビット順規約どおり恒等枝がslot 0=先頭)
        'or'/'xor'  — soft論理マージ。幅不変(innerの出力幅=入力幅が必要)
    """

    def __init__(self, inner: nn.Module, merge: str = 'concat'):
        super().__init__()
        assert merge in MERGE_MODES, f'merge は {MERGE_MODES} のいずれか: {merge}'
        self.inner = inner
        self.merge = merge
        self.merge_op = Concat() if merge == 'concat' else Add(merge)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge_op([x, self.inner(x)])

    def extra_repr(self):
        return f'merge={self.merge}'
