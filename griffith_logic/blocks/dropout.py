"""論理Dropout(研究要素)。

標準Dropoutの1/(1-p)リスケールは論理域[0,1]の意味論を壊すため行わない。
代わりに「出力単位の摂動」3方式を提供する。
"""

import torch
import torch.nn as nn

DROPOUT_MODES = ('neutral', 'flip', 'zero')


class LogicDropout(nn.Module):
    """[0,1]論理域向けDropout。訓練時のみ確率pで出力単位を摂動する。

    mode:
        'neutral' — 0.5へ固定(LUTにとって情報量最小の中立値)
        'flip'    — 1-x へ反転(論理NOT摂動)
        'zero'    — 0へ固定(標準Dropoutの論理域版)

    eval時は恒等。IR抽出(blocks_to_ir)でもスキップされ痕跡を残さない。
    勾配は非マスク位置をそのまま通す(マスク位置はneutral/zeroで0、
    flipで-1倍)。
    """

    def __init__(self, p: float = 0.1, mode: str = 'neutral'):
        super().__init__()
        assert 0.0 <= p < 1.0, f'p は [0,1) が必要: {p}'
        assert mode in DROPOUT_MODES, f'mode は {DROPOUT_MODES} のいずれか: {mode}'
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) < self.p).to(x.dtype)
        if self.mode == 'neutral':
            return x * (1.0 - mask) + 0.5 * mask
        if self.mode == 'flip':
            return x * (1.0 - mask) + (1.0 - x) * mask
        return x * (1.0 - mask)  # zero

    def extra_repr(self):
        return f'p={self.p}, mode={self.mode}'
