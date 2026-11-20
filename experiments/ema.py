"""パラメータの指数移動平均（EMA）。

DLGN での狙い:
`hard_st` / 離散化評価では、推論時のネットワークは **ゲートlogitのargmax** で決まる。
拮抗した2つのゲートが勾配ノイズで毎ステップ入れ替わると、回路そのものが大きく変わり、
評価指標が eval ごとに跳ねる（実測: mAP の eval 間 sd 0.09〜0.16）。
logit を EMA で均すと argmax の反転頻度が下がり、評価が安定することを期待する。

使い方::

    ema = ParamEMA(model, decay=0.999)
    for step in ...:
        train_step(...)
        ema.update(model)
    with ema.applied(model):          # 評価・保存の間だけ EMA 重みに差し替え
        metric = evaluate(model, ...)

`applied()` を抜けると元の重みに厳密に戻る（訓練は生の重みで継続する）。
"""

from contextlib import contextmanager

import torch


class ParamEMA:
    """浮動小数パラメータ／バッファの指数移動平均を保持する。

    Args:
        model: 対象モデル
        decay: EMA 減衰率（0 < decay < 1）。大きいほど滑らか。
        include_buffers: True でバッファ（BN統計など）も平均対象にする。
            LUT/DLGN は BN を持たないため既定 False で十分。
    """

    def __init__(self, model: torch.nn.Module, decay: float,
                 include_buffers: bool = False):
        if not 0.0 < decay < 1.0:
            raise ValueError(f'decay は (0,1) の範囲: {decay}')
        self.decay = decay
        self.include_buffers = include_buffers
        self._backup: dict[str, torch.Tensor] | None = None
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in self._tracked(model)
        }

    def _tracked(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                yield name, p
        if self.include_buffers:
            for name, b in model.named_buffers():
                if b.dtype.is_floating_point:
                    yield name, b

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """1ステップ分 EMA を進める（optimizer.step() の直後に呼ぶ）。"""
        d = self.decay
        for name, tensor in self._tracked(model):
            self.shadow[name].mul_(d).add_(tensor.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> None:
        """モデルの重みを EMA 版に差し替える（生の重みは内部に退避）。

        評価ブロックが長くインデントを変えたくない箇所向け。必ず `swap_out`
        と対で呼ぶこと（`applied()` を使えるならそちらが安全）。
        """
        if self._backup is not None:
            raise RuntimeError('swap_in が入れ子になっている（swap_out 漏れ）')
        self._backup = {name: t.detach().clone() for name, t in self._tracked(model)}
        for name, tensor in self._tracked(model):
            tensor.copy_(self.shadow[name])

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module) -> None:
        """`swap_in` で退避した生の重みへ厳密に戻す。"""
        if self._backup is None:
            raise RuntimeError('swap_in されていない')
        for name, tensor in self._tracked(model):
            tensor.copy_(self._backup[name])
        self._backup = None

    @contextmanager
    def applied(self, model: torch.nn.Module):
        """このブロックの間だけモデルの重みを EMA 版に差し替える。"""
        self.swap_in(model)
        try:
            yield model
        finally:
            self.swap_out(model)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}
