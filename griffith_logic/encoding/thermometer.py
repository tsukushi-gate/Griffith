"""Thermometer符号化(DWN: Bacellar et al., ICML 2024, Appendix B)。

実数値入力 q を昇順しきい値 t_1 < ... < t_z で単調unary codeに変換する:

    T(q) = (1_{q>t_1}, 1_{q>t_2}, ..., 1_{q>t_z}) ∈ {0,1}^z

しきい値の決め方(mode):
    uniform:      t_i = min + i·(max−min)/(z+1)(等間隔。fit未実行時は[0,1]を仮定)
    distributive: t_i = sort(data)[⌊N·i/(z+1)⌋](経験分位点、補間なし。
                  Distributive Thermometer, Bacellar et al. 2022)

公式実装(torch_dwn/binarization.py)との照合済み仕様:
    - 比較は厳密に q > t_i(≧ではない)、出力は{0,1}
    - distributiveはソート済みデータの ⌊N·i/(z+1)⌋ 番目を直接しきい値にする
    - 勾配はSTE: ∂L/∂q = Σ_i ∂L/∂T_i(しきい値比較を素通しして合算)
      (公式実装は勾配を流さないが、本実装は上流学習用にSTEを追加)
"""

import torch
import torch.nn as nn


class _ThermometerSTE(torch.autograd.Function):
    """しきい値比較のSTE: forward = 1_{x>t}、backward = ビット方向の勾配合算"""

    @staticmethod
    def forward(ctx, x, thresholds):
        # x: (..., D), thresholds: (z, D) → (..., D, z)
        return (x.unsqueeze(-1) > thresholds.T).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.sum(dim=-1), None


class ThermometerEncoder(nn.Module):
    """Thermometer符号化層。

    Args:
        bits: 特徴あたりのビット数 z(しきい値の個数)
        mode: 'uniform'(等間隔) | 'distributive'(経験分位点)
        input_bits: IR 化(export_ir)するときの入力の量子化ビット数 B。
            None(既定)では境界(IR の外)のまま。指定すると抽出器が特徴ごとの
            B ビット符号 c(LSB 先頭、位置 d·B + j)を入力とし、実数 q = c / (2^B − 1)
            (ToTensor の uint8 → [0, 1] と同じ)としてしきい値比較を B 入力 lut_layer の
            真理値表に焼く。torch の forward は従来どおり実数 q を受ける

    Attributes:
        thresholds: (bits, n_features) しきい値(fit後に確定)
    """

    MODES = ('uniform', 'distributive')

    def __init__(self, bits: int, mode: str = 'uniform', input_bits: int | None = None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'mode は {self.MODES} のいずれか: {mode}')
        if bits < 1:
            raise ValueError(f'bits は1以上: {bits}')
        if input_bits is not None and int(input_bits) < 1:
            raise ValueError(f'input_bits は1以上: {input_bits}')
        self.bits = bits
        self.mode = mode
        self.input_bits = None if input_bits is None else int(input_bits)
        self.register_buffer('thresholds', None)

    def fit(self, data: torch.Tensor) -> 'ThermometerEncoder':
        """データからしきい値を学習する。

        Args:
            data: (N, D) 学習データ(特徴ごとにしきい値を推定)

        Returns:
            self(メソッドチェーン用)
        """
        if data.dim() != 2:
            raise ValueError(f'dataは(N, D)の2次元: {tuple(data.shape)}')
        if self.mode == 'uniform':
            lo = data.min(dim=0).values
            hi = data.max(dim=0).values
            steps = torch.arange(1, self.bits + 1, dtype=data.dtype, device=data.device)
            # t_i = min + i·(max−min)/(z+1)
            self.thresholds = lo.unsqueeze(0) + steps.unsqueeze(1) * (
                (hi - lo).unsqueeze(0) / (self.bits + 1)
            )
        else:
            # 経験分位点(補間なし): sort(data)[⌊N·i/(z+1)⌋]
            sorted_data = torch.sort(data, dim=0)[0]
            n = data.shape[0]
            idx = [int(n * i / (self.bits + 1)) for i in range(1, self.bits + 1)]
            self.thresholds = sorted_data[idx].clone()
        return self

    def _resolve_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        if self.thresholds is not None:
            return self.thresholds.to(dtype=x.dtype, device=x.device)
        if self.mode == 'distributive':
            raise RuntimeError('distributiveモードはfit(data)の実行が必要')
        # uniform未fit: [0,1]範囲を仮定した既定しきい値
        steps = torch.arange(1, self.bits + 1, dtype=x.dtype, device=x.device)
        t = steps / (self.bits + 1)
        return t.unsqueeze(1).expand(self.bits, x.shape[-1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """符号化(ビット軸を分離したまま返す)。

        Args:
            x: (..., D) 実数値入力

        Returns:
            (..., D, bits) 単調unary code {0,1}
        """
        return _ThermometerSTE.apply(x, self._resolve_thresholds(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """符号化してビット軸を特徴軸に平坦化: (..., D) → (..., D*bits)"""
        code = self.encode(x)
        return code.reshape(*x.shape[:-1], x.shape[-1] * self.bits)

    def extra_repr(self):
        fitted = self.thresholds is not None
        s = f'bits={self.bits}, mode={self.mode}, fitted={fitted}'
        if self.input_bits is not None:
            s += f', input_bits={self.input_bits}'
        return s
