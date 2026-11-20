import torch


class GroupSum(torch.nn.Module):
    """
    出力ニューロンをkクラスにグループ化して合計する層。
    最終層のLUT出力を分類ロジットに変換する。

    クラス割当 `assign`(内部調査リポジトリ先行実験の GroupSum):
        contig — ニューロン i → クラス i // m(連続ブロック。difflogic / 従来の既定)
        stride — ニューロン i → クラス i mod k(k 飛び。conv 頭部で前層の局所性が
                 クラスに偏る「連続割当の罠」を避ける)
    IR では group_sum ノード自体は contig のまま、stride は手前に
    (m, k) → (k, m) の転置(permute / 恒等 lut_layer)を挟んで表す
    (export/extract.py。SpatialGroupSum の抽出と同型)。

    判定バイアス `bias`(内部調査リポジトリ先行実験の 2 クラス判定則 s1 > s0 + t):
    クラスごとの定数をクラス和に足してから τ で割る(logits = (Σ + bias) / τ)。
    NID の `s1 > s0 + t` は bias = [t, 0](または [0, −t])。IR では
    `group_sum.params.bias`(整数、省略 = 全 0)で運び、simulate が足す。HDL 生成は未対応
    (validate(target='hdl') が bias ≠ 0 を拒む。FPGA 側の終端部品か後続タスク)。
    """

    ASSIGNMENTS = ('contig', 'stride')

    def __init__(self, k: int, tau: float = 1.0, assign: str = 'contig',
                 bias=None):
        """
        Args:
            k: クラス数（出力次元）
            tau: 温度パラメータ（合計値をtauで割る）
            assign: クラス割当 'contig'(既定、従来と bitwise 同一)/ 'stride'
            bias: クラスごとの判定バイアス(長さ k。None = 全 0 で従来と bitwise 同一)。
                学習はしない(buffer)。IR に載せるには整数値であること
        """
        super().__init__()
        assert assign in self.ASSIGNMENTS, (
            f'assign は {self.ASSIGNMENTS} のいずれか: {assign!r}'
        )
        self.k = k
        self.tau = tau
        self.assign = assign
        if bias is None:
            self.register_buffer('bias', None)
        else:
            b = torch.as_tensor(bias, dtype=torch.float32).reshape(-1)
            if b.numel() != k:
                raise ValueError(f'bias の長さ({b.numel()})はクラス数 k({k})と一致が必要です')
            self.register_buffer('bias', b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % self.k == 0, (
            f'出力次元({x.shape[-1]})はクラス数({self.k})で割り切れる必要があります'
        )
        m = x.shape[-1] // self.k
        if self.assign == 'stride':
            s = x.reshape(*x.shape[:-1], m, self.k).sum(-2)
        else:
            s = x.reshape(*x.shape[:-1], self.k, m).sum(-1)
        if self.bias is not None:
            s = s + self.bias.to(dtype=s.dtype, device=s.device)
        return s / self.tau

    def bias_ints(self) -> list[int] | None:
        """IR 用の整数バイアス(全 0 なら None)。整数でない値は ValueError。"""
        if self.bias is None:
            return None
        vals = self.bias.detach().cpu().tolist()
        ints = [int(round(v)) for v in vals]
        if any(abs(v - i) > 1e-6 for v, i in zip(vals, ints, strict=True)):
            raise ValueError(f'IR に載せる GroupSum.bias は整数が必要です: {vals}')
        return ints if any(ints) else None

    def extra_repr(self):
        s = f'k={self.k}, tau={self.tau}'
        if self.assign != 'contig':
            s += f', assign={self.assign}'
        if self.bias is not None:
            s += f', bias={self.bias.tolist()}'
        return s
