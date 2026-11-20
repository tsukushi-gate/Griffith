"""ゲート付き再帰の論理ゲート化(D2): LogicLSTMCell / LogicGRUCell。

外部研究のリカレント論理ゲートNNはElman型(RDDLGNの
`k_prev = k_layers(cat([h_t, k_prev]))`)止まりで、forget/input/output
ゲートを持つLSTM/GRUセルの論理化は存在しない。本モジュールの主張は
「ゲーティング=MUX/ANDマスクであり、本来ブール演算と親和的」である。

構成:
    ゲート(f/i/o、GRUはz/r)と候補は LutStack(cat([x, h]) を入力)で計算する。
    状態更新は soft論理MUX/ANDの積形で書き、eval(hard)時にゲート出力が
    {0,1}へ離散化されると厳密なブールMUX回路に崩壊する(HDL化への布石)。

soft(train)経路は全て微分可能。hard(eval)経路はLutLayerのhard評価を
そのまま流用するため、入力と状態が{0,1}なら出力も厳密に{0,1}になる。
"""

import torch
import torch.nn as nn

from ..layers.lut import LutLayer

# 隠れ状態の中立初期値。sigmoid(0)=0.5 と同じく、論理値としての「不定」を表す
# (RDDLGN の _init_hidden 'zero' モードと同じ思想)。
_NEUTRAL_STATE = 0.5


class _LutStack(nn.Module):
    """LutLayerを depth 段積んだ最小スタック(gate/candidate の計算に使う)。

    in_dim → out_dim → ... → out_dim (計 depth 段)。
    models/rddlgn.py の LutStack と同等だが、models パッケージへの依存を
    避けるため blocks 内に自前で持つ。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        depth: int,
        n_inputs: int,
        parameterization: str,
        connections: str,
    ) -> None:
        super().__init__()
        assert depth >= 1, f'depth は1以上: {depth}'
        self.layers = nn.ModuleList()
        prev = in_dim
        for _ in range(depth):
            self.layers.append(
                LutLayer(
                    in_dim=prev,
                    out_dim=out_dim,
                    n_inputs=n_inputs,
                    parameterization=parameterization,
                    connections=connections,
                )
            )
            prev = out_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _mux(sel: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """soft論理MUX: sel=1 で a、sel=0 で b を選ぶ。

    sel*a + (1-sel)*b。{0,1}入力では厳密な2:1マルチプレクサに一致する。
    """
    return sel * a + (1.0 - sel) * b


def _and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """soft AND: a*b。{0,1}入力ではビットANDに厳密一致。"""
    return a * b


class LogicLSTMCell(nn.Module):
    """論理ゲート版LSTMセル。入力・状態は[0,1]の論理値。

    ゲート(f/i/o)と候補c̃は LutStack(cat([x, h]) を入力)で計算し、
    状態更新は soft論理MUX/ANDの積形で書く:

        c_t = mux(f_t, c_{t-1}, and(i_t, c̃_t))
            = f_t*c_{t-1} + (1-f_t)*(i_t*c̃_t)
        h_t = and(o_t, c_t) = o_t*c_t

    eval時はゲート出力(LutLayerのhard評価)が{0,1}へ離散化され、上式が
    厳密なブールMUX回路に崩壊する(入力・状態が{0,1}なら h,c も{0,1})。

    ゲート → 回路の対応(ハードウェア展開の布石):

        c_{t-1} ─────────────┐
                             │1
        f_t ──────────── sel[MUX]── c_t ──┐
                             │0            │
        i_t ──[AND]──────────┘             │
        c̃_t ──┘                          [AND]── h_t
        o_t ───────────────────────────────┘
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        depth: int = 2,
        n_inputs: int = 2,
        parameterization: str = 'softmax',
        connections: str = 'random',
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # seed 指定時は接続テーブルと重み初期化を決定的にする
        if seed is not None:
            torch.manual_seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_in = input_size + hidden_size
        stack_kwargs = dict(
            depth=depth,
            n_inputs=n_inputs,
            parameterization=parameterization,
            connections=connections,
        )
        # f=forget, i=input, o=output, c=候補c̃。全て cat([x, h]) が入力
        self.f_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)
        self.i_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)
        self.o_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)
        self.c_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x:(B,input_size), state=(h,c) 各(B,hidden_size) → (h_new, c_new)"""
        h_prev, c_prev = state
        z = torch.cat([x, h_prev], dim=-1)

        f_t = self.f_gate(z)
        i_t = self.i_gate(z)
        o_t = self.o_gate(z)
        cand = self.c_gate(z)

        c_t = _mux(f_t, c_prev, _and(i_t, cand))
        h_t = _and(o_t, c_t)
        return h_t, c_t

    def init_state(
        self, batch: int, *, device=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.full((batch, self.hidden_size), _NEUTRAL_STATE, device=device)
        c = torch.full((batch, self.hidden_size), _NEUTRAL_STATE, device=device)
        return h, c


class LogicGRUCell(nn.Module):
    """論理ゲート版GRUセル(update/resetゲート + soft MUX更新)。

    ゲート(z=update, r=reset)は LutStack(cat([x, h]) を入力)で計算し、
    候補h̃は reset を掛けた状態 cat([x, and(r, h)]) を入力に計算する:

        h̃_t = c_gate(cat([x, r_t*h_{t-1}]))
        h_t = mux(z_t, h̃_t, h_{t-1})
            = z_t*h̃_t + (1-z_t)*h_{t-1}

    eval時は z/r/h̃ が{0,1}へ離散化され、h_t が厳密なブールMUX回路に崩壊する。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        depth: int = 2,
        n_inputs: int = 2,
        parameterization: str = 'softmax',
        connections: str = 'random',
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_in = input_size + hidden_size
        stack_kwargs = dict(
            depth=depth,
            n_inputs=n_inputs,
            parameterization=parameterization,
            connections=connections,
        )
        # z=update, r=reset は cat([x, h]) が入力
        self.z_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)
        self.r_gate = _LutStack(gate_in, hidden_size, **stack_kwargs)
        # 候補は reset 適用後の状態 cat([x, r*h]) が入力
        self.h_cand = _LutStack(gate_in, hidden_size, **stack_kwargs)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x:(B,input_size), h:(B,hidden_size) → h_new:(B,hidden_size)"""
        z = torch.cat([x, h], dim=-1)
        z_t = self.z_gate(z)
        r_t = self.r_gate(z)

        reset_h = _and(r_t, h)
        cand = self.h_cand(torch.cat([x, reset_h], dim=-1))

        return _mux(z_t, cand, h)

    def init_state(self, batch: int, *, device=None) -> torch.Tensor:
        return torch.full((batch, self.hidden_size), _NEUTRAL_STATE, device=device)


def scan_cell(cell: nn.Module, xs: torch.Tensor, state=None):
    """xs:(B,T,input_size) を時間方向に走査する。

    Returns:
        (outputs, final_state)
        outputs: (B,T,hidden_size) 各ステップの出力隠れ状態 h
        final_state: セルの最終状態(LSTMは(h,c)タプル、GRUは h テンソル)
    """
    batch, seq_len, _ = xs.shape
    if state is None:
        state = cell.init_state(batch, device=xs.device)

    outputs = []
    for t in range(seq_len):
        state = cell(xs[:, t, :], state)
        h = state[0] if isinstance(state, tuple) else state
        outputs.append(h)
    return torch.stack(outputs, dim=1), state
