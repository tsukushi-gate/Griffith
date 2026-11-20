"""勾配フロー捕捉(誤差逆伝播リプレイ用)。

コアライブラリは無変更。テンソルフックで各LutLayerの
  - 層入力 x_in(forward hook)
  - 出力勾配 grad_out(output.register_hook)
  - 入力勾配 grad_x(input.register_hook、検証用)
を捕捉し、エッジ単位勾配(スロット別 grad_gathered)は
core/functions.py の backward と同一式のbutterfly逆伝播で再計算する。

正しさは「scatter_add(edge_grads) == 捕捉した grad_x」の数値等価テストで固定する
(tests/test_editor_gradcap.py)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .monitor import encode_u8


def _effective_w(layer) -> torch.Tensor | None:
    """backwardで使われる有効基底重み w_eff(out_dim, 2^N)をパラメータ化別に構成する。

    core/functions.py 各Functionのbackwardと同一:
      - softmax(16ゲート): softmax(weights) @ gate_table
      - softmax(N>2): softmax(weights)
      - hard_st: softmax(weights/τ_b) @ gate_table(STE)
      - independent: sigmoid(weights)
      - raw: weightsそのまま
      - iwp/warp: 構造が異なるため未対応(None→集約のみ)
    """
    weights = layer.weights.detach()
    param = layer.parameterization
    use_16gate = getattr(layer, '_use_16gate', False)
    gate_table = getattr(layer, 'gate_table', None)

    if param == 'softmax':
        if layer.selection == 'hard_st':
            tau_b = float(getattr(layer, 'tau_b', 1.0))
            s = F.softmax(weights / tau_b, dim=-1)
        else:
            s = F.softmax(weights, dim=-1)
        if use_16gate and gate_table is not None:
            return s @ gate_table.to(s.dtype)
        return s
    if param == 'independent':
        return torch.sigmoid(weights)
    if param == 'raw':
        return weights
    return None  # iwp / warp


def compute_edge_grads(
    layer, x_in: torch.Tensor, grad_out: torch.Tensor,
) -> torch.Tensor | None:
    """エッジ単位勾配 grad_gathered (batch, out_dim, n_inputs) を再計算する。

    functions.py backward の butterfly逆伝播ループの移植(勾配は流さない)。
    """
    w = _effective_w(layer)
    if w is None:
        return None

    with torch.no_grad():
        n_inputs = int(layer.n_inputs)
        indices = layer.indices
        batch = x_in.shape[0]
        out_dim = indices.shape[1]

        gathered = x_in[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)

        # forwardと同じbutterfly中間状態を再構築
        intermediates = []
        basis = torch.ones(batch, out_dim, 1, device=x_in.device, dtype=x_in.dtype)
        for j in range(n_inputs):
            intermediates.append(basis)
            xj = gathered[:, :, j:j + 1]
            basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)

        # butterfly逆伝播(functions.py:93-109と同一式)
        grad_basis = torch.einsum('bo,oe->boe', grad_out, w)
        grad_gathered = torch.zeros_like(gathered)
        grad_cur = grad_basis
        for j in reversed(range(n_inputs)):
            half = grad_cur.shape[-1] // 2
            grad_lo = grad_cur[..., :half]
            grad_hi = grad_cur[..., half:]
            xj = gathered[:, :, j:j + 1]
            grad_gathered[:, :, j] = ((grad_hi - grad_lo) * intermediates[j]).sum(dim=-1)
            grad_cur = grad_lo * (1.0 - xj) + grad_hi * xj

        return grad_gathered


def scatter_edge_grads(
    grad_gathered: torch.Tensor, indices: torch.Tensor, in_dim: int,
) -> torch.Tensor:
    """edge_grads → 入力次元への集約(数値等価テストでgrad_xと照合する)。"""
    batch = grad_gathered.shape[0]
    flat_indices = indices.T.reshape(-1).unsqueeze(0).expand(batch, -1)
    flat_grad = grad_gathered.reshape(batch, -1)
    grad_x = torch.zeros(
        batch, in_dim, device=grad_gathered.device, dtype=grad_gathered.dtype
    )
    grad_x.scatter_add_(1, flat_indices, flat_grad)
    return grad_x


class GradCapture:
    """テンソルフックによる勾配フロー捕捉(fullレベル時のみattachする)。

    forward hookは毎step新しいテンソルに張り直される(テンソルフックは
    計算グラフと共に解放されるためリークしない)。
    """

    def __init__(self):
        self.x_in: dict[str, torch.Tensor] = {}
        self.grad_out: dict[str, torch.Tensor] = {}
        self.grad_x: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def attach(self, lut_layers: list[tuple[str, object]]) -> None:
        for name, layer in lut_layers:
            self._handles.append(
                layer.register_forward_hook(self._make_forward_hook(name))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_forward_hook(self, name: str):
        def hook(_module, inputs, output):
            x = inputs[0]
            self.x_in[name] = x.detach()
            if x.requires_grad:
                x.register_hook(
                    lambda g, n=name: self.grad_x.__setitem__(n, g.detach())
                )
            if output.requires_grad:
                output.register_hook(
                    lambda g, n=name: self.grad_out.__setitem__(n, g.detach())
                )

        return hook


def build_grad_flow(
    step: int,
    loss: float,
    lut_layers: list[tuple[str, object]],
    capture: GradCapture,
) -> dict:
    """1ステップ分のbackwardを層分解したgrad_flowフレームを組み立てる。

    order昇順 = 損失側(後段)→入力側(前段)。フロントのBackpropPlayerが
    この順に層を発光させて「誤差が伝播していく様子」を再生する。
    """
    layers_out: list[dict] = []
    layer_grad_norm: dict[str, float] = {}

    for order, (name, layer) in enumerate(reversed(lut_layers)):
        entry: dict = {'layer': name, 'order': order}
        g_out = capture.grad_out.get(name)
        if g_out is not None:
            # ニューロン別の平均|∂L/∂y|(バッチ平均、非負)
            entry['grad_out'] = encode_u8(g_out.abs().mean(dim=0))
        g_w = layer.weights.grad
        if g_w is not None:
            entry['grad_w'] = encode_u8(g_w.detach().norm(dim=-1))
            layer_grad_norm[name] = round(float(g_w.detach().norm().item()), 6)
        x_in = capture.x_in.get(name)
        if x_in is not None and g_out is not None:
            edge = compute_edge_grads(layer, x_in, g_out)
            if edge is not None:
                # スロット別 平均|grad_gathered| (out_dim, n_inputs)
                entry['edge_grad'] = encode_u8(edge.abs().mean(dim=0))
        layers_out.append(entry)

    return {
        'type': 'grad_flow',
        'step': step,
        'loss': round(loss, 6),
        'layers': layers_out,
        'layer_grad_norm': layer_grad_norm,
    }
