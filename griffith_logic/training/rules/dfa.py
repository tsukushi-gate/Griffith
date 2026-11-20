"""DFA(Direct Feedback Alignment)学習ルール。

Lillicrap et al. 2016 / Nøkland 2016 に基づく:
出力誤差 e = ∂L/∂logits を固定ランダムフィードバック行列 B_l で
各層の出力空間へ直接射影し、局所勾配として使う。

    g_l = e @ B_l                    (batch, out_l)
    ∂L/∂θ_l ≈ ∂(local_out)/∂θ_l · g_l

実装方式(局所グラフ分離):
    - forward_pre_hook で入力を detach(上流へBP勾配を流さない)
    - モジュール自身のforwardが「detached入力→出力」の局所グラフを構築
    - forward_hook で局所出力を保持し、下流へは detach して渡す
      (このモジュールのパラメータへ大域BPが届かないことを保証)
    - loss計算後、autograd.grad(local_out, params, grad_outputs=g_l) で
      検証済みのカスタムbackward資産をそのまま再利用して勾配を注入

制約(現時点):
    - モジュール出力は2D (batch, out_dim)、または3D (batch, seq, dim) で
      あること(系列対応)。3Dの場合は内部で(batch*seq, dim)へ
      flattenして2D同様に扱う(位置をバッチ次元へ吸収するだけで、
      DFAの数式自体は変えない)。
    - block_upstream=True のみ対応(全DFA層構成が標準)

系列対応の詳細:
    - post_hookで出力が3D (B,S,D) の場合、(B*S,D) にreshapeしてstashする
      (以後の勾配計算はすべて2D空間で完結する)。
    - inject(error) の error は最終出力の誤差 (B,S,vocab) or (B,vocab)。
      各DFA層の出力形状が3Dだった場合、その層向けのフィードバック射影
      g=error@B を計算する前に error 自身も (B*S,vocab) へflattenする
      (層ごとに元のS_lが異なりうるため、layer出力のstash時に記録した
      「flatten前のbatch,seq」をkeyにして毎回対応するerrorを用意する)。
    - 出力2Dの層はerrorも2D (B,vocab) のまま使う(既存の挙動を完全維持)。
"""

import math
import zlib

import torch
import torch.nn as nn


class DFARule:
    """dfa注釈されたモジュール群のフック管理と勾配注入。"""

    def __init__(self, modules: list[tuple[str, nn.Module]]):
        self._entries = []  # (name, module)
        self._feedback: dict[str, torch.Tensor] = {}
        self._stash: dict[str, list[torch.Tensor]] = {}
        self._handles = []
        for name, module in modules:
            spec = module._learning_spec
            assert spec.block_upstream, (
                'DFA: block_upstream=False は未対応(現時点)'
            )
            self._entries.append((name, module))
            self._handles.append(module.register_forward_pre_hook(
                self._make_pre_hook()
            ))
            self._handles.append(module.register_forward_hook(
                self._make_post_hook(name)
            ))

    @property
    def modules(self):
        return [m for _, m in self._entries]

    def parameters(self):
        for _, m in self._entries:
            yield from m.parameters()

    def _make_pre_hook(self):
        def pre_hook(module, inputs):
            x = inputs[0]
            # 上流を遮断し、局所グラフの起点にする
            x = x.detach()
            if torch.is_grad_enabled():
                x.requires_grad_(True)
            return (x, *inputs[1:])
        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module, inputs, output):
            if not torch.is_grad_enabled():
                return None
            assert output.dim() in (2, 3), (
                f'DFA層 {name} の出力は2D (batch,out_dim) または'
                f'3D (batch,seq,out_dim) が必要: {tuple(output.shape)}'
            )
            if output.dim() == 3:
                b, s, d = output.shape
                flat = output.reshape(b * s, d)
                self._stash.setdefault(name, []).append((flat, (b, s)))
                # 下流へは元の3D形状のまま detach して返す(forward契約を保つ)
                return output.detach()
            self._stash.setdefault(name, []).append((output, None))
            # 大域グラフからこのモジュールを切り離す(paramはinjectで更新)
            return output.detach()
        return post_hook

    def begin_step(self):
        self._stash.clear()

    def _get_feedback(self, name: str, n_out_module: int, n_out_global: int,
                      device, dtype) -> torch.Tensor:
        key = f'{name}:{n_out_global}x{n_out_module}'
        if key not in self._feedback:
            # 固定ランダムフィードバック(学習しない)。分散1/Cでスケール。
            # seedは決定的hash(組込みhash()はプロセスごとにランダム化される)
            g = torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2 ** 31))
            B = torch.randn(n_out_global, n_out_module, generator=g)
            B = B / math.sqrt(n_out_global)
            self._feedback[key] = B.to(device=device, dtype=dtype)
        return self._feedback[key]

    def _align_error(
        self, error: torch.Tensor, local_batch: int,
        seq_shape: tuple[int, int] | None, name: str,
    ) -> torch.Tensor:
        """errorをDFA層のlocal_out先頭次元(local_batch)に合わせて2D化する。

        4通りの組み合わせを扱う(系列対応):
            (a) seq_shape=None, error 2D  — 従来通りそのまま使う。
            (b) seq_shape=None, error 3D  — 層自体は(B*S,dim)へ既にflatten
                済みで呼ばれている(例: LogicGPT.headは呼び出し側が
                x.reshape(b*s,d_model)してから渡す)が、errorは系列軸を
                保持したまま(B,S,C)で来る場合。層のbatch(=B*S)に合わせて
                errorも(B*S,C)へflattenする。
            (c) seq_shape=(b,s), error 3D — 層・error双方が系列軸を持つ
                (例: LogicMLP/LogicTransformerBlock)。同じ(b,s)である
                ことを検証してからflattenする。
            (d) seq_shape=(b,s), error 2D — errorが位置非依存(系列軸なし)
                の場合、系列方向に複製して(B*S,C)に揃える。
        """
        if seq_shape is None:
            if error.dim() == 2:
                assert error.shape[0] == local_batch, (
                    f'DFA層 {name}: error形状{tuple(error.shape)}が'
                    f'層のbatch={local_batch}と不一致'
                )
                return error
            assert error.dim() == 3, (
                f'DFA層 {name}: errorは2Dまたは3D: {tuple(error.shape)}'
            )
            b, s, c = error.shape
            assert b * s == local_batch, (
                f'DFA層 {name}: error形状{tuple(error.shape)}(B*S={b * s})が'
                f'層のbatch={local_batch}と不一致'
            )
            return error.reshape(b * s, c)

        b, s = seq_shape
        if error.dim() == 3:
            assert error.shape[0] == b and error.shape[1] == s, (
                f'DFA層 {name}: error形状{tuple(error.shape)}が'
                f'stash済み(batch={b}, seq={s})と不一致'
            )
            return error.reshape(b * s, error.shape[-1])
        # errorが2D(batch, C)のみの場合、系列方向に複製して
        # (B*S, C)に揃える(位置非依存の誤差を全位置へ配る)。
        assert error.shape[0] == b, (
            f'DFA層 {name}: error形状{tuple(error.shape)}が'
            f'stash済みbatch={b}と不一致'
        )
        return error.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)

    def inject(self, error: torch.Tensor):
        """出力誤差 e を フィードバック射影して param.grad に注入。

        Args:
            error: 最終出力の誤差。(batch, C) または系列出力
                (batch, seq, C) のいずれか(系列対応)。
                DFA層の出力が3D(post_hookで(B*S,dim)にflatten済み)の
                場合、この error も同じ (B*S, C) へflattenして使う
                (系列長SはDFA層のstash時に記録した(B,S)から取得する。
                error側のS_eとDFA層側のS_lが一致している前提。
                例: LogicGPTのhead/MLPはブロック出力と同じ系列長を保つため
                常に一致する)。
        """
        for name, module in self._entries:
            outs = self._stash.get(name, [])
            params = [p for p in module.parameters() if p.requires_grad]
            if not outs or not params:
                continue
            for local_out, seq_shape in outs:
                e = self._align_error(error, local_out.shape[0], seq_shape, name)
                B = self._get_feedback(
                    name, local_out.shape[1], e.shape[1],
                    local_out.device, local_out.dtype,
                )
                g = e @ B  # (batch[*seq], out_l)
                grads = torch.autograd.grad(
                    local_out, params, grad_outputs=g,
                    retain_graph=False, allow_unused=True,
                )
                for p, gr in zip(params, grads, strict=False):
                    if gr is None:
                        continue
                    if p.grad is None:
                        p.grad = gr.detach().clone()
                    else:
                        p.grad = p.grad + gr

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
