"""L2: LogicSeqMixerの勾配契約と参照実装照合。

固定する契約:
    - reference_causal_window / reference_soft_and のfloat64 gradcheck
      (内部点 x∈(0.1,0.9) と、境界点 x=0.0/1.0 を含むケースの両方)
    - ToeplitzTokenMixの窓gather経路の同値性:
        (a) 内部窓を公開している場合は reference_causal_window と
            TOL_F64_EQ / TOL_F64_GRAD で値・勾配が一致する
        (b) 非公開の場合も、出力位置tの入力勾配が
            reference_causal_window が定める窓の外で厳密に0であること
            (=窓gather経路が参照と同じ依存構造を持つこと)を検証する
      LutLayer自体の勾配は既存テスト(test_gradcheck_matrix等)の担保範囲のため、
      ここでは窓gather経路のみに焦点を当てる。
    - LogicSeqMixer全体の勾配疎通(埋め込み/エンコーダ/デコーダ/出力ヘッドの
      各代表パラメータにNoneでない有限勾配が届き、NaN/Infが無い)
"""

import inspect
import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD, assert_all_finite

from griffith_logic.core.reference import (
    reference_causal_window,
    reference_soft_and,
)
from griffith_logic.models.logic_seq_mixer import (
    LogicSeqMixer,
    LogicSeqMixerConfig,
    ToeplitzTokenMix,
)

_WINDOW_ATTRS = ('window', '_window', 'gather_window', '_gather_window',
                 'windows', '_windows')


def _interior(shape, seed=0):
    """(0.1, 0.9)の内部点(float64)。"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * 0.8 + 0.1


def _with_boundary(shape, seed=0):
    """内部点に境界値 0.0 / 1.0 を混ぜたfloat64テンソル。"""
    x = _interior(shape, seed=seed)
    flat = x.reshape(-1)
    flat[0] = 0.0
    flat[-1] = 1.0
    return flat.reshape(shape)


def _soft_tiny_config() -> LogicSeqMixerConfig:
    """決定的なsoft勾配のためGumbelノイズを切ったtiny構成。"""
    return replace(LogicSeqMixerConfig.tiny(), selection='soft', gumbel=False)


def _capture_token_mix(model: LogicSeqMixer, batch: int = 2):
    """モデルを1回forwardし、最初のToeplitzTokenMixと入力形状を取得する。"""
    captured: dict = {}

    def hook(module, inputs, output):
        if captured:
            return
        captured['module'] = module
        captured['shape'] = tuple(inputs[0].shape)

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, ToeplitzTokenMix)]
    if not handles:
        pytest.skip('ToeplitzTokenMixがモデル内に存在しない')

    cfg = model.config
    src = torch.zeros(batch, cfg.seq_len, dtype=torch.long)
    tgt = torch.zeros(batch, cfg.seq_len, dtype=torch.long)
    with torch.no_grad():
        model(src, tgt)
    for h in handles:
        h.remove()
    return captured['module'], captured['shape']


def _kernel_list(mix: ToeplitzTokenMix) -> list[int]:
    kernels = mix.kernels
    if isinstance(kernels, int):
        return [int(kernels)]
    return [int(k) for k in kernels]


def _max_kernel(mix: ToeplitzTokenMix) -> int:
    return max(_kernel_list(mix))


def _window_fn(mix: ToeplitzTokenMix):
    """内部窓gatherを (x, kernel_size) 呼び出しに正規化して返す(無ければNone)。"""
    for name in _WINDOW_ATTRS:
        candidate = getattr(mix, name, None)
        if not callable(candidate):
            continue
        params = inspect.signature(candidate).parameters
        if len(params) >= 2:
            return candidate
        return lambda x, _k, _fn=candidate: _fn(x)
    return None


def _reference_support(t_len: int, kernel_size: int, causal: bool) -> list[set]:
    """reference_causal_windowが定める、各位置tの依存時刻集合を返す。

    one-hot入力を全時刻掃引し「その入力が位置tの窓に現れるか」で判定する
    (窓の範囲を再実装せず、参照実装そのものを真実源にするため)。
    """
    support: list[set] = [set() for _ in range(t_len)]
    for s in range(t_len):
        x = torch.zeros(1, t_len, 1, dtype=torch.float64)
        x[0, s, 0] = 1.0
        w = reference_causal_window(x, kernel_size, causal=causal)
        for t in range(t_len):
            if float(w[0, t].abs().sum()) > 0.0:
                support[t].add(s)
    return support


class TestReferenceGradcheck:
    """参照実装のfloat64 gradcheck(内部点と境界点)。"""

    @pytest.mark.parametrize('kernel_size', [1, 3, 8])
    @pytest.mark.parametrize('causal', [True, False])
    def test_causal_window_gradcheck_interior(self, kernel_size, causal):
        x = _interior((2, 5, 3), seed=kernel_size).requires_grad_(True)

        def fn(x_):
            return reference_causal_window(x_, kernel_size, causal=causal)

        assert torch.autograd.gradcheck(fn, (x,), **GRADCHECK_KW)

    @pytest.mark.parametrize('causal', [True, False])
    def test_causal_window_gradcheck_boundary(self, causal):
        """境界点 x=0.0/1.0 を含むケース。"""
        x = _with_boundary((2, 5, 3), seed=11).requires_grad_(True)

        def fn(x_):
            return reference_causal_window(x_, 4, causal=causal)

        assert torch.autograd.gradcheck(fn, (x,), **GRADCHECK_KW)

    def test_soft_and_gradcheck_interior(self):
        a = _interior((3, 4), seed=21).requires_grad_(True)
        b = _interior((3, 4), seed=22).requires_grad_(True)
        assert torch.autograd.gradcheck(reference_soft_and, (a, b),
                                        **GRADCHECK_KW)

    def test_soft_and_gradcheck_boundary(self):
        """境界点 0.0 / 1.0 を含むケース。"""
        a = _with_boundary((3, 4), seed=23).requires_grad_(True)
        b = _with_boundary((3, 4), seed=24).requires_grad_(True)
        assert torch.autograd.gradcheck(reference_soft_and, (a, b),
                                        **GRADCHECK_KW)


class TestSoftAndReferenceEquivalence:
    """soft ANDの値・勾配が a*b の素朴経路と一致する。"""

    def test_forward_and_grads_match(self):
        a0 = _interior((3, 4), seed=31)
        b0 = _interior((3, 4), seed=32)
        grad_output = torch.randn(3, 4, dtype=torch.float64)

        ins = [t.detach().clone().requires_grad_(True) for t in (a0, b0)]
        y = reference_soft_and(*ins)
        grads = torch.autograd.grad(y, ins, grad_outputs=grad_output)

        ins_ref = [t.detach().clone().requires_grad_(True) for t in (a0, b0)]
        y_ref = ins_ref[0] * ins_ref[1]
        grads_ref = torch.autograd.grad(y_ref, ins_ref, grad_outputs=grad_output)

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        for g, g_ref in zip(grads, grads_ref):
            assert torch.allclose(g, g_ref, atol=TOL_F64_GRAD, rtol=0)


class TestTokenMixWindowPath:
    """ToeplitzTokenMixの窓gather経路が参照窓と同値であること。"""

    def test_internal_window_forward_and_grad_match(self, seed):
        """内部窓を公開している場合、値と勾配が参照窓と一致する。"""
        model = LogicSeqMixer(_soft_tiny_config()).double()
        model.train()
        mix, shape = _capture_token_mix(model)

        fn = _window_fn(mix)
        if fn is None:
            pytest.skip(
                'ToeplitzTokenMixが内部窓gatherを公開していないため、'
                'test_grad_support_matches_reference_window で代替検証する'
            )

        batch, t_len, _ = shape
        width = getattr(mix, 'group_width', shape[2])
        for kernel in _kernel_list(mix):
            x0 = _interior((batch, t_len, width), seed=41 + kernel)
            x = x0.detach().clone().requires_grad_(True)
            y = fn(x, kernel)
            grad_output = torch.randn_like(y)
            (gx,) = torch.autograd.grad(y, (x,), grad_outputs=grad_output)

            x_ref = x0.detach().clone().requires_grad_(True)
            y_ref = reference_causal_window(x_ref, kernel, causal=mix.causal)
            (gx_ref,) = torch.autograd.grad(y_ref, (x_ref,),
                                            grad_outputs=grad_output)

            assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0), \
                f'内部窓gatherの値が参照窓と不一致 (K={kernel})'
            assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), \
                f'内部窓gatherの勾配が参照窓と不一致 (K={kernel})'

    def test_grad_support_matches_reference_window(self, seed):
        """出力位置tの入力勾配が、参照窓の外で厳密に0であること。

        窓gather経路が reference_causal_window と同じ依存構造(=同じ
        Toeplitz帯)を持つことの、内部実装に依存しない同値性検証。
        """
        model = LogicSeqMixer(_soft_tiny_config()).double()
        model.train()
        mix, shape = _capture_token_mix(model)
        batch, t_len, channels = shape
        kernel = _max_kernel(mix)
        # 左側の窓境界(局所性)も検証できるよう T > K まで系列長を伸ばす
        t_len = max(t_len, kernel + 4)
        support = _reference_support(t_len, kernel, mix.causal)

        for t in range(t_len):
            x = _interior((batch, t_len, channels), seed=50 + t)
            x.requires_grad_(True)
            y = mix(x)
            (gx,) = torch.autograd.grad(y[:, t, :].sum(), (x,))
            assert_all_finite(gx, f'ToeplitzTokenMix入力勾配(t={t})')
            outside = [s for s in range(t_len) if s not in support[t]]
            for s in outside:
                assert torch.equal(gx[:, s, :], torch.zeros_like(gx[:, s, :])), \
                    (f'出力位置t={t}が窓外の時刻s={s}に依存している '
                     f'(causal={mix.causal}, K={kernel})')


class TestSeqMixerGradientFlow:
    """モデル全体の勾配疎通(Noneでない有限勾配が全段に届く)。"""

    def test_grad_reaches_all_stages(self, seed):
        model = LogicSeqMixer(_soft_tiny_config())
        model.train()
        cfg = model.config
        batch, t_len = 2, cfg.seq_len
        src = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len))
        tgt = torch.randint(0, cfg.vocab_size, (batch, t_len))

        logits, emb_reg = model(src, tgt)
        assert_all_finite(logits, 'logits')
        assert_all_finite(emb_reg, 'emb_reg_loss')
        logits.sum().backward()

        stages = {
            'src_embedding': lambda n: n.startswith('src_embedding'),
            'encoder': lambda n: n.startswith('encoder'),
            'decoder': lambda n: n.startswith('decoder'),
        }
        named = list(model.named_parameters())
        assert named, 'モデルに学習パラメータが無い'

        for stage, pred in stages.items():
            params = [(n, p) for n, p in named if pred(n) and p.requires_grad]
            assert params, f'{stage}に学習パラメータが無い'
            reached = [(n, p) for n, p in params if p.grad is not None]
            assert reached, f'{stage}にNoneでない勾配が1つも届いていない'
            for n, p in reached:
                assert_all_finite(p.grad, f'{n}の勾配')

        # 出力ヘッド(デコーダ末尾のパラメータ)にも勾配が届くこと
        decoder_params = [(n, p) for n, p in named if n.startswith('decoder')]
        head_name, head_param = decoder_params[-1]
        assert head_param.grad is not None, \
            f'出力ヘッド {head_name} に勾配が届いていない'
        assert_all_finite(head_param.grad, f'{head_name}の勾配')

        # 全体として意味のある勾配が存在する(全ゼロで「疎通」と誤認しない)
        assert any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for _, p in named
        ), '全パラメータの勾配が厳密に0(学習信号が届いていない)'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
