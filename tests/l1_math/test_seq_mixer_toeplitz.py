"""L1: LogicSeqMixerのToeplitz窓の数式性質。

固定する性質:
    - reference_causal_window が素朴ループ定義とfloat64で厳密一致
      (causal=True/False、kernel_size=1/2/3/8/16、T<Kのはみ出しケース含む)
    - Toeplitz性(shift-invariance): 入力を時間方向へ1つ右シフトすると
      出力も1つ右シフトする(先頭K段の境界領域を除く)。参照窓レベルと
      ToeplitzTokenMixのモデル出力レベルの両方で確認する
    - 因果性: causal=Trueの窓が位置tより大きい時間添字を一切含まない
      (one-hot入力の全時刻掃引で検証)
    - 二値頂点でのBoolean一致: reference_soft_and が {0,1}^2 の全4通りで
      論理ANDと厳密一致
    - soft-XORの性質: mlp_mixer._soft_xor が {0,1}^2 でXORと一致し、
      soft_xor(x,x) = 2x(1-x) が二値で0になる(MIXER-2の「XOR残差は
      自己入力で情報を消す」禁止事項の数学的根拠を固定する)

備考(代替手段の明示):
    ToeplitzTokenMix が内部の窓gather結果を公開していない場合、
    「内部窓 vs reference_causal_window の厳密一致」テストはskipし、
    参照窓自体の性質テスト + モデル出力レベルのshift-invarianceで代替する。
"""

import inspect
import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.core.reference import (
    reference_causal_window,
    reference_soft_and,
)
from griffith_logic.models.logic_seq_mixer import (
    LogicSeqMixer,
    LogicSeqMixerConfig,
    ToeplitzTokenMix,
)
from griffith_logic.models.mlp_mixer import _soft_xor

# 窓gather結果を公開している可能性のあるメソッド名(実装依存の探索)
_WINDOW_ATTRS = ('window', '_window', 'gather_window', '_gather_window',
                 'windows', '_windows')


def _naive_window(x: torch.Tensor, kernel_size: int, causal: bool,
                  pad_value: float) -> torch.Tensor:
    """(B,T,C) → (B,T,C,K) の窓gatherを素朴な三重ループで直接計算する。"""
    batch, t_len, channels = x.shape
    out = torch.full((batch, t_len, channels, kernel_size), pad_value,
                     dtype=x.dtype)
    for t in range(t_len):
        start = (t - kernel_size + 1) if causal else (t - (kernel_size - 1) // 2)
        for j in range(kernel_size):
            s = start + j
            if 0 <= s < t_len:
                out[:, t, :, j] = x[:, s, :]
    return out


def _interior(shape, seed=0):
    """(0.1, 0.9)の内部点(float64)。"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * 0.8 + 0.1


def _soft_tiny_config() -> LogicSeqMixerConfig:
    """決定的なsoft評価のためGumbelノイズを切ったtiny構成。"""
    return replace(LogicSeqMixerConfig.tiny(), selection='soft', gumbel=False)


def _capture_token_mix(model: LogicSeqMixer, batch: int = 2):
    """モデルを1回forwardし、最初のcausal ToeplitzTokenMixと入力形状を取得する。

    ToeplitzTokenMixのコンストラクタ引数は契約に含まれないため、
    実モデルから取り出して実際に流れる形状で検証する。
    """
    captured: dict = {}

    def hook(module, inputs, output):
        if captured or not getattr(module, 'causal', False):
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

    if 'module' not in captured:
        pytest.skip('causal=TrueのToeplitzTokenMixが1つも実行されなかった')
    return captured['module'], captured['shape']


def _kernel_list(mix: ToeplitzTokenMix) -> list[int]:
    """ToeplitzTokenMixが使うカーネル長の一覧。"""
    kernels = mix.kernels
    if isinstance(kernels, int):
        return [int(kernels)]
    return [int(k) for k in kernels]


def _max_kernel(mix: ToeplitzTokenMix) -> int:
    """ToeplitzTokenMixが使う最大カーネル長。"""
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


class TestReferenceCausalWindowNaive:
    """参照窓が素朴ループ定義とfloat64で厳密一致する。"""

    @pytest.mark.parametrize('kernel_size', [1, 2, 3, 8, 16])
    @pytest.mark.parametrize('causal', [True, False])
    @pytest.mark.parametrize('t_len', [1, 2, 5, 9])
    def test_matches_naive_loop(self, kernel_size, causal, t_len):
        """K > T のはみ出しケース(t_len=1,2 × K=8,16)も含めて掃引する。"""
        x = _interior((2, t_len, 3), seed=kernel_size * 10 + t_len)
        y = reference_causal_window(x, kernel_size, causal=causal)
        expected = _naive_window(x, kernel_size, causal, 0.0)
        assert y.shape == (2, t_len, 3, kernel_size), \
            f'形状契約違反: {tuple(y.shape)}'
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0), \
            f'窓gatherが素朴定義と不一致 (K={kernel_size}, causal={causal}, T={t_len})'

    @pytest.mark.parametrize('pad_value', [0.0, 0.5, 1.0])
    @pytest.mark.parametrize('causal', [True, False])
    def test_pad_value_is_used(self, pad_value, causal):
        """範囲外はpad_valueで埋まる。"""
        x = _interior((1, 3, 2), seed=7)
        y = reference_causal_window(x, 5, causal=causal, pad_value=pad_value)
        expected = _naive_window(x, 5, causal, pad_value)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)


class TestCausality:
    """causal窓は未来の時間添字を参照しない。"""

    @pytest.mark.parametrize('kernel_size', [1, 2, 3, 8, 16])
    def test_no_future_index_onehot_sweep(self, kernel_size):
        """one-hot入力の全時刻掃引: 時刻pに立てた1が t<p の窓に現れない。"""
        t_len = 6
        for p in range(t_len):
            x = torch.zeros(1, t_len, 1, dtype=torch.float64)
            x[0, p, 0] = 1.0
            w = reference_causal_window(x, kernel_size, causal=True)
            for t in range(p):
                assert torch.equal(w[0, t], torch.zeros_like(w[0, t])), \
                    f'位置t={t}の窓が未来の入力p={p}を参照している (K={kernel_size})'
            # 窓は [p, p+K-1] の範囲では非ゼロであるべき(自明性の防止)
            reachable = [t for t in range(p, min(p + kernel_size, t_len))]
            assert any(w[0, t].abs().sum() > 0 for t in reachable), \
                f'位置p={p}の入力がどの窓にも現れない (K={kernel_size})'

    @pytest.mark.parametrize('kernel_size', [2, 3, 8])
    def test_noncausal_window_is_centered(self, kernel_size):
        """causal=Falseの窓は [t-(K-1)//2, t+K//2] の中心窓である。"""
        t_len = 7
        for p in range(t_len):
            x = torch.zeros(1, t_len, 1, dtype=torch.float64)
            x[0, p, 0] = 1.0
            w = reference_causal_window(x, kernel_size, causal=False)
            for t in range(t_len):
                lo = t - (kernel_size - 1) // 2
                hi = t + kernel_size // 2
                nonzero = bool(w[0, t].abs().sum() > 0)
                assert nonzero == (lo <= p <= hi), \
                    f'中心窓の範囲が違う (t={t}, p={p}, K={kernel_size})'


class TestToeplitzShiftInvariance:
    """時間シフトに対する同変性(=重み共有によるToeplitz性)。"""

    @pytest.mark.parametrize('kernel_size', [1, 2, 3, 8])
    def test_reference_window_is_shift_equivariant(self, kernel_size):
        """参照窓レベル: 右シフト入力の窓(t) = 元入力の窓(t-1)(t>=K)。"""
        t_len = 12
        x = _interior((2, t_len, 3), seed=kernel_size)
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]

        w = reference_causal_window(x, kernel_size, causal=True)
        w_shift = reference_causal_window(x_shift, kernel_size, causal=True)

        assert torch.allclose(
            w_shift[:, kernel_size:], w[:, kernel_size - 1:t_len - 1],
            atol=TOL_F64_EQ, rtol=0,
        ), f'窓のshift-equivarianceが崩れている (K={kernel_size})'

    def test_token_mix_output_is_shift_equivariant(self, seed):
        """モデル出力レベル: ToeplitzTokenMixの出力が1時刻シフトに同変。

        位置ごとに重みが共有(=Toeplitz)でなければ成立しない性質。
        """
        model = LogicSeqMixer(_soft_tiny_config()).double()
        model.train()
        mix, shape = _capture_token_mix(model)
        batch, t_len, channels = shape
        kernel = _max_kernel(mix)
        # ToeplitzTokenMixは時間長に依存しない(重み共有)ため、シフト検証域が
        # 取れるようT > K まで系列長を伸ばして評価する
        t_len = max(t_len, kernel + 4)

        x = _interior((batch, t_len, channels), seed=123)
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]

        with torch.no_grad():
            y = mix(x)
            y_shift = mix(x_shift)

        assert y.shape == x.shape, f'形状契約違反: {tuple(y.shape)}'
        assert t_len > kernel, f'T={t_len} <= K={kernel} で検証域が無い'
        assert torch.allclose(
            y_shift[:, kernel:], y[:, kernel - 1:t_len - 1],
            atol=TOL_F64_EQ, rtol=0,
        ), 'ToeplitzTokenMixの出力が時間シフトに同変でない(重み共有が壊れている)'

    def test_internal_window_matches_reference(self, seed):
        """内部窓gatherがreference_causal_windowと厳密一致(公開時のみ)。"""
        model = LogicSeqMixer(_soft_tiny_config()).double()
        model.train()
        mix, shape = _capture_token_mix(model)

        fn = _window_fn(mix)
        if fn is None:
            pytest.skip(
                'ToeplitzTokenMixが内部窓gatherを公開していないため、'
                '参照窓の性質テストとモデル出力のshift-invarianceで代替する'
            )

        batch, t_len, _ = shape
        width = getattr(mix, 'group_width', shape[2])
        for kernel in _kernel_list(mix):
            x = _interior((batch, t_len, width), seed=321 + kernel)
            got = fn(x, kernel)
            expected = reference_causal_window(x, kernel, causal=mix.causal)
            assert torch.allclose(got, expected, atol=TOL_F64_EQ, rtol=0), \
                f'内部窓gatherがreference_causal_windowと一致しない (K={kernel})'


class TestSoftAndBinaryVertices:
    """soft ANDが二値頂点で論理ANDと厳密一致する。"""

    def test_matches_logical_and(self):
        for a_bit in (0.0, 1.0):
            for b_bit in (0.0, 1.0):
                a = torch.tensor([a_bit], dtype=torch.float64)
                b = torch.tensor([b_bit], dtype=torch.float64)
                got = reference_soft_and(a, b).item()
                expected = float(bool(a_bit) and bool(b_bit))
                assert got == expected, \
                    f'soft_and({a_bit},{b_bit}) = {got} != {expected}'

    def test_equals_product(self):
        """内部点でも a*b と厳密一致(定義の固定)。"""
        a = _interior((4, 3), seed=1)
        b = _interior((4, 3), seed=2)
        assert torch.allclose(reference_soft_and(a, b), a * b,
                              atol=TOL_F64_EQ, rtol=0)


class TestSoftXorProperties:
    """soft-XOR残差の性質(MIXER-2の禁止事項の数学的根拠)。"""

    def test_matches_logical_xor(self):
        for a_bit in (0.0, 1.0):
            for b_bit in (0.0, 1.0):
                a = torch.tensor([a_bit], dtype=torch.float64)
                b = torch.tensor([b_bit], dtype=torch.float64)
                got = _soft_xor(a, b).item()
                expected = float(bool(a_bit) != bool(b_bit))
                assert got == expected, \
                    f'soft_xor({a_bit},{b_bit}) = {got} != {expected}'

    def test_self_xor_equals_2x_1mx(self):
        """soft_xor(x,x) = 2x(1-x)。恒等式として厳密に成立する。"""
        x = _interior((5, 4), seed=3)
        assert torch.allclose(_soft_xor(x, x), 2.0 * x * (1.0 - x),
                              atol=TOL_F64_EQ, rtol=0)

    def test_self_xor_is_zero_at_binary(self):
        """二値では soft_xor(x,x) = 0。自己残差は情報を完全に消す。"""
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        assert torch.equal(_soft_xor(x, x), torch.zeros_like(x))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
