"""Transformer FFN→LogicMLP構造転写(convert/transcribe_seq)のテスト(D-attn-ffn)。

transcribe_ffn_blockはtranscribe.pyのsparsify_fanin/threshold_truth_tablesを
再利用してFFN(Linear→GELU→Linear)を転写する。GELU吸収(閾値二値化)自体は
厳密(GELU(z)>0 ⟺ z>0)だが、以下は近似が生じうる箇所であり、モジュール
docstringの規約通り0を主張しない:
    - fan-in疎化(sparsify_fanin、n_inputs<in_featuresのとき)
    - 中間ユニット部分選択(hidden_width<教師の中間幅のとき)
    - 出力層の閾値二値化(FFN出力の振幅情報は失われ符号のみ保持)

検証観点:
    (a) 内部整合性(厳密): 転写IRのsimulateが、transcribe_ffn_blockが
        実際に構築した真理値表の定義する関数と厳密に一致すること
    (b) fan-in>=in_featuresかつhidden_width=full時は厳密転写になること
        (transcribe.pyのTestExactTranscriptionと同じ論理)
    (c) LogicMLPへの書き込み(init_logic_mlp_from_ffn)が、書き込み後の
        LogicMLP(eval, hard forward)がIRのsimulateとビット厳密一致すること
    (d) 近似が生じる設定(部分選択)でのagreement率が[0,1]の妥当な値で
        報告されること(1.0を主張しない)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from griffith_logic.convert.transcribe_seq import (
    FfnTranscribeSpec,
    extract_ffn_linears,
    init_logic_mlp_from_ffn,
    transcribe_ffn_block,
)
from griffith_logic.hybrid.policy import LogicSpec
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.layers.logic_linear import LogicMLP


def _all_binary_inputs(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


class _ToyFFN(nn.Module):
    """karpathy nanoGPT `_MLP`と同じ属性名(c_fc/c_proj)を持つ最小FFN。"""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden)
        self.c_proj = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


def _reference_ffn_bits(fc: nn.Linear, proj: nn.Linear, x: np.ndarray) -> np.ndarray:
    """GELU閾値二値化+出力符号二値化の参照numpy実装(教師FFNの二値化版)。"""
    xf = x.astype(np.float64)
    w_fc = fc.weight.detach().numpy().astype(np.float64)
    b_fc = fc.bias.detach().numpy().astype(np.float64)
    hidden = ((xf @ w_fc.T + b_fc) > 0).astype(np.float64)
    w_proj = proj.weight.detach().numpy().astype(np.float64)
    b_proj = proj.bias.detach().numpy().astype(np.float64)
    out = ((hidden @ w_proj.T + b_proj) > 0).astype(np.int64)
    return out


class TestExtractFfnLinears:
    def test_c_fc_c_proj_attrs(self):
        ffn = _ToyFFN(4, 8)
        fc, proj = extract_ffn_linears(ffn)
        assert fc is ffn.c_fc
        assert proj is ffn.c_proj

    def test_sequential_two_linears(self):
        ffn = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        fc, proj = extract_ffn_linears(ffn)
        assert fc is ffn[0]
        assert proj is ffn[2]

    def test_wrong_linear_count_raises(self):
        with pytest.raises(ValueError, match='2層Linear'):
            extract_ffn_linears(nn.Sequential(nn.Linear(4, 4)))


class TestExactTranscription:
    """fan-in制約内・部分選択なしのFFNは厳密転写できる。"""

    def test_small_ffn_bitexact(self):
        """n_inputs >= max(in_features, hidden)なら両層ともfan-in疎化が
        恒等になり厳密転写(hidden層はin_features以下、output層はhidden幅
        以下のn_inputsが必要。ここではn_inputs=6=hiddenで両方満たす)。"""
        torch.manual_seed(0)
        d_model, hidden = 4, 6
        ffn = _ToyFFN(d_model, hidden).double()
        x = _all_binary_inputs(d_model)

        ir, info = transcribe_ffn_block(
            ffn, x, FfnTranscribeSpec(n_inputs=6),
        )
        assert validate(ir) == []

        out_bits = simulate(ir, x)
        ref = _reference_ffn_bits(ffn.c_fc, ffn.c_proj, x)
        assert np.array_equal(out_bits, ref), 'fan-in制約内のFFNは厳密転写'
        assert info['hidden_agreement'] == pytest.approx(1.0)
        assert info['output_agreement'] == pytest.approx(1.0)
        assert info['transcribed_hidden_width'] == hidden
        assert info['full_hidden_width'] == hidden

    def test_internal_consistency_always_holds(self):
        """fan-in疎化が近似を生む場合でも、IRのsimulateは転写が構築した
        真理値表の関数と常に厳密一致する(内部整合性、近似誤差とは独立)。"""
        torch.manual_seed(1)
        d_model, hidden = 6, 10
        ffn = _ToyFFN(d_model, hidden).double()
        x = _all_binary_inputs(d_model)

        # n_inputs=2 < in_features=6 なのでhidden層はfan-in疎化(近似)される
        ir, info = transcribe_ffn_block(ffn, x, FfnTranscribeSpec(n_inputs=2))
        assert validate(ir) == []

        # 内部整合性: simulateが「実際に選んだ真理値表・接続」を再現すること
        hidden_node = ir.node_by_id('ffn_hidden_lut')
        output_node = ir.node_by_id('ffn_output_lut')
        from griffith_logic.convert.transcribe_seq import _simulate_lut_layer
        from griffith_logic.ir.serialize import tt_from_hex

        h_bits = _simulate_lut_layer(
            x, hidden_node.params['indices'],
            tt_from_hex(hidden_node.params['truth_tables']),
        )
        expected_out = _simulate_lut_layer(
            h_bits, output_node.params['indices'],
            tt_from_hex(output_node.params['truth_tables']),
        )
        sim_out = simulate(ir, x)
        assert np.array_equal(sim_out, expected_out), (
            'simulate(ir, x)は転写が構築した真理値表列を厳密に合成した結果と一致するべき'
        )

        # 近似誤差は正直に[0,1]区間で報告され、1.0を主張しない
        assert 0.0 <= info['hidden_agreement'] <= 1.0
        assert 0.0 <= info['output_agreement'] <= 1.0


class TestUnitSelection:
    """hidden_width<教師の中間幅での部分選択(情報損失を伴う近似)。"""

    def test_partial_hidden_width_reduces_agreement_domain(self):
        torch.manual_seed(2)
        d_model, hidden = 4, 6
        ffn = _ToyFFN(d_model, hidden).double()
        x = _all_binary_inputs(d_model)

        # n_inputs=6=hiddenなので"full"側は両層ともfan-in疎化が恒等(厳密)。
        # "partial"側はhidden_width=4<6でユニット部分選択の近似が入る。
        spec_full = FfnTranscribeSpec(n_inputs=6, hidden_width=hidden)
        spec_partial = FfnTranscribeSpec(n_inputs=6, hidden_width=4)

        ir_full, info_full = transcribe_ffn_block(ffn, x, spec_full)
        ir_partial, info_partial = transcribe_ffn_block(ffn, x, spec_partial)

        assert info_full['transcribed_hidden_width'] == hidden
        assert info_partial['transcribed_hidden_width'] == 4
        assert info_partial['selected_hidden_units'].shape == (4,)
        # 部分選択は一般に出力agreementを完全な1.0未満へ下げうる(honest近似)。
        # 厳密な不等号は保証できない(たまたま選んだユニットで十分な場合が
        # あるため)、[0,1]の妥当範囲であることのみ確認する。
        assert 0.0 <= info_partial['output_agreement'] <= 1.0
        assert info_full['output_agreement'] == pytest.approx(1.0)

    def test_hidden_width_larger_than_teacher_raises(self):
        ffn = _ToyFFN(4, 6).double()
        x = _all_binary_inputs(4)
        with pytest.raises(ValueError, match='hidden_width'):
            transcribe_ffn_block(ffn, x, FfnTranscribeSpec(n_inputs=2, hidden_width=8))


class TestInitLogicMlpFromFfn:
    """LogicMLPへの重み書き込み(init_logic_mlp_from_ffn)。"""

    def _matching_logic_mlp(self, d_model: int, n_inputs: int = 2) -> LogicMLP:
        """transcribe_ffn_blockのdepth=2出力(hidden=out=d_model)と
        shapeが一致するLogicMLPを構築する(decoder='identity'でgroup_size=1、
        中間層の平均化を避ける。モジュールdocstring参照)。"""
        spec = LogicSpec(
            encoder='identity', decoder='identity', width=d_model,
            n_inputs=n_inputs, parameterization='softmax',
            connections='random', depth=2,
        )
        return LogicMLP(d_model, spec=spec)

    def test_write_then_hard_forward_matches_ir_simulate(self):
        """n_inputs=2(softmax→16ゲートモード)を使う: LogicGPTConfig既定と同じ
        n_inputs=2はweights_from_truth_tablesが全16関数を表現できる
        (N>2 softmaxは単一minterm/FALSEしか表現できないため、任意真理値表を
        書き込むにはn_inputs=2か、独立/raw/warpパラメータ化が必要。
        ir/weights.pyモジュールdocstring参照)。"""
        torch.manual_seed(3)
        d_model, hidden = 4, 4  # hidden_width==d_modelでLogicMLP shapeと一致
        ffn = _ToyFFN(d_model, hidden).double()
        x_np = _all_binary_inputs(d_model)

        logic_mlp = self._matching_logic_mlp(d_model, n_inputs=2)
        info = init_logic_mlp_from_ffn(logic_mlp, ffn, x_np)

        # 書き込み後、hard forward(eval)がIRのsimulateとビット厳密一致
        ir, _ = transcribe_ffn_block(
            ffn, x_np, FfnTranscribeSpec(n_inputs=2, hidden_width=hidden, out_width=d_model),
        )
        sim_out = simulate(ir, x_np)

        logic_mlp.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x_np, dtype=torch.float32)
            hard_out = logic_mlp(x_t).cpu().numpy()

        assert np.array_equal(hard_out.astype(np.int64), sim_out), (
            'LogicMLPへ書き込み後のhard forwardは転写IRのsimulateと厳密一致するべき'
        )
        assert 0.0 <= info['output_agreement'] <= 1.0

    def test_shape_mismatch_raises(self):
        """FFN(in=hidden=out=4)自体は自己整合的だが、logic_mlpのwidth=6と
        不一致(n_inputsも異なる)なため、restore前のshape検証で拒否される。"""
        torch.manual_seed(4)
        ffn = _ToyFFN(4, 4).double()  # in=hidden=out=4で自己整合(out_widthエラーは出ない)
        logic_mlp = self._matching_logic_mlp(d_model=6, n_inputs=2)  # d_model不一致
        with pytest.raises(ValueError, match='形状|out_width'):
            init_logic_mlp_from_ffn(logic_mlp, ffn, _all_binary_inputs(4))

    def test_depth_mismatch_raises(self):
        ffn = _ToyFFN(4, 4).double()
        spec = LogicSpec(
            encoder='identity', decoder='identity', width=4,
            n_inputs=2, parameterization='softmax', depth=3,
        )
        logic_mlp = LogicMLP(4, spec=spec)
        with pytest.raises(ValueError, match='depth=2'):
            init_logic_mlp_from_ffn(logic_mlp, ffn, _all_binary_inputs(4))


class TestSpecValidation:
    def test_bad_n_inputs(self):
        with pytest.raises(ValueError, match='n_inputs'):
            FfnTranscribeSpec(n_inputs=3)

    def test_bad_unit_select(self):
        with pytest.raises(ValueError, match='unit_select'):
            FfnTranscribeSpec(unit_select='random')
