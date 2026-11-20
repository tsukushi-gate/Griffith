"""RNNセル→LUT真理値表転写(convert/transcribe_rnn.py)の数理性質テスト。

参照意味論(transcribe_rnn.pyモジュールdocstring):
    combined = [x_t ; h_{t-1}]
    h_t_i = 1[ W_ih[i]・x_t + W_hh[i]・h_{t-1} + b_ih_i + b_hh_i > 0 ]

検証はビット/整数の厳密比較(許容誤差なし)。
"""

import numpy as np
import pytest
import torch

from griffith_logic.convert.binarize import quantize_ternary
from griffith_logic.convert.transcribe_rnn import transcribe_rnn_cell
from griffith_logic.ir.serialize import tt_from_hex


def _all_binary_inputs(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)


def _reference_cell(w_ih, w_hh, b_ih, b_hh, x, h):
    """参照意味論のnumpy実装(閾値ゲートRNNセル、素朴照合)。"""
    z = x @ w_ih.T + h @ w_hh.T + (b_ih + b_hh)
    return (z > 0).astype(np.int64)


class TestTranscribeRnnCellExact:
    """fan-in制約内(疎化不要)なら厳密転写できる。"""

    def test_output_shapes(self):
        torch.manual_seed(0)
        d_in, d_hidden = 2, 2
        w_ih = torch.randn(d_hidden, d_in, dtype=torch.float64)
        w_hh = torch.randn(d_hidden, d_hidden, dtype=torch.float64)
        b_ih = torch.randn(d_hidden, dtype=torch.float64)
        b_hh = torch.randn(d_hidden, dtype=torch.float64)

        result = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=4)
        assert result['indices'].shape == (4, d_hidden)
        assert len(result['truth_tables']) == d_hidden
        assert result['d_in'] == d_in
        assert result['d_hidden'] == d_hidden
        assert result['n_inputs'] == 4

    def test_bitexact_when_faninwithin_budget(self):
        """d_in+d_hidden <= n_inputsなら疎化が恒等になり、全入力で厳密一致する。"""
        torch.manual_seed(1)
        d_in, d_hidden = 2, 2  # combined width = 4 = n_inputs
        w_ih = torch.randn(d_hidden, d_in, dtype=torch.float64)
        w_hh = torch.randn(d_hidden, d_hidden, dtype=torch.float64)
        b_ih = torch.randn(d_hidden, dtype=torch.float64)
        b_hh = torch.randn(d_hidden, dtype=torch.float64)

        result = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=4)
        indices = result['indices']  # (4, d_hidden)
        tts = tt_from_hex(result['truth_tables'])

        combined_patterns = _all_binary_inputs(d_in + d_hidden)  # (16, 4)
        for combined in combined_patterns:
            x = combined[:d_in]
            h = combined[d_in:]
            expected = _reference_cell(
                w_ih.numpy(), w_hh.numpy(), b_ih.numpy(), b_hh.numpy(), x, h,
            )
            for i in range(d_hidden):
                # indices[:, i]はcombinedベクトル内のグローバル位置(fan-in充足時は
                # 恒等疎化なのでsetは{0,1,2,3}全体と一致するはず)。
                addr_bits = combined[indices[:, i]]
                addr = int(sum(b << j for j, b in enumerate(addr_bits)))
                bit = (tts[i] >> addr) & 1
                assert bit == expected[i], (
                    f'combined={combined} neuron={i}: '
                    f'転写tt={bit} != 参照={expected[i]}'
                )

    def test_zero_weights_gives_constant_gate(self):
        """全結合0・バイアス正なら常に1を出す真理値表になる(境界ケース)。"""
        d_in, d_hidden = 2, 2
        w_ih = torch.zeros(d_hidden, d_in, dtype=torch.float64)
        w_hh = torch.zeros(d_hidden, d_hidden, dtype=torch.float64)
        b_ih = torch.ones(d_hidden, dtype=torch.float64)
        b_hh = torch.zeros(d_hidden, dtype=torch.float64)

        result = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=4)
        tts = tt_from_hex(result['truth_tables'])
        full_mask = (1 << (1 << 4)) - 1  # n_inputs=4 → 2^4=16エントリ全て1
        for tt in tts:
            assert tt == full_mask

    def test_and_gate_semantics(self):
        """手組みAND相当のRNNセル(d_hidden=1)が真理値表で厳密にANDになる。"""
        # h_t = 1[1*x0 + 1*h_prev - 1.5 > 0] = x0 AND h_prev
        w_ih = torch.tensor([[1.0]], dtype=torch.float64)
        w_hh = torch.tensor([[1.0]], dtype=torch.float64)
        b_ih = torch.tensor([-1.5], dtype=torch.float64)
        b_hh = torch.tensor([0.0], dtype=torch.float64)

        result = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=2)
        indices = result['indices']  # (2, 1)
        tts = tt_from_hex(result['truth_tables'])
        assert len(tts) == 1

        for x0 in (0, 1):
            for h_prev in (0, 1):
                combined = np.array([x0, h_prev])
                addr_bits = combined[indices[:, 0]]
                addr = int(sum(b << j for j, b in enumerate(addr_bits)))
                bit = (tts[0] >> addr) & 1
                assert bit == (x0 & h_prev), f'x0={x0} h_prev={h_prev}'


class TestTranscribeRnnCellPruned:
    """fan-in超過(疎化発生)でも出力形状・型は常に規約通り。"""

    def test_pruned_fanin_shapes_and_calib_recalibration(self):
        torch.manual_seed(2)
        d_in, d_hidden = 6, 6  # combined width = 12 > n_inputs=4
        w_ih = torch.randn(d_hidden, d_in, dtype=torch.float64)
        w_hh = torch.randn(d_hidden, d_hidden, dtype=torch.float64)
        b_ih = torch.randn(d_hidden, dtype=torch.float64)
        b_hh = torch.randn(d_hidden, dtype=torch.float64)

        calib = np.random.RandomState(0).randint(0, 2, size=(64, d_in + d_hidden))
        result = transcribe_rnn_cell(
            w_ih, w_hh, b_ih, b_hh, n_inputs=4, calib=calib,
        )
        assert result['indices'].shape == (4, d_hidden)
        assert len(result['truth_tables']) == d_hidden
        # 疎化されたインデックスは範囲内であること
        assert result['indices'].min() >= 0
        assert result['indices'].max() < d_in + d_hidden

    def test_fanin_deficit_pads_with_zero_slot(self):
        """d_in+d_hidden < n_inputsのときスロット0でパディングされる。"""
        d_in, d_hidden = 1, 1  # combined width = 2 < n_inputs=4
        w_ih = torch.tensor([[1.0]], dtype=torch.float64)
        w_hh = torch.tensor([[1.0]], dtype=torch.float64)
        b_ih = torch.tensor([-1.5], dtype=torch.float64)
        b_hh = torch.tensor([0.0], dtype=torch.float64)

        result = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=4)
        assert result['indices'].shape == (4, d_hidden)
        # パディングスロットはindex 0(sparsify_fanin/transcribe_mlpと同じ規約)
        assert result['indices'][2:, 0].tolist() == [0, 0]


class TestTranscribeRnnCellMasks:
    """ternary量子化マスク(masks)を渡すと sparsify_from_mask 経由で
    マスク内接続のみが残る(sparsify_fanin の |W|上位k と対比)。

    参照意味論(binarize.quantize_ternary + sparsify_from_mask):
        combined重み w_combined=[w_ih|w_hh] を ternary量子化した実効重み
        (q·scale, q∈{-1,0,+1})を、mask=Trueの接続からのみ |W|上位k本選ぶ。
        maskで0にされた接続は選ばれず、その閾値ゲート出力は選外前提の値になる。
    検証はビット/整数の厳密比較(許容誤差なし)。
    """

    def test_masks_selects_masked_in_connections_only(self):
        """masksを渡すと、mask=Falseのグローバル位置はindicesに出現しない。

        d_in+d_hidden > n_inputs で疎化が必ず発生する設定にし、ternaryの
        0マスク位置がひとつもindicesへ選ばれないことを厳密に確認する。
        """
        torch.manual_seed(7)
        d_in, d_hidden = 5, 5  # combined width = 10 > n_inputs=4
        w_ih = torch.randn(d_hidden, d_in, dtype=torch.float64)
        w_hh = torch.randn(d_hidden, d_hidden, dtype=torch.float64)
        b_ih = torch.randn(d_hidden, dtype=torch.float64)
        b_hh = torch.randn(d_hidden, dtype=torch.float64)

        # 結合重みをternary量子化して0マスクを得る(combined=[w_ih|w_hh])
        w_combined = torch.cat([w_ih, w_hh], dim=1)  # (d_hidden, 10)
        _q, _scale, mask = quantize_ternary(w_combined, per='channel')

        result = transcribe_rnn_cell(
            w_ih, w_hh, b_ih, b_hh, n_inputs=4, masks=mask,
        )
        indices = result['indices']  # (4, d_hidden)
        assert indices.shape == (4, d_hidden)

        # 各ニューロンで選ばれたグローバル位置は、そのニューロンのmask=Trueに
        # 含まれるか、または埋めスロット(重み0→sparsify_from_maskは実重み0の
        # 接続を index0 で複製せず gather位置をそのまま返すが mask外)。
        # mask内非ゼロ数 >= k のニューロンは全スロットmask=True位置になる。
        mask_np = mask.numpy()
        for i in range(d_hidden):
            n_masked_in = int(mask_np[i].sum())
            sel_positions = indices[:, i]
            # k=min(4, 10)=4。mask内接続がk本以上なら選ばれた全位置がmask=True。
            if n_masked_in >= 4:
                for pos in sel_positions:
                    assert mask_np[i, int(pos)], (
                        f'neuron={i}: 選択位置{pos}がmask=False(ternary 0)なのに'
                        f'indicesへ選ばれた(sparsify_from_maskがマスクを無視)'
                    )

    def test_masks_differs_from_fanin_when_mask_excludes_top(self):
        """mask=Trueが|W|上位と食い違う手組みケースで、masksありは
        sparsify_fanin(|W|上位)と異なる接続を選ぶ(マスク尊重の厳密確認)。"""
        # d_in=3, d_hidden=1 → combined width=4(位置0,1,2=w_ih、位置3=w_hh)。
        # n_inputs=2。|W|上位2本は位置0,1(重み大)。だが mask はそれらを0
        # (枝刈り)にして位置2,3のみTrueにする → masksありは位置2,3を選ぶはず。
        w_ih = torch.tensor([[10.0, 8.0, 3.0]], dtype=torch.float64)  # 位置0,1,2
        w_hh = torch.tensor([[2.0]], dtype=torch.float64)            # 位置3(小)
        b_ih = torch.tensor([0.0], dtype=torch.float64)
        b_hh = torch.tensor([0.0], dtype=torch.float64)

        # masksなし: |W|上位2本 = 位置0,1
        res_fanin = transcribe_rnn_cell(w_ih, w_hh, b_ih, b_hh, n_inputs=2)
        assert set(res_fanin['indices'][:, 0].tolist()) == {0, 1}

        # masksあり: 位置2,3のみTrue → 位置2,3を選ぶ
        mask = torch.tensor([[False, False, True, True]])
        res_mask = transcribe_rnn_cell(
            w_ih, w_hh, b_ih, b_hh, n_inputs=2, masks=mask,
        )
        assert set(res_mask['indices'][:, 0].tolist()) == {2, 3}, (
            'masks=Trueが位置2,3なのに、それ以外が選ばれた(マスク無視)'
        )

    def test_masks_shape_mismatch_raises(self):
        """masksの列数が d_in+d_hidden と不一致なら明示エラー。"""
        w_ih = torch.zeros(2, 2, dtype=torch.float64)
        w_hh = torch.zeros(2, 2, dtype=torch.float64)
        bad_mask = torch.ones(2, 3, dtype=torch.bool)  # 列数3 != 4
        with pytest.raises(ValueError, match='masks'):
            transcribe_rnn_cell(w_ih, w_hh, None, None, n_inputs=4, masks=bad_mask)


class TestTranscribeRnnCellValidation:
    def test_bad_n_inputs(self):
        w_ih = torch.zeros(2, 2, dtype=torch.float64)
        w_hh = torch.zeros(2, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match='n_inputs'):
            transcribe_rnn_cell(w_ih, w_hh, None, None, n_inputs=3)

    def test_shape_mismatch(self):
        w_ih = torch.zeros(2, 2, dtype=torch.float64)
        w_hh = torch.zeros(3, 3, dtype=torch.float64)  # d_hiddenがw_ihと不整合
        with pytest.raises(ValueError, match='weight_hh'):
            transcribe_rnn_cell(w_ih, w_hh, None, None, n_inputs=4)

    def test_none_bias_treated_as_zero(self):
        """bias未指定(None)は0扱い(nn.RNNのbias=False相当)。"""
        w_ih = torch.tensor([[1.0]], dtype=torch.float64)
        w_hh = torch.tensor([[0.0]], dtype=torch.float64)
        result_none = transcribe_rnn_cell(w_ih, w_hh, None, None, n_inputs=2)
        zero = torch.zeros(1, dtype=torch.float64)
        result_zero = transcribe_rnn_cell(w_ih, w_hh, zero, zero, n_inputs=2)
        assert result_none['truth_tables'] == result_zero['truth_tables']
