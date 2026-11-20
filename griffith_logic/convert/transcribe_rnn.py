"""vanilla RNNセル → LUT真理値表への構造転写(RNN→RDDLGN、D-RNN1)。

`transcribe.py` が確立した閾値ゲート転写機構(`sparsify_fanin` +
`threshold_truth_tables`)を、MLPの隠れ層ではなく **RNNセルの1ステップ更新**
に適用する。

対象意味論(nn.RNN、nonlinearity='relu'相当の閾値近似):
    h_t = f(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

本モジュールが転写するのはこの `f` を `1[z > 0]`(閾値ゲート、
`transcribe.py` の隠れ層意味論と同一)に置き換えた版であり、tanh/reLUの
連続非線形性そのものではない。すなわち:

    h_t^{bin} = 1[ W_ih @ x_t^{bin} + b_ih + W_hh @ h_{t-1}^{bin} + b_hh > 0 ]

これは `[x_t ; h_{t-1}]`(結合ベクトル、幅 d_in + d_hidden)を入力とする
d_hidden個の独立な閾値ゲートの並びであり、`transcribe.py` の隠れ層1層と
数学的に同一の構造(fan-inが d_in+d_hidden に増えるだけ)。したがって
`sparsify_fanin`(|W|上位k本への疎化+バイアス再較正)と
`threshold_truth_tables`(全列挙による真理値表化)をそのまま転用できる。

出力するdictは `griffith_logic.ir.weights.restore_lut_layer` にそのまま
渡せる形式(indices: (n_inputs, out_dim) int64、truth_tables: hex文字列列)
にしてあり、RDDLGNの再帰LutStack内の対応する`LutLayer`をin-placeで
このセルの近似関数へ初期化できる(training-free構造転写)。

限界(実験スクリプト側で検証・報告する事項。本モジュールは転写のみを担当):
    - tanh/ReLUの連続非線形性 → 二値閾値ゲートへの離散化は情報を失う
      (teacher forward時の隠れ状態は連続値だが、転写後は各ステップで
      1bitへ丸められる。再帰的にステップを重ねると誤差が蓄積し得る)。
    - 疎化(fan-in > n_inputsの場合)は`sparsify_fanin`と同じ近似性質を持つ
      (見出しの接続以外の寄与をバイアスへ再較正するのみ)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..ir.serialize import tt_to_hex
from .transcribe import (
    sparsify_fanin,
    sparsify_from_mask,
    threshold_truth_tables,
)


@dataclass
class RnnCellCalib:
    """RNNセル転写用の較正データ([x_t ; h_{t-1}]結合ベクトル、0/1)。

    Attributes:
        combined: (B, d_in + d_hidden) の0/1較正入力。Noneなら較正なし
            (疎化バイアス再較正が0になる。TranscribeSpec.calib_inputsと同じ規約)。
    """

    combined: np.ndarray | torch.Tensor | None = None


def transcribe_rnn_cell(
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    *,
    n_inputs: int = 2,
    calib: np.ndarray | torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> dict:
    """vanilla RNNセルの1ステップ更新を閾値ゲートLUT網へ転写する。

    `nn.RNN`(1層、`batch_first`非依存、1タイムステップぶんのパラメータ)の
    `weight_ih_l0`/`weight_hh_l0`/`bias_ih_l0`/`bias_hh_l0` をそのまま渡せる。

    意味論(モジュールdocstring参照):
        combined = [x_t ; h_{t-1}]  (d_in + d_hidden 次元、結合順は入力→隠れ)
        h_t^{bin}_i = 1[ (W_ih[i] . x_t + W_hh[i] . h_{t-1}) + (b_ih_i + b_hh_i) > 0 ]

    Args:
        weight_ih: (d_hidden, d_in) 入力→隠れの重み(nn.RNNのweight_ih_l0)。
        weight_hh: (d_hidden, d_hidden) 隠れ→隠れの重み(weight_hh_l0)。
        bias_ih: (d_hidden,) 入力側バイアス、またはNone(0扱い)。
        bias_hh: (d_hidden,) 隠れ側バイアス、またはNone(0扱い)。
        n_inputs: 転写後LUTのfan-in(LutLayer対応値: 2/4/6)。
            d_in + d_hidden がこれを超える場合は`sparsify_fanin`で
            |W|上位n_inputs本へ疎化し、落とした接続の寄与をバイアスへ
            再較正する(transcribe.pyと同じ近似)。
        calib: (B, d_in + d_hidden) の0/1較正データ(疎化バイアス再較正用)。
            Noneなら較正なし(落とした接続の寄与を単純に無視)。
        masks: (d_hidden, d_in + d_hidden) bool の残存接続マスク
            (ternary量子化由来、combined=[x_t;h_{t-1}]の重み並び順に対応)。
            指定時は `sparsify_fanin`(|W|上位k)でなく `sparsify_from_mask`
            でmask=Trueの接続からのみ選ぶ(枝刈り済み構造を尊重、追加の
            magnitude枝刈りをしない)。マスク内非ゼロがn_inputsを超える
            ニューロンは|W|上位n_inputsへ切り捨て、切り捨て分を較正で
            バイアス再較正する。Noneなら従来の|W|上位k疎化。

    Returns:
        dict:
            'indices'      : (n_inputs, d_hidden) int64 ndarray
                (`ir.weights.restore_lut_layer` の indices 引数にそのまま渡せる)
            'truth_tables' : (d_hidden,) hex文字列list
                (`restore_lut_layer` の truth_tables_hex 引数にそのまま渡せる)
            'n_inputs'     : int(実際に使ったfan-in、= 引数のn_inputs)
            'd_in'         : int(元のd_in、combinedにおけるxブロック幅)
            'd_hidden'     : int(元のd_hidden、combinedにおけるhブロック幅)

    Raises:
        ValueError: n_inputsが2/4/6以外、または重みの形状が不整合。
    """
    if n_inputs not in (2, 4, 6):
        raise ValueError(f'n_inputsは2/4/6のみ: {n_inputs}')

    d_hidden, d_in = weight_ih.shape
    if weight_hh.shape != (d_hidden, d_hidden):
        raise ValueError(
            f'weight_hh形状 {tuple(weight_hh.shape)} != '
            f'({d_hidden}, {d_hidden})(weight_ihから推定したd_hidden)'
        )

    w_ih = weight_ih.detach().to(torch.float64)
    w_hh = weight_hh.detach().to(torch.float64)
    b_ih = (bias_ih.detach().to(torch.float64) if bias_ih is not None
            else torch.zeros(d_hidden, dtype=torch.float64))
    b_hh = (bias_hh.detach().to(torch.float64) if bias_hh is not None
            else torch.zeros(d_hidden, dtype=torch.float64))

    # combined = [x_t ; h_{t-1}] の結合重み: (d_hidden, d_in + d_hidden)
    w_combined = torch.cat([w_ih, w_hh], dim=1)
    b_combined = b_ih + b_hh

    if masks is not None:
        if tuple(masks.shape) != (d_hidden, d_in + d_hidden):
            raise ValueError(
                f'masks形状 {tuple(masks.shape)} != '
                f'({d_hidden}, {d_in + d_hidden})(d_hidden, d_in+d_hidden)'
            )

    calib_np = None
    if calib is not None:
        calib_np = np.asarray(
            calib.detach().cpu().numpy() if isinstance(calib, torch.Tensor)
            else calib,
        ).astype(np.int64)
        assert set(np.unique(calib_np)).issubset({0, 1}), '較正データは0/1のみ'
        if calib_np.shape[1] != d_in + d_hidden:
            raise ValueError(
                f'calibの列数({calib_np.shape[1]}) != d_in+d_hidden'
                f'({d_in + d_hidden})'
            )

    if masks is not None:
        indices, w_sel, bias_shift = sparsify_from_mask(
            w_combined, masks.to(w_combined.device), n_inputs,
            calib_inputs=calib_np,
        )
    else:
        indices, w_sel, bias_shift = sparsify_fanin(
            w_combined, n_inputs, calib_inputs=calib_np,
        )
    b_eff = b_combined + bias_shift

    # fan-in不足(d_in+d_hidden < n_inputs)はスロット複製(index 0)+重み0で
    # LutLayer規約(2/4/6)に合わせる(transcribe_mlpと同じパディング規約)。
    k_actual = indices.shape[0]
    if k_actual < n_inputs:
        pad = n_inputs - k_actual
        indices = np.vstack(
            [indices, np.zeros((pad, indices.shape[1]), dtype=np.int64)]
        )
        w_sel = torch.cat(
            [w_sel, torch.zeros(w_sel.shape[0], pad, dtype=w_sel.dtype)],
            dim=1,
        )

    tts = threshold_truth_tables(w_sel, b_eff)

    return {
        'indices': indices,
        'truth_tables': tt_to_hex(tts),
        'n_inputs': n_inputs,
        'd_in': d_in,
        'd_hidden': d_hidden,
    }
