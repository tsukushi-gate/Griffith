"""ハイブリッドNNの論理ゲート化レシピ。

LogicSpec は「実数レイヤを論理ゲート型レイヤに置き換えるときの
生成方針」を1つのdataclassに集約する。LogicLinear/LogicMLP、
convert_linear_to_logic / replace_mlp_with_lut が共通で使う。
"""

from dataclasses import dataclass


@dataclass
class LogicSpec:
    """論理ゲート型レイヤの生成レシピ。

    Attributes:
        n_inputs: LUTの入力数(2/4/6)
        parameterization: LutLayerのパラメータ化方式 + 'dwn'(DWNLayer使用)
        selection: 'soft' | 'hard_st'(CAGE方式。dwnでは無視)
        depth: LUT層の段数
        width: 中間層の幅(None → out_features × neurons_per_output)
        neurons_per_output: GroupSum前の出力あたり冗長ニューロン数
        encoder: 入力境界 'sigmoid' | 'binarize_ste' | 'identity' | 'thermometer'
        encoder_tau: sigmoidエンコーダの初期温度
        encoder_bits: thermometerエンコーダの特徴あたりビット数
        decoder: 出力境界 'groupsum_centered' | 'groupsum' | 'layernorm' | 'identity'
        decoder_tau: デコーダ温度(None → √グループサイズ)
        connections: 'random' | 'unique'(dwnでは第1層learnable/以降random)
        grad_factor: LUT層の勾配スケール(深い構成では>1。dwnでは無視)
    """

    n_inputs: int = 2
    parameterization: str = 'softmax'
    selection: str = 'soft'
    depth: int = 2
    width: int | None = None
    neurons_per_output: int = 8
    encoder: str = 'sigmoid'
    encoder_tau: float = 1.0
    encoder_bits: int = 8
    decoder: str = 'groupsum_centered'
    decoder_tau: float | None = None
    connections: str = 'random'
    grad_factor: float = 1.0

    def resolve_width(self, out_features: int) -> int:
        """中間幅を決定(out_featuresの倍数に切り上げ)"""
        width = self.width or out_features * self.neurons_per_output
        if width % out_features != 0:
            width = ((width // out_features) + 1) * out_features
        return width
