"""Transformer FFNブロック→LogicMLPの構造転写(D-attn-ffn)。

(experiments/convert_arith_structural_transfer.py)が実証した「構造転写(配線+ゲート
初期化)+ soft磨きはIO-KD/scratchを圧倒する」という知見を、算術回路から
Transformer FFN(`Linear→GELU→Linear`)へ拡張する。attention(softmax+q/k/v)は
文献知見(Launay 2020)通りBP-freeにできないため対象外とし、FFN部分のみを
`transcribe_mlp`(convert/transcribe.py)の機構で転写する **部分(FFN限定)構造転写**。

transcribe.pyを改造せず、`sparsify_fanin`/`threshold_truth_tables`/
`TranscribeSpec`をそのままimportして再利用する(モジュールdocstringの規約
「参照実装は書き換えない」はtranscribe.py自体には適用されないが、既存の
厳密転写機構を壊さないよう変更を加えない意図は同じ)。

`transcribe_mlp`をそのまま使わない理由(重要な設計判断):
    `transcribe_mlp`の最終層(`_final_layer_nodes`)は「argmax厳密保存の
    整数量子化class score」という分類器専用の意味論を持つ
    (transcribe.pyモジュールdocstring参照)。FFNの出力はclass scoreではなく
    残差ストリームに加算される実数特徴(のちに`FromLogic`で論理域[0,1]へ
    戻す)なので、この意味論は当てはまらない。そのためFFNの**両方の**Linear層
    (`c_fc`と`c_proj`)を`sparsify_fanin`+`threshold_truth_tables`で**均等に**
    (GELU閾値二値化として)扱う(`transcribe_mlp`の「隠れ層」意味論を両層に
    適用する。「最終層」特殊扱いはしない)。

トポロジの制約(正直に明記する近似):
    `LogicMLP`(layers/logic_linear.py)の`luts`は`spec.depth`段の**同一幅**
    `LutLayer`列である(`LogicTransformerBlock`が使う既定specはdepth=2、
    幅=d_model×mlp_ratioで統一)。教師FFNは`d_model → mlp_ratio*d_model →
    d_model`という非対称な幅を持つため、`LogicMLP`の均一幅制約とは一致しない。
    本モジュールは`spec.width == d_model`（`decoder='identity'`または
    `group_size=1`のgroupsum）に設定したLogicMLPを転写先として想定し、
    教師の中間層(`mlp_ratio*d_model`個のGELUユニット)から`d_model`個だけを
    `sparsify_fanin`と同じ|重み|上位選択の発想で部分列挙し、その部分集合の
    出力をhidden layerの真理値表として厳密転写する。これは**情報損失を伴う
    近似**(教師の中間表現の一部しか保持しない)であり、その近似誤差を
    `transcribe_ffn_block`が返す`info`辞書で較正データ上のagreement率として
    正直に報告する(0を主張しない、transcribe_conv2dと同じ規約)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex
from ..ir.weights import restore_lut_layer
from .transcribe import sparsify_fanin, threshold_truth_tables

__all__ = [
    'FfnTranscribeSpec',
    'extract_ffn_linears',
    'init_logic_mlp_from_ffn',
    'transcribe_ffn_block',
]


@dataclass
class FfnTranscribeSpec:
    """FFN転写仕様(TranscribeSpecのFFN向け薄いラッパ)。

    Attributes:
        n_inputs: LutLayerのfan-in(2/4/6のいずれか。transcribe.TranscribeSpecと同じ制約)
        hidden_width: 転写後hidden lut_layerの出力幅。Noneなら教師のc_fc.out_features
            をそのまま使う(近似なし、ただしLogicMLPへ書き込む場合は
            対象LutLayerのout_dimと一致させる必要がある。通常はLogicMLPの
            spec.widthに合わせてこれを指定し、教師中間ユニットの部分選択を行う)
        out_width: 転写後output lut_layerの出力幅。Noneなら教師のc_proj.out_features
            (=d_model)をそのまま使う
        unit_select: hidden_widthが教師のc_fc.out_featuresより小さいとき、
            どのユニット部分集合を残すかの方策。'l2_norm'(既定)=各hidden
            ユニットの(c_fc行ベクトルのL2ノルム × c_proj列ベクトルのL2ノルム)
            が大きい順に選ぶ(そのユニットが出力に与える影響の粗い指標)。
            'first'=先頭から順に選ぶ(デバッグ用)
    """

    n_inputs: int = 2
    hidden_width: int | None = None
    out_width: int | None = None
    unit_select: str = 'l2_norm'

    def __post_init__(self):
        if self.n_inputs not in (2, 4, 6):
            raise ValueError(f'n_inputsは2/4/6のみ: {self.n_inputs}')
        if self.unit_select not in ('l2_norm', 'first'):
            raise ValueError(f"unit_selectは'l2_norm'か'first': {self.unit_select}")


def extract_ffn_linears(mlp: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """FFNモジュールから(c_fc相当, c_proj相当)の2つのnn.Linearを取り出す。

    対応形状:
        - `.c_fc`/`.c_proj`属性を持つモジュール(text_teacher.py `_MLP`、
          karpathy nanoGPT形式)
        - 直下または再帰的に`nn.Linear`をちょうど2つ含む`nn.Sequential`等
          (`Linear→活性化→Linear`の任意のコンテナ)

    Returns:
        (fc, proj): fc.out_features が中間(GELU)幅、proj.out_featuresが出力幅。

    Raises:
        ValueError: nn.Linearがちょうど2つ見つからない場合
    """
    if hasattr(mlp, 'c_fc') and hasattr(mlp, 'c_proj'):
        fc, proj = mlp.c_fc, mlp.c_proj
        if not (isinstance(fc, nn.Linear) and isinstance(proj, nn.Linear)):
            raise ValueError(
                'c_fc/c_projはnn.Linearである必要がある: '
                f'{type(fc)}, {type(proj)}'
            )
        return fc, proj

    linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise ValueError(
            f'FFN転写は2層Linear(c_fc/c_proj相当)のみ対応: '
            f'{len(linears)}個のLinearが見つかった'
        )
    return linears[0], linears[1]


def _select_hidden_units(
    fc: nn.Linear, proj: nn.Linear, hidden_width: int, method: str,
) -> np.ndarray:
    """教師の中間(GELU)ユニットからhidden_width個を部分選択する。

    Returns:
        (hidden_width,) int64 ndarray — 選択された中間ユニットのindex(昇順)。
    """
    full_hidden = fc.out_features
    if hidden_width > full_hidden:
        raise ValueError(
            f'hidden_width({hidden_width})は教師の中間幅({full_hidden})以下が必要'
        )
    if hidden_width == full_hidden:
        return np.arange(full_hidden, dtype=np.int64)

    if method == 'first':
        return np.arange(hidden_width, dtype=np.int64)

    # l2_norm: 各中間ユニットjの「入力側ノルム × 出力側ノルム」が大きい順
    # (そのユニットが最終出力に与える影響の粗い指標。厳密な重要度ではないが
    # ランダム選択より原理的な近似として採用)。
    fc_norm = fc.weight.detach().to(torch.float64).norm(dim=1)   # (full_hidden,)
    proj_norm = proj.weight.detach().to(torch.float64).norm(dim=0)  # (full_hidden,)
    score = fc_norm * proj_norm
    order = torch.argsort(score, descending=True)
    sel = order[:hidden_width].sort().values
    return sel.cpu().numpy().astype(np.int64)


def transcribe_ffn_block(
    mlp: nn.Module,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
    spec: FfnTranscribeSpec | None = None,
) -> tuple[IRGraph, dict]:
    """学習済みTransformer FFN(Linear→GELU→Linear)をLUT網IRへ転写する。

    モジュールdocstring参照。GELUは発火判定の閾値二値化として吸収する
    (transcribe.pyの隠れ層意味論 y=1[Wx+b>0] と同一。GELU(z)>0 ⟺ z>0
    なのでこの吸収は厳密、近似は生じない)。近似が生じるのはhidden_width<
    教師中間幅のユニット部分選択のみ(spec.hidden_width指定時)。

    Args:
        mlp: `.c_fc`/`.c_proj`を持つFFNモジュール、または2層Linearの
            Sequential等(extract_ffn_linears参照)
        calib_inputs: (B, in_width) の0/1較正データ。fan-in疎化のバイアス
            再較正、および近似誤差(agreement)測定に使う。Noneなら
            バイアス再較正なし・agreement測定不可
        spec: 転写仕様(省略時は既定FfnTranscribeSpec、n_inputs=2で
            教師の全中間幅・全出力幅をそのまま使う=ユニット部分選択なし)

    Returns:
        (ir, info):
            ir: 入力'x'(width=in_features)→lut_layer(hidden)→lut_layer(out)
                →出力'y'(width=out_features)のIRGraph(group_sumを含まない、
                FFN出力はclass scoreでなく論理ビットのため)
            info: {
                'selected_hidden_units': 選択された中間ユニットindex,
                'hidden_agreement': 較正データ上のhidden層一致率
                    (calib_inputsがNoneならNone)、
                'output_agreement': 較正データ上の出力層一致率(同上)、
                'full_hidden_width': 教師の元中間幅,
                'transcribed_hidden_width': 転写後hidden幅,
            }

    Raises:
        ValueError: FFN形状が不正、spec制約違反
    """
    spec = spec or FfnTranscribeSpec()
    fc, proj = extract_ffn_linears(mlp)
    in_features = fc.in_features
    full_hidden = fc.out_features
    out_features = proj.out_features

    hidden_width = spec.hidden_width or full_hidden
    out_width = spec.out_width or out_features
    if out_width != out_features:
        raise ValueError(
            f'out_width({out_width})はproj.out_features({out_features})と'
            '一致させる必要がある(出力層の部分選択は未対応)'
        )

    sel = _select_hidden_units(fc, proj, hidden_width, spec.unit_select)

    h = None
    if calib_inputs is not None:
        h = np.asarray(
            calib_inputs.detach().cpu().numpy()
            if isinstance(calib_inputs, torch.Tensor) else calib_inputs
        ).astype(np.int64)
        assert set(np.unique(h)).issubset({0, 1}), '較正データは0/1のみ'
        assert h.shape[1] == in_features, (
            f'較正データ幅{h.shape[1]} != in_features{in_features}'
        )

    # ---- hidden layer: fc[sel] を fan-in疎化 + 閾値二値化(GELU吸収) ----
    w_fc = fc.weight.detach().to(torch.float64)[sel]        # (hidden_width, in_features)
    b_fc = fc.bias.detach().to(torch.float64)[sel] if fc.bias is not None \
        else torch.zeros(hidden_width, dtype=torch.float64)

    indices_h, w_sel_h, bshift_h = sparsify_fanin(w_fc, spec.n_inputs, calib_inputs=h)
    b_eff_h = b_fc + bshift_h
    k_actual = indices_h.shape[0]
    if k_actual < spec.n_inputs:
        pad = spec.n_inputs - k_actual
        indices_h = np.vstack(
            [indices_h, np.zeros((pad, indices_h.shape[1]), dtype=np.int64)]
        )
        w_sel_h = torch.cat(
            [w_sel_h, torch.zeros(w_sel_h.shape[0], pad, dtype=w_sel_h.dtype)], dim=1,
        )
    tts_h = threshold_truth_tables(w_sel_h, b_eff_h)

    hidden_agreement = None
    h_bits = None
    if h is not None:
        h_bits = _simulate_lut_layer(h, indices_h, tts_h)
        # 参照(exact GELU-threshold) vs 転写(fan-in疎化後)の一致率
        ref_h = ((h.astype(np.float64) @ w_fc.numpy().T + b_fc.numpy()[None, :]) > 0
                 ).astype(np.int64)
        hidden_agreement = float((h_bits == ref_h).mean())

    # ---- output layer: proj(fc選択に対応する列のみ) を fan-in疎化 + 閾値二値化 ----
    # proj.weightの列を選択済みhidden unitに絞り、出力自体も論理ビット
    # (threshold_truth_tablesで二値化)として扱う。FFN出力はresidualへ加算
    # される実数なので、この二値化は「教師FFNの符号情報」だけを保持する
    # 近似である(honest: 振幅情報は失われる。info['output_agreement']で
    # 較正データ上の符号一致率を報告する)。
    w_proj = proj.weight.detach().to(torch.float64)[:, sel]  # (out_features, hidden_width)
    b_proj = proj.bias.detach().to(torch.float64) if proj.bias is not None \
        else torch.zeros(out_features, dtype=torch.float64)

    indices_o, w_sel_o, bshift_o = sparsify_fanin(
        w_proj, spec.n_inputs, calib_inputs=h_bits,
    )
    b_eff_o = b_proj + bshift_o
    k_actual_o = indices_o.shape[0]
    if k_actual_o < spec.n_inputs:
        pad = spec.n_inputs - k_actual_o
        indices_o = np.vstack(
            [indices_o, np.zeros((pad, indices_o.shape[1]), dtype=np.int64)]
        )
        w_sel_o = torch.cat(
            [w_sel_o, torch.zeros(w_sel_o.shape[0], pad, dtype=w_sel_o.dtype)], dim=1,
        )
    tts_o = threshold_truth_tables(w_sel_o, b_eff_o)

    output_agreement = None
    if h_bits is not None:
        out_bits = _simulate_lut_layer(h_bits, indices_o, tts_o)
        ref_o = ((h_bits.astype(np.float64) @ w_proj.numpy().T
                  + b_proj.numpy()[None, :]) > 0).astype(np.int64)
        output_agreement = float((out_bits == ref_o).mean())

    nodes = [
        IRNode(
            id='ffn_hidden_lut',
            type='lut_layer',
            params={
                'in_width': in_features,
                'out_width': hidden_width,
                'n_inputs': spec.n_inputs,
                'indices': indices_h,
                'truth_tables': tt_to_hex(tts_h),
            },
            train_info={'kind': 'transcribe_ffn_hidden'},
        ),
        IRNode(
            id='ffn_output_lut',
            type='lut_layer',
            params={
                'in_width': hidden_width,
                'out_width': out_width,
                'n_inputs': spec.n_inputs,
                'indices': indices_o,
                'truth_tables': tt_to_hex(tts_o),
            },
            train_info={'kind': 'transcribe_ffn_output'},
        ),
    ]
    edges = [
        IREdge(src='x', dst='ffn_hidden_lut'),
        IREdge(src='ffn_hidden_lut', dst='ffn_output_lut'),
        IREdge(src='ffn_output_lut', dst='y'),
    ]
    ir = IRGraph(
        metadata={
            'source': 'convert.transcribe_seq.transcribe_ffn_block',
            'spec': {'n_inputs': spec.n_inputs, 'unit_select': spec.unit_select},
        },
        inputs=[IRPort(id='x', width=in_features, encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=out_width, semantics='ffn_output_bits')],
        nodes=nodes,
        edges=edges,
    )
    info = {
        'selected_hidden_units': sel,
        'hidden_agreement': hidden_agreement,
        'output_agreement': output_agreement,
        'full_hidden_width': full_hidden,
        'transcribed_hidden_width': hidden_width,
    }
    return ir, info


def _simulate_lut_layer(x: np.ndarray, indices: np.ndarray, tts: list[int]) -> np.ndarray:
    """(B,in)の0/1配列にIR lut_layerを適用する(transcribe.py._simulate_hiddenと同じ規約)。"""
    n = indices.shape[0]
    gathered = x[:, indices]                              # (B, n, out)
    weights = (1 << np.arange(n, dtype=np.int64))[None, :, None]
    addr = (gathered * weights).sum(axis=1)               # (B, out)
    tt_arr = np.asarray(tts, dtype=np.uint64)
    out = (tt_arr[None, :] >> addr.astype(np.uint64)) & np.uint64(1)
    return out.astype(np.int64)


def init_logic_mlp_from_ffn(
    logic_mlp,
    mlp: nn.Module,
    calib_inputs: np.ndarray | torch.Tensor | None = None,
    spec: FfnTranscribeSpec | None = None,
) -> dict:
    """LogicMLP(layers/logic_linear.py)のLutLayerスタックを教師FFNの転写で初期化する。

    `logic_mlp.luts`が(実装docstring参照)`transcribe_ffn_block`が返す2つの
    lut_layerノードと同一shape(in/out幅・n_inputs一致)であることを要求する
    (通常はspec.width==in_features==out_featuresのLogicSpecで構築した
    LogicMLPを渡す。既定のLogicTransformerBlock用spec(width=d_model*
    mlp_ratio)とはshapeが異なるため、そのままでは使えない)。

    重み書き込みは`ir/weights.py`の`restore_lut_layer`をそのまま再利用する
    (recover.pyと同じ「真理値表→LutLayer.weights合成」機構)。
    `logic_mlp.luts[i].parameterization`が何であってもrestore_lut_layerが
    対応表現(softmax16g/independent/raw/warp等)へ変換するため、
    LogicMLPのデフォルト('softmax', n_inputs=2 → 内部softmax16gモード)を
    そのまま使える。

    Args:
        logic_mlp: 初期化対象のLogicMLPインスタンス(in-place変更)。
        mlp: 転写元の教師FFN(extract_ffn_linears対応形状)。
        calib_inputs: transcribe_ffn_blockへ渡す較正データ。
        spec: transcribe_ffn_blockへ渡す転写仕様。省略時、
            logic_mlp.luts[0].out_dim/logic_mlp.luts[1].out_dimから
            hidden_width/out_widthを自動導出する。

    Returns:
        transcribe_ffn_blockのinfo辞書(近似誤差の報告に使う)。

    Raises:
        ValueError: logic_mlp.lutsが2層でない、またはshape不一致。
    """
    luts = list(logic_mlp.luts)
    if len(luts) != 2:
        raise ValueError(
            f'init_logic_mlp_from_ffnはdepth=2のLogicMLPのみ対応: '
            f'depth={len(luts)}'
        )
    hidden_layer, output_layer = luts

    if spec is None:
        spec = FfnTranscribeSpec(
            n_inputs=hidden_layer.n_inputs,
            hidden_width=hidden_layer.out_dim,
            out_width=output_layer.out_dim,
        )

    ir, info = transcribe_ffn_block(mlp, calib_inputs, spec)
    hidden_node = ir.node_by_id('ffn_hidden_lut')
    output_node = ir.node_by_id('ffn_output_lut')

    if hidden_layer.in_dim != hidden_node.params['in_width'] or \
       hidden_layer.out_dim != hidden_node.params['out_width'] or \
       hidden_layer.n_inputs != hidden_node.params['n_inputs']:
        raise ValueError(
            'logic_mlp.luts[0]の形状が転写IRのhidden層と不一致: '
            f'layer(in={hidden_layer.in_dim}, out={hidden_layer.out_dim}, '
            f'n_inputs={hidden_layer.n_inputs}) vs '
            f'ir(in={hidden_node.params["in_width"]}, '
            f'out={hidden_node.params["out_width"]}, '
            f'n_inputs={hidden_node.params["n_inputs"]})'
        )
    if output_layer.in_dim != output_node.params['in_width'] or \
       output_layer.out_dim != output_node.params['out_width'] or \
       output_layer.n_inputs != output_node.params['n_inputs']:
        raise ValueError(
            'logic_mlp.luts[1]の形状が転写IRのoutput層と不一致: '
            f'layer(in={output_layer.in_dim}, out={output_layer.out_dim}, '
            f'n_inputs={output_layer.n_inputs}) vs '
            f'ir(in={output_node.params["in_width"]}, '
            f'out={output_node.params["out_width"]}, '
            f'n_inputs={output_node.params["n_inputs"]})'
        )

    restore_lut_layer(
        hidden_layer, hidden_node.params['indices'], hidden_node.params['truth_tables'],
    )
    restore_lut_layer(
        output_layer, output_node.params['indices'], output_node.params['truth_tables'],
    )

    return info
