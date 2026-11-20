"""学習済み(または合成)DLGN IR → webbench(ブラウザ実行比較ツール)用バンドル。

W4(Track W)でconv系(SegConvDLGN / CenterNet-LUT)をブラウザ実行するための
エクスポータ。dense版(UCI-HAR)を汎用化し、IR v0.2の
conv2d / or_pool2d / concat / upsample2d / permute(upsample・
SpatialGroupSum・thermo channel permuteはネイティブノード化済み。
全空間位置展開のlut_layerではなくJIT展開ノード)を含むIRを、
ブラウザがfetchする自己完結JSONへ変換する。

出力(モード別):
    --ir <path.ir.json>       学習済みIRをwebbench形式へ変換
        tools/webbench/public/models/<label>.json
            {ir, preproc, labels, resources, eval, readout}
    --fixtures                TS整数一致テスト用ゴールデン生成
        tools/webbench/src/fixtures/conv_golden.json
            {cases:[{name, ir, in_width, n, x_bits_b64[], scores[][], ...}]}
        tools/webbench/src/fixtures/thermo_golden.json
            画像thermometer前処理のビット順契約(Py↔TS)
        tools/webbench/src/fixtures/bundle_golden.json
            deploy/bundle.py programs節(conv2d命令列)のPy↔TS言語横断契約
            {programs_cases:[{name, ir, programs}], bundle_doc: build_bundle出力}
    --demo                    デモUI(seg/webcam)用のランダム小モデル
        tools/webbench/public/models/demo_seg.json

契約(TS側 convpreproc.ts / engines/*.ts と一致させる):
    - int16復元: float = int16 * scale
    - ビットのbase64パック: bit i → byte[i>>3] のビット (i&7)(LSB-first)
    - 画像thermometer: 出力ch = rgb*T + thr、NCHW flatten
        bit index = ((rgb*T+thr)*H + y)*W + x、判定は strict greater(z > thr)
    - IR実行: griffith_logic/ir/simulate.py が真実源(conv2d/or_pool2d/
      upsample2d/permuteは lower_to_primitives展開後に実行)。TS側は
      lowerConv.ts/packedConv.ts が同一展開・ネイティブ実行を行う
"""

import argparse
import base64
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from griffith_logic.deploy.bundle import (  # noqa: E402
    attach_bundle_sizes,
    build_bundle,
    save_bundle,
)
from griffith_logic.deploy.programs import build_programs  # noqa: E402
from griffith_logic.encoding.image_features import (  # noqa: E402
    CHROMA_PAIRS,
    CURV_OFFSETS,
    EDGE_OFFSETS,
    n_output_channels,
)
from griffith_logic.export.estimate import estimate_resources  # noqa: E402
from griffith_logic.ir.schema import (  # noqa: E402
    IREdge,
    IRGraph,
    IRNode,
    IRPort,
)
from griffith_logic.ir.serialize import (  # noqa: E402
    ir_from_dict,
    ir_to_dict,
    load_ir,
)
from griffith_logic.ir.simulate import simulate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WEBBENCH = REPO / 'tools' / 'webbench'


# ---------------------------------------------------------------------------
# 汎用ビット/量子化ヘルパ(dense版から流用、TS側と契約一致)
# ---------------------------------------------------------------------------
def quantize_i16(x: np.ndarray) -> tuple[np.ndarray, float]:
    """int16量子化(共通スケール)。(i16配列, scale)を返す"""
    scale = float(np.abs(x).max()) / 32767.0 if x.size else 1.0
    if scale == 0.0:
        scale = 1.0
    xi = np.clip(np.round(x / scale), -32768, 32767).astype('<i2')
    return xi, scale


def pack_bits_b64(bits_row: np.ndarray) -> str:
    """{0,1}ベクトルをLSB-firstでバイト詰めしbase64化"""
    packed = np.packbits(bits_row.astype(np.uint8), bitorder='little')
    return base64.b64encode(packed.tobytes()).decode('ascii')


# ---------------------------------------------------------------------------
# 画像thermometer前処理(conv入力の二値エンコード契約)
# ---------------------------------------------------------------------------
def image_thermometer_params(x_train: np.ndarray, n_thresholds: int) -> dict:
    """入力画像 [N,C,H,W] からチャネル別thermometer閾値を計算する。

    標準化(チャネル別mean/std)後、等分位点をn_thresholds個取る
    (dense版 compute_thermometer_params と同一方針)。
    """
    assert x_train.ndim == 4, 'x_train は [N,C,H,W]'
    c = x_train.shape[1]
    flat = x_train.transpose(1, 0, 2, 3).reshape(c, -1).astype(np.float64)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    std[std == 0] = 1.0
    z = (flat - mean[:, None]) / std[:, None]
    qs = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    thresholds = np.stack([np.quantile(z[ch], qs) for ch in range(c)])
    return {
        'mean': mean, 'std': std, 'thresholds': thresholds,
        'n_thresholds': int(n_thresholds),
    }


def encode_image_thermometer(x: np.ndarray, params: dict) -> np.ndarray:
    """[N,C,H,W] → [N, C*T*H*W] のthermometerビット(NCHW flatten)。

    出力チャネル = c*T + thr(rgb-major)、判定は z > threshold。
    ビット index = ((c*T+thr)*H + y)*W + x。
    """
    assert x.ndim == 4
    n, c, h, w = x.shape
    mean = np.asarray(params['mean'])[None, :, None, None]
    std = np.asarray(params['std'])[None, :, None, None]
    thr = np.asarray(params['thresholds'])          # [C, T]
    t = thr.shape[1]
    z = (x - mean) / std                            # [N,C,H,W]
    # [N, C, T, H, W] → strict greater
    bits = (z[:, :, None] > thr[None, :, :, None, None]).astype(np.uint8)
    return bits.reshape(n, c * t * h * w)


# ---------------------------------------------------------------------------
# dense(時系列)thermometer前処理 — dense系エンジン(packed32等)の契約。
# TS側 preproc.ts encodeThermometer と一致: bit = ((ch*L)+t)*T+thr、z > thr
# ---------------------------------------------------------------------------
def dense_thermometer_params(x_train: np.ndarray, n_thresholds: int) -> dict:
    """[N,C,L] からチャネル別標準化+等分位thermometer閾値。"""
    assert x_train.ndim == 3
    c = x_train.shape[1]
    flat = x_train.transpose(1, 0, 2).reshape(c, -1).astype(np.float64)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    std[std == 0] = 1.0
    z = (flat - mean[:, None]) / std[:, None]
    qs = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    thresholds = np.stack([np.quantile(z[ch], qs) for ch in range(c)])
    return {'mean': mean, 'std': std, 'thresholds': thresholds}


def encode_dense_thermometer(x: np.ndarray, params: dict) -> np.ndarray:
    """[N,C,L] → [N, C*L*T]。bit index = ((ch*L)+t)*T+thr(HAR契約と同一)。"""
    assert x.ndim == 3
    n = x.shape[0]
    mean = np.asarray(params['mean'])[None, :, None]
    std = np.asarray(params['std'])[None, :, None]
    thr = np.asarray(params['thresholds'])              # [C, T]
    z = (x - mean) / std                                # [N,C,L]
    bits = (z[:, :, :, None] > thr[None, :, None, :]).astype(np.uint8)
    return bits.reshape(n, -1)                          # (ch, t, thr) 順


# ---------------------------------------------------------------------------
# 合成IR(ゴールデン/デモ用)。実モデルは学習完了後に --ir で差し込む
# ---------------------------------------------------------------------------
def build_dense_ir(seed: int, in_width: int, widths: list[int], k: int,
                   name: str = 'dense') -> IRGraph:
    """lut_layer直列+group_sum のランダムdense IR(手組み)。"""
    rng = np.random.default_rng(seed)
    nodes, edges = [], [IREdge(src='x', dst='n1')]
    prev_w = in_width
    for li, w in enumerate(widths):
        idx = rng.integers(0, prev_w, size=(2, w)).astype(np.int64)
        tts = [f'0x{int(rng.integers(0, 16)):x}' for _ in range(w)]
        nodes.append(IRNode(id=f'n{li + 1}', type='lut_layer', params={
            'in_width': prev_w, 'out_width': w, 'n_inputs': 2,
            'indices': idx, 'truth_tables': tts,
        }))
        edges.append(IREdge(src=f'n{li + 1}', dst=f'n{li + 2}'))
        prev_w = w
    gs_id = f'n{len(widths) + 1}'
    nodes.append(IRNode(id=gs_id, type='group_sum',
                        params={'in_width': prev_w, 'k': k, 'tau': 1.0}))
    edges.append(IREdge(src=gs_id, dst='y'))
    return IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=nodes, edges=edges,
    )


def build_dense_golden(seed: int = 5, n: int = 70, c: int = 4, l: int = 8,
                       t: int = 3, k: int = 3) -> dict:
    """dense系エンジン(interp/packed32/packed64/codegen)のゴールデン。

    HAR実データの代わりに合成時系列+ランダムdense IRで契約を固定する
    (n=70で32/64ワード境界をまたぐ)。modelはModelDoc形で同梱。
    """
    in_width = c * l * t
    graph = build_dense_ir(seed, in_width, [96, 72], k, name='dense_golden')
    rng = np.random.default_rng(seed + 1)
    x_train = rng.normal(size=(64, c, l)).astype(np.float32)
    x = rng.normal(size=(n, c, l)).astype(np.float32)
    params = dense_thermometer_params(x_train, t)

    xi, scale = quantize_i16(x)
    x_rest = xi.astype(np.float32) * scale
    bits = encode_dense_thermometer(x_rest, params)
    assert bits.shape[1] == in_width
    scores = simulate(graph, bits)
    model = {
        'ir': ir_to_dict(graph),
        'preproc': {
            'kind': 'thermometer', 'input_kind': 'raw',
            'mean': params['mean'].tolist(), 'std': params['std'].tolist(),
            'thresholds': params['thresholds'].tolist(),
            'bit_order': 'ch,t,thr',
        },
        'labels': [f'class{i}' for i in range(k)],
        'readout': {'kind': 'class_scores', 'k': k, 'spatial': None},
        'resources': estimate_resources(graph),
        'eval': {},
    }
    return {
        'model': model,
        'n': n, 'shape': [n, c, l], 'scale': scale,
        'x_i16': xi.reshape(n, -1).tolist(),
        'bits_b64': [pack_bits_b64(bits[i]) for i in range(n)],
        'scores': scores.astype(np.int64).tolist(),
        'preds': scores.argmax(axis=1).astype(np.int64).tolist(),
    }


def build_seg_ir(seed: int, *, in_shape, oc1, oc2, k, depth,
                 pool_k=2, up=True, pad=1, stride=1, name='seg') -> IRGraph:
    """LutConv2d(+OrPooling/UpSample)+SpatialGroupSum の小セグIRを合成する。"""
    import torch
    import torch.nn as nn

    from griffith_logic.conv import LutConv2d
    from griffith_logic.export import export_ir
    from griffith_logic.pooling import OrPooling2d
    from griffith_logic.spatial_group_sum import SpatialGroupSum
    from griffith_logic.upsample import UpSample2d

    torch.manual_seed(seed)
    c = in_shape[0]
    layers = [
        LutConv2d(c, oc1, kernel_size=3, tree_depth=depth, padding=pad,
                  stride=stride, n_inputs=2, parameterization='softmax16g'),
        OrPooling2d(kernel_size=pool_k, stride=pool_k),
    ]
    if up:
        layers.append(UpSample2d(scale_factor=pool_k))
    layers += [
        LutConv2d(oc1, oc2, kernel_size=3, tree_depth=depth, padding=1,
                  n_inputs=2, parameterization='softmax16g'),
        SpatialGroupSum(n_outputs=k, scale=1.0),
    ]
    model = nn.Sequential(*layers).eval()
    return export_ir(model, input_shape=tuple(in_shape), name=name,
                     native_conv=True)


def build_concat_ir(seed: int, in_width=24, a=32, b=16, k=3,
                    name='concat') -> IRGraph:
    """2分岐→concat→group_sum の手組みDAG(concat被覆用)。"""
    rng = np.random.default_rng(seed)

    def lut(node_id, inw, outw):
        idx = rng.integers(0, inw, size=(2, outw)).astype(np.int64)
        tts = [f'0x{int(rng.integers(0, 16)):x}' for _ in range(outw)]
        return IRNode(id=node_id, type='lut_layer', params={
            'in_width': inw, 'out_width': outw, 'n_inputs': 2,
            'indices': idx, 'truth_tables': tts,
        })

    n1 = lut('n1', in_width, a)
    n2 = lut('n2', in_width, b)
    cat = IRNode(id='cat', type='concat',
                 params={'in_widths': [a, b]})
    total = a + b
    assert total % k == 0, f'concat幅{total}がk={k}で割り切れない'
    gs = IRNode(id='gs', type='group_sum',
                params={'in_width': total, 'k': k, 'tau': 1.0})
    return IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=[n1, n2, cat, gs],
        edges=[
            IREdge(src='x', dst='n1'), IREdge(src='x', dst='n2'),
            IREdge(src='n1', dst='cat', dst_slot=0),
            IREdge(src='n2', dst='cat', dst_slot=1),
            IREdge(src='cat', dst='gs'), IREdge(src='gs', dst='y'),
        ],
    )


def _readout_meta(graph: IRGraph) -> dict:
    """出力の空間形状(seg per-pixel readout用)を推定する。

    最終group_sumのkと、直前conv/spatialの出力shapeから (n_out,H,W) を復元。
    ここでは group_sum.k = n_out*H*W と仮定し、square推定はしない
    (正確な形状はメタで供給。合成IRはtrain_infoから拾う)。
    """
    last = graph.nodes[-1]
    meta = {'kind': 'class_scores', 'k': None, 'spatial': None}
    if last.type == 'group_sum':
        meta['k'] = int(last.params['k'])
        for n in graph.nodes:
            if n.train_info.get('lowered_from') == 'SpatialGroupSum':
                meta['kind'] = 'spatial_scores'
    return meta


# ---------------------------------------------------------------------------
# ゴールデン生成
# ---------------------------------------------------------------------------
def simulate_case(graph: IRGraph, x_bits: np.ndarray) -> dict:
    scores = simulate(graph, x_bits)
    return {
        'name': graph.metadata.get('name', 'case'),
        'ir': ir_to_dict(graph),
        'in_width': int(graph.inputs[0].width),
        'out_width': int(scores.shape[1]),
        'n': int(x_bits.shape[0]),
        'x_bits_b64': [pack_bits_b64(x_bits[i]) for i in range(len(x_bits))],
        'scores': scores.astype(np.int64).tolist(),
        'preds': scores.argmax(axis=1).astype(np.int64).tolist(),
    }


# conv2dを含む被覆用seg IR specs(パディング/ストライド/深さ/プール違い)。
# build_fixture_cases(conv_golden)と build_bundle_programs_cases(bundle_golden、
#)の両方が共有する: 後者は「conv_goldenでconv2dノードを持つケース」を
# そのまま再利用したいので、specsを一箇所に集約している。
_SEG_SPECS = [
    # (name, kwargs)
    dict(name='seg_d2_pad1', in_shape=(3, 8, 8), oc1=4, oc2=6, k=2,
         depth=2, pool_k=2, up=True, pad=1, stride=1),
    dict(name='seg_d3_pool', in_shape=(2, 8, 8), oc1=3, oc2=4, k=2,
         depth=3, pool_k=2, up=True, pad=1, stride=1),
    dict(name='seg_pad0', in_shape=(3, 10, 10), oc1=4, oc2=4, k=4,
         depth=2, pool_k=2, up=True, pad=0, stride=1),
    dict(name='seg_stride2', in_shape=(2, 12, 12), oc1=4, oc2=6, k=3,
         depth=2, pool_k=2, up=True, pad=1, stride=2),
    dict(name='seg_nopool', in_shape=(3, 6, 6), oc1=5, oc2=6, k=3,
         depth=2, pool_k=1, up=False, pad=1, stride=1),
]


def build_fixture_cases(n_samples: int = 6) -> list[dict]:
    """パディング/ストライド/深さ/プールを変えた被覆ケース群。"""
    cases = []
    for i, s in enumerate(_SEG_SPECS):
        g = build_seg_ir(seed=100 + i, **s)
        rng = np.random.default_rng(1000 + i)
        x = (rng.random((n_samples, g.inputs[0].width)) > 0.5).astype(np.uint8)
        cases.append(simulate_case(g, x))
    # concat被覆
    for i, (a, b, kk) in enumerate([(32, 16, 3), (48, 24, 4), (20, 10, 2)]):
        g = build_concat_ir(seed=200 + i, in_width=24 + i * 4, a=a, b=b, k=kk,
                            name=f'concat_{i}')
        rng = np.random.default_rng(2000 + i)
        x = (rng.random((n_samples, g.inputs[0].width)) > 0.5).astype(np.uint8)
        cases.append(simulate_case(g, x))
    return cases


# ---------------------------------------------------------------------------
# 配布バンドル(griffith_logic/deploy/programs.py・bundle.py)のTS消費用
# ゴールデン。tools/webbench/src/fixtures/bundle_golden.json
# programs節(conv2d命令列)をPy⇔TSで整数厳密一致させるための言語横断契約。
# ---------------------------------------------------------------------------
def _randomize_conv_truth_tables(graph: IRGraph, seed: int) -> IRGraph:
    """conv2dノードの真理値表を一様ランダムな2入力関数に差し替える(その場変更)。

    未学習のLutConv2dは残差初期化のパススルーが支配的で、命令列化すると
    ゲートが全て畳み込まれて消える(n_instrs=0)。これはtests/deploy/
    conftest.py::randomize_conv_truth_tables と同じ観察・同じ対処
    (決定的seedでのランダム化)で、programs_cases に命令列の実行経路
    (non-empty instrs)を必ず1件は踏ませるために使う。
    """
    rng = np.random.default_rng(seed)
    for node in graph.nodes:
        if node.type != 'conv2d':
            continue
        n = len(node.params['truth_tables'])
        node.params['truth_tables'] = [
            f'0x{int(v):x}' for v in rng.integers(0, 16, size=n)
        ]
    return graph


def build_bundle_programs_cases() -> list[dict]:
    """conv2dを含む_SEG_SPECS(conv_goldenと共有)ごとのbuild_programsゴールデン。

    concat系(build_concat_ir)はconv2dノードを持たないため対象外。真理値表は
    ランダム化済み(_randomize_conv_truth_tables)なので、少なくとも一部の
    ケースでn_instrs>0の命令列が焼かれる(空振り防止、下でassertする)。
    """
    cases = []
    for i, s in enumerate(_SEG_SPECS):
        g = build_seg_ir(seed=100 + i, **s)
        _randomize_conv_truth_tables(g, seed=5000 + i)
        ir_dict = ir_to_dict(g)
        # TS側が消費する経路(irをdictへ落としてからprogramsを引く)と同じ形で
        # 生成する: 一度dictへ落としir_from_dictで往復させ、ir↔programsの
        # 整合を生成時点でも検証する(verify=Trueで縮約前後の全数照合)。
        programs = build_programs(ir_from_dict(ir_dict), verify=True)
        cases.append({'name': s['name'], 'ir': ir_dict, 'programs': programs})
    n_nonempty = sum(1 for c in cases for p in c['programs'].values()
                     if p['n_instrs'] > 0)
    assert n_nonempty > 0, (
        'bundle_golden: 全ケースの命令列がn_instrs=0(パススルー畳み込みで'
        '全消滅)。TS側のprograms消費テストが空振りになるため、'
        '_randomize_conv_truth_tables のseedを見直すこと。'
    )
    return cases


def build_bundle_doc_golden(seed: int = 6000) -> dict:
    """build_bundle(verify=True)の出力そのもののTS消費ゴールデン(画像seg系1本)。

    tests/deploy/conftest.py::_seg_doc_cached と同じ方針(小さいseg IR +
    真理値表ランダム化 + image_thermometer preproc)。demo_samples には
    実データセット(Oxford-IIIT Pet)の代わりに合成RGB1枚と、その画像に対する
    モデル自身の出力(argmaxマスク、simulateで計算)を添える。これは精度の
    正解ラベルではなく、画像モードのend-to-end検証(preproc→bundle→argmax)を
    TS側で書けるようにするための添え物。
    """
    h = w = 8
    n_thr = 1
    graph = build_seg_ir(seed=42, in_shape=(3, h, w), oc1=4, oc2=4, k=2,
                         depth=2, pool_k=2, up=True, pad=1,
                         name='bundle_golden_seg')
    _randomize_conv_truth_tables(graph, seed=seed)
    preproc = {
        'kind': 'image_thermometer', 'rgb_channels': 3, 'n_thresholds': n_thr,
        'height': h, 'width': w,
        'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0],
        'thresholds': [[0.5]] * 3,
        'bit_order': 'rgb,thr,y,x',
    }
    doc = build_model_doc(graph, preproc=preproc, labels=['a', 'b'])
    doc['readout']['spatial'] = {'n_out': 2, 'height': h, 'width': w}

    rng = np.random.default_rng(seed + 1)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)     # HWC
    x = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0   # [1,3,H,W]
    params = {
        'mean': np.asarray(preproc['mean']), 'std': np.asarray(preproc['std']),
        'thresholds': np.asarray(preproc['thresholds']),
    }
    bits = encode_image_thermometer(x, params)
    scores = simulate(graph, bits)
    mask = scores.reshape(1, 2, h * w).argmax(axis=1)[0].astype(np.uint8)
    doc['demo_samples'] = [{
        'height': h, 'width': w,
        'rgb_b64': base64.b64encode(rgb.reshape(-1).tobytes()).decode('ascii'),
        'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
    }]
    return build_bundle(doc, verify=True)


def build_bundle_golden() -> dict:
    """deploy/bundle.py の programs 節をTS側で消費するためのゴールデン。

    tools/webbench/src/fixtures/bundle_golden.json の中身:
        programs_cases  合成conv IR(_SEG_SPECS)ごとの{name, ir, programs}
        bundle_doc      build_bundle 出力そのもの(画像seg系1本)
    """
    return {
        'programs_cases': build_bundle_programs_cases(),
        'bundle_doc': build_bundle_doc_golden(),
    }


# ---------------------------------------------------------------------------
# CenterNet 検出デコード(experiments/detection_utils.py と TS detdecode.ts の
# 一致ゴールデン)。多ヘッド concat の分割メタ(DetReadout)も共有する。
# ---------------------------------------------------------------------------
def build_det_readout(graph: IRGraph, scales: dict, *, output_stride: int = 4,
                      top_k: int = 100, threshold: float = 0.1) -> dict:
    """multi_head IR の output_heads → TS detdecode の DetReadout を組む。

    各ヘッドの offset は連結スコア列内の開始位置、scale は SpatialGroupSum の
    scale(生 popcount → 実数値)。scales は {'hm','size','off': float}。
    """
    heads = graph.metadata['output_heads']
    ro_heads, offset = {}, 0
    for h in heads:
        name, shape = h['name'], h['shape']  # shape=[n_out,H,W]
        ro_heads[name] = {
            'offset': int(offset), 'n_out': int(shape[0]),
            'h': int(shape[1]), 'w': int(shape[2]),
            'scale': float(scales[name]),
        }
        offset += int(h['width'])
    return {
        'kind': 'detection', 'heads': ro_heads,
        'output_stride': int(output_stride), 'top_k': int(top_k),
        'threshold': float(threshold), 'n_classes': int(ro_heads['hm']['n_out']),
    }


def _decode_boxes_from_scores(scores_row: np.ndarray, ro: dict) -> list[dict]:
    """連結スコア列(生 popcount or 実数)→ ピクセル座標の箱列。

    detection_utils.decode_detections をそのまま呼び、TS 側 decodeDetections と
    同じ float64・-inf パディング maxpool・1e-6 タイブレークで箱を作る。
    """
    import torch
    sys.path.insert(0, str(REPO / 'experiments'))
    from detection_utils import decode_detections

    def plane(name: str) -> torch.Tensor:
        h = ro['heads'][name]
        n = h['n_out'] * h['h'] * h['w']
        seg = np.asarray(scores_row[h['offset']:h['offset'] + n], dtype=np.float64)
        seg = seg.reshape(h['n_out'], h['h'], h['w']) * h['scale']
        return torch.from_numpy(seg).to(torch.float64).unsqueeze(0)

    res = decode_detections(
        plane('hm'), plane('size'), plane('off'),
        output_stride=ro['output_stride'], top_k=ro['top_k'],
        threshold=ro['threshold'])[0]
    boxes = res['boxes'].tolist()
    scores = res['scores'].tolist()
    labels = res['labels'].tolist()
    return [
        {'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3],
         'cls': int(c), 'score': float(s)}
        for b, s, c in zip(boxes, scores, labels, strict=True)
    ]


def _tiny_centernet(seed: int, n_classes=3, k=2, canvas=16, groups=4):
    """小ランダム CenterNetLUT(ゴールデン用)。scale=1/groups で hm∈[0,1]。"""
    import torch
    sys.path.insert(0, str(REPO / 'experiments'))
    from detection_model import CenterNetLUT
    torch.manual_seed(seed)
    m = CenterNetLUT(
        n_classes=n_classes, k=k, in_channels=3, tree_depth=2, head_depth=2,
        n_inputs=2, hm_group=groups, size_group=groups, offset_group=groups,
        input_channels_per_tree=2, parameterization='softmax16g', n_layers=3,
    ).eval()
    return m, canvas


def build_det_decode_golden(seed: int = 11) -> dict:
    """純デコード一致(モデル非依存): ランダム float hm/size/off → 箱列。

    scale=1 の DetReadout に float スコアを直接与え、decode ロジックのみを
    Py↔TS で厳密照合する(topK・NMS・size/off 復元・box 変換)。
    """
    rng = np.random.default_rng(seed)
    cases = []
    for i, (C, H, W) in enumerate([(3, 8, 8), (5, 12, 12), (2, 10, 10)]):
        hw = H * W
        hm = rng.random(C * hw)
        size = rng.random(2 * hw)
        off = rng.random(2 * hw)
        scores = np.concatenate([hm, size, off])
        ro = {
            'kind': 'detection',
            'heads': {
                'hm': {'offset': 0, 'n_out': C, 'h': H, 'w': W, 'scale': 1.0},
                'size': {'offset': C * hw, 'n_out': 2, 'h': H, 'w': W,
                         'scale': 1.0},
                'off': {'offset': (C + 2) * hw, 'n_out': 2, 'h': H, 'w': W,
                        'scale': 1.0},
            },
            'output_stride': 4, 'top_k': 100, 'threshold': 0.1,
            'n_classes': C,
        }
        cases.append({
            'name': f'decode_{i}_C{C}_{H}x{W}',
            'scores': scores.tolist(),
            'readout': ro,
            'boxes': _decode_boxes_from_scores(scores, ro),
        })
    return {'cases': cases}


def build_det_e2e_golden(seed: int = 21) -> dict:
    """端到端一致: 小 CenterNetLUT → IR → 二値入力 → simulate(生 popcount)→
    多ヘッド分割 → decode。TS は ConvRuntime→detdecode で同一箱列を再現する。"""
    from griffith_logic.export import export_dag_ir
    cases = []
    for i, (nc, k, canvas, groups) in enumerate(
            [(3, 2, 16, 4), (2, 2, 24, 4)]):
        m, s = _tiny_centernet(seed + i, n_classes=nc, k=k, canvas=canvas,
                               groups=groups)
        graph = export_dag_ir(m, (3, s, s), name=f'det_e2e_{i}')
        ro = build_det_readout(
            graph,
            {'hm': m.hm_sum.scale, 'size': m.size_sum.scale,
             'off': m.off_sum.scale},
            output_stride=4, top_k=100, threshold=0.1)
        rng = np.random.default_rng(1000 + i)
        in_width = graph.inputs[0].width
        n = 3
        x = (rng.random((n, in_width)) > 0.5).astype(np.uint8)
        scores = simulate(graph, x)  # (n, total) 生 popcount
        case_frames = []
        for f in range(n):
            case_frames.append({
                'in_bits_b64': pack_bits_b64(x[f]),
                'boxes': _decode_boxes_from_scores(scores[f], ro),
            })
        cases.append({
            'name': f'det_e2e_{i}_nc{nc}_{s}x{s}',
            'ir': ir_to_dict(graph),
            'in_width': int(in_width),
            'readout': ro,
            'frames': case_frames,
        })
    return {'cases': cases}


def build_thermo_golden(seed: int = 7, n: int = 4, c: int = 3,
                        h: int = 5, w: int = 5, t: int = 3) -> dict:
    """画像thermometer前処理のビット順契約ゴールデン。"""
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(32, c, h, w)).astype(np.float32)
    x = rng.normal(size=(n, c, h, w)).astype(np.float32)
    params = image_thermometer_params(x_train, t)
    xi, scale = quantize_i16(x)
    x_rest = xi.astype(np.float32) * scale
    bits = encode_image_thermometer(x_rest, params)
    return {
        'seed': seed, 'shape': [n, c, h, w], 't': t, 'scale': scale,
        'mean': params['mean'].tolist(), 'std': params['std'].tolist(),
        'thresholds': params['thresholds'].tolist(),
        'x_i16': xi.reshape(n, -1).tolist(),
        'out_width': int(bits.shape[1]),
        'bits_b64': [pack_bits_b64(bits[i]) for i in range(n)],
    }


def build_lowlevel_golden(seed: int = 11, n: int = 4, c: int = 3,
                          h: int = 6, w: int = 7, t: int = 3) -> dict:
    """低レベル特徴前処理(edge/curv/chroma)のチャネル順契約ゴールデン。

    真実源は griffith_logic/encoding/image_features.py。TS の
    encodeImageLowLevel がこれとビット単位で一致することを convpreproc.test.ts で
    検証する。**export と TS のどちらかだけ直すと静かに壊れたモデルを配信するため、
    このゴールデンが唯一の歯止めになる。**
    """
    import torch

    from griffith_logic.encoding.image_features import encode_image_features

    rng = np.random.default_rng(seed)
    x = rng.random(size=(n, c, h, w)).astype(np.float32)  # [0,1) 画素
    edge_thr = (0.1, 0.3)
    curv_thr = (0.2,)
    chroma_thr = (0.15, 0.4)
    bits = encode_image_features(
        torch.from_numpy(x), n_thresholds=t,
        edge_thresholds=edge_thr, curv_thresholds=curv_thr,
        chroma_thresholds=chroma_thr,
    ).numpy().astype(np.uint8)
    flat = bits.reshape(n, -1)
    return {
        'seed': seed, 'shape': [n, c, h, w], 'n_thresholds': t,
        'edge_thresholds': list(edge_thr),
        'curv_thresholds': list(curv_thr),
        'chroma_thresholds': list(chroma_thr),
        'edge_offsets': [list(o) for o in EDGE_OFFSETS],
        'curv_offsets': [list(o) for o in CURV_OFFSETS],
        'chroma_pairs': [list(p) for p in CHROMA_PAIRS],
        'mean': [0.0] * c, 'std': [1.0] * c,
        'thresholds': [[(i + 1) / (t + 1) for i in range(t)] for _ in range(c)],
        'x': x.reshape(n, -1).tolist(),
        'out_channels': int(bits.shape[1]),
        'out_width': int(flat.shape[1]),
        'bits_b64': [pack_bits_b64(flat[i]) for i in range(n)],
    }


# ---------------------------------------------------------------------------
# about ブロック: ページの説明文を機械可読にする
# ---------------------------------------------------------------------------
#
# 動機: 各デモページの散文にデータセット名・学習レシピ・teacher情報が手書きされて
# いて、モデルを差し替えると説明文だけが古くなった。今後のexportではここに書き出し、
# TS側(modelMeta.ts / references.ts)が about を見つけたらそれを使う。
# **既存のモデルJSONは再生成しない**ので、TS側は about が無い場合に既存の静的散文へ
# フォールバックする(tools/webbench/src/modelMeta.ts::applyAbout)。

_DATASET_LABELS = {
    'multimnist': 'モノクロMulti-MNIST',
    'multimnist_color': 'カラーMulti-MNIST(CIFAR-10自然背景+色付き数字)',
    'fewclass': 'Pascal VOC 少クラス(person/car/dog)',
    'face': 'WIDER FACE',
    'voc': 'Pascal VOC',
    'pet': 'Oxford-IIIT Pet',
}

_TASK_LABELS = {
    'detection': '物体検出(CenterNet)',
    'segmentation': 'セマンティックセグメンテーション',
}

# about.recipe に載せる args のキー(存在するものだけ載せる)。
# 「モデルの性能を決めるレバー」だけに絞る(全argsを載せると読めない)。
_RECIPE_KEYS = (
    'k', 'wiring', 'selection', 'tau', 'hard_st', 'input_channels_per_tree',
    'n_thresholds', 'tree_depth', 'head_depth', 'n_layers', 'n_inputs',
    'parameterization', 'downsample', 'neck', 'readout_group',
    'hm_group', 'size_group', 'offset_group', 'decoupled_head', 'reg_bins',
    'edge_thresholds', 'curv_thresholds', 'chroma_thresholds',
    'ema_decay', 'grad_clip', 'epochs', 'batch_size', 'lr',
)

# teacher / KD 関連の args キー。
_TEACHER_KEYS = (
    'kd', 'kd_weight', 'kd_temperature', 'teacher_ckpt', 'teacher',
    'pseudo_labels', 'pseudo_label_ckpt',
)


def build_about(*, task: str, args_dict: dict, source_ckpt: str | None = None,
                notes: list[str] | None = None,
                wandb_run_id: str | None = None) -> dict:
    """ckpt の metadata['args'] から機械可読な背景説明(about)を作る。

    値が無いキーは**書かない**(TS側は欠損を「—」で描画するため、嘘の既定値を
    入れないことが重要)。
    """
    a = args_dict or {}
    dataset = a.get('dataset')
    about: dict = {'task': task}
    if task in _TASK_LABELS:
        about['task_label'] = _TASK_LABELS[task]
    if dataset:
        about['dataset'] = dataset
        if dataset in _DATASET_LABELS:
            about['dataset_label'] = _DATASET_LABELS[dataset]
    recipe = {k: a[k] for k in _RECIPE_KEYS
              if a.get(k) is not None and a.get(k) != ''}
    if recipe:
        about['recipe'] = recipe
    teacher = {k: a[k] for k in _TEACHER_KEYS
               if a.get(k) is not None and a.get(k) != ''}
    if teacher:
        about['teacher'] = teacher
    run_id = wandb_run_id or a.get('wandb_run_id')
    if run_id:
        about['wandb_run_id'] = str(run_id)
    if source_ckpt:
        about['source_ckpt'] = str(source_ckpt)
    if notes:
        about['notes'] = list(notes)
    about['exported_at'] = datetime.now().astimezone().isoformat()
    return about


# ---------------------------------------------------------------------------
# モデルドキュメント(webbench public/models/*.json)
# ---------------------------------------------------------------------------
def build_model_doc(graph: IRGraph, *, preproc: dict, labels: list[str],
                    eval_info: dict | None = None,
                    readout: dict | None = None,
                    about: dict | None = None) -> dict:
    ir_dict = ir_to_dict(graph)
    resources = estimate_resources(graph)
    doc = {
        'ir': ir_dict,
        'preproc': preproc,
        'labels': labels,
        'readout': readout if readout is not None else _readout_meta(graph),
        'resources': resources,
        'eval': eval_info or {},
    }
    # about は json_bytes/gzip_bytes の計測**前**に入れる
    # (配信サイズ表示が実ファイルとずれないようにするため)。
    if about is not None:
        doc['about'] = about
    model_json = json.dumps(doc, separators=(',', ':'))
    resources['json_bytes'] = len(model_json.encode())
    resources['gzip_bytes'] = len(gzip.compress(model_json.encode(), 6))
    return doc


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':')))
    print(f'  wrote {path}  ({path.stat().st_size:,}B)')


def _write_model_bundle(path: Path, doc: dict) -> None:
    """モデルdocを配布バンドルへ昇格してから public/models へ書き出す。

    以前は素のmodel doc(bundle_format/programs/preproc_descを持たない)を
    直接書いていたが、`tools/webbench/public/models/*.json` の配信物は
    全て `griffith deploy bundle` 相当のバンドルへ移行済み。エクスポータも
    以後は最初からバンドルとして書き出し、配信物と生成物の形式を揃える
    (ファイル名は変えない = strip_restricted.mjs のアローリストは影響を受けない)。

    build_model_doc が測った resources.json_bytes/gzip_bytes(バンドル前の
    素docのサイズ、demo_samples を含まない)の意味論はそのまま変えない。
    ここでは追加で resources.bundle_json_bytes/bundle_gzip_bytes を
    「実際に書き出す最終バンドル(demo_samples込み)」のサイズとして
    attach_bundle_sizes(`griffith deploy bundle` CLIと共通のヘルパ)で測る。
    dense(conv2dを持たない)IRでも build_programs は空dictを返すだけで
    build_bundle は失敗しない。

    書き出しも save_bundle(bundle.py)を再利用する。CLI経由(`griffith deploy
    bundle`)で書き出したバンドルと直列化(separators/ensure_ascii)を
    完全に揃えることで、resources.bundle_json_bytes が実ファイルの
    バイト数とここでも一致するようにする。

    resources は build_bundle 内で doc とバンドルが共有する同一dictなので、
    ここでの追記は呼び出し元の `doc['resources']` からも見える。
    """
    bundle = build_bundle(doc, verify=True)
    attach_bundle_sizes(bundle)
    save_bundle(bundle, path)
    print(f'  wrote {path}  ({path.stat().st_size:,}B)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_fixtures(args):
    cases = build_fixture_cases(n_samples=args.n)
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'conv_golden.json',
                {'cases': cases})
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'thermo_golden.json',
                build_thermo_golden())
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'lowlevel_golden.json',
                build_lowlevel_golden())
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'dense_golden.json',
                build_dense_golden())
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'det_decode_golden.json',
                build_det_decode_golden())
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'det_e2e_golden.json',
                build_det_e2e_golden())
    bundle_golden = build_bundle_golden()
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'bundle_golden.json',
                bundle_golden)
    print(f'fixtures: conv {len(cases)}ケース'
          f'(合計{sum(c["n"] for c in cases)}サンプル) + dense + thermo + det'
          f' + bundle({len(bundle_golden["programs_cases"])}ケース)')


def cmd_demo(args):
    """デモUI用のランダム小セグモデル(9ch thermometer入力 16x16)。"""
    in_shape = (args.channels, args.size, args.size)  # 二値ch数=C*T相当
    g = build_seg_ir(seed=args.seed, in_shape=in_shape, oc1=8, oc2=8,
                     k=args.classes, depth=2, pool_k=2, up=True, pad=1,
                     name='demo_seg')
    # デモは3ch RGB → T閾値 → C=3*T 二値ch。in_shape[0]=C にT倍を反映
    t = max(1, in_shape[0] // 3)
    preproc = {
        'kind': 'image_thermometer', 'rgb_channels': 3, 'n_thresholds': t,
        'height': args.size, 'width': args.size,
        # デモ用の簡易閾値(標準化なし、[0,1]画素を等間隔しきい)
        'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0],
        'thresholds': [list(np.linspace(0.25, 0.75, t)) for _ in range(3)],
        'bit_order': 'rgb,thr,y,x',
    }
    labels = [f'class{i}' for i in range(args.classes)]
    doc = build_model_doc(g, preproc=preproc, labels=labels)
    # seg per-pixel readoutの空間形状(TS側 argmaxPerPixel が使う)
    doc['readout']['spatial'] = {
        'n_out': args.classes, 'height': args.size, 'width': args.size,
    }
    _write_model_bundle(WEBBENCH / 'public' / 'models' / 'demo_seg.json', doc)
    write_manifest()
    print(f'demo seg model: in={in_shape} out_k={g.outputs[0].width} '
          f'gates={estimate_resources(g)["total_luts"]:,}')


def write_manifest() -> None:
    """モデルマニフェスト。preproc.kindでdense(dlgn)/conv(seg)/det(検出)を分類。

    検出モデル(readout.kind=='detection')は seg デモが誤ロードしないよう
    'det' へ分離する(det デモが 'det' を、seg デモが 'conv' を消費)。
    """
    models_dir = WEBBENCH / 'public' / 'models'
    dense, conv, det = [], [], []
    for p in sorted(models_dir.glob('*.json')):
        if p.name == 'index.json':
            continue
        doc = json.loads(p.read_text())
        kind = doc.get('preproc', {}).get('kind')
        # 画像系の preproc kind は複数ある(image_thermometer /
        # image_lowlevel)。ここを 'image_thermometer' 決め打ちにしていたため、
        # 低レベル特徴つきモデルが dense(dlgn)バケットへ誤分類され、
        # webcam の resolveModelFile が det から顔モデルを見つけられなくなった。
        # 新しい画像 preproc kind を足すときは必ずここにも追加すること。
        if kind in _IMAGE_PREPROC_KINDS:
            if (doc.get('readout') or {}).get('kind') == 'detection':
                det.append(p.name)
            else:
                conv.append(p.name)
        else:
            dense.append(p.name)
    manifest = {
        'dlgn': dense,
        'conv': conv,
        'det': det,
        'onnx': sorted(p.name for p in models_dir.glob('*.onnx')),
    }
    _write_json(models_dir / 'index.json', manifest)


# ---------------------------------------------------------------------------
# 学習済み検出モデル(CenterNetLUT)→ webbench
# ---------------------------------------------------------------------------
_RGB_DATASETS = ('multimnist_color', 'fewclass', 'face')

# 画像入力の preproc kind 一覧。write_manifest の分類と TS 側の
# encodeImageAuto のディスパッチが、この集合と一致している必要がある。
_IMAGE_PREPROC_KINDS = ('image_thermometer', 'image_lowlevel')


def _parse_thr_list(spec) -> tuple[float, ...]:
    """ckpt metadata の閾値指定を数値タプルへ。

    `--edge-thresholds '0.15,0.35'` は argparse の生文字列として metadata に
    入るため文字列を主に扱うが、将来 list で保存されても壊れないようにしておく。
    experiments/train_detection.py::_parse_thresholds と同一の解釈。
    """
    if not spec:
        return ()
    if isinstance(spec, list | tuple):
        return tuple(float(v) for v in spec)
    return tuple(float(v) for v in str(spec).split(',') if v.strip())


def build_centernetlut_from_ckpt(ckpt_path: str):
    """CenterNetLUT best.pt → (model, args_dict, best_metric)。

    dataset=='multimnist_color'はRGB 3ch入力(3*n_thresholds、
    thr-major)。dataset=='fewclass'(少クラス実物体検出、person/car/dog)も
    同型のRGB thermometer入力(ThresholdBinarize、thr-major)。
    dataset=='face'(WIDER FACE顔検出、1クラス)も同型のRGB thermometer入力。
    それ以外(グレースケールMultiMNIST/VOC)は従来通り1ch。

    head_type='float'(バックボーンLutConv2d+検出ヘッドfloat nn.Conv2d
    のハイブリッド)のckptはLUTのみのIR化ができない。黙って壊れたIRを吐かず
    ここで明示的にエラーにする(「学習配線exportが壊れてWebデモが動いて
    いなかった」事故の再発防止方針)。
    """
    import torch
    sys.path.insert(0, str(REPO / 'experiments'))
    from detection_model import CenterNetLUT
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = (data.get('metadata', {}) or {}).get('args', {}) or {}
    head_type = a.get('head_type', 'lut')
    if head_type != 'lut':
        raise SystemExit(
            f"\n[export非対応] {ckpt_path}: head_type={head_type!r} のckptは"
            'IRエクスポートできません。検出ヘッドがnn.Conv2d(float)のため'
            'LutConv2dのみを前提とするIR抽出(export_dag_ir)が対応していません。'
            "head-type=lut で学習したckptを使用してください。\n"
        )
    is_color = a.get('dataset') in _RGB_DATASETS
    n_rgb = 3 if is_color else 1
    # 低レベル特徴検出器(edge/curv/chroma)付きで訓練された場合、入力チャネル数は
    # n_rgb*n_thresholds では足りない。ここを間違えると state_dict のロードが
    # size mismatch で落ちる(=気付けるが、in_channels の唯一の真実源を
    # encoding.image_features に寄せて取り違いを構造的に防ぐ)。
    in_channels = n_output_channels(
        n_rgb, n_thresholds=a['n_thresholds'],
        edge_thresholds=_parse_thr_list(a.get('edge_thresholds')),
        curv_thresholds=_parse_thr_list(a.get('curv_thresholds')),
        chroma_thresholds=_parse_thr_list(a.get('chroma_thresholds')),
    )
    if a.get('dataset') == 'fewclass':
        # fewclass は n_classes を args に持たない(build_datasets内で
        # fewclass_classes から動的決定・保存)ため、クラス名列から復元する
        n_classes = len([c for c in a.get('fewclass_classes', '').split(',') if c.strip()])
    elif a.get('dataset') == 'face':
        # face(WIDER FACE)は build_datasets内で n_classes=1 固定。
        # a.get('n_classes', 10) には絶対に頼らない: --n-classes という
        # argparse引数はそもそも存在しないため、このフォールバックは常に
        # 10へ落ちる(黙って3ch→9chのin_channelsとload_state_dictが不一致し
        # 派手に落ちるので気付きやすいが、n_classesだけがズレるケースは
        # 気付きにくい罠のため専用分岐で確実に1にする)。
        n_classes = 1
    else:
        n_classes = a.get('n_classes', 10)
    model = CenterNetLUT(
        n_classes=n_classes, k=a['k'], in_channels=in_channels,
        tree_depth=a['tree_depth'], head_depth=a['head_depth'],
        n_inputs=a['n_inputs'], hm_group=a['hm_group'],
        size_group=a['size_group'], offset_group=a['offset_group'],
        input_channels_per_tree=a['input_channels_per_tree'],
        parameterization=a['parameterization'], n_layers=a['n_layers'],
        wiring=a.get('wiring', 'fixed'),
        downsample=a.get('downsample', 'orpool'),
        neck=a.get('neck', 'none'),
        decoupled_head=a.get('decoupled_head', False),
        reg_bins=a.get('reg_bins', 0),
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model, a, data.get('best_metric')


def _build_det_demo_samples(canvas: int, n_thr: int, count: int,
                            seed: int = 7, color: bool = False) -> list[dict]:
    """実 Multi-MNIST から生canvas + GT を取り出しdemo用に埋める。

    n_thresholds=0 で二値化前の生 canvas を得る(demo は canvas を描いて
    ブラウザ側で同一 thermometer 前処理を再現し推論する)。
    color=False: gray はuint8 96x96(1ch)。
    color=True: CIFAR-10背景+色付き数字のRGB canvas(uint8 96x96x3、
    rgb_b64はHWC順でpack)。
    """
    sys.path.insert(0, str(REPO / 'experiments'))
    from data.detection import MultiMNISTDetection
    ds = MultiMNISTDetection(
        mnist_root=str(REPO / 'data-mnist'), train=False, canvas_size=canvas,
        n_thresholds=0, num_samples=max(count, 1), seed=seed,
        color=color, cifar_root=str(REPO / 'data-cifar10'))
    samples = []
    for i in range(count):
        img, tgt = ds[i]  # color: (3,s,s) RGB [0,1] / gray: (1,s,s) [0,1]
        mask = tgt['mask'].bool()
        gt_boxes = tgt['gt_boxes'][mask].tolist()
        gt_labels = tgt['gt_labels'][mask].tolist()
        entry = {
            'height': canvas, 'width': canvas,
            'gt_boxes': gt_boxes, 'gt_labels': [int(v) for v in gt_labels],
        }
        if color:
            rgb = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
            rgb = rgb.contiguous().numpy()  # (s, s, 3) HWC
            entry['rgb_b64'] = base64.b64encode(rgb.reshape(-1).tobytes()) \
                .decode('ascii')
        else:
            gray = (img[0].clamp(0, 1) * 255).round().byte().numpy()  # (s,s)
            entry['gray_b64'] = base64.b64encode(gray.reshape(-1).tobytes()) \
                .decode('ascii')
        samples.append(entry)
    return samples


def _det_labels(dataset: str, args_dict: dict, n_classes: int) -> list[str]:
    """検出モデルのラベル列を dataset 種別から決定する純関数。

    face: WIDER FACE は1クラス固定で ['face']。fewclass: args_dict の
    'fewclass_classes'(カンマ区切り)を分割、空なら数値連番へフォールバック。
    それ以外(multimnist/voc/multimnist_color等): 数値文字列の連番
    (['0','1',...])。cmd_det_ckpt のラベル決定ロジックをテスト可能な純関数に
    切り出したもの(元のif/else分岐と等価)。
    """
    if dataset == 'face':
        return ['face']
    if dataset == 'fewclass':
        names = [
            c.strip() for c in args_dict.get('fewclass_classes', '').split(',')
            if c.strip()
        ]
        return names if names else [str(i) for i in range(n_classes)]
    return [str(i) for i in range(n_classes)]


def _build_widerface_det_demo_samples(
        canvas: int, count: int, *, root: str, min_box_px: float,
        max_blur: int, drop_invalid: bool, seed: int = 7) -> list[dict]:
    """WiderFaceDetection(実val split)から demo用の生RGB canvas+GTを取り出す。

    n_thresholds=0・use_rgb=False で二値化前の生RGBを得る
    (_build_fewclass_det_demo_samples と同じ方針)。ckptのargsに保存された
    フィルタ設定(face_min_box_px/face_max_blur/drop_invalid/face_root)を
    呼び出し側から引き継ぐ(引き継がないとGT破線が学習時のフィルタと
    食い違う: 学習時に除外された小さい/ぼやけた顔がdemoにだけ現れる)。
    """
    sys.path.insert(0, str(REPO / 'experiments'))
    from data.detection_face import WiderFaceDetection
    ds = WiderFaceDetection(
        root=root, train=False, img_size=canvas, n_thresholds=0,
        use_rgb=False, download=False, min_box_px=min_box_px,
        max_blur=max_blur, drop_invalid=drop_invalid)
    rng = np.random.default_rng(seed)
    n = min(count, len(ds))
    idx = sorted(int(i) for i in rng.choice(len(ds), size=n, replace=False))
    samples = []
    for i in idx:
        img, tgt = ds[i]  # img: (3,S,S) [0,1] 生RGB(二値化前)
        mask = tgt['mask'].bool()
        gt_boxes = tgt['gt_boxes'][mask].tolist()
        gt_labels = tgt['gt_labels'][mask].tolist()
        rgb = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
        rgb = rgb.contiguous().numpy()  # (S,S,3) HWC
        samples.append({
            'height': canvas, 'width': canvas,
            'gt_boxes': gt_boxes, 'gt_labels': [int(v) for v in gt_labels],
            'rgb_b64': base64.b64encode(rgb.reshape(-1).tobytes()).decode('ascii'),
        })
    return samples


def _build_fewclass_det_demo_samples(canvas: int, count: int, classes: list[str],
                                     seed: int = 7) -> list[dict]:
    """FewClassVOCDetection(実VOCテスト画像、person/car/dog)から demo用サンプルを取る。

    n_thresholds=0・use_rgb=False で二値化前の生RGB canvasを得る(_build_det_demo_samples
    のMulti-MNIST版と同じ方針)。データセット自体が該当クラスを1つ以上含む画像のみに
    フィルタ済みのため、どのサンプルにも person/car/dog のいずれかのGTが必ず入る。
    """
    sys.path.insert(0, str(REPO / 'experiments'))
    from data.detection_fewclass import FewClassVOCDetection
    ds = FewClassVOCDetection(
        root=str(REPO / 'data-voc'), train=False, img_size=canvas,
        n_thresholds=0, classes=tuple(classes), use_rgb=False, download=False)
    rng = np.random.default_rng(seed)
    n = min(count, len(ds))
    idx = sorted(int(i) for i in rng.choice(len(ds), size=n, replace=False))
    samples = []
    for i in idx:
        img, tgt = ds[i]  # img: (3,S,S) [0,1] 生RGB(二値化前)
        mask = tgt['mask'].bool()
        gt_boxes = tgt['gt_boxes'][mask].tolist()
        gt_labels = tgt['gt_labels'][mask].tolist()
        rgb = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
        rgb = rgb.contiguous().numpy()  # (S,S,3) HWC
        samples.append({
            'height': canvas, 'width': canvas,
            'gt_boxes': gt_boxes, 'gt_labels': [int(v) for v in gt_labels],
            'rgb_b64': base64.b64encode(rgb.reshape(-1).tobytes()).decode('ascii'),
        })
    return samples


def cmd_det_ckpt(args):
    """学習済み CenterNetLUT を多ヘッド DAG抽出→webbench形式へ(color・
    fewclass=少クラス実物体検出(person/car/dog)対応)。"""
    from griffith_logic.export import export_dag_ir
    model, a, best = build_centernetlut_from_ckpt(args.ckpt)
    n_thr = a['n_thresholds']
    s = a['canvas_size']
    stride = a.get('output_stride', 4)
    dataset = a.get('dataset')
    # 低レベル特徴検出器(edge/curv/chroma)付きで訓練したckptは、
    # preproc kind を 'image_lowlevel' に切り替えて配信する(TS側は
    # encodeImageLowLevel が同じブロック順で生成する)。
    _extra_feats = {k: _parse_thr_list(a.get(k)) for k in
                    ('edge_thresholds', 'curv_thresholds', 'chroma_thresholds')}
    has_lowlevel = any(_extra_feats.values())
    is_fewclass = dataset == 'fewclass'
    is_face = dataset == 'face'
    is_color = dataset in _RGB_DATASETS
    n_rgb = 3 if is_color else 1
    in_channels = n_output_channels(
        n_rgb, n_thresholds=n_thr,
        edge_thresholds=_extra_feats['edge_thresholds'],
        curv_thresholds=_extra_feats['curv_thresholds'],
        chroma_thresholds=_extra_feats['chroma_thresholds'],
    )
    print(f'CenterNetLUT k={a["k"]} in_ch={in_channels} canvas={s} color={is_color} '
          f'wiring={a.get("wiring", "fixed")} '
          f'best_hard_mAP={best:.4f} num_gates(weight-shared)={model.num_gates:,}')

    graph = export_dag_ir(model, (in_channels, s, s), name=args.label)
    # モデル→IR の忠実性をここで必ず検証(不一致なら書き出し中止)
    assert_ir_matches_torch(model, graph, (in_channels, s, s),
                            label=f'det {args.label}')
    if is_color and not has_lowlevel:
        # カラー版: dataset ThresholdBinarize が thr-major(thr*3+rgb)で
        # torch.catするため、convpreproc.ts(rgb-major)との差を
        # seg(cmd_seg_ckpt)と同じ恒等置換permuteノードで吸収する
        _prepend_input_permute(
            graph, _thermo_channel_permute_node(3, n_thr, s, s))

    # 閾値は dataset ThresholdBinarize と同一の (i+1)/(T+1)、判定 x>thr。
    # グレースケール版はconvpreproc.ts の c=1(赤ch=gray)のみ読む契約。
    thr_vals = [(i + 1) / (n_thr + 1) for i in range(n_thr)]
    if has_lowlevel:
        # 低レベル特徴検出器(Conv-DLGN §3.5 の edge/curvature 相当)付き。
        # TS 側(encodeImageLowLevel)は Python の encode_image_features と同じ
        # ブロック順 [thermo][edge][curv][chroma] / thr-major で直接生成するため、
        # thermometer 版で必要だった rgb↔thr 置換permuteは**不要**(上で抑止済み)。
        # カーネル方向・チャネル対はここで明示的に書き出し、TS 側をデータ駆動に保つ。
        preproc = {
            'kind': 'image_lowlevel', 'rgb_channels': n_rgb, 'n_thresholds': n_thr,
            'height': s, 'width': s,
            'mean': [0.0] * n_rgb, 'std': [1.0] * n_rgb,
            'thresholds': [list(thr_vals) for _ in range(n_rgb)],
            'edge_thresholds': list(_extra_feats['edge_thresholds']),
            'curv_thresholds': list(_extra_feats['curv_thresholds']),
            'chroma_thresholds': list(_extra_feats['chroma_thresholds']),
            'edge_offsets': [list(o) for o in EDGE_OFFSETS],
            'curv_offsets': [list(o) for o in CURV_OFFSETS],
            'chroma_pairs': [list(p) for p in CHROMA_PAIRS],
            'bit_order': 'block,detector,thr,sign,c,y,x',
        }
    else:
        preproc = {
            'kind': 'image_thermometer', 'rgb_channels': n_rgb, 'n_thresholds': n_thr,
            'height': s, 'width': s,
            'mean': [0.0] * n_rgb, 'std': [1.0] * n_rgb,
            'thresholds': [list(thr_vals) for _ in range(n_rgb)],
            'bit_order': 'rgb,thr,y,x',
        }
    readout = build_det_readout(
        graph,
        {'hm': model.hm_sum.scale, 'size': model.size_sum.scale,
         'off': model.off_sum.scale},
        output_stride=stride, top_k=args.top_k, threshold=args.threshold)
    labels = _det_labels(dataset, a, readout['n_classes'])
    eval_info = {'best_hard_map': float(best),
                 'num_gates_weight_shared': int(model.num_gates)}
    # 説明文を機械可読に。ページ側は about があればそれを使い、
    # 無ければ既存の静的散文にフォールバックする。
    about = build_about(
        task='detection', args_dict=a, source_ckpt=args.ckpt,
        notes=['DLGNの二値入力(thermometer / 低レベル特徴検出器)は固定・非学習。'
               'teacher CNN は学習時と同一の生入力を消費し、同一 CenterNet decode を通す。'],
    )
    doc = build_model_doc(graph, preproc=preproc, labels=labels,
                          eval_info=eval_info, readout=readout, about=about)
    if args.demo_samples > 0:
        if is_fewclass:
            doc['demo_samples'] = _build_fewclass_det_demo_samples(
                s, args.demo_samples, labels, seed=args.sample_seed)
            print(f'  demo_samples: {len(doc["demo_samples"])}枚(実VOC test、'
                  f'{"/".join(labels)}含む)')
        elif is_face:
            # ckptのargsからフィルタ設定を引き継ぐ(引き継がないと
            # GT破線が学習時のフィルタと食い違う)
            face_root = a.get('face_root') or str(REPO / 'data-widerface')
            face_min_box_px = a.get('face_min_box_px')
            if face_min_box_px is None:
                face_min_box_px = 4.0
            face_max_blur = a.get('face_max_blur')
            if face_max_blur is None:
                face_max_blur = 2
            drop_invalid = not a.get('face_keep_invalid', False)
            doc['demo_samples'] = _build_widerface_det_demo_samples(
                s, args.demo_samples, root=face_root,
                min_box_px=face_min_box_px, max_blur=face_max_blur,
                drop_invalid=drop_invalid, seed=args.sample_seed)
            print(f'  demo_samples: {len(doc["demo_samples"])}枚'
                  f'(実WIDER FACE val、min_box_px={face_min_box_px})')
        else:
            doc['demo_samples'] = _build_det_demo_samples(
                s, n_thr, args.demo_samples, seed=args.sample_seed, color=is_color)
            print(f'  demo_samples: {len(doc["demo_samples"])}枚'
                  f'(実Multi-MNIST{"-color" if is_color else ""})')

    if args.dry_run:
        print(json.dumps({'resources': doc['resources'],
                          'readout': readout}, indent=1, ensure_ascii=False))
        return
    _write_model_bundle(WEBBENCH / 'public' / 'models' / f'{args.label}.json', doc)
    write_manifest()
    r = doc['resources']
    print(f'  total_luts(unrolled)={r.get("total_luts"):,}  '
          f'json={r.get("json_bytes"):,}B  gzip={r.get("gzip_bytes"):,}B  '
          f'bundle={r.get("bundle_json_bytes"):,}B/'
          f'{r.get("bundle_gzip_bytes"):,}B(gz)  '
          f'out_width={graph.outputs[0].width:,}')


# ---------------------------------------------------------------------------
# 学習済みセグモデル(SegConvDLGN)→ webbench
# ---------------------------------------------------------------------------
def _thermo_channel_permute_node(n_rgb: int, n_thr: int, h: int, w: int):
    """rgb-major(convpreproc.ts) → thr-major(dataset/model) の恒等置換permuteノード。

    convpreproc.ts は ch=rgb*T+thr(rgb-major)でビットを並べるが、
    学習時の入力(datasets/segmentation.ThermometerEncode3ch)は
    torch.cat over thresholds = ch=thr*n_rgb+rgb(thr-major)。この差を
    IR先頭のパススルー置換で吸収し、convpreproc.ts を無改造で使う。

    全空間位置ぶんlut_layerへ展開する(旧実装は1x1x128x128級のモデルで
    JSON肥大の主因の一つだった)代わりに、入力を(n_rgb, n_thr, H, W)の
    reshape viewとみなしperm=[1,0,2,3](rgb/thr軸の入替)で表す小さい
    permuteノードにする(index無し。実行時にlower_to_primitives/ir/simulate
    がJIT計算する。旧lut_layer展開とビット厳密一致)。
    """
    return IRNode(
        id='n0_chanperm', type='permute',
        params={'in_shape': [n_rgb, n_thr, h, w], 'perm': [1, 0, 2, 3]},
        train_info={'from': 'thermo_channel_permute(rgb_major->thr_major)'},
    )


def _prepend_input_permute(graph: IRGraph, perm: IRNode) -> None:
    """入力ポート直後に置換ノードを挿入する(in-place)。"""
    in_width = int(np.prod(perm.params['in_shape']))
    assert graph.inputs[0].width == in_width, (
        graph.inputs[0].width, in_width)
    first = next(e.dst for e in graph.edges if e.src == graph.inputs[0].id)
    graph.nodes.insert(0, perm)
    graph.edges = [e for e in graph.edges
                   if not (e.src == graph.inputs[0].id and e.dst == first)]
    graph.edges.insert(0, IREdge(src=graph.inputs[0].id, dst=perm.id))
    graph.edges.insert(1, IREdge(src=perm.id, dst=first))


def assert_ir_matches_torch(model, graph, in_shape, *, n_samples: int = 3,
                            seed: int = 0, label: str = ''):
    """(再発防止)exportしたIRが PyTorch hard eval と整数厳密一致するか検証する。

    なぜここで検証するのか
    ----------------------
    抽出器が wiring='learnable' の argmax 配線解決を怠り固定 indices_flat で
    IRを組んでいたバグ(実測: seg 学習配線で画素一致54%、det でデモ検出0件)は、
    **TS⇔simulate の自己整合テストでは原理的に検出できない**。ゴールデンを
    「壊れたIR自身」から生成するため、壊れたIR同士が一致してテストが緑になる。
    モデル→IR の唯一の関門である export でここを突き合わせるのが最後の砦。

    SpatialGroupSum.scale は simulate が無視する(生 popcount を返す)ため、
    検証中だけ scale=1.0 に落として整数同士で厳密比較する。
    """
    import torch

    from griffith_logic.ir.simulate import simulate
    from griffith_logic.spatial_group_sum import SpatialGroupSum

    C, H, W = in_shape
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n_samples, C, H, W), generator=g).float()
    saved = [(m, m.scale) for m in model.modules()
             if isinstance(m, SpatialGroupSum)]
    for m, _ in saved:
        m.scale = 1.0
    try:
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict):
            heads = graph.metadata.get('output_heads') or []
            exp = torch.cat([out[h['name']].reshape(n_samples, -1)
                             for h in heads], dim=1).numpy()
        else:
            exp = out.reshape(n_samples, -1).numpy()
    finally:
        for m, s in saved:
            m.scale = s
    got = np.asarray(simulate(graph, x.flatten(1).numpy().astype(np.int64)))
    exp = np.rint(exp).astype(np.int64)
    if got.shape != exp.shape or not np.array_equal(got.astype(np.int64), exp):
        detail = (f'max|diff|={np.abs(got.astype(np.int64) - exp).max()}'
                  if got.shape == exp.shape
                  else f'shape torch={exp.shape} ir={got.shape}')
        raise SystemExit(
            f'\n[export検証NG] {label}: PyTorch hard eval ⇔ IR simulate 不一致 '
            f'({detail})。IRがモデルを再現していないため書き出しを中止した。\n'
            f'  → 抽出器(griffith_logic/export/lowering.py)がモデル設定'
            f'(例: wiring="learnable" の argmax 配線解決)を反映しているか確認せよ。')
    print(f'  ✓ export検証OK: PyTorch hard eval ⇔ IR simulate 整数厳密一致 '
          f'(n={n_samples})')


def build_segconvdlgn_from_ckpt(ckpt_path: str):
    """SegConvDLGN best.pt → (model, args_dict, best_metric)。"""
    import torch
    sys.path.insert(0, str(REPO / 'experiments'))
    from segmentation_models import SegConvDLGN
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = (data.get('metadata', {}) or {}).get('args', {}) or {}
    in_channels = 3 * a['n_thresholds']
    model = SegConvDLGN(
        in_channels=in_channels, k=a['k'], tree_depth=a['tree_depth'],
        readout_group=a['readout_group'],
        input_channels_per_tree=a['input_channels_per_tree'],
        tau=a.get('tau'),
        wiring=a.get('wiring', 'fixed'),  # 学習配線ckpt対応
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model, a, data.get('best_metric')


def _build_seg_demo_samples(img_size: int, count: int, seed: int = 7,
                            border: str = 'foreground') -> list[dict]:
    """Oxford-IIIT Pet testスプリットから実画像+GTバイナリマスクを取り出しdemo用に埋める。

    encoding='float'で生RGB(3,S,S)を得る(demoはRGB canvasを描き、ブラウザ側で
    同一thermometer前処理を再現して推論する、det の実サンプル方式に倣う)。
    mask は (S,S) の {0,1} バイナリ(1=pet)を uint8 でそのままpackする。
    """
    import torch

    from griffith_logic.datasets.segmentation import OxfordPetBinarySeg
    ds = OxfordPetBinarySeg(split='test', img_size=img_size, encoding='float',
                            border=border, download=True)
    rng = np.random.default_rng(seed)
    n = min(count, len(ds))
    idx = sorted(int(i) for i in rng.choice(len(ds), size=n, replace=False))
    samples = []
    for i in idx:
        img, mask = ds[i]  # img: (3,S,S) float[0,1], mask: (S,S) long{0,1}
        rgb = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
        rgb = rgb.contiguous().numpy()  # (S,S,3) HWC
        m = mask.to(dtype=torch.uint8).contiguous().numpy()  # (S,S)
        samples.append({
            'height': img_size, 'width': img_size,
            'rgb_b64': base64.b64encode(rgb.reshape(-1).tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(m.reshape(-1).tobytes()).decode('ascii'),
        })
    return samples


def cmd_seg_ckpt(args):
    """学習済み SegConvDLGN を DAG抽出→webbench形式へ(実モデル接続)。"""
    from griffith_logic.export import export_dag_ir
    model, a, best = build_segconvdlgn_from_ckpt(args.ckpt)
    n_thr = a['n_thresholds']
    s = a['img_size']
    in_channels = 3 * n_thr
    print(f'SegConvDLGN k={a["k"]} in_ch={in_channels} img={s} '
          f'best_hard_miou={best:.4f} num_gates(weight-shared)={model.num_gates:,}')

    graph = export_dag_ir(model, (in_channels, s, s), name=args.label)
    # モデル→IR の忠実性をここで必ず検証(不一致なら書き出し中止)
    assert_ir_matches_torch(model, graph, (in_channels, s, s),
                            label=f'seg {args.label}')
    _prepend_input_permute(
        graph, _thermo_channel_permute_node(3, n_thr, s, s))

    thr_vals = [(i + 1) / (n_thr + 1) for i in range(n_thr)]
    preproc = {
        'kind': 'image_thermometer', 'rgb_channels': 3, 'n_thresholds': n_thr,
        'height': s, 'width': s,
        # datasets/segmentation.ThermometerEncode3ch を convpreproc.ts の式
        # z=(x-mean)/std, z>thr で再現(mean=0/std=1、閾値=(i+1)/(T+1))
        'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0],
        'thresholds': [list(thr_vals) for _ in range(3)],
        'bit_order': 'rgb,thr,y,x',
    }
    labels = ['background', 'pet']
    eval_info = {'best_hard_miou': float(best),
                 'num_gates_weight_shared': int(model.num_gates)}
    # 説明文を機械可読に(ページ側は無ければ静的散文へフォールバック)。
    about = build_about(
        task='segmentation', args_dict={'dataset': 'pet', **(a or {})},
        source_ckpt=args.ckpt,
        notes=['DLGN は thermometer 二値化入力、ONNX/TF.js は生 float 入力'
               '(各ランタイムの学習時ネイティブ前処理)。'],
    )
    doc = build_model_doc(graph, preproc=preproc, labels=labels,
                          eval_info=eval_info, about=about)
    doc['readout']['spatial'] = {'n_out': 2, 'height': s, 'width': s}
    if args.demo_samples > 0:
        doc['demo_samples'] = _build_seg_demo_samples(
            s, args.demo_samples, seed=args.sample_seed,
            border=a.get('border', 'foreground'))
        print(f'  demo_samples: {len(doc["demo_samples"])}枚(実Oxford-IIIT Pet)')
    if args.dry_run:
        print(json.dumps(doc['resources'], indent=1, ensure_ascii=False))
        return
    _write_model_bundle(WEBBENCH / 'public' / 'models' / f'{args.label}.json', doc)
    write_manifest()
    r = doc['resources']
    print(f'  total_luts(unrolled)={r.get("total_luts"):,}  '
          f'json={r.get("json_bytes"):,}B  gzip={r.get("gzip_bytes"):,}B  '
          f'bundle={r.get("bundle_json_bytes"):,}B/'
          f'{r.get("bundle_gzip_bytes"):,}B(gz)')

    if args.golden_image:
        _emit_seg_e2e_golden(graph, args.golden_image, s, n_thr, thr_vals,
                             args.label)


def _emit_seg_e2e_golden(graph: IRGraph, image_paths: list[str], s: int,
                         n_thr: int, thr_vals: list[float], model_file: str):
    """実画像→canvas相当RGBA→convpreproc→simulate の端到端ゴールデン。

    TS側(seg_e2e.test.ts)が実 ConvRuntime で同一RGBAから同一マスクを
    再現できることを整数厳密に検証する(前処理〜argmaxまで端到端)。
    RGBAはPILでSxSへbilinearリサイズした確定バイト列を両側で共有する
    (canvas drawImage のリサイズ差を排除するため画素を固定する)。
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    cases = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        # PIL bilinear で SxS に固定(両側同一画素)
        t = TF.resize(TF.to_tensor(img), [s, s])       # (3,S,S) [0,1]
        rgb = (t.clamp(0, 1) * 255).round().byte().numpy()  # (3,S,S)
        hw = s * s
        rgba = np.zeros((hw, 4), dtype=np.uint8)
        for c in range(3):
            rgba[:, c] = rgb[c].reshape(-1)
        rgba[:, 3] = 255
        # Python参照: rgba→[0,1]→thermometer(rgb-major, z>thr)→simulate
        xf = rgba[:, :3].astype(np.float64) / 255.0     # (hw, 3)
        bits = np.zeros((3 * n_thr, hw), dtype=np.int64)
        for c in range(3):
            for k in range(n_thr):
                bits[c * n_thr + k] = (xf[:, c] > thr_vals[k]).astype(np.int64)
        scores = simulate(graph, bits.reshape(1, -1))
        mask = scores.reshape(1, 2, hw).argmax(axis=1)[0]
        cases.append({
            'name': Path(path).stem,
            'rgba_b64': base64.b64encode(rgba.reshape(-1).tobytes())
            .decode('ascii'),
            'expected_mask': mask.astype(np.int64).tolist(),
            'pet_fraction': float((mask == 1).mean()),
        })
    _write_json(WEBBENCH / 'src' / 'fixtures' / 'seg_e2e_golden.json',
                {'model_file': f'{model_file}.json', 'height': s, 'width': s,
                 'n_out': 2, 'cases': cases})
    print(f'  seg e2e golden: {len(cases)}枚 '
          f'(pet率 {[round(c["pet_fraction"], 3) for c in cases]})')


def cmd_ir(args):
    graph = load_ir(args.ir)
    label = args.label or graph.metadata.get('name', Path(args.ir).stem)
    # preproc は別途 --preproc-json で供給(実データ経路)。無ければ空
    preproc = json.loads(Path(args.preproc_json).read_text()) \
        if args.preproc_json else {'kind': 'raw'}
    labels = json.loads(args.labels) if args.labels else []
    doc = build_model_doc(graph, preproc=preproc, labels=labels)
    if args.dry_run:
        print(json.dumps(doc['resources'], indent=1, ensure_ascii=False))
        return
    _write_model_bundle(WEBBENCH / 'public' / 'models' / f'{label}.json', doc)
    write_manifest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    pf = sub.add_parser('fixtures', help='TS整数一致テスト用ゴールデン生成')
    pf.add_argument('-n', type=int, default=6, help='ケースあたりサンプル数')
    pf.set_defaults(func=cmd_fixtures)

    pd = sub.add_parser('demo', help='デモUI用ランダム小モデル')
    pd.add_argument('--size', type=int, default=16)
    pd.add_argument('--channels', type=int, default=9, help='二値入力ch(C*T)')
    pd.add_argument('--classes', type=int, default=2)
    pd.add_argument('--seed', type=int, default=0)
    pd.set_defaults(func=cmd_demo)

    ps = sub.add_parser('seg-ckpt', help='学習済みSegConvDLGN(best.pt)をwebbench形式へ')
    ps.add_argument('ckpt', help='SegConvDLGN best.pt パス')
    ps.add_argument('--label', default='seg_convdlgn', help='出力モデル名')
    ps.add_argument('--golden-image', nargs='*', default=None,
                    help='端到端ゴールデン用の画像パス(複数可)')
    ps.add_argument('--demo-samples', type=int, default=6,
                    help='doc に埋める実Oxford-IIIT Petテスト画像+GTマスク枚数(0で無効)')
    ps.add_argument('--sample-seed', type=int, default=7)
    ps.add_argument('--dry-run', action='store_true')
    ps.set_defaults(func=cmd_seg_ckpt)

    pdet = sub.add_parser('det-ckpt',
                          help='学習済みCenterNetLUT(best.pt)をwebbench形式へ')
    pdet.add_argument('ckpt', help='CenterNetLUT best.pt パス')
    pdet.add_argument('--label', default='det_centernet_lut', help='出力モデル名')
    pdet.add_argument('--top-k', type=int, default=100)
    pdet.add_argument('--threshold', type=float, default=0.1)
    pdet.add_argument('--demo-samples', type=int, default=6,
                      help='doc に埋める実Multi-MNISTサンプル枚数(0で無効)')
    pdet.add_argument('--sample-seed', type=int, default=7)
    pdet.add_argument('--dry-run', action='store_true')
    pdet.set_defaults(func=cmd_det_ckpt)

    pi = sub.add_parser('ir', help='学習済みIRをwebbench形式へ')
    pi.add_argument('ir', help='.ir.json パス')
    pi.add_argument('--label', default=None)
    pi.add_argument('--labels', default=None, help='JSON配列文字列')
    pi.add_argument('--preproc-json', default=None)
    pi.add_argument('--dry-run', action='store_true')
    pi.set_defaults(func=cmd_ir)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
