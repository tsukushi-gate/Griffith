"""conv2d 重み共有木の lossless 再合成。

``scripts/dc_resynth_webbench.py``のオフライン検証専用スクリプトから、
**care_sets=None(lossless)専用の経路だけ**をデプロイキット内へ移植したもの。
データセット/キャリブレーション関連(``care`` 採取・``iter_encoded`` 等の
非lossless要素)は一切持たない。

エンジンは ``export/dc_resynth.py::resynth_node_params`` で、care_sets=None
のときは各 (conv2dノード, 出力チャネル) の木を**全パターン**の真理値表に
対して再合成する: 探索が空振りなら現状維持なので構成的に等価(退化しない)
であり、原理上は精度不変。その上で本モジュールは**多重防御**として、
返す前に必ず ``ir/simulate.py`` 基準の整数一致検証を行う(ランダム入力、
および ``demo_samples`` があれば実データでも)。どちらか一つでも不一致なら
ValueError を送出し、壊れた doc を絶対に返さない。

用途: 配布バンドル化(``deploy/bundle.py::build_bundle``)の前段で、
conv2d木のゲート数を無劣化のまま削減する。CLIへの組み込みは別waveで行う
(本モジュールは単体で呼べる関数のみを提供する)。
"""

import base64
import gzip
import json
import time
from copy import deepcopy

import numpy as np

from ..export.dc_resynth import (
    ResynthStats,
    eval_tree,
    forward_graph,
    resynth_node_params,
    tree_function,
)
from ..export.estimate import estimate_resources
from ..ir.serialize import ir_from_dict
from ..ir.simulate import simulate
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits


def _decode_demo_images(demo_samples: list, preproc: dict) -> np.ndarray:
    """demo_samples の埋め込み画像を (N, C, H, W) uint8 CHW へ変換する。

    demo_samples の画像は base64 uint8 **HWC**(``scripts/export_webbench.py``
    の ``rgb_b64``/``gray_b64``)。``preproc_desc.encode_bits`` は
    [N,C,H,W] uint8 を要求する(前処理記述子は固定小数の整数演算のみで、
    浮動小数点を経由しない)ので、ここで HWC→CHW に転置するだけでよい。
    """
    c_img = len(preproc['mean'])
    imgs = []
    for sample in demo_samples:
        h = int(sample['height'])
        w = int(sample['width'])
        if 'rgb_b64' in sample:
            key = 'rgb_b64'
        elif 'gray_b64' in sample:
            key = 'gray_b64'
        else:
            raise ValueError(
                f'demo_samples に画像フィールド(rgb_b64/gray_b64)が無い: '
                f'{sorted(sample)}')
        raw = base64.b64decode(sample[key])
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size != h * w * c_img:
            raise ValueError(
                f'demo_samples[{key}] のサイズ {arr.size} が '
                f'{h}x{w}x{c_img}={h * w * c_img} と違う')
        imgs.append(arr.reshape(h, w, c_img).transpose(2, 0, 1))  # HWC -> CHW
    return np.stack(imgs, axis=0)


def _refresh_resources(doc: dict, graph, *, reuse_structure: bool = False) -> None:
    """resources を再計算する(``dc_resynth_webbench.refresh_resources`` の移植)。

    json_bytes/gzip_bytes は ``build_model_doc`` と同じ規約(demo_samples を
    含めない doc に対する測定)にそろえる。配信実体のサイズは別途
    full_bytes/full_gzip_bytes として全doc(demo_samples込み)に対して測る。
    estimate_resources が返さない旧キー(呼び出し元がdocに独自に足した値)は
    そのまま保持する。

    ``reuse_structure=True``: ``estimate_resources`` を呼ばず、doc が既に持つ
    構造由来のフィールド(total_luts / lut_counts / logic_depth / wire_count 等)を
    再利用する。conv2d 再合成は木の深さ・形・ノード形状を変えない(余った
    ゲートを tt の定数/パススルーにするだけ)ので、``lower_to_primitives`` の
    展開結果のLUT数・段数・配線数は変わらない。実測(seg_convdlgn_k32)で
    再合成前後の ``estimate_resources`` が完全一致することを確認している。
    この見積りの conv2d 展開は face で 60 秒かかるため、ここで省く。doc に
    ``total_luts`` が無い(再利用元が無い)ときは通常どおり再計算する。
    """
    old_res = doc.get('resources') or {}
    if reuse_structure and 'total_luts' in old_res:
        res = dict(old_res)
    else:
        res = estimate_resources(graph)
        for k, v in old_res.items():
            res.setdefault(k, v)
    doc['resources'] = res
    core = {k: v for k, v in doc.items() if k != 'demo_samples'}
    blob = json.dumps(core, separators=(',', ':')).encode()
    res['json_bytes'] = len(blob)
    res['gzip_bytes'] = len(gzip.compress(blob, 6))
    full = json.dumps(doc, separators=(',', ':')).encode()
    res['full_bytes'] = len(full)
    res['full_gzip_bytes'] = len(gzip.compress(full, 6))


VERIFY_MODES = ('fast', 'simulate')


def _verify_trees(old_params: dict, new_params: dict) -> None:
    """conv2d ノードの全木について、再合成前後の関数が一致することを全数確認する。

    各木の入力は高々 2^depth 個の(重複を含む)タップなので、元の木のユニーク
    タップ u 個に対する 2^u パターンの真理値表を比べれば**網羅的**に等価を
    保証できる(ランダム入力より強い)。変化していない木は読み飛ばす。
    """
    from ..ir.serialize import tt_from_hex

    def tts_of(params):
        raw = params['truth_tables']
        if raw and isinstance(raw[0], str):
            return tt_from_hex(list(raw))
        return [int(v) for v in raw]

    depth = int(old_params['tree_depth'])
    n_int = (1 << depth) - 1
    old_tts = tts_of(old_params)
    new_tts = tts_of(new_params)
    for oc in range(int(old_params['out_channels'])):
        o_row = [int(v) for v in old_params['indices_flat'][oc]]
        n_row = [int(v) for v in new_params['indices_flat'][oc]]
        o_t = old_tts[oc * n_int:(oc + 1) * n_int]
        n_t = new_tts[oc * n_int:(oc + 1) * n_int]
        if o_row == n_row and o_t == n_t:
            continue
        taps, onset = tree_function(o_row, o_t, depth)
        u = len(taps)
        n_pat = 1 << u
        full = (1 << n_pat) - 1
        mask_of = {}
        for v, pp in enumerate(taps):
            mm = 0
            for p in range(n_pat):
                if (p >> v) & 1:
                    mm |= 1 << p
            mask_of[pp] = mm
        if not set(n_row) <= set(mask_of):
            raise ValueError(
                f'resynth_doc_lossless: oc={oc} の再合成後の木が元に無い'
                'タップを参照している(出力が一致しない)')
        if eval_tree(n_row, n_t, depth, mask_of, full) != onset:
            raise ValueError(
                f'resynth_doc_lossless: oc={oc} の木の真理値表が再合成前後で'
                '一致しない(lossless再合成が壊れている可能性)')


def _verify_forward(old_graph, new_graph, bits_list) -> bool:
    """numpy 高速フォワード(conv2d を展開しない)で出力整数の一致を確かめる。

    ``forward_graph`` が未対応のノード型(lut_layer 等)を含むときは False
    を返す(呼び出し側が simulate へフォールバックする)。不一致は ValueError。
    """
    try:
        for bits in bits_list:
            old_out, _ = forward_graph(old_graph, bits)
            new_out, _ = forward_graph(new_graph, bits)
            if not np.array_equal(old_out, new_out):
                raise ValueError(
                    'resynth_doc_lossless: 高速フォワードで再合成前後の出力が'
                    '一致しない(lossless再合成が壊れている可能性)')
    except NotImplementedError:
        return False
    return True


def resynth_doc_lossless(doc: dict, *, max_cost: int = 5,
                         verify_samples: int = 8, seed: int = 0,
                         verify: str = 'simulate') -> tuple[dict, dict]:
    """docの全conv2dノードを lossless 再合成した新docと要約統計を返す。

    ``doc`` 自体は変更しない(deepcopyしてから編集する)。

    Args:
        doc: webbenchモデルdoc(または同形式のdict)。少なくとも 'ir' を持つ
        max_cost: ``resynth_node_params`` に渡す探索ゲート数上限
        verify_samples: 検証に使うランダム入力サンプル数
        seed: 検証用乱数のシード
        verify: 検証方式。ライブラリの既定は ``'simulate'``(golden model
            = ``ir/simulate.py`` との照合を既定に保つ)。CLI の
            ``bundle --compress fast`` だけが ``'fast'`` を明示して渡す
            (``compress.RESYNTH_BY_LEVEL``)。
            ``'fast'``: 木ごとの真理値表を全数照合(``_verify_trees``)
            +numpy 高速フォワード(``dc_resynth.forward_graph``)の整数一致
            (ランダム入力、および demo_samples の実データ)。conv2d を LUT へ
            展開しないので face で数秒(simulate 方式は数百秒、検証がほぼ
            全時間を占めていた)。フォワードが未対応のノード型を含む IR は
            simulate 方式へ自動フォールバックする。
            ``'simulate'``(既定): ``ir/simulate.py`` 基準の整数一致(従来どおり。
            ``--compress max`` / ``--resynth lossless`` 用)。resources の構造由来フィールドも
            ``estimate_resources`` で再計算する。

    Returns:
        (new_doc, summary)。summary は
        {'gates_before', 'gates_after', 'nodes_changed', 'n_trees',
         'n_trees_changed', 'verify', 'seconds'}

    Raises:
        ValueError: 再合成後の出力整数がランダム入力(または demo_samples の
            実データ)上で元と一致しなかった場合(=多重防御の検証に失敗。
            壊れた doc を返さないための安全弁)
    """
    if verify not in VERIFY_MODES:
        raise ValueError(
            f'verify={verify!r} は不正({"/".join(VERIFY_MODES)} のみ)')
    t_start = time.perf_counter()
    new_doc = deepcopy(doc)

    total = ResynthStats()
    nodes_changed = 0
    for node in new_doc['ir']['nodes']:
        if node['type'] != 'conv2d':
            continue
        old_params = node['params']
        new_params, st = resynth_node_params(old_params, None, max_cost=max_cost)
        if verify == 'fast':
            _verify_trees(old_params, new_params)
        node['params'] = new_params
        total.merge(st)
        if st.n_changed:
            nodes_changed += 1

    old_graph = ir_from_dict(doc['ir'])
    new_graph = ir_from_dict(new_doc['ir'])

    in_width = int(old_graph.inputs[0].width)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(verify_samples, in_width), dtype=np.uint8)
    real_bits = None
    demo_samples = doc.get('demo_samples') or []
    preproc = doc.get('preproc') or {}
    if demo_samples and preproc.get('kind') in SUPPORTED_KINDS:
        images = _decode_demo_images(demo_samples, preproc)
        plan = build_preproc_descriptors(preproc)
        real_bits = encode_bits(plan, images)

    used_mode = verify
    if verify == 'fast':
        inputs = [bits] + ([real_bits] if real_bits is not None else [])
        if not _verify_forward(old_graph, new_graph, inputs):
            used_mode = 'simulate'      # 高速フォワード未対応のノード型を含む

    if used_mode == 'simulate':
        # ---- 検証1: ランダムビット列上で厳密一致 ----
        # lossless再合成は resynth_node_params 側で構成的に等価が保証されるが、
        # 埋め込み(embed_balanced)・直列化のバグを踏んでいないことを
        # ここでも独立に確かめる(多重防御)。
        old_out = simulate(old_graph, bits)
        new_out = simulate(new_graph, bits)
        if not np.array_equal(old_out, new_out):
            raise ValueError(
                'resynth_doc_lossless: ランダム入力で再合成前後の出力が一致しない'
                '(lossless再合成が壊れている可能性)')

        # ---- 検証2: demo_samples があれば実データでも同様に確認 ----
        if real_bits is not None:
            old_out2 = simulate(old_graph, real_bits)
            new_out2 = simulate(new_graph, real_bits)
            if not np.array_equal(old_out2, new_out2):
                raise ValueError(
                    'resynth_doc_lossless: demo_samples の実データで再合成前後の'
                    '出力が一致しない(lossless再合成が壊れている可能性)')

    _refresh_resources(new_doc, new_graph,
                       reuse_structure=(used_mode == 'fast'))

    gates_before = total.cost_before
    gates_after = total.cost_after
    new_doc.setdefault('about', {})['resynth'] = {
        'mode': 'lossless',
        'max_cost': max_cost,
        'verify': used_mode,
        'gates_before': gates_before,
        'gates_after': gates_after,
    }
    summary = {
        'gates_before': gates_before,
        'gates_after': gates_after,
        'nodes_changed': nodes_changed,
        'n_trees': total.n_trees,
        'n_trees_changed': total.n_changed,
        'verify': used_mode,
        'seconds': time.perf_counter() - t_start,
    }
    return new_doc, summary
