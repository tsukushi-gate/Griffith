"""dense lut_layer 列への IR sweep の組み込み。

``ir/sweep.py``(定数伝播・縮退スロット束縛・CSE・未参照ゲート除去、SM2)は
測定用の正規化パスとして作られ、デプロイ経路には繋がっていなかった。本モジュールは
それを ``griffith deploy bundle --sweep`` として配布バンドル化の前段に挟むための
``doc → doc`` 変換を提供する(``deploy/resynth.py`` の conv2d 向け再合成と同じ
位置づけ・同じ流儀: 真実源のIRを置き換えた新docを作り、その後 ``build_bundle`` が
新IRに対して programs 等を再構築する)。

lossless の根拠と多重防御:
    sweep の正しさの定義は「任意入力で simulate(graph) == simulate(graph')」
    (tests/integration/test_ir_sweep.py)。本モジュールはその上で、返す前に
    必ず ``ir/simulate.py`` 基準の整数一致検証を行う(ランダム入力、および
    ``demo_samples`` があれば実データでも)。1件でも不一致なら ValueError を送出し、
    壊れた doc を絶対に返さない。検証は書き出し形式(``ir_to_dict`` 経由の
    シリアライズ後)に対して行う: メモリ上のグラフだけ正しくて直列化で壊れる
    型の不具合も同じ関門で捕まえる。

対象範囲:
    **dense(lut_layer + group_sum 等)専用**。conv2d/or_pool2d/upsample2d/permute
    (``export.lowering.LOWERABLE_TYPES``)を含むIRは、sweep が lut_layer 列へ展開
    してしまい conv2d の重み共有(命令列化)を失うため拒否する(conv 向けは
    ``--resynth``)。recurrent ノードも sweep のテスト範囲外なので拒否する。

層IRの構造的制約(``ir/sweep.py`` のdocstring参照): パススルー/NOTゲート(サポート1)
は層から除去されない。幅が変わるのは「消費者が全て lut_layer」の層だけ。
"""

import base64
from copy import deepcopy

import numpy as np

from ..export.lowering import LOWERABLE_TYPES
from ..ir.serialize import ir_from_dict, ir_to_dict
from ..ir.simulate import simulate
from ..ir.sweep import sweep
from ..ir.validate import validate
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits
from .resynth import _decode_demo_images, _refresh_resources

#: sweep が扱わない(=拒否する)ノード型。LOWERABLE_TYPES は sweep が展開して
#: しまうため、recurrent は本体サブグラフのsweep意味論が未検証のため。
UNSUPPORTED_TYPES = tuple(LOWERABLE_TYPES) + ('recurrent',)


def _demo_bits(doc: dict, in_width: int) -> np.ndarray | None:
    """demo_samples から実データのIR入力ビット列 [N, in_width] を作る。

    - 画像前処理(image_thermometer/image_lowlevel): 埋め込み画像を前処理記述子
      でビット化する(resynth.py と同じ経路)
    - それ以外: 各サンプルが ``bits_b64``(LSB-first packbits のbase64、
      ``scripts/export_webbench.py::pack_bits_b64`` と同形)を持つ場合のみ
      それを展開する
    読めない形式なら None(ランダム入力の検証だけが残る)。
    """
    demo_samples = doc.get('demo_samples') or []
    if not demo_samples:
        return None
    preproc = doc.get('preproc') or {}
    if preproc.get('kind') in SUPPORTED_KINDS:
        images = _decode_demo_images(demo_samples, preproc)
        plan = build_preproc_descriptors(preproc)
        return encode_bits(plan, images)
    rows = []
    for sample in demo_samples:
        if not isinstance(sample, dict) or 'bits_b64' not in sample:
            return None
        raw = np.frombuffer(base64.b64decode(sample['bits_b64']), dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder='little')
        if bits.size < in_width:
            raise ValueError(
                f'demo_samples[bits_b64] のビット数 {bits.size} が '
                f'IR入力幅 {in_width} に満たない')
        rows.append(bits[:in_width])
    return np.stack(rows, axis=0)


def sweep_doc_lossless(doc: dict, *, verify_samples: int = 8,
                       seed: int = 0) -> tuple[dict, dict]:
    """docの dense lut_layer 列を sweep した新docと要約を返す。

    ``doc`` 自体は変更しない。返す doc は ``ir`` を sweep 後のIRに差し替え、
    ``resources`` を再計算し、``about.sweep`` に sweep のレポートを記録する
    (``about.resynth`` と同じ置き場。バンドルのトップレベル ``about`` 節は
    ``build_bundle`` がそのまま引き継ぐ)。

    Args:
        doc: webbenchモデルdoc(または同形式のdict)。少なくとも 'ir' を持つ
        verify_samples: 検証に使うランダム入力サンプル数(0以下は不可)
        seed: 検証用乱数のシード

    Returns:
        (new_doc, summary)。summary は ``sweep()`` のreport
        (gates_before/gates_after/swept_gates/sweep_ratio/…)に
        ``verify_samples``(ランダム)と ``verify_demo_samples`` を加えたもの

    Raises:
        ValueError: ``ir`` が無い、conv2d等を含む、sweep後IRが validate を
            通らない、または出力整数がランダム入力(または demo_samples の
            実データ)上で元と一致しなかった場合(多重防御)
    """
    if 'ir' not in doc:
        raise ValueError("sweep_doc_lossless: doc に 'ir' がない")
    if verify_samples < 1:
        raise ValueError(
            f'sweep_doc_lossless: verify_samples は1以上が必要: {verify_samples}')

    old_graph = ir_from_dict(doc['ir'])
    bad = sorted({n.type for n in old_graph.nodes if n.type in UNSUPPORTED_TYPES})
    if bad:
        raise ValueError(
            f'--sweep は dense(lut_layer列)専用: {bad} を含むIRには適用できない'
            '(conv2d等は --resynth lossless を使う)')

    # sweep() はIRを書き換えうるので、検証用の old_graph とは別の複製に適用する
    # (sweep 自身も複製を返すが、入力不変の契約に依存しない)。
    new_graph, report = sweep(deepcopy(old_graph))
    new_ir = ir_to_dict(new_graph)
    check_graph = ir_from_dict(new_ir)      # 書き出し形式の往復後で検証する
    validate(check_graph)

    # ---- 検証1: ランダムビット列上で厳密一致 ----
    in_width = int(old_graph.inputs[0].width)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(verify_samples, in_width), dtype=np.uint8)
    if not np.array_equal(simulate(old_graph, bits), simulate(check_graph, bits)):
        raise ValueError(
            'sweep_doc_lossless: ランダム入力で sweep 前後の出力が一致しない'
            '(sweepが壊れている可能性。バンドル化を中止)')

    # ---- 検証2: demo_samples があれば実データでも同様に確認 ----
    real_bits = _demo_bits(doc, in_width)
    n_demo = 0
    if real_bits is not None:
        n_demo = int(real_bits.shape[0])
        if not np.array_equal(simulate(old_graph, real_bits),
                              simulate(check_graph, real_bits)):
            raise ValueError(
                'sweep_doc_lossless: demo_samples の実データで sweep 前後の'
                '出力が一致しない(sweepが壊れている可能性。バンドル化を中止)')

    new_doc = dict(doc)
    new_doc['ir'] = new_ir
    _refresh_resources(new_doc, check_graph)

    summary = dict(report)
    summary['verify_samples'] = int(verify_samples)
    summary['verify_demo_samples'] = n_demo
    about = dict(doc.get('about') or {})
    about['sweep'] = {'mode': 'lossless', **summary}
    new_doc['about'] = about
    return new_doc, summary
