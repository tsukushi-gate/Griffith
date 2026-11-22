"""配布バンドル v1。

**1ファイル = 全ターゲット共通の配布単位**。中身:

    bundle_format   'magicdelta-bundle' 固定
    bundle_version  'MAJOR.MINOR'(下の互換ポリシー参照)
    ir              IR(JSON dict)。真実源
    preproc         入力符号化の宣言(kind/mean/std/thresholds/bit_order…)
    programs        conv2dノードid → 命令列化済みプログラム
    preproc_desc 整数化済みの前処理記述子(追加)
    labels          クラス名
    readout         出力の読み出し方(class_scores / spatial / detection…)
    resources       ゲート数・配信サイズ等の見積り
    eval            精度等の実測値
    about           由来(ckpt/レシピ/wandb run…)
    demo_samples    デモ兼e2e検証用の実サンプル

programs / preproc_desc は ir / preproc から決定的に再構築できる**冗長
データ**であり、「端末側から最適化ロジックを排除する」ためだけに存在する。
それゆえ書き出し前に必ずセルフチェックを通す(build_bundleのverify)。

互換ポリシー(ONNXのopset運用に倣う):
    - MAJOR不一致 → ローダは**明示的に拒否**する(意味論が変わったとみなす)
    - MINORが読み手より新しい → 警告のみ(追加は後方互換として扱う)
    - IR側は ir/schema.py の IR_VERSION を別軸で持つ。バンドルは
      SUPPORTED_IR_MAJOR_MINOR に載らないIRを警告する
"""

import gzip
import json
from pathlib import Path

from ..ir.schema import IR_FORMAT, IRGraph
from ..ir.serialize import ir_from_dict
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors
from .programs import build_programs

BUNDLE_FORMAT = 'magicdelta-bundle'
BUNDLE_VERSION = '1.0'

# このリーダが意味論を理解できるIRのMAJOR.MINOR。
SUPPORTED_IR_MAJOR_MINOR = ('0.2',)

# バンドル必須キー(欠けていたらエラー)。
REQUIRED_KEYS = ('bundle_format', 'bundle_version', 'ir')


def _version_pair(text: str) -> tuple[int, int]:
    parts = str(text).split('.')
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise ValueError(f'バージョン文字列が不正: {text!r}') from exc


def build_bundle(doc: dict, *, verify: bool = True) -> dict:
    """webbenchモデルdoc(または同形式のdict)を配布バンドルv1へ昇格する。

    元のdocは変更せず、新しいdictを返す。既存のキー(preproc/labels/
    readout/resources/eval/about/demo_samples…)はそのまま引き継ぐ。

    浅いコピーなので、返り値は入力docと入れ子構造(irのdict等)を**共有**
    する(実モデルのIRは数十MBになりうるためdeepcopyしない)。返り値を
    その場で書き換える用途では呼び出し側でコピーすること。

    Args:
        doc: 少なくとも 'ir' を持つdict
        verify: conv2d命令列の全数真理値表照合を行う(既定True)。
            Falseにしてよいのは故障注入テストのときだけ

    Raises:
        ValueError: irが無い/形式不正、または命令列の等価性検証に失敗
    """
    if 'ir' not in doc:
        raise ValueError("build_bundle: doc に 'ir' がない")
    ir_dict = doc['ir']
    if ir_dict.get('format') != IR_FORMAT:
        raise ValueError(f"build_bundle: irのformatが不正: {ir_dict.get('format')}")

    graph = ir_from_dict(ir_dict)
    programs = build_programs(graph, verify=verify)

    bundle = dict(doc)
    bundle['bundle_format'] = BUNDLE_FORMAT
    bundle['bundle_version'] = BUNDLE_VERSION
    bundle['programs'] = programs

    # 画像前処理は整数記述子へコンパイルして同梱する。画像でない
    # 入力(dense系のビット列など)は対象外なので黙って省く。
    preproc = doc.get('preproc') or {}
    if preproc.get('kind') in SUPPORTED_KINDS:
        plan = build_preproc_descriptors(preproc)
        _check_preproc_geometry(plan, graph)
        bundle['preproc_desc'] = plan
    return bundle


def _check_preproc_geometry(plan: dict, graph: IRGraph) -> None:
    """前処理の出力ビット数がIRの入力幅と一致することを確かめる。

    ここがずれたバンドルは端末側で必ず壊れる(そして症状は「精度が出ない」
    という分かりにくい形で出る)ので、生成時に落とす。
    """
    if not graph.inputs:
        return
    produced = plan['channels'] * plan['height'] * plan['width']
    expected = int(graph.inputs[0].width)
    if produced != expected:
        raise ValueError(
            f'前処理の出力ビット数 {produced} '
            f"(= {plan['channels']}ch × {plan['height']} × {plan['width']}) が "
            f'IRの入力幅 {expected} と一致しない'
        )


def validate_bundle(bundle: dict, *, strict: bool = False) -> list[str]:
    """バンドルの構造・バージョン互換を検査する。

    Args:
        bundle: 検査対象
        strict: Trueなら programs をIRから再構築して**完全一致**を確認する
            (改竄・古いprograms節の検出。コストは build_bundle と同等)

    Returns:
        警告メッセージのリスト(致命的でないもの)

    Raises:
        ValueError: 必須キー欠落、MAJOR不一致、programs の構造不整合、
            strict時の再構築不一致
    """
    warnings: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in bundle:
            raise ValueError(f'バンドルに必須キーがない: {key}')
    if bundle['bundle_format'] != BUNDLE_FORMAT:
        raise ValueError(
            f"bundle_formatが不正: {bundle['bundle_format']!r} "
            f'(期待 {BUNDLE_FORMAT!r})'
        )

    want_major, want_minor = _version_pair(BUNDLE_VERSION)
    got_major, got_minor = _version_pair(bundle['bundle_version'])
    if got_major != want_major:
        raise ValueError(
            f"bundle_version {bundle['bundle_version']} は非対応"
            f'(このリーダは {want_major}.x のみ)'
        )
    if got_minor > want_minor:
        warnings.append(
            f"bundle_version {bundle['bundle_version']} はこのリーダ"
            f'({BUNDLE_VERSION})より新しい: 未知のキーは無視される'
        )

    ir_dict = bundle['ir']
    ir_major_minor = '.'.join(str(ir_dict.get('version', '')).split('.')[:2])
    if ir_major_minor not in SUPPORTED_IR_MAJOR_MINOR:
        warnings.append(
            f"IR version {ir_dict.get('version')!r} は検証済みの"
            f'{SUPPORTED_IR_MAJOR_MINOR} に含まれない'
        )

    graph = ir_from_dict(ir_dict)
    _validate_recurrent_invariants(graph)
    conv_ids = [n.id for n in graph.nodes if n.type == 'conv2d']
    programs = bundle.get('programs')
    if programs is None:
        if conv_ids:
            raise ValueError(
                f'programs節が無いのにconv2dノードがある: {conv_ids}'
            )
    else:
        missing = [nid for nid in conv_ids if nid not in programs]
        if missing:
            raise ValueError(f'programs節に無いconv2dノード: {missing}')
        extra = [nid for nid in programs if nid not in conv_ids]
        if extra:
            raise ValueError(f'conv2dノードに対応しないprograms項目: {extra}')
        for nid in conv_ids:
            _validate_program(nid, programs[nid], graph.node_by_id(nid).params)

    preproc = bundle.get('preproc') or {}
    plan = bundle.get('preproc_desc')
    if plan is None:
        if preproc.get('kind') in SUPPORTED_KINDS:
            warnings.append(
                f"preproc.kind={preproc['kind']!r} は記述子化できるのに "
                'preproc_desc が無い(端末側で前処理を実行できない)'
            )
    else:
        if len(plan['desc']) != plan['channels'] * plan['stride']:
            raise ValueError(
                f"preproc_desc: desc長 {len(plan['desc'])} が "
                f"channels×stride={plan['channels'] * plan['stride']} と違う"
            )
        if len(plan['qval']) != 256:
            raise ValueError(f"preproc_desc: qval長 {len(plan['qval'])} が256でない")
        _check_preproc_geometry(plan, graph)
        if strict and preproc.get('kind') in SUPPORTED_KINDS:
            if build_preproc_descriptors(preproc) != plan:
                raise ValueError(
                    'preproc_desc が preproc から再構築した記述子と一致しない'
                )

    if strict and conv_ids:
        rebuilt = build_programs(graph, verify=True)
        for nid in conv_ids:
            if rebuilt[nid] != programs[nid]:
                raise ValueError(
                    f'programs[{nid}] がIRから再構築した命令列と一致しない'
                    '(IRとprograms節が食い違っている)'
                )

    _check_embed_table(bundle.get('embed_table'))
    _check_folded_table('folded_n_table', bundle.get('folded_n_table'))
    _check_folded_table('folded_l_table', bundle.get('folded_l_table'))

    return warnings


def _check_embed_table(embed_table: dict | None) -> None:
    """embed_table節(RDDLGN自己回帰生成用)の構造整合を検査する。

    export/rddlgn_ir.py::extract_embed_table が作る形
    ({'vocab', 'd_emb', 'bits'}、bitsはtoken行ごとにceil(d_emb/32)語
    LSB-firstパック)からの逸脱を検出する。無ければ何もしない
    (RDDLGN以外のバンドルへの影響ゼロ)。
    """
    if embed_table is None:
        return
    for key in ('vocab', 'd_emb', 'bits'):
        if key not in embed_table:
            raise ValueError(f'embed_tableに必須キーがない: {key}')
    vocab = int(embed_table['vocab'])
    d_emb = int(embed_table['d_emb'])
    if vocab <= 0 or d_emb <= 0:
        raise ValueError(f'embed_table: vocab={vocab}, d_emb={d_emb} は正でなければならない')
    wpr = (d_emb + 31) // 32
    want_len = vocab * wpr
    got_len = len(embed_table['bits'])
    if got_len != want_len:
        raise ValueError(
            f'embed_table: bits長 {got_len} が vocab*ceil(d_emb/32)='
            f'{want_len}(={vocab}*{wpr}) と違う'
        )
    for w in embed_table['bits']:
        if not (0 <= int(w) <= 0xffffffff):
            raise ValueError(f'embed_table: bitsの語 {w!r} がu32の範囲外')


def _check_folded_table(section_name: str, table: dict | None) -> None:
    """folded_n_table/folded_l_table節(語彙折り畳み)の構造整合を検査する。

    export/rddlgn_ir.py::extract_folded_table が作る形
    ({'vocab', 'width', 'bits'}、bitsはtoken行ごとにceil(width/32)語
    LSB-firstパック)からの逸脱を検出する。_check_embed_tableと同型
    (キー名が d_emb → width になっただけ)。無ければ何もしない
    (折り畳みを見送った/対象外のバンドルへの影響ゼロ)。
    """
    if table is None:
        return
    for key in ('vocab', 'width', 'bits'):
        if key not in table:
            raise ValueError(f'{section_name}に必須キーがない: {key}')
    vocab = int(table['vocab'])
    width = int(table['width'])
    if vocab <= 0 or width <= 0:
        raise ValueError(
            f'{section_name}: vocab={vocab}, width={width} は正でなければならない')
    wpr = (width + 31) // 32
    want_len = vocab * wpr
    got_len = len(table['bits'])
    if got_len != want_len:
        raise ValueError(
            f'{section_name}: bits長 {got_len} が vocab*ceil(width/32)='
            f'{want_len}(={vocab}*{wpr}) と違う'
        )
    for w in table['bits']:
        if not (0 <= int(w) <= 0xffffffff):
            raise ValueError(f'{section_name}: bitsの語 {w!r} がu32の範囲外')


def _validate_recurrent_invariants(graph: IRGraph) -> None:
    """recurrentノードの不変条件を検査する。

    steps>=1・scan_vars幅と入力エッジ幅の整合・state_updateの網羅・
    output_source_id実在・init.kind=='const' はすべて
    `ir.validate.validate` の recurrent 検証(_check_recurrent_params/
    _check_scan_var_params)が既に行っているので、それをそのまま使う
    (ロジックを2重化しない)。recurrentノードが1つも無いバンドルでは
    何もしない(既存バンドルへの影響ゼロ)。
    """
    if not any(n.type == 'recurrent' for n in graph.nodes):
        return
    from ..ir.validate import validate as ir_validate
    ir_validate(graph, target='generic')


def _validate_program(node_id: str, prog: dict, params: dict) -> None:
    """1つの conv2d プログラムの構造を検査する(参照範囲・長さ・整合)。"""
    def fail(msg: str) -> None:
        raise ValueError(f'programs[{node_id}]: {msg}')

    oc_count = int(params['out_channels'])
    # tapのIDは「受容野内フラット位置」= in_channels*kH*kW の範囲を取る
    # (indices_flatの行長は葉の数であって受容野の大きさではない)。
    in_channels = int(params['in_shape'][0])
    kh, kw = (int(v) for v in params['kernel_size'])
    field_size = in_channels * kh * kw

    n_instrs = int(prog['n_instrs'])
    if len(prog['instrs']) != 4 * n_instrs:
        fail(f"instrs長 {len(prog['instrs'])} が 4*n_instrs={4 * n_instrs} と違う")
    if len(prog['oc_out']) != oc_count:
        fail(f"oc_out長 {len(prog['oc_out'])} が out_channels={oc_count} と違う")
    if len(prog['write_oc']) != oc_count:
        fail(f"write_oc長 {len(prog['write_oc'])} が out_channels={oc_count} と違う")
    if len(prog['write_start']) != n_instrs + 2:
        fail(
            f"write_start長 {len(prog['write_start'])} が n_instrs+2="
            f'{n_instrs + 2} と違う'
        )

    n_taps = len(prog['taps'])
    if any(not (0 <= t < field_size) for t in prog['taps']):
        fail(f'tapが受容野({field_size})の外を指している')
    if len(set(prog['taps'])) != n_taps:
        fail('tapに重複がある(dedup済みでなければならない)')

    n_slots = int(prog['n_slots'])
    flat = prog['instrs']
    for j in range(n_instrs):
        tt, a, b, slot = flat[4 * j:4 * j + 4]
        if not (0 <= tt <= 0xf):
            fail(f'命令{j}: tt={tt} が0..15の外')
        if not (0 <= slot < max(n_slots, 1)):
            fail(f'命令{j}: slot={slot} が n_slots={n_slots} の外')
        for name, ref in (('a', a), ('b', b)):
            _validate_ref(fail, f'命令{j}.{name}', ref, n_taps, j)
    for oc, ref in enumerate(prog['oc_out']):
        _validate_ref(fail, f'oc_out[{oc}]', ref, n_taps, n_instrs)
    if sorted(prog['write_oc']) != list(range(oc_count)):
        fail('write_ocが全出力チャネルの順列になっていない')


def _validate_ref(fail, where: str, ref: int, n_taps: int, limit: int) -> None:
    """ref((idx<<2)|kind)の妥当性。命令参照は自分より前(トポ順)に限る。"""
    kind = ref & 3
    idx = ref >> 2
    if kind == 0:  # CONST
        if idx not in (0, 1):
            fail(f'{where}: 定数参照 idx={idx} が0/1でない')
    elif kind == 1:  # TAP
        if not (0 <= idx < n_taps):
            fail(f'{where}: tap参照 idx={idx} が tap数 {n_taps} の外')
    elif kind == 2:  # INSTR
        if not (0 <= idx < limit):
            fail(f'{where}: 命令参照 idx={idx} が前方参照でない(limit={limit})')
    else:
        fail(f'{where}: 未知のref kind={kind}')


def attach_bundle_sizes(bundle: dict) -> dict:
    """最終バンドルの配信サイズを resources に書き込む。

    resources['bundle_json_bytes'] / resources['bundle_gzip_bytes'] に、
    この bundle をそのまま save_bundle と同じ整形(空白なしJSON、
    ensure_ascii=False)で書き出したときの実バイト数と、その gzip(-6)
    サイズを記録する。demo_samples 等、書き出し対象に含めたいキーは
    呼び出し前に bundle へ入れておくこと(ここでは bundle をそのまま
    測るだけで、キーの取捨選択はしない)。

    サイズフィールド自身がJSON長に含まれる自己参照になっている
    (フィールドの桁数が変わるとJSON長も変わる)ため、不動点反復で解く:
    値を書き込んでから直列化して測り直し、測った値が書き込んだ値と
    食い違えば書き直してもう一度測る。桁数は測るたびにしか変わらない
    ので bundle_json_bytes は通常1〜2回で収束する。

    bundle_gzip_bytes は収束の基準にしない: gzip圧縮後のサイズは
    埋め込む数値の「桁数」だけでなく「具体的な数字そのもの」にも
    (ハフマン符号長の丸めで)敏感なため、59918→59919→59918→…の
    ように桁数が変わらないまま±1バイトで振動し、厳密な不動点が
    存在しない入力が実在する(実モデルで確認済み)。そのため
    bundle_json_bytes が2回連続で同じ値になった時点(=自分自身を
    含めて直列化しても長さが変わらない、という本来の自己無矛盾条件)
    で確定とし、bundle_gzip_bytes はその最後の実測値をそのまま使う
    (真の自己参照値と最大1〜2バイトずれ得るが、配信サイズの目安
    という用途では無視できる)。bundle_json_bytes 側は実ファイルの
    バイト数と厳密に一致する(呼び出し側のテストが検証する契約)。

    Args:
        bundle: build_bundle() が返す辞書。resourcesキーが無ければ作る

    Returns:
        bundle自身(インプレースで書き換えて返す)
    """
    resources = bundle.setdefault('resources', {})
    resources.setdefault('bundle_json_bytes', 0)
    resources.setdefault('bundle_gzip_bytes', 0)
    prev_json_bytes = None
    for _ in range(8):
        payload = json.dumps(
            bundle, separators=(',', ':'), ensure_ascii=False,
        ).encode('utf-8')
        json_bytes = len(payload)
        gzip_bytes = len(gzip.compress(payload, 6))
        resources['bundle_json_bytes'] = json_bytes
        resources['bundle_gzip_bytes'] = gzip_bytes
        if json_bytes == prev_json_bytes:
            return bundle
        prev_json_bytes = json_bytes
    raise AssertionError(
        'attach_bundle_sizes: bundle_json_bytes の不動点反復が収束しなかった'
    )


def save_bundle(bundle: dict, path, *, gzip_too: bool = False,
                format: str | None = None) -> Path:
    """バンドルを書き出す(既定はJSON、空白なしの配信用整形)。

    Args:
        gzip_too: Trueなら <path>.gz も書く(配信サイズの実測用)
        format: 'json' | 'mdlb'。省略(None)時は `path` の拡張子から
            自動判定する(`.mdlb` ならバイナリ形式、それ以外はJSON。
            既存呼び出し元は全て `.json`/`.bundle.json` パスを渡すので
            挙動は変わらない)。`.mdlb` バイナリ形式の仕様・往復契約は
            `deploy/mdlb.py` のモジュールdocstring参照

    Returns:
        書き出したファイルのパス
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = format or ('mdlb' if out.suffix == '.mdlb' else 'json')
    if fmt == 'mdlb':
        from .mdlb import write_mdlb
        write_mdlb(bundle, out)
        if gzip_too:
            gz = Path(str(out) + '.gz')
            gz.write_bytes(gzip.compress(out.read_bytes(), 6))
        return out
    if fmt != 'json':
        raise ValueError(f"save_bundle: 未知のformat: {fmt!r}('json'/'mdlb'のみ)")
    payload = json.dumps(bundle, separators=(',', ':'), ensure_ascii=False)
    out.write_text(payload, encoding='utf-8')
    if gzip_too:
        gz = Path(str(out) + '.gz')
        gz.write_bytes(gzip.compress(payload.encode('utf-8'), 6))
    return out


def load_bundle(path, *, validate: bool = True, strict: bool = False) -> dict:
    """バンドルを読み込む(既定でバージョン・構造を検証する)。

    拡張子で形式を判定する: `.mdlb`/`.mdlb.gz` はバイナリ形式
    (`deploy/mdlb.py::read_mdlb`)、それ以外(`.json`/`.json.gz`等)は
    従来通りJSON。呼び出し側(`md deploy info`/`codegen-*`/`verify-*`は
    いずれも本関数経由でバンドルを読むので、この1箇所を直すだけで
    全コマンドが `.mdlb` 入力に対応する。
    """
    p = Path(path)
    name_no_gz = p.name[:-3] if p.name.endswith('.gz') else p.name
    if name_no_gz.endswith('.mdlb'):
        from .mdlb import read_mdlb
        bundle = read_mdlb(p)
    else:
        raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
        bundle = json.loads(raw.decode('utf-8'))
    if validate:
        for warning in validate_bundle(bundle, strict=strict):
            print(f'警告: {warning}')
    return bundle
