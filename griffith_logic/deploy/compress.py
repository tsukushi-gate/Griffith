"""回路圧縮(lossless ゲート削減)の自動適用(で conv も既定に)。

`griffith deploy bundle --compress {none,fast,max}`(既定 fast)は、精度不変の
ゲート削減をモデルの種類によらず**同じ意味**でかける:

    none   IR を書き換える最適化をしない
    fast   数秒〜十数秒で終わる lossless 削減を全部かける(既定)
             dense(lut_layer 列) → sweep(定数伝播・CSE・未参照除去)
             conv2d 木           → resynth(lossless 再合成、探索上限 4 ゲート、
                                   木ごとの全数真理値表照合+前向き計算で検証)
    max    fast + 時間をかける探索・検証
             conv2d 木 → resynth(探索上限 5、ir/simulate による照合も行う)

どの level でも、conv2d 木の**命令列化**(バンドルの programs 節)は変換の一部
として必ず行われる(IR は変えない)。CLI はその削減も ``about.compress.always``
に記録し、他の削減と同じ形式で表示する。

個別フラグは互換のため残し、**指定されたら --compress より優先**する
(``--sweep`` / ``--resynth lossless`` は強制適用、``--no-sweep`` は sweep を抑止)。

fail-safe の方針:
    自動適用(フラグ未指定)の途中で変換・検証が失敗したら、**その圧縮を
    スキップしてバンドル化を続行**する(圧縮無しの IR は元から正しいので、
    最適化の失敗で出荷を止めない)。失敗は about.compress.skipped に理由つきで
    残し、警告を stderr に出す。明示フラグ(``--sweep`` 等)の失敗は従来どおり
    例外のまま(ユーザーが明示した操作は黙って握りつぶさない)。

``about.compress`` の形(``info`` が 1 行で表示する)::

    {'level': 'fast', 'applied': ['sweep'], 'skipped': [],
     'gates': {'sweep': [32, 22]}}

``skipped`` には「対象はあるのに適用できなかった」パスだけを残す(conv を含まない
IR に resynth が無い、のような**そもそも対象外**は記録しない)。
``--compress none`` で何も適用されなかったときは about に書かない
(既定でなかった頃のバンドルとバイト同一)。
"""

import sys

__all__ = ['COMPRESS_LEVELS', 'sweep_skip_reason', 'resynth_skip_reason',
           'apply_compress', 'attach_compress_info', 'describe_compress']

COMPRESS_LEVELS = ('none', 'fast', 'max')

#: level ごとの conv2d resynth の設定。fast は探索上限 4・軽量検証
#: (det_face で数秒、seg_k128 で約 3 秒。上限 5 との差は -6.4% 対 -6.7% 程度)、
#: max は上限 5 に ir/simulate による照合を足す(実モデルで分オーダー)。
RESYNTH_BY_LEVEL = {'fast': {'max_cost': 4, 'verify': 'fast'},
                    'max': {'max_cost': 5, 'verify': 'simulate'}}


def _node_types(doc: dict) -> set[str]:
    return {n.get('type') for n in (doc.get('ir') or {}).get('nodes', [])}


def sweep_skip_reason(doc: dict) -> str | None:
    """sweep(dense 専用)をこの doc に自動適用できない理由。適用できるなら None。"""
    from .sweepdoc import UNSUPPORTED_TYPES

    if 'ir' not in doc:
        return "doc に 'ir' がない"
    types = _node_types(doc)
    bad = sorted(t for t in types if t in UNSUPPORTED_TYPES)
    if bad:
        return f'sweep は dense(lut_layer列)専用で {", ".join(bad)} を含む'
    if 'lut_layer' not in types:
        return 'lut_layer が無い(sweep の対象なし)'
    return None


def resynth_skip_reason(doc: dict) -> str | None:
    """resynth(conv2d 木の再合成)の対象が無い理由。あるなら None。"""
    if 'ir' not in doc:
        return "doc に 'ir' がない"
    if 'conv2d' not in _node_types(doc):
        return 'conv2d を含まない(resynth は conv2d 木専用)'
    return None


def _warn(passname: str, exc: Exception) -> str:
    reason = f'{type(exc).__name__}: {exc}'
    print(f'警告: --compress の自動 {passname} が失敗したためスキップして続行'
          f'(圧縮なしのIRは元から正しい): {reason}', file=sys.stderr)
    return f'失敗したためスキップ({reason})'


def apply_compress(doc: dict, *, level: str = 'fast',
                   sweep: bool | None = None, resynth: str | bool | None = None,
                   sweep_verify_samples: int = 8,
                   resynth_max_cost: int | None = None,
                   resynth_verify_samples: int = 8
                   ) -> tuple[dict, dict, dict]:
    """doc に圧縮パス(resynth → sweep)を level と個別指定に従って適用する。

    適用順は ``resynth → sweep``(と同じ。dense-flat と conv の命令列化は
    呼び出し側がこの後に行い、結果を ``info`` に足す)。

    Args:
        level: ``'none'|'fast'|'max'``
        sweep: True=強制適用(非対応なら ValueError。失敗も例外)、
            False=抑止(``--no-sweep``)、None=level に従う
        resynth: ``'lossless'`` なら強制適用(``ir/simulate`` 照合つき、失敗は
            例外)。False なら抑止(呼び出し側で既に再合成済みのとき)。None なら
            level ``fast``/``max`` で conv2d を含む IR に自動適用
            (設定は ``RESYNTH_BY_LEVEL``)
        resynth_max_cost: 再合成の探索ゲート数上限。None なら level の既定
            (強制適用時は 5)

    Returns:
        (new_doc, info, summaries)。info は ``about.compress`` 用の
        ``{'level', 'applied', 'skipped', 'gates'}``(gates はパスごとの
        ``[before, after]``)。summaries は ``{'resynth': ..., 'sweep': ...}``
        (適用したパスの要約。ログ用)。new_doc の ``about.compress`` へは
        書かない(呼び出し側が dense-flat 等を足してから
        `attach_compress_info` で書く)。

    Raises:
        ValueError: level 不正、強制指定のパスが失敗
    """
    from .resynth import resynth_doc_lossless
    from .sweepdoc import sweep_doc_lossless

    if level not in COMPRESS_LEVELS:
        raise ValueError(
            f'compress={level!r} は不正({"/".join(COMPRESS_LEVELS)} のみ)')
    applied: list[str] = []
    skipped: list[dict] = []
    gates: dict[str, list[int]] = {}
    summaries: dict = {}

    def _run_resynth(max_cost: int, verify: str) -> None:
        nonlocal doc
        doc, st = resynth_doc_lossless(
            doc, max_cost=max_cost, verify_samples=resynth_verify_samples,
            verify=verify)
        summaries['resynth'] = st
        applied.append('resynth')
        gates['resynth'] = [int(st['gates_before']), int(st['gates_after'])]

    # ---- resynth(conv2d 木) ----
    if resynth == 'lossless':
        _run_resynth(resynth_max_cost or 5, 'simulate')
    elif resynth not in (None, False):
        raise ValueError(f'resynth={resynth!r} は不正(lossless のみ)')
    elif resynth is None and level in RESYNTH_BY_LEVEL \
            and resynth_skip_reason(doc) is None:
        cfg = RESYNTH_BY_LEVEL[level]
        try:
            _run_resynth(resynth_max_cost or cfg['max_cost'], cfg['verify'])
        except Exception as exc:               # noqa: BLE001
            skipped.append({'pass': 'resynth', 'reason': _warn('resynth', exc)})

    # ---- sweep(dense) ----
    if sweep is True:
        doc, summaries['sweep'] = sweep_doc_lossless(
            doc, verify_samples=sweep_verify_samples)
        applied.append('sweep')
    elif sweep is False:
        if level != 'none' and 'lut_layer' in _node_types(doc):
            skipped.append({'pass': 'sweep', 'reason': '--no-sweep 指定'})
    elif level in ('fast', 'max') and 'lut_layer' in _node_types(doc):
        why = sweep_skip_reason(doc)
        if why is None:
            try:
                doc, summaries['sweep'] = sweep_doc_lossless(
                    doc, verify_samples=sweep_verify_samples)
                applied.append('sweep')
            except Exception as exc:               # noqa: BLE001
                skipped.append({'pass': 'sweep', 'reason': _warn('sweep', exc)})
        else:
            # lut_layer はあるのに他のノード(conv2d 等)のせいで掛けられない
            skipped.append({'pass': 'sweep', 'reason': why})
    if 'sweep' in summaries:
        st = summaries['sweep']
        gates['sweep'] = [int(st['gates_before']), int(st['gates_after'])]

    info = {'level': level, 'applied': applied, 'skipped': skipped}
    if gates:
        info['gates'] = gates
    return doc, info, summaries


def attach_compress_info(doc: dict, info: dict) -> dict:
    """info を ``doc['about']['compress']`` に書いた浅いコピーを返す。

    ``--compress none`` で何も適用も記録もされなかったときは doc をそのまま
    返す(圧縮導入前と同一バイトの出力を保つ)。
    """
    if info['level'] == 'none' and not info['applied'] and not info['skipped']:
        return doc
    new_doc = dict(doc)
    about = dict(doc.get('about') or {})
    about['compress'] = info
    new_doc['about'] = about
    return new_doc


def _pct(before: int, after: int) -> str:
    return f'-{100.0 * (before - after) / before:.1f}%' if before else '-0.0%'


#: 表示名(パス名 → 人が読む名前)
_PASS_LABELS = {'sweep': 'sweep', 'resynth': 'conv再合成', 'dense-flat': 'dense平坦化',
                'conv-program': 'conv命令列化'}


def describe_compress(info: dict) -> str:
    """``about.compress`` を 1 行にまとめる(bundle / info の標準出力用)。

    適用したパスはゲート数の前後つきで、dense と conv を同じ形式で並べる。
    conv は命令列化(常時)の全体の削減を先に出し、そのうち再合成の分を続ける
    (例 ``適用 conv命令列化 14,784→1,077(-92.7%、常時)、うち conv再合成
    1,279→1,077(-15.8%)``。命令列化前の 14,784 は木の名目ゲート数、1,279 は
    自明ゲートを畳んだ後の実効ゲート数)。
    """
    gates = info.get('gates') or {}
    always = info.get('always') or []
    has_program = any(item['pass'] == 'conv-program' for item in always)
    parts = []
    for item in always:
        b, a = item['gates']
        parts.append(f"{_PASS_LABELS.get(item['pass'], item['pass'])} "
                     f'{b:,}→{a:,}({_pct(b, a)}、常時)')
    for name in info.get('applied') or []:
        label = _PASS_LABELS.get(name, name)
        if name == 'resynth' and has_program:
            label = 'うち' + label
        if name in gates:
            b, a = gates[name]
            parts.append(f'{label} {b:,}→{a:,}({_pct(b, a)})')
        else:
            parts.append(label)
    text = f"圧縮(compress={info['level']}): 適用 {'、'.join(parts) or 'なし'}"
    skipped = info.get('skipped') or []
    if skipped:
        text += ' / スキップ ' + '; '.join(
            f"{s['pass']}({s['reason']})" for s in skipped)
    return text
