"""配布バンドルのバイナリ形式 `.mdlb`。

`docs/reference/deploy-bundle.md` の「`.mdlb` バイナリ形式(第2弾、未実装)」
節に凍結された仕様の実装。凍結仕様が明記するのは:

    - magic `MDLB` + u32 version、リトルエンディアン、8バイトアライン
    - セクション TLV: IR_META / TT / INDICES / PROGRAM / PREPROC_DESC /
      LABELS / READOUT
    - 全オフセットは相対。パース不要で mmap して即実行できること
    - Python 側で JSON ⇔ バイナリ双方向変換 + roundtrip テスト

以下は仕様が具体化していない点についての**このモジュールの解釈**(決定、
ここに明記する):

1. **セクションの対応**: IR中のノードパラメータで実際に使われているキー名
   (`griffith_logic/ir/*.py` 参照)をそのまま section の中身に採る。
       TT       = 各IRノードの `params['truth_tables']`(hex文字列list。
                  conv2dは2入力木の内部ノード数=out_channels*(2^depth-1)、
                  lut_layer/dwn_layerはN入力LUTなので最大64bit値になりうる)
       INDICES  = 各IRノードの `params['indices_flat']` / `params['indices']`
                  (受容野内フラット位置。ネストしたint list、形状は
                  ノード種別に依存)
       PROGRAM  = バンドルの `programs` 節(conv2d命令列化済み)。
                  実測(動機付け)でgzip後+62.6%(24モデル合計)の主犯
       PREPROC_DESC = バンドルの `preproc_desc` 節
       LABELS   = バンドルの `labels`
       READOUT  = バンドルの `readout`
       IR_META  = `ir` のうち TT/INDICES に抜き出した分を除いた残り
                  (format/version/metadata/inputs/outputs/edges、
                  各ノードのtruth_tables/indices_flat/indices以外のparams)
   TT/INDICES の対象は「その場でhex文字列list / 矩形int-nested-list という
   期待形状に一致するときだけ」抜き出す。一致しない(将来ノード種別が
   別の形で持つ、等)場合は**黙って諦めてIR_META側の一般エンコードに
   残す**(値を壊すより保守的に倒す)。

2. **spec未記載の残りキー**: 仕様策定時点には存在しなかった
   `bundle_format` / `bundle_version` / `preproc` / `resources` / `eval` /
   `about` / `demo_samples` / `embed_table` / `folded_n_table` /
   `folded_l_table`(現在で追加)を含め、上記6節に属さない**トップ
   レベルキー全て**をひとつの `EXTRA` セクションへ汎用エンコードで
   まとめる。この節名は凍結仕様には無い、追加。

3. **「dictが完全一致」の定義**(ラウンドトリップ契約): JSONは
   タプル/int辞書キー/numpy型等を表現できない(`docs/reference/
   deploy-bundle.md` 「JSON往復についての注意」参照、バンドルの正準形は
   「JSONデコード後の形」と明記されている)。本モジュールもこれに従い、

       read_mdlb(write_mdlb(bundle, path))
           == json.loads(json.dumps(bundle, ensure_ascii=False))

   を契約とする(`write_mdlb` は内部で書き込み前にこの正準化を1回行う)。
   一致判定は「dict は key の集合と各値が一致すれば順序は問わない、
   list は順序を含めて一致、int/float/bool/str/None は型ごと区別する
   (Pythonの `1 == 1.0 == True` に頼った緩い一致にしない)」という
   厳密規約で行う(テスト側 `tests/deploy/test_mdlb.py::_strict_equal`)。

4. **故障注入への耐性**: 凍結仕様は明記していないが、TLVコンテナとして
   自然な最小限の安全策(magic不一致・version不一致・セクション境界が
   ファイル範囲外・セクションCRC32不一致を例外にする)を入れる。

5. **「varint/delta符号化等、仕様が許す範囲で」(動機)の具体化**:
   仕様の「フラットバイナリ」という指示に対し、以下を実装した(いずれも
   実測に基づく、詳細は各関数のコメント参照):
       - PROGRAM節: instrsをAoS→SoA転置してから列ごとに符号化。
         tt/slot列は0..15ならニブル詰め(倍密度)。a/b列(命令オペランド
         参照)は、参照が命令(kind=INSTR)を指すとき「現在の命令位置から
         の距離」へ変換してから符号化する(実測: 命令列は直前に作った
         中間値を参照する局所性が強く、距離の中央値は3。生idx比2.2倍)。
         write_start(CSR)は差分符号化
       - 汎用値エンコードの文字列: 正準base64文字列(decode→再encodeで
         完全一致するもの)は生バイト列として埋め込む(demo_samples の
         rgb_b64 等。4/3の符号化オーバーヘッドを避ける。1文字でも
         非canonicalなら通常の文字列としてフォールバックする——
         「圧縮できるかも」より「絶対に壊さない」を優先)
   実測(det_multimnist_lut_k256、最大の実モデル): bundle.json.gz比
   1.54倍(784,630B→510,789B)。他モデルはdemo_samples(base64生画像、
   本質的に非圧縮限界に近い)比率が高いほど伸びが小さい
   (`results/deploy/mdlb_size_k58.json` 参照、2倍目標は未達)。

コンテナのバイト配置:

    offset 0   magic b'MDLB'(4B)
    offset 4   version u32 LE(=1)
    offset 8   n_sections u32 LE
    offset 12  reserved u32 LE(=0)
    offset 16  セクションディレクトリ(n_sections × 32B):
                   tag(4B ASCII) + reserved(4B) + rel_offset(u64 LE)
                   + length(u64 LE) + crc32(u32 LE) + reserved2(4B)
    offset 16+32*n_sections  ペイロード領域の先頭(8バイトアライン済み。
                   16 も 32 も8の倍数なのでここは常にアライン済み)。
                   各セクションのペイロードは `rel_offset`(ここからの
                   相対バイト数、8バイトアライン)から `length` バイト

パース不要で mmap 即実行、を汎用Python実装で厳密に体現するのは(タプル状の
命令列を直接C構造体としてmmapするような)ターゲット側ランタイムの仕事であり、
このモジュールはその前段の「JSON⇔バイナリの可逆圧縮版コンテナ」に留める。
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import struct
import zlib
from pathlib import Path

__all__ = [
    'MDLB_MAGIC', 'MDLB_VERSION',
    'write_mdlb', 'read_mdlb',
    'encode_bundle_to_mdlb', 'decode_mdlb_bytes',
]

MDLB_MAGIC = b'MDLB'
MDLB_VERSION = 1

_HEADER_SIZE = 16
_DIR_ENTRY_SIZE = 32
_SECTION_TAGS = (b'IRMT', b'TTBL', b'IDXS', b'PROG', b'PPDS', b'LABL',
                 b'RDOT', b'EXTR')

_HEX_RE = re.compile(r'^0[xX][0-9a-fA-F]+$')


# ---------------------------------------------------------------------------
# varint(unsigned LEB128)+ zigzag(符号あり)
# ---------------------------------------------------------------------------

def _write_uvarint(buf: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError(f'_write_uvarint: 負の値は書けない: {value}')
    while True:
        b = value & 0x7f
        value >>= 7
        if value:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError('_read_uvarint: バッファ終端(データが切り詰められている)')
        b = data[pos]
        pos += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _zigzag_encode(n: int) -> int:
    return (n << 1) if n >= 0 else (((-n) << 1) - 1)


def _zigzag_decode(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)


def _write_svarint(buf: bytearray, value: int) -> None:
    _write_uvarint(buf, _zigzag_encode(value))


def _read_svarint(data: bytes, pos: int) -> tuple[int, int]:
    z, pos = _read_uvarint(data, pos)
    return _zigzag_decode(z), pos


def _write_uarray(buf: bytearray, values) -> None:
    for v in values:
        _write_uvarint(buf, v)


def _read_uarray(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    out = []
    for _ in range(n):
        v, pos = _read_uvarint(data, pos)
        out.append(v)
    return out, pos


def _write_sarray(buf: bytearray, values) -> None:
    for v in values:
        _write_svarint(buf, v)


def _read_sarray(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    out = []
    for _ in range(n):
        v, pos = _read_svarint(data, pos)
        out.append(v)
    return out, pos


def _write_delta_sarray(buf: bytearray, values) -> None:
    """単調でなくても正しいが、書き出しスケジュールのように単調非減少なら
    差分がほぼ0近辺に収まり圧縮が効く(CSR用、狙いそのもの)。"""
    prev = 0
    for v in values:
        _write_svarint(buf, v - prev)
        prev = v


def _read_delta_sarray(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    out = []
    prev = 0
    for _ in range(n):
        d, pos = _read_svarint(data, pos)
        prev += d
        out.append(prev)
    return out, pos


# ---------------------------------------------------------------------------
# 文字列 + 汎用JSON値エンコード(IR_META/EXTRA/LABELS/READOUT/statsが使う)
# ---------------------------------------------------------------------------

_T_NULL = 0
_T_FALSE = 1
_T_TRUE = 2
_T_INT = 3
_T_FLOAT = 4
_T_STR = 5
_T_LIST = 6
_T_DICT = 7
_T_B64BLOB = 8

#: `demo_samples[*].rgb_b64` 等、base64文字列として埋め込まれた生バイト列を
#: 検出するしきい値・文字集合(追加。凍結仕様には無い)。デモ画像は
#: base64テキストのままだと(a) 4/3の符号化オーバーヘッド (b) 3byte単位の
#: シフトでgzipの後方参照が壊れる、の二重にサイズで損をする。1件あたり
#: タグ1byte+長さvarintの固定費があるので、短い文字列(閾値未満)は
#: 素の文字列のまま扱う。
_B64_MIN_LEN = 64
_B64_RE = re.compile(r'[A-Za-z0-9+/]*={0,2}')


def _write_str(buf: bytearray, s: str) -> None:
    b = s.encode('utf-8')
    _write_uvarint(buf, len(b))
    buf.extend(b)


def _read_str(data: bytes, pos: int) -> tuple[str, int]:
    n, pos = _read_uvarint(data, pos)
    s = data[pos:pos + n].decode('utf-8')
    return s, pos + n


def _try_encode_base64_blob(s: str) -> bytes | None:
    """sが正準なbase64文字列(decode→再encodeでbyte完全一致)なら生バイトを
    返す。1文字でも違えば(非canonical padding等)Noneを返してフォールバック
    させる — 「圧縮できるかも」より「絶対に壊さない」を優先する。"""
    n = len(s)
    if n < _B64_MIN_LEN or n % 4 != 0 or not _B64_RE.fullmatch(s):
        return None
    try:
        raw = base64.b64decode(s, validate=True)
    except (ValueError, TypeError):
        return None
    if base64.b64encode(raw).decode('ascii') != s:
        return None
    return raw


def _write_value(buf: bytearray, v) -> None:
    if v is None:
        buf.append(_T_NULL)
    elif isinstance(v, bool):
        buf.append(_T_TRUE if v else _T_FALSE)
    elif isinstance(v, int):
        buf.append(_T_INT)
        _write_svarint(buf, v)
    elif isinstance(v, float):
        buf.append(_T_FLOAT)
        buf.extend(struct.pack('<d', v))
    elif isinstance(v, str):
        raw = _try_encode_base64_blob(v)
        if raw is not None:
            buf.append(_T_B64BLOB)
            _write_uvarint(buf, len(raw))
            buf.extend(raw)
        else:
            buf.append(_T_STR)
            _write_str(buf, v)
    elif isinstance(v, list):
        buf.append(_T_LIST)
        _write_uvarint(buf, len(v))
        for item in v:
            _write_value(buf, item)
    elif isinstance(v, dict):
        buf.append(_T_DICT)
        _write_uvarint(buf, len(v))
        for k, val in v.items():
            if not isinstance(k, str):
                raise TypeError(
                    f'_write_value: dictキーが文字列でない({k!r}): '
                    'write_mdlbはjson正準化後のdictを前提とする')
            _write_str(buf, k)
            _write_value(buf, val)
    else:
        raise TypeError(
            f'_write_value: 未対応の型 {type(v)}: {v!r}(JSON安全な値のみ対応)')


def _read_value(data: bytes, pos: int):
    tag = data[pos]
    pos += 1
    if tag == _T_NULL:
        return None, pos
    if tag == _T_FALSE:
        return False, pos
    if tag == _T_TRUE:
        return True, pos
    if tag == _T_INT:
        return _read_svarint(data, pos)
    if tag == _T_FLOAT:
        (val,) = struct.unpack_from('<d', data, pos)
        return val, pos + 8
    if tag == _T_STR:
        return _read_str(data, pos)
    if tag == _T_B64BLOB:
        n, pos = _read_uvarint(data, pos)
        raw = data[pos:pos + n]
        return base64.b64encode(raw).decode('ascii'), pos + n
    if tag == _T_LIST:
        n, pos = _read_uvarint(data, pos)
        out = []
        for _ in range(n):
            item, pos = _read_value(data, pos)
            out.append(item)
        return out, pos
    if tag == _T_DICT:
        n, pos = _read_uvarint(data, pos)
        out = {}
        for _ in range(n):
            k, pos = _read_str(data, pos)
            val, pos = _read_value(data, pos)
            out[k] = val
        return out, pos
    raise ValueError(f'_read_value: 未知の型タグ {tag}(データが壊れている)')


# ---------------------------------------------------------------------------
# ネストしたint配列(indices_flat等)の shape 保存付きフラット化
# ---------------------------------------------------------------------------

def _looks_like_hex_list(value) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(x, str) and _HEX_RE.match(x) for x in value)


def _looks_like_int_nest(value) -> bool:
    if not isinstance(value, list):
        return False

    def check(v):
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            return True
        if isinstance(v, list):
            return all(check(x) for x in v)
        return False
    return check(value)


def _flatten_nested_ints(value) -> tuple[list[int], list[int]]:
    """矩形(全軸で長さ一定)のint-nested-listを (flat, shape) へ。"""
    shape: list[int] = []
    node = value
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None

    flat: list[int] = []

    def rec(v, depth):
        if depth == len(shape):
            if isinstance(v, list):
                raise ValueError('_flatten_nested_ints: 矩形でない(深さが揃わない)')
            flat.append(int(v))
            return
        if not isinstance(v, list) or len(v) != shape[depth]:
            raise ValueError('_flatten_nested_ints: 矩形でない(軸の長さが揃わない)')
        for item in v:
            rec(item, depth + 1)

    rec(value, 0)
    return flat, shape


def _unflatten_nested_ints(flat: list[int], shape: list[int]):
    it = iter(flat)

    def build(depth):
        if depth == len(shape):
            return next(it)
        return [build(depth + 1) for _ in range(shape[depth])]

    return build(0)


# ---------------------------------------------------------------------------
# IR: truth_tables / indices_flat・indices の抜き出しと再合流
# ---------------------------------------------------------------------------

def _split_ir_for_mdlb(ir_dict: dict):
    """ir dict → (meta_ir, tt_entries, idx_entries)。

    tt_entries:  [(node_id, [hex文字列, ...]), ...]
    idx_entries: [(node_id, key_name, flat:list[int], shape:list[int]), ...]
    抜き出した分は meta_ir 側の該当ノードparamsから削除する
    (浅いコピーのみ、元の ir_dict は変更しない)。
    """
    nodes = ir_dict.get('nodes') or []
    tt_entries: list[tuple[str, list[str]]] = []
    idx_entries: list[tuple[str, str, list[int], list[int]]] = []
    new_nodes = []
    for node in nodes:
        params = node.get('params') if isinstance(node, dict) else None
        node_id = node.get('id') if isinstance(node, dict) else None
        if not isinstance(params, dict) or not isinstance(node_id, str) or not node_id:
            new_nodes.append(node)
            continue

        new_params = dict(params)
        changed = False

        tt = params.get('truth_tables')
        if 'truth_tables' in params and _looks_like_hex_list(tt):
            tt_entries.append((node_id, tt))
            del new_params['truth_tables']
            changed = True

        for key in ('indices_flat', 'indices'):
            if key in params and _looks_like_int_nest(params[key]):
                try:
                    flat, shape = _flatten_nested_ints(params[key])
                except ValueError:
                    continue
                idx_entries.append((node_id, key, flat, shape))
                del new_params[key]
                changed = True
                break

        if changed:
            new_node = dict(node)
            new_node['params'] = new_params
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    meta_ir = dict(ir_dict)
    meta_ir['nodes'] = new_nodes
    return meta_ir, tt_entries, idx_entries


def _merge_ir_from_mdlb(meta_ir: dict, tt_entries, idx_entries) -> dict:
    ir_dict = dict(meta_ir)
    nodes = [dict(n) if isinstance(n, dict) else n for n in (ir_dict.get('nodes') or [])]
    by_id = {n.get('id'): i for i, n in enumerate(nodes) if isinstance(n, dict)}

    for node_id, hexlist in tt_entries:
        i = by_id[node_id]
        params = dict(nodes[i].get('params') or {})
        params['truth_tables'] = hexlist
        nodes[i] = {**nodes[i], 'params': params}

    for node_id, key, flat, shape in idx_entries:
        i = by_id[node_id]
        params = dict(nodes[i].get('params') or {})
        params[key] = _unflatten_nested_ints(flat, shape)
        nodes[i] = {**nodes[i], 'params': params}

    ir_dict['nodes'] = nodes
    return ir_dict


# ---------------------------------------------------------------------------
# TTセクション: hex文字列truth_tableの圧縮エンコード
# ---------------------------------------------------------------------------

def _write_nibbles(buf: bytearray, values: list[int]) -> None:
    """0..15の値をニブル詰め(2値/byte、偶数indexが下位ニブル)で書く。"""
    n = len(values)
    packed = bytearray((n + 1) // 2)
    for i, v in enumerate(values):
        if i % 2 == 0:
            packed[i // 2] |= (v & 0xf)
        else:
            packed[i // 2] |= (v & 0xf) << 4
    buf.extend(packed)


def _read_nibbles(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    nbytes = (n + 1) // 2
    packed = data[pos:pos + nbytes]
    pos += nbytes
    values = []
    for i in range(n):
        byte = packed[i // 2]
        nib = (byte & 0xf) if i % 2 == 0 else ((byte >> 4) & 0xf)
        values.append(nib)
    return values, pos


def _write_nibble_or_uarray(buf: bytearray, values: list[int]) -> None:
    """全値が0..15ならニブル詰め(mode=0、倍密度)、そうでなければvarint配列
    (mode=1)。conv2d(2入力木)のtt(常に4bit)向けだが、それ以外のLUT
    (N<=6→最大64bit)も安全に扱えるようフォールバックを持つ。"""
    n = len(values)
    mode = 0 if (n == 0 or max(values) <= 0xf) else 1
    buf.append(mode)
    if mode == 0:
        _write_nibbles(buf, values)
    else:
        _write_uarray(buf, values)


def _read_nibble_or_uarray(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    mode = data[pos]
    pos += 1
    if mode == 0:
        return _read_nibbles(data, pos, n)
    return _read_uarray(data, pos, n)


def _encode_tt_entry(buf: bytearray, node_id: str, hexlist: list[str]) -> None:
    _write_str(buf, node_id)
    values = [int(x, 16) for x in hexlist]
    n = len(values)
    _write_uvarint(buf, n)
    _write_nibble_or_uarray(buf, values)


def _decode_tt_entry(data: bytes, pos: int) -> tuple[str, list[str], int]:
    node_id, pos = _read_str(data, pos)
    n, pos = _read_uvarint(data, pos)
    values, pos = _read_nibble_or_uarray(data, pos, n)
    hexlist = [hex(v) for v in values]
    return node_id, hexlist, pos


# ---------------------------------------------------------------------------
# ref配列(命令オペランドa/b): (idx<<2)|kind の位置相対delta圧縮
# ---------------------------------------------------------------------------
#
# 実測(det_multimnist_lut_k256): kind=INSTR(2)の参照は現在の命令
# 位置jからの距離(中央値3、平均3.6)がidx自体(最大n_instrs、数千)より
# 桁違いに小さい(命令列は直前に作った中間値を参照する局所性が強い —
# CSE/dead除去後の木を辿る最適化パスの構造上そうなる)。CONST/TAP参照は
# jと無関係なのでidxのまま(varintは既に十分小さい: CONSTは0/1、TAPは
# taps数程度)。実測(INSTR参照58,029件、a/b列合計): 生idxの合計バイト数
# 128,571B → 距離delta化で58,031B(2.2倍)。
def _write_ref_array(buf: bytearray, refs: list[int]) -> None:
    for j, ref in enumerate(refs):
        kind = ref & 3
        idx = ref >> 2
        payload = (j - idx) if kind == 2 else idx
        _write_uvarint(buf, (payload << 2) | kind)


def _read_ref_array(data: bytes, pos: int, n: int) -> tuple[list[int], int]:
    out = []
    for j in range(n):
        code, pos = _read_uvarint(data, pos)
        kind = code & 3
        payload = code >> 2
        idx = (j - payload) if kind == 2 else payload
        out.append((idx << 2) | kind)
    return out, pos


# ---------------------------------------------------------------------------
# PROGRAMセクション: bundle['programs'] のSoA + delta圧縮エンコード
# ---------------------------------------------------------------------------

_PROGRAM_KEYS = ('taps', 'n_instrs', 'instrs', 'n_slots', 'oc_out',
                 'write_start', 'write_oc', 'stats')


def _encode_program(buf: bytearray, node_id: str, prog: dict) -> None:
    extra = set(prog.keys()) - set(_PROGRAM_KEYS)
    missing = set(_PROGRAM_KEYS) - set(prog.keys())
    if extra or missing:
        raise ValueError(
            f'encode_bundle_to_mdlb: programs[{node_id!r}] のキー集合が'
            f'program_to_dict の形と違う(不足={sorted(missing)}, '
            f'余分={sorted(extra)})')

    _write_str(buf, node_id)

    taps = [int(v) for v in prog['taps']]
    _write_uvarint(buf, len(taps))
    _write_uarray(buf, taps)

    n_instrs = int(prog['n_instrs'])
    flat = list(prog['instrs'])
    if len(flat) != 4 * n_instrs:
        raise ValueError(
            f'encode_bundle_to_mdlb: programs[{node_id!r}].instrs長 '
            f'{len(flat)} が 4*n_instrs={4 * n_instrs} と矛盾')
    _write_uvarint(buf, n_instrs)
    # AoS(tt,a,b,slot繰り返し)→SoA(列ごとに連続)へ転置してから書く。
    # 同種の値が連続する方がvarintの分布が揃い、後段gzipの繰り返し検出にも効く。
    _write_nibble_or_uarray(buf, [int(v) for v in flat[0::4]])  # tt   (常に0..15)
    _write_ref_array(buf, [int(v) for v in flat[1::4]])         # a    (ref)
    _write_ref_array(buf, [int(v) for v in flat[2::4]])         # b    (ref)
    # slot(同時生存する作業平面のインデックス)は実測でn_slots<=数十が
    # 大半(実測: 実モデル6ノードとも max<=9)。nibble_or_uarrayなら
    # 0..15に収まる限り倍密度、収まらないモデルでも安全にvarintへ落ちる。
    _write_nibble_or_uarray(buf, [int(v) for v in flat[3::4]])  # slot (>=0)

    _write_uvarint(buf, int(prog['n_slots']))

    oc_out = [int(v) for v in prog['oc_out']]
    _write_uvarint(buf, len(oc_out))
    _write_uarray(buf, oc_out)

    write_start = [int(v) for v in prog['write_start']]
    _write_uvarint(buf, len(write_start))
    # write_startはCSRで単調非減少 → 差分符号化(spec動機のvarint/delta狙いそのもの)。
    _write_delta_sarray(buf, write_start)

    write_oc = [int(v) for v in prog['write_oc']]
    _write_uvarint(buf, len(write_oc))
    _write_uarray(buf, write_oc)

    _write_value(buf, prog['stats'])


def _decode_program(data: bytes, pos: int) -> tuple[str, dict, int]:
    node_id, pos = _read_str(data, pos)

    n_taps, pos = _read_uvarint(data, pos)
    taps, pos = _read_uarray(data, pos, n_taps)

    n_instrs, pos = _read_uvarint(data, pos)
    tts, pos = _read_nibble_or_uarray(data, pos, n_instrs)
    aas, pos = _read_ref_array(data, pos, n_instrs)
    bbs, pos = _read_ref_array(data, pos, n_instrs)
    slots, pos = _read_nibble_or_uarray(data, pos, n_instrs)
    instrs: list[int] = []
    for i in range(n_instrs):
        instrs.extend((tts[i], aas[i], bbs[i], slots[i]))

    n_slots, pos = _read_uvarint(data, pos)

    n_oc, pos = _read_uvarint(data, pos)
    oc_out, pos = _read_uarray(data, pos, n_oc)

    n_ws, pos = _read_uvarint(data, pos)
    write_start, pos = _read_delta_sarray(data, pos, n_ws)

    n_woc, pos = _read_uvarint(data, pos)
    write_oc, pos = _read_uarray(data, pos, n_woc)

    stats, pos = _read_value(data, pos)

    prog = {
        'taps': taps, 'n_instrs': n_instrs, 'instrs': instrs,
        'n_slots': n_slots, 'oc_out': oc_out, 'write_start': write_start,
        'write_oc': write_oc, 'stats': stats,
    }
    return node_id, prog, pos


# ---------------------------------------------------------------------------
# PREPROC_DESCセクション
# ---------------------------------------------------------------------------

_PREPROC_DESC_REQUIRED = ('desc', 'qval', 'channels', 'image_channels',
                          'height', 'width', 'stride')
_PREPROC_DESC_SCALAR_KEYS = ('channels', 'image_channels', 'height', 'width',
                             'stride')


def _encode_preproc_desc(buf: bytearray, plan: dict) -> None:
    missing = [k for k in _PREPROC_DESC_REQUIRED if k not in plan]
    if missing:
        raise ValueError(
            f'encode_bundle_to_mdlb: preproc_descに必須キーがない: {missing}'
            '(preproc_desc.py::build_preproc_descriptors の形と違う)')

    desc = [int(v) for v in plan['desc']]
    _write_uvarint(buf, len(desc))
    _write_sarray(buf, desc)  # dy/dx等の符号付きワードを含むためzigzag

    qval = [int(v) for v in plan['qval']]
    _write_uvarint(buf, len(qval))
    _write_sarray(buf, qval)

    for key in _PREPROC_DESC_SCALAR_KEYS:
        _write_svarint(buf, int(plan[key]))

    known = set(_PREPROC_DESC_SCALAR_KEYS) | {'desc', 'qval'}
    extra = {k: v for k, v in plan.items() if k not in known}
    _write_value(buf, extra)  # counts等。将来の追加キーも汎用側で吸収する


def _decode_preproc_desc(data: bytes, pos: int) -> tuple[dict, int]:
    n_desc, pos = _read_uvarint(data, pos)
    desc, pos = _read_sarray(data, pos, n_desc)

    n_qval, pos = _read_uvarint(data, pos)
    qval, pos = _read_sarray(data, pos, n_qval)

    scalars = {}
    for key in _PREPROC_DESC_SCALAR_KEYS:
        v, pos = _read_svarint(data, pos)
        scalars[key] = v

    extra, pos = _read_value(data, pos)

    plan = {'desc': desc, 'qval': qval, **scalars, **extra}
    return plan, pos


# ---------------------------------------------------------------------------
# コンテナの組み立て/分解
# ---------------------------------------------------------------------------

def _assemble_container(sections: dict[bytes, bytes]) -> bytes:
    n = len(_SECTION_TAGS)
    header = struct.pack('<4sIII', MDLB_MAGIC, MDLB_VERSION, n, 0)
    directory = bytearray(_DIR_ENTRY_SIZE * n)
    payload = bytearray()
    for i, tag in enumerate(_SECTION_TAGS):
        data = sections[tag]
        while len(payload) % 8 != 0:  # 8バイトアライン(仕様必須)
            payload.append(0)
        rel_offset = len(payload)
        length = len(data)
        crc = zlib.crc32(data) & 0xffffffff
        struct.pack_into('<4sIQQI4x', directory, i * _DIR_ENTRY_SIZE,
                         tag, 0, rel_offset, length, crc)
        payload.extend(data)
    return header + bytes(directory) + bytes(payload)


def _disassemble_container(data: bytes) -> dict[bytes, bytes]:
    if len(data) < _HEADER_SIZE:
        raise ValueError('read_mdlb: ヘッダ未満で切り詰められている')
    magic, version, n_sections, _reserved = struct.unpack_from('<4sIII', data, 0)
    if magic != MDLB_MAGIC:
        raise ValueError(f'read_mdlb: マジックが不正: {magic!r}(期待 {MDLB_MAGIC!r})')
    if version != MDLB_VERSION:
        raise ValueError(
            f'read_mdlb: version={version} は非対応(このリーダは'
            f'{MDLB_VERSION}のみ)')

    dir_start = _HEADER_SIZE
    payload_start = dir_start + _DIR_ENTRY_SIZE * n_sections
    if len(data) < payload_start:
        raise ValueError('read_mdlb: セクションディレクトリが切り詰められている')

    sections: dict[bytes, bytes] = {}
    for i in range(n_sections):
        off = dir_start + i * _DIR_ENTRY_SIZE
        tag, _reserved2, rel_offset, length, crc = struct.unpack_from(
            '<4sIQQI', data, off)
        abs_off = payload_start + rel_offset
        if abs_off < payload_start or abs_off + length > len(data):
            raise ValueError(
                f'read_mdlb: セクション{tag!r}のoffset/lengthがファイル'
                '範囲外(壊れている)')
        blob = data[abs_off:abs_off + length]
        got_crc = zlib.crc32(blob) & 0xffffffff
        if got_crc != crc:
            raise ValueError(
                f'read_mdlb: セクション{tag!r}のチェックサム不一致'
                f'(壊れている): 期待{crc:#010x} 実測{got_crc:#010x}')
        if tag in sections:
            raise ValueError(f'read_mdlb: セクションタグ{tag!r}が重複している')
        sections[tag] = blob

    missing = [t for t in _SECTION_TAGS if t not in sections]
    if missing:
        raise ValueError(f'read_mdlb: 必須セクションが無い: {missing}')
    return sections


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------

def encode_bundle_to_mdlb(bundle: dict) -> bytes:
    """バンドルdict(JSON安全なplain型。write_mdlbが正準化を済ませる前提)を
    `.mdlb` バイナリへエンコードする。"""
    if not isinstance(bundle, dict):
        raise TypeError('encode_bundle_to_mdlb: bundleはdictでなければならない')
    ir_dict = bundle.get('ir')
    if not isinstance(ir_dict, dict):
        raise ValueError('encode_bundle_to_mdlb: bundleにirが無い')

    meta_ir, tt_entries, idx_entries = _split_ir_for_mdlb(ir_dict)

    sections: dict[bytes, bytes] = {}

    b = bytearray()
    _write_value(b, meta_ir)
    sections[b'IRMT'] = bytes(b)

    b = bytearray()
    _write_uvarint(b, len(tt_entries))
    for node_id, hexlist in tt_entries:
        _encode_tt_entry(b, node_id, hexlist)
    sections[b'TTBL'] = bytes(b)

    b = bytearray()
    _write_uvarint(b, len(idx_entries))
    for node_id, key, flat, shape in idx_entries:
        _write_str(b, node_id)
        _write_str(b, key)
        _write_uvarint(b, len(shape))
        for d in shape:
            _write_uvarint(b, d)
        _write_uvarint(b, len(flat))
        _write_uarray(b, flat)
    sections[b'IDXS'] = bytes(b)

    has_programs = 'programs' in bundle
    b = bytearray()
    b.append(1 if has_programs else 0)
    if has_programs:
        programs = bundle['programs'] or {}
        if not isinstance(programs, dict):
            raise ValueError("encode_bundle_to_mdlb: bundle['programs'] がdictでない")
        _write_uvarint(b, len(programs))
        for node_id, prog in programs.items():
            _encode_program(b, node_id, prog)
    sections[b'PROG'] = bytes(b)

    has_preproc_desc = 'preproc_desc' in bundle and bundle['preproc_desc'] is not None
    b = bytearray()
    b.append(1 if has_preproc_desc else 0)
    if has_preproc_desc:
        _encode_preproc_desc(b, bundle['preproc_desc'])
    sections[b'PPDS'] = bytes(b)

    has_labels = 'labels' in bundle
    b = bytearray()
    b.append(1 if has_labels else 0)
    if has_labels:
        _write_value(b, bundle['labels'])
    sections[b'LABL'] = bytes(b)

    has_readout = 'readout' in bundle
    b = bytearray()
    b.append(1 if has_readout else 0)
    if has_readout:
        _write_value(b, bundle['readout'])
    sections[b'RDOT'] = bytes(b)

    extra = {k: v for k, v in bundle.items()
            if k not in ('ir', 'programs', 'preproc_desc', 'labels', 'readout')}
    b = bytearray()
    _write_value(b, extra)
    sections[b'EXTR'] = bytes(b)

    return _assemble_container(sections)


def decode_mdlb_bytes(data: bytes) -> dict:
    """`.mdlb` バイナリ(bytes)→ バンドルdict。"""
    sections = _disassemble_container(data)

    meta_ir, _ = _read_value(sections[b'IRMT'], 0)

    tt_entries = []
    blob = sections[b'TTBL']
    pos = 0
    n_tt, pos = _read_uvarint(blob, pos)
    for _ in range(n_tt):
        node_id, hexlist, pos = _decode_tt_entry(blob, pos)
        tt_entries.append((node_id, hexlist))

    idx_entries = []
    blob = sections[b'IDXS']
    pos = 0
    n_idx, pos = _read_uvarint(blob, pos)
    for _ in range(n_idx):
        node_id, pos = _read_str(blob, pos)
        key, pos = _read_str(blob, pos)
        n_dims, pos = _read_uvarint(blob, pos)
        shape = []
        for _ in range(n_dims):
            d, pos = _read_uvarint(blob, pos)
            shape.append(d)
        n_flat, pos = _read_uvarint(blob, pos)
        flat, pos = _read_uarray(blob, pos, n_flat)
        idx_entries.append((node_id, key, flat, shape))

    ir_dict = _merge_ir_from_mdlb(meta_ir, tt_entries, idx_entries)

    blob = sections[b'PROG']
    has_programs = bool(blob[0]) if blob else False
    programs = None
    if has_programs:
        pos = 1
        n_prog, pos = _read_uvarint(blob, pos)
        programs = {}
        for _ in range(n_prog):
            node_id, prog, pos = _decode_program(blob, pos)
            programs[node_id] = prog

    blob = sections[b'PPDS']
    has_preproc_desc = bool(blob[0]) if blob else False
    preproc_desc = None
    if has_preproc_desc:
        preproc_desc, _ = _decode_preproc_desc(blob, 1)

    blob = sections[b'LABL']
    has_labels = bool(blob[0]) if blob else False
    labels = None
    if has_labels:
        labels, _ = _read_value(blob, 1)

    blob = sections[b'RDOT']
    has_readout = bool(blob[0]) if blob else False
    readout = None
    if has_readout:
        readout, _ = _read_value(blob, 1)

    extra, _ = _read_value(sections[b'EXTR'], 0)

    bundle = dict(extra)
    bundle['ir'] = ir_dict
    if has_programs:
        bundle['programs'] = programs
    if has_preproc_desc:
        bundle['preproc_desc'] = preproc_desc
    if has_labels:
        bundle['labels'] = labels
    if has_readout:
        bundle['readout'] = readout
    return bundle


def write_mdlb(bundle: dict, path) -> Path:
    """バンドルを `.mdlb` バイナリで書き出す。

    書き込み前に `json.loads(json.dumps(bundle))` でJSON正準化する
    (バンドルの正準形は「JSONデコード後の形」— `save_bundle`/`load_bundle`
    と同じ前提。int辞書キーの文字列化・タプル→list化などが正準化で確定する)。
    """
    canonical = json.loads(json.dumps(bundle, ensure_ascii=False))
    data = encode_bundle_to_mdlb(canonical)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def read_mdlb(path) -> dict:
    """`.mdlb`(または `.mdlb.gz`)を読み込んでバンドルdictを返す。"""
    p = Path(path)
    raw = p.read_bytes()
    if p.name.endswith('.gz'):
        raw = gzip.decompress(raw)
    return decode_mdlb_bytes(raw)
