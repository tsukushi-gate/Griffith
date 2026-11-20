"""配布バンドルのバイナリ形式 `.mdlb`のテスト。

固定する契約:
    - write_mdlb/read_mdlb は `json.loads(json.dumps(bundle))` した正準形と
      厳密一致でラウンドトリップする(型まで区別する `_strict_equal`。
      詳細規約は `griffith_logic/deploy/mdlb.py` のモジュールdocstring)
    - コンテナは magic/version/セクション境界/セクションCRC32の故障注入で
      落ちる(黙って壊れたデータを読まない)
    - `save_bundle`/`load_bundle`(bundle.py)は拡張子(`.mdlb`)で自動的に
      バイナリ形式へ切り替わり、既存の `.json` 経路は不変のまま
    - `.mdlb` から読んだバンドルは、`.json` から読んだバンドルと**バイト
      同一の生成C**を出す(codegen-c の入力として完全に等価)
"""

import gzip
import json
import struct

import pytest

from griffith_logic.deploy.bundle import build_bundle, load_bundle, save_bundle
from griffith_logic.deploy.mdlb import (
    MDLB_MAGIC,
    MDLB_VERSION,
    decode_mdlb_bytes,
    encode_bundle_to_mdlb,
    read_mdlb,
    write_mdlb,
)


def _strict_equal(a, b) -> bool:
    """dict/listの深い等価性を、型まで区別して確かめる(1==1.0やbool==int
    のような緩い一致に頼らない厳密規約。mdlb.pyモジュールdocstring参照)。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return (set(a.keys()) == set(b.keys())
                and all(_strict_equal(a[k], b[k]) for k in a))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return type(a) is type(b) and a == b
    return type(a) is type(b) and a == b


def _canonical(bundle: dict) -> dict:
    return json.loads(json.dumps(bundle, ensure_ascii=False))


class TestRoundTrip:
    """`read_mdlb(write_mdlb(bundle)) == json正準化(bundle)` を各種バンドルで確かめる。"""

    def test_seg_bundle_with_conv_programs(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        back = read_mdlb(path)
        assert _strict_equal(back, _canonical(bundle))

    def test_dense_bundle_empty_programs_and_indices_key(self, dense_doc, tmp_path):
        """conv2dが無い(lut_layer+group_sumのみ)モデル: programs=={}、
        indices(indices_flatでなく)キー、preproc_desc無しの経路を通す。"""
        bundle = build_bundle(dense_doc)
        assert bundle['programs'] == {}
        assert 'preproc_desc' not in bundle
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'lut_layer')
        assert 'indices' in node['params']
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        back = read_mdlb(path)
        assert _strict_equal(back, _canonical(bundle))
        assert back['programs'] == {}
        assert 'preproc_desc' not in back

    def test_rddlgn_bundle_with_embed_table(self, rddlgn_seq2seq_tiny_bundle_gen,
                                            tmp_path):
        """embed_table節(EXTRAセクション経由)を含むバンドルの往復。"""
        bundle = rddlgn_seq2seq_tiny_bundle_gen
        assert 'embed_table' in bundle
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        back = read_mdlb(path)
        assert _strict_equal(back, _canonical(bundle))
        assert back['embed_table'] == _canonical(bundle)['embed_table']

    def test_preserves_negative_ints_in_preproc_desc(self, seg_doc, tmp_path):
        """preproc_desc.descはdy/dx等の符号付きワードを含みうる(zigzag必須)。"""
        bundle = build_bundle(seg_doc)
        assert 'preproc_desc' in bundle
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        back = read_mdlb(path)
        assert back['preproc_desc'] == _canonical(bundle)['preproc_desc']

    def test_preserves_float_and_bool_and_none(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        bundle = dict(bundle)
        bundle['about'] = dict(bundle.get('about') or {})
        bundle['about'].update({
            'a_float': 3.5, 'a_negative_float': -0.125, 'a_bool_true': True,
            'a_bool_false': False, 'a_none': None, 'zero': 0, 'zero_float': 0.0,
        })
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        back = read_mdlb(path)
        got = back['about']
        assert got['a_float'] == 3.5 and isinstance(got['a_float'], float)
        assert got['a_negative_float'] == -0.125
        assert got['a_bool_true'] is True
        assert got['a_bool_false'] is False
        assert got['a_none'] is None
        assert got['zero'] == 0 and isinstance(got['zero'], int)
        assert got['zero_float'] == 0.0 and isinstance(got['zero_float'], float)

    def test_dict_key_order_does_not_matter_but_values_do(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        data = encode_bundle_to_mdlb(_canonical(bundle))
        back = decode_mdlb_bytes(data)
        # 同じ内容を辞書として作り直しても(挿入順が違っても)等価。
        reordered = dict(reversed(list(back.items())))
        assert _strict_equal(back, reordered)


class TestContainerFormat:
    def test_starts_with_magic_and_version(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        path = write_mdlb(bundle, tmp_path / 'm.mdlb')
        data = path.read_bytes()
        magic, version = struct.unpack_from('<4sI', data, 0)
        assert magic == MDLB_MAGIC == b'MDLB'
        assert version == MDLB_VERSION

    def test_encode_bytes_and_decode_bytes_are_inverse(self, seg_doc):
        bundle = _canonical(build_bundle(seg_doc))
        data = encode_bundle_to_mdlb(bundle)
        assert isinstance(data, bytes)
        assert _strict_equal(decode_mdlb_bytes(data), bundle)


class TestFaultInjection:
    """マジック/バージョン/セクション境界/チェックサムの破壊を検出できるか。"""

    @pytest.fixture
    def good_bytes(self, seg_doc):
        bundle = _canonical(build_bundle(seg_doc))
        return encode_bundle_to_mdlb(bundle)

    def test_bad_magic_raises(self, good_bytes):
        bad = bytearray(good_bytes)
        bad[0] = ord('X')
        with pytest.raises(ValueError, match='マジック'):
            decode_mdlb_bytes(bytes(bad))

    def test_bad_version_raises(self, good_bytes):
        bad = bytearray(good_bytes)
        struct.pack_into('<I', bad, 4, 999)
        with pytest.raises(ValueError, match='version'):
            decode_mdlb_bytes(bytes(bad))

    def test_truncated_file_raises(self, good_bytes):
        with pytest.raises(ValueError):
            decode_mdlb_bytes(good_bytes[:len(good_bytes) // 2])

    def test_too_short_for_header_raises(self, good_bytes):
        with pytest.raises(ValueError, match='ヘッダ'):
            decode_mdlb_bytes(good_bytes[:8])

    def test_checksum_corruption_raises(self, good_bytes):
        bad = bytearray(good_bytes)
        bad[-1] ^= 0xff  # 最終セクションのペイロード末尾を反転
        with pytest.raises(ValueError, match='チェックサム'):
            decode_mdlb_bytes(bytes(bad))

    def test_section_length_out_of_range_raises(self, good_bytes):
        bad = bytearray(good_bytes)
        # 先頭セクション(IRMT)のlengthフィールドをファイルサイズ超へ書き換える。
        # ディレクトリentry: offset16 = tag(4)+reserved(4)+rel_offset(8)+length(8)
        struct.pack_into('<Q', bad, 16 + 16, len(good_bytes) * 2)
        with pytest.raises(ValueError, match='範囲外'):
            decode_mdlb_bytes(bytes(bad))


class TestBundleSaveLoadIntegration:
    """bundle.py::save_bundle/load_bundleの拡張子自動判定。"""

    def test_mdlb_extension_autodetects_binary_format(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        out = save_bundle(bundle, tmp_path / 'm.mdlb')
        assert out.read_bytes()[:4] == b'MDLB'
        back = load_bundle(out)
        assert _strict_equal(back, _canonical(bundle))

    def test_json_extension_unchanged(self, seg_doc, tmp_path):
        """既存の.json経路は今回の変更で挙動もバイト列も変わらない(回帰防止)。"""
        bundle = build_bundle(seg_doc)
        out = save_bundle(bundle, tmp_path / 'm.json')
        expected = json.dumps(bundle, separators=(',', ':'), ensure_ascii=False)
        assert out.read_text(encoding='utf-8') == expected

    def test_mdlb_gz_round_trips(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        out = save_bundle(bundle, tmp_path / 'm.mdlb', gzip_too=True)
        gz = tmp_path / 'm.mdlb.gz'
        assert gz.exists()
        assert gzip.decompress(gz.read_bytes()) == out.read_bytes()
        back = load_bundle(gz)
        assert _strict_equal(back, _canonical(bundle))

    def test_explicit_format_overrides_extension(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        # .json拡張子でもformat='mdlb'を明示すればバイナリで書く。
        out = save_bundle(bundle, tmp_path / 'm.json', format='mdlb')
        assert out.read_bytes()[:4] == b'MDLB'

    def test_load_bundle_validates_mdlb_like_json(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        out = save_bundle(bundle, tmp_path / 'm.mdlb')
        # validate=Trueが既定。改竄されていなければ警告なしで読める。
        back = load_bundle(out, validate=True, strict=True)
        assert back['bundle_format'] == 'magicdelta-bundle'


class TestCliBundleFormat:
    """`griffith deploy bundle -o *.mdlb` / `--format mdlb` / `griffith deploy info`。"""

    def test_output_extension_selects_mdlb(self, seg_doc, tmp_path):
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'm.mdlb'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out)])
        assert rc == 0
        assert out.read_bytes()[:4] == b'MDLB'

    def test_format_flag_without_output_defaults_to_bundle_mdlb(self, seg_doc,
                                                                 tmp_path):
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        rc = main(['deploy', 'bundle', str(src), '--format', 'mdlb'])
        assert rc == 0
        assert (tmp_path / 'model.bundle.mdlb').exists()

    def test_info_reads_mdlb_bundle(self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        bundle_path = save_bundle(build_bundle(seg_doc), tmp_path / 'm.mdlb')
        rc = main(['deploy', 'info', str(bundle_path), '--strict'])
        assert rc == 0
        out = capsys.readouterr().out
        assert '検証: OK' in out

    def test_gzip_flag_writes_mdlb_gz(self, seg_doc, tmp_path):
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'm.mdlb'
        rc = main(['deploy', 'bundle', str(src), '-o', str(out), '--gzip'])
        assert rc == 0
        gz = tmp_path / 'm.mdlb.gz'
        assert gz.exists()
        assert gzip.decompress(gz.read_bytes()) == out.read_bytes()


class TestCodegenParityJsonVsMdlb:
    """.mdlb経由で読んだバンドルは、.json経由で読んだバンドルとバイト同一の
    生成Cを出す(griffith deploy codegen-c の入力として完全に等価であることの
    強い検証。タスク仕様が要求する「JSON経由とバイト同一の生成Cが出る
    テスト」)。"""

    def test_generated_c_is_byte_identical(self, seg_doc, tmp_path):
        from griffith_logic.deploy.cbackend import generate_c

        bundle = build_bundle(seg_doc)
        json_path = save_bundle(bundle, tmp_path / 'm.json')
        mdlb_path = save_bundle(bundle, tmp_path / 'm.mdlb')

        from_json = load_bundle(json_path)
        from_mdlb = load_bundle(mdlb_path)

        files_json = generate_c(from_json, name='model', validate=False)
        files_mdlb = generate_c(from_mdlb, name='model', validate=False)
        assert files_json.keys() == files_mdlb.keys()
        for name in files_json:
            assert files_json[name] == files_mdlb[name], f'{name} が食い違う'

    def test_generated_web_bundle_is_semantically_identical(self, seg_doc, tmp_path):
        """generate_webは受け取ったdictをそのままjson.dumpsするので、出力の
        **トップレベルkey順**はPython dictの挿入順に従う。read_mdlbは
        「dictはkeyの集合が一致すれば順序は問わない」という往復契約
        (mdlb.pyモジュールdocstring参照)であり順序までは保存しないため、
        ここではraw文字列でなく**パース後の内容**の一致を見る(C生成の
        テストの方がより強い契約=バイト同一を検証している)。"""
        from griffith_logic.deploy.webbackend import generate_web

        bundle = build_bundle(seg_doc)
        json_path = save_bundle(bundle, tmp_path / 'm.json')
        mdlb_path = save_bundle(bundle, tmp_path / 'm.mdlb')

        from_json = load_bundle(json_path)
        from_mdlb = load_bundle(mdlb_path)

        files_json = generate_web(from_json, name='model')
        files_mdlb = generate_web(from_mdlb, name='model')
        assert files_json.keys() == files_mdlb.keys()
        for name in files_json:
            assert _strict_equal(json.loads(files_json[name]),
                                 json.loads(files_mdlb[name])), f'{name} が食い違う'
