"""配布バンドルv1のフォーマット・検証テスト。

固定する契約:
    - build_bundle は元docのキーを保ったまま bundle_format/bundle_version/
      programs を足す(既存の配信JSONを壊さない)
    - programs は conv2d ノードと1対1
    - 命令列は program_to_dict / program_from_dict で往復同一
    - 互換ポリシー: MAJOR不一致は拒否、MINORが新しいだけなら警告
    - strict検証はIRから再構築して完全一致まで見る(改竄・陳腐化の検出)
"""

import copy
import json

import pytest

from griffith_logic.deploy import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    build_bundle,
    build_programs,
    load_bundle,
    program_from_dict,
    program_to_dict,
    save_bundle,
    validate_bundle,
)
from griffith_logic.export.optimize import (
    build_conv_program,
    spec_from_node_params,
)
from griffith_logic.ir.serialize import ir_from_dict


def _conv_node_ids(doc):
    return [n['id'] for n in doc['ir']['nodes'] if n['type'] == 'conv2d']


class TestBuildBundle:
    def test_adds_keys_without_touching_original(self, seg_doc):
        before = copy.deepcopy(seg_doc)
        bundle = build_bundle(seg_doc)
        assert seg_doc == before, 'build_bundleが元docを変更した'
        assert bundle['bundle_format'] == BUNDLE_FORMAT
        assert bundle['bundle_version'] == BUNDLE_VERSION
        # 既存キーはそのまま引き継がれる
        for key in ('ir', 'preproc', 'labels', 'readout', 'resources', 'eval'):
            assert bundle[key] == seg_doc[key]

    def test_programs_cover_every_conv2d_node(self, seg_doc):
        bundle = build_bundle(seg_doc)
        assert sorted(bundle['programs']) == sorted(_conv_node_ids(seg_doc))
        assert bundle['programs'], 'セグIRにconv2dが1つも無い(前提が崩れている)'

    def test_gate_reduction_is_recorded(self, seg_doc):
        bundle = build_bundle(seg_doc)
        for prog in bundle['programs'].values():
            stats = prog['stats']
            assert stats['gates_after'] <= stats['gates_before']
            assert stats['gates_after'] == prog['n_instrs']

    def test_fully_folded_conv_yields_empty_instruction_list(
            self, webbench, seg_graph_passthrough):
        """未学習のパススルー支配な木は命令列が空になる(実在するケース)。

        n_instrs=0 でも write_start 長などの不変条件は保たれ、検証を通る。
        """
        doc = webbench.build_model_doc(
            seg_graph_passthrough, preproc={}, labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        assert bundle['programs'], 'conv2dノードが無い(前提が崩れている)'
        assert all(p['n_instrs'] == 0 for p in bundle['programs'].values())
        assert validate_bundle(bundle, strict=True) == []

    def test_dense_model_has_empty_programs(self, dense_doc):
        bundle = build_bundle(dense_doc)
        assert bundle['programs'] == {}
        assert validate_bundle(bundle) == []

    def test_rejects_doc_without_ir(self):
        with pytest.raises(ValueError, match="'ir'"):
            build_bundle({'labels': []})

    def test_rejects_wrong_ir_format(self, seg_doc):
        doc = copy.deepcopy(seg_doc)
        doc['ir']['format'] = 'not-magicdelta'
        with pytest.raises(ValueError, match='format'):
            build_bundle(doc)


class TestProgramSerialization:
    def test_roundtrip_is_identity(self, seg_doc):
        graph = ir_from_dict(seg_doc['ir'])
        node = next(n for n in graph.nodes if n.type == 'conv2d')
        prog = build_conv_program(spec_from_node_params(node.params))
        d = program_to_dict(prog)
        again = program_to_dict(program_from_dict(d))
        assert again == d

    def test_from_dict_restores_instruction_fields(self, seg_doc):
        graph = ir_from_dict(seg_doc['ir'])
        node = next(n for n in graph.nodes if n.type == 'conv2d')
        prog = build_conv_program(spec_from_node_params(node.params))
        restored = program_from_dict(program_to_dict(prog))
        assert restored.instrs == prog.instrs
        assert restored.taps == prog.taps
        assert restored.oc_out == prog.oc_out
        assert restored.n_slots == prog.n_slots
        assert restored.stats == prog.stats

    def test_from_dict_rejects_truncated_instrs(self, seg_doc):
        graph = ir_from_dict(seg_doc['ir'])
        node = next(n for n in graph.nodes if n.type == 'conv2d')
        d = program_to_dict(build_conv_program(spec_from_node_params(node.params)))
        d['instrs'] = d['instrs'][:-1]
        with pytest.raises(ValueError, match='instrs'):
            program_from_dict(d)

    def test_json_payload_is_plain_ints(self, seg_doc):
        """numpy型が混ざるとjson.dumpsが落ちるので、素のintであることを固定する。"""
        programs = build_programs(ir_from_dict(seg_doc['ir']))
        json.dumps(programs)  # 例外が出なければよい
        for prog in programs.values():
            assert all(type(v) is int for v in prog['instrs'][:8])
            assert all(type(v) is int for v in prog['taps'][:8])


class TestValidateBundle:
    def test_accepts_freshly_built(self, seg_doc):
        bundle = build_bundle(seg_doc)
        assert validate_bundle(bundle) == []
        assert validate_bundle(bundle, strict=True) == []

    def test_requires_mandatory_keys(self, seg_doc):
        bundle = build_bundle(seg_doc)
        del bundle['bundle_version']
        with pytest.raises(ValueError, match='必須キー'):
            validate_bundle(bundle)

    def test_rejects_major_version_mismatch(self, seg_doc):
        bundle = build_bundle(seg_doc)
        bundle['bundle_version'] = '2.0'
        with pytest.raises(ValueError, match='非対応'):
            validate_bundle(bundle)

    def test_warns_on_newer_minor_version(self, seg_doc):
        bundle = build_bundle(seg_doc)
        major = BUNDLE_VERSION.split('.')[0]
        bundle['bundle_version'] = f'{major}.99'
        warnings = validate_bundle(bundle)
        assert any('新しい' in w for w in warnings)

    def test_warns_on_unknown_ir_version(self, seg_doc):
        bundle = build_bundle(seg_doc)
        bundle['ir']['version'] = '9.9.0'
        assert any('IR version' in w for w in validate_bundle(bundle))

    def test_rejects_missing_program_for_conv_node(self, seg_doc):
        bundle = build_bundle(seg_doc)
        victim = sorted(bundle['programs'])[0]
        del bundle['programs'][victim]
        with pytest.raises(ValueError, match='programs節に無い'):
            validate_bundle(bundle)

    def test_rejects_orphan_program(self, seg_doc):
        bundle = build_bundle(seg_doc)
        bundle['programs']['ghost'] = next(iter(bundle['programs'].values()))
        with pytest.raises(ValueError, match='対応しないprograms'):
            validate_bundle(bundle)

    def test_rejects_out_of_range_tap(self, seg_doc):
        bundle = build_bundle(seg_doc)
        prog = bundle['programs'][sorted(bundle['programs'])[0]]
        prog['taps'][0] = 10 ** 6
        with pytest.raises(ValueError, match='受容野'):
            validate_bundle(bundle)

    def test_rejects_forward_reference(self, seg_doc):
        """命令が自分より後ろの命令を参照していたら弾く(トポ順の破壊)。"""
        bundle = build_bundle(seg_doc)
        prog = bundle['programs'][sorted(bundle['programs'])[0]]
        n = prog['n_instrs']
        assert n >= 2
        prog['instrs'][1] = ((n - 1) << 2) | 2  # 命令0が末尾命令を参照
        with pytest.raises(ValueError, match='前方参照'):
            validate_bundle(bundle)

    def test_strict_detects_stale_programs(self, seg_doc):
        """故障注入: IRを書き換えるとstrictがprograms節の陳腐化を検出する。"""
        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        # 全ゲートを補元に差し替える(木の意味が確実に変わる)
        node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            for tt in node['params']['truth_tables']
        ]
        assert validate_bundle(bundle) == [], '構造検査だけでは検出できないはず'
        with pytest.raises(ValueError, match='再構築'):
            validate_bundle(bundle, strict=True)

    def test_strict_detects_rewired_taps(self, seg_doc):
        """配線(indices_flat)の書き換えもstrictで検出できる。"""
        bundle = build_bundle(seg_doc)
        node = next(n for n in bundle['ir']['nodes'] if n['type'] == 'conv2d')
        rows = node['params']['indices_flat']
        rows[0] = list(reversed(rows[0]))
        with pytest.raises(ValueError, match='再構築'):
            validate_bundle(bundle, strict=True)

    def test_dead_gate_mutation_is_invisible(self, seg_doc):
        """dead除去された枝の真理値表を変えても命令列は変わらない。

        「消えたゲートは本当に出力へ効いていない」ことの確認であり、
        strict検証が万能でない(=IRの意味変更を必ず捕まえるわけではない)
        ことの明示でもある。
        """
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        node = next(n for n in graph.nodes if n.type == 'conv2d')
        n_internal = (1 << int(node.params['tree_depth'])) - 1
        prog = build_conv_program(spec_from_node_params(node.params))

        # oc0の根ゲート(index n_internal-1)が退化して枝が死ぬ木を探し、
        # 死んだ側の葉ゲートを叩く。見つからなければこのテストは無意味なのでskip
        ir_node = next(n for n in bundle['ir']['nodes'] if n['id'] == node.id)
        tts = ir_node['params']['truth_tables']
        dead_index = None
        for oc in range(int(node.params['out_channels'])):
            base = oc * n_internal
            for g in range(n_internal - 1):
                probe = list(tts)
                probe[base + g] = hex(
                    (int(probe[base + g], 16)
                     if isinstance(probe[base + g], str) else int(probe[base + g])) ^ 0xf
                )
                params = dict(node.params, truth_tables=probe)
                if build_conv_program(spec_from_node_params(params)) == prog:
                    dead_index = base + g
                    break
            if dead_index is not None:
                break
        if dead_index is None:
            pytest.skip('この木にはdead除去されるゲートが無い')

        ir_node['params']['truth_tables'] = [
            hex((int(tt, 16) if isinstance(tt, str) else int(tt)) ^ 0xf)
            if i == dead_index else tt
            for i, tt in enumerate(tts)
        ]
        assert validate_bundle(bundle, strict=True) == []


class TestSaveLoad:
    def test_roundtrip_json(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        path = save_bundle(bundle, tmp_path / 'm.bundle.json')
        assert load_bundle(path) == json.loads(json.dumps(bundle))

    def test_roundtrip_gzip(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        save_bundle(bundle, tmp_path / 'm.bundle.json', gzip_too=True)
        gz = tmp_path / 'm.bundle.json.gz'
        assert gz.exists()
        assert load_bundle(gz) == json.loads(json.dumps(bundle))
        assert gz.stat().st_size < (tmp_path / 'm.bundle.json').stat().st_size

    def test_ir_and_programs_survive_roundtrip_exactly(self, seg_doc, tmp_path):
        """配布物の本体(ir/programs)はJSON往復で1ビットも変わらない。

        resources.lut_counts だけは**整数キーのdict**なのでJSON往復で
        キーが文字列化する(build_model_doc由来の既存仕様。TS側も文字列
        キーで読む)。バンドルの正準形はJSONデコード後の形とする。
        """
        bundle = build_bundle(seg_doc)
        loaded = load_bundle(save_bundle(bundle, tmp_path / 'm.json'))
        assert loaded['ir'] == bundle['ir']
        assert loaded['programs'] == bundle['programs']
        assert set(loaded['resources']['lut_counts']) == {
            str(k) for k in bundle['resources']['lut_counts']
        }

    def test_load_validates_by_default(self, seg_doc, tmp_path):
        bundle = build_bundle(seg_doc)
        bundle['bundle_version'] = '2.0'
        path = save_bundle(bundle, tmp_path / 'bad.json')
        with pytest.raises(ValueError, match='非対応'):
            load_bundle(path)


class TestCli:
    def test_bundle_then_info(self, seg_doc, tmp_path, capsys):
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'model.bundle.json'

        assert main(['deploy', 'bundle', str(src), '-o', str(out), '--gzip']) == 0
        assert out.exists() and (tmp_path / 'model.bundle.json.gz').exists()
        assert '削減' in capsys.readouterr().out

        assert main(['deploy', 'info', str(out), '--strict']) == 0
        assert '検証: OK' in capsys.readouterr().out

    def test_bundle_sizes_match_actual_output(self, seg_doc, tmp_path):
        """`griffith deploy bundle` が書くresources.bundle_json_bytesは、
        実際に書き出されたファイルのバイト数と厳密一致すること(自己参照の
        不動点がちゃんと収束していることの回帰テスト)。gzipサイズは
        目安値なので正の値であることだけ確認する。
        """
        from griffith_logic.cli import main

        src = tmp_path / 'model.json'
        src.write_text(json.dumps(seg_doc), encoding='utf-8')
        out = tmp_path / 'model.bundle.json'

        assert main(['deploy', 'bundle', str(src), '-o', str(out)]) == 0

        bundle = json.loads(out.read_text(encoding='utf-8'))
        resources = bundle['resources']
        assert resources['bundle_json_bytes'] == out.stat().st_size
        assert resources['bundle_gzip_bytes'] > 0

    def test_info_reports_broken_bundle(self, seg_doc, tmp_path):
        from griffith_logic.cli import main

        bundle = build_bundle(seg_doc)
        bundle['bundle_version'] = '2.0'
        path = tmp_path / 'bad.bundle.json'
        path.write_text(json.dumps(bundle), encoding='utf-8')
        assert main(['deploy', 'info', str(path)]) == 1  # ValueError → exit 1
