"""IR JSON読み込みのセキュリティ検証。

信頼できないIR JSONを `load_ir` に渡したときに、
  - $data参照によるパストラバーサル(json_dir外の読み出し)を拒否する
  - format不一致を(python -O でも消えない)ValueErrorで拒否する
  - サイズ上限超過を拒否する
ことを検証する。正常系(npzサイドカーのラウンドトリップ)も併せて守る。
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from griffith_logic.ir.schema import IR_FORMAT, IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import DEFAULT_MAX_IR_BYTES, load_ir, save_ir


def _make_graph(n: int = 4) -> IRGraph:
    """indices配列を1つ持つ最小のIRGraphを作る"""
    return IRGraph(
        metadata={'name': 'sec'},
        inputs=[IRPort(id='in', width=n)],
        outputs=[IRPort(id='out', width=1)],
        nodes=[IRNode(
            id='lut0', type='lut',
            params={'indices': np.arange(n, dtype=np.int64),
                    'truth_tables': ['0x1']},
        )],
        edges=[IREdge(src='in', dst='lut0')],
    )


def _write_doc(path, doc: dict) -> str:
    path.write_text(json.dumps(doc), encoding='utf-8')
    return str(path)


def _base_doc() -> dict:
    return {
        'format': IR_FORMAT,
        'version': '0.2',
        'metadata': {},
        'inputs': [{'id': 'in', 'width': 4}],
        'outputs': [{'id': 'out', 'width': 1}],
        'nodes': [{'id': 'lut0', 'type': 'lut',
                   'params': {}, 'train_info': {}, 'meta': {}}],
        'edges': [{'src': 'in', 'dst': 'lut0'}],
    }


class TestDataRefTraversal:
    @pytest.mark.parametrize('ref', [
        '../secret.npz#a',
        'sub/../../secret.npz#a',
        '/etc/passwd.npz#a',
        'dir/inner.npz#a',
    ])
    def test_directory_components_rejected(self, tmp_path, ref):
        """$dataにディレクトリ成分・絶対パスを含む参照は拒否する(B2)"""
        # json_dir外に本物のnpzを置いても読まれないこと
        outside = tmp_path / 'secret.npz'
        np.savez(outside, a=np.arange(3))
        ir_dir = tmp_path / 'ir'
        ir_dir.mkdir()
        doc = _base_doc()
        doc['nodes'][0]['params'] = {'indices': {'$data': ref}}
        path = _write_doc(ir_dir / 'model.json', doc)
        with pytest.raises(ValueError):
            load_ir(path)

    def test_missing_hash_rejected(self, tmp_path):
        """'#'区切りが無い$data参照はValueError(素のValueError化 / B2)"""
        doc = _base_doc()
        doc['nodes'][0]['params'] = {'indices': {'$data': 'model.npz'}}
        path = _write_doc(tmp_path / 'model.json', doc)
        with pytest.raises(ValueError, match='形式が不正'):
            load_ir(path)

    def test_empty_key_rejected(self, tmp_path):
        doc = _base_doc()
        doc['nodes'][0]['params'] = {'indices': {'$data': 'model.npz#'}}
        path = _write_doc(tmp_path / 'model.json', doc)
        with pytest.raises(ValueError):
            load_ir(path)

    def test_unknown_key_rejected(self, tmp_path):
        """npzは存在するがキーが無い場合もValueError"""
        np.savez(tmp_path / 'model.npz', other=np.arange(3))
        doc = _base_doc()
        doc['nodes'][0]['params'] = {'indices': {'$data': 'model.npz#nai'}}
        path = _write_doc(tmp_path / 'model.json', doc)
        with pytest.raises(ValueError, match='キーがありません'):
            load_ir(path)


class TestFormatCheck:
    def test_format_mismatch_raises_value_error(self, tmp_path):
        doc = _base_doc()
        doc['format'] = 'evil-format'
        path = _write_doc(tmp_path / 'model.json', doc)
        with pytest.raises(ValueError, match='formatが不正'):
            load_ir(path)

    def test_format_check_survives_optimized_mode(self, tmp_path):
        """python -O(assert無効化)でもformatチェックが効く(R3)"""
        doc = _base_doc()
        doc['format'] = 'evil-format'
        path = _write_doc(tmp_path / 'model.json', doc)
        code = (
            'from griffith_logic.ir.serialize import load_ir\n'
            f'try:\n    load_ir({path!r})\n'
            'except ValueError:\n    print("VALUEERROR")\n'
        )
        proc = subprocess.run(
            [sys.executable, '-O', '-c', code],
            capture_output=True, text=True, check=True,
        )
        assert 'VALUEERROR' in proc.stdout

    def test_non_object_json_rejected(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text('[1, 2, 3]', encoding='utf-8')
        with pytest.raises(ValueError):
            load_ir(str(path))

    def test_broken_json_raises_value_error(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text('{"format": ', encoding='utf-8')
        with pytest.raises(ValueError, match='解析に失敗'):
            load_ir(str(path))


class TestSizeLimit:
    def test_over_limit_raises_value_error(self, tmp_path):
        """max_bytes超過は読み込まずにValueError(R4)"""
        doc = _base_doc()
        path = _write_doc(tmp_path / 'model.json', doc)
        with pytest.raises(ValueError, match='サイズ上限'):
            load_ir(path, max_bytes=10)

    def test_default_limit_is_positive(self):
        assert DEFAULT_MAX_IR_BYTES > 0

    def test_deeply_nested_json_raises_value_error(self, tmp_path):
        """深いネストのJSONはRecursionErrorでなくValueErrorになる(R4)"""
        depth = 200000
        path = tmp_path / 'model.json'
        path.write_text('[' * depth + ']' * depth, encoding='utf-8')
        with pytest.raises(ValueError):
            load_ir(str(path))


class TestNormalPath:
    def test_roundtrip_with_npz_sidecar(self, tmp_path):
        """従来どおり npz サイドカー経由のラウンドトリップが動く"""
        graph = _make_graph(n=8)
        json_path = save_ir(graph, str(tmp_path / 'model.json'),
                            inline_threshold=2)
        assert (tmp_path / 'model.npz').is_file()
        loaded = load_ir(json_path)
        assert loaded.format == IR_FORMAT
        np.testing.assert_array_equal(
            np.asarray(loaded.nodes[0].params['indices']),
            np.arange(8, dtype=np.int64),
        )

    def test_roundtrip_inline(self, tmp_path):
        graph = _make_graph(n=4)
        json_path = save_ir(graph, str(tmp_path / 'model.json'))
        loaded = load_ir(json_path)
        assert list(loaded.nodes[0].params['indices']) == [0, 1, 2, 3]
