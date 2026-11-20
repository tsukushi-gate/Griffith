"""IRのJSON読み書き。

大配列(indices等)は inline_threshold を超えると .npz サイドカーに
退避し、JSON側には {"$data": "<npzファイル名>#<キー>"} 参照を残す。
truth_tables は常にhex文字列(N=6の64bit値がJSONで安全なため)。

recurrentノードの params['body'] は入れ子の IRGraph を持つ。
_params_to_doc/_params_from_dict がこれを再帰的に扱う(bodyの中の
indices/truth_tablesもnpzへ退避・復元される。npzキーは
`{外側node_id}.body.{内側node_id}_{key}` の接頭辞になる)。
in-memory契約: params['body'] は(export直後・load_ir/ir_from_dict後の
どちらでも)常に IRGraph インスタンス。JSON化されたdictのままなのは
このモジュールの中だけ。
"""

import json
import os
from pathlib import Path

import numpy as np

from .schema import IR_FORMAT, IR_VERSION, IREdge, IRGraph, IRNode, IRPort

DEFAULT_INLINE_THRESHOLD = 65536  # 要素数がこれ以下ならJSONにインライン

# IR JSONの読み込みサイズ上限(バイト)。信頼できない入力によるメモリ枯渇を防ぐ。
# 実運用のIRは大配列をnpzサイドカーへ退避するためJSON自体は数MB以下に収まる。
DEFAULT_MAX_IR_BYTES = 256 * 1024 * 1024  # 256 MiB


def _edge_to_dict(e: IREdge) -> dict:
    """エッジをdict化する。dst_slot=0はキー省略(v0.1 JSONと同一表現)"""
    d = {'src': e.src, 'dst': e.dst}
    if e.dst_slot != 0:
        d['dst_slot'] = e.dst_slot
    return d


def _edge_from_dict(e: dict) -> IREdge:
    """dictからエッジを構築する。dst_slotなし(v0.1)は0扱い"""
    return IREdge(src=e['src'], dst=e['dst'], dst_slot=int(e.get('dst_slot', 0)))


def _params_to_doc(params: dict, node_id: str, *, arrays: dict | None,
                   npz_name: str | None, threshold: int | None) -> dict:
    """paramsをJSON化可能なdictへ変換する。

    arraysがNoneなら常にインライン展開(ir_to_dict互換)。arraysがdictなら
    サイズ閾値超のndarrayをnpzへ退避する(save_ir互換)。'body'キー
    (IRGraphインスタンス)は同じ規則で再帰的に処理する。
    """
    out = {}
    for key, value in params.items():
        if key == 'body' and isinstance(value, IRGraph):
            out[key] = _graph_to_doc(value, f'{node_id}.body.', arrays=arrays,
                                     npz_name=npz_name, threshold=threshold)
        elif isinstance(value, np.ndarray):
            if (arrays is not None and threshold is not None
                    and value.size > threshold):
                data_key = f'{node_id}_{key}'
                arrays[data_key] = value
                out[key] = {'$data': f'{npz_name}#{data_key}'}
            else:
                out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _graph_to_doc(graph: IRGraph, node_id_prefix: str = '', *,
                  arrays: dict | None, npz_name: str | None,
                  threshold: int | None) -> dict:
    """IRGraphをJSON化可能なdictへ変換する(save_ir/ir_to_dict共通実装)。"""
    return {
        'format': graph.format,
        'version': graph.version,
        'metadata': graph.metadata,
        'inputs': [vars(p) for p in graph.inputs],
        'outputs': [vars(p) for p in graph.outputs],
        'nodes': [
            {
                'id': n.id,
                'type': n.type,
                'params': _params_to_doc(n.params, f'{node_id_prefix}{n.id}',
                                         arrays=arrays, npz_name=npz_name,
                                         threshold=threshold),
                'train_info': n.train_info,
                'meta': n.meta,
            }
            for n in graph.nodes
        ],
        'edges': [_edge_to_dict(e) for e in graph.edges],
    }


def save_ir(graph: IRGraph, path: str,
            inline_threshold: int = DEFAULT_INLINE_THRESHOLD) -> str:
    """IRを<path>.json(+必要なら<path>.npz)へ保存し、jsonパスを返す"""
    base = path[:-5] if path.endswith('.json') else path
    json_path = base + '.json'
    npz_name = os.path.basename(base) + '.npz'

    arrays: dict[str, np.ndarray] = {}
    doc = _graph_to_doc(graph, arrays=arrays, npz_name=npz_name,
                        threshold=inline_threshold)
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    if arrays:
        np.savez_compressed(base + '.npz', **arrays)
    return json_path


def _resolve_npz_path(ref: str, json_dir: str) -> tuple[str, str]:
    """`"<npzファイル名>#<キー>"` を (npz絶対パス, キー) へ解決する。

    信頼できないIR JSONがjson_dir外の任意ファイルを読ませる(パストラバーサル)
    ことを防ぐため、ファイル名部分はbasenameへ強制し、解決後の絶対パスが
    json_dir配下であることも検証する。
    """
    if '#' not in ref:
        raise ValueError(
            f"$data参照の形式が不正です(<npzファイル名>#<キー> が必要): {ref!r}"
        )
    npz_name, key = ref.split('#', 1)
    if not npz_name or not key:
        raise ValueError(f'$data参照のファイル名またはキーが空です: {ref!r}')

    # ディレクトリ成分は一切許可しない(../ や絶対パスの除去)
    safe_name = os.path.basename(npz_name)
    if safe_name != npz_name or safe_name in ('.', '..'):
        raise ValueError(
            f'$data参照にディレクトリ成分は指定できません: {npz_name!r}'
        )

    base = Path(json_dir).resolve()
    resolved = (base / safe_name).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f'$data参照がIRディレクトリ外を指しています: {npz_name!r}')
    return str(resolved), key


def _resolve(value, json_dir: str | None):
    if isinstance(value, dict) and '$data' in value:
        if json_dir is None:
            raise ValueError(
                "$data参照はdict経路では非対応(load_irを使用)"
            )
        ref = value['$data']
        if not isinstance(ref, str):
            raise ValueError(f'$data参照は文字列である必要があります: {ref!r}')
        npz_path, key = _resolve_npz_path(ref, json_dir)
        # allow_pickle=False: npzサイドカー経由の任意コード実行を封じる
        with np.load(npz_path, allow_pickle=False) as npz:
            if key not in npz:
                raise ValueError(f'npzにキーがありません: {ref!r}')
            return npz[key]
    return value


def _params_from_dict(params: dict, json_dir: str | None) -> dict:
    """JSONから読んだparamsをin-memory表現へ戻す('body'は再帰的にIRGraph化)。"""
    out = {}
    for key, value in params.items():
        if key == 'body' and isinstance(value, dict) and 'nodes' in value:
            out[key] = _graph_from_doc(value, json_dir)
        else:
            out[key] = _resolve(value, json_dir)
    return out


def _graph_from_doc(doc: dict, json_dir: str | None) -> IRGraph:
    """dict(JSONデコード済み、または入れ子のbody文書)からIRGraphを構築する。

    load_ir/ir_from_dict共通実装。json_dir=Noneなら$data参照はエラーに
    なる(ir_from_dictの契約: エディタHTTP経路はnpzサイドカーを使わない)。
    """
    nodes = []
    for nd in doc['nodes']:
        params = _params_from_dict(nd['params'], json_dir)
        nodes.append(IRNode(
            id=nd['id'], type=nd['type'], params=params,
            train_info=nd.get('train_info', {}), meta=nd.get('meta', {}),
        ))
    return IRGraph(
        metadata=doc.get('metadata', {}),
        inputs=[IRPort(**p) for p in doc.get('inputs', [])],
        outputs=[IRPort(**p) for p in doc.get('outputs', [])],
        nodes=nodes,
        edges=[_edge_from_dict(e) for e in doc.get('edges', [])],
        format=doc.get('format', IR_FORMAT),
        version=doc.get('version', IR_VERSION),
    )


def load_ir(json_path: str, *, max_bytes: int = DEFAULT_MAX_IR_BYTES) -> IRGraph:
    """IR JSON(+npzサイドカー)を読み込む。

    Args:
        json_path: IR JSONのパス
        max_bytes: 読み込みを許可するJSONの最大バイト数(超過でValueError)

    Raises:
        ValueError: サイズ上限超過・JSON不正・format不一致・$data参照不正
    """
    size = os.path.getsize(json_path)
    if size > max_bytes:
        raise ValueError(
            f'IR JSONがサイズ上限を超えています: {size} > {max_bytes} バイト'
            f' ({json_path})'
        )
    try:
        with open(json_path, encoding='utf-8') as f:
            doc = json.load(f)
    except RecursionError as e:  # 深くネストしたJSONによるスタック枯渇
        raise ValueError(f'IR JSONのネストが深すぎます: {json_path}') from e
    except json.JSONDecodeError as e:
        raise ValueError(f'IR JSONの解析に失敗しました: {json_path} ({e})') from e

    if not isinstance(doc, dict):
        raise ValueError(f'IR JSONのトップレベルはオブジェクトである必要があります: {json_path}')
    if doc.get('format') != IR_FORMAT:
        raise ValueError(f"formatが不正: {doc.get('format')}")
    json_dir = os.path.dirname(os.path.abspath(json_path))
    return _graph_from_doc(doc, json_dir)


def ir_to_dict(graph: IRGraph) -> dict:
    """IRGraphをJSON化可能なdictへ変換する(E2、エディタHTTP経路)。

    save_ir と同一スキーマだが、npzサイドカーへ退避せず大配列も
    常にインライン展開する(エディタはHTTPで一括送受信するため)。
    """
    return _graph_to_doc(graph, arrays=None, npz_name=None, threshold=None)


def ir_from_dict(doc: dict) -> IRGraph:
    """dict(JSONデコード済み)からIRGraphを構築する(E2、エディタHTTP経路)。

    $data参照(npzサイドカー)はエディタ経路では非対応。
    ファイルからの読み込みは load_ir を使うこと。
    """
    if doc.get('format') != IR_FORMAT:
        raise ValueError(f"formatが不正: {doc.get('format')}")
    return _graph_from_doc(doc, None)


def tt_to_hex(tt_ints: list[int]) -> list[str]:
    """真理値表整数列をhex文字列列へ"""
    return [hex(int(v)) for v in tt_ints]


def tt_from_hex(tt_hex: list) -> list[int]:
    """hex文字列列(または整数列)を整数列へ"""
    return [int(v, 16) if isinstance(v, str) else int(v) for v in tt_hex]
