"""HDLエクスポートAPI。

REST:
    POST /api/export/hdl  body: {lang: 'verilog'|'vhdl', lower: true}
        → {source, filename, lang, estimate}
        フロー: validate(target='hdl') → (lower=trueなら)lower_to_primitives
                → to_verilog/to_vhdl → estimate_resources
        blackbox到達・add(sum)等のHDL不可IRは400+日本語エラー

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入):
    from . import hdl
    app.include_router(hdl.configure(STATE))

verifyオプション(iverilog/ghdl実機での3段等価性検証)は将来課題
(export/verify.py の run_iverilog_equivalence を非同期ジョブ化して繋ぐ)。
"""

import re
import subprocess
import tempfile

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from griffith_logic.export.estimate import estimate_resources
from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.export.verify import (
    ghdl_available,
    iverilog_available,
    run_ghdl_equivalence,
    run_iverilog_equivalence,
)
from griffith_logic.export.verilog import to_verilog
from griffith_logic.export.vhdl import to_vhdl
from griffith_logic.ir import IRValidationError, ir_from_dict, validate

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入)

FILE_EXT = {'verilog': '.v', 'vhdl': '.vhdl'}

# 検証入力ベクトルを全列挙する入力幅の上限(2^この値まで全網羅、超えたら無作為抽出)
_VERIFY_ENUM_MAX_BITS = 12


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


class HdlExportRequest(BaseModel):
    lang: str = Field(default='verilog', pattern='^(verilog|vhdl)$')
    lower: bool = True  # conv2d/or_pool2dをlut_layer列へ展開してから生成する


@router.post('/api/export/hdl')
def export_hdl(req: HdlExportRequest):
    """現在のIRからHDLソースとリソース見積りを生成する。"""
    if _STATE is None or _STATE.ir is None:
        raise HTTPException(
            status_code=400, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    try:
        graph = ir_from_dict(_STATE.ir)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')

    try:
        validate(graph, target='hdl')
    except IRValidationError as e:
        raise HTTPException(status_code=400, detail=f'HDL生成不可: {e}')

    hdl_graph = lower_to_primitives(graph) if req.lower else graph

    try:
        source = to_verilog(hdl_graph) if req.lang == 'verilog' \
            else to_vhdl(hdl_graph)
    except (NotImplementedError, IRValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f'HDL生成失敗: {e}')

    name = re.sub(r'[^0-9A-Za-z_]+', '_',
                  str(graph.metadata.get('name', 'model')))
    return {
        'source': source,
        'filename': f"{name.strip('_') or 'model'}{FILE_EXT[req.lang]}",
        'lang': req.lang,
        # 見積りは元IRに対して行う(conv2d等は内部で展開して計測され、
        # loweredフラグで展開の有無を報告する)
        'estimate': estimate_resources(graph),
    }


# ノードid接頭辞(IRValidationErrorメッセージ先頭の "id: ..." 形式)の抽出用
_NODE_ID_PREFIX = re.compile(r'^([0-9A-Za-z_]+):')


def _blocker_node_id(message: str, node_ids: set[str]) -> str | None:
    """検証エラーメッセージ先頭のノードid接頭辞を取り出す(実在するidのみ)。"""
    m = _NODE_ID_PREFIX.match(message)
    if m and m.group(1) in node_ids:
        return m.group(1)
    return None


def _count_draft_nodes(graph) -> int:
    """未離散化(train_info.initialized=False)ノード数を数える。

    draftノードはランダム初期化のまま真理値表が未確定で、HDL出力前に
    build/discretizeで確定させる必要がある(build.pyのinitialized規約)。
    """
    count = 0
    for n in graph.nodes:
        ti = n.train_info if isinstance(n.train_info, dict) else {}
        if ti.get('initialized', True) is False:
            count += 1
    return count


@router.get('/api/models/current/preflight')
def preflight():
    """現IRのHDL出荷プリフライト(可否・ブロッカー・見積り・draft数)。"""
    if _STATE is None or _STATE.ir is None:
        raise HTTPException(
            status_code=400, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    try:
        graph = ir_from_dict(_STATE.ir)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')

    node_ids = {n.id for n in graph.nodes}
    blockers: list[dict] = []
    warnings: list[str] = []
    estimate = None

    # まず構造検証(genericで通らなければHDL以前に出力不可)
    try:
        warnings = validate(graph, target='generic')
    except IRValidationError as e:
        blockers.append({
            'node_id': _blocker_node_id(str(e), node_ids),
            'kind': 'structure',
            'message': str(e),
        })
    else:
        # 構造OK: HDL固有ブロッカーを全ノード列挙(validate流用の網羅版)
        for n in graph.nodes:
            if n.type == 'blackbox':
                blockers.append({
                    'node_id': n.id, 'kind': 'blackbox',
                    'message': f'{n.id}: blackboxノードはHDL生成不可'
                    '(ハイブリッドモデルは論理部分のみエクスポートすること)',
                })
            elif n.type == 'add' and n.params.get('mode') == 'sum':
                blockers.append({
                    'node_id': n.id, 'kind': 'add_sum',
                    'message': f"{n.id}: add(mode='sum')はHDL生成不可"
                    '(整数和は幅拡張が必要。論理領域の終端で処理すること)',
                })
        estimate = estimate_resources(graph)

    return {
        'hdl_exportable': len(blockers) == 0,
        'blockers': blockers,
        'warnings': warnings,
        'estimate': estimate,
        'draft_node_count': _count_draft_nodes(graph),
        'verify_available': {
            'iverilog': iverilog_available(),
            'ghdl': ghdl_available(),
        },
    }


class HdlVerifyRequest(BaseModel):
    lang: str = Field(default='verilog', pattern='^(verilog|vhdl)$')
    # 検証ベクトル数の上限(全列挙が上限を超える場合の無作為抽出数)
    max_vectors: int = Field(default=256, ge=1, le=4096)


def _verify_inputs(in_width: int, max_vectors: int) -> np.ndarray:
    """検証用の0/1入力ベクトル (n, in_width) を作る。

    2^in_width が全列挙可能なら全網羅、超えるなら無作為抽出(再現性のため固定seed)。
    """
    total = 1 << in_width
    if in_width <= _VERIFY_ENUM_MAX_BITS and total <= max_vectors:
        pats = np.arange(total, dtype=np.int64)
        bits = ((pats[:, None] >> np.arange(in_width)) & 1)
        return bits.astype(np.int64)
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(max_vectors, in_width), dtype=np.int64)


@router.post('/api/export/hdl/verify')
def verify_hdl(req: HdlVerifyRequest):
    """生成HDLとgolden model(simulate)の等価性を処理系があれば検証する。

    iverilog/ghdlが未導入の環境ではavailable=Falseを返し実行しない(オプショナル)。
    """
    if _STATE is None or _STATE.ir is None:
        raise HTTPException(
            status_code=400, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    try:
        graph = ir_from_dict(_STATE.ir)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')

    try:
        validate(graph, target='hdl')
    except IRValidationError as e:
        raise HTTPException(status_code=400, detail=f'HDL生成不可: {e}')

    available = iverilog_available() if req.lang == 'verilog' else ghdl_available()
    if not available:
        return {
            'available': False, 'passed': None, 'n_vectors': 0,
            'stdout': '', 'lang': req.lang,
        }

    if not graph.inputs:
        raise HTTPException(status_code=400, detail='入力ポートがないIRは検証不可')
    x = _verify_inputs(graph.inputs[0].width, req.max_vectors)

    runner = run_iverilog_equivalence if req.lang == 'verilog' \
        else run_ghdl_equivalence
    try:
        with tempfile.TemporaryDirectory() as workdir:
            result = runner(graph, x, workdir)
    except (subprocess.CalledProcessError, NotImplementedError,
            IRValidationError, ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f'HDL検証失敗: {e}')

    return {
        'available': True,
        'passed': bool(result['passed']),
        'n_vectors': int(result['n_vectors']),
        'stdout': result['stdout'],
        'lang': req.lang,
    }
