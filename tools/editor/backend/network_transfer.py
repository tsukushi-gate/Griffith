"""学習ネット⇔推論ネット間の重み転写API。

REST:
    POST /api/networks/transfer
        body={source_ir: dict, target_ir: dict}
        → target_ir の対象ノード(lut_layer/conv2d)へ、同名(id一致)かつ
          型・幅シグネチャが一致するsource側ノードの重みを転写して返す。
          転写結果は {target_ir, transferred, skipped} で返す。

純関数 `transfer_weights(source, target) -> dict` として実装本体を分離しており、
FastAPI非依存でも呼び出せる(単体テスト容易化)。

main.py への組み込み手順:
    from . import network_transfer
    app.include_router(network_transfer.router)
    (STATEを使わない純関数的APIのため configure() は不要)

設計:
  - `griffith_logic.ir` の ir_from_dict/ir_to_dict/validate、および
    `griffith_logic.ir.validate` の node_in_width/node_out_width を
    読み取り専用で流用する(本体非改変)。
  - 転写対象はtarget側の lut_layer/conv2d ノードのみ(TRANSFERABLE_TYPES)。
  - 幅シグネチャ(n_inputs/in_width/out_width)が一致しないノードがあれば、
    全件チェックを終えるまでraiseせず mismatches に集約し、1件でもあれば
    まとめて WeightTransferError を送出する(部分適用を絶対にしない)。
  - 転写後は validate(target_graph) を再実行し、IRとしての整合性を保証する。
"""

from __future__ import annotations

import copy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from griffith_logic.ir import ir_from_dict, ir_to_dict, validate
from griffith_logic.ir.validate import node_in_width, node_out_width

router = APIRouter()

# 重み転写の対象ノード型(学習ネット⇔推論ネットで重みを持つ層のみ)
TRANSFERABLE_TYPES = {'lut_layer', 'conv2d'}


class WeightTransferError(ValueError):
    """転写対象ノードの幅(n_inputs/in_width/out_width)不一致(全件検証後に一括判定)"""

    def __init__(self, mismatches: list[str]):
        self.mismatches = mismatches
        super().__init__('; '.join(mismatches))


def _width_signature(node) -> tuple[int, int, int]:
    """ノードの(n_inputs, in_width, out_width)シグネチャを算出する。"""
    n_inputs = int(node.params['n_inputs'])
    in_width = node_in_width(node, 0)
    out_width = node_out_width(node)
    return n_inputs, in_width, out_width


def _apply_transfer(t_node, s_node) -> None:
    """1ノード分の重みをs_nodeからt_nodeへ実際にコピーする。"""
    if t_node.type == 'lut_layer':
        t_node.params['truth_tables'] = copy.deepcopy(s_node.params['truth_tables'])
        t_node.params['indices'] = copy.deepcopy(s_node.params['indices'])
    elif t_node.type == 'conv2d':
        t_node.params['truth_tables'] = copy.deepcopy(s_node.params['truth_tables'])
        t_node.params['indices_flat'] = copy.deepcopy(s_node.params['indices_flat'])


def transfer_weights(source: dict, target: dict) -> dict:
    """学習ネット(source)の重みを推論ネット(target)へ転写する。

    Args:
        source: 転写元IR(dict)
        target: 転写先IR(dict、この構造のうち対象ノードのparamsのみ書き換える)

    Returns:
        {'target_ir': dict, 'transferred': list[str], 'skipped': list[dict]}

    Raises:
        WeightTransferError: 幅シグネチャが一致しない対象ノードが1件以上ある場合
        ValueError/KeyError/TypeError/AssertionError: 不正なIRの場合(呼び出し元で捕捉)
    """
    source_graph = ir_from_dict(source)
    target_graph = ir_from_dict(target)

    skipped: list[dict] = []
    mismatches: list[str] = []
    plan: list[tuple] = []

    for t_node in target_graph.nodes:
        if t_node.type not in TRANSFERABLE_TYPES:
            continue

        try:
            s_node = source_graph.node_by_id(t_node.id)
        except KeyError:
            skipped.append({'id': t_node.id, 'reason': 'sourceに同名ノードが存在しない'})
            continue

        if s_node.type != t_node.type:
            skipped.append({
                'id': t_node.id,
                'reason': f'型不一致(source={s_node.type}, target={t_node.type})',
            })
            continue

        t_sig = _width_signature(t_node)
        s_sig = _width_signature(s_node)
        if s_sig != t_sig:
            mismatches.append(
                f'ノード {t_node.id}: 幅不一致 '
                f'(source: n_inputs={s_sig[0]}, in_width={s_sig[1]}, out_width={s_sig[2]} / '
                f'target: n_inputs={t_sig[0]}, in_width={t_sig[1]}, out_width={t_sig[2]})'
            )
            continue

        plan.append((t_node, s_node))

    if mismatches:
        # 部分適用を絶対にしない: raiseする前にtargetへは一切書き込まない
        raise WeightTransferError(mismatches)

    transferred: list[str] = []
    for t_node, s_node in plan:
        _apply_transfer(t_node, s_node)
        transferred.append(t_node.id)

    # 転写後IRの整合性を再確認する(失敗したら想定外バグとして例外をそのまま伝播させる)
    validate(target_graph)

    return {
        'target_ir': ir_to_dict(target_graph),
        'transferred': transferred,
        'skipped': skipped,
    }


class TransferRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_ir: dict
    target_ir: dict


class SkippedItem(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    reason: str


class TransferResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    target_ir: dict
    transferred: list[str]
    skipped: list[SkippedItem]


@router.post('/api/networks/transfer', response_model=TransferResponse)
def transfer_networks_endpoint(req: TransferRequest) -> TransferResponse:
    """学習ネット⇔推論ネット間の重み転写エンドポイント。"""
    try:
        result = transfer_weights(req.source_ir, req.target_ir)
    except WeightTransferError as e:
        raise HTTPException(
            status_code=422,
            detail='幅不一致のため転写できません: ' + '; '.join(e.mismatches),
        )
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')
    return TransferResponse(
        target_ir=result['target_ir'],
        transferred=result['transferred'],
        skipped=[SkippedItem(**s) for s in result['skipped']],
    )
