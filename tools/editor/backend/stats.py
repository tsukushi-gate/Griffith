"""統計API。

REST:
    GET /api/stats
        → STATE.ir の見積り統計(export/estimate.py の estimate_resources を
          そのまま呼び出し流用)+学習パラメータ数(lut_layerの 2^n_inputs×out_width
          総和)を返す。
        blackboxノードを含む場合は算出可能な部分値のみ返し partial=true とする
        (blackboxは実数演算の外部モジュールでありIRだけからは内部構成が不明なため)。
        IR未設定は404、IRが不正(validate失敗)は422で日本語エラーを返す。

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入):
    from . import stats
    app.include_router(stats.configure(STATE))
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from griffith_logic.export.estimate import estimate_resources
from griffith_logic.export.lowering import LOWERABLE_TYPES, lower_to_primitives
from griffith_logic.ir import ir_from_dict, validate

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


class StatsResponse(BaseModel):
    """統計APIレスポンス(extra='forbid'でスキーマドリフトを検出)。"""

    model_config = ConfigDict(extra='forbid')

    total_luts: int
    lut_counts: dict[int, int]
    logic_depth: int
    wire_count: int
    group_sums: int
    node_counts: dict[str, int]
    lowered: bool
    num_params: int
    partial: bool


def _count_params(graph) -> int:
    """lut_layerノードの学習パラメータ数を集計する(2^n_inputs × out_width)。

    IR上のlut_layerはparameterization('softmax'/'independent'等)によらず
    出力ニューロン1つあたり2^n_inputs個のロジット/値を持つ(core/basis.py・
    layers/lut.pyの規約)ため、n_inputsとout_widthのみから算出できる。
    """
    total = 0
    for n in graph.nodes:
        if n.type == 'lut_layer':
            n_inputs = int(n.params['n_inputs'])
            out_width = int(n.params['out_width'])
            total += (1 << n_inputs) * out_width
    return total


@router.get('/api/stats', response_model=StatsResponse)
def stats_endpoint():
    """現在のIRのリソース見積り+学習パラメータ数を返す。"""
    if _STATE is None or _STATE.ir is None:
        raise HTTPException(
            status_code=404, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    try:
        graph = ir_from_dict(_STATE.ir)
        validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')

    estimate = estimate_resources(graph)

    # num_paramsはestimate_resourcesと同じlowering規約(conv2d/or_pool2dを
    # lut_layer列へ展開)で数える必要があるため、同じ展開処理を適用する
    # (estimate.py本体は変更せず読み取り流用のみ)。
    lowered_graph = (
        lower_to_primitives(graph)
        if any(n.type in LOWERABLE_TYPES for n in graph.nodes)
        else graph
    )
    num_params = _count_params(lowered_graph)

    partial = any(n.type == 'blackbox' for n in graph.nodes)

    return StatsResponse(
        total_luts=estimate['total_luts'],
        lut_counts=estimate['lut_counts'],
        logic_depth=estimate['logic_depth'],
        wire_count=estimate['wire_count'],
        group_sums=estimate['group_sums'],
        node_counts=estimate['node_counts'],
        lowered=estimate['lowered'],
        num_params=num_params,
        partial=partial,
    )
