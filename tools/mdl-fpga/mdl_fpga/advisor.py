"""搭載可能性の事前判定(advisor)。

LUT 数だけでは配線天井が見えない(fpga-sizing-guide R2)ため、
実測則ベースで spatial / overlay の両方式を判定し、
初心者向けの言葉(原因・次の一手)で返す。

資源単位はボードごとに異なる(Cyclone V=ALM / MAX 10=LE 等)ため、
board JSON の resources.unit と limits の `gates_per_<unit>` /
`spatial_<unit>_fill` を動的参照する(de0-cv の既存キー名は不変)。
実測則が無い未較正ボードは保守的既定で判定し、その旨を必ず明示する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir.schema import IRGraph

# 未較正ボード用の保守的既定。DE0-CV 実測則(gates_per_alm=4.0 /
# spatial_alm_fill=0.35 / spatial_gates_max=24000、fpga-sizing-guide R1-R14)を
# 他ボードへ外挿しない方針のため、その半分程度に保守化した下限側の値。
# 「搭載可能」と言って載らない事故を防ぐ側に倒す。実測較正後は boards/*.json の
# limits に記録し、こちらは使われなくなる。
CONSERVATIVE_LIMITS = {
    'gates_per_unit': 2.0,
    'spatial_fill': 0.25,
    'spatial_gates_max': 8000,
}


def count_gates(graph: IRGraph) -> int:
    """lowering 後の LUT ゲート総数(= 空間焼込みの規模指標)。"""
    lowered = lower_to_primitives(graph)
    return sum(n.params['out_width'] for n in lowered.nodes
               if n.type == 'lut_layer')


@dataclass
class Advice:
    gates: int
    spatial_ok: bool
    overlay_ok: bool | None          # None = 判定不能(overlay 上限が未定義)
    spatial_alm_needed: int          # 後方互換の名前(実体は unit 数)
    messages: list[str] = field(default_factory=list)
    unit: str = 'alm'
    calibrated: bool = True
    # 計算型配線(rot 系)で overlay 化できるか。True なら配線表が消えるので
    # 上限は limits.overlay_computed_gates_max(記述 4bit/ゲート)になる
    overlay_computed: bool = False
    overlay_reason: str = ''
    overlay_m10k: dict | None = None
    # 実際に生成される配線の持ち方('computed' / 'stored' / '' = 生成不可)。
    # stored(自由配線)は記録 32bit/ゲートなので上限は
    # limits.overlay_m10k_gates_max(アドレス指定型の実測上限)になる
    overlay_wiring: str = ''

    @property
    def spatial_units_needed(self) -> int:
        return self.spatial_alm_needed

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [f'advisor: ゲート数 {self.gates}(lowering 後)']
        lines.extend(f'  {m}' for m in self.messages)
        return '\n'.join(lines)


def advise(graph: IRGraph, board: dict) -> Advice:
    gates = count_gates(graph)
    lim = board['limits']
    res = board['resources']
    unit = res.get('unit', 'alm')
    unit_total = res[unit]

    gp = lim.get(f'gates_per_{unit}')
    fill = lim.get(f'spatial_{unit}_fill')
    calibrated = bool(lim.get('calibrated', gp is not None and fill is not None))

    msgs = []
    if gp is None or fill is None:
        calibrated = False
        gp = CONSERVATIVE_LIMITS['gates_per_unit']
        fill = CONSERVATIVE_LIMITS['spatial_fill']
    sg_max = lim.get('spatial_gates_max')
    if sg_max is None:
        calibrated = False
        sg_max = CONSERVATIVE_LIMITS['spatial_gates_max']
    if not calibrated:
        msgs.append(
            f'未較正ボード({board.get("name", "?")}): 実測則が無いため保守的'
            f'既定(gates_per_{unit}={gp} / fill={fill} / 上限 {sg_max} ゲート)'
            'で判定。実機で較正後 boards/*.json の limits に記録する')

    # R1×R2: 必要unit数 ≈ (gates/gates_per_unit) / fill(ランダム配線の実効充填率)
    unit_needed = int(gates / gp / fill)
    spatial_ok = gates <= sg_max and unit_needed <= unit_total

    wiring, reason, m10k = _overlay_plan(graph)
    computed = wiring == 'computed'
    overlay_max = lim.get('overlay_gates_max', lim.get('overlay_m10k_gates_max'))
    if computed and lim.get('overlay_computed_gates_max'):
        overlay_max = lim['overlay_computed_gates_max']
    overlay_ok: bool | None = None if overlay_max is None else gates <= overlay_max

    uname = unit.upper()
    if spatial_ok:
        msgs.append(
            f'空間焼込み(spatial): 搭載可能(必要{uname}≈{unit_needed}/'
            f'{unit_total}。実測則 R2 による)')
    else:
        msgs.append(
            f'空間焼込み(spatial): 搭載不可 — {board["name"]}'
            f'(ゲート {gates} > 上限目安 {sg_max}。'
            f'配線/配置は {uname} 充填 {fill:.0%} 相当で破綻する実測則。'
            '根拠は limits.note 参照)。')
    if overlay_ok is None:
        msgs.append(
            'オーバーレイ(overlay): このボードは overlay 上限が未定義のため'
            '判定不能(較正後に limits へ overlay_gates_max を記録)')
    elif overlay_ok:
        msgs.append(
            f'オーバーレイ(overlay): 搭載可能(M10K 律速の上限 '
            f'{overlay_max} ゲート以内)')
        if computed:
            msgs.append(
                f'  計算型配線(rot 系)として生成できる: ゲート記述 4bit/ゲート'
                f'({reason})')
        elif wiring == 'stored':
            msgs.append(
                '  自由配線なので保存型(アドレス指定型)で生成する: '
                f'ゲート記録 32bit/ゲート({reason})。上限は '
                f'{overlay_max} ゲート')
        else:
            msgs.append(
                '  ただしこの IR はオーバーレイの形(一様幅の 2 入力 FFN + '
                f'group_sum)にできない({reason})')
        if m10k:
            msgs.append(
                f'  M10K 見積り: ゲート ROM {m10k["gate_rom"]} + 活性 '
                f'{m10k["activation"]} blk(画像 ROM は枚数依存。近似値)')
    else:
        msgs.append(
            'オーバーレイ(overlay): M10K 容量超過。モデルの縮小か SDRAM 方式'
            '(v2)を検討')
    if not spatial_ok and overlay_ok:
        msgs.append('次の一手: --mode overlay であれば搭載可能')

    return Advice(gates=gates, spatial_ok=spatial_ok, overlay_ok=overlay_ok,
                  spatial_alm_needed=unit_needed, messages=msgs,
                  unit=unit, calibrated=calibrated,
                  overlay_computed=computed, overlay_reason=reason,
                  overlay_m10k=m10k, overlay_wiring=wiring)


def _overlay_plan(graph: IRGraph) -> tuple[str, str, dict | None]:
    """オーバーレイの配線の持ち方を配線の全数照合で判定する。

    計算型(rot 系の閉形式に全層一致)を優先し、外れれば保存型(自由配線)に
    落とす。どちらでも生成できない構成(一様幅でない等)は '' を返す。

    Returns:
        (配線の持ち方, 理由の 1 行, M10K 見積り or None)
    """
    from .hdlgen.overlay import OverlayUnsupported, estimate_m10k, plan_overlay

    try:
        plan = plan_overlay(graph, v=8, wiring='computed')
    except OverlayUnsupported as e:
        first = str(e).split('。')[0]
        try:
            plan = plan_overlay(graph, v=8, wiring='stored')
        except OverlayUnsupported as e2:
            return '', str(e2).split('。')[0], None
        return 'stored', first, estimate_m10k(plan)
    kinds = '/'.join(sorted({ly.kind for ly in plan.layers}))
    return 'computed', \
        (f'配線 {kinds}、V=8 で {plan.cycles_per_image} cyc/推論'), \
        estimate_m10k(plan)


def run_advise(model: str | Path, board: str | None = None) -> Advice:
    """CLI/ビューア API 用の単独実行(manifest / IR の両方を受ける)。"""
    from .board_db import resolve_board
    from .ir_io import load_graph
    from .manifest import is_manifest_file, load_manifest

    if is_manifest_file(model):
        mf = load_manifest(model)
        board = board or mf.board
        model = mf.ir_path()
    board = board or 'de0-cv'
    graph = load_graph(model)
    return advise(graph, resolve_board(board))
