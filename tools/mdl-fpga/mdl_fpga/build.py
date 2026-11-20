"""mdl-fpga build: HDL 生成 → 事前判定 → 合成 → .sof。

「見る前に合っている」原則: 合成前に必ず RTL bit 一致(presim)を通す。
advisor が「載らない」と判定したら合成に入らず言い切る(§9 エラー3分類の設計)。

2 つの実装方式:
  spatial  モデルごとに論理を焼く(最速。載る規模に配線天井)
  overlay  固定回路 + メモリ差替(容量は M10K 律速)。計算型配線(rot 系)の
           IR ならゲート記述が op 4bit だけになり、配線表・座標表が消える。
           自由配線(置換・ランダム・学習配線)の IR は保存型(記録 32bit)
           で載る。どちらにするかは overlay.wiring(既定 auto)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from griffith_logic.export.lowering import lower_to_primitives

from .advisor import advise
from .backends import FlowResult, get_backend
from .board_db import resolve_board
from .check import default_workdir, run_check
from .hdlgen.core import generate_core
from .hdlgen.ports import TopPorts
from .hdlgen.wrapper import copy_static_hdl, generate_wrapper
from .io_link import resolve_uart_link
from .ir_io import convert, load_graph
from .manifest import is_manifest_file, load_manifest
from .toolchain import Toolchain, discover


@dataclass
class BuildResult:
    ok: bool
    workdir: str = ''
    sof: str | None = None
    messages: list[str] = field(default_factory=list)
    flow: FlowResult | None = None
    board: str = ''   # flash 誘導コマンドに --board を付けるため保持

    def render(self) -> str:
        lines = [f'build: {"OK" if self.ok else "FAIL"}']
        lines.extend(f'  {m}' for m in self.messages)
        if self.flow is not None:
            lines.append(self.flow.render())
        if self.sof:
            board_opt = f' --board {self.board}' if self.board else ''
            lines.append(f'  次の一手: uv run python tools/mdl-fpga/cli.py '
                         f'flash "{self.sof}"{board_opt}')
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'workdir': self.workdir, 'sof': self.sof,
                'messages': self.messages,
                'flow': ({'stages': self.flow.stages, **self.flow.summary}
                         if self.flow is not None else None)}


def resolve_clock_hz(explicit: int | None, board_def: dict) -> int:
    """clock_hz の解決: 明示値(manifest / CLI)→ board の clock.hz → 50MHz。

    manifest がある場合は manifest の値(未指定 None なら board へ)、
    無い場合は CLI の値(未指定 None なら board へ)を explicit に渡す。
    """
    if explicit:
        return int(explicit)
    hz = (board_def.get('clock') or {}).get('hz')
    return int(hz) if hz else 50_000_000


def run_build(model: str | Path, workdir: str | Path | None = None,
              board: str = 'de0-cv', clock_hz: int | None = None,
              name: str = 'top', mode: str = 'spatial',
              skip_presim: bool = False, selftest: bool = False,
              tc: Toolchain | None = None,
              overlay_v: int | None = None,
              overlay_vectors: int | None = None,
              clock_div: int | None = None,
              overlay_wiring: str | None = None,
              presim_vectors: int = 4,
              images=None, labels=None) -> BuildResult:
    tc = tc or discover()
    res = BuildResult(ok=False)

    baud = 115200
    link = None
    observe: tuple[str, ...] = ()
    ov_v, ov_n, ov_div, ov_wiring = 8, 1000, 2, 'auto'
    if is_manifest_file(model):
        mf = load_manifest(model)
        board, clock_hz, mode = mf.board, mf.clock_hz, mf.mode
        baud = mf.effective_baud()
        link = mf.io.link
        observe = tuple(mf.debug.observation_points)
        ov_v, ov_n, ov_div = mf.overlay.v, mf.overlay.n_vectors, \
            mf.overlay.clock_div
        ov_wiring = mf.overlay.wiring
        res.messages.append(f'manifest: {model}(ir={mf.ir_path()})')
        model = mf.ir_path()
    # 明示指定(CLI/API)は manifest より優先する
    ov_v = overlay_v if overlay_v is not None else ov_v
    ov_n = overlay_vectors if overlay_vectors is not None else ov_n
    ov_div = clock_div if clock_div is not None else ov_div
    ov_wiring = overlay_wiring if overlay_wiring is not None else ov_wiring

    if mode not in ('spatial', 'overlay'):
        res.messages.append(
            f'mode={mode} は未対応(spatial | overlay)。'
            '--mode か manifest の mode を修正')
        return res

    board_def = resolve_board(board)
    res.board = board
    clock_hz = resolve_clock_hz(clock_hz, board_def)
    # uart 配線は board(物理事実)ではなく io.link(プロジェクト側の判断)から
    # 解決する。失敗(未収録ピン等)は合成前に早期に言い切る
    resolved_uart = resolve_uart_link(board_def, link)
    if resolved_uart.source == 'manifest':
        res.messages.append(
            f'io.link: uart rx={resolved_uart.rx_pin} tx={resolved_uart.tx_pin} '
            f'@ {resolved_uart.baud} baud')
    if board_def.get('tier', 3) != 1:
        res.messages.append(
            f'注意: {board_def["name"]} は Tier{board_def["tier"]}'
            '(実機未検証。動作を約束できない)')

    # 1) convert(検証・blackbox 拒否)
    conv = convert(model)
    if not conv.ok:
        res.messages.append(conv.render())
        return res

    graph = lower_to_primitives(load_graph(model))

    # 2) advisor(搭載可能性の事前判定)
    adv = advise(graph, board_def)
    res.messages.append(adv.render())
    if mode == 'spatial' and not adv.spatial_ok:
        res.messages.append('advisor 判定により合成を中止(時間の浪費を防ぐ)')
        return res
    if mode == 'overlay' and adv.overlay_ok is False:
        res.messages.append('advisor 判定により合成を中止(時間の浪費を防ぐ)')
        return res

    wd = Path(workdir) if workdir else default_workdir(model, 'build')
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)

    if mode == 'overlay':
        return _build_overlay(res, graph, board_def, wd, name=name,
                              clock_hz=clock_hz, v=ov_v, n_vectors=ov_n,
                              clock_div=ov_div, wiring=ov_wiring,
                              skip_presim=skip_presim,
                              presim_vectors=presim_vectors,
                              images=images, labels=labels,
                              selftest=selftest, tc=tc)

    # 3) 合成前 RTL bit 一致(見る前に合っている)
    if not skip_presim:
        chk = run_check(model, workdir=wd / 'presim', n_vectors=64, tc=tc)
        if chk.edges.get('ref_vs_rtl(a)') == 'SKIP' or not chk.passed:
            if 'SKIP' in chk.edges.values():
                res.messages.append('presim スキップ(iverilog 無し)。'
                                    '--skip-presim で明示的に飛ばすことも可能')
            else:
                res.messages.append('合成前 RTL 検証が FAIL。合成を中止:\n'
                                    + chk.render())
                return res
        else:
            res.messages.append(f'presim: PASS({chk.n_vectors} ベクトル)')

    # 4) HDL 一式をバックエンドのプロジェクトディレクトリへ
    backend = get_backend(board_def)
    projdir = wd / backend.project_subdir
    projdir.mkdir(parents=True, exist_ok=True)
    gen = generate_core(graph, observe=observe)
    (projdir / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
    import json as _json
    (projdir / 'name_map.json').write_text(
        _json.dumps(gen.name_map, ensure_ascii=False, indent=2),
        encoding='utf-8')
    wrap = generate_wrapper(gen.in_width, gen.out_width, top_name=name,
                            clock_hz=clock_hz, baud=baud,
                            with_selftest=selftest,
                            ports=TopPorts.from_board(board_def, resolved_uart))
    (projdir / f'{name}.v').write_text(wrap.top_verilog, encoding='utf-8')
    statics = copy_static_hdl(projdir)
    sources = [f'{name}.v', 'mdl_core.v', *statics]

    if selftest:
        from .check import expected_packed
        from .hdlgen.selftest import generate_selftest
        from .vectors import gen_inputs, pack_bits
        x = gen_inputs(gen.in_width, 256)
        if len(x) > 256:
            x = x[:256]
        exp_packed, _ = expected_packed(graph, x)
        st = generate_selftest(gen, pack_bits(x), exp_packed)
        (projdir / 'selftest.v').write_text(st.verilog, encoding='utf-8')
        sources.append('selftest.v')
        res.messages.append(
            f'selftest: {st.n_vectors} ベクトルを焼き込み'
            '(実機 LEDR[0] 点灯 = 全ベクトル bit 一致)')

    backend.write_project_files(projdir, name=name, board=board_def,
                                sources=sources, clock_hz=clock_hz,
                                groups=wrap.groups,
                                uart=resolved_uart.to_uart_dict())
    res.messages.append(
        f'HDL/QSF 生成完了: {projdir}(要求 {wrap.req_payload_bytes + 3}B / '
        f'応答 {wrap.rsp_payload_bytes + 3}B @ {baud} baud)')

    # 5) 合成フロー(バックエンド)
    avail, avail_msg = backend.available(tc)
    if not avail:
        res.messages.append(avail_msg)
        return res

    flow = backend.run_flow(projdir, name, tc=tc, board=board_def,
                            clock_hz=clock_hz)
    res.flow = flow
    res.ok = flow.ok
    res.sof = flow.sof
    return res


def _build_overlay(res: BuildResult, graph, board_def: dict, wd: Path, *,
                   name: str, clock_hz: int, v: int, n_vectors: int,
                   clock_div: int, skip_presim: bool, presim_vectors: int,
                   images, labels, selftest: bool,
                   tc: Toolchain, wiring: str = 'auto') -> BuildResult:
    """オーバーレイのビルド(生成 → presim → プロジェクト → 合成)。

    presim は overlay の bit 一致検証(ir.simulate ⇔ overlay RTL)。実ネットの
    vvp は遅いので既定は少数枚で、全 n_vectors はオンチップ ROM に焼いて
    実機(ISSP)で照合する。wiring(auto / computed / stored)でテンプレートと
    .mem の種類が決まる。
    """
    from .check import prepare_overlay_workdir, run_overlay_check
    from .hdlgen.overlay import (
        OverlayUnsupported,
        estimate_m10k,
        plan_overlay,
        wiring_label,
    )

    if selftest:
        res.messages.append(
            '注意: --selftest は overlay では無視する'
            '(検証システム自体がオンチップ自己検証なので同梱済み)')

    try:
        plan = plan_overlay(graph, v=v, wiring=wiring)
    except OverlayUnsupported as e:
        res.messages.append(f'overlay 化できない: {e}')
        return res
    est = estimate_m10k(plan, n_images=n_vectors)
    res.messages.append(
        f'overlay: V={v} W={plan.w} DEPTH={plan.depth} '
        f'gates={plan.total_gates}(配線 {wiring_label(plan)})')
    res.messages.append(
        f'  予測 {plan.cycles_per_image} cyc/推論 → '
        f'{clock_hz / clock_div / plan.cycles_per_image:.1f} inf/s '
        f'@ {clock_hz / clock_div / 1e6:.3f} MHz(clock_div={clock_div})')
    res.messages.append(
        f'  M10K 見積り(近似): ゲート ROM {est["op_rom"]} / 活性 '
        f'{est["activation"]} / 画像 ROM {est["image_rom"]} / 計 {est["total"]} '
        f'/ {board_def["resources"]["m10k_blocks"]}')
    for warn in plan.warnings:
        res.messages.append(f'  注意: {warn}')

    backend = get_backend(board_def)
    projdir = wd / backend.project_subdir
    projdir.mkdir(parents=True, exist_ok=True)

    if not skip_presim:
        n_pre = max(1, min(presim_vectors, n_vectors))
        chk = run_overlay_check(graph, wd / 'presim', n_vectors=n_pre, v=v,
                                clock_div=clock_div, wiring=plan.wiring,
                                tc=tc)
        if 'SKIP' in chk.edges.values():
            res.messages.append('presim スキップ(iverilog 無し)。'
                                '--skip-presim で明示的に飛ばすことも可能')
        elif not chk.passed:
            res.messages.append('合成前 overlay 検証が FAIL。合成を中止:\n'
                                + chk.render())
            return res
        else:
            res.messages.append(f'presim: PASS({n_pre} ベクトル、'
                                'ir.simulate ⇔ overlay RTL)')

    ports = TopPorts.from_board(board_def)
    plan, gen, sysgen, golden, used = prepare_overlay_workdir(
        graph, projdir, images=images, labels=labels, n_vectors=n_vectors,
        v=v, clock_div=clock_div, clock_hz=clock_hz, top_name=name,
        wiring=plan.wiring, ports=ports)
    rom_kind = 'ゲート記録 ROM' if plan.stored else 'op ROM'
    res.messages.append(
        f'生成完了: {projdir}({rom_kind} {plan.nrom} 本 x '
        f'{plan.gate_words} 語、検証ベクトル {len(used)} 枚)')
    if golden.get('accuracy') is not None:
        res.messages.append(
            f'  golden 精度: {golden["correct"]}/{golden["n_images"]}'
            f'({golden["accuracy"]:.2%})')

    backend.write_project_files(projdir, name=name, board=board_def,
                                sources=sysgen.sources, clock_hz=clock_hz,
                                groups=sysgen.groups, uart=None,
                                clock_div=clock_div)

    avail, avail_msg = backend.available(tc)
    if not avail:
        res.messages.append(avail_msg)
        return res
    flow = backend.run_flow(projdir, name, tc=tc, board=board_def,
                            clock_hz=clock_hz, clock_div=clock_div)
    res.flow = flow
    res.ok = flow.ok
    res.sof = flow.sof
    m10k = (flow.summary or {}).get('m10k_by_entity')
    if m10k:
        res.messages.append(
            f'fit 実測 M10K: ゲート ROM {m10k.get("gate_rom")} / 活性 '
            f'{m10k.get("activation")} / 画像 ROM {m10k.get("image_rom")} '
            f'/ 計 {m10k.get("total")}')
    if flow.ok:
        res.messages.append(
            '実機検証: flash 後に '
            f'`cli.py issp "{projdir}"` で start を叩き、'
            '一致数 / サイクル / inf/s を読む(7セグにも一致数が出る)')
    return res
