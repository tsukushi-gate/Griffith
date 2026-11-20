"""mdl-fpga 統一 CLI(構想書 §5.1 の動詞の皮)。

使い方(リポジトリ直下から):
    uv run python tools/mdl-fpga/cli.py doctor
    uv run python tools/mdl-fpga/cli.py convert model.ir.json
    uv run python tools/mdl-fpga/cli.py check  model.ir.json
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_doctor(args) -> int:
    from .doctor import run_doctor
    rep = run_doctor(probe_jtag=not args.no_jtag)
    if args.json:
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(rep.render())
    return 0


def _cmd_detect(args) -> int:
    from .detect import run_detect
    probes = []
    if not args.no_serial:
        probes.append('serial')
    probes.append('usb')
    if not args.no_jtag:
        probes.append('jtag')
    res = run_detect(probes=tuple(probes))
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(res.render())
    return 0 if res.candidates else 1


def _cmd_convert(args) -> int:
    from .ir_io import convert
    result = convert(args.model, out_path=args.out)
    print(result.render())
    return 0 if result.ok else 1


def _cmd_sim(args) -> int:
    from .check import run_sim_only
    res = run_sim_only(args.model, workdir=args.workdir, n_vectors=args.vectors,
                       seed=args.seed)
    print(res.render())
    return 0 if res.passed else 1


def _cmd_check(args) -> int:
    if args.pipelined:
        from .check import run_check_pipelined
        res = run_check_pipelined(args.model, workdir=args.workdir,
                                  n_vectors=args.vectors, seed=args.seed)
    else:
        from .check import run_check
        res = run_check(args.model, workdir=args.workdir,
                        n_vectors=args.vectors, seed=args.seed,
                        cross_check=not args.no_cross)
    print(res.render())
    return 0 if res.passed else 1


def _cmd_advise(args) -> int:
    from .advisor import run_advise
    res = run_advise(args.model, board=args.board)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(res.render())
    return 0 if (res.spatial_ok or res.overlay_ok) else 1


def _cmd_build(args) -> int:
    from .build import run_build
    res = run_build(args.model, workdir=args.workdir, board=args.board,
                    clock_hz=args.clock_hz, name=args.name, mode=args.mode,
                    skip_presim=args.skip_presim, selftest=args.selftest,
                    overlay_v=args.overlay_v,
                    overlay_vectors=args.overlay_vectors,
                    clock_div=args.clock_div,
                    overlay_wiring=args.overlay_wiring)
    print(res.render())
    return 0 if res.ok else 1


def _cmd_issp(args) -> int:
    from .issp import run_issp
    res = run_issp(args.projdir, index=args.index, timeout_ms=args.timeout_ms)
    print(res.render())
    return 0 if res.ok else 1


def _cmd_flash(args) -> int:
    from .flash import run_flash
    res = run_flash(args.bitstream, cable=args.cable, via=args.via,
                    ofl_cable=args.ofl_cable, board=args.board)
    print(res.render())
    return 0 if res.ok else 1


def _cmd_bundle(args) -> int:
    from .bundle import run_bundle
    formats = tuple(f for f, skip in (('svf', args.no_svf),
                                      ('rbf', args.no_rbf)) if not skip)
    res = run_bundle(args.workdir, manifest=args.manifest, board=args.board,
                     out=args.out, formats=formats, app=not args.no_app,
                     flasher=not args.no_flasher,
                     flasher_pack=args.flasher_pack)
    print(res.render())
    return 0 if res.ok else 1


def _cmd_view(args) -> int:
    from .viewer.server import serve
    serve(args.model, port=args.port, workdir=args.workdir,
          open_browser=not args.no_browser, pipelined=args.pipelined,
          system_dir=args.system)
    return 0


def _cmd_up(args) -> int:
    from .up import run_up
    res = run_up(args.model, board=args.board, workdir=args.workdir,
                 flash=not args.no_flash, hw_check=args.hw_check,
                 progress=print)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(res.render())
    return 0 if res.ok else 1


def _cmd_board_def(args) -> int:
    if args.action == 'fetch':
        from .boarddef.fetch import run_fetch
        res = run_fetch(args.recipe, out_dir=args.out, offline=args.offline)
        if args.json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(res.render())
        return 0 if res.ok else 1
    if args.action == 'validate':
        from .pin_validate import run_pin_validate
        res = run_pin_validate(args.board, via=args.via, quick=args.quick,
                               workdir=args.workdir)
        if args.json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(res.render())
        return res.exit_code
    return 2


def _cmd_eda(args) -> int:
    if args.action == 'plan':
        from .board_db import resolve_board
        from .eda_plan import plan_for_board
        plan = plan_for_board(resolve_board(args.board))
        if args.json:
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(plan.render())
        return 0 if plan.ready else 1
    if args.action == 'install':
        from .eda_plan import install_oss_cad_suite
        try:
            install_oss_cad_suite(dest_root=args.dir, tag=args.tag,
                                  write_config=args.write_config)
        except (FileExistsError, ValueError, RuntimeError, OSError) as e:
            print(f'[FAIL] eda install: {e}')
            return 1
        return 0
    if args.action == 'flasher-pack':
        from .flasher_pack import build_pack
        from .toolchain import TOOL_DIR
        try:
            res = build_pack(args.out or TOOL_DIR / 'runs' / 'flasher-pack')
        except (FileNotFoundError, OSError) as e:
            print(f'[FAIL] eda flasher-pack: {e}')
            return 1
        print(res.render())
        return 0
    return 2


def _cmd_hwcheck(args) -> int:
    from .hw_check import run_hw_check
    res = run_hw_check(args.model, port=args.port, n_vectors=args.vectors,
                       seed=args.seed, baud=args.baud, board=args.board)
    print(res.render())
    return 0 if res.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='mdl-fpga',
        description='griffith-logic IR の FPGA 実装支援ツール',
    )
    sub = p.add_subparsers(dest='verb', required=True)

    d = sub.add_parser('doctor', help='環境診断(ツール検出・JTAG・シリアル)')
    d.add_argument('--json', action='store_true')
    d.add_argument('--no-jtag', action='store_true', help='JTAG プローブを省略')
    d.set_defaults(fn=_cmd_doctor)

    de = sub.add_parser('detect',
                        help='接続ボードの検出(EDA 不要。JTAG/USB/シリアル署名照合)')
    de.add_argument('--json', action='store_true')
    de.add_argument('--no-serial', action='store_true', help='COM ポート列挙を省略')
    de.add_argument('--no-jtag', action='store_true',
                    help='JTAG プローブを省略(ケーブルを掴まない)')
    de.set_defaults(fn=_cmd_detect)

    c = sub.add_parser('convert', help='IR 読込・検証(blackbox 検出はここ)')
    c.add_argument('model', help='IR JSON のパス')
    c.add_argument('--out', default=None, help='検証済み IR の保存先(省略時は検証のみ)')
    c.set_defaults(fn=_cmd_convert)

    s = sub.add_parser('sim', help='RTL 全体シミュレーション(TB 自動生成)')
    s.add_argument('model')
    s.add_argument('--workdir', default=None)
    s.add_argument('--vectors', type=int, default=256)
    s.add_argument('--seed', type=int, default=0)
    s.set_defaults(fn=_cmd_sim)

    k = sub.add_parser('check', help='bit 一致 PASS/FAIL(参照シム ⇔ RTL)')
    k.add_argument('model')
    k.add_argument('--workdir', default=None)
    k.add_argument('--vectors', type=int, default=256)
    k.add_argument('--seed', type=int, default=0)
    k.add_argument('--no-cross', action='store_true',
                   help='ライブラリ生成器との相互検証(第4辺)を省略')
    k.add_argument('--pipelined', action='store_true',
                   help='パイプラインコア(各層FF)の等価性を検証')
    k.set_defaults(fn=_cmd_check)

    a = sub.add_parser('advise', help='搭載可能性の事前判定(EDA 不要、実測則ベース)')
    a.add_argument('model', help='IR JSON か manifest のパス')
    a.add_argument('--board', default=None,
                   help='ボード key か board JSON パス(draft 可。'
                        '省略時: manifest の board か de0-cv)')
    a.add_argument('--json', action='store_true')
    a.set_defaults(fn=_cmd_advise)

    b = sub.add_parser('build', help='HDL 生成 → Quartus 合成 → .sof')
    b.add_argument('model')
    b.add_argument('--workdir', default=None)
    b.add_argument('--board', default='de0-cv',
                   help='ボード key か board JSON パス(draft 可)')
    b.add_argument('--clock-hz', type=int, default=None, dest='clock_hz',
                   help='省略時 manifest → board の clock.hz → 50MHz の順で解決')
    b.add_argument('--name', default='top', help='Quartus プロジェクト名(可変)')
    b.add_argument('--mode', default='spatial', choices=['spatial', 'overlay'])
    b.add_argument('--overlay-v', type=int, default=None, dest='overlay_v',
                   help='overlay のレーン数 V(既定 manifest → 8)')
    b.add_argument('--overlay-vectors', type=int, default=None,
                   dest='overlay_vectors',
                   help='overlay の検証 ROM に焼くベクトル枚数(既定 1000)')
    b.add_argument('--clock-div', type=int, default=None, dest='clock_div',
                   choices=[1, 2, 4],
                   help='overlay のコアクロック分周比(既定 2 = 25MHz)')
    b.add_argument('--overlay-wiring', default=None, dest='overlay_wiring',
                   choices=['auto', 'computed', 'stored'],
                   help='overlay の配線の持ち方(既定 manifest → auto。'
                        'computed = 計算型のみ、stored = 保存型を強制)')
    b.add_argument('--skip-presim', action='store_true',
                   help='合成前 RTL 検証を省略(非推奨)')
    b.add_argument('--selftest', action='store_true',
                   help='オンチップ自己検証を同梱(LEDR[0]点灯=全ベクトルbit一致。'
                        'UART 不要の実機確認)')
    b.set_defaults(fn=_cmd_build)

    isp = sub.add_parser(
        'issp', help='実機 ISSP で overlay を走らせ一致数・サイクル・inf/s を読む')
    isp.add_argument('projdir',
                     help='build --mode overlay の出力(quartus プロジェクト dir)')
    isp.add_argument('--index', type=int, default=0,
                     help='ISSP インスタンス index')
    isp.add_argument('--timeout-ms', type=int, default=60000,
                     dest='timeout_ms')
    isp.set_defaults(fn=_cmd_issp)

    f = sub.add_parser('flash', help='JTAG 書込(USB-Blaster / 内蔵プログラマ)')
    f.add_argument('bitstream', help='.sof / .svf / .rbf / .fs のパス')
    f.add_argument('--cable', default='1', help='quartus_pgm のケーブル番号')
    f.add_argument('--via', default='auto',
                   choices=['auto', 'quartus', 'openfpgaloader'],
                   help='書込経路(auto=拡張子で選択。svf/rbf/fs は EDA 不要)')
    f.add_argument('--ofl-cable', default='usb-blaster', dest='ofl_cable',
                   help='openFPGALoader のケーブル名(--board 指定時は不要)')
    f.add_argument('--board', default=None,
                   help='ボード定義から openFPGALoader の -b/-c を導出'
                        '(key か board JSON パス。例: tang-nano-20k)')
    f.set_defaults(fn=_cmd_flash)

    bu = sub.add_parser('bundle',
                        help='持ち運び zip(sof+svf/rbf 変換+配線 README。'
                             'EDA 無し PC で書込可能に)')
    bu.add_argument('workdir', help='build の workdir(quartus/output_files を含む)')
    bu.add_argument('--manifest', default=None,
                    help='manifest パス(io.link の配線を README に反映)')
    bu.add_argument('--board', default=None)
    bu.add_argument('--out', default=None, help='出力 zip パス')
    bu.add_argument('--no-svf', action='store_true')
    bu.add_argument('--no-rbf', action='store_true')
    bu.add_argument('--no-app', action='store_true', dest='no_app',
                    help='スタンドアローンアプリ(app/)を同梱しない')
    bu.add_argument('--flasher-pack', default=None, dest='flasher_pack',
                    help='openFPGALoader パック(dir か zip)を flasher/ に同梱'
                         '(既定: 環境変数 MDL_FPGA_FLASHER_PACK → '
                         'local.toml の flasher_pack)')
    bu.add_argument('--no-flasher', action='store_true', dest='no_flasher',
                    help='flasher 同梱をオプトアウト(README の導入誘導のみ)')
    bu.set_defaults(fn=_cmd_bundle)

    v = sub.add_parser('view', help='可視化ビューア(クリック=1clk の対話 RTL シム)')
    v.add_argument('model')
    v.add_argument('--port', type=int, default=8788)
    v.add_argument('--workdir', default=None)
    v.add_argument('--no-browser', action='store_true')
    v.add_argument('--pipelined', action='store_true',
                   help='各層FFコアで表示(1clkごとに値が層を進む)')
    v.add_argument('--system', default=None,
                   help='分類システム生成物のディレクトリ(ROM/カウンタ込みの'
                        'システム全体を可視化。build_mnist_demo.py が生成)')
    v.set_defaults(fn=_cmd_view)

    u = sub.add_parser('up', help='一気通貫: detect→advise→build→flash(→hw-check)')
    u.add_argument('model', help='IR JSON か manifest のパス')
    u.add_argument('--board', default=None,
                   help='ボード key か board JSON パス(draft 可=合成前に'
                        'pin-validate ゲート。省略時: manifest の board か '
                        'detect で自動決定)')
    u.add_argument('--workdir', default=None)
    u.add_argument('--no-flash', action='store_true', help='書込を省略(合成まで)')
    u.add_argument('--hw-check', action='store_true', dest='hw_check',
                   help='書込後に UART 実機 bit 一致まで実行')
    u.add_argument('--json', action='store_true')
    u.set_defaults(fn=_cmd_up)

    bd = sub.add_parser('board-def',
                        help='ボード定義のオンデマンド機械生成(AUTOFLOW 工程3)')
    bds = bd.add_subparsers(dest='action', required=True)
    bf = bds.add_parser('fetch',
                        help='レシピ+外部ソースから board JSON draft を生成'
                             '(2ソース機械突合。ピン値の手打ちなし)')
    bf.add_argument('recipe', help='レシピ key(boarddef/recipes/)か JSON パス')
    bf.add_argument('--out', default=None,
                    help='出力ディレクトリ(既定: runs/boarddef/<key>-<時刻>。'
                         '既存ファイルは上書きしない)')
    bf.add_argument('--offline', action='store_true',
                    help='ネットワーク取得を禁止(references/ 等のローカルコピーのみ)')
    bf.add_argument('--json', action='store_true')
    bf.set_defaults(fn=_cmd_board_def)
    bv = bds.add_parser('validate',
                        help='ピン実在・適法性の検証(AUTOFLOW 工程4。'
                             'EDA/nextpnr に stub 合成を食わせて機械確認)')
    bv.add_argument('board', help='ボード key か board JSON のパス(draft 可)')
    bv.add_argument('--via', default='auto',
                    choices=['auto', 'gowin-oss', 'quartus'],
                    help='検証ツール(auto=board の toolchain から決定)')
    bv.add_argument('--quick', action='store_true',
                    help='quartus: map のみ(fit を省略して高速化)')
    bv.add_argument('--workdir', default=None)
    bv.add_argument('--json', action='store_true')
    bv.set_defaults(fn=_cmd_board_def)

    ed = sub.add_parser('eda',
                        help='ツールチェーン導入計画/自動導入(AUTOFLOW 工程6)')
    eds = ed.add_subparsers(dest='action', required=True)
    ep = eds.add_parser('plan', help='ボードに必要なツールと導入手順を提示')
    ep.add_argument('--board', required=True,
                    help='ボード key か board JSON パス(例: tang-nano-20k)')
    ep.add_argument('--json', action='store_true')
    ep.set_defaults(fn=_cmd_eda)
    ei = eds.add_parser('install',
                        help='ツールチェーンの完全自動導入(EULA 不要なもののみ)')
    ei.add_argument('target', choices=['oss-cad-suite'])
    ei.add_argument('--dir', default=None,
                    help='導入先(既定: ~/oss-cad-suite。既存は上書きしない)')
    ei.add_argument('--tag', default=None,
                    help='リリースタグ YYYY-MM-DD(既定: latest)')
    ei.add_argument('--write-config', action='store_true', dest='write_config',
                    help='local.toml の oss_cad_root を導入先で更新')
    ei.set_defaults(fn=_cmd_eda)
    ef = eds.add_parser('flasher-pack',
                        help='bundle 既定同梱の flasher パックを oss-cad-suite'
                             ' から生成(LICENSES 全文+NOTICE 込み)')
    ef.add_argument('--out', default=None,
                    help='出力ディレクトリ(既定: runs/flasher-pack)')
    ef.set_defaults(fn=_cmd_eda)

    h = sub.add_parser('hw-check', help='実機 bit 一致(UART 経由)')
    h.add_argument('model')
    h.add_argument('--port', default=None,
                   help='COM ポート(省略時 自動検出: --board の serial 署名 '
                        '→ CP210x)')
    h.add_argument('--vectors', type=int, default=100)
    h.add_argument('--seed', type=int, default=0)
    h.add_argument('--baud', type=int, default=115200)
    h.add_argument('--board', default=None,
                   help='ボード定義のシリアル署名でポート自動検出'
                        '(例: tang-nano-20k)')
    h.set_defaults(fn=_cmd_hwcheck)

    return p


def main(argv: list[str] | None = None) -> int:
    # cp932 コンソールで日本語メッセージが化けない/落ちないようにする
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
