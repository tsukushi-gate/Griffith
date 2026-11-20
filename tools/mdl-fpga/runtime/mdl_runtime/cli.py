"""mdl_runtime CLI(第一級の統合面。STANDALONE.md)。

- 人間向け表示に加え --json / serve-stdio(stdin コマンド→stdout JSON 行)を
  第一級で提供する= Python 以外のフレームワークからの公式 API。
- 使い方(bundle 展開先で): `python app/run.py <subcommand>`
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, flash as _flash
from .contract import find_app_dir, load_contract, load_goldens
from .link import UartLink
from .protocol import decode_scores
from .selftest import run_selftest


def _open_link(contract: dict, port: str | None) -> UartLink:
    """テストから差し替える注入点。"""
    return UartLink.from_contract(contract, port)


def _score_spec(contract: dict) -> tuple[int, int] | None:
    out = (contract.get('outputs') or [{}])[0]
    if out.get('semantics') == 'class_scores' and out.get('k'):
        return out['k'], out['score_bits']
    return None


def _describe_y(contract: dict, y: int) -> dict:
    doc: dict = {'y': f'0x{y:x}'}
    spec = _score_spec(contract)
    if spec:
        k, sb = spec
        scores = decode_scores(y, k, sb)
        doc['scores'] = scores
        doc['pred'] = max(range(k), key=lambda i: scores[i])
    return doc


def _emit(args, doc: dict, human: str) -> None:
    print(json.dumps(doc, ensure_ascii=False) if args.json else human)


def _cmd_info(args, contract) -> int:
    human = (f'{contract["name"]}: in={contract["in_width"]}bit '
             f'out={contract["out_width"]}bit '
             f'baud={contract["uart"]["baud"]} '
             f'board={contract.get("board", {}).get("key")} '
             f'(contract v{contract["contract_version"]}, '
             f'runtime {__version__})')
    _emit(args, contract, human)
    return 0


def _cmd_ports(args, contract) -> int:
    from .ports import find_port, list_candidates
    cands = list_candidates(contract)
    best = find_port(contract)
    if args.json:
        print(json.dumps({'candidates': cands, 'selected': best},
                         ensure_ascii=False))
    else:
        for c in cands:
            mark = '*' if c['matched'] else ' '
            print(f' {mark} {c["device"]}: {c["description"]}')
        print(f'自動選択: {best or "(一致なし。--port で明示指定)"}')
    return 0 if best else 1


def _cmd_flash(args, contract) -> int:
    res = _flash.run_flash(contract, args.app, exe=args.flasher,
                           persist=args.persist, dry_run=args.dry_run)
    human = ('[OK] ' if res['ok'] else '[FAIL] ') + \
        (' '.join(res.get('argv', [])) or '') + \
        ('\n' + res.get('detail', '') if res.get('detail') else '')
    _emit(args, res, human)
    return 0 if res['ok'] else 1


def _cmd_send(args, contract) -> int:
    x = int(args.x, 16) if args.x.lower().startswith('0x') else int(args.x)
    with _open_link(contract, args.port) as link:
        y = link.query(0, x)
    doc = {'x': f'0x{x:x}', **_describe_y(contract, y)}
    human = f'x=0x{x:x} -> y={doc["y"]}'
    if 'pred' in doc:
        human += f' scores={doc["scores"]} pred={doc["pred"]}'
    _emit(args, doc, human)
    return 0


def _cmd_run(args, contract) -> int:
    src = sys.stdin if args.input in (None, '-') else open(args.input,
                                                          encoding='utf-8')
    dst = sys.stdout if args.output in (None, '-') else open(args.output, 'w',
                                                             encoding='utf-8')
    try:
        with _open_link(contract, args.port) as link:
            for i, line in enumerate(src):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                x = int(line, 16) if line.lower().startswith('0x') \
                    else int(line)
                y = link.query(i, x)
                dst.write(f'0x{x:x},0x{y:x}\n')
    finally:
        if src is not sys.stdin:
            src.close()
        if dst is not sys.stdout:
            dst.close()
    return 0


def _cmd_selftest(args, contract) -> int:
    goldens = load_goldens(args.app, contract)
    with _open_link(contract, args.port) as link:
        res = run_selftest(link, goldens)
    human = (f'selftest: {"PASS" if res["passed"] else "FAIL"}'
             f'({res["n_ok"]}/{res["n"]} bit一致)')
    if res['first_fail']:
        human += f'\n  最初の不一致: {res["first_fail"]}'
    _emit(args, res, human)
    return 0 if res['passed'] else 1


def _cmd_serve_stdio(args, contract) -> int:
    """stdin: JSON 行コマンド / stdout: JSON 行応答(言語非依存 API)。"""
    link = None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                op = cmd.get('cmd')
                if op == 'quit':
                    print(json.dumps({'ok': True, 'bye': True}), flush=True)
                    break
                if op == 'info':
                    resp = {'ok': True, 'contract': contract}
                elif op == 'send':
                    if link is None:
                        link = _open_link(contract, args.port)
                    x = int(cmd['x'], 16) if str(cmd['x']).lower(
                        ).startswith('0x') else int(cmd['x'])
                    y = link.query(int(cmd.get('seq', 0)), x)
                    resp = {'ok': True, 'x': f'0x{x:x}',
                            **_describe_y(contract, y)}
                elif op == 'selftest':
                    if link is None:
                        link = _open_link(contract, args.port)
                    resp = {'ok': True,
                            **run_selftest(link,
                                           load_goldens(args.app, contract))}
                else:
                    resp = {'ok': False, 'error': f'unknown cmd: {op}'}
            except Exception as e:  # noqa: BLE001 - サーバは落とさず報告
                resp = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
            print(json.dumps(resp, ensure_ascii=False), flush=True)
    finally:
        if link is not None:
            link.close()
    return 0


def _cmd_gui(args, contract) -> int:
    try:
        import pygame  # noqa: F401
    except ImportError:
        print('pygame 未導入のため GUI を起動できない。'
              '`pip install -r app/requirements.txt` を実行するか、'
              'CUI(send / run / serve-stdio)を使う')
        return 1
    from .templates import canvas_app
    with _open_link(contract, args.port) as link:
        canvas_app.run(contract, link)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='mdl-runtime',
        description='mdl bundle スタンドアローンランタイム(io_contract 駆動)')
    p.add_argument('--app', default='.',
                   help='app ディレクトリか bundle 展開ルート(既定: .)')
    sub = p.add_subparsers(dest='verb', required=True)

    def _add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument('--json', action='store_true')
        sp.set_defaults(fn=fn)
        return sp

    _add('info', _cmd_info, '契約(入出力・ボード)の表示')
    _add('ports', _cmd_ports, 'シリアルポート一覧と自動選択')
    f = _add('flash', _cmd_flash, '書込(同梱 flasher → PATH の順に解決)')
    f.add_argument('--flasher', default=None, help='openFPGALoader を明示指定')
    f.add_argument('--persist', action='store_true',
                   help='SPI フラッシュへ永続書込(電源断でも保持。対応ボード'
                        'のみ。既定は SRAM 書込=電源断で消える)')
    f.add_argument('--dry-run', action='store_true', dest='dry_run')
    s = _add('send', _cmd_send, '1ベクトル送信(x は 0x… か 10進)')
    s.add_argument('x')
    s.add_argument('--port', default=None)
    r = sub.add_parser('run', help='バッチ実行(1行1ベクトル → "x,y" 行)')
    r.add_argument('--input', default='-', help='入力ファイル(- = stdin)')
    r.add_argument('--output', default='-', help='出力ファイル(- = stdout)')
    r.add_argument('--port', default=None)
    r.set_defaults(fn=_cmd_run, json=False)
    t = _add('selftest', _cmd_selftest,
             '実機 bit 一致セルフテスト(同梱 golden 再生)')
    t.add_argument('--port', default=None)
    v = sub.add_parser('serve-stdio',
                       help='stdin JSON コマンド / stdout JSON 行'
                            '(他フレームワーク連携用 API)')
    v.add_argument('--port', default=None)
    v.set_defaults(fn=_cmd_serve_stdio, json=True)
    g = sub.add_parser('gui', help='pygame GUI(手描き/ビットトグル+出力表示)')
    g.add_argument('--port', default=None)
    g.set_defaults(fn=_cmd_gui, json=False)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = build_parser().parse_args(argv)
    try:
        contract = load_contract(find_app_dir(args.app))
    except (FileNotFoundError, ValueError) as e:
        print(f'[FAIL] {e}')
        return 2
    args.app = str(find_app_dir(args.app))
    try:
        return args.fn(args, contract)
    except ImportError as e:
        print(f'[FAIL] 依存が未導入: {e}\n'
              '       `pip install -r app/requirements.txt` を実行してから'
              '再実行する(README の環境構築手順を参照)')
        return 2
    except (TimeoutError, ConnectionError) as e:
        print(f'[FAIL] {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
