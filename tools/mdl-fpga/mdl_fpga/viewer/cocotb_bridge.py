"""cocotb ブリッジ(シミュレータ内で走る側)。

vvp が VPI 経由で Python を埋め込み、本モジュールの cocotb テストが
TCP(127.0.0.1:$MDL_FPGA_BRIDGE_PORT)でビューアサーバに接続して
JSON 行プロトコルのコマンドループを回す。シミュレーション時間は
コマンドが来たときだけ進む(=対話ステップ実行)。

メタ駆動の信号解決(v3): 各ノードの `path`(DUT 相対のドット径路)を
ハンドルへ解決して**セッション中キャッシュ**する(VPI 探索は信号ごとに
1回だけ)。path が無い旧メタは `u_core.<wire>` へフォールバックし、
コアモードの後方互換を保つ。システムモード(allow_set_x=false)では
set_x を拒否する(入力は ROM 供給)。

先行実験の実証パターンを踏襲:
  - クロックは Timer で手動駆動(_tick)
  - X/Z 安全読み出し(未初期化 reg は X → 0 に解決しつつ報告)
  - パックドレジスタはベクトル一括読み+Python スライス(handle 添字は不可)
"""

import json
import os
import socket

import cocotb
from cocotb.triggers import Timer


def _u(v) -> tuple[int, bool]:
    """X/Z 耐性の unsigned 読み出し。(値, X/Zを含んだか) を返す。"""
    try:
        return int(v), False
    except (ValueError, TypeError):
        s = ''.join('0' if c in 'xzXZuU-' else c for c in str(v))
        return (int(s, 2) if s else 0), True


async def _tick(dut):
    dut.clk.value = 1
    await Timer(1, unit='ns')
    dut.clk.value = 0
    await Timer(1, unit='ns')


def _resolve(dut, path: str):
    """DUT 相対のドット径路を cocotb ハンドルへ解決する。"""
    h = dut
    for part in path.split('.'):
        h = getattr(h, part)
    return h


def _build_handles(dut, meta: dict) -> dict[str, object]:
    """表示キー → ハンドル(セッション中キャッシュ)。"""
    paths: dict[str, str] = {}
    for n in meta['nodes']:
        paths[n['id']] = n.get('path') or ('u_core.' + n['wire'])
    extra = meta.get('extra_signals') or \
        {'x': 'xin', 'y_comb': 'y_comb', 'y': 'y_reg'}
    paths.update(extra)

    handle_by_path: dict[str, object] = {}
    handles: dict[str, object] = {}
    for key, path in paths.items():
        if path not in handle_by_path:
            handle_by_path[path] = _resolve(dut, path)
        handles[key] = handle_by_path[path]
    return handles


def _read_values(handles: dict) -> tuple[dict, list]:
    """{キー: hex文字列} と X/Z を含んだキー一覧(同一ハンドルは1回読み)。"""
    cache: dict[int, tuple[int, bool]] = {}
    values: dict[str, str] = {}
    xz: list[str] = []
    for key, handle in handles.items():
        hid = id(handle)
        if hid not in cache:
            cache[hid] = _u(handle.value)
        val, has_xz = cache[hid]
        values[key] = f'{val:x}'
        if has_xz:
            xz.append(key)
    return values, xz


@cocotb.test()
async def bridge(dut):
    port = int(os.environ['MDL_FPGA_BRIDGE_PORT'])
    sock = socket.create_connection(('127.0.0.1', port), timeout=30)
    sock.settimeout(None)   # コマンド待ちは無期限(サーバ都合で閉じられる)
    f = sock.makefile('rw', encoding='utf-8', newline='\n')

    # 接続後の失敗は必ずサーバへ報告してから死ぬ(無言切断のデバッグ地獄回避)
    try:
        with open(os.environ['MDL_FPGA_VIEW_META'], encoding='utf-8') as mf:
            meta = json.load(mf)
        allow_set_x = bool(meta.get('allow_set_x', True))
        handles = _build_handles(dut, meta)

        # 初期状態: クロック Low・リセット解除(・入力 0)
        dut.clk.value = 0
        dut.rst.value = 0
        if allow_set_x:
            dut.x_in.value = 0
        await Timer(1, unit='ns')
    except Exception as e:  # noqa: BLE001
        f.write(json.dumps({'ok': False, 'hello': True,
                            'fatal': f'{type(e).__name__}: {e}'}) + '\n')
        f.flush()
        f.close()
        sock.close()
        raise

    cycle = 0
    f.write(json.dumps({'ok': True, 'hello': True,
                        'n_signals': len(handles)}) + '\n')
    f.flush()

    while True:
        line = f.readline()
        if not line:
            break
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            f.write(json.dumps({'ok': False, 'error': 'bad json'}) + '\n')
            f.flush()
            continue

        op = cmd.get('cmd')
        resp: dict = {'ok': True}
        try:
            if op == 'quit':
                f.write(json.dumps({'ok': True, 'bye': True}) + '\n')
                f.flush()
                break
            elif op == 'reset':
                dut.rst.value = 1
                await _tick(dut)
                await _tick(dut)
                dut.rst.value = 0
                await Timer(1, unit='ns')
                cycle = 0
            elif op == 'set_x':
                if not allow_set_x:
                    raise ValueError('入力は ROM 供給(システムモード)。'
                                     'step で時間を進めてください')
                dut.x_in.value = int(cmd['hex'], 16)
                await Timer(1, unit='ns')
            elif op == 'step':
                for _ in range(max(1, int(cmd.get('n', 1)))):
                    await _tick(dut)
                    cycle += 1
            elif op == 'step_until':
                # 指定信号(スライス適用後)の値が変化するまで進める(上限付き)
                key = cmd['key']
                handle = handles.get(key)
                if handle is None:
                    raise ValueError(f'不明な監視対象: {key}')
                sl = cmd.get('slice')
                max_steps = max(1, int(cmd.get('max', 1000)))

                def _watched(h=handle, s=sl) -> int:
                    v, _ = _u(h.value)
                    if s:
                        msb, lsb = int(s[0]), int(s[1])
                        v = (v >> lsb) & ((1 << (msb - lsb + 1)) - 1)
                    return v

                start_val = _watched()
                steps = 0
                changed = False
                while steps < max_steps:
                    await _tick(dut)
                    cycle += 1
                    steps += 1
                    if _watched() != start_val:
                        changed = True
                        break
                resp['steps'] = steps
                resp['changed'] = changed
                resp['watch'] = key
            elif op == 'read':
                pass   # 値だけ返す
            else:
                resp = {'ok': False, 'error': f'unknown cmd: {op}'}
        except Exception as e:  # noqa: BLE001 - ブリッジは落とさず報告する
            resp = {'ok': False, 'error': f'{type(e).__name__}: {e}'}

        if resp.get('ok'):
            values, xz = _read_values(handles)
            resp.update({'cycle': cycle, 'values': values, 'xz': xz})
        f.write(json.dumps(resp) + '\n')
        f.flush()

    f.close()
    sock.close()
